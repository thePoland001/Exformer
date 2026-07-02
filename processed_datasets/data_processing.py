#!/usr/bin/env python
# encoding: utf-8
import argparse
import os
import sys
from scipy import stats
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from datetime import datetime, timedelta
from tqdm import tqdm
from utils.ext_utils import (initial_seed, parse_kv_argfile, load_config,
                       gen_month_tag, gen_time_feature, cos_date, sin_date, RnnDataset,
                       log_std_normalization, log_std_normalization_with_stats,
                       standard_normalization, standard_normalization_with_stats)


class DataGenerate:

    def __init__(self, data_path, args):

        # All data: Train + Validation + Test
        self.all_input_data = pd.read_csv(args.data_path + args.stream_sensor + ".csv", sep="\t")
        self.all_input_data.columns = ["id", "datetime", "value"]
        self.all_input_data.sort_values("datetime", inplace=True)
        self.all_input_data.reset_index(drop=True, inplace=True)
        self.kruskal = args.oversampling

        # Rain / auxiliary sensor data (used when watershed >= 1, mirrors DS.py R_X branch)
        self.all_R_input_data = None
        if args.watershed >= 1:
            self.all_R_input_data = pd.read_csv(args.data_path + args.rain_sensor + ".csv", sep="\t")
            self.all_R_input_data.columns = ["id", "datetime", "value"]
            self.all_R_input_data.sort_values("datetime", inplace=True)
            self.all_R_input_data.reset_index(drop=True, inplace=True)

        self.sensor_all_data, self.all_data, self.all_data_time = None, None, None
        self.sensor_all_data_norm, self.sensor_all_data_norm_list = None, None

        self.all_month, self.all_day, self.all_hour = None, None, None
        self.all_tag, self.all_cos_d, self.all_sin_d = None, None, None

        # some parameters
        self.data_lens = args.input_len + args.output_len + 1

        # GMM3 — fit on raw data (DS.py approach)
        self.gm3 = GaussianMixture(n_components=3)
        self.gm3_train_recover_prob = None
        self.gm3_min_thres, self.gm3_max_thres = None, None

        self.gmm0 = GaussianMixture(n_components=3)
        self.gmm0_train_recover_prob, self.gmm0_means = None, None

        # Train data
        self.sensor_train_data_norm_list, self.sensor_train_data_norm = None, None
        self.train_data, self.train_data_time = None, None
        self.mean, self.std = None, None          # log_std normalization stats
        self.stdn_mean, self.stdn_std = None, None  # standard normalization stats
        self.log_std_train_data_norm = None       # alias kept for label building in get_train_data
        self.standard_train_data_norm = None      # standard-normed train series for label building
        self.train_tag, self.train_cos_d, self.train_sin_d = None, None, None
        self.train_month, self.train_day, self.train_hour = None, None, None
        self.Train_DataSets = None
        self.x_train, self.y_train = None, None
        self.train_data_loader = None

        self.x_normal_train, self.y_normal_train = None, None

        self.h_value = []           # all kruskal H values seen during sampling
        self.sampled_h_value = []   # H values for accepted samples

        self.R_sensor_data_norm, self.R_mean, self.R_std = None, None, None
        self.R_data, self.R_sensor_data_norm1 = None, None

        # Validation data
        self.val_points = []

        # Test data
        self.test_data_index = []

        # Get data
        self.read_train_dataset(args)
        self.get_val_data_index(args)
        self.get_train_data(args)
        self.get_test_data_index(args)

        val_x, val_y = self.get_batch_data(self.val_points, args)
        test_x, test_y = self.get_batch_data(self.test_data_index, args)

        save_dir = os.path.join(args.outf, args.name, f"in{args.input_len}_out{args.output_len}")
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "train_x.npy"), self.x_train)
        np.save(os.path.join(save_dir, "train_y.npy"), self.y_train)
        # np.save(os.path.join(save_dir, "train_x_normal.npy"), self.x_normal_train)
        # np.save(os.path.join(save_dir, "train_y_normal.npy"), self.y_normal_train)
        np.save(os.path.join(save_dir, "val_x.npy"), val_x)
        np.save(os.path.join(save_dir, "val_y.npy"), val_y)
        np.save(os.path.join(save_dir, "test_x.npy"), test_x)
        np.save(os.path.join(save_dir, "test_y.npy"), test_y)

        # normalization stats: log_std + standard + R
        # mean_std_mini = {'mean': self.mean, 'std': self.std,
        #                  'stdn_mean': self.stdn_mean, 'stdn_std': self.stdn_std,
        #                  'R_mean': self.R_mean, 'R_std': self.R_std}

        mean_std_mini = {'mean': self.mean, 'std': self.std,
                         'stdn_mean': self.stdn_mean, 'stdn_std': self.stdn_std,
                         'R_mean': self.R_mean, 'R_std': self.R_std
                         }
        torch.save(mean_std_mini, os.path.join(save_dir, "mean_std_mini.pt"))

        print('DataGenerate initialized with reservoir sensor:', args.stream_sensor)

    def read_train_dataset(self, args):

        trainX = self.all_input_data[["datetime", "value"]]

        # read sensor data to vector
        train_start_num = trainX[trainX["datetime"] == args.train_start_point].index.values[
            0]  # start_point: start time of the train set
        print("for sensor ", args.stream_sensor, "train_start_num is: ", train_start_num)
        # foot label of train_end
        train_length = (trainX[trainX["datetime"] == args.train_end_point].index.values[
                            0] - train_start_num)  # train_point: end time of the train set
        print("train set length is : ", train_length)

        sensor_data = trainX[
                      train_start_num: train_length + train_start_num]
        data = np.array(sensor_data["value"].fillna(np.nan))
        data_time = np.array(sensor_data["datetime"].fillna(np.nan))

        # DS.py preprocessing: log_std normalization as primary
        sensor_train_data_norm, mean, std = log_std_normalization(data)
        sensor_train_data_norm_list = [[ff] for ff in sensor_train_data_norm]

        # standard normalization (on top of log_std, as in original data_processing)
        standard_train_data_norm, stdn_mean, stdn_std = standard_normalization(data)
        standard_train_data_norm_list = [[ff] for ff in standard_train_data_norm]

        self.train_data, self.train_data_time = data, data_time
        self.sensor_train_data_norm = sensor_train_data_norm
        self.mean, self.std = mean, std
        self.stdn_mean, self.stdn_std = stdn_mean, stdn_std
        self.log_std_train_data_norm = sensor_train_data_norm   # alias for label building in get_train_data
        self.standard_train_data_norm = standard_train_data_norm  # for label building in get_train_data

        # GMM gm3: fit on raw data (DS.py approach)
        gmm_input = sensor_train_data_norm  # gmm0 still uses log_std normed values

        clean_data = []
        for ii in range(len(data)):
            if (data[ii] is not None) and (np.isnan(data[ii]) != 1):
                clean_data.append(data[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)

        # dataset-wise gmm gm3 for dim=1
        self.gm3.fit(sensor_data_prob)

        save_dir = os.path.join(args.outf, args.name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.gm3, os.path.join(save_dir, "train_GM3.pt"))

        gm_means = np.squeeze(self.gm3.means_)
        gm3_z0 = np.min(gm_means)
        gm3_z1 = np.median(gm_means)
        gm3_z2 = np.max(gm_means)

        gm3_thre1 = (gm3_z0 + gm3_z1) / 2
        gm3_thre2 = (gm3_z1 + gm3_z2) / 2

        self.gm3_min_thres, self.gm3_max_thres = gm3_thre1, gm3_thre2

        print("gm3.means are: ", gm_means)
        print("gm3 thresholds are: {} {}, and min, median, max are: {} {}, {}".format(gm3_thre1, gm3_thre2, gm3_z0, gm3_z1, gm3_z2))  # 打印阈值
        print("gm3.covariances are: {}, and gm3.weights are: {}".format(self.gm3.covariances_, self.gm3.weights_))

        gm3_weights = self.gm3.weights_
        gm3_prob3 = self.gm3.predict_proba(sensor_data_prob)

        # computing a score to gently highlight the extreme values
        gm3_prob_in_distribution = (gm3_prob3[:, 0] * gm3_weights[0] +
                                    gm3_prob3[:, 1] * gm3_weights[1] +
                                    gm3_prob3[:, 2] * gm3_weights[2])
        gm3_prob_like_outlier = 1 - gm3_prob_in_distribution
        gm3_prob_like_outlier = gm3_prob_like_outlier.reshape((len(sensor_data_prob), 1))  # shape(训练集长度, 1)

        recover_data = []
        temp = [0.5]   # DS.py: fill NaN positions with neutral outlier score
        jj = 0
        for ii in range(len(data)):
            if (data[ii] is not None) and (np.isnan(data[ii]) != 1):
                recover_data.append(gm3_prob_like_outlier[jj])
                jj = jj + 1
            else:
                recover_data.append(temp)
        gm3_prob_like_outlier = np.array(recover_data, np.float32).reshape(len(data), 1)

        self.gm3_train_recover_prob = gm3_prob_like_outlier  # 保存 gm3 的恢复概率


        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, gm3_prob_like_outlier),1)  # dim=1, from gm3  # 把“离群分数”拼成新的特征：in dim=1. (1阶差分归一化，离群分数，)

        clean_data = []
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                clean_data.append(gmm_input[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)

        series = []
        random.seed(args.val_seed)
        for ggg in range(200000):
            g0 = random.randint(0, len(gmm_input) - args.output_len)
            if not np.isnan(gmm_input[g0]).any():
                series.append([gmm_input[g0]])

        self.gmm0.fit(np.array(series).reshape(-1, 1))

        torch.save(self.gmm0, os.path.join(save_dir, "train_GMM0.pt"))

        gmm0_means = np.squeeze(self.gmm0.means_)
        print("gmm0.means are: {}, and gmm0.weights are: {}".format(gmm0_means, self.gmm0.weights_))
        gmm0_weights3 = self.gmm0.weights_


        data_prob30 = self.gmm0.predict_proba(sensor_data_prob)

        order1 = np.argmax(gmm0_weights3)
        d0 = data_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmin(gmm0_weights3)
        d1 = data_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("new order is, ", order1, order2, order3)

        data_prob3 = np.concatenate((d0, d1), 1)
        data_prob3 = np.concatenate((data_prob3, data_prob30[:, order3].reshape(-1, 1)), 1)  # # data_prob3.shape (训练集长度, 3)，每行是一个样本在 3 个成分上的后验概率

        recover_prob = []
        temp = np.zeros(np.array(data_prob3[0]).shape)
        jj = 0
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                recover_prob.append(data_prob3[jj])
                jj = jj + 1
            else:
                recover_prob.append(temp)
        recover_prob = np.array(recover_prob, np.float32)

        self.gmm0_train_recover_prob = recover_prob

        # dims 2, 3, 4: gmm0 ordered posteriors
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, recover_prob[:, 0:1]), 1)
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, recover_prob[:, 1:2]), 1)
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, recover_prob[:, 2:3]), 1)

        # dim 5: standard normalization
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, standard_train_data_norm_list), 1)

        # dims 6-10: time features
        data_time_str = data_time.astype(str)
        data_time_pd = pd.to_datetime(data_time_str)

        year = data_time_pd.year
        month = data_time_pd.month
        day = data_time_pd.day
        hour = data_time_pd.hour
        minute = data_time_pd.minute

        time_features = np.stack([year, month, day, hour, minute], axis=1)

        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, time_features), 1)


        # R_data: DS.py watershed branch─
        if args.watershed >= 1:
            # external rain / auxiliary sensor data (DS.py opt_hinter_dim >= 1 branch)
            R_trainX = self.all_R_input_data[["datetime", "value"]]
            R_start_num = R_trainX[R_trainX["datetime"] == args.train_start_point].index.values[0]
            R_train_length = (R_trainX[R_trainX["datetime"] == args.train_end_point].index.values[0] - R_start_num)
            R_sensor_data = R_trainX[R_start_num: R_train_length + R_start_num]
            self.R_data = np.array(R_sensor_data["value"].fillna(np.nan))
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(self.R_data)
            self.R_sensor_data_norm1 = [[ff] for ff in self.R_sensor_data_norm]

            # NOW add rain as column 11
            R_train_norm_list = np.array([[ff] for ff in self.R_sensor_data_norm])
            sensor_train_data_norm_list = np.concatenate(
                (sensor_train_data_norm_list, R_train_norm_list), 1
            )
        else:
            # gm3 outlier probability as auxiliary signal (DS.py else branch)
            self.R_data = gm3_prob_like_outlier
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(self.R_data)
            self.R_sensor_data_norm1 = gm3_prob_like_outlier.squeeze()
            self.R_sensor_data_norm = self.R_sensor_data_norm1  # shape (len(data),)

        self.sensor_train_data_norm_list = sensor_train_data_norm_list

        print("sensor_train_data_norm_list, ", sensor_train_data_norm_list)
        print("Finish prob indicator generating.")

        tag = gen_month_tag(sensor_data)  #
        month, day, hour = gen_time_feature(sensor_data)

        self.train_tag = tag
        self.train_month, self.train_day, self.train_hour = month, day, hour  # 保存月份、日期、小时特征

        cos_d = cos_date(month, day, hour)  #  cos_d shape: (N_train, )
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(month, day, hour)  #  sin_d shape: (N_train, )
        sin_d = [[x] for x in sin_d]

        self.train_cos_d, self.train_sin_d = cos_d, sin_d  #

    def get_val_data_index(self, args):

        val_points = []

        near_len = args.output_len
        random.seed(args.val_seed)

        counts_val = 0

        while counts_val < args.val_size:

            i = random.randint(args.output_len, len(self.train_data) - self.data_lens - 1)

            if (
                    (not np.isnan(self.train_data[i: i + self.data_lens]).any())        # DS.py: checks raw data
                    and (not np.isnan(self.R_data[i: i + self.data_lens]).any())        # DS.py: also checks R_data
                    and (
                        self.train_tag[i + args.input_len] <= -9                        # DS.py threshold
                        or -6 < self.train_tag[i + args.input_len] < 0                 # DS.py threshold
                        or 2 <= self.train_tag[i + args.input_len] <= 3
                    )
            ):
                self.train_tag[i + args.input_len] = 2  # tag 2 means in validation set

                for k in range(near_len):
                    self.train_tag[i + args.input_len - k] = 3  # tag 3 means near points of validation set
                    self.train_tag[i + args.input_len + k] = 3

                point = self.train_data_time[i + args.input_len]
                val_points.append([point])

                counts_val = counts_val + 1

        self.val_points = val_points

        val_name = "%s" % (args.model)
        file_name = os.path.join(args.outf, val_name, "val", f"in{args.input_len}_out{args.output_len}_validation_timestamps_24avg.tsv")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pd_temp = pd.DataFrame(data=val_points, columns=["Hold Out Start"])
        pd_temp.to_csv(file_name, sep="\t")
        print("val set saved to : ", file_name)


    def get_train_data(self, args):
        DATA, Label = [], []
        q = args.output_len // 4  # quarter of output window for kruskal split

        random.seed(args.train_seed)
        ii = 0
        while ii < args.train_volume:

            i = random.randint(args.output_len, len(self.train_data) - 31 * args.output_len - 1)

            # DS.py: check both raw data and R_data for NaNs, and tag thresholds
            if (
                (not np.isnan(self.train_data[i: i + self.data_lens]).any())
                and (not np.isnan(self.R_data[i: i + self.data_lens]).any())
                and (
                    self.train_tag[i + args.input_len] <= -9
                    or -6 < self.train_tag[i + args.input_len] < 0
                )
            ):
                data0 = np.array(
                    self.sensor_train_data_norm_list[i: (i + args.input_len)]
                ).reshape(args.input_len, -1)

                b = i + args.input_len
                e = i + args.input_len + args.output_len

                # dim 0: log_std normed GT
                label00 = np.array(self.sensor_train_data_norm[b:e])
                label0 = [[ff] for ff in label00]

                # dim 1: cos date
                label2 = cos_date(self.train_month[b:e], self.train_day[b:e], self.train_hour[b:e])
                label2 = [[ff] for ff in label2]

                # dim 2: sin date
                label3 = sin_date(self.train_month[b:e], self.train_day[b:e], self.train_hour[b:e])
                label3 = [[ff] for ff in label3]

                # dim 3: raw GT lag-1 (pre-window)
                label4 = np.array(self.train_data[(b - 1):(e - 1)]).reshape(-1, 1)

                # dim 4: raw GT
                label5 = np.array(self.train_data[b:e]).reshape(-1, 1)
                label01 = label5.squeeze()  # used for kruskal

                # dim 5: log_std normed GT (alias, kept for compatibility)
                label6 = np.array(self.log_std_train_data_norm[b:e])
                label6 = [[ff] for ff in label6]

                # dim 6: R past outlier score
                label7 = np.array(self.R_sensor_data_norm[(b - args.output_len): b])
                label7 = [[ff] for ff in label7]

                # dim 7: R current outlier score
                label8 = np.array(self.R_sensor_data_norm[b:e])
                label8 = [[ff] for ff in label8]

                # dim 8: standard normed GT
                label9 = np.array(self.standard_train_data_norm[b:e])
                label9 = [[ff] for ff in label9]

                label = np.concatenate((label0, label2), 1)
                label = np.concatenate((label, label3), 1)
                label = np.concatenate((label, label4), 1)
                label = np.concatenate((label, label5), 1)
                label = np.concatenate((label, label6), 1)
                label = np.concatenate((label, label7), 1)
                label = np.concatenate((label, label8), 1)
                label = np.concatenate((label, label9), 1)

                # kruskal H-stat filtering (DS.py logic)
                if (
                    (label01[:q] == label01[q:2*q]).all()
                    and (label01[2*q:3*q] == label01[3*q:]).all()
                    and (label01[q:2*q] == label01[2*q:3*q]).all()
                ):
                    h_stat = 0
                else:
                    h_stat, _ = stats.kruskal(
                        label01[:q], label01[q:2*q], label01[2*q:3*q], label01[3*q:]
                    )

                self.h_value.append(h_stat)

                if h_stat > self.kruskal:
                    DATA.append(data0)
                    Label.append(label)
                    self.train_tag[i + args.input_len] = 4
                    ii += 1
                    self.sampled_h_value.append(h_stat)
                else:
                    kk = random.randint(0, 99)
                    if kk <= args.event_focus_level:
                        DATA.append(data0)
                        Label.append(label)
                        self.train_tag[i + args.input_len] = 4
                        ii += 1
                        self.sampled_h_value.append(h_stat)

        print("DATA shape, ", np.array(DATA).shape)
        print("Label shape, ", np.array(Label).shape)

        self.Train_DataSets = DATA
        self.x_train = np.array(DATA, np.float32)
        self.y_train = np.array(Label, np.float32)

        # No normal/extreme split in DS.py; alias to full set
        self.x_normal_train = self.x_train
        self.y_normal_train = self.y_train

        train_data_tensor = RnnDataset(DATA, Label)
        self.train_data_loader = DataLoader(
            train_data_tensor,
            args.batchsize,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: x,
        )

    def refresh_dataset(self, args):

        all_train = self.all_input_data

        # read sensor data to vector
        start_num = all_train[all_train["datetime"] == args.train_start_point].index.values[0]
        print("for sensor ", args.stream_sensor, "start_num is: ", start_num)
        # foot label of train_end
        train_end = (all_train[all_train["datetime"] == args.train_end_point].index.values[0] - start_num)
        print("train set length is : ", train_end)

        k = all_train[all_train["datetime"] == args.test_end].index.values[0]
        self.sensor_all_data = all_train[start_num:k]

        # --------------------------------------------------
        # --------------------------------------------------
        self.all_data = np.array(self.sensor_all_data["value"].fillna(np.nan))
        self.all_data_time = np.array(self.sensor_all_data["datetime"].fillna(np.nan))

        # DS.py preprocessing: log_std normalization with training stats─
        self.sensor_all_data_norm = log_std_normalization_with_stats(self.all_data, self.mean, self.std)
        self.sensor_all_data_norm_list = [[ff] for ff in self.sensor_all_data_norm]

        # --------------------------------------------------
        # gm3 was trained on raw data; feed raw all_data for inference
        gmm_input = self.sensor_all_data_norm   # gmm0 still uses log_std values

        clean_data = []
        for ii in range(len(self.all_data)):
            if (self.all_data[ii] is not None) and (np.isnan(self.all_data[ii]) != 1):
                clean_data.append(self.all_data[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)

        data_prob3 = self.gm3.predict_proba(sensor_data_prob)  # (N,3)
        weights3 = self.gm3.weights_

        prob_in_distribution3 = (data_prob3[:, 0] * weights3[0] + data_prob3[:, 1] * weights3[1] + data_prob3[:, 2] * weights3[2])

        prob_like_outlier3 = 1 - prob_in_distribution3
        prob_like_outlier3 = prob_like_outlier3.reshape((len(sensor_data_prob), 1))

        recover_data = []
        temp = [0.5]   # DS.py NaN fill
        jj = 0
        for ii in range(len(self.all_data)):
            if (self.all_data[ii] is not None) and (np.isnan(self.all_data[ii]) != 1):
                recover_data.append(prob_like_outlier3[jj])
                jj = jj + 1
            else:
                recover_data.append(temp)
        prob_like_outlier3 = np.array(recover_data, np.float32).reshape(len(self.all_data), 1)

        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, prob_like_outlier3), 1)

        clean_data = []
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                clean_data.append(gmm_input[ii])
        sensor_all_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)

        self.gmm0_means = np.squeeze(self.gmm0.means_)
        weights3 = self.gmm0.weights_
        all_data_prob30 = self.gmm0.predict_proba(sensor_all_data_prob)

        order1 = np.argmax(weights3)
        d0 = all_data_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmin(weights3)
        d1 = all_data_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("new order is, ", order1, order2, order3)

        data_prob3 = np.concatenate((d0, d1), 1)
        data_prob3 = np.concatenate((data_prob3, all_data_prob30[:, order3].reshape(-1, 1)), 1)

        recover_prob = []
        temp = np.zeros(np.array(data_prob3[0]).shape)
        jj = 0
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                recover_prob.append(data_prob3[jj])
                jj = jj + 1
            else:
                recover_prob.append(temp)
        recover_prob = np.array(recover_prob, np.float32).reshape(len(gmm_input), -1)

        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, recover_prob[:, 0:1]), 1)
        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, recover_prob[:, 1:2]), 1)
        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, recover_prob[:, 2:3]), 1)

        # dim 5: standard normalization (using training stats)
        standnorm_all_data = standard_normalization_with_stats(self.all_data, self.stdn_mean, self.stdn_std)
        standnorm_all_data_list = [[ff] for ff in standnorm_all_data]
        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, standnorm_all_data_list), 1)

        # dims 6-10: time features
        all_data_time_str = self.all_data_time.astype(str)
        all_data_time_pd = pd.to_datetime(all_data_time_str)

        year = all_data_time_pd.year
        month = all_data_time_pd.month
        day = all_data_time_pd.day
        hour = all_data_time_pd.hour
        minute = all_data_time_pd.minute

        all_time_features = np.stack([year, month, day, hour, minute], axis=1)

        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, all_time_features), 1)

        # R_data for val/test: DS.py watershed branch─
        if args.watershed >= 1:
            R_all_trainX = self.all_R_input_data[["datetime", "value"]]
            R_start_num = R_all_trainX[R_all_trainX["datetime"] == args.train_start_point].index.values[0]
            R_k = R_all_trainX[R_all_trainX["datetime"] == args.test_end].index.values[0]
            R_sensor_all_data = R_all_trainX[R_start_num:R_k]
            self.R_all_data = np.array(R_sensor_all_data["value"].fillna(np.nan))
            self.R_all_sensor_data_norm = log_std_normalization_with_stats(self.R_all_data, self.R_mean, self.R_std)

            R_all_norm_list = np.array([[ff] for ff in self.R_all_sensor_data_norm])
            self.sensor_all_data_norm_list = np.concatenate(
                (self.sensor_all_data_norm_list, R_all_norm_list), 1
            )

        else:
            # recompute gm3 outlier prob on all_data using training gm3
            clean_data = []
            for ii in range(len(self.all_data)):
                if (self.all_data[ii] is not None) and (np.isnan(self.all_data[ii]) != 1):
                    clean_data.append(self.all_data[ii])
            all_data_prob_input = np.array(clean_data, np.float32).reshape(-1, 1)
            dan_weights3 = self.gm3.weights_
            dan_data_prob3 = self.gm3.predict_proba(all_data_prob_input)
            dan_prob_like_outlier3 = 1 - (
                dan_data_prob3[:, 0] * dan_weights3[0]
                + dan_data_prob3[:, 1] * dan_weights3[1]
                + dan_data_prob3[:, 2] * dan_weights3[2]
            )
            dan_prob_like_outlier3 = dan_prob_like_outlier3.reshape(len(all_data_prob_input), 1)
            recover = []
            temp_fill = [0.5]
            jj = 0
            for ii in range(len(self.all_data)):
                if (self.all_data[ii] is not None) and (np.isnan(self.all_data[ii]) != 1):
                    recover.append(dan_prob_like_outlier3[jj])
                    jj += 1
                else:
                    recover.append(temp_fill)
            self.R_all_data = np.array(recover, np.float32).reshape(len(self.all_data), 1)
            self.R_all_sensor_data_norm = log_std_normalization_with_stats(self.R_all_data, self.R_mean, self.R_std)


        self.R_all_sensor_data_norm1 = np.array(self.R_all_sensor_data_norm).squeeze()
        self.R_all_sensor_data_norm = self.R_all_sensor_data_norm1  # shape (len(all_data),)

        print("Finish prob indicator updating.")


        self.all_tag = gen_month_tag(self.sensor_all_data)
        self.all_month, self.all_day, self.all_hour = gen_time_feature(self.sensor_all_data)  # update

        cos_d = cos_date(self.all_month, self.all_day, self.all_hour)
        self.all_cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.all_month, self.all_day, self.all_hour)
        self.all_sin_d = [[x] for x in sin_d]

        # self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, cos_d, sin_d), axis=1)

    def get_test_data_index(self, args):
        """
        Get the test data as a DataLoader object.

        :param args: Arguments containing batch size and other parameters.
        :return: DataLoader object for test data.
        """

        test_points = []
        self.refresh_dataset(args)

        start_num = self.all_input_data[self.all_input_data["datetime"] == args.train_start_point].index.values[0]

        begin_num = (
                    self.all_input_data[self.all_input_data["datetime"] == args.test_start].index.values[0] - start_num)

        end_num = (self.all_input_data[self.all_input_data["datetime"] == args.test_end].index.values[0] - start_num)

        iterval = 16  # DS.py: iterate every 16 steps

        for i in range(int((end_num - begin_num - args.output_len) / iterval)):
            point = self.all_data_time[begin_num + i * iterval]

            if not np.isnan(
                    np.array(
                        self.all_data[
                            begin_num + i * iterval - args.input_len: begin_num + i * iterval + args.output_len]
                    )
            ).any():
                test_points.append([point])

        self.test_data_index = test_points
        # ------------------------------------------------------------------
        print("Finish getting test data")

    def get_batch_data(self, time_point_list, args):
        results = []
        all_data = self.sensor_all_data.sort_values("datetime").copy()
        all_data = all_data.reset_index(drop=True)

        for t in tqdm(time_point_list, desc="Processing time points"):
            if isinstance(t, list):
                time_point = t[0]
            else:
                time_point = t
            result = self.get_single_data(time_point, all_data, args)
            if result:
                results.append(result)

        if not results:
            return (
                np.empty((0, args.input_len, 10)),  # x_tests
                np.empty((0, args.output_len, 5)),  # y_tests
            )

        x_tests = []
        logstd_gts = []
        dan_now_outliers = []
        dan_pre_outliers = []
        norm_gts = []
        ts_features = []
        pre_gts = []
        gts = []
        stdn_gts = []

        for x_test, logstd_norm_y_test, dan_now_y_prob_like_outlier3, dan_pre_y_prob_like_outlier3, norm_gt, ts_f, pre_gt, gt, stdn_norm_y_test in results:
            x_tests.append(x_test)
            norm_gts.append(np.expand_dims(norm_gt, axis=0))
            ts_features.append(ts_f)
            pre_gts.append(pre_gt)
            gts.append(np.expand_dims(gt.reshape(args.output_len, 1), axis=0))
            logstd_gts.append(np.expand_dims(logstd_norm_y_test, axis=0))
            dan_now_outliers.append(np.expand_dims(dan_now_y_prob_like_outlier3, axis=0))
            dan_pre_outliers.append(np.expand_dims(dan_pre_y_prob_like_outlier3, axis=0))
            stdn_gts.append(np.expand_dims(stdn_norm_y_test, axis=0))

        x_tests_np = np.concatenate(x_tests, axis=0)
        norm_gts_np = np.concatenate(norm_gts, axis=0)
        ts_features_np = np.concatenate(ts_features, axis=0)
        pre_gts_np = np.repeat(np.array(pre_gts)[:, None], args.output_len, axis=1)[..., None]
        gts_np = np.concatenate(gts, axis=0)
        logstd_gts_np = np.concatenate(logstd_gts, axis=0)
        dan_now_outliers_np = np.concatenate(dan_now_outliers, axis=0)
        dan_pre_outliers_np = np.concatenate(dan_pre_outliers, axis=0)
        stdn_gts_np = np.concatenate(stdn_gts, axis=0)

        # y dim layout: 0=log_std GT, 1-2=cos/sin ts, 3=pre_gt, 4=raw GT,
        #               5=log_std GT (compat), 6=R_pre, 7=R_now, 8=stdn GT
        y_tests_np = np.concatenate(
            [norm_gts_np, ts_features_np, pre_gts_np, gts_np,
             logstd_gts_np, dan_pre_outliers_np, dan_now_outliers_np, stdn_gts_np],
            axis=-1
        )

        return x_tests_np, y_tests_np

    def get_single_data(self, time_point, all_data, args):

        try:
            point = all_data[all_data["datetime"] == time_point].index.values[0]
        except IndexError:
            print(f"Time point {time_point} not found in data.")
            return None

        iloc_point = all_data.index.get_loc(point)
        if iloc_point + args.output_len > len(all_data) or iloc_point < args.input_len:
            print(f"Time point {time_point} is out of valid range.")
            return None

        reservoir_data = all_data[point - args.input_len: point]["value"].values.tolist()
        pre_gt = np.array(all_data[point - 1: point]["value"])
        pre_gt = pre_gt[0]
        gt = np.array(all_data[point: point + args.output_len]["value"])

        if pre_gt is None:
            print("pre_gt is None, please fill it or switch to another time point.")
        NN = np.isnan(reservoir_data).any()
        if NN:
            print("There is None value in the input sequence.")

        test_month = []
        test_day = []
        test_hour = []
        test_year = []
        test_minute = []

        new_time = datetime.strptime(time_point, "%Y-%m-%d %H:%M:%S")
        for i in range(args.output_len):
            new_time_temp = new_time + timedelta(minutes=30)
            new_time = new_time.strftime("%Y-%m-%d %H:%M:%S")

            year = int(new_time[0:4])
            month = int(new_time[5:7])
            day = int(new_time[8:10])
            hour = int(new_time[11:13])
            minute = int(new_time[14:16])

            test_month.append(month)
            test_day.append(day)
            test_hour.append(hour)
            test_year.append(year)
            test_minute.append(minute)

            new_time = new_time_temp

        y2 = cos_date(test_month, test_day, test_hour)
        y2 = [[ff] for ff in y2]

        y3 = sin_date(test_month, test_day, test_hour)
        y3 = [[ff] for ff in y3]

        test_ts_features = np.array([np.concatenate((y2, y3), 1)])
        test_timestamp_features = np.array(
            [np.stack([test_year, test_month, test_day, test_hour, test_minute], axis=1)])

        # dim 0: log_std normalized input (DS.py primary normalization)
        x_test = np.array(log_std_normalization_with_stats(reservoir_data, self.mean, self.std), np.float32).reshape(
            args.input_len, -1)
        norm_y_test = np.array(
            log_std_normalization_with_stats(all_data[point: point + args.output_len]["value"].values.tolist(),
                                             self.mean, self.std), np.float32).reshape(args.output_len, -1)

        # dim 1: gm3 outlier prob (gm3 trained on raw data)
        gmm_input = x_test
        weights3 = self.gm3.weights_
        data_prob3 = self.gm3.predict_proba(np.array(reservoir_data, np.float32).reshape(-1, 1))
        prob_in_distribution3 = (
                data_prob3[:, 0] * weights3[0]
                + data_prob3[:, 1] * weights3[1]
                + data_prob3[:, 2] * weights3[2]
        )
        prob_like_outlier3 = 1 - prob_in_distribution3
        prob_like_outlier3 = prob_like_outlier3.reshape(-1, 1)
        prob_like_outlier3 = np.array(prob_like_outlier3, np.float32).reshape(-1, 1)
        x_test = np.concatenate((x_test, prob_like_outlier3), 1)

        self.gmm0_means = np.squeeze(self.gmm0.means_)
        weights3 = self.gmm0.weights_
        data_prob30 = self.gmm0.predict_proba(np.array(gmm_input)[:, 0:1].reshape(-1, 1))
        order1 = np.argmax(weights3)
        d0 = data_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmin(weights3)
        d1 = data_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        data_prob3 = np.concatenate((d0, d1), 1)
        data_prob3 = np.concatenate((data_prob3, data_prob30[:, order3].reshape(-1, 1)), 1)
        recover_prob = np.array(data_prob3, np.float32)
        x_test = np.concatenate((x_test, recover_prob[:, 0:1]), 1)
        x_test = np.concatenate((x_test, recover_prob[:, 1:2]), 1)
        x_test = np.concatenate((x_test, recover_prob[:, 2:3]), 1)

        # dim 5: standard normalization (using training stats)
        stdn_x_test = np.array(standard_normalization_with_stats(reservoir_data, self.stdn_mean, self.stdn_std),
                               np.float32).reshape(args.input_len, -1)
        x_test = np.concatenate((x_test, stdn_x_test), 1)

        # stdn and original-data dims removed (DS.py preprocessing has no equivalent)
        x_timestamp = all_data[point - args.input_len: point]["datetime"].values
        data_time_str = x_timestamp.astype(str)
        data_time_pd = pd.to_datetime(data_time_str)

        year = data_time_pd.year
        month = data_time_pd.month
        day = data_time_pd.day
        hour = data_time_pd.hour
        minute = data_time_pd.minute

        x_time_features = np.stack([year, month, day, hour, minute], axis=1)
        x_test = np.concatenate((x_test, x_time_features), 1)

        # add rain as col 11 for val/test
        if args.watershed >= 1:
            iloc_point = all_data.index.get_loc(point)
            R_x = np.array(
                self.R_all_sensor_data_norm[iloc_point - args.input_len: iloc_point],
                np.float32
            ).reshape(args.input_len, -1)
            x_test = np.concatenate((x_test, R_x), 1)

        x_test = np.array([x_test])  # add batch dim: (1, input_len, feature_dim)

        # labels
        # label dim 0 (norm_y_test already computed above as log_std)
        # label dim 5: log_std norm (same source as dim 0, kept for compat)
        logstd_norm_y_test = norm_y_test.copy()

        # label dim 6: past R outlier score (R_all_sensor_data_norm pre-output window)
        iloc_point = all_data.index.get_loc(point)
        pre_start = iloc_point - args.output_len
        pre_end = iloc_point
        dan_pre_y_prob_like_outlier3 = np.array(
            self.R_all_sensor_data_norm[pre_start:pre_end], np.float32
        ).reshape(args.output_len, -1)

        # label dim 7: current R outlier score (R_all_sensor_data_norm output window)
        now_start = iloc_point
        now_end = iloc_point + args.output_len
        dan_now_y_prob_like_outlier3 = np.array(
            self.R_all_sensor_data_norm[now_start:now_end], np.float32
        ).reshape(args.output_len, -1)

        # label dim 8: standard normalization ground truth
        stdn_norm_y_test = np.array(
            standard_normalization_with_stats(
                all_data[point: point + args.output_len]["value"].values.tolist(),
                self.stdn_mean, self.stdn_std
            ), np.float32
        ).reshape(args.output_len, -1)

        return x_test, logstd_norm_y_test, dan_now_y_prob_like_outlier3, dan_pre_y_prob_like_outlier3, norm_y_test, test_ts_features, pre_gt, gt, stdn_norm_y_test


def data_generation(task_name: str, arg_file_path: str = None):
    parser = argparse.ArgumentParser(description=task_name)

    # default settings

    # dataset and sampling parameters
    parser.add_argument("--data_path", type=str, default="./data/datasets/watershed/raw/", help="path to the dataset")
    parser.add_argument("--stream_sensor", default="Ross_S_fixed", help="stream dataset", )
    parser.add_argument("--name", type=str, default="Ross_withRain", help="name of the experiment")
    parser.add_argument("--rain_sensor", type=str, default="Ross_R_fixed", help="rain sensor name")

    parser.add_argument("--train_seed", type=int, default=1010, help="random seed for train sampling")
    parser.add_argument("--test_seed", type=int, default=2000, help="random seed for test sampling")
    parser.add_argument("--val_seed", type=int, default=2007, help="random seed for val sampling")
    parser.add_argument("--train_volume", type=int, default=30000, help="train set size")
    parser.add_argument("--val_size", type=int, default=120, help="validation set size")

    parser.add_argument("--train_start_point", type=str, default="1988-01-01 14:30:00",
                        help="start time of the train set", )
    parser.add_argument("--train_end_point", type=str, default="2021-08-31 23:30:00",
                        help="end time of the train set", )
    parser.add_argument("--test_start", type=str, default="2021-09-01 00:30:00", help="start time of the test set", )
    parser.add_argument("--test_end", type=str, default="2022-05-31 23:30:00", help="end time of the test set", )

    parser.add_argument("--oversampling", type=int, default=80, help="[unused] kept for config file compat")
    parser.add_argument("--event_focus_level", type=int, default=18, help="0-99; probability (pct) of accepting samples below kruskal threshold")

    # input and output parameters
    parser.add_argument("--input_len", type=int, default=1440, help="length of input vector")
    parser.add_argument("--output_len", type=int, default=288, help="length of output vector")

    # model parameters
    parser.add_argument("--model", type=str, default="Ross_withRain", help="model label")

    # training parameters
    parser.add_argument("--batchsize", type=int, default=48, help="batch size of train data")
    parser.add_argument("--lradj", type=str, default="type4", help="learning rate adjustment policy")
    parser.add_argument("--mode", type=str, default="train", help="set it to train or inference with an existing pt_file", )
    parser.add_argument("--arg_file", type=str, required=True, default="", help=".txt file. If set, reset the default parameters defined in this file.", )
    parser.add_argument("--save", type=int, default=0, help="1 if save the predicted file of testset, else 0", )
    parser.add_argument("--outf", default="./data/datasets/watershed/", help="output folder")
    parser.add_argument('--use_gpu', default=True, help='use gpu or not')

    parser.add_argument("--gpu_id", type=int, default=1, help="gpu ids: e.g. 0. use -1 for CPU")
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--watershed", type=int, default=1, help="watershed index")

    # cli_args = []
    #
    # file_args = []
    # if arg_file_path and os.path.isfile(arg_file_path):
    #     file_args = parse_kv_argfile(arg_file_path)
    #
    # args = parser.parse_args(file_args + cli_args)

    if arg_file_path:
        file_args = load_config(arg_file_path)
        args = parser.parse_args(file_args)
    else:
        args, _ = parser.parse_known_args()
        if args.arg_file:
            file_args = load_config(args.arg_file)
            args = parser.parse_args(file_args + sys.argv[1:])
        else:
            args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_data = DataGenerate(args.data_path, args)

    print(f"Initializing data generation for position:{args.name}!")

if __name__ == '__main__':
    # initial_seed(2025)

    # config_path = f"../records/mcann_configs/{pos}.txt"
    data_generation('test')
