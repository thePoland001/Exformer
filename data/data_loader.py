import os
from logging import root

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from utils.tools import get_statistical, get_statistical_dan
from utils.scale import StandardNorm
from utils.ext_utils import log_std_denorm_dataset
from utils.misc import fprint


import warnings

warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, features='M',
                 data_split=[0.7, 0.1, 0.2], scale=True, scale_statistic=None, target='OT', timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.scale = scale
        # self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if (self.data_split[0] < 1):
            train_num = int(len(df_raw) * self.data_split[0])
            test_num = int(len(df_raw) * self.data_split[2])
            val_num = len(df_raw) - train_num - test_num
        else:
            train_num = self.data_split[0]
            val_num = self.data_split[1]
            test_num = self.data_split[2]
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num + val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        # assume, first col = timestamp, last col = label
        cols_data = df_raw.columns[1:-1]  # all feature columns
        col_label = df_raw.columns[-1]  # the label column

        df_data = df_raw[cols_data]
        df_label = df_raw[[col_label]]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)

        else:
            data = df_data.values
        data_label = df_label.values  # raw labels, no scaling

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_label = data_label[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        label_y = self.data_label[s_begin:s_end]  # corresponding labels

        seq_x_mark = 6.5
        seq_y_mark = 5.3
        cycle_index = 6
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y

    def __len__(self):
        return len(self.data_x) - self.in_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_DAN_Watershed(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, features='M',
                 target='OT', timeenc=0, freq='h', cycle=None, dan_norm_type='std', anorm_thres=0.9,
                 merge_to_series=False, scale_statistic=None, Scale=None, watershed=1):
        assert flag in ['train', 'val', 'test']
        assert size is not None and len(size) == 3, "size must be [seq_len, label_len, pred_len]"

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.dan_norm_type = dan_norm_type
        self.merge_to_series = merge_to_series
        self.scale_statistic = scale_statistic
        self.watershed = watershed
        self.anorm_thres = anorm_thres
        print(self.anorm_thres)

        self.data_x = None
        self.data_x_label = None
        self.data_y_full = None
        self.data_x_sel = None
        self.data_y_target = None
        self.series_x = None
        self.series_y = None
        self.series_label = None
        self.series_cycle = None
        self.anomaly = None
        self.mean = None
        self.std = None

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        base_dir = os.path.join(self.root_path, self.data_path, f'in{self.seq_len}_out{self.pred_len}')
        x_path = os.path.join(base_dir, f"{self.flag}_x.npy")
        y_path = os.path.join(base_dir, f"{self.flag}_y.npy")
        x_label_path = os.path.join(base_dir, f"{self.flag}_labels.npz")

        self.data_x = np.load(x_path).astype(np.float32)      # (N, seq_len, Cx)
        self.data_y_full = np.load(y_path).astype(np.float32) # (N, out_len, Cy)
        # self.data_x_label =  np.load(x_label_path)['labels'].astype(np.float32)  # (N, seq_len)


        y_target_col = None
        # ---------------- X Columns ---------------- | ---------------- Y Columns ----------------
        # [0]  log_std_value                          | [0]  log_std_GT
        # [1]  gm3_outlier_score                      | [1]  cos(date)
        # [2]  gmm0_post_high_weight                  | [2]  sin(date)
        # [3]  gmm0_post_low_weight                   | [3]  GT_lag1_or_preGT
        # [4]  gmm0_post_remaining                    | [4]  raw_GT
        # [5]  std_norm_value                         | [5]  log_std_GT_dup
        # [6]  year                                   | [6]  rain_log_std_past_window
        # [7]  month                                  | [7]  rain_log_std_current_window
        # [8]  day                                    | [8]  standard_norm_GT
        # [9]  hour                                   |
        # [10] minute                                 |
        # [11] minute                                 |
        if self.dan_norm_type == 'std':
            x_stream_col = 5
            y_target_col = 8
        elif self.dan_norm_type == 'log-std':
            x_stream_col = 0
            y_target_col = 0
        elif self.dan_norm_type == 'ori':
            x_stream_col = 6
            y_target_col = 4

            # encoder input: stream + rain if available
        if self.watershed >= 1:
            rain_col = 11  # rain added during data generation
            print(self.data_x.shape)
            self.data_x_sel = self.data_x[:, :, [x_stream_col, rain_col]]  # (N, seq_len, 2)
        else:
            self.data_x_sel = self.data_x[:, :, [x_stream_col]]  # (N, seq_len, 1)


        self.data_y_target = self.data_y_full[:, :, [y_target_col]]  # (N, out_len, 1)
        stat_file = os.path.join(base_dir, "mean_std_mini.pt")
        if os.path.isfile(stat_file):
            train_mean, train_std = get_statistical_dan(base_dir, self.dan_norm_type)
            self.scale_norm = StandardNorm(mean=train_mean, std=train_std)
            self.mean = train_mean
            self.std = train_std

        self.anomaly = (self.data_x[:, :, 1:2] > self.anorm_thres).astype(np.float32)


    def __getitem__(self, index):
        seq_x = self.data_x_sel[index]   # (seq_len, 1)
        seq_y = self.data_y_target[index]   # (pred_len, 1)
        # dummy values
        seq_x_mark = 6.5
        seq_y_mark = 5.3
        cycle_index = 6
        # label_y = self.data_x_label[index]
        label_y = self.anomaly[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y

    def __len__(self):
        return self.data_x_sel.shape[0]

    def inverse_transform(self, data, norm_type=None, part='test', obj='predict'):
        if self.dan_norm_type == 'std' and self.scale_norm is not None:
            use_norm_type = self.dan_norm_type if norm_type is None else norm_type
            return self.scale_norm.inverse_transform(data, use_norm_type, part, obj)
        else:
            return log_std_denorm_dataset(self.mean, self.std, data)

