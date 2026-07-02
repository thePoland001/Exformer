from sympy import false

from data.data_provider import data_provider
from exp.exp_basic import Exp_Basic
from models.my_method import dozerformer_Linear, dozerformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, process_one_batch
from utils.metrics import metric, MAPE, metric_g, truncate_to_dan

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
import warnings
import wandb
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args.device = self.device
        # self.norm_type = self.args.norm_type
        # self.dan_norm_type = args.dan_norm_type
        if args.load_pretrained_model:
            self._load_pretrain_model()

    def _build_model(self):
        model_dict = {
            'dozerformer_Linear': dozerformer_Linear,
            'dozerformer': dozerformer
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.99))
        return optimizer

    def _select_LR_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.args.train_epochs)
        return scheduler

    def _select_criterion(self):
        criterion = nn.L1Loss(reduction='mean') if self.args.loss == 'L1' else nn.MSELoss(reduction='mean')
        return criterion


        return loss_high + loss_low + loss_base
    #
    # def _compute_loss(self, preds, targets):
    #     criterion_1 = nn.L1Loss(reduction='mean')
    #     criterion_2 = nn.MSELoss(reduction='mean')
    #     loss = criterion_1(preds, targets) + criterion_2(preds, targets)
    #     return loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        scheduler = self._select_LR_scheduler(optimizer)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            self.model.train()

            train_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_label) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = process_one_batch(self.model, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_label, self.args)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()


            train_loss = np.average(train_loss)

            # Run validation and test
            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch)
            test_loss = self.vali(test_data, test_loader, criterion, epoch)
            # Save learning rate before update
            lr_current = scheduler.get_last_lr()[0]
            scheduler.step(epoch)
            self.model.epoch += 1

            # Wandb Save
            training_vals = {
                'Epoch': epoch,
                'Learning_Rate_realtime': lr_current,
                'train_loss': train_loss,
                'vali_loss': vali_loss,
                'test_loss': test_loss
            }
            wandb.log(training_vals) if self.args.wandb == True else None

            print("Epoch: {:3d}, Steps: {:3d}, cost time: {:5.2f} | Train Loss: {:5.4f} Vali Loss: {:5.4f} Test Loss: {:5.4f}".format(
                epoch + 1, train_steps, (time.time() - epoch_time), train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        return self.model

    def vali(self, vali_data, vali_loader, criterion, epoch=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_label) in enumerate(vali_loader):
                outputs, batch_y = process_one_batch(self.model, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_label,  self.args)

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                loss = criterion(outputs, batch_y)

                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        pred_raws = []
        true_raws = []
        pred_norms = []
        true_norms = []
        reservoir_datasets = {
            'Coyote', 'Lexington', 'Almaden', 'Stevens_Creek', 'Vasona',
        }

        watershed_datasets = {
            'Ross_noRain', 'Ross', 'Ross_S_fixed', 'SFC400',
            'Saratoga', 'Saratoga_S_fixed', 'Saratoga_noRain',
            'SFC', 'SFC_S_fixed', 'SFC_noRain',
            'UpperPen', 'UpperPen_S_fixed', 'UpperPen_noRain',
            'MTS_npy'
        }

        folder_path = './results/' + setting + '/'
        ext_path = 'ext_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            def _to_numpy(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().numpy()
                return np.asarray(x)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_label) in enumerate(test_loader):
                outputs, batch_y = process_one_batch(self.model, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_label, self.args)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # rmse_raw = np.sqrt(np.mean((outputs - batch_y) ** 2))
                pred_raw = test_data.inverse_transform(outputs)
                true_raw = test_data.inverse_transform(batch_y)
                # ← ADD: clip negative predictions (DAN style)
                if self.args.data in watershed_datasets:
                    pred_raw = np.clip(pred_raw, 0, None)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                pred_raws.append(pred_raw)
                true_raws.append(true_raw)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        pred_raws = np.concatenate(pred_raws, axis=0)
        true_raws = np.concatenate(true_raws, axis=0)

        # np.save(ext_path + self.args.data + '/pred_raws.npy', pred_raws)
        # np.save(ext_path +  self.args.data + '/true_raws.npy', true_raws)

        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)

        metric_dict = {
            'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape,
            'mspe': mspe, 'corr': corr,
        }

        extreme_metric_datasets = reservoir_datasets | watershed_datasets
        show_extended_metrics = self.args.data in extreme_metric_datasets

        if show_extended_metrics:
            if self.args.data in watershed_datasets:
                from utils.metrics_dan import compute_metrics_dan
                from utils.ext_dan import compute_metrics_same
                dan_metrics = compute_metrics_dan(pred_raws, true_raws, window_size=self.args.pred_len)
                # dan_metrics = compute_metrics_same(pred_raws.squeeze(-1))
                metric_dict = dan_metrics
                # metric_line = 'rmse_3d:{}, mape:{}'.format(np.array(dan_metrics[0][0]), np.array(dan_metrics[1][0]))
                metric_line = 'For 3d, rmse:{}, mape:{}, For 4h, rmse:{}, mape:{}'.format(dan_metrics['rmse_3d'], dan_metrics['mape_3d'],
                                                                                          dan_metrics['rmse_4h'], dan_metrics['mape_4h'])
            elif self.args.data in reservoir_datasets:
                from utils.metrics_dan import compute_metrics_reservoir

                reservoir_metrics = compute_metrics_reservoir(pred_raws, true_raws)
                metric_dict = reservoir_metrics
                metric_line = 'rmse:{}, mape:{}'.format(reservoir_metrics['rmse'], reservoir_metrics['mape'])

        else:
            metric_line = 'mse:{}, mae:{}'.format(mse, mae)

        wandb.log(metric_dict) if self.args.wandb == True else None

        print(metric_line)
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write(metric_line)
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return


