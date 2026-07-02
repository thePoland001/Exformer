import argparse
import os
import torch
from exp.exp_main_finetune import Exp_Main
import random
import numpy as np
import os
import sys
import time
import yaml
from utils.tools import string_split
import wandb
from utils.ext_utils import load_config

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

def main():
    def str2bool(v):
        return str(v).lower() in ('true', '1', 'yes')
    parser = argparse.ArgumentParser(description='Dozerformer')
    parser.add_argument('--arg_file', type=str, default=None, help='Path to YAML config')
    parser.add_argument('--mode', default='finetune', type=str, help='Name of model to train, options: [pretrain, finetune, Transformer]')
    parser.add_argument('--data', type=str, required=False, default='Ross_noRain',
                        help='name of dataset')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='dozerformer_Linear',
                        help='model name, options: [tsformer, Informer, Transformer]')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Data parameters
    parser.add_argument('--root_path', type=str, default='./data/datasets/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1_labeled.csv', help='location of the data file')
    parser.add_argument('--seq_len', type=int, default=1440, help='input sequence length for encoder, look back window')
    parser.add_argument('--label_len', type=int, default=96, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=288, help='prediction sequence length, horizon')
    parser.add_argument('--features', type=str, default='M', choices=['S', 'M'],
                        help='features S is univariate, M is multivariate')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--data_split', type=str, default='0.6,0.2,0.2',
                        help='train/val/test split, can be ratio or number')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')

    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=8, help='encoder input size')
    parser.add_argument('--decoder_embed_dim', type=int, default=8, help='encoder input size')
    parser.add_argument('--n_heads', type=int, default=4, help='number of multihead attention')
    parser.add_argument('--encoder_depth', type=int, default=2, help='The layers of transformer encoder')
    parser.add_argument('--decoder_depth', type=int, default=1, help='The layers of transformer decoder')
    # Those transformers used this in encoder and decoder as the parameter for conv1d, MLP
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of MLP in transformer')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_size', type=int, default=8, help='patch sizes of hierarchical architecture')
    parser.add_argument('--moving_avg', type=str, default='13, 17', help='window size of moving average')

    # CNN parameters
    parser.add_argument('--CNN_embed_dim', type=int, default=8, help='CNN trend model embedding dimension')

    # DozerAttention parameters
    parser.add_argument('--local_window', type=int, default=2, help='The size of local window')
    parser.add_argument('--stride', type=int, default=4, help='The stride interval sparse attention. If set to 24, interval will be 24.')
    parser.add_argument('--rand_rate', type=int, default=0.1, help='The rate of random attention')

    # Ablation paprameters
    parser.add_argument('--vary_len', type=int, default=1, help='The start varying length, if 1 input equals output')
    parser.add_argument('--attn', type=str, default='dozer', help='model supports different attention mechanism [prob, AutoCorr, FedAttn]')
    parser.add_argument('--factor', type=int, default=1, help='attn factor. Autoformer')
    parser.add_argument('--Fedformer_version', type=str, default='None', help='Fouriers, Wavelets')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=48, help='batch size of train input data')
    parser.add_argument('--seed', type=int, default=2023, help='Random Seed')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--loss', type=str, default='L1', help='dropout')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0', help='multiple gpu')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='optimizer learning rate')
    parser.add_argument('--lradj', type=int, default=3, help='adjust learning rate')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--load_pretrained_model', type=bool, default=False, help='flag for wether load encoder from pretrained model')

    parser.add_argument('--wandb', type=str2bool, default=True,
                        help='flag for whether use wandb')
    parser.add_argument('--abla_type', type=str, default='False', help='ablation study type')

    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--cycle', type=int, default=24, help='cycle length')
    parser.add_argument('--norm_type', type=str, default='std', choices=['all', 'ori', 'std'],
                        help='channel selection mode for MTS_npy dataset')
    parser.add_argument('--dan_norm_type', type=str, default='log-std', choices=['log-std', 'std', 'ori'],
                        help='normalization mode for Dan watershed datasets')
    parser.add_argument('--merge_to_series', type=str2bool, default=False,
                        help='flatten (N,T,C)->(N*T,C) and use sliding windows for MTS_npy')
    # Dan watershed date/sampling split options.
    parser.add_argument('--start_point', type=str, default=None, help='start time for training timeline')
    parser.add_argument('--train_point', type=str, default=None, help='end time for training timeline')
    parser.add_argument('--test_start', type=str, default=None, help='start time for test timeline')
    parser.add_argument('--test_end', type=str, default=None, help='end time for test timeline')
    parser.add_argument('--train_seed', type=int, default=1010, help='seed for train window sampling')
    parser.add_argument('--val_seed', type=int, default=2007, help='seed for val window sampling')
    parser.add_argument('--train_volume', type=int, default=30000, help='number of sampled train windows')
    parser.add_argument('--val_size', type=int, default=120, help='number of sampled val windows')
    parser.add_argument('--test_stride', type=int, default=16, help='test window stride')
    parser.add_argument('--rain_data_path', type=str, default=None, help='rain dataset file for Dan watershed loader')
    parser.add_argument('--watershed', type=int, default=1, help='1: use rain signal, 0: use GMM outlier indicator')
    parser.add_argument('--oversampling', type=float, default=80, help='Kruskal H threshold for Dan train sampling')
    parser.add_argument('--event_focus_level', type=int, default=18, help='random acceptance percent when H threshold not met')


    #fusion
    parser.add_argument('--fusion', type=str, default='SUM', help='[SUM, EIA, ADT]')
    parser.add_argument('--u_size', type=int, default=2, help='u')
    parser.add_argument('--f_version', type=str, default="sk_v2", help='u')
    parser.add_argument('--anorm_thres', type=float, default=0.5, help='threshold for anomaly score when using outlier detection')

    parser.add_argument('--mask', type=str, default='dozer', help='type of sparse mask')

    parser.add_argument('--exp_run', type=str, default='saratoga_withrain', help='identifier for experiments')
    parser.add_argument('--patch_thres', type=int, default=1, help='type of sparse mask')
    parser.add_argument('--notes', type=str, default='', help='notes for the experiment')

    args = parser.parse_args()

    if args.arg_file is not None:
        with open(args.arg_file, "r") as f:
            yaml_cfg = yaml.safe_load(f)

        for key, value in yaml_cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)

    watershed_datasets = {
        'Ross_noRain', 'Ross', 'SFC400',
         'Saratoga',
        'SFC', 'SFC_noRain',
         'UpperPen', 'UpperPen_noRain',
    }
    no_rain_datasets = {'Ross_noRain', 'Saratoga_noRain', 'SFC_noRain', 'UpperPen_noRain'}

    if args.data in watershed_datasets:
        if args.start_point is None:
            args.start_point = '1988-01-01 14:30:00'
        if args.train_point is None:
            args.train_point = '2021-08-31 23:30:00'
        if args.test_start is None:
            args.test_start = '2021-09-01 00:30:00'
        if args.test_end is None:
            args.test_end = '2022-05-31 23:30:00'
    if args.data in no_rain_datasets:
        args.watershed = 0

    # fix the seed for reproducibility, default 2023
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    args.moving_avg = [int(i) for i in args.moving_avg.split(', ')]
    args.lradj = 'CosineAnnealing'

    # Model's parameters
    args.decoder_embed_dim = args.embed_dim
    args.d_ff = args.embed_dim*args.n_heads if args.d_ff == None else args.d_ff
    args.output_attention = False

    # FedAttn (FourierBlock) parameters - only set when needed
    if args.attn == 'FedAttn':
        args.modes = getattr(args, 'modes', 64)
        args.mode_select = getattr(args, 'mode_select', 'random')

    # args.wandb = False

    # Optimization's parameters
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # Multi gpu
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'ETTh1_labeled': {'data': 'ETT-data/ETTh1_labeled.csv', 'data_dim': 7, 'split': [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24]},
        'ETTh2_labeled': {'data': 'ETT-data/ETTh2_labeled.csv', 'data_dim': 7,
                          'split': [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24]},
        'ETTh1': {'data': 'ETT-data/ETTh1.csv', 'data_dim': 7, 'split': [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24]},
        'ETTh2': {'data': 'ETT-data/ETTh2.csv', 'data_dim': 7, 'split': [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24]},
        'ETTm1': {'data': 'ETT-data/ETTm1.csv', 'data_dim': 7, 'split': [4 * 12 * 30 * 24, 4 * 4 * 30 * 24, 4 * 4 * 30 * 24]},
        'ETTm2': {'data': 'ETT-data/ETTm2.csv', 'data_dim': 7, 'split': [4 * 12 * 30 * 24, 4 * 4 * 30 * 24, 4 * 4 * 30 * 24]},
        'ETTm1_labeled': {'data': 'ETT-data/ETTm1_labeled.csv', 'data_dim': 7,
                  'split': [4 * 12 * 30 * 24, 4 * 4 * 30 * 24, 4 * 4 * 30 * 24]},
        'ETTm2_labeled': {'data': 'ETT-data/ETTm2_labeled.csv', 'data_dim': 7,
                  'split': [4 * 12 * 30 * 24, 4 * 4 * 30 * 24, 4 * 4 * 30 * 24]},

        'electricity': {'data': 'electricity/electricity.csv', 'data_dim': 321, 'split': [0.7, 0.1, 0.2]},
        'electricity_labeled': {'data': 'electricity/electricity_labeled.csv', 'data_dim': 321, 'split': [0.7, 0.1, 0.2]},
        'Weather': {'data': 'weather/weather.csv', 'data_dim': 21, 'split': [0.7, 0.1, 0.2]},
        'Weather_labeled': {'data': 'weather/weather_labeled.csv', 'data_dim': 21, 'split': [0.7, 0.1, 0.2]},
        'ILI': {'data': 'national_illness.csv', 'data_dim': 7, 'split': [0.7, 0.1, 0.2]},
        'Traffic': {'data': 'STEE/traffic.csv', 'data_dim': 862, 'split': [0.7, 0.1, 0.2]},
        'Exchange': {'data': 'exchange_rate/exchange_rate.csv', 'data_dim': 8, 'split': [0.7, 0.1, 0.2]},
        'Exchange_labeled': {'data': 'exchange_rate/exchange_rate_labeled.csv', 'data_dim': 8, 'split': [0.7, 0.1, 0.2]},
        'Ross_noRain': {'data': 'watershed/Ross_noRain', 'data_dim': 1, 'split': [0.7, 0.1, 0.2]},
        'Ross': {'data': 'watershed/Ross_withRain', 'data_dim': 2, 'split': [0.7, 0.1, 0.2]},
        'Saratoga': {'data': 'watershed/Saratoga_withRain', 'data_dim': 2, 'split': [0.7, 0.1, 0.2]},
        'Saratoga_noRain': {'data': 'watershed/Saratoga_S_fixed.csv', 'data_dim': 1, 'split': [0.7, 0.1, 0.2]},
        'SFC': {'data': 'watershed/SFC_withRain', 'data_dim': 2, 'split': [0.7, 0.1, 0.2]},
        'SFC400': {'data': 'watershed/SFC400', 'data_dim': 2, 'split': [0.7, 0.1, 0.2]},
        'SFC_noRain': {'data': 'watershed/SFC_S_fixed.csv', 'data_dim': 1, 'split': [0.7, 0.1, 0.2]},
        'UpperPen': {'data': 'watershed/UpperPen_withRain', 'data_dim': 2, 'split': [0.7, 0.1, 0.2]},
        'UpperPen_noRain': {'data': 'watershed/UpperPen_S_fixed.csv', 'data_dim': 1, 'split': [0.7, 0.1, 0.2]},
        'Coyote': {'data': 'reservoir/Coyote/in360_out72_ro8', 'data_dim': 1, 'split': [0.7, 0.1, 0.2]},
        'Lexington': {'data': 'reservoir/Lexington/in360_out72_ro8', 'data_dim': 1, 'split': [0.7, 0.1, 0.2]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.data_dim = data_info['data_dim']
        args.data_split = data_info['split']
    else:
        args.data_split = string_split(args.data_split)

    if args.wandb==True:
        wandb.login()
        wandb.init(project="e-Dozerformer", config=args)

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_segl{}_dm{}_nh{}_el{}_dl{}_mask_{}'.format(
                args.mode,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.patch_size,
                args.embed_dim,
                args.n_heads,
                args.encoder_depth,
                args.decoder_depth,
                args.mask,
                )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_segl{}_dm{}_nh{}_el{}_dl{}_mask_{}'.format(
            args.mode,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.patch_size,
            args.embed_dim,
            args.n_heads,
            args.encoder_depth,
            args.decoder_depth,
            args.mask,
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    print('Finished Training')
