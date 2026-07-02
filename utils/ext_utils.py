#!/usr/bin/env python
# encoding: utf-8
import os
import numpy as np
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import math

import matplotlib.pyplot as plt
from scipy.stats import norm, kruskal
from scipy.stats import skew, kurtosis

from torch.utils.data import TensorDataset

def initial_seed(seed: int = 10):
    """ Fix seed for random number generator. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def load_config(filepath):
    args = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)  # split on first '=' only
            args.append(f'--{key.strip()}')
            args.append(value.strip())
    return args



def r_log_std_normalization(sensor_data_val):
    data = sensor_data_val

    data1 = data[1:]
    data2 = [0 for _ in data1]
    for i in range(len(data) - 1):
        if data[i] > 0:
            data2[i] = data1[i] - data[i]
        else:

            data2[i] = (data1[i] + 1e-8) - (data[i] + 1e-8)
    data = data2

    c = np.array([1] + data)
    mean = np.nanmean(c)
    print("mean is: ", mean)
    std  = np.nanstd(c)
    print("std is ", std)
    c = (c - mean) / std

    mini = 0
    return c, mean, std, mini


def normalize_diff_with_stats(sensor_data_val, mean, std):
    """ Normalize the sensor data using a first-order difference and z-score normalization."""

    data = sensor_data_val
    # diff
    data1 = data[1:]
    data2 = [0 for i in data1]
    for i in range(len(data) - 1):
        if data[i] > 0:
            data2[i] = data1[i] - data[i]
        else:
            data2[i] = (data1[i] + 1e-8) - (data[i] + 1e-8)
    data = data2

    c = np.array([1] + data)

    # norm
    c = (c - mean) / std
    return c


def r_log_std_denorm_dataset(mean, std, predict_y0, y_pre):

    # de-norm
    a2 = predict_y0
    a2 = [ii * std + mean for ii in a2]
    a3 = np.zeros(len(a2))
    a3[0] = a2[0] + y_pre
    for ii in range(len(a2) - 1):
        a3[ii + 1] = a3[ii] + a2[ii + 1]
    return a3


def std_denorm_dataset(predict_y0, pre_y, mean, std):

    a2 = r_log_std_denorm_dataset(mean, std, 0, predict_y0, pre_y)

    return a2


def log_std_normalization(sensor_data_val):
    a = np.log(np.array(sensor_data_val) + 1)
    c = a
    mean = np.nanmean(c)
    # print("mean is: ", mean)
    std = np.nanstd(c)
    # print("std is ", std)
    c = (c - mean) / std

    return c, mean, std

def log_std_normalization_with_stats(sensor_data_val, mean=None, std=None):


    c = np.log(np.array(sensor_data_val) + 1)
    c = (c - mean) / std

    return c

def log_std_denorm_dataset(mean, std, predict_y0):
    a2 = predict_y0
    a2 = [ii * std + mean for ii in a2]
    a3 = a2
    a3 = [((np.e) ** ii) - 1 for ii in a3]

    return np.array(a3)

# def standard_normalization(x):
#
#     x = np.array(x)
#     mean = np.mean(x)
#     std = np.std(x)
#     x_norm = (x - mean) / std
#     return x_norm, mean, std


def standard_normalization(x):
    x = np.array(x)

    mean = np.nanmean(x)
    std = np.nanstd(x)
    x_norm = (x - mean) / std

    return x_norm, mean, std

# def standard_denormalization(x_norm, mean, std):
#
#     x = x_norm * std + mean
#     return x

def standard_denormalization(norm_data, mean, std):
    norm_data = np.array(norm_data)
    return norm_data * std + mean


