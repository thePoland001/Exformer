import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred = np.squeeze(pred)
    true = np.squeeze(true)
    return mean_absolute_percentage_error(np.array(true) + 1, np.array(pred) + 1)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric_g(pred, true, window_size=288, short_window=16):
    pred = np.array(pred)
    true = np.array(true)
    ll = int(len(pred) / window_size)
    rmse_all1, mape_all1 = [], []
    rmse_all2, mape_all2 = [], []
    for i in range(ll):
        p_full = pred[i * window_size: (i + 1) * window_size]
        g_full = true[i * window_size: (i + 1) * window_size]
        _, _, rmse1, mape1 = _metric_per_window(p_full, g_full)
        rmse_all1.append(rmse1)
        mape_all1.append(mape1)

        p_short = pred[i * window_size: i * window_size + short_window]
        g_short = true[i * window_size: i * window_size + short_window]
        _, _, rmse2, mape2 = _metric_per_window(p_short, g_short)
        rmse_all2.append(rmse2)
        mape_all2.append(mape2)

    rmse1 = np.around(np.mean(rmse_all1), 2)
    mape1 = np.around(np.mean(mape_all1), 3)
    rmse2 = np.around(np.mean(rmse_all2), 2)
    mape2 = np.around(np.mean(mape_all2), 3)
    return rmse1, mape1, rmse2, mape2


def compute_metrics_dan(pred_raws, true_raws, window_size=288):
    pred_flat, true_flat = truncate_to_dan(pred_raws, true_raws, window_size)
    rmse_3d, mape_3d, rmse_4h, mape_4h = metric_g(pred_flat, true_flat, window_size)

    return {
        'rmse_3d': rmse_3d,
        'mape_3d': mape_3d,
        'rmse_4h': rmse_4h,
        'mape_4h': mape_4h,
    }

def metric_rolling_reservoir(pred, true, rm=16, inter=72):
    pred = np.array(pred)
    true = np.array(true)
    ll = int(len(pred) / inter)
    pre_all = []
    gt_all = []
    for i in range(ll):
        start = i * inter
        end = start + rm
        if end <= len(pred):
            pre_all.extend(pred[start:end])
            gt_all.extend(true[start:end])
    _, _, rmse, mape = _metric_per_window(np.array(pre_all), np.array(gt_all))
    rmse = np.around(rmse, 2)
    mape = np.around(mape, 3)
    return rmse, mape

def metric_rolling_reservoir1(pred, true, rm=16, inter=72):
    pred = np.array(pred)
    true = np.array(true)
    ll = int(len(pred) / inter)
    rmse_all = []
    mape_all = []
    for i in range(ll):
        start = i * inter
        end = start + rm
        if end <= len(pred):
            _, _, rmse, mape = _metric_per_window(pred[start:end], true[start:end])
            rmse_all.append(rmse)
            mape_all.append(mape)
    rmse = np.around(np.mean(np.array(rmse_all)), 2)
    mape = np.around(np.mean(np.array(mape_all)), 3)
    return rmse, mape

def compute_metrics_reservoir(pred_flat, true_flat, window_size=72):
    pred_flat, true_flat = truncate_to_dan(pred_flat, true_flat, window_size)

    rmse_roll_full, mape_roll_full = metric_rolling_reservoir(
        pred_flat, true_flat, rm=window_size, inter=window_size
    )
    rmse_roll_8h, mape_roll_8h = metric_rolling_reservoir(
        pred_flat, true_flat, rm=8, inter=window_size
    )

    return {
        'rmse': rmse_roll_full,
        'mape': mape_roll_full,
        'rmse_8h': rmse_roll_8h,
        'mape_8h': mape_roll_8h,
    }

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true)
    return mae, mse, rmse, mape, mspe, corr


def _metric_per_window(pred, true):
    """DAN's metric() — per-window stats (exact copy from DAN metric.py)"""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    return mae, mse, rmse, mape


# def metric_g(pred, true, window_size=288):
#     """DAN's metric_g — exact replication.
#
#     Splits pred/true into windows of window_size,
#     computes RMSE and MAPE per window, then averages.
#
#     Args:
#         pred: flattened predictions in RAW space, shape (N*window_size,)
#         true: flattened ground truth in RAW space, shape (N*window_size,)
#         window_size: prediction length (default 288 = 3 days of 15-min)
#
#     Returns:
#         (rmse, mape): averaged across windows, rounded to 2 and 3 decimals
#     """
#     pred = np.array(pred)
#     true = np.array(true)
#     ll = int(len(pred) / window_size)
#     rmse_all = []
#     mape_all = []
#     for i in range(ll):
#         p = pred[i * window_size: (i + 1) * window_size]
#         g = true[i * window_size: (i + 1) * window_size]
#         mae, mse, rmse, mape = _metric_per_window(p, g)
#         rmse_all.append(rmse)
#         mape_all.append(mape)
#     return np.around(np.mean(np.array(rmse_all)), 2), np.around(np.mean(np.array(mape_all)), 3)


def truncate_to_dan(pred, true, window_size):
    """DAN's compute_metrics truncation — round down to nearest 100 windows.

    DAN does: if ind >= count - count % 100: break
    This drops the last (count % 100) windows.

    Args:
        pred: array shape (N, window_size, C) or (N*window_size,)
        true: array shape (N, window_size, C) or (N*window_size,)
        window_size: prediction length (default 288)

    Returns:
        (pred_flat, true_flat): truncated and flattened
    """
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    n_windows = len(pred_flat) // window_size
    n_use = n_windows - n_windows % 100
    n_values = n_use * window_size
    return pred_flat[:n_values], true_flat[:n_values]



# def metric_rolling(pre, gt):
#     pre = np.array(pre)
#     gt = np.array(gt)
#     ll = int(len(pre)/288)
#     rmse_all1 = []
#     mape_all1 = []
#     rmse_all2 = []
#     mape_all2 = []
#     for i in range(ll):
#         _, _, rmse1, mape1 = metric(pre[i*288:(i*288+288)], gt[i*288:(i*288+288)])
#         _, _, rmse2, mape2 = metric(pre[i*288:(i*288+16)], gt[i*288:(i*288+16)])
#         rmse_all1.append(rmse1)
#         mape_all1.append(mape1)
#         rmse_all2.append(rmse2)
#         mape_all2.append(mape2)
#     rmse1 = np.around(np.mean(np.array(rmse_all1)),2)
#     mape1 = np.around(np.mean(np.array(mape_all1)),3)
#     print("For rolling prediction: 3 days")
#     print("RMSE: ", rmse1)
#     print("MAPE: ", mape1)
#     rmse2 = np.around(np.mean(np.array(rmse_all2)),2)
#     mape2 = np.around(np.mean(np.array(mape_all2)),3)
#     print("For rolling prediction: 4 hours")
#     print("RMSE: ", rmse2)
#     print("MAPE: ", mape2)
#     return rmse1, mape1, rmse2, mape2
#
# def compute_metrics_dan(pred_raws, true_raws, window_size=288):
#     """Full DAN compute_metrics pipeline — single call.
#
#     1. Truncate to nearest 100 windows
#     2. Compute per-window RMSE and MAPE then average
#
#     Args:
#         pred_raws: denormalized predictions, shape (N, window_size, C)
#         true_raws: denormalized ground truth, shape (N, window_size, C)
#         window_size: prediction length (default 288)
#
#     Returns:
#         dict with 'rmse' and 'mape'
#     """
#     # Step 2: truncate to nearest 100 windows
#     pred_flat, true_flat = truncate_to_dan(pred_raws, true_raws, window_size)
#     # # 3-day (full window)
#     # rmse, mape = metric_g(pred_flat, true_flat, window_size)
#
#     # 4-hour (short window)
#     rmse_3d, mape_3d, rmse_4h, mape_4h = metric_rolling(pred_flat, true_flat)
#
#     return {
#         'rmse_3day': rmse_3d,
#         'mape_3day': mape_3d,
#         'rmse_4hour': rmse_4h,
#         'mape_4hour': mape_4h,
#     }