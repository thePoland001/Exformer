import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def save_model(epoch, lr, model, model_dir, model_name='pems08', horizon=12):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_name = os.path.join(model_dir, model_name+str(horizon)+'.bin')
    torch.save(
        {
        'epoch': epoch,
        'lr': lr,
        'model': model.state_dict(),
        }, file_name)
    print('save model in ',file_name)


def load_model(model, model_dir, model_name='pems08', horizon=12):
    if not model_dir:
        return
    file_name = os.path.join(model_dir, model_name+str(horizon)+'.bin') 

    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        print('loaded the model...', file_name, 'now lr:', lr, 'now epoch:', epoch)
    return model, lr, epoch

# No warm up.
# 这样学习率下降的慢，加一个除2的
def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj==1:
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj==2:
        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10:0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }
    elif args.lradj==3:
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj==4:
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    # Fixed learning rate
    else:
        lr_adjust = {}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
    return lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def test_draw(tensor):
    import matplotlib.pyplot as plt

    size = tensor.size()
    if len(size) == 3:
        tensor = tensor[0, :, :].detach().cpu().numpy()
    if len(size) == 4:
        tensor = tensor[0, 0, :, :].detach().cpu().numpy()

    plt.figure()
    for i in range(1, 5):
        plt.plot(tensor[-i, :], label='feature -{}'.format(i), linewidth=2)
    plt.legend()
    plt.title('test')
    plt.show()

def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list

# def get_statistical(file_path):
#     statistics_data = torch.load(os.path.join(file_path, "mean_std_mini.pt"))
#
#     train_diff_mean = statistics_data['diff_mean']
#     train_diff_std = statistics_data['diff_std']
#     train_min = statistics_data['mini']
#     train_mean = statistics_data['stdn_mean']
#     train_std = statistics_data['stdn_std']
#
#     return train_diff_mean, train_diff_std, train_min, train_mean, train_std

def get_statistical(file_path):

    stats_path = os.path.join(file_path, "mean_std_mini.pt")
    try:
        statistics_data = torch.load(stats_path, map_location='cpu', weights_only=False)
    except TypeError:
        # for older torch versions that don't support weights_only
        statistics_data = torch.load(stats_path, map_location='cpu')

    train_diff_mean = statistics_data['diff_mean']
    train_diff_std = statistics_data['diff_std']
    train_min = statistics_data['mini']
    train_mean = statistics_data['stdn_mean']
    train_std = statistics_data['stdn_std']

    return train_diff_mean, train_diff_std, train_min, train_mean, train_std


def get_statistical_dan(file_path, norm_type):
    stats_path = os.path.join(file_path, "mean_std_mini.pt")
    try:
        statistics_data = torch.load(stats_path, map_location='cpu', weights_only=False)
    except TypeError:
        # for older torch versions that don't support weights_only
        statistics_data = torch.load(stats_path, map_location='cpu')

    if norm_type == 'std':
        train_mean = statistics_data['stdn_mean']
        train_std = statistics_data['stdn_std']
    elif norm_type == 'log-std':
        train_mean = statistics_data['mean']
        train_std = statistics_data['std']

    return train_mean, train_std

from fvcore.nn import FlopCountAnalysis

# def Cal_FLOPs(model, batch_x, dec_inp, batch_label):
#   # model = model.cuda()
#
#
#   # print(model)
#
#   # inputs = torch.randn(32, 100, 55).cuda()
#
#   flops = FlopCountAnalysis(model, (batch_x, dec_inp, batch_label))
#   # print("unsupported operations: ", unsupported_operations)
#
#   n_param = sum([p.nelement() for p in model.parameters()])
#
#   print(f'FLOPs: {flops.total() / 1e9:.4f} GFLOPs')
#   print("FLOPs by operator:", flops.by_operator())
#
#   print(f'Params:{n_param}')
#   return 0

from thop import profile
import torch

def Cal_FLOPs(model, batch_x, dec_inp, batch_label):

    # # Example inputs (ensure correct shapes and device)
    # batch_x = torch.randn(32, 720, 7).cuda()       # [batch, seq_len, features]
    # dec_inp = torch.randn(32, 192, 7).cuda()       # [batch, pred_len, features]
    # batch_label = torch.randn(32, 720, 1).cuda()   # target or labels
    #
    # # Ensure model is on the same device
    # model = model.cuda()
    #
    # # thop expects tuple of all inputs
    # inputs = (batch_x, dec_inp, batch_label)
    #
    # # Compute FLOPs and parameters
    # flops, params = profile(model, inputs=inputs)
    #
    # # Convert to readable units
    # print(f"FLOPs: {flops}, Params: {params / 1e6:.3f} M")
    return 0

# attn_mask: bool tensor, True = attend, False = don't attend
# output attn_bias: same dtype/device as q, with 0 for allowed, big negative for blocked
def mask_to_bias(attn_mask: torch.Tensor, q: torch.Tensor, neg_val: float | None = None):
    attn_mask = attn_mask.to(device=q.device)

    if neg_val is None:
        # safe default for fp16/bf16 kernels (avoid -inf sometimes)
        neg_val = -1e4 if q.dtype == torch.float16 else -1e9

    attn_bias = torch.zeros_like(attn_mask, dtype=q.dtype, device=q.device)
    attn_bias = attn_bias.masked_fill(~attn_mask, neg_val)
    return attn_bias




def process_one_batch(model, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_label, args):
    batch_x = batch_x.float().to(args.device)
    batch_y = batch_y.float().to(args.device)
    batch_label = batch_label.float().to(args.device)
    batch_cycle = batch_cycle.int().to(args.device)

    # decoder input
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)
    Cal_FLOPs(model, batch_x, dec_inp, batch_label)

    # encoder - decoder
    if args.use_amp:
        with torch.cuda.amp.autocast():
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, batch_y_mark, dec_inp, batch_label, batch_cycle)[0]
            else:
                outputs = model(batch_x, batch_x_mark, batch_y_mark, dec_inp, batch_label, batch_cycle)
    else:
        if args.output_attention:
            outputs, attns = model(batch_x, batch_x_mark, batch_y_mark, dec_inp, batch_label, batch_cycle)
        else:
            outputs = model(batch_x, batch_x_mark, batch_y_mark, dec_inp, batch_label, batch_cycle)

        f_dim = -1 if args.features == 'S' else 0

        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:]

    return outputs, batch_y