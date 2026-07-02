from models.my_method.Attentions.attn_type import ProbAttention, AutoCorrelation, FourierBlock
from models.my_method.Attentions.flash_dozer import DozerAttention


def build_prob(configs, **kwargs):
    return ProbAttention(
        False, factor=5,
        attention_dropout=configs.dropout,
        output_attention=configs.output_attention
    )

def build_dozer(configs, **kwargs):
    return DozerAttention(
        configs.local_window, configs.stride, configs.rand_rate,
        configs.vary_len, kwargs['encoder_segment'].seg_num, kwargs['in_channel'],
        False, mask=configs.mask,
        attention_dropout=configs.dropout,
        output_attention=configs.output_attention
    )



def build_autocorrelation(configs, **kwargs):
    return AutoCorrelation(
        False, factor=1,
        attention_dropout=configs.dropout,
        output_attention=configs.output_attention
    )


def build_fourier(configs, **kwargs):
    seg_num = configs.seq_len // configs.patch_size
    return FourierBlock(
        in_channels=configs.embed_dim * configs.patch_size,
        out_channels=configs.embed_dim * configs.patch_size,
        seq_len=seg_num,
        n_heads=configs.n_heads,
        modes=configs.modes,
        mode_select_method=configs.mode_select
    )


ATTENTION_MAP = {
    'prob': build_prob,
    'dozer': build_dozer,
    'AutoCorr': build_autocorrelation,
    'FedAttn': build_fourier,
}