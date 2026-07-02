import torch
from torch import nn
from einops import rearrange

from models.my_method.dozerformer_EncDec import dozerformer_Encoder

from models.REVIN import RevIN
from models.my_method.build_model_util import series_decomp_multi, series_decomp_multi_learnable

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mode = configs.mode
        assert self.mode in ["pretrain", 'finetune', "forecasting"], "Error mode."
        self.patch_size = configs.patch_size
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.in_channel = configs.data_dim
        self.embed_dim = configs.embed_dim
        # self.mask_ratio = configs.mask_ratio
        self.encoder_depth = configs.encoder_depth
        self.decoder_depth = configs.decoder_depth
        self.decoder_embed_dim = configs.decoder_embed_dim
        self.dropout = configs.dropout
        self.epoch = 0
        self.fusion = configs.fusion
        # self.d_model = 256
        configs.activation = 'gelu'
        self.watershed = configs.watershed

        self.revin_layer = RevIN(self.in_channel, affine=True, subtract_last=False)

        # Decomposition
        self.decomp_multi = series_decomp_multi(configs.moving_avg)
        # Seasonal encoder and decoder
        self.encoder_seasonal = dozerformer_Encoder(configs, mode='Seasonal')

        self.output_layer_2 = nn.Conv2d(in_channels=self.embed_dim,
                                      out_channels=1,
                                      kernel_size=(1, 1))
        self.output_layer_1 = nn.Linear(configs.seq_len, configs.pred_len)
        self.trend_model = nn.Linear(configs.seq_len, configs.pred_len)

        if self.fusion == 'EIA':
            self.attention_mlp = nn.Sequential(
                nn.Linear(self.in_channel * 2, self.in_channel),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.in_channel, self.in_channel),
                nn.Sigmoid()
            )
            self._init_eia_weights()


    def _init_eia_weights(self):
        for layer in self.attention_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x_enc, x_mark_enc, seq_y_mark, x_dec, x_label, phase,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
                ) -> torch.tensor:

        x_norm = self.revin_layer(x_enc, 'norm')

        x_enc, trend_enc = self.decomp_multi(x_norm)

        # Encoder
        encoder_output = self.encoder_seasonal(x_enc, x_label, x_mark_enc, phase)

        encoder_output = self.encoder_seasonal.encoder_segment.concat(encoder_output)
        encoder_output = rearrange(encoder_output, 'b emb seq_len ts_d -> b emb ts_d seq_len')

        seasonal_predict = self.output_layer_1(encoder_output)
        seasonal_predict = self.output_layer_2(seasonal_predict)
        seasonal_predict = rearrange(seasonal_predict, 'b 1 ts_d seq_len -> b seq_len ts_d')

        # # Trend
        trend_enc = rearrange(trend_enc, 'b seq_len ts_d -> b ts_d seq_len')
        trend_predict = self.trend_model(trend_enc)
        trend_predict = rearrange(trend_predict, 'b ts_d seq_len -> b seq_len ts_d')

        # Concate Trend and Seasonal
        final_predict = seasonal_predict + trend_predict

        # Inverse Revin
        final_predict = self.revin_layer(final_predict, 'denorm')  # (b, pred_len, 2)
        if self.watershed:
            final_predict = final_predict[:, :, 0:1]  # (b, pred_len, 1) stream only
        return final_predict

