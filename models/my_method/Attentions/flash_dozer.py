import torch
import torch.nn as nn
from math import sqrt
from einops import repeat
from flash_sparse_attn import flash_sparse_attn_func_auto
from models.my_method.Attentions.attention_masking import SparseMask, show_mask


class DozerAttention(nn.Module):
    def __init__(self, local_window, stride, rand_rate, vary_len, pred_len,
                 in_channel, mask_flag=True, scale=None, mask='dozer',
                 attention_dropout=0.1, output_attention=False):
        super(DozerAttention, self).__init__()
        self.scale = scale
        self.local_window = local_window
        self.stride = stride
        self.rand_rate = rand_rate
        self.vary_len = vary_len
        self.mask_flag = mask_flag
        self.pred_len = pred_len
        self.in_channel = in_channel
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.mask = mask
        # self.register_buffer("flops_accum", torch.zeros(1))


    def forward(self, queries, keys, values, x_label, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        batch_size, _, _ = x_label.shape
        orig_dtype = queries.dtype

        scale = self.scale or 1. / sqrt(D)
        if L_Q == L_K:
            # mask types: 'extreme_mask', 'dozer', 'dozer_ext_only', 'dozer_ext_0', 'dozer_ext_null', 'dozer_AND_ext'

            sparse_mask = SparseMask(x_label, self.local_window, self.stride, queries.device, batch_size, L_Q, L_K)
            base_mask = sparse_mask.generate_mask(mask=self.mask)

            dozer_np = sparse_mask.visualize_mask(mask='dozer')
            masks = sparse_mask.visualize_mask(mask='all')
            extreme_np = masks['extreme_mask']
            # Normal query attend to Dozer keys, extreme query attend to dozer keys and extreme keys
            # This is the implementation of Sanjeev on Feb 19.
            dozer_ext_only = masks['dozer_ext_only']

            # Normal query attend keys from Dozer and Extreme after AND operator, key has to be True in both attention matrix to be select.
            # This is the implementation Yifan asked to implement on Feb 19.
            dozer_ext_0 = masks['dozer_ext_0']

            # Extreme query doesn't attend any keys. All False in rows (queries) with extreme label (1)
            dozer_ext_null = masks['dozer_ext_null']

            # This implementation apply AND operation to Dozer and Extreme Mask directly. Possibly for extreme query, there will be no key selected.
            # But, in usual, this has little differences with dozer_ext_only and dozer_ext_0
            dozer_AND_ext = masks['dozer_AND_ext']
            full_mask = masks['full_mask']
            dozer_ext_0_v1 = masks['dozer_ext_0_v1']
            dozer_v1 = masks['dozer_v1']
            adapt_dozer_mask = repeat(base_mask, 'b seg_num c -> (b ts_d) seg_num c', ts_d=self.in_channel)
            attn_mask = adapt_dozer_mask.unsqueeze(1).expand(-1, H, -1, -1)

        flash_sparse_attn_func = flash_sparse_attn_func_auto(backend="cuda")
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        queries = queries.to(target_dtype)
        keys    = keys.to(target_dtype)
        values  = values.to(target_dtype)

        # active = attn_mask.to(torch.bool).sum()
        # flops = 2 * H * active * D

        attn = flash_sparse_attn_func(
            query=queries,
            key=keys,
            value=values,
            attn_mask=attn_mask,  # bool, [B, H, L_Q, L_K]
            attn_bias=None,
            softmax_scale=scale,
        )
        # self.flops_accum = flops / 1e6

        attn = attn.to(orig_dtype)

        if self.output_attention:
            return attn, None
        return attn, None


class DozerAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(DozerAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, x_label, attn_mask):
        x = torch.clone(queries)
        # Batch size, Seq len, embed_dim
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Batch size, Seq len, head, embed_dim/head
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            x_label,
            attn_mask
        )

        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out, attn


