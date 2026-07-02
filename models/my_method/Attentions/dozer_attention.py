from math import sqrt
import torch.nn as nn
import torch
from models.my_method.Attentions.attention_masking import TriangularCausalMask
import numpy as np
from einops import einsum, rearrange, repeat
from flash_sparse_attn import flash_sparse_attn_func_auto
from flash_sparse_attn.utils.mask import create_mask

# class DozerAttention(nn.Module):
#     def __init__(self, local_window, stride, rand_rate, vary_len, pred_len, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
#         super(DozerAttention, self).__init__()
#         self.scale = scale
#
#         self.local_window = local_window
#         self.stride = stride
#         self.rand_rate = rand_rate
#         self.vary_len = vary_len
#         self.mask_flag = mask_flag
#         self.pred_len = pred_len
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)
#
#     def forward(self, queries, keys, values, attn_mask):
#         # Batch size, Seq len, Head, dim/head
#         B, L_Q, H, D = queries.shape
#         _, L_K, _, _ = keys.shape
#         scale = self.scale or 1. / sqrt(D)
#
#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)
#
#         sparse_mask = torch.zeros(L_Q, L_K, device=scores.device)
#         # Self Attention
#         if L_Q == L_K:
#             if self.local_window:
#                 for w_idx in range(self.local_window//2+1):
#                     sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
#                     sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)
#
#             if self.stride:
#                 stride = self.stride + 1
#                 for w_idx in range(0, L_Q, stride):
#                     sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
#                     sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)
#
#         # Cross Attention
#         if L_Q != L_K:
#             # 1. local
#             if self.local_window:
#                 local_window = self.local_window//2 if self.local_window>1 else self.local_window
#                 sparse_mask[:, -local_window:] = 1
#
#             # 2. Stride
#             if self.stride:
#                 start_index = L_K - L_Q//2
#                 stride = self.stride + 1
#                 for w_idx in range(start_index, L_K, stride):
#                     sparse_mask = torch.diagonal_scatter(sparse_mask,
#                                                          torch.ones(len(torch.diagonal(sparse_mask, w_idx))),
#                                                          w_idx)
#                 for w_idx in range(start_index, -L_K, -stride):
#                     sparse_mask = torch.diagonal_scatter(sparse_mask,
#                                                          torch.ones(len(torch.diagonal(sparse_mask, w_idx))),
#                                                          w_idx)
#
#             if self.vary_len or type(self.vary_len) is int:
#                 start_index = -self.pred_len+self.vary_len-1
#                 var_len_mask = torch.tril(torch.ones(L_Q, L_K, device=scores.device), diagonal=start_index)
#                 var_len_mask = torch.flip(var_len_mask, [1])
#                 sparse_mask = torch.where((sparse_mask + var_len_mask) >= 1, 1, 0)
#
#         scores = scores * sparse_mask
#
#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L_Q, device=queries.device)
#             # attn_mask is bool
#             scores.masked_fill_(attn_mask.mask, -np.inf)
#         b = scores[0, 0, :, :].detach().cpu().numpy()
#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)
#         if self.output_attention:
#             return (V.contiguous(), A)
#         else:
#             return (V.contiguous(), None)

class DozerAttention(nn.Module):
    def __init__(self, local_window, stride, rand_rate, vary_len, pred_len, in_channel, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
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

    def forward(self, queries, keys, values, x_label, attn_mask):
        # Batch size, Seq len, Head, dim/head
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        batch_size, _, _ = x_label.shape


        scale = self.scale or 1. / sqrt(D)
        dozer_mask = torch.zeros(L_Q, L_K, device=queries.device)

        adapt_mask = torch.zeros(batch_size, L_Q, L_K, device=queries.device)
        # Self Attention
        if L_Q == L_K:
            if self.local_window:
                for w_idx in range(self.local_window//2+1):
                    dozer_mask = torch.diagonal_scatter(dozer_mask, torch.ones(L_Q - w_idx), w_idx)
                    dozer_mask = torch.diagonal_scatter(dozer_mask, torch.ones(L_Q - w_idx), -w_idx)

            if self.stride:
                stride = self.stride + 1
                for w_idx in range(0, L_Q, stride):
                    dozer_mask = torch.diagonal_scatter(dozer_mask, torch.ones(L_Q - w_idx), w_idx)
                    dozer_mask = torch.diagonal_scatter(dozer_mask, torch.ones(L_Q - w_idx), -w_idx)

                # Self Attention
            # If there are more time steps than thereshold, we can define the entire patch as extreme.
            # patch_labels = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1,
            #                  0]
            # for i in range(len(patch_labels)):
            #     for j in range(len(patch_labels)):
            #         if patch_labels[i] == patch_labels[j]:
            #             dozer_mask[i, j] = 1
            #create sparse mask (batch, num_segments, num_segments) for e.g (32, 30, 30)
            #write one more loop that iterate over batch to create different sparse mask for each batch

            # extreme adaptive mask, content-aware, depend upon the labels
            for b_idx in range(batch_size):
                for i in range(L_Q):
                    for j in range(L_K):
                        if x_label[b_idx, i, 0] == x_label[b_idx, j, 0]:
                            adapt_mask[b_idx, i, j] = 1

        dozer_mask = dozer_mask.unsqueeze(0)  # shape (1, L_Q, l_K)
        # elementwise multiplication (AND operation)

        adapt_dozer_mask = adapt_mask * dozer_mask
        # a = adapt_dozer_mask[0].detach().cpu().numpy()
        # to match with x data points dimension, expand over (batch * channels)
        adapt_dozer_mask = repeat(adapt_dozer_mask, 'b seg_num c -> (b ts_d) seg_num c', ts_d=self.in_channel)



        scores = torch.zeros(B, H, L_Q, L_K).to(queries.device)
        # # TO-DO: revise to calculate different mask across different batch dimension
        for b in range(B):
            for i in range(L_Q):
                seleted_keys_idxs = rearrange(adapt_dozer_mask[b, i, :].nonzero(), 'dim1 dim2 -> (dim1 dim2)')
                scores[b:b+1, :, i:i+1, seleted_keys_idxs] = torch.einsum("blhe,bshe->bhls", queries[b:b+1, i:i+1, :, :], keys[b:b+1, seleted_keys_idxs, :, :])


        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L_Q, device=queries.device)
            # attn_mask is bool
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



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