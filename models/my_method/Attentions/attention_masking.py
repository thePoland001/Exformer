import torch
import matplotlib.pyplot as plt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class ExtremeMask():
    """
    Builds a content-aware mask where attention is allowed only between tokens
    that contains the same label.
    """
    def __init__(self, x_label, device=None):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0]               # (B, L)
            mask = labels.unsqueeze(2).eq(labels.unsqueeze(1))  # (B, L, L) bool
            self._mask = mask.to(device) # (B,L,L)

    @property
    def mask(self):
        return self._mask


class DozerExtremeOnlyMask():
    """
    Combines (OR operator) an extremes-only content mask with a structural dozer mask.
    """
    def __init__(self, x_label, dozer_mask, device=None, extreme_value=1):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0].to(torch.long)      # (B, L)
            extreme = (labels == extreme_value)          # (B, L) bool
            extreme_mask = extreme.unsqueeze(2) & extreme.unsqueeze(1)  # (B,L,L)

            # normalize dozer_mask to (B,L,L)
            if dozer_mask.dim() == 2:                    # (L,L)
                dozer_b = dozer_mask.unsqueeze(0).expand(extreme_mask.size(0), -1, -1)
            else:
                raise ValueError(f"Unexpected dozer_mask shape: {dozer_mask.shape}. Both mask should be of the same shape")

            base = extreme_mask | dozer_b                # (B,L,L)
            self._mask = base.to(device)   # (B,L,L)

    @property
    def mask(self):
        return self._mask


class ExtremeDozerSparseMask:
    """
    Sparse mask:
        1) extreme_mask & dozer_mask
        2) remove query rows where label == 1
    """
    def __init__(self, extreme_mask, dozer_mask, x_label, device=None):

        if device is None:
            device = x_label.device

        with torch.no_grad():
            B, L, _ = extreme_mask.shape

            # Normalize dozer mask
            if dozer_mask.dim() == 2:  # (L, L)
                dozer_mask = dozer_mask.unsqueeze(0).expand(B, -1, -1)

            # Step 1: structural AND
            combined_mask = extreme_mask & dozer_mask  # (B,L,L)

            # Step 2: remove query rows where label == 1
            query_keep = (x_label.squeeze(-1) == 0).unsqueeze(-1)  # (B,L,1)
            combined_mask = combined_mask & query_keep  # broadcast

            self._mask = combined_mask.to(device)  # (B,1,L,L)

    @property
    def mask(self):
        return self._mask

class ConditionalExtremeDozerMask:
    """
    Row-wise conditional mask:

    For each batch b and query index i:
      if label[b,i] == 0 (normal query):
          row = dozer_mask[b,i,:] & extreme_0_1[b,i,:]
      else (extreme query):
          row = extreme_1[b,i,:]

      extreme_0_1  : (B, L, L) bool   (your content/extreme constraint)
      extreme_1 : (B, L, L) bool   (e.g., extreme<->extreme)
    """
    def __init__(self, x_label, dozer_mask, extreme_0_1, extreme_1, device=None):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            B, L, _ = extreme_0_1.shape

            # labels: (B, L) bool where True means "extreme query"
            labels = x_label[..., 0].to(torch.long)          # (B,L)
            is_extreme_q = (labels == 1)                     # (B,L) bool

            # normalize dozer_mask to (B, L, L)
            if dozer_mask.dim() == 2:                        # (L,L)
                dozer_b = dozer_mask.unsqueeze(0).expand(B, -1, -1)
            else:
                raise ValueError(f"Unexpected dozer_mask shape: {dozer_mask.shape}")

            # normal rows: dozer & extreme_0_1
            normal_rows = dozer_b & extreme_0_1              # (B,L,L)

            # choose per-row using broadcasting
            # selector: (B,L,1) -> broadcast across keys (L)
            sel = is_extreme_q.unsqueeze(-1)                 # (B,L,1)
            combined = torch.where(sel, extreme_1, normal_rows)  # (B,L,L)

            self._mask = combined.to(device)     # (B,L,L)

    @property
    def mask(self):
        return self._mask


class ExtremeOnlyMask:
    """
    Builds an attention mask that is True only when BOTH query and key tokens are extreme (label==1).
    """
    def __init__(self, x_label, device=None, extreme_value=1):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0].to(torch.long)          # (B, L)
            extreme = (labels == extreme_value)              # (B, L) bool
            mask = extreme.unsqueeze(2) & extreme.unsqueeze(1)  # (B, L, L) bool
            self._mask = mask.to(device)        # (B, L, L)

    @property
    def mask(self):
        return self._mask

def show_mask(m):
    return m.detach().cpu().numpy()

def build_dozer_mask(L_Q, L_K, local_window=None, stride=None, device=None):
    device = device or "cpu"
    with torch.no_grad():
        i = torch.arange(L_Q, device=device)[:, None]   # [L_Q, 1]
        j = torch.arange(L_K, device=device)[None, :]   # [1, L_K]
        d = (i - j).abs()                               # [L_Q, L_K]

        mask = torch.zeros((L_Q, L_K), device=device, dtype=torch.bool)

        if local_window:
            w = local_window // 2
            mask |= (d <= w)

        if stride:
            s = stride + 1
            mask |= (d % s == 0)

        return mask

def build_dozer_mask_v1(L_Q, L_K, x_label=None, local_window=None, stride=None, device=None, cap=None):
    """
    Args:
        L_Q, L_K     : query/key sequence lengths
        x_label      : (batch, L_K, 1) float/bool — marks extreme positions
        local_window : local attention window width
        stride       : stride step for seasonal (normal queries only)
        device       : torch device

    Returns:
        If x_label is None → (L_Q, L_K)       bool mask
        If x_label given  → (batch, L_Q, L_K) bool mask
            • normal query rows  : local window  +  stride
            • extreme query rows : local window  +  all extreme-key columns
    """
    device = device or "cpu"

    with torch.no_grad():
        i = torch.arange(L_Q, device=device)[:, None]   # (L_Q, 1)
        j = torch.arange(L_K, device=device)[None, :]   # (1,  L_K)
        d = (i - j).abs()                               # (L_Q, L_K)

        # shared local component
        local_mask = torch.zeros((L_Q, L_K), device=device, dtype=torch.bool)
        if local_window:
            w = local_window // 2
            local_mask |= (d <= w)

        # normal-query mask  (local + stride)─
        normal_mask = local_mask.clone()
        if stride:
            normal_mask |= (d % (stride + 1) == 0)

        # no x_label → return shared static mask
        if x_label is None:
            return normal_mask                           # (L_Q, L_K)

        # batch-aware mask
        B = x_label.shape[0]
        x_label = x_label.to(device)                    # (B, L_K, 1)

        # which KEY positions are extreme?  (B, 1, L_K)
        extreme_key_cols = (x_label.squeeze(-1) > 0).unsqueeze(1)

        # which QUERY positions are extreme?  (B, L_Q, 1)
        # assumes L_Q == L_K (same sequence); adjust indexing if they differ
        extreme_query_rows = (x_label.squeeze(-1) > 0).unsqueeze(2)

        # expand static masks to batch dim
        normal_mask_b = normal_mask.unsqueeze(0).expand(B, -1, -1)   # (B, L_Q, L_K)
        local_mask_b  = local_mask .unsqueeze(0).expand(B, -1, -1)   # (B, L_Q, L_K)


        # extreme-query mask: local window  +  attend to every extreme key
        extreme_mask_b = local_mask_b | extreme_key_cols              # (B, L_Q, L_K)

        # stitch: extreme rows get extreme_mask, normal rows get normal_mask
        mask = torch.where(extreme_query_rows, extreme_mask_b, normal_mask_b)

        return mask                                      # (B, L_Q, L_K)

def generate_full_mask(B, L_Q, L_K, device=None):
    with torch.no_grad():
        mask = torch.ones((L_Q, L_K), device=device, dtype=torch.bool)
        return mask.repeat(B, 1, 1)



class ExtremeAndDozerMask:
    """
    Content-aware Extreme mask AND dozer) mask.
    """
    def __init__(self, x_label, dozer_mask, device=None):
        if device is None:
            device = x_label.device

        with torch.no_grad():
            labels = x_label[..., 0]  # (B, L)

            # (B, L, L): True when same label
            adapt_mask = labels.unsqueeze(2).eq(labels.unsqueeze(1))

            # normalize dozer_mask to (B, L, L)
            if dozer_mask.dim() == 2:                 # (L, L)
                B = labels.shape[0]
                dozer_b = dozer_mask.unsqueeze(0).expand(B, -1, -1)
            else:
                raise ValueError(f"Unexpected dozer_mask shape: {dozer_mask.shape}")

            base_mask = adapt_mask & dozer_b          # (B, L, L)
            self._mask = base_mask.to(device)  # (B,L,L)

    @property
    def mask(self):
        return self._mask


class SparseMask:
    def __init__(self, x_label, local_window, stride, device, B, L_Q, L_K):
        self.x_label = x_label
        self.local_window = local_window
        self.stride = stride
        self.device = device
        self.B = B
        self.L_Q = L_Q
        self.L_K = L_K
        self._mask = None

    def generate_mask(self, mask='dozer'):
        if mask == 'extreme_mask':
            self._mask = ExtremeMask(self.x_label).mask

        else:
            dozer_mask = build_dozer_mask(
                self.L_Q, self.L_K,
                local_window=self.local_window,
                stride=self.stride,
                device=self.device
            )

            if mask == 'dozer':
                self._mask = dozer_mask.unsqueeze(0).repeat(self.B, 1, 1)
            elif mask == "dozer_v1":
                self._mask = build_dozer_mask_v1(
                self.L_Q, self.L_K,
                x_label=self.x_label,
                local_window=self.local_window,
                stride=self.stride,
                device=self.device
            )
            elif mask == "dozer_v2":
                self._mask = build_dozer_mask_v1(
                    self.L_Q, self.L_K,
                    x_label=self.x_label,
                    local_window=self.local_window,
                    stride=self.stride,
                    device=self.device,
                    cap="local"
                    )
            elif mask == 'dozer_ext_only':
                self._mask = DozerExtremeOnlyMask(self.x_label, dozer_mask).mask
            elif mask == 'dozer_ext_0':
                extreme_0_1 = ExtremeMask(self.x_label).mask
                extreme_1 = ExtremeOnlyMask(self.x_label).mask
                self._mask = ConditionalExtremeDozerMask(self.x_label, dozer_mask, extreme_0_1, extreme_1).mask
            elif mask == 'dozer_ext_null':
                self._mask = ExtremeDozerSparseMask(ExtremeMask(self.x_label).mask, dozer_mask, self.x_label).mask
            elif mask == 'dozer_AND_ext':
                self._mask = ExtremeAndDozerMask(self.x_label, dozer_mask).mask
            elif mask == 'dozer_ext_0_v1':
                self._mask = build_dozer_ext_0_v1(self.L_Q, self.L_K, x_label=self.x_label,
                                                local_window=self.local_window,
                                                stride=self.stride, device=self.device)
            elif mask == 'full_mask':
                self._mask = generate_full_mask(self.B, self.L_Q, self.L_K, self.device)
            else:
                raise ValueError(f"Unknown mask type: {mask}")

        return self._mask

    def visualize_mask(self, mask='dozer'):
        MASK_TYPES = ['extreme_mask', 'dozer', 'dozer_ext_only', 'full_mask', 'dozer_ext_0',
                      'dozer_ext_null', 'dozer_AND_ext', 'dozer_ext_0_v1', 'dozer_v1', 'dozer_v2']

        if mask == 'all':
            results = {}
            for mask_type in MASK_TYPES:
                m = self.generate_mask(mask_type)
                results[mask_type] = m.detach().cpu().numpy()
            return results

        m = self.generate_mask(mask)
        return m.detach().cpu().numpy()

        # # If batched (3D), take the first batch element
        # mask_2d = self._mask[0] if self._mask.dim() == 3 else self._mask
        # mask_np = mask_2d.cpu().numpy()
        #
        # plt.figure(figsize=(10, 8))
        # plt.imshow(mask_np, aspect='auto', cmap='Blues', interpolation='none')
        # plt.colorbar(label='Mask Value')
        # plt.title("TEst")
        # plt.xlabel('Key Position (L_K)')
        # plt.ylabel('Query Position (L_Q)')
        # plt.tight_layout()
        # plt.show()

import torch

def build_local_band_mask(L_Q, L_K, local_window, device):
    """
    Returns a (L_Q, L_K) bool mask for the local window band only.
    """
    if not local_window:
        # If no local window specified, treat as "no restriction"
        return torch.ones((L_Q, L_K), device=device, dtype=torch.bool)

    i = torch.arange(L_Q, device=device)[:, None]
    j = torch.arange(L_K, device=device)[None, :]
    d = (i - j).abs()
    w = local_window // 2
    return (d <= w)

# B, L, _ = extreme_0_1.shape
#
# combined_mask = torch.zeros_like(extreme_0_1)
#
# for b in range(B):
#     for i in range(L):
#         if labels[b, i, 0] == 1:
#             continue
#         combined_mask[b, i, :] = extreme_0_1[b, i, :] & dozer_mask[b, i, :]

# labels = x_label[:, :, 0]  # [batch_size, L_Q]

# # Yifan's implementation
# extreme_0_1 = labels.unsqueeze(2).eq(labels.unsqueeze(1))
# batch_size = extreme_0_1.shape[0]
# dozer_mask = dozer_mask.repeat(batch_size, 1, 1)
# a_0_1 = extreme_0_1.detach().cpu().numpy()
# for i in range(extreme_0_1.shape[1]):
#     if labels[i] == 1:
#         continue
#     combined_mask = extreme_0_1[:, i, :] & dozer_mask[:, i, :]
# c_0_1 = combined_mask.detach().cpu().numpy()


def build_extreme_mask(L_Q, L_K, x_label, local_window=None, device=None):
    """
    Extreme queries attend to:
      1. ALL extreme keys globally (regardless of distance)
      2. ALL keys within local window (normal or extreme, for local context)
    """
    device = device or "cpu"
    B = x_label.shape[0]

    with torch.no_grad():
        i = torch.arange(L_Q, device=device)[:, None]
        j = torch.arange(L_K, device=device)[None, :]
        d = (i - j).abs()

        # all extreme key positions globally
        is_extreme_k = (x_label.squeeze(-1) == 1).unsqueeze(1)   # (B, 1, L_K)

        # local window for surrounding context
        local = torch.zeros((L_Q, L_K), device=device, dtype=torch.bool)
        if local_window:
            w = local_window // 2
            local |= (d <= w)
        local = local.unsqueeze(0)                                 # (1, L_Q, L_K)

        extreme_mask = is_extreme_k | local                        # (B, L_Q, L_K)

    return extreme_mask

def build_dozer_ext_0_v1(L_Q, L_K, x_label=None, local_window=None, stride=None, device=None):
    """
    Args:
        L_Q, L_K     : query/key sequence lengths
        x_label      : (batch, L_K, 1) float/bool — marks extreme positions
        local_window : local attention window width
        stride       : stride step for seasonal (normal queries only)
        device       : torch device

    Returns:
        If x_label is None → (L_Q, L_K)       bool mask
        If x_label given  → (batch, L_Q, L_K) bool mask
            • normal query rows  : dozer (local + stride) with extreme key columns blocked
            • extreme query rows : local window OR all extreme-extreme key columns (no stride)
    """
    device = device or "cpu"

    with torch.no_grad():
        i = torch.arange(L_Q, device=device)[:, None]
        j = torch.arange(L_K, device=device)[None, :]
        d = (i - j).abs()

        # position-based masks
        local_mask = torch.zeros((L_Q, L_K), device=device, dtype=torch.bool)
        if local_window:
            w = local_window // 2
            local_mask |= (d <= w)

        normal_mask = local_mask.clone()
        if stride:
            normal_mask |= (d % (stride + 1) == 0)

        # no x_label → return static dozer mask─
        if x_label is None:
            return normal_mask                               # (L_Q, L_K)

        # label-based masks─
        B = x_label.shape[0]
        x_label = x_label.to(device)                        # (B, L_K, 1)

        extreme_k = (x_label.squeeze(-1) > 0)               # (B, L_K)
        extreme_q = extreme_k                                # (B, L_Q)  assumes L_Q == L_K

        # extreme query i attends to extreme key j
        extreme_extreme = extreme_q.unsqueeze(2) & extreme_k.unsqueeze(1)  # (B, L_Q, L_K)

        # expand position masks to batch
        normal_mask_b = normal_mask.unsqueeze(0).expand(B, -1, -1)         # (B, L_Q, L_K)
        local_mask_b  = local_mask .unsqueeze(0).expand(B, -1, -1)         # (B, L_Q, L_K)

        # normal queries: dozer mask with extreme key columns blocked
        normal_mask_full  = normal_mask_b & ~extreme_k.unsqueeze(1)        # (B, L_Q, L_K)

        # extreme queries: local window OR all extreme keys globally (no stride)
        extreme_mask_full = local_mask_b  |  extreme_extreme               # (B, L_Q, L_K)

        # stitch on query type
        extreme_query_rows = extreme_q.unsqueeze(2)                        # (B, L_Q, 1)
        mask = torch.where(extreme_query_rows, extreme_mask_full, normal_mask_full)

        return mask                                                         # (B, L_Q, L_K)

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask