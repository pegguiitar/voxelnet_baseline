import torch
import torch.nn as nn
import torch.nn.functional as F

# Same tradeoff as models/sparse_ops.py's CONV_MODE: "padded" batches every
# slot into one attention call (fewer/bigger GPU launches); "loop" attends
# one slot at a time (less wasted compute on padding, measured faster on
# CPU). Time both on the actual Colab GPU before trusting either default.
ATTENTION_MODE = "loop"  # "padded" or "loop" -- default "loop" because that's the one actually
                          # measured faster so far (on CPU); try "padded" on the real Colab GPU

# This one is a model-quality comparison, not a speed toggle: "softmax" is
# the standard scaled-dot-product attention we designed around; "linear" is
# the kernel-feature-map attention (Katharopoulos et al., elu(x)+1) that
# FSHNet's own SlotFormer actually uses. Same slot grouping either way --
# only the in-slot attention math changes. Compare val loss/recall between
# the two before picking one for a long run.
ATTENTION_KIND = "softmax"  # "softmax" or "linear"
LINEAR_ATTN_EPS = 1e-6


def sinusoidal_pe(coord_vals: torch.Tensor, dim: int, temperature: float) -> torch.Tensor:
    """Standard Transformer sin/cos positional encoding for a single axis.
    coord_vals: (N,) -> (N, dim)."""
    device = coord_vals.device
    inv_freq = temperature ** (2 * (torch.arange(dim, device=device) // 2) / dim)
    embed = coord_vals[:, None] / inv_freq[None, :]
    pe = torch.zeros(coord_vals.shape[0], dim, device=device, dtype=coord_vals.dtype)
    pe[:, 0::2] = embed[:, 0::2].sin()
    pe[:, 1::2] = embed[:, 1::2].cos()
    return pe


class SFLayer(nn.Module):
    """One axial slot-attention layer. `direction` (0/1/2 = x/y/z) is the ONLY
    axis that gets windowed into `win_size` bins; the other two axes are
    unconstrained within a slot, which is what gives this its long receptive
    field (see design discussion: this is not the same as a compact 3D window).

    Positional encoding is added to Q/K only (not V), following SST -- and
    covers all 3 axes (summed, not concatenated) rather than just the free
    ones, since it's cheap and keeps every layer's code path identical.
    """

    def __init__(self, channels, num_heads, win_size, direction, shift, temperature=10000, ffn_ratio=4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.win_size = win_size
        self.direction = direction
        self.shift = shift
        self.temperature = temperature

        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels * ffn_ratio, channels),
        )

    def _positional_encoding(self, coords):
        pe = torch.zeros(coords.shape[0], self.channels, device=coords.device, dtype=torch.float32)
        for axis in (1, 2, 3):
            pe = pe + sinusoidal_pe(coords[:, axis].float(), self.channels, self.temperature)
        return pe

    def _slot_ids(self, coords):
        shift_amt = self.win_size // 2 if self.shift else 0
        win_coord = (coords[:, self.direction + 1] + shift_amt).div(self.win_size, rounding_mode="floor")
        slot_key = torch.stack([coords[:, 0], win_coord], dim=1)
        _, slot_ids = torch.unique(slot_key, dim=0, return_inverse=True)
        return slot_ids

    def forward(self, features, coords):
        N = features.shape[0]
        if N == 0:
            return features

        slot_ids = self._slot_ids(coords)
        num_slots = int(slot_ids.max().item()) + 1
        pe = self._positional_encoding(coords)

        x = self.norm1(features)
        q_in, k_in, v_in = self.qkv(x).chunk(3, dim=-1)
        q = (q_in + pe).view(N, self.num_heads, self.head_dim)
        k = (k_in + pe).view(N, self.num_heads, self.head_dim)
        v = v_in.view(N, self.num_heads, self.head_dim)

        if ATTENTION_MODE == "loop":
            attn_out = self._attend_loop(q, k, v, slot_ids, num_slots)
        else:
            attn_out = self._attend_padded(q, k, v, slot_ids, num_slots)

        attn_out = self.proj(attn_out.reshape(N, self.channels))
        features = features + attn_out
        features = features + self.ffn(self.norm2(features))
        return features

    def _attend_loop(self, q, k, v, slot_ids, num_slots):
        attn_out = torch.zeros_like(v)
        order = torch.argsort(slot_ids)
        sorted_slots = slot_ids[order]
        counts = torch.bincount(sorted_slots, minlength=num_slots)
        offsets = torch.cumsum(counts, dim=0) - counts
        for s in range(num_slots):
            m = counts[s].item()
            if m == 0:
                continue
            idx = order[offsets[s]: offsets[s] + m]
            qs, ks, vs = q[idx].permute(1, 0, 2), k[idx].permute(1, 0, 2), v[idx].permute(1, 0, 2)
            if ATTENTION_KIND == "softmax":
                scores = (qs @ ks.transpose(-1, -2)) / (self.head_dim ** 0.5)
                weights = torch.softmax(scores, dim=-1)
                out = weights @ vs
            else:
                out = self._linear_attend(qs, ks, vs)
            attn_out[idx] = out.permute(1, 0, 2)
        return attn_out

    def _linear_attend(self, qs, ks, vs, key_mask=None):
        """Katharopoulos et al. linear attention: replace softmax(QK^T)V with a
        positive kernel feature map phi(x)=elu(x)+1, so QK^T never gets formed --
        phi(K)^T @ V is computed once per slot and reused for every query in it.
        qs/ks/vs: (..., seq, head_dim). key_mask (optional): (..., seq, 1) with
        1.0 for real positions / 0.0 for padding -- zeroed *after* phi(), since
        phi(0) = elu(0)+1 = 1, not 0, so un-masked padding would otherwise leak
        a phantom key/value into the sum."""
        phi_q = F.elu(qs) + 1
        phi_k = F.elu(ks) + 1
        if key_mask is not None:
            phi_k = phi_k * key_mask
            vs = vs * key_mask
        kv = phi_k.transpose(-1, -2) @ vs                            # (..., head_dim, head_dim)
        k_sum = phi_k.sum(dim=-2, keepdim=True)                      # (..., 1, head_dim)
        z = 1.0 / (phi_q @ k_sum.transpose(-1, -2) + LINEAR_ATTN_EPS)  # (..., seq, 1)
        return (phi_q @ kv) * z

    def _attend_padded(self, q, k, v, slot_ids, num_slots):
        """Pads every slot to the same size and runs ONE batched attention
        over the (num_slots, max_slot_size) grid -- fewer, larger GPU kernel
        launches than one attention call per slot, at the cost of wasted
        FLOPs on padding when slot sizes are uneven."""
        N = q.shape[0]
        order = torch.argsort(slot_ids)
        sorted_slots = slot_ids[order]
        counts = torch.bincount(sorted_slots, minlength=num_slots)
        offsets = torch.cumsum(counts, dim=0) - counts
        max_m = int(counts.max().item())

        within_slot_pos = torch.arange(N, device=q.device) - offsets[sorted_slots]
        padded_idx = torch.full((num_slots, max_m), -1, dtype=torch.long, device=q.device)
        padded_idx[sorted_slots, within_slot_pos] = order
        pad_mask = padded_idx >= 0
        safe_idx = padded_idx.clamp(min=0)

        q_pad = q[safe_idx].permute(0, 2, 1, 3)
        k_pad = k[safe_idx].permute(0, 2, 1, 3)
        v_pad = v[safe_idx].permute(0, 2, 1, 3)

        if ATTENTION_KIND == "softmax":
            scores = (q_pad @ k_pad.transpose(-1, -2)) / (self.head_dim ** 0.5)
            key_mask = pad_mask[:, None, None, :]
            scores = scores.masked_fill(~key_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            out_pad = (weights @ v_pad).permute(0, 2, 1, 3)
        else:
            key_mask = pad_mask[:, None, :, None].to(q_pad.dtype)  # (S,1,max_m,1), broadcasts over heads/head_dim
            out_pad = self._linear_attend(q_pad, k_pad, v_pad, key_mask=key_mask).permute(0, 2, 1, 3)

        attn_out = torch.zeros_like(v)
        flat_idx = padded_idx.reshape(-1)
        flat_out = out_pad.reshape(-1, self.num_heads, self.head_dim)
        valid = flat_idx >= 0
        attn_out[flat_idx[valid]] = flat_out[valid]
        return attn_out


class SlotFormerBackbone(nn.Module):
    def __init__(self, channels, win_size, num_cycles=2, num_heads=4, temperature=10000):
        super().__init__()
        directions = [0, 1, 2] * num_cycles
        layers = []
        for i, d in enumerate(directions):
            layers.append(SFLayer(channels, num_heads, win_size, d, shift=(i % 2 == 1), temperature=temperature))
        self.layers = nn.ModuleList(layers)

    def forward(self, features, coords):
        for layer in self.layers:
            features = layer(features, coords)
        return features
