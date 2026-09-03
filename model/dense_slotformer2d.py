"""dense_slotformer2d.py - dense-grid analog of slotformer.py's windowed axial
attention (SlotFormerBackbone with num_axes=2), for the dense counterpart of
exp1_single_stage_bev (dense_baseline_exp1_bev). Operates on a (B,C,H,W) dense
tensor instead of sparse (N,C)+coords -- same axial-window idea (window along x OR
y, alternating every layer, shifted every other layer like Swin), just implemented
via pad+reshape instead of slot_id grouping, since every grid position exists here
(no "active voxel" sparsity to hash into slots).

Simplification vs slotformer.py: positional encoding is added to the INPUT features
(before norm+qkv, affecting Q/K/V together) instead of only to Q/K after the qkv
projection -- a standard alternative (many ViT-style models do it this way), chosen
here because windowing the PE tensor identically to Q/K's post-projection shape
would need the same pad/reshape logic duplicated for a tensor with no batch dim.
Doesn't change what's being measured (compute cost of dense vs sparse windowed
attention), just where exactly the position signal enters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from slotformer import sinusoidal_pe


def _positional_encoding_2d(H: int, W: int, channels: int, device, temperature: float = 10000) -> torch.Tensor:
    y_idx = torch.arange(H, device=device).float()
    x_idx = torch.arange(W, device=device).float()
    pe_y = sinusoidal_pe(y_idx, channels, temperature)  # (H,C)
    pe_x = sinusoidal_pe(x_idx, channels, temperature)  # (W,C)
    return pe_y[:, None, :] + pe_x[None, :, :]  # (H,W,C)


class _DenseAxialWindowLayer(nn.Module):
    """One axial-window attention layer on a dense (B,C,H,W) grid -- dense analog of
    slotformer.SFLayer(num_axes=2). `direction` 0 windows along W (x), attending
    freely over the full H (y) within each window; direction 1 windows along H (y),
    attending freely over the full W (x) -- matching SFLayer's "only one axis is
    windowed, the other is unconstrained" design (long receptive field on the free
    axis, not a compact 2D window)."""

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

    def forward(self, feat2d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat2d.shape
        x = feat2d.permute(0, 2, 3, 1)  # (B,H,W,C)
        x = x + _positional_encoding_2d(H, W, C, feat2d.device, self.temperature).unsqueeze(0)

        win = self.win_size
        shift_amt = win // 2 if self.shift else 0
        shift_dim = 2 if self.direction == 0 else 1  # W axis or H axis, within (B,H,W,C)
        if shift_amt:
            x = torch.roll(x, shifts=-shift_amt, dims=shift_dim)

        if self.direction == 0:  # window along W, attend freely over H within each window
            pad_w = (win - W % win) % win
            if pad_w:
                x = F.pad(x, (0, 0, 0, pad_w))
            Wp = W + pad_w
            nwin = Wp // win
            xw = x.view(B, H, nwin, win, C).permute(0, 2, 1, 3, 4).reshape(B * nwin, H * win, C)
        else:  # window along H, attend freely over W within each window
            pad_h = (win - H % win) % win
            if pad_h:
                x = F.pad(x, (0, 0, 0, 0, 0, pad_h))
            Hp = H + pad_h
            nwin = Hp // win
            xw = x.view(B, nwin, win, W, C).reshape(B * nwin, win * W, C)

        N = xw.shape[1]
        h = self.norm1(xw)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(-1, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, N, self.channels)
        xw = xw + self.proj(out)
        xw = xw + self.ffn(self.norm2(xw))

        if self.direction == 0:
            x = xw.view(B, nwin, H, win, C).permute(0, 2, 1, 3, 4).reshape(B, H, Wp, C)
            x = x[:, :, :W, :]
        else:
            x = xw.view(B, nwin, win, W, C).reshape(B, Hp, W, C)
            x = x[:, :H, :, :]

        if shift_amt:
            x = torch.roll(x, shifts=shift_amt, dims=shift_dim)

        return x.permute(0, 3, 1, 2)  # back to (B,C,H,W)


class DenseSlotFormerBackbone2D(nn.Module):
    def __init__(self, channels, win_size, num_cycles=2, num_heads=4, temperature=10000):
        super().__init__()
        directions = [0, 1] * num_cycles  # x,y cycling -- matches slotformer.SlotFormerBackbone(num_axes=2)
        self.layers = nn.ModuleList([
            _DenseAxialWindowLayer(channels, num_heads, win_size, d, shift=(i % 2 == 1), temperature=temperature)
            for i, d in enumerate(directions)
        ])

    def forward(self, feat2d: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            feat2d = layer(feat2d)
        return feat2d
