"""Minimal sparse 3D conv primitives built entirely on core PyTorch.

No spconv / torch_scatter dependency: both need a CUDA-matched compiled
extension and are a common source of broken Colab setups. Everything here
uses only torch.scatter_reduce_ / index_add_ / advanced indexing, so it runs
identically on CPU (for local testing) and CUDA (Colab), with no extra installs.

A "sparse tensor" here is just:
  features: (N, C) float
  coords:   (N, 4) long, columns = [batch_idx, x_idx, y_idx, z_idx]
  index_grid: (B, X, Y, Z) long, dense lookup -> row index into features, or -1
"""
import torch
import torch.nn as nn


def build_index_grid(coords: torch.Tensor, batch_size: int, grid_size, device=None) -> torch.Tensor:
    X, Y, Z = grid_size
    device = device or coords.device
    grid = torch.full((batch_size, X, Y, Z), -1, dtype=torch.long, device=device)
    if coords.shape[0] > 0:
        grid[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = torch.arange(
            coords.shape[0], device=device
        )
    return grid


# Two implementations, same signature/output: vectorized batches all k^3
# kernel offsets into one gather+einsum (fewer, bigger GPU kernel launches --
# helps when launch overhead dominates, i.e. on GPU); loop processes one
# offset at a time (smaller peak memory, no k^3 blowup -- measured faster on
# CPU, where there's no launch-overhead penalty to amortize away).
CONV_MODE = "vectorized"  # "vectorized" or "loop" -- measured on a real GPU (RTX 2070, via
                          # benchmark_backbone.py on the actual sonar dataset): "loop" took
                          # ~1880ms/frame for this backbone, "vectorized" ~266ms/frame (7x) --
                          # launch-overhead dominates on GPU as expected, so "vectorized" is now
                          # the default. "loop" is still what's faster on CPU-only setups.
                          # (SparseConv3dDown's output-coord candidate search had its own,
                          # separate k^3 Python loop that CONV_MODE never touched -- vectorizing
                          # that too, see SparseConv3dDown.forward below, took the same benchmark
                          # down to ~44ms/frame, on par with an equivalent dense nn.Conv3d backbone
                          # (~48ms/frame) despite <0.1% voxel occupancy.)


def scatter_to_bev(features: torch.Tensor, coords: torch.Tensor, grid_size, batch_size: int,
                    channels: int) -> torch.Tensor:
    """Scatter a sparse (N,C) tensor to dense (B,C,D,H,W) via coords, then merge z
    into the channel dim -> (B,C*D,H,W). The one shared implementation of the
    "collapse a sparse backbone's output into a BEV feature map" step every
    sparse_bev_*.py experiment backbone does at the end of its forward -- lets
    each backbone return a ready-to-use dense BEV tensor directly instead of the
    caller having to scatter/reshape sparse tensors itself."""
    D, H, W = grid_size
    dense = features.new_zeros(batch_size, channels, D, H, W)
    if coords.shape[0] > 0:
        b, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        dense[b, :, z, y, x] = features
    return dense.reshape(batch_size, channels * D, H, W)


def _as_axis_tuple(v):
    """int -> (v,v,v); already-a-3-tuple -> unchanged. Lets stride/padding be given
    either isotropically (old call sites, unchanged behavior) or per-axis (e.g.
    stride=(2,1,1) for a z-only downsample, matching coords' column order)."""
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v, v)


def _sparse_conv_core(in_features, in_coords, in_index_grid, in_grid_size, out_coords,
                       weight, bias, stride, padding, dilation=1):
    if CONV_MODE == "loop":
        return _sparse_conv_core_loop(in_features, in_coords, in_index_grid, in_grid_size, out_coords, weight, bias, stride, padding, dilation)
    return _sparse_conv_core_vectorized(in_features, in_coords, in_index_grid, in_grid_size, out_coords, weight, bias, stride, padding, dilation)


def _sparse_conv_core_loop(in_features, in_coords, in_index_grid, in_grid_size, out_coords,
                            weight, bias, stride, padding, dilation=1):
    k = weight.shape[0]
    Nout = out_coords.shape[0]
    Cout = weight.shape[-2]
    out = torch.zeros(Nout, Cout, device=in_features.device, dtype=in_features.dtype)
    X, Y, Z = in_grid_size
    s0, s1, s2 = _as_axis_tuple(stride)
    p0, p1, p2 = _as_axis_tuple(padding)
    for kx in range(k):
        for ky in range(k):
            for kz in range(k):
                ix = out_coords[:, 1] * s0 + kx * dilation - p0
                iy = out_coords[:, 2] * s1 + ky * dilation - p1
                iz = out_coords[:, 3] * s2 + kz * dilation - p2
                in_bounds = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
                if not in_bounds.any():
                    continue
                b = out_coords[in_bounds, 0]
                row_idx = in_index_grid[b, ix[in_bounds], iy[in_bounds], iz[in_bounds]]
                has_nb = row_idx >= 0
                if not has_nb.any():
                    continue
                dest = in_bounds.clone()
                dest[in_bounds] = has_nb
                gathered = in_features[row_idx[has_nb]]
                w = weight[kx, ky, kz]
                out[dest] = out[dest] + gathered @ w.t()
    if bias is not None:
        out = out + bias
    return out


def _sparse_conv_core_vectorized(in_features, in_coords, in_index_grid, in_grid_size, out_coords,
                                  weight, bias, stride, padding, dilation=1):
    """Vectorized over all k^3 kernel offsets at once (one gather + one einsum)
    instead of a Python loop -- far fewer, larger GPU kernel launches, which
    matters a lot more than raw FLOPs here since every offset's per-op cost
    is tiny relative to launch overhead."""
    k = weight.shape[0]
    device = in_features.device
    Nout = out_coords.shape[0]
    Cout, Cin = weight.shape[-2], weight.shape[-1]
    if Nout == 0:
        return torch.zeros(0, Cout, device=device, dtype=in_features.dtype)

    ar = torch.arange(k, device=device)
    offsets = torch.stack(torch.meshgrid(ar, ar, ar, indexing="ij"), dim=-1).reshape(-1, 3) * dilation  # (K,3)
    K = offsets.shape[0]
    X, Y, Z = in_grid_size
    s0, s1, s2 = _as_axis_tuple(stride)
    p0, p1, p2 = _as_axis_tuple(padding)

    ix = out_coords[None, :, 1] * s0 + offsets[:, 0:1] - p0  # (K,Nout)
    iy = out_coords[None, :, 2] * s1 + offsets[:, 1:2] - p1
    iz = out_coords[None, :, 3] * s2 + offsets[:, 2:3] - p2
    ib = out_coords[None, :, 0].expand(K, Nout)

    in_bounds = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
    row_idx = in_index_grid[ib, ix.clamp(0, X - 1), iy.clamp(0, Y - 1), iz.clamp(0, Z - 1)]  # (K,Nout)
    valid = in_bounds & (row_idx >= 0)
    safe_row_idx = row_idx.clamp(min=0)

    gathered = in_features[safe_row_idx.reshape(-1)].reshape(K, Nout, Cin)
    gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)

    w = weight.reshape(K, Cout, Cin)
    out = torch.einsum("kod,knd->no", w, gathered)
    if bias is not None:
        out = out + bias
    return out


class SubMConv3d(nn.Module):
    """Submanifold conv: output lives on the exact same active coords as input.

    dilation>1 spaces the k^3 taps `dilation` voxels apart (padding grows to
    dilation*(k//2) to keep the active set exactly unchanged, same as dilation=1) --
    grows receptive field without adding a downsample stage, so it doesn't touch
    effective resolution (VOXEL_SIZE * backbone stride) the way a deeper/more-strided
    backbone would. See models/backbone3d.py's module docstring for why effective
    resolution is the thing that actually broke precision/recall before."""

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, dilation=1):
        super().__init__()
        k = kernel_size
        self.weight = nn.Parameter(torch.empty(k, k, k, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight.reshape(-1, in_channels), a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.k = k
        self.dilation = dilation
        self.padding = dilation * (k // 2)

    def forward(self, features, coords, index_grid, grid_size):
        out = _sparse_conv_core(
            features, coords, index_grid, grid_size, coords,
            self.weight, self.bias, stride=1, padding=self.padding, dilation=self.dilation,
        )
        return out, coords, index_grid  # active set unchanged


class SparseConv3dDown(nn.Module):
    """Strided conv that can shrink the active set (used for the stem downsample).

    stride/padding: either a single int (isotropic, same value for all 3 coord
    columns -- the original behavior) or a 3-tuple, one value per coords column
    (column order must match whatever convention the caller's coords use -- e.g.
    this repo's [batch,z_idx,y_idx,x_idx] means stride[0] applies to z, stride[1] to
    y, stride[2] to x). A z-only downsample (x,y untouched) is stride=(2,1,1)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True):
        super().__init__()
        k = kernel_size
        self.weight = nn.Parameter(torch.empty(k, k, k, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight.reshape(-1, in_channels), a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.k = k
        self.stride = _as_axis_tuple(stride)
        self.padding = _as_axis_tuple(padding)

    @staticmethod
    def output_grid_size(grid_size, kernel_size, stride, padding):
        stride = _as_axis_tuple(stride)
        padding = _as_axis_tuple(padding)
        return tuple((g + 2 * padding[i] - kernel_size) // stride[i] + 1 for i, g in enumerate(grid_size))

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        out_grid_size = self.output_grid_size(grid_size, self.k, self.stride, self.padding)
        device = features.device
        # candidate output coords: invert in_coord = out*stride + koff - padding for every
        # active input voxel and every kernel offset, keep integer/in-range results. Batches
        # all k^3 offsets into one vectorized pass instead of a 27-iteration Python loop --
        # on GPU that loop was k^3 sequential small launches *every* downsample stage, which
        # (per benchmark_backbone.py against the real dataset on an RTX 2070) was a big chunk
        # of why the sparse backbone was slower than an equivalent dense nn.Conv3d one despite
        # <0.1% voxel occupancy; this arithmetic is cheap enough that batching it costs nothing
        # extra on CPU either.
        k = self.k
        s0, s1, s2 = self.stride
        p0, p1, p2 = self.padding
        if coords.shape[0] == 0:
            out_coords = torch.zeros((0, 4), dtype=torch.long, device=device)
        else:
            # For an axis with stride==1 AND "same"-style padding (padding==k//2,
            # e.g. x,y in a z-only downsample), every kernel offset maps to a
            # DIFFERENT output coordinate (out = in - koff + padding), so looping
            # over the full k offsets there -- as the isotropic case correctly does
            # when every axis is actually being strided -- would instead "dilate"
            # the active set outward by up to k//2 voxels on that axis EVERY stage,
            # even though nothing is being downsampled there. Left unchecked this
            # compounds across stages (measured: 800 synthetic input voxels -> CUDA
            # OOM by the 4th stage). Using only the center tap (koff=padding, giving
            # out=in exactly) for such axes keeps output support on that axis
            # IDENTICAL to input support -- true submanifold behavior, matching
            # SubMConv3d -- while the actual gather in _sparse_conv_core still uses
            # the full k^3 kernel window regardless of how out_coords was built, so
            # the conv's receptive field is unaffected; only which positions get
            # computed changes. No-op for the isotropic (all-axes-strided) case.
            #
            # Requiring padding==k//2 (not just stride==1) matters: a "valid"-style
            # stride==1 layer (padding=0, e.g. model.ConvMiddleLayers' middle conv,
            # kernel=3/stride=1/padding=0 on z to shrink D by 2 without striding)
            # is genuinely a shrinking conv, not a resolution-preserving one -- the
            # center-tap trick would wrongly claim out=in (same support/size as
            # input) instead of discovering the true, smaller valid-conv output
            # domain, silently corrupting results instead of erroring. Falling back
            # to the full k-offset search there is correct AND still bounded (the
            # output domain this produces is inherently smaller than the input's,
            # so it can't runaway-dilate the way a same-padding stride==1 axis would).
            ar = torch.arange(k, device=device)
            off0 = ar if not (s0 == 1 and p0 == k // 2) else torch.tensor([p0], device=device)
            off1 = ar if not (s1 == 1 and p1 == k // 2) else torch.tensor([p1], device=device)
            off2 = ar if not (s2 == 1 and p2 == k // 2) else torch.tensor([p2], device=device)
            offsets = torch.stack(torch.meshgrid(off0, off1, off2, indexing="ij"), dim=-1).reshape(-1, 3)  # (K,3)

            numer_0 = coords[None, :, 1] - offsets[:, 0:1] + p0  # (K,N)
            numer_1 = coords[None, :, 2] - offsets[:, 1:2] + p1
            numer_2 = coords[None, :, 3] - offsets[:, 2:3] + p2
            div = (numer_0 % s0 == 0) & (numer_1 % s1 == 0) & (numer_2 % s2 == 0)

            o0 = numer_0 // s0
            o1 = numer_1 // s1
            o2 = numer_2 // s2
            ob = coords[None, :, 0].expand_as(o0)

            in_range = (
                (o0 >= 0) & (o0 < out_grid_size[0]) &
                (o1 >= 0) & (o1 < out_grid_size[1]) &
                (o2 >= 0) & (o2 < out_grid_size[2])
            )
            valid = (div & in_range).reshape(-1)

            cand = torch.stack([ob, o0, o1, o2], dim=-1).reshape(-1, 4)[valid]
            out_coords = torch.zeros((0, 4), dtype=torch.long, device=device) if cand.shape[0] == 0 \
                else torch.unique(cand, dim=0)

        out_features = _sparse_conv_core(
            features, coords, index_grid, grid_size, out_coords,
            self.weight, self.bias, stride=self.stride, padding=self.padding,
        )
        out_index_grid = build_index_grid(out_coords, batch_size, out_grid_size, device=device)
        return out_features, out_coords, out_index_grid, out_grid_size


def _sparse_conv_core_transposed(child_features, child_index_grid, child_grid_size,
                                  parent_coords, weight, bias, stride, padding):
    """The transpose of SparseConv3dDown's forward: gathers from the coarser
    ("child") sparse tensor into a caller-given ("parent") coordinate set, using
    the exact same in_coord = out*stride + koff - padding relationship
    SparseConv3dDown.forward already uses to derive its own output coords -- just
    run with parent_coords as the query/output side instead of being solved for.
    Because parent_coords is supplied by the caller (typically cached from the
    paired SparseConv3dDown call this is inverting) rather than discovered by
    scanning for occupied neighbors, this is a *paired* inverse, not a generative
    transposed conv: it never invents a coordinate that wasn't in the given
    parent set -- which is what lets a skip-connection merge be a plain
    index-aligned concat (see SparseInverseConv3d below) instead of a
    coordinate-hash join. Ported from the 3d_point_cloud sibling project.

    weight: (k,k,k,out_ch,in_ch) where in_ch indexes child_features' channels
    and out_ch indexes the channels this call produces for parent_coords."""
    k = weight.shape[0]
    device = child_features.device
    Nout = parent_coords.shape[0]
    Cout, Cin = weight.shape[-2], weight.shape[-1]
    if Nout == 0:
        return torch.zeros(0, Cout, device=device, dtype=child_features.dtype)

    ar = torch.arange(k, device=device)
    offsets = torch.stack(torch.meshgrid(ar, ar, ar, indexing="ij"), dim=-1).reshape(-1, 3)  # (K,3)
    K = offsets.shape[0]
    X, Y, Z = child_grid_size

    numer_x = parent_coords[None, :, 1] - offsets[:, 0:1] + padding  # (K,Nout)
    numer_y = parent_coords[None, :, 2] - offsets[:, 1:2] + padding
    numer_z = parent_coords[None, :, 3] - offsets[:, 2:3] + padding
    ib = parent_coords[None, :, 0].expand(K, Nout)

    div = (numer_x % stride == 0) & (numer_y % stride == 0) & (numer_z % stride == 0)
    cx, cy, cz = numer_x // stride, numer_y // stride, numer_z // stride
    in_bounds = (cx >= 0) & (cx < X) & (cy >= 0) & (cy < Y) & (cz >= 0) & (cz < Z)
    valid = div & in_bounds

    row_idx = child_index_grid[ib, cx.clamp(0, X - 1), cy.clamp(0, Y - 1), cz.clamp(0, Z - 1)]
    valid = valid & (row_idx >= 0)
    safe_row_idx = row_idx.clamp(min=0)

    gathered = child_features[safe_row_idx.reshape(-1)].reshape(K, Nout, Cin)
    gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)

    w = weight.reshape(K, Cout, Cin)
    out = torch.einsum("kod,knd->no", w, gathered)
    if bias is not None:
        out = out + bias
    return out


class SparseInverseConv3d(nn.Module):
    """Paired inverse of SparseConv3dDown -- deliberately NOT a generative
    transposed conv (the kind that grows the active set outward from scratch,
    like MinkowskiEngine's generative ConvolutionTranspose). This layer only
    ever writes to the exact `parent_coords` the caller passes in -- normally
    the coords cached from the matching SparseConv3dDown call it's inverting.
    Same (kernel_size, stride, padding) as that call is required for the
    coordinate arithmetic to invert correctly. Ported from the 3d_point_cloud
    sibling project (backbone3d_down_slot_up.py's decoder stage uses this)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        k = kernel_size
        self.weight = nn.Parameter(torch.empty(k, k, k, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight.reshape(-1, in_channels), a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.k = k

    def forward(self, features, coords, index_grid, grid_size,
                parent_coords, parent_index_grid, parent_grid_size, stride, padding):
        out = _sparse_conv_core_transposed(
            features, index_grid, grid_size, parent_coords,
            self.weight, self.bias, stride=stride, padding=padding,
        )
        return out, parent_coords, parent_index_grid  # active set becomes the parent's, unchanged henceforth


def make_norm1d(norm_type, channels, group_size=8):
    """"batch" (default, unchanged behavior) or "group" -- GroupNorm's statistics
    are computed per-sample (independent of how many active voxels other samples
    in the batch happen to have), unlike BatchNorm1d which pools statistics over
    ALL active voxels across the whole batch. At small batch sizes (this project's
    sparse configs run batch=2-8 on an 8GB card) with active-voxel counts that swing
    widely frame to frame (n_pos/voxel counts observed jumping several-fold between
    steps), BatchNorm's per-step statistics can be noisy -- GroupNorm is the standard
    fix point-cloud/sparse-conv work (e.g. VoxelNeXt, SECOND-family) reaches for here.
    group_size=8 channels/group is a common default; channels must be divisible by it."""
    if norm_type == "group":
        assert channels % group_size == 0, f"channels={channels} not divisible by group_size={group_size}"
        return nn.GroupNorm(channels // group_size, channels)
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    raise ValueError(f"unknown norm_type: {norm_type!r} (expected 'batch' or 'group')")


class SparseBasicBlock(nn.Module):
    """Two SubMConv3d + norm + ReLU with a residual connection (ResNet-style)."""

    def __init__(self, channels, kernel_size=3, dilation=1, norm_type="batch"):
        super().__init__()
        self.conv1 = SubMConv3d(channels, channels, kernel_size, bias=False, dilation=dilation)
        self.bn1 = make_norm1d(norm_type, channels)
        self.conv2 = SubMConv3d(channels, channels, kernel_size, bias=False, dilation=dilation)
        self.bn2 = make_norm1d(norm_type, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features, coords, index_grid, grid_size):
        identity = features
        out, _, _ = self.conv1(features, coords, index_grid, grid_size)
        out = self.relu(self.bn1(out))
        out, _, _ = self.conv2(out, coords, index_grid, grid_size)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out, coords, index_grid


# ---- scatter helpers (replace what torch_scatter would provide) ----

def scatter_sum(src, index, dim_size):
    shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(src, index, dim_size):
    s = scatter_sum(src, index, dim_size)
    ones = torch.ones(src.shape[0], device=src.device, dtype=src.dtype)
    cnt = scatter_sum(ones, index, dim_size).clamp(min=1)
    return s / cnt.view(-1, *([1] * (s.dim() - 1)))


def scatter_max(src, index, dim_size):
    shape = (dim_size,) + src.shape[1:]
    init = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    # out-of-place scatter_reduce (no trailing "_"): scatter_max is called repeatedly
    # inside the VFE's PFN stack, and mutating a tensor in place after it already has
    # autograd history attached corrupts the saved state scatter_reduce's backward needs.
    out = init.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out


def scatter_softmax(logits, index, dim_size):
    """Softmax over groups defined by `index`, along dim 0. logits: (N,) or (N,H)."""
    max_per_group = scatter_max(logits, index, dim_size)
    shifted = logits - max_per_group[index]
    exp = shifted.exp()
    denom = scatter_sum(exp, index, dim_size).clamp(min=1e-12)
    return exp / denom[index]
