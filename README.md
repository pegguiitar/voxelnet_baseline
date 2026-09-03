# voxelnet_baseline

VoxelNet (Zhou & Tuzel, 2018) reproduction for sonar-based 3D diver detection, plus
a backbone-swap experiment (`sparse` branch, this branch) that keeps VoxelNet's
original head design but replaces its dense VFE->ConvMiddleLayers[z-collapse]->
RPNBackbone[2D] pipeline with a genuinely-3D sparse backbone that never collapses
height into channels.

## Original dense pipeline (`main` branch / `model/model.py`)

`StackedVFE` (paper's VFE-1(7,32)->VFE-2(32,128)->FCN(128,128)+max-pool) -> dense
scatter into a `(B,128,D,H,W)` grid -> `ConvMiddleLayers` (3x Conv3D, z-only stride,
D=10->2, this is where height gets folded into channels) -> reshape to 2D BEV ->
`RPNBackbone` (3-block downsample + deconv-upsample-concat FPN neck, paper's Fig.4)
-> `RPNCenterHead` (CenterPoint-style anchor-free: heatmap/offset/z/dim/rot6D/density).

## Sparse backbone experiment (this branch)

`sparse_voxelnet.py`'s `SparseVoxelNet` reuses `StackedVFE` and `SparseCenterHead`
(RPNCenterHead's head design, dimension-transformed from dense 2D conv to sparse
per-voxel ops -- see `sparse_center_head.py`'s docstring) unchanged, and swaps only
the middle section for `backbone3d_down_slot_up.SparseDownSlotUpBackbone`:

```
VFE (unchanged) -> N-stage sparse downsample encoder -> SlotFormer (bottleneck)
    -> M-stage sparse upsample decoder -> SparseCenterHead (unchanged)
```

Ported from the `3d-point-cloud` sibling project's SlotFormer-backbone comparison
series (`models/backbone3d_down_slot_up.py`/`slotformer.py`/`sparse_ops.py`) --
see that repo's README for the full architecture rationale and the other 3
SlotFormer-backbone variants it also tried.

Current config (`config.py`'s `SPARSE_DOWN_SLOT_UP_*` constants):

| knob | value | meaning |
|---|---|---|
| `STAGE_CHANNELS` | `(64, 96, 128, 128)` | 4 downsample stages |
| `UPSAMPLE_STAGES` | `4` | fully restores resolution (`total_stride=1`) |
| `SLOTFORMER_NUM_CYCLES` | `2` | 6L (two full x/y/z passes) |
| `SLOTFORMER_WIN_SIZE` | `3` | sized for this backbone's bottleneck effective resolution (`VOXEL_SIZE(0.1) * stride^4(16) = 1.6m/voxel`) to cover roughly the same ~4.8m physical window the sibling project's single-stage backbone used |
| `SPARSE_VOXEL_SIZE` | `(0.1, 0.1, 0.1)` | matches the dense baseline's own effective x,y resolution (see `config.py`'s comment) |

**Not yet measured on real GPU hardware** at the time this was pushed -- only
verified structurally via `smoke_sparse_voxelnet.py` (synthetic data, CPU). This
specific combination (4-stage down + 4-stage full restore + SlotFormer(6L), at a
fine 0.1m voxel size) is architecturally similar to the `3d-point-cloud` project's
`unet` branch, which measured 3+ hours/epoch even on an A100 -- time a handful of
real steps before committing to a full run (see "Measuring before training" below).

## Fully-sparse BEV experiments (`model/experiments/`)

**2026-09-03, second redesign.** The first BEV redesign (exp1/2/3 each ending in
"scatter to dense `(B,C,H,W)` -> dense 2D `RPNBackbone`/`RPNCenterHead`") still paid
full dense-grid cost for that entire 2D stage no matter how cheap the sparse
encoder in front of it was -- the same waste `dense_baseline_bev`'s
`ConvMiddleLayers`-alone comparison measured (dense ~0.74 it/s/7.9GB vs sparse
~6.2 it/s/3.4GB) turned out to apply to the much bigger, shared 16-layer 2D
backbone too. This redesign never materializes a dense tensor anywhere, closer to
VoxelNeXt's (Chen et al., CVPR 2023) "stay sparse through the head" approach --
adapted onto this codebase's own components (spconv, this repo's VFE/config/head
conventions) rather than porting VoxelNeXt's exact architecture. All 3 share the
same first step (`zdown_to_sparse2d.ZDownTo2D`: z-only-stride spconv.SparseConv3d
down to D=1, so every surviving `(batch,y,x)` has at most one row and dropping z
needs no merge, just a column drop -- see that module's docstring for why
`sparse_ops.restrict_xy_support` is needed alongside it: spconv's `SparseConv3d`,
unlike `SubMConv3d`, doesn't restrict a stride==1 axis to its input support on its
own, so a naive z-only-stride stage "dilates" x,y outward every stage) and the same
sparse head (`sparse_head_bev.SparseBEVCenterHead` -- `RPNCenterHead`'s head set as
`nn.Linear` instead of `nn.Conv2d(...,kernel_size=1)`, same dimension-transform
trick `sparse_center_head.py` uses for the fully-3D experiment above), then differ
in exactly one thing each:

| folder | after z-compression | x,y ever downsampled? | idea |
|---|---|---|---|
| `exp3_conv_middle_bev` | straight to the head | no | minimal baseline, closest to VoxelNeXt's "stay sparse, predict directly" |
| `exp1_single_stage_bev` | `slotformer.SlotFormerBackbone(num_axes=2)` (windowed attention, x,y only -- z is already gone) | no | "does adding attention alone help?" |
| `exp2_down_slot_up_bev` | isotropic 3D encoder first (`backbone3d.Sparse3DBackbone`, x,y,z together) *then* z-compress the bottleneck, *then* `backbone2d_sparse.Sparse2DBackbone` (a self-contained sparse 2D U-Net, spconv `SparseConv2d`/`SparseInverseConv2d`) | **yes** | "does a real spatial (2D) backbone help beyond attention?" |

(exp2's encoder->decoder split is a deliberate workaround, not the original
"encoder downsamples x,y,z, decoder restores x,y only" idea verbatim:
`spconv.SparseInverseConv2d`/`3d` can only invert a conv call that used the *exact
same* kernel/stride/padding it's paired with via `indice_key`, so a decoder that
undoes only 2 of an isotropic 3D encoder's 3 downsampled axes isn't directly
expressible. Finishing z-compression at the bottleneck first, then handing off to a
backbone that is *only* 2D from there on, gets the same practical effect: "the
decoder only ever touches x,y" is true by construction, since there's no z axis
left for it to touch.)

`slotformer.py`'s `SFLayer`/`SlotFormerBackbone` gained a `num_axes` parameter
(default 3, unchanged) for this: the original 3-axis (x,y,z) direction-cycling
assumes 4-column `[batch,z,y,x]` coords and would index out of bounds on the
3-column `[batch,y,x]` coords these experiments have post-z-compression;
`num_axes=2` cycles only (x,y) so every attention layer does useful windowed work
instead of wasting 1 of every 3 layers on a degenerate constant axis.

Because exp1/exp3 never downsample x,y at all, their head runs at the *full* input
x,y resolution (`head_grid_size` = `SPARSE_BEV_GRID_SIZE`'s own W,H) -- exp2's head
runs at a coarser resolution (isotropic encoder's stride, times
`Sparse2DBackbone`'s own /2). This is an inherent consequence of the 3 designs
answering different questions, not a bug -- don't expect their AP numbers to be
directly comparable without accounting for it.

Each `voxelnet.py`'s `forward()` returns `(pred_dict, coords, batch_size)` (sparse,
variable-length) instead of a dense 6-tuple -- targets/loss/decode come from
`sparse_head_bev.py` (`build_sparse_bev_targets`/`sparse_bev_center_loss`/
`decode_sparse_bev_boxes`, the 2D-after-height-compression analog of
`sparse_center_head.py`'s fully-3D versions: same CenterPoint nearest-active-cell
assignment idea, but z comes back as a regressed value instead of a spatial index),
and per-epoch val AP from `eval_sparse_bev.py`, not `sparse_bev_head.py`/
`eval_bev.py` (those stay in use by `dense_baseline_bev`, which still needs a real
dense tensor to mirror `model.ConvMiddleLayers` faithfully).

Run from inside each experiment's own folder:

```bash
cd model/experiments/exp1_single_stage_bev   # or exp2_down_slot_up_bev / exp3_conv_middle_bev
python smoke_test.py                          # structural check, synthetic data, no dataset needed
python train.py --ckpt_dir checkpoints_exp1_single_stage_bev --batch_size <N> --epochs 20
```

**Real-data measurements (batch_size=4, RTX 2070 8GB)**, comparing this redesign
against the previous (dense-2D-head) one at the same experiment:

| experiment | previous (dense 2D head) | this redesign (fully sparse) |
|---|---|---|
| exp3_conv_middle_bev | ~6.2-6.3 it/s, 3.4GB | **~11-12 it/s, ~580MB-3.3GB** |
| exp1_single_stage_bev | ~3.2-3.5 it/s, ~7-8GB | ~834MB (single-batch check; not yet measured over a sustained run) |
| exp2_down_slot_up_bev | ~2.5 it/s, ~7GB | ~582MB (single-batch check; not yet measured over a sustained run) |

exp3's ~2x additional speedup (on top of the earlier sparse_ops.py->spconv
migration's own ~4.4x) and the memory drop from GB to sub-GB is exactly the
"dense 2D backbone was still the bottleneck" hypothesis confirmed -- removing it
entirely, not just making its input sparse, is what did this.

### Backend: spconv, not this repo's own sparse_ops.py

Every sparse backbone in this repo (the fully-3D experiment above, and all 3
BEV experiments) is built on **spconv** (traveller59/spconv2, package
`spconv-cu126`) instead of a from-scratch sparse conv implementation.
`sparse_ops.py` originally held one (`SparseConv3dDown`/`SubMConv3d`/
`SparseInverseConv3d`), written to avoid a compiled-CUDA-extension dependency (a
common source of broken Colab setups) -- but profiling
(`experiments/profile_pipeline.py`, per-submodule forward/backward timing via
hooks) found its hand-rolled backward pass ate ~80% of total step time in an
earlier exp3 design, ~23x its own forward cost. Switching to spconv's real CUDA
kernels fixed this directly (measured ~4.4x real-training speedup on its own,
before the fully-sparse redesign above added another ~2x on top), since it
happened to already be installed and working in this venv. The from-scratch
primitives were removed once nothing used them anymore; `sparse_ops.py` now only
holds `build_index_grid` (still used by `sparse_center_head.py`),
`scatter_to_bev` (still used by `dense_baseline_bev`), and `yx_key`/
`restrict_xy_support` (new, see `zdown_to_sparse2d.py`'s docstring above).

### Per-epoch validation metrics (`eval_bev.py` / `eval_sparse_bev.py`)

Every `train.py` in `experiments/` logs the exact same per-epoch val metric set
the confirmed dense-pipeline baseline uses (`model/train.py`'s `VAL_LOG_IOUS`/
`compute_ap`): AP3D, precision, and recall at IoU 0.25/0.30/0.35/0.40/0.50 (0.35 is
the baseline's own primary/reported threshold), via Monte-Carlo OBB IoU
(`eval_voxelnet.iou_3d_obb`) on a capped, fixed-seed sample of the val split
(`--val_ap_max_frames`, default 500 -- scoring the full ~8000-frame val split every
epoch would dominate total training time). Stdout prints the AP@{0.30,0.35,0.40}
subset each epoch; the full 15 columns (`ap_iou25`, `precision_iou25`,
`recall_iou25`, ... `ap_iou50`, `precision_iou50`, `recall_iou50`) land in
`loss_history.csv`'s `phase="val"` rows. Disable with `--val_ap_every_n_epochs 0`
if you just want loss curves. `dense_baseline_bev`/exp1/2/3 (dense-BEV-head design,
now superseded) use `eval_bev.py`; the current exp1/2/3 (fully sparse) use
`eval_sparse_bev.py` -- same metric definitions, different decode function
underneath (`sparse_bev_head.decode_bev_center_boxes` vs.
`sparse_head_bev.decode_sparse_bev_boxes`).

## Setup

```bash
cd model
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your driver's CUDA
pip install spconv-cu126   # match your CUDA version instead if not cu126 -- see spconv's PyPI page for the list
# (no separate requirements.txt yet on this branch -- torch/spconv/tqdm/pyyaml/numpy cover it)
```

`sparse_voxelnet.py` and every `experiments/*_bev/voxelnet.py` need spconv (see
"Backend: spconv" above); the fully-dense `main` branch pipeline doesn't.

Dataset: `sonar_diver_dataset.py` reads directly from the sibling
`labeling-tool-main/dataset` (i.e. this repo and `labeling-tool-main` need to be
sibling directories) -- `TEST_SCENES`/`TRAINVAL_SCENES` are hardcoded near the top
of that file, mirroring the `3d-point-cloud` project's split.

## Structural sanity check (no dataset needed)

```bash
python smoke_sparse_voxelnet.py
```
Confirms forward/target-building/loss/backward/decode all run cleanly on synthetic
data before touching the real dataset.

## Measuring before training

Same methodology as the `3d-point-cloud` sibling project: run a real training
process for a few minutes / 250+ steps and watch `nvidia-smi` (or
`torch.cuda.max_memory_reserved()`) for memory creep, not just an isolated few-step
probe -- reserved memory on this codebase's wildly variable per-frame active-voxel
counts has been observed to climb for a couple minutes before plateauing.

## Train

```bash
python train_sparse.py --ckpt_dir checkpoints_sparse --batch_size <N> --epochs 20
```

`--batch_size`/other hyperparameters are **not tuned yet** for this backbone --
measure real GPU memory/speed first (see above) rather than trusting `config.py`'s
defaults, which were only sanity-checked on CPU.

`train_sparse.py` evaluates AP@IoU(0.30/0.35/0.40) on the test split after every
epoch by default (`--test_eval_every_n_epochs 1`) and logs everything (per-step
losses + per-epoch AP) to `<ckpt_dir>/loss_history.csv`.
