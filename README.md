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

## BEV backbone experiments (`model/experiments/`)

A second comparison series, alongside the fully-sparse-head experiment above. All
three replace `SparseCenterHead` with `RPNCenterHead` (the original dense pipeline's
head, unchanged) fed by a dense BEV feature map -- but they answer two DIFFERENT
questions, so read the backbone column below rather than assuming all 3 are directly
comparable to each other:

- **exp1/exp2 ("which sparse backbone structure is best?")**: each wraps a whole
  sparse 3D backbone (single-stage vs. 4-stage down/SlotFormer/up) so the backbone
  itself scatters to dense + merges z into channels at the end
  (`sparse_ops.scatter_to_bev`) and returns a ready BEV feature map. Same
  `SPARSE_BEV_*` voxelization and the same collapse method across both, so the
  comparison isolates backbone STRUCTURE alone.
- **exp3 ("what if ONLY the dense pipeline's z-collapse step were sparse?")**: not a
  backbone-structure swap at all -- `model.ConvMiddleLayers` (dense pipeline's own
  3-layer z-collapse) is reproduced layer-for-layer (same channels, same per-layer
  kernel/stride/padding) using `SparseConv3dDown` instead of `nn.Conv3d`. VFE and
  `RPNCenterHead`/`RPNBackbone` are the exact same dense-pipeline code, and there's no
  SlotFormer or extra backbone stages -- this isolates the effect of sparsity in that
  one component alone, as close to an apples-to-apples dense-vs-sparse ablation as
  this codebase gets.

Each experiment is self-contained (`voxelnet.py`/`train.py`/`smoke_test.py`, one
`<Name>BEVBackbone` class + one `SparseBEV<Name>VoxelNet` class per file) but shares
`config.py`, `sparse_ops.py`, `model.py`, `sparse_bev_head.py` from `model/` (the
parent directory):

| folder | backbone class | VoxelNet class | SlotFormer | x,y downsampled by backbone? |
|---|---|---|---|---|
| `exp1_single_stage_bev` | `SingleStageBEVBackbone` (wraps `backbone3d.Sparse3DBackbone`, 1 stage, `SPARSE_BACKBONE_*`) | `SparseBEVSingleStageVoxelNet` | external, 2 cycles (6L) | yes (isotropic stride) |
| `exp2_down_slot_up_bev` | `DownSlotUpBEVBackbone` (wraps `backbone3d_down_slot_up.SparseDownSlotUpBackbone`, 4-stage down + 4-stage full restore, `SPARSE_BEV_*`) | `SparseBEVDownSlotUpVoxelNet` | built into the backbone, at the bottleneck | yes (isotropic stride), but decoder restores back to input resolution |
| `exp3_conv_middle_bev` | `ConvMiddleBEVBackbone` (sparse mirror of `model.ConvMiddleLayers`, `SPARSE_BEV_CONVMID_CHANNELS`) | `SparseBEVConvMiddleVoxelNet` | none (dense baseline has none either) | **no** -- x,y's SIZE is unchanged (stride=1, "same" padding, exactly like the dense layer it mirrors); only D shrinks |

Run from inside each experiment's own folder (each inserts `model/`'s path via
`sys.path` so `import config`/`from model import ...`/etc. resolve to the shared
parent modules):

```bash
cd model/experiments/exp1_single_stage_bev   # or exp2_down_slot_up_bev / exp3_conv_middle_bev
python smoke_test.py                          # structural check, synthetic data, no dataset needed
python train.py --ckpt_dir checkpoints_exp1_single_stage_bev --batch_size <N> --epochs 20
```

### Backend: spconv, not this repo's own sparse_ops.py

All 3 backbones above are now built on **spconv** (traveller59/spconv2, package
`spconv-cu126`) instead of this repo's own from-scratch sparse conv primitives
(`sparse_ops.py`'s `SparseConv3dDown`/`SubMConv3d`/`SparseInverseConv3d`).
`sparse_ops.py` was originally written to avoid a compiled-CUDA-extension
dependency (a common source of broken Colab setups -- see that file's own
docstring) -- but `experiments/profile_pipeline.py` (per-submodule forward/backward
timing via hooks) found its hand-rolled backward pass ate **~80% of total step
time** in exp3, ~23x its own forward cost (a normal dense conv's backward is
roughly 2x its forward), and that cost didn't shrink even when the surrounding
backbone had almost no other compute (exp3 is just 3 conv layers). Switching to
spconv's real CUDA kernels fixed this directly, since it happened to already be
installed and working in this venv:

| experiment | before (sparse_ops.py) | after (spconv) | speedup |
|---|---|---|---|
| exp3_conv_middle_bev | ~1.4 it/s, 4.3GB | ~6.2-6.3 it/s, 3.4GB | ~4.4x |
| exp1_single_stage_bev | (not measured pre-spconv) | ~3.2-3.5 it/s, ~7-8GB | -- |
| exp2_down_slot_up_bev | (not measured pre-spconv) | ~2.5 it/s, ~7GB | -- |
| dense_baseline_bev (real `nn.Conv3d`, for comparison) | -- | ~0.74 it/s, 7.9GB | exp3 now ~8.4x faster than dense |

(batch_size=4, RTX 2070 8GB, real sonar data -- all measured over 250+ real
training steps per this repo's "measure before committing" methodology, not
synthetic-data smoke tests.) `sparse_ops.py`'s `build_index_grid` and
`scatter_to_bev` are still used (the scatter step was never the bottleneck --
its backward is a plain gather over unique indices, not the duplicate-index
scatter-add pattern that was actually slow) but the from-scratch conv primitives
are dead code now that backbone3d.py/backbone3d_down_slot_up.py both use spconv.

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
