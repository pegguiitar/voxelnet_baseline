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

## Setup

```bash
cd model
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your driver's CUDA
# (no separate requirements.txt yet on this branch -- torch/tqdm/pyyaml/numpy cover it)
```

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
