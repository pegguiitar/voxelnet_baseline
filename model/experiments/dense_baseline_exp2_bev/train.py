"""train.py (dense_baseline_exp2_bev) - trains DenseBEVDownSlotUpVoxelNet.

Uses the scene-level 3-way split (TRAIN_SCENES/VAL_SCENES/TEST_SCENES,
sonar_diver_dataset.py) -- val is held out at the scene level, same as test.

Density aux-head supervision is skipped (points=None passed to build_bev_targets)
-- sonar_diver_dataset.py's collate_fn doesn't currently expose raw
pre-voxelization points, only the (M,13) gt_boxes tensor. The density_head still
runs (near-zero extra cost, matches RPNCenterHead's design -- see model.py), just
isn't supervised; add a "points" field to __getitem__/collate_fn if this aux loss
turns out to matter.

Usage:
    python train.py --ckpt_dir checkpoints_dense_baseline_exp2_bev
    python train.py --ckpt_dir checkpoints_dense_baseline_exp2_bev --resume checkpoints_dense_baseline_exp2_bev/last.pth
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from sonar_diver_dataset import SonarDiverDataset, collate_fn
from sparse_bev_head import build_bev_targets, decode_bev_center_boxes
from center_loss import center_voxelnet_loss
from eval_bev import evaluate_bev_ap, IOU_THRESHOLDS as VAL_AP_IOU_THRESHOLDS, PRINT_IOUS as VAL_AP_PRINT_IOUS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voxelnet import DenseBEVDownSlotUpVoxelNet  # noqa: E402

LOSS_KEYS = ["hm_loss", "reg_loss", "offset_loss", "z_loss", "dim_loss", "rot_loss", "density_loss"]
AP_KEYS = []  # ap_iou25, precision_iou25, recall_iou25, ap_iou30, ... (matches
for _t in VAL_AP_IOU_THRESHOLDS:  # model/train.py's own per-epoch val metric set, VAL_LOG_IOUS)
    _tag = int(round(_t * 100))
    AP_KEYS += [f"ap_iou{_tag}", f"precision_iou{_tag}", f"recall_iou{_tag}"]


class LossLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self.file = open(self.path, "a", newline="")
        self.writer = csv.writer(self.file)
        if is_new:
            self.writer.writerow(["phase", "epoch", "step", "lr", "n_pos", "total"] + LOSS_KEYS + AP_KEYS)
            self.file.flush()

    def log(self, phase, epoch, step, lr, total, stats, metrics_by_thresh=None):
        metrics_by_thresh = metrics_by_thresh or {}
        ap_cols = []
        for t in VAL_AP_IOU_THRESHOLDS:
            ap_cols += list(metrics_by_thresh.get(t, ("", "", "")))
        self.writer.writerow([phase, epoch, step, lr, stats.get("n_pos", ""), total] +
                              [stats.get(k, "") for k in LOSS_KEYS] + ap_cols)
        self.file.flush()

    def close(self):
        self.file.close()


def build_dataloader(split, batch_size, shuffle, num_workers):
    ds = SonarDiverDataset(
        split, point_cloud_range=config.SPARSE_BEV_POINT_CLOUD_RANGE, voxel_size=config.SPARSE_BEV_VOXEL_SIZE,
        max_points_per_voxel=config.SPARSE_BEV_MAX_POINTS_PER_VOXEL, max_voxels=config.SPARSE_BEV_MAX_VOXELS,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                       collate_fn=collate_fn, drop_last=shuffle, pin_memory=True,
                       persistent_workers=(num_workers > 0))


def run_step(model, batch, device):
    voxel_features = batch["voxel_features"].to(device, non_blocking=True)
    num_points = batch["num_points"].to(device, non_blocking=True)
    coords = batch["coords"].to(device, non_blocking=True)
    gt_boxes_list = batch["gt_boxes"]  # list[B] of (M_b,13) CPU tensors -- targets built on CPU (numpy)

    heatmap, offset, z, dim, rot, density = model(voxel_features, num_points, coords)

    targets = [build_bev_targets(gb.numpy(), None, model.head_grid_size, model.head_stride, model.pc_range)
               for gb in gt_boxes_list]
    heatmap_t = torch.from_numpy(np.stack([t["heatmap"][0] for t in targets])).unsqueeze(1).to(device)
    reg_mask_t = torch.from_numpy(np.stack([t["reg_mask"] for t in targets])).to(device)
    offset_t = torch.from_numpy(np.stack([t["offset"] for t in targets])).to(device)
    z_t = torch.from_numpy(np.stack([t["z"] for t in targets])).to(device)
    dim_t = torch.from_numpy(np.stack([t["dim"] for t in targets])).to(device)
    rot_t = torch.from_numpy(np.stack([t["rot"] for t in targets])).to(device)

    loss, stats = center_voxelnet_loss(
        heatmap, offset, z, dim, rot,
        heatmap_t, reg_mask_t, offset_t, z_t, dim_t, rot_t,
    )
    return loss, stats, (heatmap, offset, z, dim, rot, density)


@torch.no_grad()
def run_validation(model, val_loader, device):
    model.eval()
    sums = {k: 0.0 for k in LOSS_KEYS}
    total_sum, n = 0.0, 0
    for batch in val_loader:
        loss, stats, _ = run_step(model, batch, device)
        total_sum += loss.item()
        for k in LOSS_KEYS:
            sums[k] += stats.get(k, 0.0)
        n += 1
    model.train()
    n = max(n, 1)
    avg_stats = {k: v / n for k, v in sums.items()}
    return total_sum / n, avg_stats


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, epoch_complete):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step,
        "epoch_complete": epoch_complete,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="checkpoints_dense_baseline_exp2_bev")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--pct_start", type=float, default=0.1, help="OneCycleLR warmup fraction")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_every_epochs", type=int, default=1)
    parser.add_argument("--ckpt_every_steps", type=int, default=1000)
    parser.add_argument("--val_ap_every_n_epochs", type=int, default=1,
                         help="0 disables -- compute AP3D@IoU(0.30/0.35/0.40) on (a subsample of) the val split")
    parser.add_argument("--val_ap_max_frames", type=int, default=500,
                         help="cap frames scored per AP check (Monte-Carlo IoU is expensive; 0 = whole val split)")
    parser.add_argument("--val_ap_score_thresh", type=float, default=0.1)
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found -- this will be very slow.")

    train_loader = build_dataloader("train", args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_dataloader("val", args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"train batches/epoch: {len(train_loader)}  val batches: {len(val_loader)}")

    model = DenseBEVDownSlotUpVoxelNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  out_grid_size(D,H,W): {model.out_grid_size}  "
          f"head_grid_size(W'',H''): {model.head_grid_size}  head_stride: {model.head_stride}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    start_epoch, global_step = 0, 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1 if ckpt.get("epoch_complete") else ckpt["epoch"]
        global_step = ckpt["step"]
        print(f"resumed from {args.resume} at epoch {start_epoch}, step {global_step}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.pct_start, div_factor=25,
        last_epoch=(global_step - 1) if ckpt is not None else -1,
    )

    ckpt_dir = Path(args.ckpt_dir)
    log_path = Path(args.log_file) if args.log_file else ckpt_dir / "loss_history.csv"
    logger = LossLogger(log_path)
    print(f"logging to {log_path}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs - 1}")
        for batch in pbar:
            loss, stats, _ = run_step(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += loss.item()
            lr = scheduler.get_last_lr()[0]
            logger.log("train", epoch, global_step, lr, loss.item(), stats)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "hm": f"{stats['hm_loss']:.3f}",
                               "n_pos": stats["n_pos"], "lr": f"{lr:.2e}"})

            if args.ckpt_every_steps and global_step % args.ckpt_every_steps == 0:
                save_checkpoint(ckpt_dir / f"step_{global_step}.pth", model, optimizer, scheduler,
                                 epoch, global_step, epoch_complete=False)

        epoch_time = time.time() - epoch_t0
        avg_train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_stats = run_validation(model, val_loader, device)

        metrics_by_thresh = None
        if args.val_ap_every_n_epochs and (epoch + 1) % args.val_ap_every_n_epochs == 0:
            max_frames = None if not args.val_ap_max_frames else args.val_ap_max_frames
            metrics_by_thresh, ap_time = evaluate_bev_ap(
                model, val_loader, device, score_thresh=args.val_ap_score_thresh, max_frames=max_frames)

        logger.log("val", epoch, global_step, "", val_loss, val_stats, metrics_by_thresh=metrics_by_thresh)
        msg = f"epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} time={epoch_time:.1f}s"
        if metrics_by_thresh is not None:
            # stdout stays terse (PRINT_IOUS subset, AP only); the full IOU_THRESHOLDS
            # set (AP+precision+recall each) still goes to the CSV via logger.log above.
            ap_str = "  ".join(f"AP@{t:.2f}={metrics_by_thresh[t][0]:.4f}" for t in VAL_AP_PRINT_IOUS)
            msg += f"  ({ap_time:.1f}s) {ap_str}"
        print(msg)

        if args.ckpt_every_epochs and (epoch + 1) % args.ckpt_every_epochs == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch}.pth", model, optimizer, scheduler,
                             epoch, global_step, epoch_complete=True)

    save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scheduler, args.epochs - 1, global_step, epoch_complete=True)
    logger.close()
    print("training complete.")


if __name__ == "__main__":
    main()
