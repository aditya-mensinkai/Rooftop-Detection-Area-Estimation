"""
train.py — Training & Validation Pipeline
SolarSense Platform | IEEE YESIST12 WePOWER Track 2026

Two-phase training strategy:
    Phase 1 (epochs 1–10)  : Encoder frozen — decoder warmup with high LR
    Phase 2 (epochs 11–40) : Full fine-tune with differential LR
                             (encoder 10× lower LR than decoder)

Run:
    python train.py --image_dir dataset/images --mask_dir dataset/masks

Key outputs saved to --output_dir (default: runs/solarsense/):
    best_model.pth          ← best validation IoU checkpoint
    last_model.pth          ← latest epoch checkpoint
    train_log.csv           ← per-epoch metrics
    viz/epoch_<N>.png       ← image | mask | prediction grids
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import RoofDataset, split_dataset
from loss import CombinedLoss
from metrics import MetricTracker
from model import UNetResNet34


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SolarSense Training Pipeline")
    p.add_argument("--image_dir",    default="dataset_final/images", help="Path to .tif images")
    p.add_argument("--mask_dir",     default="dataset_final/masks",  help="Path to .npy masks")
    p.add_argument("--output_dir",   default="runs/solarsense",    help="Where to save outputs")
    p.add_argument("--epochs",       type=int,   default=40,       help="Total training epochs")
    p.add_argument("--freeze_epochs",type=int,   default=10,       help="Epochs to freeze encoder")
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--img_size",     type=int,   default=256)
    p.add_argument("--lr_decoder",   type=float, default=1e-3,     help="Decoder LR (Phase 1 & 2)")
    p.add_argument("--lr_encoder",   type=float, default=1e-4,     help="Encoder LR (Phase 2 only)")
    p.add_argument("--val_split",    type=float, default=0.15)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--gsd",          type=float, default=0.5,      help="Ground Sampling Distance (m/px)")
    p.add_argument("--viz_every",    type=int,   default=5,        help="Save viz every N epochs")
    p.add_argument("--resume",       default=None,                 help="Path to checkpoint to resume from")
    p.add_argument("--filter_empty", action="store_true", default=True, help="Filter out empty masks (no buildings)")
    p.add_argument("--keep_empty_ratio", type=float, default=0.2,  help="Ratio of empty masks to keep (0-1)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # Force CPU temporarily
    print("  Using CPU (training will be slow)")
    return torch.device("cpu")


def save_checkpoint(
    model: UNetResNet34,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "metrics":     metrics,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: UNetResNet34,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load checkpoint; returns the epoch to resume from."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    epoch = ckpt.get("epoch", 0)
    print(f"  Resumed from {path} (epoch {epoch})")
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet denormalization constants
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(t: torch.Tensor) -> np.ndarray:
    """Convert normalised tensor (3,H,W) → uint8 HWC numpy array."""
    img = t.cpu() * _STD + _MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def save_viz(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    epoch: int,
    save_dir: Path,
    n: int = 4,
) -> None:
    """
    Save a side-by-side visualisation grid:
        Column 1 : Input satellite image (denormalised)
        Column 2 : Ground-truth mask (white = rooftop)
        Column 3 : Predicted mask (after sigmoid + threshold)
        Column 4 : Overlay (prediction in red over image)

    Args:
        images   : (B, 3, H, W) normalised image tensor
        masks    : (B, 1, H, W) binary ground-truth
        logits   : (B, 1, H, W) raw model output
        epoch    : Current epoch number (used in filename)
        save_dir : Directory to save the PNG
        n        : Number of samples to visualise (≤ batch size)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, images.size(0))
    preds = (torch.sigmoid(logits.cpu()) > 0.5).float()

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D indexing

    col_titles = ["Satellite Image", "Ground Truth", "Prediction", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row in range(n):
        img_np  = denorm(images[row])
        gt_np   = masks[row, 0].cpu().numpy()
        pred_np = preds[row, 0].numpy()

        # Overlay: red channel boosted where prediction = 1
        overlay = img_np.copy()
        overlay[pred_np == 1, 0] = np.clip(overlay[pred_np == 1, 0].astype(int) + 80, 0, 255)
        overlay[pred_np == 1, 1] = np.clip(overlay[pred_np == 1, 1].astype(int) - 30, 0, 255)
        overlay[pred_np == 1, 2] = np.clip(overlay[pred_np == 1, 2].astype(int) - 30, 0, 255)

        axes[row, 0].imshow(img_np)
        axes[row, 1].imshow(gt_np,   cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(overlay)

        for col in range(4):
            axes[row, col].axis("off")

    plt.suptitle(f"SolarSense — Epoch {epoch}", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = save_dir / f"epoch_{epoch:03d}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Viz saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: UNetResNet34,
    loader: DataLoader,
    criterion: CombinedLoss,
    tracker: MetricTracker,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
    phase: str,
) -> tuple[float, dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one full epoch (train or val).

    Returns:
        mean_loss   : Average total loss over the epoch
        metrics     : Dict from MetricTracker.compute()
        last_images : Image tensor from the final batch (for viz)
        last_masks  : Mask tensor from the final batch
        last_logits : Logit tensor from the final batch
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    tracker.reset()

    running_loss = 0.0
    last_images = last_masks = last_logits = None

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            # Forward pass (mixed precision when scaler is available)
            if scaler is not None:
                with autocast():
                    logits = model(images)
                    total_loss, _ = criterion(logits, masks)
            else:
                logits = model(images)
                total_loss, _ = criterion(logits, masks)

            if is_train:
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            running_loss += total_loss.item()
            tracker.update(logits.detach(), masks)

            # Keep last batch for visualisation
            last_images = images.detach().cpu()
            last_masks  = masks.detach().cpu()
            last_logits = logits.detach().cpu()

    mean_loss = running_loss / len(loader)
    metrics   = tracker.compute()
    return mean_loss, metrics, last_images, last_masks, last_logits


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device    = get_device()
    out_dir   = Path(args.output_dir)
    viz_dir   = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n[1/5] Building datasets …")
    train_stems, val_stems = split_dataset(
        args.image_dir, val_split=args.val_split, seed=args.seed
    )
    print(f"  Train: {len(train_stems)} | Val: {len(val_stems)}")

    train_ds = RoofDataset(
        args.image_dir, args.mask_dir,
        file_stems=train_stems, augment=True,
        target_size=args.img_size, gsd=args.gsd,
        filter_empty=args.filter_empty,
        keep_empty_ratio=args.keep_empty_ratio,
    )
    val_ds = RoofDataset(
        args.image_dir, args.mask_dir,
        file_stems=val_stems, augment=False,
        target_size=args.img_size, gsd=args.gsd,
        filter_empty=args.filter_empty,
        keep_empty_ratio=args.keep_empty_ratio,
    )

    print(f"[INFO] Training samples: {len(train_ds)}")
    print(f"[INFO] Validation samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print("\n[2/5] Building model …")
    model = UNetResNet34(pretrained=True, dropout=args.dropout).to(device)

    # ── Optimiser (Phase 1 — decoder only) ──────────────────────────────────
    print("\n[3/5] Configuring optimiser …")
    optimizer = AdamW(model.get_decoder_params(), lr=args.lr_decoder, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision scaler (CUDA only)
    scaler = GradScaler() if device.type == "cuda" else None

    # Freeze encoder for Phase 1
    model.freeze_encoder()
    print(f"  Phase 1: encoder frozen for {args.freeze_epochs} epochs")

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)

    # ── Loss & metrics ───────────────────────────────────────────────────────
    criterion    = CombinedLoss().to(device)
    train_tracker = MetricTracker()
    val_tracker   = MetricTracker()

    # ── CSV logger ───────────────────────────────────────────────────────────
    log_path = out_dir / "train_log.csv"
    log_fields = ["epoch", "phase", "loss", "iou", "dice", "pixel_acc",
                  "precision", "recall", "f1", "lr", "elapsed_s"]

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val_iou = 0.0
    print(f"\n[4/5] Training for {args.epochs} epochs …\n")
    print("=" * 65)

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # ── Phase transition ─────────────────────────────────────────────────
        if epoch == args.freeze_epochs + 1:
            model.unfreeze_encoder()
            # Rebuild optimiser with differential LR
            optimizer = AdamW(
                [
                    {"params": model.get_decoder_params(), "lr": args.lr_decoder},
                    {"params": model.get_encoder_params(), "lr": args.lr_encoder},
                ],
                weight_decay=1e-4,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-6)
            print(f"\n  ✦ Phase 2: encoder unfrozen (decoder LR={args.lr_decoder}, encoder LR={args.lr_encoder})\n")

        # ── Train ────────────────────────────────────────────────────────────
        train_loss, train_metrics, _, _, _ = run_epoch(
            model, train_loader, criterion, train_tracker,
            optimizer, scaler, device, phase="train",
        )
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────────
        val_loss, val_metrics, last_imgs, last_masks, last_logits = run_epoch(
            model, val_loader, criterion, val_tracker,
            None, None, device, phase="val",
        )

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Console summary ───────────────────────────────────────────────────
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  "
            f"Val IoU: {val_metrics['iou']:.4f}  "
            f"Val Dice: {val_metrics['dice']:.4f}  "
            f"LR: {current_lr:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        # ── CSV log ───────────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            for phase, loss, mets in [
                ("train", train_loss, train_metrics),
                ("val",   val_loss,   val_metrics),
            ]:
                writer.writerow({
                    "epoch": epoch, "phase": phase, "loss": round(loss, 6),
                    "iou":       round(mets.get("iou",       0), 6),
                    "dice":      round(mets.get("dice",      0), 6),
                    "pixel_acc": round(mets.get("pixel_acc", 0), 6),
                    "precision": round(mets.get("precision", 0), 6),
                    "recall":    round(mets.get("recall",    0), 6),
                    "f1":        round(mets.get("f1",        0), 6),
                    "lr":        current_lr,
                    "elapsed_s": round(elapsed, 2),
                })

        # ── Checkpoints ───────────────────────────────────────────────────────
        # Always save latest
        save_checkpoint(model, optimizer, epoch, val_metrics, out_dir / "last_model.pth")

        # Save best val IoU
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            save_checkpoint(model, optimizer, epoch, val_metrics, out_dir / "best_model.pth")
            print(f"  ★ New best val IoU: {best_val_iou:.4f} → best_model.pth")

        # ── Visualisation ─────────────────────────────────────────────────────
        if epoch % args.viz_every == 0 or epoch == 1 or epoch == args.epochs:
            save_viz(last_imgs, last_masks, last_logits, epoch, viz_dir)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"[5/5] Training complete.")
    print(f"  Best val IoU : {best_val_iou:.4f}")
    print(f"  Best model   : {out_dir / 'best_model.pth'}")
    print(f"  Training log : {log_path}")
    print(f"  Viz grids    : {viz_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    train(args)
