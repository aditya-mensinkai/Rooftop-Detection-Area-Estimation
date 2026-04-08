"""
rooftop_dataset.py — PyTorch Dataset for Rooftop Segmentation
SolarSense Platform | IEEE YESIST12 WePOWER Track 2026

Dataset structure expected:
    dataset/
    ├── images/   *.tif   (256×256, RGB satellite images)
    └── masks/    *.npy   (256×256, binary 0/1 masks)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _random_hflip(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask  = np.fliplr(mask)
    return image, mask


def _random_vflip(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask  = np.flipud(mask)
    return image, mask


def _random_rotate90(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    k = random.choice([0, 1, 2, 3])
    if k > 0:
        image = TF.rotate(image, angle=k * 90)
        mask  = np.rot90(mask, k=k)
    return image, mask


def _color_jitter(image: Image.Image) -> Image.Image:
    """Mild colour jitter to simulate different imaging conditions."""
    image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
    image = TF.adjust_contrast(image,   contrast_factor=random.uniform(0.8, 1.2))
    image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))
    return image


def _random_crop(
    image: Image.Image,
    mask: np.ndarray,
    target_size: int,
    min_scale: float = 0.75,
) -> tuple[Image.Image, np.ndarray]:
    """Random crop then resize back to target_size — simulates zoom variation."""
    w, h = image.size
    scale = random.uniform(min_scale, 1.0)
    crop_w, crop_h = int(w * scale), int(h * scale)
    x0 = random.randint(0, w - crop_w)
    y0 = random.randint(0, h - crop_h)
    image = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    image = image.resize((target_size, target_size), Image.BILINEAR)
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    mask = np.array(mask_img.resize((target_size, target_size), Image.NEAREST))
    return image, mask


def _gaussian_noise(image: Image.Image, std: float = 0.02) -> Image.Image:
    """Add Gaussian noise to simulate sensor noise in satellite imagery."""
    arr = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _gaussian_blur(image: Image.Image) -> Image.Image:
    """Mild Gaussian blur to simulate atmospheric scattering."""
    radius = random.uniform(0.5, 1.5)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _random_shadow(image: Image.Image) -> Image.Image:
    """Simulate cloud / building shadow as a random dark polygon strip."""
    arr = np.array(image).astype(np.float32)
    w, h = image.size
    # Random horizontal or vertical shadow band
    if random.random() > 0.5:
        x0 = random.randint(0, w - 1)
        width = random.randint(w // 8, w // 3)
        factor = random.uniform(0.5, 0.8)
        arr[:, x0:x0 + width] *= factor
    else:
        y0 = random.randint(0, h - 1)
        height = random.randint(h // 8, h // 3)
        factor = random.uniform(0.5, 0.8)
        arr[y0:y0 + height, :] *= factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# ImageNet normalisation stats
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class RooftopDataset(Dataset):
    """
    Dataset for paired satellite images and binary rooftop masks.

    Args:
        image_dir   : Directory containing .tif images.
        mask_dir    : Directory containing .npy binary masks.
        file_stems  : Optional list of file stems (without extension) to use.
                      If None, all images in image_dir are used.
        augment     : Enable training-time augmentation.
        target_size : Resize images/masks to this square size.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        file_stems: list[str] | None = None,
        augment: bool = False,
        target_size: int = 256,
        gsd: float = 0.5,
    ) -> None:
        self.image_dir   = Path(image_dir)
        self.mask_dir    = Path(mask_dir)
        self.augment     = augment
        self.target_size = target_size
        self.gsd         = gsd

        if file_stems is not None:
            self.stems = file_stems
        else:
            self.stems = []
            for p in sorted(self.image_dir.glob("*.tif")):
                mask_path = self.mask_dir / f"{p.stem}.npy"
                if mask_path.exists():
                    self.stems.append(p.stem)

        if not self.stems:
            raise ValueError("No valid image-mask pairs found!")

        print(f"[Dataset] Loaded {len(self.stems)} valid pairs")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys:
                'image' : (3, H, W) float tensor, ImageNet-normalised.
                'mask'  : (1, H, W) float tensor with values 0 or 1.
                'stem'  : filename stem (str) for tracking.
        """
        stem = self.stems[idx]

        # ── Load image ──────────────────────────────────────────────────────
        img_path = self.image_dir / f"{stem}.tif"
        image = Image.open(img_path).convert("RGB")
        if image.size != (self.target_size, self.target_size):
            image = image.resize((self.target_size, self.target_size), Image.BILINEAR)

        # ── Load mask ───────────────────────────────────────────────────────
        mask_path = self.mask_dir / f"{stem}.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = np.load(str(mask_path)).astype(np.float32)
        if mask.shape != (self.target_size, self.target_size):
            mask = np.array(
                Image.fromarray(mask).resize(
                    (self.target_size, self.target_size), Image.NEAREST
                )
            )
        mask = (mask > 0.5).astype(np.float32)   # ensure binary

        if mask.sum() == 0:
            # log but allow (do not crash)
            print(f"[Warning] Empty mask: {mask_path.name}")

        # ── Augmentation ────────────────────────────────────────────────────
        if self.augment:
            image, mask = _random_hflip(image, mask)
            image, mask = _random_vflip(image, mask)
            image, mask = _random_rotate90(image, mask)
            image = _color_jitter(image)
            if random.random() > 0.5:
                image, mask = _random_crop(image, mask, self.target_size)
            if random.random() > 0.5:
                image = _gaussian_noise(image)
            if random.random() > 0.5:
                image = _gaussian_blur(image)
            if random.random() > 0.3:
                image = _random_shadow(image)

        # ── To tensor & normalise ────────────────────────────────────────────
        image_t = TF.to_tensor(image)                    # [0, 1], (3, H, W)
        image_t = TF.normalize(image_t, _MEAN, _STD)

        mask = np.ascontiguousarray(mask)
        mask_t = torch.from_numpy(mask).unsqueeze(0)     # (1, H, W)

        return {"image": image_t, "mask": mask_t, "stem": stem, "gsd": self.gsd}


# ---------------------------------------------------------------------------
# Train / val split helper
# ---------------------------------------------------------------------------

def split_dataset(
    image_dir: str | Path,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Randomly split file stems into train and val lists.

    Args:
        image_dir : Directory with .tif images.
        val_split : Fraction to use for validation (0 < val_split < 1).
        seed      : Random seed for reproducibility.

    Returns:
        (train_stems, val_stems) lists of file stems.
    """
    stems = sorted(p.stem for p in Path(image_dir).glob("*.tif"))
    rng   = random.Random(seed)
    rng.shuffle(stems)

    n_val  = max(1, int(len(stems) * val_split))
    val_stems   = stems[:n_val]
    train_stems = stems[n_val:]
    return train_stems, val_stems


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    image_dir = "dataset/images"
    mask_dir  = "dataset/masks"

    try:
        train_stems, val_stems = split_dataset(image_dir, val_split=0.15)
        print(f"Train samples : {len(train_stems)}")
        print(f"Val   samples : {len(val_stems)}")

        ds = RooftopDataset(image_dir, mask_dir, train_stems, augment=True)
        sample = ds[0]
        print(f"Image shape   : {sample['image'].shape}")
        print(f"Mask  shape   : {sample['mask'].shape}")
        print(f"Mask  unique  : {sample['mask'].unique().tolist()}")
        print("Dataset OK ✓")

    except FileNotFoundError as e:
        print(f"[dataset.py] Dataset not found — {e}", file=sys.stderr)
        print("Create the dataset directory structure and re-run.")
