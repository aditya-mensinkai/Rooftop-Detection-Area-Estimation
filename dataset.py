#!/usr/bin/env python3
"""
Production-Ready PyTorch Dataset for Rooftop Segmentation

SpaceNet Dataset Pipeline - Prepares .tif images and .npy masks for U-Net training.

Features:
- Strict filename matching (image.tif ↔ mask.npy)
- Clean logging with [INFO]/[WARNING]/[ERROR] prefixes
- Dataset validation with auto-detection of wrong directories
- Empty mask warnings (kept for training)
- 3-panel visualization (image, mask, overlay)
- Synchronized augmentation (same transforms for image & mask)

Usage:
    # Basic usage
    from dataset import RoofDataset, get_dataloader

    dataset = RoofDataset(
        image_dir='dataset_final/images',
        mask_dir='dataset_final/masks',
        target_size=256
    )
    train_loader = get_dataloader(dataset, batch_size=8)

    # With augmentation
    dataset = RoofDataset(
        image_dir='dataset_final/images',
        mask_dir='dataset_final/masks',
        target_size=256,
        augment=True
    )

    # CLI testing with visualization
    python dataset.py --image_dir dataset_final/images --mask_dir dataset_final/masks --visualize
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

# Configure logging with clean format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# IMAGE LOADING
# ============================================================================

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file.

    Uses rasterio for .tif files, PIL for other formats.

    Args:
        image_path: Path to image file

    Returns:
        Image array (H, W, 3), uint8, values in [0, 255]
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

    # Use rasterio for TIFF files
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        return _load_tiff(image_path)
    else:
        return _load_pil_image(image_path)


def _load_tiff(image_path: Path) -> np.ndarray:
    """Load TIFF image using rasterio."""
    try:
        import rasterio

        with rasterio.open(image_path) as src:
            if src.count >= 3:
                rgb = np.dstack([src.read(i) for i in range(1, 4)])
            elif src.count == 1:
                gray = src.read(1)
                rgb = np.stack([gray] * 3, axis=-1)
            else:
                raise ValueError(f"[ERROR] Unsupported band count: {src.count}")

            # Ensure uint8
            if rgb.dtype != np.uint8:
                if rgb.max() > 255:
                    rgb = (rgb / rgb.max() * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)

            return rgb

    except ImportError:
        raise ImportError("[ERROR] rasterio required. Install: pip install rasterio")


def _load_pil_image(image_path: Path) -> np.ndarray:
    """Load image using PIL."""
    try:
        from PIL import Image

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img, dtype=np.uint8)

    except Exception as e:
        raise ValueError(f"[ERROR] Failed to load image {image_path}: {e}")


# ============================================================================
# MASK LOADING
# ============================================================================

def load_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """
    Load binary mask from file.

    Supports .npy (numpy) and .png formats.

    Args:
        mask_path: Path to mask file

    Returns:
        Binary mask array (H, W), values in {0, 1} as float32
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"[ERROR] Mask not found: {mask_path}")

    if mask_path.suffix == '.npy':
        return _load_npy_mask(mask_path)
    elif mask_path.suffix == '.png':
        return _load_png_mask(mask_path)
    else:
        raise ValueError(f"[ERROR] Unsupported mask format: {mask_path.suffix}")


def _load_npy_mask(mask_path: Path) -> np.ndarray:
    """Load mask from numpy file."""
    try:
        mask = np.load(mask_path)

        # Ensure 2D
        if len(mask.shape) > 2:
            mask = mask.squeeze()

        # Ensure binary (0 or 1) as float32
        mask = (mask > 0).astype(np.float32)

        return mask

    except Exception as e:
        raise ValueError(f"[ERROR] Corrupt .npy file: {mask_path} - {e}")


def _load_png_mask(mask_path: Path) -> np.ndarray:
    """Load mask from PNG file."""
    try:
        from PIL import Image

        img = Image.open(mask_path).convert('L')
        mask = np.array(img, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        return mask

    except Exception as e:
        raise ValueError(f"[ERROR] Failed to load mask {mask_path}: {e}")


# ============================================================================
# PREPROCESSING
# ============================================================================

def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Union[int, Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image and mask to target size."""
    try:
        import cv2

        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        # Resize image with linear interpolation
        img_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # Resize mask with nearest neighbor (preserve binary values)
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        return img_resized, mask_resized

    except ImportError:
        # Fallback to PIL
        from PIL import Image

        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        img_pil = Image.fromarray(image)
        img_resized = np.array(img_pil.resize(target_size, Image.BILINEAR))

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_resized = np.array(mask_pil.resize(target_size, Image.NEAREST))
        mask_resized = (mask_resized > 127).astype(np.float32)

        return img_resized, mask_resized


def preprocess_image(image: np.ndarray, target_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Preprocess image for model input.

    Args:
        image: Input image (H, W, 3), uint8
        target_size: Target size for resizing

    Returns:
        Preprocessed image (3, H, W), float32, values in [0, 1]
    """
    # Resize
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    image_resized, _ = resize_image_and_mask(image, np.zeros_like(image[:, :, 0]), target_size)

    # Normalize to [0, 1]
    image_norm = image_resized.astype(np.float32) / 255.0

    # Convert HWC to CHW
    image_tensor = np.transpose(image_norm, (2, 0, 1))

    return image_tensor


def preprocess_mask(mask: np.ndarray, target_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Preprocess mask for model input.

    Args:
        mask: Input mask (H, W), values in {0, 1}
        target_size: Target size for resizing

    Returns:
        Preprocessed mask (1, H, W), float32, values in {0, 1}
    """
    # Resize with nearest neighbor
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    _, mask_resized = resize_image_and_mask(np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8), mask, target_size)

    # Ensure binary
    mask_binary = (mask_resized > 0.5).astype(np.float32)

    # Add channel dimension
    mask_tensor = mask_binary[np.newaxis, :, :]

    return mask_tensor


# ============================================================================
# AUGMENTATION
# ============================================================================

class Augmentation:
    """
    Data augmentation for image and mask.

    Applies same geometric transforms to both image and mask.
    Color transforms (brightness) only apply to image.
    """

    def __init__(self, horizontal_flip: bool = True, vertical_flip: bool = True,
                 rotation: bool = True, brightness: bool = True, p: float = 0.5):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.brightness = brightness
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to image and mask."""
        import random

        # Horizontal flip (geometric - apply to both)
        if self.horizontal_flip and random.random() < self.p:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Vertical flip (geometric - apply to both)
        if self.vertical_flip and random.random() < self.p:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # Rotation (geometric - apply to both)
        if self.rotation and random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
            image = np.rot90(image, k=k).copy()
            mask = np.rot90(mask, k=k).copy()

        # Brightness (color - image only)
        if self.brightness and random.random() < self.p:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        return image, mask


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class RoofDataset:
    """
    Production-ready PyTorch Dataset for rooftop segmentation.

    Features strict filename matching and comprehensive validation.

    Args:
        image_dir: Directory containing .tif images
        mask_dir: Directory containing .npy masks
        target_size: Target size for resizing (default: 256)
        augment: Whether to apply data augmentation (default: False)
        transform: Optional additional transforms
        validate: Whether to validate dataset on init (default: True)

    Returns:
        Tuple of (image_tensor, mask_tensor) where:
        - image_tensor: (3, H, W), float32, values in [0, 1]
        - mask_tensor: (1, H, W), float32, values in {0, 1}
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        target_size: Union[int, Tuple[int, int]] = 256,
        augment: bool = False,
        transform: Optional[Callable] = None,
        validate: bool = True
    ):
        """Initialize dataset with strict file matching."""
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.target_size = target_size
        self.transform = transform
        self.augment = Augmentation() if augment else None

        # Validate directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"[ERROR] Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"[ERROR] Mask directory not found: {self.mask_dir}")

        # Build valid pairs using strict filename matching
        self.pairs = self._build_valid_pairs()

        # Dataset validation
        if len(self.pairs) < 100:
            self._handle_small_dataset()

        # Log summary
        self._log_summary(augment)

        # Optional validation
        if validate:
            self._validate_dataset()

    def _build_valid_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Build list of valid image-mask pairs using strict filename matching.

        Only includes pairs where BOTH image.tif and mask.npy exist.
        """
        # Get all image files (.tif only for SpaceNet)
        image_files = {}
        for ext in ['.tif', '.tiff']:
            for f in self.image_dir.glob(f'*{ext}'):
                image_files[f.stem] = f

        # Get all mask files (.npy only)
        mask_files = {}
        for f in self.mask_dir.glob('*.npy'):
            mask_files[f.stem] = f

        # Find intersection (strict matching)
        valid_stems = set(image_files.keys()) & set(mask_files.keys())

        # Build sorted pairs list
        pairs = [(image_files[stem], mask_files[stem]) for stem in sorted(valid_stems)]

        # Store counts for summary
        self._total_images = len(image_files)
        self._total_masks = len(mask_files)
        self._unmatched_images = set(image_files.keys()) - set(mask_files.keys())
        self._unmatched_masks = set(mask_files.keys()) - set(image_files.keys())

        return pairs

    def _handle_small_dataset(self):
        """Handle case where dataset has fewer than 100 samples."""
        logger.error(f"[ERROR] Dataset too small! Only {len(self.pairs)} valid pairs found.")
        logger.error(f"[ERROR] Expected: thousands of samples for training.")
        logger.error(f"[ERROR] Image directory: {self.image_dir}")
        logger.error(f"[ERROR] Mask directory: {self.mask_dir}")

        # Check for dataset_final
        parent = self.image_dir.parent
        if parent.name != 'dataset_final':
            alt_path = parent / 'dataset_final'
            if alt_path.exists():
                logger.info(f"[INFO] Found alternative: {alt_path}")
                logger.info(f"[INFO] Try: --image_dir {alt_path}/images --mask_dir {alt_path}/masks")

        # Log unmatched files
        if self._unmatched_images:
            logger.warning(f"[WARNING] {len(self._unmatched_images)} images without masks")
            logger.warning(f"[WARNING] Examples: {list(self._unmatched_images)[:3]}")
        if self._unmatched_masks:
            logger.warning(f"[WARNING] {len(self._unmatched_masks)} masks without images")
            logger.warning(f"[WARNING] Examples: {list(self._unmatched_masks)[:3]}")

        raise ValueError(f"[ERROR] Dataset too small ({len(self.pairs)} samples) — check directory paths")

    def _log_summary(self, augment: bool):
        """Log dataset initialization summary."""
        logger.info(f"[INFO] Total images: {self._total_images}")
        logger.info(f"[INFO] Total masks: {self._total_masks}")
        logger.info(f"[INFO] Valid pairs: {len(self.pairs)}")

        if self._unmatched_images:
            logger.warning(f"[WARNING] Images without masks: {len(self._unmatched_images)}")
        if self._unmatched_masks:
            logger.warning(f"[WARNING] Masks without images: {len(self._unmatched_masks)}")

        logger.info(f"[INFO] Image directory: {self.image_dir}")
        logger.info(f"[INFO] Mask directory: {self.mask_dir}")
        logger.info(f"[INFO] Target size: {self.target_size}")
        logger.info(f"[INFO] Augmentation: {'enabled' if augment else 'disabled'}")
        logger.info(f"[INFO] Dataset initialized successfully")

    def _validate_dataset(self):
        """Validate dataset by loading first sample."""
        if len(self.pairs) == 0:
            raise ValueError("[ERROR] No valid image-mask pairs found")

        try:
            sample_image, sample_mask = self[0]
            assert sample_image.shape[0] == 3, f"Image should have 3 channels"
            assert sample_mask.shape[0] == 1, f"Mask should have 1 channel"
            assert sample_image.shape[1:] == sample_mask.shape[1:], "Shapes should match"
            logger.info(f"[INFO] Dataset validation passed")
        except Exception as e:
            logger.error(f"[ERROR] Dataset validation failed: {e}")
            raise

    def __len__(self) -> int:
        """Return number of valid pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        if idx >= len(self):
            raise IndexError(f"[ERROR] Index {idx} out of range")

        image_path, mask_path = self.pairs[idx]

        # Load image
        try:
            image = load_image(image_path)
        except Exception as e:
            logger.error(f"[ERROR] Failed to load image {image_path}: {e}")
            raise

        # Load mask
        try:
            mask = load_mask(mask_path)
        except Exception as e:
            logger.error(f"[ERROR] Failed to load mask {mask_path}: {e}")
            raise

        # Validate alignment
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"[ERROR] Shape mismatch: image={image.shape[:2]}, mask={mask.shape[:2]}")

        # Check for empty mask
        if mask.sum() == 0:
            logger.warning(f"[WARNING] Empty mask: {mask_path.name}")

        # Apply augmentation (before resizing for quality)
        if self.augment is not None:
            try:
                image, mask = self.augment(image, mask)
            except Exception as e:
                logger.error(f"[ERROR] Augmentation failed: {e}")
                raise

        # Preprocess
        try:
            image_tensor = preprocess_image(image, self.target_size)
            mask_tensor = preprocess_mask(mask, self.target_size)
        except Exception as e:
            logger.error(f"[ERROR] Preprocessing failed: {e}")
            raise

        # Apply additional transforms if provided
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor


# ============================================================================
# DATALOADER
# ============================================================================

def get_dataloader(
    dataset: RoofDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    **kwargs
):
    """
    Create PyTorch DataLoader.

    Args:
        dataset: RoofDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        pin_memory: Pin memory for GPU
        **kwargs: Additional arguments

    Returns:
        DataLoader instance
    """
    try:
        from torch.utils.data import DataLoader

        if len(dataset) < batch_size:
            logger.warning(f"[WARNING] Dataset size ({len(dataset)}) < batch_size ({batch_size})")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )

    except ImportError:
        raise ImportError("[ERROR] PyTorch required. Install: pip install torch")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_sample(dataset: RoofDataset, idx: int = 0, save_path: Optional[str] = None):
    """
    Create 3-panel visualization: image, mask, overlay.

    Args:
        dataset: RoofDataset instance
        idx: Sample index
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("[ERROR] matplotlib required. Install: pip install matplotlib")
        return

    if idx >= len(dataset):
        logger.error(f"[ERROR] Index {idx} out of range")
        return

    # Get sample
    image_tensor, mask_tensor = dataset[idx]
    image_path, mask_path = dataset.pairs[idx]

    # Convert back for visualization
    # Image: (3, H, W) -> (H, W, 3)
    image = np.transpose(image_tensor, (1, 2, 0))
    # Mask: (1, H, W) -> (H, W)
    mask = mask_tensor[0]

    # Create overlay (red mask on image)
    overlay = image.copy()
    overlay[mask > 0.5] = [1.0, 0.0, 0.0]  # Red

    # Calculate coverage
    coverage = mask.sum() / mask.size * 100
    pixel_count = int(mask.sum())

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Image: {image_path.name}")
    axes[0].axis('off')

    # Binary mask
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Mask: {mask_path.name}")
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nCoverage: {coverage:.1f}% | Pixels: {pixel_count}")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"[INFO] Saved visualization to: {save_path}")

    plt.show()

    # Print statistics
    logger.info(f"[INFO] Sample {idx}: {image_path.name}")
    logger.info(f"[INFO]   Image shape: {image.shape}")
    logger.info(f"[INFO]   Mask shape: {mask.shape}")
    logger.info(f"[INFO]   Coverage: {coverage:.1f}%")
    logger.info(f"[INFO]   Building pixels: {pixel_count}")


# ============================================================================
# MAIN / CLI
# ============================================================================

def main():
    """Main function for dataset testing."""
    parser = argparse.ArgumentParser(
        description='Production-Ready RoofDataset Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test
    python dataset.py --image_dir dataset_final/images --mask_dir dataset_final/masks

    # With visualization
    python dataset.py --image_dir dataset_final/images --mask_dir dataset_final/masks --visualize

    # With augmentation
    python dataset.py --image_dir dataset_final/images --mask_dir dataset_final/masks --augment

    # Batch test
    python dataset.py --image_dir dataset_final/images --mask_dir dataset_final/masks --batch_size 8
        """
    )

    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing .tif images')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory containing .npy masks')
    parser.add_argument('--target_size', type=int, default=256,
                        help='Target size (default: 256)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable augmentation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (default: 0)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show 3-panel visualization')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Save visualization to path')
    parser.add_argument('--test_idx', type=int, default=0,
                        help='Sample index to test (default: 0)')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("ROOFTOP SEGMENTATION DATASET - PRODUCTION PIPELINE")
    logger.info("=" * 70)

    # ============================================================
    # TEST 1: Dataset Initialization
    # ============================================================
    logger.info("")
    logger.info("[INFO] Test 1: Dataset Initialization")
    logger.info("-" * 70)

    try:
        dataset = RoofDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            target_size=args.target_size,
            augment=args.augment,
            validate=True
        )
    except Exception as e:
        logger.error(f"[ERROR] Dataset initialization failed: {e}")
        sys.exit(1)

    # CLI safety check
    if len(dataset) < 100:
        logger.error(f"[ERROR] Dataset too small ({len(dataset)} samples)!")
        logger.error(f"[ERROR] Check if you meant dataset_final instead of dataset")
        sys.exit(1)

    # ============================================================
    # TEST 2: Single Sample Loading
    # ============================================================
    logger.info("")
    logger.info("[INFO] Test 2: Single Sample Loading")
    logger.info("-" * 70)

    try:
        image, mask = dataset[args.test_idx]
        logger.info(f"[INFO] Sample {args.test_idx}:")
        logger.info(f"[INFO]   Image shape: {image.shape}, dtype: {image.dtype}")
        logger.info(f"[INFO]   Mask shape: {mask.shape}, dtype: {mask.dtype}")
        logger.info(f"[INFO]   Image range: [{image.min():.3f}, {image.max():.3f}]")
        logger.info(f"[INFO]   Mask coverage: {mask.sum()/mask.size*100:.1f}%")
        logger.info(f"[INFO] ✓ Single sample loading passed")
    except Exception as e:
        logger.error(f"[ERROR] Single sample loading failed: {e}")
        sys.exit(1)

    # ============================================================
    # TEST 3: Visualization
    # ============================================================
    if args.visualize:
        logger.info("")
        logger.info("[INFO] Test 3: Visualization")
        logger.info("-" * 70)

        try:
            visualize_sample(dataset, idx=args.test_idx, save_path=args.save_path)
            logger.info(f"[INFO] ✓ Visualization passed")
        except Exception as e:
            logger.error(f"[ERROR] Visualization failed: {e}")
            sys.exit(1)

    # ============================================================
    # TEST 4: DataLoader Test
    # ============================================================
    logger.info("")
    logger.info("[INFO] Test 4: DataLoader Test")
    logger.info("-" * 70)

    try:
        dataloader = get_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        batch_count = 0
        for batch_images, batch_masks in dataloader:
            batch_count += 1

            coverage = batch_masks.sum() / batch_masks.numel() * 100
            logger.info(f"[INFO] Batch {batch_count}:")
            logger.info(f"[INFO]   Images shape: {batch_images.shape}")
            logger.info(f"[INFO]   Masks shape: {batch_masks.shape}")
            logger.info(f"[INFO]   Coverage: {coverage:.1f}%")

            if batch_count >= 2:
                break

        logger.info(f"[INFO] ✓ DataLoader test passed ({batch_count} batches)")

    except Exception as e:
        logger.error(f"[ERROR] DataLoader test failed: {e}")
        sys.exit(1)

    # ============================================================
    # Summary
    # ============================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 70)
    logger.info(f"[INFO] Dataset ready for training with {len(dataset)} samples")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
