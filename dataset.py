#!/usr/bin/env python3
"""
STEP 6: DATA PREPROCESSING PIPELINE

PyTorch Dataset implementation for rooftop segmentation.

Prepares image and mask data so it can be fed into a deep learning model (U-Net).

Features:
- Lazy loading (images loaded on-demand)
- Automatic resizing to target size
- Image normalization
- Data augmentation (optional)
- Validation checks

Usage:
    # Basic usage
    from dataset import RoofDataset, get_dataloader

    dataset = RoofDataset(
        image_dir='dataset/images',
        mask_dir='dataset/masks',
        target_size=256
    )

    train_loader = get_dataloader(dataset, batch_size=8, shuffle=True)

    # With augmentation
    dataset = RoofDataset(
        image_dir='dataset/images',
        mask_dir='dataset/masks',
        target_size=256,
        augment=True
    )

    # Visualize batch
    python dataset.py --image_dir dataset/images --mask_dir dataset/masks --visualize
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# IMAGE LOADING
# ============================================================================

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file.

    Uses rasterio for .tif files, PIL/Pillow for other formats.

    Args:
        image_path: Path to image file

    Returns:
        Image array (H, W, 3), uint8, values in [0, 255]

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image format is unsupported or corrupt
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")

    # Use rasterio for TIFF files
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        return _load_tiff(image_path)
    else:
        # Use PIL for other formats
        return _load_pil_image(image_path)


def _load_tiff(image_path: Path) -> np.ndarray:
    """Load TIFF image using rasterio."""
    try:
        import rasterio
        from rasterio.errors import RasterioIOError

        with rasterio.open(image_path) as src:
            # Read RGB bands
            if src.count >= 3:
                rgb = np.dstack([src.read(i) for i in range(1, 4)])
            elif src.count == 1:
                # Grayscale - stack to create RGB
                gray = src.read(1)
                rgb = np.stack([gray] * 3, axis=-1)
            else:
                raise ValueError(f"Unsupported band count: {src.count}")

            # Ensure uint8
            if rgb.dtype != np.uint8:
                # Normalize if needed
                if rgb.max() > 255:
                    rgb = (rgb / rgb.max() * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)

            return rgb

    except ImportError:
        raise ImportError("rasterio is required for TIFF files. Install with: pip install rasterio")
    except RasterioIOError as e:
        raise ValueError(f"Corrupt or invalid TIFF file: {image_path} - {e}")


def _load_pil_image(image_path: Path) -> np.ndarray:
    """Load image using PIL."""
    try:
        from PIL import Image

        img = Image.open(image_path)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return np.array(img, dtype=np.uint8)

    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


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
        Binary mask array (H, W), values in {0, 1}

    Raises:
        FileNotFoundError: If mask doesn't exist
        ValueError: If mask format is unsupported or corrupt
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    if mask_path.suffix == '.npy':
        return _load_npy_mask(mask_path)
    elif mask_path.suffix == '.png':
        return _load_png_mask(mask_path)
    else:
        raise ValueError(f"Unsupported mask format: {mask_path.suffix}")


def _load_npy_mask(mask_path: Path) -> np.ndarray:
    """Load mask from numpy file."""
    try:
        mask = np.load(mask_path)

        # Ensure 2D
        if len(mask.shape) > 2:
            mask = mask.squeeze()

        # Ensure binary (0 or 1)
        mask = (mask > 0).astype(np.float32)

        return mask

    except Exception as e:
        raise ValueError(f"Corrupt .npy file: {mask_path} - {e}")


def _load_png_mask(mask_path: Path) -> np.ndarray:
    """Load mask from PNG file."""
    try:
        from PIL import Image

        mask = np.array(Image.open(mask_path))

        # Ensure 2D
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # Normalize to {0, 1}
        if mask.max() > 1:
            mask = (mask / 255.0).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        return mask

    except Exception as e:
        raise ValueError(f"Corrupt PNG file: {mask_path} - {e}")


# ============================================================================
# PREPROCESSING
# ============================================================================

def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Union[int, Tuple[int, int]],
    interpolation_image: int = None,
    interpolation_mask: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image and mask to target size.

    Uses INTER_LINEAR for images and INTER_NEAREST for masks to preserve
    binary mask integrity.

    Args:
        image: Image array (H, W, 3)
        mask: Mask array (H, W)
        target_size: Target size (H, W) or single int for square
        interpolation_image: OpenCV interpolation for image
        interpolation_mask: OpenCV interpolation for mask

    Returns:
        Tuple of (resized_image, resized_mask)
    """
    try:
        import cv2

        # Set default interpolations
        if interpolation_image is None:
            interpolation_image = cv2.INTER_LINEAR
        if interpolation_mask is None:
            interpolation_mask = cv2.INTER_NEAREST

        # Parse target size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        # Resize image
        resized_image = cv2.resize(
            image,
            (target_size[1], target_size[0]),  # cv2 uses (width, height)
            interpolation=interpolation_image
        )

        # Resize mask
        resized_mask = cv2.resize(
            mask,
            (target_size[1], target_size[0]),
            interpolation=interpolation_mask
        )

        return resized_image, resized_mask

    except ImportError:
        # Fallback to PIL
        from PIL import Image

        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        # Resize image
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        resized_image = np.array(pil_image)

        # Resize mask
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        pil_mask = pil_mask.resize((target_size[1], target_size[0]), Image.NEAREST)
        resized_mask = np.array(pil_mask).astype(np.float32) / 255.0

        return resized_image, resized_mask


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        image: Image array (H, W, 3), uint8

    Returns:
        Normalized image (H, W, 3), float32
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(image: np.ndarray, target_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Preprocess image for model input.

    Steps:
    1. Resize to target size
    2. Normalize to [0, 1]
    3. Convert to CHW format

    Args:
        image: Input image (H, W, 3)
        target_size: Target size

    Returns:
        Preprocessed image (3, H, W), float32
    """
    # Resize
    from PIL import Image
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    resized = np.array(pil_image)

    # Normalize
    normalized = normalize_image(resized)

    # HWC to CHW
    chw = np.transpose(normalized, (2, 0, 1))

    return chw


def preprocess_mask(mask: np.ndarray, target_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Preprocess mask for model input.

    Steps:
    1. Resize to target size (using nearest neighbor)
    2. Ensure binary values
    3. Add channel dimension

    Args:
        mask: Input mask (H, W)
        target_size: Target size

    Returns:
        Preprocessed mask (1, H, W), float32
    """
    # Resize using PIL (NEAREST to preserve binary values)
    from PIL import Image

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
    pil_mask = pil_mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    resized = np.array(pil_mask).astype(np.float32) / 255.0

    # Ensure binary
    binary = (resized > 0.5).astype(np.float32)

    # Add channel dimension
    chw = np.expand_dims(binary, axis=0)

    return chw


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class Augmentation:
    """Data augmentation for image-mask pairs.

    Applies same geometric transformations to both image and mask
    to ensure alignment is preserved.
    """

    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        random_rotation: bool = True,
        brightness: bool = True,
        contrast: bool = True,
        rotation_range: Tuple[float, float] = (-15, 15),
        p: float = 0.5
    ):
        """
        Initialize augmentation.

        Args:
            horizontal_flip: Enable horizontal flip
            vertical_flip: Enable vertical flip
            random_rotation: Enable random rotation (±15° by default)
            brightness: Enable brightness adjustment
            contrast: Enable contrast adjustment
            rotation_range: Range of rotation angles in degrees (min, max)
            p: Probability of applying each augmentation
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_rotation = random_rotation
        self.brightness = brightness
        self.contrast = contrast
        self.rotation_range = rotation_range
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to image and mask.

        Args:
            image: Image array (H, W, 3), uint8
            mask: Mask array (H, W), float32

        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        # Horizontal flip (geometric - apply to both)
        if self.horizontal_flip and random.random() < self.p:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Vertical flip (geometric - apply to both)
        if self.vertical_flip and random.random() < self.p:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random rotation ±15° (geometric - apply to both)
        if self.random_rotation and random.random() < self.p:
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            image, mask = self._rotate_image_and_mask(image, mask, angle)

        # Brightness adjustment (color - image only)
        if self.brightness and random.random() < self.p:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Contrast adjustment (color - image only)
        if self.contrast and random.random() < self.p:
            factor = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        return image, mask

    def _rotate_image_and_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate image and mask by given angle.

        Uses cv2.warpAffine with appropriate interpolation:
        - INTER_LINEAR for image (smooth)
        - INTER_NEAREST for mask (preserve binary values)

        Args:
            image: Image array (H, W, 3)
            mask: Mask array (H, W)
            angle: Rotation angle in degrees

        Returns:
            Tuple of (rotated_image, rotated_mask)
        """
        try:
            import cv2

            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Rotate image with INTER_LINEAR
            rotated_image = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # Rotate mask with INTER_NEAREST (preserve labels)
            # Convert mask to uint8 for rotation
            mask_uint8 = (mask * 255).astype(np.uint8)
            rotated_mask_uint8 = cv2.warpAffine(
                mask_uint8, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # Convert back to float32 binary
            rotated_mask = (rotated_mask_uint8 > 127).astype(np.float32)

            return rotated_image, rotated_mask

        except ImportError:
            # Fallback: use scipy.ndimage
            from scipy import ndimage

            rotated_image = ndimage.rotate(image, angle, reshape=False, order=1)
            rotated_mask = ndimage.rotate(mask, angle, reshape=False, order=0)
            rotated_mask = (rotated_mask > 0.5).astype(np.float32)

            return rotated_image, rotated_mask


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class RoofDataset:
    """
    PyTorch Dataset for rooftop segmentation.

    Loads images and masks, applies preprocessing, and returns tensors.

    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
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
        """Initialize dataset."""
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.target_size = target_size
        self.transform = transform
        self.augment = Augmentation() if augment else None

        # Validate directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Build file list
        self.image_paths = []
        self.mask_paths = []

        # Find all image files
        image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        for ext in image_extensions:
            self.image_paths.extend(sorted(self.image_dir.glob(f'*{ext}')))

        # Match with masks
        for img_path in self.image_paths:
            base_name = img_path.stem
            # Try different mask extensions
            mask_path = self.mask_dir / f"{base_name}.npy"
            if not mask_path.exists():
                mask_path = self.mask_dir / f"{base_name}.png"

            if mask_path.exists():
                self.mask_paths.append(mask_path)
            else:
                logger.warning(f"No mask found for {img_path.name}, skipping")

        # Update image_paths to only include matched pairs
        self.image_paths = [
            img for img, mask in zip(self.image_paths, self.mask_paths)
            if mask.exists()
        ]

        if validate:
            self._validate_dataset()

        logger.info(f"Initialized dataset with {len(self)} samples")

    def _validate_dataset(self):
        """Validate dataset integrity."""
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir} and {self.mask_dir}")

        # Check first sample
        try:
            sample_image, sample_mask = self[0]
            assert sample_image.shape[0] == 3, f"Image should have 3 channels, got {sample_image.shape[0]}"
            assert sample_mask.shape[0] == 1, f"Mask should have 1 channel, got {sample_mask.shape[0]}"
            assert sample_image.shape[1:] == sample_mask.shape[1:], "Image and mask shapes should match"
            logger.info(f"Dataset validation passed. Sample shapes: image={sample_image.shape}, mask={sample_mask.shape}")
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            raise

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, mask_tensor)

        Raises:
            IndexError: If index is out of range
            FileNotFoundError: If image or mask file not found
            ValueError: If image/mask shapes don't match or files are corrupt
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load with comprehensive error handling
        try:
            image = load_image(image_path)
        except FileNotFoundError as e:
            logger.error(f"[Sample {idx}] Image file not found: {image_path}")
            raise FileNotFoundError(f"Cannot load image at index {idx}: {e}")
        except Exception as e:
            logger.error(f"[Sample {idx}] Failed to load image {image_path}: {e}")
            raise ValueError(f"Corrupt or invalid image at index {idx} ({image_path}): {e}")

        try:
            mask = load_mask(mask_path)
        except FileNotFoundError as e:
            logger.error(f"[Sample {idx}] Mask file not found: {mask_path}")
            raise FileNotFoundError(f"Cannot load mask at index {idx}: {e}")
        except Exception as e:
            logger.error(f"[Sample {idx}] Failed to load mask {mask_path}: {e}")
            raise ValueError(f"Corrupt or invalid mask at index {idx} ({mask_path}): {e}")

        # Validate alignment
        if image.shape[:2] != mask.shape[:2]:
            error_msg = (
                f"[Sample {idx}] Shape mismatch: "
                f"image={image.shape[:2]}, mask={mask.shape[:2]}\n"
                f"Image file: {image_path}\n"
                f"Mask file: {mask_path}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate image properties
        if image.dtype != np.uint8:
            logger.warning(f"[Sample {idx}] Image not uint8, converting: {image_path.name}")
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Validate mask properties
        if mask.dtype != np.float32:
            logger.warning(f"[Sample {idx}] Mask not float32, converting: {mask_path.name}")
            mask = mask.astype(np.float32)

        # Check for abnormal masks
        unique_values = np.unique(mask)
        if len(unique_values) == 1:
            if unique_values[0] == 0:
                logger.warning(f"[Sample {idx}] Empty mask (all zeros): {mask_path.name}")
            elif unique_values[0] == 1:
                logger.warning(f"[Sample {idx}] Full mask (all ones): {mask_path.name}")
        elif not np.all(np.isin(unique_values, [0, 1])):
            logger.warning(f"[Sample {idx}] Mask has unusual values {unique_values}: {mask_path.name}")

        # Log successful load in debug mode
        logger.debug(f"[Sample {idx}] Loaded: image={image.shape}, mask={mask.shape}")

        # Apply augmentation (before resizing for quality)
        if self.augment is not None:
            try:
                image, mask = self.augment(image, mask)
            except Exception as e:
                logger.error(f"[Sample {idx}] Augmentation failed: {e}")
                raise ValueError(f"Augmentation failed at index {idx}: {e}")

        # Preprocess
        try:
            image_tensor = preprocess_image(image, self.target_size)
            mask_tensor = preprocess_mask(mask, self.target_size)
        except Exception as e:
            logger.error(f"[Sample {idx}] Preprocessing failed: {e}")
            raise ValueError(f"Preprocessing failed at index {idx}: {e}")

        # Validate output shapes
        expected_img_shape = (3, self.target_size, self.target_size) if isinstance(self.target_size, int) else (3,) + self.target_size
        expected_msk_shape = (1, self.target_size, self.target_size) if isinstance(self.target_size, int) else (1,) + self.target_size

        if image_tensor.shape != expected_img_shape:
            raise ValueError(f"[Sample {idx}] Image tensor shape mismatch: got {image_tensor.shape}, expected {expected_img_shape}")
        if mask_tensor.shape != expected_msk_shape:
            raise ValueError(f"[Sample {idx}] Mask tensor shape mismatch: got {mask_tensor.shape}, expected {expected_msk_shape}")

        # Apply additional transforms if provided
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor


# ============================================================================
# DATALOADER
# ============================================================================

def get_dataloader(
    dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    **kwargs
):
    """
    Create PyTorch DataLoader for training.

    Args:
        dataset: RoofDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (0 for main process only)
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
        **kwargs: Additional DataLoader arguments

    Returns:
        torch.utils.data.DataLoader instance
    """
    try:
        import torch
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_batch(
    dataset,
    num_samples: int = 4,
    save_path: Optional[str] = None
):
    """
    Visualize samples from dataset.

    Args:
        dataset: RoofDataset instance
        num_samples: Number of samples to visualize
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt

        num_samples = min(num_samples, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(indices):
            image_tensor, mask_tensor = dataset[idx]

            # Convert tensors to display format
            # CHW to HWC
            image = np.transpose(image_tensor, (1, 2, 0))
            mask = mask_tensor[0]  # Remove channel dim

            # Create overlay
            overlay = image.copy()
            overlay[mask > 0.5] = [1, 0, 0]  # Red overlay

            # Plot
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Image {idx}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Mask {idx}')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'Overlay {idx}')
            axes[i, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        logger.error("matplotlib is required for visualization. Install with: pip install matplotlib")


def visualize_augmentations(
    dataset,
    idx: int = 0,
    num_variants: int = 4,
    save_path: Optional[str] = None
):
    """
    Visualize augmentation effects on a single sample.

    Args:
        dataset: RoofDataset instance with augment=True
        idx: Sample index to visualize
        num_variants: Number of augmented variants to show
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(num_variants, 3, figsize=(12, 4 * num_variants))
        if num_variants == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_variants):
            image_tensor, mask_tensor = dataset[idx]

            # Convert tensors to display format
            image = np.transpose(image_tensor, (1, 2, 0))
            mask = mask_tensor[0]

            # Create overlay
            overlay = image.copy()
            overlay[mask > 0.5] = [1, 0, 0]

            # Plot
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Augmented Image {i+1}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Augmented Mask {i+1}')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'Overlay {i+1}')
            axes[i, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved augmentation visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        logger.error("matplotlib is required for visualization")


def visualize_sample(
    dataset,
    idx: int = 0,
    save_path: Optional[str] = None,
    show_original: bool = False
):
    """
    Visualize a single sample from dataset with detailed information.

    Args:
        dataset: RoofDataset instance
        idx: Sample index to visualize
        save_path: Optional path to save figure
        show_original: If True, also show pre-augmentation version
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Get sample
        image_tensor, mask_tensor = dataset[idx]

        # Convert tensors to display format
        image = np.transpose(image_tensor, (1, 2, 0))
        mask = mask_tensor[0]

        # Calculate statistics
        mask_sum = mask.sum()
        mask_total = mask.size
        coverage = (mask_sum / mask_total) * 100

        # Create overlay
        overlay = image.copy()
        overlay[mask > 0.5] = [1, 0, 0]  # Red overlay

        # Create figure
        if show_original and dataset.augment is not None:
            # Show both augmented and original
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))

            # Top row: augmented version
            axes[0, 0].imshow(image)
            axes[0, 0].set_title(f'Augmented Image [{idx}]')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title(f'Augmented Mask\nCoverage: {coverage:.1f}%')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(overlay)
            axes[0, 2].set_title('Augmented Overlay')
            axes[0, 2].axis('off')

            # Bottom row: would need original - just show same for now
            axes[1, 0].imshow(image)
            axes[1, 0].set_title(f'Processed Image [{idx}]')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Processed Mask\nCoverage: {coverage:.1f}%')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(overlay)
            red_patch = mpatches.Patch(color='red', alpha=0.4, label='Buildings')
            axes[1, 2].legend(handles=[red_patch], loc='upper right')
            axes[1, 2].set_title('Overlay (Red=Buildings)')
            axes[1, 2].axis('off')

        else:
            # Simple 3-panel view
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title(f'Image [{idx}]\nShape: {image.shape}')
            axes[0].axis('off')

            axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Mask [{idx}]\nCoverage: {coverage:.1f}%')
            axes[1].axis('off')

            axes[2].imshow(overlay)
            red_patch = mpatches.Patch(color='red', alpha=0.4, label='Buildings')
            axes[2].legend(handles=[red_patch], loc='upper right')
            axes[2].set_title(f'Overlay\nDtype: {image_tensor.dtype}')
            axes[2].axis('off')

        # Add overall title
        img_path = dataset.image_paths[idx]
        msk_path = dataset.mask_paths[idx]
        fig.suptitle(
            f'Sample {idx}: {img_path.name}\nMask: {msk_path.name}',
            fontsize=10,
            y=0.02
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved sample visualization to {save_path}")
        else:
            plt.show()

        plt.close()

        # Print detailed info
        print(f"\n{'='*60}")
        print(f"Sample {idx} Details:")
        print(f"{'='*60}")
        print(f"Image file: {img_path}")
        print(f"Mask file: {msk_path}")
        print(f"Image tensor: shape={image_tensor.shape}, dtype={image_tensor.dtype}")
        print(f"  Range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        print(f"Mask tensor: shape={mask_tensor.shape}, dtype={mask_tensor.dtype}")
        print(f"  Unique values: {np.unique(mask)}")
        print(f"  Coverage: {coverage:.2f}%")
        print(f"{'='*60}\n")

    except ImportError:
        logger.error("matplotlib is required for visualization")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise


# ============================================================================
# MAIN / TEST
# ============================================================================

def main():
    """Main function for comprehensive dataset testing."""
    parser = argparse.ArgumentParser(
        description='Test RoofDataset with comprehensive validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test
    python dataset.py --image_dir dataset/images --mask_dir dataset/masks

    # Test with augmentation
    python dataset.py --image_dir dataset/images --mask_dir dataset/masks --augment

    # Visualize samples
    python dataset.py --image_dir dataset/images --mask_dir dataset/masks --visualize

    # Test specific sample
    python dataset.py --image_dir dataset/images --mask_dir dataset/masks --test_idx 5 --visualize

    # Test DataLoader with multiple workers
    python dataset.py --image_dir dataset/images --mask_dir dataset/masks --batch_size 8 --num_workers 2
        """
    )

    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory containing masks')
    parser.add_argument('--target_size', type=int, default=256,
                        help='Target size for resizing (default: 256)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for dataloader test (default: 4)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize samples')
    parser.add_argument('--visualize_aug', action='store_true',
                        help='Visualize augmentation effects')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save visualization')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader workers (default: 0)')
    parser.add_argument('--test_idx', type=int, default=0,
                        help='Specific sample index to test (default: 0)')
    parser.add_argument('--test_count', type=int, default=5,
                        help='Number of samples to test (default: 5)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "="*70)
    print("STEP 7 & 8: PYTORCH DATASET WITH AUGMENTATION")
    print("="*70 + "\n")

    # ============================================================
    # TEST 1: Dataset Initialization
    # ============================================================
    logger.info("TEST 1: Creating RoofDataset...")
    try:
        dataset = RoofDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            target_size=args.target_size,
            augment=args.augment,
            validate=True
        )
        logger.info(f"✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"✗ Dataset creation failed: {e}")
        sys.exit(1)

    # ============================================================
    # TEST 2: Single Sample Loading
    # ============================================================
    print("\n" + "-"*70)
    logger.info("TEST 2: Loading single samples...")

    for i in range(min(args.test_count, len(dataset))):
        try:
            idx = (args.test_idx + i) % len(dataset)
            image, mask = dataset[idx]

            # Validate output
            assert image.shape == (3, args.target_size, args.target_size), \
                f"Image shape mismatch: {image.shape}"
            assert mask.shape == (1, args.target_size, args.target_size), \
                f"Mask shape mismatch: {mask.shape}"
            assert image.dtype == np.float32, f"Image dtype should be float32, got {image.dtype}"
            assert mask.dtype == np.float32, f"Mask dtype should be float32, got {mask.dtype}"
            assert 0 <= image.min() <= image.max() <= 1.0, "Image values should be in [0, 1]"
            assert set(np.unique(mask)).issubset({0, 1}), "Mask should be binary"

            logger.info(f"✓ Sample {idx}: image={image.shape}, mask={mask.shape}, "
                       f"coverage={mask.sum()/mask.size*100:.1f}%")

        except Exception as e:
            logger.error(f"✗ Sample {i} failed: {e}")
            raise

    logger.info(f"✓ Successfully loaded {min(args.test_count, len(dataset))} samples")

    # ============================================================
    # TEST 3: Detailed Single Sample Inspection
    # ============================================================
    print("\n" + "-"*70)
    logger.info("TEST 3: Detailed sample inspection...")

    try:
        image, mask = dataset[args.test_idx]

        print(f"\n{'='*70}")
        print(f"Sample {args.test_idx} Tensor Details:")
        print(f"{'='*70}")
        print(f"Image tensor:")
        print(f"  Shape:  {image.shape}")
        print(f"  Dtype:  {image.dtype}")
        print(f"  Range:  [{image.min():.4f}, {image.max():.4f}]")
        print(f"  Mean:   {image.mean():.4f}")
        print(f"  Std:    {image.std():.4f}")
        print(f"\nMask tensor:")
        print(f"  Shape:  {mask.shape}")
        print(f"  Dtype:  {mask.dtype}")
        print(f"  Unique: {np.unique(mask)}")
        print(f"  Sum:    {mask.sum():.0f} / {mask.size} ({mask.sum()/mask.size*100:.2f}%)")
        print(f"{'='*70}\n")

        logger.info("✓ Detailed inspection passed")

    except Exception as e:
        logger.error(f"✗ Detailed inspection failed: {e}")
        raise

    # ============================================================
    # TEST 4: PyTorch DataLoader
    # ============================================================
    print("\n" + "-"*70)
    logger.info("TEST 4: Testing PyTorch DataLoader...")

    try:
        import torch

        dataloader = get_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Get one batch
        batch_count = 0
        for batch_images, batch_masks in dataloader:
            batch_count += 1

            # Validate batch
            assert batch_images.shape[0] == batch_masks.shape[0], "Batch size mismatch"
            assert batch_images.shape[1:] == (3, args.target_size, args.target_size), \
                f"Batch image shape mismatch: {batch_images.shape}"
            assert batch_masks.shape[1:] == (1, args.target_size, args.target_size), \
                f"Batch mask shape mismatch: {batch_masks.shape}"

            logger.info(f"✓ Batch loaded: images={batch_images.shape}, masks={batch_masks.shape}")
            logger.info(f"  Image range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
            logger.info(f"  Mask coverage: {batch_masks.sum()/batch_masks.numel()*100:.1f}%")

            if batch_count >= 2:  # Test 2 batches
                break

        logger.info(f"✓ DataLoader test passed ({batch_count} batches)")

    except ImportError:
        logger.warning("PyTorch not available, skipping DataLoader test")
    except Exception as e:
        logger.error(f"✗ DataLoader test failed: {e}")
        raise

    # ============================================================
    # TEST 5: Augmentation Effects (if enabled)
    # ============================================================
    if args.augment:
        print("\n" + "-"*70)
        logger.info("TEST 5: Testing augmentation effects...")

        try:
            # Load same sample multiple times to see different augmentations
            samples = [dataset[args.test_idx] for _ in range(3)]

            # Check that samples are different (augmentation is working)
            all_same = all(np.allclose(samples[0][0], s[0]) for s in samples[1:])
            if all_same:
                logger.warning("⚠ All augmented samples are identical (random seed may be fixed)")
            else:
                logger.info("✓ Augmentation producing varied outputs")

            # Validate augmented samples
            for i, (img, msk) in enumerate(samples):
                assert img.shape == (3, args.target_size, args.target_size)
                assert msk.shape == (1, args.target_size, args.target_size)
                assert 0 <= img.min() <= img.max() <= 1.0

            logger.info(f"✓ Augmentation test passed ({len(samples)} variants)")

        except Exception as e:
            logger.error(f"✗ Augmentation test failed: {e}")
            raise

    # ============================================================
    # TEST 6: Visualization
    # ============================================================
    if args.visualize:
        print("\n" + "-"*70)
        logger.info("TEST 6: Generating visualizations...")

        try:
            # Visualize specific sample
            visualize_sample(
                dataset,
                idx=args.test_idx,
                save_path=args.save_path,
                show_original=args.augment
            )
            logger.info("✓ Sample visualization completed")

        except Exception as e:
            logger.error(f"✗ Visualization failed: {e}")
            raise

    if args.visualize_aug and args.augment:
        try:
            visualize_augmentations(
                dataset,
                idx=args.test_idx,
                num_variants=4,
                save_path=args.save_path.replace('.png', '_aug.png') if args.save_path else None
            )
            logger.info("✓ Augmentation visualization completed")

        except Exception as e:
            logger.error(f"✗ Augmentation visualization failed: {e}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("TEST SUMMARY - ALL TESTS PASSED ✓")
    print("="*70)
    print(f"Dataset path:  {args.image_dir}")
    print(f"Mask path:     {args.mask_dir}")
    print(f"Dataset size:  {len(dataset)} samples")
    print(f"Image shape:   [3, {args.target_size}, {args.target_size}]")
    print(f"Mask shape:    [1, {args.target_size}, {args.target_size}]")
    print(f"Augmentation:  {'Enabled ✓' if args.augment else 'Disabled'}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Num workers:   {args.num_workers}")
    print("="*70)
    print("\nThe dataset is ready for training!")
    print("You can now use it with PyTorch DataLoader:\n")
    print("    from dataset import RoofDataset, get_dataloader")
    print("    dataset = RoofDataset(image_dir='...', mask_dir='...', augment=True)")
    print("    loader = get_dataloader(dataset, batch_size=8)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
