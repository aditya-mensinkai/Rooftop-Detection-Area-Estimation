#!/usr/bin/env python3
"""
STEP 4: VISUALIZE MASKS (CRITICAL DEBUG STEP)

Verify that generated masks are correctly aligned with satellite images before training.

Usage:
    # Single sample visualization
    python visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks --num_samples 1

    # Batch visualization with specific mode
    python visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks \
                              --num_samples 5 --mode overlay

    # Interactive mode
    python visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks \
                              --interactive --mode side_by_side

    # Full dataset validation
    python visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks \
                              --validate_all --output_dir validation_report/
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('visualization.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MaskStats:
    """Statistics for a single mask."""
    total_pixels: int
    building_pixels: int
    coverage_percent: float
    is_empty: bool
    is_full: bool
    is_abnormal: bool
    abnormal_reason: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating an image-mask pair."""
    image_path: Path
    mask_path: Path
    image_id: str
    is_valid: bool = False
    image_shape: Optional[Tuple[int, ...]] = None
    mask_shape: Optional[Tuple[int, ...]] = None
    mask_stats: Optional[MaskStats] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# IMAGE LOADING
# ============================================================================

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load satellite image using rasterio (for .tif) or cv2 (for .png/.jpg).

    Args:
        image_path: Path to image file

    Returns:
        Normalized RGB image array (H, W, 3), values in [0, 1]

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image format is unsupported or corrupt
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")

    logger.info(f"Loading image: {image_path}")

    # Use rasterio for .tif files
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        return _load_tiff(image_path)
    # Use PIL/cv2 for other formats
    elif image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        return _load_standard_image(image_path)
    else:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")


def _load_tiff(image_path: Path) -> np.ndarray:
    """Load TIFF image using rasterio."""
    import rasterio
    from rasterio.errors import RasterioIOError

    try:
        with rasterio.open(image_path) as src:
            # Read RGB bands (1, 2, 3 correspond to R, G, B)
            if src.count >= 3:
                rgb = np.dstack([src.read(i) for i in range(1, 4)])
            elif src.count == 1:
                # Grayscale - stack to create RGB
                gray = src.read(1)
                rgb = np.stack([gray] * 3, axis=-1)
            else:
                raise ValueError(f"Unsupported band count: {src.count}")

            # Normalize using 99th percentile clipping (matches training preprocessing)
            rgb = rgb.astype(np.float32)
            percentile_99 = np.percentile(rgb, 99)
            if percentile_99 > 0:
                rgb = np.clip(rgb / percentile_99, 0, 1)
            else:
                rgb = np.clip(rgb, 0, 1)

            return rgb

    except RasterioIOError as e:
        raise ValueError(f"Corrupt or invalid TIFF file: {image_path} - {e}")


def _load_standard_image(image_path: Path) -> np.ndarray:
    """Load PNG/JPG using PIL/OpenCV."""
    try:
        from PIL import Image

        img = Image.open(image_path)
        rgb = np.array(img).astype(np.float32) / 255.0

        # Ensure RGB format
        if len(rgb.shape) == 2:
            # Grayscale
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[2] == 4:
            # RGBA - drop alpha
            rgb = rgb[:, :, :3]

        return rgb

    except Exception as e:
        raise ValueError(f"Failed to load image: {image_path} - {e}")


# ============================================================================
# MASK LOADING
# ============================================================================

def load_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """
    Load binary mask from .npy or .png file.

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

    logger.info(f"Loading mask: {mask_path}")

    if mask_path.suffix == '.npy':
        return _load_npy_mask(mask_path)
    elif mask_path.suffix == '.png':
        return _load_png_mask(mask_path)
    else:
        raise ValueError(f"Unsupported mask format: {mask_path.suffix}")


def _load_npy_mask(mask_path: Path) -> np.ndarray:
    """Load mask from .npy file."""
    try:
        mask = np.load(mask_path)

        # Ensure 2D
        if len(mask.shape) > 2:
            mask = mask.squeeze()

        # Ensure binary
        mask = (mask > 0).astype(np.uint8)

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
            mask = (mask / 255.0).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        return mask

    except Exception as e:
        raise ValueError(f"Corrupt PNG file: {mask_path} - {e}")


# ============================================================================
# ALIGNMENT & VALIDATION
# ============================================================================

def check_alignment(image: np.ndarray, mask: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Verify that image and mask have compatible dimensions.

    Args:
        image: Image array (H, W, C) or (H, W)
        mask: Mask array (H, W)

    Returns:
        Tuple of (is_aligned, error_message)
        error_message is None if aligned
    """
    image_shape = image.shape[:2]  # Get H, W (ignore channels)
    mask_shape = mask.shape

    if image_shape != mask_shape:
        return False, f"Shape mismatch: image {image_shape} vs mask {mask_shape}"

    return True, None


def calculate_mask_stats(mask: np.ndarray) -> MaskStats:
    """
    Calculate statistics for a binary mask.

    Args:
        mask: Binary mask array (H, W)

    Returns:
        MaskStats object with computed statistics
    """
    total_pixels = mask.size
    building_pixels = int(np.sum(mask > 0))
    coverage_percent = (building_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    is_empty = building_pixels == 0
    is_full = building_pixels == total_pixels

    # Determine if mask is abnormal
    is_abnormal = False
    abnormal_reason = None

    if is_empty:
        is_abnormal = True
        abnormal_reason = "Empty mask (no buildings)"
    elif is_full:
        is_abnormal = True
        abnormal_reason = "Fully white mask (all buildings)"
    elif coverage_percent < 1.0:
        is_abnormal = True
        abnormal_reason = f"Very small coverage ({coverage_percent:.2f}% < 1%)"
    elif coverage_percent > 90.0:
        is_abnormal = True
        abnormal_reason = f"Very large coverage ({coverage_percent:.2f}% > 90%)"

    return MaskStats(
        total_pixels=total_pixels,
        building_pixels=building_pixels,
        coverage_percent=coverage_percent,
        is_empty=is_empty,
        is_full=is_full,
        is_abnormal=is_abnormal,
        abnormal_reason=abnormal_reason
    )


def validate_pair(image_path: Path, mask_path: Path) -> ValidationResult:
    """
    Validate an image-mask pair comprehensively.

    Args:
        image_path: Path to image file
        mask_path: Path to mask file

    Returns:
        ValidationResult with all validation information
    """
    image_id = extract_image_id(image_path.name)
    result = ValidationResult(
        image_path=image_path,
        mask_path=mask_path,
        image_id=image_id
    )

    warnings_list = []

    # Try to load image
    try:
        image = load_image(image_path)
        result.image_shape = image.shape
    except Exception as e:
        result.is_valid = False
        result.error_message = str(e)
        logger.error(f"Failed to load image {image_path}: {e}")
        return result

    # Try to load mask
    try:
        mask = load_mask(mask_path)
        result.mask_shape = mask.shape
    except Exception as e:
        result.is_valid = False
        result.error_message = str(e)
        logger.error(f"Failed to load mask {mask_path}: {e}")
        return result

    # Check alignment
    aligned, error = check_alignment(image, mask)
    if not aligned:
        result.is_valid = False
        result.error_message = error
        logger.error(f"Alignment error for {image_id}: {error}")
        return result

    # Calculate mask statistics
    result.mask_stats = calculate_mask_stats(mask)

    # Check for abnormal masks
    if result.mask_stats.is_abnormal:
        warnings_list.append(result.mask_stats.abnormal_reason)
        logger.warning(f"Abnormal mask detected: {image_id} - {result.mask_stats.abnormal_reason}")

    result.warnings = warnings_list
    result.is_valid = True

    logger.info(f"Validated {image_id}: {result.mask_stats.coverage_percent:.2f}% building coverage")

    return result


# ============================================================================
# IMAGE ID EXTRACTION
# ============================================================================

def extract_image_id(filename: str) -> str:
    """
    Extract image ID from filename.

    Args:
        filename: Filename like 'img1.tif' or 'SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1.tif'

    Returns:
        Image ID like 'img1'
    """
    # Try regex pattern first
    pattern = re.compile(r'img\d+', re.IGNORECASE)
    match = pattern.search(filename)

    if match:
        return match.group(0).lower()

    # Fallback: use stem
    stem = Path(filename).stem
    if '_img' in stem:
        return 'img' + stem.split('_img')[1].split('.')[0].lower()

    return stem.lower()


def find_matching_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Find all matching image-mask pairs in directories.

    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks

    Returns:
        List of tuples (image_path, mask_path, image_id)
    """
    pairs = []

    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return pairs

    if not mask_dir.exists():
        logger.error(f"Mask directory not found: {mask_dir}")
        return pairs

    # Build mask lookup (image_id -> mask_path)
    mask_lookup = {}
    for mask_file in mask_dir.iterdir():
        if mask_file.suffix in ['.npy', '.png']:
            mask_id = extract_image_id(mask_file.name)
            mask_lookup[mask_id] = mask_file

    # Match images to masks
    for image_file in image_dir.iterdir():
        if image_file.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            image_id = extract_image_id(image_file.name)

            if image_id in mask_lookup:
                pairs.append((image_file, mask_lookup[image_id], image_id))
            else:
                logger.warning(f"No mask found for image: {image_file.name}")

    logger.info(f"Found {len(pairs)} matching pairs")
    return pairs


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4,
                   color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    Create overlay of mask on image.

    Args:
        image: RGB image array (H, W, 3), values in [0, 1]
        mask: Binary mask array (H, W)
        alpha: Transparency of mask overlay
        color: RGB tuple for mask color (default red)

    Returns:
        Overlay image array (H, W, 3)
    """
    overlay = image.copy()

    # Normalize color to [0, 1]
    color_norm = np.array(color) / 255.0

    # Create colored mask
    mask_3ch = np.stack([mask] * 3, axis=-1)
    color_mask = mask_3ch * color_norm

    # Blend
    overlay = image * (1 - mask_3ch * alpha) + color_mask * alpha

    return np.clip(overlay, 0, 1)


def create_blended(image: np.ndarray, mask: np.ndarray,
                   image_weight: float = 0.7, mask_weight: float = 0.3) -> np.ndarray:
    """
    Create blended view of image and mask.

    Args:
        image: RGB image array (H, W, 3), values in [0, 1]
        mask: Binary mask array (H, W)
        image_weight: Weight for image
        mask_weight: Weight for mask

    Returns:
        Blended image array (H, W, 3)
    """
    # Convert mask to 3-channel grayscale
    mask_3ch = np.stack([mask] * 3, axis=-1)

    # Blend
    blended = image * image_weight + mask_3ch * mask_weight

    return np.clip(blended, 0, 1)


def visualize_sample(image_path: Path, mask_path: Path, mode: str = 'all',
                     figsize: Tuple[int, int] = (15, 5), save_path: Optional[Path] = None) -> bool:
    """
    Create visualization for a single image-mask pair.

    Args:
        image_path: Path to image file
        mask_path: Path to mask file
        mode: 'side_by_side', 'overlay', 'blended', or 'all'
        figsize: Figure size for matplotlib
        save_path: Optional path to save figure

    Returns:
        True if successful, False otherwise
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    try:
        # Load data
        image = load_image(image_path)
        mask = load_mask(mask_path)

        # Validate
        aligned, error = check_alignment(image, mask)
        if not aligned:
            logger.error(f"Cannot visualize: {error}")
            return False

        # Calculate stats
        stats = calculate_mask_stats(mask)

        # Create figure based on mode
        if mode == 'side_by_side':
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Original image
            axes[0].imshow(image)
            axes[0].set_title(f'Original Image\n{image_path.name}')
            axes[0].axis('off')

            # Mask
            axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Mask\nCoverage: {stats.coverage_percent:.2f}%')
            axes[1].axis('off')

        elif mode == 'overlay':
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            overlay = create_overlay(image, mask)
            ax.imshow(overlay)
            ax.set_title(f'Overlay View\n{image_path.name}\nCoverage: {stats.coverage_percent:.2f}%')
            ax.axis('off')

            # Add legend
            red_patch = mpatches.Patch(color='red', alpha=0.4, label='Buildings')
            ax.legend(handles=[red_patch], loc='upper right')

        elif mode == 'blended':
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            blended = create_blended(image, mask)
            ax.imshow(blended)
            ax.set_title(f'Blended View\n{image_path.name}\nCoverage: {stats.coverage_percent:.2f}%')
            ax.axis('off')

        else:  # 'all' - show all 3 views
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Mask
            im = axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Binary Mask\n{stats.building_pixels:,} building pixels')
            axes[1].axis('off')

            # Overlay
            overlay = create_overlay(image, mask)
            axes[2].imshow(overlay)
            axes[2].set_title(f'Overlay (Red=Buildings)\nCoverage: {stats.coverage_percent:.2f}%')
            axes[2].axis('off')

            # Add legend to overlay
            red_patch = mpatches.Patch(color='red', alpha=0.4, label='Buildings')
            axes[2].legend(handles=[red_patch], loc='upper right')

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()

        plt.close(fig)
        return True

    except Exception as e:
        logger.error(f"Visualization failed for {image_path}: {e}")
        return False


def visualize_random_samples(image_dir: Path, mask_dir: Path, num_samples: int = 5,
                             mode: str = 'all', output_dir: Optional[Path] = None,
                             interactive: bool = False) -> int:
    """
    Visualize random samples from the dataset.

    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        num_samples: Number of samples to visualize
        mode: Visualization mode
        output_dir: Optional directory to save visualizations
        interactive: If True, wait for keypress between samples

    Returns:
        Number of successful visualizations
    """
    import matplotlib.pyplot as plt

    pairs = find_matching_pairs(image_dir, mask_dir)

    if not pairs:
        logger.error("No matching pairs found")
        return 0

    if len(pairs) < num_samples:
        logger.warning(f"Requested {num_samples} samples but only {len(pairs)} available")
        num_samples = len(pairs)

    # Randomly select samples
    selected_indices = np.random.choice(len(pairs), num_samples, replace=False)

    success_count = 0

    for idx in selected_indices:
        image_path, mask_path, image_id = pairs[idx]

        logger.info(f"Visualizing {image_id}: {image_path.name}")

        # Validate first
        result = validate_pair(image_path, mask_path)

        if not result.is_valid:
            logger.error(f"Validation failed for {image_id}: {result.error_message}")
            continue

        # Print stats
        if result.mask_stats:
            print(f"\n{'='*60}")
            print(f"Sample: {image_id}")
            print(f"Image: {result.image_shape}")
            print(f"Mask coverage: {result.mask_stats.coverage_percent:.2f}%")
            print(f"Building pixels: {result.mask_stats.building_pixels:,}")
            if result.warnings:
                print(f"⚠️  Warnings: {', '.join(result.warnings)}")
            print(f"{'='*60}\n")

        # Create visualization
        save_path = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{image_id}_viz.png"

        if visualize_sample(image_path, mask_path, mode=mode, save_path=save_path):
            success_count += 1

        # Interactive mode
        if interactive:
            if output_dir:
                print(f"Saved to {save_path}")
                response = input("Press Enter for next, 'q' to quit, or 's' to skip: ").lower()
            else:
                response = input("Press Enter for next, 'q' to quit, or 's' to skip: ").lower()

            if response == 'q':
                logger.info("User quit interactive mode")
                break
            elif response == 's':
                continue

    return success_count


# ============================================================================
# BATCH VALIDATION
# ============================================================================

def validate_all_pairs(image_dir: Path, mask_dir: Path,
                       output_dir: Optional[Path] = None) -> Dict:
    """
    Validate all image-mask pairs and generate a report.

    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        output_dir: Optional directory to save validation report

    Returns:
        Dictionary with validation summary
    """
    pairs = find_matching_pairs(image_dir, mask_dir)

    results = {
        'total_pairs': len(pairs),
        'valid': 0,
        'invalid': 0,
        'empty_masks': 0,
        'full_masks': 0,
        'abnormal_masks': 0,
        'coverage_stats': [],
        'errors': [],
        'warnings': []
    }

    validation_results = []

    print(f"\nValidating {len(pairs)} pairs...\n")

    for i, (image_path, mask_path, image_id) in enumerate(pairs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(pairs)}...")

        result = validate_pair(image_path, mask_path)
        validation_results.append(result)

        if result.is_valid:
            results['valid'] += 1

            if result.mask_stats:
                results['coverage_stats'].append(result.mask_stats.coverage_percent)

                if result.mask_stats.is_empty:
                    results['empty_masks'] += 1
                if result.mask_stats.is_full:
                    results['full_masks'] += 1
                if result.mask_stats.is_abnormal:
                    results['abnormal_masks'] += 1
                    results['warnings'].append({
                        'image_id': image_id,
                        'reason': result.mask_stats.abnormal_reason
                    })
        else:
            results['invalid'] += 1
            results['errors'].append({
                'image_id': image_id,
                'error': result.error_message
            })

    # Calculate coverage statistics
    if results['coverage_stats']:
        coverage_array = np.array(results['coverage_stats'])
        results['coverage_summary'] = {
            'mean': float(np.mean(coverage_array)),
            'median': float(np.median(coverage_array)),
            'min': float(np.min(coverage_array)),
            'max': float(np.max(coverage_array)),
            'std': float(np.std(coverage_array))
        }

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs:      {results['total_pairs']}")
    print(f"Valid:            {results['valid']}")
    print(f"Invalid:          {results['invalid']}")
    print(f"Empty masks:      {results['empty_masks']}")
    print(f"Full masks:       {results['full_masks']}")
    print(f"Abnormal masks:   {results['abnormal_masks']}")

    if 'coverage_summary' in results:
        print(f"\nCoverage Statistics:")
        print(f"  Mean:   {results['coverage_summary']['mean']:.2f}%")
        print(f"  Median: {results['coverage_summary']['median']:.2f}%")
        print(f"  Range:  {results['coverage_summary']['min']:.2f}% - {results['coverage_summary']['max']:.2f}%")
        print(f"  Std:    {results['coverage_summary']['std']:.2f}%")

    print(f"{'='*60}\n")

    # Save report
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved validation report to {report_path}")

        # Save detailed CSV
        import csv
        csv_path = output_dir / 'validation_details.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'is_valid', 'error_message', 'coverage_percent',
                           'building_pixels', 'is_empty', 'is_full', 'is_abnormal', 'warnings'])

            for result in validation_results:
                writer.writerow([
                    result.image_id,
                    result.is_valid,
                    result.error_message or '',
                    result.mask_stats.coverage_percent if result.mask_stats else 0,
                    result.mask_stats.building_pixels if result.mask_stats else 0,
                    result.mask_stats.is_empty if result.mask_stats else False,
                    result.mask_stats.is_full if result.mask_stats else False,
                    result.mask_stats.is_abnormal if result.mask_stats else False,
                    '; '.join(result.warnings)
                ])

        logger.info(f"Saved detailed CSV to {csv_path}")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='STEP 4: Visualize masks and verify alignment with images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize 5 random samples with all views
    python visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks --num_samples 5

    # Interactive mode with overlay only
    python visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks -i -m overlay

    # Validate entire dataset
    python visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks --validate_all

    # Save visualizations to directory
    python visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks \
                              --num_samples 10 --output_dir visualizations/
        """
    )

    parser.add_argument('--image_dir', type=Path, required=True,
                        help='Directory containing images (.tif, .png, .jpg)')
    parser.add_argument('--mask_dir', type=Path, required=True,
                        help='Directory containing masks (.npy, .png)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples to visualize (default: 5)')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['side_by_side', 'overlay', 'blended', 'all'],
                        help='Visualization mode (default: all)')
    parser.add_argument('--output_dir', type=Path,
                        help='Directory to save visualizations')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode (press key to continue)')
    parser.add_argument('--validate_all', action='store_true',
                        help='Validate all pairs and generate report')
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Validate directories
    if not args.image_dir.exists():
        logger.error(f"Image directory not found: {args.image_dir}")
        sys.exit(1)

    if not args.mask_dir.exists():
        logger.error(f"Mask directory not found: {args.mask_dir}")
        sys.exit(1)

    try:
        if args.validate_all:
            # Full validation mode
            validate_all_pairs(args.image_dir, args.mask_dir, args.output_dir)
        else:
            # Visualization mode
            success = visualize_random_samples(
                args.image_dir,
                args.mask_dir,
                num_samples=args.num_samples,
                mode=args.mode,
                output_dir=args.output_dir,
                interactive=args.interactive
            )

            print(f"\n{'='*60}")
            print(f"Successfully visualized {success}/{args.num_samples} samples")
            print(f"{'='*60}\n")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
