#!/usr/bin/env python3
"""
STEP 5: BUILD DATASET STRUCTURE (FINAL FORMAT)

Organize images and masks into a clean, consistent dataset structure for training.

This is the FINAL STEP before training - ensures all images have matching masks
and the dataset is ready for PyTorch/TensorFlow DataLoader.

Target Structure:
    dataset/
    ├── images/
    │   ├── img1.tif
    │   ├── img2.tif
    │   └── ...
    └── masks/
        ├── img1.npy
        ├── img2.npy
        └── ...

Critical Rule: Image and mask filenames MUST match exactly (img1.tif ↔ img1.npy)

Usage:
    # Basic usage
    python build_dataset.py --input_json matched_pairs.json --mask_src dataset/masks

    # With custom directories
    python build_dataset.py \
        --input_json matched_pairs.json \
        --image_src SN2_Vegas/PS-RGB \
        --mask_src dataset/masks \
        --output_dir dataset

    # Verify existing dataset
    python build_dataset.py --verify_only --output_dir dataset
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dataset_build.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CopyResult:
    """Result of copying a single image-mask pair."""
    base_name: str
    image_src: Path
    mask_src: Path
    image_dst: Path
    mask_dst: Path
    success: bool
    skipped: bool
    error_message: Optional[str] = None
    image_size: Optional[int] = None
    mask_size: Optional[int] = None


@dataclass
class DatasetSummary:
    """Summary of dataset building operation."""
    total_pairs: int = 0
    valid_pairs: int = 0
    copied_pairs: int = 0
    skipped_pairs: int = 0
    failed_pairs: int = 0
    missing_masks: int = 0
    missing_images: int = 0
    corrupt_files: int = 0
    skipped_details: List[Dict] = field(default_factory=list)


# ============================================================================
# FILENAME EXTRACTION
# ============================================================================

def extract_base_name(file_path: Union[str, Path]) -> str:
    """
    Extract base name from filename.

    Examples:
        SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1.tif → img1
        SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img1.geojson → img1
        img1.npy → img1
        img1.tif → img1

    Args:
        file_path: Path to file

    Returns:
        Base name (e.g., 'img1')
    """
    file_path = Path(file_path)
    stem = file_path.stem

    # Try regex pattern first
    pattern = re.compile(r'img\d+', re.IGNORECASE)
    match = pattern.search(stem)

    if match:
        return match.group(0).lower()

    # Fallback: if '_img' in name
    if '_img' in stem:
        return 'img' + stem.split('_img')[1].split('.')[0].lower()

    return stem.lower()


def get_image_id_from_pair(pair: Dict[str, str]) -> Tuple[str, Path, Path]:
    """
    Extract image ID and paths from a matched pair.

    Args:
        pair: Dict with 'image' and 'label' keys

    Returns:
        Tuple of (image_id, image_path, label_path)
    """
    image_path = Path(pair['image'])
    label_path = Path(pair['label'])
    image_id = extract_base_name(image_path)

    return image_id, image_path, label_path


# ============================================================================
# VALIDATION
# ============================================================================

def validate_file(file_path: Path, min_size: int = 100) -> Tuple[bool, Optional[str]]:
    """
    Validate that a file exists and is not corrupt.

    Args:
        file_path: Path to file
        min_size: Minimum file size in bytes (default 100)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    try:
        file_size = file_path.stat().st_size
        if file_size < min_size:
            return False, f"File too small ({file_size} bytes): {file_path}"
    except OSError as e:
        return False, f"Cannot read file stats: {file_path} - {e}"

    return True, None


def validate_pair(image_path: Path, mask_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate an image-mask pair before copying.

    Checks:
    1. Image exists and is valid
    2. Mask exists and is valid
    3. Image ID matches mask ID

    Args:
        image_path: Path to image file
        mask_path: Path to mask file

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate image
    image_valid, image_error = validate_file(image_path)
    if not image_valid:
        return False, f"Image error: {image_error}"

    # Validate mask
    mask_valid, mask_error = validate_file(mask_path)
    if not mask_valid:
        return False, f"Mask error: {mask_error}"

    # Check name consistency
    image_id = extract_base_name(image_path)
    mask_id = extract_base_name(mask_path)

    if image_id != mask_id:
        return False, f"Name mismatch: image='{image_id}' vs mask='{mask_id}'"

    return True, None


def check_file_corruption(file_path: Path, file_type: str = 'image') -> Tuple[bool, Optional[str]]:
    """
    Check if a file is corrupt by attempting to read its header.

    Args:
        file_path: Path to file
        file_type: 'image' or 'mask'

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if file_type == 'mask' and file_path.suffix == '.npy':
            # Try to load numpy file header
            import numpy as np
            # Load with mmap_mode to avoid reading full array
            arr = np.load(file_path, mmap_mode='r')
            arr.shape  # Access shape to validate
            return True, None
        elif file_type == 'image':
            # For images, just check file size and permissions
            file_size = file_path.stat().st_size
            if file_size == 0:
                return False, "File is empty"
            return True, None
    except Exception as e:
        return False, f"Corrupt {file_type} file: {e}"

    return True, None


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def copy_file_with_validation(src: Path, dst: Path, validate: bool = True) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Copy a file with validation.

    Args:
        src: Source file path
        dst: Destination file path
        validate: Whether to validate file after copying

    Returns:
        Tuple of (success, error_message, file_size)
    """
    try:
        # Check if source and destination are the same file
        if src.resolve() == dst.resolve():
            logger.warning(f"[SKIP] Source and destination are the same file: {src}")
            file_size = src.stat().st_size
            return True, None, file_size

        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src, dst)

        # Verify copy succeeded
        if not dst.exists():
            return False, "Copy verification failed: destination file does not exist", None

        dst_size = dst.stat().st_size

        if validate:
            src_size = src.stat().st_size
            if src_size != dst_size:
                return False, f"Size mismatch: src={src_size}, dst={dst_size}", None

        return True, None, dst_size

    except PermissionError as e:
        return False, f"Permission denied: {e}", None
    except OSError as e:
        return False, f"OS error during copy: {e}", None
    except Exception as e:
        return False, f"Unexpected error: {e}", None


def copy_pair(image_src: Path, mask_src: Path, output_dirs: Dict[str, Path],
              base_name: str, skip_existing: bool = True) -> CopyResult:
    """
    Copy an image-mask pair to the dataset directory.

    Args:
        image_src: Source image path
        mask_src: Source mask path
        output_dirs: Dict with 'images' and 'masks' keys
        base_name: Base name for output files
        skip_existing: If True, skip if destination files exist

    Returns:
        CopyResult with operation details
    """
    # Define destination paths
    image_ext = image_src.suffix  # Preserve original extension (.tif)
    mask_ext = mask_src.suffix    # Preserve original extension (.npy)

    image_dst = output_dirs['images'] / f"{base_name}{image_ext}"
    mask_dst = output_dirs['masks'] / f"{base_name}{mask_ext}"

    result = CopyResult(
        base_name=base_name,
        image_src=image_src,
        mask_src=mask_src,
        image_dst=image_dst,
        mask_dst=mask_dst,
        success=False,
        skipped=False
    )

    # Check if source and destination are the same for masks
    if mask_src.resolve() == mask_dst.resolve():
        logger.warning(f"[SKIP] Mask src/dst identical: {mask_src.name}")
        # Mark as success since the file is already in place
        result.success = True
        result.skipped = True
        result.image_size = image_src.stat().st_size if image_src.exists() else None
        result.mask_size = mask_src.stat().st_size if mask_src.exists() else None
        return result

    # Check if source and destination are the same for images
    if image_src.resolve() == image_dst.resolve():
        logger.warning(f"[SKIP] Image src/dst identical: {image_src.name}")
        result.success = True
        result.skipped = True
        result.image_size = image_src.stat().st_size if image_src.exists() else None
        result.mask_size = mask_src.stat().st_size if mask_src.exists() else None
        return result

    # Check if already exists
    if skip_existing and image_dst.exists() and mask_dst.exists():
        result.skipped = True
        result.success = True
        result.image_size = image_dst.stat().st_size
        result.mask_size = mask_dst.stat().st_size
        logger.info(f"Skipped (exists): {base_name}")
        return result

    # Validate pair before copying
    is_valid, error_msg = validate_pair(image_src, mask_src)
    if not is_valid:
        result.error_message = error_msg
        logger.error(f"Validation failed for {base_name}: {error_msg}")
        return result

    # Check for corruption
    img_valid, img_error = check_file_corruption(image_src, 'image')
    if not img_valid:
        result.error_message = img_error
        logger.error(f"Corrupt image {base_name}: {img_error}")
        return result

    mask_valid, mask_error = check_file_corruption(mask_src, 'mask')
    if not mask_valid:
        result.error_message = mask_error
        logger.error(f"Corrupt mask {base_name}: {mask_error}")
        return result

    # Copy image
    img_success, img_error, img_size = copy_file_with_validation(image_src, image_dst)
    if not img_success:
        result.error_message = f"Image copy failed: {img_error}"
        logger.error(f"Failed to copy image {base_name}: {img_error}")
        return result

    result.image_size = img_size

    # Copy mask
    mask_success, mask_error, mask_size = copy_file_with_validation(mask_src, mask_dst)
    if not mask_success:
        result.error_message = f"Mask copy failed: {mask_error}"
        # Clean up image if mask failed
        if image_dst.exists():
            image_dst.unlink()
        logger.error(f"Failed to copy mask {base_name}: {mask_error}")
        return result

    result.mask_size = mask_size
    result.success = True

    logger.info(f"Copied: {base_name} (image: {img_size:,} bytes, mask: {mask_size:,} bytes)")

    return result


# ============================================================================
# DATASET BUILDING
# ============================================================================

def build_dataset_structure(input_json: Path, image_src: Path, mask_src: Path,
                            output_root: Path, skip_existing: bool = True) -> DatasetSummary:
    """
    Build the clean dataset structure from matched pairs.

    Args:
        input_json: Path to matched_pairs.json
        image_src: Source directory for images (or None to use paths from JSON)
        mask_src: Source directory for masks
        output_root: Root directory for output dataset
        skip_existing: Skip files that already exist

    Returns:
        DatasetSummary with operation statistics
    """
    summary = DatasetSummary()

    # Create output directories
    output_dirs = {
        'images': output_root / 'images',
        'masks': output_root / 'masks'
    }

    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

    # Load matched pairs
    if not input_json.exists():
        logger.error(f"Input JSON not found: {input_json}")
        return summary

    try:
        with open(input_json, 'r') as f:
            pairs = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_json}: {e}")
        return summary
    except Exception as e:
        logger.error(f"Failed to load {input_json}: {e}")
        return summary

    summary.total_pairs = len(pairs)
    logger.info(f"Loaded {len(pairs)} pairs from {input_json}")

    # Process each pair
    skipped_list = []

    for i, pair in enumerate(pairs):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing pair {i + 1}/{len(pairs)}...")

        # Extract information
        image_id, image_path, _ = get_image_id_from_pair(pair)

        # Determine image source path
        if image_src:
            # Look for image in specified source directory
            potential_paths = [
                image_src / image_path.name,
                image_src / f"{image_path.stem}.tif",
                image_src / f"{image_path.stem}.TIF",
            ]
            actual_image_path = None
            for p in potential_paths:
                if p.exists():
                    actual_image_path = p
                    break

            if actual_image_path is None:
                logger.warning(f"Image not found in {image_src}: {image_path.name}")
                summary.missing_images += 1
                skipped_list.append({
                    'image_id': image_id,
                    'reason': 'Image file not found',
                    'expected': str(image_path)
                })
                continue
        else:
            actual_image_path = image_path

        # Determine mask path
        # Try multiple possible mask locations
        potential_mask_paths = [
            mask_src / f"{image_id}.npy",
            mask_src / f"{image_id}.png",
            Path(str(actual_image_path).replace('.tif', '.npy').replace('PS-RGB', 'masks')),
        ]

        actual_mask_path = None
        for p in potential_mask_paths:
            if p.exists():
                actual_mask_path = p
                break

        if actual_mask_path is None:
            logger.warning(f"Mask not found for {image_id}")
            summary.missing_masks += 1
            skipped_list.append({
                'image_id': image_id,
                'reason': 'Mask file not found',
                'image_path': str(actual_image_path)
            })
            continue

        # Validate and copy pair
        is_valid, error_msg = validate_pair(actual_image_path, actual_mask_path)
        if not is_valid:
            logger.warning(f"Skipping {image_id}: {error_msg}")
            summary.skipped_pairs += 1
            skipped_list.append({
                'image_id': image_id,
                'reason': error_msg,
                'image_path': str(actual_image_path),
                'mask_path': str(actual_mask_path)
            })
            continue

        summary.valid_pairs += 1

        # Copy the pair
        result = copy_pair(
            actual_image_path,
            actual_mask_path,
            output_dirs,
            image_id,
            skip_existing=skip_existing
        )

        if result.success:
            if result.skipped:
                summary.skipped_pairs += 1
            else:
                summary.copied_pairs += 1
        else:
            summary.failed_pairs += 1
            skipped_list.append({
                'image_id': image_id,
                'reason': result.error_message,
                'image_path': str(actual_image_path),
                'mask_path': str(actual_mask_path)
            })

    summary.skipped_details = skipped_list

    return summary


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_dataset(dataset_path: Path) -> Dict:
    """
    Verify that the dataset is correctly structured.

    Checks:
    1. Equal number of images and masks
    2. Matching filenames
    3. No missing pairs
    4. Files are valid (not empty/corrupt)

    Args:
        dataset_path: Path to dataset root (contains images/ and masks/)

    Returns:
        Dict with verification results
    """
    logger.info(f"Verifying dataset at: {dataset_path}")

    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # Check directories exist
    if not images_dir.exists():
        results['valid'] = False
        results['errors'].append(f"Images directory not found: {images_dir}")
        return results

    if not masks_dir.exists():
        results['valid'] = False
        results['errors'].append(f"Masks directory not found: {masks_dir}")
        return results

    # Get all files
    image_files = sorted([f for f in images_dir.iterdir() if f.is_file()])
    mask_files = sorted([f for f in masks_dir.iterdir() if f.is_file()])

    results['stats']['num_images'] = len(image_files)
    results['stats']['num_masks'] = len(mask_files)

    # Check counts match
    if len(image_files) != len(mask_files):
        results['valid'] = False
        results['errors'].append(
            f"Count mismatch: {len(image_files)} images vs {len(mask_files)} masks"
        )

    # Extract base names
    image_names = {extract_base_name(f): f for f in image_files}
    mask_names = {extract_base_name(f): f for f in mask_files}

    # Find mismatches
    images_without_masks = set(image_names.keys()) - set(mask_names.keys())
    masks_without_images = set(mask_names.keys()) - set(image_names.keys())

    if images_without_masks:
        results['valid'] = False
        missing_list = list(images_without_masks)[:5]
        results['errors'].append(
            f"[MISMATCH] {len(images_without_masks)} images without masks. First 5: {missing_list}"
        )

    if masks_without_images:
        results['valid'] = False
        missing_list = list(masks_without_images)[:5]
        results['errors'].append(
            f"[MISMATCH] {len(masks_without_images)} masks without images. First 5: {missing_list}"
        )

    # Check for empty files
    empty_images = [f.name for f in image_files if f.stat().st_size == 0]
    empty_masks = [f.name for f in mask_files if f.stat().st_size == 0]

    if empty_images:
        results['warnings'].append(f"Empty image files: {empty_images}")

    if empty_masks:
        results['warnings'].append(f"Empty mask files: {empty_masks}")

    # Check for corrupt mask files
    corrupt_masks = []
    for mask_file in mask_files:
        if mask_file.suffix == '.npy':
            valid, error = check_file_corruption(mask_file, 'mask')
            if not valid:
                corrupt_masks.append(f"{mask_file.name}: {error}")

    if corrupt_masks:
        results['valid'] = False
        results['errors'].extend(corrupt_masks)

    # Calculate matching pairs
    matching_pairs = set(image_names.keys()) & set(mask_names.keys())
    results['stats']['matching_pairs'] = len(matching_pairs)

    # Sample check
    if matching_pairs:
        sample_id = list(matching_pairs)[0]
        sample_image = image_names[sample_id]
        sample_mask = mask_names[sample_id]

        results['stats']['sample_pair'] = {
            'id': sample_id,
            'image': sample_image.name,
            'mask': sample_mask.name
        }

    # Final status
    if results['valid'] and not results['warnings']:
        logger.info("[OK] Dataset verification PASSED")
    elif results['valid']:
        logger.warning("[OK] Dataset verification PASSED with warnings")
    else:
        logger.error("[FAILED] Dataset verification FAILED")

    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_summary_report(summary: DatasetSummary, output_dir: Path):
    """
    Generate summary JSON and skipped files list.

    Args:
        summary: DatasetSummary object
        output_dir: Directory for output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate JSON summary
    report = {
        'total_pairs': summary.total_pairs,
        'valid_pairs': summary.valid_pairs,
        'copied_pairs': summary.copied_pairs,
        'skipped_pairs': summary.skipped_pairs,
        'failed_pairs': summary.failed_pairs,
        'missing_masks': summary.missing_masks,
        'missing_images': summary.missing_images,
        'success_rate': summary.copied_pairs / summary.total_pairs * 100 if summary.total_pairs > 0 else 0
    }

    report_path = output_dir / 'dataset_summary.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved dataset summary to {report_path}")

    # Generate skipped files list
    if summary.skipped_details:
        skipped_path = output_dir / 'skipped_files.txt'
        with open(skipped_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SKIPPED FILES REPORT\n")
            f.write("=" * 60 + "\n\n")

            for item in summary.skipped_details:
                f.write(f"Image ID: {item['image_id']}\n")
                f.write(f"Reason: {item['reason']}\n")
                if 'image_path' in item:
                    f.write(f"Image: {item['image_path']}\n")
                if 'mask_path' in item:
                    f.write(f"Mask: {item['mask_path']}\n")
                f.write("-" * 40 + "\n")

        logger.info(f"Saved skipped files list to {skipped_path}")


def print_summary(summary: DatasetSummary, output_dir: Path = None):
    """Print formatted summary to console."""
    print("\n" + "=" * 60)
    print("DATASET BUILD SUMMARY")
    print("=" * 60)
    print(f"Total pairs in JSON:    {summary.total_pairs}")
    print(f"Valid pairs found:      {summary.valid_pairs}")
    print(f"Successfully copied:    {summary.copied_pairs}")
    print(f"Skipped (exist):        {summary.skipped_pairs}")
    print(f"Failed:                 {summary.failed_pairs}")
    print(f"Missing masks:          {summary.missing_masks}")
    print(f"Missing images:         {summary.missing_images}")
    if summary.total_pairs > 0:
        success_rate = summary.copied_pairs / summary.total_pairs * 100
        print(f"Success rate:           {success_rate:.1f}%")

    # Calculate final dataset size
    if output_dir and output_dir.exists():
        try:
            images_dir = output_dir / 'images'
            masks_dir = output_dir / 'masks'
            image_dir_size = sum(f.stat().st_size for f in images_dir.iterdir() if f.is_file()) if images_dir.exists() else 0
            mask_dir_size = sum(f.stat().st_size for f in masks_dir.iterdir() if f.is_file()) if masks_dir.exists() else 0
            total_size_mb = (image_dir_size + mask_dir_size) / (1024 * 1024)
            print(f"Total dataset size:     {total_size_mb:.2f} MB")
        except Exception:
            pass

    print("=" * 60 + "\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='STEP 5: Build clean dataset structure for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build dataset with default settings
    python build_dataset.py --input_json matched_pairs.json --mask_src dataset/masks

    # With custom image source
    python build_dataset.py --input_json matched_pairs.json \\
        --image_src SN2_Vegas/PS-RGB \\
        --mask_src dataset/masks \\
        --output_dir dataset

    # Verify existing dataset only
    python build_dataset.py --verify_only --output_dir dataset

    # Force re-copy (don't skip existing)
    python build_dataset.py --input_json matched_pairs.json --mask_src dataset/masks --no_skip

    # Safe mode (skip if src==dst, verify only)
    python build_dataset.py --input_json matched_pairs.json --mask_src dataset/masks --safe_mode
        """
    )

    parser.add_argument('--input_json', type=Path, default='matched_pairs.json',
                        help='Path to matched_pairs.json (default: matched_pairs.json)')
    parser.add_argument('--image_src', type=Path, default=None,
                        help='Source directory for images (default: use paths from JSON)')
    parser.add_argument('--mask_src', type=Path, required=True,
                        help='Source directory for masks (required)')
    parser.add_argument('--output_dir', type=Path, default='dataset',
                        help='Output directory for dataset (default: dataset)')
    parser.add_argument('--verify_only', action='store_true',
                        help='Only verify existing dataset, skip building')
    parser.add_argument('--no_skip', action='store_true',
                        help='Do not skip existing files (force re-copy)')
    parser.add_argument('--safe_mode', action='store_true',
                        help='Safe mode: skip copying if src==dst, only verify and log')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Verify-only mode
    if args.verify_only:
        if not args.output_dir.exists():
            logger.error(f"Dataset directory not found: {args.output_dir}")
            sys.exit(1)

        results = verify_dataset(args.output_dir)

        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"Valid: {results['valid']}")
        print(f"Images: {results['stats'].get('num_images', 0)}")
        print(f"Masks: {results['stats'].get('num_masks', 0)}")
        print(f"Matching pairs: {results['stats'].get('matching_pairs', 0)}")

        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")

        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")

        print("=" * 60 + "\n")

        sys.exit(0 if results['valid'] else 1)

    # Build dataset
    logger.info("Starting dataset build...")
    logger.info(f"Input JSON: {args.input_json}")
    logger.info(f"Mask source: {args.mask_src}")
    logger.info(f"Output directory: {args.output_dir}")

    if not args.input_json.exists():
        logger.error(f"Input JSON not found: {args.input_json}")
        sys.exit(1)

    if not args.mask_src.exists():
        logger.error(f"Mask source directory not found: {args.mask_src}")
        sys.exit(1)

    # Safe mode validation and source/destination collision check
    output_mask_dir = args.output_dir / "masks"
    output_image_dir = args.output_dir / "images"

    if args.safe_mode:
        logger.info("Running in SAFE MODE - will skip copying identical files")

    if args.mask_src.resolve() == output_mask_dir.resolve():
        logger.warning("=" * 60)
        logger.warning("DETECTED: Mask source equals destination directory!")
        logger.warning(f"Source: {args.mask_src}")
        logger.warning(f"Destination: {output_mask_dir}")
        logger.warning("Masks will be skipped (already in place)")
        logger.warning("=" * 60)

    if args.image_src and args.image_src.resolve() == output_image_dir.resolve():
        logger.warning("=" * 60)
        logger.warning("DETECTED: Image source equals destination directory!")
        logger.warning(f"Source: {args.image_src}")
        logger.warning(f"Destination: {output_image_dir}")
        logger.warning("Images will be skipped (already in place)")
        logger.warning("=" * 60)

    # Build dataset
    summary = build_dataset_structure(
        args.input_json,
        args.image_src,
        args.mask_src,
        args.output_dir,
        skip_existing=not args.no_skip
    )

    # Generate reports
    generate_summary_report(summary, args.output_dir)

    # Print summary
    print_summary(summary, args.output_dir)

    # Verify the built dataset
    logger.info("Verifying built dataset...")
    verify_results = verify_dataset(args.output_dir)

    if not verify_results['valid']:
        logger.error("Dataset verification failed! Check errors above.")
        sys.exit(1)

    logger.info("Dataset built and verified successfully!")
    logger.info(f"Dataset location: {args.output_dir}")
    logger.info(f"Images: {args.output_dir / 'images'}")
    logger.info(f"Masks: {args.output_dir / 'masks'}")

    sys.exit(0)


if __name__ == '__main__':
    main()
