#!/usr/bin/env python3
"""
SpaceNet Dataset Batch Mask Generator (STEP 3)

Efficiently generates segmentation masks for the entire dataset using
batch processing with multiprocessing support.

This script reads matched_pairs.json and processes all image-label pairs
to create binary masks suitable for U-Net training.

Usage:
    python generate_masks_batch.py --input_json matched_pairs.json --output_dir dataset/masks
    python generate_masks_batch.py --resize 256 --format npy --debug --num_workers 4
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import traceback

# Delay imports until after dependency check
# import numpy as np
# import rasterio
# import geopandas as gpd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mask_generation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class ProcessingResult:
    """Result of processing a single image-label pair."""
    image_id: str
    image_path: str
    label_path: str
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    buildings_count: int = 0
    mask_shape: Optional[Tuple[int, int]] = None


@dataclass
class BatchStatistics:
    """Statistics for the entire batch processing job."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_buildings: int = 0
    total_time: float = 0.0
    failed_files: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    try:
        import rasterio
    except ImportError:
        missing.append("rasterio")
    try:
        import geopandas as gpd
    except ImportError:
        missing.append("geopandas")
    try:
        from shapely import geometry
    except ImportError:
        missing.append("shapely")
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing.append("matplotlib")

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install numpy rasterio geopandas shapely Pillow matplotlib")
        sys.exit(1)

    logger.info("All dependencies verified ✓")


def load_image_metadata(image_path: Path) -> Dict[str, Any]:
    """
    Load satellite image metadata using rasterio.

    Args:
        image_path: Path to the .tif image file

    Returns:
        Dictionary with height, width, transform, crs, etc.
    """
    import rasterio

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with rasterio.open(image_path) as src:
        return {
            'height': src.height,
            'width': src.width,
            'transform': src.transform,
            'crs': src.crs,
            'count': src.count,
            'dtype': src.dtypes[0],
            'profile': src.profile
        }


def load_and_clean_geojson(geojson_path: Path, target_crs: Optional[Any] = None) -> Tuple[Optional[List], int]:
    """
    Load GeoJSON and clean geometries using buffer(0) fix.

    Args:
        geojson_path: Path to the .geojson file
        target_crs: Target CRS for reprojection

    Returns:
        Tuple of (geometries list or None, count of valid geometries)
    """
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    if geojson_path.stat().st_size == 0:
        logger.warning(f"Empty GeoJSON file: {geojson_path}")
        return None, 0

    try:
        gdf = gpd.read_file(geojson_path)

        if len(gdf) == 0:
            logger.warning(f"No features in GeoJSON: {geojson_path}")
            return None, 0

        # Fix invalid geometries
        invalid_count = 0
        original_count = len(gdf)

        def fix_geometry(geom):
            nonlocal invalid_count
            if geom is None:
                return None
            if not geom.is_valid:
                invalid_count += 1
                try:
                    fixed = geom.buffer(0)
                    if fixed.is_valid and not fixed.is_empty:
                        return fixed
                except Exception:
                    pass
                return None
            return geom

        gdf['geometry'] = gdf['geometry'].apply(fix_geometry)
        gdf = gdf[gdf['geometry'].notna()]

        if invalid_count > 0:
            logger.debug(f"Fixed {invalid_count} invalid geometries in {geojson_path.name}")

        # Reproject if needed
        if target_crs is not None and gdf.crs != target_crs:
            try:
                gdf = gdf.to_crs(target_crs)
            except Exception as e:
                logger.warning(f"Failed to reproject CRS: {e}")

        geometries = gdf['geometry'].tolist()
        return geometries, len(geometries)

    except Exception as e:
        logger.error(f"Error loading GeoJSON {geojson_path}: {e}")
        return None, 0


def rasterize_buildings(
    geometries: List[Any],
    shape: Tuple[int, int],
    transform: Any,
    fill_value: int = 0,
    default_value: int = 1
) -> 'np.ndarray':
    """
    Rasterize building polygons into a binary mask.

    Args:
        geometries: List of shapely geometries
        shape: Output shape (height, width)
        transform: Affine transform matrix
        fill_value: Background value
        default_value: Building value

    Returns:
        Binary mask array
    """
    import numpy as np
    from rasterio import features

    valid_shapes = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if hasattr(geom, 'geoms'):
            for part in geom.geoms:
                if part.is_valid and not part.is_empty:
                    valid_shapes.append((part, default_value))
        elif hasattr(geom, 'exterior'):
            if geom.is_valid and not geom.is_empty:
                valid_shapes.append((geom, default_value))

    if not valid_shapes:
        return np.full(shape, fill_value, dtype=np.uint8)

    mask = features.rasterize(
        shapes=valid_shapes,
        out_shape=shape,
        transform=transform,
        fill=fill_value,
        dtype=np.uint8
    )

    return mask


def resize_mask(mask: 'np.ndarray', target_size: Tuple[int, int]) -> 'np.ndarray':
    """Resize mask to target dimensions using nearest neighbor."""
    import numpy as np
    from PIL import Image

    img = Image.fromarray(mask)
    resized = img.resize((target_size[1], target_size[0]), Image.NEAREST)
    return np.array(resized, dtype=np.uint8)


def create_mask(
    image_path: Path,
    geojson_path: Path,
    resize: Optional[Tuple[int, int]] = None
) -> Tuple[Optional['np.ndarray'], Dict[str, Any]]:
    """
    Create binary mask from image and GeoJSON.

    Args:
        image_path: Path to satellite image
        geojson_path: Path to GeoJSON labels
        resize: Optional target size (height, width)

    Returns:
        Tuple of (mask array or None, metadata dict)
    """
    import numpy as np

    metadata = {
        'image_path': str(image_path),
        'geojson_path': str(geojson_path),
        'success': False,
        'buildings_count': 0,
        'error': None
    }

    try:
        img_meta = load_image_metadata(image_path)
        metadata['image_height'] = img_meta['height']
        metadata['image_width'] = img_meta['width']
        metadata['crs'] = str(img_meta['crs'])

        geometries, count = load_and_clean_geojson(
            geojson_path,
            target_crs=img_meta['crs']
        )
        metadata['buildings_count'] = count

        if geometries is None or count == 0:
            mask_shape = (img_meta['height'], img_meta['width'])
            mask = np.zeros(mask_shape, dtype=np.uint8)
            metadata['empty_mask'] = True
        else:
            mask_shape = (img_meta['height'], img_meta['width'])
            mask = rasterize_buildings(
                geometries,
                mask_shape,
                img_meta['transform']
            )
            metadata['empty_mask'] = False

        if resize is not None:
            mask = resize_mask(mask, resize)
            metadata['resized'] = True
        else:
            metadata['resized'] = False

        metadata['success'] = True
        metadata['mask_shape'] = mask.shape
        return mask, metadata

    except Exception as e:
        metadata['error'] = str(e)
        return None, metadata


def save_mask(mask: 'np.ndarray', output_path: Path, format: str = 'npy') -> bool:
    """
    Save mask to file in specified format.

    Args:
        mask: Binary mask array
        output_path: Output file path
        format: 'npy' or 'png'

    Returns:
        True if successful, False otherwise
    """
    import numpy as np
    from PIL import Image

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'npy':
            np.save(output_path, mask)
        elif format == 'png':
            mask_img = (mask * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return True
    except Exception as e:
        logger.error(f"Failed to save mask to {output_path}: {e}")
        return False


def process_single_pair(
    pair: Dict[str, str],
    output_dir: Path,
    format: str = 'npy',
    resize: Optional[Tuple[int, int]] = None
) -> ProcessingResult:
    """
    Process a single image-label pair.

    Args:
        pair: Dict with 'image' and 'label' keys
        output_dir: Directory to save mask
        format: Output format ('npy' or 'png')
        resize: Optional target size (height, width)

    Returns:
        ProcessingResult with status and metadata
    """
    import numpy as np

    start_time = time.time()

    image_path = Path(pair['image'])
    geojson_path = Path(pair['label'])

    # Extract image ID
    image_id = image_path.stem
    if '_img' in image_id:
        base_id = 'img' + image_id.split('_img')[1]
    else:
        base_id = image_id

    result = ProcessingResult(
        image_id=base_id,
        image_path=str(image_path),
        label_path=str(geojson_path)
    )

    # Check for existing output
    output_path = output_dir / f"{base_id}.{format}"

    try:
        # Create mask
        mask, metadata = create_mask(image_path, geojson_path, resize=resize)

        if mask is None:
            result.success = False
            result.error_message = metadata.get('error', 'Unknown error')
            result.processing_time = time.time() - start_time
            return result

        # Save mask
        if save_mask(mask, output_path, format):
            result.success = True
            result.output_path = str(output_path)
            result.buildings_count = metadata.get('buildings_count', 0)
            result.mask_shape = mask.shape
        else:
            result.success = False
            result.error_message = "Failed to save mask file"

    except Exception as e:
        result.success = False
        result.error_message = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Exception processing {base_id}: {e}")
        logger.debug(traceback.format_exc())

    result.processing_time = time.time() - start_time
    return result


def visualize_samples(
    results: List[ProcessingResult],
    num_samples: int = 5,
    output_dir: Optional[Path] = None
) -> None:
    """
    Create visualizations for random successful samples.

    Args:
        results: List of processing results
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    import numpy as np
    import rasterio
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    successful = [r for r in results if r.success and r.output_path]
    if not successful:
        logger.warning("No successful results to visualize")
        return

    samples = np.random.choice(successful, min(num_samples, len(successful)), replace=False)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for result in samples:
        try:
            image_path = Path(result.image_path)
            mask_path = Path(result.output_path)

            # Load image
            with rasterio.open(image_path) as src:
                if src.count >= 3:
                    rgb = np.dstack([src.read(i) for i in range(1, 4)])
                    rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
                else:
                    rgb = src.read(1)
                    rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
                    rgb = np.stack([rgb] * 3, axis=-1)

            # Load mask
            if mask_path.suffix == '.npy':
                mask = np.load(mask_path)
            else:
                from PIL import Image
                mask = np.array(Image.open(mask_path)) / 255.0

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(rgb)
            axes[0].set_title(f'Original: {result.image_id}')
            axes[0].axis('off')

            axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Mask ({result.buildings_count} buildings)')
            axes[1].axis('off')

            axes[2].imshow(rgb)
            axes[2].imshow(mask, alpha=0.4, cmap='Reds', vmin=0, vmax=1)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.tight_layout()

            if output_dir:
                viz_path = output_dir / f"{result.image_id}_viz.png"
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved visualization: {viz_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.warning(f"Failed to visualize {result.image_id}: {e}")


def generate_all_masks(
    json_path: Path,
    output_dir: Path,
    format: str = 'npy',
    resize: Optional[int] = None,
    num_workers: int = 1,
    debug: bool = False,
    viz_dir: Optional[Path] = None
) -> BatchStatistics:
    """
    Main pipeline: Generate masks for all image-label pairs.

    Args:
        json_path: Path to matched_pairs.json
        output_dir: Directory to save masks
        format: Output format ('npy' or 'png')
        resize: Optional target size (square)
        num_workers: Number of parallel workers (1 = sequential)
        debug: Enable debug visualizations
        viz_dir: Directory for visualizations

    Returns:
        BatchStatistics with processing results
    """
    import numpy as np
    from tqdm import tqdm

    stats = BatchStatistics()
    stats.start_time = datetime.now()

    # Load pairs
    logger.info(f"Loading pairs from {json_path}")
    with open(json_path, 'r') as f:
        pairs = json.load(f)

    stats.total = len(pairs)
    logger.info(f"Loaded {stats.total} pairs")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Format: {format}")
    if resize:
        logger.info(f"Resize to: {resize}x{resize}")
    logger.info(f"Workers: {num_workers}")

    output_dir.mkdir(parents=True, exist_ok=True)

    resize_tuple = (resize, resize) if resize else None
    results: List[ProcessingResult] = []

    # Process pairs
    if num_workers > 1:
        # Parallel processing
        logger.info(f"Processing with {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_pair = {
                executor.submit(
                    process_single_pair,
                    pair,
                    output_dir,
                    format,
                    resize_tuple
                ): pair for pair in pairs
            }

            for future in tqdm(as_completed(future_to_pair), total=len(pairs), desc="Processing"):
                result = future.result()
                results.append(result)

                if result.success:
                    stats.successful += 1
                    stats.total_buildings += result.buildings_count
                else:
                    stats.failed += 1
                    stats.failed_files.append(f"{result.image_id}: {result.error_message}")
    else:
        # Sequential processing
        logger.info("Processing sequentially...")
        for pair in tqdm(pairs, desc="Generating masks"):
            result = process_single_pair(pair, output_dir, format, resize_tuple)
            results.append(result)

            if result.success:
                stats.successful += 1
                stats.total_buildings += result.buildings_count
            else:
                stats.failed += 1
                stats.failed_files.append(f"{result.image_id}: {result.error_message}")

    stats.end_time = datetime.now()
    stats.total_time = (stats.end_time - stats.start_time).total_seconds()

    # Save failed files log
    if stats.failed_files:
        failed_log_path = output_dir.parent / "failed_files.txt"
        with open(failed_log_path, 'w') as f:
            for line in stats.failed_files:
                f.write(f"{line}\n")
        logger.info(f"Saved failed files list to {failed_log_path}")

    # Generate visualizations in debug mode
    if debug and viz_dir:
        logger.info("Generating debug visualizations...")
        visualize_samples(results, num_samples=5, output_dir=viz_dir)

    return stats


def print_summary(stats: BatchStatistics) -> None:
    """Print processing summary statistics."""
    print("\n" + "=" * 70)
    print("BATCH MASK GENERATION SUMMARY")
    print("=" * 70)
    print(f"Start time:        {stats.start_time}")
    print(f"End time:          {stats.end_time}")
    print(f"Total time:        {stats.total_time:.2f} seconds ({stats.total_time/60:.2f} minutes)")
    print("-" * 70)
    print(f"Total pairs:       {stats.total}")
    print(f"Successful masks:  {stats.successful}")
    print(f"Failed masks:      {stats.failed}")
    print(f"Success rate:      {(stats.successful/stats.total*100):.1f}%" if stats.total > 0 else "N/A")
    print("-" * 70)
    if stats.successful > 0:
        avg_buildings = stats.total_buildings / stats.successful
        print(f"Total buildings:   {stats.total_buildings}")
        print(f"Avg buildings/img: {avg_buildings:.1f}")
    if stats.failed > 0:
        print(f"Failed files log:  failed_files.txt")
    print("=" * 70)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks for SpaceNet dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_json',
        type=Path,
        default=Path('matched_pairs.json'),
        help='Input JSON file with matched pairs'
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('dataset/masks'),
        help='Output directory for masks'
    )

    parser.add_argument(
        '--format',
        choices=['npy', 'png'],
        default='npy',
        help='Output mask format'
    )

    parser.add_argument(
        '--resize',
        type=int,
        default=None,
        help='Resize masks to square of this size (e.g., 256)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers (1 = sequential)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with visualizations'
    )

    parser.add_argument(
        '--viz_dir',
        type=Path,
        default=Path('dataset/visualizations'),
        help='Directory for debug visualizations'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process only first N pairs (for testing)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check dependencies
    check_dependencies()

    # Validate input
    if not args.input_json.exists():
        logger.error(f"Input file not found: {args.input_json}")
        sys.exit(1)

    # Load and optionally limit pairs
    with open(args.input_json, 'r') as f:
        pairs = json.load(f)

    if args.limit and args.limit < len(pairs):
        logger.info(f"Limiting to first {args.limit} pairs")
        pairs = pairs[:args.limit]

        # Save limited pairs to temp file for processing
        temp_json = Path('temp_limited_pairs.json')
        with open(temp_json, 'w') as f:
            json.dump(pairs, f)
        input_path = temp_json
    else:
        input_path = args.input_json

    # Run batch processing
    stats = generate_all_masks(
        json_path=input_path,
        output_dir=args.output_dir,
        format=args.format,
        resize=args.resize,
        num_workers=args.num_workers,
        debug=args.debug,
        viz_dir=args.viz_dir if args.debug else None
    )

    # Print summary
    print_summary(stats)

    # Cleanup temp file
    if input_path.name == 'temp_limited_pairs.json':
        input_path.unlink(missing_ok=True)

    # Exit with appropriate code
    sys.exit(0 if stats.failed == 0 else 1)


if __name__ == '__main__':
    main()
