#!/usr/bin/env python3
"""
SpaceNet Dataset GeoJSON to Mask Converter

Converts building footprint polygons (GeoJSON) into binary segmentation masks
aligned with satellite images for rooftop segmentation ML pipelines.

Usage:
    python generate_masks.py --pairs-file matched_pairs.json --output-dir dataset/masks
    python generate_masks.py --pairs-file matched_pairs.json --resize 256
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Delay numpy import until after dependency check
# import numpy as np  # Imported within functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress warnings from dependencies
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
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
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install rasterio geopandas shapely tqdm")
        sys.exit(1)


def load_image_metadata(image_path: Path) -> Dict[str, Any]:
    """
    Load satellite image metadata using rasterio.

    Args:
        image_path: Path to the .tif image file

    Returns:
        Dictionary containing:
            - height: Image height in pixels
            - width: Image width in pixels
            - transform: Affine transform matrix
            - crs: Coordinate Reference System
            - bounds: Image bounds

    Raises:
        FileNotFoundError: If image file doesn't exist
        rasterio.errors.RasterioIOError: If file cannot be read
    """
    import rasterio

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with rasterio.open(image_path) as src:
        metadata = {
            'height': src.height,
            'width': src.width,
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'count': src.count,  # Number of bands
            'dtype': src.dtypes[0],
            'profile': src.profile
        }

    return metadata


def load_and_clean_geojson(
    geojson_path: Path,
    target_crs: Optional[Any] = None
) -> Tuple[Optional[Any], int]:
    """
    Load GeoJSON and clean geometries.

    Fixes invalid geometries using buffer(0) and ensures correct CRS.

    Args:
        geojson_path: Path to the .geojson file
        target_crs: Target CRS to reproject to (optional)

    Returns:
        Tuple of (geometries or None, count of valid geometries)

    Raises:
        FileNotFoundError: If GeoJSON file doesn't exist
    """
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    # Check if file is empty
    if geojson_path.stat().st_size == 0:
        logger.warning(f"Empty GeoJSON file: {geojson_path}")
        return None, 0

    try:
        # Load GeoJSON
        gdf = gpd.read_file(geojson_path)

        if len(gdf) == 0:
            logger.warning(f"No features in GeoJSON: {geojson_path}")
            return None, 0

        # Fix invalid geometries using buffer(0) trick
        invalid_count = 0
        original_count = len(gdf)

        def fix_geometry(geom):
            """Fix invalid geometry using buffer(0)."""
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

        # Remove None geometries
        gdf = gdf[gdf['geometry'].notna()]

        if invalid_count > 0:
            logger.debug(f"Fixed {invalid_count} invalid geometries in {geojson_path.name}")

        if len(gdf) == 0:
            logger.warning(f"No valid geometries after cleaning: {geojson_path}")
            return None, 0

        # Reproject to target CRS if needed
        if target_crs is not None and gdf.crs != target_crs:
            try:
                gdf = gdf.to_crs(target_crs)
            except Exception as e:
                logger.warning(f"Failed to reproject CRS: {e}")

        # Extract geometries list
        geometries = gdf['geometry'].tolist()
        valid_count = len(geometries)

        if valid_count < original_count:
            logger.debug(
                f"Removed {original_count - valid_count} invalid geometries, "
                f"{valid_count} remaining"
            )

        return geometries, valid_count

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {geojson_path}: {e}")
        return None, 0
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
        geometries: List of shapely geometries (Polygons or MultiPolygons)
        shape: Output shape (height, width)
        transform: Affine transform matrix from rasterio
        fill_value: Background value (default: 0)
        default_value: Building value (default: 1)

    Returns:
        Binary mask array with shape (height, width)
    """
    import numpy as np
    from rasterio import features

    # Prepare shapes for rasterization
    # Filter out any remaining invalid geometries
    valid_shapes = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if hasattr(geom, 'geoms'):
            # MultiPolygon - add each part
            for part in geom.geoms:
                if part.is_valid and not part.is_empty:
                    valid_shapes.append((part, default_value))
        elif hasattr(geom, 'exterior'):
            # Polygon
            if geom.is_valid and not geom.is_empty:
                valid_shapes.append((geom, default_value))

    if not valid_shapes:
        # Return empty mask
        return np.full(shape, fill_value, dtype=np.uint8)

    # Rasterize
    mask = features.rasterize(
        shapes=valid_shapes,
        out_shape=shape,
        transform=transform,
        fill=fill_value,
        dtype=np.uint8
    )

    return mask


def resize_mask(
    mask: 'np.ndarray',
    target_size: Tuple[int, int],
    interpolation: str = 'nearest'
) -> 'np.ndarray':
    """
    Resize mask to target dimensions.

    Args:
        mask: Input binary mask
        target_size: Target size (height, width)
        interpolation: Interpolation method

    Returns:
        Resized mask
    """
    import numpy as np
    from PIL import Image

    if interpolation == 'nearest':
        method = Image.NEAREST
    elif interpolation == 'bilinear':
        method = Image.BILINEAR
    else:
        method = Image.NEAREST

    img = Image.fromarray(mask)
    resized = img.resize((target_size[1], target_size[0]), method)
    return np.array(resized, dtype=np.uint8)


def create_mask(
    image_path: Path,
    geojson_path: Path,
    resize: Optional[Tuple[int, int]] = None
) -> Tuple[Optional['np.ndarray'], Dict[str, Any]]:
    """
    Create binary mask from image and GeoJSON.

    Full pipeline:
    1. Load image metadata
    2. Load and clean GeoJSON
    3. Rasterize buildings
    4. Optionally resize

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
        # Load image metadata
        img_meta = load_image_metadata(image_path)
        metadata['image_height'] = img_meta['height']
        metadata['image_width'] = img_meta['width']
        metadata['crs'] = str(img_meta['crs'])

        # Load and clean GeoJSON
        geometries, count = load_and_clean_geojson(
            geojson_path,
            target_crs=img_meta['crs']
        )
        metadata['buildings_count'] = count

        if geometries is None or count == 0:
            # No buildings - create empty mask
            mask_shape = (img_meta['height'], img_meta['width'])
            mask = np.zeros(mask_shape, dtype=np.uint8)
            metadata['empty_mask'] = True
        else:
            # Rasterize buildings
            mask_shape = (img_meta['height'], img_meta['width'])
            mask = rasterize_buildings(
                geometries,
                mask_shape,
                img_meta['transform']
            )
            metadata['empty_mask'] = False

        # Resize if requested
        if resize is not None:
            original_shape = mask.shape
            mask = resize_mask(mask, resize)
            metadata['resized'] = True
            metadata['original_shape'] = original_shape
            metadata['resized_shape'] = mask.shape
        else:
            metadata['resized'] = False

        metadata['success'] = True
        return mask, metadata

    except Exception as e:
        metadata['error'] = str(e)
        logger.error(f"Failed to create mask for {image_path.name}: {e}")
        return None, metadata


def save_mask(
    mask: 'np.ndarray',
    output_path: Path,
    format: str = 'npy'
) -> bool:
    """
    Save mask to file.

    Args:
        mask: Binary mask array
        output_path: Output file path
        format: 'npy' or 'png'

    Returns:
        True if successful, False otherwise
    """
    import numpy as np

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'npy':
            np.save(output_path, mask)
        elif format == 'png':
            from PIL import Image
            # Scale to 0-255 for PNG
            mask_img = (mask * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return True
    except Exception as e:
        logger.error(f"Failed to save mask to {output_path}: {e}")
        return False


def visualize_sample(
    image_path: Path,
    mask: 'np.ndarray',
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize image with overlaid mask.

    Args:
        image_path: Path to the image file
        mask: Binary mask array
        save_path: Optional path to save the visualization
    """
    import numpy as np
    import rasterio
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Load image
    with rasterio.open(image_path) as src:
        # Read RGB bands (assuming bands 1,2,3 are RGB)
        if src.count >= 3:
            rgb = np.dstack([src.read(i) for i in range(1, 4)])
            # Normalize for display
            rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
        else:
            rgb = src.read(1)
            rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
            rgb = np.stack([rgb] * 3, axis=-1)

    # Create figure with side-by-side and overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Mask only
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Mask (Buildings: {mask.sum()} pixels)')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(rgb)
    axes[2].imshow(mask, alpha=0.4, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    # Add legend
    red_patch = mpatches.Patch(color='red', alpha=0.4, label='Buildings')
    axes[2].legend(handles=[red_patch], loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def process_batch(
    pairs: List[Dict[str, str]],
    output_dir: Path,
    format: str = 'npy',
    resize: Optional[int] = None,
    visualize: bool = False,
    visualize_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Process all image-label pairs and generate masks.

    Args:
        pairs: List of dicts with 'image' and 'label' keys
        output_dir: Directory to save masks
        format: Output format ('npy' or 'png')
        resize: Optional target size (square, e.g., 256)
        visualize: Whether to create visualization samples
        visualize_dir: Directory to save visualizations

    Returns:
        Statistics dictionary
    """
    import numpy as np
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    if visualize and visualize_dir:
        visualize_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': len(pairs),
        'successful': 0,
        'failed': 0,
        'empty_masks': 0,
        'total_buildings': 0
    }

    resize_tuple = (resize, resize) if resize else None

    logger.info(f"Processing {len(pairs)} pairs...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Format: {format}")
    if resize:
        logger.info(f"Resizing to: {resize}x{resize}")

    # Select random samples for visualization
    viz_indices = set()
    if visualize:
        viz_count = min(3, len(pairs))
        viz_indices = set(np.random.choice(len(pairs), viz_count, replace=False))

    for i, pair in enumerate(tqdm(pairs, desc="Generating masks")):
        image_path = Path(pair['image'])
        geojson_path = Path(pair['label'])

        # Generate output filename
        image_id = image_path.stem.replace('.tif', '').replace('.TIF', '')
        # Clean up the ID - remove the prefix and keep imgX part
        if '_img' in image_id:
            base_id = 'img' + image_id.split('_img')[1]
        else:
            base_id = image_id

        output_path = output_dir / f"{base_id}.{format}"

        # Create mask
        mask, metadata = create_mask(image_path, geojson_path, resize=resize_tuple)

        if mask is None:
            stats['failed'] += 1
            continue

        # Save mask
        if save_mask(mask, output_path, format):
            stats['successful'] += 1
            if metadata.get('empty_mask'):
                stats['empty_masks'] += 1
            stats['total_buildings'] += metadata['buildings_count']
        else:
            stats['failed'] += 1
            continue

        # Visualize if selected
        if visualize and i in viz_indices and visualize_dir:
            viz_path = visualize_dir / f"{base_id}_viz.png"
            try:
                visualize_sample(image_path, mask, save_path=viz_path)
            except Exception as e:
                logger.warning(f"Visualization failed for {base_id}: {e}")

    return stats


def print_summary(stats: Dict[str, Any]) -> None:
    """Print processing summary statistics."""
    print("\n" + "=" * 60)
    print("MASK GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total pairs processed:    {stats['total']}")
    print(f"Successful masks:         {stats['successful']}")
    print(f"Failed conversions:       {stats['failed']}")
    print(f"Empty masks (no buildings): {stats['empty_masks']}")
    if stats['successful'] > 0:
        avg_buildings = stats['total_buildings'] / stats['successful']
        print(f"Total buildings:          {stats['total_buildings']}")
        print(f"Avg buildings per image:  {avg_buildings:.1f}")
    success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"Success rate:             {success_rate:.1f}%")
    print("=" * 60)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert SpaceNet GeoJSON labels to binary segmentation masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--pairs-file',
        type=Path,
        default=Path('matched_pairs.json'),
        help='JSON file with image-label pairs'
    )

    parser.add_argument(
        '--output-dir',
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
        '--visualize',
        action='store_true',
        help='Generate visualization samples'
    )

    parser.add_argument(
        '--visualize-dir',
        type=Path,
        default=Path('dataset/visualizations'),
        help='Directory for visualization samples'
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

    # Load pairs file
    if not args.pairs_file.exists():
        logger.error(f"Pairs file not found: {args.pairs_file}")
        sys.exit(1)

    try:
        with open(args.pairs_file, 'r') as f:
            pairs = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in pairs file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading pairs file: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(pairs)} pairs from {args.pairs_file}")

    # Limit if requested
    if args.limit and args.limit < len(pairs):
        logger.info(f"Limiting to first {args.limit} pairs")
        pairs = pairs[:args.limit]

    # Process batch
    stats = process_batch(
        pairs=pairs,
        output_dir=args.output_dir,
        format=args.format,
        resize=args.resize,
        visualize=args.visualize,
        visualize_dir=args.visualize_dir
    )

    # Print summary
    print_summary(stats)

    # Exit with appropriate code
    sys.exit(0 if stats['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
