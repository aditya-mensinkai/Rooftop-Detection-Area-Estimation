# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpaceNet Rooftop Segmentation Pipeline - Converts satellite imagery (.tif) and GeoJSON building footprints into binary segmentation masks (.npy) for U-Net training.

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Pipeline

**STEP 1: Match images with labels**
```bash
python3 match_dataset.py --verbose
# Output: matched_pairs.json, missing_labels.txt, missing_images.txt
```

**STEP 2: Generate masks (single/debug mode)**
```bash
python3 generate_masks.py --limit 10 --resize 256 --visualize
```

**STEP 3: Batch mask generation (production)**
```bash
# Test run
python3 generate_masks_batch.py --limit 10 --debug --num_workers 1

# Full production with 4 workers
python3 generate_masks_batch.py --num_workers 4 --resize 256 --format npy
```

### Environment Variables (GDAL Optimization)
```bash
export GDAL_CACHEMAX=1024
export GDAL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
```

## Code Architecture

### Three-Stage Pipeline

| Script | Purpose | Key Class/Function |
|--------|---------|-------------------|
| `match_dataset.py` | Match .tif images with .geojson labels by image ID | `DatasetMatcher.scan_directories()`, `DatasetMatcher.find_matches()` |
| `generate_masks.py` | Single-pair mask generator for testing/debugging | `create_mask()`, `rasterize_buildings()` |
| `generate_masks_batch.py` | Batch processing with multiprocessing | `generate_all_masks()`, `process_single_pair()`, `ProcessingResult` dataclass |

### Core Workflow (mask generation)

1. **Load image metadata** (`load_image_metadata()`): Uses rasterio to extract height, width, CRS, affine transform
2. **Load and clean GeoJSON** (`load_and_clean_geojson()`): Uses geopandas; fixes invalid geometries with `buffer(0)`; handles CRS reprojection
3. **Rasterize** (`rasterize_buildings()`): Uses `rasterio.features.rasterize` to burn polygons into binary mask
4. **Resize** (optional): Uses PIL.Image.NEAREST for 256x256 resizing
5. **Save**: `.npy` format (numpy binary) or `.png` (visual)

### Key Design Patterns

- **Delayed imports**: Heavy geospatial libraries (rasterio, geopandas) are imported inside functions, not at module level
- **Dataclasses**: `ProcessingResult` and `BatchStatistics` for structured result tracking
- **ProcessPoolExecutor**: Parallel processing in `generate_masks_batch.py` with graceful error handling per worker
- **Logging**: File + console logging; `mask_generation.log` created by batch script
- **Validation**: `validate_pair()` checks file existence and non-zero size before processing

### Data Flow

```
SN2_Vegas/PS-RGB/*.tif + SN2_Vegas/geojson_buildings/*.geojson
    ↓ (match_dataset.py)
matched_pairs.json
    ↓ (generate_masks_batch.py)
dataset/masks/*.npy + mask_generation.log + failed_files.txt
```

### Naming Convention

- Image ID extraction: regex `img\d+` from filenames like `SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1.tif` → `img1`
- Output mask naming: `img1.npy` (matches image ID)

### Error Handling Strategy

- **Per-file isolation**: Each pair processed in try-except; failures logged but don't stop batch
- **Failed files log**: `failed_files.txt` contains list of failed conversions with reasons
- **Geometry repair**: Invalid polygons auto-fixed with `buffer(0)`; unrepairable ones skipped
- **Memory safety**: One file at a time per worker; `--resize` recommended for large images

### Output Formats

- **npy**: Binary numpy array (0=background, 1=building). Use for training.
- **png**: Grayscale image (0-255). Use for visual inspection.

### Testing Pattern

Always test with `--limit N` before full runs:
```bash
python3 generate_masks_batch.py --limit 10 --debug --num_workers 1
```

### Dependencies Note

- `rasterio` requires GDAL system library
- `geopandas` requires Fiona which needs GDAL
- macOS: `brew install gdal` before pip install
- Ubuntu: `sudo apt-get install libgdal-dev` before pip install
