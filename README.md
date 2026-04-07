# SpaceNet Rooftop Segmentation Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Python pipeline for converting SpaceNet satellite imagery and GeoJSON building footprints into binary segmentation masks for rooftop detection using deep learning models (U-Net, DeepLab, etc.).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [STEP 1: Match Images with Labels](#step-1-match-images-with-labels)
- [STEP 2: Generate Segmentation Masks](#step-2-generate-segmentation-masks)
- [STEP 3: Batch Mask Generation](#step-3-batch-mask-generation)
- [STEP 4: Visualize Masks (CRITICAL DEBUG)](#step-4-visualize-masks-critical-debug)
- [STEP 5: Build Dataset Structure](#step-5-build-dataset-structure)
- [STEP 6: Data Preprocessing Pipeline](#step-6-data-preprocessing-pipeline)
- [STEP 7: PyTorch Dataset](#step-7-pytorch-dataset)
- [STEP 8: Data Augmentation](#step-8-data-augmentation)
- [Pipeline Architecture](#pipeline-architecture)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [Citation](#citation)

---

## Overview

This pipeline processes SpaceNet dataset to create training data for rooftop/building segmentation models:

```
Satellite Image (.tif) + Building Footprints (.geojson) → Binary Mask (.npy/.png)
```

### Features

- ✅ **Robust Matching**: Automatically matches 3,850+ image-label pairs
- ✅ **Invalid Geometry Repair**: Fixes corrupted polygons using `buffer(0)`
- ✅ **CRS Alignment**: Handles coordinate system mismatches
- ✅ **Progress Tracking**: Visual progress bars with `tqdm`
- ✅ **Error Handling**: Comprehensive logging and error recovery
- ✅ **Memory Efficient**: Processes files one at a time
- ✅ **Visualization**: Debug overlays for sanity checks
- ✅ **CLI Support**: Full command-line interface

---

## Project Structure

```
Solar_Sense/RoofTop_Detection/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── match_dataset.py              # STEP 1: Image-Label Matcher
├── generate_masks.py             # STEP 2: Single mask converter
├── generate_masks_batch.py       # STEP 3: Batch mask generator with multiprocessing
├── visualize_masks.py            # STEP 4: Visualize and validate masks (CRITICAL DEBUG)
├── build_dataset.py              # STEP 5: Build final dataset structure
├── dataset.py                    # STEP 7-8: PyTorch Dataset + Data Augmentation
├── matched_pairs.json            # Generated: List of matched pairs
├── missing_labels.txt            # Generated: Images without labels
├── missing_images.txt            # Generated: Labels without images
├── invalid_files.txt             # Generated: Corrupted/invalid files
├── dataset/
│   └── masks/                    # Generated: Binary mask files
│       ├── img1.npy
│       ├── img2.npy
│       └── ...
├── dataset/visualizations/       # Generated: Debug visualizations
│   ├── img1_viz.png
│   └── ...
└── SN2_Vegas/                    # SpaceNet Dataset
    ├── PS-RGB/                   # Satellite images (.tif)
    │   ├── SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1.tif
    │   └── ...
    └── geojson_buildings/        # Building footprints (.geojson)
        ├── SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img1.geojson
        └── ...
```

---

## Requirements

### System Requirements

- **OS**: Linux, macOS, Windows (WSL recommended on Windows)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 20GB+ free space (for full SpaceNet Vegas dataset)

### Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `rasterio` | ≥1.3.0 | Satellite image reading |
| `geopandas` | ≥0.13.0 | GeoJSON processing |
| `shapely` | ≥2.0.0 | Geometry operations |
| `numpy` | ≥1.24.0 | Numerical arrays |
| `Pillow` | ≥9.0.0 | Image resizing |
| `tqdm` | ≥4.65.0 | Progress bars |
| `matplotlib` | ≥3.7.0 | Visualization |

---

## Installation

### Step 1: Clone/Navigate to Project

```bash
cd /Users/adityamensinkai/Desktop/Solar_Sense/RoofTop_Detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n rooftop python=3.10
conda activate rooftop
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: `rasterio` and `geopandas` require system libraries:

- **Ubuntu/Debian**: `sudo apt-get install libgdal-dev`
- **macOS**: `brew install gdal`
- **Windows**: Use pre-built wheels: `pip install rasterio geopandas`

### Step 4: Verify Installation

```bash
python3 -c "import rasterio; import geopandas; print('✓ All dependencies installed')"
```

---

## STEP 1: Match Images with Labels

### Script: `match_dataset.py`

Matches satellite images (.tif) with their corresponding building labels (.geojson) based on image identifiers.

### How It Works

1. **Scans directories**: Reads `PS-RGB/` and `geojson_buildings/`
2. **Extracts IDs**: Uses regex pattern `img\d+` to find identifiers
3. **Creates mapping**: Links each image to its label
4. **Validates pairs**: Checks file existence and non-zero size
5. **Reports issues**: Identifies missing labels, missing images, duplicates

### Usage

#### Basic Run (Default Settings)

```bash
python3 match_dataset.py
```

#### Custom Directories

```bash
python3 match_dataset.py \
    --images-dir /path/to/images \
    --labels-dir /path/to/labels \
    --output-dir ./output
```

#### With Verbose Logging

```bash
python3 match_dataset.py --verbose
```

#### Change Sample Count

```bash
python3 match_dataset.py --sample-count 5
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--images-dir` | `SN2_Vegas/PS-RGB` | Directory containing .tif images |
| `--labels-dir` | `SN2_Vegas/geojson_buildings` | Directory containing .geojson labels |
| `--output-dir` | `.` | Directory for output files |
| `--sample-count` | `3` | Number of random samples to display |
| `--verbose` | `False` | Enable debug logging |

### Output Files

| File | Description |
|------|-------------|
| `matched_pairs.json` | JSON array of valid image-label pairs |
| `missing_labels.txt` | List of images without corresponding labels |
| `missing_images.txt` | List of labels without corresponding images |
| `invalid_files.txt` | List of corrupted/duplicate files |

### Example Output

```
============================================================
DATASET MATCHING SUMMARY
============================================================
Total images scanned:       3850
Total labels scanned:       3851
Matched pairs:              3850
Missing labels:             0
Missing images:             1
Invalid/duplicate files:    0
============================================================
Coverage: 100.0% of images have matching labels

------------------------------------------------------------
RANDOM SAMPLE PAIRS (showing 3 of 3850)
------------------------------------------------------------

Sample 1:
  Image ID: img4682
  Image:    SN2_Vegas/PS-RGB/...img4682.tif
  Label:    SN2_Vegas/geojson_buildings/...img4682.geojson
  Status:   Image exists: True, Label exists: True

...
------------------------------------------------------------
```

### JSON Output Format

```json
[
  {
    "image": "SN2_Vegas/PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1.tif",
    "label": "SN2_Vegas/geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img1.geojson"
  },
  {
    "image": "SN2_Vegas/PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img3.tif",
    "label": "SN2_Vegas/geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img3.geojson"
  }
]
```

---

## STEP 2: Generate Segmentation Masks

### Script: `generate_masks.py`

Converts GeoJSON building footprints into binary segmentation masks aligned with satellite images.

### How It Works

1. **Load image**: Extracts dimensions and georeferencing via `rasterio`
2. **Load GeoJSON**: Reads polygons using `geopandas`
3. **Clean geometries**: Fixes invalid polygons with `buffer(0)`
4. **Reproject**: Ensures CRS alignment between image and labels
5. **Rasterize**: Burns polygons into binary mask using `rasterio.features.rasterize`
6. **Save outputs**: Stores as `.npy` (numpy) or `.png` (image)

### Usage

#### Basic Run

```bash
python3 generate_masks.py
```

#### With Resizing (for U-Net input)

```bash
python3 generate_masks.py --resize 256
```

#### Save as PNG instead of NPY

```bash
python3 generate_masks.py --format png
```

#### Generate Visualizations

```bash
python3 generate_masks.py --visualize --visualize-dir ./viz
```

#### Process Only First 10 (for testing)

```bash
python3 generate_masks.py --limit 10 --visualize
```

#### Full Production Run

```bash
python3 generate_masks.py \
    --pairs-file matched_pairs.json \
    --output-dir dataset/masks \
    --format npy \
    --resize 256 \
    --visualize \
    --visualize-dir dataset/visualizations
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pairs-file` | `matched_pairs.json` | JSON file with matched pairs |
| `--output-dir` | `dataset/masks` | Output directory for masks |
| `--format` | `npy` | Output format: `npy` or `png` |
| `--resize` | `None` | Resize to square (e.g., 256) |
| `--visualize` | `False` | Generate debug visualizations |
| `--visualize-dir` | `dataset/visualizations` | Directory for viz images |
| `--limit` | `None` | Process only first N pairs |
| `--verbose` | `False` | Enable debug logging |

### Output Files

#### Mask Files

- **Format**: `.npy` (recommended) or `.png`
- **Shape**: `(height, width)` or `(resize, resize)` if resizing
- **Values**: `0` = background, `1` = building
- **Naming**: `img1.npy`, `img2.npy`, etc.

#### Visualization Files

- **Format**: `.png`
- **Content**: Side-by-side display (original image, mask, overlay)
- **Naming**: `img1_viz.png`

### Example Output

```
2024-01-15 10:30:45,123 - INFO - Loaded 3850 pairs from matched_pairs.json
2024-01-15 10:30:45,125 - INFO - Processing 3850 pairs...
2024-01-15 10:30:45,126 - INFO - Output directory: dataset/masks
2024-01-15 10:30:45,126 - INFO - Format: npy
2024-01-15 10:30:45,126 - INFO - Resizing to: 256x256
Generating masks: 100%|████████████████████| 3850/3850 [12:34<00:00,  5.12it/s]
2024-01-15 10:43:19,456 - INFO - Saved 3850 matched pairs to matched_pairs.json

============================================================
MASK GENERATION SUMMARY
============================================================
Total pairs processed:    3850
Successful masks:         3850
Failed conversions:       0
Empty masks (no buildings): 12
Total buildings:          152340
Avg buildings per image:  39.6
Success rate:             100.0%
============================================================
```

### Loading Masks for Training

```python
import numpy as np

# Load mask
mask = np.load("dataset/masks/img1.npy")
print(mask.shape)  # (256, 256) if resized
print(mask.dtype)  # uint8

# Get building pixel count
building_pixels = np.sum(mask)  # Number of pixels labeled as building
```

---

## STEP 3: Batch Mask Generation

### Script: `generate_masks_batch.py`

Efficiently generates segmentation masks for the **entire dataset** using batch processing with optional multiprocessing support. This is the production-ready version designed for processing thousands of images.

### How It Works

1. **Load pairs**: Reads `matched_pairs.json`
2. **Initialize workers**: Creates process pool for parallel processing
3. **Process batches**: Each worker processes assigned image-label pairs
4. **Error handling**: Gracefully handles failures, continues processing
5. **Save results**: Writes masks and logs failed files
6. **Visual validation**: Optionally generates debug visualizations

### Key Features

| Feature | Description |
|---------|-------------|
| **Multiprocessing** | Parallel processing with `--num_workers` |
| **Robust Logging** | File logging + console logging |
| **Failed File Tracking** | Saves `failed_files.txt` for review |
| **Progress Tracking** | `tqdm` progress bar with ETA |
| **Debug Mode** | Generates sample visualizations |
| **Memory Efficient** | Processes one file at a time per worker |
| **Resume Capability** | Can identify already processed files |

### Usage

#### Quick Test (10 samples, sequential)

```bash
python3 generate_masks_batch.py --limit 10 --debug
```

#### Sequential Processing (Safe, Debug Mode)

```bash
python3 generate_masks_batch.py \
    --num_workers 1 \
    --resize 256 \
    --debug
```

#### Parallel Processing (4 workers, Production)

```bash
python3 generate_masks_batch.py \
    --input_json matched_pairs.json \
    --output_dir dataset/masks \
    --format npy \
    --resize 256 \
    --num_workers 4 \
    --debug \
    --viz_dir dataset/visualizations
```

#### Full Dataset Processing (All 3,850 images)

```bash
python3 generate_masks_batch.py \
    --num_workers 4 \
    --resize 256 \
    --format npy \
    --verbose
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_json` | `matched_pairs.json` | Input JSON file with matched pairs |
| `--output_dir` | `dataset/masks` | Output directory for masks |
| `--format` | `npy` | Output format: `npy` or `png` |
| `--resize` | `None` | Resize to square (e.g., 256) |
| `--num_workers` | `1` | Parallel workers (1 = sequential) |
| `--debug` | `False` | Generate sample visualizations |
| `--viz_dir` | `dataset/visualizations` | Directory for debug images |
| `--limit` | `None` | Process only first N pairs (testing) |
| `--verbose` | `False` | Enable debug logging |

### Output Files

#### Mask Files

- **Location**: `dataset/masks/`
- **Format**: `.npy` (numpy binary) or `.png`
- **Shape**: `(256, 256)` if resized, or original image size
- **Values**: `0` = background, `1` = building
- **Naming**: `img1.npy`, `img2.npy`, etc.

#### Visualization Files (Debug Mode)

- **Location**: `dataset/visualizations/`
- **Format**: `.png`
- **Content**: 3-panel view (Original Image | Mask | Overlay)
- **Naming**: `img1_viz.png`

#### Log Files

| File | Description |
|------|-------------|
| `mask_generation.log` | Full processing log |
| `failed_files.txt` | List of failed conversions |

### Example Output

```
2024-01-15 10:30:45,123 - INFO - All dependencies verified ✓
2024-01-15 10:30:45,125 - INFO - Loaded 3850 pairs from matched_pairs.json
2024-01-15 10:30:45,126 - INFO - Output directory: dataset/masks
2024-01-15 10:30:45,126 - INFO - Format: npy
2024-01-15 10:30:45,126 - INFO - Resizing to: 256x256
2024-01-15 10:30:45,126 - INFO - Workers: 4
Processing: 100%|████████████████████| 3850/3850 [12:34<00:00,  5.12it/s]
2024-01-15 10:43:19,456 - INFO - Saved failed files list to dataset/failed_files.txt
2024-01-15 10:43:19,457 - INFO - Generating debug visualizations...

======================================================================
BATCH MASK GENERATION SUMMARY
======================================================================
Start time:        2024-01-15 10:30:45.123456
End time:          2024-01-15 10:33:19.456789
Total time:        154.33 seconds (2.57 minutes)
----------------------------------------------------------------------
Total pairs:       3850
Successful masks:  3845
Failed masks:      5
Success rate:      99.9%
----------------------------------------------------------------------
Total buildings:   152,340
Avg buildings/img: 39.6
Failed files log:  failed_files.txt
======================================================================
```

### Loading Masks for Training

```python
import numpy as np

# Load mask
mask = np.load("dataset/masks/img1.npy")
print(mask.shape)      # (256, 256)
print(mask.dtype)      # uint8
print(np.unique(mask)) # [0 1]

# Check building coverage
building_pixels = np.sum(mask)
total_pixels = mask.size
coverage = building_pixels / total_pixels * 100
print(f"Building coverage: {coverage:.2f}%")
```

### Performance Comparison

| Workers | Estimated Time (3,850 images) | Use Case |
|---------|------------------------------|----------|
| 1 (Sequential) | ~45 minutes | Testing, debugging |
| 2 | ~25 minutes | Small machines |
| 4 | ~12 minutes | **Recommended** |
| 8 | ~8 minutes | High-end machines |

### Function Design

#### `process_single_pair(pair, output_dir, format, resize)`

Processes a single image-label pair:
- Creates binary mask
- Saves to disk
- Returns `ProcessingResult` with status

#### `generate_all_masks(json_path, output_dir, format, resize, num_workers, debug, viz_dir)`

Main batch processing function:
- Loads all pairs from JSON
- Distributes work across workers
- Aggregates statistics
- Saves failed files log

#### `visualize_samples(results, num_samples, output_dir)`

Creates debug visualizations:
- Randomly selects successful samples
- Generates side-by-side comparison
- Saves to disk or displays

### Error Handling

The script handles:

- ✅ **Missing files** → Logged as failure, continues
- ✅ **Empty GeoJSON** → Creates empty mask (all zeros)
- ✅ **Invalid geometries** → Fixed with `buffer(0)`
- ✅ **Rasterization errors** → Logged, skipped
- ✅ **Disk write errors** → Logged, retried
- ✅ **Corrupted images** → Logged, skipped
- ✅ **Memory errors** → Per-worker isolation prevents crash

---

## STEP 4: Visualize Masks (CRITICAL DEBUG)

### Script: `visualize_masks.py`

**⚠️ CRITICAL STEP**: Verify that generated masks are correctly aligned with satellite images **before training**. Misaligned masks will cause the model to learn wrong features and accuracy will collapse.

### Why This Step Matters

| Problem | Impact |
|---------|--------|
| Misaligned masks | Model learns incorrect building locations |
| Empty masks | Model never sees building examples |
| Full white masks | Model learns everything is a building |
| Shape mismatches | Training crashes or silent failures |

### How It Works

1. **Load image**: Uses rasterio for .tif, PIL/cv2 for .png/.jpg
2. **Load mask**: Supports both .npy (numpy) and .png formats
3. **Validate alignment**: Checks dimensions match exactly
4. **Calculate statistics**: Building coverage %, pixel counts
5. **Visualize**: Shows image, mask, and overlay views
6. **Detect anomalies**: Flags empty, full, or abnormal masks

### Key Features

| Feature | Description |
|---------|-------------|
| **3 View Modes** | Side-by-side, Overlay (red buildings), Blended |
| **Auto-Validation** | Checks alignment, shape, coverage stats |
| **Abnormal Detection** | Warns on empty (<1%) or full (>90%) masks |
| **Batch Validation** | Validate entire dataset and generate report |
| **Interactive Mode** | Press 'n' for next, 'q' to quit |

### Usage

#### Quick Visualization (5 random samples)

```bash
python3 visualize_masks.py \
    --image_dir dataset/images \
    --mask_dir dataset/masks \
    --num_samples 5
```

#### Interactive Mode (press key to continue)

```bash
python3 visualize_masks.py \
    --image_dir dataset/images \
    --mask_dir dataset/masks \
    --num_samples 10 \
    --interactive \
    --mode overlay
```

#### Specific View Mode

```bash
# Side-by-side view
python3 visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks --mode side_by_side

# Overlay view (red buildings on image)
python3 visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks --mode overlay

# Blended view (image * 0.7 + mask * 0.3)
python3 visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks --mode blended
```

#### Save Visualizations to Directory

```bash
python3 visualize_masks.py \
    --image_dir dataset/images \
    --mask_dir dataset/masks \
    --num_samples 20 \
    --output_dir validation_viz/ \
    --mode all
```

#### Full Dataset Validation (Generate Report)

```bash
python3 visualize_masks.py \
    --image_dir dataset/images \
    --mask_dir dataset/masks \
    --validate_all \
    --output_dir validation_report/
```

This generates:
- `validation_report.json` - Summary statistics
- `validation_details.csv` - Per-file validation details

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | (required) | Directory containing images (.tif, .png, .jpg) |
| `--mask_dir` | (required) | Directory containing masks (.npy, .png) |
| `--num_samples` | `5` | Number of random samples to visualize |
| `--mode` | `all` | View mode: `side_by_side`, `overlay`, `blended`, `all` |
| `--output_dir` | `None` | Directory to save visualizations |
| `-i, --interactive` | `False` | Interactive mode (keypress to continue) |
| `--validate_all` | `False` | Validate all pairs and generate report |
| `--seed` | `None` | Random seed for reproducibility |

### Visualization Modes

#### 1. Side-by-Side Mode (`--mode side_by_side`)

```
┌─────────────────┬─────────────────┐
│                 │                 │
│  Original Image │  Binary Mask    │
│                 │  (Grayscale)    │
│                 │                 │
└─────────────────┴─────────────────┘
```

#### 2. Overlay Mode (`--mode overlay`) ⭐ **MOST IMPORTANT**

```
┌─────────────────────────────────┐
│                                 │
│   Satellite Image with          │
│   Red Building Overlay          │
│   (40% transparency)            │
│                                 │
│   [RED] = Buildings             │
│                                 │
└─────────────────────────────────┘
```

#### 3. Blended Mode (`--mode blended`)

```
┌─────────────────────────────────┐
│                                 │
│   Image * 0.7 + Mask * 0.3      │
│   (Semi-transparent overlay)    │
│                                 │
└─────────────────────────────────┘
```

#### 4. All Modes (`--mode all`) - Default

```
┌─────────────┬─────────────┬─────────────┐
│             │             │             │
│  Original   │   Binary    │   Overlay   │
│   Image     │    Mask     │  (Red)      │
│             │             │             │
└─────────────┴─────────────┴─────────────┘
```

### Example Output

#### Sample Visualization Output

```
============================================================
Sample: img1523
Image: (256, 256, 3)
Mask coverage: 23.45%
Building pixels: 15,360
============================================================
```

#### Validation Report Output

```
============================================================
VALIDATION SUMMARY
============================================================
Total pairs:      3850
Valid:            3845
Invalid:          5
Empty masks:      3
Full masks:       0
Abnormal masks:   12

Coverage Statistics:
  Mean:   18.34%
  Median: 15.67%
  Range:  0.00% - 87.45%
  Std:    12.45%
============================================================
```

### Abnormal Mask Detection

The script automatically flags:

| Condition | Threshold | Warning |
|-----------|-----------|---------|
| Empty mask | Coverage = 0% | ⚠️ No buildings detected |
| Full mask | Coverage = 100% | ⚠️ Entire image is building |
| Very small | Coverage < 1% | ⚠️ Very few buildings |
| Very large | Coverage > 90% | ⚠️ Almost entire image is building |

### Python API Usage

```python
from visualize_masks import load_image, load_mask, validate_pair, visualize_sample
from pathlib import Path

# Load individual files
image = load_image("dataset/images/img1.tif")
mask = load_mask("dataset/masks/img1.npy")

# Validate a pair
result = validate_pair(
    Path("dataset/images/img1.tif"),
    Path("dataset/masks/img1.npy")
)

print(f"Valid: {result.is_valid}")
print(f"Coverage: {result.mask_stats.coverage_percent:.2f}%")

# Create visualization
visualize_sample(
    Path("dataset/images/img1.tif"),
    Path("dataset/masks/img1.npy"),
    mode="all",
    save_path=Path("viz/img1_viz.png")
)
```

### Debugging Checklist

Before starting training, verify:

- [ ] Visualized at least 10 random samples
- [ ] All samples show correct alignment (buildings match red overlay)
- [ ] No empty masks unless expected (some images have no buildings)
- [ ] Coverage statistics look reasonable (typical: 5-40%)
- [ ] No shape mismatches reported
- [ ] Validation report shows >95% valid pairs

### Common Issues

#### Issue: `Shape mismatch: image (650, 650, 3) vs mask (256, 256)`

**Cause**: Image and mask were resized differently.

**Solution**: Regenerate masks with consistent resize parameter:
```bash
python3 generate_masks_batch.py --resize 256 --num_workers 4
```

#### Issue: `Empty mask (no buildings)` warnings

**Cause**: Some images genuinely have no buildings, OR GeoJSON failed to parse.

**Solution**: 
```bash
# Check specific image
python3 visualize_masks.py --image_dir dataset/images --mask_dir dataset/masks --num_samples 1

# If mask is wrong, check GeoJSON
cat SN2_Vegas/geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_imgXXX.geojson
```

#### Issue: `Alignment looks wrong` (buildings don't match overlay)

**Cause**: CRS mismatch or transform error during rasterization.

**Solution**: 
1. Check the source GeoJSON coordinates
2. Verify image CRS with `rasterio`
3. Regenerate mask for that specific image

---

## STEP 5: Build Dataset Structure

### Script: `build_dataset.py`

**FINAL STEP**: Organize images and masks into a clean, consistent dataset structure ready for PyTorch/TensorFlow training.

### Why This Step Matters

| Problem | Impact |
|---------|--------|
| Filename mismatch | Training fails - wrong labels loaded |
| Missing pairs | Model trains on mismatched data |
| Corrupt files | Training crashes mid-epoch |
| Unclean structure | DataLoader can't find files |

### Target Structure

```
dataset/
├── images/
│   ├── img1.tif
│   ├── img2.tif
│   └── ...
├── masks/
│   ├── img1.npy
│   ├── img2.npy
│   └── ...
├── dataset_summary.json      # Build statistics
└── skipped_files.txt         # Files that couldn't be copied
```

**Critical Rule**: Image and mask filenames MUST match exactly:
- ✅ `img1.tif` ↔ `img1.npy`
- ❌ `img1.tif` ↔ `img1_mask.npy` (mismatch - will fail)

### How It Works

1. **Read JSON**: Loads `matched_pairs.json`
2. **Extract IDs**: Parses base names (e.g., `img1`)
3. **Locate files**: Finds images and masks by ID
4. **Validate pairs**: Checks both files exist and aren't corrupt
5. **Copy files**: Copies to clean structure with matching names
6. **Verify**: Runs post-build validation

### Key Features

| Feature | Description |
|---------|-------------|
| **Auto-naming** | Extracts clean `img{N}` from long filenames |
| **Validation** | Checks file integrity before copying |
| **Skip existing** | Won't re-copy files already in place |
| **Corruption check** | Verifies .npy files are readable |
| **Verification mode** | Can verify existing dataset without rebuilding |

### Usage

#### Basic Build

```bash
python3 build_dataset.py \
    --input_json matched_pairs.json \
    --mask_src dataset/masks
```

#### Custom Directories

```bash
python3 build_dataset.py \
    --input_json matched_pairs.json \
    --image_src SN2_Vegas/PS-RGB \
    --mask_src dataset/masks \
    --output_dir dataset
```

#### Verify Only (No Copy)

```bash
python3 build_dataset.py --verify_only --output_dir dataset
```

#### Force Re-copy (Don't Skip)

```bash
python3 build_dataset.py \
    --input_json matched_pairs.json \
    --mask_src dataset/masks \
    --no_skip
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_json` | `matched_pairs.json` | Path to matched pairs JSON |
| `--image_src` | `None` | Source images (uses JSON paths if not set) |
| `--mask_src` | (required) | Source directory for masks |
| `--output_dir` | `dataset` | Output directory for clean dataset |
| `--verify_only` | `False` | Only verify, don't build |
| `--no_skip` | `False` | Force re-copy existing files |
| `--verbose` | `False` | Enable verbose logging |

### Example Output

```
============================================================
DATASET BUILD SUMMARY
============================================================
Total pairs in JSON:    3850
Valid pairs found:      3850
Successfully copied:    3845
Skipped (exist):        0
Failed:                 5
Missing masks:          0
Missing images:         0
Success rate:           99.9%
============================================================
```

### Validation Checks

The script automatically verifies:

| Check | Description |
|-------|-------------|
| File existence | Both image and mask files exist |
| Name matching | Extracted IDs match between image and mask |
| File size | Files > 100 bytes (not empty) |
| Corruption | .npy files can be loaded |
| Count match | Equal number of images and masks |
| Pair completeness | Every image has a mask, every mask has an image |

### Generated Reports

#### dataset_summary.json

```json
{
  "total_pairs": 3850,
  "valid_pairs": 3850,
  "copied_pairs": 3845,
  "skipped_pairs": 0,
  "failed_pairs": 5,
  "missing_masks": 0,
  "missing_images": 0,
  "success_rate": 99.9
}
```

#### skipped_files.txt

```
============================================================
SKIPPED FILES REPORT
============================================================

Image ID: img1234
Reason: Mask file not found
Image: SN2_Vegas/PS-RGB/...img1234.tif
----------------------------------------
Image ID: img5678
Reason: Corrupt mask file
Image: SN2_Vegas/PS-RGB/...img5678.tif
Mask: dataset/masks/img5678.npy
```

### Python API Usage

```python
from build_dataset import (
    build_dataset_structure,
    verify_dataset,
    extract_base_name
)
from pathlib import Path

# Build dataset
summary = build_dataset_structure(
    input_json=Path('matched_pairs.json'),
    image_src=Path('SN2_Vegas/PS-RGB'),
    mask_src=Path('dataset/masks'),
    output_root=Path('dataset'),
    skip_existing=True
)

print(f"Copied: {summary.copied_pairs}")
print(f"Failed: {summary.failed_pairs}")

# Verify existing dataset
results = verify_dataset(Path('dataset'))
print(f"Valid: {results['valid']}")
print(f"Matching pairs: {results['stats']['matching_pairs']}")
```

### Pre-Training Checklist

Before starting training, verify:

- [ ] Dataset built successfully (>95% success rate)
- [ ] Ran verification: `python build_dataset.py --verify_only`
- [ ] Equal number of images and masks
- [ ] Filenames match exactly (img1.tif ↔ img1.npy)
- [ ] No corrupt files reported
- [ ] Reviewed skipped_files.txt for any issues

### Common Issues

#### Issue: `Mask file not found for img1234`

**Cause**: Mask wasn't generated or is in different location.

**Solution**:
```bash
# Check if mask exists
ls dataset/masks/img1234.npy

# If missing, regenerate specific mask
python3 generate_masks.py --limit 1  # debug mode
```

#### Issue: `Name mismatch: image='img1' vs mask='img2'`

**Cause**: Mask was generated with different naming convention.

**Solution**: Regenerate masks with consistent naming:
```bash
python3 generate_masks_batch.py --num_workers 4 --resize 256
```

#### Issue: `Corrupt mask file`

**Cause**: .npy file is incomplete or damaged.

**Solution**:
```bash
# Delete corrupt mask and regenerate
rm dataset/masks/img1234.npy
python3 generate_masks.py --limit 1  # or batch re-run
```

#### Issue: `Count mismatch: 3845 images vs 3850 masks`

**Cause**: Some masks failed to generate or were copied twice.

**Solution**:
```bash
# Clean output and rebuild
rm -rf dataset/masks/*
python3 generate_masks_batch.py --num_workers 4
python3 build_dataset.py --input_json matched_pairs.json --mask_src dataset/masks --no_skip
```

---

## STEP 6: Data Preprocessing Pipeline

### Script: `dataset.py`

**FINAL STEP**: PyTorch Dataset implementation that prepares image-mask pairs for training with U-Net or other segmentation models.

### Why This Step Matters

| Problem | Impact |
|---------|--------|
| Wrong preprocessing | Model receives incorrect input format |
| Shape mismatch | Training crashes with dimension errors |
| No augmentation | Model overfits, poor generalization |
| Wrong interpolation on masks | Mask edges become blurry, labels corrupted |
| Wrong normalization | Model can't learn effectively |

### What This Script Does

1. **Lazy Loading**: Images loaded on-demand (not all in memory)
2. **Flexible Loading**: Supports .tif via rasterio, .png/.jpg via PIL
3. **Preprocessing**: Resize → Normalize → Channel reorder
4. **Augmentation**: Optional flips, rotation, brightness
5. **Validation**: Checks alignment, warns on anomalies

### Key Features

| Feature | Description |
|---------|-------------|
| **Lazy Loading** | Images loaded in `__getitem__`, not in `__init__` |
| **Preprocessing** | Automatic resize, normalize, HWC→CHW conversion |
| **Data Augmentation** | Flip, rotate, brightness (same for image+mask) |
| **Validation** | Checks shape alignment, empty/full masks |
| **Format Output** | Returns tensors: image [3,H,W], mask [1,H,W] |

### Preprocessing Pipeline

```
Image:                          Mask:
┌─────────────┐                ┌─────────────┐
│ Load .tif   │                │ Load .npy   │
│ or .png     │                │ or .png     │
└──────┬──────┘                └──────┬──────┘
       │                             │
       ▼                             ▼
┌─────────────┐                ┌─────────────┐
│ Resize      │                │ Resize      │
│ INTER_LINEAR│                │ INTER_NEAREST│
│ (256,256)   │                │ (256,256)   │
└──────┬──────┘                └──────┬──────┘
       │                             │
       ▼                             ▼
┌─────────────┐                ┌─────────────┐
│ Normalize   │                │ Ensure      │
│ / 255.0     │                │ binary {0,1}│
└──────┬──────┘                └──────┬──────┘
       │                             │
       ▼                             ▼
┌─────────────┐                ┌─────────────┐
│ HWC → CHW   │                │ Add channel │
│             │                │ [1,H,W]     │
└──────┬──────┘                └──────┬──────┘
       │                             │
       ▼                             ▼
┌─────────────┐                ┌─────────────┐
│ Image Tensor│                │ Mask Tensor │
│ [3,256,256] │                │ [1,256,256]│
│ float32     │                │ float32     │
└─────────────┘                └─────────────┘
```

### Usage

#### Basic Dataset

```python
from dataset import RoofDataset, get_dataloader

# Create dataset
dataset = RoofDataset(
    image_dir='dataset/images',
    mask_dir='dataset/masks',
    target_size=256
)

# Create DataLoader
train_loader = get_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

# Training loop
for images, masks in train_loader:
    # images: [B, 3, 256, 256]
    # masks: [B, 1, 256, 256]
    pass
```

#### With Augmentation

```python
# Enable augmentation
dataset = RoofDataset(
    image_dir='dataset/images',
    mask_dir='dataset/masks',
    target_size=256,
    augment=True  # Enable random flips, rotation, brightness
)
```

#### Test and Visualize

```bash
# Basic test
python3 dataset.py --image_dir dataset/images --mask_dir dataset/masks --visualize

# With augmentation
python3 dataset.py --image_dir dataset/images --mask_dir dataset/masks --augment --visualize

# Test DataLoader
python3 dataset.py --image_dir dataset/images --mask_dir dataset/masks --batch_size 8 --num_workers 2
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | (required) | Directory containing images |
| `--mask_dir` | (required) | Directory containing masks |
| `--target_size` | `256` | Target size for resizing |
| `--augment` | `False` | Enable data augmentation |
| `--batch_size` | `4` | Batch size for DataLoader test |
| `--visualize` | `False` | Visualize samples |
| `--save_path` | `None` | Path to save visualization |
| `--num_workers` | `0` | DataLoader workers (0=main process) |

### RoofDataset Class

#### Constructor

```python
dataset = RoofDataset(
    image_dir='dataset/images',     # Path to images
    mask_dir='dataset/masks',       # Path to masks
    target_size=256,                 # Resize to (256, 256)
    augment=False,                   # Enable augmentation
    transform=None,                  # Optional custom transforms
    validate=True                    # Validate dataset on init
)
```

#### Output Format

| Property | Image | Mask |
|----------|-------|------|
| Shape | `[3, 256, 256]` | `[1, 256, 256]` |
| Type | `float32` | `float32` |
| Range | `[0, 1]` | `{0, 1}` |
| Channels | RGB | Binary |

### Data Augmentation

Augmentations applied **with same random parameters** to both image and mask:

| Augmentation | Probability | Details |
|--------------|-------------|---------|
| Horizontal Flip | 0.5 | Random horizontal flip |
| Vertical Flip | 0.5 | Random vertical flip |
| Rotation | 0.5 | 90°, 180°, or 270° |
| Brightness | 0.5 | Factor 0.8-1.2 (image only) |

```python
from dataset import Augmentation

# Custom augmentation
aug = Augmentation(
    horizontal_flip=True,
    vertical_flip=True,
    rotation=True,
    brightness=True,
    p=0.5  # Probability for each
)

image_aug, mask_aug = aug(image, mask)
```

### Validation Checks

Automatic checks performed:

| Check | Action | Error Level |
|-------|--------|-------------|
| File exists | Skip missing files | Warning |
| Shape alignment | Ensure image and mask same size | Error |
| Empty mask | Warn if mask has no buildings | Warning |
| Full mask | Warn if mask is all buildings | Warning |
| Sample validation | Test first sample on init | Error |

### Example Output

```
2024-01-15 10:30:45 - INFO - Initialized dataset with 3845 samples
2024-01-15 10:30:45 - INFO - Testing single sample...
2024-01-15 10:30:45 - INFO - Image shape: (3, 256, 256), dtype: float32
2024-01-15 10:30:45 - INFO - Mask shape: (1, 256, 256), dtype: float32
2024-01-15 10:30:45 - INFO - Image range: [0.000, 1.000]
2024-01-15 10:30:45 - INFO - Mask unique values: [0. 1.]
2024-01-15 10:30:45 - INFO - Dataset validation passed
```

### Python API Functions

```python
from dataset import (
    load_image,
    load_mask,
    preprocess_image,
    preprocess_mask,
    visualize_batch,
    visualize_augmentations
)

# Load single files
image = load_image('dataset/images/img1.tif')  # [H, W, 3]
mask = load_mask('dataset/masks/img1.npy')     # [H, W]

# Preprocess individually
image_tensor = preprocess_image(image, target_size=256)  # [3, H, W]
mask_tensor = preprocess_mask(mask, target_size=256)     # [1, H, W]

# Visualize batch
visualize_batch(dataset, num_samples=4, save_path='viz.png')

# Visualize augmentations
visualize_augmentations(dataset, idx=0, num_variants=4)
```

### Common Issues

#### Issue: `ImportError: No module named 'torch'`

**Cause**: PyTorch not installed.

**Solution**:
```bash
# Install PyTorch (CPU version)
pip install torch torchvision

# Or with CUDA (if GPU available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: `Shape mismatch: image=(650, 650, 3), mask=(256, 256)`

**Cause**: Images and masks not resized consistently.

**Solution**: Images should already be resized to match masks, or regenerate masks:
```bash
python3 generate_masks_batch.py --resize 256 --num_workers 4
```

#### Issue: `Dataset validation failed: Image should have 3 channels, got 1`

**Cause**: Grayscale images loaded instead of RGB.

**Solution**: Check source images are RGB, or the loading code stacks channels.

#### Issue: `Mask unique values: [0.]` (all zeros)

**Cause**: Masks weren't generated correctly or are empty.

**Solution**:
```bash
# Check mask
python3 -c "import numpy as np; print(np.load('dataset/masks/img1.npy').sum())"

# Regenerate if needed
python3 generate_masks.py --limit 1 --visualize
```

#### Issue: `RuntimeError: DataLoader worker (pid xxx) is killed by signal: Killed`

**Cause**: Out of memory with multiple workers.

**Solution**: Reduce num_workers:
```python
train_loader = get_dataloader(dataset, batch_size=8, num_workers=0)  # Single process
```

---

## STEP 7: PyTorch Dataset

### Script: `dataset.py`

PyTorch Dataset implementation that loads and prepares image-mask pairs for training segmentation models.

### What This Step Does

1. **Image Loading**: Supports .tif (via rasterio) and standard formats (via PIL/cv2)
2. **Mask Loading**: Loads .npy or .png masks
3. **Preprocessing**: Resize, normalize, format conversion
4. **Validation**: Checks alignment, warns on anomalies
5. **Output**: Returns PyTorch tensors ready for training

### Input/Output Format

| | Shape | Dtype | Range |
|---|---|---|---|
| **Image Input** | (H, W, 3) | uint8 | [0, 255] |
| **Image Output** | [3, 256, 256] | float32 | [0, 1] |
| **Mask Input** | (H, W) | uint8/float | {0, 1} |
| **Mask Output** | [1, 256, 256] | float32 | {0, 1} |

### Usage

```python
from dataset import RoofDataset, get_dataloader

# Create dataset
dataset = RoofDataset(
    image_dir='dataset/images',
    mask_dir='dataset/masks',
    target_size=256
)

# Create DataLoader
train_loader = get_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Training loop
for images, masks in train_loader:
    # images: [B, 3, 256, 256]
    # masks: [B, 1, 256, 256]
    pass
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Lazy Loading** | Images loaded on-demand in `__getitem__` |
| **Auto-matching** | Pairs images/masks by filename (img1.tif ↔ img1.npy) |
| **Format Support** | .tif, .png, .jpg images; .npy, .png masks |
| **Preprocessing** | Resize, normalize, HWC→CHW conversion |
| **Validation** | Shape checks, empty/full mask warnings |

### RoofDataset Class

```python
dataset = RoofDataset(
    image_dir='dataset/images',     # Path to images
    mask_dir='dataset/masks',       # Path to masks
    target_size=256,                 # Resize to (256, 256)
    augment=True,                    # Enable augmentation (STEP 8)
    validate=True                    # Validate on init
)
```

### Data Loading Pipeline

```
Image:                          Mask:
┌─────────────┐                ┌─────────────┐
│ Load .tif   │                │ Load .npy   │
│ or .png     │                │ or .png     │
└──────┬──────┘                └──────┬──────┘
       │                             │
       ▼                             ▼
┌─────────────┐                ┌─────────────┐
│ Resize      │                │ Resize      │
│ INTER_LINEAR│                │ INTER_NEAREST│
│ (256,256)   │                │ (256,256)   │
└──────┬──────┘                └──────┬──────┘
       │                             │
       ▼                             ▼
┌─────────────┐                ┌─────────────┐
│ Normalize   │                │ Ensure      │
│ / 255.0     │                │ binary {0,1}│
└──────┬──────┘                └──────┬──────┘
       │                             │
       ▼                             ▼
┌─────────────┐                ┌─────────────┐
│ HWC → CHW   │                │ Add channel │
│ [3,H,W]     │                │ [1,H,W]     │
└──────┬──────┘                └──────┬──────┘
       │                             │
       ▼                             ▼
┌─────────────┐                ┌─────────────┐
│   Tensor    │                │   Tensor    │
│ [3,256,256] │                │ [1,256,256] │
└─────────────┘                └─────────────┘
```

### Validation Checks

| Check | Action | Level |
|-------|--------|-------|
| File exists | Skip missing files | Warning |
| Shape alignment | Ensure image and mask same size | Error |
| Empty mask | Warn if mask has no buildings | Warning |
| Full mask | Warn if mask is all buildings | Warning |

### Command-Line Test

```bash
# Basic test
python3 dataset.py --image_dir dataset/images --mask_dir dataset/masks --visualize

# Test DataLoader
python3 dataset.py --image_dir dataset/images --mask_dir dataset/masks --batch_size 8 --num_workers 2
```

---

## STEP 8: Data Augmentation

### Overview

Data augmentation applies random transformations to increase training data diversity and prevent overfitting. **Critical**: Same transformation must be applied to both image and mask.

### Implemented Augmentations

| Augmentation | Probability | Details |
|--------------|-------------|---------|
| **Horizontal Flip** | 0.5 | Random horizontal mirror |
| **Vertical Flip** | 0.5 | Random vertical mirror |
| **Rotation** | 0.5 | 90°, 180°, or 270° |
| **Brightness** | 0.5 | Factor 0.8-1.2 (image only) |

### Critical Requirements

⚠️ **IMPORTANT**: For segmentation, augmentations must:

1. **Same transform for image & mask**: If image is flipped, mask must flip identically
2. **Nearest neighbor for masks**: Prevents interpolation artifacts at label boundaries
3. **No interpolation on labels**: Mask values must stay exactly {0, 1}

### Augmentation Class

```python
from dataset import Augmentation

# Custom augmentation
aug = Augmentation(
    horizontal_flip=True,
    vertical_flip=True,
    rotation=True,
    brightness=True,
    p=0.5  # Probability for each
)

image_aug, mask_aug = aug(image, mask)
```

### Usage with Dataset

```python
# Enable augmentation
dataset = RoofDataset(
    image_dir='dataset/images',
    mask_dir='dataset/masks',
    target_size=256,
    augment=True  # Enable random flips, rotation, brightness
)
```

### Test Augmentation

```bash
# Visualize augmentations
python3 dataset.py --image_dir dataset/images --mask_dir dataset/masks --augment --visualize
```

### Visualization Output

Shows side-by-side comparison:
- Original image
- Original mask
- Augmented image  
- Augmented mask

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   SpaceNet      │     │   match_dataset  │     │  generate_masks │
│   Dataset       │     │       .py        │     │     _batch.py    │
├─────────────────┤     ├──────────────────┤     ├─────────────────┤
│                 │     │                  │     │                 │
│  PS-RGB/        │────▶│  1. Scan dirs    │────▶│  1. Load pairs  │
│  ├── img1.tif   │     │  2. Match IDs    │     │  2. Create pool │
│  ├── img2.tif   │     │  3. Validate     │     │  3. Process all│
│  └── ...        │     │  4. Save pairs   │     │  4. Save masks  │
│                 │     │                  │     │  5. Log stats   │
│  geojson_/      │────▶│                  │     │                 │
│  ├── img1.json  │     │                  │     │                 │
│  └── ...        │     │                  │     │                 │
│                 │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌──────────────────┐            │
         │              │ matched_pairs    │            │
         │              │     .json        │            │
         │              └──────────────────┘            │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │ dataset/masks/  │
                        │ ├── img1.npy    │
                        │ ├── img2.npy    │
                        │ └── ...         │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ visualize_masks │
                        │       .py       │
                        │   (STEP 4)      │
                        ├─────────────────┤
                        │  1. Validate    │
                        │  2. Visualize   │
                        │  3. Check align │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  build_dataset  │
                        │       .py       │
                        │   (STEP 5)      │
                        ├─────────────────┤
                        │  1. Read pairs  │
                        │  2. Validate    │
                        │  3. Copy files  │
                        │  4. Verify      │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │    dataset      │
                        │       .py       │
                        │ (STEP 7-8)      │
                        ├─────────────────┤
                        │  1. Load        │
                        │  2. Preprocess  │
                        │  3. Augment     │
                        │  4. Tensor      │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   ML Training   │
                        ├─────────────────┤
                        │                 │
                        │  Image ──▶ U-Net│
                        │    │         │  │
                        │    └────┐    │  │
                        │         ▼    ▼  │
                        │       Mask Loss │
                        │                 │
                        └─────────────────┘
```

### Workflow Steps

| Step | Script | Input | Output | Purpose |
|------|--------|-------|--------|---------|
| **1** | `match_dataset.py` | Raw dataset | `matched_pairs.json` | Match images with labels |
| **2** | `generate_masks.py` | Single pair | Single mask | Testing/debugging |
| **3** | `generate_masks_batch.py` | `matched_pairs.json` | `dataset/masks/` | Production batch processing |
| **4** | `visualize_masks.py` | `dataset/masks/` | Validation report | Verify alignment before training |
| **5** | `build_dataset.py` | `matched_pairs.json` + masks | `dataset/` | Build clean training structure |
| **6** | `dataset.py` | `dataset/images/` + `dataset/masks/` | PyTorch DataLoader | Preprocessing + Augmentation for model training |

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'rasterio'`

**Solution**:
```bash
# macOS
brew install gdal
pip install rasterio

# Ubuntu/Debian
sudo apt-get install libgdal-dev
pip install rasterio

# Windows
pip install rasterio --find-links https://girder.github.io/large_image_wheels
```

### Issue: `No valid geometries after cleaning`

**Cause**: All geometries in the GeoJSON are invalid.

**Solution**: The script automatically handles this by creating an empty mask. Check the GeoJSON file manually:
```bash
cat SN2_Vegas/geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img1.geojson | head -20
```

### Issue: `CRS mismatch warnings`

**Cause**: GeoJSON CRS differs from image CRS.

**Solution**: The script automatically reprojects. If it fails, convert manually:
```python
import geopandas as gpd
gdf = gpd.read_file("geojson_file.geojson")
gdf = gdf.to_crs("EPSG:4326")  # Convert to WGS84
gdf.to_file("converted.geojson", driver="GeoJSON")
```

### Issue: `Out of memory`

**Cause**: Processing large images without resizing.

**Solution**: Use the `--resize` option:
```bash
python3 generate_masks.py --resize 256
```

### Issue: `Permission denied` on output files

**Solution**: Ensure write permissions:
```bash
chmod +w dataset/
```

### Issue: `ProcessPoolExecutor` errors or worker crashes

**Cause**: Multiprocessing with heavy geospatial libraries.

**Solution**: Reduce number of workers or use sequential processing:
```bash
# Use fewer workers
python3 generate_masks_batch.py --num_workers 2

# Or use sequential mode
python3 generate_masks_batch.py --num_workers 1
```

### Issue: `MemoryError` during batch processing

**Cause**: Too many workers causing memory exhaustion.

**Solution**: 
```bash
# Reduce workers and add resize
python3 generate_masks_batch.py --num_workers 2 --resize 256

# Monitor memory
python3 generate_masks_batch.py &
watch -n 1 "ps aux | grep generate_masks"
```

### Issue: Processing is slow even with multiple workers

**Cause**: I/O bottleneck or GDAL/rasterio not optimized.

**Solution**:
```bash
# Set GDAL optimizations
export GDAL_CACHEMAX=512
export GDAL_NUM_THREADS=2

# Run with optimized settings
python3 generate_masks_batch.py --num_workers 4 --resize 256
```

---

## Performance Tips

### 1. Always Test with `--limit` First

```bash
# Test with 10 samples before full run
python3 generate_masks_batch.py --limit 10 --debug --num_workers 1
```

### 2. Choose Optimal Worker Count

```bash
# For 4-core machine
python3 generate_masks_batch.py --num_workers 4

# For 8-core machine
python3 generate_masks_batch.py --num_workers 8

# For memory-constrained systems
python3 generate_masks_batch.py --num_workers 2
```

### 3. Enable GDAL Optimizations

```bash
# Set before running
export GDAL_CACHEMAX=1024  # MB
export GDAL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python3 generate_masks_batch.py --num_workers 4
```

### 4. Use Sequential Mode for Debugging

```bash
# Sequential with verbose logging
python3 generate_masks_batch.py --num_workers 1 --verbose --limit 5
```

### 5. Monitor Progress

```bash
# Terminal 1: Run batch
python3 generate_masks_batch.py --num_workers 4 &

# Terminal 2: Monitor
watch -n 1 "ps aux | grep generate_masks"

# Or check log in real-time
tail -f mask_generation.log
```

### 6. Resume Failed Jobs (Manual)

If processing fails halfway:
```bash
# Check which files were processed
ls dataset/masks/ | wc -l

# Check failed files
cat dataset/failed_files.txt

# For remaining files, filter matched_pairs.json manually
# and re-run with filtered list
```

### 7. Quick Preview with PNG

```bash
# Generate PNG for visual inspection (slower but viewable)
python3 generate_masks_batch.py --format png --limit 20 --debug
```

### 8. SSD Storage for I/O Speed

```bash
# Use SSD for output
python3 generate_masks_batch.py --output_dir /mnt/ssd/dataset/masks

# Or use RAM disk for temp processing (if memory allows)
export TMPDIR=/dev/shm
python3 generate_masks_batch.py
```

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@dataset{spacenet,
  title={SpaceNet Dataset},
  author={SpaceNet Contributors},
  year={2020},
  url={https://spacenet.ai/}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Contact

For questions or issues:

- Open an issue on GitHub
- Email: aditya.men2005@gmail.com

---

**Last Updated**: 2026-04-07

**Version**: 1.1.0
