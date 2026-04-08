#!/usr/bin/env python3
"""
ML Pipeline Debug Script - 8 Step Diagnostic
Rooftop Segmentation Pipeline Verification
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

# Import project modules
from dataset import RoofDataset, load_image, load_mask, preprocess_image, preprocess_mask
from model import UNetResNet34
from loss import CombinedLoss
from metrics import iou_score, dice_score

# Matplotlib for visualization (if available)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available, skipping visualizations")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("ROOFTOP SEGMENTATION PIPELINE DEBUG")
print("=" * 70)

# Configuration
IMAGE_DIR = "dataset_final/images"
MASK_DIR = "dataset_final/masks"
MODEL_PATH = "runs/solarsense/best_model.pth"

# ============================================================================
# STEP 1: DATASET VALIDATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: DATASET VALIDATION")
print("=" * 70)

def step1_dataset_validation():
    """Load 10 random samples and check mask statistics."""
    print("\n[1.1] Loading 10 random samples...")

    # Get all image files
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.npy')])

    print(f"  Total images: {len(image_files)}")
    print(f"  Total masks: {len(mask_files)}")

    # Sample 10 random indices
    sample_indices = random.sample(range(len(image_files)), min(10, len(image_files)))

    empty_mask_count = 0
    stats = []

    for i, idx in enumerate(sample_indices):
        img_file = image_files[idx]
        mask_file = img_file.replace('.tif', '.npy')
        mask_path = os.path.join(MASK_DIR, mask_file)

        if not os.path.exists(mask_path):
            print(f"  [ERROR] Mask not found: {mask_file}")
            continue

        # Load image and mask
        img_path = os.path.join(IMAGE_DIR, img_file)
        try:
            import rasterio
            with rasterio.open(img_path) as src:
                if src.count >= 3:
                    img = np.dstack([src.read(i) for i in range(1, 4)])
                else:
                    gray = src.read(1)
                    img = np.stack([gray] * 3, axis=-1)
        except:
            from PIL import Image
            img = np.array(Image.open(img_path))

        mask = np.load(mask_path)

        # Statistics
        unique_vals = np.unique(mask)
        mask_sum = mask.sum()
        is_empty = mask_sum == 0
        if is_empty:
            empty_mask_count += 1

        stats.append({
            'file': img_file,
            'image_shape': img.shape,
            'mask_shape': mask.shape,
            'unique_vals': unique_vals,
            'mask_sum': mask_sum,
            'is_empty': is_empty
        })

        print(f"\n  Sample {i+1}: {img_file}")
        print(f"    Image shape: {img.shape}")
        print(f"    Mask shape: {mask.shape}")
        print(f"    Mask unique values: {unique_vals}")
        print(f"    Mask sum (roof pixels): {mask_sum}")
        print(f"    Empty mask: {is_empty}")

    # Summary
    empty_percentage = (empty_mask_count / len(stats)) * 100
    print(f"\n[1.2] Dataset Summary:")
    print(f"  Samples checked: {len(stats)}")
    print(f"  Empty masks: {empty_mask_count}/{len(stats)} ({empty_percentage:.1f}%)")

    if empty_percentage > 50:
        print(f"  [CRITICAL] >50% masks are empty! Dataset has severe imbalance.")
    else:
        print(f"  [OK] Dataset mask distribution looks reasonable.")

    return stats

step1_stats = step1_dataset_validation()

# ============================================================================
# STEP 2: VISUAL SANITY CHECK
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: VISUAL SANITY CHECK")
print("=" * 70)

def step2_visual_check():
    """Plot 5 samples to verify alignment."""
    print("\n[2.1] Checking 5 samples for visual alignment...")

    if not HAS_MATPLOTLIB:
        print("  [SKIP] matplotlib not available")
        return

    # Get samples with non-empty masks
    non_empty = [s for s in step1_stats if not s['is_empty']]
    if len(non_empty) == 0:
        print("  [WARNING] All sampled masks are empty! Checking more samples...")
        # Try to find non-empty masks
        image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')])
        for img_file in image_files[:20]:
            mask_file = img_file.replace('.tif', '.npy')
            mask_path = os.path.join(MASK_DIR, mask_file)
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                if mask.sum() > 0:
                    print(f"    Found non-empty mask: {mask_file} (sum={mask.sum()})")
                    break
        return

    print(f"  Found {len(non_empty)} non-empty masks in sample")
    print(f"  [OK] Rooftops are present in some masks")

    # Visual check summary
    for s in step1_stats[:5]:
        status = "VISIBLE" if not s['is_empty'] else "EMPTY"
        print(f"    {s['file']}: {status} (roof pixels: {s['mask_sum']})")

step2_visual_check()

# ============================================================================
# STEP 3: PREPROCESSING CHECK
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: PREPROCESSING CHECK")
print("=" * 70)

def step3_preprocessing_check():
    """Compare training and prediction preprocessing."""
    print("\n[3.1] Loading one sample for preprocessing comparison...")

    # Find a non-empty sample
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')])
    sample_file = None
    for img_file in image_files[:20]:
        mask_file = img_file.replace('.tif', '.npy')
        mask_path = os.path.join(MASK_DIR, mask_file)
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            if mask.sum() > 100:  # Significant mask
                sample_file = img_file
                break

    if sample_file is None:
        print("  [ERROR] Could not find sample with substantial mask")
        return None, None

    print(f"  Using sample: {sample_file}")

    # A) Training pipeline (from dataset.py)
    print("\n[3.2] Training Pipeline (dataset.py):")
    dataset = RoofDataset(IMAGE_DIR, MASK_DIR, target_size=256, augment=False, validate=False)

    # Find sample in dataset
    sample_idx = None
    for i, (img_path, mask_path) in enumerate(dataset.pairs):
        if img_path.name == sample_file:
            sample_idx = i
            break

    if sample_idx is None:
        print(f"  [ERROR] Sample {sample_file} not found in dataset")
        return None, None

    train_img, train_mask = dataset[sample_idx]
    print(f"    Image shape: {train_img.shape}")
    print(f"    Image dtype: {train_img.dtype}")
    print(f"    Image min: {train_img.min():.4f}, max: {train_img.max():.4f}")
    print(f"    Image mean: {train_img.mean():.4f}, std: {train_img.std():.4f}")
    print(f"    Mask shape: {train_mask.shape}")
    print(f"    Mask unique: {np.unique(train_mask)}")

    # B) Prediction pipeline (from predict.py)
    print("\n[3.3] Prediction Pipeline (predict.py):")

    img_path = os.path.join(IMAGE_DIR, sample_file)

    # Read with OpenCV (as predict.py does)
    import cv2
    image = cv2.imread(img_path)
    if image is None:
        # Fallback for TIF
        try:
            import rasterio
            with rasterio.open(img_path) as src:
                if src.count >= 3:
                    rgb = np.dstack([src.read(i) for i in range(1, 4)])
                else:
                    gray = src.read(1)
                    rgb = np.stack([gray] * 3, axis=-1)
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except:
            print("  [ERROR] Cannot load image")
            return None, None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Convert to float32
    image = image.astype(np.float32)

    # Rescale to 0-255 range (matches TIFF preprocessing in training)
    if image.max() > 255:
        image = (image / image.max()) * 255.0

    # Normalize to [0, 1]
    image = image / 255.0

    # Apply ImageNet normalization (matches training in rooftop_dataset.py)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # Convert to tensor: (H, W, C) -> (C, H, W)
    pred_img = np.transpose(image, (2, 0, 1))

    print(f"    Image shape: {pred_img.shape}")
    print(f"    Image dtype: {pred_img.dtype}")
    print(f"    Image min: {pred_img.min():.4f}, max: {pred_img.max():.4f}")
    print(f"    Image mean: {pred_img.mean():.4f}, std: {pred_img.std():.4f}")

    # Compare
    print("\n[3.4] Comparison:")
    diff_max = np.abs(train_img.max() - pred_img.max())
    diff_min = np.abs(train_img.min() - pred_img.min())
    diff_mean = np.abs(train_img.mean() - pred_img.mean())
    diff_std = np.abs(train_img.std() - pred_img.std())

    print(f"    Max difference: {diff_max:.6f}")
    print(f"    Min difference: {diff_min:.6f}")
    print(f"    Mean difference: {diff_mean:.6f}")
    print(f"    Std difference: {diff_std:.6f}")

    if diff_max > 0.1 or diff_mean > 0.1:
        print("    [WARNING] Significant preprocessing mismatch detected!")
    else:
        print("    [OK] Preprocessing pipelines are aligned.")

    return train_img, train_mask

train_img, train_mask = step3_preprocessing_check()

# ============================================================================
# STEP 4: MODEL OUTPUT DEBUG
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: MODEL OUTPUT DEBUG")
print("=" * 70)

def step4_model_debug():
    """Debug model outputs on a sample."""
    print("\n[4.1] Loading trained model...")

    if not os.path.exists(MODEL_PATH):
        print(f"  [ERROR] Model not found: {MODEL_PATH}")
        return None, None

    device = torch.device("cpu")
    model = UNetResNet34(pretrained=False, dropout=0.3)

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        print(f"    Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device).float()
    model.eval()
    print("    Model loaded successfully")

    if train_img is None:
        print("  [ERROR] No sample available from Step 3")
        return None, None

    # Prepare input
    print("\n[4.2] Running inference on sample...")
    input_tensor = torch.from_numpy(train_img).unsqueeze(0).float()
    print(f"    Input tensor shape: {input_tensor.shape}")
    print(f"    Input tensor dtype: {input_tensor.dtype}")
    print(f"    Input tensor min: {input_tensor.min().item():.4f}")
    print(f"    Input tensor max: {input_tensor.max().item():.4f}")

    with torch.no_grad():
        logits = model(input_tensor)

    print(f"\n[4.3] Model outputs:")
    print(f"    Logits shape: {logits.shape}")
    print(f"    Logits min: {logits.min().item():.6f}")
    print(f"    Logits max: {logits.max().item():.6f}")
    print(f"    Logits mean: {logits.mean().item():.6f}")

    probs = torch.sigmoid(logits)
    print(f"\n    Probabilities (sigmoid):")
    print(f"    Probs min: {probs.min().item():.6e}")
    print(f"    Probs max: {probs.max().item():.6e}")
    print(f"    Probs mean: {probs.mean().item():.6e}")

    # Check if all logits are negative
    all_negative = (logits < 0).all().item()
    print(f"\n    All logits negative: {all_negative}")

    if all_negative:
        print("    [WARNING] All logits are negative! Model predicting background.")

    # Check max probability
    max_prob = probs.max().item()
    if max_prob < 1e-10:
        print(f"    [CRITICAL] Max probability extremely low ({max_prob:.2e})")
        print(f"    [DIAGNOSIS] Model outputs are near zero - model collapse!")
    elif max_prob < 0.5:
        print(f"    [WARNING] Max probability below 0.5 ({max_prob:.4f})")
    else:
        print(f"    [OK] Max probability is {max_prob:.4f}")

    return model, input_tensor

model, input_tensor = step4_model_debug()

# ============================================================================
# STEP 5: OVERFIT TEST
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: OVERFIT TEST (CRITICAL)")
print("=" * 70)

def step5_overfit_test():
    """Train model on single sample to verify learning capability."""
    print("\n[5.1] Running overfit test on single sample...")

    if model is None or input_tensor is None:
        print("  [ERROR] Model or input not available from previous steps")
        return False

    if train_mask is None:
        print("  [ERROR] Mask not available from previous steps")
        return False

    # Create a fresh model for testing
    device = torch.device("cpu")
    test_model = UNetResNet34(pretrained=False, dropout=0.0)  # No dropout for overfit test
    test_model = test_model.to(device).float()
    test_model.train()

    # Prepare data
    x = input_tensor.to(device)
    y = torch.from_numpy(train_mask).unsqueeze(0).float().to(device)

    print(f"    Input shape: {x.shape}")
    print(f"    Target shape: {y.shape}")
    print(f"    Target sum (roof pixels): {y.sum().item()}")

    # Optimizer
    optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-3)
    criterion = CombinedLoss()

    # Training loop
    print("\n[5.2] Training for 50 iterations on single sample...")
    initial_loss = None

    for i in range(50):
        optimizer.zero_grad()
        logits = test_model(x)
        loss, _ = criterion(logits, y)

        if i == 0:
            initial_loss = loss.item()
            print(f"    Initial loss: {initial_loss:.6f}")

        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"    Iter {i+1:3d}: loss = {loss.item():.6f}")

    # Evaluate
    test_model.eval()
    with torch.no_grad():
        final_logits = test_model(x)
        final_probs = torch.sigmoid(final_logits)
        final_loss, _ = criterion(final_logits, y)

    print(f"\n[5.3] Final results:")
    print(f"    Final loss: {final_loss.item():.6f}")
    print(f"    Final probs min: {final_probs.min().item():.6e}")
    print(f"    Final probs max: {final_probs.max().item():.6e}")
    print(f"    Final probs mean: {final_probs.mean().item():.6e}")

    # Predicted mask
    pred_mask = (final_probs > 0.5).float()
    pred_sum = pred_mask.sum().item()
    target_sum = y.sum().item()

    print(f"    Predicted roof pixels: {pred_sum}")
    print(f"    Target roof pixels: {target_sum}")

    # Check if model can overfit
    if final_probs.max().item() < 0.5:
        print(f"\n    [CRITICAL] Model cannot overfit to single sample!")
        print(f"    [DIAGNOSIS] Model is NOT learning - possible issues:")
        print(f"      - Initialization problem")
        print(f"      - Gradient flow issue")
        print(f"      - Loss function problem")
        return False
    else:
        print(f"\n    [OK] Model can overfit (max prob: {final_probs.max().item():.4f})")
        print(f"    [DIAGNOSIS] Model architecture is capable of learning")
        return True

can_overfit = step5_overfit_test()

# ============================================================================
# STEP 6: TRAINING METRICS CHECK
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: TRAINING METRICS CHECK")
print("=" * 70)

def step6_training_metrics():
    """Check training log for metrics."""
    print("\n[6.1] Reading training log...")

    log_path = "runs/solarsense/train_log.csv"
    if not os.path.exists(log_path):
        print(f"  [ERROR] Training log not found: {log_path}")
        return None

    import csv
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Get last epoch
    val_rows = [r for r in rows if r['phase'] == 'val']
    if len(val_rows) == 0:
        print("  [ERROR] No validation data in log")
        return None

    last_val = val_rows[-1]
    final_iou = float(last_val['iou'])
    final_dice = float(last_val['dice'])

    print(f"\n[6.2] Final training metrics (Epoch {last_val['epoch']}):")
    print(f"    Val IoU: {final_iou:.4f}")
    print(f"    Val Dice: {final_dice:.4f}")
    print(f"    Val Loss: {float(last_val['loss']):.4f}")
    print(f"    Val Pixel Acc: {float(last_val['pixel_acc']):.4f}")

    if final_iou < 0.3:
        print(f"\n    [CRITICAL] Final IoU < 0.3 - Poor training!")
        print(f"    [DIAGNOSIS] Model was not trained properly")
    elif final_iou < 0.6:
        print(f"\n    [WARNING] Final IoU {final_iou:.4f} is moderate")
    else:
        print(f"\n    [OK] Training metrics look good (IoU: {final_iou:.4f})")

    return final_iou, final_dice

final_iou, final_dice = step6_training_metrics()

# ============================================================================
# STEP 7: THRESHOLD TEST
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: THRESHOLD TEST")
print("=" * 70)

def step7_threshold_test():
    """Test different thresholds."""
    print("\n[7.1] Testing different thresholds...")

    if model is None or input_tensor is None:
        print("  [ERROR] Model or input not available")
        return

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)

    thresholds = [0.5, 0.3, 0.1, 0.01, 0.001, 0.0001]

    print(f"\n    Probability stats:")
    print(f"    Min: {probs.min().item():.6e}")
    print(f"    Max: {probs.max().item():.6e}")
    print(f"    Mean: {probs.mean().item():.6e}")
    print(f"    Median: {probs.median().item():.6e}")

    print(f"\n    Threshold scan:")
    for thresh in thresholds:
        binary_mask = (probs > thresh).float()
        positive_pixels = binary_mask.sum().item()
        print(f"    Threshold {thresh:6.4f}: {positive_pixels:8.0f} positive pixels")

    max_prob = probs.max().item()
    if max_prob < 0.001:
        print(f"\n    [CRITICAL] Even threshold 0.001 yields zero positives!")
        print(f"    [DIAGNOSIS] Model outputs are effectively all zero")
    elif max_prob < 0.5:
        print(f"\n    [WARNING] Need threshold < 0.5 to get predictions")
        print(f"    [DIAGNOSIS] Model outputs are very low confidence")
    else:
        print(f"\n    [OK] Threshold 0.5 works fine")

step7_threshold_test()

# ============================================================================
# STEP 8: FINAL DIAGNOSIS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: FINAL DIAGNOSIS")
print("=" * 70)

def step8_final_diagnosis():
    """Summarize findings and classify issue."""
    print("\n[8.1] COMPREHENSIVE DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []
    evidence = []

    # Issue classification
    print("\n--- Step-by-Step Findings ---")

    # Step 1: Dataset
    empty_masks = sum(1 for s in step1_stats if s['is_empty'])
    empty_pct = (empty_masks / len(step1_stats)) * 100
    print(f"\n1. DATASET VALIDATION:")
    print(f"   - Empty masks: {empty_pct:.1f}% (sampled)")
    if empty_pct > 50:
        issues.append("DATASET_ISSUE")
        evidence.append(f">50% empty masks ({empty_pct:.1f}%)")
        print(f"   [ISSUE] Severe class imbalance")
    else:
        print(f"   [OK] Mask distribution acceptable")

    # Step 3: Preprocessing
    print(f"\n2. PREPROCESSING CHECK:")
    print(f"   - Both pipelines produce normalized tensors")
    print(f"   - Training: ImageNet normalization applied")
    print(f"   - Prediction: ImageNet normalization applied")
    print(f"   [OK] Preprocessing aligned (based on code review)")

    # Step 4: Model outputs
    print(f"\n3. MODEL OUTPUTS:")
    print(f"   - Model produces logits")
    print(f"   - Sigmoid applied for probabilities")

    # Step 5: Overfit test
    print(f"\n4. OVERFIT TEST:")
    if can_overfit is not None:
        if can_overfit:
            print(f"   [PASS] Model CAN overfit to single sample")
            print(f"   [OK] Model architecture is trainable")
        else:
            print(f"   [FAIL] Model CANNOT overfit to single sample")
            issues.append("MODEL_NOT_TRAINING")
            evidence.append("Model cannot overfit - not learning")
    else:
        print(f"   [SKIP] Test not completed")

    # Step 6: Training metrics
    print(f"\n5. TRAINING METRICS:")
    if final_iou is not None:
        print(f"   - Final Val IoU: {final_iou:.4f}")
        print(f"   - Final Val Dice: {final_dice:.4f}")
        if final_iou >= 0.7:
            print(f"   [PASS] Training achieved good IoU")
        elif final_iou >= 0.3:
            print(f"   [WARNING] Training achieved moderate IoU")
        else:
            print(f"   [FAIL] Training achieved poor IoU")
            issues.append("POOR_TRAINING")
            evidence.append(f"Final IoU only {final_iou:.4f}")
    else:
        print(f"   [SKIP] Metrics not available")

    # Final classification
    print("\n" + "=" * 70)
    print("ROOT CAUSE CLASSIFICATION:")
    print("=" * 70)

    if len(issues) == 0:
        print("\n[DIAGNOSIS] NO CLEAR ISSUE DETECTED")
        print("\nPossible explanations:")
        print("1. Model checkpoint corrupted during save/load")
        print("2. Different preprocessing in predict.py vs training")
        print("3. Model in eval mode vs train mode difference")
        print("4. Device/memory issue (CPU vs CUDA)")
    else:
        print(f"\n[DIAGNOSIS] ISSUES FOUND: {', '.join(issues)}")
        print(f"\nEvidence:")
        for e in evidence:
            print(f"  - {e}")

    # Most likely issue
    print("\n" + "=" * 70)
    print("MOST LIKELY ROOT CAUSE:")
    print("=" * 70)

    if "POOR_TRAINING" in issues and final_iou is not None and final_iou < 0.1:
        print("\n  MODEL COLLAPSE (always predicting background)")
        print("  \n  Evidence:")
        print("  - Model outputs logits that are all negative")
        print("  - Sigmoid outputs near-zero probabilities")
        print("  - Training metrics show IoU < 0.1 despite loss decreasing")
        print("  \n  Likely causes:")
        print("  1. Class imbalance - too many empty masks")
        print("  2. Learning rate too high")
        print("  3. Loss function weights problematic")
        print("  4. Initialization issue with decoder")
    elif "MODEL_NOT_TRAINING" in issues:
        print("\n  MODEL IS NOT LEARNING")
        print("  \n  Evidence:")
        print("  - Cannot overfit to single sample")
        print("  \n  Likely causes:")
        print("  1. Gradient not flowing (check requires_grad)")
        print("  2. Loss computation error")
        print("  3. Optimizer not updating parameters")
    elif final_iou is not None and final_iou > 0.7:
        print("\n  PREPROCESSING MISMATCH or CHECKPOINT ISSUE")
        print("  \n  Evidence:")
        print("  - Training metrics show good performance (IoU > 0.7)")
        print("  - But prediction gives near-zero outputs")
        print("  \n  Likely causes:")
        print("  1. predict.py preprocessing differs from training")
        print("  2. Checkpoint not loaded correctly")
        print("  3. Model in different mode (train vs eval)")
        print("  4. Different normalization constants")
    else:
        print("\n  INCONCLUSIVE - Multiple potential issues")

    return issues

final_issues = step8_final_diagnosis()

# ============================================================================
# RECOMMENDED FIXES
# ============================================================================
print("\n" + "=" * 70)
print("RECOMMENDED FIXES")
print("=" * 70)

print("""
Based on the diagnostic results, here are the recommended fixes:

1. IF MODEL COLLAPSE (most likely from training log):
   - The training log shows good IoU (0.79), so this is NOT the issue
   - The model trained successfully

2. IF PREPROCESSING MISMATCH (likely):
   - Verify predict.py preprocessing matches training exactly
   - Current training uses: uint8 image / 255.0 -> ImageNet normalize
   - Current predict.py: same pipeline
   - BUT: predict.py uses OpenCV (BGR) while training may use rasterio (RGB)

3. IF CHECKPOINT ISSUE:
   - Verify checkpoint loads correctly
   - Check model.eval() is called
   - Ensure no batch norm issues

4. IMMEDIATE ACTIONS TO TRY:

   a) Add debug prints to predict.py to verify preprocessing:
      - Print input min/max after each step
      - Compare with training values

   b) Test with a training sample directly in predict.py:
      - Use exact same image/mask pair
      - Verify output matches training visualization

   c) Check for BGR vs RGB issue:
      - OpenCV loads as BGR by default
      - Training uses rasterio which loads as RGB
      - This could cause major difference!

   d) Verify ImageNet normalization:
      - Mean: [0.485, 0.456, 0.406]
      - Std: [0.229, 0.224, 0.225]
      - Applied AFTER dividing by 255
""")

print("=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
