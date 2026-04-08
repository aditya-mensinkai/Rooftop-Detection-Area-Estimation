"""
Rooftop Segmentation Inference Script

Predicts rooftop segmentation mask from a single satellite image,
calculates area metrics, and estimates solar potential.

Usage:
    python predict.py --image_path sample.tif --gsd 0.5 --state Karnataka

Output:
    - outputs/mask.png    : Binary segmentation mask
    - outputs/viz.png     : Side-by-side visualization
    - Console             : Area and solar metrics
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from area_utils import area_to_solar_metrics, pixels_to_area
from model import UNetResNet34

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ImageNet normalization constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Model input size
TARGET_SIZE = 256


def load_model(model_path: str, device: torch.device) -> UNetResNet34:
    """
    Load trained UNetResNet34 model from checkpoint.

    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Initialize model (pretrained=False since we're loading trained weights)
    model = UNetResNet34(pretrained=False, dropout=0.3)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"  Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)

    # 🔥 FORCE EVERYTHING TO FLOAT32
    model = model.to(device).float()

    # 🔥 EXTRA SAFETY: Ensure all parameters are float32
    for param in model.parameters():
        param.data = param.data.float()

    model.eval()

    logger.info(f"  Model loaded successfully on {device}")
    return model


def preprocess(image_path: str, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for inference.

    Args:
        image_path: Path to input image
        device: Device to move tensor to

    Returns:
        Tuple of (preprocessed tensor [1,3,256,256], original image array)
    """
    logger.info(f"Preprocessing image: {image_path}")

    # Load image using rasterio to match training pipeline (dataset.py)
    import rasterio
    with rasterio.open(image_path) as src:
        if src.count >= 3:
            image = np.dstack([src.read(i) for i in range(1, 4)])
        elif src.count == 1:
            gray = src.read(1)
            image = np.stack([gray] * 3, axis=-1)
        else:
            raise ValueError(f"Unsupported band count: {src.count}")

        # Store original for visualization (before any processing)
        if image.dtype != np.uint8:
            if image.max() > 255:
                original = (image / image.max() * 255).astype(np.uint8)
            else:
                original = image.astype(np.uint8)
        else:
            original = image.copy()

        # Ensure uint8 for consistent processing (matches dataset.py)
        if image.dtype != np.uint8:
            if image.max() > 255:
                image = (image / image.max() * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

    # Resize to target size using PIL (matches dataset.py preprocessing)
    from PIL import Image
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    image = np.array(image_pil)

    # Convert to float32
    image = image.astype(np.float32)

    # Normalize to [0, 1] (matches training in dataset.py)
    image = image / 255.0

    # Convert to tensor: (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)

    # Add batch dimension: (1, C, H, W) and ensure float32
    image_tensor = image_tensor.unsqueeze(0).to(device).float()

    logger.info(f"  Image shape: {original.shape} -> tensor {image_tensor.shape}")
    return image_tensor, original


def predict(model: UNetResNet34, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run inference on preprocessed image.

    Args:
        model: Loaded UNetResNet34 model
        image_tensor: Preprocessed image tensor [1,3,H,W]

    Returns:
        Binary mask tensor [H,W] with values 0 or 1
    """
    logger.info("Running inference...")

    # 🔍 DEBUG: Print dtypes and input range
    print("  Model dtype:", next(model.parameters()).dtype)
    print("  Input dtype:", image_tensor.dtype)
    print("  Input min:", image_tensor.min().item())
    print("  Input max:", image_tensor.max().item())

    # 🔥 EXTRA SAFETY: Ensure float32
    image_tensor = image_tensor.float()

    with torch.no_grad():
        # Forward pass (returns logits)
        logits = model(image_tensor)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # 🔍 DEBUG: Print max probability
        print("  Max probability:", probs.max().item())

        # Threshold at 0.3 to get binary mask (lower threshold for better sensitivity)
        binary_mask = (probs > 0.3).float()

    # Remove batch and channel dimensions -> (H, W)
    binary_mask = binary_mask.squeeze()

    logger.info(f"  Prediction complete. Mask shape: {binary_mask.shape}")
    return binary_mask


def postprocess(pred_mask: torch.Tensor) -> np.ndarray:
    """
    Convert prediction tensor to numpy array.

    Args:
        pred_mask: Binary mask tensor [H,W]

    Returns:
        Binary numpy array [H,W] with values 0 or 1
    """
    # Move to CPU and convert to numpy
    mask_np = pred_mask.cpu().numpy()

    # Ensure binary values
    mask_np = (mask_np > 0).astype(np.uint8)

    return mask_np


def save_outputs(
    original: np.ndarray,
    binary_mask: np.ndarray,
    output_dir: str,
) -> Tuple[str, str]:
    """
    Save mask and visualization images.

    Args:
        original: Original image array [H,W,3]
        binary_mask: Binary mask array [H,W] with values 0 or 1
        output_dir: Directory to save outputs

    Returns:
        Tuple of (mask_path, viz_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save mask as grayscale image (0-255)
    mask_path = os.path.join(output_dir, "mask.png")
    mask_img = (binary_mask * 255).astype(np.uint8)
    Image.fromarray(mask_img, mode="L").save(mask_path)
    logger.info(f"  Saved mask: {mask_path}")

    # Create side-by-side visualization
    # Resize original to match mask size if needed
    if original.shape[:2] != binary_mask.shape:
        original = cv2.resize(original, (binary_mask.shape[1], binary_mask.shape[0]))

    # Convert original to PIL Image
    original_pil = Image.fromarray(original)

    # Create overlay
    mask_rgba = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)
    mask_rgba[binary_mask == 1] = [255, 0, 0, 128]  # Red semi-transparent
    mask_overlay = Image.fromarray(mask_rgba)
    overlay_img = Image.alpha_composite(
        original_pil.convert("RGBA"),
        mask_overlay
    ).convert("RGB")

    # Create side-by-side visualization
    viz_width = binary_mask.shape[1] * 3
    viz_height = binary_mask.shape[0]
    viz_img = Image.new("RGB", (viz_width, viz_height))

    # Original image
    viz_img.paste(original_pil, (0, 0))

    # Predicted mask (grayscale -> RGB)
    mask_rgb = Image.fromarray(mask_img).convert("RGB")
    viz_img.paste(mask_rgb, (binary_mask.shape[1], 0))

    # Overlay
    viz_img.paste(overlay_img, (binary_mask.shape[1] * 2, 0))

    # Save visualization
    viz_path = os.path.join(output_dir, "viz.png")
    viz_img.save(viz_path)
    logger.info(f"  Saved visualization: {viz_path}")

    return mask_path, viz_path


def print_results(area_info: dict, solar_info: dict) -> None:
    """
    Print formatted prediction results.

    Args:
        area_info: Output from pixels_to_area()
        solar_info: Output from area_to_solar_metrics()
    """
    # Format currency
    cost_inr = solar_info["estimated_system_cost_inr"]
    cost_formatted = f"₹{cost_inr:,.0f}"

    print("\n" + "=" * 40)
    print("===== PREDICTION RESULTS =====")
    print("=" * 40)
    print(f"Roof pixels:     {area_info['roof_pixels']:,.0f}")
    print(f"Total area:      {area_info['total_roof_area_m2']:,.1f} m²")
    print(f"Usable area:     {area_info['usable_area_m2']:,.1f} m²")
    print()
    print("=" * 40)
    print("===== SOLAR ESTIMATION =====")
    print("=" * 40)
    print(f"Capacity:        {solar_info['system_kw_capacity']:.2f} kW")
    print(f"Annual energy:   {solar_info['annual_kwh']:,.0f} kWh")
    print(f"System cost:     {cost_formatted}")
    print(f"CO₂ saved:       {solar_info['co2_offset_kg_per_year']:,.0f} kg/year")
    print("=" * 40 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rooftop Segmentation Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image_path sample.tif
  python predict.py --image_path sample.tif --gsd 0.3 --state Maharashtra
        """,
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input satellite image",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/solarsense/best_model.pth",
        help="Path to trained model checkpoint (default: runs/solarsense/best_model.pth)",
    )

    parser.add_argument(
        "--gsd",
        type=float,
        default=0.5,
        help="Ground Sampling Distance in meters per pixel (default: 0.5)",
    )

    parser.add_argument(
        "--state",
        type=str,
        default="Karnataka",
        help="Indian state for solar calculations (default: Karnataka)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified.",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load model
        model = load_model(args.model_path, device)

        # Preprocess image
        image_tensor, original = preprocess(args.image_path, device)

        # Run prediction
        pred_mask_tensor = predict(model, image_tensor)

        # Postprocess
        binary_mask = postprocess(pred_mask_tensor)

        # Calculate area
        logger.info("Calculating area metrics...")
        area_info = pixels_to_area(binary_mask, gsd=args.gsd)

        # Calculate solar metrics
        logger.info(f"Calculating solar metrics for {args.state}...")
        solar_info = area_to_solar_metrics(area_info["usable_area_m2"], state=args.state)

        # Save outputs
        logger.info(f"Saving outputs to {args.output_dir}/")
        save_outputs(original, binary_mask, args.output_dir)

        # Print results
        print_results(area_info, solar_info)

        logger.info("Prediction complete!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
