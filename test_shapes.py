"""
tests/test_shapes.py — Unit Tests for SolarSense ML Pipeline
SolarSense Platform | IEEE YESIST12 WePOWER Track 2026

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import UNetResNet34, DecoderBlock
from loss import DiceLoss, FocalLoss, CombinedLoss
from metrics import (
    iou_score,
    dice_score,
    pixel_accuracy,
    precision_recall_f1,
    MetricTracker,
)
from area_utils import pixels_to_area, area_to_solar_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> UNetResNet34:
    """Instantiate model once for the entire test module."""
    m = UNetResNet34(pretrained=False, dropout=0.3)
    m.eval()
    return m


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Random (B=2) batch of images and binary masks."""
    torch.manual_seed(0)
    images  = torch.randn(2, 3, 256, 256)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
    return images, targets


# ---------------------------------------------------------------------------
# Model shape tests
# ---------------------------------------------------------------------------

class TestModelShapes:
    def test_output_shape(self, model: UNetResNet34) -> None:
        """Model output must be (B, 1, H, W)."""
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 256, 256), f"Got {out.shape}"

    def test_output_no_nan(self, model: UNetResNet34) -> None:
        """Model must not produce NaN logits."""
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "NaN in model output"

    def test_single_image(self, model: UNetResNet34) -> None:
        """Batch size 1 must work without errors."""
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_decoder_block_shape(self) -> None:
        """DecoderBlock output shape must match expectations."""
        block = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        x    = torch.randn(2, 512, 8,  8)
        skip = torch.randn(2, 256, 16, 16)
        with torch.no_grad():
            out = block(x, skip)
        assert out.shape == (2, 256, 16, 16)


# ---------------------------------------------------------------------------
# Freeze / unfreeze tests
# ---------------------------------------------------------------------------

class TestFreezeUnfreeze:
    def test_freeze_encoder(self, model: UNetResNet34) -> None:
        """After freeze_encoder(), all encoder params must have requires_grad=False."""
        model.freeze_encoder()
        frozen = [p for p in model.get_encoder_params() if p.requires_grad]
        assert len(frozen) == 0, f"{len(frozen)} encoder params still require grad"

    def test_unfreeze_encoder(self, model: UNetResNet34) -> None:
        """After unfreeze_encoder(), all encoder params must have requires_grad=True."""
        model.freeze_encoder()
        model.unfreeze_encoder()
        unfrozen = [p for p in model.get_encoder_params() if not p.requires_grad]
        assert len(unfrozen) == 0, f"{len(unfrozen)} encoder params still frozen"

    def test_decoder_params_always_trainable(self, model: UNetResNet34) -> None:
        """Decoder params should always be trainable regardless of encoder state."""
        model.freeze_encoder()
        frozen_dec = [p for p in model.get_decoder_params() if not p.requires_grad]
        assert len(frozen_dec) == 0, "Decoder params should not be frozen"
        model.unfreeze_encoder()   # restore state

    def test_param_count_consistency(self, model: UNetResNet34) -> None:
        """encoder + decoder params should equal total model params."""
        total    = sum(p.numel() for p in model.parameters())
        enc      = sum(p.numel() for p in model.get_encoder_params())
        dec      = sum(p.numel() for p in model.get_decoder_params())
        assert enc + dec == total, (
            f"Encoder ({enc}) + Decoder ({dec}) ≠ Total ({total})"
        )


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def test_dice_loss_scalar(self, batch: tuple) -> None:
        logits, targets = batch
        loss = DiceLoss()(logits, targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_dice_loss_range(self, batch: tuple) -> None:
        logits, targets = batch
        loss = DiceLoss()(logits, targets)
        assert 0.0 <= loss.item() <= 1.0, f"Dice loss {loss.item()} out of [0, 1]"

    def test_focal_loss_scalar(self, batch: tuple) -> None:
        logits, targets = batch
        loss = FocalLoss()(logits, targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_focal_loss_positive(self, batch: tuple) -> None:
        logits, targets = batch
        loss = FocalLoss()(logits, targets)
        assert loss.item() >= 0.0

    def test_combined_loss_scalar(self, batch: tuple) -> None:
        logits, targets = batch
        total, comps = CombinedLoss()(logits, targets)
        assert total.ndim == 0
        assert not torch.isnan(total)

    def test_combined_loss_components(self, batch: tuple) -> None:
        logits, targets = batch
        total, comps = CombinedLoss()(logits, targets)
        for key in ("total", "bce", "dice", "focal"):
            assert key in comps, f"Missing component: {key}"
            assert not np.isnan(comps[key]), f"NaN in component: {key}"

    def test_combined_loss_weights_sum(self) -> None:
        with pytest.raises(AssertionError):
            CombinedLoss(bce_weight=0.5, dice_weight=0.5, focal_weight=0.5)

    def test_perfect_prediction_low_loss(self) -> None:
        """Near-perfect logits should yield very low Dice loss."""
        targets = torch.ones(1, 1, 32, 32)
        logits  = torch.full_like(targets, 10.0)   # very high → sigmoid ≈ 1
        loss = DiceLoss()(logits, targets)
        assert loss.item() < 0.01, f"Expected low loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_iou_range(self, batch: tuple) -> None:
        logits, targets = batch
        iou = iou_score(logits, targets)
        assert 0.0 <= iou <= 1.0

    def test_dice_range(self, batch: tuple) -> None:
        logits, targets = batch
        d = dice_score(logits, targets)
        assert 0.0 <= d <= 1.0

    def test_pixel_accuracy_range(self, batch: tuple) -> None:
        logits, targets = batch
        acc = pixel_accuracy(logits, targets)
        assert 0.0 <= acc <= 1.0

    def test_precision_recall_f1_keys(self, batch: tuple) -> None:
        logits, targets = batch
        result = precision_recall_f1(logits, targets)
        for key in ("precision", "recall", "f1"):
            assert key in result
            assert 0.0 <= result[key] <= 1.0

    def test_perfect_iou(self) -> None:
        """Identical prediction and target should give IoU ≈ 1."""
        targets = torch.ones(2, 1, 64, 64)
        logits  = torch.full_like(targets, 10.0)
        assert iou_score(logits, targets) > 0.99

    def test_metric_tracker_accumulates(self, batch: tuple) -> None:
        logits, targets = batch
        tracker = MetricTracker()
        tracker.update(logits, targets)
        tracker.update(logits, targets)
        results = tracker.compute()
        for key in ("iou", "dice", "pixel_acc", "precision", "recall", "f1"):
            assert key in results

    def test_metric_tracker_reset(self, batch: tuple) -> None:
        logits, targets = batch
        tracker = MetricTracker()
        tracker.update(logits, targets)
        tracker.reset()
        results = tracker.compute()
        assert results == {}, f"Expected empty after reset, got {results}"

    def test_metric_tracker_consistency(self, batch: tuple) -> None:
        """Tracker mean over identical batches should equal single-batch metric."""
        logits, targets = batch
        expected_iou = iou_score(logits, targets)

        tracker = MetricTracker()
        for _ in range(5):
            tracker.update(logits, targets)
        results = tracker.compute()

        assert abs(results["iou"] - expected_iou) < 1e-5


# ---------------------------------------------------------------------------
# Area utility tests
# ---------------------------------------------------------------------------

class TestAreaUtils:
    def test_pixels_to_area_positive(self) -> None:
        """Area metrics must all be positive for a non-zero mask."""
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[80:160, 80:160] = 1.0
        result = pixels_to_area(mask, gsd=0.5)
        assert result["total_roof_area_m2"] > 0
        assert result["usable_area_m2"] > 0
        assert result["estimated_kw_capacity"] > 0
        assert result["estimated_monthly_kwh"] > 0

    def test_pixels_to_area_empty_mask(self) -> None:
        """All-zero mask should give zero area."""
        mask   = np.zeros((256, 256), dtype=np.float32)
        result = pixels_to_area(mask)
        assert result["roof_pixels"] == 0
        assert result["total_roof_area_m2"] == 0.0

    def test_pixels_to_area_gsd_scaling(self) -> None:
        """Doubling GSD should quadruple area (area ∝ gsd²)."""
        mask = np.ones((64, 64), dtype=np.float32)
        r1 = pixels_to_area(mask, gsd=0.5)
        r2 = pixels_to_area(mask, gsd=1.0)
        ratio = r2["total_roof_area_m2"] / r1["total_roof_area_m2"]
        assert abs(ratio - 4.0) < 1e-3, f"Expected 4×, got {ratio}×"

    def test_area_to_solar_metrics_keys(self) -> None:
        result = area_to_solar_metrics(100.0, state="Tamil Nadu")
        for key in ("system_kw_capacity", "annual_kwh",
                    "monthly_kwh_by_month", "estimated_system_cost_inr",
                    "co2_offset_kg_per_year"):
            assert key in result

    def test_area_to_solar_metrics_monthly_length(self) -> None:
        result = area_to_solar_metrics(100.0)
        assert len(result["monthly_kwh_by_month"]) == 12

    def test_area_to_solar_metrics_physical(self) -> None:
        """Annual generation should be positive and plausible."""
        result = area_to_solar_metrics(50.0, state="Rajasthan")
        assert result["annual_kwh"] > 0
        assert result["co2_offset_kg_per_year"] > 0
        assert result["estimated_system_cost_inr"] > 0

    def test_usable_factor_applied(self) -> None:
        mask = np.ones((100, 100), dtype=np.float32)
        r075 = pixels_to_area(mask, gsd=0.5, usable_factor=0.75)
        r050 = pixels_to_area(mask, gsd=0.5, usable_factor=0.50)
        ratio = r075["usable_area_m2"] / r050["usable_area_m2"]
        assert abs(ratio - 1.5) < 1e-3


# ---------------------------------------------------------------------------
# Run directly for quick feedback
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))
