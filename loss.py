"""
loss.py — Loss Functions for Rooftop Segmentation
SolarSense Platform | IEEE YESIST12 WePOWER Track 2026

Losses:
    BCEWithLogitsLoss  — standard binary cross-entropy
    DiceLoss           — overlap-based loss (robust to class imbalance)
    FocalLoss          — penalty-boosted loss for small rooftops
    CombinedLoss       — weighted sum of all three
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Sigmoid is applied inside this class — pass raw logits.

    Args:
        smooth : Laplace smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model output.
            targets : (B, 1, H, W) binary ground-truth mask (0 or 1).

        Returns:
            Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)
        probs_flat   = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        sum_pred     = probs_flat.sum(dim=1)
        sum_target   = targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (sum_pred + sum_target + self.smooth)
        return (1.0 - dice).mean()


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for binary segmentation — designed to up-weight hard, rare
    positive examples (small rooftops vs. large background).

    Args:
        alpha : Weighting factor for the positive class.
        gamma : Focusing parameter (higher = more focus on hard examples).
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) raw model output.
            targets : (B, 1, H, W) binary ground-truth mask.

        Returns:
            Scalar Focal loss.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p_t = torch.exp(-bce)                            # probability of correct class
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Weighted combination of BCE + Dice + Focal losses.

    Default weights are tuned for satellite rooftop segmentation:
        BCE   0.3  — global structure
        Dice  0.5  — segmentation quality
        Focal 0.2  — small-rooftop recovery

    Args:
        bce_weight   : Weight for BCE component.
        dice_weight  : Weight for Dice component.
        focal_weight : Weight for Focal component.
        pos_weight   : Optional positive-class weight tensor for BCE
                       (useful when background >> foreground).
        dice_smooth  : Smooth constant for DiceLoss.
        focal_alpha  : Alpha for FocalLoss.
        focal_gamma  : Gamma for FocalLoss.
    """

    def __init__(
        self,
        bce_weight: float   = 0.3,
        dice_weight: float  = 0.5,
        focal_weight: float = 0.2,
        pos_weight: torch.Tensor | None = None,
        dice_smooth: float  = 1.0,
        focal_alpha: float  = 0.8,
        focal_gamma: float  = 2.0,
    ) -> None:
        super().__init__()
        assert abs(bce_weight + dice_weight + focal_weight - 1.0) < 1e-5, \
            "Loss weights must sum to 1.0"

        self.w_bce   = bce_weight
        self.w_dice  = dice_weight
        self.w_focal = focal_weight

        self.bce   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice  = DiceLoss(smooth=dice_smooth)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            logits  : (B, 1, H, W) raw model output.
            targets : (B, 1, H, W) binary ground-truth mask.

        Returns:
            total_loss : Scalar combined loss tensor (for .backward()).
            components : Dict with keys 'total', 'bce', 'dice', 'focal'.
        """
        targets = targets.float()

        l_bce   = self.bce(logits, targets)
        l_dice  = self.dice(logits, targets)
        l_focal = self.focal(logits, targets)

        total = self.w_bce * l_bce + self.w_dice * l_dice + self.w_focal * l_focal

        components = {
            "total": total.detach().item(),
            "bce":   l_bce.detach().item(),
            "dice":  l_dice.detach().item(),
            "focal": l_focal.detach().item(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Unit test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, W = 4, 256, 256
    logits  = torch.randn(B, 1, H, W)
    targets = torch.randint(0, 2, (B, 1, H, W)).float()

    print("Running loss unit tests …\n")

    # BCE
    bce_fn = nn.BCEWithLogitsLoss()
    l_bce = bce_fn(logits, targets)
    assert not torch.isnan(l_bce), "BCE returned NaN"
    print(f"  BCEWithLogitsLoss : {l_bce.item():.6f}  ✓")

    # Dice
    dice_fn = DiceLoss()
    l_dice = dice_fn(logits, targets)
    assert not torch.isnan(l_dice) and 0.0 <= l_dice.item() <= 1.0
    print(f"  DiceLoss          : {l_dice.item():.6f}  ✓")

    # Focal
    focal_fn = FocalLoss()
    l_focal = focal_fn(logits, targets)
    assert not torch.isnan(l_focal)
    print(f"  FocalLoss         : {l_focal.item():.6f}  ✓")

    # Combined
    combined_fn = CombinedLoss()
    total, comps = combined_fn(logits, targets)
    assert not torch.isnan(total)
    print(f"  CombinedLoss      : {total.item():.6f}  ✓")
    print(f"    components → {comps}")

    print("\nAll loss tests passed ✓")
