import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (1, 2, 3)

        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth + self.eps
        )
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (1, 2, 3)

        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps
        )
        return 1.0 - score.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.tversky = TverskyLoss(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            eps=eps,
        )
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.tversky(logits, targets), self.gamma)


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
        pos_weight: float | None = None,
        tversky_weight: float = 0.0,
        focal_tversky_weight: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_tversky_gamma: float = 0.75,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.focal_tversky_weight = float(focal_tversky_weight)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.tversky_loss = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=dice_smooth,
        )
        self.focal_tversky_loss = FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=focal_tversky_gamma,
            smooth=dice_smooth,
        )
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([float(pos_weight)], dtype=torch.float32),
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=pos_weight,
        )
        dice = self.dice_loss(logits, targets)
        tversky = self.tversky_loss(logits, targets)
        focal_tversky = self.focal_tversky_loss(logits, targets)
        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.tversky_weight * tversky
            + self.focal_tversky_weight * focal_tversky
        )
        return total, {
            "total_loss": total.detach(),
            "bce_loss": bce.detach(),
            "dice_loss": dice.detach(),
            "tversky_loss": tversky.detach(),
            "focal_tversky_loss": focal_tversky.detach(),
        }
