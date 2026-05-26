import torch


@torch.no_grad()
def confusion_from_probabilities(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    preds = probs >= threshold
    target = targets >= 0.5

    tp = (preds & target).sum(dtype=torch.float64)
    fp = (preds & ~target).sum(dtype=torch.float64)
    fn = (~preds & target).sum(dtype=torch.float64)
    tn = (~preds & ~target).sum(dtype=torch.float64)
    return torch.stack([tp, fp, fn, tn])


@torch.no_grad()
def confusion_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    return confusion_from_probabilities(
        torch.sigmoid(logits),
        targets,
        threshold=threshold,
    )


def metrics_from_confusion(
    confusion: torch.Tensor,
    eps: float = 1e-7,
) -> dict[str, float]:
    tp, fp, fn, tn = [value.item() for value in confusion.detach().cpu()]

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }
