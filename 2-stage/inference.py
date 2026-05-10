import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for a trained 2D U-Net checkpoint on one .npy CT slice."
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--input", required=True, type=str, help="Path to input .npy slice.")
    parser.add_argument("--output-mask", default=None, type=str)
    parser.add_argument("--output-prob", default=None, type=str)
    parser.add_argument(
        "--threshold",
        default=None,
        type=float,
        help="Override probability threshold. Defaults to checkpoint best_threshold.",
    )
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--img-size", default=None, type=int)
    return parser.parse_args()


def load_checkpoint(path: str | Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def as_chw_float32(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3:
        if array.shape[0] == 1:
            pass
        elif array.shape[-1] == 1:
            array = np.moveaxis(array, -1, 0)
        else:
            raise ValueError(f"Unsupported input shape: {array.shape}")
    else:
        raise ValueError(f"Unsupported input shape: {array.shape}")
    return array.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get(
        "model",
        {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 32,
            "bilinear": False,
        },
    )
    img_size = args.img_size or int(config.get("data", {}).get("img_size", 512))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(
            checkpoint.get(
                "best_threshold",
                config.get("metrics", {}).get("threshold", 0.5),
            )
        )
    )

    image = np.load(args.input, allow_pickle=False)
    image = as_chw_float32(image)
    expected_shape = (1, img_size, img_size)
    if tuple(image.shape) != expected_shape:
        raise ValueError(f"Input has shape {image.shape}, expected {expected_shape}")

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    mask = (prob >= threshold).astype(np.uint8)

    input_path = Path(args.input)
    output_mask = Path(args.output_mask) if args.output_mask else input_path.with_name(
        f"{input_path.stem}_pred_mask.npy"
    )
    output_prob = Path(args.output_prob) if args.output_prob else input_path.with_name(
        f"{input_path.stem}_pred_prob.npy"
    )
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    output_prob.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mask, mask)
    np.save(output_prob, prob)

    print(f"Saved mask: {output_mask}")
    print(f"Saved probability map: {output_prob}")
    print(f"Threshold: {threshold}")


if __name__ == "__main__":
    main()
