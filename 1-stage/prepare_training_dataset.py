import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# =========================
# SETTINGS
# =========================
# These defaults are used when running:
# python 1-stage/prepare_training_dataset.py
DATA_DIR = PROJECT_ROOT / "preprocessed_npy"
OUTPUT_DIR = PROJECT_ROOT / "prepared_npy"
CONFIG_PATH = SCRIPT_DIR / "prepare_training_config.yaml"

MODE = "patch"  # "patch" or "full_slice"
IMG_SIZE = 512
PATCH_SIZE = 256
PATCH_CENTER_JITTER = 64
NEGATIVE_RATIO = 1.0
AUGMENTATIONS_PER_POSITIVE = 4
AUGMENTATIONS_PER_NEGATIVE = 2
SEED = 2004
TORCH_NUM_THREADS = 4
OVERWRITE_OUTPUT = True

REQUIRED_MANIFEST_COLUMNS = {
    "split",
    "case_id",
    "z",
    "has_tumor",
    "image_path",
    "mask_path",
    "original_image_path",
    "original_mask_path",
}

REQUIRED_COLUMNS_ORDER = [
    "split",
    "case_id",
    "z",
    "has_tumor",
    "image_path",
    "mask_path",
    "original_image_path",
    "original_mask_path",
]

EXTRA_COLUMNS = [
    "prepared_mode",
    "prepared_sample_id",
    "prepared_variant",
    "is_augmented",
    "source_split",
    "source_case_id",
    "source_z",
    "source_image_path",
    "source_mask_path",
]

DEFAULT_AUGMENTATION = {
    "enabled": True,
    "horizontal_flip_p": 0.5,
    "rotation_degrees": 10.0,
    "shift_fraction": 0.05,
    "scale_limit": 0.10,
    "intensity_scale": 0.10,
    "intensity_shift": 0.05,
    "gaussian_noise_std": 0.01,
    "elastic_p": 0.2,
    "elastic_alpha": 4.0,
    "elastic_kernel_size": 17,
    "clip_image": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an offline train dataset with balanced slices, train-only "
            "augmentations, and optional tumor-centered patches."
        )
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
    )
    parser.add_argument("--mode", choices=["patch", "full_slice"], default=MODE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--patch-center-jitter", type=int, default=PATCH_CENTER_JITTER)
    parser.add_argument("--negative-ratio", type=float, default=NEGATIVE_RATIO)
    parser.add_argument(
        "--augmentations-per-positive",
        type=int,
        default=AUGMENTATIONS_PER_POSITIVE,
    )
    parser.add_argument(
        "--augmentations-per-negative",
        type=int,
        default=AUGMENTATIONS_PER_NEGATIVE,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--torch-num-threads", type=int, default=TORCH_NUM_THREADS)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=OVERWRITE_OUTPUT)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)

    cwd_path = (Path.cwd() / path).resolve(strict=False)
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve(strict=False)


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)

    cwd_path = (Path.cwd() / path).resolve(strict=False)
    if cwd_path.exists():
        return cwd_path
    return (SCRIPT_DIR / path).resolve(strict=False)


def load_augmentation_config(config_path: str | Path) -> dict[str, Any]:
    config = dict(DEFAULT_AUGMENTATION)
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict) and isinstance(loaded.get("augmentation"), dict):
            config.update(loaded["augmentation"])
    return config


def load_splits(data_dir: str | Path) -> dict[str, Any]:
    splits_path = Path(data_dir) / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path}")
    with splits_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(data_dir: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(data_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_MANIFEST_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                "manifest.csv is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"manifest.csv is empty: {manifest_path}")
    return rows


def path_candidates(
    raw_path: str | Path,
    data_dir: Path,
    manifest_dir: Path,
    split: str,
    case_id: str,
    kind: str,
) -> list[Path]:
    raw = Path(str(raw_path))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)

    for marker in (data_dir.name, "preprocessed_npy"):
        if marker in raw.parts:
            idx = raw.parts.index(marker)
            if idx + 1 < len(raw.parts):
                candidates.append(data_dir / Path(*raw.parts[idx + 1 :]))

    folder = "images" if kind == "image" else "masks"
    candidates.append(data_dir / split / case_id / folder / raw.name)

    if not raw.is_absolute():
        candidates.extend([manifest_dir / raw, Path.cwd() / raw, data_dir / raw])

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.expanduser().resolve(strict=False))
        if normalized not in seen:
            seen.add(normalized)
            unique.append(Path(normalized))
    return unique


def validate_resolved_sample_path(
    path: str | Path,
    split: str,
    case_id: str,
    kind: str,
) -> None:
    path = Path(path).expanduser().resolve(strict=False)
    folder = "images" if kind == "image" else "masks"
    parts = path.parts
    expected = (str(split), str(case_id), folder)
    for index in range(0, max(len(parts) - 2, 0)):
        if parts[index : index + 3] == expected:
            return
    raise ValueError(
        "Data leakage risk: resolved manifest path does not match row split/case. "
        f"kind={kind}, split={split!r}, case_id={case_id!r}, path={path}"
    )


def validate_manifest_reference(
    raw_path: str | Path,
    split: str,
    case_id: str,
    kind: str,
) -> None:
    raw = Path(str(raw_path))
    folder = "images" if kind == "image" else "masks"
    parts = raw.parts
    expected = (str(split), str(case_id), folder)

    for index in range(0, max(len(parts) - 2, 0)):
        if parts[index : index + 3] == expected:
            return

    split_names = {"train", "val", "test"}
    for index in range(0, max(len(parts) - 2, 0)):
        if parts[index] in split_names and parts[index + 2] == folder:
            raise ValueError(
                "Data leakage risk: manifest path points to a different "
                f"split/case than the row metadata. kind={kind}, "
                f"split={split!r}, case_id={case_id!r}, raw_path={raw_path!r}"
            )


def resolve_manifest_path(
    raw_path: str | Path,
    data_dir: str | Path,
    manifest_dir: str | Path,
    split: str,
    case_id: str,
    kind: str,
) -> Path:
    candidates = path_candidates(
        raw_path=raw_path,
        data_dir=Path(data_dir),
        manifest_dir=Path(manifest_dir),
        split=split,
        case_id=case_id,
        kind=kind,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    formatted = "\n  - ".join(str(path) for path in candidates[:6])
    raise FileNotFoundError(
        f"Cannot resolve {kind} path from manifest value {raw_path!r}. "
        f"Checked:\n  - {formatted}"
    )


def prepare_manifest_rows(
    rows: list[dict[str, Any]],
    data_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_dir = Path(data_dir).expanduser().resolve(strict=False)
    prepared: list[dict[str, Any]] = []
    for row in rows:
        split = str(row["split"])
        case_id = str(row["case_id"])
        validate_manifest_reference(
            row["image_path"],
            split=split,
            case_id=case_id,
            kind="image",
        )
        validate_manifest_reference(
            row["mask_path"],
            split=split,
            case_id=case_id,
            kind="mask",
        )
        image_path = resolve_manifest_path(
            row["image_path"],
            data_dir=data_dir,
            manifest_dir=data_dir,
            split=split,
            case_id=case_id,
            kind="image",
        )
        mask_path = resolve_manifest_path(
            row["mask_path"],
            data_dir=data_dir,
            manifest_dir=data_dir,
            split=split,
            case_id=case_id,
            kind="mask",
        )
        validate_resolved_sample_path(
            image_path,
            split=split,
            case_id=case_id,
            kind="image",
        )
        validate_resolved_sample_path(
            mask_path,
            split=split,
            case_id=case_id,
            kind="mask",
        )
        updated = dict(row)
        updated["_image_path"] = str(image_path)
        updated["_mask_path"] = str(mask_path)
        prepared.append(updated)

    return prepared, {
        "manifest_rows": len(rows),
        "checked_image_files": len(rows),
        "checked_mask_files": len(rows),
        "missing_files": 0,
    }


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row["split"])
        if split not in result:
            raise ValueError(f"Unexpected split in manifest.csv: {split!r}")
        result[split].append(row)
    return result


def _split_case_ids(splits: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        values = splits.get(split, [])
        if not isinstance(values, list):
            raise ValueError(f"splits.json field {split!r} must be a list")
        result[split] = {str(value) for value in values}
    return result


def validate_split_integrity(
    rows: list[dict[str, Any]],
    splits: dict[str, Any],
    max_examples: int = 20,
) -> dict[str, Any]:
    split_case_ids = _split_case_ids(splits)

    case_to_splits: dict[str, set[str]] = {}
    for split, case_ids in split_case_ids.items():
        for case_id in case_ids:
            case_to_splits.setdefault(case_id, set()).add(split)
    split_duplicates = {
        case_id: sorted(split_names)
        for case_id, split_names in sorted(case_to_splits.items())
        if len(split_names) > 1
    }
    if split_duplicates:
        examples = dict(list(split_duplicates.items())[:max_examples])
        raise ValueError(
            "Data leakage risk: splits.json assigns the same case_id to "
            f"multiple splits: {examples}"
        )

    manifest_case_splits: dict[str, set[str]] = {}
    mismatched_rows: list[dict[str, str]] = []
    for row in rows:
        split = str(row["split"])
        case_id = str(row["case_id"])
        if split not in split_case_ids:
            raise ValueError(f"Unexpected split in manifest.csv: {split!r}")

        manifest_case_splits.setdefault(case_id, set()).add(split)
        if split_case_ids[split] and case_id not in split_case_ids[split]:
            mismatched_rows.append(
                {"split": split, "case_id": case_id, "z": str(row.get("z", ""))}
            )

    manifest_duplicates = {
        case_id: sorted(split_names)
        for case_id, split_names in sorted(manifest_case_splits.items())
        if len(split_names) > 1
    }
    if manifest_duplicates:
        examples = dict(list(manifest_duplicates.items())[:max_examples])
        raise ValueError(
            "Data leakage risk: manifest.csv contains the same case_id in "
            f"multiple splits: {examples}"
        )
    if mismatched_rows:
        raise ValueError(
            "Data leakage risk: manifest.csv split/case_id rows do not match "
            f"splits.json. First examples: {mismatched_rows[:max_examples]}"
        )

    return {
        "splits_json_case_overlap": False,
        "manifest_case_overlap": False,
        "manifest_matches_splits_json": True,
    }


def as_chw_float32(array: np.ndarray, name: str = "array") -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3:
        if array.shape[0] == 1:
            pass
        elif array.shape[-1] == 1:
            array = np.moveaxis(array, -1, 0)
        else:
            raise ValueError(
                f"{name} must have shape [H, W], [1, H, W] or [H, W, 1], "
                f"got {array.shape}"
            )
    else:
        raise ValueError(
            f"{name} must have shape [H, W], [1, H, W] or [H, W, 1], "
            f"got {array.shape}"
        )
    return array.astype(np.float32, copy=False)


def build_balanced_train_rows(
    train_rows: list[dict[str, Any]],
    negative_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positive_rows = [row for row in train_rows if to_int(row["has_tumor"]) == 1]
    negative_rows = [row for row in train_rows if to_int(row["has_tumor"]) == 0]
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")

    requested_negative = int(round(len(positive_rows) * float(negative_ratio)))
    used_negative = min(requested_negative, len(negative_rows))
    rng = np.random.default_rng(int(seed))
    selected_negative_rows: list[dict[str, Any]] = []
    if used_negative > 0:
        selected_indices = rng.choice(len(negative_rows), size=used_negative, replace=False)
        selected_negative_rows = [negative_rows[int(index)] for index in selected_indices]

    balanced_rows = [*positive_rows, *selected_negative_rows]
    if balanced_rows:
        order = rng.permutation(len(balanced_rows))
        balanced_rows = [balanced_rows[int(index)] for index in order]

    return balanced_rows, {
        "source_train_slices": len(train_rows),
        "source_positive_slices": len(positive_rows),
        "source_negative_slices": len(negative_rows),
        "train_negative_ratio": float(negative_ratio),
        "requested_negative_slices": requested_negative,
        "used_positive_slices": len(positive_rows),
        "used_negative_slices": used_negative,
        "balanced_train_slices": len(balanced_rows),
        "used_all_positive_slices": True,
        "used_all_available_negative_slices": used_negative == len(negative_rows),
    }


def crop_with_padding(
    image: np.ndarray,
    mask: np.ndarray,
    center_y: int,
    center_x: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = image.shape
    crop_size = int(size)
    half = crop_size // 2
    y0 = int(center_y) - half
    x0 = int(center_x) - half
    y1 = y0 + crop_size
    x1 = x0 + crop_size

    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(height, y1)
    src_x1 = min(width, x1)
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    image_crop = np.zeros((image.shape[0], crop_size, crop_size), dtype=np.float32)
    mask_crop = np.zeros((mask.shape[0], crop_size, crop_size), dtype=np.float32)
    if src_y1 > src_y0 and src_x1 > src_x0:
        image_crop[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[
            :,
            src_y0:src_y1,
            src_x0:src_x1,
        ]
        mask_crop[:, dst_y0:dst_y1, dst_x0:dst_x1] = mask[
            :,
            src_y0:src_y1,
            src_x0:src_x1,
        ]
    return image_crop, mask_crop


def sample_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    center_jitter: int,
    max_attempts: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = image.shape
    positive_pixels = np.argwhere(mask[0] > 0.5)
    if positive_pixels.size == 0:
        center_y = int(np.random.randint(0, height))
        center_x = int(np.random.randint(0, width))
        return crop_with_padding(image, mask, center_y, center_x, patch_size)

    y_min, x_min = positive_pixels.min(axis=0)
    y_max, x_max = positive_pixels.max(axis=0)
    bbox_center_y = int(round((int(y_min) + int(y_max)) / 2.0))
    bbox_center_x = int(round((int(x_min) + int(x_max)) / 2.0))
    jitter = max(int(center_jitter), 0)

    for _ in range(max_attempts):
        center_y = bbox_center_y
        center_x = bbox_center_x
        if jitter > 0:
            center_y += int(np.random.randint(-jitter, jitter + 1))
            center_x += int(np.random.randint(-jitter, jitter + 1))
        image_crop, mask_crop = crop_with_padding(
            image,
            mask,
            center_y=center_y,
            center_x=center_x,
            size=patch_size,
        )
        if mask_crop.sum() > 0:
            return image_crop, mask_crop

    return crop_with_padding(image, mask, bbox_center_y, bbox_center_x, patch_size)


def affine_augment(
    image: torch.Tensor,
    mask: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = image.shape
    max_degrees = float(config.get("rotation_degrees", 0.0))
    max_shift = float(config.get("shift_fraction", 0.0))
    max_scale = float(config.get("scale_limit", 0.0))
    if max_degrees <= 0 and max_shift <= 0 and max_scale <= 0:
        return image, mask

    angle = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
    scale = 1.0
    if max_scale > 0:
        scale = float(np.random.uniform(1.0 - max_scale, 1.0 + max_scale))
    shift_x = float(np.random.uniform(-max_shift, max_shift)) if max_shift > 0 else 0.0
    shift_y = float(np.random.uniform(-max_shift, max_shift)) if max_shift > 0 else 0.0

    cos_value = float(np.cos(angle) / scale)
    sin_value = float(np.sin(angle) / scale)
    theta = image.new_tensor(
        [
            [cos_value, -sin_value, -2.0 * shift_x],
            [sin_value, cos_value, -2.0 * shift_y],
        ]
    ).unsqueeze(0)
    grid = F.affine_grid(theta, size=(1, 1, height, width), align_corners=True)
    image = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    mask = F.grid_sample(
        mask.unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    return image, mask


def elastic_augment(
    image: torch.Tensor,
    mask: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = float(config.get("elastic_p", 0.0))
    if probability <= 0 or np.random.random() >= probability:
        return image, mask

    alpha = float(config.get("elastic_alpha", 0.0))
    kernel_size = int(config.get("elastic_kernel_size", 17))
    if alpha <= 0 or kernel_size <= 1:
        return image, mask
    if kernel_size % 2 == 0:
        kernel_size += 1

    _, height, width = image.shape
    displacement = torch.randn((1, 2, height, width), dtype=image.dtype)
    displacement = F.avg_pool2d(
        displacement,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    displacement = displacement / displacement.abs().amax().clamp_min(1e-6)
    displacement[:, 0] *= 2.0 * alpha / max(width - 1, 1)
    displacement[:, 1] *= 2.0 * alpha / max(height - 1, 1)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, width, dtype=image.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    grid = grid + displacement.permute(0, 2, 3, 1)
    image = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    mask = F.grid_sample(
        mask.unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    return image, mask


def apply_train_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if np.random.random() < float(config.get("horizontal_flip_p", 0.0)):
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=2)

    image_tensor = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32)))
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
    image_tensor, mask_tensor = affine_augment(image_tensor, mask_tensor, config)
    image_tensor, mask_tensor = elastic_augment(image_tensor, mask_tensor, config)
    image = image_tensor.numpy()
    mask = (mask_tensor.numpy() > 0.5).astype(np.float32, copy=False)

    intensity_scale = float(config.get("intensity_scale", 0.0))
    if intensity_scale > 0:
        scale = np.random.uniform(1.0 - intensity_scale, 1.0 + intensity_scale)
        image = image * np.float32(scale)

    intensity_shift = float(config.get("intensity_shift", 0.0))
    if intensity_shift > 0:
        shift = np.random.uniform(-intensity_shift, intensity_shift)
        image = image + np.float32(shift)

    noise_std = float(config.get("gaussian_noise_std", 0.0))
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)
        image = image + noise

    if bool(config.get("clip_image", True)):
        image = np.clip(image, 0.0, 1.0)
    return image, mask


def set_sample_seed(seed: int) -> None:
    bounded_seed = int(seed) % (2**32 - 1)
    random.seed(bounded_seed)
    np.random.seed(bounded_seed)
    torch.manual_seed(bounded_seed)


def to_int(value: Any) -> int:
    return int(float(value))


def load_pair(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    image = as_chw_float32(np.load(row["_image_path"], allow_pickle=False), "image")
    mask = as_chw_float32(np.load(row["_mask_path"], allow_pickle=False), "mask")
    mask = (mask > 0.5).astype(np.float32, copy=False)
    return image, mask


def validate_source_shape(
    image: np.ndarray,
    mask: np.ndarray,
    row: dict[str, Any],
    img_size: int,
) -> None:
    expected = (1, int(img_size), int(img_size))
    if tuple(image.shape) != expected:
        raise ValueError(
            f"Image {row['_image_path']} has shape {image.shape}, expected {expected}"
        )
    if tuple(mask.shape) != expected:
        raise ValueError(
            f"Mask {row['_mask_path']} has shape {mask.shape}, expected {expected}"
        )


def make_sample(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str,
    patch_size: int,
    patch_center_jitter: int,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "patch":
        return sample_patch(
            image=image,
            mask=mask,
            patch_size=patch_size,
            center_jitter=patch_center_jitter,
        )
    return image.copy(), mask.copy()


def save_pair(
    image: np.ndarray,
    mask: np.ndarray,
    image_path: Path,
    mask_path: Path,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, image[0].astype(np.float32, copy=False))
    np.save(mask_path, (mask[0] > 0.5).astype(np.uint8, copy=False))


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_output_row(
    source_row: dict[str, Any],
    split: str,
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    prepared_mode: str,
    sample_id: str,
    variant: str,
    is_augmented: bool,
    has_tumor: int,
) -> dict[str, Any]:
    output = dict(source_row)
    output["split"] = split
    output["has_tumor"] = int(has_tumor)
    output["image_path"] = relpath(image_path, output_dir)
    output["mask_path"] = relpath(mask_path, output_dir)
    output["prepared_mode"] = prepared_mode
    output["prepared_sample_id"] = sample_id
    output["prepared_variant"] = variant
    output["is_augmented"] = int(is_augmented)
    output["source_split"] = source_row["split"]
    output["source_case_id"] = source_row["case_id"]
    output["source_z"] = source_row["z"]
    output["source_image_path"] = source_row.get("image_path", "")
    output["source_mask_path"] = source_row.get("mask_path", "")
    return output


def train_output_paths(
    output_dir: Path,
    row: dict[str, Any],
    sample_index: int,
    variant: str,
) -> tuple[Path, Path, str]:
    case_id = str(row["case_id"])
    z_value = to_int(row["z"])
    sample_id = f"{case_id}_z{z_value:04d}_s{sample_index:06d}_{variant}"
    image_path = output_dir / "train" / case_id / "images" / f"{sample_id}.npy"
    mask_path = output_dir / "train" / case_id / "masks" / f"{sample_id}.npy"
    return image_path, mask_path, sample_id


def prepare_train_split(
    train_rows: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    augmentation_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    balanced_rows, balance_stats = build_balanced_train_rows(
        train_rows,
        negative_ratio=float(args.negative_ratio),
        seed=int(args.seed),
    )
    output_rows: list[dict[str, Any]] = []
    base_samples = 0
    augmented_samples = 0
    positive_output_samples = 0
    negative_output_samples = 0

    iterator = tqdm(balanced_rows, desc="prepare train", unit="slice")
    for sample_index, row in enumerate(iterator):
        source_is_positive = to_int(row["has_tumor"]) == 1
        image, mask = load_pair(row)
        validate_source_shape(image, mask, row, int(args.img_size))
        mask_is_positive = bool(mask.sum() > 0)
        if source_is_positive != mask_is_positive:
            raise ValueError(
                f"Manifest has_tumor mismatch for {row['case_id']} z={row['z']}: "
                f"manifest={int(source_is_positive)}, mask_positive={int(mask_is_positive)}"
            )

        variants = [("base", False)]
        augment_count = (
            int(args.augmentations_per_positive)
            if source_is_positive
            else int(args.augmentations_per_negative)
        )
        for aug_index in range(max(augment_count, 0)):
            variants.append((f"aug{aug_index + 1:02d}", True))

        for variant_index, (variant, is_augmented) in enumerate(variants):
            sample_seed = int(args.seed) + sample_index * 1009 + variant_index * 9176
            set_sample_seed(sample_seed)
            image_sample, mask_sample = make_sample(
                image=image,
                mask=mask,
                mode=str(args.mode),
                patch_size=int(args.patch_size),
                patch_center_jitter=int(args.patch_center_jitter),
            )

            if is_augmented and bool(augmentation_config.get("enabled", True)):
                image_before_aug = image_sample.copy()
                mask_before_aug = mask_sample.copy()
                for attempt in range(10):
                    set_sample_seed(sample_seed + attempt)
                    aug_image, aug_mask = apply_train_augmentations(
                        image=image_before_aug,
                        mask=mask_before_aug,
                        config=augmentation_config,
                    )
                    if not source_is_positive or aug_mask.sum() > 0:
                        image_sample, mask_sample = aug_image, aug_mask
                        break
                else:
                    image_sample, mask_sample = image_before_aug, mask_before_aug

            has_tumor = int(mask_sample.sum() > 0)
            if source_is_positive and has_tumor == 0:
                raise RuntimeError(
                    f"Prepared positive sample lost tumor pixels: "
                    f"{row['case_id']} z={row['z']} variant={variant}"
                )

            image_path, mask_path, sample_id = train_output_paths(
                output_dir=output_dir,
                row=row,
                sample_index=sample_index,
                variant=variant,
            )
            save_pair(image_sample, mask_sample, image_path, mask_path)
            output_rows.append(
                build_output_row(
                    source_row=row,
                    split="train",
                    image_path=image_path,
                    mask_path=mask_path,
                    output_dir=output_dir,
                    prepared_mode=str(args.mode),
                    sample_id=sample_id,
                    variant=variant,
                    is_augmented=is_augmented,
                    has_tumor=has_tumor,
                )
            )
            base_samples += int(not is_augmented)
            augmented_samples += int(is_augmented)
            positive_output_samples += int(has_tumor == 1)
            negative_output_samples += int(has_tumor == 0)

    stats = {
        **balance_stats,
        "output_train_samples": len(output_rows),
        "output_train_base_samples": base_samples,
        "output_train_augmented_samples": augmented_samples,
        "output_train_positive_samples": positive_output_samples,
        "output_train_negative_samples": negative_output_samples,
        "prepared_mode": args.mode,
        "patch_size": args.patch_size if args.mode == "patch" else None,
        "patch_center_jitter": args.patch_center_jitter if args.mode == "patch" else None,
        "augmentations_per_positive": args.augmentations_per_positive,
        "augmentations_per_negative": args.augmentations_per_negative,
    }
    return output_rows, stats


def copy_eval_split(
    split: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    positive = 0
    negative = 0
    for row in tqdm(rows, desc=f"copy {split}", unit="slice"):
        case_id = str(row["case_id"])
        source_image = Path(row["_image_path"])
        source_mask = Path(row["_mask_path"])
        image_path = output_dir / split / case_id / "images" / source_image.name
        mask_path = output_dir / split / case_id / "masks" / source_mask.name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_path)
        shutil.copy2(source_mask, mask_path)

        has_tumor = to_int(row["has_tumor"])
        positive += int(has_tumor == 1)
        negative += int(has_tumor == 0)
        output_rows.append(
            build_output_row(
                source_row=row,
                split=split,
                image_path=image_path,
                mask_path=mask_path,
                output_dir=output_dir,
                prepared_mode="full_slice",
                sample_id=f"{case_id}_z{to_int(row['z']):04d}",
                variant="original",
                is_augmented=False,
                has_tumor=has_tumor,
            )
        )
    stats = {
        "samples": len(output_rows),
        "positive_samples": positive,
        "negative_samples": negative,
        "prepared_mode": "full_slice",
        "augmentation": False,
    }
    return output_rows, stats


def manifest_fieldnames(source_rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for column in REQUIRED_COLUMNS_ORDER:
        ordered.append(column)
        seen.add(column)

    for row in source_rows:
        for key in row.keys():
            if key.startswith("_") or key in seen:
                continue
            ordered.append(key)
            seen.add(key)
    for column in EXTRA_COLUMNS:
        if column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def write_manifest(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    if args.negative_ratio < 0:
        raise ValueError("--negative-ratio must be non-negative")
    if args.augmentations_per_positive < 0 or args.augmentations_per_negative < 0:
        raise ValueError("augmentation counts must be non-negative")

    torch.set_num_threads(max(int(args.torch_num_threads), 1))
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    config_path = resolve_config_path(args.config)

    print("Training dataset preparation settings:")
    print(f"  data_dir: {data_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  config: {config_path}")
    print(f"  mode: {args.mode}")
    print(f"  patch_size: {args.patch_size if args.mode == 'patch' else 'disabled'}")
    print(f"  negative_ratio: {args.negative_ratio}")
    print(f"  augmentations_per_positive: {args.augmentations_per_positive}")
    print(f"  augmentations_per_negative: {args.augmentations_per_negative}")
    print(f"  seed: {args.seed}")
    print(f"  overwrite: {args.overwrite}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Set OVERWRITE_OUTPUT = True or use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentation_config = load_augmentation_config(config_path)
    splits = load_splits(data_dir)
    rows = load_manifest(data_dir)
    prepared_rows, file_report = prepare_manifest_rows(
        rows=rows,
        data_dir=data_dir,
    )
    split_integrity_report = validate_split_integrity(prepared_rows, splits)
    rows_by_split = split_rows(prepared_rows)

    train_rows, train_stats = prepare_train_split(
        train_rows=rows_by_split["train"],
        output_dir=output_dir,
        args=args,
        augmentation_config=augmentation_config,
    )
    val_rows, val_stats = copy_eval_split("val", rows_by_split["val"], output_dir)
    test_rows, test_stats = copy_eval_split("test", rows_by_split["test"], output_dir)

    output_manifest_rows = [*train_rows, *val_rows, *test_rows]
    write_manifest(
        output_dir / "manifest.csv",
        output_manifest_rows,
        manifest_fieldnames(output_manifest_rows),
    )

    split_payload = dict(splits)
    split_payload["prepared_dataset"] = {
        "source_data_dir": str(data_dir),
        "mode": args.mode,
        "img_size": args.img_size,
        "patch_size": args.patch_size if args.mode == "patch" else None,
        "patch_center_jitter": args.patch_center_jitter if args.mode == "patch" else None,
        "train_negative_ratio": args.negative_ratio,
        "augmentations_per_positive": args.augmentations_per_positive,
        "augmentations_per_negative": args.augmentations_per_negative,
        "seed": args.seed,
        "val_test_unchanged": True,
    }
    write_json(output_dir / "splits.json", split_payload)

    summary = {
        "source_data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "file_verification": file_report,
        "split_integrity": split_integrity_report,
        "augmentation": augmentation_config,
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "total_output_samples": len(output_manifest_rows),
    }
    write_json(output_dir / "preparation_summary.json", summary)
    write_json(
        output_dir / "preparation_config.json",
        {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "mode": args.mode,
            "img_size": args.img_size,
            "patch_size": args.patch_size,
            "patch_center_jitter": args.patch_center_jitter,
            "negative_ratio": args.negative_ratio,
            "augmentations_per_positive": args.augmentations_per_positive,
            "augmentations_per_negative": args.augmentations_per_negative,
            "seed": args.seed,
            "augmentation": augmentation_config,
        },
    )

    print(f"Prepared dataset: {output_dir}")
    print(f"Train samples: {train_stats['output_train_samples']}")
    print(f"Val samples: {val_stats['samples']}")
    print(f"Test samples: {test_stats['samples']}")


if __name__ == "__main__":
    main()
