import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


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


def _to_int(value: Any) -> int:
    return int(float(value))


def load_splits(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    splits_path = data_dir / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path}")

    with splits_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(data_dir: str | Path) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.csv"
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


def _path_candidates(
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
        candidates.extend(
            [
                manifest_dir / raw,
                Path.cwd() / raw,
                data_dir / raw,
            ]
        )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.expanduser().resolve(strict=False))
        if normalized not in seen:
            seen.add(normalized)
            unique.append(Path(normalized))
    return unique


def resolve_manifest_path(
    raw_path: str | Path,
    data_dir: str | Path,
    manifest_dir: str | Path,
    split: str,
    case_id: str,
    kind: str,
    must_exist: bool = True,
) -> Path:
    candidates = _path_candidates(
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

    if must_exist:
        formatted = "\n  - ".join(str(path) for path in candidates[:6])
        raise FileNotFoundError(
            f"Cannot resolve {kind} path from manifest value {raw_path!r}. "
            f"Checked:\n  - {formatted}"
        )
    return candidates[0]


def prepare_manifest_rows(
    rows: list[dict[str, Any]],
    data_dir: str | Path,
    validate_files: bool = True,
    max_missing_report: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_dir = Path(data_dir).expanduser().resolve(strict=False)
    manifest_dir = data_dir
    prepared: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for row in rows:
        split = str(row["split"])
        case_id = str(row["case_id"])
        image_path = resolve_manifest_path(
            row["image_path"],
            data_dir=data_dir,
            manifest_dir=manifest_dir,
            split=split,
            case_id=case_id,
            kind="image",
            must_exist=False,
        )
        mask_path = resolve_manifest_path(
            row["mask_path"],
            data_dir=data_dir,
            manifest_dir=manifest_dir,
            split=split,
            case_id=case_id,
            kind="mask",
            must_exist=False,
        )

        if validate_files:
            if not image_path.exists():
                missing.append(
                    {
                        "kind": "image",
                        "split": split,
                        "case_id": case_id,
                        "z": str(row["z"]),
                        "path": str(image_path),
                    }
                )
            if not mask_path.exists():
                missing.append(
                    {
                        "kind": "mask",
                        "split": split,
                        "case_id": case_id,
                        "z": str(row["z"]),
                        "path": str(mask_path),
                    }
                )

        updated = dict(row)
        updated["_image_path"] = str(image_path)
        updated["_mask_path"] = str(mask_path)
        prepared.append(updated)

    report = {
        "manifest_rows": len(rows),
        "checked_image_files": len(rows),
        "checked_mask_files": len(rows),
        "missing_files": len(missing),
        "missing_examples": missing[:max_missing_report],
    }

    if validate_files and missing:
        examples = "\n".join(
            f"{item['kind']} | {item['split']} | {item['case_id']} | "
            f"z={item['z']} | {item['path']}"
            for item in missing[:max_missing_report]
        )
        raise FileNotFoundError(
            f"Missing {len(missing)} files referenced by manifest.csv. "
            f"First examples:\n{examples}"
        )

    return prepared, report


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row["split"])
        if split not in result:
            raise ValueError(f"Unexpected split in manifest.csv: {split!r}")
        result[split].append(row)
    return result


def compute_split_statistics(
    rows: list[dict[str, Any]],
    splits: dict[str, Any],
) -> dict[str, Any]:
    stats: dict[str, Any] = {"splits_json": {}, "manifest": {}}
    for split in ("train", "val", "test"):
        split_case_ids = splits.get(split, [])
        stats["splits_json"][split] = {
            "patients": len(split_case_ids),
            "case_ids": split_case_ids,
        }

        split_rows_list = [row for row in rows if row["split"] == split]
        case_ids = sorted({str(row["case_id"]) for row in split_rows_list})
        positive = sum(_to_int(row["has_tumor"]) for row in split_rows_list)
        total = len(split_rows_list)
        stats["manifest"][split] = {
            "patients": len(case_ids),
            "case_ids": case_ids,
            "slices": total,
            "positive_slices": positive,
            "negative_slices": total - positive,
        }

    settings = splits.get("settings", {})
    if settings:
        stats["preprocessing_settings"] = settings
    return stats


def _as_chw_float32(array: np.ndarray, name: str) -> np.ndarray:
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


def inspect_npy_samples(
    rows_by_split: dict[str, list[dict[str, Any]]],
    img_size: int | None,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        if not rows:
            info[split] = {"samples": 0}
            continue

        row = rows[0]
        image = np.load(row["_image_path"], allow_pickle=False)
        mask = np.load(row["_mask_path"], allow_pickle=False)
        image_chw = _as_chw_float32(image, "image")
        mask_chw = _as_chw_float32(mask, "mask")

        if img_size is not None:
            expected = (1, int(img_size), int(img_size))
            if tuple(image_chw.shape) != expected:
                raise ValueError(
                    f"Unexpected image shape for {row['_image_path']}: "
                    f"{image_chw.shape}, expected {expected}"
                )
            if tuple(mask_chw.shape) != expected:
                raise ValueError(
                    f"Unexpected mask shape for {row['_mask_path']}: "
                    f"{mask_chw.shape}, expected {expected}"
                )

        info[split] = {
            "image_path": row["_image_path"],
            "mask_path": row["_mask_path"],
            "image_shape_raw": list(image.shape),
            "mask_shape_raw": list(mask.shape),
            "image_shape_tensor": list(image_chw.shape),
            "mask_shape_tensor": list(mask_chw.shape),
            "image_dtype": str(image.dtype),
            "mask_dtype": str(mask.dtype),
        }
    return info


def compute_mask_pixel_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_pixels = 0
    total_pixels = 0

    for row in rows:
        mask = np.load(row["_mask_path"], allow_pickle=False)
        mask = _as_chw_float32(mask, "mask") > 0.5
        positive_pixels += int(mask.sum())
        total_pixels += int(mask.size)

    negative_pixels = total_pixels - positive_pixels
    positive_fraction = positive_pixels / max(total_pixels, 1)
    raw_pos_weight = negative_pixels / max(positive_pixels, 1)
    return {
        "slices": len(rows),
        "total_pixels": total_pixels,
        "positive_pixels": positive_pixels,
        "negative_pixels": negative_pixels,
        "positive_fraction": positive_fraction,
        "raw_pos_weight": raw_pos_weight,
    }


def _apply_train_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if np.random.random() < float(config.get("horizontal_flip_p", 0.0)):
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=2)

    if np.random.random() < float(config.get("vertical_flip_p", 0.0)):
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    if np.random.random() < float(config.get("rotate90_p", 0.0)):
        k = int(np.random.randint(1, 4))
        image = np.rot90(image, k=k, axes=(1, 2))
        mask = np.rot90(mask, k=k, axes=(1, 2))

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


class LungTumorNpyDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        data_dir: str | Path,
        img_size: int = 512,
        validate_files: bool = False,
        augment: bool = False,
        augmentation_config: dict[str, Any] | None = None,
    ) -> None:
        if rows and "_image_path" not in rows[0]:
            rows, _ = prepare_manifest_rows(
                rows=rows,
                data_dir=data_dir,
                validate_files=validate_files,
            )
        self.rows = rows
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.augmentation_config = augmentation_config or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = np.load(row["_image_path"], allow_pickle=False)
        mask = np.load(row["_mask_path"], allow_pickle=False)

        image = _as_chw_float32(image, "image")
        mask = _as_chw_float32(mask, "mask")
        mask = (mask > 0.5).astype(np.float32, copy=False)

        expected_shape = (1, self.img_size, self.img_size)
        if tuple(image.shape) != expected_shape:
            raise ValueError(
                f"Image {row['_image_path']} has shape {image.shape}, "
                f"expected {expected_shape}"
            )
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"Mask {row['_mask_path']} has shape {mask.shape}, "
                f"expected {expected_shape}"
            )

        if self.augment:
            image, mask = _apply_train_augmentations(
                image,
                mask,
                self.augmentation_config,
            )

        z_value = _to_int(row["z"])
        has_tumor = _to_int(row["has_tumor"])
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "case_id": row["case_id"],
            "z": z_value,
            "has_tumor": has_tumor,
            "image_path": row["_image_path"],
            "mask_path": row["_mask_path"],
        }


class BalancedPositiveNegativeSampler(Sampler[int]):
    """Samples train slices with a fixed positive/negative ratio, DDP-aware."""

    def __init__(
        self,
        dataset: LungTumorNpyDataset,
        positive_fraction: float = 0.5,
        samples_per_epoch: int | None = None,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= float(positive_fraction) <= 1.0:
            raise ValueError("positive_fraction must be in [0, 1]")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset = dataset
        self.positive_fraction = float(positive_fraction)
        self.rank = int(rank)
        self.num_replicas = int(num_replicas)
        self.seed = int(seed)
        self.epoch = 0

        labels = [_to_int(row["has_tumor"]) for row in dataset.rows]
        self.positive_indices = [idx for idx, label in enumerate(labels) if label == 1]
        self.negative_indices = [idx for idx, label in enumerate(labels) if label == 0]

        requested_total = len(dataset) if samples_per_epoch is None else int(samples_per_epoch)
        if requested_total <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.num_samples = int(np.ceil(requested_total / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.positive_indices and self.negative_indices:
            positive_count = int(round(self.total_size * self.positive_fraction))
            negative_count = self.total_size - positive_count
        elif self.positive_indices:
            positive_count = self.total_size
            negative_count = 0
        elif self.negative_indices:
            positive_count = 0
            negative_count = self.total_size
        else:
            return iter([])

        indices: list[int] = []
        if positive_count > 0:
            selected = torch.randint(
                low=0,
                high=len(self.positive_indices),
                size=(positive_count,),
                generator=generator,
            ).tolist()
            indices.extend(self.positive_indices[idx] for idx in selected)
        if negative_count > 0:
            selected = torch.randint(
                low=0,
                high=len(self.negative_indices),
                size=(negative_count,),
                generator=generator,
            ).tolist()
            indices.extend(self.negative_indices[idx] for idx in selected)

        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[idx] for idx in order]
        return iter(shuffled[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def statistics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "positive_fraction": self.positive_fraction,
            "positive_slices": len(self.positive_indices),
            "negative_slices": len(self.negative_indices),
            "samples_per_epoch_total": self.total_size,
            "samples_per_epoch_per_rank": self.num_samples,
            "num_replicas": self.num_replicas,
        }


class DistributedEvalSampler(DistributedSampler):
    """Distributed sequential sampler without duplicated validation/test samples."""

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=0,
            drop_last=False,
        )

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        return iter(indices[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        dataset_len = len(self.dataset)
        if dataset_len <= self.rank:
            return 0
        return (dataset_len - 1 - self.rank) // self.num_replicas + 1
