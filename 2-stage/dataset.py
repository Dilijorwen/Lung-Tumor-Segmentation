import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
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
    source_mismatches: list[dict[str, str]] = []
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

        source_split = str(row.get("source_split", "")).strip()
        source_case_id = str(row.get("source_case_id", "")).strip()
        if source_split and source_split != split:
            source_mismatches.append(
                {
                    "split": split,
                    "case_id": case_id,
                    "source_split": source_split,
                    "source_case_id": source_case_id,
                    "z": str(row.get("z", "")),
                }
            )
        if source_case_id and source_case_id != case_id:
            source_mismatches.append(
                {
                    "split": split,
                    "case_id": case_id,
                    "source_split": source_split,
                    "source_case_id": source_case_id,
                    "z": str(row.get("z", "")),
                }
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
    if source_mismatches:
        raise ValueError(
            "Data leakage risk: prepared manifest source split/case does not "
            f"match output split/case. First examples: {source_mismatches[:max_examples]}"
        )

    return {
        "splits_json_case_overlap": False,
        "manifest_case_overlap": False,
        "manifest_matches_splits_json": True,
        "prepared_source_matches_output_split": True,
    }


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


def as_chw_float32(array: np.ndarray, name: str = "array") -> np.ndarray:
    return _as_chw_float32(array=array, name=name)


def inspect_npy_samples(
    rows_by_split: dict[str, list[dict[str, Any]]],
    img_size: int | None = None,
    img_sizes_by_split: dict[str, int | None] | None = None,
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

        expected_size = (
            img_sizes_by_split.get(split, img_size)
            if img_sizes_by_split is not None
            else img_size
        )
        if expected_size is not None:
            expected = (1, int(expected_size), int(expected_size))
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


class LungTumorNpyDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        data_dir: str | Path,
        img_size: int = 512,
        validate_files: bool = False,
    ) -> None:
        if rows and "_image_path" not in rows[0]:
            rows, _ = prepare_manifest_rows(
                rows=rows,
                data_dir=data_dir,
                validate_files=validate_files,
            )
        self.rows = rows
        self.img_size = int(img_size)

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
