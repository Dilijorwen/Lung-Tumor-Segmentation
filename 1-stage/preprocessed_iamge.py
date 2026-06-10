from pathlib import Path
import csv
import json
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



DATA_DIR = Path("..")
IMAGES_DIR = DATA_DIR / "imagesTr"
MASKS_DIR = DATA_DIR / "labelsTr"

OUT_DIR = DATA_DIR / "preprocessed_npy"

IMG_SIZE = 512
SEED = 2004

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

WINDOW_MIN = -1000
WINDOW_MAX = 400

TRAIN_NEGATIVE_RATIO = 1.0
SAVE_ALL_VAL_TEST_SLICES = True



random.seed(SEED)
np.random.seed(SEED)


def get_nifti_files(folder: Path):
    return sorted([
        p for p in folder.glob("*.nii.gz")
        if not p.name.startswith("._")
    ])


def get_case_id(path: Path):
    return path.name.replace(".nii.gz", "")


def window_normalize_ct(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, WINDOW_MIN, WINDOW_MAX)
    image = (image - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    return image.astype(np.float32)


def resize_image_slice(slice_2d: np.ndarray, img_size: int) -> np.ndarray:
    x = torch.from_numpy(slice_2d).float()
    x = x.unsqueeze(0).unsqueeze(0)

    x = F.interpolate(
        x,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False
    )

    return x.squeeze(0).squeeze(0).numpy().astype(np.float32)


def resize_mask_slice(slice_2d: np.ndarray, img_size: int) -> np.ndarray:
    x = torch.from_numpy(slice_2d).float()
    x = x.unsqueeze(0).unsqueeze(0)

    x = F.interpolate(
        x,
        size=(img_size, img_size),
        mode="nearest"
    )

    x = x.squeeze(0).squeeze(0).numpy()
    x = (x > 0).astype(np.uint8)

    return x


def get_selected_slices(mask: np.ndarray, split_name: str):
    positive_slices = []
    negative_slices = []

    for z in range(mask.shape[2]):
        if mask[:, :, z].sum() > 0:
            positive_slices.append(z)
        else:
            negative_slices.append(z)

    if split_name == "train":
        n_negative = int(len(positive_slices) * TRAIN_NEGATIVE_RATIO)

        if n_negative > 0 and len(negative_slices) > 0:
            selected_negative = random.sample(
                negative_slices,
                min(n_negative, len(negative_slices))
            )
        else:
            selected_negative = []

        selected_slices = positive_slices + selected_negative
        selected_slices = sorted(selected_slices)

    else:
        if SAVE_ALL_VAL_TEST_SLICES:
            selected_slices = list(range(mask.shape[2]))
        else:
            selected_slices = positive_slices

    return selected_slices, positive_slices, negative_slices


def save_patient_slices(
    image_path: Path,
    mask_path: Path,
    split_name: str,
    out_dir: Path,
    img_size: int
):
    case_id = get_case_id(image_path)

    print(f"[{split_name}] {case_id}")

    image_obj = nib.as_closest_canonical(nib.load(image_path))
    mask_obj = nib.as_closest_canonical(nib.load(mask_path))

    image = image_obj.get_fdata().astype(np.float32)
    mask = mask_obj.get_fdata().astype(np.uint8)

    mask = (mask > 0).astype(np.uint8)

    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: {case_id}: image={image.shape}, mask={mask.shape}"
        )

    image = window_normalize_ct(image)

    selected_slices, positive_slices, negative_slices = get_selected_slices(
        mask,
        split_name
    )

    patient_dir = out_dir / split_name / case_id
    images_out_dir = patient_dir / "images"
    masks_out_dir = patient_dir / "masks"

    images_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for z in selected_slices:
        ct_slice = image[:, :, z]
        mask_slice = mask[:, :, z]

        ct_slice = resize_image_slice(ct_slice, img_size)
        mask_slice = resize_mask_slice(mask_slice, img_size)

        image_out_path = images_out_dir / f"z{z:04d}.npy"
        mask_out_path = masks_out_dir / f"z{z:04d}.npy"

        np.save(image_out_path, ct_slice.astype(np.float32))
        np.save(mask_out_path, mask_slice.astype(np.uint8))

        has_tumor = int(mask_slice.sum() > 0)

        rows.append({
            "split": split_name,
            "case_id": case_id,
            "z": z,
            "has_tumor": has_tumor,
            "image_path": str(image_out_path),
            "mask_path": str(mask_out_path),
            "original_image_path": str(image_path),
            "original_mask_path": str(mask_path),
        })

    patient_info = {
        "case_id": case_id,
        "split": split_name,
        "original_shape": list(image.shape),
        "saved_slices": len(selected_slices),
        "positive_slices_total": len(positive_slices),
        "negative_slices_total": len(negative_slices),
        "selected_slices": selected_slices,
        "img_size": img_size,
        "window_min": WINDOW_MIN,
        "window_max": WINDOW_MAX,
    }

    with open(patient_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(patient_info, f, ensure_ascii=False, indent=4)

    print(
        f"[{split_name}] {case_id}: "
        f"saved={len(selected_slices)}, "
        f"positive_total={len(positive_slices)}, "
        f"negative_total={len(negative_slices)}"
    )

    return rows


def validate_patient_splits(splits_payload):
    case_to_splits = {}
    for split_name in ("train", "val", "test"):
        for case_id in splits_payload.get(split_name, []):
            case_to_splits.setdefault(case_id, set()).add(split_name)

    duplicates = {
        case_id: sorted(split_names)
        for case_id, split_names in sorted(case_to_splits.items())
        if len(split_names) > 1
    }
    if duplicates:
        raise ValueError(
            "Data leakage risk: the same patient appears in multiple splits: "
            f"{duplicates}"
        )



image_files = get_nifti_files(IMAGES_DIR)
mask_files = get_nifti_files(MASKS_DIR)

image_dict = {get_case_id(p): p for p in image_files}
mask_dict = {get_case_id(p): p for p in mask_files}

case_ids = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

volume_pairs = [(image_dict[cid], mask_dict[cid]) for cid in case_ids]

print("Всего размеченных 3D-объёмов:", len(volume_pairs))

if len(volume_pairs) == 0:
    raise RuntimeError("Не найдены пары imagesTr + labelsTr")



train_pairs, temp_pairs = train_test_split(
    volume_pairs,
    test_size=(1.0 - TRAIN_SIZE),
    random_state=SEED
)

relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)

val_pairs, test_pairs = train_test_split(
    temp_pairs,
    test_size=relative_test_size,
    random_state=SEED
)

print("Train patients:", len(train_pairs))
print("Val patients:", len(val_pairs))
print("Test patients:", len(test_pairs))



OUT_DIR.mkdir(parents=True, exist_ok=True)

splits = {
    "train": [get_case_id(p[0]) for p in train_pairs],
    "val": [get_case_id(p[0]) for p in val_pairs],
    "test": [get_case_id(p[0]) for p in test_pairs],
    "settings": {
        "img_size": IMG_SIZE,
        "seed": SEED,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "window_min": WINDOW_MIN,
        "window_max": WINDOW_MAX,
        "train_negative_ratio": TRAIN_NEGATIVE_RATIO,
        "save_all_val_test_slices": SAVE_ALL_VAL_TEST_SLICES,
    }
}
validate_patient_splits(splits)

with open(OUT_DIR / "splits.json", "w", encoding="utf-8") as f:
    json.dump(splits, f, ensure_ascii=False, indent=4)



all_rows = []

for split_name, pairs in [
    ("train", train_pairs),
    ("val", val_pairs),
    ("test", test_pairs),
]:
    for image_path, mask_path in pairs:
        rows = save_patient_slices(
            image_path=image_path,
            mask_path=mask_path,
            split_name=split_name,
            out_dir=OUT_DIR,
            img_size=IMG_SIZE
        )
        all_rows.extend(rows)



manifest_path = OUT_DIR / "manifest.csv"

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "split",
        "case_id",
        "z",
        "has_tumor",
        "image_path",
        "mask_path",
        "original_image_path",
        "original_mask_path",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)


def count_rows(split_name):
    return [r for r in all_rows if r["split"] == split_name]


train_rows = count_rows("train")
val_rows = count_rows("val")
test_rows = count_rows("test")

print("\nГотово.")
print("Папка:", OUT_DIR)
print("Manifest:", manifest_path)

print("Train slices:", len(train_rows))
print("Val slices:", len(val_rows))
print("Test slices:", len(test_rows))

print("Train positive slices:", sum(r["has_tumor"] == 1 for r in train_rows))
print("Val positive slices:", sum(r["has_tumor"] == 1 for r in val_rows))
print("Test positive slices:", sum(r["has_tumor"] == 1 for r in test_rows))
