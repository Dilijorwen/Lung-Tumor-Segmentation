from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # важно: до import pyplot

import numpy as np
import matplotlib.pyplot as plt


# =========================
# НАСТРОЙКИ
# =========================

PREPROCESSED_DIR = Path("preprocessed_npy")

# Можно писать:
# "lung_001"
# или просто "1"
PATIENT_ID = "lung_001"

# None -> искать пациента автоматически в train / val / test
SPLIT = None

# Скорость слайд-шоу
PAUSE_SECONDS = 0.01

# True  -> показывать только срезы с опухолью
# False -> показывать все сохранённые срезы пациента
SHOW_ONLY_TUMOR_SLICES = False

# True -> крутить по кругу
LOOP = True


def normalize_patient_id(patient_id: str) -> str:
    patient_id = str(patient_id).strip()

    if patient_id.isdigit():
        return f"lung_{int(patient_id):03d}"

    return patient_id


def find_patient_dir(preprocessed_dir: Path, patient_id: str, split: str | None):
    if split is not None:
        patient_dir = preprocessed_dir / split / patient_id

        if not patient_dir.exists():
            raise FileNotFoundError(f"Пациент не найден: {patient_dir}")

        return patient_dir, split

    for split_name in ["train", "val", "test"]:
        patient_dir = preprocessed_dir / split_name / patient_id

        if patient_dir.exists():
            return patient_dir, split_name

    raise FileNotFoundError(
        f"Пациент {patient_id} не найден в train / val / test"
    )


def load_slice_pairs(patient_dir: Path):
    images_dir = patient_dir / "images"
    masks_dir = patient_dir / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Нет папки images: {images_dir}")

    if not masks_dir.exists():
        raise FileNotFoundError(f"Нет папки masks: {masks_dir}")

    image_files = sorted(images_dir.glob("z*.npy"))

    pairs = []

    for image_path in image_files:
        mask_path = masks_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Нет маски для {image_path.name}: {mask_path}")

        pairs.append((image_path, mask_path))

    if len(pairs) == 0:
        raise RuntimeError(f"Нет .npy-срезов в {images_dir}")

    return pairs


def load_image_and_mask(image_path: Path, mask_path: Path):
    image = np.load(image_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.uint8)

    masked_tumor = np.ma.masked_where(mask == 0, mask)
    has_tumor = int(mask.sum() > 0)

    return image, mask, masked_tumor, has_tumor


patient_id = normalize_patient_id(PATIENT_ID)

patient_dir, split_name = find_patient_dir(
    PREPROCESSED_DIR,
    patient_id,
    SPLIT
)

pairs = load_slice_pairs(patient_dir)

if SHOW_ONLY_TUMOR_SLICES:
    tumor_pairs = []

    for image_path, mask_path in pairs:
        mask = np.load(mask_path).astype(np.uint8)

        if mask.sum() > 0:
            tumor_pairs.append((image_path, mask_path))

    pairs = tumor_pairs

if len(pairs) == 0:
    raise RuntimeError("Нет срезов для отображения.")

print("Пациент:", patient_id)
print("Split:", split_name)
print("Папка:", patient_dir)
print("Срезов для показа:", len(pairs))



plt.ion()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ax1, ax2 = axes

try:
    fig.canvas.manager.set_window_title(f"{patient_id} | {split_name}")
except Exception:
    pass

image, mask, masked_tumor, has_tumor = load_image_and_mask(
    pairs[0][0],
    pairs[0][1]
)

left_image = ax1.imshow(image, cmap="gray", vmin=0, vmax=1)
right_image = ax2.imshow(image, cmap="gray", vmin=0, vmax=1)
mask_overlay = ax2.imshow(masked_tumor, cmap="Reds", alpha=0.45, vmin=0, vmax=1)

ax1.axis("off")
ax2.axis("off")

while True:
    for idx, (image_path, mask_path) in enumerate(pairs):
        if not plt.fignum_exists(fig.number):
            break

        image, mask, masked_tumor, has_tumor = load_image_and_mask(
            image_path,
            mask_path
        )

        z_name = image_path.stem

        left_image.set_data(image)
        right_image.set_data(image)
        mask_overlay.set_data(masked_tumor)

        ax1.set_title(f"{patient_id} | {z_name} | без маски")
        ax2.set_title(
            f"{patient_id} | {z_name} | с маской | tumor={has_tumor} | "
            f"{idx + 1}/{len(pairs)}"
        )

        fig.tight_layout()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        plt.pause(PAUSE_SECONDS)

    if not LOOP:
        break

    if not plt.fignum_exists(fig.number):
        break

plt.ioff()
plt.show()