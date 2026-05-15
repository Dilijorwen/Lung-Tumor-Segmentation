# 3-stage: local prediction visualization

Этот этап нужен для локальной проверки обученной модели. Он не обучает модель и не требует Docker.

## Что нужно локально

- `best_model.pth`, например `3-stage/best_model.pth`;
- один или несколько `.npy` срезов из `preprocessed_npy/`, например:

```text
preprocessed_npy/test/lung_096/images/z0092.npy
preprocessed_npy/test/lung_096/masks/z0092.npy
```

## Установка окружения

Из корня репозитория:

```bash
python3 -m venv .venv-3stage
source .venv-3stage/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 3-stage/requirements.txt
```

## Запуск notebook

```bash
jupyter notebook 3-stage/visualize_prediction.ipynb
```

Notebook загружает U-Net из `2-stage/model.py`, поэтому запускать его нужно вместе с репозиторием. По умолчанию он сначала ищет checkpoint в `3-stage/best_model.pth`, затем в `/Users/daniil/best_run/run_20260511_061207/checkpoints/best_model.pth`.

Для новых checkpoint с `model.name: SMPUnet`, `model.name: SMPUnetPlusPlus` или `model.name: SMPAttentionUnet` нужна зависимость `segmentation-models-pytorch`. Она уже добавлена в `3-stage/requirements.txt`. Старые checkpoint с `model.name: UNet2D` продолжают открываться через legacy-модель в `2-stage/model.py`.

Срез выбирается через `manifest.csv`. В ячейке выбора можно поменять `SPLIT`, `SAMPLE`, `CASE_ID` и `Z`, либо вручную указать `image_path` и `mask_path`.

Порог берётся из `best_threshold` внутри checkpoint. Если его там нет, используется `0.5`.

## Результаты

В `3-stage/predictions/` будут сохранены:

- `*_prediction.png` — CT slice, ground truth, prediction overlay, probability map;
- `*_pred_mask.npy` — бинарная предсказанная маска;
- `*_pred_prob.npy` — probability map;
- `*_summary.json` — пути, threshold и slice-level метрики, если передана ground truth mask.
