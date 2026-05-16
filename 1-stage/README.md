# 1-stage: preprocessing

Эта папка содержит первый этап pipeline.

- `preprocessed_iamge.py` готовит данные из 3D NIfTI: windowing HU, resize до `512x512`, выбор positive/negative срезов, сохранение `.npy`.
- `prepare_training_dataset.py` делает второй шаг подготовки: балансирует train positive/negative, режет train на patch `256x256`, применяет train-аугментации offline и копирует val/test без изменений.
- `check.py` визуально проверяет подготовленные `.npy`-срезы и маски.

Ожидаемый результат этапа:

```text
preprocessed_npy/
├── train/
├── val/
├── test/
├── splits.json
└── manifest.csv
```

После этого нужно создать готовый training dataset:

```bash
python 1-stage/prepare_training_dataset.py \
  --data-dir preprocessed_npy \
  --output-dir prepared_npy \
  --mode patch \
  --patch-size 256 \
  --negative-ratio 1.0 \
  --augmentations-per-positive 1 \
  --augmentations-per-negative 1 \
  --seed 42 \
  --overwrite
```

Результат:

```text
prepared_npy/
├── train/
├── val/
├── test/
├── splits.json
├── manifest.csv
├── preparation_config.json
└── preparation_summary.json
```

Именно `prepared_npy/manifest.csv` используется вторым этапом обучения.
