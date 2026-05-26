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
python 1-stage/prepare_training_dataset.py
```

По умолчанию создаётся `prepared_npy/` с `patch_size=256`, `augmentations_per_positive=4` и `augmentations_per_negative=2`.

Отдельный эксперимент с `patch_size=384` нужно писать в другую папку:

```bash
python 1-stage/prepare_training_dataset.py \
  --output-dir prepared_npy_patch384 \
  --patch-size 384 \
  --patch-center-jitter 96
```

Основные параметры меняются в верхнем блоке `SETTINGS` внутри `prepare_training_dataset.py`: пути, режим `patch/full_slice`, `patch_size`, баланс negative, число offline-аугментаций, seed и overwrite. Параметры самих аугментаций лежат в `prepare_training_config.yaml`.

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
