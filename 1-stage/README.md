# 1-stage: preprocessing

Эта папка содержит первый этап pipeline.

- `preprocessed_iamge.py` готовит данные из 3D NIfTI: windowing HU, resize до `512x512`, выбор positive/negative срезов, сохранение `.npy`.
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

Именно `preprocessed_npy/manifest.csv` используется вторым этапом обучения.
