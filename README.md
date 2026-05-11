# Lung-Tumor-Segmentation

Проект разделён на этапы.

## 1-stage: preprocessing

Папка `1-stage/` содержит код подготовки данных:

- `1-stage/preprocessed_iamge.py` — разбиение пациентов, windowing HU, resize до `512x512`, сохранение `.npy`, `splits.json`, `manifest.csv`.
- `1-stage/check.py` — визуальная проверка подготовленных срезов и масок.

Результат этапа: `preprocessed_npy/` с `manifest.csv`, `splits.json` и `.npy`-срезами.

## 2-stage: training

Папка `2-stage/` содержит текущий pipeline обучения 2D U-Net:

- `train_ddp.py` — DDP-обучение через `torchrun`, validation/test, чекпоинты, логи.
- `dataset.py` — чтение `manifest.csv` и загрузка `.npy`.
- `model.py` — 2D U-Net.
- `losses.py` — `BCEWithLogitsLoss + DiceLoss`.
- `metrics.py` — Dice/F1, Precision, Recall.
- `utils_logging.py` — run-директории, отдельные log-файлы, JSON/YAML/CSV.
- `inference.py` — инференс одного `.npy`-среза через `best_model.pth`.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `config.yaml` — контейнерный запуск.

В `2-stage` включены улучшения обучения без изменения U-Net: train-аугментации, автоматический `pos_weight` для BCE по пиксельному дисбалансу train masks и подбор лучшего порога по validation Dice.

Запуск обучения:

```bash
torchrun --nproc_per_node=4 2-stage/train_ddp.py \
  --data-dir /workspace/data/preprocessed_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4 \
  --img-size 512
```

`--batch-size-per-gpu` означает batch size на один процесс/GPU. При 4 GPU и значении `8` полный batch size равен `32`.

## Docker

```bash
docker build -t lung-unet-ddp:latest 2-stage

docker run --gpus all --rm -it \
  -v /server/path/preprocessed_npy:/workspace/data/preprocessed_npy \
  -v /server/path/outputs:/workspace/outputs \
  lung-unet-ddp:latest \
  torchrun --nproc_per_node=4 train_ddp.py \
    --data-dir /workspace/data/preprocessed_npy \
    --output-dir /workspace/outputs \
    --epochs 100 \
    --batch-size-per-gpu 8 \
    --lr 1e-4
```

Compose-вариант:

```bash
docker compose -f 2-stage/docker-compose.yml up --build
```

Перед запуском замените `/server/path/preprocessed_npy` и `/server/path/outputs` в `2-stage/docker-compose.yml` на реальные пути сервера.

## Артефакты обучения

Каждый запуск пишет отдельную директорию `outputs/<тип>/run_YYYYMMDD_HHMMSS/`:

Полное обучение сохраняется в `outputs/train/run_YYYYMMDD_HHMMSS/`, smoke/debug запуск на 1 эпоху и 1 GPU автоматически сохраняется в `outputs/test/run_YYYYMMDD_HHMMSS/`.

- `logs/data_loading.log`
- `logs/hyperparameters.log`
- `logs/training.log`
- `logs/validation.log`
- `logs/testing.log`
- `logs/best_model.log`
- `logs/errors.log`
- `checkpoints/best_model.pth`
- `checkpoints/last_model.pth`
- `metrics/history.csv`
- `metrics/history.json`
- `metrics/best_model_metrics.json`
- `metrics/test_metrics.json`
- `metrics/split_statistics.json`
- `config.yaml`

## Инференс

```bash
python 2-stage/inference.py \
  --checkpoint outputs/train/run_YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
  --input preprocessed_npy/test/lung_001/images/z0000.npy \
  --output-mask pred_mask.npy \
  --output-prob pred_prob.npy
```

## 3-stage: local visualization

Папка `3-stage/` содержит Jupyter notebook для локального просмотра предсказаний `best_model.pth` на `.npy`-срезах. Notebook показывает CT, ground truth, predicted mask и probability map, а также сохраняет PNG, `.npy`-маску, `.npy`-probability map и JSON-summary.

Запуск:

```bash
python3 -m venv .venv-3stage
source .venv-3stage/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 3-stage/requirements.txt
jupyter notebook 3-stage/visualize_prediction.ipynb
```
