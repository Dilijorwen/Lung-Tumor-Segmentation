# Lung-Tumor-Segmentation

Проект разделён на этапы.

## 1-stage: preprocessing

Папка `1-stage/` содержит код подготовки данных:

- `1-stage/preprocessed_iamge.py` — разбиение пациентов, windowing HU, resize до `512x512`, сохранение `.npy`, `splits.json`, `manifest.csv`.
- `1-stage/prepare_training_dataset.py` — offline-подготовка train dataset: баланс positive/negative, patch sampling `256x256`, train-аугментации на диск, копирование val/test без изменений.
- `1-stage/check.py` — визуальная проверка подготовленных срезов и масок.

Первый шаг этапа создаёт `preprocessed_npy/` с `manifest.csv`, `splits.json` и `.npy`-срезами.

Второй шаг этапа создаёт `prepared_npy/`, который уже отправляется на сервер и используется обучением:

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

## 2-stage: training

Папка `2-stage/` содержит текущий pipeline обучения 2D U-Net:

- `train_ddp.py` — DDP-обучение через `torchrun`, validation/test, чекпоинты, логи.
- `dataset.py` — чтение `manifest.csv` и загрузка `.npy`.
- `model.py` — библиотечный 2D U-Net через `segmentation_models_pytorch` с legacy-совместимостью для старых checkpoint.
- `losses.py` — `BCEWithLogitsLoss + DiceLoss`.
- `metrics.py` — Dice/F1, Precision, Recall.
- `utils_logging.py` — run-директории, отдельные log-файлы, JSON/YAML/CSV.
- `inference.py` — инференс одного `.npy`-среза через `best_model.pth`.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `config.yaml`, `config_unetplusplus.yaml`, `config_attention_unet.yaml` — контейнерный запуск.

В `2-stage` используется библиотечный U-Net через `segmentation_models_pytorch`: обычный `Unet`, `UnetPlusPlus` и Attention U-Net на базе `Unet` с `decoder_attention_type: scse`. Для Attention U-Net включён `ddp.find_unused_parameters`, чтобы DDP корректно работал с attention-блоками SMP. Все варианты используют `resnet34` encoder без pretrained weights.

`2-stage` не делает балансировку, patch sampling или аугментации. Он читает `prepared_npy/`, созданный в `1-stage`.

Запуск обучения:

```bash
torchrun --nproc_per_node=4 2-stage/train_ddp.py \
  --data-dir /workspace/data/prepared_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4 \
  --img-size 512
```

Запуск U-Net++:

```bash
torchrun --nproc_per_node=4 2-stage/train_ddp.py \
  --config 2-stage/config_unetplusplus.yaml \
  --data-dir /workspace/data/prepared_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4 \
  --img-size 512
```

Запуск Attention U-Net:

```bash
torchrun --nproc_per_node=4 2-stage/train_ddp.py \
  --config 2-stage/config_attention_unet.yaml \
  --data-dir /workspace/data/prepared_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4 \
  --img-size 512
```

`--batch-size-per-gpu` означает batch size на один процесс/GPU. При 4 GPU и значении `8` полный batch size равен `32`.

Ключевые параметры обучения:

```bash
--focal-tversky-weight 0.5
--tversky-beta 0.7
--threshold-candidates 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90
```

## Docker

```bash
docker build -t lung-unet-ddp:latest 2-stage

docker run --gpus all --rm -it \
  -v /server/path/prepared_npy:/workspace/data/prepared_npy \
  -v /server/path/outputs:/workspace/outputs \
  lung-unet-ddp:latest \
  torchrun --nproc_per_node=4 train_ddp.py \
    --data-dir /workspace/data/prepared_npy \
    --output-dir /workspace/outputs \
    --epochs 100 \
    --batch-size-per-gpu 8 \
    --lr 1e-4
```

Compose-вариант:

```bash
docker compose -f 2-stage/docker-compose.yml up --build
```

Перед запуском замените `/server/path/prepared_npy` и `/server/path/outputs` в `2-stage/docker-compose.yml` на реальные пути сервера.

## Артефакты обучения

Каждый запуск пишет отдельную директорию `outputs/<тип>/<модель>/run_YYYYMMDD_HHMMSS/`:

Полное обучение U-Net сохраняется в `outputs/train/unet/run_YYYYMMDD_HHMMSS/`, U-Net++ — в `outputs/train/unet++/run_YYYYMMDD_HHMMSS/`, Attention U-Net — в `outputs/train/unet_attention/run_YYYYMMDD_HHMMSS/`. Smoke/debug запуск на 1 эпоху и 1 GPU автоматически сохраняется в `outputs/test/<модель>/run_YYYYMMDD_HHMMSS/`.

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
  --checkpoint outputs/train/unet/run_YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
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
