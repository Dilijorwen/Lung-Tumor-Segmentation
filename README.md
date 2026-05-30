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
python 1-stage/prepare_training_dataset.py
```

По умолчанию train расширяется offline сильнее: `augmentations_per_positive=4`, `augmentations_per_negative=2`. Отдельный эксперимент с patch `384x384` готовится в другую папку:

```bash
python 1-stage/prepare_training_dataset.py \
  --output-dir prepared_npy_patch384 \
  --patch-size 384 \
  --patch-center-jitter 96
```

Для обучения на `prepared_npy_patch384` во втором этапе нужно передать `--train-img-size 384`. Основные параметры меняются в верхнем блоке `SETTINGS` внутри `1-stage/prepare_training_dataset.py`. Параметры самих аугментаций лежат в `1-stage/prepare_training_config.yaml`.

## 2-stage: training

Папка `2-stage/` содержит текущий pipeline обучения 2D U-Net:

- `train_ddp.py` — DDP-обучение через `torchrun`, validation/test, чекпоинты, логи.
- `dataset.py` — чтение `manifest.csv` и загрузка `.npy`.
- `model.py` — библиотечные модели: U-Net/U-Net++ через `segmentation_models_pytorch`, Attention U-Net через `MONAI`, legacy-совместимость для старых checkpoint.
- `losses.py` — `BCEWithLogitsLoss + DiceLoss`.
- `metrics.py` — Dice, Precision, Recall.
- `utils_logging.py` — run-директории, отдельные log-файлы, JSON/YAML/CSV.
- `inference.py` — инференс одного `.npy`-среза через `best_model.pth`.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `config.yaml`, `config_unetplusplus.yaml`, `config_attention_unet.yaml` — контейнерный запуск.

В `2-stage` используется библиотечный U-Net через `segmentation_models_pytorch`: обычный `Unet` и `UnetPlusPlus` с `resnet34` encoder без pretrained weights. Attention U-Net реализован отдельно через `monai.networks.nets.AttentionUnet` с attention gates на skip connections; это не SMP `decoder_attention_type: scse`.

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
--early-stopping-patience 20
--early-stopping-min-delta 0.001
--early-stopping-min-epochs 30
# threshold_candidates по умолчанию: 0.025..0.950, шаг 0.025
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

Полное обучение U-Net сохраняется в `outputs/train/unet/run_YYYYMMDD_HHMMSS/`, U-Net++ — в `outputs/train/unet++/run_YYYYMMDD_HHMMSS/`, Attention U-Net — в `outputs/train/attention_unet/run_YYYYMMDD_HHMMSS/`. Smoke/debug запуск на 1 эпоху и 1 GPU автоматически сохраняется в `outputs/test/<модель>/run_YYYYMMDD_HHMMSS/`.

- `logs/data_loading.log`
- `logs/hyperparameters.log`
- `logs/training.log`
- `logs/validation.log`
- `logs/testing.log`
- `logs/best_model.log`
- `logs/errors.log`
- `checkpoints/best_model.pth`
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

Папка `3-stage/` содержит Jupyter notebook для локального просмотра слайд-шоу предсказаний. Положите модель в `3-stage/best_model.pth`. КТ-срезы и маски остаются в корневой папке `preprocessed_npy/` или `prepared_npy/`. Notebook загружает все срезы выбранного пациента и показывает Play-кнопку со slider. Каждый кадр состоит из трёх слоёв: CT-срез, синяя истинная маска и красная предсказанная маска. Для текущего кадра выводятся `Dice`, `Precision` и `Recall`.

Запуск:

```bash
python3 -m venv .venv-3stage
source .venv-3stage/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 3-stage/requirements.txt
jupyter notebook 3-stage/visualize_prediction.ipynb
```
