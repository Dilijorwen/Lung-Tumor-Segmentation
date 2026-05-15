# 2-stage: DDP training

Эта папка содержит второй этап pipeline: обучение 2D U-Net для бинарной сегментации опухолей лёгких на подготовленных `.npy`-срезах.

## Состав

- `train_ddp.py` — основной training script с DDP.
- `dataset.py` — чтение `splits.json`, `manifest.csv`, проверка путей и загрузка `.npy`.
- `model.py` — библиотечный 2D U-Net через `segmentation_models_pytorch` и legacy `UNet2D` для старых checkpoint.
- `losses.py` — `BCEWithLogitsLoss + DiceLoss`.
- `metrics.py` — Dice/F1, Precision, Recall.
- `utils_logging.py` — логи, CSV/JSON/YAML, run-директории.
- `inference.py` — инференс одного `.npy`-среза.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `config.yaml`, `config_unetplusplus.yaml`, `config_attention_unet.yaml` — Docker-запуск.

Обычная U-Net задаётся в `config.yaml` как `SMPUnet`:

- `segmentation_models_pytorch.Unet`;
- encoder: `resnet34`;
- `encoder_weights: null`, чтобы Docker/server запуск не скачивал pretrained weights;
- `in_channels: 1`, `out_channels: 1`;
- output остаётся logits, sigmoid применяется только в метриках/inference.

U-Net++ задаётся в `config_unetplusplus.yaml` как `SMPUnetPlusPlus`:

- `segmentation_models_pytorch.UnetPlusPlus`;
- тот же `resnet34` encoder;
- те же вход/выход: `[1, 512, 512] -> [1, 512, 512]`.

Attention U-Net задаётся в `config_attention_unet.yaml` как `SMPAttentionUnet`:

- `segmentation_models_pytorch.Unet`;
- `decoder_attention_type: scse`;
- тот же `resnet34` encoder;
- те же вход/выход: `[1, 512, 512] -> [1, 512, 512]`.

Для старых checkpoint сохранена legacy-совместимость: если в checkpoint config указано `model.name: UNet2D`, `build_model` создаст прежнюю самописную U-Net.

Улучшения обучения вынесены в training pipeline:

- balanced positive/negative sampling train-срезов;
- train-аугментации: horizontal flip, brightness/contrast shift, gaussian noise;
- автоматический `pos_weight` для BCE, рассчитанный по train masks и ограниченный `pos_weight_max`;
- `BCE + Dice + FocalTversky`, где Tversky сильнее штрафует false negatives;
- подбор лучшего threshold по validation Dice, сохранение `best_threshold` в checkpoint;
- inference по умолчанию использует `best_threshold` из checkpoint.

## Запуск

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --data-dir /workspace/data/preprocessed_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4 \
  --img-size 512
```

Полные запуски пишутся в `outputs/train/<модель>/run_YYYYMMDD_HHMMSS/`. Smoke/debug запуск на 1 эпоху и 1 GPU автоматически пишется в `outputs/test/<модель>/run_YYYYMMDD_HHMMSS/`.

Архитектура добавляется вторым уровнем директории:

- U-Net: `outputs/train/unet/run_YYYYMMDD_HHMMSS/`;
- U-Net++: `outputs/train/unet++/run_YYYYMMDD_HHMMSS/`;
- Attention U-Net: `outputs/train/unet_attention/run_YYYYMMDD_HHMMSS/`;
- smoke/debug: `outputs/test/unet/...`, `outputs/test/unet++/...` или `outputs/test/unet_attention/...`.

U-Net++:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --config config_unetplusplus.yaml \
  --data-dir /workspace/data/preprocessed_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4
```

То же самое можно задать без отдельного config:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --model-architecture unet++ \
  --data-dir /workspace/data/preprocessed_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4
```

Attention U-Net:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --config config_attention_unet.yaml \
  --data-dir /workspace/data/preprocessed_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4
```

То же самое через CLI:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --model-architecture unet_attention \
  --data-dir /workspace/data/preprocessed_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4
```

## Улучшения качества

По умолчанию включено:

- `sampling.balanced_train: true`;
- `sampling.positive_fraction: 0.5`;
- `loss.focal_tversky_weight: 0.5`;
- `loss.tversky_alpha: 0.3`;
- `loss.tversky_beta: 0.7`;
- threshold search от `0.05` до `0.90` с шагом `0.05`.

Параметры можно менять из CLI:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --data-dir /workspace/data/preprocessed_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --positive-fraction 0.6 \
  --focal-tversky-weight 0.75 \
  --tversky-beta 0.8
```

Отключить balanced sampling:

```bash
--no-balanced-sampling
```

Smoke test:

```bash
torchrun --nproc_per_node=1 train_ddp.py \
  --data-dir /workspace/data/preprocessed_npy \
  --output-dir /workspace/outputs \
  --epochs 1 \
  --batch-size-per-gpu 2 \
  --num-workers 2
```

Лучший checkpoint сохраняется как `checkpoints/best_model.pth`, а метрики эпохи, на которой он был выбран, дополнительно пишутся в `logs/best_model.log` и `metrics/best_model_metrics.json`.

## Docker

Из корня репозитория:

```bash
docker build -t lung-unet-ddp:latest 2-stage
```

```bash
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
