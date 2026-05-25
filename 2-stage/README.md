# 2-stage: DDP training

Эта папка содержит второй этап pipeline: обучение 2D U-Net для бинарной сегментации опухолей лёгких на подготовленных `.npy`-срезах.

## Состав

- `train_ddp.py` — основной training script с DDP.
- `dataset.py` — чтение `splits.json`, `manifest.csv`, проверка путей и загрузка `.npy`.
- `model.py` — библиотечные модели: U-Net/U-Net++ через `segmentation_models_pytorch`, Attention U-Net через `MONAI` и legacy `UNet2D` для старых checkpoint.
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
- train может идти на patch `[1, 256, 256]`, val/test остаются `[1, 512, 512]`.

Attention U-Net задаётся в `config_attention_unet.yaml` как `MONAIAttentionUnet`:

- `monai.networks.nets.AttentionUnet`;
- attention gates используются на skip connections, это не SMP `decoder_attention_type: scse`;
- базовые каналы: `[64, 128, 256, 512, 1024]`;
- `ddp.find_unused_parameters: false`, все параметры участвуют в forward;
- train может идти на patch `[1, 256, 256]`, val/test остаются `[1, 512, 512]`.

Для старых checkpoint сохранена legacy-совместимость: если в checkpoint config указано `model.name: UNet2D`, `build_model` создаст прежнюю самописную U-Net.

Формирование train/val/test делается в `1-stage/prepare_training_dataset.py` до обучения:

- скрипт читает исходный `preprocessed_npy/`;
- train берётся только из `preprocessed_npy/train`;
- используются все positive-срезы train и negative-срезы в соотношении `1:1`;
- train заранее сохраняется в `prepared_npy/train` как patch `256x256`;
- positive patch центрируется около bounding box опухоли с jitter;
- negative patch берётся как случайный crop фона;
- train-аугментации применяются offline и сохраняются на диск;
- val/test копируются в `prepared_npy/val` и `prepared_npy/test` без аугментаций.

Улучшения обучения в `2-stage`:

- автоматический `pos_weight` для BCE, рассчитанный по train masks и ограниченный `pos_weight_max`;
- `BCE + Dice + FocalTversky`, где Tversky сильнее штрафует false negatives;
- подбор лучшего threshold по validation Dice, сохранение `best_threshold` в checkpoint;
- inference по умолчанию использует `best_threshold` из checkpoint.

## Запуск

Перед запуском `2-stage` должен уже существовать `prepared_npy/`, созданный первым этапом.

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --data-dir /workspace/data/prepared_npy \
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
- Attention U-Net: `outputs/train/attention_unet/run_YYYYMMDD_HHMMSS/`;
- smoke/debug: `outputs/test/unet/...`, `outputs/test/unet++/...` или `outputs/test/attention_unet/...`.

U-Net++:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --config config_unetplusplus.yaml \
  --data-dir /workspace/data/prepared_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4
```

То же самое можно задать без отдельного config:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --model-architecture unet++ \
  --data-dir /workspace/data/prepared_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4
```

Attention U-Net:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --config config_attention_unet.yaml \
  --data-dir /workspace/data/prepared_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4
```

То же самое через CLI:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --model-architecture attention_unet \
  --data-dir /workspace/data/prepared_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --lr 1e-4
```

## Улучшения качества

В training config по умолчанию:

- `train.img_size: 256`;
- `val.img_size: 512`;
- `test.img_size: 512`;
- `loss.focal_tversky_weight: 0.5`;
- `loss.tversky_alpha: 0.3`;
- `loss.tversky_beta: 0.7`;
- threshold search от `0.05` до `0.90` с шагом `0.05`.

Параметры качества можно менять из CLI:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --data-dir /workspace/data/prepared_npy \
  --output-dir /workspace/outputs \
  --epochs 100 \
  --batch-size-per-gpu 8 \
  --focal-tversky-weight 0.75 \
  --tversky-beta 0.8
```

Smoke test:

```bash
torchrun --nproc_per_node=1 train_ddp.py \
  --data-dir /workspace/data/prepared_npy \
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
