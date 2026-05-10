# 2-stage: DDP training

Эта папка содержит второй этап pipeline: обучение 2D U-Net для бинарной сегментации опухолей лёгких на подготовленных `.npy`-срезах.

## Состав

- `train_ddp.py` — основной training script с DDP.
- `dataset.py` — чтение `splits.json`, `manifest.csv`, проверка путей и загрузка `.npy`.
- `model.py` — 2D U-Net.
- `losses.py` — `BCEWithLogitsLoss + DiceLoss`.
- `metrics.py` — Dice/F1, IoU, Precision, Recall.
- `utils_logging.py` — логи, CSV/JSON/YAML, run-директории.
- `inference.py` — инференс одного `.npy`-среза.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `config.yaml` — Docker-запуск.

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

Полные запуски пишутся в `outputs/train/run_YYYYMMDD_HHMMSS/`.
Smoke/debug запуск на 1 эпоху и 1 GPU автоматически пишется в `outputs/test/run_YYYYMMDD_HHMMSS/`.

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
