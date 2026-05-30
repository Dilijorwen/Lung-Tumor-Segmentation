# 3-stage: local prediction visualization

Этот этап нужен для локальной проверки обученной модели. Он не обучает модель и не требует Docker.

## Что нужно локально

- положить в папку `3-stage/` ровно три файла:

```text
3-stage/best_model.pth
3-stage/image.npy
3-stage/mask.npy
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

Notebook загружает U-Net из `2-stage/model.py`, поэтому запускать его нужно вместе с репозиторием. Модель всегда берётся только из `3-stage/best_model.pth`.

Для checkpoint с `model.name: SMPUnet` или `model.name: SMPUnetPlusPlus` нужна зависимость `segmentation-models-pytorch`. Для нового `model.name: MONAIAttentionUnet` нужна зависимость `monai`. Обе зависимости уже добавлены в `3-stage/requirements.txt`. Старые checkpoint с `model.name: UNet2D` продолжают открываться через legacy-модель в `2-stage/model.py`.

Основная визуализация — единое наложение на CT-срез:

- синяя область — настоящая маска опухоли (`ground truth`);
- красная область — предсказанная моделью маска (`prediction`).

В заголовке изображения выводятся `Dice`, `Precision`, `Recall` и использованный threshold. Порог берётся из `best_threshold` внутри checkpoint. Если его там нет, используется `0.5`.
