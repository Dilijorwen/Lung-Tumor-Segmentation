# 3-stage: local prediction slideshow

Этот этап нужен для локального просмотра предсказаний обученной модели. Он не обучает модель и не требует Docker.

## Что нужно локально

Положить модель в папку `3-stage/`:

```text
3-stage/best_model.pth
```

КТ-срезы и маски остаются в корневой папке `preprocessed_npy/` или `prepared_npy/`, например:

```text
preprocessed_npy/test/lung_096/images/z0092.npy
preprocessed_npy/test/lung_096/masks/z0092.npy
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

Notebook загружает модель из `3-stage/best_model.pth` и архитектуру из `2-stage/model.py`.

В первой кодовой ячейке выберите набор данных:

```python
DATASET_DIR = REPO_ROOT / "preprocessed_npy"  # или "prepared_npy"
SPLIT = "test"
POSITIVE_CASE_ONLY = True
RANDOM_SEED = None
INFERENCE_BATCH_SIZE = 8
```

Notebook прочитает `manifest.csv`, случайно выберет пациента из указанного split, найдёт все пары `.npy` в папках `images/` и `masks/`, выполнит inference для всех срезов, построит графики для сравнения моделей и сформирует встроенное HTML5-видео. По умолчанию выбираются только пациенты с хотя бы одним опухолевым срезом. Установите `POSITIVE_CASE_ONLY = False`, если нужны также пациенты без опухоли. Для повторяемого случайного выбора установите, например, `RANDOM_SEED = 2004`.

Каждый кадр состоит из трёх наложенных слоёв:

1. CT-срез — серый фон.
2. Истинная маска опухоли — синяя область.
3. Предсказанная моделью маска — красная область.

Для текущего среза показываются `Dice`, `Precision` и `Recall`, а в заголовке видео дополнительно выводятся агрегированные volume-метрики по выбранному пациенту. Пустые срезы без истинной и предсказанной опухоли отображаются как `n/a`, чтобы они не выглядели как нулевое качество сегментации. После inference также печатаются агрегированные метрики и диагностические счётчики по всему объёму пациента. Threshold берётся из `best_threshold` внутри checkpoint; если его нет, используется `0.5`.

Дополнительно notebook строит:

- сравнение U-Net, UNet++ и Attention U-Net по Dice coefficient, Precision и Recall на test;
- динамику train/validation Dice и loss;
- статичные примеры предсказанных масок на тестовых КТ-срезах.

Для формирования HTML5-видео нужен `ffmpeg`. На macOS при необходимости установите его:

```bash
brew install ffmpeg
```
