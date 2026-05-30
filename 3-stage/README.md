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

В первой кодовой ячейке выберите пациента:

```python
DATASET_DIR = REPO_ROOT / "preprocessed_npy"  # или "prepared_npy"
SPLIT = "test"
CASE_ID = "lung_096"
INFERENCE_BATCH_SIZE = 8
```

Notebook сам найдёт все пары `.npy` в папках `images/` и `masks/`, выполнит inference для всех срезов и покажет интерактивное слайд-шоу с Play-кнопкой и slider.

Каждый кадр состоит из трёх наложенных слоёв:

1. CT-срез — серый фон.
2. Истинная маска опухоли — синяя область.
3. Предсказанная моделью маска — красная область.

Для текущего среза показываются `Dice`, `Precision` и `Recall`. После inference также печатаются агрегированные метрики по всему объёму пациента. Threshold берётся из `best_threshold` внутри checkpoint; если его нет, используется `0.5`.
