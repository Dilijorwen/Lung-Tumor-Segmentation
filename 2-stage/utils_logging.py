import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def create_run_directories(
    output_dir: str | Path,
    run_name: str | None = None,
    run_group: str | None = None,
    model_group: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path_parts = [output_dir]
    if run_group is not None:
        path_parts.append(Path(run_group))
    if model_group is not None:
        path_parts.append(Path(model_group))
    path_parts.append(Path(run_name))
    run_dir = Path(*path_parts)

    dirs = {
        "run_dir": run_dir,
        "logs_dir": run_dir / "logs",
        "checkpoints_dir": run_dir / "checkpoints",
        "metrics_dir": run_dir / "metrics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_file_logger(
    name: str,
    log_path: str | Path,
    enabled: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def log_dict(
    logger: logging.Logger,
    data: dict[str, Any],
    title: str | None = None,
) -> None:
    if title:
        logger.info(title)
    for key, value in data.items():
        if isinstance(value, dict):
            logger.info("%s:", key)
            for nested_key, nested_value in value.items():
                logger.info("  %s: %s", nested_key, nested_value)
        else:
            logger.info("%s: %s", key, value)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_history_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_bce_loss",
        "train_dice_loss",
        "train_tversky_loss",
        "train_focal_tversky_loss",
        "train_dice",
        "val_loss",
        "val_bce_loss",
        "val_dice_loss",
        "val_tversky_loss",
        "val_focal_tversky_loss",
        "val_dice",
        "val_best_dice",
        "val_best_threshold",
        "val_best_precision",
        "val_best_recall",
        "val_precision",
        "val_recall",
        "lr",
        "epoch_time_sec",
        "gpu_memory_gb",
        "early_stop_no_improve_epochs",
        "early_stop_triggered",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
