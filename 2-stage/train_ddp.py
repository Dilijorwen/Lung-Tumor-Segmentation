import argparse
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import (
    DistributedEvalSampler,
    LungTumorNpyDataset,
    compute_mask_pixel_statistics,
    compute_split_statistics,
    inspect_npy_samples,
    load_manifest,
    load_splits,
    prepare_manifest_rows,
    split_rows,
)
from losses import BCEDiceLoss
from metrics import confusion_from_probabilities, metrics_from_confusion
from model import build_model, model_output_name
from utils_logging import (
    create_run_directories,
    log_dict,
    save_json,
    save_yaml,
    setup_file_logger,
    write_history_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def default_config() -> dict[str, Any]:
    return {
        "data": {
            "data_dir": "prepared_npy",
            "img_size": 512,
            "validate_files": True,
        },
        "output": {
            "output_dir": "outputs",
            "run_name": None,
            "run_type": "auto",
        },
        "model": {
            "name": "SMPUnet",
            "architecture": "unet",
            "library": "segmentation_models_pytorch",
            "encoder_name": "resnet34",
            "encoder_weights": None,
            "encoder_depth": 5,
            "in_channels": 1,
            "out_channels": 1,
            "decoder_channels": [256, 128, 64, 32, 16],
            "decoder_use_batchnorm": True,
            "decoder_attention_type": None,
            "sync_batchnorm": False,
        },
        "loss": {
            "name": "BCEWithLogitsLoss + DiceLoss + FocalTverskyLoss",
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "dice_smooth": 1.0,
            "tversky_weight": 0.0,
            "focal_tversky_weight": 0.5,
            "tversky_alpha": 0.3,
            "tversky_beta": 0.7,
            "focal_tversky_gamma": 0.75,
            "pos_weight": None,
            "auto_pos_weight": True,
            "pos_weight_max": 20.0,
            "pos_weight_min": 1.0,
        },
        "train": {
            "img_size": 256,
        },
        "val": {
            "img_size": 512,
        },
        "test": {
            "img_size": 512,
        },
        "training": {
            "epochs": 100,
            "batch_size_per_gpu": 8,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "min_learning_rate": 1e-6,
            "num_workers": 8,
            "seed": 2004,
            "mixed_precision": True,
            "grad_clip_norm": 0.0,
            "resume": None,
        },
        "early_stopping": {
            "enabled": True,
            "monitor": "val_best_dice",
            "mode": "max",
            "patience": 20,
            "min_delta": 0.001,
            "min_epochs": 30,
        },
        "metrics": {
            "threshold": 0.5,
            "threshold_candidates": [
                round(0.025 * index, 3) for index in range(1, 39)
            ],
        },
        "ddp": {
            "backend": "nccl",
            "find_unused_parameters": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 2D U-Net for binary lung tumor segmentation with DDP."
    )
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--run-type",
        choices=["auto", "train", "test"],
        default=None,
        help="Output subfolder: train for full runs, test for smoke/debug runs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Route outputs to test/ and default to one epoch when --epochs is omitted.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size-per-gpu",
        "--batch-size",
        dest="batch_size_per_gpu",
        type=int,
        default=None,
        help="Batch size on each GPU/process.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--train-img-size", type=int, default=None)
    parser.add_argument("--val-img-size", type=int, default=None)
    parser.add_argument("--test-img-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--model-architecture",
        choices=[
            "unet",
            "unet++",
            "unetplusplus",
            "unet_attention",
            "unet-attention",
            "attention_unet",
            "attention-unet",
        ],
        default=None,
        help="Model architecture: unet, unet++, or attention_unet.",
    )
    parser.add_argument("--encoder-name", type=str, default=None)
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default=None,
        help="SMP encoder weights, for example imagenet. Use none/null for random init.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Legacy UNet2D only. Ignored by SMPUnet.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-candidates",
        type=str,
        default=None,
        help="Comma-separated validation thresholds, for example 0.1,0.2,0.3,0.4,0.5.",
    )
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--focal-tversky-weight", type=float, default=None)
    parser.add_argument("--tversky-beta", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=None)

    early_stop_group = parser.add_mutually_exclusive_group()
    early_stop_group.add_argument(
        "--early-stopping",
        dest="early_stopping_enabled",
        action="store_true",
    )
    early_stop_group.add_argument(
        "--no-early-stopping",
        dest="early_stopping_enabled",
        action="store_false",
    )
    parser.set_defaults(early_stopping_enabled=None)

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="mixed_precision", action="store_true")
    amp_group.add_argument("--no-amp", dest="mixed_precision", action="store_false")
    parser.set_defaults(mixed_precision=None)

    return parser.parse_args()


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return loaded


def build_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = default_config()
    config_path = Path(args.config).expanduser().resolve(strict=False)
    config_dir = config_path.parent
    deep_update(cfg, load_config(config_path))

    for section, key in (("data", "data_dir"), ("output", "output_dir")):
        value = cfg.get(section, {}).get(key)
        if value is not None and not Path(str(value)).expanduser().is_absolute():
            cfg[section][key] = str((config_dir / str(value)).resolve(strict=False))

    overrides = {
        ("data", "data_dir"): args.data_dir,
        ("data", "img_size"): args.img_size,
        ("train", "img_size"): args.train_img_size,
        ("val", "img_size"): args.val_img_size,
        ("test", "img_size"): args.test_img_size,
        ("output", "output_dir"): args.output_dir,
        ("output", "run_name"): args.run_name,
        ("output", "run_type"): args.run_type,
        ("model", "architecture"): args.model_architecture,
        ("model", "encoder_name"): args.encoder_name,
        ("model", "encoder_weights"): args.encoder_weights,
        ("model", "base_channels"): args.base_channels,
        ("training", "epochs"): args.epochs,
        ("training", "batch_size_per_gpu"): args.batch_size_per_gpu,
        ("training", "learning_rate"): args.lr,
        ("training", "weight_decay"): args.weight_decay,
        ("training", "num_workers"): args.num_workers,
        ("training", "seed"): args.seed,
        ("training", "mixed_precision"): args.mixed_precision,
        ("training", "scheduler"): args.scheduler,
        ("training", "grad_clip_norm"): args.grad_clip_norm,
        ("training", "resume"): args.resume,
        ("metrics", "threshold"): args.threshold,
        ("loss", "focal_tversky_weight"): args.focal_tversky_weight,
        ("loss", "tversky_beta"): args.tversky_beta,
        ("early_stopping", "enabled"): args.early_stopping_enabled,
        ("early_stopping", "patience"): args.early_stopping_patience,
        ("early_stopping", "min_delta"): args.early_stopping_min_delta,
        ("early_stopping", "min_epochs"): args.early_stopping_min_epochs,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            cfg[section][key] = value

    if args.smoke_test:
        cfg["output"]["run_type"] = "test"
        if args.epochs is None:
            cfg["training"]["epochs"] = 1

    if args.threshold_candidates is not None:
        cfg["metrics"]["threshold_candidates"] = [
            float(value.strip())
            for value in args.threshold_candidates.split(",")
            if value.strip()
        ]

    if args.model_architecture is not None:
        architecture = model_output_name({"architecture": args.model_architecture})
        if architecture in {"unet", "unet++"}:
            cfg["model"]["decoder_attention_type"] = None
    else:
        architecture = model_output_name(cfg["model"])
    cfg["model"]["architecture"] = architecture
    if architecture == "unet++":
        cfg["model"]["name"] = "SMPUnetPlusPlus"
        cfg["model"]["library"] = "segmentation_models_pytorch"
    elif architecture == "attention_unet":
        cfg["model"]["name"] = "MONAIAttentionUnet"
        cfg["model"]["library"] = "monai"
        cfg["model"].setdefault("spatial_dims", 2)
        cfg["model"].setdefault("channels", [64, 128, 256, 512, 1024])
        cfg["model"].setdefault("strides", [2, 2, 2, 2])
        cfg["model"].setdefault("kernel_size", 3)
        cfg["model"].setdefault("up_kernel_size", 3)
        cfg["model"].setdefault("dropout", 0.0)
        cfg["model"].pop("decoder_attention_type", None)
    elif architecture == "unet":
        cfg["model"]["name"] = "SMPUnet"
        cfg["model"]["library"] = "segmentation_models_pytorch"
    elif architecture == "unet2d":
        cfg["model"]["name"] = "UNet2D"
        cfg["model"]["library"] = "legacy"

    if str(cfg["training"]["scheduler"]).lower() == "none":
        cfg["training"]["scheduler"] = None
    return cfg


def get_threshold_candidates(cfg: dict[str, Any]) -> list[float]:
    values = cfg["metrics"].get("threshold_candidates", [])
    if values is None:
        return []
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    thresholds = sorted({float(value) for value in values})
    primary_threshold = float(cfg["metrics"].get("threshold", 0.5))
    if thresholds and primary_threshold not in thresholds:
        thresholds.append(primary_threshold)
        thresholds = sorted(set(thresholds))
    for threshold in thresholds:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(
                "All threshold candidates must be between 0 and 1, "
                f"got {threshold}"
            )
    return thresholds


def maybe_configure_auto_pos_weight(
    cfg: dict[str, Any],
    train_rows: list[dict[str, Any]],
    is_main: bool,
) -> dict[str, Any] | None:
    if not bool(cfg["loss"].get("auto_pos_weight", False)):
        return None
    if cfg["loss"].get("pos_weight") is not None:
        return {
            "mode": "manual",
            "used_pos_weight": float(cfg["loss"]["pos_weight"]),
        }

    payload: list[dict[str, Any] | None] = [None]
    if is_main:
        stats = compute_mask_pixel_statistics(train_rows)
        min_weight = float(cfg["loss"].get("pos_weight_min", 1.0))
        max_weight = float(cfg["loss"].get("pos_weight_max", 20.0))
        raw_weight = float(stats["raw_pos_weight"])
        used_weight = min(max(raw_weight, min_weight), max_weight)
        stats["mode"] = "auto"
        stats["pos_weight_min"] = min_weight
        stats["pos_weight_max"] = max_weight
        stats["used_pos_weight"] = used_weight
        stats["pos_weight_was_clipped"] = used_weight != raw_weight
        payload[0] = stats

    if distributed_is_initialized():
        dist.broadcast_object_list(payload, src=0)

    stats = payload[0]
    if stats is None:
        raise RuntimeError("Failed to compute automatic pos_weight.")
    cfg["loss"]["pos_weight"] = float(stats["used_pos_weight"])
    cfg["loss"]["auto_pos_weight_stats"] = stats
    return stats


def resolve_run_group(cfg: dict[str, Any], world_size: int) -> str:
    run_type = str(cfg["output"].get("run_type", "auto")).lower()
    if run_type not in {"auto", "train", "test"}:
        raise ValueError(f"Unsupported output.run_type: {run_type!r}")
    if run_type != "auto":
        return run_type

    epochs = int(cfg["training"]["epochs"])
    return "test" if epochs <= 1 and world_size == 1 else "train"


def resolve_model_group(cfg: dict[str, Any]) -> str:
    return model_output_name(cfg["model"])


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(cfg: dict[str, Any]) -> dict[str, Any]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device,
            "backend": None,
        }

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = str(cfg["ddp"].get("backend", "nccl"))
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    return {
        "distributed": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "backend": backend,
    }


def cleanup_distributed() -> None:
    if distributed_is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_worker_init_fn(seed: int, rank: int):
    def _init_fn(worker_id: int) -> None:
        worker_seed = int(seed) + int(rank) * 1000 + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn


def get_split_img_size(cfg: dict[str, Any], split: str) -> int:
    value = cfg.get(split, {}).get("img_size", None)
    if value is None:
        value = cfg["data"]["img_size"]
    return int(value)


def build_loader(
    dataset: LungTumorNpyDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sampler=None,
    shuffle: bool = False,
    seed: int = 2004,
    rank: int = 0,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "worker_init_fn": make_worker_init_fn(seed, rank),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = cfg["training"].get("scheduler")
    if scheduler_name is None:
        return None

    scheduler_name = str(scheduler_name).lower()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["training"]["epochs"]),
            eta_min=float(cfg["training"].get("min_learning_rate", 0.0)),
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def tracker_to_metrics(tracker: torch.Tensor) -> dict[str, float]:
    if distributed_is_initialized():
        dist.all_reduce(tracker, op=dist.ReduceOp.SUM)

    sample_count = max(float(tracker[5].item()), 1.0)
    confusion = tracker[6:10]
    metrics = metrics_from_confusion(confusion)
    metrics.update(
        {
            "total_loss": float(tracker[0].item() / sample_count),
            "bce_loss": float(tracker[1].item() / sample_count),
            "dice_loss": float(tracker[2].item() / sample_count),
            "tversky_loss": float(tracker[3].item() / sample_count),
            "focal_tversky_loss": float(tracker[4].item() / sample_count),
            "samples": int(tracker[5].item()),
        }
    )
    return metrics


def reduce_threshold_tracker(
    threshold_tracker: torch.Tensor,
    threshold_candidates: list[float],
) -> dict[str, Any]:
    if distributed_is_initialized():
        dist.all_reduce(threshold_tracker, op=dist.ReduceOp.SUM)

    threshold_metrics: dict[str, dict[str, float]] = {}
    best_threshold = float(threshold_candidates[0])
    best_metrics: dict[str, float] | None = None

    for index, threshold in enumerate(threshold_candidates):
        metrics = metrics_from_confusion(threshold_tracker[index])
        threshold_metrics[f"{threshold:.3f}"] = metrics
        if best_metrics is None or metrics["dice"] > best_metrics["dice"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return {
        "threshold_metrics": threshold_metrics,
        "best_threshold": best_threshold,
        "best_dice": best_metrics["dice"],
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    mixed_precision: bool,
    threshold: float,
    rank: int,
    description: str,
    grad_clip_norm: float = 0.0,
    threshold_candidates: list[float] | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    tracker = torch.zeros(10, dtype=torch.float64, device=device)
    threshold_tracker = None
    if threshold_candidates:
        threshold_tracker = torch.zeros(
            (len(threshold_candidates), 4),
            dtype=torch.float64,
            device=device,
        )

    iterator = tqdm(
        loader,
        desc=description,
        disable=rank != 0,
        leave=False,
        dynamic_ncols=True,
    )

    for batch in iterator:
        images = batch["image"].to(device=device, non_blocking=True)
        masks = batch["mask"].to(device=device, non_blocking=True)
        batch_size = int(images.size(0))

        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type=device.type,
                enabled=mixed_precision and device.type == "cuda",
            ):
                logits = model(images)
                loss, loss_items = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                assert scaler is not None
                scaler.scale(loss).backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(grad_clip_norm),
                    )
                scaler.step(optimizer)
                scaler.update()

        probabilities = torch.sigmoid(logits.detach())
        masks_detached = masks.detach()
        confusion = confusion_from_probabilities(
            probabilities,
            masks_detached,
            threshold=threshold,
        ).to(device=device)

        if threshold_tracker is not None:
            for threshold_index, threshold_candidate in enumerate(threshold_candidates):
                threshold_tracker[threshold_index] += confusion_from_probabilities(
                    probabilities,
                    masks_detached,
                    threshold=threshold_candidate,
                ).to(device=device)

        tracker[0] += float(loss_items["total_loss"].item()) * batch_size
        tracker[1] += float(loss_items["bce_loss"].item()) * batch_size
        tracker[2] += float(loss_items["dice_loss"].item()) * batch_size
        tracker[3] += float(loss_items["tversky_loss"].item()) * batch_size
        tracker[4] += float(loss_items["focal_tversky_loss"].item()) * batch_size
        tracker[5] += batch_size
        tracker[6:10] += confusion

        if rank == 0:
            iterator.set_postfix(
                loss=f"{loss_items['total_loss'].item():.4f}",
                dice=f"{metrics_from_confusion(confusion)['dice']:.4f}",
            )

    metrics = tracker_to_metrics(tracker)
    if threshold_tracker is not None and threshold_candidates:
        metrics.update(
            reduce_threshold_tracker(
                threshold_tracker=threshold_tracker,
                threshold_candidates=threshold_candidates,
            )
        )
    return metrics


def get_peak_gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    memory = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / (1024**3)],
        dtype=torch.float64,
        device=device,
    )
    if distributed_is_initialized():
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    return float(memory.item())


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_val_dice: float,
    best_threshold: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
        "best_threshold": float(best_threshold),
        "config": config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def format_epoch_metrics(prefix: str, metrics: dict[str, float]) -> str:
    parts = [
        f"{prefix}_loss={metrics['total_loss']:.6f}",
        f"{prefix}_bce_loss={metrics['bce_loss']:.6f}",
        f"{prefix}_dice_loss={metrics['dice_loss']:.6f}",
        f"{prefix}_tversky_loss={metrics['tversky_loss']:.6f}",
        f"{prefix}_focal_tversky_loss={metrics['focal_tversky_loss']:.6f}",
        f"{prefix}_dice={metrics['dice']:.6f}",
        f"{prefix}_precision={metrics['precision']:.6f}",
        f"{prefix}_recall={metrics['recall']:.6f}",
    ]
    if "best_dice" in metrics:
        parts.extend(
            [
                f"{prefix}_best_dice={metrics['best_dice']:.6f}",
                f"{prefix}_best_threshold={metrics['best_threshold']:.3f}",
                f"{prefix}_best_precision={metrics['best_precision']:.6f}",
                f"{prefix}_best_recall={metrics['best_recall']:.6f}",
            ]
        )
    return ", ".join(parts)


def main() -> None:
    args = parse_args()
    cfg = build_effective_config(args)
    dist_info: dict[str, Any] = {}
    error_logger = None

    try:
        dist_info = setup_distributed(cfg)
        rank = int(dist_info["rank"])
        world_size = int(dist_info["world_size"])
        local_rank = int(dist_info["local_rank"])
        device: torch.device = dist_info["device"]
        is_main = rank == 0

        seed_everything(int(cfg["training"]["seed"]), rank=rank)
        run_group = resolve_run_group(cfg, world_size)
        model_group = resolve_model_group(cfg)
        cfg["output"]["run_type"] = run_group
        cfg["output"]["model_group"] = model_group

        if is_main:
            run_dirs = create_run_directories(
                cfg["output"]["output_dir"],
                cfg["output"].get("run_name"),
                run_group=run_group,
                model_group=model_group,
            )
            run_dir_str = str(run_dirs["run_dir"])
        else:
            run_dirs = {}
            run_dir_str = ""

        if distributed_is_initialized():
            payload = [run_dir_str]
            dist.broadcast_object_list(payload, src=0)
            run_dir_str = payload[0]
            if not is_main:
                run_dir = Path(run_dir_str)
                run_dirs = {
                    "run_dir": run_dir,
                    "logs_dir": run_dir / "logs",
                    "checkpoints_dir": run_dir / "checkpoints",
                    "metrics_dir": run_dir / "metrics",
                }
            dist.barrier()

        run_dir = Path(run_dir_str)
        logs_dir = run_dirs["logs_dir"]
        checkpoints_dir = run_dirs["checkpoints_dir"]
        metrics_dir = run_dirs["metrics_dir"]

        data_logger = setup_file_logger(
            "data_loading",
            logs_dir / "data_loading.log",
            enabled=is_main,
        )
        hyper_logger = setup_file_logger(
            "hyperparameters",
            logs_dir / "hyperparameters.log",
            enabled=is_main,
        )
        training_logger = setup_file_logger(
            "training",
            logs_dir / "training.log",
            enabled=is_main,
        )
        validation_logger = setup_file_logger(
            "validation",
            logs_dir / "validation.log",
            enabled=is_main,
        )
        testing_logger = setup_file_logger(
            "testing",
            logs_dir / "testing.log",
            enabled=is_main,
        )
        best_model_logger = setup_file_logger(
            "best_model",
            logs_dir / "best_model.log",
            enabled=is_main,
        )
        error_logger = setup_file_logger(
            "errors",
            logs_dir / "errors.log",
            enabled=is_main,
        )

        data_dir = Path(cfg["data"]["data_dir"]).expanduser().resolve(strict=False)
        output_dir = Path(cfg["output"]["output_dir"]).expanduser().resolve(strict=False)
        cfg["data"]["data_dir"] = str(data_dir)
        cfg["output"]["output_dir"] = str(output_dir)
        cfg["runtime"] = {
            "distributed": bool(dist_info["distributed"]),
            "backend": dist_info["backend"],
            "world_size": world_size,
            "local_rank": local_rank,
            "device": str(device),
            "total_batch_size": int(cfg["training"]["batch_size_per_gpu"]) * world_size,
            "run_group": run_group,
            "model_group": model_group,
            "split_img_sizes": {
                "train": get_split_img_size(cfg, "train"),
                "val": get_split_img_size(cfg, "val"),
                "test": get_split_img_size(cfg, "test"),
            },
        }

        if is_main:
            print(f"Run directory: {run_dir}", flush=True)

        splits = load_splits(data_dir)
        manifest_rows = load_manifest(data_dir)
        prepared_rows, file_report = prepare_manifest_rows(
            manifest_rows,
            data_dir=data_dir,
            validate_files=bool(cfg["data"].get("validate_files", True)),
        )
        rows_by_split = split_rows(prepared_rows)

        for split_name in ("train", "val", "test"):
            if not rows_by_split[split_name]:
                raise ValueError(f"No rows for split={split_name!r} in manifest.csv")

        train_rows = rows_by_split["train"]
        train_selection_stats = {
            "source": "manifest.csv",
            "note": "balance, augmentation and patch sampling are prepared in 1-stage",
            "train_slices": len(train_rows),
            "positive_slices": sum(int(float(row["has_tumor"])) for row in train_rows),
            "negative_slices": sum(1 - int(float(row["has_tumor"])) for row in train_rows),
        }
        if not train_rows:
            raise ValueError("Train rows are empty.")

        split_stats = compute_split_statistics(prepared_rows, splits)
        split_stats["train_selection"] = train_selection_stats
        split_img_sizes = {
            "train": get_split_img_size(cfg, "train"),
            "val": get_split_img_size(cfg, "val"),
            "test": get_split_img_size(cfg, "test"),
        }
        sample_info = inspect_npy_samples(
            rows_by_split,
            img_sizes_by_split=split_img_sizes,
        )
        auto_pos_weight_stats = maybe_configure_auto_pos_weight(
            cfg=cfg,
            train_rows=train_rows,
            is_main=is_main,
        )
        if auto_pos_weight_stats is not None:
            split_stats["train_pixel_statistics"] = auto_pos_weight_stats

        threshold = float(cfg["metrics"]["threshold"])
        threshold_candidates = get_threshold_candidates(cfg)
        cfg["metrics"]["threshold_candidates"] = threshold_candidates

        if is_main:
            save_yaml(cfg, run_dir / "config.yaml")
            data_logger.info("data_dir: %s", data_dir)
            data_logger.info("manifest_path: %s", data_dir / "manifest.csv")
            data_logger.info("splits_path: %s", data_dir / "splits.json")
            log_dict(data_logger, split_stats, title="split statistics")
            log_dict(data_logger, file_report, title="file verification")
            log_dict(data_logger, sample_info, title="sample npy shapes and dtypes")
            save_json(split_stats, metrics_dir / "split_statistics.json")

        train_dataset = LungTumorNpyDataset(
            train_rows,
            data_dir=data_dir,
            img_size=get_split_img_size(cfg, "train"),
        )
        val_dataset = LungTumorNpyDataset(
            rows_by_split["val"],
            data_dir=data_dir,
            img_size=get_split_img_size(cfg, "val"),
        )
        test_dataset = LungTumorNpyDataset(
            rows_by_split["test"],
            data_dir=data_dir,
            img_size=get_split_img_size(cfg, "test"),
        )

        if dist_info["distributed"]:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(cfg["training"]["seed"]),
                drop_last=False,
            )
        else:
            train_sampler = None

        if dist_info["distributed"]:
            val_sampler = DistributedEvalSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
            )
            test_sampler = DistributedEvalSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
            )
        else:
            val_sampler = None
            test_sampler = None

        batch_size = int(cfg["training"]["batch_size_per_gpu"])
        num_workers = int(cfg["training"]["num_workers"])
        train_loader = build_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            seed=int(cfg["training"]["seed"]),
            rank=rank,
        )
        val_loader = build_loader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            sampler=val_sampler,
            shuffle=False,
            seed=int(cfg["training"]["seed"]),
            rank=rank,
        )
        test_loader = build_loader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            sampler=test_sampler,
            shuffle=False,
            seed=int(cfg["training"]["seed"]),
            rank=rank,
        )

        model = build_model(cfg["model"])
        if (
            bool(cfg["model"].get("sync_batchnorm", False))
            and dist_info["distributed"]
            and device.type == "cuda"
        ):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)

        if dist_info["distributed"]:
            find_unused_parameters = bool(
                cfg.get("ddp", {}).get("find_unused_parameters", False)
            )
            if device.type == "cuda":
                model = DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters,
                )
            else:
                model = DistributedDataParallel(
                    model,
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters,
                )

        criterion = BCEDiceLoss(
            bce_weight=float(cfg["loss"]["bce_weight"]),
            dice_weight=float(cfg["loss"]["dice_weight"]),
            dice_smooth=float(cfg["loss"]["dice_smooth"]),
            pos_weight=cfg["loss"].get("pos_weight"),
            tversky_weight=float(cfg["loss"].get("tversky_weight", 0.0)),
            focal_tversky_weight=float(
                cfg["loss"].get("focal_tversky_weight", 0.0)
            ),
            tversky_alpha=float(cfg["loss"].get("tversky_alpha", 0.3)),
            tversky_beta=float(cfg["loss"].get("tversky_beta", 0.7)),
            focal_tversky_gamma=float(
                cfg["loss"].get("focal_tversky_gamma", 0.75)
            ),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        scheduler = create_scheduler(optimizer, cfg)
        scaler = torch.cuda.amp.GradScaler(
            enabled=bool(cfg["training"]["mixed_precision"]) and device.type == "cuda"
        )

        start_epoch = 1
        best_val_dice = -1.0
        best_threshold = threshold
        resume_path = cfg["training"].get("resume")
        if resume_path:
            checkpoint = load_checkpoint(resume_path, map_location=device)
            unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
            best_threshold = float(checkpoint.get("best_threshold", threshold))

        if is_main:
            hyperparameters = {
                "architecture": cfg["model"],
                "img_size": cfg["data"]["img_size"],
                "split_img_sizes": split_img_sizes,
                "batch_size_per_gpu": batch_size,
                "total_batch_size": batch_size * world_size,
                "epochs": cfg["training"]["epochs"],
                "learning_rate": cfg["training"]["learning_rate"],
                "optimizer": cfg["training"]["optimizer"],
                "loss_function": cfg["loss"]["name"],
                "scheduler": cfg["training"].get("scheduler"),
                "seed": cfg["training"]["seed"],
                "num_workers": num_workers,
                "mixed_precision": cfg["training"]["mixed_precision"],
                "ddp_world_size": world_size,
                "run_group": run_group,
                "model_group": model_group,
                "train": cfg["train"],
                "val": cfg["val"],
                "test": cfg["test"],
                "loss": cfg["loss"],
                "early_stopping": cfg["early_stopping"],
                "threshold": threshold,
                "threshold_candidates": threshold_candidates,
            }
            log_dict(hyper_logger, hyperparameters, title="hyperparameters")

        history: list[dict[str, Any]] = []
        epochs = int(cfg["training"]["epochs"])
        grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 0.0))
        best_model_metrics: dict[str, Any] | None = None
        early_cfg = cfg.get("early_stopping", {})
        early_stopping_enabled = bool(early_cfg.get("enabled", False))
        early_stopping_monitor = str(early_cfg.get("monitor", "val_best_dice"))
        early_stopping_mode = str(early_cfg.get("mode", "max")).lower()
        early_stopping_patience = int(early_cfg.get("patience", 20))
        early_stopping_min_delta = float(early_cfg.get("min_delta", 0.001))
        early_stopping_min_epochs = int(early_cfg.get("min_epochs", 30))
        if early_stopping_monitor != "val_best_dice":
            raise ValueError(
                "Only early_stopping.monitor=val_best_dice is supported, "
                f"got {early_stopping_monitor!r}"
            )
        if early_stopping_mode != "max":
            raise ValueError(
                "Only early_stopping.mode=max is supported, "
                f"got {early_stopping_mode!r}"
            )
        if early_stopping_patience < 1:
            raise ValueError("early_stopping.patience must be >= 1")
        if early_stopping_min_epochs < 1:
            raise ValueError("early_stopping.min_epochs must be >= 1")
        early_stopping_best = best_val_dice
        no_improve_epochs = 0

        for epoch in range(start_epoch, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            epoch_start = time.time()
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                mixed_precision=bool(cfg["training"]["mixed_precision"]),
                threshold=threshold,
                rank=rank,
                description=f"epoch {epoch}/{epochs} train",
                grad_clip_norm=grad_clip_norm,
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=scaler,
                mixed_precision=bool(cfg["training"]["mixed_precision"]),
                threshold=threshold,
                rank=rank,
                description=f"epoch {epoch}/{epochs} val",
                threshold_candidates=threshold_candidates,
            )

            selection_dice = float(val_metrics.get("best_dice", val_metrics["dice"]))
            selection_threshold = float(
                val_metrics.get("best_threshold", threshold)
            )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(selection_dice)
                else:
                    scheduler.step()

            epoch_time = time.time() - epoch_start
            lr = float(optimizer.param_groups[0]["lr"])
            gpu_memory_gb = get_peak_gpu_memory_gb(device)

            is_best = selection_dice > best_val_dice
            if is_best:
                best_val_dice = selection_dice
                best_threshold = selection_threshold
            has_meaningful_improvement = (
                selection_dice > early_stopping_best + early_stopping_min_delta
            )
            if has_meaningful_improvement:
                early_stopping_best = selection_dice
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            early_stop_triggered = (
                early_stopping_enabled
                and epoch >= early_stopping_min_epochs
                and no_improve_epochs >= early_stopping_patience
            )

            if is_main:
                training_logger.info(
                    "epoch=%d | %s | lr=%.8g | epoch_time_sec=%.2f | "
                    "gpu_memory_gb=%.3f",
                    epoch,
                    format_epoch_metrics("train", train_metrics),
                    lr,
                    epoch_time,
                    gpu_memory_gb,
                )
                validation_logger.info(
                    "epoch=%d | %s | saved_best_model=%s",
                    epoch,
                    format_epoch_metrics("val", val_metrics),
                    bool(is_best),
                )

                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["total_loss"],
                    "train_bce_loss": train_metrics["bce_loss"],
                    "train_dice_loss": train_metrics["dice_loss"],
                    "train_tversky_loss": train_metrics["tversky_loss"],
                    "train_focal_tversky_loss": train_metrics["focal_tversky_loss"],
                    "train_dice": train_metrics["dice"],
                    "val_loss": val_metrics["total_loss"],
                    "val_bce_loss": val_metrics["bce_loss"],
                    "val_dice_loss": val_metrics["dice_loss"],
                    "val_tversky_loss": val_metrics["tversky_loss"],
                    "val_focal_tversky_loss": val_metrics["focal_tversky_loss"],
                    "val_dice": val_metrics["dice"],
                    "val_best_dice": val_metrics.get("best_dice"),
                    "val_best_threshold": val_metrics.get("best_threshold"),
                    "val_best_precision": val_metrics.get("best_precision"),
                    "val_best_recall": val_metrics.get("best_recall"),
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "lr": lr,
                    "epoch_time_sec": epoch_time,
                    "gpu_memory_gb": gpu_memory_gb,
                    "early_stop_no_improve_epochs": no_improve_epochs,
                    "early_stop_triggered": early_stop_triggered,
                }
                history.append(row)
                write_history_csv(history, metrics_dir / "history.csv")
                save_json(history, metrics_dir / "history.json")

                if is_best:
                    best_checkpoint_path = checkpoints_dir / "best_model.pth"
                    save_checkpoint(
                        best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_val_dice=best_val_dice,
                        best_threshold=best_threshold,
                        config=cfg,
                    )
                    best_model_metrics = {
                        "checkpoint_path": str(best_checkpoint_path),
                        "epoch": epoch,
                        "best_val_dice": best_val_dice,
                        "best_threshold": best_threshold,
                        "train": train_metrics,
                        "validation": val_metrics,
                        "lr": lr,
                        "epoch_time_sec": epoch_time,
                        "gpu_memory_gb": gpu_memory_gb,
                        "run_group": run_group,
                        "model_group": model_group,
                    }
                    save_json(
                        best_model_metrics,
                        metrics_dir / "best_model_metrics.json",
                    )
                    best_model_logger.info(
                        "saved_best_model | epoch=%d | checkpoint_path=%s | "
                        "best_val_dice=%.6f | best_threshold=%.3f | %s | %s | lr=%.8g | "
                        "epoch_time_sec=%.2f | gpu_memory_gb=%.3f",
                        epoch,
                        best_checkpoint_path,
                        best_val_dice,
                        best_threshold,
                        format_epoch_metrics("train", train_metrics),
                        format_epoch_metrics("val", val_metrics),
                        lr,
                        epoch_time,
                        gpu_memory_gb,
                    )
                if early_stop_triggered:
                    training_logger.info(
                        "early_stopping | stopped=True | epoch=%d | "
                        "best_val_dice=%.6f | early_stopping_best=%.6f | "
                        "no_improve_epochs=%d | patience=%d | min_delta=%.6f | "
                        "min_epochs=%d",
                        epoch,
                        best_val_dice,
                        early_stopping_best,
                        no_improve_epochs,
                        early_stopping_patience,
                        early_stopping_min_delta,
                        early_stopping_min_epochs,
                    )

            if distributed_is_initialized():
                dist.barrier()
            if early_stop_triggered:
                break

        if distributed_is_initialized():
            dist.barrier()

        best_checkpoint_path = checkpoints_dir / "best_model.pth"
        best_checkpoint = load_checkpoint(best_checkpoint_path, map_location=device)
        unwrap_model(model).load_state_dict(best_checkpoint["model_state_dict"])
        test_threshold = float(best_checkpoint.get("best_threshold", best_threshold))

        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            mixed_precision=bool(cfg["training"]["mixed_precision"]),
            threshold=test_threshold,
            rank=rank,
            description="test",
        )

        if is_main:
            testing_logger.info("checkpoint_path: %s", best_checkpoint_path)
            testing_logger.info("%s", format_epoch_metrics("test", test_metrics))
            if best_model_metrics is None:
                best_model_metrics = {
                    "checkpoint_path": str(best_checkpoint_path),
                    "epoch": int(best_checkpoint.get("epoch", 0)),
                    "best_val_dice": float(best_checkpoint.get("best_val_dice", 0.0)),
                    "best_threshold": test_threshold,
                   "run_group": run_group,
                    "model_group": model_group,
                }
            best_model_metrics["test"] = test_metrics
            best_model_metrics["test_threshold"] = test_threshold
            save_json(best_model_metrics, metrics_dir / "best_model_metrics.json")
            best_model_logger.info(
                "test_evaluation | checkpoint_path=%s | threshold=%.3f | %s",
                best_checkpoint_path,
                test_threshold,
                format_epoch_metrics("test", test_metrics),
            )
            test_output = {
                "checkpoint_path": str(best_checkpoint_path),
                "best_model": best_model_metrics,
                "threshold": test_threshold,
                "total_loss": test_metrics["total_loss"],
                "loss": test_metrics["total_loss"],
                "bce_loss": test_metrics["bce_loss"],
                "dice_loss": test_metrics["dice_loss"],
                "tversky_loss": test_metrics["tversky_loss"],
                "focal_tversky_loss": test_metrics["focal_tversky_loss"],
                "dice": test_metrics["dice"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "specificity": test_metrics["specificity"],
                "samples": test_metrics["samples"],
            }
            save_json(test_output, metrics_dir / "test_metrics.json")

        if distributed_is_initialized():
            dist.barrier()

    except Exception:
        if error_logger is not None:
            error_logger.error(traceback.format_exc())
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
