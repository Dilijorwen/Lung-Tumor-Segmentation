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
    compute_split_statistics,
    inspect_npy_samples,
    load_manifest,
    load_splits,
    prepare_manifest_rows,
    split_rows,
)
from losses import BCEDiceLoss
from metrics import confusion_from_logits, metrics_from_confusion
from model import build_model
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
            "data_dir": "preprocessed_npy",
            "img_size": 512,
            "validate_files": True,
        },
        "output": {
            "output_dir": "outputs",
            "run_name": None,
            "checkpoint_every": 1,
        },
        "model": {
            "name": "UNet2D",
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 32,
            "bilinear": False,
            "sync_batchnorm": False,
        },
        "loss": {
            "name": "BCEWithLogitsLoss + DiceLoss",
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "dice_smooth": 1.0,
            "pos_weight": None,
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
            "seed": 42,
            "mixed_precision": True,
            "grad_clip_norm": 0.0,
            "resume": None,
        },
        "metrics": {
            "threshold": 0.5,
        },
        "ddp": {
            "backend": "nccl",
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
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)

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
        ("output", "output_dir"): args.output_dir,
        ("output", "run_name"): args.run_name,
        ("output", "checkpoint_every"): args.checkpoint_every,
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
    }
    for (section, key), value in overrides.items():
        if value is not None:
            cfg[section][key] = value

    if str(cfg["training"]["scheduler"]).lower() == "none":
        cfg["training"]["scheduler"] = None
    return cfg


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

    return _init_fn


def build_loader(
    dataset: LungTumorNpyDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sampler=None,
    shuffle: bool = False,
    seed: int = 42,
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

    sample_count = max(float(tracker[3].item()), 1.0)
    confusion = tracker[4:8]
    metrics = metrics_from_confusion(confusion)
    metrics.update(
        {
            "total_loss": float(tracker[0].item() / sample_count),
            "bce_loss": float(tracker[1].item() / sample_count),
            "dice_loss": float(tracker[2].item() / sample_count),
            "samples": int(tracker[3].item()),
        }
    )
    return metrics


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
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    tracker = torch.zeros(8, dtype=torch.float64, device=device)

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

        confusion = confusion_from_logits(
            logits.detach(),
            masks.detach(),
            threshold=threshold,
        ).to(device=device)

        tracker[0] += float(loss_items["total_loss"].item()) * batch_size
        tracker[1] += float(loss_items["bce_loss"].item()) * batch_size
        tracker[2] += float(loss_items["dice_loss"].item()) * batch_size
        tracker[3] += batch_size
        tracker[4:8] += confusion

        if rank == 0:
            iterator.set_postfix(
                loss=f"{loss_items['total_loss'].item():.4f}",
                dice=f"{metrics_from_confusion(confusion)['dice']:.4f}",
            )

    return tracker_to_metrics(tracker)


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
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
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
    return (
        f"{prefix}_loss={metrics['total_loss']:.6f}, "
        f"{prefix}_bce_loss={metrics['bce_loss']:.6f}, "
        f"{prefix}_dice_loss={metrics['dice_loss']:.6f}, "
        f"{prefix}_dice={metrics['dice']:.6f}, "
        f"{prefix}_f1={metrics['f1_score']:.6f}, "
        f"{prefix}_iou={metrics['iou']:.6f}, "
        f"{prefix}_precision={metrics['precision']:.6f}, "
        f"{prefix}_recall={metrics['recall']:.6f}"
    )


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

        if is_main:
            run_dirs = create_run_directories(
                cfg["output"]["output_dir"],
                cfg["output"].get("run_name"),
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
        }

        if is_main:
            print(f"Run directory: {run_dir}", flush=True)
            save_yaml(cfg, run_dir / "config.yaml")

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

        split_stats = compute_split_statistics(prepared_rows, splits)
        sample_info = inspect_npy_samples(
            rows_by_split,
            img_size=int(cfg["data"]["img_size"]),
        )

        if is_main:
            data_logger.info("data_dir: %s", data_dir)
            data_logger.info("manifest_path: %s", data_dir / "manifest.csv")
            data_logger.info("splits_path: %s", data_dir / "splits.json")
            log_dict(data_logger, split_stats, title="split statistics")
            log_dict(data_logger, file_report, title="file verification")
            log_dict(data_logger, sample_info, title="sample npy shapes and dtypes")
            save_json(split_stats, metrics_dir / "split_statistics.json")

        train_dataset = LungTumorNpyDataset(
            rows_by_split["train"],
            data_dir=data_dir,
            img_size=int(cfg["data"]["img_size"]),
        )
        val_dataset = LungTumorNpyDataset(
            rows_by_split["val"],
            data_dir=data_dir,
            img_size=int(cfg["data"]["img_size"]),
        )
        test_dataset = LungTumorNpyDataset(
            rows_by_split["test"],
            data_dir=data_dir,
            img_size=int(cfg["data"]["img_size"]),
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
            train_sampler = None
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
            if device.type == "cuda":
                model = DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                )
            else:
                model = DistributedDataParallel(model, broadcast_buffers=False)

        criterion = BCEDiceLoss(
            bce_weight=float(cfg["loss"]["bce_weight"]),
            dice_weight=float(cfg["loss"]["dice_weight"]),
            dice_smooth=float(cfg["loss"]["dice_smooth"]),
            pos_weight=cfg["loss"].get("pos_weight"),
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

        if is_main:
            hyperparameters = {
                "architecture": cfg["model"],
                "img_size": cfg["data"]["img_size"],
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
            }
            log_dict(hyper_logger, hyperparameters, title="hyperparameters")

        history: list[dict[str, Any]] = []
        epochs = int(cfg["training"]["epochs"])
        threshold = float(cfg["metrics"]["threshold"])
        grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 0.0))
        checkpoint_every = int(cfg["output"].get("checkpoint_every", 1))

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
            )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["dice"])
                else:
                    scheduler.step()

            epoch_time = time.time() - epoch_start
            lr = float(optimizer.param_groups[0]["lr"])
            gpu_memory_gb = get_peak_gpu_memory_gb(device)

            is_best = val_metrics["dice"] > best_val_dice
            if is_best:
                best_val_dice = val_metrics["dice"]

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

                save_checkpoint(
                    checkpoints_dir / "last_model.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_dice=best_val_dice,
                    config=cfg,
                )
                if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                    save_checkpoint(
                        checkpoints_dir / f"epoch_{epoch:04d}.pth",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_val_dice=best_val_dice,
                        config=cfg,
                    )
                if is_best:
                    save_checkpoint(
                        checkpoints_dir / "best_model.pth",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_val_dice=best_val_dice,
                        config=cfg,
                    )

                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["total_loss"],
                    "train_bce_loss": train_metrics["bce_loss"],
                    "train_dice_loss": train_metrics["dice_loss"],
                    "train_dice": train_metrics["dice"],
                    "train_f1_score": train_metrics["f1_score"],
                    "train_iou": train_metrics["iou"],
                    "val_loss": val_metrics["total_loss"],
                    "val_bce_loss": val_metrics["bce_loss"],
                    "val_dice_loss": val_metrics["dice_loss"],
                    "val_dice": val_metrics["dice"],
                    "val_f1_score": val_metrics["f1_score"],
                    "val_iou": val_metrics["iou"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "lr": lr,
                    "epoch_time_sec": epoch_time,
                    "gpu_memory_gb": gpu_memory_gb,
                }
                history.append(row)
                write_history_csv(history, metrics_dir / "history.csv")
                save_json(history, metrics_dir / "history.json")

            if distributed_is_initialized():
                dist.barrier()

        if distributed_is_initialized():
            dist.barrier()

        best_checkpoint_path = checkpoints_dir / "best_model.pth"
        best_checkpoint = load_checkpoint(best_checkpoint_path, map_location=device)
        unwrap_model(model).load_state_dict(best_checkpoint["model_state_dict"])

        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            mixed_precision=bool(cfg["training"]["mixed_precision"]),
            threshold=threshold,
            rank=rank,
            description="test",
        )

        if is_main:
            testing_logger.info("checkpoint_path: %s", best_checkpoint_path)
            testing_logger.info("%s", format_epoch_metrics("test", test_metrics))
            test_output = {
                "checkpoint_path": str(best_checkpoint_path),
                "total_loss": test_metrics["total_loss"],
                "loss": test_metrics["total_loss"],
                "bce_loss": test_metrics["bce_loss"],
                "dice_loss": test_metrics["dice_loss"],
                "dice": test_metrics["dice"],
                "f1_score": test_metrics["f1_score"],
                "iou": test_metrics["iou"],
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
