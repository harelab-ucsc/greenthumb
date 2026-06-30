"""
benchmark.py

Author:
    nubby
    Taylor Kergan

Date:
    30 Jun 2026

Version:
    1.0.1
"""
from __future__ import annotations

import argparse
import copy
import logging
import random
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb

"""
from .data_pipeline import build_data_bundle
from .models import LSTMClassifier, TemporalConvNetClassifier, TransformerClassifier
"""
from data_pipeline import build_data_bundle
from models import LSTMClassifier, TemporalConvNetClassifier, TransformerClassifier


spr_max = 1000

@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32 
    lr: float = 3e-4
    weight_decay: float = 1e-3
    patience: int = 30
    grad_clip: float = 1.0


@dataclass
class TrainingHistory:
    epoch_indices: List[int] = field(default_factory=list)
    train_epoch_loss: List[float] = field(default_factory=list)
    train_epoch_acc: List[float] = field(default_factory=list)
    val_epoch_loss: List[float] = field(default_factory=list)
    val_epoch_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    train_batch_loss: List[float] = field(default_factory=list)
    train_batch_step: List[int] = field(default_factory=list)
    best_epoch: Optional[int] = None
    test_loss: Optional[float] = None
    test_acc: Optional[float] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    history: Optional[TrainingHistory] = None,
    split: str = "train",
    global_step: int = 0,
    wandb_run: Optional[object] = None,
    wandb_prefix: str = "",
) -> Tuple[float, float, int]:
    """Runs a single training or evaluation epoch and logs batch metrics if requested."""
    #spr_thresh = 30  # 30 PSI of absolute error is acceptable.
    spr_thresh = 30 / spr_max  # 30 PSI of absolute error is acceptable.
    if train:
        if optimizer is None:
            raise ValueError("Optimizer must be provided when train=True.")
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    next_step = global_step

    for batch_idx, (inputs, lengths, labels, _) in enumerate(dataloader):
        step_id = global_step + batch_idx
        inputs = inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if history is not None:
                history.train_batch_loss.append(loss.item())
                history.train_batch_step.append(step_id)

            if wandb_run is not None:
                wandb_run.log(
                    {f"{wandb_prefix}/train_batch_loss": loss.item()},
                    step=step_id,
                )

        total_loss += loss.item() * labels.size(0)
        preds = logits
        #errors_abs = abs(preds - logits)
        # Use only the first layer of SPRs for predictions.
        errors_abs = abs(preds[:,0] - labels[:,0])

        total_correct += (errors_abs <= spr_thresh).sum().item()
        total_examples += labels.size(0) * 3
        next_step = step_id + 1

    avg_loss = total_loss / max(1, total_examples)
    avg_acc = total_correct / max(1, total_examples)
    return avg_loss, avg_acc, next_step


def train_and_evaluate(
    model: nn.Module,
    bundle,
    device: torch.device,
    config: TrainingConfig,
    *,
    model_name: str,
    wandb_run: Optional[object] = None,
) -> Tuple[Dict[str, float], TrainingHistory]:
    train_loader = bundle.dataloader("train", batch_size=config.batch_size)
    val_loader = bundle.dataloader("val", batch_size=config.batch_size)
    test_loader = bundle.dataloader("test", batch_size=config.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = TrainingHistory()

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    global_step = 0

    if wandb_run is not None:
        wandb_run.watch(model, log="gradients", log_freq=200)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, global_step = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            train=True,
            optimizer=optimizer,
            grad_clip=config.grad_clip,
            history=history,
            split="train",
            global_step=global_step,
            wandb_run=wandb_run,
            wandb_prefix=model_name,
        )
        val_loss, val_acc, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            train=False,
            split="val",
        )
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        history.epoch_indices.append(epoch)
        history.train_epoch_loss.append(train_loss)
        history.train_epoch_acc.append(train_acc)
        history.val_epoch_loss.append(val_loss)
        history.val_epoch_acc.append(val_acc)
        history.learning_rates.append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} best_val_acc={best_val_acc:.3f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    f"{model_name}/train_loss": train_loss,
                    f"{model_name}/train_acc": train_acc,
                    f"{model_name}/val_loss": val_loss,
                    f"{model_name}/val_acc": val_acc,
                    f"{model_name}/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        if epochs_no_improve >= config.patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    history.best_epoch = best_epoch

    test_loss, test_acc, _ = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        train=False,
        split="test",
    )
    history.test_loss = test_loss
    history.test_acc = test_acc

    if wandb_run is not None:
        wandb_run.log(
            {
                f"{model_name}/best_val_acc": best_val_acc,
                f"{model_name}/test_acc": test_acc,
                f"{model_name}/test_loss": test_loss,
                f"{model_name}/best_epoch": best_epoch,
            },
            step=max(history.epoch_indices, default=0),
        )

    return (
        {
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "best_epoch": best_epoch,
        },
        history,
    )


def save_history(history: TrainingHistory, output_dir: Path, model_name: str) -> Path:
    """Serialises training history to JSON for downstream analysis."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    history_path = model_dir / "history.json"
    payload = {
        "epoch_indices": history.epoch_indices,
        "train_epoch_loss": history.train_epoch_loss,
        "train_epoch_acc": history.train_epoch_acc,
        "val_epoch_loss": history.val_epoch_loss,
        "val_epoch_acc": history.val_epoch_acc,
        "learning_rates": history.learning_rates,
        "train_batch_loss": history.train_batch_loss,
        "train_batch_step": history.train_batch_step,
        "best_epoch": history.best_epoch,
        "test_loss": history.test_loss,
        "test_acc": history.test_acc,
    }
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return history_path


def save_plots(history: TrainingHistory, output_dir: Path, model_name: str) -> None:
    """Generates publication-ready loss/accuracy plots."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if history.epoch_indices:
        with plt.style.context("seaborn-v0_8"):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(history.epoch_indices, history.train_epoch_loss, marker="o", label="Train")
            axes[0].plot(history.epoch_indices, history.val_epoch_loss, marker="o", label="Validation")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Cross-Entropy Loss")
            axes[0].set_title(f"{model_name.upper()} Loss")
            axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)
            axes[0].legend()

            axes[1].plot(history.epoch_indices, history.train_epoch_acc, marker="o", label="Train")
            axes[1].plot(history.epoch_indices, history.val_epoch_acc, marker="o", label="Validation")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title(f"{model_name.upper()} Accuracy")
            axes[1].set_ylim(0.0, 1.05)
            axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(model_dir / f"{model_name}_epoch_metrics.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    if history.train_batch_step:
        with plt.style.context("seaborn-v0_8"):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history.train_batch_step, history.train_batch_loss, color="#1f77b4", linewidth=1.0)
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Batch Loss")
            ax.set_title(f"{model_name.upper()} Training Batch Loss")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(model_dir / f"{model_name}_train_batch_loss.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

def set_up(
        no_cuda: bool,
        output_dir: Path,
        seed: int,
        ) -> torch.device:
    # Select device for training.
    device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        )
    print(f"Using device: {device}")

    # Set random seed.
    set_seed(seed)

    # Set up output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    return device

def benchmark(
        batch_size: int,
        data_dir: Path,
        epochs: int,
        lr: float,
        models: list[str],
        no_cuda: bool,
        output_dir: Path,
        patience: float,
        seed: int,
        stride: int,
        use_wandb: bool,
        wandb_entity: str,
        wandb_group: str,
        wandb_mode: str,
        wandb_project: str,
        wandb_run_prefix: str,
        weight_decay: float,
        window_size: int,
    ):
    """
    benchmark(...)

    Main training pipeline entry point.
    """
    device = set_up(
            no_cuda=no_cuda,
            output_dir=output_dir,
            seed=seed,
        )

    # Create data bundle for training/evals.
    data_bundle = build_data_bundle(
        data_dir,
        seed=seed,
        window_size=window_size,
        stride=stride,
        mode="b1"   # NOTE: b1,qcat
    )
    input_dim = len(data_bundle.feature_names)
    num_classes = 3

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )

    available_models = {
        "lstm": LSTMClassifier(input_dim=input_dim, num_classes=num_classes),
        "tcn": TemporalConvNetClassifier(input_dim=input_dim, num_classes=num_classes),
        "transformer": TransformerClassifier(input_dim=input_dim, num_classes=num_classes),
    }

    unknown = set(models) - set(available_models.keys())
    if unknown:
        raise ValueError(f"Unknown model names requested: {', '.join(sorted(unknown))}")
    models = {name: available_models[name] for name in models}

    dataset_summary = {
        "train_samples": len(data_bundle.train),
        "val_samples": len(data_bundle.val),
        "test_samples": len(data_bundle.test),
        "feature_dim": input_dim,
        "window_size": window_size,
        "stride": stride,
    }

    print(
        "Dataset sequences -> "
        f"train: {dataset_summary['train_samples']}, "
        f"val: {dataset_summary['val_samples']}, "
        f"test: {dataset_summary['test_samples']}, "
        f"feature_dim: {dataset_summary['feature_dim']}, "
        f"window_size: {dataset_summary['window_size']}, "
        f"stride: {dataset_summary['stride']}"
    )

    results: Dict[str, Dict[str, float]] = {}
    histories: Dict[str, TrainingHistory] = {}

    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        layers = list(model.modules())
        num_layers = len(layers)
        print(f"\n=== Training {name.upper()} ({param_count} params, {num_layers} layers) ===")
        model.to(device)

        wandb_run = None
        if use_wandb and wandb_mode != "disabled":
            if wandb is None:
                raise RuntimeError(
                    "Weights & Biases is not installed. Install it with 'pip install wandb' to enable logging."
                )
            run_name = f"{wandb_run_prefix}-{name}-seed{seed}"
            run_config = {
                **dataset_summary,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "model": name,
                "params": param_count,
            }
            wandb_run = wandb.init(  # type: ignore[call-arg]
                project=wandb_project,
                entity=wandb_entity,
                group=wandb_group,
                name=run_name,
                config=run_config,
                mode=wandb_mode,
                reinit=True,
            )

        outcomes, history = train_and_evaluate(
            model,
            data_bundle,
            device,
            config,
            model_name=name,
            wandb_run=wandb_run,
        )
        results[name] = outcomes
        histories[name] = history
        save_history(history, output_dir, name)
        save_plots(history, output_dir, name)

        print(
            f"{name.upper()} best_val_acc={outcomes['best_val_acc']:.3f} "
            f"test_acc={outcomes['test_acc']:.3f} test_loss={outcomes['test_loss']:.4f} "
            f"(best_epoch={outcomes['best_epoch']})"
        )

        if wandb_run is not None:
            wandb_run.finish()

    print("\n=== Summary ===")
    for name, metrics in results.items():
        print(
            f"{name.upper():12s} | val_acc={metrics['best_val_acc']:.3f} "
            f"| test_acc={metrics['test_acc']:.3f} | best_epoch={metrics['best_epoch']}"
        )
        print(f"Artifacts saved to {output_dir / name}")


if __name__ == "__main__":
    # Arg parsing.
    parser = argparse.ArgumentParser(
            description="Benchmark sequence models on DyRET terrain dataset."
        )
    parser.add_argument(
            "--data-dir",
            type=Path,
            default=Path("data"),
            help="Path to the data directory."
        )
    parser.add_argument(
            "--epochs",
            type=int,
            default=30,
            help="Maximum number of training epochs."
        )
    parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Mini-batch size."
        )
    parser.add_argument(
            "--lr",
            type=float,
            default=3e-4,
            help="Initial learning rate."
        )
    parser.add_argument(
            "--weight-decay",
            type=float,
            default=1e-3,
            help="Weight decay for AdamW."
        )
    parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed."
        )
    parser.add_argument(
            "--patience",
            type=int,
            default=6,
            help="Early stopping patience."
        )
    parser.add_argument(
            "--models",
            nargs="+",
            default=["lstm", "tcn", "transformer"],
            help="Subset of models to train (choices: lstm, tcn, transformer).",
        )
    parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("artifacts"),
            help="Directory to store plots and training histories.",
        )
    parser.add_argument(
            "--window-size",
            type=int,
            default=None,
            help="Optional sliding window size (timesteps) applied within each gait step.",
        )
    parser.add_argument(
            "--stride",
            type=int,
            default=None,
            help="Stride (timesteps) between sliding windows; defaults to window size for non-overlapping segments.",
        )
    parser.add_argument(
            "--use-wandb",
            action="store_true",
            help="Enable Weights & Biases logging.",
        )
    parser.add_argument(
            "--wandb-project",
            type=str,
            default="dyret-terrain",
            help="Weights & Biases project name.",
        )
    parser.add_argument(
            "--wandb-entity",
            type=str,
            default=None,
            help="Weights & Biases entity (team or username).",
        )
    parser.add_argument(
            "--wandb-group",
            type=str,
            default=None,
            help="Optional Weights & Biases group label for the runs.",
        )
    parser.add_argument(
            "--wandb-run-prefix",
            type=str,
            default="terrain",
            help="Prefix for generated Weights & Biases run names.",
        )
    parser.add_argument(
            "--wandb-mode",
            type=str,
            choices=["online", "offline", "disabled"],
            default="online",
            help=("Weights & Biases mode (use 'offline' when working without "
                  "network)."),
        )
    parser.add_argument(
            "--no-cuda",
            action="store_true",
            help="Force CPU execution even if CUDA is available."
        )
    args = parser.parse_args()

    if args.stride is not None and args.window_size is None:
        parser.error("--stride requires --window-size.")
    if args.window_size is not None and args.window_size <= 0:
        parser.error("--window-size must be a positive integer.")
    if args.stride is not None and args.stride <= 0:
        parser.error("--stride must be a positive integer.")
    if args.use_wandb and args.wandb_mode == "disabled":
        parser.error("--use-wandb cannot be combined with --wandb-mode "
                     "disabled.")

    benchmark(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            epochs=args.epochs,
            lr=args.lr,
            models=args.models,
            no_cuda=args.no_cuda
            output_dir=args.output_dir,
            patience=args.patience,
            seed=args.seed,
            stride=args.stride,
            use_wandb=args.use_wandb,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_run_prefix=args.wandb_run_prefix,
            weight_decay=args.weight_decay,
            window_size=args.window_size,
        )
