"""
benchmark.py

Author:
    nubby
    Taylor Kergan

Date:
    21 Jul 2026

Version:
    1.2.1
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import logging
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

import wandb

from data_pipeline import build_data_bundles, FullDataBundle
from models import LSTMEstimator, TemporalConvNetEstimator, TransformerEstimator

# Library configs.
matplotlib.use("Agg")
logger = logging.getLogger("greenthumb")


# Useful macros/global variables.
spr_frac_e_max = 0.11   # Fractional percentage of SPR for success.
sbd_rmse_max = 0.03     # RMSE prediction threshold for success.

# Dataclasses.
@dataclass
class EpochStats:
    """
    EpochStats

    Statistics about the performance of a split for a given target. Unless
    otherwise indicated, these statistics refer to "true" targets, as opposed
    to normalized targets.

    Args:
        ...
        n_preds (int)   Number of predictions made (per timestep).
        ...
    """
    accuracy_mean: float
    loss_mean: float
    mse: float
    n_examples: int
    percent_e_mean: float
    percent_e_std: float
    rmse: float
    target: str
    accuracy_std: float = 0.0
    loss_std: float = 0.0

@dataclass
class TrainingConfig:
    """
    TrainingConfig

    Args:
        ...
        train_mean  (float) Initialize to 0, but must be set elsewhere.
        train_std   (float) Initialize to 0, but must be set elsewhere.
        ...
    """
    no_cuda: bool
    num_steps: int
    target: str
    batch_size: int = 32 
    classic_mode: bool = False
    epochs: int = 30
    grad_clip: float = 1.0
    loocv: bool = False
    lr: float = 3e-4
    patience: int = 30
    train_mean: float = 0.0
    train_std: float = 0.0
    weight_decay: float = 1e-3

@dataclass
class TrainingHistory:
    """
    TrainingHistory
    """
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

@dataclass
class WandbConfig:
    """
    WandbConfig
    """
    use_wandb: bool
    wandb_entity: str
    wandb_group: str
    wandb_mode: str
    wandb_project: str
    wandb_run_prefix: str


# Functional stuff.
def set_seed(seed: int):
    """
    set_seed(seed)

    Random seed setting for data pipeline.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_up(
        no_cuda: bool,
        output_dir: Path,
        seed: int,
    ) -> torch.device:
    """
    set_up(no_cuda, output_dir, seed) -> device
    """
    # Select device for training.
    device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        )
    logger.info(f"Using device: {device}")

    # Set random seed.
    set_seed(seed)

    # Set up output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    return device

def save_trial_splits(
        assignment: Dict,
        idx: int,
        output_dir: Path,
        seed: int,
        training_config: TrainingConfig
    ):
    """
    save_trial_splits(data_bundle, idx, output_dir, training_config)

    Save splits from a given seed.
    """
    # First, create a "splits" directory if it does not already exist.
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Extract information to create a full path.
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    path = splits_dir / ("-".join([
        ts,
        str(seed),
        "splits",
        str(idx)
    ]) + ".csv")

    # Next, add a header line.
    header = ",".join([
        "Trial Label",
        "Number of Steps Used",
        "Split"
    ]) + "\n"
    with open(path, "a+") as sp:
        sp.write(header)

    # Last, write which dataset was used in which split.
    for key in assignment.keys():
        line = f"{key[2]},{assignment[key]}\n"
        with open(path, "a+") as sp:
            sp.write(line)

def run_epoch(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        target: str,
        train_mean: Tuple[float],
        train_std: Tuple[float],
        *,
        train: bool,
        optimizer: torch.optim.Optimizer | None = None,
        grad_clip: float = 1.0,
        history: Optional[TrainingHistory] = None,
        split: str = "train",
        global_step: int = 0,
        wandb_run: Optional[object] = None,
        wandb_prefix: str = "",
    ) -> Tuple[EpochStats, int]:
    """
    run_epoch(...) -> (epoch_stats, idx_next_step)

    Runs a single training or evaluation epoch and logs batch metrics if
    requested.

    Args:
        ...
        target      (str)           [spr, sbd]
        train_mean  (Tuple[float])  Mean values for each layer for success
                                    criteria evaluation.
        train_std   (Tuple[float])  STD of values for each layer for success
                                    criteria evaluation.
        ...
    """
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
        # NOTE: Derivation of success criteria below:
        #
        #   For SPR targets, success is within 11% of real value:
        #
        #   z = (x-m)/std -> x  = std*z +m
        #                    x^ = std*z^+m
        #   * For success: abs((x-x^)/x) <= 0.11, where
        #       in which z = normalized label, z^ = predicted value,
        #       x = true label, and x^ = predicted label.
        #
        #   For SBD targets, success is <= RMSE = 0.12g/cm^3:
        #
        #   * For success: sqrt(mean(y-x)^2) <= 0.12
        #
        #   Percent error and RMSE mean/STD are reported as well.

        total_loss += loss.item() * labels.size(0)
        preds = logits

        # Recover targets and predictions from z-scores for use in evals.
        # NOTE: Use only the first soil layer for predictions (currently).
        z_hat = preds[:,0]
        z = labels[:,0]
        y = train_std * z_hat + train_mean
        x = train_std * z + train_mean

        # Stats for nerds. Also evals.
        errors_abs = abs(y - x)
        errors_perc = errors_abs / x
        errors_perc_mean = torch.mean(errors_perc)
        errors_perc_std = torch.std(errors_perc)
        errors_mse = torch.mean((y - x)**2)
        errors_rmse = torch.sqrt(errors_mse)

        if (target == "spr"):
            # SPR score.
            total_correct += (errors_perc <= spr_frac_e_max).sum().item()
        else:
            # SBD score.
            total_correct += len(logits) if errors_rmse <= sbd_rmse_max else 0

        total_examples += labels.size(0)
        next_step = step_id + 1

    avg_loss = total_loss / max(1, total_examples)
    avg_acc = total_correct / max(1, total_examples)

    # Report statistics from epoch.
    epoch_stats = EpochStats(
            accuracy_mean=avg_acc,
            loss_mean=avg_loss,
            n_examples=total_examples,
            mse=errors_mse.item(),
            percent_e_mean=errors_perc_mean.item(),
            percent_e_std=errors_perc_std.item(),
            rmse=errors_rmse.item(),
            target=target,
        )

    return epoch_stats, next_step


def train_and_evaluate(
        model: nn.Module,
        data_bundle: FullDataBundle,
        device: torch.device,
        loocv: bool,
        training_config: TrainingConfig,
        *,
        model_name: str,
        target: str,
        wandb_run: Optional[object] = None,
    ) -> Tuple[Dict[str, float], TrainingHistory]:
    """
    train_and_evaluate(...) -> {
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "best_epoch": best_epoch,
        },
        history
    """
    train_loader = data_bundle.dataloader(
            batch_size=training_config.batch_size,
            split="train",
        )
    if not loocv:
        val_loader = data_bundle.dataloader(
                batch_size=training_config.batch_size,
                split="val",
            )
    else:
        val_loader = None
    test_loader = data_bundle.dataloader(
            batch_size=training_config.batch_size,
            split="test",
        )

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.lr,
            weight_decay=training_config.weight_decay
        )
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
    best_val_rmse = 100000.0            # Highest possible (and then some).
    best_val_percent_error = 100.0      # Highest possible.
    best_epoch = 0
    epochs_no_improve = 0
    global_step = 0

    if wandb_run is not None:
        wandb_run.watch(model, log="gradients", log_freq=200)

    for epoch in range(1, training_config.epochs + 1):
        train_stats, global_step = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            target=target,
            train_mean=training_config.train_mean,
            train_std=training_config.train_std,
            train=True,
            optimizer=optimizer,
            grad_clip=training_config.grad_clip,
            history=history,
            split="train",
            global_step=global_step,
            wandb_run=wandb_run,
            wandb_prefix=model_name,
        )
        # Unzip training stats.
        (
            train_loss,
            train_acc,
            train_rmse,
            train_perc_e_mean,
            train_perc_e_std
        ) = (
                train_stats.loss_mean,
                train_stats.accuracy_mean,
                train_stats.rmse,
                train_stats.percent_e_mean,
                train_stats.percent_e_std
            )
        
        # Validate model.
        if not loocv:
            val_stats, _ = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                target=target,
                train_mean=training_config.train_mean,
                train_std=training_config.train_std,
                train=False,
                split="val",
            )
            # Unzip validation test stats.
            val_loss, val_acc, val_rmse, val_perc_e_mean, val_perc_e_std = (
                    val_stats.loss_mean,
                    val_stats.accuracy_mean,
                    val_stats.rmse,
                    val_stats.percent_e_mean,
                    val_stats.percent_e_std
                )
            # TODO(nubby)
            scheduler.step(val_acc)

        # Evaluate model performance outside of loss.
        # TODO(nubby):  Chat with some friends about minimizing loss versus
        #               these weird evaluations.
        if not loocv:
            if (val_perc_e_mean < best_val_percent_error):
                # Evaluation for SPR (drop "mean" for ease of use).
                best_val_percent_error = val_perc_e_mean
                if not training_config.classic_mode:
                    if (target == "spr"):
                        best_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if (val_rmse < best_val_rmse):
                # Evaluation for SBD.
                best_val_rmse = val_rmse
                if not training_config.classic_mode:
                    if (target == "sbd"):
                        best_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if (val_acc > best_val_acc):
                # Evaluation for classic mode.
                best_val_acc = val_acc
                if training_config.classic_mode:
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        else:
            if (train_perc_e_mean < best_val_percent_error):
                # Evaluation for SPR (drop "mean" for ease of use).
                best_val_percent_error = train_perc_e_mean
                if not training_config.classic_mode:
                    if (target == "spr"):
                        best_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if (train_rmse < best_val_rmse):
                # Evaluation for SBD.
                best_val_rmse = train_rmse
                if not training_config.classic_mode:
                    if (target == "sbd"):
                        best_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if (train_acc > best_val_acc):
                # Evaluation for classic mode.
                best_val_acc = train_acc
                if training_config.classic_mode:
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        """
        if (val_acc > best_val_acc):
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        """

        history.epoch_indices.append(epoch)
        history.train_epoch_loss.append(train_loss)
        history.train_epoch_acc.append(train_acc)
        if not loocv:
            history.val_epoch_loss.append(val_loss)
            history.val_epoch_acc.append(val_acc)
        else:
            history.val_epoch_loss.append(train_loss)
            history.val_epoch_acc.append(train_acc)
            # For logging.
            val_loss = train_loss
            val_acc = train_acc
            val_rmse = train_rmse
            val_perc_e_mean = train_perc_e_mean
            val_perc_e_std = train_perc_e_std
        history.learning_rates.append(optimizer.param_groups[0]["lr"])

        logger.info(
            f"Epoch {epoch:02d} (Train)\t"
            f"| train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} "
            f"train_rmse={train_rmse:.4f} "
            f"train_perc_e_mean={train_perc_e_mean:.3f} "
            f"train_perc_e_std={train_perc_e_std:.3f}\n"
            f"Epoch {epoch:02d} (Val)\t\t"
            f"| val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} "
            f"val_rmse={val_rmse:.4f} "
            f"val_perc_e_mean={val_perc_e_mean:.3f} "
            f"val_perc_e_std={val_perc_e_std:.3f} "
            f"best_val_rmse={best_val_rmse:.3f} "
            f"best_val_percent_error={best_val_percent_error:.3f}"
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

        if epochs_no_improve >= training_config.patience:
            logger.info("Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    history.best_epoch = best_epoch

    test_stats, _ = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            target=target,
            train_mean=training_config.train_mean,
            train_std=training_config.train_std,
            train=False,
            split="test",
        )
    # Unzip test statistics.
    test_loss, test_acc, test_rmse, test_perc_e_mean, test_perc_e_std = (
            test_stats.loss_mean,
            test_stats.accuracy_mean,
            test_stats.rmse,
            test_stats.percent_e_mean,
            test_stats.percent_e_std
        )
    # TODO(nubby):  Expand history tracking of other metrics.
    history.test_loss = test_loss
    history.test_acc = test_acc

    if wandb_run is not None:
        wandb_run.log(
            {
                f"{model_name}/best_val_acc": best_val_acc,
                f"{model_name}/test_acc": test_acc,
                f"{model_name}/test_loss": test_loss,
                f"{model_name}/test_rmse": test_rmse,
                f"{model_name}/test_perc_e_mean": test_perc_e_mean,
                f"{model_name}/test_perc_e_std": test_perc_e_std,
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
            "test_rmse": test_rmse,
            "test_perc_e_mean": test_perc_e_mean,
            "test_perc_e_std": test_perc_e_std
        },
        history,
    )


def save_history(
        history: TrainingHistory,
        output_dir: Path,
        model_name: str,
        label: str
    ) -> Path:
    """
    save_history(history, output_dir, model_name) -> history_path

    Serialises training history to JSON for downstream analysis.
    """
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    history_path = model_dir / f"{label}_history.json"
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


def save_plots(
        history: TrainingHistory,
        output_dir: Path,
        model_name: str,
        label: str
    ):
    """
    save_plots(history, output_dir, model_name)

    Generates publication-ready loss/accuracy plots.
    """
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if history.epoch_indices:
        with plt.style.context("seaborn-v0_8"):
            fig, axes = plt.subplots(1, 1, figsize=(12, 4))

            axes.plot(history.epoch_indices, history.train_epoch_loss, marker="o", label="Train")
            #axes.plot(history.epoch_indices, history.val_epoch_loss, marker="o", label="Validation")
            axes.set_xlabel("Epoch")
            axes.set_ylabel("RMSE Loss")
            axes.set_title(f"{model_name.upper()} Loss")
            axes.grid(True, which="both", linestyle="--", linewidth=0.5)
            axes.legend()

            """
            axes[1].plot(history.epoch_indices, history.train_epoch_acc, marker="o", label="Train")
            #axes[1].plot(history.epoch_indices, history.val_epoch_acc, marker="o", label="Validation")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title(f"{model_name.upper()} Accuracy")
            axes[1].set_ylim(0.0, 1.05)
            axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)
            axes[1].legend()
            """

            fig.tight_layout()
            fig.savefig(model_dir / f"{label}_{model_name}_epoch_metrics.png", dpi=300, bbox_inches="tight")
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
            fig.savefig(model_dir / f"{label}_{model_name}_train_batch_loss.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

def _configure_wandb(
        dataset_summary: dict,
        name: str,
        param_count: int,
        training_config: TrainingConfig,
        wandb_config: WandbConfig
    ):
    """
    _configure_wandb(...)
    """
    wandb_run = None
    if wandb_config.use_wandb and wandb_config.wandb_mode != "disabled":
        if wandb is None:
            # Check if imported wandb lib is found.
            raise RuntimeError(
                "Weights & Biases is not installed. Install it with "
                "'pip install wandb' to enable logging."
            )
        run_name = (f"{wandb_config.wandb_run_prefix}-"
                    f"{name}-"
                    f"seed{wandb_config.seed}")
        run_config = {
            **dataset_summary,
            "epochs": training_config.epochs,
            "batch_size": training_config.batch_size,
            "lr": training_config.lr,
            "weight_decay": training_config.weight_decay,
            "patience": training_config.patience,
            "model": name,
            "params": param_count,
        }
        wandb_run = wandb.init(  # type: ignore[call-arg]
            project=wandb_config.wandb_project,
            entity=wandb_config.wandb_entity,
            group=wandb_config.wandb_group,
            name=run_name,
            config=run_config,
            mode=wandb_config.wandb_mode,
            reinit=True,
        )

    return wandb_run

def _print_model_results(model_results: Dict[str, float], name: str, seed: int):
    """
    _print_model_results(model_results)
    """
    logger.info(
            f"{name.upper()}, seed={seed}: \n"
            f"best_val_acc={model_results['best_val_acc']:.3f} "
            f"test_acc={model_results['test_acc']:.3f} "
            f"test_loss={model_results['test_loss']:.4f} "
            f"test_rmse={model_results['test_rmse']:.3f} "
            f"test_perc_error_mean={model_results['test_perc_e_mean']:.4f} "
            f"test_perc_error_std={model_results['test_perc_e_std']:.4f} "
            f"(best_epoch={model_results['best_epoch']})"
        )


def _save_model_results(
        idx: int,
        model_results: Dict[str, float],
        name: str,
        output_dir: Path,
        seed: int,
        training_config: TrainingConfig
    ) -> str:
    """
    _save_model_results(
        model_results, name, output_dir, training_config) -> label
    """
    logger.info(
            f"Saving results for {name} (seed={seed}) to {output_dir}."
        )
    path = output_dir / "model_results.csv"
    now = datetime.now(timezone.utc)

    # If the results file and/or directory do not yet exist, add/create them.
    if not output_dir.is_dir():
        os.makedirs(output_dir)

    if not path.exists():
        line = ",".join([
                "Timestamp",
                "ModelName",
                "Seed",
                "TrainingIndex",
                "BestValidationAccuracy",
                "TestAccuracy",
                "TestLoss",
                "TestRMSE",
                "TestPercentErrorMean",
                "TestPercentErrorSTD",
                "BestEpochNumber",
                "Target",
                "BatchSize",
                "UseClassicMode",
                "NumEpochs",
                "GradClip",
                "LearningRate",
                "Patience",
                "TrainingMean",
                "TrainingSTD",
                "WeightDecay"
            ]) + "\n"
        with open(path, "a+") as rp:
            rp.write(line)

    # Generate the timestamp and label.
    ts = now.strftime("%Y%m%d_%H%M%S")
    classic = "classic" if training_config.classic_mode else "alt"
    label = "-".join([
        ts,
        name,
        str(seed),
        str(idx),
        training_config.target,
        classic,
        str(training_config.batch_size),
        str(training_config.epochs),
        str(training_config.grad_clip),
        str(training_config.lr),
        str(training_config.patience),
        str(training_config.weight_decay)
    ])

    # Write a line.
    line = ",".join([
            ts,
            name,
            str(seed),
            str(idx),
            str(model_results["best_val_acc"]),
            str(model_results["test_acc"]),
            str(model_results["test_loss"]),
            str(model_results["test_rmse"]),
            str(model_results["test_perc_e_mean"]),
            str(model_results["test_perc_e_std"]),
            str(model_results["best_epoch"]),
            training_config.target,
            str(training_config.batch_size),
            str(training_config.classic_mode),
            str(training_config.epochs),
            str(training_config.grad_clip),
            str(training_config.lr),
            str(training_config.patience),
            str(training_config.train_mean),
            str(training_config.train_std),
            str(training_config.weight_decay)
        ]) + "\n"
    with open(path, "a+") as rp:
        rp.write(line)

    return label

def _report_model_results(
        debug_mode: bool,
        idx: int,
        model_history: TrainingHistory,
        model_results: Dict[str, float],
        name: str,
        output_dir: Path,
        seed: int,
        training_config: TrainingConfig
    ):
    """
    _report_model_results(results)

    Print and save model outputs.
    """
    _print_model_results(model_results=model_results, name=name, seed=seed)
    if not debug_mode:
        label = _save_model_results(
                idx=idx,
                model_results=model_results,
                output_dir=output_dir,
                name=name,
                seed=seed,
                training_config=training_config
            )
        # TODO:
        save_history(
                history=model_history,
                output_dir=output_dir,
                model_name=name,
                label=label
            )
        save_plots(
                history=model_history,
                output_dir=output_dir,
                model_name=name,
                label=label
            )

def run_benchmark_model(
        data_bundle: FullDataBundle,
        dataset_summary: dict,
        debug_mode: bool,
        device: torch.device,
        histories: Dict[str, TrainingHistory],
        idx: int,
        loocv: bool,
        model: nn.Module,
        name: str,
        output_dir: Path,
        results: Dict[str, Dict[str, float]],
        seed: int,
        training_config: TrainingConfig,
        wandb_config: WandbConfig
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, TrainingHistory]]:
    """
    run_benchmark_model()

    Begin benchmark evaluations for a given model.
    """
    # Gather model parameters.
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layers = list(model.modules())
    num_layers = len(layers)

    logger.info(f"\n=== Training {name.upper()} "
                f"({param_count} params, "
                f"{num_layers} layers) ===")

    # Move model to device buffer.
    model.to(device)

    # Configure weights and biases.
    wandb_run = _configure_wandb(
            dataset_summary=dataset_summary,
            name=name,
            param_count=param_count,
            training_config=training_config,
            wandb_config=wandb_config
        )

    # Run the prepped model through the pipeline.
    outcomes, history = train_and_evaluate(
            model=model,
            data_bundle=data_bundle,
            device=device,
            loocv=loocv,
            model_name=name,
            training_config=training_config,
            target=training_config.target,
            wandb_run=wandb_run,
        )

    # Save and report results.
    results[name] = outcomes
    histories[name] = history
    _report_model_results(
            debug_mode=debug_mode,
            idx=idx,
            model_history=history,
            model_results=outcomes,
            name=name,
            output_dir=output_dir,
            seed=seed,
            training_config=training_config
        )

    if wandb_run is not None:
        wandb_run.finish()
    
    return results, histories

def _print_benchmark_results(
        idx: int,
        n: int,
        results: Dict[str, Dict[str, float]],
        seed: int
    ):
    """
    _print_benchmark_results(results, seed)
    """
    logger.info(f"\n=== Summary (seed: {seed}, fold {idx}/{n}) ===")
    for name, metrics in results.items():
        logger.info(
            f"{name.upper():12s} | val_acc={metrics['best_val_acc']:.3f} "
            f"| test_acc={metrics['test_acc']:.3f} "
            f"| best_epoch={metrics['best_epoch']} "
            f"| test_rmse={metrics['test_rmse']:.3f} "
            f"| test_perc_e_mean={metrics['test_perc_e_mean']:.3f} "
            f"| test_perc_e_std={metrics['test_perc_e_std']:.3f}"
        )

def analyze_benchmark_results(
        output_dir: Path,
        results: List[Dict[str, Dict[str, float]]],
        seed: int
    ):
    # To save output.
    path = output_dir / "results.csv"

    n = len(results)

    # First gather RMSE from each fold.
    rmses = {}
    for result in results:
        for model, metrics in result.items():
            # Add new array if not already added.
            if model not in rmses.keys():
                rmses[model] = []
            rmses[model].append(metrics["test_rmse"])

    # Now get statistics for each model.
    for model, vals in rmses.items():
        print(f"\tRESULTS: {model} - {vals}")
        rmse_avg = np.mean(vals)
        rmse_median = np.median(vals)
        rmse_std = np.std(vals)
        logger.info(
            f"{str(seed)} | {model}"
            f"| n={n} "
            f"| rmse_avg={rmse_avg:.4f} "
            f"| rmse_median={rmse_median:.4f} "
            f"| rmse_std={rmse_std:.4f} "
        )

        # Next, add a header line.
        if not path.exists():
            header = ",".join([
                "ModelName",
                "Seed",
                "N",
                "RMSE (mean)",
                "RMSE (median)",
                "RMSE (std)"
            ]) + "\n"
            with open(path, "a+") as sp:
                sp.write(header)

        # Last, write results.
        line = ",".join([
            f"{model}",
            f"{str(seed)}",
            f"{n}",
            f"{rmse_avg:.4f}",
            f"{rmse_median:.4f}",
            f"{rmse_std:.4f}" + "\n"
        ])
        with open(path, "a+") as rp:
            rp.write(line)


def run_benchmark(
        data_dir: str,
        debug_mode: bool,
        loocv: bool,
        models: list[str],
        num_steps: int,
        output_dir: Path,
        seed: int,
        stride: int,
        target: str,
        training_config: TrainingConfig,
        wandb_config: WandbConfig,
        wet: bool,
        window_size: int,
    ):
    """
    run_benchmark(...)

    Perform a single benchmark evaluation.
    """
    # Set up device for training with a different seed for each eval.
    device = set_up(
            no_cuda=training_config.no_cuda,
            output_dir=output_dir,
            seed=seed,
        )

    # Create data bundle for train/val/test splits for each trial (with
    # different random seed for thoroughness).
    # TODO(nubby):  Further divide individual trials, either by a fixed size or
    #               by steps.
    if loocv:
        val_frac = 0.0
        n_scenes = 1000 # Update with trial bundling.
    else:
        val_frac = 0.2
        n_scenes = 1

    data_bundles, split_assignments = build_data_bundles(
            data_dir=data_dir,
            num_steps=num_steps,
            seed=seed,
            stride=stride,
            target=target,
            val_frac=val_frac,
            wet=wet,
            window_size=window_size
        )

    # Initialize bulk results/histories here, then pass with evals.
    results: List[Dict[str, Dict[str, float]]] = []
    for idx, bundle in enumerate(data_bundles):
        # Extract mean/STD from training set for later use.
        training_config.train_mean = bundle.train_mean
        training_config.train_std = bundle.train_std

        input_dim = len(bundle.feature_names)

        # Number of soil layers to predict (from the top layer down).
        soil_layers = 1

        # Set up models.
        available_models = {
            "lstm": LSTMEstimator(input_dim=input_dim, num_targets=soil_layers),
            "tcn": TemporalConvNetEstimator(
                input_dim=input_dim,
                num_targets=soil_layers
            ),
            "transformer": TransformerEstimator(
                input_dim=input_dim,
                num_targets=soil_layers,
                max_chunk_len=window_size
            )
        }

        # Verify model selection is valid, then load them.
        unknown = set(models) - set(available_models.keys())
        if unknown:
            raise ValueError(
                f"Unknown model names requested: {', '.join(sorted(unknown))}"
            )
        models = {name: available_models[name] for name in models}

        # Summarize dataset.
        dataset_summary = {
            "train_samples": len(bundle.train),
            "val_samples": len(bundle.val) if not loocv else 0,
            "test_samples": len(bundle.test),
            "feature_dim": input_dim,
            "window_size": window_size,
            "stride": stride,
        }

        logger.info(
            "Dataset sequences -> "
            f"train: {dataset_summary['train_samples']}, "
            f"val: {dataset_summary['val_samples']}, "
            f"test: {dataset_summary['test_samples']}, "
            f"feature_dim: {dataset_summary['feature_dim']}, "
            f"window_size: {dataset_summary['window_size']}, "
            f"stride: {dataset_summary['stride']}\n\n"
        )

        # Save trial splits.
        save_trial_splits(
                assignment=split_assignments[idx],
                idx=idx,
                output_dir=output_dir,
                seed=seed,
                training_config=training_config
            )

        # Run each model through a round of the benchmark.
        for name, model in models.items():
            trial_results: Dict[str, Dict[str, float]] = {}
            histories: Dict[str, TrainingHistory] = {}

            trial_results, histories = run_benchmark_model(
                    data_bundle=bundle,
                    dataset_summary=dataset_summary,
                    debug_mode=debug_mode,
                    device=device,
                    histories=histories,
                    idx=idx,
                    loocv=loocv,
                    model=model,
                    name=name,
                    output_dir=output_dir,
                    results=trial_results,
                    seed=seed,
                    training_config=training_config,
                    wandb_config=wandb_config
                )

            # Report benchmark results for each seed.
            _print_benchmark_results(
                    idx=idx+1,
                    n=len(data_bundles),
                    results=trial_results,
                    seed=seed
                )

            # Add results here for final evaluation.
            results.append(trial_results)

    # Analyze results here.
    analyze_benchmark_results(output_dir=output_dir, results=results, seed=seed)


def _report_results_summary(
        output_dir: Path
    ):
    """
    _report_results_summary(output_dir)

    Summarize and print results/stats for each model's performance (as
    specified by user).
    """
    pass

def benchmark(
        batch_size: int,
        classic_mode: bool,
        data_dir: str,
        debug_mode: bool,
        epochs: int,
        loocv: bool,
        lr: float,
        models: list[str],
        n_evals: int,
        no_cuda: bool,
        num_steps: int,
        output_dir: Path,
        patience: float,
        seed: int,
        stride: int,
        target: str,
        use_wandb: bool,
        wandb_entity: str,
        wandb_group: str,
        wandb_mode: str,
        wandb_project: str,
        wandb_run_prefix: str,
        weight_decay: float,
        wet: bool,
        window_size: int,
        verbose: bool = False
    ):
    """
    benchmark(...)

    Main training pipeline entry point. Perform benchmark evaluations with
    different seeds to generate useful run statistics.
    """
    # Create configs that do not change between runs.
    wandb_config = WandbConfig(
            use_wandb=use_wandb,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            wandb_mode=wandb_mode,
            wandb_project=wandb_project,
            wandb_run_prefix=wandb_run_prefix
        )
    training_config = TrainingConfig(
            batch_size=batch_size,
            classic_mode=classic_mode,
            epochs=epochs,
            loocv=loocv,
            lr=lr,
            no_cuda=no_cuda,
            num_steps=num_steps,
            patience=patience,
            target=target,
            weight_decay=weight_decay
        )

    for i in range(n_evals):
        # Each benchmark iteration, only change the seed programmatically.
        run_benchmark(
            data_dir=data_dir,
            debug_mode=debug_mode,
            loocv=loocv,
            models=models,
            num_steps=num_steps,
            output_dir=output_dir,
            seed=seed+i,
            stride=stride,
            target=target,
            training_config=training_config,
            wandb_config=wandb_config,
            wet=wet,
            window_size=window_size,
        )

    logger.info(
            (f"Completed {n_evals} tests. Goodbye.") if not verbose else 
            (f"Bye nub.")
        )


if __name__ == "__main__":
    # Logger setup.
    logging.basicConfig(level=logging.INFO)

    # Arg parsing.
    parser = argparse.ArgumentParser(
            description="Benchmark sequence models on DyRET terrain dataset."
        )
    parser.add_argument(
            "--data-dir",
            type=str,
            default="processed",
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
            default=2000,
            help=("Optional sliding window size (timesteps) applied within each"
                  " sequence/chunk/step used.")
        )
    parser.add_argument(
            "--stride",
            type=int,
            default=None,
            help=("Stride (timesteps) between sliding windows; defaults to "
                  "window size for non-overlapping segments.")
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
    # TODO(nubby): Allow for dual prediction?
    parser.add_argument(
            "--target",
            type=str,
            default="spr",
            help="Target choice for training/evals [spr, sbd].",
        )
    parser.add_argument(
            "--classic-mode",
            action="store_true",
            help=("Evaluate model performance based on number of correct "
                  "predictions made, versus RMSE (for SBD) and percent error "
                  "mean (for SPR) in standard mode.")
        )
    parser.add_argument(
            "-n",
            type=int,
            default=1,
            help=("Number of benchmark iterations to run.")
        )
    parser.add_argument(
            "--debug",
            action="store_true",
            help=("Enable debug mode.")
        )
    parser.add_argument(
            "--num-steps",
            type=int,
            default=-1,
            help=("Number of steps per trial; 0 uses a sliding window, -1 uses "
                  "all available steps found.")
        )
    parser.add_argument(
            "--wet",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=("Use VWC as a feature in training.")
        )
    parser.add_argument(
            "--loocv",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=("Use leave-one-out cross validation.")
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
    if args.target not in ["spr", "sbd"]:
        parser.error("--target must be a either spr or sbd.")

    benchmark(
            batch_size=args.batch_size,
            classic_mode=args.classic_mode,
            data_dir=args.data_dir,
            debug_mode=args.debug,
            epochs=args.epochs,
            loocv=args.loocv,
            lr=args.lr,
            models=args.models,
            n_evals=args.n,
            no_cuda=args.no_cuda,
            num_steps=args.num_steps,
            output_dir=args.output_dir,
            patience=args.patience,
            seed=args.seed,
            stride=args.stride,
            target=args.target,
            use_wandb=args.use_wandb,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_run_prefix=args.wandb_run_prefix,
            weight_decay=args.weight_decay,
            wet=args.wet,
            window_size=args.window_size,
        )
