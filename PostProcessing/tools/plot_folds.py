"""
File:
    plot_folds.py

Description:
    Generate variance plots for training loss and test accuracy from saved
    history JSON files.

Author:
    nubby

Date:
    02 Sep 2026

Version:
    0.1.1
"""
import argparse
import csv
import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys

from dataclasses import dataclass
from datetime import datetime
from io import StringIO

logger = logging.getLogger("greenthumb_plotter")


# Global LUT for test label and experimental conditions.
#   (COMP_IDX, VWC_IDX)
LUT_EXPERIMENTS = {
        "20260518_145400-inPlace": (0,0),
        "20260518_145400-walk": (0,0),
        "20260518_145400-inPlaceTry": (0,0), 
        "20260518_170400-walk": (1,0),
        "20260518_170400-inPlaceTry": (1,0), 
        "20260518_170400-inPlace": (1,0),
        "20260520_135800-walk1": (2,0),
        "20260520_135800-walk0": (2,0),
        "20260520_135800-inPlace": (2,0),
        "20260529_160600-walk": (0,1),
        "20260529_160600-inPlaceManual1": (0,1),
        "20260529_160600-inPlace": (0,1),
        "20260529_173500-inPlaceManual": (1,1),
        "20260529_173500-walk": (1,1),
        "20260529_173500-inPlace": (1,1)
    }


def _check_and_get_input_dirs(
        input_dir: str) -> tuple[str, tuple[str, str, str]]:
    """
    _check_and_get_input_dirs(input_dir) -> (splits_dir, models_dirs)
    """
    logging.info(f"Verifying directory structure for {input_dir}...")
    if os.path.isdir(input_dir):
        splits_dir = os.path.join(input_dir, "splits")
        lstm_dir = os.path.join(input_dir, "lstm")
        tcn_dir = os.path.join(input_dir, "tcn")
        trans_dir = os.path.join(input_dir, "transformer")

    # Check that all required directories exist.
    for d in (splits_dir, lstm_dir, tcn_dir, trans_dir):
        if not os.path.isdir(d):
            logging.error(f"{d} does not exist; aborting")
            exit(1)

    logging.info("\tDONE.\n")
    return (splits_dir, (lstm_dir, tcn_dir, trans_dir))

def _load_splits_file(path: str) -> dict:
    """
    _load_splits_file(path) -> splits

    Returns:
        splits  (dict)
            {
                "train": [DATASET_LABELS],
                "test": [DATASET_LABELS]
            }
    """
    # TODO: Fix the bug in "benchmark.py" which adds an extra column, making
    #       this not a valid CSV file.
    splits = {
            "train": [],
            "test": []
        }
    
    # Load all lines.
    lines = []
    with open(path, "r+") as sfp:
        for line in sfp:
            lines.append(line.strip())

    # Remove the header.
    lines = lines[1:]

    # Sort each line into splits by CSV-enforced structure.
    for line in lines:
        parts = line.split(",")
        if len(parts) != 2:
            logging.warning(f"{path} is not a valid split file; skipping...")
            return None
        condition = LUT_EXPERIMENTS[parts[0]]
        if condition not in splits[parts[1]]:
            splits[parts[1]].append(condition)
    
    return splits

def _load_assignments(splits_dir: str) -> dict:
    """
    _load_assignments(splits_dir) -> assignments

    Returns:
        assignments (dict)  Structure of
            {
                <FOLD_#>: {
                    "train": [DATASET_LABELS],
                    "test": [DATASET_LABELS]
                },
                ...
            }
    """
    logging.info(f"Loading splits for each fold from {splits_dir}...")
    assignments = {}

    # Get all splits files.
    fns_splits = [
            f for f in os.listdir(splits_dir) if (
                os.path.isfile(os.path.join(splits_dir, f))
            )
        ]
    for fn in fns_splits:
        path = os.path.join(splits_dir, fn)

        # Digest each file into its fold and splits.
        parts = fn.split("-")
        # Enforce file structure.
        if len(parts) != 4:
            logging.warning(f"Splits file {path} invalid; skipping...")
            continue

        # 0: Timestamp - 1: Seed - 2: "splits" - 3: Fold #.csv .
        fold = int(parts[3].split(".")[0])
        # NOTE: At this point, we optimize the process if we assume all splits
        #       are identical for the same fold number.
        if fold not in assignments.keys():
            splits = _load_splits_file(path=path)
            if not splits:
                continue

            # Add splits to fold.
            assignments[fold] = splits

    logging.info("\tDONE.\n")
    return assignments

def _load_history_files(base_dir: str) -> dict:
    """
    _load_history_files(base_dir) -> model_histories

    Returns:
        {
            <FOLD_#>: [
                {
                    "epoch_indices": [...],
                    "train_epoch_loss": [...],
                    "learning_rates": [...],
                    "train_batch_loss": [...],
                    "train_batch_step": [...],
                    "best_epoch": [...],
                    "test_loss": float
                },
                ...
            ],
            ...
        }
    """
    histories = {}

    # Get all history files.
    fns_histories = [
            f for f in os.listdir(base_dir) if (
                os.path.isfile(os.path.join(base_dir, f))
            ) and (
                f.endswith(".json")
            )
        ]

    for fn in fns_histories:
        path = os.path.join(base_dir, fn)

        # Digest each file into its fold and splits.
        parts = fn.split("-")
        # Enforce file structure.
        if len(parts) != 12:
            logging.warning(f"History file {path} invalid; skipping...")
            continue

        # Load file.
        data = {}
        with open(path, "r+") as hfp:
            data = json.load(hfp)

        # Check that all required keys are found.
        required_keys = [
                "epoch_indices",
                "train_epoch_loss",
                "learning_rates",
                "train_batch_loss",
                "train_batch_step",
                "best_epoch",
                "test_loss"
            ]
        if not all(key in data.keys() for key in required_keys):
            logging.warning(f"History file {path} incomplete; skipping...")
            continue

        # 0: Timestamp - 1: Model Name - 2: Seed - 3: Fold # - ...
        fold = int(parts[3])
        if fold not in histories.keys():
            histories[fold] = []

        # Add to history.
        histories[fold].append({key: data[key] for key in required_keys})

    return histories

def _load_lstm_histories(lstm_dir: str) -> dict:
    """
    _load_lstm_histories(lstm_dir) -> folds 

    Returns:
        {
            <FOLD_#>: {
            },
            ...
        }
    """
    logging.info(f"Loading LSTM histories for each fold from {lstm_dir}...")
    histories = _load_history_files(base_dir=lstm_dir)
    logging.info("\tDONE.\n")

    return histories

def _load_tcn_histories(tcn_dir: str) -> dict:
    """
    _load_tcn_histories(tcn_dir) -> folds 

    Returns:
        {
            <FOLD_#>: {
            },
            ...
        }
    """
    logging.info(f"Loading TCN histories for each fold from {tcn_dir}...")
    histories = _load_history_files(base_dir=tcn_dir)
    logging.info("\tDONE.\n")

    return histories

def _load_trans_histories(trans_dir: str) -> dict:
    """
    _load_trans_histories(trans_dir) -> folds 

    Returns:
        {
            <FOLD_#>: {
            },
            ...
        }
    """
    logging.info(
            f"Loading Transformer histories for each fold from {trans_dir}...")
    histories = _load_history_files(base_dir=trans_dir)
    logging.info("\tDONE.\n")

    return histories

def _load_model_histories(models_dirs: tuple[str, str, str]) -> dict:
    """
    _load_model_histories(models_dirs) -> histories:

    Returns:
        {
            "lstm": {
                <FOLD_#>: {
                },
                ...
            },
            "tcn": {...},
            "transformer": {...}
        }
    """
    histories = {}
    histories["lstm"] = _load_lstm_histories(lstm_dir=models_dirs[0])
    histories["tcn"] = _load_tcn_histories(tcn_dir=models_dirs[0])
    histories["transformer"] = _load_trans_histories(trans_dir=models_dirs[0])

    return histories

def _load_results(input_dir: str) -> dict:
    """
    _load_results(input_dir) -> results

    Returns:
        {
            "lstm": {
                <FOLD_#>: {
                    "test_rmse": []
                },
                ...
            },
            "tcn": {...},
            "transformer": {...}
        }
    """
    results = {}

    # Check that "model_results.csv" file exists and is properly formatted.
    path = os.path.join(input_dir, "model_results.csv")
    if not os.path.isfile(path):
        logging.error(f"Model results not found at {path}; aborting.")
        exit(1)

    required_keys = [
            "test_rmse"
        ]
    """
    if not all(key in data.keys() for key in required_keys):
        logging.error(f"Model results file {path} incomplete; aborting.")
        exit(1)
    """

    # Load all model results.
    df = pd.read_csv(path)
    models = df["ModelName"].unique().tolist()
    folds = df["TrainingIndex"].astype(int).unique().tolist()
    for model in models:
        results[model] = {}
        for fold in folds:
            results[model][fold] = df.loc[
                    (df["ModelName"] == model) &
                    (df["TrainingIndex"].astype(int) == fold),
                    "TestRMSE"].astype(float).values

    return results

def load_model_results(input_dir: str) -> tuple[dict, dict, dict]:
    """
    load_model_results(input_dir) -> results:
    """
    # Check that directory has proper stucture.
    splits_dir, models_dirs = _check_and_get_input_dirs(input_dir=input_dir)

    # Locate and load all assignment breakdowns.
    assignments = _load_assignments(splits_dir=splits_dir)

    # Locate and load all saved model histories.
    # NOTE: This is only useful when plotting training/loss histories.
    #histories = _load_model_histories(models_dirs=models_dirs)
    histories = None

    # Locate and load saved model results.
    results = _load_results(input_dir=input_dir)

    return assignments, histories, results

def plot_results_models_comp(
        assignments: dict,
        output_dir: str,
        results: dict,
        save: bool
    ):
    """
    plot_results_models_comp(...)

    Plot each model's performance for each fold, side-by-side.
    """
    # Plot non-ablated model performance, side-by-side for each fold.
    for model, folds in results.items():
        plt.figure(figsize=(6,4))
        plt.boxplot(
                [folds[key] for key in folds.keys()],
                [assignments[fold]["test"] for fold in folds.keys()],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="red", linewidth=2)
            )
        plt.title(f"Test (model={model})", fontsize=18)
        plt.ylabel("RMSE", fontsize=14)

    if not save:
        plt.show()
    else:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        plt.savefig(os.path.join(output_dir, "ensemble.pdf"))

def plot_results_ens_boxes(
        assignments: list,
        folds: list,
        mean: float,
        models: list,
        output_dir: str,
        results_vwc: dict,
        results_no_vwc: dict,
        save: bool,
        title: str
    ):
    """
    plot_results_ens_boxes()

    ...
    """
    # Plot average model performance for each fold, side-by-side.
    ens_vwc = {}
    ens_no_vwc = {}

    # X locations for scenario groups + within-group offsets.
    x = np.arange(len(folds))
    offset = 0.20
    box_width = 0.32
    ylim = 0.14
    rmse_goal = 0.074

    # Generate ensemble data for each fold.
    for fold in folds: 
        ens_vwc[fold] = np.concatenate([
            results_vwc[model][fold] for model in models
            ])
        ens_no_vwc[fold] = np.concatenate([
            results_no_vwc[model][fold] for model in models
            ])

    # Group data from each scenario.
    fig, ax = plt.subplots()
    bp_vwc = ax.boxplot(
            [ens_vwc[fold] for fold in folds],
            boxprops=dict(facecolor="blue", color="black"),
            medianprops=dict(color="black", linewidth=2),
            patch_artist=True,
            positions = x - offset,
            widths=box_width
        )
    bp_no_vwc = ax.boxplot(
            [ens_no_vwc[fold] for fold in folds],
            boxprops=dict(facecolor="gold", color="black"),
            medianprops=dict(color="black", linewidth=2),
            patch_artist=True,
            positions = x + offset,
            widths=box_width
        )
    ax.set_xlabel("Scenario Label", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(assignments)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.set_ylim(top=ylim)
    ax.set_title(title, fontsize=16)
    plt.axhline(
            y=rmse_goal,
            color="red",
            label="Target",
            linestyle="--"
        )
    plt.text(
            x=0.01,
            y=rmse_goal+0.001,
            s="0.074",
            color="red",
            fontsize=12
        )
    plt.tight_layout()

    ax.legend(
            [bp_vwc["boxes"][0], bp_no_vwc["boxes"][0]],
            ["Standard", "VWC Ablated"],
            loc="upper left",
            frameon=True
        )
    if not save:
        plt.show()
    else:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        plt.savefig(os.path.join(output_dir, "ensemble.pdf"))

def plot_results_ensemble(
        assignments: dict,
        output_dir: str,
        results_no_vwc: dict,
        results_vwc: dict,
        save: bool,
    ):
    """
    plot_results_ensemble(...)

    Generate plots for model ensembles in wet and dry conditions.
    """
    title_wet = f"Ensemble Performance (Wet) - VWC Ablation Comparison"
    title_dry = f"Ensemble Performance (Dry) - VWC Ablation Comparison"

    # NOTE: Models and folds are the same between non-/ablated tests.
    models = list(results_vwc.keys())
    folds = list(results_vwc[models[0]].keys())
    folds_wet = [fold for fold in folds if assignments[fold]["test"][0][1] > 0]
    folds_dry = [fold for fold in folds if assignments[fold]["test"][0][1] == 0]
    assignments_wet = [assignments[fold]["test"][0] for fold in folds_wet]
    assignments_dry = [assignments[fold]["test"][0] for fold in folds_dry]

    # Get the mean of all data from all folds.
    rmse_avg = np.mean(np.concatenate([
        results_vwc[model][fold] for model in models for fold in folds
        ]))
    
    # Plot ablation study in dry settings.
    plot_results_ens_boxes(
            assignments=assignments_dry,
            folds=folds_dry,
            mean=rmse_avg,
            models=models,
            output_dir=output_dir,
            results_vwc=results_vwc,
            results_no_vwc=results_no_vwc,
            save=save,
            title=title_dry
        )

    # Plot ablation study in wet settings.
    plot_results_ens_boxes(
            assignments=assignments_wet,
            folds=folds_wet,
            mean=rmse_avg,
            models=models,
            output_dir=output_dir,
            results_vwc=results_vwc,
            results_no_vwc=results_no_vwc,
            save=save,
            title=title_wet
        )

def plot_results(
        assignments: dict,
        histories: dict,
        output_dir: str,
        results_vwc: dict,
        results_no_vwc: dict,
        save: bool,
        ensemble: bool = False
    ):
    """
    plot_results(assigments, histories, output_dir, save)
    """
    # TODO: Plot non-/ablated results side-by-side for each fold.
    plot_results_models_comp(
            assignments=assignments,
            output_dir=output_dir,
            results=results_vwc,
            save=save
        )

    # Generate plots for ensemble model comparisons in wet and dry conditions.
    plot_results_ensemble(
            assignments=assignments,
            output_dir=output_dir,
            results_no_vwc=results_no_vwc,
            results_vwc=results_vwc,
            save=save
        )

def plot_folds(
        input_dir_vwc: str,
        input_dir_no_vwc: str,
        output_dir: str = ""
    ):
    """
    plot_folds(input_dir_vwc, input_dir_no_vwc, output_dir)

    Generate variance plots based on "history.json" files contained within
    input_dir, then save those plots to output_dir (if specified).
    """
    save = True
    if not output_dir:
        # Display plots if no output is specified.
        save = False

    # Find, load, and group all model run histories.
    assignments, histories, results_vwc = load_model_results(
            input_dir=input_dir_vwc
        )
    _, _, results_no_vwc = load_model_results(input_dir=input_dir_no_vwc)

    # Generate plots.
    plot_results(
            assignments=assignments,
            ensemble=True,
            histories=histories,
            output_dir=output_dir,
            results_vwc=results_vwc,
            results_no_vwc=results_no_vwc,
            save=save,
        )

if __name__ == "__main__":
    # Logger setup.
    logging.basicConfig(level=logging.INFO)

    # Parse args for later triage.
    parser = argparse.ArgumentParser(
        description=("Generate variance plots.")
    )
    parser.add_argument(
            "--input-dir-vwc",
            type=str,
            default="artifacts-wet/",
            help="Path to input directory containing GreenThumb outputs with vwc."
        )
    parser.add_argument(
            "--input-dir-no-vwc",
            type=str,
            default="artifacts-no_wet/",
            help="Path to input directory containing GreenThumb outputs without vwc."
        )
    parser.add_argument(
            "--output-dir",
            type=str,
            default="",
            help="Path to output directory."
        )
    args = parser.parse_args()

    plot_folds(
            input_dir_vwc=args.input_dir_vwc,
            input_dir_no_vwc=args.input_dir_no_vwc,
            output_dir=args.output_dir
        )
