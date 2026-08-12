"""
File:
    plot_folds.py

Description:
    Generate variance plots for training loss and test accuracy from saved
    history JSON files.

Author:
    nubby

Date:
    11 Aug 2026

Version:
    0.0.1
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

def plot_results(
        assignments: dict,
        histories: dict,
        output_dir: str,
        results: dict,
        save: bool
    ):
    """
    plot_results(assigments, histories, output_dir, save)
    """
    for model, folds in results.items():
        plt.figure(figsize=(6,4))
        plt.boxplot(
                [folds[key] for key in folds.keys()],
                [assignments[fold]["test"] for fold in folds.keys()],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="red", linewidth=2)
            )
        plt.title(f"Test (model={model})")
        plt.ylabel("RMSE")
        plt.show()
        for fold in folds.keys():
            split = assignments[fold]
            test_mean_fold = np.mean(folds[fold])
            test_std_fold = np.std(folds[fold])

def plot_folds(
        input_dir: str,
        output_dir: str = ""
    ):
    """
    plot_folds(input_dir, output_dir)

    Generate variance plots based on "history.json" files contained within
    input_dir, then save those plots to output_dir (if specified).
    """
    save = True
    if not output_dir:
        # Display plots if no output is specified.
        save = False

    # Find, load, and group all model run histories.
    assignments, histories, results = load_model_results(input_dir=input_dir)

    # Generate plots.
    plot_results(
            assignments=assignments,
            histories=histories,
            output_dir=output_dir,
            results=results,
            save=save
        )

if __name__ == "__main__":
    # Logger setup.
    logging.basicConfig(level=logging.INFO)

    # Parse args for later triage.
    parser = argparse.ArgumentParser(
        description=("Generate variance plots.")
    )
    parser.add_argument(
            "--input-dir",
            type=str,
            default="artifacts/",
            help="Path to input directory containing all GreenThumb outputs."
        )
    parser.add_argument(
            "--output-dir",
            type=str,
            default="",
            help="Path to output directory."
        )
    args = parser.parse_args()

    plot_folds(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
