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

    logging.info("\tDONE.")
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
        splits[parts[1]].append(parts[0])
    
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

    logging.info("\tDONE.")
    return assignments

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
    logging.info("\tDONE.")

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
    logging.info("\tDONE.")

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
    logging.info("\tDONE.")

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

def load_model_results(input_dir: str) -> dict:
    """
    load_model_results(input_dir) -> results:
    """
    # Check that directory has proper stucture.
    splits_dir, models_dirs = _check_and_get_input_dirs(input_dir)

    # Locate and load all assignment breakdowns.
    assignments = _load_assignments(splits_dir=splits_dir)

    # Locate and load all saved model histories.
    histories = _load_model_histories(models_dirs=models_dirs)

    # Group all model histories by fold number and model name.
    pass

def plot_results(output_dir: str, results: dict, save: bool):
    """
    plot_results(output_dir, results, save)
    """
    pass

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
    results = load_model_results(input_dir=input_dir)

    # Generate plots.
    plot_results(
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
