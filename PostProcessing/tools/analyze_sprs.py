#!/bin/python3
"""
File:
    analyze_sprs.py

Description:
    Plot all collected SPR data collected as a histogram.

Author:
    jLab
    HARE Lab

Date:
    25 Jan 2026

Version:
    0.0.9
"""
import argparse
import csv
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def _read_csv(path: str) -> pd.DataFrame:
    """Read CSV file and return a dataset."""
    return pd.read_csv(path)

def _get_penetrometer_labels(dataset: pd.DataFrame, num_layers: int) -> dict:
    """Generate labels from penetrometer data."""
    # Each layer == round(depth/3)".
    soil_layer_stats = {}
    for layer_index in range(num_layers):
        mask = dataset["sample layer"].isin([layer_index+1])
        df_filtered = dataset.loc[mask, "psi"]
        soil_layer_stats[layer_index+1] = {
            "depth_in": (layer_index+1) * 3,
            "mean": df_filtered.mean(),
            "median": df_filtered.median(),
            "std": df_filtered.std()
        }

    return soil_layer_stats

def _load_dataset_info(
        info: dict,
        base_path: str,
        file_names: list[str]
    ) -> dict:
    """Capture metadata about each file in a dataset."""
    # Regex for file names.
    pen_pattern = re.compile(r"soil_pen") 

    for name in file_names:
        # Only read CSV files.
        if not name.lower().endswith(".csv"):
            continue
        rel_path = os.path.join(base_path, name)

        # Capture only robot and recorded penetrometer data.
        pen_match = re.search(pen_pattern, rel_path)
        if pen_match:
            info["path"]["pen_csv"] = rel_path

    return info

def _load_dataset(base_path: str, file_names: list) -> dict:
    """
    Load all SPR-containing data from provided files.
    """
    # All metadata and datasets related to a given test session.
    dataset_info = {
        "label": base_path.split("-")[-1],
        "path": {
            "pen_csv": ""
        },
        "dataset": {
            "penetrometer": None
        }
    }
    
    # Get metadata about the dataset.
    dataset_info = _load_dataset_info(
        info=dataset_info,
        base_path=base_path,
        file_names=file_names
    )

    try:
        dataset_info["dataset"]["penetrometer"] = _read_csv(
                dataset_info["path"]["pen_csv"])
    except Exception as e:
        return dataset_info

    return dataset_info


def load_spr_files(path_base: str) -> list[dict]:
    """
    Import and group raw CSV SPR datasets from data collection trials.
    """
    spr_datasets = []
    assert os.path.isdir(path_base), (
        "ERROR: Data input directory does not exist!"
    )
    for base, _, files in os.walk(path_base):
        info = _load_dataset(base_path=base, file_names=files)
        # Only append info if it corresponds to a directory containing all 
        # required datasets.
        try:
            if not info["dataset"]["penetrometer"].empty:
                print(f"Adding {info['label']}!")
                spr_datasets.append(info)
        except AttributeError:
            pass

    return spr_datasets

def plot_spr(dataset_info: dict):
    pass

def plot_all_sprs(dataset_infos: list[dict], output_dir: str):
    """
    Plot histograms of SPRs at each measured layer, with sand and grass
    separated. Heavy assistance from Perplexity.ai.
    """
    color_grass = "g"
    color_sand = "o"

    # Extract datasets from info.
    datasets = [info["dataset"]["penetrometer"] for info in dataset_infos]
    
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)

    # Make sure axes is an array.
    if n == 1:
        axes = [axes]
    for i, (ds, ax) in enumerate(zip(datasets, axes)):
        # Group values by label in first column.
        grouped = ds.groupby(ds.iloc[:, 0])[ds.columns[1]]

        # Ensure boxes appear in label order 1-4 (if they exist).
        labels = sorted(grouped.groups.keys())
        data = [grouped.get_group(label).values for label in labels]
        for i, stuff in enumerate(data):
            print(f"{i}: Mean - {np.mean(stuff)}; STD - {np.std(stuff)}")

        ax.boxplot(
            data,
            tick_labels=[str(int(label) * 3)+"\"" for label in labels]
        )
        n_samples = sum([len(grouped.groups[l]) for l in labels])
        print(f"{dataset_infos[i]['label']}: n = {n_samples}")
        ax.set_title(f"{dataset_infos[i]['label']} (n={n_samples})")
        ax.set_xlabel("Layer Depth")
        if i == 0:
            ax.set_ylabel("SPR (PSI)")

    fig.suptitle("Distribution of Measured SPRs: SPR Variation Study")
    #fig.suptitle("Distribution of Measured SPRs: Prelim Study")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(
        f"{output_dir}/{timestamp}-sprs.pdf",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)

def analyze_sprs(
        path_input: str,
        path_output: str
    ):
    """
    analyze_sprs(path_input, path_output)

    Generate a histogram with range bins recorded SPRs.
    """
    # Comb through each directory to find each applicable data file.
    print(f"Input Directory:\t./{path_input}/\r\n"
          f"Output Directory:\t./{path_output}/")
    combined_sprs = load_spr_files(path_base=path_input)
    plot_all_sprs(dataset_infos=combined_sprs, output_dir=path_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate informative stats on SPR files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="sprs/",
        help="Path to directory containing SPR data files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sprs/",
        help="Path to directory in which to output statistics and graphics."
    )
    args = parser.parse_args()
    analyze_sprs(path_input=args.input_dir, path_output=args.output_dir)
