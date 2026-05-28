"""
preprocess_b1_data.py

Author:
    nubby

Date:
    11 Dec 2025

Version:
    1.0.0
"""
import argparse
import copy as cp
import csv
import matplotlib
import numpy as np
import os
import pandas as pd
import re

# TODO: Add "eval_id" to output dataset.
# TODO: Debug strange directory naming convention.

def _read_csv(path: str) -> pd.DataFrame:
    """Read CSV file and return a dataset."""
    # NOTE: May miss data due to bad lines being skipped.
    #return pd.read_csv(path, on_bad_lines="skip")
    return pd.read_csv(path)

def _write_dataset(dataset: pd.DataFrame, path: str):
    # First create any required parent directories.
    pdir = os.path.dirname(path)
    os.makedirs(pdir, exist_ok=True)

    # Write it!
    dataset.to_csv(path, index=False)
    print(f"SUCCESS:\tWrote {path}!")

def _load_dataset_info(
        info: dict,
        base_path: str,
        file_names: list[str]
    ) -> dict:
    """Capture metadata about each file in a dataset."""
    # Regex for file names.
    robot_pattern = re.compile(r"\d{6}_\d{6}")  # We get the timestamp from
                                                # online data colletion files.
    pen_pattern = re.compile(r"soil_pen") 

    for name in file_names:
        # Only read CSV files.
        if not name.lower().endswith(".csv"):
            continue
        rel_path = os.path.join(base_path, name)

        # Capture only robot and recorded penetrometer data.
        robot_match = re.search(robot_pattern, rel_path)
        pen_match = re.search(pen_pattern, rel_path)
        if robot_match:
            info["timestamp"] = robot_match.group()
            info["path"]["robot_csv"] = rel_path
        elif pen_match:
            info["path"]["pen_csv"] = rel_path

    return info

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

# TODO(nubby): Add GPS/timestamp alignment here?
def _label_proprioceptive_dataset(info: dict) -> pd.DataFrame:
    """Add SPR labels to a proprioceptive dataset."""
    assert (
        not info["dataset"]["penetrometer"].empty and
        not info["dataset"]["proprioceptive"].empty
    ), "WARNING: Dataset is missing required data; skipping..."

    # First generate labels from penetrometer data.
    num_layers = 3  # Define the max soil depth we would like to dive.
    soil_layer_stats = _get_penetrometer_labels(
        dataset=info["dataset"]["penetrometer"],
        num_layers=num_layers
    )

    # Now add mean info as labels to the "combined" dataset.
    df_combined = cp.deepcopy(info["dataset"]["proprioceptive"])
    for i in range(num_layers):
        df_combined[f"Mean SPR {i+1} (PSI)"] = soil_layer_stats[i+1]["mean"]
        df_combined[f"STD SPR {i+1} (PSI)"] = soil_layer_stats[i+1]["std"]

    info["dataset"]["combined"] = df_combined

    return info

def _load_dataset(base_path: str, file_names: list[str]) -> dict:
    """Read all data from a given base path."""
    # All metadata and datasets related to a given test session.
    dataset_info = {
        "label": base_path.split("-")[-1],
        "path": {
            "pen_csv": "",
            "robot_csv": "" 
        },
        "dataset": {
            "combined": None,
            "imu": None,
            "mtorques": None,
            "penetrometer": None,
            "proprioceptive": None
        },
        "timestamp": ""
    }
    
    # Get metadata about the dataset.
    dataset_info = _load_dataset_info(
        info=dataset_info,
        base_path=base_path,
        file_names=file_names
    )

    try:
        dataset_info["dataset"]["proprioceptive"] = _read_csv(
            dataset_info["path"]["robot_csv"])
        dataset_info["dataset"]["penetrometer"] = _read_csv(
                dataset_info["path"]["pen_csv"])
    except Exception as e:
        return dataset_info

    # Combine all robot and penetrometer datasets.
    try:
        dataset_info = _label_proprioceptive_dataset(info=dataset_info)
    except AssertionError:
        pass

    return dataset_info

def _split_dataset(dataset: pd.DataFrame) -> dict:
    """Divide a combined dataset into labeld IMU and motor torque datasets."""
    imu_keys = [
            "time (ms)",
            "IMUAccx",
            "IMUAccy",
            "IMUAccz",
            "IMUGyrroll",
            "IMUGyrpitch",
            "IMUGyryaw",
            "IMUQw",
            "IMUQx",
            "IMUQy",
            "IMUQz",
            "SPR 3in (PSI)",
            "SPR 6in (PSI)",
            "SPR 9in (PSI)"
        ]
    torque_keys = [
            "time (ms)",
            "frHip t (Nm)",
            "FLHipT (Nm)",
            "RRHipT (Nm)",
            "RLHipT (Nm)",
            "FRThighT (Nm)",
            "FLThighT (Nm)",
            "RRThighT (Nm)",
            "RLThighT (Nm)",
            "FRCalfT (Nm)",
            "FLCalfT (Nm)",
            "RRCalfT (Nm)",
            "RLCalfT (Nm)",
            "SPR 3in (PSI)",
            "SPR 6in (PSI)",
            "SPR 9in (PSI)"
        ]
    return info

def load_new_datasets(path_base: str) -> list[dict]:
    """
    Import and group raw CSV datasets from data collection trials.
    """
    verbose_datasets = []
    assert os.path.isdir(path_base), "ERROR: Data input directory does not exist!"
    for base, _, files in os.walk(path_base):
        info = _load_dataset(base_path=base, file_names=files)
        # Only append info if it corresponds to a directory containing all 
        # required datasets.
        try:
            if not info["dataset"]["combined"].empty:
                print(f"Adding {info['label']}!")
                verbose_datasets.append(info)
        except AttributeError:
            pass

    return verbose_datasets

# TODO
def plot_steps(csv_data: dict):
    print(csv_data["timestamp"], csv_data["path"])

def preprocess_b1_data(path_input_dir: str, path_output_dir: str):
    # Comb through each directory to find each applicable data file.
    print(f"Input Directory:\t./{path_input_dir}/\r\n"
          f"Output Directory:\t./{path_output_dir}/")
    verbose_datasets = load_new_datasets(path_input_dir)
    assert any(
        [len(info) > 0 for info in verbose_datasets]
    ), "ERROR: No input CSV files found!"

    [_write_dataset(
        dataset=info["dataset"]["combined"],
        path=os.path.join(
            path_output_dir,
            info["label"],
            info["timestamp"],
            "combined.csv"
        )
     ) for info in verbose_datasets]

    # We want to enforce naming conventions as follows:
    #   + <DATE>_<TIME>-<LABEL>-b1_imu.csv
    #   + <DATE>_<TIME>-<LABEL>-b1_joints.csv
    #   + <DATE>_<TIME>-<LABEL>_soilPen-labels.csv
    #
    # In the above, "b1_imu" corresponds to IMU data (accelerations, quaternions),
    # "b1_joints" are torques from each joint in Nm, and "soilPen-labels" are
    # ground truth soil compaction measurements, averaged at each scene for each layer.
    """
    for dataset_info in verbose_datasets:
        #_load_raw_data(info_data_input)
        plot_steps(dataset_info)
    """
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw data from dog into model for use by GreenThumb."
    )
    parser.add_argument("--input_dir", type=str, default="tmp")
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()
    preprocess_b1_data(
        path_input_dir=args.input_dir,
        path_output_dir=args.output_dir
    )
