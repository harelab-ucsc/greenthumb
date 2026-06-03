"""
preprocess_dataset.py

Author:
    HARE Lab
    jLab
    nubby

Date:
    2 Jun 2026

Version:
    1.0.2
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
def label_dataset(info: dict) -> pd.DataFrame:
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

def label_datasets(
        core_labels: pd.DataFrame,
        base_path: str,
        pen_labels: pd.DataFrame
    ) -> list[dict]:
    """
    label_datasets()
    """
    return []

def _load_dataset(base_path: str, file_names: list[str]) -> list[dict]:
    """
    Read all data from a given base path and return a list of processed, labeled
    datasets as pandas DataFrames with accompanying metadata.
    """
    # All metadata and datasets related to a given test session.
    dataset_info = {
        "label": base_path.split("-")[-1],
        "path": {
            "cores": "",
            "pens": "",
            "b1": [],
            "teros12": []
        },
        "dataset": {
            "combined": None,
            "core": None,
            "imu": None,
            "j_angles": None,
            "j_dangles": None,
            "j_d2angles": None,
            "j_torques": None,
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
        dataset_info["dataset"]["core"] = _read_csv(
                dataset_info["path"]["core_csv"])
    except Exception as e:
        return dataset_info

    # Attach GT labels to datasets.
    try:
        processed_datasets = label_datasets(
                core_labels=core_labels,
                pen_labels=pen_labels,
                data_in_dir=data_in_dir
            )
    except AssertionError:
        pass

    return dataset_info 

def _split_dataset(dataset: pd.DataFrame) -> dict:
    """Divide a combined dataset into labeld IMU and motor torque datasets."""
    # TODO: Why is this unused?
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
            "SBD 4in (g/cm^3)",
            "SBD 7in (g/cm^3)",
            "SBD 10in (g/cm^3)",
            "SPR 4in (PSI)",
            "SPR 7in (PSI)",
            "SPR 10in (PSI)",
            "VWC 4in (%)",
            "VWC 7in (%)",
            "VWC 10in (%)"
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
            "FLHipQ (Nm)",
            "RRHipQ (Nm)",
            "RLHipQ (Nm)",
            "FRThighQ (Nm)",
            "FLThighQ (Nm)",
            "RRThighQ (Nm)",
            "RLThighQ (Nm)",
            "FRCalfQ (Nm)",
            "FLCalfQ (Nm)",
            "RRCalfQ (Nm)",
            "RLCalfQ (Nm)",
            "FLHipdQ (Nm)",
            "RRHipdQ (Nm)",
            "RLHipdQ (Nm)",
            "FRThighdQ (Nm)",
            "FLThighdQ (Nm)",
            "RRThighdQ (Nm)",
            "RLThighdQ (Nm)",
            "FRCalfdQ (Nm)",
            "FLCalfdQ (Nm)",
            "RRCalfdQ (Nm)",
            "RLCalfdQ (Nm)",
            "FLHipd2Q (Nm)",
            "RRHipd2Q (Nm)",
            "RLHipd2Q (Nm)",
            "FRThighd2Q (Nm)",
            "FLThighd2Q (Nm)",
            "RRThighd2Q (Nm)",
            "RLThighd2Q (Nm)",
            "FRCalfd2Q (Nm)",
            "FLCalfd2Q (Nm)",
            "RRCalfd2Q (Nm)",
            "RLCalfd2Q (Nm)",
            "SBD 4in (g/cm^3)",
            "SBD 7in (g/cm^3)",
            "SBD 10in (g/cm^3)"
            "SPR 4in (PSI)",
            "SPR 7in (PSI)",
            "SPR 10in (PSI)",
            "VWC 4in (%)",
            "VWC 7in (%)",
            "VWC 10in (%)"
        ]
    return info

def check_previously_run_datasets(path_output_dir: str) -> list[str]:
    """
    check_previously_run_datasets(path_output_dir) -> [to_skip]

    Load dataset labels that have already been processed.
    """
    # TODO.
    return []

def _estimate_depth_pen_labels(
    df: pd.DataFrame,
    depths: list[int],
    interpolation: str = "linear"
) -> pd.DataFrame:
    """
    _estimate_depth_pen_labels(df, depths) -> df

    Args:
        interpolation   (str)   [linear]
            * linear == least squares fit for compaction value.
    """
    # Group each entry by date and compaction level first.
    grouped = df.groupby([
        "Date",
        "Compaction Level (index)"
    ])

    for group in grouped:
        x = group[1]["Depth (inches)"].to_numpy(dtype=float)
        y = group[1]["SPR (PSI, average)"].to_numpy(dtype=float)
        m, b = np.polyfit(x, y, 1)
        for d in depths:
            if (interpolation == "linear"):
                # In linear regression fit, PSI can slip below 0 PSI, which
                # is impossible; thus, enforce a minimum condition here
                # during the fit.
                new_psi = m*d + b if (m*d + b > 0) else 0
            else:
                # TODO: Better understand non-linear/exponential trends in
                #       increase of SPR here.
                new_psi = 0
            new_entry = pd.DataFrame([{
                "Date": group[0][0],
                "Compaction Level (index)": group[0][1],
                "Depth (inches)": d,
                "SPR (PSI, average)": new_psi,
                "SPR (PSI, variance)": None,
                "Sample Size (# jabs)": 0
            }])
            df = pd.concat(
                [df, new_entry],
                join="inner",
                ignore_index=True
            )
    return df

def load_pen_labels(path_base: str) -> pd.DataFrame:
    """
    load_pen_labels(path_base) -> pen_df
    """
    # Load all values from the label file.
    path_pen_labels = "/".join([
        path_base,
        "pen_labels.csv"
    ])
    pen_df = _read_csv(path_pen_labels)

    # NOTE: The below three actions could be addressed with updates to the data
    #       input pipeline.
    # Clean up "Depth" column (to make this an int rather than str).
    vals = pen_df["Depth"].str.extract(r"^(\d+)\s+(.*)$")
    pen_df["Depth"] = vals[0]
    # Replace the "Depth" label for one with units.
    pen_df = pen_df.rename(columns={"Depth": "Depth (inches)"})
    # Replace the "Average PSI" label for one that is more descriptive.
    # NOTE: This is unecessary since we do it again below.
    #pen_df = pen_df.rename(columns={"Average PSI": "SPR (PSI, average)"})

    # Calculate SPR stats for each depth.
    pen_cols = [
            "Penetrometer PSI (sample 1)",
            "Penetrometer PSI (sample 2)",
            "Penetrometer PSI (sample 3)",
            "Penetrometer PSI (sample 4)",
            "Penetrometer PSI (sample 5)"
    ]
    pen_df["SPR (PSI, average)"] = pen_df[pen_cols].mean(axis=1)
    pen_df["SPR (PSI, variance)"] = pen_df[pen_cols].var(axis=1)
    pen_df["Sample Size (# jabs)"] = pen_df[pen_cols].notna().sum(axis=1)

    # Generate new labels (if not currently present) for [4, 7, 10]" range.
    new_depths = [4, 7, 10]
    pen_df = _estimate_depth_pen_labels(df=pen_df, depths=new_depths)

    return pen_df

def convert_df_core_to_avgs(df: pd.DataFrame) -> pd.DataFrame:
    """
    convert_df_core_to_avgs(df) -> df

    Merge ground-truth info for VWC and SBD from soil cores into averages,
    variances, and ns.
    """
    # Start with a new, empty DataFrame.
    df_out = None

    # Group each entry by date, compaction level, and depth.
    grouped = df.groupby([
        "Date",
        "Compaction Level (index)",
        "Depth (inches)"
    ])

    for group in grouped:
        # Average VWCs and SBDs from cores, collecting info on variance and
        # sample size as we go.
        n = len(group[1])
        vwc_avg = group[1]["VWC (%, derived)"].mean()
        vwc_var = group[1]["VWC (%, derived)"].var()
        sbd_avg = group[1]["Bulk density (g/cm^3)"].mean()
        sbd_var = group[1]["Bulk density (g/cm^3)"].var()
        new_entry = pd.DataFrame([{
            "Date": group[0][0],
            "Compaction Level (index)": group[0][1],
            "Depth (inches)": group[0][2],
            "Sample Size (# cores)": n,
            "Core VWC (%, average)": vwc_avg,
            "Core VWC (%, variance)": vwc_var,
            "SBD (g/cm^3, average)": sbd_avg,
            "SBD (g/cm^3, variance)": sbd_var
        }])
        df_out = pd.concat(
                [df_out, new_entry],
                join="inner",
                ignore_index=True
            )
    return df_out

def load_core_labels(path_base: str) -> pd.DataFrame:
    """
    load_core_labels(path_base) -> core_df
    """
    path_core_labels = "/".join([
        path_base,
        "core_labels.csv"
    ])
    core_df = _read_csv(path_core_labels)

    # Calculate average SBD for each depth for each compaction level on a given
    # date.
    core_df = convert_df_core_to_avgs(core_df)

    # TODO: Allow for inference of compaction levels at specific depths?
    return core_df

def load_teros12_data(path_base: str) -> pd.DataFrame:
    """
    load_teros12_data(path_base) -> teros12_df
    """
    return None

def load_datasets(path_base: str, to_skip: list = []) -> list[dict]:
    """
    load_datasets(path_base, to_skip) -> datasets

    Import, label, and trim raw B1 CSV datasets from data collection trials.
    """
    # Ensure that the base path actually exists.
    assert (
        os.path.isdir(path_base)
    ), (
        "ERROR: Data input directory does not exist!"
    )

    # First load labels from the penetrometer (SPR) and soil cores (SBD, VWC).
    core_df = load_core_labels(path_base=path_base)
    pen_df = load_pen_labels(path_base=path_base)

    # Next load and combine [valid] sensor feeds from TEROS-12 sensors on all
    # days.
    teros12_df = load_teros12_data(path_base=path_base)
    for base, _, files in os.walk(path_base):
        print(base)
        print(files)
        print()
        """
        info = _load_dataset(base_path=base, file_names=files)
        # Only append info if it corresponds to a directory containing all 
        # required datasets.
        try:
            if not info["dataset"]["combined"].empty:
                print(f"Adding {info['label']}!")
                verbose_datasets.append(info)
        except AttributeError:
            pass
        """
    exit()

    return None, None, core_df, pen_df, None

def preprocess_dataset(
        path_input_dir: str,
        path_output_dir: str,
        force: bool = False
    ):
    """
    preprocess_dataset(
            path_input_dir,
            path_output_dir
        )

    Description:
        Clean and organize data from controlled compaction studies into joint,
        IMU, and label files.

        We want to enforce naming conventions as follows:
            + <DATE>_<TIME>-<LABEL>-b1_imu.csv
            + <DATE>_<TIME>-<LABEL>-b1_joints.csv
            + <DATE>_<TIME>-<LABEL>-labels.csv

        In the above, "b1_imu" corresponds to IMU data (accelerations,
        quaternions), "b1_joints" are joint-related readings, including
        (torques, angles, d_angle, d^2_angle), and "labels" include (SPR, SBD)
        ground-truth soil compaction measurements, averaged at each scene for
        each layer.

        Each "LABEL" above should follow "SETTING_COMPACTION" format, e.g.
        "SETTING" = "Lab", "COMPACTION" = "0" (for loose soil).
    """
    print(f"Input Directory:\t./{path_input_dir}/\r\n"
          f"Output Directory:\t./{path_output_dir}/\r\n"
    )

    # Check which datasets have already been formatted to skip.
    to_skip = []
    if not force:
        to_skip += check_previously_run_datasets(path_output_dir)

    # Process all desired datasets as follows:
    #   1. Extract ground truth labels for SBD and SPR as DataFrames.
    #   2. Combine all [valid] TEROS-12 data streams into one DataFrame.
    #   3. Extract each B1 sensor feed and provide metadata about start/stop
    #       times, test setting, etc.
    #   4. Do the same as (3) for Chipotle scans.
    (b1_datasets,
     chipotle_datasets,
     core_df,
     pen_df,
     teros12_dataset) = load_datasets(
            path_base=path_input_dir,
            to_skip=to_skip
        )

    exit()
    assert (
        any(
            [len(info) > 0 for info in verbose_datasets]
        )
    ), "ERROR: No input CSV files found!"

    # Output properly formatted dataset as described above.
    [_write_dataset(
        dataset=info["dataset"]["combined"],
        path=os.path.join(
            path_output_dir,
            info["label"],
            info["timestamp"],
            "combined.csv"
        )
     ) for info in verbose_datasets]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw sensor data into a form usable by GreenThumb."
    )
    parser.add_argument("--input_dir", type=str, default="../Data")
    parser.add_argument("--output_dir", type=str, default="processed")
    parser.add_argument(
        "--force",
        type=bool,
        default=False,
        help="Force script to reprocess data?"
    )
    args = parser.parse_args()
    preprocess_dataset(
        path_input_dir=args.input_dir,
        path_output_dir=args.output_dir,
        force=args.force
    )
