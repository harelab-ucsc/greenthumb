"""
preprocess_dataset.py

Author:
    HARE Lab
    jLab
    nubby

Date:
    30 Jun 2026

Version:
    1.0.10
"""
import argparse
import copy as cp
import csv
import matplotlib
import numpy as np
import os
import pandas as pd
import re

from datetime import datetime, timedelta
from io import StringIO


# Useful macros.
_RECURSION_MAX = 5                          # Number of recursions deep.
_T_TEROS12_DIFF_MAX = 2 * 60 * 60 * 1000    # 2 hours.


# TODO: Add "eval_id" to output dataset.
# TODO: Debug strange directory naming convention.
# TODO: Separate code for each device type into specialized submodules/classes.

def _read_csv(path: str, ds_type: str = "") -> pd.DataFrame:
    """
    Read CSV file and return a dataset.

    Args:
        path    (str)
        ds_type (str)   Dataset type label. Accepted labels include:
                        + teros12   - TEROS-12 data.
                        + b1        - Unitree B1 data.
                        All other values will default to normal CSV format.
    """
    if ds_type == "teros12":
        data_rows = []
        with open(path, "r+") as csvfp:
            reader = csv.reader(csvfp)
            # The first two rows of these files contains metadata and should
            # be skipped.
            for row in reader:
                if (len(row) <= 1):
                    continue
                data_rows.append(row)
        return pd.DataFrame(data=data_rows[1:], columns=data_rows[0])
    elif ds_type == "b1":
        with open(path, "r") as fp:
            data_array = [
                line for i, line in enumerate(fp) if i == 0 or "," in line
            ]
        # Convert CSV array into pandas DataFrame.
        try:
            return pd.read_csv(StringIO("".join(data_array)))
        except Exception as e:
            print(f"Could not read {path}!")
            print(e)
            exit()
    else:
        # NOTE: May miss data due to bad lines being skipped with the below.
        #return pd.read_csv(path, on_bad_lines="skip")
        return pd.read_csv(path)

def _write_dataset(dataset: pd.DataFrame, path: str):
    # First create any required parent directories.
    pdir = os.path.dirname(path)
    os.makedirs(pdir, exist_ok=True)

    # Write it!
    dataset.to_csv(path, index=False)
    print(f"SUCCESS:\tWrote {path}!")

def _get_b1_df_label(df: pd.DataFrame) -> str:
    """
    _get_b1_df_label(df) -> label

    Form a dataset label using the start time, trial label, compaction level,
    and wetness level.
    """
    t_start = df["Timestamp (Epoch-UTC-ms)"].values[0]
    label_b1 = df["Dataset Label"].values[0]
    idx_comp = df["Compaction Level (index)"].values[0]
    idx_wet = df["SWC Level (index)"].values[0]

    return "-".join([
        t_start,
        label_b1,
        "c"+idx_comp,
        "w"+idx_wet
    ])

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
        # NOTE: These "VWC (%, derived)" values are actually GWC.
        gwc_avg = group[1]["VWC (%, derived)"].mean()
        gwc_var = group[1]["VWC (%, derived)"].var()
        sbd_avg = group[1]["Bulk density (g/cm^3)"].mean()
        sbd_var = group[1]["Bulk density (g/cm^3)"].var()
        new_entry = pd.DataFrame([{
            "Date": group[0][0],
            "Compaction Level (index)": group[0][1],
            "Depth (inches)": group[0][2],
            "Sample Size (# cores)": n,
            "Core GWC (%, average)": gwc_avg,
            "Core GWC (%, variance)": gwc_var,
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
        try:
            m, b = np.polyfit(x, y, 1)
        except Exception as e:
            print(x)
            print(y)
            print(e)
            exit()
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
                "SPR (PSI, variance)": 0,
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
    # TODO: Allow this to accept integer depth values as well.
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

def teros12_file_is_valid(path: str) -> bool:
    """
    teros12_file_is_valid(path) -> valid?
    """
    return True

def ingest_teros12_data_from_path(path: str) -> pd.DataFrame:
    """
    ingest_teros12_data_from_path(path) -> teros12_df
    """
    return None

def _get_teros12_paths(path_base: str) -> list[str]:
    """
    _get_teros12_paths(path_base) -> paths
    """
    paths_teros12 = []
    # Sift through each path in search of TEROS-12 data. Once you find a
    # "teros12" directory, copy the full paths to a returned list.
    for base, _, files in os.walk(path_base):
        # Enter only the "teros12" folders.
        if "teros12" in base:
            for file in files:
                # Only ingest valid TEROS-12 data files.
                fpath = "/".join([base, file])
                if (teros12_file_is_valid(fpath)):
                    paths_teros12.append(fpath)
    return paths_teros12


def _get_teros12_ds_labels_from_path(path: str) -> dict:
    """
    _get_teros12_ds_labels_from_path(path) -> ds_labels_dict

    TEROS-12 file paths are expected to follow the following format:

        <DEPTH>inch-teros12-<DATE>_<TIME>-sensor_<SENSOR #>.csv

    This method extracts depth (inches), date, time, and sensor #.
    """
    labels_raw = path.split("/")[-1].split("-")

    # Assign metadata labels based on label location.
    try:
        l_depth = int(labels_raw[0].split("inch")[0])
    except ValueError:
        print(f"WARNING: Could not get TEROS-12 depth label from {path};"
              f" skipping...")
        return None

    try:
        l_datetime_start = datetime.strptime(labels_raw[2], "%Y%m%d_%H%M%S")
    except ValueError:
        print(f"WARNING: Could not get TEROS-12 start time label from {path};"
              f" skipping...")
        return None

    try:
        l_sensor_idx = labels_raw[-1].split("_")[-1].split(".")[0]
    except ValueError as e:
        print(f"WARNING: Could not get TEROS-12 sensor index label from {path};"
              f" skipping...")
        return None

    # Correct depths if improper depth recorded.
    ## NOTE: After 20260515, all tests were done with 4", 7", and 10" depths.
    l_depths_to_change = [3, 6, 9]
    if (l_datetime_start >= datetime(year=2026, month=5, day=15)):
        # All improperly-labeled depths should only need to be incremented by 1.
        if l_depth in l_depths_to_change:
            l_depth += 1

    return {
            "Depth (inches)": l_depth,
            "Datetime Start": l_datetime_start,
            "Sensor Index": l_sensor_idx
        }

def _extract_labeled_teros12_dfs_from_paths(
        paths: list[str]
    ) -> list[pd.DataFrame]:
    """
    _extract_labeled_teros12_dfs_from_paths(paths) -> dfs
    """
    dfs = []
    for path in paths:
        # Extract labels from path.
        ds_labels_dict = _get_teros12_ds_labels_from_path(path)

        # Extract/Format CSV data from each dataset as DataFrames.
        try:
            df = _read_csv(path, ds_type="teros12")
        except pd.errors.ParserError:
            print(f"WARNING: Dataset at {path} invalid CSV.")
            continue

        # Label each DataFrame.
        for label in ds_labels_dict.keys():
            df[label] = ds_labels_dict[label]

        # Append DataFrame.
        dfs.append(df)

    return dfs

def _group_teros12_dfs_by_session(
        dfs: list[pd.DataFrame]
    ) -> list[pd.DataFrame]:
    """
    _group_teros12_dfs_by_session(dfs) -> dfs
    """
    dfs_out = []        # Holder for all final, grouped DataFrames.
    df_groups = [] 
    t_delta_thresh = 5  # Number of seconds between starting data collection
                        # sessions to consider a file as part of the same
                        # dataset.
    for df in dfs:
        # Create groups of start timestamps based on their proximity to others.
        l_start_datetime = df["Datetime Start"][0]
        # It is sufficient to check against the first entry of each group based
        # on threshold.
        grouped = False
        for group in df_groups:
            group_datetime = group[0]["Datetime Start"][0]
            if (abs((
                l_start_datetime - group_datetime
                ).seconds) <= t_delta_thresh):
                # Add to existing group if start timestamp within threshold.
                group.append(df)
                grouped = True
        if not grouped:
            # Create a new group if not within threshold of any group.
            df_groups.append([df])
    
    # Now combine members of each group into single DataFrames.
    for group in df_groups:
        df_grouped = pd.concat(
                group,
                join="inner",
                ignore_index=True
            )
        dfs_out.append(df_grouped)

    return dfs_out


def _filter_teros12_dfs(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    _filter_teros12_dfs(dfs) -> dfs

    Scrap TEROS-12 datasets which are invalid for the following reasons:
        + Estimated VWC exceeds wetness threshold or fails to meet dryness
          threshold.
        + More?
    """
    # Filter criteria.
    vwc_max = 0.30
    vwc_min = 0.04

    dfs_filtered = []

    dfs_no_start = len(dfs)

    for df in dfs:
        # Filter out VWCs outside of the bounds of acceptable by collecting
        # session labels (remove all from that set).
        if (float(df["VWC"].max()) > vwc_max):
            l_datetime = datetime.strftime(df["Datetime Start"][0],
                                           "%Y%m%d_%H%M%S")
            print(f"Filtering out data from {l_datetime}: > VWC_max...")
        elif (float(df["VWC"].min()) < vwc_min):
            l_datetime = datetime.strftime(df["Datetime Start"][0],
                                           "%Y%m%d_%H%M%S")
            print(f"Filtering out data from {l_datetime}: < VWC_min...")
        else:
            dfs_filtered.append(df)

    dfs_no_end = len(dfs_filtered)
    print(f"\r\nSUCCESS: Filtered out {dfs_no_start - dfs_no_end} TEROS-12 "
          f"datasets!")

    return dfs_filtered

def _format_teros12_dfs(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    _format_teros12_dfs(dfs) -> dfs_formatted
    """
    dfs_reformatted = []

    for df in dfs:
        # Create a new, empty DF.
        df_tmp = pd.DataFrame()
        # Replace generic "timestamp" label with UTC Epoch label (in ms).
        df_tmp["Timestamp (Epoch-UTC-ms)"] = (
            pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S")
                .dt.tz_localize("America/Los_Angeles")
                .dt.tz_convert("UTC")
                .astype("int64") // 1_000_000
            )
        df.rename(
                columns={"timestamp": "Timestamp (Epoch-UTC-ms)"},
                inplace=True
            )
        df["Timestamp (Epoch-UTC-ms)"] = df_tmp["Timestamp (Epoch-UTC-ms)"]
        
        # Drop all 'nan' timestamp rows.
        df = df.dropna(subset=["Timestamp (Epoch-UTC-ms)"])

        # Next, reformat VWCs to be floats.
        df["VWC"] = df["VWC"].astype(float)

        dfs_reformatted.append(df)

    return dfs

def extract_teros12_data_from_paths(paths: list[str]) -> pd.DataFrame:
    """
    ingest_teros12_data_from_path(paths) -> df

    Convert a list of TEROS-12 data paths to one DataFrame.
    """
    # First generate a list of TEROS-12 labeled DataFrames.
    dfs_teros12_all = _extract_labeled_teros12_dfs_from_paths(paths=paths)

    # Group together TEROS-12 DataFrames by session.
    dfs_teros12_grouped = _group_teros12_dfs_by_session(dfs=dfs_teros12_all)

    # Filter out invalid datasets.
    dfs_teros12_filtered = _filter_teros12_dfs(dfs=dfs_teros12_grouped)

    # Format each TEROS-12 dataset to match other DFs.
    dfs_teros12_formatted = _format_teros12_dfs(dfs=dfs_teros12_filtered)

    # Merge together all TEROS-12 DataFrames into one (with labels), sorted
    # by timestamp.
    df_teros12 = pd.concat(
            dfs_teros12_formatted,
            join="inner",
            ignore_index=True
        ).sort_values(by="Timestamp (Epoch-UTC-ms)")

    return df_teros12

def load_teros12_data(path_base: str) -> pd.DataFrame:
    """
    load_teros12_data(path_base) -> teros12_df
    """
    # Get the paths of each TEROS-12 data file.
    paths_teros12 = _get_teros12_paths(path_base=path_base)

    # Extract sensor data and metadata from each valid file path and convert to
    # a single DataFrame.
    df_teros12 = extract_teros12_data_from_paths(paths_teros12)

    return df_teros12 

def b1_file_is_valid(path: str) -> bool:
    """
    b1_file_is_valid(path) -> is_valid
    """
    # TODO
    return True

def _get_b1_paths(path_base: str) -> list[str]:
    """
    _get_b1_paths(path_base) -> paths
    """
    paths_b1 = []
    # Sift through each path in search of B1 datastreams. Once you find a
    # "b1" directory, copy the full paths to a returned list.
    for base, _, files in os.walk(path_base):
        # Enter only the "b1" folders.
        if "b1" in base:
            for file in files:
                # Only ingest valid B1 data files.
                fpath = "/".join([base, file])
                if (b1_file_is_valid(fpath)):
                    paths_b1.append(fpath)
    return paths_b1

def _get_b1_ds_labels_from_path(path: str) -> dict:
    """
    _get_b1_ds_labels_from_path(path) -> labels:
    """
    labels_raw = path.split("/")[-1].split("-")

    # Look-up tables (LUTs) for various labels.

    ## Indicates the number of times that the box was sprayed before data was
    ## collected.
    lut_dampness = {
            "fieldDry": 0,
            "fieldDamp1": 2
        }

    # Assign metadata labels based on label location.
    try:
        l_comp_level = int(labels_raw[-1].split(".")[0])
    except ValueError:
        print(f"WARNING: Could not get compaction level from {path};"
              f" skipping...")
        return None

    try:
        l_name = "-".join([labels_raw[0], labels_raw[1]]) 
    except ValueError:
        print(f"WARNING: Could not get B1 dataset label from {path};"
              f" skipping...")
        return None

    try:
        l_wet_level = lut_dampness[labels_raw[-2]]
    except ValueError as e:
        print(f"WARNING: Could not get dampness level from {path};"
              f" skipping...")
        return None

    try:
        l_date = datetime.strptime(
                labels_raw[0], "%Y%m%d_%H%M%S"
            ).strftime("%-m/%-d/%Y")
    except ValueError:
        print(f"WARNING: Could not get B1 dataset date label from {path};"
              f" skipping...")
        return None

    return {
            "Compaction Level (index)": l_comp_level,
            "Dataset Label": l_name,
            "Date": l_date,
            "SWC Level (index)": l_wet_level
        }

def scrub_b1_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    scrub_df(df) -> df

    Description:
        Excise both columns that never change in value and long portions of
        data which remain 0 (i.e. at the beginning).
        
        Copied from `step_detector.py` almost verbatim.

    Args:
        df  (pd.DataFrame)

    Returns:
        df  (pd.DataFrame)
    """
    # Start with only numeric columns.
    numeric = df.select_dtypes(include=["number"])

    # First remove constant and 0.0 columns (from Perplexity).
    df = df.loc[:, df.nunique(dropna=False) > 1].copy()
    df = df.loc[:, df.var(numeric_only=True) != 0]

    # Next, delete datapoints which remain 0.0 for a while (i.e. at the
    # start of data collection).
    zero_threshold = 0.9    # Fraction of columns that can be 0.0 to be dropped.
    if not numeric.empty:
        zero_fraction = (numeric == 0.0).mean(axis=1)
        df = df.loc[zero_fraction < zero_threshold].copy()

    return df


def _extract_labeled_b1_dfs_from_paths(paths: list[str]) -> list[pd.DataFrame]:
    """
    _extract_labeled_b1_dfs_from_paths(paths) -> dfs_b1
    """
    dfs = []
    for path in paths:
        # Pull metadata from path names.
        ds_labels_dict = _get_b1_ds_labels_from_path(path)

        # Extract/Format CSV data from each dataset as DataFrames.
        try:
            df = _read_csv(path, ds_type="b1")
        except pd.errors.ParserError:
            print(f"WARNING: Dataset at {path} invalid CSV.")
            continue

        # Trim both empty data columns (in the case of early test data) and
        # long sequences of nothing (as in the beginning).
        df = scrub_b1_df(df)

        # Rename the timestamps column with something more descriptive.
        df.rename(
                columns={"time (ms)": "Timestamp (Epoch-UTC-ms)"},
                inplace=True
            )

        # Label each DataFrame.
        for label in ds_labels_dict.keys():
            df[label] = ds_labels_dict[label]

        # Append DataFrame.
        dfs.append(df)

    return dfs

def _filter_b1_dfs(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    _filter_b1_dfs(dfs) -> dfs_b1
    """
    # TODO: Incorporate filtering methods from `step_detector` module.
    return dfs

def extract_b1_data_from_paths(paths: list[str]) -> list[pd.DataFrame]:
    """
    extract_b1_data_from_paths(paths) -> dfs_b1
    """
    # Generate labeled B1 DataFrames from paths.
    dfs_b1_all = _extract_labeled_b1_dfs_from_paths(paths=paths)

    # Filter out invalid DataFrames.
    dfs_b1_filtered = _filter_b1_dfs(dfs=dfs_b1_all)

    return dfs_b1_filtered

def load_b1_datasets(path_base: str) -> list[pd.DataFrame]:
    """
    load_b1_datasets(path_base) -> b1_dfs

    Each B1 DataFrame generated herein is appropriately polished.
    """
    # Get the paths of each B1 data file.
    paths_b1 = _get_b1_paths(path_base=path_base)

    # Extract sensor data and metadata from each valid file path and convert to
    # labeled dataframes.
    dfs_b1 = extract_b1_data_from_paths(paths_b1)
    
    print(f"SUCCESS: Ingested {len(dfs_b1)} B1 datasets!")

    return dfs_b1


def load_chipotle_datasets(path_base: str) -> pd.DataFrame:
    """
    load_chipotle_datasets(path_base) -> teros12_df
    """
    # TODO: For multimodal models.
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
    df_cores = load_core_labels(path_base=path_base)
    df_pen = load_pen_labels(path_base=path_base)

    # Next load and combine [valid] sensor feeds from TEROS-12 sensors on all
    # days.
    df_teros12 = load_teros12_data(path_base=path_base)

    # Load all B1 and Chipotle sensor streams into their own datasets with
    # metadata attached (i.e. as dicts with structure:
    #                           {"df": df, "date": <DATE>, ...}, etc).
    dfs_b1 = load_b1_datasets(path_base=path_base)
    dfs_chipotle = load_chipotle_datasets(path_base=path_base)

    return (dfs_b1,
            dfs_chipotle,
            df_cores,
            df_pen,
            df_teros12)


def _label_get_sbd_from_date_comp_index(
        df_cores: pd.DataFrame,
        l_date: str,
        l_compaction_level: int
    ) -> (float, float, float):
    """
    _label_get_sbd_from_date_comp_index(...) -> labels

    Extract labels from core DF; please note:
      + Only the first entry for each compaction level will be considered
          (there should be only one average.
      + VWC index labels are not included in core metadata, so they are not
          used for references here.
    """
    try:
        l_sbd_0 = df_cores.loc[
                (df_cores["Depth (inches)"] == 0) &
                (df_cores["Date"] == l_date) &
                (df_cores["Compaction Level (index)"] == str(l_compaction_level)),
                ["SBD (g/cm^3, average)"]
            ].values[0][0]
    except IndexError:
        print(f"WARNING: Cannot find SBD from cores at 0\" for {l_date}, "
              f"compaction level {l_compaction_level}; marking as -1.")
        l_sbd_0 = -1

    try:
        l_sbd_4 = df_cores.loc[
                (df_cores["Depth (inches)"] == 4) &
                (df_cores["Date"] == l_date) &
                (df_cores["Compaction Level (index)"] == str(l_compaction_level)),
                ["SBD (g/cm^3, average)"]
            ].values[0][0]
    except IndexError:
        print(f"WARNING: Cannot find SBD from cores at 4\" for {l_date}, "
              f"compaction level {l_compaction_level}; marking as -1.")
        l_sbd_4 = -1

    try:
        l_sbd_7 = df_cores.loc[
                (df_cores["Depth (inches)"] == 7) &
                (df_cores["Date"] == l_date) &
                (df_cores["Compaction Level (index)"] == str(l_compaction_level)),
                ["SBD (g/cm^3, average)"]
            ].values[0][0]
    except IndexError:
        print(f"WARNING: Cannot find SBD from cores at 7\" for {l_date}, "
              f"compaction level {l_compaction_level}; marking as -1.")
        l_sbd_7 = -1

    return (l_sbd_0, l_sbd_4, l_sbd_7)

def _label_get_spr_from_date_comp_index(
        df_pen: pd.DataFrame,
        l_date: str,
        l_compaction_level: int
    ) -> (float, float, float):
    """
    _label_get_spr_from_date_comp_index(...) -> labels

    Extract labels from pen DF; please note:
      + Only the first entry for each compaction level will be considered
          (there should be only one average).
      + VWC index labels are not included in core metadata, so they are not
          used for references here.
    """
    try:
        l_spr_3 = df_pen.loc[
                (df_pen["Depth (inches)"].astype(int) == 3) &
                (df_pen["Date"] == l_date) &
                (df_pen["Compaction Level (index)"] == l_compaction_level),
                ["SPR (PSI, average)"]
            ].values[0][0]
    except IndexError:
        print(f"WARNING: Cannot find SBD from pen at 0\" for {l_date}, "
              f"compaction level {l_compaction_level}; marking as -1.")
        l_spr_0 = -1

    try:
        l_spr_4 = df_pen.loc[
                (df_pen["Depth (inches)"].astype(int) == 4) &
                (df_pen["Date"] == l_date) &
                (df_pen["Compaction Level (index)"] == l_compaction_level),
                ["SPR (PSI, average)"]
            ].values[0][0]
    except IndexError:
        print(f"WARNING: Cannot find SBD from pen at 4\" for {l_date}, "
              f"compaction level {l_compaction_level}; marking as -1.")
        l_spr_4 = -1

    try:
        l_spr_7 = df_pen.loc[
                (df_pen["Depth (inches)"].astype(int) == 7) &
                (df_pen["Date"] == l_date) &
                (df_pen["Compaction Level (index)"] == l_compaction_level),
                ["SPR (PSI, average)"]
            ].values[0][0]
    except IndexError:
        print(f"WARNING: Cannot find SBD from pen at 7\" for {l_date}, "
              f"compaction level {l_compaction_level}; marking as -1.")
        l_spr_7 = -1

    return (l_spr_3, l_spr_4, l_spr_7)

def _label_b1_compaction(
        df_b1: pd.DataFrame,
        df_cores: pd.DataFrame, 
        df_pen: pd.DataFrame
    ) -> pd.DataFrame:
    """
    _label_b1_compaction(...) -> df_labeled
    """
    # TODO: Get rid of ',' in column headers.
    # Match ground truth soil compaction info to B1 dataset date, compaction
    # level, and soil moisture level.
    l_date = df_b1["Date"].values[0]
    l_compaction_level = df_b1["Compaction Level (index)"].values[0]
    l_swc_level = df_b1["SWC Level (index)"].values[0]

    l_sbd_0, l_sbd_4, l_sbd_7 = _label_get_sbd_from_date_comp_index(
            df_cores=df_cores,
            l_date=l_date,
            l_compaction_level=l_compaction_level
        )

    # TODO: Consider adding more SPR labels since 0" is basically unuseable.
    l_spr_3, l_spr_4, l_spr_7 = _label_get_spr_from_date_comp_index(
            df_pen=df_pen,
            l_date=l_date,
            l_compaction_level=l_compaction_level
        )

    # Merge datasets now that full compaction info is absorbed.
    df_b1[[
        "SBD (g/mL-avg-0in)",
        "SBD (g/mL-avg-4in)",
        "SBD (g/mL-avg-7in)",
        "SPR (PSI-avg-3in)",
        "SPR (PSI-avg-4in)",
        "SPR (PSI-avg-7in)"
    ]] = [
            l_sbd_0,
            l_sbd_4,
            l_sbd_7,
            l_spr_3,
            l_spr_4,
            l_spr_7
        ]
    
    return df_b1

def _get_teros12_start_index_from_t_start(
        depth: int,
        rows: tuple[float, float],
        t_start: float
    ) -> (int, list):
    """
    _label_b1_wetness_get_rows_by_depth_and_t_start(...) -> index
    """
    idx_t12 = 0

    # Iterate through until we reach the start index (do not go past).
    while float(rows[idx_t12][0]) <= t_start:
        idx_t12 += 1
        if (idx_t12 >= len(rows)):
            # Raise an index error if start time is beyond all measurments.
            print(idx_t12)
            print(t_start)
            print(rows[-1])
            print(len(rows))
            raise IndexError

    return idx_t12 - 1

def _get_prev_index(t_ref: float, t_list: list[float]) -> int:
    """
    _get_prev_index(t_ref, t_list) -> idx_start

    Find the index of the timestamp directly before a given timestamp.
    Assumes that t_list is sorted.
    """
    for i, t in enumerate(t_list):
        if t <= t_ref:
            return i

def _get_t_vwc_rows_teros12_from_df(
        df: pd.DataFrame,
        depth: int
    ) -> list[float, float]:
    """
    _get_t_vwc_rows_teros12_from_df(df, depth) -> list[t, vwc]
    """
    # Get timestamps paired with VWC and depth label from TEROS-12 dataset.
    rows = df.loc[df["Depth (inches)"].astype(int) == depth,
                  ["Timestamp (Epoch-UTC-ms)", "VWC"]
              ].values

    return rows

def _label_b1_wetness_get_t12_idx(
        idx: int,
        rows: tuple[float, float],
        t: float,
        rec_n: int = 0
    ) -> tuple[int, bool]:
    """
    _label_b1_wetness_get_t12_idx(...) -> (idx_t12, day_jump?) 

    Recursive function for finding the current interval in which B1 data exists.
    """
    assert (rec_n < _RECURSION_MAX), (f"ERROR: Beyond max recursion!")

    t_prev = rows[idx][0]
    t_next = rows[idx+1][0]

    if (t_next - t_prev >= _T_TEROS12_DIFF_MAX):
        # If TEROS-12 streaming began after a run began on a day, print a
        # warning and use the first datapoint of the day until we catch up.
        day_1 = pd.to_datetime(
                int(t_prev), unit="ms", utc=True
            ).tz_convert("America/Los_Angeles").date()
        day_2 = pd.to_datetime(
                int(t_next), unit="ms", utc=True
            ).tz_convert("America/Los_Angeles").date()
        if (day_1 != day_2):
            # On differing days, increment the index and use timestamps from
            # that day.
            # TODO: Test that this doesn't produce some wild artifacts from
            #       linear fitting.
            print(f"WARNING: TEROS-12 data jumps {day_1} to {day_2}; using "
                  f"first datapoint from {day_2}.")
            return (idx, True)
        else:
            # If within the same day, print a warning.
            print(f"WARNING: TEROS-12 data jumps from {t_prev} "
                  f"(VWC: {rows[idx][1]}) to {t_next} (VWC: {rows[idx+1][1]}).")

    assert(t_next > t_prev), (
            f"ERROR: Something is up with TEROS-12 data at {idx}:\n",
            f"\t{t_prev} -> {t_next}")

    if (t >= t_prev) and (t < t_next):
        # Hit the right interval.
        return (idx, False)
    elif (t < t_prev):
        # Go backwards here.
        return _label_b1_wetness_get_t12_idx(
            idx=idx-1,
            rows=rows,
            t=t,
            rec_n=rec_n+1
        )
    elif (t >= t_next):
        # Go forwards here.
        return _label_b1_wetness_get_t12_idx(
            idx=idx+1,
            rows=rows,
            t=t,
            rec_n=rec_n+1
        )

def _label_b1_wetness_get_teros12_labels(
        depth: int,
        df_b1: pd.DataFrame,
        df_teros12: pd.DataFrame,
        t_b1: tuple[float]
    ) -> pd.DataFrame:
    """
    _label_b1_wetness_get_teros12_labels(...) -> df

    Assume B1 timestamps are sorted properly.
    """
    # Get pairs of t and VWC.
    rows_t12 = _get_t_vwc_rows_teros12_from_df(df=df_teros12, depth=depth)

    # Find the starting TEROS-12 timestamp index to save on time.
    try:
        idx_t12 = _get_teros12_start_index_from_t_start(
                depth=depth,
                rows=rows_t12,
                t_start=t_b1[0]
            )
    except IndexError as e:
        print(rows_t12[-1])
        print(f"WARNING: Missing TEROS-12 data ({depth}in) for B1 runs at "
              f"{t_b1[0]}! Skipping...")
        return df_b1

    # Iteratively fill in VWC values throughout the the time of data collection:
    #   + Each additional datapoint is smoothly connected to the following one
    #       using a variety of methods.
    #           (currently supported: [linear])

    for t in t_b1:
        # NOTE: Need to constrain against long jumps in TEROS-12 time.
        # Start by locating previous/next TEROS-12 rows.
        idx_t12, day_jump = _label_b1_wetness_get_t12_idx(
                idx=idx_t12,
                rows=rows_t12,
                t=t
            )

        if day_jump:
            # If data jumps a day, use the next TEROS-12 datapoint until we
            # catch up.
            vwc_now_est = rows_t12[idx_t12+1][1]
        else:
            # If all is as it should be, linearly interpolate:
            # Unzip TEROS-12 rows for math.
            [t_t12_prev, vwc_prev] = [float(i) for i in rows_t12[idx_t12]]
            [t_t12_next, vwc_next] = [float(i) for i in rows_t12[idx_t12+1]]
            
            # Said math.
            m = (vwc_next - vwc_prev) / (t_t12_next - t_t12_prev)        
            vwc_now_est = vwc_prev + m * (t - t_t12_prev)

        # Add estimated VWC with t.
        df_b1.loc[df_b1["Timestamp (Epoch-UTC-ms)"]  == t,
                  f"Est VWC (%-{depth}in)"] = vwc_now_est

    return df_b1

def _label_b1_wetness(
        df_b1: pd.DataFrame,
        df_teros12: pd.DataFrame
    ) -> pd.DataFrame:
    """
    _label_b1_wetness(...) -> df_labeled

    Since our TEROS-12 data are sampled every 10s, dates and times must be
    matched to align moisture data with each B1 DF.
    """
    # Get timestamps from B1 dataset.
    # NOTE: In the future, B1 data will not be synchronized with TEROS-12 data.
    #       Alternative methods of alignment will need to be explored.
    t_b1 = sorted(df_b1["Timestamp (Epoch-UTC-ms)"].astype(float).values)

    # Get TEROS-12 data for alignment with B1 DF.
    try:
        df_b1 = _label_b1_wetness_get_teros12_labels(
                depth=4,
                df_b1=df_b1,
                df_teros12=df_teros12,
                t_b1=t_b1
            )
        df_b1 = _label_b1_wetness_get_teros12_labels(
                depth=7,
                df_b1=df_b1,
                df_teros12=df_teros12,
                t_b1=t_b1
            )
        df_b1 = _label_b1_wetness_get_teros12_labels(
                depth=10,
                df_b1=df_b1,
                df_teros12=df_teros12,
                t_b1=t_b1
            )

    except AssertionError as e:
        print(e)
        print(f"B1 time start:\t{t_b1[0]};\n"
              f"B1 time end:\t{t_b1[-1]};\n")
        print(df_teros12)
        exit()
    
    return df_b1

def label_dfs_b1(
        dfs_b1: list[pd.DataFrame],
        df_cores: pd.DataFrame,
        df_pen: pd.DataFrame,
        df_teros12: pd.DataFrame
    ):
    """
    label_dfs_b1(...) -> dfs_b1_labeled
    """
    dfs_b1_labeled = []
    # Iterate through each B1 dataset.
    for df in dfs_b1:
        # Label SPRs and BDs from core and penetrometer measurements for each
        # B1 dataset based date/time, compaction level, and moisture level.
        df_labeled = _label_b1_compaction(
            df_b1=df,
            df_cores=df_cores,
            df_pen=df_pen
        )
        
        # Add labels found from the TEROS12 DF to each B1 dataset.
        df_labeled = _label_b1_wetness(
            df_b1=df,
            df_teros12=df_teros12
        )

        dfs_b1_labeled.append(df_labeled)

    print(f"SUCCESS: Labeled {len(dfs_b1_labeled)} B1 datasets!")

    return dfs_b1_labeled 


def _write_labeled_df(df: pd.DataFrame, path_base: str):
    """
    _write_labeled_df(df)

    Write preprocessed output to a programmatically-generated path.
    """
    label_df = _get_b1_df_label(df=df)
    path_out = ("/").join([path_base, label_df+".csv"])
    print(f"STATUS: Writing {label_df} to {path_out}...")
    _write_dataset(dataset=df, path=label_df)
    print("SUCCESS: DONE.")

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
    (dfs_b1,
     dfs_chipotle,
     df_cores,
     df_pen,
     df_teros12) = load_datasets(
            path_base=path_input_dir,
            to_skip=to_skip
        )

    # Label B1 and Chipotle datasets (or other sensing modes under test).
    dfs_b1_out = label_dfs_b1(
        dfs_b1=dfs_b1,
        df_cores=df_cores,
        df_pen=df_pen,
        df_teros12=df_teros12)

    # Write fully-processed, labeled datasets to files for further processing/
    # learning/training.
    for df in dfs_b1_out:
        _write_labeled_df(df, path_base=path_output_dir)


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
