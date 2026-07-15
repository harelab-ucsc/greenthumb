#!/usr/bin/env python3
"""
step_detector.py

Date:
    14 Jul 2026

Version:
    0.1.0
"""
import csv
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from dataclasses import dataclass
from datetime import datetime
from io import StringIO

logger = logging.getLogger("greenthumb")


class B1Step(object):
    """
    B1Step

    Mostly-dataclass for holding information about each B1 step event. Please
    note that step isolation and analysis must be handled elsewhere.
    """
    def __init__(
            self,
            df: pd.DataFrame,
            leg: str,
            idx_step: int,
            trial_label: str,
            ts_end: datetime,
            ts_start: datetime,
        ):
        """
        Args:
            df          (pd.DataFrame)  Pandas DF for step data.
            leg         (str)           [fr,fl,rr,rl].
            idx_step    (int)           Index of step within session/motion.
            trial_label (str)           Name of data collection session.
            ts_end      (datetime)      Last timestamp of step.
            ts_start    (datetime)      First timestamp of step.
        """
        self.df = df
        self.leg = leg
        self.end_effector = end_effector
        self.ts_end = ts_end
        self.ts_start = ts_start

    def write(self, output_dir: str):
        """
        write(output_dir)

        Write contained DF to file based on its properties.
        """
        fname = "-".join([
            self.trial_label,
            self.leg,
            str(self.idx_step)
        ]) + ".csv"
        path = os.path.join([output_dir, fname])
        self.df.to_csv(path, index=False)


# Module-level variables.
_EPSILON = 0.0001
_THRESHOLD_TS_MS_STEP = 50


# Meat.
def _filter_step_by_max_and_leg(df: pd.DataFrame, leg: str) -> tuple[datetime]:
    """
    _filter_step_by_max_and_leg(df) -> ts

    Leg options:    [fr, fl, rr, rl]
    """
    lut_leg = {
            "fr": ["FRKneeQ (rad)", "FRKneedQ (rps)"],
            "fl": ["FLKneeQ (rad)", "FLKneedQ (rps)"],
            "rr": ["RRKneeQ (rad)", "RRKneedQ (rps)"],
            "rl": ["RLKneeQ (rad)", "RLKneedQ (rps)"]
        }
    cols = ["Timestamp (Epoch-UTC-ms)"]
    cols += lut_leg[leg.lower()]

    df_filtered = df[cols]
    ts_filtered = df_filtered.loc[
            df_filtered[lut_leg[leg][1]].abs() < _EPSILON
        ]
    leg_ts = ts_filtered["Timestamp (Epoch-UTC-ms)"].astype("int64").values
    leg_angles = ts_filtered[lut_leg[leg][0]].values

    # Find the time between each step max.
    idx = 0
    delta = 0
    step_ts = []
    step_angles = []
    while (idx < len(ts_filtered) - 1):
        delta = leg_ts[idx+1] - leg_ts[idx]
        if (delta > _THRESHOLD_TS_MS_STEP):
            step_ts.append(leg_ts[idx])
            step_angles.append(leg_angles[idx])
        idx += 1

    return step_ts

def find_steps_by_knee_angle(df: pd.DataFrame) -> dict:
    """
    find_steps_by_knee_angle(df) -> step_event_times

    Returns:
        
    """
    step_dts = []

    # Filter the dataset for some efficiency (maybe).
    cols = [
            "Timestamp (Epoch-UTC-ms)",
            "FRKneeQ (rad)",
            "FLKneeQ (rad)",
            "RRKneeQ (rad)",
            "RLKneeQ (rad)",
            "FRKneedQ (rps)",
            "FLKneedQ (rps)",
            "RRKneedQ (rps)",
            "RLKneedQ (rps)"
        ]
    df_filtered = df[cols]

    # Gather timestamps for each leg.
    fr_step_events = _filter_step_by_max_and_leg(df=df_filtered, leg="fr")
    fl_step_events = _filter_step_by_max_and_leg(df=df_filtered, leg="fl")
    rr_step_events = _filter_step_by_max_and_leg(df=df_filtered, leg="rr")
    rl_step_events = _filter_step_by_max_and_leg(df=df_filtered, leg="rl")

    return {
        "fr": fr_step_events,
        "fl": fl_step_events,
        "rr": rr_step_events,
        "rl": rl_step_events
    }


def find_steps_by_da_dt_max(df: pd.DataFrame) -> tuple[datetime]:
    """
    find_steps_by_da_dt_max(df) -> step_event_times
    """
    logging.warning("Finding steps by (da/dt)_max is WIP.")
    return []

def get_steps_from_b1_df(
        df: pd.DataFrame,
        method: str = "knee_angle"
    ) -> list[B1Step]:
    """
    get_steps_from_b1_df(df) -> steps

    Break a provided DataFrame into individual steps based on B1 proprioceptive
    sensor streams.
    """
    # Start by defining method of finding "step events" within DF:
    #   * knee_angle:   When the knee reach max angle, define this as "contact".
    #   * da_dt_max:    When the local absolute vertical acceleration is
    #                   maximized (TODO).
    #
    # After finding the timestamps of all of these events, define a "step" as
    # the DF containing all data from +/- half of the min time between these
    # events.
    lut_step_isolator = {
            "knee_angle": find_steps_by_knee_angle,
            "da_dt_max": find_steps_by_da_dt_max
        }

    # Gather timestamps for each leg.
    step_events = lut_step_isolator[method](df=df)

    # Convert step events into steps.
    # TODO
    exit()


def print_df_stats(df: pd.DataFrame):
    """
    print_df_stats(df)

    Description:
        Print out statistics related to a provided dataframe, including:
            + Min values (for each column).
            + Max values (for each column).

    Args:
        df  (pd.DataFrame)
    """
    for column in df.columns:
        try:
            col_min = df[column].min()
            col_max = df[column].max()
            logging.info(f"{column}: min={col_min}, max={col_max}")
        except Exception as e:
            logging.error(f"{column}: could not compute min/max ({e})")

def plot_data_columns(
        df: pd.DataFrame,
        n: int,
        cols: list[str] = [],
        start_index: int = 0
    ):
    """
    plot_data_columns(df, n)

    Plot the first n datapoints of each column, side-by-side.

    Args:
        df          (pd.DataFrame)
        n           (int)           Number of datapoints to plot.
        cols        (list[str])     Names of columns desired to plot; if empty,
                                    plot all.
        start_index (int)           Starting data entry index.
    """
    # Take the slice based on the starting index.
    # TODO: Check that there are enough frames.
    df_chunk = df.iloc[start_index:start_index + n]

    # Select only desired columns if provided.
    if len(cols) > 0:
        df_chunk = df_chunk.loc[:, cols]

    # Only keep numeric columns.
    df_chunk = df_chunk.select_dtypes(include="number")

    # Get the total number of rows.
    num_cols = 3
    num_rows = math.ceil(len(df_chunk.columns) / num_cols)

    axes = df_chunk.plot(
            subplots=True,
            layout=(num_rows, num_cols),
            figsize=(5*num_cols, 3*num_rows),
            sharex=True,
            legend=True
        )

    plt.tight_layout()
    plt.show()

def scrub_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    scrub_df(df) -> df

    Description:
        Excise both columns that never change in value and long portions of
        data which remain 0 (i.e. at the beginning).

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

def extract_steps_from_df(df: pd.DataFrame) -> list[B1Step]:
    """
    extract_steps_from_df(df) -> steps

    Description:
        Isolate individual steps from a recorded dataframe and return a list
        of "step" events for future training.

    Args:
        df      (pd.DataFrame)

    Returns:
        steps   (list[B1Step])
    """
    # First, trim both empty data columns (in the case of early test data) and
    # long sequences of nothing (as in the beginning).
    df = scrub_df(df)

    print_df_stats(df)

    cols = [
#            "FRHipdQ (rps)",
#            "FRThighdQ (rps)",
#            "FRKneedQ (rps)",
#            "FLHipdQ (rps)",
#            "FLThighdQ (rps)",
#            "FLKneedQ (rps)",
#            "RLHipdQ (rps)",
#            "RLThighdQ (rps)",
#            "RLKneedQ (rps)",
#            "RRHipdQ (rps)",
#            "RRThighdQ (rps)",
#            "RRKneedQ (rps)"
            "FRHipT (Nm)",
            "FRThighT (Nm)",
            "FRKneeT (Nm)",
            "FLHipT (Nm)",
            "FLThighT (Nm)",
            "FLKneeT (Nm)",
            "RLHipT (Nm)",
            "RLThighT (Nm)",
            "RLKneeT (Nm)",
            "RRHipT (Nm)",
            "RRThighT (Nm)",
            "RRKneeT (Nm)",
            "IMUAccz"
        ]
    plot_data_columns(df=df, cols=cols, n=1000, start_index=10000)

def _read_csv(path: str) -> pd.DataFrame:
    """
    _read_csv(path) -> df

    Description:
        Import, clean, and convert a CSV file into a pandas DataFrame.

    Args:
        path    (str)

    Returns:
        df      (pd.DataFrame)
    """
    # Import the raw CSV file as an array.
    with open(path, "r") as fp:
        data_array = [
            line for i, line in enumerate(fp) if i == 0 or "," in line
        ]
        """
        for line in fp:
            # Filter out invalid rows.
            csv_row = line.split(",")
            if len(csv_row) == 65:
                data_array.append(csv_row)
            else:
                print(csv_row)
        """

    # Convert CSV array into pandas DataFrame.
    df = pd.read_csv(StringIO("".join(data_array)))

    return df

def step_detector():
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <input.csv>")
        sys.exit(1)

    # Convert the input CSV file into a Pandas DataFrame.
    csv_path = sys.argv[1]
    df = _read_csv(csv_path)

    #print_df_stats(df)
    #extract_steps_from_df(df)
    get_steps_from_b1_df(df)

if __name__ == "__main__":
    step_detector()
