#!/usr/bin/env python3
"""
step_detector.py

Date:
    17 May 2026

Version:
    0.0.1
"""
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from io import StringIO


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
            print(f"{column}: min={col_min}, max={col_max}")
        except Exception as e:
            print(f"{column}: could not compute min/max ({e})")

def plot_data_columns(
        df: pd.DataFrame,
        n: int,
        start_index: int = 0
    ):
    """
    plot_data_columns(df, n)

    Plot the first n datapoints of each column, side-by-side.

    Args:
        df          (pd.DataFrame)
        n           (int)           Number of datapoints to plot.
        start_index (int)           Starting data entry index.
    """
    # Take the slice based on the starting index.
    # TODO: Check that there are enough frames.
    df_chunk = df.iloc[start_index:start_index + n]

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
            legend=False
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

    # First remove constant columns (from Perplexity).
    #df = df.loc[:, df.nunique(dropna=False) > 1].copy()
    df = df.loc[:, df.var(numeric_only=True) != 0]
    # A couple of columns may remain, so do something special for them.
    """
    columns_to_drop = []
    for column in df.columns:
        c_var = df[column].var()
        if c_var == 0.0:
            columns_to_drop.append(column)
    """

    # Next, delete datapoints which remain 0.0 for a while (i.e. at the
    # start of data collection).
    zero_threshold = 0.9    # Fraction of columns that can be 0.0 to be dropped.
    if not numeric.empty:
        zero_fraction = (numeric == 0.0).mean(axis=1)
        df = df.loc[zero_fraction < zero_threshold].copy()

    return df

def extract_steps_from_df(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    extract_steps_from_df(df)

    Description:
        Isolate individual steps from a recorded dataframe and return a list
        of "step" events for future training.

    Args:
        df      (pd.DataFrame)

    Returns:
        steps   (list[pd.DataFrame])
    """
    # First, trim both empty data columns (in the case of early test data) and
    # long sequences of nothing (as in the beginning).
    df = scrub_df(df)
    print_df_stats(df)
    plot_data_columns(df, 40)

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
    extract_steps_from_df(df)

if __name__ == "__main__":
    step_detector()
