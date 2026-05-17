#!/usr/bin/env python3
"""
step_detector.py

Date:
    11 May 2026

Version:
    0.0.1
"""
import sys
import pandas as pd


def print_df_stats(df: pd.DataFrame):
    for column in df.columns:
        try:
            col_min = df[column].min()
            col_max = df[column].max()
            print(f"{column}: min={col_min}, max={col_max}")
        except Exception as e:
            print(f"{column}: could not compute min/max ({e})")

def step_detector():
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <input.csv>")
        sys.exit(1)

    # Convert the input CSV file into a Pandas DataFrame.
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    print_df_stats(df)

if __name__ == "__main__":
    step_detector()
