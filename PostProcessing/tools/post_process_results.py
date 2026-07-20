"""
File:
    post_process_results.py

Description:
    Analyze the results of training/eval splits for a given set of runs.

Author:
    nubby

Date:
    20 Jul 2026

Version:
    0.0.1
"""
import argparse
import csv
import logging

logger = logging.getLogger("greenthumb")


def _load_csv(path: str) -> dict:
    """
    _load_csv(path) -> data
    """
    data = {}
    with open(path) as cfp:
        reader = csv.DictReader(cfp)
        for row in reader:
            for key, val in row.items():
                if key not in data.keys():
                    data[key] = []
                data[key].append(val)
    return data 

def split_alt_and_classic_runs(data: dict) -> tuple[dict, dict]:
    """
    split_alt_and_classic_runs(data) -> alt, classic
    """
    data_alt = {}
    data_classic = {}

    # Populate each dict with keys and empty arrays to start.
    for key in data.keys():
        data_alt[key] = []
        data_classic[key] = []

    # Find all classic entries and alternate entries.
    n_runs = len(data["Timestamp"])
    i = 0
    while (i < n_runs):
        if data["UseClassicMode"][i]:
            for key in data_classic.keys():
                data_classic[key].append(data[key][i])
        else:
            for key in data_alt.keys():
                data_alt[key].append(data[key][i])
        i += 1
    
    return data_alt, data_classic

def post_process_results(path_results: str):
    """
    post_process_results(path_results)
    """
    # First load all results into a dict.
    data_all = _load_csv(path=path_results)

    # Separate "classic mode" and "alternate" runs.
    data_alt, data_classic = split_alt_and_classic_runs(data=data_all)
    print(data_alt)
    exit()

    # Then get basic stats for each set's columns.

if __name__ == "__main__":
    # Logger setup.
    logging.basicConfig(level=logging.INFO)

    # Arg parsing.
    parser = argparse.ArgumentParser(
            description="Digest run results."
        )

    parser.add_argument(
            "--path-results",
            type=str,
            default="artifacts/results.csv",
            help="Path to the file containing run results."
        )
    args = parser.parse_args()

    post_process_results(path_results=args.path_results)
