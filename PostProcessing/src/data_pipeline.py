"""
data_pipeline.py

Author:
    nubby
    Taylor Kergan

Date:
    4 Aug 2026

Version:
    1.0.8
"""
from __future__ import annotations

from dataclasses import dataclass

import glob
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import warnings

from step_detector import get_steps_from_b1_df


# NOTE / TODO: Currently, each "combined.csv" dataset needs to be:
#   1. Run through the preprocessing pipeline (after adding an "eval_id" column
#       and making sure that column labels match the below) to produce your
#       "combined.csv" file as usual.
#   2. Enter each file and divide each dataset into at least 10 parts. This is
#       due to the pipeline currently requiring at least 3 evals for each
#       dataset, and it expects 10 for some reason. <-- Figure this out.
# Feature definitions lifted from the CSV headers after stripping whitespace.
IMU_FEATURES_QCAT: Sequence[str] = (
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
    "lin_acc_x",
    "lin_acc_y",
    "lin_acc_z",
    "ori_x",
    "ori_y",
    "ori_z",
    "ori_w",
)

"""
    "forwardSpeed (m/s)",
    "sideSpeed (m/s)",
    "rotateSpeed (m?/s)",
    "yawSpeed(rad/s)",
"""
# Feature definitions from B1 for now.
FEATURES_IMU: Sequence[str] = (
        "IMUQw",
        "IMUQx",
        "IMUQy",
        "IMUQz",
        "IMUAccx",
        "IMUAccy",
        "IMUAccz"
    )

# Ground reaction force features.
FORCE_FEATURES: Sequence[str] = (
        "FL_x (N)",
        "FL_y (N)",
        "FL_z (N)",
        "FR_x (N)",
        "FR_y (N)",
        "FR_z (N)",
        "BR_x (N)",
        "BR_y (N)",
        "BR_z (N)",
        "BL_x (N)",
        "BL_y (N)",
        "BL_z (N)",
    )

# Joint angle features.
FEATURES_JANGLE: Sequence[str] = (
        "FRHipQ (rad)",
        "FRThighQ (rad)",
        "FRKneeQ (rad)",
        "FLHipQ (rad)",
        "FLThighQ (rad)",
        "FLKneeQ (rad)",
        "RRHipQ (rad)",
        "RRThighQ (rad)",
        "RRKneeQ (rad)",
        "RLHipQ (rad)",
        "RLThighQ (rad)",
        "RLKneeQ (rad)",
    )

# Joint angle velocity features.
FEATURES_DJANGLE: Sequence[str] = (
        "FRHipdQ (rps)",
        "FRThighdQ (rps)",
        "FRKneedQ (rps)",
        "FLHipdQ (rps)",
        "FLThighdQ (rps)",
        "FLKneedQ (rps)",
        "RRHipdQ (rps)",
        "RRThighdQ (rps)",
        "RRKneedQ (rps)",
        "RLHipdQ (rps)",
        "RLThighdQ (rps)",
        "RLKneedQ (rps)",
    )

# Joint angle acceleration features.
FEATURES_D2JANGLE: Sequence[str] = (
        "FRHipd2Q (rps^2)",
        "FRThighd2Q (rps^2)",
        "FRKneed2Q (rps^2)",
        "FLHipd2Q (rps^2)",
        "FLThighd2Q (rps^2)",
        "FLKneed2Q (rps^2)",
        "RRHipd2Q (rps^2)",
        "RRThighd2Q (rps^2)",
        "RRKneed2Q (rps^2)",
        "RLHipd2Q (rps^2)",
        "RLThighd2Q (rps^2)",
        "RLKneed2Q (rps^2)",
    )

# Joint torque features.
FEATURES_JTORQUE: Sequence[str] = (
        "FRHipT (Nm)",
        "FRThighT (Nm)",
        "FRKneeT (Nm)",
        "FLHipT (Nm)",
        "FLThighT (Nm)",
        "FLKneeT (Nm)",
        "RRHipT (Nm)",
        "RRThighT (Nm)",
        "RRKneeT (Nm)",
        "RLHipT (Nm)",
        "RLThighT (Nm)",
        "RLKneeT (Nm)",
    )

# VWC features from TEROS-12 sensors.
FEATURES_VWC: Sequence[str] = (
        "Est VWC (%-4in)",
        "Est VWC (%-7in)",
        "Est VWC (%-10in)"
    )

# Labels.
LABELS_SBD: Sequence[str] = (
        "SBD (g/mL-avg-0in)",
        #"SBD (g/mL-avg-4in)",
        #"SBD (g/mL-avg-7in)",
    )
LABELS_SPR: Sequence[str] = (
        "SPR (PSI-avg-3in)",
        #"SPR (PSI-avg-4in)",
        #"SPR (PSI-avg-7in)",
    )

@dataclass(frozen=True)
class SampleMetadata:
    """
    SampleMetadata

    Lightweight description of where each sequence originated.

    Members:
        idx_compaction  (int)   Number of compaction events completed.
        idx_wetness     (int)   Number of wetting events completed.
        step            (int)
        trial           (str)   Unique trial name.
        length          (int)
        window          (int)
    """
    idx_compaction: int
    idx_wetness: int
    length: int
    step: int
    trial: str
    window: int = 0  # Index of the sliding window within the step.


@dataclass
class RawSequenceDataset:
    """
    RawSequenceDataset

    Holds the un-normalised sensor sequences before splitting.
    """
    sequences: List[np.ndarray]
    labels: np.ndarray
    lengths: np.ndarray
    metadata: List[SampleMetadata]
    feature_names: Sequence[str]

    def __len__(self) -> int:
        return len(self.sequences)


class SequenceDataset(Dataset):
    """
    SequenceDataset(Dataset)

    Torch dataset that stores variable-length sequences along with metadata.
    """
    def __init__(
        self,
        sequences: List[torch.Tensor],
        labels: torch.Tensor,
        lengths: torch.Tensor,
        metadata: List[SampleMetadata],
    ) -> None:
        if not (len(sequences) == len(labels) == len(lengths) == len(metadata)):
            raise ValueError("Dataset inputs must have matching lengths.")
        self._sequences = sequences
        self._labels = labels
        self._lengths = lengths
        self._metadata = metadata

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            SampleMetadata]:
        return (
            self._sequences[idx],
            self._lengths[idx],
            self._labels[idx],
            self._metadata[idx],
        )

    @staticmethod
    def collate_fn(
        batch: Sequence[Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            SampleMetadata
        ]]
    ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Sequence[SampleMetadata]
        ]:
        """Pads variable-length sequences per batch."""
        sequences, lengths, labels, metadata = zip(*batch)
        padded = pad_sequence(sequences, batch_first=True)  # shape (B, T_max, F)
        lengths_tensor = torch.stack(lengths, dim=0)
        labels_tensor = torch.stack(labels, dim=0)
        metadata = list(metadata)
        return padded, lengths_tensor, labels_tensor, metadata


@dataclass
class FullDataBundle:
    """
    FullDataBundle

    Container for train/val/test datasets and their shared statistics.
    """
    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    feature_names: Sequence[str]
    train_mean: float
    train_std: float

    def dataloader(
        self,
        split: str,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True
    ) -> DataLoader:
        dataset = {
                "train": self.train,
                "val": self.val,
                "test": self.test
            }[split]
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle if split == "train" else False,
            num_workers=num_workers,
            collate_fn=SequenceDataset.collate_fn,
        )


def _read_sensor_csv(path: Path) -> pd.DataFrame:
    """Loads a sensor CSV and strips whitespace from column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "eval_id" not in df.columns:
        raise ValueError(f"CSV {path} is missing expected 'eval_id' column.")
    return df.sort_values("time (ms)").reset_index(drop=True)

def _get_df_labels(df: pd.DataFrame, target: str) -> Tuple[
        Tuple[Tuple[float], Tuple[float]],
        int, int, str
    ]:
    """
    _get_df_labels(df) -> labels, idx_compaction, idx_wetness, trial

    Returns:
        labels          (tuple[tuple(float), tuple(float)]) [SBD, SPR] labels.
        idx_compaction  (int)
        idx_wetness     (int)
        trial           (str)                               Unique trial name.
        target          (str)                               ["spr", "sbd"]
    """
    # Group all label features.
    target_lut = {
            "sbd": list(LABELS_SBD),
            "spr": list(LABELS_SPR)
        }

    # Extract only the first label for each column (same throughout).
    try:
        labels = df[target_lut[target]].to_numpy(dtype=np.float32)[0]
    except KeyError as e:
        print(e)
        print(f"ERROR: Could not find column, {target} for use as target.")
        exit()

    # Extract compaction, wetness, and trial label for the DF.
    idx_compaction = df["Compaction Level (index)"].astype(int).values[0]
    idx_wetness = df["SWC Level (index)"].astype(int).values[0]
    trial = df["Dataset Label"].astype(str).values[0]

    return labels, idx_compaction, idx_wetness, trial

def _segment_trial_by_steps(
        df: pd.DataFrame,
        num_steps: int,
        step_len: int,
        trial_label: str,
        wet: bool = True
    ) -> Tuple[List[np.ndarray], int]:
    """
    _segment_trial_by_steps(df, num_steps) -> (segments, step_len)

    Aligns a trial by truncating to the shared number of steps.

    Note:
        Is this wise? What happens if each trial has different numbers of steps?
    """
    # TODO: Add back windowing.
    """
    min_len = len(df)
    if min_len < num_steps:
        raise ValueError(
            f"Trial too short ({min_len}) to segment into {num_steps} steps."
        )
    seg_len = min_len // num_steps
    usable_len = seg_len * num_steps
    """

    # Split trial's DF into smaller DFs for each step.
    steps = get_steps_from_b1_df(
            df=df,
            trial_label=trial_label,
            step_len=step_len
        )

    # Only include desired features (omit timestamps, etc).
    cols = []
    cols += list(FEATURES_IMU)
    cols += list(FEATURES_JTORQUE)
    cols += list(FEATURES_JANGLE)
    cols += list(FEATURES_DJANGLE)
    cols += list(FEATURES_D2JANGLE)
    if wet:
        cols += list(FEATURES_VWC)

    dfs_steps = [step.df[cols].to_numpy() for step in steps]
    
    # Select number of steps based on the argument provided.
    if (num_steps == -1 or len(steps) <= num_steps):
        # Use all found steps if specified or not enough are available.
        return dfs_steps, step_len

    # Else, return the specified number of steps.
    return dfs_steps[:num_steps], step_len

def _window_segment(
        segment: np.ndarray,
        window_size: int | None,
        stride: int | None
    ) -> List[np.ndarray]:
    """
    _window_segment(segment, window_size, stride) -> windows

    Generates sliding windows for a single step segment.
    """
    if window_size is None:
        return [segment]
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    window = min(window_size, len(segment))
    hop = stride if stride is not None else window
    if hop <= 0:
        raise ValueError("stride must be a positive integer when provided.")

    windows: List[np.ndarray] = []
    for start in range(0, len(segment) - window + 1, hop):
        windows.append(segment[start : start + window])

    if not windows:
        windows.append(segment[-window:])
    return windows

def _load_raw_b1_dataset(
        data_file_paths: list[str],
        num_trials: int = 10,
        num_steps: int = 8,
        step_len: int = 1000,
        stride: int | None = None,
        target: str = "spr",
        wet: bool = True,
        window_size: int | None = None,
    ) -> Tuple[
            List[np.ndarray],
            List[int],
            List[np.ndarray],
            List[SampleMetadata]
        ]:
    """
    _load_raw_b1_dataset(...) -> (sequences, labels, lengths, metadata)

    Load selected B1 datasets with SPR/SBD labels.
    """
    sequences: List[np.ndarray] = []
    labels: List[float] = []
    lengths: List[int] = []
    metadata: List[SampleMetadata] = []

    # TODO: Re-encode this framework to pair temporally aligned SPR values with
    #       other sensor data (and GPS).
    for path in data_file_paths:
        # Load DFs from each trial.
        df = pd.read_csv(path)

        # Get labels from DFs.
        these_labels, idx_compaction, idx_wetness, trial = _get_df_labels(
                df=df,
                target=target
            )
        
        # Split DFs into temporal data segments.
        segments, segment_len = _segment_trial_by_steps(
                df=df,
                num_steps=num_steps,
                step_len=step_len,
                trial_label=trial,
                wet=wet
            )

        # NOTE: "step" == "segement".
        for step_idx, segment in enumerate(segments):
            # Now we divide each step/segment into rolling windows.
            windows = _window_segment(segment, window_size, stride)
            for window_idx, window_segment in enumerate(windows):
                # Add each window segment with paired labels.
                sequences.append(window_segment)
                labels.append(these_labels)
                lengths.append(len(window_segment))
                metadata.append(
                    SampleMetadata(
                        idx_compaction=idx_compaction,
                        idx_wetness=idx_wetness,
                        length=len(window_segment),
                        step=step_idx,
                        trial=trial,
                        window=window_idx
                    )
                )

    return (sequences, labels, lengths, metadata)

def _ls_data_file_paths(
        data_dir: Path
    ) -> list[str]:
    """
    _ls_data_file_paths(data_dir) -> data_file_paths
    """
    paths = []
    with os.scandir(data_dir) as dp:
        for path in dp:
            if (path.is_file() and path.name.endswith(".csv")):
                paths.append(path)

    # Raise an error if no data files found.
    if len(paths) == 0:
        raise FileNotFoundError

    return paths

def load_raw_dataset(
        data_dir: str,
        num_trials: int = 10,
        num_steps: int = 8,
        step_len: int = 1000,
        stride: int | None = None,
        target: str = "spr",
        wet: bool = True,
        window_size: int | None = None,
    ) -> RawSequenceDataset:
    """
    load_raw_dataset(...) -> RawSequenceDataset

    Parses and aligns the CSV files into per-step sequences of IMU quaternions,
    joint angles, and joint torques.
    """
    # TODO: Step length and window size are the same currently (under most
    #       conditions.
    sequences: List[np.ndarray] = []
    labels: List[List[int]] = []
    lengths: List[int] = []
    metadata: List[SampleMetadata] = []
    feature_names: List[str] = []

    # First load available datasets from the selected directory.
    data_file_paths = _ls_data_file_paths(data_dir=data_dir)

    # Shape feature vector and load data based on selections.
    raw = _load_raw_b1_dataset(
            data_file_paths=data_file_paths,
            num_trials=num_trials,
            num_steps=num_steps,
            step_len=step_len,
            stride=stride,
            target=target,
            wet=wet,
            window_size=window_size,
        )
    sequences += raw[0]
    labels += raw[1]
    lengths += raw[2]
    metadata += raw[3]

    # Create a full list of feature names.
    # TODO(nubby): Can this be merged in earlier for efficiency?
    feature_names += list(FEATURES_JTORQUE)
    feature_names += list(FEATURES_JANGLE)
    feature_names += list(FEATURES_DJANGLE)
    feature_names += list(FEATURES_D2JANGLE)
    if wet:
        feature_names += list(FEATURES_VWC)
    feature_names += list(FEATURES_IMU)

    #labels=np.asarray(labels, dtype=np.int64),

    return RawSequenceDataset(
        sequences=sequences,
        labels=np.array(labels),
        lengths=np.asarray(lengths, dtype=np.int64),
        metadata=metadata,
        feature_names=feature_names
    )

def _compute_split_counts(
        n_trials: int,
        train_frac: float,
        val_frac: float
    ) -> Tuple[int, int, int]:
    """
    _compute_split_counts(n_trials, train_frac, val_frac) -> (train, val, test)

    Computes integer trial counts per split with basic safeguards.
    Set "val_frac" = 0 for leave-one-out cross validation.
    """
    if n_trials <= 2:
        raise ValueError(
                "Need at least three trials to form train/val/test splits."
            )
    val = int(round(n_trials * val_frac))
    if val == 0:
        train = n_trials - 1
    else:
        train = max(1, int(round(n_trials * train_frac)))
    if train + val >= n_trials:
        train = max(1, n_trials - 2)
        val = 1
    test = n_trials - train - val
    if test <= 0:
        if val > 1:
            val -= 1
        else:
            train = max(1, train - 1)
        test = n_trials - train - val
    return train, val, test

def _assign_trial_splits(
        metadata: Sequence[SampleMetadata],
        train_frac: float,
        val_frac: float,
        rng: np.random.Generator,
    ) -> Tuple[Dict[Tuple[int, int], str]]:
    """
    _assign_trial_splits(metadata, train_frac, val_frac, rng) -> assigned_splits

    Assigns each (compaction, wetness) tuple to a split (train, val, test).
    Set "val_frac" = 0 to use leave-one-out-cross validation.

    """
    # TODO(nubby): Consider splitting up trials into subtrials to increase
    #               number of datapoints available for training. NOTE that this
    #               would also split up temporally linked datasets, which may
    #               be undesireable.
    # First group trials under compaction events and soil wetness events.
    by_step = True      # Set to True to make splits by number of steps.
    by_scene = False    # Set to True to group datasets from the same scene.
    by_key: Dict[Tuple[int, int], List[int]] = {}
    for meta in metadata:
        key = (meta.idx_compaction, meta.idx_wetness)
        # Prevent raising of exceptions when key not present.
        by_key.setdefault(key, [])
        if meta.trial not in by_key[key]:
            by_key[key].append(meta.trial)

    # Intelligently divide trials by number of steps in each scenario?
    division: Dict[Tuple[int, int], int] = {}
    n_scenes = len(by_key.keys())
    for key in by_key.keys():
        # Get the number of total entries for each key.
        n = len([
            meta for meta in metadata if key == (
                meta.idx_compaction, meta.idx_wetness
            )])
        division[key] = n
    
    # From total number of steps, find number of steps to provide for each
    # split.
    """
    total_steps = sum([division[key] for key in division.keys()])
    train_steps = np.ceil(total_steps * train_frac)
    val_steps = np.floor(total_steps * val_frac)
    """
    if val_frac == 0:
        assignments = []
        assignment: Dict[Tuple[int, int, str], str] = {}
        # For LOOCV, train and evaluate on every possible scene config.
        for idx in range(n_scenes):
            keys = list(by_key.keys())
            test_key = keys[idx]
            test_trials = by_key[test_key]
            for trial in test_trials:
                assignment[(test_key[0], test_key[1], trial)] = "test"
            for key, trials in by_key.items():
                if key != test_key:
                    for trial in trials:
                        assignment[(key[0], key[1], trial)] = "train"
            # Add this assignment.
            assignments.append(assignment)
        return assignments
    else:
        # Assign trials from each compaction/wetness level randomly to splits.
        assignment: Dict[Tuple[int, int, str], str] = {}
        if not by_scene:
            for key, trials in by_key.items():
                # Randomly shuffle trials for each key value, then split accordingly.
                trials_copy = list(trials)
                rng.shuffle(trials_copy)
                train_count, val_count, _ = _compute_split_counts(
                        len(trials_copy), train_frac, val_frac
                    )
                train_trials = set(trials_copy[:train_count])
                val_trials = set(trials_copy[train_count : train_count + val_count])
                for trial in trials:
                    if trial in train_trials:
                        assignment[(key[0], key[1], trial)] = "train"
                    elif trial in val_trials:
                        assignment[(key[0], key[1], trial)] = "val"
                    else:
                        assignment[(key[0], key[1], trial)] = "test"
        else:
            # If grouping by scenario (indexed by (SWC, compaction index)), assign
            # by key rather than by trial.
            keys = list(by_key.keys())
            # Randomly shuffle keys (each "scenario"), then split accordingly.
            rng.shuffle(keys)
            train_count, val_count, _ = _compute_split_counts(
                    len(keys), train_frac, val_frac
                )
            train_keys = set(keys[:train_count])
            val_keys = set(keys[train_count : train_count + val_count])
            for key, trials in by_key.items():
                if key in train_keys:
                    for trial in trials:
                        assignment[(key[0], key[1], trial)] = "train"
                elif key in val_keys:
                    for trial in trials:
                        assignment[(key[0], key[1], trial)] = "val"
                else:
                    for trial in trials:
                        assignment[(key[0], key[1], trial)] = "test"
        return [assignment]

def _compute_feature_stats(
        sequences: Sequence[np.ndarray],
        indices: Iterable[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    _compute_feature_stats(sequences, indices) -> [mean, std]

    Derives mean/std from the provided subset of a sequence for training.
    """
    stacked = np.concatenate([sequences[i] for i in indices], axis=0)
    mean = stacked.mean(axis=0, dtype=np.float64)
    std = stacked.std(axis=0, dtype=np.float64)
    # Replace extremely low STDs with 1.0 to avoid division by 0.
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def _compute_label_stats(
        labels: Sequence[np.ndarray],
        indices: Iterable[int]
        ) -> Tuple[float, float]:
    """
    _compute_label_stats(labels, indices) -> [mean, std]

    Derives mean/std from the provided labels. Can be thought of as fewer dims
    than _compute_feature_stats() above.
    """
    stacked = np.concatenate([labels[i] for i in indices], axis=0)
    mean = stacked.mean(axis=0, dtype=np.float64)
    std = stacked.std(axis=0, dtype=np.float64)
    # Replace extremely low STDs with 1.0 to avoid division by 0.
    if (std < 1e-6):
        std = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def build_data_bundles(
        data_dir: str,
        num_steps: int,
        seed: int = 13,
        stride: int | None = None,
        target: str = "spr",
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        wet: bool = True,
        window_size: int | None = None
    ) -> Tuple[FullDataBundle, Dict]:
    """
    build_data_bundle(...) -> FullDataBundle

    Loads raw data, performs splits, and returns ready-to-use torch datasets.
    Data/Features are not yet selectable, but targets can be selected as either:
        + SPR 
        + SBD

    Todo:
        * Make SWC as a label/feature selectable.
    """
    # Default step length to 1000 samples.
    step_len = window_size if window_size else 1000

    # Start by loading the selected datasets.
    raw = load_raw_dataset(
        data_dir=data_dir,
        num_steps=num_steps,
        step_len=step_len,
        stride=stride,
        target=target.lower(),
        wet=wet,
        window_size=window_size
    )

    # Random seed-based assignment for replicability.
    rng = np.random.default_rng(seed)

    # Randomly-split trials into (train, val, test) splits based on trial
    # labels.
    assignments = _assign_trial_splits(
        metadata=raw.metadata,
        train_frac=train_frac,
        val_frac=val_frac,
        rng=rng
    )

    bundles = []
    # Lightweight reference structure using integer indices rather than unique
    # metadata entries.
    for assignment in assignments:
        split_indices: Dict[str, List[int]] = {
                "train": [], "val": [], "test": []
            }
        for idx, meta in enumerate(raw.metadata):
            split = assignment[
                    (meta.idx_compaction, meta.idx_wetness, meta.trial)
                ]
            split_indices[split].append(idx)

        # Generate stats from training dataset for use in normalization.
        feature_mean_np, feature_std_np = _compute_feature_stats(
            raw.sequences,
            split_indices["train"]
        )
        labels_mean, labels_std = _compute_label_stats(
            labels=raw.labels,
            indices=split_indices["train"]
        )

        def make_dataset(indices: List[int]) -> SequenceDataset:
            """
            make_dataset(indices) -> SequenceDataset

            Shape formatted sequences into tensors for ingestion into torch by
            assigned split indices.
            """
            # Return "None" for validation set if using leave-one-out.
            if len(indices) == 0:
                return None
            seq_tensors: List[torch.Tensor] = []
            lengths_tensors: List[torch.Tensor] = []
            labels_tensors: List[torch.Tensor] = []
            metadata_subset: List[SampleMetadata] = []

            for idx in indices:
                seq = raw.sequences[idx]
                label = raw.labels[idx]
                # First normalize each sequence based on the training dataset stats.
                # TODO(nubby): Is this just the z-score?
                norm_seq = (seq - feature_mean_np) / feature_std_np
                seq_tensors.append(torch.from_numpy(norm_seq.astype(np.float32)))
                lengths_tensors.append(
                    torch.tensor(raw.lengths[idx], dtype=torch.long)
                )
                # Next normalize labels here.
                norm_label = (label - labels_mean) / labels_std
                labels_tensors.append(
                    torch.tensor(norm_label, dtype=torch.float)
                )
                metadata_subset.append(raw.metadata[idx])

            return SequenceDataset(
                sequences=seq_tensors,
                labels=torch.stack(labels_tensors),
                lengths=torch.stack(lengths_tensors),
                metadata=metadata_subset
            )

        # Convert splits into SequenceDataset for ingestion by torch.
        train_ds = make_dataset(split_indices["train"])
        val_ds = make_dataset(split_indices["val"])
        test_ds = make_dataset(split_indices["test"])

        # Generate dataset-scale statistics.
        feature_mean = torch.from_numpy(feature_mean_np)
        feature_std = torch.from_numpy(feature_std_np)

        bundles.append(FullDataBundle(
                train=train_ds,
                val=val_ds,
                test=test_ds,
                feature_mean=feature_mean,
                feature_std=feature_std,
                feature_names=raw.feature_names,
                train_mean=labels_mean,
                train_std=labels_std,
            ))

    return bundles, assignments
