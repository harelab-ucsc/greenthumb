"""
data_pipeline.py

Author:
    nubby
    Taylor Kergan

Date:
    1 Jul 2026

Version:
    1.0.2
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import warnings


spr_max = 1000

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
IMU_FEATURES_B1: Sequence[str] = (
        "IMUQw",
        "IMUQx",
        "IMUQy",
        "IMUQz"
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
JANGLE_FEATURES: Sequence[str] = (
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

# Joint torque features.
JTORQUE_FEATURES: Sequence[str] = (
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

# Labels.
LABELS: Sequence[str] = (
        "SBD (g/mL-avg-0in)",
        #"SBD (g/mL-avg-4in)",
        #"SBD (g/mL-avg-7in)",
        "SPR (PSI-avg-3in)",
        #"SPR (PSI-avg-4in)",
        #"SPR (PSI-avg-7in)",
        "Est VWC (%-4in)",
        "Est VWC (%-7in)",
        "Est VWC (%-10in)"
    )


@dataclass(frozen=True)
class SampleMetadata:
    """Lightweight description of where each sequence originated."""

    terrain: int
    speed: int
    trial: int
    step: int
    length: int
    window: int = 0  # index of the sliding window within the step


@dataclass
class RawSequenceDataset:
    """Holds the un-normalised sensor sequences before splitting."""

    sequences: List[np.ndarray]
    labels: np.ndarray
    lengths: np.ndarray
    metadata: List[SampleMetadata]
    feature_names: Sequence[str]

    def __len__(self) -> int:
        return len(self.sequences)


class SequenceDataset(Dataset):
    """Torch dataset that stores variable-length sequences along with metadata."""

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SampleMetadata]:
        return (
            self._sequences[idx],
            self._lengths[idx],
            self._labels[idx],
            self._metadata[idx],
        )

    @staticmethod
    def collate_fn(
        batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SampleMetadata]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[SampleMetadata]]:
        """Pads variable-length sequences per batch."""
        sequences, lengths, labels, metadata = zip(*batch)
        padded = pad_sequence(sequences, batch_first=True)  # shape (B, T_max, F)
        lengths_tensor = torch.stack(lengths, dim=0)
        labels_tensor = torch.stack(labels, dim=0)
        return padded, lengths_tensor, labels_tensor, metadata


@dataclass
class TerrainDataBundle:
    """Container for train/val/test datasets and their shared statistics."""

    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    feature_names: Sequence[str]

    def dataloader(
        self, split: str, batch_size: int, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        dataset = {"train": self.train, "val": self.val, "test": self.test}[split]
        return DataLoader(
            dataset,
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


# TODO(nubby): Add joint torques.
def _segment_trial_imu_force(
    imu_trial: pd.DataFrame,
    force_trial: pd.DataFrame,
    num_steps: int,
) -> Tuple[List[np.ndarray], int]:
    """Aligns an IMU and force trial by truncating to the shared length."""
    min_len = min(len(imu_trial), len(force_trial))
    if min_len < num_steps:
        raise ValueError(f"Trial too short ({min_len}) to segment into {num_steps} steps.")
    step_len = min_len // num_steps
    usable_len = step_len * num_steps

    imu_cols = list(IMU_FEATURES_QCAT)
    force_cols = list(FORCE_FEATURES)
    imu_values = imu_trial.loc[: usable_len - 1, imu_cols].to_numpy(dtype=np.float32)
    force_values = force_trial.loc[: usable_len - 1, force_cols].to_numpy(dtype=np.float32)
    #jtorque_values = jtorque_trial.loc[: usable_len - 1, jtorque_cols].to_numpy(dtype=np.float32)
    combined = np.concatenate([
        imu_values,
        force_values,
    ], axis=1)

    segments: List[np.ndarray] = []
    for step_idx in range(num_steps):
        start = step_idx * step_len
        end = start + step_len
        segments.append(combined[start:end])
    return segments, step_len

def _segment_trial_combined(
    combined_trial: pd.DataFrame,
    num_steps: int,
) -> Tuple[List[np.ndarray], int]:
    """Aligns a trial by truncating to the shared length."""
    # TODO: This is basically just bloatware RN.
    min_len = len(combined_trial)
    if min_len < num_steps:
        raise ValueError(f"Trial too short ({min_len}) to segment into {num_steps} steps.")
    step_len = min_len // num_steps
    usable_len = step_len * num_steps

    imu_cols = list(IMU_FEATURES_B1)
    jtorque_cols = list(JTORQUE_FEATURES)
    imu_values = combined_trial.loc[: usable_len - 1, imu_cols].to_numpy(dtype=np.float32)
    jtorque_values = combined_trial.loc[: usable_len - 1, jtorque_cols].to_numpy(dtype=np.float32)
    combined = np.concatenate([
        imu_values,
        jtorque_values,
    ], axis=1)

    segments: List[np.ndarray] = []
    for step_idx in range(num_steps):
        start = step_idx * step_len
        end = start + step_len
        segments.append(combined[start:end])
    return segments, step_len


def _window_segment(
    segment: np.ndarray,
    window_size: int | None,
    stride: int | None,
) -> List[np.ndarray]:
    """Generates sliding windows for a single step segment."""
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

def _load_raw_qcat_dataset(
        terrains: List[str],
        data_dir: Path,
        num_trials: int = 10,
        num_steps: int = 8,
        window_size: int | None = None,
        stride: int | None = None,
    ) -> tuple:
    """
    _load_raw_qcat_dataset(...) -> tuple

    Load selected QCAT datasets with SPR labels.
    """
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    lengths: List[int] = []
    metadata: List[SampleMetadata] = []

    # Create a LUT for decoding and labeling QCAT datasets with SPR estimates.
    # Each SPR listed has linearly-separated depths of 3" * (index + 1) deep.
    spr = lambda spr0, d, r : spr0 + d*r
    QCAT_LABELS_LUT = {
        "concrete": {
            "index": 0,
            "SPR": [1, 1, 1]
        },
        "grass": {
            "index": 1,
            "SPR": [
                0.2395,
                0.2390,
                0.336
            ]
        },
        "gravel": {
            "index": 2,
            "SPR": [0, 0, 0]
        },
        "mulch": {
            "index": 3,
            "SPR": [0, 0, 0]
        },
        "dirt": {
            "index": 4,
            "SPR": [
                spr(0.200, 0, 0.100),
                spr(0.200, 1, 0.100),
                spr(0.200, 2, 0.100)
            ]
        },
        "sand": {
            "index": 5,
            "SPR": [    # Using averages from Natural Bridges State Beach.
                0.0655,
                0.1570,
                0.2565
            ]
        },
    }
    speeds = (1, 2, 3, 4, 5, 6) # Gather data from all speeds.

    # Load datasets from each trial for the desired terrains.
    for terrain in terrains:
        t_index = QCAT_LABELS_LUT[terrain]["index"]
        terrain_sprs = np.array(QCAT_LABELS_LUT[terrain]["SPR"]) / spr_max # Norm.
        #terrain_sprs = np.array((QCAT_LABELS_LUT[terrain]["SPR"])
        #terrain_sprs = np.array(np.log(QCAT_LABELS_LUT[terrain]["SPR"]))
        for speed in speeds:
            imu_path = data_dir / f"{t_index}_{speed}_legSensors_imu.csv"
            force_path = data_dir / f"{t_index}_{speed}_legSensors_raw.csv"
            if not imu_path.exists() or not force_path.exists():
                raise FileNotFoundError(f"Missing sensor files for terrain "
                                        f"{terrain}, speed {speed}.")

            imu_df = _read_sensor_csv(imu_path)
            force_df = _read_sensor_csv(force_path)

            imu_trials = sorted(imu_df["eval_id"].unique())
            force_trials = sorted(force_df["eval_id"].unique())
            common_trials = [
                trial for trial in imu_trials if trial in force_trials
            ]

            if len(common_trials) != num_trials:
                warnings.warn(
                    f"Expected {num_trials} trials but found "
                    f"{len(common_trials)} for terrain {terrain}, speed "
                    f"{speed}. Proceeding with shared trials only."
                )

            for trial in common_trials:
                imu_trial = imu_df[
                    imu_df["eval_id"] == trial
                ].reset_index(drop=True)
                force_trial = force_df[
                    force_df["eval_id"] == trial
                ].reset_index(drop=True)
                segments, step_len = _segment_trial_imu_force(
                    imu_trial,
                    force_trial,
                    num_steps
                )

                for step_idx, segment in enumerate(segments):
                    windows = _window_segment(segment, window_size, stride)
                    for window_idx, window_segment in enumerate(windows):
                        sequences.append(window_segment)
                        labels.append(np.asarray(terrain_sprs))
                        """
                        label = torch.zeros_like(torch.Tensor(terrain_sprs))
                        label[(terrain_sprs >= 200) & (terrain_sprs < 300)] = 1
                        label[terrain_sprs >= 300 ]= 2
                        labels.append(label)
                        """
                        lengths.append(len(window_segment))
                        metadata.append(
                            SampleMetadata(
                                terrain=t_index,
                                speed=speed,
                                trial=int(trial),
                                step=step_idx,
                                length=len(window_segment),
                                window=window_idx,
                            )
                        )
    return (sequences, labels, lengths, metadata)

def _load_raw_b1_dataset(
        data_dir: Path,
        num_trials: int = 10,
        num_steps: int = 8,
        window_size: int | None = None,
        stride: int | None = None,
    ) -> tuple:
    """
    _load_raw_b1_dataset(...) -> (sequences, labels, lengths, metadata)

    Load selected B1 datasets with SPR/SBD labels.
    """
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    lengths: List[int] = []
    metadata: List[SampleMetadata] = []

    # Load datasets from each trial.
    # TODO: Re-encode this framework to pair temporally aligned SPR values with
    #       other sensor data (and GPS).
    # TODO(nubby, 7/1/2026): Reformat this to use all data in Data/ folder.
    for terrain in terrains:
        combined_path = data_dir / f"{t_index}_combined.csv"
        if not combined_path.exists():
            raise FileNotFoundError(f"Missing sensor files for terrain "
                                    f"{terrain}.")
        # Read B1 sensor file with IMU and joint torques.
        combined_df = _read_sensor_csv(combined_path)

        combined_trials = sorted(combined_df["eval_id"].unique())

        if len(combined_trials) != num_trials:
            warnings.warn(
                f"Expected {num_trials} trials but found "
                f"{len(combined_trials)} for terrain {terrain}. "
                f"Proceeding with shared trials only."
            )

        for trial in combined_trials:
            combined_trial = combined_df[
                combined_df["eval_id"] == trial
            ].reset_index(drop=True)
            terrain_sprs = [
                combined_trial["Mean SPR 1 (PSI)"].mean() / spr_max,
                combined_trial["Mean SPR 2 (PSI)"].mean() / spr_max,
                combined_trial["Mean SPR 3 (PSI)"].mean() / spr_max
            ]
            segments, step_len = _segment_trial_combined(
                combined_trial,
                num_steps
            )

            for step_idx, segment in enumerate(segments):
                windows = _window_segment(segment, window_size, stride)
                for window_idx, window_segment in enumerate(windows):
                    sequences.append(window_segment)
                    labels.append(np.asarray(terrain_sprs))
                    """
                    label = torch.zeros_like(torch.Tensor(terrain_sprs))
                    label[(terrain_sprs >= 200) & (terrain_sprs < 300)] = 1
                    label[terrain_sprs >= 300 ]= 2
                    labels.append(label)
                    """
                    lengths.append(len(window_segment))
                    metadata.append(
                        SampleMetadata(
                            terrain=t_index,
                            speed=speed,
                            trial=int(trial),
                            step=step_idx,
                            length=len(window_segment),
                            window=window_idx,
                        )
                    )
    return (sequences, labels, lengths, metadata)


# TODO(nubby): Allow for selective/combined input of B1 data.
# TODO(nubby): Change num_classes to 3 for "NC", "IDC", and "C".
def load_raw_dataset(
        data_dir: Path,
        num_classes: int = 6,
        speeds: Sequence[int] = (1, 2, 3, 4, 5, 6),
        num_trials: int = 10,
        num_steps: int = 8,
        window_size: int | None = None,
        stride: int | None = None,
    ) -> RawSequenceDataset:
    """
    load_raw_dataset(...) -> RawSequenceDataset

    Parses and aligns the CSV files into per-step sequences of IMU quaternions,
    joint angles, and joint torques.
    """
    sequences: List[np.ndarray] = []
    labels: List[List[int]] = []
    lengths: List[int] = []
    metadata: List[SampleMetadata] = []

    # Shape feature vector and load data based on training mode (always use IMU
    # data).
    raw = _load_raw_b1_dataset(
            data_dir=data_dir,
            num_trials=num_trials,
            num_steps=num_steps,
            window_size=window_size,
            stride=stride
        )
    sequences += raw[0]
    labels += raw[1]
    lengths += raw[2]
    metadata += raw[3]

    feature_names: List[str] = []

    # TODO
    feature_names += list(JTORQUE_FEATURES)
    feature_names += list(JANGLES_FEATURES)
    feature_names += list(IMU_FEATURES_B1)

    #labels=np.asarray(labels, dtype=np.int64),

    return RawSequenceDataset(
        feature_names=feature_names,
        labels=np.array(labels),
        lengths=np.asarray(lengths, dtype=np.int64),
        metadata=metadata,
        sequences=sequences
    )

def _compute_split_counts(n_trials: int, train_frac: float, val_frac: float) -> Tuple[int, int, int]:
    """Computes integer trial counts per split with basic safeguards."""
    if n_trials <= 2:
        raise ValueError("Need at least three trials to form train/val/test splits.")
    train = max(1, int(round(n_trials * train_frac)))
    val = max(1, int(round(n_trials * val_frac)))
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
) -> Dict[Tuple[int, int, int], str]:
    """Assigns each (terrain, speed, trial) tuple to a split."""
    # First group trials under terrain and speed.
    by_key: Dict[Tuple[int, int], List[int]] = {}
    for meta in metadata:
        key = (meta.terrain, meta.speed)
        by_key.setdefault(key, [])
        if meta.trial not in by_key[key]:
            by_key[key].append(meta.trial)

    assignment: Dict[Tuple[int, int, int], str] = {}
    for key, trials in by_key.items():
        trials_copy = list(trials)
        rng.shuffle(trials_copy)
        train_count, val_count, _ = _compute_split_counts(len(trials_copy), train_frac, val_frac)
        train_trials = set(trials_copy[:train_count])
        val_trials = set(trials_copy[train_count : train_count + val_count])
        for trial in trials:
            if trial in train_trials:
                assignment[(key[0], key[1], trial)] = "train"
            elif trial in val_trials:
                assignment[(key[0], key[1], trial)] = "val"
            else:
                assignment[(key[0], key[1], trial)] = "test"
    return assignment


def _compute_feature_stats(
    sequences: Sequence[np.ndarray],
    indices: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Derives mean/std from the provided subset."""
    stacked = np.concatenate([sequences[i] for i in indices], axis=0)
    mean = stacked.mean(axis=0, dtype=np.float64)
    std = stacked.std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def build_data_bundle(
        data_dir: Path,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        seed: int = 13,
        window_size: int | None = None,
        stride: int | None = None,
        mode: str = "qcat",
    ) -> TerrainDataBundle:
    """
    build_data_bundle(...) -> TerrainDataBundle

    Loads raw data, performs splits, and returns ready-to-use torch datasets.
    Data/Features used in training are selectable with the "mode" flag:
        + mode: "qcat"      <- Open-source dataset only.
        + mode: "b1"        <- Dataset collected from our B1 robot.
        + mode: "combined"  <- [WIP] Combined dataset.
    """
    # Start by loading the selected datasets.
    raw = load_raw_dataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        mode=mode
    )

    # Random seed-based assignment for replicability.
    rng = np.random.default_rng(seed)

    assignment = _assign_trial_splits(
        metadata=raw.metadata,
        train_frac=train_frac,
        val_frac=val_frac,
        rng=rng
    )

    split_indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for idx, meta in enumerate(raw.metadata):
        split = assignment[(meta.terrain, meta.speed, meta.trial)]
        split_indices[split].append(idx)

    feature_mean_np, feature_std_np = _compute_feature_stats(
        raw.sequences,
        split_indices["train"]
    )

    def make_dataset(indices: List[int]) -> SequenceDataset:
        seq_tensors: List[torch.Tensor] = []
        lengths_tensors: List[torch.Tensor] = []
        labels_tensors: List[torch.Tensor] = []
        metadata_subset: List[SampleMetadata] = []
        for idx in indices:
            seq = raw.sequences[idx]
            norm_seq = (seq - feature_mean_np) / feature_std_np
            seq_tensors.append(torch.from_numpy(norm_seq.astype(np.float32)))
            lengths_tensors.append(
                torch.tensor(raw.lengths[idx], dtype=torch.long)
            )
            # TODO(nubby): Normalize here for some reason? Make this suck less.
            labels_tensors.append(
                torch.tensor(raw.labels[idx], dtype=torch.float)
            )
            metadata_subset.append(raw.metadata[idx])
        return SequenceDataset(
            seq_tensors,
            torch.stack(labels_tensors),
            torch.stack(lengths_tensors),
            metadata_subset
        )

    train_ds = make_dataset(split_indices["train"])
    val_ds = make_dataset(split_indices["val"])
    test_ds = make_dataset(split_indices["test"])

    feature_mean = torch.from_numpy(feature_mean_np)
    feature_std = torch.from_numpy(feature_std_np)

    return TerrainDataBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=raw.feature_names,
    )
