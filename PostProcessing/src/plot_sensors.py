from __future__ import annotations

import argparse
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_pipeline import FORCE_FEATURES, IMU_FEATURES


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)

TERRAIN_NAMES: Dict[int, str] = {
    0: "Concrete",
    1: "Grass",
    2: "Gravel",
    3: "Mulch",
    4: "Dirt",
    5: "Sand",
}

FORCE_LEG_ORDER = ("FL", "FR", "BR", "BL")


@dataclass
class StepSample:
    terrain: int
    speed: int
    trial: int
    step: int
    time: np.ndarray  # seconds, origin shifted to zero
    imu: pd.DataFrame
    force: pd.DataFrame

    @property
    def terrain_name(self) -> str:
        return TERRAIN_NAMES.get(self.terrain, f"Terrain {self.terrain}")


def _read_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def extract_step(
    data_dir: Path,
    terrain: int,
    speed: int,
    trial: int,
    step: int,
    *,
    num_steps: int = 8,
) -> StepSample:
    imu_path = data_dir / f"{terrain}_{speed}_legSensors_imu.csv"
    force_path = data_dir / f"{terrain}_{speed}_legSensors_raw.csv"
    if not imu_path.exists() or not force_path.exists():
        raise FileNotFoundError(f"Missing sensor files for terrain={terrain} speed={speed}.")

    imu_df = _read_sensor_csv(imu_path)
    force_df = _read_sensor_csv(force_path)

    imu_trial = imu_df[imu_df["eval_id"] == trial].reset_index(drop=True)
    force_trial = force_df[force_df["eval_id"] == trial].reset_index(drop=True)

    if imu_trial.empty or force_trial.empty:
        raise ValueError(
            f"No data found for terrain={terrain}, speed={speed}, trial={trial}. "
            "Try a different trial index."
        )

    min_len = min(len(imu_trial), len(force_trial))
    if min_len < num_steps:
        raise ValueError(
            f"Trial shorter than required steps: len={min_len}, num_steps={num_steps}."
        )
    step_len = min_len // num_steps
    start = step * step_len
    end = start + step_len
    if end > min_len:
        raise ValueError(
            f"Requested step {step} exceeds available length (min_len={min_len}, step_len={step_len})."
        )

    imu_trim = imu_trial.iloc[:min_len]
    force_trim = force_trial.iloc[:min_len]

    time_ms = imu_trim["time (ms)"].to_numpy(dtype=np.float64)[start:end]
    time_sec = (time_ms - time_ms[0]) / 1000.0

    imu_cols = list[str](IMU_FEATURES)
    force_cols = list[str](FORCE_FEATURES)

    imu_step = imu_trim.iloc[start:end][imu_cols].reset_index(drop=True)
    force_step = force_trim.iloc[start:end][force_cols].reset_index(drop=True)

    return StepSample(
        terrain=terrain,
        speed=speed,
        trial=trial,
        step=step,
        time=time_sec,
        imu=imu_step,
        force=force_step,
    )


def choose_samples(
    data_dir: Path,
    terrains: Sequence[int],
    speed: int,
    trial: int,
    step: int,
) -> List[StepSample]:
    samples: List[StepSample] = []
    for terrain in terrains:
        try:
            sample = extract_step(data_dir, terrain, speed, trial, step)
        except ValueError:
            # Fallback: try the first available trial if requested one missing
            fallback_trial = 0 if trial != 0 else 1
            sample = extract_step(data_dir, terrain, speed, fallback_trial, step)
        samples.append(sample)
    return samples


def plot_force_panel(samples: Sequence[StepSample], output_dir: Path) -> Path:
    n = len(samples)
    rows = int(np.ceil(n / 2))
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    colors = {
        "FL": "#2E86AB",
        "FR": "#F18F01",
        "BR": "#C73E1D",
        "BL": "#6C9A8B",
    }

    for ax in axes.flat:
        ax.axis("off")

    legend_handles = None
    legend_labels = None

    for idx, sample in enumerate(samples):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("on")
        for leg in FORCE_LEG_ORDER:
            z_col = f"{leg}_z (N)"
            if z_col not in sample.force.columns:
                continue
            line, = ax.plot(
                sample.time,
                sample.force[z_col],
                label=f"{leg} $F_z$",
                linewidth=2.0,
                color=colors.get(leg),
            )
        if legend_handles is None and ax.get_legend_handles_labels()[0]:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(f"{sample.terrain_name} (class {sample.terrain})", pad=12)
        ax.set_ylabel("Force (N)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        if r == rows - 1:
            ax.set_xlabel("Time (s)")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=False,
            ncol=2,
        )

    fig.suptitle("Vertical Ground Reaction Forces per Step", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 0.92, 0.97])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "force_panel.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_imu_panel(samples: Sequence[StepSample], output_dir: Path) -> Path:
    n = len(samples)
    rows = int(np.ceil(n / 2))
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    imu_groups: Sequence[Tuple[str, Sequence[str], str]] = (
        ("Angular Velocity", ("ang_vel_x", "ang_vel_y", "ang_vel_z"), "rad/s"),
        ("Linear Acceleration", ("lin_acc_x", "lin_acc_y", "lin_acc_z"), "m/s²"),
    )

    palette = {
        "ang_vel_x": "#1f77b4",
        "ang_vel_y": "#ff7f0e",
        "ang_vel_z": "#2ca02c",
        "lin_acc_x": "#d62728",
        "lin_acc_y": "#9467bd",
        "lin_acc_z": "#8c564b",
    }

    for ax in axes.flat:
        ax.axis("off")

    legend_handles = None
    legend_labels = None

    for idx, sample in enumerate(samples):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("on")

        for _, cols_group, _ in imu_groups:
            for col in cols_group:
                if col not in sample.imu.columns:
                    continue
                ax.plot(
                    sample.time,
                    sample.imu[col],
                    label=col,
                    linewidth=2.0,
                    color=palette.get(col, None),
                )
        if legend_handles is None and ax.get_legend_handles_labels()[0]:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(f"{sample.terrain_name} (class {sample.terrain})", pad=12)
        ax.set_ylabel("Signal")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        if r == rows - 1:
            ax.set_xlabel("Time (s)")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
            ncol=2,
        )

    fig.suptitle("Body IMU Signals per Step", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 0.92, 0.97])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "imu_panel.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise DyRET sensor inputs for publication-ready figures.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--terrains",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Terrain class IDs to include in the panels.",
    )
    parser.add_argument("--speed", type=int, default=3, help="Robot speed index (1-6).")
    parser.add_argument("--trial", type=int, default=0, help="Trial (eval_id) to visualise.")
    parser.add_argument("--step", type=int, default=0, help="Step segment within the trial (0-7).")
    parser.add_argument("--num-steps", type=int, default=8, help="Number of steps per trial recording.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory for output figures.")
    args = parser.parse_args()

    samples = choose_samples(args.data_dir, args.terrains, args.speed, args.trial, args.step)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    force_path = plot_force_panel(samples, output_dir)
    imu_path = plot_imu_panel(samples, output_dir)

    print(f"Saved force panel to {force_path}")
    print(f"Saved IMU panel to {imu_path}")


if __name__ == "__main__":
    main()
