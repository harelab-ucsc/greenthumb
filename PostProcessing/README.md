# GreenThumb - PostProcessing


## Description

This directory contains all software related to analyzing data collected as part of the GreenThumb project.
Insights to be derived include model performance (accuracy, uncertainty) as well as generated soil maps.

### Pipeline layout

- `src/`
  - `data_pipeline.py`: Parse CSVs, align IMU+force, split trials, normalize features, build torch datasets.
  - `models.py`: `LSTMClassifier`, `TemporalConvNetClassifier`, `TransformerClassifier`.
  - `benchmark.py`: Train/evaluate models with early stopping and LR scheduling; saves plots and JSON logs.
  - `plot_sensors.py`: Generate IMU/force panels.
- `data/`: Preprocessed, labeled data are placed here for use by the pipeline.
- `figures/`: Saved panels (e.g., `imu_panel.png`, `force_panel.png`).
- `artifacts_temp/`: Precomputed training histories and plots for LSTM/TCN/Transformer.
- `requirements.txt`: Python dependencies (see Install).

---

## Install

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Alternatively, install essentials directly:

```bash
pip install torch numpy pandas matplotlib seaborn wandb
```

Optional: disable online W&B logging:

```bash
export WANDB_MODE=offline
```

---

## Dataset: Comedy

When working with data fresh from a Unitree B1 robot (specifically from HARE Lab's Comedy), it will need
to be trimmed and labeled to be used as model input. Do so using with the following:

1. Copy collected data into the `tmp/` folder.

2. ```bash
python3 -m src.preprocess_b1_data --input-dir tmp --output-dir data 
```

And that's all. Now the data in the `data/` directory is ready for training/validation.

---

## Quick start

### 1) Visualize sensor panels

```bash
python -m src.plot_sensors --data-dir data --output-dir figures \
  --terrains 0 1 2 3 4 5 --speed 3 --trial 0 --step 0
```

Outputs:
- `figures/force_panel.png`
- `figures/imu_panel.png`

### 2) Train benchmarks (LSTM, TCN, Transformer)

```bash
python -m src.benchmark --data-dir data --output-dir artifacts_temp \
  --models lstm tcn transformer --epochs 15 --batch-size 32 --lr 3e-4 \
  --patience 6 --seed 42
```

Common flags:
- `--window-size N` and `--stride S` for intra-step sliding windows (optional)
- `--use-wandb` plus `--wandb-project`, `--wandb-entity`, `--wandb-group`, `--wandb-run-prefix`
- `--no-cuda` to force CPU

Outputs per model under `artifacts_temp/<model>/`:
- `history.json` (metrics over time)
- `<model>_epoch_metrics.png` (train/val loss+acc)
- `<model>_train_batch_loss.png` (batch loss curve)

---

## Results and artifacts

Precomputed artifacts (for reference) are provided:

- LSTM: `artifacts_temp/lstm/lstm_epoch_metrics.png`
- TCN: `artifacts_temp/tcn/tcn_epoch_metrics.png`
- Transformer: `artifacts_temp/transformer/transformer_epoch_metrics.png`

To generate these results:
```bash
python -m src.benchmark --data-dir data --output-dir artifacts_temp \
  --models lstm --epochs 15 --batch-size 32 --lr 3e-4 \
  --patience 6 --seed 42

python -m src.benchmark --data-dir data --output-dir artifacts_temp \
  --models tcn --epochs 15 --batch-size 32 --lr 3e-4 \
  --patience 6 --seed 42

python -m src.benchmark --data-dir data --output-dir artifacts_temp \
  --models transformer --epochs 15 --batch-size 32 --lr 3e-4 \
  --patience 6 --seed 42
```

Figures (built from the CSVs):
- IMU panel: `figures/imu_panel.png`
- Force panel: `figures/force_panel.png`

You can regenerate all of the above using the commands in Quick start.

---

## Notes

- Features are normalized using training split statistics.
- Trial-level splits are stratified per compaction level and soil moisture content.

### Supported Platforms

- Unitree B1


## Authors

HARE Lab

nubby

Taylor Kergan


## Date

28 May 2026


## Version

0.1.1

