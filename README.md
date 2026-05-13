# PSR: Physics-Structured Regression for Cross-Task Robot Anomaly Detection

Source code for:
> **Physics-Structured Residual Monitoring for Cross-Task Anomaly Detection in Reconfigurable Robot Manufacturing Cells**

## Overview

PSR decomposes joint motor current into physically interpretable dynamics terms using Newton–Euler mechanics. The residual suppresses task-dependent variations and exposes health-related deviations, enabling cross-task anomaly detection without target-task retraining or external sensors.

The pipeline is evaluated under strict leave-one-task-out (LOTO) cross-validation across four manufacturing tasks (T1 pick-and-place, T2 assembly press-fit, T3 palletizing, T4 bin-pick with 60° wrist reorientation).

## Pipeline

### Core scripts (PSR model, baselines, statistical tests)

| Script | Description |
|--------|-------------|
| `featureextraction.py` | Per-cycle statistical features from HDF5 recordings |
| `PSRresidualmonitoring.py` | Per-joint PSR model via OLS, residuals, R² |
| `Full Physics Term Ablation under Strict LOTO.py` | Physics-term ablation under strict LOTO (M2 / M3 / M4 plus single-term conditions) |
| `Ablationstudy.py` | Per-anomaly ablation and feature-group ranking |
| `ConvolutionalAutoencoderBaseline.py` | Conv-AE baseline |
| `Baselines.py` | LSTM-VAE and GMM baselines |
| `Statisticaltests.py` | BCa bootstrap CIs and DeLong tests |
| `Robustness.py` | Monte Carlo DH parameter perturbation |
| `ComputationBenchmarks.py` | Training time, inference latency, model size |

### T4 fold extension

| Script | Description |
|--------|-------------|
| `T4_psr_extension.py` | PSR-family detectors and physics-term ablation at T4 |
| `T4_baselines.py` | Conv-AE, LSTM-VAE, Raw Z-Score at T4 (FIXED_LEN computed over the 4-task cycle pool) |
| `T4_operating_points.py` | Precision / recall / F1 at FPR ≤ 0.05 for T4; final 4-fold Table 3 and Supp S4 aggregation |

## Data Availability

The experimental data will be made available upon reasonable request. Contact: Prof. Wei Qin (wqin@sjtu.edu.cn).

## License

MIT
