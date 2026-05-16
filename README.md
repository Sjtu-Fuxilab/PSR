# PSR: Physics-Structured Regression for Cross-Task Robot Anomaly Detection

Source code for:
> Cross-task anomaly detection in reconfigurable industrial robot systems based on physics-structured regression of joint motor currents

## Overview

PSR decomposes joint motor current into physically interpretable dynamics terms using Newton–Euler mechanics. The residual suppresses task-dependent variations and exposes health-related deviations, enabling cross-task anomaly detection without target-task retraining or external sensors. All results are produced under strict leave-one-task-out (LOTO) evaluation across four tasks.

## Pipeline

The notebooks are run in numerical order. Notebooks 1–4 collect raw data; notebook 5 extracts features; notebooks 6–17 produce the manuscript tables and reviewer-response analyses; notebooks 18–19 build the final figures and tables.

| Notebook | Produces |
|---|---|
| `01_data_collection_T1.ipynb` | T1 (pick-and-place) raw HDF5 recordings |
| `02_data_collection_T2.ipynb` | T2 (assembly press-fit) raw HDF5 recordings |
| `03_data_collection_T3.ipynb` | T3 (palletizing) raw HDF5 recordings |
| `04_data_collection_T4.ipynb` | T4 (bin-pick with wrist reorientation) raw HDF5 recordings |
| `05_feature_extraction.ipynb` | `features.csv` — 110-dim per-cycle feature vectors |
| `06_psr_regression_quality_4fold.ipynb` | Table 3, Supplementary Table S1 |
| `07_statistical_tests_4fold.ipynb` | Table 5, Supplementary Table S3 |
| `08_baseline_convae_4fold.ipynb` | Table 5 Conv-AE row |
| `09_baseline_lstmvae_gmm_4fold.ipynb` | Table 5 LSTM-VAE and GMM rows |
| `10_physics_term_ablation_4fold.ipynb` | Table 6 |
| `11_parameter_robustness_4fold.ipynb` | Table 4 |
| `12_operating_points_4fold.ipynb` | Table 8 |
| `13_feature_group_auroc_4fold.ipynb` | Supplementary Table S2 |
| `14_inertia_friction_diagnostic.ipynb` | Methodology validation (P37) |
| `15_term_variance_decomposition.ipynb` | Variance decomposition narrative (P95) |
| `16_spectral_feature_ablation.ipynb` | Supplementary Table S6 |
| `17_noise_robustness.ipynb` | Supplementary Table S7 |
| `18_figures.ipynb` | Figures 3, 4, 6 |
| `19_build_tables.ipynb` | Final formatted Table 5 and Supplementary Table S3 |
| `20_verify_ols_vs_ridge.ipynb` | OLS vs Ridge methodology check |

## Joint indexing

Internal column names and printouts use Python 0-indexed joint labels `J0`…`J5`. The manuscript uses 1-indexed labels `J1`…`J6`. The mapping is direct (`J0` → `J1` base, `J1` → `J2` shoulder, …, `J5` → `J6` wrist 3). Numerical values are identical under either convention.

## Data

Data acquisition uses the RTDE interface to a UR5 CB3 industrial robot at 125 Hz. Raw recordings (joint position, velocity, motor current, cycle index) are stored as HDF5 files. Experimental data are available upon reasonable request: Prof. Wei Qin (wqin@sjtu.edu.cn).

## Environment

Python 3.11. Required packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `h5py`, `matplotlib`, `torch` (for Conv-AE and LSTM-VAE baselines), `rtde_receive` (data collection only).

Edit the `ROOT` constant at the top of each analysis notebook to point to your local data directory. The directory structure expected is `<ROOT>/Lab_Data/<task>/<condition>/*.h5` for raw recordings and `<ROOT>/Processed_Data/` for intermediate CSV outputs.

## License

MIT.
