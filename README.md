# PSR: Physics-Structured Regression for Cross-Task Robot Anomaly Detection

Source code for:

> **Cross-task anomaly detection in reconfigurable industrial robot systems based on physics-structured regression of joint motor currents**

## Overview

PSR decomposes joint motor current into physically interpretable dynamics terms using Newton–Euler mechanics. The residual suppresses task-dependent variations and exposes health-related deviations, enabling cross-task anomaly detection without target-task retraining or external sensors.

## Pipeline

| Script | Description |
|--------|-------------|
| `01_feature_extraction.py` | Per-cycle statistical features from HDF5 recordings |
| `02_psr_model.py` | Per-joint PSR model via OLS, residuals, R² |
| `03_physics_ablation.py` | Nine ablation conditions under strict LOTO |
| `04_robustness.py` | Monte Carlo DH parameter perturbation |
| `05_statistical_tests.py` | BCa bootstrap CIs and DeLong tests |
| `06_baseline_convae.py` | Conv-AE baseline |
| `07_baseline_lstmvae_gmm.py` | LSTM-VAE and GMM baselines |
| `08_per_anomaly_benchmark.py` | Per-anomaly AUROC and computational benchmarks |
| `09_operating_points.py` | Operating-point metrics at FPR ≤ 0.05 |

## Data Availability

The experimental data will be made available upon reasonable request. Contact: Prof. Wei Qin (wqin@sjtu.edu.cn).

## Citation

```bibtex
@article{zafar2026psr,
  title={Cross-task anomaly detection in reconfigurable industrial robot systems based on physics-structured regression of joint motor currents},
  author={Zafar, Sanwal Ahmad and Qin, Wei},
  journal={Journal of Manufacturing Systems},
  year={2026}
}
```

## License

MIT
