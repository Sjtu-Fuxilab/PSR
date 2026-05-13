# T4_operating_points.py
# Operating-point metrics at FPR <= 0.05 for T4, plus final 4-fold aggregation

# %% Cell 1: Configuration

import os, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT    = r"D:\Research\R"
OUT     = os.path.join(ROOT, "P_Data")
FPR_CAP = 0.05

print(f"Operating point: FPR <= {FPR_CAP}")
print(f"Inputs from   : {OUT}")

# %% Cell 2: Load T4 scores from upstream notebooks

with open(os.path.join(OUT, "T4_psr_scores.pkl"), "rb") as fh:
    psr_scores = pickle.load(fh)
with open(os.path.join(OUT, "T4_baseline_scores.pkl"), "rb") as fh:
    base_scores = pickle.load(fh)

assert np.array_equal(psr_scores["y_true"], base_scores["y_true"]), \
    "PSR and baseline notebooks disagree on T4 y_true ordering"
y_T4 = psr_scores["y_true"]
print(f"T4 test cycles: {len(y_T4)}  ({(y_T4==0).sum()} healthy, {(y_T4==1).sum()} anomaly)")

T4_SCORES = {
    "PSR Z-Score":     psr_scores["PSR_ZScore"],
    "PSR OC-SVM":      psr_scores["PSR_OCSVM"],
    "PSR IsoForest":   psr_scores["PSR_IsoForest"],
    "GMM (PSR feat.)": psr_scores["GMM"],
    "Conv-AE":         base_scores["Conv-AE"],
    "LSTM-VAE":        base_scores["LSTM-VAE"],
    "Raw Z-Score":     base_scores["Raw Z-Score"],
}

# %% Cell 3: Operating-point function

def operating_point(y_true, y_score, fpr_cap=FPR_CAP):
    """Smallest threshold (most permissive) such that FPR <= fpr_cap on the held-out fold."""
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.unique(y_score)[::-1]
    best = None
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tn   = int(((pred == 0) & (y_true == 0)).sum())
        fp   = int(((pred == 1) & (y_true == 0)).sum())
        n_healthy = tn + fp
        fpr = fp / n_healthy if n_healthy > 0 else 0.0
        if fpr <= fpr_cap:
            prec = precision_score(y_true, pred, zero_division=0)
            rec  = recall_score(y_true, pred, zero_division=0)
            f1   = f1_score(y_true, pred, zero_division=0)
            best = (prec, rec, f1, fpr, thr)
        else:
            break
    if best is None:
        # No threshold met the cap; report tightest-threshold metrics.
        pred = (y_score >= thresholds[0]).astype(int)
        tn   = int(((pred == 0) & (y_true == 0)).sum())
        fp   = int(((pred == 1) & (y_true == 0)).sum())
        n_healthy = tn + fp
        fpr  = fp / n_healthy if n_healthy > 0 else 0.0
        prec = precision_score(y_true, pred, zero_division=0)
        rec  = recall_score(y_true, pred, zero_division=0)
        f1   = f1_score(y_true, pred, zero_division=0)
        return prec, rec, f1, fpr, thresholds[0]
    return best

# %% Cell 4: Operating points for all 7 methods at T4

print("T4 operating-point metrics at FPR <= 0.05:")
print(f"  {'Method':<18}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'FPR':>6}")

t4_rows = []
for method, scores in T4_SCORES.items():
    prec, rec, f1, fpr, thr = operating_point(y_T4, scores, FPR_CAP)
    t4_rows.append(dict(method=method, fold="T4",
                        precision=round(prec, 4), recall=round(rec, 4),
                        f1=round(f1, 4), fpr=round(fpr, 4),
                        threshold=round(float(thr), 6)))
    print(f"  {method:<18}  {prec:>6.4f}  {rec:>6.4f}  {f1:>6.4f}  {fpr:>6.4f}")

df_t4_op = pd.DataFrame(t4_rows)
df_t4_op.to_csv(os.path.join(OUT, "T4_operating_points.csv"), index=False, float_format="%.4f")

# %% Cell 5: Final 4-fold Table 3 (aggregate AUROC)

# Canonical 3-fold values from the published Table 3. Adjust if the 3-fold
# pipeline produces slightly different numbers locally; the manuscript values
# are the reference.
CANONICAL_3FOLD = {
    "PSR Z-Score":     {"T1": (0.966, 0.946, 0.978), "T2": (0.904, 0.873, 0.928), "T3": (0.648, 0.594, 0.698)},
    "PSR OC-SVM":      {"T1": (0.996, 0.981, 1.000), "T2": (0.831, 0.790, 0.864), "T3": (0.305, 0.274, 0.334)},
    "PSR IsoForest":   {"T1": (0.897, 0.867, 0.922), "T2": (0.928, 0.900, 0.949), "T3": (0.635, 0.581, 0.687)},
    "GMM (PSR feat.)": {"T1": (0.970, 0.951, 0.983), "T2": (0.874, 0.834, 0.907), "T3": (0.500, 0.442, 0.563)},
    "Conv-AE":         {"T1": (0.047, 0.029, 0.073), "T2": (0.949, 0.923, 0.967), "T3": (0.747, 0.694, 0.794)},
    "LSTM-VAE":        {"T1": (0.742, 0.693, 0.779), "T2": (0.256, 0.202, 0.286), "T3": (0.123, 0.095, 0.173)},
    "Raw Z-Score":     {"T1": (0.792, 0.750, 0.833), "T2": (0.025, 0.012, 0.048), "T3": (0.211, 0.168, 0.261)},
}

df_t4_psr  = pd.read_csv(os.path.join(OUT, "T4_psr_family_aggregate.csv"))
df_t4_base = pd.read_csv(os.path.join(OUT, "T4_baselines_aggregate.csv"))
df_t4_all  = pd.concat([df_t4_psr, df_t4_base], ignore_index=True)

rows_4fold = []
for method, folds_3 in CANONICAL_3FOLD.items():
    t4_row = df_t4_all[df_t4_all.method == method].iloc[0]
    a4, lo4, hi4 = float(t4_row.auroc), float(t4_row.ci_lo), float(t4_row.ci_hi)
    mean_4fold = round(float(np.mean([folds_3["T1"][0], folds_3["T2"][0], folds_3["T3"][0], a4])), 3)
    rows_4fold.append(dict(method=method,
                           T1_auroc=folds_3["T1"][0], T1_ci_lo=folds_3["T1"][1], T1_ci_hi=folds_3["T1"][2],
                           T2_auroc=folds_3["T2"][0], T2_ci_lo=folds_3["T2"][1], T2_ci_hi=folds_3["T2"][2],
                           T3_auroc=folds_3["T3"][0], T3_ci_lo=folds_3["T3"][1], T3_ci_hi=folds_3["T3"][2],
                           T4_auroc=round(a4, 4),    T4_ci_lo=round(lo4, 4),    T4_ci_hi=round(hi4, 4),
                           mean_4fold=mean_4fold))

df_table3 = pd.DataFrame(rows_4fold)
df_table3.to_csv(os.path.join(OUT, "Table3_aggregate_AUROC_4fold.csv"),
                 index=False, float_format="%.4f")
print(df_table3[["method","T1_auroc","T2_auroc","T3_auroc","T4_auroc","mean_4fold"]].to_string(index=False))

# %% Cell 6: Final 4-fold Supp S4 (per-anomaly AUROC)

# Canonical 3-fold per-anomaly values from the published Supp Table S4.
CANONICAL_S4_3FOLD = {
    "PSR Z-Score":     {"A2": {"T1": 0.996, "T2": 0.994, "T3": 0.438},
                        "A3": {"T1": 0.995, "T2": 0.995, "T3": 0.998},
                        "A5": {"T1": 0.916, "T2": 0.754, "T3": 0.624}},
    "PSR OC-SVM":      {"A2": {"T1": 0.997, "T2": 0.995, "T3": 0.163},
                        "A3": {"T1": 0.997, "T2": 0.995, "T3": 0.495},
                        "A5": {"T1": 0.994, "T2": 0.557, "T3": 0.320}},
    "PSR IsoForest":   {"A2": {"T1": 0.990, "T2": 0.991, "T3": 0.419},
                        "A3": {"T1": 0.993, "T2": 0.993, "T3": 0.966},
                        "A5": {"T1": 0.740, "T2": 0.822, "T3": 0.632}},
    "GMM (PSR feat.)": {"A2": {"T1": 0.998, "T2": 0.996, "T3": 0.667},
                        "A3": {"T1": 0.995, "T2": 0.995, "T3": 0.000},
                        "A5": {"T1": 0.924, "T2": 0.671, "T3": 0.667}},
    "Conv-AE":         {"A2": {"T1": 0.079, "T2": 0.925, "T3": 0.360},
                        "A3": {"T1": 0.032, "T2": 0.998, "T3": 1.000},
                        "A5": {"T1": 0.025, "T2": 0.940, "T3": 0.965}},
    "LSTM-VAE":        {"A2": {"T1": 0.975, "T2": 0.017, "T3": 0.216},
                        "A3": {"T1": 0.044, "T2": 0.975, "T3": 0.038},
                        "A5": {"T1": 0.975, "T2": 0.016, "T3": 0.088}},
    "Raw Z-Score":     {"A2": {"T1": 0.455, "T2": 0.025, "T3": 0.564},
                        "A3": {"T1": 0.994, "T2": 0.025, "T3": 0.000},
                        "A5": {"T1": 0.995, "T2": 0.025, "T3": 0.000}},
}

df_t4_psr_per  = pd.read_csv(os.path.join(OUT, "T4_psr_family_per_anomaly.csv"))
df_t4_base_per = pd.read_csv(os.path.join(OUT, "T4_baselines_per_anomaly.csv"))
df_t4_per_all  = pd.concat([df_t4_psr_per, df_t4_base_per], ignore_index=True)

def t4_value(method, anomaly):
    sub = df_t4_per_all[(df_t4_per_all.method == method) & (df_t4_per_all.anomaly == anomaly)]
    return float(sub.auroc.iloc[0]) if len(sub) else float("nan")

s4_rows = []
for method in CANONICAL_S4_3FOLD:
    row = {"method": method}
    vals_for_mean = []
    for anom in ["A2","A3","A5"]:
        row[f"{anom}_T1"] = CANONICAL_S4_3FOLD[method][anom]["T1"]
        row[f"{anom}_T2"] = CANONICAL_S4_3FOLD[method][anom]["T2"]
        row[f"{anom}_T3"] = CANONICAL_S4_3FOLD[method][anom]["T3"]
        row[f"{anom}_T4"] = round(t4_value(method, anom), 4)
        vals_for_mean.extend([row[f"{anom}_T1"], row[f"{anom}_T2"], row[f"{anom}_T3"], row[f"{anom}_T4"]])
    row["per_anom_mean"] = round(np.nanmean(vals_for_mean), 3)
    s4_rows.append(row)

df_s4 = pd.DataFrame(s4_rows)
df_s4.to_csv(os.path.join(OUT, "SuppS4_per_anomaly_AUROC_4fold.csv"), index=False, float_format="%.4f")
print(df_s4[["method","A2_T4","A3_T4","A5_T4","per_anom_mean"]].to_string(index=False))

# %% Cell 7: Cross-check against manuscript-locked means

LOCKED_MEAN = {"PSR Z-Score": 0.842, "PSR OC-SVM": 0.752, "PSR IsoForest": 0.814,
               "GMM (PSR feat.)": 0.830, "Conv-AE": 0.495, "LSTM-VAE": 0.442,
               "Raw Z-Score": 0.470}

print(f"\n  {'Method':<18}  {'Computed':<10}  {'Manuscript':<12}  {'Delta':<8}")
for method, manuscript in LOCKED_MEAN.items():
    computed = float(df_table3[df_table3.method == method].mean_4fold.iloc[0])
    delta = computed - manuscript
    status = "OK" if abs(delta) < 0.005 else "DRIFT"
    print(f"  {method:<18}  {computed:<10.3f}  {manuscript:<12.3f}  {delta:+.3f}  {status}")

print("\nDrift <= 0.005 is within stochastic noise for neural-network baselines.")
print("Drift > 0.02 means T4 did not reproduce the manuscript; investigate.")
