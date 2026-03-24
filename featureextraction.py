# %% Cell 1 — Configuration
import h5py, os, glob
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT     = r"D:\Research\R"
BASE     = os.path.join(ROOT, "L_Data")
OUT_DIR  = os.path.join(ROOT, "P_Data")
FIG_SUP  = os.path.join(ROOT, "M_Data", "Figures", "Supplementary")
RATE_HZ  = 125
MIN_CYCLE_SAMPLES = 200
TRIM_FIRST_LAST   = True

for d in [OUT_DIR, FIG_SUP]:
    os.makedirs(d, exist_ok=True)

JOINT_NAMES = [
    "J0_base", "J1_shoulder", "J2_elbow",
    "J3_wrist1", "J4_wrist2", "J5_wrist3",
]

REGISTRY = {
    "T1_healthy":    ("T1_PickPlace/Healthy",  "UR5_T1_healthy_180cyc_*.h5",        "T1", "healthy", "none",    0.0),
    "T2_healthy":    ("T2_Assembly/Healthy",    "UR5_T2_healthy_180cyc_*.h5",        "T2", "healthy", "none",    0.0),
    "T3_healthy":    ("T3_Palletize/Healthy",   "UR5_T3_healthy_183cyc_*.h5",        "T3", "healthy", "none",    0.0),
    "T1_A2_0.5kg":   ("T1_PickPlace/A2", "UR5_T1_A2_0.5kg_gripper_40cyc_*.h5",  "T1", "A2", "0.5kg",   0.5),
    "T1_A2_1kg":     ("T1_PickPlace/A2", "UR5_T1_A2_1kg_gripper_40cyc_*.h5",    "T1", "A2", "1kg",     1.0),
    "T1_A2_2kg":     ("T1_PickPlace/A2", "UR5_T1_A2_2kg_gripper_40cyc_*.h5",    "T1", "A2", "2kg",     2.0),
    "T1_A3_10wraps": ("T1_PickPlace/A3", "UR5_T1_A3_1band_40cyc_*.h5",          "T1", "A3", "10wraps", 0.0),
    "T1_A3_17wraps": ("T1_PickPlace/A3", "UR5_T1_A3_3bands_40cyc_*.h5",         "T1", "A3", "17wraps", 0.0),
    "T1_A5_20mm":    ("T1_PickPlace/A5", "UR5_T1_A5_20mm_40cyc_*.h5",           "T1", "A5", "20mm",    0.0),
    "T1_A5_50mm":    ("T1_PickPlace/A5", "UR5_T1_A5_50mm_40cyc_*.h5",           "T1", "A5", "50mm",    0.0),
    "T1_A5_100mm":   ("T1_PickPlace/A5", "UR5_T1_A5_100mm_40cyc_*.h5",          "T1", "A5", "100mm",   0.0),
    "T2_A2_1.5kg":   ("T2_Assembly/A2", "UR5_T2_A2_1.5kg_gripper_40cyc_*.h5",   "T2", "A2", "0.5kg",   0.5),
    "T2_A2_2kg":     ("T2_Assembly/A2", "UR5_T2_A2_2kg_gripper_40cyc_*.h5",     "T2", "A2", "1kg",     1.0),
    "T2_A2_3kg":     ("T2_Assembly/A2", "UR5_T2_A2_3kg_gripper_40cyc_*.h5",     "T2", "A2", "2kg",     2.0),
    "T2_A3_7duct":   ("T2_Assembly/A3", "UR5_T2_A3_light_duct_40cyc_*_214735.h5",  "T2", "A3", "7wraps",  0.0),
    "T2_A3_14duct":  ("T2_Assembly/A3", "UR5_T2_A3_medium_duct_40cyc_*_225508.h5", "T2", "A3", "14wraps", 0.0),
    "T2_A5_20mm":    ("T2_Assembly/A5", "UR5_T2_A5_20mm_40cyc_*.h5",            "T2", "A5", "20mm",    0.0),
    "T2_A5_50mm":    ("T2_Assembly/A5", "UR5_T2_A5_50mm_40cyc_*.h5",            "T2", "A5", "50mm",    0.0),
    "T2_A5_100mm":   ("T2_Assembly/A5", "UR5_T2_A5_100mm_40cyc_*.h5",           "T2", "A5", "100mm",   0.0),
    "T3_A2_3.5kg":   ("T3_Palletize/A2", "UR5_T3_A2_3.5kg_gripper_33cyc_*.h5",  "T3", "A2", "0.5kg",   0.5),
    "T3_A2_4kg":     ("T3_Palletize/A2", "UR5_T3_A2_4kg_gripper_33cyc_*.h5",    "T3", "A2", "1kg",     1.0),
    "T3_A2_5kg":     ("T3_Palletize/A2", "UR5_T3_A2_4.5kg_gripper_33cyc_*.h5",  "T3", "A2", "2kg",     2.0),
    "T3_A3_14duct":  ("T3_Palletize/A3", "UR5_T3_A3_medium_duct_33cyc_*.h5",    "T3", "A3", "14wraps", 0.0),
    "T3_A3_7duct":   ("T3_Palletize/A3", "UR5_T3_A3_light_duct_33cyc_*.h5",     "T3", "A3", "7wraps",  0.0),
    "T3_A5_20mm":    ("T3_Palletize/A5", "UR5_T3_A5_20mm_33cyc_*.h5",           "T3", "A5", "20mm",    0.0),
    "T3_A5_50mm":    ("T3_Palletize/A5", "UR5_T3_A5_50mm_33cyc_*.h5",           "T3", "A5", "50mm",    0.0),
    "T3_A5_100mm":   ("T3_Palletize/A5", "UR5_T3_A5_100mm_33cyc_*.h5",          "T3", "A5", "100mm",   0.0),
}

SEVERITY_ORDER = {
    "none": 0,
    "0.5kg": 1, "1kg": 2, "2kg": 3,
    "7wraps": 1, "10wraps": 2, "14wraps": 3, "17wraps": 4,
    "20mm": 1, "50mm": 2, "100mm": 3,
}

print(f"Registry: {len(REGISTRY)} canonical files")


# %% Cell 2 — Resolve paths
paths = {}
for tag, (subdir, pattern, *_) in REGISTRY.items():
    hits = glob.glob(os.path.join(BASE, subdir, pattern))
    if hits:
        paths[tag] = hits[0]
    else:
        print(f"  MISSING: {tag}  ({subdir}/{pattern})")

assert len(paths) == len(REGISTRY), f"Only {len(paths)}/{len(REGISTRY)} files found"
print(f"All {len(paths)} files resolved.")


# %% Cell 3 — Cycle segmentation
def cycles_from_labels(cycle_arr):
    """Segment using per-sample cycle_number labels stored in HDF5."""
    segments, start, cur = [], 0, cycle_arr[0]
    for i in range(1, len(cycle_arr)):
        if cycle_arr[i] != cur:
            if cur > 0:
                segments.append((start, i, int(cur)))
            cur, start = cycle_arr[i], i
    if cur > 0:
        segments.append((start, len(cycle_arr), int(cur)))
    return segments


def cycles_from_tcp(tcp, home_r_mm=15.0, far_mm=30.0, min_n=500):
    """Fallback: detect cycles via TCP Euclidean distance from first sample."""
    home = tcp[0, :3]
    d = np.linalg.norm(tcp[:, :3] - home, axis=1) * 1000
    near = d < home_r_mm
    segs, active, s, cn = [], False, 0, 0
    for i in range(1, len(d)):
        if not active and not near[i] and d[i] > far_mm:
            active, s = True, i
        elif active and near[i]:
            if (i - s) >= min_n:
                cn += 1
                segs.append((s, i, cn))
            active = False
    return segs


# %% Cell 4 — Feature computation
def cycle_features(seg):
    """Compute 110-dim feature vector for a (N, 6) current segment."""
    f = {"n_samples": seg.shape[0], "duration_sec": seg.shape[0] / RATE_HZ}

    for j in range(6):
        p = JOINT_NAMES[j]
        s = seg[:, j]
        d = np.diff(s)

        f[f"{p}_mean"]         = np.mean(s)
        f[f"{p}_std"]          = np.std(s)
        f[f"{p}_min"]          = np.min(s)
        f[f"{p}_max"]          = np.max(s)
        f[f"{p}_range"]        = np.ptp(s)
        f[f"{p}_rms"]          = np.sqrt(np.mean(s**2))
        f[f"{p}_abs_mean"]     = np.mean(np.abs(s))
        f[f"{p}_skew"]         = float(sp_stats.skew(s))
        f[f"{p}_kurtosis"]     = float(sp_stats.kurtosis(s))
        f[f"{p}_p05"]          = np.percentile(s, 5)
        f[f"{p}_p25"]          = np.percentile(s, 25)
        f[f"{p}_p50"]          = np.percentile(s, 50)
        f[f"{p}_p75"]          = np.percentile(s, 75)
        f[f"{p}_p95"]          = np.percentile(s, 95)
        f[f"{p}_iqr"]          = f[f"{p}_p75"] - f[f"{p}_p25"]
        f[f"{p}_diff_std"]     = np.std(d)
        f[f"{p}_diff_abs_mean"]= np.mean(np.abs(d))

    f["total_rms"]     = np.sqrt(np.mean(seg**2))
    f["total_abs_max"] = np.max(np.abs(seg))
    return f

n_feats = len(cycle_features(np.random.randn(1000, 6)))
print(f"Features per cycle: {n_feats}")


# %% Cell 5 — Main extraction loop
rows, summaries = [], []

for tag in sorted(paths):
    subdir, pattern, task, anomaly, severity, extra_mass = REGISTRY[tag]
    fp = paths[tag]

    with h5py.File(fp, "r") as f:
        cur = f["actual_current"][:]
        if "cycle_number" in f:
            segs = cycles_from_labels(f["cycle_number"][:])
            method = "labels"
        else:
            segs = cycles_from_tcp(f["actual_TCP_pose"][:],
                                   home_r_mm=float(f.attrs.get("home_radius_mm", 15.0)))
            method = "tcp"

    if TRIM_FIRST_LAST and len(segs) > 2:
        segs = segs[1:-1]

    n_ok = 0
    for s, e, cn in segs:
        if (e - s) < MIN_CYCLE_SAMPLES:
            continue
        feat = cycle_features(cur[s:e])
        feat.update(dict(
            tag=tag, task=task, anomaly=anomaly, severity=severity,
            severity_order=SEVERITY_ORDER.get(severity, 0),
            extra_mass_kg=extra_mass,
            is_anomaly=int(anomaly != "healthy"),
            cycle_num=cn, file=os.path.basename(fp),
        ))
        rows.append(feat)
        n_ok += 1

    summaries.append(dict(tag=tag, task=task, anomaly=anomaly,
                          severity=severity, cycles_used=n_ok, method=method))
    print(f"  {tag:<20s}  {n_ok:>4d} cycles  ({method})")

print(f"\nTotal: {len(rows)} cycle-feature vectors from {len(paths)} files")


# %% Cell 6 — Build and save DataFrame
df = pd.DataFrame(rows)

for col in df.select_dtypes(include=["string", "object", "category"]).columns:
    df[col] = df[col].astype("object")

meta = ["tag", "task", "anomaly", "severity", "severity_order",
        "extra_mass_kg", "is_anomaly", "cycle_num", "file",
        "n_samples", "duration_sec"]
feats = sorted(c for c in df.columns if c not in meta)
df = df[meta + feats]

csv_path = os.path.join(OUT_DIR, "features.csv")
h5_path  = os.path.join(OUT_DIR, "features.h5")

df.to_csv(csv_path, index=False, float_format="%.6f")
df.to_hdf(h5_path, key="features", mode="w", format="table")

print(f"Shape: {df.shape[0]} cycles x {df.shape[1]} columns ({len(feats)} features)")
print(f"Saved: {csv_path}  ({os.path.getsize(csv_path)/1e6:.1f} MB)")
print(f"Saved: {h5_path}  ({os.path.getsize(h5_path)/1e6:.1f} MB)")


# %% Cell 7 — Cycle count verification
sdf = pd.DataFrame(summaries)
print(sdf.to_string(index=False))

low = sdf[sdf["cycles_used"] < 25]
if len(low):
    print(f"\nWARNING: {len(low)} files with < 25 usable cycles")
else:
    print(f"\nAll files have >= 25 usable cycles.")


# %% Cell 8 — Row counts by condition
print("\nCycles per task:")
print(df.groupby("task").size().to_string())
print("\nCycles per condition:")
print(df.groupby(["task", "anomaly", "severity"]).size().to_string())


# %% Cell 9 — QC Figure 1: J2 feature distributions by anomaly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("J2 Elbow Current — Distribution by Anomaly Type", fontsize=13)

for col, task in enumerate(["T1", "T2", "T3"]):
    tm = df["task"] == task
    for anom, color in [("healthy", "#1b9e77"), ("A2", "#d95f02"), ("A3", "#7570b3"), ("A5", "#e7298a")]:
        sub = df[tm & (df["anomaly"] == anom)]
        if len(sub) == 0:
            continue
        axes[0, col].hist(sub["J2_elbow_mean"], bins=30, alpha=0.45, label=anom, color=color, density=True)
        axes[1, col].hist(sub["J2_elbow_std"],  bins=30, alpha=0.45, label=anom, color=color, density=True)
    axes[0, col].set_title(f"{task} — J2 mean")
    axes[1, col].set_title(f"{task} — J2 std")
    axes[0, col].legend(fontsize=7)
    axes[1, col].legend(fontsize=7)
    axes[1, col].set_xlabel("Current (A)")

axes[0, 0].set_ylabel("Density")
axes[1, 0].set_ylabel("Density")
plt.tight_layout()
fig.savefig(os.path.join(FIG_SUP, "NB7_QC_distributions.png"), dpi=1200)
fig.savefig(os.path.join(FIG_SUP, "NB7_QC_distributions.pdf"))
plt.show()
print(f"Saved: {FIG_SUP}/NB7_QC_distributions.png/.pdf")


# %% Cell 10 — QC Figure 2: Severity scaling boxplots
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle("Anomaly Severity Scaling — Per-Cycle Features", fontsize=13)

configs = [
    ("A2", "J2_elbow_mean", "J2 mean (A)",  ["healthy", "0.5kg", "1kg", "2kg"]),
    ("A3", "J2_elbow_std",  "J2 std (A)",   ["healthy", "7wraps", "10wraps", "14wraps", "17wraps"]),
    ("A5", "J2_elbow_mean", "J2 mean (A)",  ["healthy", "20mm", "50mm", "100mm"]),
]

for row, (anom, feat_col, ylabel, order) in enumerate(configs):
    for col, task in enumerate(["T1", "T2", "T3"]):
        ax = axes[row, col]
        subset = df[(df["task"] == task) & (df["anomaly"].isin(["healthy", anom]))].copy()
        subset["label"] = subset.apply(
            lambda r: "healthy" if r["anomaly"] == "healthy" else r["severity"], axis=1)
        cats = [o for o in order if o in subset["label"].values]
        subset["label"] = pd.Categorical(subset["label"], categories=cats, ordered=True)
        subset.boxplot(column=feat_col, by="label", ax=ax, grid=False,
                       boxprops=dict(linewidth=1.2), medianprops=dict(color="red"))
        ax.set_title(f"{task} — {anom}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if col == 0 else "")
        fig.suptitle("Anomaly Severity Scaling — Per-Cycle Features", fontsize=13)

plt.tight_layout()
fig.savefig(os.path.join(FIG_SUP, "NB7_QC_severity.png"), dpi=1200)
fig.savefig(os.path.join(FIG_SUP, "NB7_QC_severity.pdf"))
plt.show()
print(f"Saved: {FIG_SUP}/NB7_QC_severity.png/.pdf")


# %% Cell 11 — QC Figure 3: Healthy baseline stability across cycles
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Healthy Baseline Stability Over Cycles", fontsize=13)

healthy = df[df["anomaly"] == "healthy"]
for j, jn in enumerate(["J1_shoulder", "J2_elbow"]):
    for col, task in enumerate(["T1", "T2", "T3"]):
        ax = axes[j, col]
        sub = healthy[healthy["task"] == task]
        m = sub[f"{jn}_mean"].mean()
        ax.plot(sub["cycle_num"], sub[f"{jn}_mean"], ".", alpha=0.25, ms=2, color="#333")
        ax.axhline(m, color="red", ls="--", lw=0.8, label=f"mean={m:.3f}A")
        ax.set_title(f"{task} — {jn}", fontsize=10)
        ax.set_xlabel("Cycle" if j == 1 else "")
        ax.set_ylabel("Mean current (A)" if col == 0 else "")
        ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(FIG_SUP, "NB7_QC_stability.png"), dpi=1200)
fig.savefig(os.path.join(FIG_SUP, "NB7_QC_stability.pdf"))
plt.show()
print(f"Saved: {FIG_SUP}/NB7_QC_stability.png/.pdf")


# %% Cell 12 — Descriptive statistics table
summary = df.groupby(["task", "anomaly", "severity"]).agg(
    n=("cycle_num", "count"),
    J1_mean=("J1_shoulder_mean", "mean"),
    J2_mean=("J2_elbow_mean", "mean"),
    J2_std=("J2_elbow_std", "mean"),
    total_rms=("total_rms", "mean"),
).round(4)

summary.to_csv(os.path.join(OUT_DIR, "feature_summary.csv"))
print(summary.to_string())
print(f"\nNB complete. {df.shape[0]} cycles x {len(feats)} features → {OUT_DIR}")
print(f"Fig → {FIG_SUP}")
