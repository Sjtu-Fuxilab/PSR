import os
import glob
import warnings
import numpy as np
import pandas as pd
import h5py
import scipy.stats as sst
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
np.random.seed(42)

# PATHS
ROOT    = r"D:\Research\R"
BASE    = os.path.join(ROOT, "L_Data")
OUT     = os.path.join(ROOT, "P_Data")
os.makedirs(OUT, exist_ok=True)

AGG_PATH   = os.path.join(OUT, "NBg_ablation_aggregate.csv")
ANOM_PATH  = os.path.join(OUT, "NBg_ablation_per_anomaly.csv")
RMSE_PATH  = os.path.join(OUT, "NBg_psr_rmse.csv")

# CONSTANTS  — identical to NB10d
TASKS        = ["T1", "T2", "T3"]
TASK_PAYLOAD = {"T1": 0.0, "T2": 1.0, "T3": 3.0}
PAYLOAD_COM  = np.array([0.0, 0.0, 0.05])
GRAVITY      = np.array([0.0, 0.0, -9.81])
RATE         = 125
SUBSAMPLE    = 4
MIN_SAMP     = 200
N_BOOT       = 1000        
N_PCA        = 50          
# UR5 CB3 PARAMETERS
UR5_DH_A     = [0,        -0.42500, -0.39225, 0,       0,       0      ]
UR5_DH_D     = [0.089159,  0,        0,        0.10915, 0.09465, 0.0823 ]
UR5_DH_ALPHA = [np.pi/2,   0,        0,        np.pi/2, -np.pi/2, 0     ]
UR5_MASS     = [3.7000, 8.3930, 2.2750, 1.2190, 1.2190, 0.1879]
UR5_COM      = [
    [ 0.0,     -0.02561,  0.00193],
    [ 0.21250,  0.0,      0.11336],
    [ 0.11993,  0.0,      0.02650],
    [ 0.0,     -0.00180,  0.01634],
    [ 0.0,      0.00180,  0.01634],
    [ 0.0,      0.0,     -0.00116],
]


def dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,    ca,     d   ],
        [0,   0,     0,      1   ]
    ])


def gravity_torque(q, payload_mass=0.0):
    """Numerical gravity torque via finite-difference FK. Identical to NB10d."""
    T = [np.eye(4)]
    for i in range(6):
        T.append(T[-1] @ dh_transform(
            UR5_DH_A[i], UR5_DH_D[i], UR5_DH_ALPHA[i], q[i]))
    com_world = [(T[i+1] @ np.array([*UR5_COM[i], 1.0]))[:3] for i in range(6)]
    masses = list(UR5_MASS)
    if payload_mass > 0:
        masses.append(payload_mass)
        com_world.append((T[6] @ np.array([*PAYLOAD_COM, 1.0]))[:3])
    tau_g = np.zeros(6)
    dq = 1e-6
    for i in range(6):
        qp = q.copy(); qp[i] += dq
        Tp = [np.eye(4)]
        for jj in range(6):
            Tp.append(Tp[-1] @ dh_transform(
                UR5_DH_A[jj], UR5_DH_D[jj], UR5_DH_ALPHA[jj], qp[jj]))
        for jj in range(len(masses)):
            cp = ((Tp[jj+1] @ np.array([*UR5_COM[jj], 1.0]))[:3]
                  if jj < 6
                  else (Tp[6] @ np.array([*PAYLOAD_COM, 1.0]))[:3])
            tau_g[i] += masses[jj] * GRAVITY @ ((cp - com_world[jj]) / dq)
    return tau_g


# FILE REGISTRY
REGISTRY = {
    "T1_healthy":    ("T1_PickPlace/Healthy",  "UR5_T1_healthy_180cyc_*.h5",
                      "T1", "healthy", "none", 0.0),
    "T2_healthy":    ("T2_Assembly/Healthy",    "UR5_T2_healthy_180cyc_*.h5",
                      "T2", "healthy", "none", 0.0),
    "T3_healthy":    ("T3_Palletize/Healthy",   "UR5_T3_healthy_183cyc_*.h5",
                      "T3", "healthy", "none", 0.0),
    "T1_A2_0.5kg":   ("T1_PickPlace/A2", "UR5_T1_A2_0.5kg_gripper_40cyc_*.h5",
                      "T1", "A2", "0.5kg", 0.5),
    "T1_A2_1kg":     ("T1_PickPlace/A2", "UR5_T1_A2_1kg_gripper_40cyc_*.h5",
                      "T1", "A2", "1kg",   1.0),
    "T1_A2_2kg":     ("T1_PickPlace/A2", "UR5_T1_A2_2kg_gripper_40cyc_*.h5",
                      "T1", "A2", "2kg",   2.0),
    "T1_A3_10wraps": ("T1_PickPlace/A3", "UR5_T1_A3_1band_40cyc_*.h5",
                      "T1", "A3", "10wraps", 0.0),
    "T1_A3_17wraps": ("T1_PickPlace/A3", "UR5_T1_A3_3bands_40cyc_*.h5",
                      "T1", "A3", "17wraps", 0.0),
    "T1_A5_20mm":    ("T1_PickPlace/A5", "UR5_T1_A5_20mm_40cyc_*.h5",
                      "T1", "A5", "20mm", 0.0),
    "T1_A5_50mm":    ("T1_PickPlace/A5", "UR5_T1_A5_50mm_40cyc_*.h5",
                      "T1", "A5", "50mm", 0.0),
    "T1_A5_100mm":   ("T1_PickPlace/A5", "UR5_T1_A5_100mm_40cyc_*.h5",
                      "T1", "A5", "100mm", 0.0),
    "T2_A2_1.5kg":   ("T2_Assembly/A2", "UR5_T2_A2_1.5kg_gripper_40cyc_*.h5",
                      "T2", "A2", "1.5kg", 1.5),
    "T2_A2_2kg":     ("T2_Assembly/A2", "UR5_T2_A2_2kg_gripper_40cyc_*.h5",
                      "T2", "A2", "2kg",   2.0),
    "T2_A2_3kg":     ("T2_Assembly/A2", "UR5_T2_A2_3kg_gripper_40cyc_*.h5",
                      "T2", "A2", "3kg",   3.0),
    "T2_A3_7duct":   ("T2_Assembly/A3", "UR5_T2_A3_light_duct_40cyc_*.h5",
                      "T2", "A3", "7duct", 0.0),
    "T2_A3_14duct":  ("T2_Assembly/A3",
                      "UR5_T2_A3_medium_duct_40cyc_*_225508.h5",
                      "T2", "A3", "14duct", 0.0),
    "T2_A5_20mm":    ("T2_Assembly/A5", "UR5_T2_A5_20mm_40cyc_*.h5",
                      "T2", "A5", "20mm", 0.0),
    "T2_A5_50mm":    ("T2_Assembly/A5", "UR5_T2_A5_50mm_40cyc_*.h5",
                      "T2", "A5", "50mm", 0.0),
    "T2_A5_100mm":   ("T2_Assembly/A5", "UR5_T2_A5_100mm_40cyc_*.h5",
                      "T2", "A5", "100mm", 0.0),
    "T3_A2_3.5kg":   ("T3_Palletize/A2",
                      "UR5_T3_A2_3.5kg_gripper_33cyc_*.h5",
                      "T3", "A2", "3.5kg", 3.5),
    "T3_A2_4kg":     ("T3_Palletize/A2", "UR5_T3_A2_4kg_gripper_33cyc_*.h5",
                      "T3", "A2", "4kg",   4.0),
    "T3_A2_5kg":     ("T3_Palletize/A2",
                      "UR5_T3_A2_4.5kg_gripper_33cyc_*.h5",
                      "T3", "A2", "5kg",   5.0),
    "T3_A3_7duct":   ("T3_Palletize/A3",
                      "UR5_T3_A3_light_duct_33cyc_*_222457.h5",
                      "T3", "A3", "7duct", 0.0),
    "T3_A3_14duct":  ("T3_Palletize/A3",
                      "UR5_T3_A3_medium_duct_33cyc_*_205648.h5",
                      "T3", "A3", "14duct", 0.0),
    "T3_A5_20mm":    ("T3_Palletize/A5",
                      "UR5_T3_A5_20mm_33cyc_*_172334.h5",
                      "T3", "A5", "20mm", 0.0),
    "T3_A5_50mm":    ("T3_Palletize/A5",
                      "UR5_T3_A5_50mm_33cyc_*_164447.h5",
                      "T3", "A5", "50mm", 0.0),
    "T3_A5_100mm":   ("T3_Palletize/A5",
                      "UR5_T3_A5_100mm_33cyc_*_160716.h5",
                      "T3", "A5", "100mm", 0.0),
}

# PHYSICS TERM DEFINITIONS
# Each phi_fn(tau_g, qd, sgn, qdd) → 1D array = one row of the regressor matrix
# w[-1] is always the bias term; w[0] is tau_g weight when gravity is present.

PHI_FNS = {
    "bias_only":        lambda tg, qd, sg, qa: np.array([1.0]),
    "inertia_only":     lambda tg, qd, sg, qa: np.array([qa, 1.0]),
    "friction_only":    lambda tg, qd, sg, qa: np.array([qd, sg, 1.0]),
    "gravity_only":     lambda tg, qd, sg, qa: np.array([tg, 1.0]),
    "M3_grav_vel_fric": lambda tg, qd, sg, qa: np.array([tg, qd, sg, 1.0]),
    "M4_full":          lambda tg, qd, sg, qa: np.array([tg, qd, sg, qa, 1.0]),
}

# Which conditions have a gravity term (needed for grav_resid computation)?
HAS_GRAVITY = {
    "bias_only": False, "inertia_only": False, "friction_only": False,
    "gravity_only": True, "M3_grav_vel_fric": True, "M4_full": True,
}

# All 8 conditions in order for the output table
ALL_CONDITIONS = [
    "no_physics",
    "no_physics_pca50",
    "bias_only",
    "inertia_only",
    "friction_only",
    "gravity_only",
    "M3_grav_vel_fric",
    "M4_full",
]

ANOMALY_TYPES = ["A2", "A3", "A5"]

# UTILITY FUNCTIONS

def bootstrap_auroc_bca(y_true, y_score, n_boot=N_BOOT, seed=42):
    """BCa bootstrap 95% CI for AUROC. Identical to NBd."""
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    auroc_obs = roc_auc_score(y_true, y_score)
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; ys = y_score[idx]
        boot[b] = roc_auc_score(yt, ys) if 0 < yt.sum() < n else auroc_obs
    prop = np.clip(np.mean(boot < auroc_obs), 1e-6, 1 - 1e-6)
    z0   = sst.norm.ppf(prop)
    jack = np.zeros(n)
    for i in range(n):
        idx_j = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        yt_j = y_true[idx_j]; ys_j = y_score[idx_j]
        jack[i] = (roc_auc_score(yt_j, ys_j)
                   if 0 < yt_j.sum() < len(yt_j) else auroc_obs)
    jm  = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a   = num / den if den != 0 else 0.0
    ci  = {}
    for label, za in [("lo", sst.norm.ppf(0.025)), ("hi", sst.norm.ppf(0.975))]:
        p = sst.norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
        ci[label] = float(np.quantile(boot, np.clip(p, 0.001, 0.999)))
    return float(auroc_obs), ci["lo"], ci["hi"]


def zscore_scores(Xtr, Xte):
    """Z-Score anomaly scorer. Identical to NB10d."""
    mu = Xtr.mean(0); sg = Xtr.std(0) + 1e-8
    return np.abs((Xte - mu) / sg).mean(1)


# STEP 1 — Load all cycles from HDF5
print("=" * 65)
print("NB10g — Full Physics Term Ablation (Strict LOTO)")
print("=" * 65)
print("\n[Step 1] Loading HDF5 data...")

all_cycles = []
for key, (subdir, pattern, task, anomaly, severity, _) in REGISTRY.items():
    matches = sorted(glob.glob(os.path.join(BASE, subdir, pattern)))
    if not matches:
        print(f"  WARNING — not found: {key}")
        continue
    with h5py.File(matches[0], "r") as f:
        cnum    = f["cycle_number"][:].astype(int).ravel()
        q_all   = f["actual_q"][:]
        qd_all  = f["actual_qd"][:]
        cur_all = f["actual_current"][:]
    is_anom = 0 if anomaly == "healthy" else 1
    for c in np.unique(cnum[cnum > 0]):
        mask = cnum == c
        if mask.sum() >= MIN_SAMP:
            all_cycles.append({
                "q":         q_all[mask],
                "qd":        qd_all[mask],
                "current":   cur_all[mask],
                "task":      task,
                "anomaly":   anomaly,
                "severity":  severity,
                "is_anomaly": is_anom,
            })

healthy_cycles = [c for c in all_cycles if c["is_anomaly"] == 0]
print(f"  Total cycles: {len(all_cycles)} | Healthy: {len(healthy_cycles)}")
for t in TASKS:
    nh = sum(1 for c in healthy_cycles if c["task"] == t)
    na = sum(1 for c in all_cycles     if c["task"] == t and c["is_anomaly"] == 1)
    print(f"    {t}: {nh} healthy, {na} anomaly")


# STEP 2 — Precompute gravity torques (one-time, cached per cycle)
print("\n[Step 2] Precomputing gravity torques (one-time; ~30 min)...")
total_cycs = len(all_cycles)
for ci, cyc in enumerate(all_cycles):
    payload = TASK_PAYLOAD[cyc["task"]]
    q_a  = cyc["q"]
    qd_a = cyc["qd"]
    N    = len(q_a)
    idx  = list(range(0, N, SUBSAMPLE))
    n_sub = len(idx)
    tau_g_arr = np.zeros((n_sub, 6))
    qdd_arr   = np.zeros((n_sub, 6))
    for ti, t in enumerate(idx):
        tau_g_arr[ti] = gravity_torque(q_a[t], payload_mass=payload)
        for j in range(6):
            qdd_arr[ti, j] = ((qd_a[t+1, j] - qd_a[t-1, j]) * RATE / 2.0
                              if 0 < t < N - 1 else 0.0)
    cyc["tau_g_cached"] = tau_g_arr
    cyc["qdd_cached"]   = qdd_arr
    cyc["sub_idx"]      = idx
    if (ci + 1) % 100 == 0 or (ci + 1) == total_cycs:
        print(f"  {ci+1}/{total_cycs} cycles precomputed...")

print("  Gravity torque precomputation complete.")


# STEP 3 — Precompute raw 104-dim features for no_physics conditions
# Raw feature set matches NB7_feature_extraction.py exactly:
# 17 statistical features × 6 joints + 2 cross-joint correlations = 104.
# Features: mean, std, min, max, range, rms, skew, kurtosis, mad,
#           peak-to-peak, crest_factor, zcr, energy,
#           dom_fft_freq, dom_fft_mag, spectral_centroid, spectral_bandwidth.

print("\n[Step 3] Extracting raw 104-dim features for no_physics conditions...")

def extract_raw_features(cycle):
    cur = cycle["current"]   # shape (N, 6)
    f = {}
    for j in range(6):
        s = cur[:, j]
        N = len(s)
        rms = np.sqrt(np.mean(s**2))
        peak = np.abs(s).max()
        f[f"J{j}_mean"]         = s.mean()
        f[f"J{j}_std"]          = s.std()
        f[f"J{j}_min"]          = s.min()
        f[f"J{j}_max"]          = s.max()
        f[f"J{j}_range"]        = s.max() - s.min()
        f[f"J{j}_rms"]          = rms
        f[f"J{j}_skew"]         = float(sst.skew(s))
        f[f"J{j}_kurtosis"]     = float(sst.kurtosis(s))
        f[f"J{j}_mad"]          = float(np.mean(np.abs(s - s.mean())))
        f[f"J{j}_peak2peak"]    = peak * 2.0
        f[f"J{j}_crest"]        = float(peak / rms) if rms > 0 else 0.0
        # Zero-crossing rate
        zcr = float(np.sum(np.diff(np.sign(s - s.mean())) != 0)) / N
        f[f"J{j}_zcr"]          = zcr
        f[f"J{j}_energy"]       = float(np.sum(s**2))
        # FFT-based features
        freqs  = np.fft.rfftfreq(N, d=1.0/RATE)
        mag    = np.abs(np.fft.rfft(s - s.mean()))
        if mag.sum() > 0:
            f[f"J{j}_dom_fft_freq"] = float(freqs[np.argmax(mag)])
            f[f"J{j}_dom_fft_mag"]  = float(mag.max())
            weights = mag / mag.sum()
            f[f"J{j}_spec_centroid"] = float(np.dot(freqs, weights))
            f[f"J{j}_spec_bw"]       = float(
                np.sqrt(np.dot((freqs - f[f"J{j}_spec_centroid"])**2, weights)))
        else:
            f[f"J{j}_dom_fft_freq"] = 0.0
            f[f"J{j}_dom_fft_mag"]  = 0.0
            f[f"J{j}_spec_centroid"] = 0.0
            f[f"J{j}_spec_bw"]       = 0.0
    # Cross-joint correlations
    f["J1J2_corr"] = float(np.corrcoef(cur[:, 1], cur[:, 2])[0, 1])
    f["J2J3_corr"] = float(np.corrcoef(cur[:, 2], cur[:, 3])[0, 1])
    return f

# Attach raw features to each cycle
RAW_FEAT_COLS = None
for cyc in all_cycles:
    rf = extract_raw_features(cyc)
    cyc["raw_features"] = rf
    if RAW_FEAT_COLS is None:
        RAW_FEAT_COLS = sorted(rf.keys())

print(f"  Raw features extracted: {len(RAW_FEAT_COLS)} features per cycle.")


# HELPER FUNCTIONS — physics feature extraction using cached torques

def fit_psr_weights(cycles, phi_fn):
    """Fit OLS PSR weights per joint using cached torques."""
    Phi = [[] for _ in range(6)]
    I   = [[] for _ in range(6)]
    for cyc in cycles:
        qd_a  = cyc["qd"]
        cur   = cyc["current"]
        tau_g = cyc["tau_g_cached"]
        qdd   = cyc["qdd_cached"]
        for ti, t in enumerate(cyc["sub_idx"]):
            for j in range(6):
                phi_j = phi_fn(tau_g[ti, j], qd_a[t, j],
                               np.sign(qd_a[t, j]), qdd[ti, j])
                Phi[j].append(phi_j)
                I[j].append(cur[t, j])
    weights = {}
    for j in range(6):
        w, _, _, _ = np.linalg.lstsq(
            np.array(Phi[j]), np.array(I[j]), rcond=None)
        weights[j] = w
    return weights


def psr_rmse_on_cycles(cycles, phi_fn, weights):
    """Compute per-joint RMSE and R² on a set of cycles using fitted weights.
    Used for regression quality reporting on held-out healthy test cycles.
    """
    res_sums  = np.zeros(6)
    ss_tot    = np.zeros(6)
    n_pts     = np.zeros(6)
    cur_means = np.zeros(6)
    cur_counts = np.zeros(6)
    # Pass 1: compute current means for R²
    for cyc in cycles:
        cur = cyc["current"]
        for j in range(6):
            for t in cyc["sub_idx"]:
                cur_means[j]  += cur[t, j]
                cur_counts[j] += 1
    cur_means /= np.maximum(cur_counts, 1)
    # Pass 2: compute residuals
    for cyc in cycles:
        qd_a  = cyc["qd"]
        cur   = cyc["current"]
        tau_g = cyc["tau_g_cached"]
        qdd   = cyc["qdd_cached"]
        for ti, t in enumerate(cyc["sub_idx"]):
            for j in range(6):
                phi_j = phi_fn(tau_g[ti, j], qd_a[t, j],
                               np.sign(qd_a[t, j]), qdd[ti, j])
                resid = cur[t, j] - phi_j @ weights[j]
                res_sums[j]  += resid**2
                ss_tot[j]    += (cur[t, j] - cur_means[j])**2
                n_pts[j]     += 1
    rmse = np.sqrt(res_sums / np.maximum(n_pts, 1))
    r2   = 1.0 - res_sums / np.maximum(ss_tot, 1e-12)
    return rmse, r2


def extract_psr_features(cycles, phi_fn, weights, has_gravity):
    """Extract 50-dim PSR feature vector per cycle.
    Matches NB10d / NB10e feature extraction exactly.
    For conditions without a gravity term, grav_resid = full_resid.
    """
    rows = []
    for cyc in cycles:
        qd_a  = cyc["qd"]
        cur   = cyc["current"]
        tau_g = cyc["tau_g_cached"]
        qdd   = cyc["qdd_cached"]
        n_sub = len(cyc["sub_idx"])
        res   = np.zeros((n_sub, 6))
        gr    = np.zeros((n_sub, 6))
        for ti, t in enumerate(cyc["sub_idx"]):
            for j in range(6):
                phi_j = phi_fn(tau_g[ti, j], qd_a[t, j],
                               np.sign(qd_a[t, j]), qdd[ti, j])
                res[ti, j] = cur[t, j] - phi_j @ weights[j]
                if has_gravity:
                    # gravity-specific residual: remove only tau_g + bias
                    w = weights[j]
                    gr[ti, j] = cur[t, j] - (w[0] * tau_g[ti, j] + w[-1])
                else:
                    # no gravity term — grav_resid equals full residual
                    gr[ti, j] = res[ti, j]
        f = {}
        for j in range(6):
            r = res[:, j]; g = gr[:, j]
            f[f"J{j}_resid_mean"]      = r.mean()
            f[f"J{j}_resid_std"]       = r.std()
            f[f"J{j}_resid_rms"]       = np.sqrt(np.mean(r**2))
            f[f"J{j}_resid_max"]       = np.abs(r).max()
            f[f"J{j}_resid_skew"]      = float(sst.skew(r))
            f[f"J{j}_resid_kurtosis"]  = float(sst.kurtosis(r))
            f[f"J{j}_grav_resid_std"]  = g.std()
            f[f"J{j}_grav_resid_rms"]  = np.sqrt(np.mean(g**2))
        f["total_resid_rms"] = np.sqrt(np.mean(res**2))
        f["J1J2_resid_corr"] = float(
            np.corrcoef(res[:, 1], res[:, 2])[0, 1] if len(res) > 2 else 0.0)
        f["task"]       = cyc["task"]
        f["anomaly"]    = cyc["anomaly"]
        f["is_anomaly"] = cyc["is_anomaly"]
        rows.append(f)
    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns
                 if c not in ("task", "anomaly", "is_anomaly")]
    return df, feat_cols


# CHECKPOINT HELPERS

def load_existing_agg():
    if os.path.exists(AGG_PATH):
        return pd.read_csv(AGG_PATH)
    return pd.DataFrame()


def load_existing_anom():
    if os.path.exists(ANOM_PATH):
        return pd.read_csv(ANOM_PATH)
    return pd.DataFrame()


def load_existing_rmse():
    if os.path.exists(RMSE_PATH):
        return pd.read_csv(RMSE_PATH)
    return pd.DataFrame()


def condition_done(df, condition):
    if df.empty or "condition" not in df.columns:
        return False
    return condition in df["condition"].values


def save_rows(df_existing, new_rows, path):
    df_new = pd.DataFrame(new_rows)
    if not df_existing.empty:
        combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        combined = df_new
    combined.to_csv(path, index=False)
    return combined


# MAIN ABLATION LOOP
print("\n" + "=" * 65)
print("Full Ablation — 8 conditions × 3 LOTO folds")
print("=" * 65)

df_agg  = load_existing_agg()
df_anom = load_existing_anom()
df_rmse = load_existing_rmse()

# Collect columns for feature matrices
PSR_FEAT_COLS_50 = (
    [f"J{j}_{s}" for j in range(6)
     for s in ["resid_mean", "resid_std", "resid_rms", "resid_max",
               "resid_skew", "resid_kurtosis",
               "grav_resid_std", "grav_resid_rms"]]
    + ["total_resid_rms", "J1J2_resid_corr"]
)

for condition in ALL_CONDITIONS:

    if condition_done(df_agg, condition):
        print(f"\n  [CHECKPOINT] {condition} already complete — skipping.")
        continue

    print(f"\n  ---- Condition: {condition} ----")
    agg_rows  = []
    anom_rows = []
    rmse_rows = []

    for test_task in TASKS:
        tr_tasks  = [t for t in TASKS if t != test_task]
        tr_cycs   = [c for c in all_cycles if c["task"] in tr_tasks]
        te_cycs   = [c for c in all_cycles if c["task"] == test_task]
        tr_healthy = [c for c in tr_cycs if c["is_anomaly"] == 0]
        te_healthy = [c for c in te_cycs if c["is_anomaly"] == 0]

        # Feature extraction per condition

        if condition == "no_physics":
            # Raw 104-dim features — no fold-specific fitting required
            Xtr = np.array([[c["raw_features"][k] for k in RAW_FEAT_COLS]
                            for c in tr_healthy])
            X_te_all = np.array([[c["raw_features"][k] for k in RAW_FEAT_COLS]
                                 for c in te_cycs])
            y_te = np.array([c["is_anomaly"] for c in te_cycs])
            anom_te = [c["anomaly"] for c in te_cycs]
            feat_label = "raw104"

        elif condition == "no_physics_pca50":
            # PCA-50 fitted strictly on training-fold healthy features
            Xtr_raw = np.array([[c["raw_features"][k] for k in RAW_FEAT_COLS]
                                for c in tr_healthy])
            X_te_raw = np.array([[c["raw_features"][k] for k in RAW_FEAT_COLS]
                                 for c in te_cycs])
            scaler  = StandardScaler().fit(Xtr_raw)
            pca     = PCA(n_components=N_PCA, random_state=42)
            pca.fit(scaler.transform(Xtr_raw))
            Xtr = pca.transform(scaler.transform(Xtr_raw))
            X_te_all = pca.transform(scaler.transform(X_te_raw))
            y_te = np.array([c["is_anomaly"] for c in te_cycs])
            anom_te = [c["anomaly"] for c in te_cycs]
            feat_label = "pca50"

        else:
            # Physics condition: fit PSR weights on training-fold healthy only
            phi_fn     = PHI_FNS[condition]
            has_grav   = HAS_GRAVITY[condition]
            weights    = fit_psr_weights(tr_healthy, phi_fn)

            # Extract 50-dim residual features on all test cycles
            df_te, feat_cols = extract_psr_features(
                te_cycs, phi_fn, weights, has_grav)
            # Extract on training healthy for Z-Score reference
            df_tr, _ = extract_psr_features(
                tr_healthy, phi_fn, weights, has_grav)

            Xtr       = df_tr[PSR_FEAT_COLS_50].values
            X_te_all  = df_te[PSR_FEAT_COLS_50].values
            y_te      = df_te["is_anomaly"].values
            anom_te   = df_te["anomaly"].tolist()
            feat_label = "psr50"

            # PSR RMSE on held-out healthy test cycles
            rmse_vals, r2_vals = psr_rmse_on_cycles(te_healthy, phi_fn, weights)
            for j in range(6):
                rmse_rows.append({
                    "condition": condition,
                    "test_task": test_task,
                    "joint":     f"J{j}",
                    "rmse":      float(rmse_vals[j]),
                    "r2":        float(r2_vals[j]),
                })

        # Z-Score anomaly scoring
        scores = zscore_scores(Xtr, X_te_all)

        # Aggregate AUROC
        auroc, lo, hi = bootstrap_auroc_bca(y_te, scores)
        agg_rows.append({
            "condition": condition,
            "test_task": test_task,
            "feat_type": feat_label,
            "auroc":     round(auroc, 4),
            "ci_lo":     round(lo, 4),
            "ci_hi":     round(hi, 4),
            "n_healthy": int((y_te == 0).sum()),
            "n_anomaly": int((y_te == 1).sum()),
        })
        print(f"    {test_task}  aggregate AUROC = {auroc:.4f} [{lo:.4f}, {hi:.4f}]")

        # Per-anomaly-type AUROC
        for anom_type in ANOMALY_TYPES:
            # Test set: healthy(test_task) + anomaly_type(test_task)
            mask_pa = np.array(
                [(c["is_anomaly"] == 0 or c["anomaly"] == anom_type)
                 for c in te_cycs])
            y_pa    = y_te[mask_pa]
            sc_pa   = scores[mask_pa]
            if y_pa.sum() == 0:
                continue
            auroc_pa, lo_pa, hi_pa = bootstrap_auroc_bca(y_pa, sc_pa)
            anom_rows.append({
                "condition":   condition,
                "test_task":   test_task,
                "anomaly_type": anom_type,
                "auroc":       round(auroc_pa, 4),
                "ci_lo":       round(lo_pa, 4),
                "ci_hi":       round(hi_pa, 4),
            })
        print(f"          per-anomaly: A2={anom_rows[-3]['auroc']:.4f}  "
              f"A3={anom_rows[-2]['auroc']:.4f}  A5={anom_rows[-1]['auroc']:.4f}")

    # Save after each condition completes
    df_agg  = save_rows(df_agg,  agg_rows,  AGG_PATH)
    df_anom = save_rows(df_anom, anom_rows, ANOM_PATH)
    if rmse_rows:
        df_rmse = save_rows(df_rmse, rmse_rows, RMSE_PATH)
    print(f"  [CHECKPOINT] {condition} saved.")


# SUMMARY TABLES
print("\n" + "=" * 65)
print("NB10g — Ablation Results Summary")
print("=" * 65)

df_agg  = pd.read_csv(AGG_PATH)
df_anom = pd.read_csv(ANOM_PATH)

# Aggregate table: condition × task with mean
pivot = df_agg.pivot_table(
    index="condition", columns="test_task", values="auroc")
pivot = pivot.reindex(ALL_CONDITIONS)
pivot["mean"] = pivot[TASKS].mean(axis=1)

print("\n  Aggregate AUROC (Z-Score, strict LOTO):")
print(f"  {'Condition':<22} {'T1':>8} {'T2':>8} {'T3':>8} {'Mean':>8}")
print("  " + "-" * 56)
for cond in ALL_CONDITIONS:
    if cond not in pivot.index:
        continue
    row = pivot.loc[cond]
    print(f"  {cond:<22} {row['T1']:>8.4f} {row['T2']:>8.4f} "
          f"{row['T3']:>8.4f} {row['mean']:>8.4f}")

# Per-anomaly table: for M4_full vs key ablation conditions
print("\n  Per-anomaly AUROC for selected conditions (mean across 3 tasks):")
print(f"  {'Condition':<22} {'A2':>8} {'A3':>8} {'A5':>8} {'Mean':>8}")
print("  " + "-" * 56)
df_anom_mean = df_anom.groupby(["condition", "anomaly_type"])["auroc"].mean()
for cond in ALL_CONDITIONS:
    if cond not in df_anom["condition"].values:
        continue
    vals = {}
    for at in ANOMALY_TYPES:
        try:
            vals[at] = df_anom_mean.loc[(cond, at)]
        except KeyError:
            vals[at] = float("nan")
    mean_a = np.nanmean(list(vals.values()))
    print(f"  {cond:<22} {vals.get('A2', float('nan')):>8.4f} "
          f"{vals.get('A3', float('nan')):>8.4f} "
          f"{vals.get('A5', float('nan')):>8.4f} {mean_a:>8.4f}")

# PSR RMSE on test-fold healthy: mean across joints per condition × task
if os.path.exists(RMSE_PATH):
    df_rmse = pd.read_csv(RMSE_PATH)
    pivot_rmse = df_rmse.groupby(["condition", "test_task"])["rmse"].mean().unstack()
    print("\n  Mean per-joint RMSE on test-fold healthy cycles (physics conditions):")
    print(f"  {'Condition':<22} {'T1':>8} {'T2':>8} {'T3':>8}")
    print("  " + "-" * 48)
    for cond in PHI_FNS.keys():
        if cond not in pivot_rmse.index:
            continue
        row = pivot_rmse.loc[cond]
        t1  = row.get("T1", float("nan"))
        t2  = row.get("T2", float("nan"))
        t3  = row.get("T3", float("nan"))
        print(f"  {cond:<22} {t1:>8.4f} {t2:>8.4f} {t3:>8.4f}")
    # Add M4_full
    for cond in ["gravity_only", "M3_grav_vel_fric", "M4_full"]:
        pass  # Already printed above via PHI_FNS.keys()

print("\n" + "=" * 65)
print("NBg COMPLETE")
