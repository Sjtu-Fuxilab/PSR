import os, glob, warnings
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# PATHS
ROOT = r"D:\Research\R"
BASE = os.path.join(ROOT, "L_Data")
OUT  = os.path.join(ROOT, "P_Data")
SUPP = os.path.join(ROOT, "M_Data", "Figures", "Supplementary")
for d in [OUT, SUPP]:
    os.makedirs(d, exist_ok=True)

# UR5 CB3 DYNAMICS - exact parameters from NB
UR5_DH_A     = [0, -0.42500, -0.39225, 0, 0, 0]
UR5_DH_D     = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
UR5_DH_ALPHA = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
UR5_MASS     = [3.7000, 8.3930, 2.2750, 1.2190, 1.2190, 0.1879]
UR5_COM      = [
    [0.0,     -0.02561,  0.00193],
    [0.21250,  0.0,      0.11336],
    [0.11993,  0.0,      0.02650],
    [0.0,     -0.00180,  0.01634],
    [0.0,      0.00180,  0.01634],
    [0.0,      0.0,     -0.00116],
]
GRAVITY = np.array([0, 0, -9.81])
TASK_PAYLOAD = {"T1": 0.0, "T2": 1.0, "T3": 3.0}
PAYLOAD_COM  = np.array([0, 0, 0.05])


def dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,    sa,    ca,    d   ],
        [0,    0,     0,     1   ]
    ])


def gravity_torque(q, payload_mass=0.0):
    """Gravity torques via numerical Jacobian (exact method from NB8v4)."""
    T = [np.eye(4)]
    for i in range(6):
        T.append(T[-1] @ dh_transform(UR5_DH_A[i], UR5_DH_D[i],
                                       UR5_DH_ALPHA[i], q[i]))
    masses = UR5_MASS[:]
    coms   = [np.array(UR5_COM[i]) for i in range(6)]
    if payload_mass > 0:
        masses.append(payload_mass)
        coms.append((T[6] @ np.array([*PAYLOAD_COM, 1.0]))[:3])

    com_world = []
    for i in range(6):
        c_local = np.array([*UR5_COM[i], 1.0])
        com_world.append((T[i+1] @ c_local)[:3])
    if payload_mass > 0:
        com_world.append((T[6] @ np.array([*PAYLOAD_COM, 1.0]))[:3])

    tau_g = np.zeros(6)
    dq    = 1e-6
    for i in range(6):
        qp = q.copy(); qp[i] += dq
        Tp = [np.eye(4)]
        for j in range(6):
            Tp.append(Tp[-1] @ dh_transform(UR5_DH_A[j], UR5_DH_D[j],
                                             UR5_DH_ALPHA[j], qp[j]))
        for j in range(len(masses)):
            if j < 6:
                cp = (Tp[j+1] @ np.array([*UR5_COM[j], 1.0]))[:3]
            else:
                cp = (Tp[6] @ np.array([*PAYLOAD_COM, 1.0]))[:3]
            dp_dq    = (cp - com_world[j]) / dq
            tau_g[i] += masses[j] * GRAVITY @ dp_dq
    return tau_g


# FILE REGISTRY - exact from NB
REGISTRY = {
    "T1_healthy":    ("T1_PickPlace/Healthy",  "UR5_T1_healthy_180cyc_*.h5",          "T1", "healthy", "none",    0.0),
    "T2_healthy":    ("T2_Assembly/Healthy",    "UR5_T2_healthy_180cyc_*.h5",          "T2", "healthy", "none",    0.0),
    "T3_healthy":    ("T3_Palletize/Healthy",   "UR5_T3_healthy_183cyc_*.h5",          "T3", "healthy", "none",    0.0),
    "T1_A2_0.5kg":   ("T1_PickPlace/A2",        "UR5_T1_A2_0.5kg_gripper_40cyc_*.h5", "T1", "A2",      "0.5kg",   0.5),
    "T1_A2_1kg":     ("T1_PickPlace/A2",        "UR5_T1_A2_1kg_gripper_40cyc_*.h5",   "T1", "A2",      "1kg",     1.0),
    "T1_A2_2kg":     ("T1_PickPlace/A2",        "UR5_T1_A2_2kg_gripper_40cyc_*.h5",   "T1", "A2",      "2kg",     2.0),
    "T1_A3_10wraps": ("T1_PickPlace/A3",        "UR5_T1_A3_1band_40cyc_*.h5",         "T1", "A3",      "10wraps", 0.0),
    "T1_A3_17wraps": ("T1_PickPlace/A3",        "UR5_T1_A3_3bands_40cyc_*.h5",        "T1", "A3",      "17wraps", 0.0),
    "T1_A5_20mm":    ("T1_PickPlace/A5",        "UR5_T1_A5_20mm_40cyc_*.h5",          "T1", "A5",      "20mm",    0.0),
    "T1_A5_50mm":    ("T1_PickPlace/A5",        "UR5_T1_A5_50mm_40cyc_*.h5",          "T1", "A5",      "50mm",    0.0),
    "T1_A5_100mm":   ("T1_PickPlace/A5",        "UR5_T1_A5_100mm_40cyc_*.h5",         "T1", "A5",      "100mm",   0.0),
    "T2_A2_1.5kg":   ("T2_Assembly/A2",         "UR5_T2_A2_1.5kg_gripper_40cyc_*.h5", "T2", "A2",      "0.5kg",   0.5),
    "T2_A2_2kg":     ("T2_Assembly/A2",         "UR5_T2_A2_2kg_gripper_40cyc_*.h5",   "T2", "A2",      "1kg",     1.0),
    "T2_A2_3kg":     ("T2_Assembly/A2",         "UR5_T2_A2_3kg_gripper_40cyc_*.h5",   "T2", "A2",      "2kg",     2.0),
    "T2_A3_7duct":   ("T2_Assembly/A3",         "UR5_T2_A3_light_duct_40cyc_*_214735.h5",  "T2", "A3", "7wraps",  0.0),
    "T2_A3_14duct":  ("T2_Assembly/A3",         "UR5_T2_A3_medium_duct_40cyc_*_225508.h5", "T2", "A3", "14wraps", 0.0),
    "T2_A5_20mm":    ("T2_Assembly/A5",         "UR5_T2_A5_20mm_40cyc_*.h5",          "T2", "A5",      "20mm",    0.0),
    "T2_A5_50mm":    ("T2_Assembly/A5",         "UR5_T2_A5_50mm_40cyc_*.h5",          "T2", "A5",      "50mm",    0.0),
    "T2_A5_100mm":   ("T2_Assembly/A5",         "UR5_T2_A5_100mm_40cyc_*.h5",         "T2", "A5",      "100mm",   0.0),
    "T3_A2_3.5kg":   ("T3_Palletize/A2",        "UR5_T3_A2_3.5kg_gripper_33cyc_*.h5", "T3", "A2",      "0.5kg",   0.5),
    "T3_A2_4kg":     ("T3_Palletize/A2",        "UR5_T3_A2_4kg_gripper_33cyc_*.h5",   "T3", "A2",      "1kg",     1.0),
    "T3_A2_5kg":     ("T3_Palletize/A2",        "UR5_T3_A2_4.5kg_gripper_33cyc_*.h5", "T3", "A2",      "2kg",     2.0),
    "T3_A3_14duct":  ("T3_Palletize/A3",        "UR5_T3_A3_medium_duct_33cyc_*.h5",   "T3", "A3",      "14wraps", 0.0),
    "T3_A3_7duct":   ("T3_Palletize/A3",        "UR5_T3_A3_light_duct_33cyc_*.h5",    "T3", "A3",      "7wraps",  0.0),
    "T3_A5_20mm":    ("T3_Palletize/A5",        "UR5_T3_A5_20mm_33cyc_*.h5",          "T3", "A5",      "20mm",    0.0),
    "T3_A5_50mm":    ("T3_Palletize/A5",        "UR5_T3_A5_50mm_33cyc_*.h5",          "T3", "A5",      "50mm",    0.0),
    "T3_A5_100mm":   ("T3_Palletize/A5",        "UR5_T3_A5_100mm_33cyc_*.h5",         "T3", "A5",      "100mm",   0.0),
}

# STEP 1 - Load all HDF5 cycles
RATE      = 125
SUBSAMPLE = 4
MIN_SAMP  = 200

print("=" * 65)
print("NB8v5 - PSR Component Ablation Study")
print("=" * 65)
print("\n[Step 1] Loading canonical HDF5 files...")

all_cycles = []
for key, (subdir, pattern, task, anomaly, severity, _) in REGISTRY.items():
    matches = sorted(glob.glob(os.path.join(BASE, subdir, pattern)))
    if not matches:
        print("  WARNING  Not found: " + key)
        continue
    fpath = matches[0]
    with h5py.File(fpath, "r") as f:
        cnum    = f["cycle_number"][:].astype(int).ravel()
        q       = f["actual_q"][:]
        qd      = f["actual_qd"][:]
        current = f["actual_current"][:]

    is_anomaly = 0 if anomaly == "healthy" else 1
    tag = "healthy" if anomaly == "healthy" else anomaly + "_" + severity

    n_cyc = 0
    for c in np.unique(cnum[cnum > 0]):
        mask = cnum == c
        if mask.sum() < MIN_SAMP:
            continue
        all_cycles.append({
            "q": q[mask], "qd": qd[mask], "current": current[mask],
            "task": task, "anomaly": anomaly, "severity": severity,
            "is_anomaly": is_anomaly, "tag": tag,
        })
        n_cyc += 1
    print("  OK  " + key + ": " + str(n_cyc) + " cycles")

print("\n  Total cycles: " + str(len(all_cycles)))
print("  Healthy: " + str(sum(1 for c in all_cycles if c["is_anomaly"] == 0)))
print("  Anomaly: " + str(sum(1 for c in all_cycles if c["is_anomaly"] == 1)))

# ABLATION CONDITIONS
# Regressor indices in 5-vector: [gravity, viscous, coulomb, inertia, bias]
ABLATION_CONDITIONS = {
    "full_PSR":      [0, 1, 2, 3, 4],
    "gravity_only":  [0, 4],
    "friction_only": [1, 2, 4],
    "inertia_only":  [3, 4],
    "bias_only":     [4],
}

# STEP 2 - PSR fitting and residual computation

def fit_psr(healthy_cycles, active_regs):
    """
    Fit PSR using only active regressor indices.
    Uses lstsq on subsampled healthy data (SUBSAMPLE=4, matching NB8v4).
    Returns dict {joint_idx: weight_vector (len=5)}.
    """
    train_Phi = {j: [] for j in range(6)}
    train_I   = {j: [] for j in range(6)}

    for ci, cyc in enumerate(healthy_cycles):
        payload = TASK_PAYLOAD[cyc["task"]]
        q       = cyc["q"]
        qd      = cyc["qd"]
        current = cyc["current"]
        N       = len(q)

        for t in range(0, N, SUBSAMPLE):
            tau_g = gravity_torque(q[t], payload_mass=payload)
            for j in range(6):
                qdd_j = ((qd[t+1, j] - qd[t-1, j]) * RATE / 2.0
                         if 0 < t < N-1 else 0.0)
                phi = np.array([tau_g[j], qd[t,j], np.sign(qd[t,j]), qdd_j, 1.0])
                train_Phi[j].append(phi)
                train_I[j].append(current[t, j])

        if (ci + 1) % 50 == 0:
            print("    " + str(ci+1) + "/" + str(len(healthy_cycles))
                  + " healthy cycles processed")

    psr_weights = {}
    for j in range(6):
        Phi_j = np.array(train_Phi[j])
        I_j   = np.array(train_I[j])
        cols  = list(active_regs)
        w_sub, _, _, _ = np.linalg.lstsq(Phi_j[:, cols], I_j, rcond=None)
        w_full = np.zeros(5)
        for k, idx in enumerate(cols):
            w_full[idx] = w_sub[k]
        psr_weights[j] = w_full

    return psr_weights


def compute_residuals_cycle(cyc, psr_weights):
    """
    Compute residuals for one cycle (subsampled).
    Returns:
      res      (N_sub, 6) — full residual: actual - PSR_predicted
      grav_res (N_sub, 6) — gravity residual: actual - (w[0]*tau_g + w[4])
    Gravity residual retains friction/inertia signal and is specifically
    sensitive to mass anomalies (A2). Matches NB8v4 decomposition exactly.
    """
    payload = TASK_PAYLOAD[cyc["task"]]
    q       = cyc["q"]
    qd      = cyc["qd"]
    current = cyc["current"]
    N       = len(q)

    indices  = list(range(0, N, SUBSAMPLE))
    res      = np.zeros((len(indices), 6))
    grav_res = np.zeros((len(indices), 6))
    for ti, t in enumerate(indices):
        tau_g = gravity_torque(q[t], payload_mass=payload)
        for j in range(6):
            qdd_j = ((qd[t+1,j] - qd[t-1,j]) * RATE / 2.0
                     if 0 < t < N-1 else 0.0)
            phi            = np.array([tau_g[j], qd[t,j], np.sign(qd[t,j]), qdd_j, 1.0])
            w              = psr_weights[j]
            res[ti, j]     = current[t, j] - phi @ w
            # gravity residual: remove only gravity+bias, leave friction/inertia in
            grav_res[ti, j] = current[t, j] - (w[0]*tau_g[j] + w[4])
    return res, grav_res


# STEP 3 - Feature extraction
FEAT_COLS = (
    [f"J{j}_{s}" for j in range(6)
     for s in ["resid_mean", "resid_std", "resid_rms", "resid_max",
               "resid_skew", "resid_kurtosis",
               "grav_resid_std", "grav_resid_rms"]]
    + ["total_resid_rms", "J1J2_resid_corr"]
)
META_COLS = ["task", "anomaly", "severity", "is_anomaly", "tag"]
TASKS     = ["T1", "T2", "T3"]


def extract_features(res, grav_res):
    """
    50 features matching features_residual.csv from NB8v4 exactly.
    res      (N, 6): full residual = actual - PSR_predicted
    grav_res (N, 6): gravity residual = actual - (w[0]*tau_g + w[4])
    grav_resid_std/rms use grav_res, NOT res — this is the key fix.
    """
    feats = {}
    for j in range(6):
        r  = res[:, j]
        gr = grav_res[:, j]
        s  = pd.Series(r)
        feats[f"J{j}_resid_mean"]     = float(r.mean())
        feats[f"J{j}_resid_std"]      = float(r.std())
        feats[f"J{j}_resid_rms"]      = float(np.sqrt(np.mean(r**2)))
        feats[f"J{j}_resid_max"]      = float(np.max(np.abs(r)))
        feats[f"J{j}_resid_skew"]     = float(s.skew())
        feats[f"J{j}_resid_kurtosis"] = float(s.kurtosis())
        feats[f"J{j}_grav_resid_std"] = float(gr.std())          # gravity-specific
        feats[f"J{j}_grav_resid_rms"] = float(np.sqrt(np.mean(gr**2)))  # gravity-specific
    feats["total_resid_rms"] = float(np.sqrt(np.mean(res**2)))
    feats["J1J2_resid_corr"] = float(
        np.corrcoef(res[:,1], res[:,2])[0,1] if len(res) > 2 else 0.0)
    return feats


def build_feature_df(all_cycles, psr_weights):
    rows = []
    for cyc in all_cycles:
        res, grav_res = compute_residuals_cycle(cyc, psr_weights)
        feats = extract_features(res, grav_res)
        feats.update(task=cyc["task"], anomaly=cyc["anomaly"],
                     severity=cyc["severity"], is_anomaly=cyc["is_anomaly"],
                     tag=cyc["tag"])
        rows.append(feats)
    return pd.DataFrame(rows)


# STEP 4 - Anomaly detectors

def zscore_det(X_tr, X_te):
    mu = X_tr.mean(0); sig = X_tr.std(0) + 1e-8
    return np.abs((X_te - mu) / sig).mean(1)


def ocsvm_det(X_tr, X_te):
    sc = StandardScaler().fit(X_tr)
    clf = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    clf.fit(sc.transform(X_tr))
    return -clf.decision_function(sc.transform(X_te))


def if_det(X_tr, X_te):
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    clf.fit(X_tr)
    return -clf.decision_function(X_te)


def pca_det(X_tr, X_te, nc=10):
    sc  = StandardScaler().fit(X_tr)
    Xtr = sc.transform(X_tr); Xte = sc.transform(X_te)
    nc  = min(nc, Xtr.shape[1], Xtr.shape[0])
    pca = PCA(n_components=nc).fit(Xtr)
    rec = pca.inverse_transform(pca.transform(Xte))
    return np.sqrt(((Xte - rec)**2).mean(1))


DETECTORS = {
    "Z-Score":    zscore_det,
    "OC-SVM":     ocsvm_det,
    "Iso-Forest": if_det,
    "PCA-Recon":  pca_det,
}


def loto_evaluate(df, feat_cols=None):
    if feat_cols is None:
        feat_cols = FEAT_COLS
    records = []
    for test_task in TASKS:
        tr_tasks = [t for t in TASKS if t != test_task]
        tr = df[df["task"].isin(tr_tasks) & (df["is_anomaly"] == 0)]
        te = df[df["task"] == test_task]
        if len(tr) == 0 or len(te) == 0 or te["is_anomaly"].nunique() < 2:
            continue
        Xtr = tr[feat_cols].values
        Xte = te[feat_cols].values
        yte = te["is_anomaly"].values
        for dname, dfn in DETECTORS.items():
            try:
                auroc = roc_auc_score(yte, dfn(Xtr, Xte))
            except Exception:
                auroc = np.nan
            records.append(dict(detector=dname, fold_test_task=test_task,
                                auroc=auroc))
    return pd.DataFrame(records)


def loto_per_anomaly(df, feat_cols=None):
    if feat_cols is None:
        feat_cols = FEAT_COLS
    records = []
    for test_task in TASKS:
        tr_tasks = [t for t in TASKS if t != test_task]
        tr  = df[df["task"].isin(tr_tasks) & (df["is_anomaly"] == 0)]
        Xtr = tr[feat_cols].values
        for anom in ["A2", "A3", "A5"]:
            te_h = df[(df["task"] == test_task) & (df["is_anomaly"] == 0)]
            te_a = df[(df["task"] == test_task) & (df["anomaly"] == anom)]
            if len(te_h) == 0 or len(te_a) == 0:
                continue
            te  = pd.concat([te_h, te_a])
            yte = te["is_anomaly"].values
            try:
                auroc = roc_auc_score(yte, zscore_det(Xtr, te[feat_cols].values))
            except Exception:
                auroc = np.nan
            records.append(dict(fold_test_task=test_task,
                                anomaly_type=anom, auroc=auroc))
    return pd.DataFrame(records)

# STEP 5 - Run ablation
print("\n[Step 5] Running PSR ablation (5 conditions)...")
healthy_cycles = [c for c in all_cycles if c["is_anomaly"] == 0]
print("  Healthy cycles for fitting: " + str(len(healthy_cycles)))

ablation_summary  = {}
ablation_per_anom = {}
rmse_records      = []

for cond_name, active_regs in ABLATION_CONDITIONS.items():
    print("\n  Condition: " + cond_name + "  regressors=" + str(active_regs))
    psr_w = fit_psr(healthy_cycles, active_regs)

    # Per-joint RMSE on healthy data
    all_res = np.vstack([compute_residuals_cycle(c, psr_w)[0] for c in healthy_cycles])
    rmse    = np.sqrt(np.mean(all_res**2, axis=0))
    for j, r in enumerate(rmse):
        rmse_records.append(dict(condition=cond_name, joint="J"+str(j), rmse_A=r))
    print("  RMSE: " + "  ".join(f"J{j}={v:.3f}" for j, v in enumerate(rmse)))

    print("  Extracting features for " + str(len(all_cycles)) + " cycles...")
    df = build_feature_df(all_cycles, psr_w)
    ablation_summary[cond_name]  = loto_evaluate(df)
    ablation_per_anom[cond_name] = loto_per_anomaly(df)

    zs = ablation_summary[cond_name][
        ablation_summary[cond_name]["detector"] == "Z-Score"]["auroc"].values
    print("  Z-Score AUROC: " +
          "  ".join(f"{t}={v:.3f}" for t, v in zip(TASKS, zs)) +
          f"  Mean={np.nanmean(zs):.3f}")

# STEP 6 helpers — LOTO with per-fold PCA projection

def _loto_pca50(df, feat_cols, n_components=50):
    
    records = []
    nc = min(n_components, len(feat_cols))
    for test_task in TASKS:
        tr_tasks = [t for t in TASKS if t != test_task]
        tr = df[df["task"].isin(tr_tasks) & (df["is_anomaly"] == 0)]
        te = df[df["task"] == test_task]
        if len(tr) == 0 or len(te) == 0 or te["is_anomaly"].nunique() < 2:
            continue
        sc  = StandardScaler().fit(tr[feat_cols].values)
        pca = PCA(n_components=nc, random_state=42).fit(sc.transform(tr[feat_cols].values))
        Xtr = pca.transform(sc.transform(tr[feat_cols].values))
        Xte = pca.transform(sc.transform(te[feat_cols].values))
        yte = te["is_anomaly"].values
        for dname, dfn in DETECTORS.items():
            try:
                auroc = roc_auc_score(yte, dfn(Xtr, Xte))
            except Exception:
                auroc = np.nan
            records.append(dict(detector=dname, fold_test_task=test_task, auroc=auroc))
    return pd.DataFrame(records)


def _loto_pca50_per_anomaly(df, feat_cols, n_components=50):
    """Per-anomaly LOTO AUROC with PCA-50 reduction, Z-Score only."""
    records = []
    nc = min(n_components, len(feat_cols))
    for test_task in TASKS:
        tr_tasks = [t for t in TASKS if t != test_task]
        tr  = df[df["task"].isin(tr_tasks) & (df["is_anomaly"] == 0)]
        sc  = StandardScaler().fit(tr[feat_cols].values)
        pca = PCA(n_components=nc, random_state=42).fit(sc.transform(tr[feat_cols].values))
        Xtr = pca.transform(sc.transform(tr[feat_cols].values))
        for anom in ["A2", "A3", "A5"]:
            te_h = df[(df["task"] == test_task) & (df["is_anomaly"] == 0)]
            te_a = df[(df["task"] == test_task) & (df["anomaly"] == anom)]
            if len(te_h) == 0 or len(te_a) == 0:
                continue
            te  = pd.concat([te_h, te_a])
            Xte = pca.transform(sc.transform(te[feat_cols].values))
            yte = te["is_anomaly"].values
            try:
                auroc = roc_auc_score(yte, zscore_det(Xtr, Xte))
            except Exception:
                auroc = np.nan
            records.append(dict(fold_test_task=test_task, anomaly_type=anom, auroc=auroc))
    return pd.DataFrame(records)


# STEP 6 - No-physics baseline from features.csv
print("\n[Step 6] No-physics baseline (features.csv)...")
feat_csv = os.path.join(OUT, "features.csv")
no_phys_loto       = None
no_phys_pa         = None
no_phys_pca50_loto = None
no_phys_pca50_pa   = None

if os.path.exists(feat_csv):
    raw_df = pd.read_csv(feat_csv)
    # features.csv (NB7) has extra non-feature cols beyond standard META_COLS:
    # severity_order, extra_mass_kg, cycle_num, file, n_samples, duration_sec
    # Exclude all non-numeric / bookkeeping cols from the feature set.
    NON_FEAT = {"tag", "task", "anomaly", "severity", "is_anomaly",
                "severity_order", "extra_mass_kg", "cycle_num",
                "file", "n_samples", "duration_sec"}
    raw_feats = [c for c in raw_df.columns if c not in NON_FEAT]

    # is_anomaly stored as string '0'/'1' in features.csv — cast to int
    raw_df["is_anomaly"] = raw_df["is_anomaly"].astype(int)

    # More features favours this baseline for Z-Score (more task signal = more noise
    no_phys_loto = loto_evaluate(raw_df, feat_cols=raw_feats)
    no_phys_pa   = loto_per_anomaly(raw_df, feat_cols=raw_feats)

    zs = no_phys_loto[no_phys_loto["detector"] == "Z-Score"]["auroc"].values
    print("  No-physics (104 feat) Z-Score: " +
          "  ".join(f"{t}={v:.3f}" for t, v in zip(TASKS, zs)) +
          f"  Mean={np.nanmean(zs):.3f}")

    no_phys_pca50_loto = _loto_pca50(raw_df, raw_feats, n_components=50)
    no_phys_pca50_pa   = _loto_pca50_per_anomaly(raw_df, raw_feats, n_components=50)

    zs2 = no_phys_pca50_loto[no_phys_pca50_loto["detector"] == "Z-Score"]["auroc"].values
    print("  No-physics PCA-50 Z-Score: " +
          "  ".join(f"{t}={v:.3f}" for t, v in zip(TASKS, zs2)) +
          f"  Mean={np.nanmean(zs2):.3f}")
else:
    print("  WARNING: features.csv not found - run NB7 first.")

# STEP 7 - Save results
print("\n[Step 7] Saving results tables...")
all_cond_names = list(ABLATION_CONDITIONS.keys())
if no_phys_loto is not None:
    all_cond_names = ["no_physics", "no_physics_pca50"] + all_cond_names


def _get_loto_df(cond):
    if cond == "no_physics":        return no_phys_loto
    if cond == "no_physics_pca50":  return no_phys_pca50_loto
    return ablation_summary.get(cond)


def _get_pa_df(cond):
    if cond == "no_physics":        return no_phys_pa
    if cond == "no_physics_pca50":  return no_phys_pca50_pa
    return ablation_per_anom.get(cond)


summary_rows = []
for cond in all_cond_names:
    df_r = _get_loto_df(cond)
    if df_r is None: continue
    for det in DETECTORS:
        sub    = df_r[df_r["detector"] == det]["auroc"].values
        summary_rows.append(dict(
            condition=cond, detector=det,
            T1_auroc=sub[0] if len(sub) > 0 else np.nan,
            T2_auroc=sub[1] if len(sub) > 1 else np.nan,
            T3_auroc=sub[2] if len(sub) > 2 else np.nan,
            mean_auroc=np.nanmean(sub)))
pd.DataFrame(summary_rows).to_csv(
    os.path.join(OUT, "results_ablation.csv"), index=False)

pa_rows = []
for cond in all_cond_names:
    df_pa = _get_pa_df(cond)
    if df_pa is None: continue
    for _, row in df_pa.iterrows():
        pa_rows.append(dict(condition=cond,
                            fold_test_task=row["fold_test_task"],
                            anomaly_type=row["anomaly_type"],
                            auroc=row["auroc"]))
pd.DataFrame(pa_rows).to_csv(
    os.path.join(OUT, "results_ablation_per_anomaly.csv"), index=False)
pd.DataFrame(rmse_records).to_csv(
    os.path.join(OUT, "results_ablation_psr_rmse.csv"), index=False)
print("  Saved 3 CSV files to Processed_Data/")

# STEP 8 - Figures
print("\n[Step 8] Generating figures...")
conds_present = [c for c in COND_ORDER if c in all_cond_names]


def get_loto(cond):
    return _get_loto_df(cond)


def get_pa(cond):
    return _get_pa_df(cond)


# Figure 1: Ablation summary
fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 70/25.4))

ax = axes[0]
x  = np.arange(len(conds_present)); w = 0.22
fold_colors = [C["blue"], C["orange"], C["green"]]
for fi, fold_task in enumerate(TASKS):
    vals = []
    for cond in conds_present:
        df_r = get_loto(cond)
        if df_r is None: vals.append(np.nan); continue
        v = df_r[(df_r["detector"]=="Z-Score") &
                 (df_r["fold_test_task"]==fold_task)]["auroc"]
        vals.append(v.values[0] if len(v) else np.nan)
    ax.bar(x + (fi-1)*w, vals, w, label="Test "+fold_task,
           color=fold_colors[fi], edgecolor="white", lw=0.4)
ax.axhline(0.5, color="#AAAAAA", lw=0.7, ls="--", label="Chance")
ax.set_xticks(x)
ax.set_xticklabels([CONDITION_LABELS[c] for c in conds_present],
                   rotation=20, ha="right", fontsize=5)
ax.set_ylabel("Cross-task AUROC (Z-Score)")
ax.set_ylim(0, 1.08)
ax.legend(ncol=2, fontsize=4)
ax.set_title("(a) Per-fold AUROC - Z-Score", fontsize=7,
             fontweight="bold", loc="left")
ax.grid(True, axis="y", lw=0.3, alpha=0.4)

ax2 = axes[1]
focus = [c for c in ["no_physics","bias_only","gravity_only","full_PSR"]
         if c in conds_present]
xd = np.arange(len(DETECTORS)); wd = 0.8 / len(focus)
for ci, cond in enumerate(focus):
    df_r = get_loto(cond)
    if df_r is None: continue
    means = [df_r[df_r["detector"]==d]["auroc"].mean()
             for d in DETECTORS]
    ax2.bar(xd + (ci - len(focus)/2 + 0.5)*wd, means, wd,
            label=CONDITION_LABELS[cond], color=COND_COLORS[cond],
            edgecolor="white", lw=0.4)
ax2.axhline(0.5, color="#AAAAAA", lw=0.7, ls="--")
ax2.set_xticks(xd)
ax2.set_xticklabels(list(DETECTORS.keys()), rotation=15, ha="right", fontsize=5)
ax2.set_ylabel("Mean cross-task AUROC")
ax2.set_ylim(0, 1.08)
ax2.legend(fontsize=4)
ax2.set_title("(b) All detectors - key conditions", fontsize=7,
              fontweight="bold", loc="left")
ax2.grid(True, axis="y", lw=0.3, alpha=0.4)

fig.suptitle("NB8v5 - PSR Component Ablation: Cross-Task AUROC by Physics Term",
             fontsize=8, fontweight="bold")
plt.tight_layout()
savefig(fig, "NB8v5_ablation_summary")

# Figure 2: Per-anomaly heatmap
fig2, axes2 = plt.subplots(1, 3, figsize=(180/25.4, 60/25.4))
for ai, anom in enumerate(["A2", "A3", "A5"]):
    ax  = axes2[ai]
    mat = np.full((len(conds_present), 3), np.nan)
    for ci, cond in enumerate(conds_present):
        df_pa = get_pa(cond)
        if df_pa is None: continue
        for ti, task in enumerate(TASKS):
            v = df_pa[(df_pa["anomaly_type"]==anom) &
                      (df_pa["fold_test_task"]==task)]["auroc"]
            if len(v): mat[ci, ti] = v.values[0]
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(TASKS, fontsize=6)
    ax.set_yticks(range(len(conds_present)))
    ax.set_yticklabels([CONDITION_LABELS[c] for c in conds_present], fontsize=5)
    ax.set_title(f"({chr(97+ai)}) " + ["Added mass (A2)","Friction (A3)","TCP offset (A5)"][ai],
                 fontsize=7, fontweight="bold", loc="left")
    for ci in range(len(conds_present)):
        for ti in range(3):
            v = mat[ci, ti]
            if not np.isnan(v):
                ax.text(ti, ci, f"{v:.2f}", ha="center", va="center",
                        fontsize=4.5, fontweight="bold",
                        color="black" if 0.25 < v < 0.85 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8, label="AUROC")
fig2.suptitle("NB8v5 - Per-anomaly AUROC: Ablation x Task (Z-Score)",
              fontsize=8, fontweight="bold")
plt.tight_layout()
savefig(fig2, "NB8v5_ablation_per_anomaly")

# Figure 3: PSR RMSE
rmse_df   = pd.DataFrame(rmse_records)
conds_rmse = [c for c in list(ABLATION_CONDITIONS.keys())
              if c in rmse_df["condition"].unique()]
fig3, ax3 = plt.subplots(figsize=(180/25.4, 60/25.4))
xj = np.arange(6); wr = 0.8 / len(conds_rmse)
for ci, cond in enumerate(conds_rmse):
    vals = [rmse_df[(rmse_df["condition"]==cond) &
                    (rmse_df["joint"]==f"J{j}")]["rmse_A"].values
            for j in range(6)]
    vals = [v[0] if len(v) else np.nan for v in vals]
    ax3.bar(xj + (ci - len(conds_rmse)/2 + 0.5)*wr, vals, wr,
            label=CONDITION_LABELS[cond], color=COND_COLORS[cond],
            edgecolor="white", lw=0.4)
ax3.set_xticks(xj)
ax3.set_xticklabels(["J0 Base","J1 Shoulder","J2 Elbow",
                     "J3 Wrist1","J4 Wrist2","J5 Wrist3"], fontsize=5)
ax3.set_ylabel("Prediction RMSE (A)")
ax3.set_title("(a) PSR prediction RMSE by condition - healthy cycles",
              fontsize=7, fontweight="bold", loc="left")
ax3.legend(ncol=3, fontsize=4)
ax3.grid(True, axis="y", lw=0.3, alpha=0.4)
fig3.suptitle("NB8v5 - PSR Fit Quality by Ablation Condition",
              fontsize=8, fontweight="bold")
plt.tight_layout()
savefig(fig3, "NB8v5_ablation_psr_fit")

# STEP 9 - Print summary table
print("\n" + "="*65)
print("ABLATION RESULTS - Z-Score LOTO AUROC")
print("="*65)
print(f"{'Condition':<16}  {'T1':>6}  {'T2':>6}  {'T3':>6}  {'Mean':>6}")
print("-"*50)
for cond in COND_ORDER:
    if cond not in all_cond_names: continue
    df_r = get_loto(cond)
    if df_r is None: continue
    zdf = df_r[df_r["detector"] == "Z-Score"]
    def fv(task):
        v = zdf[zdf["fold_test_task"]==task]["auroc"].values
        return v[0] if len(v) else np.nan
    t1, t2, t3 = fv("T1"), fv("T2"), fv("T3")
    mark = " <-- FULL MODEL" if cond == "full_PSR" else \
           " <-- NO-PHYSICS BASELINE" if cond == "no_physics" else ""
    print(f"{cond:<16}  {t1:>6.3f}  {t2:>6.3f}  {t3:>6.3f}  "
          f"{np.nanmean([t1,t2,t3]):>6.3f}{mark}")
print("="*65)
print("\nNext: NB8v6 - PSR Robustness to DH Parameter Uncertainty")
