import os, glob, warnings, copy
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

# CONFIGURATION — adjust N_MC and PERT_LEVELS to trade runtime for precision
N_MC        = 5      # Monte Carlo seeds per perturbation level
PERT_LEVELS = [0.00, 0.05, 0.10, 0.20]   # eps values (fractional)
PERT_DECOMP = 0.10   # eps for geometry_only / inertial_only decomposition
SUBSAMPLE   = 8      # timestep subsampling (NB8v5 used 4; 8 halves runtime)
RATE        = 125    # Hz
MIN_SAMP    = 200    # minimum samples to count a cycle

# Expected runtime at SUBSAMPLE=8:
#   N_fits = N_MC * len(PERT_LEVELS) + 2 (decomp) + 1 (nominal already at eps=0)
#   N_fits = 5*4 + 2 + 0 = 22 (eps=0 uses N_MC seeds but they're identical)
#   In practice: (N_MC*(len(PERT_LEVELS)-1) + 2) unique fits ≈ 17 fits
#   At ~34 min per fit (SUBSAMPLE=8) ≈ 9-10 hours

TASKS = ["T1", "T2", "T3"]

# PATHS
ROOT = r"D:\Research\R"
BASE = os.path.join(ROOT, "L_Data")
OUT  = os.path.join(ROOT, "P_Data")
SUPP = os.path.join(ROOT, "M_Data", "Figures", "Supplementary")
for d in [OUT, SUPP]:
    os.makedirs(d, exist_ok=True)

# NOMINAL UR5 CB3 PARAMETERS
NOM_DH_A     = np.array([0.0, -0.42500, -0.39225,  0.0,     0.0,     0.0    ])
NOM_DH_D     = np.array([0.089159, 0.0,  0.0,      0.10915, 0.09465, 0.0823 ])
NOM_DH_ALPHA = np.array([np.pi/2,  0.0,  0.0,      np.pi/2, -np.pi/2, 0.0   ])
NOM_MASS     = np.array([3.7000, 8.3930, 2.2750, 1.2190, 1.2190, 0.1879])
NOM_COM      = np.array([
    [ 0.0,      -0.02561,  0.00193],
    [ 0.21250,   0.0,      0.11336],
    [ 0.11993,   0.0,      0.02650],
    [ 0.0,      -0.00180,  0.01634],
    [ 0.0,       0.00180,  0.01634],
    [ 0.0,       0.0,     -0.00116],
])
GRAVITY      = np.array([0, 0, -9.81])
TASK_PAYLOAD = {"T1": 0.0, "T2": 1.0, "T3": 3.0}
PAYLOAD_COM  = np.array([0, 0, 0.05])


def sample_perturbed_params(eps, rng, perturb_geometry=True, perturb_inertial=True):
    """
    Sample one set of perturbed parameters.
    Each parameter drawn independently: param * U(1-eps, 1+eps).
    alpha not perturbed (discrete joint axis orientations are exact by design).
    """
    def perturb(arr):
        factors = rng.uniform(1.0 - eps, 1.0 + eps, size=arr.shape)
        return arr * factors

    dh_a  = perturb(NOM_DH_A)   if perturb_geometry else NOM_DH_A.copy()
    dh_d  = perturb(NOM_DH_D)   if perturb_geometry else NOM_DH_D.copy()
    mass  = perturb(NOM_MASS)   if perturb_inertial else NOM_MASS.copy()
    com   = perturb(NOM_COM)    if perturb_inertial else NOM_COM.copy()

    # alpha is exact (joint axis directions are not a calibration quantity)
    alpha = NOM_DH_ALPHA.copy()
    return dh_a, dh_d, alpha, mass, com


# PARAMETERISED GRAVITY TORQUE
def dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,    sa,    ca,    d   ],
        [0,    0,     0,     1   ]
    ])


def gravity_torque_param(q, dh_a, dh_d, dh_alpha, mass, com, payload_mass=0.0):
    """
    Gravity torques via numerical Jacobian.
    All parameters passed explicitly — no global state.
    Same algorithm as NB8v4/v5 gravity_torque().
    """
    T = [np.eye(4)]
    for i in range(6):
        T.append(T[-1] @ dh_transform(dh_a[i], dh_d[i], dh_alpha[i], q[i]))

    com_world = []
    for i in range(6):
        c_local = np.array([*com[i], 1.0])
        com_world.append((T[i+1] @ c_local)[:3])
    if payload_mass > 0:
        com_world.append((T[6] @ np.array([*PAYLOAD_COM, 1.0]))[:3])

    all_masses = list(mass)
    if payload_mass > 0:
        all_masses.append(payload_mass)

    tau_g = np.zeros(6)
    dq    = 1e-6
    for i in range(6):
        qp = q.copy(); qp[i] += dq
        Tp = [np.eye(4)]
        for j in range(6):
            Tp.append(Tp[-1] @ dh_transform(dh_a[j], dh_d[j], dh_alpha[j], qp[j]))
        for j in range(len(all_masses)):
            if j < 6:
                cp = (Tp[j+1] @ np.array([*com[j], 1.0]))[:3]
            else:
                cp = (Tp[6] @ np.array([*PAYLOAD_COM, 1.0]))[:3]
            dp_dq    = (cp - com_world[j]) / dq
            tau_g[i] += all_masses[j] * GRAVITY @ dp_dq
    return tau_g


# FILE REGISTRY
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
    "T2_A2_1.5kg":   ("T2_Assembly/A2",         "UR5_T2_A2_1.5kg_gripper_40cyc_*.h5", "T2", "A2",      "1.5kg",   1.5),
    "T2_A2_2kg":     ("T2_Assembly/A2",         "UR5_T2_A2_2kg_gripper_40cyc_*.h5",   "T2", "A2",      "2kg",     2.0),
    "T2_A2_3kg":     ("T2_Assembly/A2",         "UR5_T2_A2_3kg_gripper_40cyc_*.h5",   "T2", "A2",      "3kg",     3.0),
    "T2_A3_7duct":   ("T2_Assembly/A3",         "UR5_T2_A3_light_duct_40cyc_*.h5",    "T2", "A3",      "7duct",   0.0),
    "T2_A3_14duct":  ("T2_Assembly/A3",         "UR5_T2_A3_medium_duct_40cyc_*_225508.h5", "T2", "A3", "14duct", 0.0),
    "T2_A5_20mm":    ("T2_Assembly/A5",         "UR5_T2_A5_20mm_40cyc_*.h5",          "T2", "A5",      "20mm",    0.0),
    "T2_A5_50mm":    ("T2_Assembly/A5",         "UR5_T2_A5_50mm_40cyc_*.h5",          "T2", "A5",      "50mm",    0.0),
    "T2_A5_100mm":   ("T2_Assembly/A5",         "UR5_T2_A5_100mm_40cyc_*.h5",         "T2", "A5",      "100mm",   0.0),
    "T3_A2_3.5kg":   ("T3_Palletize/A2",        "UR5_T3_A2_3.5kg_gripper_33cyc_*.h5", "T3", "A2",      "3.5kg",   3.5),
    "T3_A2_4kg":     ("T3_Palletize/A2",        "UR5_T3_A2_4kg_gripper_33cyc_*.h5",   "T3", "A2",      "4kg",     4.0),
    "T3_A2_5kg":     ("T3_Palletize/A2",        "UR5_T3_A2_4.5kg_gripper_33cyc_*.h5", "T3", "A2",      "5kg",     5.0),
    "T3_A3_14duct":  ("T3_Palletize/A3",        "UR5_T3_A3_medium_duct_33cyc_*_205648.h5", "T3", "A3", "14duct", 0.0),
    "T3_A3_7duct":   ("T3_Palletize/A3",        "UR5_T3_A3_light_duct_33cyc_*_222457.h5",  "T3", "A3", "7duct",  0.0),
    "T3_A5_20mm":    ("T3_Palletize/A5",        "UR5_T3_A5_20mm_33cyc_*_172334.h5",   "T3", "A5",      "20mm",    0.0),
    "T3_A5_50mm":    ("T3_Palletize/A5",        "UR5_T3_A5_50mm_33cyc_*_164447.h5",   "T3", "A5",      "50mm",    0.0),
    "T3_A5_100mm":   ("T3_Palletize/A5",        "UR5_T3_A5_100mm_33cyc_*_160716.h5",  "T3", "A5",      "100mm",   0.0),
}

# STEP 1 — Load data
print("=" * 65)
print("NB8v6 - PSR Robustness to Model Parameter Uncertainty")
print("=" * 65)
print(f"\nConfiguration: N_MC={N_MC}, PERT_LEVELS={PERT_LEVELS}, SUBSAMPLE={SUBSAMPLE}")
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
    print(f"  OK  {key}: {n_cyc} cycles")

healthy_cycles = [c for c in all_cycles if c["is_anomaly"] == 0]
print(f"\n  Total: {len(all_cycles)} cycles  |  Healthy: {len(healthy_cycles)}")


# STEP 2 — PSR fitting with explicit parameters
def fit_psr_param(healthy_cycles, dh_a, dh_d, dh_alpha, mass, com):
    """
    Fit full PSR (all 5 regressors) using supplied physics parameters.
    Returns weight dict {joint: array(5)}.
    """
    train_Phi = {j: [] for j in range(6)}
    train_I   = {j: [] for j in range(6)}

    for ci, cyc in enumerate(healthy_cycles):
        payload = TASK_PAYLOAD[cyc["task"]]
        q_arr   = cyc["q"]
        qd_arr  = cyc["qd"]
        cur     = cyc["current"]
        N       = len(q_arr)

        for t in range(0, N, SUBSAMPLE):
            tau_g = gravity_torque_param(q_arr[t], dh_a, dh_d, dh_alpha,
                                         mass, com, payload_mass=payload)
            for j in range(6):
                qdd_j = ((qd_arr[t+1, j] - qd_arr[t-1, j]) * RATE / 2.0
                         if 0 < t < N-1 else 0.0)
                phi = np.array([tau_g[j], qd_arr[t,j],
                                np.sign(qd_arr[t,j]), qdd_j, 1.0])
                train_Phi[j].append(phi)
                train_I[j].append(cur[t, j])

        if (ci + 1) % 100 == 0:
            print(f"    {ci+1}/{len(healthy_cycles)} healthy cycles fitted")

    psr_w = {}
    for j in range(6):
        w, _, _, _ = np.linalg.lstsq(
            np.array(train_Phi[j]), np.array(train_I[j]), rcond=None)
        psr_w[j] = w
    return psr_w


# STEP 3 — Residual computation and feature extraction
FEAT_COLS = (
    [f"J{j}_{s}" for j in range(6)
     for s in ["resid_mean", "resid_std", "resid_rms", "resid_max",
               "resid_skew", "resid_kurtosis",
               "grav_resid_std", "grav_resid_rms"]]
    + ["total_resid_rms", "J1J2_resid_corr"]
)
META_COLS = ["task", "anomaly", "severity", "is_anomaly", "tag"]


def compute_residuals_param(cyc, psr_w, dh_a, dh_d, dh_alpha, mass, com):
    """
    Full residual and gravity residual for one cycle.
    Uses perturbed parameters for gravity torque computation.
    Returns (res, grav_res), each (N_sub, 6).
    """
    payload  = TASK_PAYLOAD[cyc["task"]]
    q_arr    = cyc["q"]
    qd_arr   = cyc["qd"]
    cur      = cyc["current"]
    N        = len(q_arr)
    indices  = list(range(0, N, SUBSAMPLE))
    res      = np.zeros((len(indices), 6))
    grav_res = np.zeros((len(indices), 6))

    for ti, t in enumerate(indices):
        tau_g = gravity_torque_param(q_arr[t], dh_a, dh_d, dh_alpha,
                                     mass, com, payload_mass=payload)
        for j in range(6):
            qdd_j = ((qd_arr[t+1,j] - qd_arr[t-1,j]) * RATE / 2.0
                     if 0 < t < N-1 else 0.0)
            phi          = np.array([tau_g[j], qd_arr[t,j],
                                     np.sign(qd_arr[t,j]), qdd_j, 1.0])
            w            = psr_w[j]
            res[ti, j]   = cur[t, j] - phi @ w
            grav_res[ti, j] = cur[t, j] - (w[0]*tau_g[j] + w[4])
    return res, grav_res


def extract_features(res, grav_res):
    """50 features matching features_residual.csv from NB8v4 exactly."""
    import scipy.stats as sst
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
        feats[f"J{j}_grav_resid_std"] = float(gr.std())
        feats[f"J{j}_grav_resid_rms"] = float(np.sqrt(np.mean(gr**2)))
    feats["total_resid_rms"] = float(np.sqrt(np.mean(res**2)))
    feats["J1J2_resid_corr"] = float(
        np.corrcoef(res[:,1], res[:,2])[0,1] if len(res) > 2 else 0.0)
    return feats


def build_feature_df_param(all_cycles, psr_w, dh_a, dh_d, dh_alpha, mass, com):
    rows = []
    for cyc in all_cycles:
        res, grav_res = compute_residuals_param(
            cyc, psr_w, dh_a, dh_d, dh_alpha, mass, com)
        feats = extract_features(res, grav_res)
        feats.update(task=cyc["task"], anomaly=cyc["anomaly"],
                     severity=cyc["severity"], is_anomaly=cyc["is_anomaly"],
                     tag=cyc["tag"])
        rows.append(feats)
    return pd.DataFrame(rows)


# STEP 4 — Anomaly detectors and LOTO evaluation
def zscore_det(X_tr, X_te):
    mu = X_tr.mean(0); sig = X_tr.std(0) + 1e-8
    return np.abs((X_te - mu) / sig).mean(1)


def ocsvm_det(X_tr, X_te):
    sc  = StandardScaler().fit(X_tr)
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


def loto_evaluate(df):
    records = []
    for test_task in TASKS:
        tr_tasks = [t for t in TASKS if t != test_task]
        tr  = df[df["task"].isin(tr_tasks) & (df["is_anomaly"] == 0)]
        te  = df[df["task"] == test_task]
        if len(tr) == 0 or len(te) == 0 or te["is_anomaly"].nunique() < 2:
            continue
        Xtr = tr[FEAT_COLS].values
        Xte = te[FEAT_COLS].values
        yte = te["is_anomaly"].values
        for dname, dfn in DETECTORS.items():
            try:
                auroc = roc_auc_score(yte, dfn(Xtr, Xte))
            except Exception:
                auroc = np.nan
            records.append(dict(detector=dname, fold_test_task=test_task, auroc=auroc))
    return pd.DataFrame(records)


def loto_per_anomaly_zscore(df):
    records = []
    for test_task in TASKS:
        tr_tasks = [t for t in TASKS if t != test_task]
        tr  = df[df["task"].isin(tr_tasks) & (df["is_anomaly"] == 0)]
        Xtr = tr[FEAT_COLS].values
        for anom in ["A2", "A3", "A5"]:
            te_h = df[(df["task"] == test_task) & (df["is_anomaly"] == 0)]
            te_a = df[(df["task"] == test_task) & (df["anomaly"] == anom)]
            if len(te_h) == 0 or len(te_a) == 0:
                continue
            te  = pd.concat([te_h, te_a])
            yte = te["is_anomaly"].values
            try:
                auroc = roc_auc_score(yte, zscore_det(Xtr, te[FEAT_COLS].values))
            except Exception:
                auroc = np.nan
            records.append(dict(fold_test_task=test_task, anomaly_type=anom, auroc=auroc))
    return pd.DataFrame(records)


# STEP 5 — Monte Carlo perturbation experiment
print("\n[Step 5] Running Monte Carlo perturbation experiment...")
detail_records = []   # per-seed, per-fold, per-detector
pa_records     = []   # per-seed, per-fold, per-anomaly (Z-Score only)

n_total_fits = N_MC * len(PERT_LEVELS) + 2  # +2 for decomp conditions
fit_count = 0

for eps in PERT_LEVELS:
    print(f"\n  Perturbation level: {eps*100:.0f}%  ({N_MC} seeds)")
    seed_aurocs = []

    n_seeds = 1 if eps == 0.0 else N_MC  # nominal: 1 seed (deterministic)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 1000)
        dh_a, dh_d, dh_alpha, mass, com = sample_perturbed_params(
            eps, rng, perturb_geometry=True, perturb_inertial=True)

        fit_count += 1
        print(f"    Seed {seed+1}/{n_seeds}  (fit {fit_count}/{n_total_fits})")

        psr_w = fit_psr_param(healthy_cycles, dh_a, dh_d, dh_alpha, mass, com)

        # RMSE on healthy (for reporting)
        all_res = np.vstack([
            compute_residuals_param(c, psr_w, dh_a, dh_d, dh_alpha, mass, com)[0]
            for c in healthy_cycles])
        rmse = np.sqrt(np.mean(all_res**2, axis=0))
        print("    RMSE: " + "  ".join(f"J{j}={v:.3f}" for j, v in enumerate(rmse)))

        print(f"    Extracting features for {len(all_cycles)} cycles...")
        df = build_feature_df_param(all_cycles, psr_w, dh_a, dh_d, dh_alpha, mass, com)

        loto_df = loto_evaluate(df)
        pa_df   = loto_per_anomaly_zscore(df)

        zs = loto_df[loto_df["detector"] == "Z-Score"]["auroc"].values
        print("    Z-Score AUROC: " +
              "  ".join(f"{t}={v:.3f}" for t, v in zip(TASKS, zs)) +
              f"  Mean={np.nanmean(zs):.3f}")

        for _, row in loto_df.iterrows():
            detail_records.append(dict(
                eps=eps, seed=seed,
                detector=row["detector"],
                fold_test_task=row["fold_test_task"],
                auroc=row["auroc"],
                condition="all_params"))

        for _, row in pa_df.iterrows():
            pa_records.append(dict(
                eps=eps, seed=seed,
                fold_test_task=row["fold_test_task"],
                anomaly_type=row["anomaly_type"],
                auroc=row["auroc"],
                condition="all_params"))

# STEP 6 — Decomposition experiment at eps=10%
print(f"\n[Step 6] Decomposition at eps={PERT_DECOMP*100:.0f}%: geometry vs inertial...")
N_DECOMP = min(3, N_MC)   # 3 seeds sufficient for decomposition

for decomp_label, geom, inert in [
        ("geometry_only", True,  False),
        ("inertial_only", False, True)]:
    print(f"\n  Condition: {decomp_label}")
    for seed in range(N_DECOMP):
        rng = np.random.default_rng(seed + 2000)
        dh_a, dh_d, dh_alpha, mass, com = sample_perturbed_params(
            PERT_DECOMP, rng,
            perturb_geometry=geom, perturb_inertial=inert)

        fit_count += 1
        print(f"    Seed {seed+1}/{N_DECOMP}  (fit {fit_count}/{n_total_fits})")

        psr_w  = fit_psr_param(healthy_cycles, dh_a, dh_d, dh_alpha, mass, com)
        df     = build_feature_df_param(all_cycles, psr_w, dh_a, dh_d, dh_alpha, mass, com)
        loto_df = loto_evaluate(df)

        zs = loto_df[loto_df["detector"] == "Z-Score"]["auroc"].values
        print("    Z-Score AUROC: " +
              "  ".join(f"{t}={v:.3f}" for t, v in zip(TASKS, zs)) +
              f"  Mean={np.nanmean(zs):.3f}")

        for _, row in loto_df.iterrows():
            detail_records.append(dict(
                eps=PERT_DECOMP, seed=seed,
                detector=row["detector"],
                fold_test_task=row["fold_test_task"],
                auroc=row["auroc"],
                condition=decomp_label))

# STEP 7 — Aggregate and save results
print("\n[Step 7] Saving results...")
detail_df = pd.DataFrame(detail_records)
detail_df.to_csv(os.path.join(OUT, "results_robustness_detail.csv"), index=False)

# Summary: mean ± std AUROC per (eps, condition, detector)
summary_rows = []
for (eps, cond, det), grp in detail_df.groupby(["eps", "condition", "detector"]):
    per_fold = grp.groupby("fold_test_task")["auroc"].mean()
    mean_auroc = per_fold.mean()
    std_across_seeds = grp.groupby("seed")["auroc"].mean().std()
    summary_rows.append(dict(
        eps_pct=eps*100, condition=cond, detector=det,
        T1=per_fold.get("T1", np.nan),
        T2=per_fold.get("T2", np.nan),
        T3=per_fold.get("T3", np.nan),
        mean_auroc=mean_auroc,
        std_auroc=std_across_seeds))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT, "results_robustness.csv"), index=False)
print("  Saved: results_robustness.csv")
print("  Saved: results_robustness_detail.csv")

# STEP 8 — Figures
print("\n[Step 8] Generating figures...")
eps_pcts = [e*100 for e in PERT_LEVELS]

# --- Figure 1: Main robustness — AUROC vs perturbation level ---
fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 70/25.4))

# Panel a: Z-Score per fold + mean
ax = axes[0]
fold_colors = {"T1": C["blue"], "T2": C["orange"], "T3": C["green"]}
main_df = detail_df[detail_df["condition"] == "all_params"]
main_zs = main_df[main_df["detector"] == "Z-Score"]

for task in TASKS:
    fold_vals, fold_stds = [], []
    for eps in PERT_LEVELS:
        grp = main_zs[(main_zs["eps"] == eps) &
                      (main_zs["fold_test_task"] == task)]["auroc"]
        fold_vals.append(grp.mean())
        fold_stds.append(grp.std() if len(grp) > 1 else 0)
    ax.errorbar(eps_pcts, fold_vals, yerr=fold_stds,
                label=f"Test {task}", color=fold_colors[task],
                marker="o", markersize=3, capsize=2, linewidth=0.8)

mean_vals, mean_stds = [], []
for eps in PERT_LEVELS:
    grp = main_zs[main_zs["eps"] == eps].groupby("seed")["auroc"].mean()
    mean_vals.append(grp.mean())
    mean_stds.append(grp.std() if len(grp) > 1 else 0)
ax.errorbar(eps_pcts, mean_vals, yerr=mean_stds,
            label="Mean LOTO", color="black", marker="s", markersize=4,
            capsize=2, linewidth=1.0, linestyle="--")

ax.axhline(0.5, color="#AAAAAA", lw=0.7, ls=":", label="Chance (0.5)")
ax.axvline(10, color="#DDAAAA", lw=0.5, ls="--", alpha=0.6,
           label="±10% (paper threshold)")
ax.set_xlabel("Parameter perturbation (±%)")
ax.set_ylabel("Cross-task AUROC (Z-Score)")
ax.set_ylim(0, 1.08)
ax.set_xticks(eps_pcts)
ax.set_xticklabels([f"±{e:.0f}%" for e in eps_pcts])
ax.legend(ncol=2, fontsize=4)
ax.set_title("(a) AUROC vs. model parameter error — Z-Score detector",
             fontsize=7, fontweight="bold", loc="left")
ax.grid(True, axis="y", lw=0.3, alpha=0.4)

# Panel b: all detectors at mean AUROC
ax2 = axes[1]
det_colors = {"Z-Score": C["blue"], "OC-SVM": C["green"],
              "Iso-Forest": C["orange"], "PCA-Recon": C["red"]}
for det in DETECTORS:
    vals, stds = [], []
    for eps in PERT_LEVELS:
        grp = main_df[(main_df["detector"] == det) &
                      (main_df["eps"] == eps)].groupby("seed")["auroc"].mean()
        vals.append(grp.mean())
        stds.append(grp.std() if len(grp) > 1 else 0)
    ax2.errorbar(eps_pcts, vals, yerr=stds,
                 label=det, color=det_colors[det],
                 marker="o", markersize=3, capsize=2, linewidth=0.8)

ax2.axhline(0.5, color="#AAAAAA", lw=0.7, ls=":")
ax2.axvline(10, color="#DDAAAA", lw=0.5, ls="--", alpha=0.6)
ax2.set_xlabel("Parameter perturbation (±%)")
ax2.set_ylabel("Mean cross-task AUROC")
ax2.set_ylim(0, 1.08)
ax2.set_xticks(eps_pcts)
ax2.set_xticklabels([f"±{e:.0f}%" for e in eps_pcts])
ax2.legend(fontsize=4)
ax2.set_title("(b) All detectors — mean LOTO AUROC",
              fontsize=7, fontweight="bold", loc="left")
ax2.grid(True, axis="y", lw=0.3, alpha=0.4)

fig.suptitle(
    "NB8v6 — PSR Robustness: AUROC Under DH Parameter and Mass Perturbation",
    fontsize=8, fontweight="bold")
plt.tight_layout()
savefig(fig, "NB8v6_robustness_main")

# --- Figure 2: Geometry vs Inertial decomposition ---
fig2, ax3 = plt.subplots(figsize=(90/25.4, 65/25.4))
decomp_labels = {
    "all_params":     f"All params (±{PERT_DECOMP*100:.0f}%)",
    "geometry_only":  f"Geometry only (±{PERT_DECOMP*100:.0f}%)",
    "inertial_only":  f"Inertial only (±{PERT_DECOMP*100:.0f}%)",
}
decomp_colors = {
    "all_params":    C["red"],
    "geometry_only": C["sky"],
    "inertial_only": C["orange"],
}
decomp_df = detail_df[
    (detail_df["detector"] == "Z-Score") &
    (detail_df["eps"] == PERT_DECOMP)]

x = np.arange(3)
w = 0.25
for ci, (cond, clabel) in enumerate(decomp_labels.items()):
    sub = decomp_df[decomp_df["condition"] == cond]
    vals = [sub[sub["fold_test_task"] == t]["auroc"].mean() for t in TASKS]
    stds = [sub[sub["fold_test_task"] == t]["auroc"].std() for t in TASKS]
    stds = [s if not np.isnan(s) else 0 for s in stds]
    ax3.bar(x + (ci-1)*w, vals, w, yerr=stds, capsize=2,
            label=clabel, color=decomp_colors[cond],
            edgecolor="white", linewidth=0.4, error_kw={"linewidth": 0.6})

ax3.set_xticks(x)
ax3.set_xticklabels(TASKS, fontsize=6)
ax3.set_ylabel("AUROC (Z-Score)")
ax3.set_ylim(0, 1.08)
ax3.legend(fontsize=4)
ax3.set_title(f"(a) Geometry vs. inertial sensitivity — ±{PERT_DECOMP*100:.0f}%",
              fontsize=7, fontweight="bold", loc="left")
ax3.grid(True, axis="y", lw=0.3, alpha=0.4)

fig2.suptitle("NB8v6 — Parameter Sensitivity Decomposition",
              fontsize=8, fontweight="bold")
plt.tight_layout()
savefig(fig2, "NB8v6_robustness_decomp")

# --- Figure 3: Per-anomaly robustness ---
pa_df_all = pd.DataFrame(pa_records)
fig3, axes3 = plt.subplots(1, 3, figsize=(180/25.4, 60/25.4))
anom_nice = {"A2": "Added mass (A2)", "A3": "Friction (A3)", "A5": "TCP offset (A5)"}

for ai, anom in enumerate(["A2", "A3", "A5"]):
    ax = axes3[ai]
    sub = pa_df_all[pa_df_all["anomaly_type"] == anom]
    for task in TASKS:
        vals, stds = [], []
        for eps in PERT_LEVELS:
            grp = sub[(sub["eps"] == eps) &
                      (sub["fold_test_task"] == task)]["auroc"]
            vals.append(grp.mean())
            stds.append(grp.std() if len(grp) > 1 else 0)
        ax.errorbar(eps_pcts, vals, yerr=stds,
                    label=f"Test {task}", color=fold_colors[task],
                    marker="o", markersize=3, capsize=2, linewidth=0.8)
    ax.axhline(0.5, color="#AAAAAA", lw=0.7, ls=":")
    ax.axvline(10, color="#DDAAAA", lw=0.5, ls="--", alpha=0.6)
    ax.set_xticks(eps_pcts)
    ax.set_xticklabels([f"±{e:.0f}%" for e in eps_pcts])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUROC (Z-Score)" if ai == 0 else "")
    ax.set_xlabel("Perturbation (±%)")
    ax.set_title(f"({chr(97+ai)}) {anom_nice[anom]}",
                 fontsize=7, fontweight="bold", loc="left")
    ax.legend(fontsize=4)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)

fig3.suptitle("NB8v6 — Per-Anomaly AUROC Under Parameter Perturbation",
              fontsize=8, fontweight="bold")
plt.tight_layout()
savefig(fig3, "NB8v6_robustness_per_anomaly")

# STEP 9 — Summary table
print("\n" + "=" * 65)
print("ROBUSTNESS RESULTS — Z-Score LOTO AUROC (mean ± std across seeds)")
print("=" * 65)
print(f"{'Level':<8}  {'T1':>10}  {'T2':>10}  {'T3':>10}  {'Mean':>10}")
print("-" * 58)

main_zs = detail_df[
    (detail_df["condition"] == "all_params") &
    (detail_df["detector"] == "Z-Score")]

for eps in PERT_LEVELS:
    grp = main_zs[main_zs["eps"] == eps]
    t1  = grp[grp["fold_test_task"] == "T1"]["auroc"]
    t2  = grp[grp["fold_test_task"] == "T2"]["auroc"]
    t3  = grp[grp["fold_test_task"] == "T3"]["auroc"]
    means = grp.groupby("seed")["auroc"].mean()
    tag = " <-- nominal" if eps == 0.0 else ""
    print(f"±{eps*100:2.0f}%  "
          f"  {t1.mean():5.3f}±{t1.std():4.3f}"
          f"  {t2.mean():5.3f}±{t2.std():4.3f}"
          f"  {t3.mean():5.3f}±{t3.std():4.3f}"
          f"  {means.mean():5.3f}±{means.std():4.3f}{tag}")

print("=" * 65)
print("\nDecomposition at ±10% (Z-Score, mean AUROC):")
for cond in ["all_params", "geometry_only", "inertial_only"]:
    sub = detail_df[(detail_df["condition"] == cond) &
                    (detail_df["eps"] == PERT_DECOMP) &
                    (detail_df["detector"] == "Z-Score")]
    means = sub.groupby("seed")["auroc"].mean()
    print(f"  {cond:<20}: {means.mean():.3f} ± {means.std():.3f}")

print("\nNB8v6 complete. Next: NB9 — Publication Figures + Statistical Tests")
