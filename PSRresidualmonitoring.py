# %% Cell 1 — Inspect HDF5 structure
import os, glob, warnings, h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = r"D:\Research\R"
BASE    = os.path.join(ROOT, "L_Data")
OUT     = os.path.join(ROOT, "P_Data")
FIG_SUP = os.path.join(ROOT, "M_Data", "Figures", "Supplementary")

for d in [OUT, FIG_SUP]:
    os.makedirs(d, exist_ok=True)

# Check what signals are available
sample_file = glob.glob(os.path.join(BASE, "T1_PickPlace", "Healthy", "*.h5"))[0]
print(f"Inspecting: {os.path.basename(sample_file)}\n")
with h5py.File(sample_file, "r") as f:
    print("Available datasets:")
    for key in sorted(f.keys()):
        ds = f[key]
        print(f"  {key:<30s}  shape={ds.shape}  dtype={ds.dtype}")

REQUIRED = ["actual_q", "actual_qd", "actual_current", "cycle_number"]
with h5py.File(sample_file, "r") as f:
    available = list(f.keys())
    for req in REQUIRED:
        status = "✅" if req in available else "❌ MISSING"
        print(f"\n  {req}: {status}")

    if all(r in available for r in REQUIRED):
        print("\nSample data (first 3 rows):")
        for sig in ["actual_q", "actual_qd", "actual_current"]:
            print(f"  {sig}: {f[sig][:3]}")
        print("\n✅ All required signals present. Physics residual approach is feasible.")
    else:
        missing = [r for r in REQUIRED if r not in available]
        print(f"\n❌ Missing: {missing}")
        print("Cannot compute physics residuals without joint positions/velocities.")
        print("Check if signals are stored under different names.")


# %% Cell 2 — UR5 CB3 kinematics and gravity torque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UR5_DH = {
    "a":     [0, -0.42500, -0.39225, 0, 0, 0],
    "d":     [0.089159, 0, 0, 0.10915, 0.09465, 0.0823],
    "alpha": [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0],
}

UR5_MASS = [3.7000, 8.3930, 2.2750, 1.2190, 1.2190, 0.1879]

UR5_COM = [
    [0.0,     -0.02561, 0.00193],
    [0.21250,  0.0,     0.11336],
    [0.11993,  0.0,     0.02650],
    [0.0,     -0.00180, 0.01634],
    [0.0,      0.00180, 0.01634],
    [0.0,      0.0,    -0.00116],
]

GRAVITY = np.array([0, 0, -9.81])


def dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])


def gravity_torque(q, payload_mass=0.0, payload_com=np.array([0, 0, 0.05])):
    a = UR5_DH["a"]
    d = UR5_DH["d"]
    alpha = UR5_DH["alpha"]

    T = [np.eye(4)]
    for i in range(6):
        Ti = dh_transform(a[i], d[i], alpha[i], q[i])
        T.append(T[-1] @ Ti)

    com_positions = []
    for i in range(6):
        com_local = np.array([*UR5_COM[i], 1.0])
        com_base = T[i+1] @ com_local
        com_positions.append(com_base[:3])

    if payload_mass > 0:
        payload_local = np.array([*payload_com, 1.0])
        payload_base = T[6] @ payload_local
        com_positions.append(payload_base[:3])
        masses = UR5_MASS + [payload_mass]
    else:
        masses = UR5_MASS

    tau_g = np.zeros(6)
    dq = 1e-6
    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += dq

        T_plus = [np.eye(4)]
        for j in range(6):
            Tj = dh_transform(a[j], d[j], alpha[j], q_plus[j])
            T_plus.append(T_plus[-1] @ Tj)

        for j in range(len(masses)):
            if j < 6:
                com_local = np.array([*UR5_COM[j], 1.0])
                com_plus = T_plus[j+1] @ com_local
                com_orig = com_positions[j]
            else:
                com_local = np.array([*payload_com, 1.0])
                com_plus = T_plus[6] @ com_local
                com_orig = com_positions[j]

            dp_dq = (com_plus[:3] - com_orig) / dq
            tau_g[i] += masses[j] * GRAVITY @ dp_dq

    return tau_g


q_home = np.zeros(6)
tau_test = gravity_torque(q_home, payload_mass=0)
print("Gravity torque at home (no payload):")
print(f"  J0={tau_test[0]:.2f}  J1={tau_test[1]:.2f}  J2={tau_test[2]:.2f} Nm")
print(f"  J3={tau_test[3]:.2f}  J4={tau_test[4]:.2f}  J5={tau_test[5]:.2f} Nm")

tau_test_1kg = gravity_torque(q_home, payload_mass=1.0)
print("\nGravity torque at home (1 kg payload):")
print(f"  J1={tau_test_1kg[1]:.2f}  (Δ={tau_test_1kg[1]-tau_test[1]:.2f} Nm from added mass)")


# %% Cell 3 — Load all cycles with positions, velocities, and currents
RATE = 125

REGISTRY = {
    "T1_healthy":    ("T1_PickPlace/Healthy",  "UR5_T1_healthy_180cyc_*.h5",        "T1","healthy","none",   0.0),
    "T2_healthy":    ("T2_Assembly/Healthy",    "UR5_T2_healthy_180cyc_*.h5",        "T2","healthy","none",   0.0),
    "T3_healthy":    ("T3_Palletize/Healthy",   "UR5_T3_healthy_183cyc_*.h5",        "T3","healthy","none",   0.0),
    "T1_A2_0.5kg":   ("T1_PickPlace/A2","UR5_T1_A2_0.5kg_gripper_40cyc_*.h5",  "T1","A2","0.5kg",  0.5),
    "T1_A2_1kg":     ("T1_PickPlace/A2","UR5_T1_A2_1kg_gripper_40cyc_*.h5",    "T1","A2","1kg",    1.0),
    "T1_A2_2kg":     ("T1_PickPlace/A2","UR5_T1_A2_2kg_gripper_40cyc_*.h5",    "T1","A2","2kg",    2.0),
    "T1_A3_10wraps": ("T1_PickPlace/A3","UR5_T1_A3_1band_40cyc_*.h5",          "T1","A3","10wraps",0.0),
    "T1_A3_17wraps": ("T1_PickPlace/A3","UR5_T1_A3_3bands_40cyc_*.h5",         "T1","A3","17wraps",0.0),
    "T1_A5_20mm":    ("T1_PickPlace/A5","UR5_T1_A5_20mm_40cyc_*.h5",           "T1","A5","20mm",   0.0),
    "T1_A5_50mm":    ("T1_PickPlace/A5","UR5_T1_A5_50mm_40cyc_*.h5",           "T1","A5","50mm",   0.0),
    "T1_A5_100mm":   ("T1_PickPlace/A5","UR5_T1_A5_100mm_40cyc_*.h5",          "T1","A5","100mm",  0.0),
    "T2_A2_1.5kg":   ("T2_Assembly/A2","UR5_T2_A2_1.5kg_gripper_40cyc_*.h5",   "T2","A2","0.5kg",  0.5),
    "T2_A2_2kg":     ("T2_Assembly/A2","UR5_T2_A2_2kg_gripper_40cyc_*.h5",     "T2","A2","1kg",    1.0),
    "T2_A2_3kg":     ("T2_Assembly/A2","UR5_T2_A2_3kg_gripper_40cyc_*.h5",     "T2","A2","2kg",    2.0),
    "T2_A3_7duct":   ("T2_Assembly/A3","UR5_T2_A3_light_duct_40cyc_*_214735.h5","T2","A3","7wraps", 0.0),
    "T2_A3_14duct":  ("T2_Assembly/A3","UR5_T2_A3_medium_duct_40cyc_*_225508.h5","T2","A3","14wraps",0.0),
    "T2_A5_20mm":    ("T2_Assembly/A5","UR5_T2_A5_20mm_40cyc_*.h5",            "T2","A5","20mm",   0.0),
    "T2_A5_50mm":    ("T2_Assembly/A5","UR5_T2_A5_50mm_40cyc_*.h5",            "T2","A5","50mm",   0.0),
    "T2_A5_100mm":   ("T2_Assembly/A5","UR5_T2_A5_100mm_40cyc_*.h5",           "T2","A5","100mm",  0.0),
    "T3_A2_3.5kg":   ("T3_Palletize/A2","UR5_T3_A2_3.5kg_gripper_33cyc_*.h5",  "T3","A2","0.5kg",  0.5),
    "T3_A2_4kg":     ("T3_Palletize/A2","UR5_T3_A2_4kg_gripper_33cyc_*.h5",    "T3","A2","1kg",    1.0),
    "T3_A2_5kg":     ("T3_Palletize/A2","UR5_T3_A2_4.5kg_gripper_33cyc_*.h5",  "T3","A2","2kg",    2.0),
    "T3_A3_14duct":  ("T3_Palletize/A3","UR5_T3_A3_medium_duct_33cyc_*.h5",    "T3","A3","14wraps",0.0),
    "T3_A3_7duct":   ("T3_Palletize/A3","UR5_T3_A3_light_duct_33cyc_*.h5",     "T3","A3","7wraps", 0.0),
    "T3_A5_20mm":    ("T3_Palletize/A5","UR5_T3_A5_20mm_33cyc_*.h5",           "T3","A5","20mm",   0.0),
    "T3_A5_50mm":    ("T3_Palletize/A5","UR5_T3_A5_50mm_33cyc_*.h5",           "T3","A5","50mm",   0.0),
    "T3_A5_100mm":   ("T3_Palletize/A5","UR5_T3_A5_100mm_33cyc_*.h5",          "T3","A5","100mm",  0.0),
}

TASK_PAYLOAD = {"T1": 0.0, "T2": 1.0, "T3": 3.0}


def load_cycles_full(filepath):
    with h5py.File(filepath, "r") as f:
        q = f["actual_q"][:]
        qd = f["actual_qd"][:]
        current = f["actual_current"][:]
        cn = f["cycle_number"][:].ravel()

    cycles = []
    labels = np.unique(cn)
    labels = labels[labels > 0]
    for c in labels:
        mask = cn == c
        if mask.sum() >= 50:
            cycles.append({
                "q": q[mask], "qd": qd[mask], "current": current[mask],
                "n_samples": int(mask.sum()),
            })
    return cycles


all_cycles = []
for tag, (subdir, pattern, task, anomaly, severity, extra_mass) in REGISTRY.items():
    hits = glob.glob(os.path.join(BASE, subdir, pattern))
    if not hits:
        print(f"  MISSING: {tag}")
        continue

    raw_cycles = load_cycles_full(hits[0])
    if len(raw_cycles) > 2:
        raw_cycles = raw_cycles[1:-1]

    for i, cyc in enumerate(raw_cycles):
        cyc.update(task=task, anomaly=anomaly, severity=severity,
                   extra_mass=extra_mass, is_anomaly=int(anomaly != "healthy"),
                   tag=tag, cycle_idx=i)
        all_cycles.append(cyc)

meta = pd.DataFrame([{k:v for k,v in c.items() if k not in ["q","qd","current"]}
                      for c in all_cycles])
print(f"Loaded {len(all_cycles)} cycles with full state (q, qd, current)")
print(meta.groupby(["task","anomaly"]).size().unstack(fill_value=0))


# %% Cell 4 — Physics-Structured Regression (PSR)
from scipy.signal import savgol_filter

def build_physics_regressors(q, qd, payload_mass=0.0, rate=RATE):
    N = len(q)
    Phi = np.zeros((N, 6, 5))

    for t in range(N):
        tau_g = gravity_torque(q[t], payload_mass=payload_mass)
        for j in range(6):
            Phi[t, j, 0] = tau_g[j]
            Phi[t, j, 1] = qd[t, j]
            Phi[t, j, 2] = np.sign(qd[t, j])
            Phi[t, j, 4] = 1.0

    for j in range(6):
        if N > 15:
            qd_smooth = savgol_filter(qd[:, j], window_length=min(15, N//2*2-1), polyorder=3)
            qdd = np.gradient(qd_smooth, 1.0/rate)
        else:
            qdd = np.gradient(qd[:, j], 1.0/rate)
        Phi[:, j, 3] = qdd

    return Phi


print("Building physics regressors from healthy data...")
print("  (This takes a few minutes — computing gravity torque per timestep)")

SUBSAMPLE = 4

train_Phi = {j: [] for j in range(6)}
train_I = {j: [] for j in range(6)}

healthy_cycles = [c for c in all_cycles if c["is_anomaly"] == 0]

for ci, c in enumerate(healthy_cycles):
    task = c["task"]
    payload = TASK_PAYLOAD[task]

    q_sub = c["q"][::SUBSAMPLE]
    qd_full = c["qd"]
    cur_sub = c["current"][::SUBSAMPLE]

    N_sub = len(q_sub)
    for t in range(N_sub):
        tau_g = gravity_torque(q_sub[t], payload_mass=payload)
        t_full = t * SUBSAMPLE
        for j in range(6):
            if t_full > 0 and t_full < len(qd_full) - 1:
                qdd_j = (qd_full[t_full+1, j] - qd_full[t_full-1, j]) * RATE / 2
            else:
                qdd_j = 0.0
            phi = np.array([tau_g[j], qd_full[t_full, j],
                            np.sign(qd_full[t_full, j]), qdd_j, 1.0])
            train_Phi[j].append(phi)
            train_I[j].append(cur_sub[t, j] if t < len(cur_sub) else c["current"][t_full, j])

    if (ci + 1) % 50 == 0:
        print(f"  {ci+1}/{len(healthy_cycles)} healthy cycles processed")

psr_weights = {}
for j in range(6):
    Phi_j = np.array(train_Phi[j])
    I_j = np.array(train_I[j])
    w, residuals, rank, sv = np.linalg.lstsq(Phi_j, I_j, rcond=None)
    psr_weights[j] = w

    I_pred = Phi_j @ w
    rmse = np.sqrt(np.mean((I_j - I_pred)**2))
    print(f"  J{j}: RMSE={rmse:.4f}A  weights=[gravity={w[0]:.4f}, "
          f"viscous={w[1]:.4f}, coulomb={w[2]:.4f}, inertia={w[3]:.6f}, bias={w[4]:.4f}]")

print("\nPSR model fitted on all healthy data.")
print("The gravity weight tells us the current-to-torque conversion factor.")
print("The Coulomb/viscous weights capture baseline friction.")


# %% Cell 5 — Compute residuals for ALL cycles
print("Computing residuals for all cycles...")

def compute_residual_features(cycle, psr_weights, payload_mass, subsample=4):
    q = cycle["q"]
    qd = cycle["qd"]
    current = cycle["current"]
    N = len(q)

    residuals = np.zeros((N // subsample, 6))
    gravity_resid = np.zeros((N // subsample, 6))
    friction_resid = np.zeros((N // subsample, 6))

    for ti, t in enumerate(range(0, N - subsample + 1, subsample)):
        tau_g = gravity_torque(q[t], payload_mass=payload_mass)

        for j in range(6):
            if t > 0 and t < N - 1:
                qdd_j = (qd[t+1, j] - qd[t-1, j]) * RATE / 2
            else:
                qdd_j = 0.0

            phi = np.array([tau_g[j], qd[t, j], np.sign(qd[t, j]), qdd_j, 1.0])
            w = psr_weights[j]
            i_pred = phi @ w
            i_actual = current[t, j]

            residuals[ti, j] = i_actual - i_pred
            gravity_resid[ti, j] = i_actual - (w[0]*tau_g[j] + w[4])
            friction_resid[ti, j] = i_actual - i_pred + (w[1]*qd[t,j] + w[2]*np.sign(qd[t,j]))

    feats = {}
    for j in range(6):
        jn = f"J{j}"
        r = residuals[:, j]
        feats[f"{jn}_resid_mean"] = np.mean(r)
        feats[f"{jn}_resid_std"] = np.std(r)
        feats[f"{jn}_resid_rms"] = np.sqrt(np.mean(r**2))
        feats[f"{jn}_resid_max"] = np.max(np.abs(r))
        feats[f"{jn}_resid_skew"] = float(pd.Series(r).skew())
        feats[f"{jn}_resid_kurtosis"] = float(pd.Series(r).kurtosis())

        gr = gravity_resid[:, j]
        feats[f"{jn}_grav_resid_std"] = np.std(gr)
        feats[f"{jn}_grav_resid_rms"] = np.sqrt(np.mean(gr**2))

    feats["total_resid_rms"] = np.sqrt(np.mean(residuals**2))
    feats["J1J2_resid_corr"] = np.corrcoef(residuals[:, 1], residuals[:, 2])[0, 1] \
        if len(residuals) > 2 else 0.0

    return feats


all_feats = []
for ci, c in enumerate(all_cycles):
    task = c["task"]
    payload = TASK_PAYLOAD[task]

    feats = compute_residual_features(c, psr_weights, payload)
    feats.update(task=task, anomaly=c["anomaly"], severity=c["severity"],
                 is_anomaly=c["is_anomaly"], tag=c["tag"])
    all_feats.append(feats)

    if (ci + 1) % 100 == 0:
        print(f"  {ci+1}/{len(all_cycles)} cycles processed")

resid_df = pd.DataFrame(all_feats)
RESID_FEATS = [c for c in resid_df.columns if c not in
               ["task","anomaly","severity","is_anomaly","tag"]]

print(f"\nResidual features: {len(RESID_FEATS)} per cycle")
print(f"Cycles: {len(resid_df)}")


# %% Cell 6 — Quantify task invariance: residual features vs raw features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score

raw_df = pd.read_csv(os.path.join(OUT, "features.csv"))
RAW_FEATS = [c for c in raw_df.columns if c not in
             ["tag","task","anomaly","severity","severity_order",
              "extra_mass_kg","is_anomaly","cycle_num","file","n_samples","duration_sec"]]

healthy_mask_res = resid_df["is_anomaly"] == 0
healthy_mask_raw = raw_df["is_anomaly"] == 0

scaler_res = StandardScaler()
X_res_h = scaler_res.fit_transform(resid_df[healthy_mask_res][RESID_FEATS].values)
y_task_h = resid_df[healthy_mask_res]["task"].values

scaler_raw = StandardScaler()
X_raw_h = scaler_raw.fit_transform(raw_df[healthy_mask_raw][RAW_FEATS].values)
y_task_raw = raw_df[healthy_mask_raw]["task"].values

clf = LogisticRegression(max_iter=2000, random_state=42)

acc_raw = cross_val_score(clf, X_raw_h, y_task_raw, cv=5, scoring="accuracy").mean()
acc_res = cross_val_score(clf, X_res_h, y_task_h, cv=5, scoring="accuracy").mean()

le = LabelEncoder()
sil_raw = silhouette_score(X_raw_h, le.fit_transform(y_task_raw))
sil_res = silhouette_score(X_res_h, le.fit_transform(y_task_h))

print("--- TASK INVARIANCE (healthy only) ---\n")
print(f"  {'Feature set':<25s}  {'Task Acc':<12s}  {'Silhouette':<12s}")
print(f"  {'-'*50}")
print(f"  {'Raw features':<25s}  {acc_raw:<12.3f}  {sil_raw:<12.3f}")
print(f"  {'Residual features':<25s}  {acc_res:<12.3f}  {sil_res:<12.3f}")
print(f"  {'Random chance':<25s}  {'0.333':<12s}")
print(f"\n  Improvement: task_acc {acc_raw:.3f} → {acc_res:.3f} "
      f"({(acc_raw-acc_res)/acc_raw*100:.1f}% reduction)")


# %% Cell 7 — Cross-task anomaly detection with residual features
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

print("\n--- CROSS-TASK ANOMALY DETECTION ---\n")

TASKS = ["T1", "T2", "T3"]

def zscore_auroc(X_train, X_test, y_test):
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    scores = np.mean(((X_test - mu) / std)**2, axis=1)
    if len(np.unique(y_test)) < 2:
        return np.nan
    return roc_auc_score(y_test, scores)

print("Feature Set: RAW (104 features)")
for test_task in TASKS:
    train_tasks = [t for t in TASKS if t != test_task]
    mask_tr = (raw_df.task.isin(train_tasks)) & (raw_df.is_anomaly == 0)
    mask_te = raw_df.task == test_task

    X_tr = StandardScaler().fit_transform(raw_df[mask_tr][RAW_FEATS].values)
    X_te_scaler = StandardScaler().fit(raw_df[mask_tr][RAW_FEATS].values)
    X_te = X_te_scaler.transform(raw_df[mask_te][RAW_FEATS].values)
    y_te = raw_df[mask_te]["is_anomaly"].values

    auroc = zscore_auroc(X_tr, X_te, y_te)
    print(f"  Train: {'+'.join(train_tasks)} → Test: {test_task}  AUROC={auroc:.3f}")

raw_aurocs = []
for test_task in TASKS:
    train_tasks = [t for t in TASKS if t != test_task]
    mask_tr = (raw_df.task.isin(train_tasks)) & (raw_df.is_anomaly == 0)
    mask_te = raw_df.task == test_task
    sc = StandardScaler().fit(raw_df[mask_tr][RAW_FEATS].values)
    X_tr = sc.transform(raw_df[mask_tr][RAW_FEATS].values)
    X_te = sc.transform(raw_df[mask_te][RAW_FEATS].values)
    y_te = raw_df[mask_te]["is_anomaly"].values
    raw_aurocs.append(zscore_auroc(X_tr, X_te, y_te))

print(f"\nFeature Set: PHYSICS RESIDUAL ({len(RESID_FEATS)} features)")
resid_aurocs = []
for test_task in TASKS:
    train_tasks = [t for t in TASKS if t != test_task]
    mask_tr = (resid_df.task.isin(train_tasks)) & (resid_df.is_anomaly == 0)
    mask_te = resid_df.task == test_task

    sc = StandardScaler().fit(resid_df[mask_tr][RESID_FEATS].values)
    X_tr = sc.transform(resid_df[mask_tr][RESID_FEATS].values)
    X_te = sc.transform(resid_df[mask_te][RESID_FEATS].values)
    y_te = resid_df[mask_te]["is_anomaly"].values

    auroc = zscore_auroc(X_tr, X_te, y_te)
    resid_aurocs.append(auroc)
    print(f"  Train: {'+'.join(train_tasks)} → Test: {test_task}  AUROC={auroc:.3f}")

print(f"\nAverage cross-task AUROC:")
print(f"  Raw features:      {np.mean(raw_aurocs):.3f}")
print(f"  Residual features: {np.mean(resid_aurocs):.3f}")


# %% Cell 8 — Per-anomaly-type and severity breakdown (cross-task)
print("\n--- CROSS-TASK AUROC BY ANOMALY TYPE AND SEVERITY ---\n")

cross_detail = []

for test_task in TASKS:
    train_tasks = [t for t in TASKS if t != test_task]
    mask_tr = (resid_df.task.isin(train_tasks)) & (resid_df.is_anomaly == 0)
    mask_te = resid_df.task == test_task

    sc = StandardScaler().fit(resid_df[mask_tr][RESID_FEATS].values)
    X_tr = sc.transform(resid_df[mask_tr][RESID_FEATS].values)

    test_data = resid_df[mask_te].copy()
    test_data["score"] = np.mean(((sc.transform(test_data[RESID_FEATS].values)
                                    - X_tr.mean(axis=0)) / (X_tr.std(axis=0)+1e-8))**2, axis=1)

    for anom in ["A2","A3","A5"]:
        sub = test_data[(test_data.anomaly == anom) | (test_data.is_anomaly == 0)]
        if sub.is_anomaly.sum() > 0 and len(sub.is_anomaly.unique()) > 1:
            auroc = roc_auc_score(sub.is_anomaly, sub.score)
            cross_detail.append(dict(test_task=test_task, anomaly=anom, auroc=auroc))
            print(f"  {test_task} {anom}: AUROC={auroc:.3f}")

    for sev in test_data[test_data.is_anomaly == 1]["severity"].unique():
        sub = test_data[(test_data.severity == sev) | (test_data.is_anomaly == 0)]
        if sub.is_anomaly.sum() > 0 and len(sub.is_anomaly.unique()) > 1:
            auroc = roc_auc_score(sub.is_anomaly, sub.score)
            cross_detail.append(dict(test_task=test_task, severity=sev, auroc=auroc))

detail_df = pd.DataFrame(cross_detail)
if "severity" in detail_df.columns and detail_df["severity"].notna().any():
    print("\nPer-severity (averaged across tasks):")
    sev_pivot = detail_df.dropna(subset=["severity"]).groupby("severity")["auroc"].mean().round(3)
    print(sev_pivot.to_string())


# %% Cell 9 — Model comparison: baselines + PI-SBD on residual features
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PISBDResid(nn.Module):
    def __init__(self, d_in, d_lat=12, n_tasks=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 48), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, d_lat))
        self.bias = nn.Embedding(n_tasks, d_lat)
        nn.init.zeros_(self.bias.weight)
        self.dec = nn.Sequential(nn.Linear(d_lat, 24), nn.ReLU(),
                                 nn.Linear(24, 48), nn.ReLU(), nn.Linear(48, d_in))
        n_grav = 12
        self.grav_head = nn.Sequential(nn.Linear(d_lat, 16), nn.ReLU(),
                                       nn.Linear(16, n_grav))
        n_fric = 6
        self.fric_head = nn.Sequential(nn.Linear(d_lat, 8), nn.ReLU(),
                                       nn.Linear(8, n_fric))

    def forward(self, x, tid):
        z = self.enc(x)
        x_rec = self.dec(z + self.bias(tid))
        grav_pred = self.grav_head(z)
        fric_pred = self.fric_head(z)
        return x_rec, z, grav_pred, fric_pred


def train_pisbd(X_train, tasks_train, feat_cols, epochs=200, lr=1e-3,
                lam_g=0.3, lam_f=0.2, lam_b=0.01):
    tmap = {"T1": 0, "T2": 1, "T3": 2}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train).astype(np.float32)
    tids = np.array([tmap[t] for t in tasks_train], dtype=np.int64)

    grav_idx = [i for i, c in enumerate(feat_cols) if "grav_resid" in c]
    fric_idx = [i for i, c in enumerate(feat_cols) if "resid_std" in c and "grav" not in c]

    d_in = Xs.shape[1]
    net = PISBDResid(d_in, d_lat=12, n_tasks=3).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xs), torch.from_numpy(tids)),
                    batch_size=64, shuffle=True)
    net.train()
    for ep in range(epochs):
        for xb, tb in dl:
            xb, tb = xb.to(DEVICE), tb.to(DEVICE)
            xr, z, gp, fp = net(xb, tb)
            loss = nn.MSELoss()(xr, xb)
            if len(grav_idx) > 0:
                loss += lam_g * nn.MSELoss()(gp[:, :len(grav_idx)], xb[:, grav_idx])
            if len(fric_idx) > 0:
                loss += lam_f * nn.MSELoss()(fp[:, :len(fric_idx)], xb[:, fric_idx])
            loss += lam_b * torch.mean(net.bias.weight**2)
            opt.zero_grad(); loss.backward(); opt.step()

    return net, scaler, tmap, grav_idx, fric_idx


def score_pisbd(net, scaler, X_test, tasks_test, tmap, grav_idx, fric_idx,
                lam_g=0.3, lam_f=0.2):
    Xs = scaler.transform(X_test).astype(np.float32)
    tids = np.array([tmap[t] for t in tasks_test], dtype=np.int64)
    net.eval()
    with torch.no_grad():
        xt = torch.from_numpy(Xs).to(DEVICE)
        tt = torch.from_numpy(tids).to(DEVICE)
        xr, z, gp, fp = net(xt, tt)
        rec_err = torch.mean((xt - xr)**2, dim=1).cpu().numpy()
        grav_err = torch.mean((gp[:, :len(grav_idx)] - xt[:, grav_idx])**2,
                              dim=1).cpu().numpy() if grav_idx else 0
        fric_err = torch.mean((fp[:, :len(fric_idx)] - xt[:, fric_idx])**2,
                              dim=1).cpu().numpy() if fric_idx else 0
    return rec_err + lam_g * grav_err + lam_f * fric_err


print("\n--- CROSS-TASK MODEL COMPARISON (residual features) ---\n")

MODEL_RESULTS = []

for test_task in TASKS:
    train_tasks = [t for t in TASKS if t != test_task]
    mask_tr = (resid_df.task.isin(train_tasks)) & (resid_df.is_anomaly == 0)
    mask_te = resid_df.task == test_task

    X_tr_raw = resid_df[mask_tr][RESID_FEATS].values
    X_te_raw = resid_df[mask_te][RESID_FEATS].values
    y_te = resid_df[mask_te]["is_anomaly"].values
    tasks_tr = resid_df[mask_tr]["task"].values
    tasks_te = resid_df[mask_te]["task"].values

    print(f"  Train: {'+'.join(train_tasks)} → Test: {test_task}")

    sc = StandardScaler().fit(X_tr_raw)
    X_tr_s = sc.transform(X_tr_raw)
    X_te_s = sc.transform(X_te_raw)
    scores = np.mean(((X_te_s - X_tr_s.mean(0)) / (X_tr_s.std(0)+1e-8))**2, axis=1)
    auroc_zs = roc_auc_score(y_te, scores)
    MODEL_RESULTS.append(dict(model="Z-Score", test_task=test_task, auroc=auroc_zs))

    clf_if = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    clf_if.fit(X_tr_s)
    scores_if = -clf_if.decision_function(X_te_s)
    auroc_if = roc_auc_score(y_te, scores_if)
    MODEL_RESULTS.append(dict(model="Isolation Forest", test_task=test_task, auroc=auroc_if))

    pca = PCA(n_components=0.95).fit(X_tr_s)
    scores_pca = np.mean((X_te_s - pca.inverse_transform(pca.transform(X_te_s)))**2, axis=1)
    auroc_pca = roc_auc_score(y_te, scores_pca)
    MODEL_RESULTS.append(dict(model="PCA Recon.", test_task=test_task, auroc=auroc_pca))

    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(X_tr_s)
    scores_svm = -ocsvm.decision_function(X_te_s)
    auroc_svm = roc_auc_score(y_te, scores_svm)
    MODEL_RESULTS.append(dict(model="OC-SVM", test_task=test_task, auroc=auroc_svm))

    net, sc_pisbd, tmap, grav_idx, fric_idx = train_pisbd(
        X_tr_raw, tasks_tr, RESID_FEATS, epochs=200)
    scores_pisbd = score_pisbd(net, sc_pisbd, X_te_raw, tasks_te,
                                tmap, grav_idx, fric_idx)
    auroc_pisbd = roc_auc_score(y_te, scores_pisbd)
    MODEL_RESULTS.append(dict(model="PI-SBD", test_task=test_task, auroc=auroc_pisbd))

    print(f"    Z-Score={auroc_zs:.3f}  IF={auroc_if:.3f}  PCA={auroc_pca:.3f}  "
          f"OC-SVM={auroc_svm:.3f}  PI-SBD={auroc_pisbd:.3f}")

model_df = pd.DataFrame(MODEL_RESULTS)
print(f"\nAverage cross-task AUROC:")
print(model_df.groupby("model")["auroc"].mean().sort_values(ascending=False).round(3).to_string())


# %% Cell 10 — Comparison figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x = np.arange(3)
w = 0.35
ax.bar(x - w/2, raw_aurocs, w, label="Raw features", color="#d95f02", alpha=0.8)
ax.bar(x + w/2, resid_aurocs, w, label="Physics residual", color="#1b9e77", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"Test: {t}" for t in TASKS])
ax.set_ylabel("AUROC")
ax.set_title("Cross-Task Transfer: Raw vs Physics Residual")
ax.axhline(0.5, color="grey", ls=":", lw=0.8)
ax.legend()
ax.set_ylim(0, 1.05)

ax = axes[1]
colors_m = {"Z-Score":"#a6cee3", "Isolation Forest":"#b2df8a", "PCA Recon.":"#fb9a99",
            "OC-SVM":"#fdbf6f", "PI-SBD":"#e31a1c"}
model_names = ["Z-Score","Isolation Forest","PCA Recon.","OC-SVM","PI-SBD"]
x = np.arange(3)
w = 0.15
for mi, mn in enumerate(model_names):
    sub = model_df[model_df.model == mn]
    vals = [sub[sub.test_task == t]["auroc"].values[0] for t in TASKS]
    ax.bar(x + (mi - 2)*w, vals, w, label=mn, color=colors_m[mn], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([f"Test: {t}" for t in TASKS])
ax.set_ylabel("AUROC")
ax.set_title("Cross-Task: Model Comparison (Residual Features)")
ax.axhline(0.5, color="grey", ls=":", lw=0.8)
ax.legend(fontsize=7, loc="lower right")
ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(os.path.join(FIG_SUP, "NB8_physics_residual.png"), dpi=1200)
fig.savefig(os.path.join(FIG_SUP, "NB8_physics_residual.pdf"))
plt.show()
print(f"Saved: {FIG_SUP}/NB8_physics_residual.png/.pdf")


# %% Cell 11 — Residual waveform visualisation (for publication)
fig, axes = plt.subplots(3, 4, figsize=(18, 10))

for row, task in enumerate(TASKS):
    h_cyc = [c for c in all_cycles if c["task"]==task and c["anomaly"]=="healthy"][0]
    payload = TASK_PAYLOAD[task]

    for col, (condition, color) in enumerate([
        ("healthy","#1b9e77"), ("A2","#d95f02"), ("A3","#7570b3"), ("A5","#e7298a")]):

        ax = axes[row, col]
        if condition == "healthy":
            cyc = h_cyc
        else:
            candidates = [c for c in all_cycles if c["task"]==task and c["anomaly"]==condition]
            if not candidates:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue
            cyc = sorted(candidates, key=lambda x: x["extra_mass"], reverse=True)[0]

        N = min(cyc["n_samples"], 500)
        resid_j2 = np.zeros(N // SUBSAMPLE)
        for ti, t in enumerate(range(0, N - SUBSAMPLE + 1, SUBSAMPLE)):
            tau_g = gravity_torque(cyc["q"][t], payload_mass=payload)
            qd_t = cyc["qd"][t]
            qdd_2 = (cyc["qd"][min(t+1,N-1), 2] - cyc["qd"][max(t-1,0), 2]) * RATE / 2
            phi = np.array([tau_g[2], qd_t[2], np.sign(qd_t[2]), qdd_2, 1.0])
            i_pred = phi @ psr_weights[2]
            resid_j2[ti] = cyc["current"][t, 2] - i_pred

        t_axis = np.arange(len(resid_j2)) * SUBSAMPLE / RATE
        ax.plot(t_axis, resid_j2, color=color, linewidth=0.5)
        ax.set_title(f"{task} {condition}", fontsize=9)
        ax.set_xlim(0, t_axis[-1] if len(t_axis) > 0 else 1)
        if col == 0: ax.set_ylabel("J2 residual (A)")
        if row == 2: ax.set_xlabel("Time (s)")

plt.suptitle("Physics Residual (J2 Elbow): Healthy vs Anomaly Waveforms", fontsize=13)
plt.tight_layout()
plt.show()
print(f"Saved: {FIG_SUP}/NB8_residual_waveforms.png/.pdf")


# %% Cell 12 — Save results
resid_df.to_csv(os.path.join(OUT, "features_residual.csv"), index=False)
model_df.to_csv(os.path.join(OUT, "results_cross_task_residual.csv"), index=False)
pd.DataFrame(cross_detail).to_csv(os.path.join(OUT, "results_cross_detail_residual.csv"),
                                   index=False)

print(f"\nSaved: {OUT}/features_residual.csv")
print(f"Saved: {OUT}/results_cross_task_residual.csv")
print(f"Saved: {OUT}/results_cross_detail_residual.csv")
print(f"Figures: {FIG_SUP}/NB8_*.png/.pdf")
print(f"\nNB complete.")
