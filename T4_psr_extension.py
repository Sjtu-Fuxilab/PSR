# T4_psr_extension.py
# 4-fold extension of the PSR pipeline. Refits PSR weights on T1+T2+T3 healthy
# and evaluates PSR Z-Score, OC-SVM, IsoForest, and GMM at T4.

# %% Cell 1: Configuration

import os, glob, warnings, pickle
import numpy as np
import pandas as pd
import h5py
import scipy.stats as sst
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = r"D:\Research\R"
BASE = os.path.join(ROOT, "L_Data")
OUT  = os.path.join(ROOT, "P_Data")
os.makedirs(OUT, exist_ok=True)

RATE      = 125
SUBSAMPLE = 4
MIN_SAMP  = 50
N_BOOT    = 10000

TASKS        = ["T1", "T2", "T3", "T4"]
TASK_PAYLOAD = {"T1": 0.0, "T2": 1.0, "T3": 3.0, "T4": 2.0}

REGISTRY = {
    "T1_healthy":    ("T1_PickPlace/Healthy",   "UR5_T1_healthy_180cyc_*.h5",         "T1","healthy","none",  0.0),
    "T1_A2_0.5kg":   ("T1_PickPlace/A2",        "UR5_T1_A2_0.5kg_gripper_40cyc_*.h5", "T1","A2","0.5kg",      0.5),
    "T1_A2_1kg":     ("T1_PickPlace/A2",        "UR5_T1_A2_1kg_gripper_40cyc_*.h5",   "T1","A2","1kg",        1.0),
    "T1_A2_2kg":     ("T1_PickPlace/A2",        "UR5_T1_A2_2kg_gripper_40cyc_*.h5",   "T1","A2","2kg",        2.0),
    "T1_A3_10wraps": ("T1_PickPlace/A3",        "UR5_T1_A3_1band_40cyc_*.h5",         "T1","A3","10wraps",    0.0),
    "T1_A3_17wraps": ("T1_PickPlace/A3",        "UR5_T1_A3_3bands_40cyc_*.h5",        "T1","A3","17wraps",    0.0),
    "T1_A5_20mm":    ("T1_PickPlace/A5",        "UR5_T1_A5_20mm_40cyc_*.h5",          "T1","A5","20mm",       0.0),
    "T1_A5_50mm":    ("T1_PickPlace/A5",        "UR5_T1_A5_50mm_40cyc_*.h5",          "T1","A5","50mm",       0.0),
    "T1_A5_100mm":   ("T1_PickPlace/A5",        "UR5_T1_A5_100mm_40cyc_*.h5",         "T1","A5","100mm",      0.0),
    "T2_healthy":    ("T2_Assembly/Healthy",    "UR5_T2_healthy_180cyc_*.h5",            "T2","healthy","none", 0.0),
    "T2_A2_1.5kg":   ("T2_Assembly/A2",         "UR5_T2_A2_1.5kg_gripper_40cyc_*.h5",    "T2","A2","1.5kg",     0.5),
    "T2_A2_2kg":     ("T2_Assembly/A2",         "UR5_T2_A2_2kg_gripper_40cyc_*.h5",      "T2","A2","2kg",       1.0),
    "T2_A2_3kg":     ("T2_Assembly/A2",         "UR5_T2_A2_3kg_gripper_40cyc_*.h5",      "T2","A2","3kg",       2.0),
    "T2_A3_7duct":   ("T2_Assembly/A3",         "UR5_T2_A3_light_duct_40cyc_*_214735.h5","T2","A3","7duct",     0.0),
    "T2_A3_14duct":  ("T2_Assembly/A3",         "UR5_T2_A3_medium_duct_40cyc_*_225508.h5","T2","A3","14duct",   0.0),
    "T2_A5_20mm":    ("T2_Assembly/A5",         "UR5_T2_A5_20mm_40cyc_*.h5",             "T2","A5","20mm",      0.0),
    "T2_A5_50mm":    ("T2_Assembly/A5",         "UR5_T2_A5_50mm_40cyc_*.h5",             "T2","A5","50mm",      0.0),
    "T2_A5_100mm":   ("T2_Assembly/A5",         "UR5_T2_A5_100mm_40cyc_*.h5",            "T2","A5","100mm",     0.0),
    "T3_healthy":    ("T3_Palletize/Healthy",   "UR5_T3_healthy_183cyc_*.h5",            "T3","healthy","none", 0.0),
    "T3_A2_3.5kg":   ("T3_Palletize/A2",        "UR5_T3_A2_3.5kg_gripper_33cyc_*.h5",    "T3","A2","3.5kg",     0.5),
    "T3_A2_4kg":     ("T3_Palletize/A2",        "UR5_T3_A2_4kg_gripper_33cyc_*.h5",      "T3","A2","4kg",       1.0),
    "T3_A2_5kg":     ("T3_Palletize/A2",        "UR5_T3_A2_4.5kg_gripper_33cyc_*.h5",    "T3","A2","5kg",       2.0),
    "T3_A3_7duct":   ("T3_Palletize/A3",        "UR5_T3_A3_light_duct_33cyc_*_222457.h5","T3","A3","7duct",     0.0),
    "T3_A3_14duct":  ("T3_Palletize/A3",        "UR5_T3_A3_medium_duct_33cyc_*_205648.h5","T3","A3","14duct",   0.0),
    "T3_A5_20mm":    ("T3_Palletize/A5",        "UR5_T3_A5_20mm_33cyc_*_172334.h5",      "T3","A5","20mm",      0.0),
    "T3_A5_50mm":    ("T3_Palletize/A5",        "UR5_T3_A5_50mm_33cyc_*_164447.h5",      "T3","A5","50mm",      0.0),
    "T3_A5_100mm":   ("T3_Palletize/A5",        "UR5_T3_A5_100mm_33cyc_*_160716.h5",     "T3","A5","100mm",     0.0),
    "T4_healthy":    ("T4_BinReorient/healthy", "UR5_T4_healthy_session2_35cyc_*.h5",    "T4","healthy","none", 0.0),
    "T4_A2_0.5kg":   ("T4_BinReorient/anomaly", "UR5_T4_A2_0.5kg_35cyc_*.h5",            "T4","A2","0.5kg",     0.5),
    "T4_A2_1kg":     ("T4_BinReorient/anomaly", "UR5_T4_A2_1kg_35cyc_*.h5",              "T4","A2","1kg",       1.0),
    "T4_A2_2kg":     ("T4_BinReorient/anomaly", "UR5_T4_A2_2kg_35cyc_*.h5",              "T4","A2","2kg",       2.0),
    "T4_A3_7duct":   ("T4_BinReorient/anomaly", "UR5_T4_A3_7wraps_35cyc_*.h5",           "T4","A3","7duct",     0.0),
    "T4_A3_14duct":  ("T4_BinReorient/anomaly", "UR5_T4_A3_14wraps_35cyc_*.h5",          "T4","A3","14duct",    0.0),
    "T4_A5_20mm":    ("T4_BinReorient/anomaly", "UR5_T4_A5_20mm_35cyc_*.h5",             "T4","A5","20mm",      0.0),
    "T4_A5_50mm":    ("T4_BinReorient/anomaly", "UR5_T4_A5_50mm_35cyc_*.h5",             "T4","A5","50mm",      0.0),
    "T4_A5_100mm":   ("T4_BinReorient/anomaly", "UR5_T4_A5_100mm_35cyc_*.h5",            "T4","A5","100mm",     0.0),
}

print(f"Registry: {len(REGISTRY)} entries across {len(TASKS)} tasks.")

# %% Cell 2: UR5 kinematics and gravity torque

UR5_DH = {"a":     [0, -0.42500, -0.39225, 0, 0, 0],
          "d":     [0.089159, 0, 0, 0.10915, 0.09465, 0.0823],
          "alpha": [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]}
UR5_MASS = [3.7000, 8.3930, 2.2750, 1.2190, 1.2190, 0.1879]
UR5_COM = [[0.0,    -0.02561, 0.00193],
           [0.21250, 0.0,     0.11336],
           [0.11993, 0.0,     0.02650],
           [0.0,    -0.00180, 0.01634],
           [0.0,     0.00180, 0.01634],
           [0.0,     0.0,    -0.00116]]
GRAVITY = np.array([0, 0, -9.81])

def dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0,   sa,     ca,    d   ],
                     [0,   0,      0,     1   ]])

def gravity_torque(q, payload_mass=0.0, payload_com=np.array([0, 0, 0.05])):
    a, d, alpha = UR5_DH["a"], UR5_DH["d"], UR5_DH["alpha"]
    T = [np.eye(4)]
    for i in range(6):
        T.append(T[-1] @ dh_transform(a[i], d[i], alpha[i], q[i]))
    com_positions = [(T[i+1] @ np.array([*UR5_COM[i], 1.0]))[:3] for i in range(6)]
    if payload_mass > 0:
        com_positions.append((T[6] @ np.array([*payload_com, 1.0]))[:3])
        masses = UR5_MASS + [payload_mass]
    else:
        masses = UR5_MASS
    tau_g = np.zeros(6)
    dq = 1e-6
    for i in range(6):
        q_plus = q.copy(); q_plus[i] += dq
        T_plus = [np.eye(4)]
        for j in range(6):
            T_plus.append(T_plus[-1] @ dh_transform(a[j], d[j], alpha[j], q_plus[j]))
        for j in range(len(masses)):
            if j < 6:
                com_plus = (T_plus[j+1] @ np.array([*UR5_COM[j], 1.0]))[:3]
            else:
                com_plus = (T_plus[6] @ np.array([*payload_com, 1.0]))[:3]
            dp_dq = (com_plus - com_positions[j]) / dq
            tau_g[i] += masses[j] * GRAVITY @ dp_dq
    return tau_g

# %% Cell 3: Load all cycles

def load_cycles(filepath):
    with h5py.File(filepath, "r") as f:
        q       = f["actual_q"][:]
        qd      = f["actual_qd"][:]
        current = f["actual_current"][:]
        cn      = f["cycle_number"][:].ravel()
    cycles = []
    for c in np.unique(cn[cn > 0]):
        m = cn == c
        if m.sum() >= MIN_SAMP:
            cycles.append({"q": q[m], "qd": qd[m], "current": current[m], "n_samples": int(m.sum())})
    return cycles

print("Loading all cycles...")
all_cycles = []
for tag, (subdir, pattern, task, anomaly, severity, extra_mass) in REGISTRY.items():
    hits = glob.glob(os.path.join(BASE, subdir, pattern))
    if not hits:
        print(f"  MISSING: {tag}")
        continue
    raw = load_cycles(hits[0])
    if len(raw) > 2:
        raw = raw[1:-1]
    for i, cyc in enumerate(raw):
        cyc.update(task=task, anomaly=anomaly, severity=severity,
                   extra_mass=extra_mass, is_anomaly=int(anomaly != "healthy"),
                   tag=tag, cycle_idx=i)
        all_cycles.append(cyc)

meta = pd.DataFrame([{k:v for k,v in c.items() if k not in ["q","qd","current"]} for c in all_cycles])
print(f"Loaded {len(all_cycles)} cycles")
print(meta.groupby(["task","anomaly"]).size().unstack(fill_value=0))

# %% Cell 4: Fit PSR weights on T1+T2+T3 healthy

print("Fitting PSR weights on T1+T2+T3 healthy...")
train_Phi = {j: [] for j in range(6)}
train_I   = {j: [] for j in range(6)}
healthy_train = [c for c in all_cycles if c["is_anomaly"] == 0 and c["task"] in ("T1","T2","T3")]

for ci, c in enumerate(healthy_train):
    payload = TASK_PAYLOAD[c["task"]]
    q_sub   = c["q"][::SUBSAMPLE]
    qd_full = c["qd"]
    cur_sub = c["current"][::SUBSAMPLE]
    for t in range(len(q_sub)):
        tau_g  = gravity_torque(q_sub[t], payload_mass=payload)
        t_full = t * SUBSAMPLE
        for j in range(6):
            if 0 < t_full < len(qd_full) - 1:
                qdd_j = (qd_full[t_full+1, j] - qd_full[t_full-1, j]) * RATE / 2
            else:
                qdd_j = 0.0
            phi = np.array([tau_g[j], qd_full[t_full, j], np.sign(qd_full[t_full, j]), qdd_j, 1.0])
            train_Phi[j].append(phi)
            train_I[j].append(cur_sub[t, j] if t < len(cur_sub) else c["current"][t_full, j])
    if (ci + 1) % 50 == 0:
        print(f"  {ci+1}/{len(healthy_train)} healthy cycles")

psr_weights = {}
for j in range(6):
    Phi_j = np.array(train_Phi[j])
    I_j   = np.array(train_I[j])
    w, *_ = np.linalg.lstsq(Phi_j, I_j, rcond=None)
    psr_weights[j] = w
    rmse = np.sqrt(np.mean((I_j - Phi_j @ w)**2))
    print(f"  J{j}: RMSE={rmse:.4f}A  [g={w[0]:.4f}, v={w[1]:.4f}, c={w[2]:.4f}, i={w[3]:.6f}, b={w[4]:.4f}]")

# %% Cell 5: Compute residual features for all cycles

def residual_features(cycle, weights, payload_mass, subsample=SUBSAMPLE):
    q, qd, current = cycle["q"], cycle["qd"], cycle["current"]
    N = len(q)
    n_sub = N // subsample
    residuals     = np.zeros((n_sub, 6))
    gravity_resid = np.zeros((n_sub, 6))
    for ti, t in enumerate(range(0, N - subsample + 1, subsample)):
        tau_g = gravity_torque(q[t], payload_mass=payload_mass)
        for j in range(6):
            if 0 < t < N - 1:
                qdd_j = (qd[t+1, j] - qd[t-1, j]) * RATE / 2
            else:
                qdd_j = 0.0
            phi    = np.array([tau_g[j], qd[t, j], np.sign(qd[t, j]), qdd_j, 1.0])
            w      = weights[j]
            i_pred = phi @ w
            residuals[ti, j]     = current[t, j] - i_pred
            gravity_resid[ti, j] = current[t, j] - (w[0]*tau_g[j] + w[4])
    feats = {}
    for j in range(6):
        jn = f"J{j}"
        r  = residuals[:, j]
        gr = gravity_resid[:, j]
        feats[f"{jn}_resid_mean"]     = np.mean(r)
        feats[f"{jn}_resid_std"]      = np.std(r)
        feats[f"{jn}_resid_rms"]      = np.sqrt(np.mean(r**2))
        feats[f"{jn}_resid_max"]      = np.max(np.abs(r))
        feats[f"{jn}_resid_skew"]     = float(pd.Series(r).skew())
        feats[f"{jn}_resid_kurtosis"] = float(pd.Series(r).kurtosis())
        feats[f"{jn}_grav_resid_std"] = np.std(gr)
        feats[f"{jn}_grav_resid_rms"] = np.sqrt(np.mean(gr**2))
    feats["total_resid_rms"] = np.sqrt(np.mean(residuals**2))
    feats["J1J2_resid_corr"] = (np.corrcoef(residuals[:, 1], residuals[:, 2])[0, 1]
                                if len(residuals) > 2 else 0.0)
    return feats, residuals

print("Computing residual features for all cycles...")
all_feats = []
for ci, c in enumerate(all_cycles):
    feats, _ = residual_features(c, psr_weights, TASK_PAYLOAD[c["task"]])
    feats.update(task=c["task"], anomaly=c["anomaly"], severity=c["severity"],
                 is_anomaly=c["is_anomaly"], tag=c["tag"])
    all_feats.append(feats)
    if (ci + 1) % 100 == 0:
        print(f"  {ci+1}/{len(all_cycles)} cycles")

resid_df = pd.DataFrame(all_feats)
RESID_FEATS = [c for c in resid_df.columns if c not in ["task","anomaly","severity","is_anomaly","tag"]]

# %% Cell 6: Per-joint R2 and RMSE at T4

t4_healthy = [c for c in all_cycles if c["task"] == "T4" and c["is_anomaly"] == 0]
print(f"T4 healthy cycles: {len(t4_healthy)}")

per_joint_rows = []
joint_R2 = []
for j in range(6):
    y_true_list, y_pred_list = [], []
    for c in t4_healthy:
        q, qd, current = c["q"], c["qd"], c["current"]
        for t in range(0, len(q) - SUBSAMPLE + 1, SUBSAMPLE):
            tau_g = gravity_torque(q[t], payload_mass=TASK_PAYLOAD["T4"])
            if 0 < t < len(q) - 1:
                qdd_j = (qd[t+1, j] - qd[t-1, j]) * RATE / 2
            else:
                qdd_j = 0.0
            phi = np.array([tau_g[j], qd[t, j], np.sign(qd[t, j]), qdd_j, 1.0])
            y_true_list.append(current[t, j])
            y_pred_list.append(phi @ psr_weights[j])
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    joint_R2.append(r2)
    per_joint_rows.append(dict(fold="T4", joint=f"J{j+1}", R2=r2, RMSE_A=rmse))
    print(f"  J{j+1}: R2={r2:+.4f}  RMSE={rmse:.4f}A")

R2_min      = float(min(joint_R2))
R2_mean     = float(np.mean(joint_R2))
worst_joint = f"J{int(np.argmin(joint_R2)) + 1}"
print(f"\nT4 mean R2 = {R2_mean:.3f}, R2_min = {R2_min:+.3f} at {worst_joint}")
pd.DataFrame(per_joint_rows).to_csv(os.path.join(OUT, "T4_per_joint_fit.csv"), index=False, float_format="%.4f")

# %% Cell 7: BCa bootstrap AUROC

def bootstrap_auroc_bca(y_true, y_score, n_boot=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(42)
    n = len(y_true)
    auroc_obs = roc_auc_score(y_true, y_score)
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        boot[b] = roc_auc_score(yt, ys) if (0 < yt.sum() < n) else auroc_obs
    prop = np.clip(np.mean(boot < auroc_obs), 1e-6, 1 - 1e-6)
    z0   = sst.norm.ppf(prop)
    jack = np.zeros(n)
    for i in range(n):
        idx_j = np.concatenate([np.arange(i), np.arange(i+1, n)])
        yt_j, ys_j = y_true[idx_j], y_score[idx_j]
        jack[i] = roc_auc_score(yt_j, ys_j) if (0 < yt_j.sum() < len(yt_j)) else auroc_obs
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    ci = {}
    for label, z_a in [("lo", sst.norm.ppf(0.025)), ("hi", sst.norm.ppf(0.975))]:
        p = sst.norm.cdf(z0 + (z0 + z_a) / (1 - a * (z0 + z_a)))
        ci[label] = float(np.quantile(boot, np.clip(p, 0.001, 0.999)))
    return float(auroc_obs), ci["lo"], ci["hi"]

# %% Cell 8: PSR-family aggregate AUROC at T4

tr_mask = (resid_df.task.isin(["T1","T2","T3"])) & (resid_df.is_anomaly == 0)
te_mask = resid_df.task == "T4"
X_tr = resid_df[tr_mask][RESID_FEATS].values
X_te = resid_df[te_mask][RESID_FEATS].values
y_te = resid_df[te_mask]["is_anomaly"].values

sc     = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr)
X_te_s = sc.transform(X_te)

# PSR Z-Score
mu, std = X_tr_s.mean(0), X_tr_s.std(0); std[std == 0] = 1
s_zs = np.mean(((X_te_s - mu) / std) ** 2, axis=1)
auroc_zs, lo_zs, hi_zs = bootstrap_auroc_bca(y_te, s_zs)
print(f"PSR Z-Score    AUROC = {auroc_zs:.4f}  [{lo_zs:.4f}, {hi_zs:.4f}]")

# PSR OC-SVM
ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(X_tr_s)
s_svm = -ocsvm.decision_function(X_te_s)
auroc_svm, lo_svm, hi_svm = bootstrap_auroc_bca(y_te, s_svm)
print(f"PSR OC-SVM     AUROC = {auroc_svm:.4f}  [{lo_svm:.4f}, {hi_svm:.4f}]")

# PSR IsoForest
ifo  = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1).fit(X_tr_s)
s_if = -ifo.decision_function(X_te_s)
auroc_if, lo_if, hi_if = bootstrap_auroc_bca(y_te, s_if)
print(f"PSR IsoForest  AUROC = {auroc_if:.4f}  [{lo_if:.4f}, {hi_if:.4f}]")

# GMM with BIC over K = 1..8
best_bic, best_K = np.inf, 1
for K in range(1, 9):
    g = GaussianMixture(n_components=K, covariance_type="full", random_state=42, max_iter=200).fit(X_tr_s)
    if g.bic(X_tr_s) < best_bic:
        best_bic, best_K = g.bic(X_tr_s), K
gmm   = GaussianMixture(n_components=best_K, covariance_type="full", random_state=42, max_iter=200).fit(X_tr_s)
s_gmm = -gmm.score_samples(X_te_s)
auroc_gmm, lo_gmm, hi_gmm = bootstrap_auroc_bca(y_te, s_gmm)
print(f"GMM (K={best_K})    AUROC = {auroc_gmm:.4f}  [{lo_gmm:.4f}, {hi_gmm:.4f}]")

agg_rows = [
    dict(test_task="T4", method="PSR Z-Score",     n_healthy=int((y_te==0).sum()), n_anomaly=int((y_te==1).sum()),
         auroc=round(auroc_zs,4),  ci_lo=round(lo_zs,4),  ci_hi=round(hi_zs,4),  ci_width=round(hi_zs-lo_zs,4)),
    dict(test_task="T4", method="PSR OC-SVM",      n_healthy=int((y_te==0).sum()), n_anomaly=int((y_te==1).sum()),
         auroc=round(auroc_svm,4), ci_lo=round(lo_svm,4), ci_hi=round(hi_svm,4), ci_width=round(hi_svm-lo_svm,4)),
    dict(test_task="T4", method="PSR IsoForest",   n_healthy=int((y_te==0).sum()), n_anomaly=int((y_te==1).sum()),
         auroc=round(auroc_if,4),  ci_lo=round(lo_if,4),  ci_hi=round(hi_if,4),  ci_width=round(hi_if-lo_if,4)),
    dict(test_task="T4", method="GMM (PSR feat.)", n_healthy=int((y_te==0).sum()), n_anomaly=int((y_te==1).sum()),
         auroc=round(auroc_gmm,4), ci_lo=round(lo_gmm,4), ci_hi=round(hi_gmm,4), ci_width=round(hi_gmm-lo_gmm,4)),
]
pd.DataFrame(agg_rows).to_csv(os.path.join(OUT, "T4_psr_family_aggregate.csv"), index=False, float_format="%.4f")

# %% Cell 9: Per-anomaly AUROC at T4 for PSR family

test_data = resid_df[te_mask].copy()
test_data["PSR_ZScore"]    = s_zs
test_data["PSR_OCSVM"]     = s_svm
test_data["PSR_IsoForest"] = s_if
test_data["GMM"]           = s_gmm

per_anom_rows = []
for label, col in [("PSR Z-Score","PSR_ZScore"), ("PSR OC-SVM","PSR_OCSVM"),
                   ("PSR IsoForest","PSR_IsoForest"), ("GMM (PSR feat.)","GMM")]:
    for anom in ["A2","A3","A5"]:
        sub = test_data[(test_data.anomaly == anom) | (test_data.is_anomaly == 0)]
        if len(sub.is_anomaly.unique()) < 2:
            continue
        auroc, lo, hi = bootstrap_auroc_bca(sub.is_anomaly.values, sub[col].values)
        per_anom_rows.append(dict(method=label, anomaly=anom, fold="T4",
                                  n_healthy=int((sub.is_anomaly == 0).sum()),
                                  n_anomaly=int((sub.is_anomaly == 1).sum()),
                                  auroc=round(auroc, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
        print(f"  {label:<18} {anom}: AUROC = {auroc:.4f}  [{lo:.4f}, {hi:.4f}]")
pd.DataFrame(per_anom_rows).to_csv(os.path.join(OUT, "T4_psr_family_per_anomaly.csv"), index=False, float_format="%.4f")

# %% Cell 10: Physics-term ablation at T4

# Active regressor indices for each ablation condition.
# Columns of the regressor matrix: 0=gravity, 1=viscous, 2=Coulomb, 3=inertia, 4=bias.
# Raw conditions are handled separately in Cell 11.
ABLATIONS = {
    "Bias only":     [4],
    "Inertia":       [3, 4],
    "Friction":      [1, 2, 4],
    "Gravity":       [0, 4],
    "M2 (G+V+B)":    [0, 1, 4],
    "M3 (G+V+F+B)":  [0, 1, 2, 4],
    "M4 (Full)":     [0, 1, 2, 3, 4],
}

def zscore_score(X_tr, X_te):
    sc_ = StandardScaler().fit(X_tr)
    Xs_tr = sc_.transform(X_tr); Xs_te = sc_.transform(X_te)
    mu, std = Xs_tr.mean(0), Xs_tr.std(0); std[std == 0] = 1
    return np.mean(((Xs_te - mu) / std) ** 2, axis=1)

ablation_rows = []
for cond, active in ABLATIONS.items():
    masked = {j: psr_weights[j].copy() for j in range(6)}
    for j in range(6):
        for k in range(5):
            if k not in active:
                masked[j][k] = 0.0
    feats_list = []
    for c in all_cycles:
        feats, _ = residual_features(c, masked, TASK_PAYLOAD[c["task"]])
        feats.update(task=c["task"], is_anomaly=c["is_anomaly"], anomaly=c["anomaly"])
        feats_list.append(feats)
    df_abl = pd.DataFrame(feats_list)
    feat_cols = [k for k in df_abl.columns if k not in ("task","is_anomaly","anomaly")]
    tr = (df_abl.task.isin(["T1","T2","T3"])) & (df_abl.is_anomaly == 0)
    te = df_abl.task == "T4"
    s_te = zscore_score(df_abl[tr][feat_cols].values, df_abl[te][feat_cols].values)
    y_te_ = df_abl[te]["is_anomaly"].values
    auroc, lo, hi = bootstrap_auroc_bca(y_te_, s_te)
    ablation_rows.append(dict(condition=cond, fold="T4",
                              auroc=round(auroc, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    print(f"  {cond:<14} AUROC = {auroc:.4f}  [{lo:.4f}, {hi:.4f}]")
pd.DataFrame(ablation_rows).to_csv(os.path.join(OUT, "T4_ablation_psr.csv"), index=False, float_format="%.4f")

# %% Cell 11: Raw-feature ablation conditions at T4

def raw_features(cycle):
    cur = cycle["current"]; f = {}
    for j in range(6):
        s = cur[:, j]; d = np.diff(s)
        p = f"J{j}"
        f[f"{p}_mean"]    = np.mean(s);  f[f"{p}_std"]   = np.std(s)
        f[f"{p}_min"]     = np.min(s);   f[f"{p}_max"]   = np.max(s)
        f[f"{p}_range"]   = np.ptp(s);   f[f"{p}_rms"]   = np.sqrt(np.mean(s**2))
        f[f"{p}_abs_mean"]= np.mean(np.abs(s))
        f[f"{p}_skew"]    = float(sst.skew(s))
        f[f"{p}_kurt"]    = float(sst.kurtosis(s))
        f[f"{p}_p05"]     = np.percentile(s, 5)
        f[f"{p}_p25"]     = np.percentile(s, 25)
        f[f"{p}_p50"]     = np.percentile(s, 50)
        f[f"{p}_p75"]     = np.percentile(s, 75)
        f[f"{p}_p95"]     = np.percentile(s, 95)
        f[f"{p}_iqr"]     = f[f"{p}_p75"] - f[f"{p}_p25"]
        f[f"{p}_dstd"]    = np.std(d)
        f[f"{p}_dabsm"]   = np.mean(np.abs(d))
    f["total_rms"]     = np.sqrt(np.mean(cur**2))
    f["total_abs_max"] = np.max(np.abs(cur))
    return f

raw_rows = []
for c in all_cycles:
    r = raw_features(c)
    r.update(task=c["task"], is_anomaly=c["is_anomaly"])
    raw_rows.append(r)
df_raw   = pd.DataFrame(raw_rows)
raw_cols = [k for k in df_raw.columns if k not in ("task","is_anomaly")]
tr = (df_raw.task.isin(["T1","T2","T3"])) & (df_raw.is_anomaly == 0)
te = df_raw.task == "T4"
y_te_ = df_raw[te]["is_anomaly"].values

s_raw = zscore_score(df_raw[tr][raw_cols].values, df_raw[te][raw_cols].values)
auroc_raw, lo_raw, hi_raw = bootstrap_auroc_bca(y_te_, s_raw)
print(f"  No physics (raw)  AUROC = {auroc_raw:.4f}  [{lo_raw:.4f}, {hi_raw:.4f}]")

sc_p   = StandardScaler().fit(df_raw[tr][raw_cols].values)
Xtr_p  = sc_p.transform(df_raw[tr][raw_cols].values)
Xte_p  = sc_p.transform(df_raw[te][raw_cols].values)
pca    = PCA(n_components=50, random_state=42).fit(Xtr_p)
Xtr_pc = pca.transform(Xtr_p)
Xte_pc = pca.transform(Xte_p)
mu, std = Xtr_pc.mean(0), Xtr_pc.std(0); std[std == 0] = 1
s_pca = np.mean(((Xte_pc - mu) / std) ** 2, axis=1)
auroc_pca, lo_pca, hi_pca = bootstrap_auroc_bca(y_te_, s_pca)
print(f"  No phys. PCA-50   AUROC = {auroc_pca:.4f}  [{lo_pca:.4f}, {hi_pca:.4f}]")

pd.DataFrame([
    dict(condition="No physics (raw)", fold="T4", auroc=round(auroc_raw,4), ci_lo=round(lo_raw,4), ci_hi=round(hi_raw,4)),
    dict(condition="No phys. PCA-50",  fold="T4", auroc=round(auroc_pca,4), ci_lo=round(lo_pca,4), ci_hi=round(hi_pca,4)),
]).to_csv(os.path.join(OUT, "T4_ablation_raw.csv"), index=False, float_format="%.4f")

# %% Cell 12: Save scores for downstream notebooks

psr_scores_T4 = {
    "PSR_ZScore":    s_zs,
    "PSR_OCSVM":     s_svm,
    "PSR_IsoForest": s_if,
    "GMM":           s_gmm,
    "y_true":        y_te,
    "te_cycles":     [{"task": c["task"], "anomaly": c["anomaly"],
                       "severity": c["severity"], "is_anomaly": c["is_anomaly"]}
                       for c in all_cycles if c["task"] == "T4"],
}
with open(os.path.join(OUT, "T4_psr_scores.pkl"), "wb") as fh:
    pickle.dump(psr_scores_T4, fh)

print(f"\nDone. T4 healthy: {len(t4_healthy)},  T4 anomaly: {int((resid_df[te_mask].is_anomaly == 1).sum())}")
print(f"T4 R2_min = {R2_min:+.3f} at {worst_joint}")
print(f"PSR Z-Score T4 AUROC = {auroc_zs:.4f}")
print(f"Next: run T4_baselines.py")
