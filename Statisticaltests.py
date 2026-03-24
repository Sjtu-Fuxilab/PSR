import os, glob, time, warnings
import numpy as np
import pandas as pd
import h5py
import scipy.stats as sst
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")
np.random.seed(42)

# PATHS
ROOT = r"D:\Research\R"
BASE = os.path.join(ROOT, "L_Data")
OUT  = os.path.join(ROOT, "P_Data")
os.makedirs(OUT, exist_ok=True)

# CONSTANTS
TASKS        = ["T1", "T2", "T3"]
TASK_PAYLOAD = {"T1": 0.0, "T2": 1.0, "T3": 3.0}
PAYLOAD_COM  = np.array([0, 0, 0.05])
GRAVITY      = np.array([0, 0, -9.81])
RATE         = 125
SUBSAMPLE    = 4
MIN_SAMP     = 200
N_BOOT       = 10000
TPR_TARGETS  = [0.80, 0.90, 0.95]

SEVERITY_RANK = {
    "none":0,
    "0.5kg":1, "1kg":2,   "2kg":3,
    "1.5kg":1, "3kg":3,
    "3.5kg":1, "4kg":2,   "5kg":3,
    "10wraps":1,"17wraps":2,
    "7duct":1,  "14duct":2,
    "20mm":1,   "50mm":2,  "100mm":3,
}

# UR5 CB3 PARAMETERS
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
    T = [np.eye(4)]
    for i in range(6):
        T.append(T[-1] @ dh_transform(
            UR5_DH_A[i], UR5_DH_D[i], UR5_DH_ALPHA[i], q[i]))
    com_world = [(T[i+1] @ np.array([*UR5_COM[i], 1.0]))[:3]
                 for i in range(6)]
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
            cp = (Tp[jj+1] @ np.array([*UR5_COM[jj], 1.0]))[:3] if jj < 6 \
                 else (Tp[6] @ np.array([*PAYLOAD_COM, 1.0]))[:3]
            tau_g[i] += masses[jj] * GRAVITY @ ((cp - com_world[jj]) / dq)
    return tau_g


# FILE REGISTRY
REGISTRY = {
    "T1_healthy":    ("T1_PickPlace/Healthy",  "UR5_T1_healthy_180cyc_*.h5",
                      "T1","healthy","none",0.0),
    "T2_healthy":    ("T2_Assembly/Healthy",    "UR5_T2_healthy_180cyc_*.h5",
                      "T2","healthy","none",0.0),
    "T3_healthy":    ("T3_Palletize/Healthy",   "UR5_T3_healthy_183cyc_*.h5",
                      "T3","healthy","none",0.0),
    "T1_A2_0.5kg":   ("T1_PickPlace/A2","UR5_T1_A2_0.5kg_gripper_40cyc_*.h5",
                      "T1","A2","0.5kg",0.5),
    "T1_A2_1kg":     ("T1_PickPlace/A2","UR5_T1_A2_1kg_gripper_40cyc_*.h5",
                      "T1","A2","1kg",1.0),
    "T1_A2_2kg":     ("T1_PickPlace/A2","UR5_T1_A2_2kg_gripper_40cyc_*.h5",
                      "T1","A2","2kg",2.0),
    "T1_A3_10wraps": ("T1_PickPlace/A3","UR5_T1_A3_1band_40cyc_*.h5",
                      "T1","A3","10wraps",0.0),
    "T1_A3_17wraps": ("T1_PickPlace/A3","UR5_T1_A3_3bands_40cyc_*.h5",
                      "T1","A3","17wraps",0.0),
    "T1_A5_20mm":    ("T1_PickPlace/A5","UR5_T1_A5_20mm_40cyc_*.h5",
                      "T1","A5","20mm",0.0),
    "T1_A5_50mm":    ("T1_PickPlace/A5","UR5_T1_A5_50mm_40cyc_*.h5",
                      "T1","A5","50mm",0.0),
    "T1_A5_100mm":   ("T1_PickPlace/A5","UR5_T1_A5_100mm_40cyc_*.h5",
                      "T1","A5","100mm",0.0),
    "T2_A2_1.5kg":   ("T2_Assembly/A2","UR5_T2_A2_1.5kg_gripper_40cyc_*.h5",
                      "T2","A2","1.5kg",1.5),
    "T2_A2_2kg":     ("T2_Assembly/A2","UR5_T2_A2_2kg_gripper_40cyc_*.h5",
                      "T2","A2","2kg",2.0),
    "T2_A2_3kg":     ("T2_Assembly/A2","UR5_T2_A2_3kg_gripper_40cyc_*.h5",
                      "T2","A2","3kg",3.0),
    "T2_A3_7duct":   ("T2_Assembly/A3","UR5_T2_A3_light_duct_40cyc_*.h5",
                      "T2","A3","7duct",0.0),
    "T2_A3_14duct":  ("T2_Assembly/A3","UR5_T2_A3_medium_duct_40cyc_*_225508.h5",
                      "T2","A3","14duct",0.0),
    "T2_A5_20mm":    ("T2_Assembly/A5","UR5_T2_A5_20mm_40cyc_*.h5",
                      "T2","A5","20mm",0.0),
    "T2_A5_50mm":    ("T2_Assembly/A5","UR5_T2_A5_50mm_40cyc_*.h5",
                      "T2","A5","50mm",0.0),
    "T2_A5_100mm":   ("T2_Assembly/A5","UR5_T2_A5_100mm_40cyc_*.h5",
                      "T2","A5","100mm",0.0),
    "T3_A2_3.5kg":   ("T3_Palletize/A2","UR5_T3_A2_3.5kg_gripper_33cyc_*.h5",
                      "T3","A2","3.5kg",3.5),
    "T3_A2_4kg":     ("T3_Palletize/A2","UR5_T3_A2_4kg_gripper_33cyc_*.h5",
                      "T3","A2","4kg",4.0),
    "T3_A2_5kg":     ("T3_Palletize/A2","UR5_T3_A2_4.5kg_gripper_33cyc_*.h5",
                      "T3","A2","5kg",5.0),
    "T3_A3_7duct":   ("T3_Palletize/A3","UR5_T3_A3_light_duct_33cyc_*_222457.h5",
                      "T3","A3","7duct",0.0),
    "T3_A3_14duct":  ("T3_Palletize/A3","UR5_T3_A3_medium_duct_33cyc_*_205648.h5",
                      "T3","A3","14duct",0.0),
    "T3_A5_20mm":    ("T3_Palletize/A5","UR5_T3_A5_20mm_33cyc_*_172334.h5",
                      "T3","A5","20mm",0.0),
    "T3_A5_50mm":    ("T3_Palletize/A5","UR5_T3_A5_50mm_33cyc_*_164447.h5",
                      "T3","A5","50mm",0.0),
    "T3_A5_100mm":   ("T3_Palletize/A5","UR5_T3_A5_100mm_33cyc_*_160716.h5",
                      "T3","A5","100mm",0.0),
}

# STATISTICAL FUNCTIONS

def bootstrap_auroc_bca(y_true, y_score, n_boot=N_BOOT, rng=None):
    """BCa bootstrap 95% CI for AUROC (Efron 1987). Cycle-level resampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y_true)
    auroc_obs = roc_auc_score(y_true, y_score)
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; ys = y_score[idx]
        boot[b] = roc_auc_score(yt, ys) \
            if (0 < yt.sum() < n) else auroc_obs
    prop = np.clip(np.mean(boot < auroc_obs), 1e-6, 1 - 1e-6)
    z0   = sst.norm.ppf(prop)
    jack = np.zeros(n)
    for i in range(n):
        idx_j = np.concatenate([np.arange(i), np.arange(i+1, n)])
        yt_j = y_true[idx_j]; ys_j = y_score[idx_j]
        jack[i] = roc_auc_score(yt_j, ys_j) \
            if (0 < yt_j.sum() < len(yt_j)) else auroc_obs
    jm  = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a   = num / den if den != 0 else 0.0
    ci = {}
    for label, z_a in [("lo", sst.norm.ppf(0.025)),
                        ("hi", sst.norm.ppf(0.975))]:
        p = sst.norm.cdf(z0 + (z0 + z_a) / (1 - a * (z0 + z_a)))
        ci[label] = float(np.quantile(boot, np.clip(p, 0.001, 0.999)))
    return float(auroc_obs), ci["lo"], ci["hi"]


def delong_paired(y_true, scores_a, scores_b):
    """
    DeLong's nonparametric paired test for two correlated AUROCs.
    DeLong et al. (1988) Biometrics 44:837-845.
    Returns (auc_a, auc_b, delta, z, p).
    """
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    np_, nn = len(pos), len(neg)

    def placements(sc):
        V10 = np.array([
            np.mean(sc[p] > sc[neg]) + 0.5 * np.mean(sc[p] == sc[neg])
            for p in pos])
        V01 = np.array([
            np.mean(sc[pos] > sc[n]) + 0.5 * np.mean(sc[pos] == sc[n])
            for n in neg])
        return V10.mean(), V10, V01

    auc_a, V10a, V01a = placements(scores_a)
    auc_b, V10b, V01b = placements(scores_b)
    S10 = np.cov(np.stack([V10a, V10b])) / np_
    S01 = np.cov(np.stack([V01a, V01b])) / nn
    S   = S10 + S01
    delta    = auc_a - auc_b
    var_diff = S[0,0] + S[1,1] - 2*S[0,1]
    if var_diff <= 0:
        return auc_a, auc_b, delta, np.nan, np.nan
    z = delta / np.sqrt(var_diff)
    p = 2 * (1 - sst.norm.cdf(abs(z)))
    return auc_a, auc_b, delta, z, p


def sig_stars(p):
    if np.isnan(p): return "n.d."
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"


def operating_point(y_true, y_score, tpr_target):
    """Find threshold where TPR >= tpr_target. Returns (tpr, fpr, thr, prec)."""
    fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, y_score)
    idx  = min(np.searchsorted(tpr_arr, tpr_target), len(tpr_arr)-1)
    tp   = tpr_arr[idx]; fp = fpr_arr[idx]
    thr  = float(thr_arr[idx]) if idx < len(thr_arr) else np.nan
    pred = (y_score >= thr).astype(int)
    tp_a = int(((pred==1) & (y_true==1)).sum())
    fp_a = int(((pred==1) & (y_true==0)).sum())
    prec = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else np.nan
    return float(tp), float(fp), thr, float(prec)


# STEP 1 -- Computational benchmark (inference)
print("=" * 65)
print("NB10 -- Statistical Tests  [v2: strict per-fold PSR fitting]")
print("=" * 65)
print("\n[Step 1] Computational benchmark (inference)...")

matches_t1  = sorted(glob.glob(
    os.path.join(BASE, "T1_PickPlace/Healthy", "UR5_T1_healthy_180cyc_*.h5")))
bench_rows  = []

if matches_t1:
    with h5py.File(matches_t1[0], "r") as f:
        cnum_b = f["cycle_number"][:].astype(int).ravel()
        q_b    = f["actual_q"][:]
        qd_b   = f["actual_qd"][:]
        cur_b  = f["actual_current"][:]
    bench_cycs = []
    for c in np.unique(cnum_b[cnum_b > 0]):
        mask = cnum_b == c
        if mask.sum() >= MIN_SAMP:
            bench_cycs.append({"q": q_b[mask], "qd": qd_b[mask],
                                "current": cur_b[mask]})
        if len(bench_cycs) >= 5:
            break
    dummy_w = {j: np.array([0.1, 1.5, 0.5, 0.2, 0.3]) for j in range(6)}
    cyc0   = bench_cycs[0]
    N_cyc  = len(cyc0["q"])
    N_REPS = 50
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        for t in range(0, N_cyc, SUBSAMPLE):
            tau_g = gravity_torque(cyc0["q"][t], payload_mass=0.0)
            for j in range(6):
                qdd_j = ((cyc0["qd"][t+1,j] - cyc0["qd"][t-1,j]) * RATE / 2.0
                         if 0 < t < N_cyc-1 else 0.0)
                phi = np.array([tau_g[j], cyc0["qd"][t,j],
                                np.sign(cyc0["qd"][t,j]), qdd_j, 1.0])
                _ = cyc0["current"][t, j] - phi @ dummy_w[j]
    inf_ms = (time.perf_counter() - t0) / N_REPS * 1000
    print(f"  Inference: {inf_ms:.1f} ms/cycle  "
          f"(N={N_cyc} samples, SUBSAMPLE={SUBSAMPLE})")
    bench_rows.append(dict(operation="Single-cycle PSR inference",
                           value=round(inf_ms, 1), unit="ms",
                           notes=f"N={N_cyc} samples, SUBSAMPLE={SUBSAMPLE}, "
                                 "includes gravity_torque numerical diff"))
    bench_rows.append(dict(operation="PSR model storage",
                           value=30*8, unit="bytes",
                           notes="30 float64 weights (6 joints x 5 regressors)"))

pd.DataFrame(bench_rows).to_csv(
    os.path.join(OUT, "NB10_compute_benchmark.csv"), index=False)

# STEP 2 -- Load all data
print("\n[Step 2] Loading data...")
all_cycles = []
for key, (subdir, pattern, task, anomaly, severity, _) in REGISTRY.items():
    matches = sorted(glob.glob(os.path.join(BASE, subdir, pattern)))
    if not matches:
        print(f"  WARNING  Not found: {key}")
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
                "q": q_all[mask], "qd": qd_all[mask],
                "current": cur_all[mask],
                "task": task, "anomaly": anomaly,
                "severity": severity, "is_anomaly": is_anom,
            })

healthy_cycles = [c for c in all_cycles if c["is_anomaly"] == 0]
print(f"  Total: {len(all_cycles)} | Healthy: {len(healthy_cycles)}")

# STEP 3 -- PSR per-fold fit function + training time benchmark

print("\n[Step 3] PSR per-fold fit function; timing one representative fold...")


def fit_psr_fold(healthy_cycs):
    """
    Fit PSR OLS weights per joint.
    Per-cycle TASK_PAYLOAD[cyc["task"]] -- never a pooled or mean payload.
    Returns dict psr_w where psr_w[j] is (5,) weight vector for joint j.
    """
    train_Phi = {j: [] for j in range(6)}
    train_I   = {j: [] for j in range(6)}
    for cyc in healthy_cycs:
        payload = TASK_PAYLOAD[cyc["task"]]
        q_a = cyc["q"]; qd_a = cyc["qd"]; cur = cyc["current"]
        N   = len(q_a)
        for t in range(0, N, SUBSAMPLE):
            tau_g = gravity_torque(q_a[t], payload_mass=payload)
            for j in range(6):
                qdd_j = ((qd_a[t+1,j] - qd_a[t-1,j]) * RATE / 2.0
                         if 0 < t < N-1 else 0.0)
                train_Phi[j].append(
                    np.array([tau_g[j], qd_a[t,j], np.sign(qd_a[t,j]),
                              qdd_j, 1.0]))
                train_I[j].append(cur[t, j])
    psr_w = {}
    for j in range(6):
        w, _, _, _ = np.linalg.lstsq(
            np.array(train_Phi[j]), np.array(train_I[j]), rcond=None)
        psr_w[j] = w
    return psr_w


_bench_tr = [c for c in healthy_cycles if c["task"] in ["T2", "T3"]]
t_fit_start = time.perf_counter()
_ = fit_psr_fold(_bench_tr)
fit_s = time.perf_counter() - t_fit_start
n_samp_bench = sum(len(range(0, len(c["q"]), SUBSAMPLE)) for c in _bench_tr)
print(f"  Fold fit time: {fit_s:.1f} s  "
      f"({len(_bench_tr)} cycles, {n_samp_bench:,} subsampled samples)")
bench_rows.append(dict(
    operation="PSR training (per LOTO fold, 2 tasks)",
    value=round(fit_s, 1), unit="s",
    notes=f"{len(_bench_tr)} healthy cycles, {n_samp_bench:,} subsampled samples; "
          "per-cycle task-specific payload"))
pd.DataFrame(bench_rows).to_csv(
    os.path.join(OUT, "NB10_compute_benchmark.csv"), index=False)
print("  NB10_compute_benchmark.csv updated")
del _bench_tr

# STEP 4 -- Feature extraction functions
PSR_COLS = ([f"J{j}_{s}" for j in range(6)
             for s in ["resid_mean","resid_std","resid_rms","resid_max",
                       "resid_skew","resid_kurtosis",
                       "grav_resid_std","grav_resid_rms"]]
            + ["total_resid_rms","J1J2_resid_corr"])

RAW_COLS = ([f"J{j}_{s}" for j in range(6)
             for s in ["raw_mean","raw_std","raw_rms"]]
            + ["total_raw_rms"])


def extract_psr(cyc, psr_w):
    """
    Extract 50-dim PSR feature vector for one cycle.
    psr_w must be the fold-specific weights from fit_psr_fold.
    """
    payload = TASK_PAYLOAD[cyc["task"]]
    q_a = cyc["q"]; qd_a = cyc["qd"]; cur = cyc["current"]
    N   = len(q_a)
    idx = list(range(0, N, SUBSAMPLE))
    res = np.zeros((len(idx), 6))
    gr  = np.zeros((len(idx), 6))
    for ti, t in enumerate(idx):
        tau_g = gravity_torque(q_a[t], payload_mass=payload)
        for j in range(6):
            qdd_j = ((qd_a[t+1,j] - qd_a[t-1,j]) * RATE / 2.0
                     if 0 < t < N-1 else 0.0)
            phi = np.array([tau_g[j], qd_a[t,j],
                            np.sign(qd_a[t,j]), qdd_j, 1.0])
            res[ti, j] = cur[t, j] - phi @ psr_w[j]
            gr[ti, j]  = cur[t, j] - (psr_w[j][0]*tau_g[j] + psr_w[j][4])
    f = {}
    for j in range(6):
        r = res[:, j]; g = gr[:, j]
        f[f"J{j}_resid_mean"]     = r.mean()
        f[f"J{j}_resid_std"]      = r.std()
        f[f"J{j}_resid_rms"]      = np.sqrt(np.mean(r**2))
        f[f"J{j}_resid_max"]      = np.abs(r).max()
        f[f"J{j}_resid_skew"]     = float(sst.skew(r))
        f[f"J{j}_resid_kurtosis"] = float(sst.kurtosis(r))
        f[f"J{j}_grav_resid_std"] = g.std()
        f[f"J{j}_grav_resid_rms"] = np.sqrt(np.mean(g**2))
    f["total_resid_rms"] = np.sqrt(np.mean(res**2))
    f["J1J2_resid_corr"] = float(np.corrcoef(res[:,1], res[:,2])[0,1]
                                  if len(res) > 2 else 0.0)
    return np.array([f[k] for k in PSR_COLS])


def extract_raw(cyc):
    """Extract 19-dim raw current feature vector for one cycle."""
    cur = cyc["current"]
    idx = list(range(0, len(cur), SUBSAMPLE))
    c   = cur[idx]
    f   = {}
    for j in range(6):
        f[f"J{j}_raw_mean"] = c[:, j].mean()
        f[f"J{j}_raw_std"]  = c[:, j].std()
        f[f"J{j}_raw_rms"]  = np.sqrt(np.mean(c[:, j]**2))
    f["total_raw_rms"] = np.sqrt(np.mean(c**2))
    return np.array([f[k] for k in RAW_COLS])


# STEP 5: Detector functions

def zscore(Xtr, Xte):
    mu = Xtr.mean(0); sg = Xtr.std(0) + 1e-8
    return np.abs((Xte - mu) / sg).mean(1)

def ocsvm(Xtr, Xte):
    sc  = StandardScaler().fit(Xtr)
    clf = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    clf.fit(sc.transform(Xtr))
    return -clf.decision_function(sc.transform(Xte))

def isoforest(Xtr, Xte):
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    clf.fit(Xtr)
    return -clf.decision_function(Xte)

# (scorer_fn, feature_type): feature_type "PSR" uses PSR features, "Raw" uses raw
DETECTORS = {
    "PSR_ZScore":    (zscore,     "PSR"),
    "PSR_OCSVM":     (ocsvm,      "PSR"),
    "PSR_IsoForest": (isoforest,  "PSR"),
    "Raw_ZScore":    (zscore,     "Raw"),
}

# STEP 6 -- LOTO evaluation: per-fold PSR fitting + feature extraction
print("\n[Step 6] LOTO evaluation (strict per-fold PSR fitting)...")

agg_scores  = {}   # (task, det)         -> (y_true, y_score)
anom_scores = {}   # (task, anom, det)   -> (y_true, y_score)
sev_scores  = {}   # (task, anom, sev)   -> (y_true, y_score)  [PSR_ZScore]

for test_task in TASKS:
    tr_tasks   = [t for t in TASKS if t != test_task]
    tr_healthy = [c for c in healthy_cycles if c["task"] in tr_tasks]
    te_cycles  = [c for c in all_cycles     if c["task"] == test_task]
    te_h_idx   = [i for i, c in enumerate(te_cycles) if c["is_anomaly"] == 0]

    print(f"\n  Fold: test={test_task}  train={tr_tasks}  "
          f"tr_n={len(tr_healthy)}  te_n={len(te_cycles)}")

    # -- Fit PSR on training tasks only --
    psr_w_fold = fit_psr_fold(tr_healthy)

    # -- Extract features: training (for detector fit) and test (for scoring) --
    print(f"    Extracting PSR features...", end=" ", flush=True)
    Xtr_psr = np.array([extract_psr(c, psr_w_fold) for c in tr_healthy])
    Xte_psr = np.array([extract_psr(c, psr_w_fold) for c in te_cycles])
    print("done PSR", end=" ", flush=True)
    Xtr_raw = np.array([extract_raw(c) for c in tr_healthy])
    Xte_raw = np.array([extract_raw(c) for c in te_cycles])
    print("done Raw")
    y_te = np.array([c["is_anomaly"] for c in te_cycles])

    feat_map = {"PSR": (Xtr_psr, Xte_psr), "Raw": (Xtr_raw, Xte_raw)}

    for det_name, (det_fn, ftype) in DETECTORS.items():
        Xtr, Xte = feat_map[ftype]

        # Aggregate
        scores = det_fn(Xtr, Xte)
        agg_scores[(test_task, det_name)] = (y_te, scores)
        auroc = roc_auc_score(y_te, scores)
        print(f"    {det_name:15s} | aggregate AUROC={auroc:.4f}")

        # Per-anomaly type
        for anom in ["A2", "A3", "A5"]:
            anom_idx = [i for i, c in enumerate(te_cycles)
                        if c["anomaly"] == anom]
            if not anom_idx:
                continue
            comb_idx = te_h_idx + anom_idx
            Xte_a = Xte[comb_idx]
            yte_a = np.array([te_cycles[i]["is_anomaly"] for i in comb_idx])
            sc_a  = det_fn(Xtr, Xte_a)
            anom_scores[(test_task, anom, det_name)] = (yte_a, sc_a)

        # Per-severity (PSR_ZScore only -- for operating point table)
        if det_name == "PSR_ZScore":
            sc_h = scores[te_h_idx]
            seen_sev = set()
            for i, c in enumerate(te_cycles):
                if c["is_anomaly"] == 0:
                    continue
                key = (c["anomaly"], c["severity"])
                seen_sev.add(key)
            for (anom, sev) in seen_sev:
                sv_idx = [i for i, c in enumerate(te_cycles)
                          if c["anomaly"] == anom and c["severity"] == sev]
                yt_s  = np.concatenate([np.zeros(len(te_h_idx)),
                                        np.ones(len(sv_idx))])
                sc_cs = np.concatenate([sc_h, scores[sv_idx]])
                sev_scores[(test_task, anom, sev)] = (yt_s, sc_cs)

# STEP 7 -- Bootstrap BCa CIs
print(f"\n[Step 7] Bootstrap BCa CIs (N_BOOT={N_BOOT})...")
rng = np.random.default_rng(42)

# 7a -- Aggregate
agg_ci_rows = []
for (task, det), (yt, ys) in agg_scores.items():
    auroc, lo, hi = bootstrap_auroc_bca(yt, ys, rng=rng)
    agg_ci_rows.append(dict(
        test_task=task, detector=det,
        n_healthy=int((yt==0).sum()), n_anomaly=int((yt==1).sum()),
        auroc=round(auroc,4), ci_lo=round(lo,4), ci_hi=round(hi,4),
        ci_width=round(hi-lo,4)))
    print(f"  {task} | {det:15s} | {auroc:.4f} [{lo:.4f}, {hi:.4f}]")

pd.DataFrame(agg_ci_rows).to_csv(
    os.path.join(OUT, "NB10_bootstrap_ci_aggregate.csv"), index=False)
print("  Saved: NB10_bootstrap_ci_aggregate.csv")

# 7b -- Per anomaly type
anom_ci_rows = []
for (task, anom, det), (yt, ys) in anom_scores.items():
    auroc, lo, hi = bootstrap_auroc_bca(yt, ys, rng=rng)
    anom_ci_rows.append(dict(
        test_task=task, anomaly_type=anom, detector=det,
        n_healthy=int((yt==0).sum()), n_anomaly=int((yt==1).sum()),
        auroc=round(auroc,4), ci_lo=round(lo,4), ci_hi=round(hi,4),
        ci_width=round(hi-lo,4)))

pd.DataFrame(anom_ci_rows).sort_values(
    ["anomaly_type","test_task","detector"]).to_csv(
    os.path.join(OUT, "NB10_bootstrap_ci_per_anomaly.csv"), index=False)
print("  Saved: NB10_bootstrap_ci_per_anomaly.csv")

print("\n  Per-anomaly BCa CI (PSR_OCSVM -- proposed method):")
print(f"  {'Task':3} {'Anomaly':5} {'AUROC':>8}  {'95% BCa CI':>20}")
for row in sorted(anom_ci_rows, key=lambda r: (r["anomaly_type"], r["test_task"])):
    if row["detector"] != "PSR_OCSVM":
        continue
    print(f"  {row['test_task']:3} {row['anomaly_type']:5} "
          f"{row['auroc']:8.4f}  [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]")

# STEP 8 -- DeLong paired tests
print("\n[Step 8] DeLong paired tests...")
delong_rows = []

# 8a -- PSR_OCSVM vs Raw_ZScore (main paper)
print("\n  Aggregate: PSR_OCSVM vs Raw_ZScore  [MAIN TABLE]")
for task in TASKS:
    yt       = agg_scores[(task, "PSR_OCSVM")][0]
    ys_ocsvm = agg_scores[(task, "PSR_OCSVM")][1]
    ys_raw   = agg_scores[(task, "Raw_ZScore")][1]
    au, ab, d, z, p = delong_paired(yt, ys_ocsvm, ys_raw)
    stars = sig_stars(p)
    print(f"  {task}: OCSVM={au:.4f}  Raw={ab:.4f}  "
          f"delta={d:+.4f}  z={z:.2f}  p={p:.3e}  {stars}")
    delong_rows.append(dict(
        scope="aggregate", test_task=task, anomaly_type="all",
        comparison="PSR_OCSVM vs Raw_ZScore",
        auc_a=round(au,4), auc_b=round(ab,4),
        delta=round(d,4), z=round(z,3),
        p_value=round(p,6) if not np.isnan(p) else np.nan,
        significance=stars))

# 8b -- PSR_OCSVM vs PSR_ZScore, PSR_IsoForest (supplementary ablation)
print("\n  Aggregate: PSR_OCSVM vs PSR ablation variants  [SUPPLEMENTARY]")
for task in TASKS:
    yt       = agg_scores[(task, "PSR_OCSVM")][0]
    ys_ocsvm = agg_scores[(task, "PSR_OCSVM")][1]
    for comp_det in ["PSR_ZScore", "PSR_IsoForest"]:
        ys_c = agg_scores[(task, comp_det)][1]
        au, ab, d, z, p = delong_paired(yt, ys_ocsvm, ys_c)
        stars = sig_stars(p)
        print(f"  {task}: OCSVM={au:.4f}  {comp_det}={ab:.4f}  "
              f"delta={d:+.4f}  p={p:.3e}  {stars}")
        delong_rows.append(dict(
            scope="aggregate", test_task=task, anomaly_type="all",
            comparison=f"PSR_OCSVM vs {comp_det}",
            auc_a=round(au,4), auc_b=round(ab,4),
            delta=round(d,4), z=round(z,3),
            p_value=round(p,6) if not np.isnan(p) else np.nan,
            significance=stars))

# 8c -- Per anomaly type: PSR_OCSVM vs Raw_ZScore
print("\n  Per anomaly: PSR_OCSVM vs Raw_ZScore")
for task in TASKS:
    for anom in ["A2", "A3", "A5"]:
        if (task, anom, "PSR_OCSVM") not in anom_scores:
            continue
        yt       = anom_scores[(task, anom, "PSR_OCSVM")][0]
        ys_ocsvm = anom_scores[(task, anom, "PSR_OCSVM")][1]
        ys_raw   = anom_scores[(task, anom, "Raw_ZScore")][1]
        au, ab, d, z, p = delong_paired(yt, ys_ocsvm, ys_raw)
        stars = sig_stars(p)
        print(f"  {task} {anom}: OCSVM={au:.4f}  Raw={ab:.4f}  "
              f"delta={d:+.4f}  z={z:.2f}  p={p:.3e}  {stars}")
        delong_rows.append(dict(
            scope="per_anomaly", test_task=task, anomaly_type=anom,
            comparison="PSR_OCSVM vs Raw_ZScore",
            auc_a=round(au,4), auc_b=round(ab,4),
            delta=round(d,4), z=round(z,3),
            p_value=round(p,6) if not np.isnan(p) else np.nan,
            significance=stars))

pd.DataFrame(delong_rows).to_csv(
    os.path.join(OUT, "NB10_delong_tests.csv"), index=False)
print(f"\n  Saved: NB10_delong_tests.csv  ({len(delong_rows)} rows)")

# STEP 9 -- Operating points (per anomaly, per severity, PSR_ZScore)
print(f"\n[Step 9] Operating points at TPR={TPR_TARGETS}...")
op_rows = []

for (task, anom, sev), (yt, ys) in sev_scores.items():
    auroc    = roc_auc_score(yt, ys)
    sev_rank = SEVERITY_RANK.get(sev, -1)
    for tpr_t in TPR_TARGETS:
        tpr_v, fpr_v, thr_v, prec_v = operating_point(yt, ys, tpr_t)
        op_rows.append(dict(
            test_task=task, anomaly_type=anom, severity=sev,
            severity_rank=sev_rank, auroc=round(auroc,4),
            tpr_target=tpr_t,
            achieved_tpr=round(tpr_v,4),
            fpr=round(fpr_v,4),
            false_alarms_per_100=round(fpr_v*100,1),
            threshold=round(thr_v,4) if thr_v is not None else np.nan,
            precision=round(prec_v,4) if not np.isnan(prec_v) else np.nan,
            n_healthy=int((yt==0).sum()),
            n_anomaly=int((yt==1).sum()),
        ))

op_df = pd.DataFrame(op_rows).sort_values(
    ["anomaly_type","severity_rank","test_task","tpr_target"])
op_df.to_csv(os.path.join(OUT, "NB10_operating_points.csv"), index=False)
print(f"  Saved: NB10_operating_points.csv  ({len(op_df)} rows)")

print(f"\n  Operating point at TPR=0.95 (PSR_ZScore):")
print(f"  {'Task':3} {'Anom':4} {'Severity':10} {'AUROC':>7}  "
      f"{'FPR':>7}  {'FA/100':>7}  {'Precision':>10}")
print("  " + "-" * 60)
for _, row in op_df[op_df["tpr_target"]==0.95].sort_values(
        ["anomaly_type","severity_rank","test_task"]).iterrows():
    flag = " !!" if row["fpr"] > 0.20 else ""
    print(f"  {row['test_task']:3} {row['anomaly_type']:4} "
          f"{row['severity']:10} {row['auroc']:7.4f}  "
          f"{row['fpr']:7.4f}  {row['false_alarms_per_100']:7.1f}  "
          f"{row['precision']:10.4f}{flag}")

# STEP 10 -- Summary
print("\n" + "=" * 65)
print("NB10 COMPLETE  [v2 -- strict per-fold PSR fitting]")
print("=" * 65)

print("\n  AUROC summary (strict LOTO):")
print(f"  {'Detector':15s}  {'T1':>8}  {'T2':>8}  {'T3':>8}  {'Mean':>8}")
print("  " + "-" * 55)
ci_df = pd.DataFrame(agg_ci_rows)
for det in ["PSR_OCSVM", "PSR_ZScore", "PSR_IsoForest", "Raw_ZScore"]:
    sub  = ci_df[ci_df["detector"] == det]
    vals = {r["test_task"]: r["auroc"] for _, r in sub.iterrows()}
    t1   = vals.get("T1", np.nan)
    t2   = vals.get("T2", np.nan)
    t3   = vals.get("T3", np.nan)
    mn   = np.nanmean([t1, t2, t3])
    print(f"  {det:15s}  {t1:8.4f}  {t2:8.4f}  {t3:8.4f}  {mn:8.4f}")

print("\n[Outputs written to Processed_Data/ -- all v1 corrupted files overwritten]")
for fname in ["NB10_compute_benchmark.csv",
              "NB10_bootstrap_ci_aggregate.csv",
              "NB10_bootstrap_ci_per_anomaly.csv",
              "NB10_delong_tests.csv",
              "NB10_operating_points.csv"]:
    fpath = os.path.join(OUT, fname)
    if os.path.exists(fpath):
        nrows = len(pd.read_csv(fpath))
        print(f"  OK  {fname}  ({nrows} rows)")
    else:
        print(f"  MISSING  {fname}")

print("\n[No figures. No formatted tables.]")
print("Next: NB10c (patched) -> NB10d -> NB11.")
