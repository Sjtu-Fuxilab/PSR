# DEPENDENCY CHECK
import sys
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} — {'GPU' if torch.cuda.is_available() else 'CPU'}")
except ImportError:
    print("ERROR: PyTorch not found. Install with: pip install torch")
    print("       CPU-only install: pip install torch --index-url "
          "https://download.pytorch.org/whl/cpu")
    sys.exit(1)

import os, glob, warnings, time
import numpy as np
import pandas as pd
import h5py
import scipy.stats as sst
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# PATHS
ROOT = r"D:\Research\R"
BASE = os.path.join(ROOT, "L_Data")
OUT  = os.path.join(ROOT, "P_Data")
os.makedirs(OUT, exist_ok=True)

# CONSTANTS
TASKS        = ["T1", "T2", "T3"]
TASK_PAYLOAD = {"T1": 0.0, "T2": 1.0, "T3": 3.0}
RATE         = 125
SUBSAMPLE    = 4        # must match NB10
MIN_SAMP     = 200      # must match NB10
N_BOOT       = 10000
EPOCHS       = 60
BATCH_SIZE   = 32
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FILE REGISTRY
REGISTRY = {
    "T1_healthy":    ("T1_PickPlace/Healthy",  "UR5_T1_healthy_180cyc_*.h5",
                      "T1","healthy","none"),
    "T2_healthy":    ("T2_Assembly/Healthy",    "UR5_T2_healthy_180cyc_*.h5",
                      "T2","healthy","none"),
    "T3_healthy":    ("T3_Palletize/Healthy",   "UR5_T3_healthy_183cyc_*.h5",
                      "T3","healthy","none"),
    "T1_A2_0.5kg":   ("T1_PickPlace/A2","UR5_T1_A2_0.5kg_gripper_40cyc_*.h5",
                      "T1","A2","0.5kg"),
    "T1_A2_1kg":     ("T1_PickPlace/A2","UR5_T1_A2_1kg_gripper_40cyc_*.h5",
                      "T1","A2","1kg"),
    "T1_A2_2kg":     ("T1_PickPlace/A2","UR5_T1_A2_2kg_gripper_40cyc_*.h5",
                      "T1","A2","2kg"),
    "T1_A3_10wraps": ("T1_PickPlace/A3","UR5_T1_A3_1band_40cyc_*.h5",
                      "T1","A3","10wraps"),
    "T1_A3_17wraps": ("T1_PickPlace/A3","UR5_T1_A3_3bands_40cyc_*.h5",
                      "T1","A3","17wraps"),
    "T1_A5_20mm":    ("T1_PickPlace/A5","UR5_T1_A5_20mm_40cyc_*.h5",
                      "T1","A5","20mm"),
    "T1_A5_50mm":    ("T1_PickPlace/A5","UR5_T1_A5_50mm_40cyc_*.h5",
                      "T1","A5","50mm"),
    "T1_A5_100mm":   ("T1_PickPlace/A5","UR5_T1_A5_100mm_40cyc_*.h5",
                      "T1","A5","100mm"),
    "T2_A2_1.5kg":   ("T2_Assembly/A2","UR5_T2_A2_1.5kg_gripper_40cyc_*.h5",
                      "T2","A2","1.5kg"),
    "T2_A2_2kg":     ("T2_Assembly/A2","UR5_T2_A2_2kg_gripper_40cyc_*.h5",
                      "T2","A2","2kg"),
    "T2_A2_3kg":     ("T2_Assembly/A2","UR5_T2_A2_3kg_gripper_40cyc_*.h5",
                      "T2","A2","3kg"),
    "T2_A3_7duct":   ("T2_Assembly/A3","UR5_T2_A3_light_duct_40cyc_*.h5",
                      "T2","A3","7duct"),
    "T2_A3_14duct":  ("T2_Assembly/A3","UR5_T2_A3_medium_duct_40cyc_*_225508.h5",
                      "T2","A3","14duct"),
    "T2_A5_20mm":    ("T2_Assembly/A5","UR5_T2_A5_20mm_40cyc_*.h5",
                      "T2","A5","20mm"),
    "T2_A5_50mm":    ("T2_Assembly/A5","UR5_T2_A5_50mm_40cyc_*.h5",
                      "T2","A5","50mm"),
    "T2_A5_100mm":   ("T2_Assembly/A5","UR5_T2_A5_100mm_40cyc_*.h5",
                      "T2","A5","100mm"),
    "T3_A2_3.5kg":   ("T3_Palletize/A2","UR5_T3_A2_3.5kg_gripper_33cyc_*.h5",
                      "T3","A2","3.5kg"),
    "T3_A2_4kg":     ("T3_Palletize/A2","UR5_T3_A2_4kg_gripper_33cyc_*.h5",
                      "T3","A2","4kg"),
    "T3_A2_5kg":     ("T3_Palletize/A2","UR5_T3_A2_4.5kg_gripper_33cyc_*.h5",
                      "T3","A2","5kg"),
    "T3_A3_7duct":   ("T3_Palletize/A3","UR5_T3_A3_light_duct_33cyc_*_222457.h5",
                      "T3","A3","7duct"),
    "T3_A3_14duct":  ("T3_Palletize/A3","UR5_T3_A3_medium_duct_33cyc_*_205648.h5",
                      "T3","A3","14duct"),
    "T3_A5_20mm":    ("T3_Palletize/A5","UR5_T3_A5_20mm_33cyc_*_172334.h5",
                      "T3","A5","20mm"),
    "T3_A5_50mm":    ("T3_Palletize/A5","UR5_T3_A5_50mm_33cyc_*_164447.h5",
                      "T3","A5","50mm"),
    "T3_A5_100mm":   ("T3_Palletize/A5","UR5_T3_A5_100mm_33cyc_*_160716.h5",
                      "T3","A5","100mm"),
}

# STATISTICAL FUNCTIONS
def bootstrap_auroc_bca(y_true, y_score, n_boot=N_BOOT, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y_true)
    auroc_obs = roc_auc_score(y_true, y_score)
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; ys = y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            boot[b] = auroc_obs
        else:
            boot[b] = roc_auc_score(yt, ys)
    prop = np.clip(np.mean(boot < auroc_obs), 1e-6, 1 - 1e-6)
    z0   = sst.norm.ppf(prop)
    jack = np.zeros(n)
    for i in range(n):
        idx_j = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        yt_j  = y_true[idx_j]; ys_j = y_score[idx_j]
        jack[i] = roc_auc_score(yt_j, ys_j) \
            if (0 < yt_j.sum() < len(yt_j)) else auroc_obs
    jm  = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a   = num / den if den != 0 else 0.0
    ci_lo = ci_hi = None
    for z_a, attr in [(sst.norm.ppf(0.025), "lo"),
                      (sst.norm.ppf(0.975), "hi")]:
        p = sst.norm.cdf(z0 + (z0 + z_a) / (1 - a * (z0 + z_a)))
        p = np.clip(p, 0.001, 0.999)
        if attr == "lo":
            ci_lo = float(np.quantile(boot, p))
        else:
            ci_hi = float(np.quantile(boot, p))
    return float(auroc_obs), ci_lo, ci_hi


def delong_paired(y_true, scores_a, scores_b):
    """DeLong et al. (1988) Biometrics 44:837-845."""
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
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
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


# CONV-AE ARCHITECTURE

class ConvAutoencoder(nn.Module):
    """
    1D convolutional autoencoder for multivariate time-series anomaly detection.
    Input shape: (batch, 6, fixed_len)  — channels-first convention.
    Score: mean per-cycle MSE between input and reconstruction.

    Architecture chosen to be comparable to published baselines (Wang et al.
    JMS 2024 uses a similar depth). Kernel sizes chosen to capture both local
    (joint-level) and medium-range (cycle-phase) temporal patterns.
    """
    def __init__(self, fixed_len: int, n_channels: int = 6):
        super().__init__()
        self.fixed_len  = fixed_len
        self.n_channels = n_channels

        # Encoder: two 2× pooling stages → spatial compression factor 4
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),                        # len // 2
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),                        # len // 4
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder: two 2× upsampling stages to restore original length
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, n_channels, kernel_size=7, padding=3),
        )

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        # Ensure reconstructed length matches input (may differ by 1 due to
        # integer division in MaxPool1d with odd lengths)
        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = x_hat[..., :x.shape[-1]]
        return x_hat

    def reconstruction_score(self, x):
        """Per-sample mean MSE score. Higher = more anomalous."""
        with torch.no_grad():
            x_hat = self.forward(x)
            # MSE per sample: mean over channels and time
            mse = ((x - x_hat) ** 2).mean(dim=[1, 2])
        return mse.cpu().numpy()


# STEP 1 — Load data
print("=" * 65)
print("NB — Convolutional Autoencoder Baseline")
print("=" * 65)
print(f"\nDevice: {DEVICE}")
print("\n[Step 1] Loading data...")

all_cycles = []
for key, (subdir, pattern, task, anomaly, severity) in REGISTRY.items():
    matches = sorted(glob.glob(os.path.join(BASE, subdir, pattern)))
    if not matches:
        print(f"  WARNING  Not found: {key}")
        continue
    with h5py.File(matches[0], "r") as f:
        cnum    = f["cycle_number"][:].astype(int).ravel()
        cur_all = f["actual_current"][:]
    is_anom = 0 if anomaly == "healthy" else 1
    for c in np.unique(cnum[cnum > 0]):
        mask = cnum == c
        if mask.sum() >= MIN_SAMP:
            all_cycles.append({
                "current":    cur_all[mask],
                "task":       task,
                "anomaly":    anomaly,
                "severity":   severity,
                "is_anomaly": is_anom,
            })

healthy_cycles = [c for c in all_cycles if c["is_anomaly"] == 0]
print(f"  Total: {len(all_cycles)} | Healthy: {len(healthy_cycles)}")

# STEP 2 — Determine FIXED_LEN from subsampled cycle lengths
print("\n[Step 2] Determining FIXED_LEN...")

sub_lengths = []
for cyc in all_cycles:
    n_sub = len(range(0, len(cyc["current"]), SUBSAMPLE))
    sub_lengths.append(n_sub)

sub_lengths = np.array(sub_lengths)
p5  = int(np.percentile(sub_lengths, 5))
p50 = int(np.percentile(sub_lengths, 50))
print(f"  Subsampled lengths: min={sub_lengths.min()}  "
      f"p5={p5}  median={p50}  max={sub_lengths.max()}")

# Round down to nearest power of 2 for clean encoder/decoder arithmetic
# MaxPool1d(2) twice → length must be divisible by 4
FIXED_LEN = max(p5 - (p5 % 4), 64)
print(f"  FIXED_LEN={FIXED_LEN}  "
      f"(cycles shorter than this will be zero-padded, "
      f"longer will be truncated)")
n_padded   = int(np.sum(sub_lengths < FIXED_LEN))
n_truncated = int(np.sum(sub_lengths > FIXED_LEN))
print(f"  Padded: {n_padded}/{len(all_cycles)}  "
      f"Truncated: {n_truncated}/{len(all_cycles)}")


def cycle_to_tensor(cyc, fixed_len=FIXED_LEN):
    """
    Convert a cycle's raw current to a fixed-length tensor.
    Shape: (n_channels, fixed_len) — channels-first for Conv1d.
    Subsampling matches NB10 (SUBSAMPLE=4).
    Normalisation: per-cycle per-joint zero-mean unit-variance.
    """
    cur   = cyc["current"]
    idx   = list(range(0, len(cur), SUBSAMPLE))
    ts    = cur[idx]                          # (n_sub, 6)
    # Normalise per-channel (prevents large-payload channels dominating loss)
    mu    = ts.mean(0, keepdims=True)
    sg    = ts.std(0, keepdims=True) + 1e-8
    ts    = (ts - mu) / sg
    # Truncate or pad to FIXED_LEN
    if len(ts) >= fixed_len:
        ts = ts[:fixed_len]
    else:
        pad  = np.zeros((fixed_len - len(ts), ts.shape[1]))
        ts   = np.vstack([ts, pad])
    return ts.T.astype(np.float32)            # (6, fixed_len)


# STEP 3 — LOTO training and evaluation
print("\n[Step 3] LOTO training and scoring...")

# agg_scores[(test_task)] = (y_true, y_score)
# anom_scores[(test_task, anom)] = (y_true, y_score)
agg_scores  = {}
anom_scores = {}

for test_task in TASKS:
    tr_tasks = [t for t in TASKS if t != test_task]
    print(f"\n  Fold: test={test_task}  train={tr_tasks}")

    # Build training tensors
    tr_tensors = [cycle_to_tensor(c)
                  for c in healthy_cycles if c["task"] in tr_tasks]
    X_tr = torch.tensor(np.stack(tr_tensors)).to(DEVICE)
    tr_ds = TensorDataset(X_tr)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       drop_last=False)
    print(f"  Training samples: {len(X_tr)}")

    # Train Conv-AE 
    model = ConvAutoencoder(FIXED_LEN).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    t0 = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for (xb,) in tr_dl:
            opt.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_tr)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS}  loss={epoch_loss:.6f}")
    train_s = time.perf_counter() - t0
    print(f"  Training time: {train_s:.1f} s")

    # Score test cycles
    model.eval()
    te_cycles = [c for c in all_cycles if c["task"] == test_task]
    scores    = []
    labels    = []
    anomalies = []
    severities = []

    for cyc in te_cycles:
        x = torch.tensor(cycle_to_tensor(cyc)).unsqueeze(0).to(DEVICE)
        sc = model.reconstruction_score(x)[0]
        scores.append(sc)
        labels.append(cyc["is_anomaly"])
        anomalies.append(cyc["anomaly"])
        severities.append(cyc["severity"])

    y_true  = np.array(labels,    dtype=int)
    y_score = np.array(scores,    dtype=float)
    auroc   = roc_auc_score(y_true, y_score)
    print(f"  Aggregate AUROC ({test_task}): {auroc:.4f}")

    agg_scores[test_task] = (y_true, y_score)

    # Per-anomaly scores 
    h_mask   = np.array([c["task"] == test_task and c["is_anomaly"] == 0
                         for c in all_cycles])
    h_scores = y_score[y_true == 0]

    for anom in ["A2", "A3", "A5"]:
        a_mask   = (np.array(anomalies) == anom) & (y_true == 1)
        if a_mask.sum() == 0:
            continue
        a_scores = y_score[a_mask]
        yt_a     = np.concatenate([np.zeros(len(h_scores)),
                                   np.ones(len(a_scores))])
        ys_a     = np.concatenate([h_scores, a_scores])
        au_a     = roc_auc_score(yt_a, ys_a)
        anom_scores[(test_task, anom)] = (yt_a, ys_a)
        print(f"    {test_task} {anom}: AUROC={au_a:.4f}")

# STEP 4 — Bootstrap BCa CIs
print(f"\n[Step 4] Bootstrap BCa CIs (N_BOOT={N_BOOT})...")
rng = np.random.default_rng(42)

agg_ci_rows = []
for task in TASKS:
    yt, ys = agg_scores[task]
    auroc, lo, hi = bootstrap_auroc_bca(yt, ys, rng=rng)
    agg_ci_rows.append(dict(
        test_task=task,
        method="Conv-AE (raw current)",
        n_healthy=int((yt==0).sum()), n_anomaly=int((yt==1).sum()),
        auroc=round(auroc,4), ci_lo=round(lo,4), ci_hi=round(hi,4),
        ci_width=round(hi-lo,4)))
    print(f"  {task}: AUROC={auroc:.4f} [{lo:.4f}, {hi:.4f}]")

pd.DataFrame(agg_ci_rows).to_csv(
    os.path.join(OUT, "NB10b_convae_auroc_aggregate.csv"), index=False)
print(f"  Saved: NB10b_convae_auroc_aggregate.csv")

anom_ci_rows = []
for (task, anom), (yt, ys) in anom_scores.items():
    auroc, lo, hi = bootstrap_auroc_bca(yt, ys, rng=rng)
    anom_ci_rows.append(dict(
        test_task=task, anomaly_type=anom,
        method="Conv-AE (raw current)",
        n_healthy=int((yt==0).sum()), n_anomaly=int((yt==1).sum()),
        auroc=round(auroc,4), ci_lo=round(lo,4), ci_hi=round(hi,4),
        ci_width=round(hi-lo,4)))

anom_ci_df = pd.DataFrame(anom_ci_rows).sort_values(
    ["anomaly_type","test_task"])
anom_ci_df.to_csv(
    os.path.join(OUT, "NB10b_convae_auroc_per_anomaly.csv"), index=False)
print(f"  Saved: NB10b_convae_auroc_per_anomaly.csv")

# STEP 5 — DeLong test: PSR Z-Score vs Conv-AE (aggregate per task)
print("\n[Step 5] Re-computing PSR Z-Score scores for paired DeLong test...")

import scipy.stats as sst_

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
PAYLOAD_COM = np.array([0, 0, 0.05])
GRAVITY     = np.array([0, 0, -9.81])


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


# Reload full HDF5 data (need q, qd for PSR)
print("  Reloading full kinematics for PSR re-computation...")
full_cycles = []
for key, (subdir, pattern, task, anomaly, severity) in REGISTRY.items():
    matches = sorted(glob.glob(os.path.join(BASE, subdir, pattern)))
    if not matches:
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
            full_cycles.append({
                "q": q_all[mask], "qd": qd_all[mask],
                "current": cur_all[mask],
                "task": task, "anomaly": anomaly,
                "severity": severity, "is_anomaly": is_anom,
            })

full_healthy = [c for c in full_cycles if c["is_anomaly"] == 0]

PSR_COLS = ([f"J{j}_{s}" for j in range(6)
             for s in ["resid_mean","resid_std","resid_rms","resid_max",
                       "resid_skew","resid_kurtosis",
                       "grav_resid_std","grav_resid_rms"]]
            + ["total_resid_rms","J1J2_resid_corr"])


def extract_psr_features(cyc, psr_w):
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
        f[f"J{j}_resid_skew"]     = float(sst_.skew(r))
        f[f"J{j}_resid_kurtosis"] = float(sst_.kurtosis(r))
        f[f"J{j}_grav_resid_std"] = g.std()
        f[f"J{j}_grav_resid_rms"] = np.sqrt(np.mean(g**2))
    f["total_resid_rms"]  = np.sqrt(np.mean(res**2))
    f["J1J2_resid_corr"]  = float(np.corrcoef(res[:,1], res[:,2])[0,1]
                                   if len(res) > 2 else 0.0)
    return f


# Fit global PSR
print("  Fitting PSR (global, all healthy cycles)...")
train_Phi = {j: [] for j in range(6)}
train_I   = {j: [] for j in range(6)}
for cyc in full_healthy:
    payload = TASK_PAYLOAD[cyc["task"]]
    q_a = cyc["q"]; qd_a = cyc["qd"]; cur = cyc["current"]
    N = len(q_a)
    for t in range(0, N, SUBSAMPLE):
        tau_g = gravity_torque(q_a[t], payload_mass=payload)
        for j in range(6):
            qdd_j = ((qd_a[t+1,j] - qd_a[t-1,j]) * RATE / 2.0
                     if 0 < t < N-1 else 0.0)
            train_Phi[j].append(np.array([tau_g[j], qd_a[t,j],
                                          np.sign(qd_a[t,j]), qdd_j, 1.0]))
            train_I[j].append(cur[t, j])
psr_w = {}
for j in range(6):
    w, _, _, _ = np.linalg.lstsq(
        np.array(train_Phi[j]), np.array(train_I[j]), rcond=None)
    psr_w[j] = w

# Extract PSR features
psr_rows = []
for cyc in full_cycles:
    f = extract_psr_features(cyc, psr_w)
    f.update(task=cyc["task"], anomaly=cyc["anomaly"],
             severity=cyc["severity"], is_anomaly=cyc["is_anomaly"])
    psr_rows.append(f)
psr_df = pd.DataFrame(psr_rows)

# LOTO Z-Score scoring (PSR)
psr_agg_scores = {}
for test_task in TASKS:
    tr_tasks = [t for t in TASKS if t != test_task]
    Xtr = psr_df[psr_df["task"].isin(tr_tasks) &
                 (psr_df["is_anomaly"]==0)][PSR_COLS].values
    Xte = psr_df[psr_df["task"]==test_task][PSR_COLS].values
    yte = psr_df[psr_df["task"]==test_task]["is_anomaly"].values
    mu  = Xtr.mean(0); sg = Xtr.std(0) + 1e-8
    ys  = np.abs((Xte - mu) / sg).mean(1)
    psr_agg_scores[test_task] = (yte, ys)
    auroc = roc_auc_score(yte, ys)
    print(f"  PSR Z-Score {test_task}: AUROC={auroc:.4f}  "
          f"(should match NB10: ~{[0.860,0.953,0.989][TASKS.index(test_task)]:.3f})")

# STEP 6 — DeLong: PSR Z-Score vs Conv-AE (paired, same test cycles)
print("\n[Step 6] DeLong paired test: PSR Z-Score vs Conv-AE...")

delong_rows = []
for task in TASKS:
    yt_psr, ys_psr = psr_agg_scores[task]
    yt_cae, ys_cae = agg_scores[task]
    # Verify same test set (should be identical by construction)
    assert np.array_equal(yt_psr, yt_cae), \
        f"Test set mismatch for {task}: PSR and Conv-AE must score same cycles"
    auc_psr, auc_cae, delta, z, p = delong_paired(yt_psr, ys_psr, ys_cae)
    stars = sig_stars(p)
    print(f"  {task}: PSR={auc_psr:.4f}  Conv-AE={auc_cae:.4f}  "
          f"Δ={delta:+.4f}  z={z:.2f}  p={p:.3e}  {stars}")
    delong_rows.append(dict(
        test_task=task,
        comparison="PSR_ZScore vs Conv-AE",
        auc_psr=round(auc_psr,4), auc_cae=round(auc_cae,4),
        delta_auc=round(delta,4),
        z_stat=round(z,3),
        p_value=round(p,6) if not np.isnan(p) else np.nan,
        significance=stars))

pd.DataFrame(delong_rows).to_csv(
    os.path.join(OUT, "NB10b_delong_psr_vs_convae.csv"), index=False)
print(f"  Saved: NB10b_delong_psr_vs_convae.csv")

# STEP 7 — Summary
print("\n" + "=" * 65)
print("NB10b COMPLETE")
print("=" * 65)

print("\n[Conv-AE vs PSR Z-Score — Aggregate AUROC]")
print(f"  {'Task':3}  {'Conv-AE':>22}  {'PSR Z-Score':>22}")
print("  " + "-" * 50)
for row in agg_ci_rows:
    task     = row["test_task"]
    psr_row  = [r for r in [] if r["test_task"]==task]  # placeholder
    cae_str  = f"{row['auroc']:.4f} [{row['ci_lo']:.4f},{row['ci_hi']:.4f}]"
    yt, ys   = psr_agg_scores[task]
    psr_au   = roc_auc_score(yt, ys)
    print(f"  {task}  Conv-AE={cae_str}  PSR={psr_au:.4f}")

print("\n[Outputs written to Processed_Data/]")
for f in ["NB10b_convae_auroc_aggregate.csv",
          "NB10b_convae_auroc_per_anomaly.csv",
          "NB10b_delong_psr_vs_convae.csv"]:
    fpath = os.path.join(OUT, f)
    if os.path.exists(fpath):
        nrows = len(pd.read_csv(fpath))
        print(f"  ✓  {f}  ({nrows} rows)")
    else:
        print(f"  ✗  {f}  MISSING")

print("\n[No figures generated — NB11 reads these CSVs for Table 1]")
print("\nNBb fills Gap 1 (comparison to published baseline).")
print("Conv-AE (Wang et al. JMS 2024 analogue) now appears as Row 3 in Table 1.")
