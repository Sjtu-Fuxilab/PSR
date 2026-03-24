# DEPENDENCIES
import sys
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} -- "
          f"{'GPU' if torch.cuda.is_available() else 'CPU'}")
except ImportError:
    print("ERROR: PyTorch not found. Install: pip install torch")
    sys.exit(1)

import os, glob, warnings, time
import numpy as np
import pandas as pd
import h5py
import scipy.stats as sst
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

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
SUBSAMPLE    = 4
MIN_SAMP     = 200
N_BOOT       = 10000
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LSTMVAE_EPOCHS   = 80
LSTMVAE_BATCH    = 32
LSTMVAE_LR       = 1e-3
LSTMVAE_HIDDEN   = 64
LSTMVAE_LATENT   = 32
LSTMVAE_N_LAYERS = 2
LSTMVAE_BETA     = 1.0

GMM_MAX_COMP = 8
GMM_MAX_ITER = 500
GMM_N_INIT   = 5

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
    """Bootstrap BCa 95% CI on AUROC."""
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
            np.mean(sc[p] > sc[neg]) + 0.5*np.mean(sc[p] == sc[neg])
            for p in pos])
        V01 = np.array([
            np.mean(sc[pos] > sc[n]) + 0.5*np.mean(sc[pos] == sc[n])
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


# PSR PHYSICS
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


# LSTM-VAE ARCHITECTURE
class LSTMEncoder(nn.Module):
    def __init__(self, n_channels, hidden_dim, latent_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True,
                            bidirectional=True)
        self.fc_mu     = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_fwd = h[-2]; h_bwd = h[-1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)
        return self.fc_mu(h_cat), self.fc_logvar(h_cat)


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_channels, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.fc_init    = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, n_channels)

    def forward(self, z, seq_len):
        z_rep = z.unsqueeze(1).repeat(1, seq_len, 1)
        h0 = torch.tanh(self.fc_init(z)).unsqueeze(0).repeat(self.n_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(z_rep, (h0, c0))
        return self.fc_out(out)


class LSTMVAE(nn.Module):
    def __init__(self, n_channels=6,
                 hidden_dim=LSTMVAE_HIDDEN,
                 latent_dim=LSTMVAE_LATENT,
                 n_layers=LSTMVAE_N_LAYERS,
                 beta=LSTMVAE_BETA):
        super().__init__()
        self.beta    = beta
        self.encoder = LSTMEncoder(n_channels, hidden_dim, latent_dim, n_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, n_channels, n_layers)

    def reparameterise(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z, x.shape[1]), mu, logvar

    def elbo_loss(self, x):
        x_hat, mu, logvar = self.forward(x)
        recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kl, recon.item()

    @torch.no_grad()
    def reconstruction_score(self, x):
        self.eval()
        x_hat, _, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=[1, 2]).cpu().numpy()


# STEP 1: Load data
print("=" * 65)
print("NB10c -- LSTM-VAE and GMM Baselines  [v2: patched]")
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
        q_all   = f["actual_q"][:]
        qd_all  = f["actual_qd"][:]
    is_anom = 0 if anomaly == "healthy" else 1
    for c in np.unique(cnum[cnum > 0]):
        mask = cnum == c
        if mask.sum() >= MIN_SAMP:
            all_cycles.append({
                "current": cur_all[mask], "q": q_all[mask], "qd": qd_all[mask],
                "task": task, "anomaly": anomaly,
                "severity": severity, "is_anomaly": is_anom,
            })

healthy_cycles = [c for c in all_cycles if c["is_anomaly"] == 0]
print(f"  Total cycles: {len(all_cycles)} | Healthy: {len(healthy_cycles)}")
for t in TASKS:
    nh = sum(1 for c in all_cycles if c["task"]==t and c["is_anomaly"]==0)
    na = sum(1 for c in all_cycles if c["task"]==t and c["is_anomaly"]==1)
    print(f"    {t}: {nh} healthy, {na} anomaly")

# STEP 2: Sequence preparation (for LSTM-VAE)
print("\n[Step 2] Determining FIXED_LEN for LSTM-VAE sequences...")

sub_lengths = np.array([len(range(0, len(c["current"]), SUBSAMPLE))
                        for c in all_cycles])
p5        = int(np.percentile(sub_lengths, 5))
FIXED_LEN = max(p5 - (p5 % 4), 64)
print(f"  Subsampled lengths: min={sub_lengths.min()}  "
      f"p5={p5}  median={int(np.median(sub_lengths))}  max={sub_lengths.max()}")
print(f"  FIXED_LEN = {FIXED_LEN}")
print(f"  Padded: {int((sub_lengths < FIXED_LEN).sum())}/{len(all_cycles)}  "
      f"Truncated: {int((sub_lengths > FIXED_LEN).sum())}/{len(all_cycles)}")


def cycle_to_sequence(cyc, fixed_len=FIXED_LEN):
    """Fixed-length normalised sequence (fixed_len, 6) for LSTM input."""
    cur = cyc["current"]
    idx = list(range(0, len(cur), SUBSAMPLE))
    ts  = cur[idx].astype(np.float32)
    mu  = ts.mean(0, keepdims=True); sg = ts.std(0, keepdims=True) + 1e-8
    ts  = (ts - mu) / sg
    if len(ts) >= fixed_len:
        return ts[:fixed_len]
    pad = np.zeros((fixed_len - len(ts), ts.shape[1]), dtype=np.float32)
    return np.vstack([ts, pad])


# STEP 3: PSR fitting + feature extraction (per LOTO fold)
print("\n[Step 3] PSR fitting and feature extraction (per LOTO fold)...")

PSR_COLS = ([f"J{j}_{s}" for j in range(6)
             for s in ["resid_mean","resid_std","resid_rms","resid_max",
                       "resid_skew","resid_kurtosis",
                       "grav_resid_std","grav_resid_rms"]]
            + ["total_resid_rms","J1J2_resid_corr"])

RAW_COLS = ([f"J{j}_{s}" for j in range(6)
             for s in ["raw_mean","raw_std","raw_rms"]]
            + ["total_raw_rms"])


def fit_psr(healthy_cycs):
    """
    Fit PSR Ridge weights per joint.
    FIX v2: uses per-cycle TASK_PAYLOAD[cyc["task"]] -- never a pooled payload.
    Returns W (6, 5) weight matrix.
    """
    lam = 1e-4
    W   = np.zeros((6, 5))
    for j in range(6):
        Phi_j = []
        y_j   = []
        for cyc in healthy_cycs:
            payload = TASK_PAYLOAD[cyc["task"]]   # <-- FIX: per-cycle payload
            q_a = cyc["q"]; qd_a = cyc["qd"]; cur = cyc["current"]
            N   = len(q_a)
            for t in range(0, N, SUBSAMPLE):
                tau_g = gravity_torque(q_a[t], payload_mass=payload)
                qdd_j = ((qd_a[min(t+1,N-1),j] - qd_a[max(t-1,0),j])
                         * RATE / 2.0)
                Phi_j.append(np.array([tau_g[j], qd_a[t,j],
                                       np.sign(qd_a[t,j]), qdd_j, 1.0]))
                y_j.append(cur[t, j])
        Phi_arr = np.array(Phi_j)
        y_arr   = np.array(y_j)
        W[j] = np.linalg.solve(
            Phi_arr.T @ Phi_arr + lam * np.eye(5), Phi_arr.T @ y_arr)
    return W


def extract_psr_features(cyc, psr_w):
    """Extract 50-dim PSR feature vector using fold-specific weights."""
    payload = TASK_PAYLOAD[cyc["task"]]
    q_a = cyc["q"]; qd_a = cyc["qd"]; cur = cyc["current"]
    N   = len(q_a)
    idx = list(range(0, N, SUBSAMPLE))
    res = np.zeros((len(idx), 6))
    gr  = np.zeros((len(idx), 6))
    for ti, t in enumerate(idx):
        tau_g = gravity_torque(q_a[t], payload_mass=payload)
        for j in range(6):
            qdd_j = ((qd_a[min(t+1,N-1),j] - qd_a[max(t-1,0),j])
                     * RATE / 2.0)
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


def extract_raw_features(cyc):
    """Extract 19-dim raw current feature vector."""
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


# Cache per-fold features
# fold_features[test_task] = (y_true, X_psr_test, anomaly_labels,
#                              X_psr_train, X_raw_test, X_raw_train)
fold_features = {}

for test_task in TASKS:
    tr_tasks   = [t for t in TASKS if t != test_task]
    tr_healthy = [c for c in healthy_cycles if c["task"] in tr_tasks]
    te_cycles  = [c for c in all_cycles     if c["task"] == test_task]

    print(f"  Fitting PSR for LOTO fold test={test_task}...", end=" ", flush=True)
    W = fit_psr(tr_healthy)   # per-cycle payload inside fit_psr
    print("done")

    psr_te   = np.array([extract_psr_features(c, W) for c in te_cycles])
    psr_tr   = np.array([extract_psr_features(c, W) for c in tr_healthy])
    raw_te   = np.array([extract_raw_features(c)    for c in te_cycles])
    raw_tr   = np.array([extract_raw_features(c)    for c in tr_healthy])
    y_true   = np.array([c["is_anomaly"]             for c in te_cycles])
    anom_lbl = [c["anomaly"]                         for c in te_cycles]

    fold_features[test_task] = (y_true, psr_te, anom_lbl, psr_tr, raw_te, raw_tr)
    print(f"    {test_task}: {len(te_cycles)} test cycles, "
          f"PSR feature matrix {psr_te.shape}")

# STEP 4 -- GMM fitting and scoring (BIC component selection)
print("\n[Step 4] GMM fitting and scoring (BIC component selection)...")

gmm_agg_scores  = {}
gmm_anom_scores = {}

for test_task in TASKS:
    tr_tasks = [t for t in TASKS if t != test_task]
    y_true, X_psr_test, anom_lbl, X_psr_train, _, _ = fold_features[test_task]

    # Normalise features
    sc  = StandardScaler().fit(X_psr_train)
    Xtr = sc.transform(X_psr_train)
    Xte = sc.transform(X_psr_test)

    # BIC-based component selection
    print(f"  {test_task}: BIC component selection (1-{GMM_MAX_COMP})...", end=" ")
    bic_scores = []
    for k in range(1, GMM_MAX_COMP + 1):
        gmm_k = GaussianMixture(n_components=k, max_iter=GMM_MAX_ITER,
                                n_init=GMM_N_INIT, random_state=42)
        gmm_k.fit(Xtr)
        bic_scores.append(gmm_k.bic(Xtr))
    best_k = np.argmin(bic_scores) + 1
    print(f"best n_components={best_k} (BIC={bic_scores[best_k-1]:.1f})")

    gmm = GaussianMixture(n_components=best_k, max_iter=GMM_MAX_ITER,
                          n_init=GMM_N_INIT, random_state=42)
    gmm.fit(Xtr)

    # Anomaly score: negative log-likelihood (higher = more anomalous)
    y_score = -gmm.score_samples(Xte)
    auroc   = roc_auc_score(y_true, y_score)
    print(f"  {test_task}: Aggregate AUROC={auroc:.4f}")
    gmm_agg_scores[test_task] = (y_true, y_score)

    h_scores  = y_score[y_true == 0]
    anom_arr  = np.array(anom_lbl)
    for anom in ["A2", "A3", "A5"]:
        a_mask = (anom_arr == anom) & (y_true == 1)
        if a_mask.sum() == 0:
            continue
        yt_a = np.concatenate([np.zeros(h_scores.sum()==0 or len(h_scores)),
                               np.ones(a_mask.sum())])
        yt_a = np.concatenate([np.zeros(len(h_scores)), np.ones(a_mask.sum())])
        ys_a = np.concatenate([h_scores, y_score[a_mask]])
        au_a = roc_auc_score(yt_a, ys_a)
        gmm_anom_scores[(test_task, anom)] = (yt_a, ys_a)
        print(f"    {test_task} {anom}: AUROC={au_a:.4f}")

# STEP 5: LSTM-VAE training and scoring
print(f"\n[Step 5] LSTM-VAE training and scoring "
      f"(epochs={LSTMVAE_EPOCHS}, lr={LSTMVAE_LR}, "
      f"hidden={LSTMVAE_HIDDEN}, latent={LSTMVAE_LATENT})...")

lstmvae_agg_scores  = {}
lstmvae_anom_scores = {}

for test_task in TASKS:
    tr_tasks   = [t for t in TASKS if t != test_task]
    tr_healthy = [c for c in healthy_cycles if c["task"] in tr_tasks]
    te_cycles  = [c for c in all_cycles     if c["task"] == test_task]
    print(f"\n  Fold: test={test_task}  train={tr_tasks}")

    tr_seqs = [cycle_to_sequence(c) for c in tr_healthy]
    X_tr    = torch.tensor(np.stack(tr_seqs)).to(DEVICE)
    tr_dl   = DataLoader(TensorDataset(X_tr), batch_size=LSTMVAE_BATCH,
                         shuffle=True, drop_last=False)
    print(f"  Training samples: {len(X_tr)}")

    model     = LSTMVAE(n_channels=6).to(DEVICE)
    opt       = optim.Adam(model.parameters(), lr=LSTMVAE_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=LSTMVAE_EPOCHS, eta_min=1e-5)

    t0 = time.perf_counter()
    model.train()
    for epoch in range(LSTMVAE_EPOCHS):
        epoch_loss = 0.0
        for (xb,) in tr_dl:
            opt.zero_grad()
            loss, _ = model.elbo_loss(xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_tr)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{LSTMVAE_EPOCHS}  "
                  f"loss={epoch_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
    print(f"  Training time: {time.perf_counter()-t0:.1f} s")

    model.eval()
    scores     = []
    labels     = []
    anom_types = []
    for cyc in te_cycles:
        seq = cycle_to_sequence(cyc)
        x   = torch.tensor(seq).unsqueeze(0).to(DEVICE)
        scores.append(model.reconstruction_score(x)[0])
        labels.append(cyc["is_anomaly"])
        anom_types.append(cyc["anomaly"])

    y_true  = np.array(labels,  dtype=int)
    y_score = np.array(scores,  dtype=float)
    auroc   = roc_auc_score(y_true, y_score)
    print(f"  Aggregate AUROC ({test_task}): {auroc:.4f}")
    lstmvae_agg_scores[test_task] = (y_true, y_score)

    h_scores = y_score[y_true == 0]
    anom_arr = np.array(anom_types)
    for anom in ["A2", "A3", "A5"]:
        a_mask = (anom_arr == anom) & (y_true == 1)
        if a_mask.sum() == 0:
            continue
        yt_a = np.concatenate([np.zeros(len(h_scores)), np.ones(a_mask.sum())])
        ys_a = np.concatenate([h_scores, y_score[a_mask]])
        lstmvae_anom_scores[(test_task, anom)] = (yt_a, ys_a)
        print(f"    {test_task} {anom}: AUROC={roc_auc_score(yt_a, ys_a):.4f}")

# STEP 6: PSR_OCSVM, PSR_ZScore, and Raw_ZScore scoring per fold
print("\n[Step 6] PSR_OCSVM / PSR_ZScore / Raw_ZScore LOTO scores...")

psr_ocsvm_scores   = {}   # (test_task) -> (y_true, y_score)
psr_zscore_scores  = {}
raw_zscore_scores  = {}

for test_task in TASKS:
    y_true, X_psr_test, _, X_psr_train, X_raw_test, X_raw_train = \
        fold_features[test_task]

    # PSR_OCSVM
    sc_psr   = StandardScaler().fit(X_psr_train)
    clf_svm  = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    clf_svm.fit(sc_psr.transform(X_psr_train))
    ys_ocsvm = -clf_svm.decision_function(sc_psr.transform(X_psr_test))
    psr_ocsvm_scores[test_task]  = (y_true, ys_ocsvm)
    print(f"  {test_task}: PSR_OCSVM  AUROC={roc_auc_score(y_true, ys_ocsvm):.4f}")

    # PSR_ZScore (kept for supplementary; will not appear in main DeLong table)
    mu_psr  = X_psr_train.mean(0); sg_psr = X_psr_train.std(0) + 1e-8
    ys_zpsr = np.abs((X_psr_test - mu_psr) / sg_psr).mean(1)
    psr_zscore_scores[test_task] = (y_true, ys_zpsr)
    print(f"  {test_task}: PSR_ZScore AUROC={roc_auc_score(y_true, ys_zpsr):.4f}"
          "  [supplementary only -- see protocol note]")

    # Raw_ZScore
    mu_raw  = X_raw_train.mean(0); sg_raw = X_raw_train.std(0) + 1e-8
    ys_zraw = np.abs((X_raw_test - mu_raw) / sg_raw).mean(1)
    raw_zscore_scores[test_task] = (y_true, ys_zraw)
    print(f"  {test_task}: Raw_ZScore AUROC={roc_auc_score(y_true, ys_zraw):.4f}")

# STEP 7 -- Bootstrap BCa CIs
print(f"\n[Step 7] Bootstrap BCa CIs (N_BOOT={N_BOOT})...")
rng = np.random.default_rng(42)

# LSTM-VAE aggregate
lstmvae_agg_rows = []
for task in TASKS:
    yt, ys = lstmvae_agg_scores[task]
    auroc, lo, hi = bootstrap_auroc_bca(yt, ys, rng=rng)
    lstmvae_agg_rows.append(dict(
        test_task=task, method="LSTM-VAE (raw current)",
        n_healthy=int((yt==0).sum()), n_anomaly=int((yt==1).sum()),
        auroc=round(auroc,4), ci_lo=round(lo,4), ci_hi=round(hi,4),
        ci_width=round(hi-lo,4)))
    print(f"  LSTM-VAE  {task}: {auroc:.4f} [{lo:.4f}, {hi:.4f}]")

pd.DataFrame(lstmvae_agg_rows).to_csv(
    os.path.join(OUT, "NB10c_lstmvae_auroc_aggregate.csv"), index=False)
print("  Saved: NB10c_lstmvae_auroc_aggregate.csv")

# LSTM-VAE per-anomaly
lstmvae_anom_rows = []
for (task, anom), (yt, ys) in lstmvae_anom_scores.items():
    auroc, lo, hi = bootstrap_auroc_bca(yt, ys, rng=rng)
    lstmvae_anom_rows.append(dict(
        test_task=task, anomaly_type=anom, method="LSTM-VAE (raw current)",
        n_healthy=int((yt==0).sum()), n_anomaly=int((yt==1).sum()),
        auroc=round(auroc,4), ci_lo=round(lo,4), ci_hi=round(hi,4)))

pd.DataFrame(lstmvae_anom_rows).sort_values(["anomaly_type","test_task"]).to_csv(
    os.path.join(OUT, "NB10c_lstmvae_auroc_per_anomaly.csv"), index=False)
print("  Saved: NB10c_lstmvae_auroc_per_anomaly.csv")

# GMM aggregate
gmm_agg_rows = []
for task in TASKS:
    yt, ys = gmm_agg_scores[task]
    auroc, lo, hi = bootstrap_auroc_bca(yt, ys, rng=rng)
    gmm_agg_rows.append(dict(
        test_task=task, method="GMM (PSR features)",
        n_healthy=int((yt==0).sum()), n_anomaly=int((yt==1).sum()),
        auroc=round(auroc,4), ci_lo=round(lo,4), ci_hi=round(hi,4),
        ci_width=round(hi-lo,4)))
    print(f"  GMM       {task}: {auroc:.4f} [{lo:.4f}, {hi:.4f}]")

pd.DataFrame(gmm_agg_rows).to_csv(
    os.path.join(OUT, "NB10c_gmm_auroc_aggregate.csv"), index=False)
print("  Saved: NB10c_gmm_auroc_aggregate.csv")

# GMM per-anomaly
gmm_anom_rows = []
for (task, anom), (yt, ys) in gmm_anom_scores.items():
    auroc, lo, hi = bootstrap_auroc_bca(yt, ys, rng=rng)
    gmm_anom_rows.append(dict(
        test_task=task, anomaly_type=anom, method="GMM (PSR features)",
        n_healthy=int((yt==0).sum()), n_anomaly=int((yt==1).sum()),
        auroc=round(auroc,4), ci_lo=round(lo,4), ci_hi=round(hi,4)))

pd.DataFrame(gmm_anom_rows).sort_values(["anomaly_type","test_task"]).to_csv(
    os.path.join(OUT, "NB10c_gmm_auroc_per_anomaly.csv"), index=False)
print("  Saved: NB10c_gmm_auroc_per_anomaly.csv")


# STEP 8: Paired DeLong tests: PSR_OCSVM vs LSTM-VAE, GMM, Raw_ZScore
print("\n[Step 8] Paired DeLong tests (PSR_OCSVM vs all external baselines)...")

# Helper to build a DeLong row dict
def delong_row(task, auc_ocsvm, auc_base, delta, z, p, baseline_name):
    return dict(test_task=task,
                auc_psr_ocsvm=round(auc_ocsvm, 4),
                auc_baseline=round(auc_base, 4),
                baseline=baseline_name,
                delta=round(delta, 4),
                z_stat=round(z, 3) if not np.isnan(z) else np.nan,
                p_value=round(p, 6) if not np.isnan(p) else np.nan,
                sig=sig_stars(p))

# PSR_OCSVM vs LSTM-VAE
delong_lstmvae_rows = []
print(f"\n  PSR_OCSVM vs LSTM-VAE:")
print(f"  {'Task':3}  {'AUC_OCSVM':>10}  {'AUC_LSTMVAE':>11}  "
      f"{'delta':>8}  {'z':>7}  {'p':>10}  sig")
for task in TASKS:
    yt_o, ys_o = psr_ocsvm_scores[task]
    yt_l, ys_l = lstmvae_agg_scores[task]
    assert np.array_equal(yt_o, yt_l), f"y_true mismatch at {task}"
    auc_o, auc_l, delta, z, p = delong_paired(yt_o, ys_o, ys_l)
    row = delong_row(task, auc_o, auc_l, delta, z, p, "LSTM-VAE")
    delong_lstmvae_rows.append(row)
    print(f"  {task}   {auc_o:.4f}       {auc_l:.4f}       "
          f"{delta:+.4f}  {z:7.2f}  {p:.6f}  {sig_stars(p)}")

pd.DataFrame(delong_lstmvae_rows).to_csv(
    os.path.join(OUT, "NB10c_delong_ocsvm_vs_lstmvae.csv"), index=False)
print("  Saved: NB10c_delong_ocsvm_vs_lstmvae.csv")

# --- PSR_OCSVM vs GMM ---
delong_gmm_rows = []
print(f"\n  PSR_OCSVM vs GMM:")
print(f"  {'Task':3}  {'AUC_OCSVM':>10}  {'AUC_GMM':>8}  "
      f"{'delta':>8}  {'z':>7}  {'p':>10}  sig")
for task in TASKS:
    yt_o, ys_o = psr_ocsvm_scores[task]
    yt_g, ys_g = gmm_agg_scores[task]
    assert np.array_equal(yt_o, yt_g)
    auc_o, auc_g, delta, z, p = delong_paired(yt_o, ys_o, ys_g)
    row = delong_row(task, auc_o, auc_g, delta, z, p, "GMM")
    delong_gmm_rows.append(row)
    print(f"  {task}   {auc_o:.4f}    {auc_g:.4f}   "
          f"{delta:+.4f}  {z:7.2f}  {p:.6f}  {sig_stars(p)}")

pd.DataFrame(delong_gmm_rows).to_csv(
    os.path.join(OUT, "NB10c_delong_ocsvm_vs_gmm.csv"), index=False)
print("  Saved: NB10c_delong_ocsvm_vs_gmm.csv")

# --- PSR_OCSVM vs Raw_ZScore ---
delong_raw_rows = []
print(f"\n  PSR_OCSVM vs Raw_ZScore:")
print(f"  {'Task':3}  {'AUC_OCSVM':>10}  {'AUC_RAW':>8}  "
      f"{'delta':>8}  {'z':>7}  {'p':>10}  sig")
for task in TASKS:
    yt_o, ys_o = psr_ocsvm_scores[task]
    yt_r, ys_r = raw_zscore_scores[task]
    assert np.array_equal(yt_o, yt_r)
    auc_o, auc_r, delta, z, p = delong_paired(yt_o, ys_o, ys_r)
    row = delong_row(task, auc_o, auc_r, delta, z, p, "Raw_ZScore")
    delong_raw_rows.append(row)
    print(f"  {task}   {auc_o:.4f}    {auc_r:.4f}   "
          f"{delta:+.4f}  {z:7.2f}  {p:.6f}  {sig_stars(p)}")

pd.DataFrame(delong_raw_rows).to_csv(
    os.path.join(OUT, "NB10c_delong_ocsvm_vs_rawzscore.csv"), index=False)
print("  Saved: NB10c_delong_ocsvm_vs_rawzscore.csv")


# STEP 9: Comparison table
print("\n[Step 9] Assembling full comparison table...")

nb10_agg  = pd.read_csv(os.path.join(OUT, "NB10_bootstrap_ci_aggregate.csv"))
nb10b_agg = pd.read_csv(os.path.join(OUT, "NB10b_convae_auroc_aggregate.csv"))

table_rows = []

# PSR OC-SVM (proposed) -- from patched NB10
for _, row in nb10_agg[nb10_agg["detector"] == "PSR_OCSVM"].iterrows():
    table_rows.append({
        "Method": "PSR OC-SVM (proposed)",
        "Task": row["test_task"], "AUROC": row["auroc"],
        "95pct_BCa_CI": f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
        "Physics": "Yes"})

# GMM (PSR features) -- NB10c
for r in gmm_agg_rows:
    table_rows.append({
        "Method": "GMM (PSR features)",
        "Task": r["test_task"], "AUROC": r["auroc"],
        "95pct_BCa_CI": f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
        "Physics": "Yes"})

# Conv-AE (raw current) -- NBb
for _, row in nb10b_agg.iterrows():
    table_rows.append({
        "Method": "Conv-AE (raw current)",
        "Task": row["test_task"], "AUROC": row["auroc"],
        "95pct_BCa_CI": f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
        "Physics": "No"})

# LSTM-VAE (raw current) -- NB10c
for r in lstmvae_agg_rows:
    table_rows.append({
        "Method": "LSTM-VAE (raw current)",
        "Task": r["test_task"], "AUROC": r["auroc"],
        "95pct_BCa_CI": f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
        "Physics": "No"})

# Raw Z-Score -- from patched NB10
for _, row in nb10_agg[nb10_agg["detector"] == "Raw_ZScore"].iterrows():
    table_rows.append({
        "Method": "Raw Z-Score (no physics)",
        "Task": row["test_task"], "AUROC": row["auroc"],
        "95pct_BCa_CI": f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
        "Physics": "No"})

comp_df = pd.DataFrame(table_rows)

# Pivot to wide format
comp_wide = comp_df.pivot_table(
    index="Method", columns="Task", values="AUROC",
    aggfunc="first").reset_index()
comp_wide["Mean_AUROC"] = comp_wide[TASKS].mean(axis=1).round(4)
comp_wide = comp_wide.sort_values("Mean_AUROC", ascending=False)

comp_df.to_csv(os.path.join(OUT, "NB10c_comparison_table_full.csv"), index=False)
print("  Saved: NB10c_comparison_table_full.csv")

print("\n" + "=" * 70)
print("FULL COMPARISON TABLE")
print("=" * 70)
print(f"{'Method':<30} {'T1':>8} {'T2':>8} {'T3':>8} {'Mean':>8}")
print("-" * 70)
for _, row in comp_wide.iterrows():
    print(f"  {row['Method']:<28} "
          f"{row.get('T1', np.nan):>8.4f} "
          f"{row.get('T2', np.nan):>8.4f} "
          f"{row.get('T3', np.nan):>8.4f} "
          f"{row['Mean_AUROC']:>8.4f}")
print("=" * 70)

print("\nNBc complete.")
print(f"Outputs written to: {OUT}")
print("Files produced (v2):")
for fn in [
    "csv",
    ]:
    fpath = os.path.join(OUT, fn)
    if os.path.exists(fpath):
        print(f"  OK  {fn}")
    else:
        print(f"  WRITTEN  {fn}")
