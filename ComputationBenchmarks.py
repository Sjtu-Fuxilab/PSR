import os, glob, pickle, time, sys, warnings
import numpy as np
import pandas as pd
import h5py
import scipy.stats as sst
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
np.random.seed(42)

# PyTorch import (required for LSTM-VAE)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} — "
          f"{'GPU' if torch.cuda.is_available() else 'CPU'}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("ERROR: PyTorch not found.  Install: pip install torch")
    sys.exit(1)

# PATHS
ROOT    = r"D:\Research\R"
BASE    = os.path.join(ROOT, "L_Data")
OUT     = os.path.join(ROOT, "P_Data")
NB10B   = os.path.join(OUT, "NB.csv")

SCORES_PKL_PSR  = os.path.join(OUT, "NB10e_psr_gmm_raw_scores.pkl")
SCORES_PKL_LSTM = os.path.join(OUT, "NB10e_lstmvae_scores.pkl")

OUT_AUROC   = os.path.join(OUT, "NB10e_per_anomaly_auroc.csv")
OUT_DELONG  = os.path.join(OUT, "NB10e_delong_per_anomaly.csv")
OUT_TIMING  = os.path.join(OUT, "NB10e_timing.csv")

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

# LSTM-VAE hyper-parameters
LSTMVAE_EPOCHS   = 80
LSTMVAE_BATCH    = 32
LSTMVAE_LR       = 1e-3
LSTMVAE_HIDDEN   = 64
LSTMVAE_LATENT   = 32
LSTMVAE_N_LAYERS = 2
LSTMVAE_BETA     = 1.0

# GMM
GMM_MAX_COMP = 8
GMM_MAX_ITER = 500
GMM_N_INIT   = 5

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

# FEATURE COLUMN NAMES
PSR_COLS = ([f"J{j}_{s}" for j in range(6)
             for s in ["resid_mean", "resid_std", "resid_rms", "resid_max",
                       "resid_skew", "resid_kurtosis",
                       "grav_resid_std", "grav_resid_rms"]]
            + ["total_resid_rms", "J1J2_resid_corr"])    # 50-dim

RAW_COLS = ([f"J{j}_{s}" for j in range(6)
             for s in ["raw_mean", "raw_std", "raw_rms"]]
            + ["total_raw_rms"])                          # 19-dim

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

# PHYSICS FUNCTIONS

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


def fit_psr_fold(train_cycles):
    """
    OLS fit of M4 PSR weights for one LOTO fold.
    Uses per-cycle TASK_PAYLOAD[cyc["task"]] — identical to NB10_statistical_tests.py.
    Returns psr_w: list of 6 weight vectors [tau_g, qd, sign(qd), qdd, 1].
    """
    rows = {j: [] for j in range(6)}
    for cyc in train_cycles:
        payload = TASK_PAYLOAD[cyc["task"]]
        q_a, qd_a, cur = cyc["q"], cyc["qd"], cyc["current"]
        N = len(q_a)
        for t in range(0, N, SUBSAMPLE):
            tau_g = gravity_torque(q_a[t], payload_mass=payload)
            for j in range(6):
                qdd_j = ((qd_a[t+1,j] - qd_a[t-1,j]) * RATE / 2.0
                         if 0 < t < N-1 else 0.0)
                phi = [tau_g[j], qd_a[t,j], np.sign(qd_a[t,j]), qdd_j, 1.0]
                rows[j].append(phi + [cur[t, j]])
    psr_w = []
    for j in range(6):
        A = np.array(rows[j])
        X, y = A[:, :5], A[:, 5]
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        psr_w.append(w)
    return psr_w


def extract_psr(cyc, psr_w):
    """50-dim PSR feature vector for one cycle (identical to NB10_statistical_tests)."""
    payload = TASK_PAYLOAD[cyc["task"]]
    q_a, qd_a, cur = cyc["q"], cyc["qd"], cyc["current"]
    N = len(q_a)
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
    """19-dim raw current feature vector for one cycle."""
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


# DETECTOR FUNCTIONS

def score_zscore(Xtr, Xte):
    mu = Xtr.mean(0); sg = Xtr.std(0) + 1e-8
    return np.abs((Xte - mu) / sg).mean(1)


def score_ocsvm(Xtr, Xte):
    sc  = StandardScaler().fit(Xtr)
    clf = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    clf.fit(sc.transform(Xtr))
    return -clf.decision_function(sc.transform(Xte))


def score_isoforest(Xtr, Xte):
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    clf.fit(Xtr)
    return -clf.decision_function(Xte)


def score_gmm(Xtr, Xte):
    sc  = StandardScaler().fit(Xtr)
    bic = [GaussianMixture(n_components=k, covariance_type="full",
                           max_iter=GMM_MAX_ITER, n_init=GMM_N_INIT,
                           random_state=42).fit(sc.transform(Xtr)).bic(sc.transform(Xtr))
           for k in range(1, GMM_MAX_COMP + 1)]
    best_k = np.argmin(bic) + 1
    gmm = GaussianMixture(n_components=best_k, covariance_type="full",
                          max_iter=GMM_MAX_ITER, n_init=GMM_N_INIT,
                          random_state=42)
    gmm.fit(sc.transform(Xtr))
    return -gmm.score_samples(sc.transform(Xte))


# LSTM-VAE
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
        self.n_layers = n_layers
        self.fc_init  = nn.Linear(latent_dim, hidden_dim)
        self.lstm     = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim,
                                num_layers=n_layers, batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_channels)

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


def cycle_to_sequence(cyc, fixed_len):
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


def train_lstmvae(train_seqs, device, fixed_len):
    """Train LSTM-VAE on a list of (fixed_len, 6) sequences. Returns model."""
    X = torch.tensor(np.stack(train_seqs), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X),
        batch_size=LSTMVAE_BATCH, shuffle=True)
    model = LSTMVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LSTMVAE_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    model.train()
    for ep in range(LSTMVAE_EPOCHS):
        for (xb,) in loader:
            xb = xb.to(device)
            loss, _ = model.elbo_loss(xb)
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
    return model


def score_lstmvae(model, seqs, device):
    """Reconstruction scores for a list of (fixed_len, 6) sequences."""
    X = torch.tensor(np.stack(seqs), dtype=torch.float32).to(device)
    return model.reconstruction_score(X)


# STATISTICAL FUNCTIONS

def bootstrap_auroc_bca(y_true, y_score, n_boot=N_BOOT, rng=None):
    """BCa bootstrap 95% CI for AUROC (identical to NB10 / NB10d)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y_true)
    auroc_obs = roc_auc_score(y_true, y_score)
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; ys = y_score[idx]
        boot[b] = roc_auc_score(yt, ys) if (0 < yt.sum() < n) else auroc_obs
    prop = np.clip(np.mean(boot < auroc_obs), 1e-6, 1 - 1e-6)
    z0   = sst.norm.ppf(prop)
    jack = np.zeros(n)
    for i in range(n):
        idx_j = np.concatenate([np.arange(i), np.arange(i+1, n)])
        yt_j  = y_true[idx_j]; ys_j = y_score[idx_j]
        jack[i] = (roc_auc_score(yt_j, ys_j)
                   if (0 < yt_j.sum() < len(yt_j)) else auroc_obs)
    jm  = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a   = num / den if den != 0 else 0.0
    ci  = {}
    for label, z_a in [("lo", sst.norm.ppf(0.025)),
                        ("hi", sst.norm.ppf(0.975))]:
        p = sst.norm.cdf(z0 + (z0 + z_a) / (1 - a * (z0 + z_a)))
        ci[label] = float(np.quantile(boot, np.clip(p, 0.001, 0.999)))
    return float(auroc_obs), ci["lo"], ci["hi"]


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
    var_diff = S[0,0] + S[1,1] - 2*S[0,1]
    if var_diff <= 0:
        return auc_a, auc_b, delta, np.nan, np.nan
    z = delta / np.sqrt(var_diff)
    p = 2 * (1 - sst.norm.cdf(abs(z)))
    return auc_a, auc_b, delta, z, p


def sig_stars(p):
    if np.isnan(p):  return "n.d."
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return "ns"


# STEP 1 — Load data
print("=" * 65)
print("NBe — Per-Anomaly AUROC Comparison and Computation Benchmarks")
print("=" * 65)
print("\n[Step 1] Loading HDF5 data...")

all_cycles = []
for key, (subdir, pattern, task, anomaly, severity) in REGISTRY.items():
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
print(f"  Total cycles loaded: {len(all_cycles)} | Healthy: {len(healthy_cycles)}")
for t in TASKS:
    nh = sum(1 for c in healthy_cycles if c["task"] == t)
    na = sum(1 for c in all_cycles if c["task"] == t and c["is_anomaly"] == 1)
    print(f"    {t}: {nh} healthy, {na} anomaly")

# STEP 2 — LSTM-VAE sequence length
print("\n[Step 2] Determining FIXED_LEN for LSTM-VAE sequences...")
sub_lengths = np.array([len(range(0, len(c["current"]), SUBSAMPLE))
                        for c in all_cycles])
p5        = int(np.percentile(sub_lengths, 5))
FIXED_LEN = max(p5 - (p5 % 4), 64)
print(f"  Subsampled lengths: min={sub_lengths.min()}  "
      f"p5={p5}  median={int(np.median(sub_lengths))}  max={sub_lengths.max()}")
print(f"  FIXED_LEN = {FIXED_LEN}")

# STEP 3 — PSR / GMM / Raw Z-Score LOTO loop (per-anomaly scores)
# CHECKPOINT: load partial results if they exist.
if os.path.exists(SCORES_PKL_PSR):
    with open(SCORES_PKL_PSR, "rb") as f:
        psr_scores = pickle.load(f)
    completed_folds_psr = set(psr_scores.keys())
    print(f"\n[Step 3] Loaded PSR/GMM/Raw checkpoint: "
          f"{sorted(completed_folds_psr)} already done.")
else:
    psr_scores = {}
    completed_folds_psr = set()
    print("\n[Step 3] PSR/GMM/Raw LOTO loop (per-anomaly, strict LOTO)...")

for test_task in TASKS:
    if test_task in completed_folds_psr:
        print(f"  {test_task}: skipping (checkpoint)")
        continue

    print(f"\n  ---- Fold: test={test_task} ----")
    tr_healthy = [c for c in healthy_cycles if c["task"] != test_task]
    te_cycles  = [c for c in all_cycles    if c["task"] == test_task]
    te_h_idx   = [i for i, c in enumerate(te_cycles) if c["is_anomaly"] == 0]

    print(f"    Train healthy: {len(tr_healthy)}  "
          f"Test (healthy+anom): {len(te_cycles)}")

    # PSR weights for this fold
    t0 = time.perf_counter()
    psr_w = fit_psr_fold(tr_healthy)
    psr_fit_time = time.perf_counter() - t0
    print(f"    PSR fit: {psr_fit_time:.1f}s", end="  ")

    # Extract PSR features
    t0 = time.perf_counter()
    Xtr_psr = np.array([extract_psr(c, psr_w) for c in tr_healthy])
    Xte_psr = np.array([extract_psr(c, psr_w) for c in te_cycles])
    psr_feat_time = time.perf_counter() - t0
    print(f"PSR features: {psr_feat_time:.1f}s", end="  ")

    # Extract raw features
    t0 = time.perf_counter()
    Xtr_raw = np.array([extract_raw(c) for c in tr_healthy])
    Xte_raw = np.array([extract_raw(c) for c in te_cycles])
    raw_feat_time = time.perf_counter() - t0
    print(f"Raw features: {raw_feat_time:.1f}s")

    # Detector training times
    t0 = time.perf_counter()
    _mu_zs = Xtr_psr.mean(0); _sg_zs = Xtr_psr.std(0) + 1e-8
    zs_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _sc_oc = StandardScaler().fit(Xtr_psr)
    _clf_oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    _clf_oc.fit(_sc_oc.transform(Xtr_psr))
    ocsvm_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _clf_if = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    _clf_if.fit(Xtr_psr)
    iso_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _sc_gm = StandardScaler().fit(Xtr_psr)
    _bic = [GaussianMixture(n_components=k, covariance_type="full",
                            max_iter=GMM_MAX_ITER, n_init=GMM_N_INIT,
                            random_state=42).fit(_sc_gm.transform(Xtr_psr)).bic(
                                _sc_gm.transform(Xtr_psr))
            for k in range(1, GMM_MAX_COMP + 1)]
    _best_k = np.argmin(_bic) + 1
    _clf_gm = GaussianMixture(n_components=_best_k, covariance_type="full",
                              max_iter=GMM_MAX_ITER, n_init=GMM_N_INIT,
                              random_state=42)
    _clf_gm.fit(_sc_gm.transform(Xtr_psr))
    gmm_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _mu_rz = Xtr_raw.mean(0); _sg_rz = Xtr_raw.std(0) + 1e-8
    raw_train_time = time.perf_counter() - t0

    print(f"    Detector train times — ZScore:{zs_train_time:.3f}s  "
          f"OC-SVM:{ocsvm_train_time:.2f}s  IsoForest:{iso_train_time:.2f}s  "
          f"GMM:{gmm_train_time:.2f}s  RawZS:{raw_train_time:.3f}s")

    # Score all test cycles
    t_inf0 = time.perf_counter()
    sc_zscore    = score_zscore(Xtr_psr, Xte_psr)
    sc_ocsvm     = score_ocsvm(Xtr_psr, Xte_psr)
    sc_isoforest = score_isoforest(Xtr_psr, Xte_psr)
    sc_gmm       = score_gmm(Xtr_psr, Xte_psr)
    sc_rawzs     = score_zscore(Xtr_raw, Xte_raw)
    infer_time   = (time.perf_counter() - t_inf0) / len(te_cycles) * 1000  # ms/cycle

    print(f"    Inference: {infer_time:.1f} ms/cycle (avg over {len(te_cycles)} cycles)")

    # Per-anomaly score dicts
    fold_data = {
        "psr_w":          psr_w,
        "Xtr_psr":        Xtr_psr,
        "Xtr_raw":        Xtr_raw,
        "te_cycles":      te_cycles,
        "te_h_idx":       te_h_idx,
        "scores": {
            "PSR_ZScore":    sc_zscore,
            "PSR_OC-SVM":   sc_ocsvm,
            "PSR_IsoForest": sc_isoforest,
            "GMM":           sc_gmm,
            "Raw_ZScore":    sc_rawzs,
        },
        "train_times": {
            "PSR_ZScore":    psr_fit_time + psr_feat_time + zs_train_time,
            "PSR_OC-SVM":   psr_fit_time + psr_feat_time + ocsvm_train_time,
            "PSR_IsoForest": psr_fit_time + psr_feat_time + iso_train_time,
            "GMM":           psr_fit_time + psr_feat_time + gmm_train_time,
            "Raw_ZScore":    raw_feat_time + raw_train_time,
        },
        "infer_ms_per_cycle": infer_time,
    }
    psr_scores[test_task] = fold_data

    with open(SCORES_PKL_PSR, "wb") as f:
        pickle.dump(psr_scores, f)
    print(f"    Checkpoint saved: {SCORES_PKL_PSR}")

# STEP 4 — LSTM-VAE LOTO loop (per-anomaly scores)
if os.path.exists(SCORES_PKL_LSTM):
    with open(SCORES_PKL_LSTM, "rb") as f:
        lstm_scores = pickle.load(f)
    completed_folds_lstm = set(lstm_scores.keys())
    print(f"\n[Step 4] Loaded LSTM-VAE checkpoint: "
          f"{sorted(completed_folds_lstm)} already done.")
else:
    lstm_scores = {}
    completed_folds_lstm = set()
    print("\n[Step 4] LSTM-VAE LOTO loop (per-anomaly, strict LOTO)...")
    print(f"  Device: {DEVICE}  | Epochs: {LSTMVAE_EPOCHS} | FIXED_LEN: {FIXED_LEN}")

for test_task in TASKS:
    if test_task in completed_folds_lstm:
        print(f"  {test_task}: skipping (checkpoint)")
        continue

    print(f"\n  ---- Fold: test={test_task} ----")
    tr_healthy = [c for c in healthy_cycles if c["task"] != test_task]
    te_cycles  = [c for c in all_cycles    if c["task"] == test_task]

    train_seqs = [cycle_to_sequence(c, FIXED_LEN) for c in tr_healthy]
    test_seqs  = [cycle_to_sequence(c, FIXED_LEN) for c in te_cycles]

    print(f"    Training LSTM-VAE on {len(train_seqs)} sequences...")
    t0 = time.perf_counter()
    model = train_lstmvae(train_seqs, DEVICE, FIXED_LEN)
    lstm_train_time = time.perf_counter() - t0
    print(f"    LSTM-VAE train: {lstm_train_time:.1f}s")

    t0 = time.perf_counter()
    sc_lstm = score_lstmvae(model, test_seqs, DEVICE)
    lstm_infer_ms = (time.perf_counter() - t0) / len(te_cycles) * 1000

    # Model size in bytes (state dict)
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    lstm_model_bytes = buf.tell()

    lstm_scores[test_task] = {
        "te_cycles":          te_cycles,
        "scores":             sc_lstm,
        "train_time_s":       lstm_train_time,
        "infer_ms_per_cycle": lstm_infer_ms,
        "model_bytes":        lstm_model_bytes,
    }
    with open(SCORES_PKL_LSTM, "wb") as f:
        pickle.dump(lstm_scores, f)
    print(f"    LSTM-VAE scores: min={sc_lstm.min():.4f}  max={sc_lstm.max():.4f}")
    print(f"    Checkpoint saved: {SCORES_PKL_LSTM}")

# STEP 5 — BCa CIs and per-anomaly AUROC table
print(f"\n[Step 5] Bootstrap BCa CIs (N_BOOT={N_BOOT})...")
rng = np.random.default_rng(42)

PHYSICS_FLAG = {
    "PSR_ZScore":     "Yes",
    "PSR_OC-SVM":    "Yes",
    "PSR_IsoForest":  "Yes",
    "GMM":            "Yes",
    "Conv-AE":        "No",
    "LSTM-VAE":       "No",
    "Raw_ZScore":     "No",
}

auroc_rows = []

# --- PSR / GMM / Raw methods ---
for test_task in TASKS:
    fold = psr_scores[test_task]
    te_cycles = fold["te_cycles"]
    te_h_idx  = fold["te_h_idx"]
    y_all     = np.array([c["is_anomaly"] for c in te_cycles])

    for anom in ["A2", "A3", "A5"]:
        anom_idx = [i for i, c in enumerate(te_cycles) if c["anomaly"] == anom]
        if not anom_idx:
            continue
        comb_idx = te_h_idx + anom_idx
        y_anom   = np.array([te_cycles[i]["is_anomaly"] for i in comb_idx])
        n_h      = int((y_anom == 0).sum())
        n_a      = int((y_anom == 1).sum())

        for method, sc_all in fold["scores"].items():
            sc_anom = sc_all[comb_idx]
            auroc, lo, hi = bootstrap_auroc_bca(y_anom, sc_anom, rng=rng)
            auroc_rows.append(dict(
                test_task=test_task, anomaly_type=anom, method=method,
                physics=PHYSICS_FLAG[method],
                n_healthy=n_h, n_anomaly=n_a,
                auroc=round(auroc, 4),
                ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                ci_width=round(hi - lo, 4)))
            print(f"  {test_task} | {anom} | {method:15s} | "
                  f"AUROC={auroc:.4f} [{lo:.4f}, {hi:.4f}]")

# --- LSTM-VAE ---
for test_task in TASKS:
    fold = lstm_scores[test_task]
    te_cycles = fold["te_cycles"]
    te_h_idx  = [i for i, c in enumerate(te_cycles) if c["is_anomaly"] == 0]
    sc_all    = fold["scores"]

    for anom in ["A2", "A3", "A5"]:
        anom_idx = [i for i, c in enumerate(te_cycles) if c["anomaly"] == anom]
        if not anom_idx:
            continue
        comb_idx = te_h_idx + anom_idx
        y_anom   = np.array([te_cycles[i]["is_anomaly"] for i in comb_idx])
        sc_anom  = sc_all[comb_idx]
        n_h      = int((y_anom == 0).sum())
        n_a      = int((y_anom == 1).sum())
        auroc, lo, hi = bootstrap_auroc_bca(y_anom, sc_anom, rng=rng)
        auroc_rows.append(dict(
            test_task=test_task, anomaly_type=anom, method="LSTM-VAE",
            physics="No",
            n_healthy=n_h, n_anomaly=n_a,
            auroc=round(auroc, 4),
            ci_lo=round(lo, 4), ci_hi=round(hi, 4),
            ci_width=round(hi - lo, 4)))
        print(f"  {test_task} | {anom} | {'LSTM-VAE':15s} | "
              f"AUROC={auroc:.4f} [{lo:.4f}, {hi:.4f}]")

# --- Conv-AE: load from NB10b (already valid under strict LOTO) ---
if os.path.exists(NB10B):
    convae_df = pd.read_csv(NB10B)
    print(f"\n  Loading Conv-AE per-anomaly from NB10b ({len(convae_df)} rows)...")
    for _, row in convae_df.iterrows():
        auroc_rows.append(dict(
            test_task=row["test_task"],
            anomaly_type=row["anomaly_type"],
            method="Conv-AE",
            physics="No",
            n_healthy=int(row["n_healthy"]),
            n_anomaly=int(row["n_anomaly"]),
            auroc=round(float(row["auroc"]), 4),
            ci_lo=round(float(row["ci_lo"]), 4),
            ci_hi=round(float(row["ci_hi"]), 4),
            ci_width=round(float(row["ci_hi"]) - float(row["ci_lo"]), 4)))
        print(f"  {row['test_task']} | {row['anomaly_type']} | {'Conv-AE':15s} | "
              f"AUROC={row['auroc']:.4f}")
else:
    print(f"\n  WARNING: {NB10B} not found — Conv-AE rows not included.")

# Save
df_auroc = pd.DataFrame(auroc_rows).sort_values(
    ["anomaly_type", "test_task", "method"]).reset_index(drop=True)
df_auroc.to_csv(OUT_AUROC, index=False)
print(f"\n  Saved: NB10e_per_anomaly_auroc.csv  ({len(df_auroc)} rows)")

# STEP 6 — DeLong tests per (task, anomaly): PSR_ZScore vs all baselines
print("\n[Step 6] DeLong tests per (task, anomaly)...")

BASELINE_METHODS = ["PSR_OC-SVM", "PSR_IsoForest", "GMM",
                    "Conv-AE", "LSTM-VAE", "Raw_ZScore"]
delong_rows = []

# Build per-anomaly score lookup including Conv-AE
# (PSR/GMM/Raw from psr_scores; LSTM-VAE from lstm_scores; Conv-AE needs raw scores)
# Note: Conv-AE raw scores not available here; we skip DeLong for Conv-AE
# (only BCa CIs). This matches NB10 where Conv-AE was handled separately.

for test_task in TASKS:
    fold_psr  = psr_scores[test_task]
    fold_lstm = lstm_scores[test_task]
    te_cycles = fold_psr["te_cycles"]
    te_h_idx  = fold_psr["te_h_idx"]

    for anom in ["A2", "A3", "A5"]:
        anom_idx = [i for i, c in enumerate(te_cycles) if c["anomaly"] == anom]
        if not anom_idx:
            continue
        comb_idx = te_h_idx + anom_idx
        y_anom   = np.array([te_cycles[i]["is_anomaly"] for i in comb_idx])

        sc_a = fold_psr["scores"]["PSR_ZScore"][comb_idx]

        # PSR vs other PSR/GMM/Raw
        for method_b in ["PSR_OC-SVM", "PSR_IsoForest", "GMM", "Raw_ZScore"]:
            sc_b = fold_psr["scores"][method_b][comb_idx]
            auc_a, auc_b, delta, z, p = delong_paired(y_anom, sc_a, sc_b)
            delong_rows.append(dict(
                test_task=test_task, anomaly_type=anom,
                method_a="PSR_ZScore", method_b=method_b,
                auc_a=round(auc_a, 4), auc_b=round(auc_b, 4),
                delta=round(delta, 4), z=round(z, 3) if not np.isnan(z) else np.nan,
                p=round(p, 4) if not np.isnan(p) else np.nan,
                sig=sig_stars(p)))

        # PSR vs LSTM-VAE
        # Confirm te_cycles ordering matches — LSTM uses same te_cycles
        lstm_te = fold_lstm["te_cycles"]
        # Sanity: same cycles (check anomaly labels match)
        if len(lstm_te) == len(te_cycles):
            sc_b_lstm = fold_lstm["scores"][comb_idx]
            auc_a, auc_b, delta, z, p = delong_paired(y_anom, sc_a, sc_b_lstm)
            delong_rows.append(dict(
                test_task=test_task, anomaly_type=anom,
                method_a="PSR_ZScore", method_b="LSTM-VAE",
                auc_a=round(auc_a, 4), auc_b=round(auc_b, 4),
                delta=round(delta, 4), z=round(z, 3) if not np.isnan(z) else np.nan,
                p=round(p, 4) if not np.isnan(p) else np.nan,
                sig=sig_stars(p)))

df_delong = pd.DataFrame(delong_rows).sort_values(
    ["anomaly_type", "test_task", "method_b"]).reset_index(drop=True)
df_delong.to_csv(OUT_DELONG, index=False)
print(f"  Saved: NB10e_delong_per_anomaly.csv  ({len(df_delong)} rows)")

# STEP 7 — Timing benchmarks
print("\n[Step 7] Computation benchmarks...")

# Average train time across folds for PSR-based methods
import sys as _sys

timing_rows = []

# PSR-based methods: aggregate train time from Step 3
for method in ["PSR_ZScore", "PSR_OC-SVM", "PSR_IsoForest", "GMM", "Raw_ZScore"]:
    train_times = [psr_scores[t]["train_times"][method] for t in TASKS]
    infer_times = [psr_scores[t]["infer_ms_per_cycle"] for t in TASKS]
    # Model size: tiny for all PSR methods (weights only)
    # Rough estimate: PSR weights = 6 joints × 5 floats × 8 bytes + scaler
    if method == "PSR_ZScore":
        model_bytes = 6 * 5 * 8 + 50 * 8 * 2  # PSR weights + mu/sigma (50-dim)
    elif method == "PSR_OC-SVM":
        model_bytes = 6 * 5 * 8 + 100 * 50 * 8   # PSR weights + support vectors (rough)
    elif method == "PSR_IsoForest":
        model_bytes = 6 * 5 * 8 + 200 * 50 * 8   # PSR weights + 200 trees (rough)
    elif method == "GMM":
        model_bytes = 6 * 5 * 8 + 8 * 50 * 50 * 8  # PSR weights + GMM covariance matrices
    else:  # Raw_ZScore
        model_bytes = 19 * 8 * 2   # mu/sigma for 19-dim raw features

    timing_rows.append(dict(
        method=method,
        physics=PHYSICS_FLAG[method],
        device="CPU",
        train_time_s=round(np.mean(train_times), 2),
        infer_ms_per_cycle=round(np.mean(infer_times), 3),
        model_bytes=model_bytes))
    print(f"  {method:15s} | train={np.mean(train_times):.1f}s  "
          f"infer={np.mean(infer_times):.1f}ms/cyc  "
          f"size={model_bytes} B")

# LSTM-VAE
lstm_train_times = [lstm_scores[t]["train_time_s"] for t in TASKS]
lstm_infer_times = [lstm_scores[t]["infer_ms_per_cycle"] for t in TASKS]
lstm_model_bytes = [lstm_scores[t]["model_bytes"] for t in TASKS]
timing_rows.append(dict(
    method="LSTM-VAE",
    physics="No",
    device=str(DEVICE),
    train_time_s=round(np.mean(lstm_train_times), 2),
    infer_ms_per_cycle=round(np.mean(lstm_infer_times), 3),
    model_bytes=int(np.mean(lstm_model_bytes))))
print(f"  {'LSTM-VAE':15s} | train={np.mean(lstm_train_times):.1f}s  "
      f"infer={np.mean(lstm_infer_times):.1f}ms/cyc  "
      f"size={int(np.mean(lstm_model_bytes))} B")

convae_bench = os.path.join(OUT, "NB.csv")
_convae_timing = dict(method="Conv-AE", physics="No",
                      device="see NB10b", train_time_s=None,
                      infer_ms_per_cycle=None, model_bytes=None)
if os.path.exists(convae_bench):
    try:
        bench_df = pd.read_csv(convae_bench)
        # Try common column name variants for the method identifier
        method_col = None
        for candidate in ["method", "Method", "detector", "Detector", "name"]:
            if candidate in bench_df.columns:
                method_col = candidate
                break
        if method_col is not None:
            mask = bench_df[method_col].astype(str).str.contains(
                "Conv|convae", case=False, na=False)
            convae_row = bench_df[mask]
            if not convae_row.empty:
                r = convae_row.iloc[0].to_dict()
                # Map common column name variants to expected keys
                def _get(row, *keys, default=None):
                    for k in keys:
                        if k in row and pd.notna(row[k]):
                            return row[k]
                    return default
                _convae_timing = dict(
                    method="Conv-AE",
                    physics="No",
                    device=str(_get(r, "device", "Device", default="CPU")),
                    train_time_s=round(float(_get(r, "train_time_s", "train_s",
                                                  "training_s", default=0)), 2),
                    infer_ms_per_cycle=round(float(_get(r, "infer_ms_per_cycle",
                                                        "infer_ms", "inference_ms",
                                                        default=0)), 3),
                    model_bytes=int(_get(r, "model_bytes", "model_size_bytes",
                                         default=0)))
    except Exception as e:
        print(f"  NOTE: Could not parse NB10_compute_benchmark.csv ({e}); "
              f"Conv-AE timing set to None.")
timing_rows.append(_convae_timing)
print(f"  {'Conv-AE':15s} | train={_convae_timing['train_time_s']}s  "
      f"infer={_convae_timing['infer_ms_per_cycle']}ms/cyc  "
      f"size={_convae_timing['model_bytes']} B")

df_timing = pd.DataFrame(timing_rows)
df_timing.to_csv(OUT_TIMING, index=False)
print(f"\n  Saved: NB10e_timing.csv  ({len(df_timing)} rows)")

# STEP 8 — Summary tables
print("\n" + "=" * 65)
print("NBe SUMMARY — Per-Anomaly AUROC (mean across tasks per anomaly)")
print("=" * 65)

METHOD_ORDER = ["PSR_ZScore", "PSR_OC-SVM", "PSR_IsoForest", "GMM",
                "Conv-AE", "LSTM-VAE", "Raw_ZScore"]

print(f"\n{'Method':18} {'Phys':4} | "
      f"{'A2 T1':8} {'A2 T2':8} {'A2 T3':8} | "
      f"{'A3 T1':8} {'A3 T2':8} {'A3 T3':8} | "
      f"{'A5 T1':8} {'A5 T2':8} {'A5 T3':8} | "
      f"{'Mean':8}")

for method in METHOD_ORDER:
    sub = df_auroc[df_auroc["method"] == method]
    vals = []
    row_str = f"{method:18} {PHYSICS_FLAG.get(method,'?'):4} | "
    for anom in ["A2", "A3", "A5"]:
        for task in TASKS:
            r = sub[(sub["test_task"] == task) & (sub["anomaly_type"] == anom)]
            if r.empty:
                row_str += f"{'N/A':8} "
                vals.append(np.nan)
            else:
                v = float(r["auroc"].iloc[0])
                row_str += f"{v:8.4f} "
                vals.append(v)
        row_str += "| "
    mean_v = np.nanmean(vals)
    row_str += f"{mean_v:8.4f}"
    print(row_str)

print(f"\n{'':18} {'':4} | " + "".join(
    [f"{'T1':8} {'T2':8} {'T3':8} | " for _ in ["A2","A3","A5"]]) +
    f"{'All':8}")

print("\nDeLong summary (PSR_ZScore vs baselines, aggregate across folds):")
print(f"{'Method B':20} {'sig@A2':10} {'sig@A3':10} {'sig@A5':10}")
for mb in ["PSR_OC-SVM", "PSR_IsoForest", "GMM", "LSTM-VAE", "Raw_ZScore"]:
    sigs = []
    for anom in ["A2", "A3", "A5"]:
        sub_d = df_delong[(df_delong["method_b"] == mb) &
                          (df_delong["anomaly_type"] == anom)]
        p_vals = sub_d["p"].dropna().values
        # Combined Fisher p (product of test tasks)
        if len(p_vals) > 0:
            chi2 = -2 * np.sum(np.log(np.clip(p_vals, 1e-300, 1)))
            p_combined = 1 - sst.chi2.cdf(chi2, df=2 * len(p_vals))
            sigs.append(sig_stars(p_combined))
        else:
            sigs.append("—")
    print(f"  {mb:20} {sigs[0]:10} {sigs[1]:10} {sigs[2]:10}")

print("\n" + "=" * 65)
print("NBe COMPLETE")
