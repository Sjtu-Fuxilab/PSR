# T4_baselines.py
# Data-driven baselines at T4: Conv-AE, LSTM-VAE, Raw Z-Score.
# FIXED_LEN computed over the full 4-task pool (matches manuscript).

# %% Cell 1: Configuration

import os, glob, time, warnings, pickle
import numpy as np
import pandas as pd
import h5py
import scipy.stats as sst
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42)

ROOT = r"D:\Research\R"
BASE = os.path.join(ROOT, "L_Data")
OUT  = os.path.join(ROOT, "P_Data")
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

RATE      = 125
SUBSAMPLE = 4
MIN_SAMP  = 200
N_BOOT    = 10000
TASKS     = ["T1", "T2", "T3", "T4"]

CONVAE_EPOCHS    = 60
CONVAE_BATCH     = 32
CONVAE_LR        = 1e-3
LSTMVAE_EPOCHS   = 80
LSTMVAE_BATCH    = 32
LSTMVAE_LR       = 1e-3
LSTMVAE_HIDDEN   = 64
LSTMVAE_LATENT   = 32
LSTMVAE_N_LAYERS = 2
LSTMVAE_BETA     = 1.0

REGISTRY = {
    "T1_healthy":    ("T1_PickPlace/Healthy",   "UR5_T1_healthy_180cyc_*.h5",         "T1","healthy"),
    "T1_A2_0.5kg":   ("T1_PickPlace/A2",        "UR5_T1_A2_0.5kg_gripper_40cyc_*.h5", "T1","A2"),
    "T1_A2_1kg":     ("T1_PickPlace/A2",        "UR5_T1_A2_1kg_gripper_40cyc_*.h5",   "T1","A2"),
    "T1_A2_2kg":     ("T1_PickPlace/A2",        "UR5_T1_A2_2kg_gripper_40cyc_*.h5",   "T1","A2"),
    "T1_A3_10wraps": ("T1_PickPlace/A3",        "UR5_T1_A3_1band_40cyc_*.h5",         "T1","A3"),
    "T1_A3_17wraps": ("T1_PickPlace/A3",        "UR5_T1_A3_3bands_40cyc_*.h5",        "T1","A3"),
    "T1_A5_20mm":    ("T1_PickPlace/A5",        "UR5_T1_A5_20mm_40cyc_*.h5",          "T1","A5"),
    "T1_A5_50mm":    ("T1_PickPlace/A5",        "UR5_T1_A5_50mm_40cyc_*.h5",          "T1","A5"),
    "T1_A5_100mm":   ("T1_PickPlace/A5",        "UR5_T1_A5_100mm_40cyc_*.h5",         "T1","A5"),
    "T2_healthy":    ("T2_Assembly/Healthy",    "UR5_T2_healthy_180cyc_*.h5",            "T2","healthy"),
    "T2_A2_1.5kg":   ("T2_Assembly/A2",         "UR5_T2_A2_1.5kg_gripper_40cyc_*.h5",    "T2","A2"),
    "T2_A2_2kg":     ("T2_Assembly/A2",         "UR5_T2_A2_2kg_gripper_40cyc_*.h5",      "T2","A2"),
    "T2_A2_3kg":     ("T2_Assembly/A2",         "UR5_T2_A2_3kg_gripper_40cyc_*.h5",      "T2","A2"),
    "T2_A3_7duct":   ("T2_Assembly/A3",         "UR5_T2_A3_light_duct_40cyc_*_214735.h5","T2","A3"),
    "T2_A3_14duct":  ("T2_Assembly/A3",         "UR5_T2_A3_medium_duct_40cyc_*_225508.h5","T2","A3"),
    "T2_A5_20mm":    ("T2_Assembly/A5",         "UR5_T2_A5_20mm_40cyc_*.h5",             "T2","A5"),
    "T2_A5_50mm":    ("T2_Assembly/A5",         "UR5_T2_A5_50mm_40cyc_*.h5",             "T2","A5"),
    "T2_A5_100mm":   ("T2_Assembly/A5",         "UR5_T2_A5_100mm_40cyc_*.h5",            "T2","A5"),
    "T3_healthy":    ("T3_Palletize/Healthy",   "UR5_T3_healthy_183cyc_*.h5",            "T3","healthy"),
    "T3_A2_3.5kg":   ("T3_Palletize/A2",        "UR5_T3_A2_3.5kg_gripper_33cyc_*.h5",    "T3","A2"),
    "T3_A2_4kg":     ("T3_Palletize/A2",        "UR5_T3_A2_4kg_gripper_33cyc_*.h5",      "T3","A2"),
    "T3_A2_5kg":     ("T3_Palletize/A2",        "UR5_T3_A2_4.5kg_gripper_33cyc_*.h5",    "T3","A2"),
    "T3_A3_7duct":   ("T3_Palletize/A3",        "UR5_T3_A3_light_duct_33cyc_*_222457.h5","T3","A3"),
    "T3_A3_14duct":  ("T3_Palletize/A3",        "UR5_T3_A3_medium_duct_33cyc_*_205648.h5","T3","A3"),
    "T3_A5_20mm":    ("T3_Palletize/A5",        "UR5_T3_A5_20mm_33cyc_*_172334.h5",      "T3","A5"),
    "T3_A5_50mm":    ("T3_Palletize/A5",        "UR5_T3_A5_50mm_33cyc_*_164447.h5",      "T3","A5"),
    "T3_A5_100mm":   ("T3_Palletize/A5",        "UR5_T3_A5_100mm_33cyc_*_160716.h5",     "T3","A5"),
    "T4_healthy":    ("T4_BinReorient/healthy", "UR5_T4_healthy_session2_35cyc_*.h5",    "T4","healthy"),
    "T4_A2_0.5kg":   ("T4_BinReorient/anomaly", "UR5_T4_A2_0.5kg_35cyc_*.h5",            "T4","A2"),
    "T4_A2_1kg":     ("T4_BinReorient/anomaly", "UR5_T4_A2_1kg_35cyc_*.h5",              "T4","A2"),
    "T4_A2_2kg":     ("T4_BinReorient/anomaly", "UR5_T4_A2_2kg_35cyc_*.h5",              "T4","A2"),
    "T4_A3_7duct":   ("T4_BinReorient/anomaly", "UR5_T4_A3_7wraps_35cyc_*.h5",           "T4","A3"),
    "T4_A3_14duct":  ("T4_BinReorient/anomaly", "UR5_T4_A3_14wraps_35cyc_*.h5",          "T4","A3"),
    "T4_A5_20mm":    ("T4_BinReorient/anomaly", "UR5_T4_A5_20mm_35cyc_*.h5",             "T4","A5"),
    "T4_A5_50mm":    ("T4_BinReorient/anomaly", "UR5_T4_A5_50mm_35cyc_*.h5",             "T4","A5"),
    "T4_A5_100mm":   ("T4_BinReorient/anomaly", "UR5_T4_A5_100mm_35cyc_*.h5",            "T4","A5"),
}
print(f"Registry: {len(REGISTRY)} entries.")

# %% Cell 2: BCa bootstrap AUROC

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

# %% Cell 3: Load all cycles

def load_cycles(filepath, task, anomaly):
    with h5py.File(filepath, "r") as f:
        cnum    = f["cycle_number"][:].astype(int).ravel()
        cur_all = f["actual_current"][:]
    is_anom = 0 if anomaly == "healthy" else 1
    out = []
    for c in np.unique(cnum[cnum > 0]):
        m = cnum == c
        if m.sum() >= MIN_SAMP:
            out.append({"current": cur_all[m], "task": task,
                        "anomaly": anomaly, "is_anomaly": is_anom})
    return out

print("Loading all cycles...")
all_cycles = []
for tag, (subdir, pattern, task, anomaly) in REGISTRY.items():
    hits = glob.glob(os.path.join(BASE, subdir, pattern))
    if not hits:
        print(f"  MISSING: {tag}")
        continue
    all_cycles.extend(load_cycles(hits[0], task, anomaly))

print(f"Loaded {len(all_cycles)} cycles")
for t in TASKS:
    nh = sum(1 for c in all_cycles if c["task"] == t and c["is_anomaly"] == 0)
    na = sum(1 for c in all_cycles if c["task"] == t and c["is_anomaly"] == 1)
    print(f"  {t}: {nh} healthy + {na} anomaly")

# %% Cell 4: FIXED_LEN over the 4-task pool

sub_lengths = np.array([len(range(0, len(c["current"]), SUBSAMPLE)) for c in all_cycles])
p5         = int(np.percentile(sub_lengths, 5))
FIXED_LEN  = max(p5 - (p5 % 4), 64)
print(f"FIXED_LEN (4-task pool): {FIXED_LEN}  (p5 = {p5})")

def cycle_to_tensor(cyc, fixed_len=FIXED_LEN):
    cur = cyc["current"]
    idx = list(range(0, len(cur), SUBSAMPLE))
    ts  = cur[idx].astype(np.float32)
    mu  = ts.mean(0, keepdims=True); sg = ts.std(0, keepdims=True) + 1e-8
    ts  = (ts - mu) / sg
    if len(ts) >= fixed_len:
        ts = ts[:fixed_len]
    else:
        pad = np.zeros((fixed_len - len(ts), ts.shape[1]), dtype=np.float32)
        ts  = np.vstack([ts, pad])
    return ts.T  # (channels, time)

def cycle_to_sequence(cyc, fixed_len=FIXED_LEN):
    return cycle_to_tensor(cyc, fixed_len).T  # (time, channels)

# %% Cell 5: Conv-AE at T4

class ConvAutoencoder(nn.Module):
    def __init__(self, fixed_len, n_channels=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,  8, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, n_channels, kernel_size=7, padding=3),
        )
    def forward(self, x):
        x_hat = self.decoder(self.encoder(x))
        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = x_hat[..., :x.shape[-1]]
        return x_hat
    @torch.no_grad()
    def reconstruction_score(self, x):
        self.eval()
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=[1, 2]).cpu().numpy()

print("Conv-AE T4 fold...")
healthy_cycles = [c for c in all_cycles if c["is_anomaly"] == 0]
tr_tensors = [cycle_to_tensor(c) for c in healthy_cycles if c["task"] in ("T1","T2","T3")]
X_tr = torch.tensor(np.stack(tr_tensors)).to(DEVICE)
dl_tr = DataLoader(TensorDataset(X_tr), batch_size=CONVAE_BATCH, shuffle=True)
print(f"  Training cycles: {len(X_tr)}")

torch.manual_seed(42); np.random.seed(42)
convae = ConvAutoencoder(FIXED_LEN).to(DEVICE)
opt    = optim.Adam(convae.parameters(), lr=CONVAE_LR)
crit   = nn.MSELoss()

t0 = time.perf_counter()
convae.train()
for ep in range(CONVAE_EPOCHS):
    eloss = 0.0
    for (xb,) in dl_tr:
        opt.zero_grad(); loss = crit(convae(xb), xb); loss.backward(); opt.step()
        eloss += loss.item() * len(xb)
    if (ep + 1) % 20 == 0:
        print(f"  Epoch {ep+1:3d}/{CONVAE_EPOCHS}  loss = {eloss/len(X_tr):.6f}")
print(f"  Train time: {time.perf_counter()-t0:.1f}s")

convae.eval()
te_cycles_T4 = [c for c in all_cycles if c["task"] == "T4"]
scores_conv, y_T4 = [], []
for cyc in te_cycles_T4:
    x = torch.tensor(cycle_to_tensor(cyc)).unsqueeze(0).to(DEVICE)
    scores_conv.append(float(convae.reconstruction_score(x)[0]))
    y_T4.append(cyc["is_anomaly"])
scores_conv = np.array(scores_conv); y_T4 = np.array(y_T4)

auroc_conv, lo_conv, hi_conv = bootstrap_auroc_bca(y_T4, scores_conv)
print(f"  Conv-AE T4 AUROC = {auroc_conv:.4f}  [{lo_conv:.4f}, {hi_conv:.4f}]")

# %% Cell 6: LSTM-VAE at T4

class LSTMEncoder(nn.Module):
    def __init__(self, n_channels, hidden_dim, latent_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True)
        self.fc_mu     = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc_mu(h_cat), self.fc_logvar(h_cat)

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_channels, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.fc_init  = nn.Linear(latent_dim, hidden_dim)
        self.lstm     = nn.LSTM(latent_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_channels)
    def forward(self, z, seq_len):
        z_rep = z.unsqueeze(1).repeat(1, seq_len, 1)
        h0    = torch.tanh(self.fc_init(z)).unsqueeze(0).repeat(self.n_layers, 1, 1)
        c0    = torch.zeros_like(h0)
        out, _ = self.lstm(z_rep, (h0, c0))
        return self.fc_out(out)

class LSTMVAE(nn.Module):
    def __init__(self, n_channels=6, hidden_dim=LSTMVAE_HIDDEN, latent_dim=LSTMVAE_LATENT,
                 n_layers=LSTMVAE_N_LAYERS, beta=LSTMVAE_BETA):
        super().__init__()
        self.beta    = beta
        self.encoder = LSTMEncoder(n_channels, hidden_dim, latent_dim, n_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, n_channels, n_layers)
    def reparam(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparam(mu, logvar)
        return self.decoder(z, x.shape[1]), mu, logvar
    def elbo_loss(self, x):
        x_hat, mu, logvar = self.forward(x)
        recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kl
    @torch.no_grad()
    def reconstruction_score(self, x):
        self.eval()
        x_hat, _, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=[1, 2]).cpu().numpy()

print("LSTM-VAE T4 fold...")
tr_seqs = [cycle_to_sequence(c) for c in healthy_cycles if c["task"] in ("T1","T2","T3")]
X_tr_seq = torch.tensor(np.stack(tr_seqs), dtype=torch.float32)
dl_seq   = DataLoader(TensorDataset(X_tr_seq), batch_size=LSTMVAE_BATCH, shuffle=True)
print(f"  Training sequences: {len(X_tr_seq)}")

torch.manual_seed(42); np.random.seed(42)
lstmvae   = LSTMVAE().to(DEVICE)
opt       = optim.Adam(lstmvae.parameters(), lr=LSTMVAE_LR)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

t0 = time.perf_counter()
lstmvae.train()
for ep in range(LSTMVAE_EPOCHS):
    eloss = 0.0
    for (xb,) in dl_seq:
        xb = xb.to(DEVICE)
        loss = lstmvae.elbo_loss(xb)
        opt.zero_grad(); loss.backward(); opt.step()
        eloss += loss.item() * len(xb)
    scheduler.step()
    if (ep + 1) % 20 == 0:
        print(f"  Epoch {ep+1:3d}/{LSTMVAE_EPOCHS}  loss = {eloss/len(X_tr_seq):.6f}")
print(f"  Train time: {time.perf_counter()-t0:.1f}s")

lstmvae.eval()
te_seqs = np.stack([cycle_to_sequence(c) for c in te_cycles_T4])
X_te    = torch.tensor(te_seqs, dtype=torch.float32).to(DEVICE)
scores_lstm = lstmvae.reconstruction_score(X_te)

auroc_lstm, lo_lstm, hi_lstm = bootstrap_auroc_bca(y_T4, scores_lstm)
print(f"  LSTM-VAE T4 AUROC = {auroc_lstm:.4f}  [{lo_lstm:.4f}, {hi_lstm:.4f}]")

# %% Cell 7: Raw Z-Score at T4

def raw_zscore_features(cycle):
    cur = cycle["current"]; f = {}
    for j in range(6):
        s = cur[:, j]
        f[f"J{j}_mean"] = float(np.mean(s))
        f[f"J{j}_std"]  = float(np.std(s))
        f[f"J{j}_rms"]  = float(np.sqrt(np.mean(s**2)))
    f["total_rms"] = float(np.sqrt(np.mean(cur**2)))
    return f

print("Raw Z-Score T4 fold...")
rows_raw = []
for c in all_cycles:
    r = raw_zscore_features(c)
    r.update(task=c["task"], anomaly=c["anomaly"], is_anomaly=c["is_anomaly"])
    rows_raw.append(r)
df_raw   = pd.DataFrame(rows_raw)
RAW_COLS = [k for k in df_raw.columns if k not in ("task","anomaly","is_anomaly")]
print(f"  Raw features per cycle: {len(RAW_COLS)}")

tr_mask = (df_raw.task.isin(["T1","T2","T3"])) & (df_raw.is_anomaly == 0)
te_mask = df_raw.task == "T4"
X_tr_raw = df_raw[tr_mask][RAW_COLS].values
X_te_raw = df_raw[te_mask][RAW_COLS].values
y_te_raw = df_raw[te_mask]["is_anomaly"].values

mu  = X_tr_raw.mean(0); std = X_tr_raw.std(0); std[std == 0] = 1
scores_raw = np.mean(((X_te_raw - mu) / std) ** 2, axis=1)
auroc_raw, lo_raw, hi_raw = bootstrap_auroc_bca(y_te_raw, scores_raw)
print(f"  Raw Z-Score T4 AUROC = {auroc_raw:.4f}  [{lo_raw:.4f}, {hi_raw:.4f}]")

# %% Cell 8: Per-anomaly AUROC at T4 for baselines

# Sanity check: anomaly labels must be aligned across the three baselines.
anom_per_cyc = [c["anomaly"] for c in te_cycles_T4]
df_raw_T4 = df_raw[te_mask].reset_index(drop=True)
assert all(a == b for a, b in zip(anom_per_cyc, df_raw_T4["anomaly"].tolist())), \
    "Anomaly label ordering mismatch between Conv-AE/LSTM-VAE cycles and Raw Z-Score cycles."

per_anom_rows = []
for method, scores in [("Conv-AE", scores_conv), ("LSTM-VAE", scores_lstm),
                       ("Raw Z-Score", scores_raw)]:
    for anom in ["A2","A3","A5"]:
        idx = [i for i, c in enumerate(te_cycles_T4)
               if c["anomaly"] == anom or c["anomaly"] == "healthy"]
        y_t = y_T4[idx]
        y_s = scores[idx]
        if len(np.unique(y_t)) < 2:
            continue
        auroc, lo, hi = bootstrap_auroc_bca(y_t, y_s)
        per_anom_rows.append(dict(method=method, anomaly=anom, fold="T4",
                                  n_healthy=int((y_t == 0).sum()),
                                  n_anomaly=int((y_t == 1).sum()),
                                  auroc=round(auroc, 4),
                                  ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
        print(f"  {method:<13} {anom}: AUROC = {auroc:.4f}  [{lo:.4f}, {hi:.4f}]")
pd.DataFrame(per_anom_rows).to_csv(os.path.join(OUT, "T4_baselines_per_anomaly.csv"),
                                    index=False, float_format="%.4f")

# %% Cell 9: Save aggregate AUROC and per-cycle scores

agg_rows = []
for method, scores in [("Conv-AE", scores_conv), ("LSTM-VAE", scores_lstm),
                       ("Raw Z-Score", scores_raw)]:
    auroc, lo, hi = bootstrap_auroc_bca(y_T4, scores)
    agg_rows.append(dict(test_task="T4", method=method,
                         n_healthy=int((y_T4 == 0).sum()),
                         n_anomaly=int((y_T4 == 1).sum()),
                         auroc=round(auroc, 4), ci_lo=round(lo, 4),
                         ci_hi=round(hi, 4), ci_width=round(hi - lo, 4)))
pd.DataFrame(agg_rows).to_csv(os.path.join(OUT, "T4_baselines_aggregate.csv"),
                              index=False, float_format="%.4f")

with open(os.path.join(OUT, "T4_baseline_scores.pkl"), "wb") as fh:
    pickle.dump({"Conv-AE":     scores_conv,
                 "LSTM-VAE":    scores_lstm,
                 "Raw Z-Score": scores_raw,
                 "y_true":      y_T4,
                 "te_cycles":   [{"task": c["task"], "anomaly": c["anomaly"],
                                  "is_anomaly": c["is_anomaly"]}
                                  for c in te_cycles_T4]}, fh)

print(f"\nConv-AE T4 AUROC     = {auroc_conv:.4f}")
print(f"LSTM-VAE T4 AUROC    = {auroc_lstm:.4f}")
print(f"Raw Z-Score T4 AUROC = {auroc_raw:.4f}")
print(f"Next: run T4_operating_points.py")
