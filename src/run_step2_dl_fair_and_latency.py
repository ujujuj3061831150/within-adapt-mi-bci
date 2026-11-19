"""
Step-2 Deep Learning Fair Multi-Seed Benchmark & Online Single-Trial Inference Latency
===================================================================================
Harmonizes ALL DL baselines with the exact strictly causal preprocessing (CAR + 4th-order
causal Butterworth `sosfilt` 8-30 Hz in microvolts uV), multi-seed evaluation (seeds 42,
123, 2025), and subject-level validation early stopping (15% held-out train subjects,
patience=12, max_epochs=50):
  1. Single-process standalone 109-fold CPU runtime for `Within-Adapt` (apples-to-apples
     against single-GPU 109-fold `Online EA + EEGNet-8,2`)
  2. Single-trial online inference latency (500 timed iterations after 50 warm-up):
     pre-trained `Online EA (W=8) + EEGNet-8,2` on CPU (1-thread / default) & RTX 4060 GPU
     including CAR + causal `sosfilt` + W=8 EA matrix inverse-sqrt + forward pass
  3. Fair 3-seed causal `sosfilt` benchmarks on EEGBCI (N=109), BCI-IV-2a (Bin & 4-Cls LOSO),
     and Chang2025 (Bin & 3-Cls LOSO), reporting BOTH `trials[15:]` AND Trial-0 Cold-Start
     (`trials[0:]` and `[0:15]`) for `Online EA + EEGNet-8,2` with Bayesian EA prior!
"""

from __future__ import annotations

import os
import random
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import mne
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import dl_loaders_causal as dl_load
import eegbci_reviewer_ablations as eeg_ab
import run_step2_riemannian_master as step2
import unified_within_adapt_coral_benchmark as uni

OUT_NPZ = config.RESULTS_DIR / "step2_dl_and_latency.npz"

SEEDS = [42, 123, 2025]
EVAL_START_IDX = 15
CAUSAL_WINDOW = 8
TAU1_PRIOR = 2.0
EA_REG_EPS = 1e-6
EIG_EPS = 1e-12

MAX_EPOCHS = 50
PATIENCE = 12
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-2


def causal_sosfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    zi_base = sosfilt_zi(sos)
    zi = zi_base[:, None, :] * x[:, :1][None, :, :]
    out, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return out


def sym_inv_sqrtm(mat: np.ndarray) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, EIG_EPS)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


class EEGNet82(nn.Module):
    def __init__(self, n_chans: int, n_times: int, n_classes: int, kernel_length: int) -> None:
        super().__init__()
        F1, D, F2 = 8, 2, 16
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25),
        )
        with torch.no_grad():
            flat_dim = self.block2(self.block1(torch.zeros(1, 1, n_chans, n_times))).numel()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block2(self.block1(x)).flatten(start_dim=1))


class ShallowConvNet(nn.Module):
    def __init__(self, n_chans: int, n_times: int, n_classes: int) -> None:
        super().__init__()
        self.conv_temp = nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False)
        self.conv_spat = nn.Conv2d(40, 40, kernel_size=(n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7))
        self.drop = nn.Dropout(0.5)
        with torch.no_grad():
            z = self.pool(torch.log(torch.clamp(self.bn(self.conv_spat(self.conv_temp(torch.zeros(1, 1, n_chans, n_times)))) ** 2, min=1e-6)))
            flat_dim = z.numel()
        self.clf = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_spat(self.conv_temp(x))
        h = self.bn(h) ** 2
        h = torch.log(torch.clamp(self.pool(h), min=1e-6))
        return self.clf(self.drop(h).flatten(start_dim=1))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Causal uV Deep Learning Trial Loaders
# ==============================================================================
def compute_ea_subject_blocks(epochs: np.ndarray, n_chans: int, n_times: int) -> dict:
    """epochs shape (N, ch, T) in uV."""
    # Static Z-score
    mu = epochs.mean(axis=-1, keepdims=True)
    sd = epochs.std(axis=-1, keepdims=True) + 1e-6
    x_stat = ((epochs - mu) / sd)[:, None, :, :].astype(np.float32)

    # Spatial covariance per trial
    xc = epochs - mu
    covs = np.einsum("nct,ndt->ncd", xc, xc) / float(n_times) + EA_REG_EPS * np.eye(n_chans)[None, :, :]
    r_sub = covs.mean(axis=0)
    r_inv = sym_inv_sqrtm(r_sub)
    x_tr_ea = np.einsum("ij,njt->nit", r_inv, epochs)[:, None, :, :].astype(np.float32)
    return {"x_stat": x_stat, "x_tr_ea": x_tr_ea, "raw": epochs.astype(np.float32), "covs": covs.astype(np.float64), "r_sub": r_sub.astype(np.float64)}


def apply_causal_ea_stream(epochs: np.ndarray, covs: np.ndarray, r_src_mean: np.ndarray) -> np.ndarray:
    """Strictly causal EA over [0:] using Bayesian source prior at t<8 and pure W=8 at t>=8."""
    n_trials = len(epochs)
    out = np.empty((n_trials, 1, epochs.shape[1], epochs.shape[2]), dtype=np.float32)
    for t in range(n_trials):
        if t == 0:
            r_t = r_src_mean
        elif t < CAUSAL_WINDOW:
            r_t = (covs[:t].sum(axis=0) + TAU1_PRIOR * r_src_mean) / (t + TAU1_PRIOR)
        else:
            r_t = covs[t - CAUSAL_WINDOW : t].mean(axis=0)
        r_inv = sym_inv_sqrtm(r_t)
        out[t, 0] = r_inv @ epochs[t]
    return out


def dl_worker_eegbci(sid: int) -> dict:
    mne.set_log_level("WARNING")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sub_str = f"S{sid:03d}"
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=160.0, output="sos")
    s0, s1 = int(round(0.5 * 160.0)), int(round(0.5 * 160.0)) + 480
    ep_list, y_list = [], []
    for run in eeg_ab.MI_RUNS:
        raw = mne.io.read_raw_edf(eeg_ab.DATA_DIR / sub_str / f"{sub_str}R{run:02d}.edf", preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)
        if abs(raw.info["sfreq"] - 160.0) > 1e-3:
            raw.resample(160.0, verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        raw.pick([ch for ch in eeg_ab.CORE_CHANNELS if ch in raw.ch_names])
        events, ev_all = mne.events_from_annotations(raw, verbose=False)
        mi_id = {k: v for k, v in ev_all.items() if k in ("T1", "T2")}
        data_uv = raw.get_data() * 1e6
        vev = events[np.isin(events[:, 2], [mi_id["T1"], mi_id["T2"]])]
        vev = vev[(vev[:, 0] + s0 >= 0) & (vev[:, 0] + s1 <= data_uv.shape[1])]
        f_ca = causal_sosfilt(sos, data_uv)
        ep_list.append(np.stack([f_ca[:, ev[0] + s0 : ev[0] + s1] for ev in vev], axis=0))
        y_list.append((vev[:, 2] != mi_id["T1"]).astype(np.int64))
    epochs = np.concatenate(ep_list, 0)
    blk = compute_ea_subject_blocks(epochs, 13, 480)
    blk["y"] = np.concatenate(y_list, 0)
    return blk


def dl_worker_b2a(sid: int) -> dict:
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=250.0, output="sos")
    s0 = int(round((b2a.CUE_OFFSET_SEC + 0.5) * 250.0))
    s1 = s0 + 750
    out = {}
    for sn in ("T", "E"):
        mat = sio.loadmat(b2a.DATA_DIR / f"A{sid:02d}{sn}.mat", simplify_cells=True)
        ep_list, y_list = [], []
        for rd in mat["data"]:
            tr = np.asarray(rd["trial"], dtype=np.int64)
            if tr.size == 0:
                continue
            X = np.asarray(rd["X"], dtype=np.float64)[:, :22]
            X_car = (X - X.mean(1, keepdims=True)).T
            f_ca = causal_sosfilt(sos, X_car)
            ep_list.append(np.stack([f_ca[:, (i - 1) + s0 : (i - 1) + s1] for i in tr], 0))
            y_list.append(np.asarray(rd["y"], dtype=np.int64) - 1)
        ep = np.concatenate(ep_list, 0)
        blk = compute_ea_subject_blocks(ep, 22, 750)
        blk["y"] = np.concatenate(y_list, 0)
        out[sn] = blk
    return out


def dl_worker_chang(subj_dir: Path) -> dict:
    sid = subj_dir.name
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=1000.0, output="sos")
    s0, s1 = int(round(0.5 * 250.0)), int(round(0.5 * 250.0)) + 750
    ep_list, y_list = [], []
    for sp in sorted(subj_dir.glob(f"{sid}_MI[0-9]*.set")):
        eeg = sio.loadmat(sp, simplify_cells=True)["EEG"]
        n_ch, n_pts = int(eeg["nbchan"]), int(eeg["pnts"])
        ch_names = [c["labels"] for c in eeg["chanlocs"]]
        pick_idx = [ch_names.index(c) for c in ch25.CORE12_CHANNELS]
        raw = np.fromfile(sp.with_suffix(".fdt"), dtype=np.float32).reshape((n_ch, n_pts), order="F").astype(np.float64)
        raw = raw - raw.mean(0, keepdims=True)
        f250 = causal_sosfilt(sos, raw[pick_idx])[:, ::4]
        evs = eeg["event"]
        if isinstance(evs, dict):
            evs = [evs]
        lats, ys = [], []
        for ev in evs:
            tv = int(ev["type"])
            if tv in (1, 2, 3):
                s250 = int(round((float(ev["latency"]) - 1.0) / 4.0))
                if s250 + s1 <= f250.shape[1]:
                    lats.append(s250)
                    ys.append(tv - 1)
        ep_list.append(np.stack([f250[:, s + s0 : s + s1] for s in lats], 0))
        y_list.append(np.asarray(ys, dtype=np.int64))
    ep = np.concatenate(ep_list, 0)
    blk = compute_ea_subject_blocks(ep, 12, 750)
    blk["y"] = np.concatenate(y_list, 0)
    return blk


def subset_dl_block(blk: dict, allowed: tuple[int, ...]) -> dict:
    mask = np.isin(blk["y"], allowed)
    # Remap labels 0..K-1
    y_m = blk["y"][mask]
    uniq = sorted(allowed)
    y_map = np.zeros_like(y_m)
    for new_i, old_v in enumerate(uniq):
        y_map[y_m == old_v] = new_i
    raw_sub = blk["raw"][mask]
    return compute_ea_subject_blocks(raw_sub.astype(np.float64), raw_sub.shape[1], raw_sub.shape[2]) | {"y": y_map}


def train_eval_val_stop(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    make_model,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    set_seeds(seed)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    Xte = torch.from_numpy(X_te).to(device)

    best_loss = float("inf")
    best_state = None
    no_imp = 0
    n_tr = Xt.shape[0]

    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, BATCH_SIZE):
            idx = perm[s : s + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = float(crit(model(Xv), yv).item())
        if vloss < best_loss - 1e-4:
            best_loss = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(dim=1).cpu().numpy()
    return preds == y_te


def benchmark_ea_eegnet_latency(n_chans: int, n_times: int, sfreq: float, name: str, device_cuda: torch.device) -> dict[str, float]:
    rng = np.random.default_rng(42)
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
    zi = np.repeat(sosfilt_zi(sos)[:, None, :], n_chans, axis=1)
    klen = 64 if sfreq == 160.0 else 100
    raw_ep = rng.normal(size=(n_chans, n_times)) * 10.0
    cov_buf = [np.eye(n_chans) * 10.0 + EA_REG_EPS * np.eye(n_chans) for _ in range(CAUSAL_WINDOW)]

    res = {}
    for dev_label, dev in [("cpu", torch.device("cpu")), ("gpu", device_cuda)]:
        if dev.type == "cuda" and not torch.cuda.is_available():
            continue
        model = EEGNet82(n_chans, n_times, 2, kernel_length=klen).to(dev).eval()
        lats = []
        with torch.no_grad():
            for it in range(550):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                x_car = raw_ep - raw_ep.mean(0, keepdims=True)
                x_filt, zi = sosfilt(sos, x_car, axis=-1, zi=zi)
                xc = x_filt - x_filt.mean(-1, keepdims=True)
                c_cur = (xc @ xc.T) / float(n_times) + EA_REG_EPS * np.eye(n_chans)
                r_inv = sym_inv_sqrtm(np.mean(cov_buf, axis=0))
                x_ea = (r_inv @ x_filt)[None, None, :, :].astype(np.float32)
                ten = torch.from_numpy(x_ea).to(dev)
                _ = model(ten).argmax(dim=1).item()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                cov_buf.append(c_cur); cov_buf.pop(0)
                if it >= 50:
                    lats.append((t1 - t0) * 1000.0)
        a = np.asarray(lats)
        print(f"  Online EA+EEGNet [{name} | {dev_label.upper()}]: {a.mean():.2f}+-{a.std():.2f} ms (P50={np.median(a):.2f}, P95={np.percentile(a,95):.2f})")
        res[f"{dev_label}_mean"] = float(a.mean())
        res[f"{dev_label}_std"] = float(a.std())
        res[f"{dev_label}_p50"] = float(np.median(a))
        res[f"{dev_label}_p95"] = float(np.percentile(a, 95))
    return res


def benchmark_single_proc_wa_109(eeg_subs_ca: list[dict]) -> float:
    """Single-process 109-fold standalone time for Within-Adapt (strictly matching single-GPU batch)."""
    t0 = time.perf_counter()
    rec_f = [eeg_ab.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(0))) for b in eeg_subs_ca]
    rec_h = [eeg_ab.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(0))) for b in eeg_subs_ca]
    for i in range(109):
        tr = [j for j in range(109) if j != i]
        y_tr = np.concatenate([eeg_subs_ca[j]["y"] for j in tr])
        pf = PCA(n_components=30, random_state=42).fit(np.concatenate([rec_f[j] for j in tr]))
        ph = PCA(n_components=30, random_state=42).fit(np.concatenate([rec_h[j] for j in tr]))
        zf_l, zh_l = [pf.transform(rec_f[j]) for j in tr], [ph.transform(rec_h[j]) for j in tr]
        wfi, sfc, sfs = uni.fit_within_operator(np.concatenate(zf_l), y_tr)
        whi, shc, shs = uni.fit_within_operator(np.concatenate(zh_l), y_tr)
        Xtr = np.concatenate([np.concatenate([step2.within_adapt_s(zf @ wfi, zf @ wfi, sfc, sfs, 1.0), step2.within_adapt_s(zh @ whi, zh @ whi, shc, shs, 1.0)], 1) for zf, zh in zip(zf_l, zh_l)])
        clf = LogisticRegression(solver="newton-cg", max_iter=5000, C=0.05, random_state=42).fit(Xtr, y_tr)
        te = eeg_subs_ca[i]
        n_te = len(te["y"])
        Xte = np.empty((n_te - 15, 60))
        for oi, t in enumerate(range(15, n_te)):
            mf = b2a.sym_expm(te["log_f"][t - 8 : t].mean(0))
            mh = b2a.sym_expm(te["log_h"][t - 8 : t].mean(0))
            vf = pf.transform(eeg_ab.project_tangent(te["cov_f"][t - 8 : t + 1], mf)) @ wfi
            vh = ph.transform(eeg_ab.project_tangent(te["cov_h"][t - 8 : t + 1], mh)) @ whi
            Xte[oi, :30] = step2.within_adapt_s(vf[:-1], vf[-1:], sfc, sfs, 1.0)[0]
            Xte[oi, 30:] = step2.within_adapt_s(vh[:-1], vh[-1:], shc, shs, 1.0)[0]
        _ = clf.predict(Xte)
    dt = time.perf_counter() - t0
    print(f"  Single-Process 109-fold Standalone Runtime (Within-Adapt): {dt:.2f} s")
    return float(dt)


def eval_dl_dataset(
    tag: str,
    subj_blocks: list[list[dict]],  # per subject list of train blocks
    test_blocks: list[dict],
    n_chans: int,
    n_times: int,
    n_classes: int,
    kernel_len: int,
    seeds: list[int],
    device: torch.device,
    models_to_run: list[str] | None = None,
) -> dict[str, np.ndarray]:
    n_sub = len(test_blocks)
    out = {}
    if models_to_run is None:
        models_to_run = ["Static_EEGNet", "Online_EA_EEGNet"]

    for mname in models_to_run:
        acc_15 = np.empty((len(seeds), n_sub), dtype=np.float64)
        acc_0 = np.empty((len(seeds), n_sub), dtype=np.float64)
        acc_b = np.empty((len(seeds), n_sub), dtype=np.float64)
        t_start = time.perf_counter()

        for si, seed in enumerate(seeds):
            t_s0 = time.perf_counter()
            rng_split = np.random.default_rng(seed)
            for i in range(n_sub):
                tr_sub_ids = [j for j in range(n_sub) if j != i]
                rng_split.shuffle(tr_sub_ids)
                n_val = max(1, int(round(0.15 * len(tr_sub_ids))))
                val_ids, tr_ids = tr_sub_ids[:n_val], tr_sub_ids[n_val:]

                tr_b = [b for j in tr_ids for b in subj_blocks[j]]
                val_b = [b for j in val_ids for b in subj_blocks[j]]
                te_b = test_blocks[i]

                if mname == "Online_EA_EEGNet":
                    xkey = "x_tr_ea"
                    r_src = np.mean([b["r_sub"] for b in tr_b], axis=0)
                    X_te = apply_causal_ea_stream(te_b["raw"], te_b["covs"], r_src)
                else:
                    xkey = "x_stat"
                    X_te = te_b["x_stat"]

                X_tr = np.concatenate([b[xkey] for b in tr_b], 0)
                y_tr = np.concatenate([b["y"] for b in tr_b], 0)
                X_val = np.concatenate([b[xkey] for b in val_b], 0)
                y_val = np.concatenate([b["y"] for b in val_b], 0)
                y_te = te_b["y"]

                if mname == "Static_ShallowConvNet":
                    maker = lambda: ShallowConvNet(n_chans, n_times, n_classes)
                else:
                    maker = lambda: EEGNet82(n_chans, n_times, n_classes, kernel_len)

                correct = train_eval_val_stop(X_tr, y_tr, X_val, y_val, X_te, y_te, maker, seed + i, device)
                acc_15[si, i] = float(np.mean(correct[EVAL_START_IDX:]))
                acc_0[si, i] = float(np.mean(correct))
                acc_b[si, i] = float(np.mean(correct[:EVAL_START_IDX]))

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_seed = time.perf_counter() - t_s0
            if si == 0:
                out[f"{tag}__{mname}__time_seed0"] = np.asarray([dt_seed])

        mean_15 = acc_15.mean(axis=0)
        mean_0 = acc_0.mean(axis=0)
        mean_b = acc_b.mean(axis=0)
        out[f"{tag}__{mname}__acc15_seeds"] = acc_15
        out[f"{tag}__{mname}__acc0_seeds"] = acc_0
        out[f"{tag}__{mname}__accburn_seeds"] = acc_b
        out[f"{tag}__{mname}__acc15"] = mean_15
        out[f"{tag}__{mname}__acc0"] = mean_0
        out[f"{tag}__{mname}__accburn"] = mean_b
        print(
            f"  [{tag} | {mname:21s}] ({len(seeds)} seeds, total={time.perf_counter()-t_start:.1f}s) | "
            f"[15:]={mean_15.mean()*100:5.2f}% (seed_means={[round(x*100,2) for x in acc_15.mean(1)]}) | "
            f"[0:]={mean_0.mean()*100:5.2f}% (burn={mean_b.mean()*100:5.2f}%)"
        )
    return out


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print(f"STEP-2 DL MULTI-SEED (HARMONIZED CAUSAL sosfilt) & LATENCY | Torch={torch.__version__}, Device={device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    try:
        cpu_name = subprocess.check_output(["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
        print(f"  CPU: {cpu_name}")
    except Exception:
        pass
    print("=" * 96)

    store = {}

    # 1. Single-trial online latency benchmarks for Online EA + EEGNet
    print("\n[ONLINE SINGLE-TRIAL INFERENCE LATENCY BENCHMARK (CAR + causal sosfilt + W=8 EA + EEGNet)]")
    for k, v in benchmark_ea_eegnet_latency(13, 480, 160.0, "EEGBCI 13ch 160Hz", device).items():
        store[f"lat_dl_13__{k}"] = np.asarray([v])
    for k, v in benchmark_ea_eegnet_latency(22, 750, 250.0, "BCI-IV-2a 22ch 250Hz", device).items():
        store[f"lat_dl_22__{k}"] = np.asarray([v])

    # 2. Load causal sosfilt uV DL datasets + Riemannian EEGBCI for exact 1-process batch timing
    with ProcessPoolExecutor(max_workers=8) as ex:
        b2a_dl = list(ex.map(dl_load.dl_worker_b2a, range(1, 10)))
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor(max_workers=8) as ex:
        ch_dl = list(ex.map(dl_load.dl_worker_chang, ch_dirs))
    with ProcessPoolExecutor(max_workers=8) as ex:
        eeg_dl = list(ex.map(dl_load.dl_worker_eegbci, range(1, 110)))
        eeg_ca = list(ex.map(step2.worker_eegbci, range(1, 110)))
    print(f"Loaded causal DL datasets in {time.perf_counter()-t0:.1f}s")

    store["wa_1proc_109_time"] = np.asarray([benchmark_single_proc_wa_109(eeg_ca)])

    # 3. BCI-IV-2a (3 seeds) & Chang2025 (3 seeds) & EEGBCI (3 seeds for EEGNet/EA+EEGNet, 1 seed ShallowConvNet)
    print("\n[FAIR MULTI-SEED DL BENCHMARKS (CAR + causal sosfilt + val-split early stopping)]")
    for tag, cset, k_len, n_t, n_ch, subs_raw, is_b2a in [
        ("2a_bin_loso", (0, 1), 100, 750, 22, b2a_dl, True),
        ("2a_4cls_loso", (0, 1, 2, 3), 100, 750, 22, b2a_dl, True),
        ("chang_bin", (0, 1), 100, 750, 12, ch_dl, False),
        ("chang_3cls", (0, 1, 2), 100, 750, 12, ch_dl, False),
    ]:
        if is_b2a:
            s_blk = [[subset_dl_block(s["T"], cset), subset_dl_block(s["E"], cset)] for s in subs_raw]
            te_blk = [subset_dl_block(s["E"], cset) for s in subs_raw]
        else:
            s_blk = [[subset_dl_block(s, cset)] for s in subs_raw]
            te_blk = [b[0] for b in s_blk]
        res = eval_dl_dataset(tag, s_blk, te_blk, n_ch, n_t, len(cset), k_len, SEEDS, device)
        store.update(res)

    # EEGBCI (N=109): 3 seeds for Static EEGNet & Online EA + EEGNet; 1 seed ShallowConvNet
    eeg_sblk = [[s] for s in eeg_dl]
    res_eeg = eval_dl_dataset("eegbci_bin_loso", eeg_sblk, eeg_dl, 13, 480, 2, 64, SEEDS, device, models_to_run=["Static_EEGNet", "Online_EA_EEGNet"])
    store.update(res_eeg)
    res_sh = eval_dl_dataset("eegbci_bin_loso", eeg_sblk, eeg_dl, 13, 480, 2, 64, [42], device, models_to_run=["Static_ShallowConvNet"])
    store.update(res_sh)

    np.savez_compressed(OUT_NPZ, **store)
    print(f"\nSaved DL & Latency results to {OUT_NPZ} in {time.perf_counter()-t0:.1f}s total.")


if __name__ == "__main__":
    main()
