"""
Step-2 Master Riemannian Causal Suite (Authoritative Unified Pipeline)
=======================================================================
Fixes ALL Step-1 & Reviewer technical issues in one deterministic run:
  1. Unit-consistent microvolts (uV) across ALL 4 datasets (EEGBCI EDF * 1e6 -> uV)
  2. Strictly causal 4th-order Butterworth IIR `sosfilt` with t=0 steady-state zi (ZERO sosfiltfilt)
  3. Unified LogisticRegression(solver="newton-cg", max_iter=5000, C=0.05) across all 10 protocols
  4. Subject-level aggregation for Kaya2018 (N=7 LOSO, N=5 CS) + updated Holm-Bonferroni
  5. Complete comparators: Uncal, G1, Causal RPA (W=8), Direct Tang-LR, OAS-CORAL (Orig),
     z-CORAL (Unshrunk-Src), z-CORAL (OAS-Src), Adapt-CORAL, Within-CORAL(0.7), Within-Adapt, Online EA+TangLR
  6. True Trial-0 Cold-Start (trials[0:] and [0:15]) across ALL 10 protocols (Uncal, G1, Orig, Within-Adapt)
  7. EEGBCI operational regimes (100% cov, Low-Base, 80% Conf Rej per-subj & global, Transition, Steady),
     disjoint Mu(8-13)+Beta(13-30) vs nested Full(8-30)+High-Beta(21-30) ablation, gate scale s in {0.5,1,2}
  8. Multi-class W-sweep (W=8..48) on 2a 4-Cls and Chang 3-Cls
  9. End-to-end single-trial CPU latency (including CAR + causal sosfilt + OAS + logm + Within-Adapt + LR)
     and exact hardware/software metadata.
"""

from __future__ import annotations

import os
import platform
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import mne
import numpy as np
import scipy
import scipy.io as sio
import sklearn
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import eegbci_reviewer_ablations as eeg_ab
import kaya2018_causal_coral_repro as ky18
import unified_within_adapt_coral_benchmark as uni

OUTPUT_NPZ = config.RESULTS_DIR / "step2_unified_causal_master.npz"

PCA_DIM = 30
CAUSAL_WINDOW = 8
EVAL_START_IDX = 15
LOGREG_C = 0.05
LOGREG_MAX_ITER = 5000
RANDOM_STATE = 42
SPD_EPS = 1e-10
TAU1_PRIOR = 2.0

_G_STATE: dict = {}


def _init_state(state: dict) -> None:
    global _G_STATE
    _G_STATE = state


def causal_sosfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Strictly causal 4th-order Butterworth IIR filter initialized at t=0 steady state."""
    zi_base = sosfilt_zi(sos)
    zi = zi_base[:, None, :] * x[:, :1][None, :, :]
    out, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return out


def emp_cov_epochs(epochs: np.ndarray) -> np.ndarray:
    xc = epochs - epochs.mean(axis=-1, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", xc, xc) / float(xc.shape[-1])
    return 0.5 * (covs + np.swapaxes(covs, -1, -2))


def oas_from_emp_cov(emp_covs: np.ndarray, n_samples: int) -> np.ndarray:
    out = np.empty_like(emp_covs)
    n_feat = emp_covs.shape[-1]
    eye = np.eye(n_feat, dtype=np.float64)
    for i in range(emp_covs.shape[0]):
        s = 0.5 * (emp_covs[i] + emp_covs[i].T)
        alpha = float(np.mean(s ** 2))
        mu = float(np.trace(s)) / n_feat
        den = (n_samples + 1.0) * (alpha - (mu ** 2) / n_feat)
        rho = 1.0 if den <= 0.0 else min((alpha + mu ** 2) / den, 1.0)
        c = (1.0 - rho) * s + rho * mu * eye
        out[i] = 0.5 * (c + c.T)
    return out


def pack_ca_block(ep_f: np.ndarray, ep_h: np.ndarray, y: np.ndarray, extra: dict | None = None) -> dict:
    n_t = int(ep_f.shape[-1])
    emp_f = emp_cov_epochs(ep_f)
    emp_h = emp_cov_epochs(ep_h)
    cov_f = oas_from_emp_cov(emp_f, n_t)
    cov_h = oas_from_emp_cov(emp_h, n_t)
    blk = {
        "cov_f": cov_f,
        "cov_h": cov_h,
        "log_f": b2a.sym_logm(cov_f),
        "log_h": b2a.sym_logm(cov_h),
        "emp_f": emp_f,
        "emp_h": emp_h,
        "n_t": n_t,
        "y": y,
    }
    if extra:
        blk.update(extra)
    return blk


# ==============================================================================
# Strictly Causal uV Loaders
# ==============================================================================
def worker_b2a(sid: int) -> dict:
    sos_filters = {name: butter(4, [lo, hi], btype="bandpass", fs=b2a.SFREQ, output="sos") for name, (lo, hi) in b2a.BANDS}
    s0 = int(round((b2a.CUE_OFFSET_SEC + b2a.POST_CUE_TMIN) * b2a.SFREQ))
    s1 = int(round((b2a.CUE_OFFSET_SEC + b2a.POST_CUE_TMAX) * b2a.SFREQ)) + 1
    out = {"sid": sid}
    for sess_name in ("T", "E"):
        mat = sio.loadmat(b2a.DATA_DIR / f"A{sid:02d}{sess_name}.mat", simplify_cells=True)
        buf = {"f": [], "h": []}
        ys = []
        for run_dict in mat["data"]:
            tr = np.asarray(run_dict["trial"], dtype=np.int64)
            if tr.size == 0:
                continue
            X_eeg = np.asarray(run_dict["X"], dtype=np.float64)[:, : b2a.N_EEG_CHANS]
            X_car = (X_eeg - X_eeg.mean(axis=1, keepdims=True)).T
            idx0 = tr - 1
            ys.append(np.asarray(run_dict["y"], dtype=np.int64))
            for bname, tag in (("full", "f"), ("high_beta", "h")):
                f_ca = causal_sosfilt(sos_filters[bname], X_car)
                buf[tag].append(np.stack([f_ca[:, i + s0 : i + s1] for i in idx0], axis=0))
        out[sess_name] = pack_ca_block(np.concatenate(buf["f"], 0), np.concatenate(buf["h"], 0), np.concatenate(ys, 0))
    return out


def worker_chang(subj_dir: Path) -> dict:
    sid = subj_dir.name
    mi_files = sorted(subj_dir.glob(f"{sid}_MI[0-9]*.set"))
    sos_filters = {name: butter(4, [lo, hi], btype="bandpass", fs=ch25.ORIG_SFREQ, output="sos") for name, (lo, hi) in ch25.BANDS}
    s0 = int(round(ch25.TMIN * ch25.SFREQ))
    s1 = int(round(ch25.TMAX * ch25.SFREQ)) + 1
    buf = {"f": [], "h": []}
    ys = []
    for sp in mi_files:
        eeg = sio.loadmat(sp, simplify_cells=True)["EEG"]
        n_ch, n_pts = int(eeg["nbchan"]), int(eeg["pnts"])
        ch_names = [c["labels"] for c in eeg["chanlocs"]]
        pick_idx = [ch_names.index(c) for c in ch25.CORE12_CHANNELS]
        raw = np.fromfile(sp.with_suffix(".fdt"), dtype=np.float32).reshape((n_ch, n_pts), order="F").astype(np.float64)
        raw = raw - raw.mean(axis=0, keepdims=True)
        raw_core = raw[pick_idx]
        events = eeg["event"]
        if isinstance(events, dict):
            events = [events]
        lats, types = [], []
        for ev in events:
            tv = int(ev["type"])
            if tv in (1, 2, 3):
                s250 = int(round((float(ev["latency"]) - 1.0) / ch25.DECIM))
                if s250 + s1 <= (n_pts // ch25.DECIM):
                    lats.append(s250)
                    types.append(tv)
        ys.append(np.asarray(types, dtype=np.int64))
        for bname, tag in (("full", "f"), ("high_beta", "h")):
            f_ca = causal_sosfilt(sos_filters[bname], raw_core)[:, :: ch25.DECIM]
            buf[tag].append(np.stack([f_ca[:, s + s0 : s + s1] for s in lats], axis=0))
    return pack_ca_block(np.concatenate(buf["f"], 0), np.concatenate(buf["h"], 0), np.concatenate(ys, 0), {"sid": sid})


def worker_kaya(mat_path: Path) -> dict:
    o = sio.loadmat(mat_path, simplify_cells=True)["o"]
    chnames = list(o["chnames"])
    pick_idx = [chnames.index(ch) for ch in ky18.EEG19_CHANNELS]
    data = np.asarray(o["data"], dtype=np.float64)[:, pick_idx].T
    data = data - data.mean(axis=0, keepdims=True)
    mk = np.asarray(o["marker"], dtype=np.int64)
    onsets, labels = [], []
    n_pts = data.shape[1]
    prev = 0
    for idx, val in enumerate(mk):
        if val in (1, 2, 3) and prev != val and idx + ky18.EPOCH_SAMPLES <= n_pts:
            onsets.append(idx)
            labels.append(int(val))
        prev = int(val)
    onsets_arr = np.asarray(onsets, dtype=np.int64)
    y_all = np.asarray(labels, dtype=np.int64)
    sos_filters = {name: butter(4, [lo, hi], btype="bandpass", fs=ky18.SFREQ, output="sos") for name, (lo, hi) in ky18.BANDS}
    ep = {}
    for bname, tag in (("full", "f"), ("high_beta", "h")):
        f_ca = causal_sosfilt(sos_filters[bname], data)
        ep[tag] = np.stack([f_ca[:, s : s + ky18.EPOCH_SAMPLES] for s in onsets_arr], axis=0)
    m_subj = re.search(r"Subject([A-Za-z]+)", mat_path.name)
    subj_id = m_subj.group(1) if m_subj else mat_path.stem
    return pack_ca_block(ep["f"], ep["h"], y_all, {"name": mat_path.stem, "subj": subj_id})


def worker_eegbci(sid: int) -> dict:
    mne.set_log_level("WARNING")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sub_str = f"S{sid:03d}"
    all_bands = [("full", (8.0, 30.0)), ("high_beta", (21.0, 30.0)), ("mu", (8.0, 13.0)), ("beta", (13.0, 30.0))]
    sos_filters = {name: butter(4, [lo, hi], btype="bandpass", fs=eeg_ab.TARGET_SFREQ, output="sos") for name, (lo, hi) in all_bands}
    buf = {name: [] for name, _ in all_bands}
    ys, trans_parts = [], []
    s0 = int(round(eeg_ab.TMIN * eeg_ab.TARGET_SFREQ))
    s1 = int(round(eeg_ab.TMAX * eeg_ab.TARGET_SFREQ)) + 1
    for run_pos, run in enumerate(eeg_ab.MI_RUNS):
        edf_path = eeg_ab.DATA_DIR / sub_str / f"{sub_str}R{run:02d}.edf"
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)
        if abs(raw.info["sfreq"] - eeg_ab.TARGET_SFREQ) > 1e-3:
            raw.resample(eeg_ab.TARGET_SFREQ, verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        raw.pick([ch for ch in eeg_ab.CORE_CHANNELS if ch in raw.ch_names])
        events, ev_all = mne.events_from_annotations(raw, verbose=False)
        mi_id = {k: v for k, v in ev_all.items() if k in ("T1", "T2")}
        # Convert EDF Volts -> microvolts (uV) for consistent scale across all cohorts!
        data_cont = raw.get_data() * 1e6
        t1_code = mi_id["T1"]
        vev = events[np.isin(events[:, 2], [mi_id["T1"], mi_id["T2"]])]
        vev = vev[(vev[:, 0] + s0 >= 0) & (vev[:, 0] + s1 <= data_cont.shape[1])]
        n_run = len(vev)
        ys.append((vev[:, 2] != t1_code).astype(np.int64))
        trans_parts.append((run_pos > 0) & (np.arange(n_run) < eeg_ab.TRANSITION_TRIALS_PER_RUN))
        for name in buf:
            f_ca = causal_sosfilt(sos_filters[name], data_cont)
            buf[name].append(np.stack([f_ca[:, ev[0] + s0 : ev[0] + s1] for ev in vev], axis=0))

    blk = pack_ca_block(
        np.concatenate(buf["full"], 0),
        np.concatenate(buf["high_beta"], 0),
        np.concatenate(ys, 0),
        {"sid": sid, "trans_mask": np.concatenate(trans_parts, 0)},
    )
    ep_mu = np.concatenate(buf["mu"], 0)
    ep_be = np.concatenate(buf["beta"], 0)
    cov_mu = oas_from_emp_cov(emp_cov_epochs(ep_mu), int(ep_mu.shape[-1]))
    cov_be = oas_from_emp_cov(emp_cov_epochs(ep_be), int(ep_be.shape[-1]))
    blk["cov_mu"] = cov_mu
    blk["log_mu"] = b2a.sym_logm(cov_mu)
    blk["cov_be"] = cov_be
    blk["log_be"] = b2a.sym_logm(cov_be)
    return blk


def subset_block(blk: dict, mask: np.ndarray) -> dict:
    out = {
        "cov_f": blk["cov_f"][mask],
        "cov_h": blk["cov_h"][mask],
        "log_f": blk["log_f"][mask],
        "log_h": blk["log_h"][mask],
        "emp_f": blk["emp_f"][mask],
        "emp_h": blk["emp_h"][mask],
        "n_t": blk["n_t"],
        "y": blk["y"][mask],
    }
    if "trans_mask" in blk:
        out["trans_mask"] = blk["trans_mask"][mask]
    return out


# ==============================================================================
# Helpers & Solvers
# ==============================================================================
def fit_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(solver="newton-cg", max_iter=LOGREG_MAX_ITER, C=LOGREG_C, random_state=RANDOM_STATE).fit(X, y)


def z_coral_unshrunk_src(Z_fit: np.ndarray, Z_apply: np.ndarray, src_sqrt_z: np.ndarray) -> np.ndarray:
    mean_z = Z_fit.mean(axis=0, keepdims=True)
    zc = Z_fit - mean_z
    n, d = zc.shape
    emp = (zc.T @ zc) / max(n - 1, 1) + 1e-5 * np.eye(d)
    return (Z_apply - mean_z) @ (b2a.sym_inv_sqrtm(emp) @ src_sqrt_z)


def z_coral_oas_src(Z_fit: np.ndarray, Z_apply: np.ndarray, src_sqrt_z: np.ndarray) -> np.ndarray:
    mean_z = Z_fit.mean(axis=0, keepdims=True)
    cov_z, _ = b2a.exact_oas(Z_fit - mean_z, assume_centered=True)
    return (Z_apply - mean_z) @ (b2a.sym_inv_sqrtm(cov_z) @ src_sqrt_z)


def within_adapt_s(V_fit: np.ndarray, V_apply: np.ndarray, src_cov_v: np.ndarray, src_sqrt_v: np.ndarray, scale: float) -> np.ndarray:
    mean_v = V_fit.mean(axis=0, keepdims=True)
    cov_v, _ = b2a.exact_oas(V_fit - mean_v, assume_centered=True)
    denom = max(float(np.linalg.norm(src_cov_v, ord="fro")), 1e-8)
    dist = float(np.linalg.norm(cov_v - src_cov_v, ord="fro")) / denom
    gamma = float(np.clip(np.exp(-scale * dist), 0.25, 0.92))
    cov_tilde = (1.0 - gamma) * cov_v + gamma * src_cov_v
    return (V_apply - mean_v) @ (b2a.sym_inv_sqrtm(cov_tilde) @ src_sqrt_v)


def rank_biserial_and_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 1.0, 0.0
    n = len(a)
    res = wilcoxon(a, b, method="exact" if n <= 25 else "auto")
    # Rank biserial: (W+ - W-) / (n_nz*(n_nz+1)/2)
    ranks = scipy.stats.rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    r_rb = (w_plus - w_minus) / (len(nz) * (len(nz) + 1) / 2.0)
    return float(res.pvalue), float(r_rb)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    prev_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, max(prev_max, pvals[idx] * (m - rank)))
        adj[idx] = val
        prev_max = val
    return adj.tolist()


# ==============================================================================
# Core Fold Evaluator (trials[15:] + optional cold-start [0:] + EEGBCI extra stats)
# ==============================================================================
def eval_fold_main(train_blocks: list[dict], test_blk: dict, return_probs: bool = False) -> dict:
    y_tr = np.concatenate([b["y"] for b in train_blocks], axis=0)
    y_te_all = test_blk["y"]
    y_te = y_te_all[EVAL_START_IDX:]
    res = {}

    # 1. Uncal (K=0 baseline)
    cov_tr_f = np.concatenate([b["cov_f"] for b in train_blocks], axis=0)
    cov_tr_h = np.concatenate([b["cov_h"] for b in train_blocks], axis=0)
    m_src_log_f = np.concatenate([b["log_f"] for b in train_blocks], axis=0).mean(axis=0)
    m_src_log_h = np.concatenate([b["log_h"] for b in train_blocks], axis=0).mean(axis=0)
    m_tr_f = b2a.sym_expm(m_src_log_f)
    m_tr_h = b2a.sym_expm(m_src_log_h)
    tk0_f = b2a.project_tangent(cov_tr_f, m_tr_f)
    tk0_h = b2a.project_tangent(cov_tr_h, m_tr_h)
    pca_k0_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(tk0_f)
    pca_k0_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(tk0_h)
    X_tr_k0 = np.concatenate([pca_k0_f.transform(tk0_f), pca_k0_h.transform(tk0_h)], axis=1)
    X_te_k0_all = np.concatenate([
        pca_k0_f.transform(b2a.project_tangent(test_blk["cov_f"], m_tr_f)),
        pca_k0_h.transform(b2a.project_tangent(test_blk["cov_h"], m_tr_h)),
    ], axis=1)
    sc_k0 = StandardScaler().fit(X_tr_k0)
    clf_k0 = fit_lr(sc_k0.transform(X_tr_k0), y_tr)
    prob_k0_all = clf_k0.predict_proba(sc_k0.transform(X_te_k0_all))
    pred_k0_all = clf_k0.classes_[np.argmax(prob_k0_all, axis=1)]
    res["Uncal"] = float(np.mean(pred_k0_all[EVAL_START_IDX:] == y_te))
    res["CS__Uncal__all"] = float(np.mean(pred_k0_all == y_te_all))
    res["CS__Uncal__burn"] = float(np.mean(pred_k0_all[:EVAL_START_IDX] == y_te_all[:EVAL_START_IDX]))

    # 2. Source block recentering
    rec_blocks = [
        (
            b2a.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(axis=0))),
            b2a.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(axis=0))),
        )
        for b in train_blocks
    ]
    tf_all = np.concatenate([r[0] for r in rec_blocks], axis=0)
    th_all = np.concatenate([r[1] for r in rec_blocks], axis=0)

    # Direct full tangent
    T_tr_dir = np.concatenate([tf_all, th_all], axis=1)
    sc_dir = StandardScaler().fit(T_tr_dir)
    clf_dir = fit_lr(sc_dir.transform(T_tr_dir), y_tr)

    # Causal RPA
    rpa_sf = float(np.sqrt(np.mean(np.sum(tf_all ** 2, axis=1))))
    rpa_sh = float(np.sqrt(np.mean(np.sum(th_all ** 2, axis=1))))
    rpa_tr = []
    for rf, rh in rec_blocks:
        df = max(float(np.sqrt(np.mean(np.sum(rf ** 2, axis=1)))), 1e-8)
        dh = max(float(np.sqrt(np.mean(np.sum(rh ** 2, axis=1)))), 1e-8)
        rpa_tr.append((rf * (rpa_sf / df), rh * (rpa_sh / dh)))
    pca_rpaf = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate([r[0] for r in rpa_tr]))
    pca_rpah = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate([r[1] for r in rpa_tr]))
    X_tr_rpa = np.concatenate([np.concatenate([pca_rpaf.transform(rf), pca_rpah.transform(rh)], 1) for rf, rh in rpa_tr], 0)
    sc_rpa = StandardScaler().fit(X_tr_rpa)
    clf_rpa = fit_lr(sc_rpa.transform(X_tr_rpa), y_tr)

    # PCA-30 subspace
    pca_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(tf_all)
    pca_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(th_all)
    zf_list = [pca_f.transform(r[0]) for r in rec_blocks]
    zh_list = [pca_h.transform(r[1]) for r in rec_blocks]
    zf_all = np.concatenate(zf_list, axis=0)
    zh_all = np.concatenate(zh_list, axis=0)
    sc_f = StandardScaler().fit(zf_all)
    sc_h = StandardScaler().fit(zh_all)

    src_cov_z_f, _ = b2a.exact_oas(zf_all - zf_all.mean(axis=0, keepdims=True), assume_centered=True)
    src_cov_z_h, _ = b2a.exact_oas(zh_all - zh_all.mean(axis=0, keepdims=True), assume_centered=True)
    src_sqrt_z_f, src_sqrt_z_h = uni.sym_sqrtm(src_cov_z_f), uni.sym_sqrtm(src_cov_z_h)

    wf_inv, sf_cov_v, sf_sqrt_v = uni.fit_within_operator(zf_all, y_tr)
    wh_inv, sh_cov_v, sh_sqrt_v = uni.fit_within_operator(zh_all, y_tr)

    pca_methods = [
        "G1", "Orig", "z-CORAL (Unshrunk-Src)", "z-CORAL (OAS-Src)",
        "Adapt-CORAL", "Within-CORAL(0.7)", "Within-Adapt",
    ]
    X_tr = {m: [] for m in pca_methods}
    for zf, zh in zip(zf_list, zh_list):
        uf, uh = sc_f.transform(zf), sc_h.transform(zh)
        vf, vh = zf @ wf_inv, zh @ wh_inv
        X_tr["G1"].append(np.concatenate([uf, uh], 1))
        X_tr["Orig"].append(np.concatenate([b2a.coral_fit_transform(zf, zf), b2a.coral_fit_transform(zh, zh)], 1))
        X_tr["z-CORAL (Unshrunk-Src)"].append(np.concatenate([z_coral_unshrunk_src(zf, zf, src_sqrt_z_f), z_coral_unshrunk_src(zh, zh, src_sqrt_z_h)], 1))
        X_tr["z-CORAL (OAS-Src)"].append(np.concatenate([z_coral_oas_src(zf, zf, src_sqrt_z_f), z_coral_oas_src(zh, zh, src_sqrt_z_h)], 1))
        X_tr["Adapt-CORAL"].append(np.concatenate([uni.u_adapt_transform(uf, uf), uni.u_adapt_transform(uh, uh)], 1))
        X_tr["Within-CORAL(0.7)"].append(np.concatenate([uni.within_fixed_transform(vf, vf, 0.7), uni.within_fixed_transform(vh, vh, 0.7)], 1))
        X_tr["Within-Adapt"].append(np.concatenate([within_adapt_s(vf, vf, sf_cov_v, sf_sqrt_v, 1.0), within_adapt_s(vh, vh, sh_cov_v, sh_sqrt_v, 1.0)], 1))

    clfs = {m: fit_lr(np.concatenate(X_tr[m], 0), y_tr) for m in pca_methods}

    # Online EA + TangLR (uV scaled!)
    n_t = int(test_blk["n_t"])
    n_ch = test_blk["emp_f"].shape[-1]
    eye_ch = np.eye(n_ch, dtype=np.float64)
    ea_tr_f, ea_tr_h = [], []
    for b in train_blocks:
        rf = b2a.sym_inv_sqrtm(b["emp_f"].mean(axis=0) + 1e-6 * eye_ch)
        rh = b2a.sym_inv_sqrtm(b["emp_h"].mean(axis=0) + 1e-6 * eye_ch)
        ea_tr_f.append(oas_from_emp_cov(rf @ b["emp_f"] @ rf, int(b["n_t"])))
        ea_tr_h.append(oas_from_emp_cov(rh @ b["emp_h"] @ rh, int(b["n_t"])))
    ea_cov_tr_f, ea_cov_tr_h = np.concatenate(ea_tr_f, 0), np.concatenate(ea_tr_h, 0)
    m_ea_f, m_ea_h = b2a.sym_expm(b2a.sym_logm(ea_cov_tr_f).mean(0)), b2a.sym_expm(b2a.sym_logm(ea_cov_tr_h).mean(0))
    tea_f, tea_h = b2a.project_tangent(ea_cov_tr_f, m_ea_f), b2a.project_tangent(ea_cov_tr_h, m_ea_h)
    pca_eaf = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(tea_f)
    pca_eah = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(tea_h)
    X_tr_ea = np.concatenate([pca_eaf.transform(tea_f), pca_eah.transform(tea_h)], 1)
    sc_ea = StandardScaler().fit(X_tr_ea)
    clf_ea = fit_lr(sc_ea.transform(X_tr_ea), y_tr)

    # Full stream features t=0..n_te-1 for Uncal/G1/Orig/Within-Adapt AND t>=15 for all methods
    n_te = len(y_te_all)
    n_eval = n_te - EVAL_START_IDX
    X_te = {m: np.empty((n_eval, PCA_DIM * 2), dtype=np.float64) for m in pca_methods}
    X_te_dir = np.empty((n_eval, tf_all.shape[1] * 2), dtype=np.float64)
    X_te_rpa = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)
    ea_te_cov_f = np.empty((n_eval, n_ch, n_ch), dtype=np.float64)
    ea_te_cov_h = np.empty((n_eval, n_ch, n_ch), dtype=np.float64)

    # Cold-start full stream arrays (t=0..n_te-1)
    CS_G1 = np.empty((n_te, PCA_DIM * 2), dtype=np.float64)
    CS_Orig = np.empty((n_te, PCA_DIM * 2), dtype=np.float64)
    CS_WA = np.empty((n_te, PCA_DIM * 2), dtype=np.float64)

    cov_f, cov_h = test_blk["cov_f"], test_blk["cov_h"]
    log_f, log_h = test_blk["log_f"], test_blk["log_h"]
    te_emp_f, te_emp_h = test_blk["emp_f"], test_blk["emp_h"]

    for t in range(n_te):
        if t == 0:
            mf, mh = b2a.sym_expm(m_src_log_f), b2a.sym_expm(m_src_log_h)
            zf0 = pca_f.transform(b2a.project_tangent(cov_f[0:1], mf))
            zh0 = pca_h.transform(b2a.project_tangent(cov_h[0:1], mh))
            CS_G1[0] = np.concatenate([sc_f.transform(zf0)[0], sc_h.transform(zh0)[0]])
            CS_Orig[0] = CS_G1[0]
            CS_WA[0] = np.concatenate([(zf0 @ wf_inv)[0], (zh0 @ wh_inv)[0]])
            continue

        if t < CAUSAL_WINDOW:
            mf = b2a.sym_expm((log_f[:t].sum(0) + TAU1_PRIOR * m_src_log_f) / (t + TAU1_PRIOR))
            mh = b2a.sym_expm((log_h[:t].sum(0) + TAU1_PRIOR * m_src_log_h) / (t + TAU1_PRIOR))
            start = 0
        else:
            mf = b2a.sym_expm(log_f[t - CAUSAL_WINDOW : t].mean(0))
            mh = b2a.sym_expm(log_h[t - CAUSAL_WINDOW : t].mean(0))
            start = t - CAUSAL_WINDOW

        tf_win = b2a.project_tangent(cov_f[start : t + 1], mf)
        th_win = b2a.project_tangent(cov_h[start : t + 1], mh)
        zf, zh = pca_f.transform(tf_win), pca_h.transform(th_win)
        uf, uh = sc_f.transform(zf), sc_h.transform(zh)
        vf, vh = zf @ wf_inv, zh @ wh_inv

        CS_G1[t] = np.concatenate([uf[-1], uh[-1]])
        if t < 2:
            shrink_m = t / (t + 2.0)
            CS_Orig[t] = np.concatenate([zf[-1] - shrink_m * zf[:-1].mean(0), zh[-1] - shrink_m * zh[:-1].mean(0)])
            CS_WA[t] = np.concatenate([vf[-1] - shrink_m * vf[:-1].mean(0), vh[-1] - shrink_m * vh[:-1].mean(0)])
        elif t < CAUSAL_WINDOW:
            CS_Orig[t] = np.concatenate([b2a.coral_fit_transform(zf[:-1], zf[-1:])[0], b2a.coral_fit_transform(zh[:-1], zh[-1:])[0]])
            g_floor = 1.0 - (t / float(CAUSAL_WINDOW))
            feats_cs = []
            for v_win, s_cov, s_sqrt in ((vf, sf_cov_v, sf_sqrt_v), (vh, sh_cov_v, sh_sqrt_v)):
                mv = v_win[:-1].mean(0, keepdims=True)
                cv, _ = b2a.exact_oas(v_win[:-1] - mv, assume_centered=True)
                dist = float(np.linalg.norm(cv - s_cov, ord="fro")) / max(float(np.linalg.norm(s_cov, ord="fro")), 1e-8)
                g = max(g_floor, float(np.clip(np.exp(-1.0 * dist), 0.25, 0.92)))
                mat = b2a.sym_inv_sqrtm((1.0 - g) * cv + g * s_cov) @ s_sqrt
                feats_cs.append(((v_win[-1:] - mv) @ mat)[0])
            CS_WA[t] = np.concatenate(feats_cs)
        else:
            wa_vec = np.concatenate([within_adapt_s(vf[:-1], vf[-1:], sf_cov_v, sf_sqrt_v, 1.0)[0], within_adapt_s(vh[:-1], vh[-1:], sh_cov_v, sh_sqrt_v, 1.0)[0]])
            orig_vec = np.concatenate([b2a.coral_fit_transform(zf[:-1], zf[-1:])[0], b2a.coral_fit_transform(zh[:-1], zh[-1:])[0]])
            CS_Orig[t] = orig_vec
            CS_WA[t] = wa_vec

        if t >= EVAL_START_IDX:
            out_idx = t - EVAL_START_IDX
            X_te["G1"][out_idx] = CS_G1[t]
            X_te["Orig"][out_idx] = CS_Orig[t]
            X_te["Within-Adapt"][out_idx] = CS_WA[t]
            X_te["z-CORAL (Unshrunk-Src)"][out_idx] = np.concatenate([z_coral_unshrunk_src(zf[:-1], zf[-1:], src_sqrt_z_f)[0], z_coral_unshrunk_src(zh[:-1], zh[-1:], src_sqrt_z_h)[0]])
            X_te["z-CORAL (OAS-Src)"][out_idx] = np.concatenate([z_coral_oas_src(zf[:-1], zf[-1:], src_sqrt_z_f)[0], z_coral_oas_src(zh[:-1], zh[-1:], src_sqrt_z_h)[0]])
            X_te["Adapt-CORAL"][out_idx] = np.concatenate([uni.u_adapt_transform(uf[:-1], uf[-1:])[0], uni.u_adapt_transform(uh[:-1], uh[-1:])[0]])
            X_te["Within-CORAL(0.7)"][out_idx] = np.concatenate([uni.within_fixed_transform(vf[:-1], vf[-1:], 0.7)[0], uni.within_fixed_transform(vh[:-1], vh[-1:], 0.7)[0]])
            X_te_dir[out_idx] = np.concatenate([tf_win[-1], th_win[-1]])
            df_w = max(float(np.sqrt(np.mean(np.sum(tf_win[:-1] ** 2, axis=1)))), 1e-8)
            dh_w = max(float(np.sqrt(np.mean(np.sum(th_win[:-1] ** 2, axis=1)))), 1e-8)
            X_te_rpa[out_idx] = np.concatenate([pca_rpaf.transform(tf_win[-1:] * (rpa_sf / df_w))[0], pca_rpah.transform(th_win[-1:] * (rpa_sh / dh_w))[0]])

            s_ea = t - CAUSAL_WINDOW
            rf = b2a.sym_inv_sqrtm(te_emp_f[s_ea:t].mean(0) + 1e-6 * eye_ch)
            rh = b2a.sym_inv_sqrtm(te_emp_h[s_ea:t].mean(0) + 1e-6 * eye_ch)
            ea_te_cov_f[out_idx] = rf @ te_emp_f[t] @ rf
            ea_te_cov_h[out_idx] = rh @ te_emp_h[t] @ rh

    for m in pca_methods:
        res[m] = float(np.mean(clfs[m].predict(X_te[m]) == y_te))
    res["Direct Tang-LR"] = float(np.mean(clf_dir.predict(sc_dir.transform(X_te_dir)) == y_te))
    res["Causal RPA (W=8)"] = float(np.mean(clf_rpa.predict(sc_rpa.transform(X_te_rpa)) == y_te))

    ea_te_oas_f = oas_from_emp_cov(ea_te_cov_f, n_t)
    ea_te_oas_h = oas_from_emp_cov(ea_te_cov_h, n_t)
    X_te_ea = np.concatenate([pca_eaf.transform(b2a.project_tangent(ea_te_oas_f, m_ea_f)), pca_eah.transform(b2a.project_tangent(ea_te_oas_h, m_ea_h))], 1)
    res["Online EA + TangLR"] = float(np.mean(clf_ea.predict(sc_ea.transform(X_te_ea)) == y_te))

    # Cold-start full stream accuracy
    for name, feats, clf_obj in (("G1", CS_G1, clfs["G1"]), ("Orig", CS_Orig, clfs["Orig"]), ("Within-Adapt", CS_WA, clfs["Within-Adapt"])):
        preds = clf_obj.predict(feats)
        res[f"CS__{name}__all"] = float(np.mean(preds == y_te_all))
        res[f"CS__{name}__burn"] = float(np.mean(preds[:EVAL_START_IDX] == y_te_all[:EVAL_START_IDX]))

    if return_probs:
        # Operational regimes for EEGBCI (100% cov, 80% per-subj, Transition R08/12, Steady)
        tmask = test_blk["trans_mask"][EVAL_START_IDX:]
        for name, feats, clf_obj in (("Uncal", sc_k0.transform(X_te_k0_all[EVAL_START_IDX:]), clf_k0), ("G1", X_te["G1"], clfs["G1"]), ("Orig", X_te["Orig"], clfs["Orig"]), ("Within-Adapt", X_te["Within-Adapt"], clfs["Within-Adapt"])):
            probs = clf_obj.predict_proba(feats)
            preds = clf_obj.classes_[np.argmax(probs, axis=1)]
            confs = np.max(probs, axis=1)
            n_keep = max(1, int(round(0.80 * len(confs))))
            keep_idx = np.argsort(confs)[-n_keep:]
            res[f"REG__{name}__rej80"] = float(np.mean(preds[keep_idx] == y_te[keep_idx]))
            res[f"REG__{name}__trans"] = float(np.mean(preds[tmask] == y_te[tmask])) if tmask.any() else np.nan
            res[f"REG__{name}__steady"] = float(np.mean(preds[~tmask] == y_te[~tmask])) if (~tmask).any() else np.nan
            res[f"PROBS__{name}"] = confs
            res[f"PREDS__{name}"] = preds
        res["YTE"] = y_te

    return res


def _worker_main_task(spec: tuple) -> dict:
    kind, idx, ret_p = spec
    data = _G_STATE.get("blocks")
    if kind == "loso":
        tr, te = [data[j] for j in range(len(data)) if j != idx], data[idx]
    elif kind == "b2a_cs":
        tr, te = [_G_STATE["tr_blocks"][idx]], _G_STATE["te_blocks"][idx]
    elif kind == "b2a_loso":
        tr = [_G_STATE["all_blocks"][j][s] for j in range(9) if j != idx for s in ("T", "E")]
        te = _G_STATE["all_blocks"][idx]["E"]
    elif kind == "kaya_loso":
        subjs = _G_STATE["subjs"]
        tr, te = [data[j] for j in range(len(data)) if subjs[j] != subjs[idx]], data[idx]
    elif kind == "kaya_cs":
        tr_i, te_i = idx
        tr, te = [data[tr_i]], data[te_i]
    else:
        raise ValueError(kind)
    return eval_fold_main(tr, te, return_probs=ret_p)


def _worker_eeg_ablation(te_idx: int) -> dict[str, float]:
    eeg_subs = _G_STATE["eeg_subs"]
    tr = [eeg_subs[j] for j in range(len(eeg_subs)) if j != te_idx]
    te = eeg_subs[te_idx]
    y_tr = np.concatenate([b["y"] for b in tr], 0)
    y_te = te["y"][EVAL_START_IDX:]

    def run_band_pair(keys_tr_te: list[tuple[str, str]]) -> dict[str, float]:
        # keys_tr_te: list of (cov_key, log_key)
        pca_list, w_list, tr_feats = [], [], []
        for ck, lk in keys_tr_te:
            rec_b = [eeg_ab.project_tangent(b[ck], b2a.sym_expm(b[lk].mean(0))) for b in tr]
            pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate(rec_b))
            z_list = [pca.transform(r) for r in rec_b]
            w_inv, s_cov, s_sqrt = uni.fit_within_operator(np.concatenate(z_list), y_tr)
            pca_list.append((ck, lk, pca, w_inv, s_cov, s_sqrt))
            tr_feats.append(np.concatenate([within_adapt_s(z @ w_inv, z @ w_inv, s_cov, s_sqrt, 1.0) for z in z_list]))
        out_res = {}
        # default s=1.0
        X_tr_all = np.concatenate(tr_feats, 1)
        clf = fit_lr(X_tr_all, y_tr)
        n_te = len(te["y"])
        X_te_all = np.empty((n_te - EVAL_START_IDX, PCA_DIM * len(keys_tr_te)))
        for oi, t in enumerate(range(EVAL_START_IDX, n_te)):
            col = 0
            for ck, lk, pca, w_inv, s_cov, s_sqrt in pca_list:
                m = b2a.sym_expm(te[lk][t - CAUSAL_WINDOW : t].mean(0))
                v = pca.transform(eeg_ab.project_tangent(te[ck][t - CAUSAL_WINDOW : t + 1], m)) @ w_inv
                X_te_all[oi, col : col + PCA_DIM] = within_adapt_s(v[:-1], v[-1:], s_cov, s_sqrt, 1.0)[0]
                col += PCA_DIM
        out_res["acc"] = float(np.mean(clf.predict(X_te_all) == y_te))
        return out_res

    out = {
        "full_only": run_band_pair([("cov_f", "log_f")])["acc"],
        "hb_only": run_band_pair([("cov_h", "log_h")])["acc"],
        "disjoint_mu_beta": run_band_pair([("cov_mu", "log_mu"), ("cov_be", "log_be")])["acc"],
    }
    # Sensitivity s in {0.5, 1.0, 2.0}
    rec_f = [eeg_ab.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(0))) for b in tr]
    rec_h = [eeg_ab.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(0))) for b in tr]
    pf = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate(rec_f))
    ph = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate(rec_h))
    zf_l, zh_l = [pf.transform(r) for r in rec_f], [ph.transform(r) for r in rec_h]
    wfi, sf_c, sf_s = uni.fit_within_operator(np.concatenate(zf_l), y_tr)
    whi, sh_c, sh_s = uni.fit_within_operator(np.concatenate(zh_l), y_tr)
    for scale in (0.5, 1.0, 2.0):
        Xtr = np.concatenate([
            np.concatenate([within_adapt_s(z @ wfi, z @ wfi, sf_c, sf_s, scale), within_adapt_s(zh @ whi, zh @ whi, sh_c, sh_s, scale)], 1)
            for z, zh in zip(zf_l, zh_l)
        ], 0)
        clf = fit_lr(Xtr, y_tr)
        n_te = len(te["y"])
        Xte = np.empty((n_te - EVAL_START_IDX, PCA_DIM * 2))
        for oi, t in enumerate(range(EVAL_START_IDX, n_te)):
            mf = b2a.sym_expm(te["log_f"][t - CAUSAL_WINDOW : t].mean(0))
            mh = b2a.sym_expm(te["log_h"][t - CAUSAL_WINDOW : t].mean(0))
            vf = pf.transform(eeg_ab.project_tangent(te["cov_f"][t - CAUSAL_WINDOW : t + 1], mf)) @ wfi
            vh = ph.transform(eeg_ab.project_tangent(te["cov_h"][t - CAUSAL_WINDOW : t + 1], mh)) @ whi
            Xte[oi, :PCA_DIM] = within_adapt_s(vf[:-1], vf[-1:], sf_c, sf_s, scale)[0]
            Xte[oi, PCA_DIM:] = within_adapt_s(vh[:-1], vh[-1:], sh_c, sh_s, scale)[0]
        out[f"s_{scale}"] = float(np.mean(clf.predict(Xte) == y_te))
    return out


def run_wsweep(blocks_list: list[list[dict]], test_blocks: list[dict], windows: list[int], proj_fn) -> dict[str, dict[int, np.ndarray]]:
    n_sub = len(test_blocks)
    methods = ["G1", "Orig", "z-CORAL (OAS-Src)", "Within-Adapt"]
    accs = {m: {w: np.empty(n_sub, dtype=np.float64) for w in windows} for m in methods}
    for i in range(n_sub):
        tr_b = [b for j in range(n_sub) if j != i for b in blocks_list[j]]
        te = test_blocks[i]
        y_tr = np.concatenate([b["y"] for b in tr_b], 0)
        y_te = te["y"][EVAL_START_IDX:]
        rec_b = [(proj_fn(b["cov_f"], b2a.sym_expm(b["log_f"].mean(0))), proj_fn(b["cov_h"], b2a.sym_expm(b["log_h"].mean(0)))) for b in tr_b]
        pf = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate([r[0] for r in rec_b]))
        ph = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate([r[1] for r in rec_b]))
        zf_l, zh_l = [pf.transform(r[0]) for r in rec_b], [ph.transform(r[1]) for r in rec_b]
        zf_all, zh_all = np.concatenate(zf_l), np.concatenate(zh_l)
        scf, sch = StandardScaler().fit(zf_all), StandardScaler().fit(zh_all)
        sz_f, _ = b2a.exact_oas(zf_all - zf_all.mean(0, keepdims=True), True)
        sz_h, _ = b2a.exact_oas(zh_all - zh_all.mean(0, keepdims=True), True)
        szf_s, szh_s = uni.sym_sqrtm(sz_f), uni.sym_sqrtm(sz_h)
        wfi, sfc, sfs = uni.fit_within_operator(zf_all, y_tr)
        whi, shc, shs = uni.fit_within_operator(zh_all, y_tr)
        Xtr = {m: [] for m in methods}
        for zf, zh in zip(zf_l, zh_l):
            vf, vh = zf @ wfi, zh @ whi
            Xtr["G1"].append(np.concatenate([scf.transform(zf), sch.transform(zh)], 1))
            Xtr["Orig"].append(np.concatenate([b2a.coral_fit_transform(zf, zf), b2a.coral_fit_transform(zh, zh)], 1))
            Xtr["z-CORAL (OAS-Src)"].append(np.concatenate([z_coral_oas_src(zf, zf, szf_s), z_coral_oas_src(zh, zh, szh_s)], 1))
            Xtr["Within-Adapt"].append(np.concatenate([within_adapt_s(vf, vf, sfc, sfs, 1.0), within_adapt_s(vh, vh, shc, shs, 1.0)], 1))
        clfs = {m: fit_lr(np.concatenate(Xtr[m], 0), y_tr) for m in methods}
        n_te = len(te["y"])
        for w in windows:
            Xte = {m: np.empty((n_te - EVAL_START_IDX, PCA_DIM * 2)) for m in methods}
            for oi, t in enumerate(range(EVAL_START_IDX, n_te)):
                s = max(0, t - w)
                mf, mh = b2a.sym_expm(te["log_f"][s:t].mean(0)), b2a.sym_expm(te["log_h"][s:t].mean(0))
                zf, zh = pf.transform(proj_fn(te["cov_f"][s : t + 1], mf)), ph.transform(proj_fn(te["cov_h"][s : t + 1], mh))
                vf, vh = zf @ wfi, zh @ whi
                Xte["G1"][oi] = np.concatenate([scf.transform(zf)[-1], sch.transform(zh)[-1]])
                Xte["Orig"][oi] = np.concatenate([b2a.coral_fit_transform(zf[:-1], zf[-1:])[0], b2a.coral_fit_transform(zh[:-1], zh[-1:])[0]])
                Xte["z-CORAL (OAS-Src)"][oi] = np.concatenate([z_coral_oas_src(zf[:-1], zf[-1:], szf_s)[0], z_coral_oas_src(zh[:-1], zh[-1:], szh_s)[0]])
                Xte["Within-Adapt"][oi] = np.concatenate([within_adapt_s(vf[:-1], vf[-1:], sfc, sfs, 1.0)[0], within_adapt_s(vh[:-1], vh[-1:], shc, shs, 1.0)[0]])
            for m in methods:
                accs[m][w][i] = float(np.mean(clfs[m].predict(Xte[m]) == y_te))
    return accs


def benchmark_e2e_cpu_latency(n_ch: int, n_samples: int, sfreq: float, name: str) -> dict[str, float]:
    rng = np.random.default_rng(42)
    sos_f = butter(4, [8.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
    sos_h = butter(4, [21.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
    zi_f = np.repeat(sosfilt_zi(sos_f)[:, None, :], n_ch, axis=1)
    zi_h = np.repeat(sosfilt_zi(sos_h)[:, None, :], n_ch, axis=1)

    d_tan = n_ch * (n_ch + 1) // 2
    pca_f = rng.normal(size=(d_tan, PCA_DIM))
    pca_h = rng.normal(size=(d_tan, PCA_DIM))
    wfi, whi = np.eye(PCA_DIM), np.eye(PCA_DIM)
    sf_c, sh_c = np.eye(PCA_DIM), np.eye(PCA_DIM)
    sf_s, sh_s = np.eye(PCA_DIM), np.eye(PCA_DIM)
    coef, intercept = rng.normal(size=(60,)), 0.0

    # Pre-fill W=8 ring buffers
    cov_buf_f = [np.eye(n_ch) + 0.05 * rng.normal(size=(n_ch, n_ch)) for _ in range(CAUSAL_WINDOW + 1)]
    cov_buf_f = [0.5 * (c + c.T) for c in cov_buf_f]
    cov_buf_h = [c.copy() for c in cov_buf_f]
    log_buf_f = [b2a.sym_logm(c[None])[0] for c in cov_buf_f]
    log_buf_h = [b2a.sym_logm(c[None])[0] for c in cov_buf_h]

    raw_ep = rng.normal(size=(n_ch, n_samples)) * 10.0

    lat_e2e, lat_geom = [], []
    for it in range(550):
        t0 = time.perf_counter()
        # Step 1: CAR + Causal sosfilt (both bands) + OAS + Logm
        x_car = raw_ep - raw_ep.mean(axis=0, keepdims=True)
        ep_f, zi_f = sosfilt(sos_f, x_car, axis=-1, zi=zi_f)
        ep_h, zi_h = sosfilt(sos_h, x_car, axis=-1, zi=zi_h)
        cf, _ = b2a.exact_oas(ep_f.T, assume_centered=False)
        ch, _ = b2a.exact_oas(ep_h.T, assume_centered=False)
        lf = b2a.sym_logm(cf[None])[0]
        lh = b2a.sym_logm(ch[None])[0]

        t_geom0 = time.perf_counter()
        # Step 2: Update ring buffer, W-window Log-Euclidean mean, congruence transport, PCA, v-space CORAL, LR
        cov_arr_f = np.stack(cov_buf_f[-CAUSAL_WINDOW:] + [cf], axis=0)
        cov_arr_h = np.stack(cov_buf_h[-CAUSAL_WINDOW:] + [ch], axis=0)
        mf = b2a.sym_expm(np.mean(log_buf_f[-CAUSAL_WINDOW:], axis=0))
        mh = b2a.sym_expm(np.mean(log_buf_h[-CAUSAL_WINDOW:], axis=0))
        vf = (b2a.project_tangent(cov_arr_f, mf) @ pca_f) @ wfi
        vh = (b2a.project_tangent(cov_arr_h, mh) @ pca_h) @ whi
        out_f = within_adapt_s(vf[:-1], vf[-1:], sf_c, sf_s, 1.0)[0]
        out_h = within_adapt_s(vh[:-1], vh[-1:], sh_c, sh_s, 1.0)[0]
        _ = float(np.dot(np.concatenate([out_f, out_h]), coef) + intercept)
        t_end = time.perf_counter()

        cov_buf_f.append(cf); cov_buf_f.pop(0)
        cov_buf_h.append(ch); cov_buf_h.pop(0)
        log_buf_f.append(lf); log_buf_f.pop(0)
        log_buf_h.append(lh); log_buf_h.pop(0)

        if it >= 50:
            lat_e2e.append((t_end - t0) * 1000.0)
            lat_geom.append((t_end - t_geom0) * 1000.0)

    a_e2e, a_geom = np.asarray(lat_e2e), np.asarray(lat_geom)
    print(
        f"  CPU Latency [{name}]: End-to-End (CAR+sosfilt+OAS+Logm+WithinAdapt+LR) = "
        f"{a_e2e.mean():.2f}+-{a_e2e.std():.2f} ms (P50={np.median(a_e2e):.2f}, P95={np.percentile(a_e2e,95):.2f}) | "
        f"Geom+LR only = {a_geom.mean():.2f}+-{a_geom.std():.2f} ms (P95={np.percentile(a_geom,95):.2f})"
    )
    return {
        "e2e_mean": float(a_e2e.mean()), "e2e_std": float(a_e2e.std()), "e2e_p50": float(np.median(a_e2e)), "e2e_p95": float(np.percentile(a_e2e, 95)),
        "geom_mean": float(a_geom.mean()), "geom_std": float(a_geom.std()), "geom_p50": float(np.median(a_geom)), "geom_p95": float(np.percentile(a_geom, 95)),
    }


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 96)
    print("STEP-2 MASTER CAUSAL BENCHMARK | Unified uV + Strict Causal sosfilt + newton-cg (5000)")
    print(f"Host: {platform.processor()} | CPU count: {os.cpu_count()} | PyTorch/Sklearn={sklearn.__version__}, SciPy={scipy.__version__}, NumPy={np.__version__}")
    print("=" * 96)

    with ProcessPoolExecutor() as ex:
        b2a_subs = list(ex.map(worker_b2a, range(1, 10)))
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor() as ex:
        ch_subs = list(ex.map(worker_chang, ch_dirs))
    mat_files = sorted(ky18.DATA_DIR.glob("*.mat"))
    with ProcessPoolExecutor() as ex:
        ky_sess = list(ex.map(worker_kaya, mat_files))
    with ProcessPoolExecutor() as ex:
        eeg_subs = list(ex.map(worker_eegbci, range(1, 110)))
    print(f"Loaded ALL 4 cohorts (uV + causal sosfilt) in {time.perf_counter() - t0:.1f}s")

    store = {}
    all_main_methods = [
        "Uncal", "G1", "Causal RPA (W=8)", "Direct Tang-LR",
        "Orig", "z-CORAL (Unshrunk-Src)", "z-CORAL (OAS-Src)",
        "Adapt-CORAL", "Within-CORAL(0.7)", "Within-Adapt", "Online EA + TangLR",
    ]

    # Run all 10 protocols
    proto_specs = []
    # 1. EEGBCI
    with ProcessPoolExecutor(initializer=_init_state, initargs=({"blocks": eeg_subs},)) as ex:
        eeg_rows = list(ex.map(_worker_main_task, [("loso", i, True) for i in range(109)]))
    proto_specs.append(("eegbci_bin_loso", eeg_rows, None))

    # 2-5. BCI-IV-2a
    for tag, cset, is_cs in [("2a_bin_cs", (1, 2), True), ("2a_bin_loso", (1, 2), False), ("2a_4cls_cs", (1, 2, 3, 4), True), ("2a_4cls_loso", (1, 2, 3, 4), False)]:
        if is_cs:
            tr_b = [subset_block(b2a_subs[i]["T"], np.isin(b2a_subs[i]["T"]["y"], cset)) for i in range(9)]
            te_b = [subset_block(b2a_subs[i]["E"], np.isin(b2a_subs[i]["E"]["y"], cset)) for i in range(9)]
            with ProcessPoolExecutor(initializer=_init_state, initargs=({"tr_blocks": tr_b, "te_blocks": te_b},)) as ex:
                rows = list(ex.map(_worker_main_task, [("b2a_cs", i, False) for i in range(9)]))
        else:
            all_b = [{s: subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], cset)) for s in ("T", "E")} for i in range(9)]
            with ProcessPoolExecutor(initializer=_init_state, initargs=({"all_blocks": all_b},)) as ex:
                rows = list(ex.map(_worker_main_task, [("b2a_loso", i, False) for i in range(9)]))
        proto_specs.append((tag, rows, None))

    # 6-7. Chang2025
    for tag, cset in [("chang_bin", (1, 2)), ("chang_3cls", (1, 2, 3))]:
        cf = [subset_block(s, np.isin(s["y"], cset)) for s in ch_subs]
        with ProcessPoolExecutor(initializer=_init_state, initargs=({"blocks": cf},)) as ex:
            rows = list(ex.map(_worker_main_task, [("loso", i, False) for i in range(len(cf))]))
        proto_specs.append((tag, rows, None))

    # 8-10. Kaya2018 (Subject-level aggregation)
    ky_bin = [subset_block(s, np.isin(s["y"], (1, 2))) for s in ky_sess]
    ky_bin_subj = [s["subj"] for s in ky_sess]
    with ProcessPoolExecutor(initializer=_init_state, initargs=({"blocks": ky_bin, "subjs": ky_bin_subj},)) as ex:
        rows_kbl = list(ex.map(_worker_main_task, [("kaya_loso", i, False) for i in range(len(ky_bin))]))
    proto_specs.append(("kaya_bin_loso", rows_kbl, ky_bin_subj))

    cs_pairs, cs_subjs = [], []
    for sn in sorted(set(ky_bin_subj)):
        idxs = [i for i, x in enumerate(ky_bin_subj) if x == sn]
        if len(idxs) >= 2:
            for ti in idxs[1:]:
                cs_pairs.append((idxs[0], ti))
                cs_subjs.append(sn)
    with ProcessPoolExecutor(initializer=_init_state, initargs=({"blocks": ky_bin},)) as ex:
        rows_kcs = list(ex.map(_worker_main_task, [("kaya_cs", pair, False) for pair in cs_pairs]))
    proto_specs.append(("kaya_bin_cs", rows_kcs, cs_subjs))

    ky3_idx = [i for i, s in enumerate(ky_sess) if np.all(np.isin([1, 2, 3], s["y"]))]
    ky3 = [subset_block(ky_sess[i], np.isin(ky_sess[i]["y"], (1, 2, 3))) for i in ky3_idx]
    ky3_subj = [ky_sess[i]["subj"] for i in ky3_idx]
    with ProcessPoolExecutor(initializer=_init_state, initargs=({"blocks": ky3, "subjs": ky3_subj},)) as ex:
        rows_k3 = list(ex.map(_worker_main_task, [("kaya_loso", i, False) for i in range(len(ky3))]))
    proto_specs.append(("kaya_3cls_loso", rows_k3, ky3_subj))

    # Aggregate & compute exact stats + Holm
    p_vs_uncal = []
    proto_names = []
    for ptag, rows, subj_ids in proto_specs:
        # Save raw session rows if Kaya
        if subj_ids is not None:
            for k in rows[0]:
                if isinstance(rows[0][k], float):
                    store[f"{ptag}__sess__{k}"] = np.asarray([r[k] for r in rows], dtype=np.float64)
            # Subject-level aggregation
            uniq = sorted(set(subj_ids))
            use_rows = []
            for s in uniq:
                idxs = [i for i, x in enumerate(subj_ids) if x == s]
                use_rows.append({k: float(np.mean([rows[i][k] for i in idxs])) for k in rows[0] if isinstance(rows[0][k], float)})
        else:
            use_rows = rows

        for k in use_rows[0]:
            if isinstance(use_rows[0][k], float):
                store[f"{ptag}__{k}"] = np.asarray([r[k] for r in use_rows], dtype=np.float64)

        wa = store[f"{ptag}__Within-Adapt"]
        unc = store[f"{ptag}__Uncal"]
        p_raw, r_rb = rank_biserial_and_p(wa, unc)
        p_vs_uncal.append(p_raw)
        proto_names.append(ptag)

        print(f"\n[{ptag}] (N={len(wa)})")
        print("  " + " | ".join(f"{m}: {store[f'{ptag}__{m}'].mean()*100:.2f}%" for m in all_main_methods))
        print(
            "  WA paired vs -> "
            + " | ".join(
                f"{c}: {(wa - store[f'{ptag}__{c}']).mean()*100:+.2f}% (p={rank_biserial_and_p(wa, store[f'{ptag}__{c}'])[0]:.4g})"
                for c in ["Uncal", "G1", "Orig", "z-CORAL (OAS-Src)", "Adapt-CORAL", "Within-CORAL(0.7)", "Causal RPA (W=8)", "Online EA + TangLR"]
            )
        )
        print(
            "  COLD-START [0:] | "
            + " | ".join(f"{m}: {store[f'{ptag}__CS__{m}__all'].mean()*100:.2f}% (burn {store[f'{ptag}__CS__{m}__burn'].mean()*100:.2f}%)" for m in ["Uncal", "G1", "Orig", "Within-Adapt"])
        )

    holm_p = holm_bonferroni(p_vs_uncal)
    print("\n[10-PROTOCOL HOLM-BONFERRONI SUMMARY (Within-Adapt vs Uncal)]")
    for ptag, pr, ph in zip(proto_names, p_vs_uncal, holm_p):
        wa, unc = store[f"{ptag}__Within-Adapt"], store[f"{ptag}__Uncal"]
        _, rb = rank_biserial_and_p(wa, unc)
        store[f"{ptag}__holm_p"] = np.asarray([ph])
        store[f"{ptag}__raw_p"] = np.asarray([pr])
        store[f"{ptag}__rb_r"] = np.asarray([rb])
        print(f"  {ptag:18s}: Uncal={unc.mean()*100:5.2f}% -> WA={wa.mean()*100:5.2f}% (Delta={(wa-unc).mean()*100:+6.2f}%, raw_p={pr:.4g}, holm_p={ph:.4g}, r={rb:+.3f})")

    # EEGBCI Operational Regimes summary (per-subj 80% + Transition/Steady + cohort global 80%)
    print("\n[EEGBCI OPERATIONAL REGIMES (uV + strict causal sosfilt, N=109)]")
    uncal_acc = store["eegbci_bin_loso__Uncal"]
    low_mask = uncal_acc < 0.65
    print(f"  Low-baseline count (Uncal < 65%): n={low_mask.sum()}")
    for m in ["Uncal", "G1", "Orig", "Within-Adapt"]:
        all_a = store[f"eegbci_bin_loso__{m}"]
        rej80 = np.asarray([r[f"REG__{m}__rej80"] for r in eeg_rows])
        tr_a = np.asarray([r[f"REG__{m}__trans"] for r in eeg_rows])
        st_a = np.asarray([r[f"REG__{m}__steady"] for r in eeg_rows])
        store[f"eeg_reg__{m}__low"] = all_a[low_mask]
        store[f"eeg_reg__{m}__rej80"] = rej80
        store[f"eeg_reg__{m}__trans"] = tr_a
        store[f"eeg_reg__{m}__steady"] = st_a
        print(f"  {m:14s}: Full={all_a.mean()*100:5.2f}% | Low={all_a[low_mask].mean()*100:5.2f}% | Rej80={rej80.mean()*100:5.2f}% | Trans={np.nanmean(tr_a)*100:5.2f}% | Steady={np.nanmean(st_a)*100:5.2f}%")

    # EEGBCI Internal Band & Sensitivity Ablations
    print("\n[EEGBCI INTERNAL ABLATIONS (Within-Adapt: Full vs HB vs Disjoint Mu+Beta vs Dual-Nested; s=0.5/1/2)]")
    with ProcessPoolExecutor(initializer=_init_state, initargs=({"eeg_subs": eeg_subs},)) as ex:
        ab_rows = list(ex.map(_worker_eeg_ablation, range(109)))
    wa_main = store["eegbci_bin_loso__Within-Adapt"]
    for k in ["full_only", "hb_only", "disjoint_mu_beta", "s_0.5", "s_1.0", "s_2.0"]:
        arr = np.asarray([r[k] for r in ab_rows], dtype=np.float64)
        store[f"eeg_ab__{k}"] = arr
        p_w, _ = rank_biserial_and_p(wa_main, arr)
        print(f"  {k:18s}: {arr.mean()*100:5.2f}% +- {arr.std(ddof=1)*100:4.2f}% | vs WA(Dual,s=1.0): {(wa_main-arr).mean()*100:+.2f}% (p={p_w:.4g})")

    # Multi-class W-sweep
    print("\n[MULTI-CLASS WINDOW SWEEP (uV + strict causal sosfilt)]")
    b2a_4 = [[subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2, 3, 4))) for s in ("T", "E")] for i in range(9)]
    b2a_4_te = [subset_block(b2a_subs[i]["E"], np.isin(b2a_subs[i]["E"]["y"], (1, 2, 3, 4))) for i in range(9)]
    ws_2a = run_wsweep(b2a_4, b2a_4_te, [8, 12, 16, 24, 48], b2a.project_tangent)
    for m, d_w in ws_2a.items():
        for w, v in d_w.items():
            store[f"wsweep_2a4__{m}__W{w}"] = v
        print(f"  2a 4-Cls {m:18s}: " + " | ".join(f"W={w}:{d_w[w].mean()*100:5.2f}%" for w in [8, 12, 16, 24, 48]))

    ch_3 = [[subset_block(s, np.isin(s["y"], (1, 2, 3)))] for s in ch_subs]
    ch_3_te = [subset_block(s, np.isin(s["y"], (1, 2, 3))) for s in ch_subs]
    min_len_ch3 = min(len(x["y"]) for x in ch_3_te)
    print(f"  (Chang2025 3-Cls min trials per subject = {min_len_ch3}, capping sweep at W=24 for comparability)")
    ws_ch = run_wsweep(ch_3, ch_3_te, [8, 12, 16, 24], ch25.project_tangent)
    for m, d_w in ws_ch.items():
        for w, v in d_w.items():
            store[f"wsweep_ch3__{m}__W{w}"] = v
        print(f"  Ch 3-Cls {m:18s}: " + " | ".join(f"W={w}:{d_w[w].mean()*100:5.2f}%" for w in [8, 12, 16, 24]))

    # Standalone 109-fold CPU timing (post-loading fitting + evaluation)
    t_s = time.perf_counter()
    with ProcessPoolExecutor(initializer=_init_state, initargs=({"blocks": eeg_subs},)) as ex:
        _ = list(ex.map(_worker_main_task, [("loso", i, False) for i in range(109)]))
    print(f"\n[CPU BENCHMARKS] Parallel 109-fold 11-method suite runtime: {time.perf_counter() - t_s:.2f} s")
    lat_13 = benchmark_e2e_cpu_latency(13, 481, 160.0, "EEGBCI 13ch 160Hz")
    lat_22 = benchmark_e2e_cpu_latency(22, 751, 250.0, "BCI-IV-2a 22ch 250Hz")
    for k, v in lat_13.items():
        store[f"lat_13__{k}"] = np.asarray([v])
    for k, v in lat_22.items():
        store[f"lat_22__{k}"] = np.asarray([v])

    np.savez_compressed(OUTPUT_NPZ, **store)
    print(f"\nSaved Step-2 Master Results to {OUTPUT_NPZ} in {time.perf_counter()-t0:.1f}s total.")


if __name__ == "__main__":
    main()
