"""
Step-1 Probe: Strict Causal IIR (sosfilt) vs Zero-Phase (sosfiltfilt),
Missing Baselines (z-CORAL SrcColor, Causal RPA, Direct-Tang, Online EA+CovLR),
Kaya Subject-Level Inference, Unified newton-cg Solver, and Bursty-Stream Stress Tests.
"""

from __future__ import annotations

import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import mne
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import eegbci_reviewer_ablations as eeg_ab
import kaya2018_causal_coral_repro as ky18
import unified_within_adapt_coral_benchmark as uni

OUTPUT_NPZ = config.RESULTS_DIR / "probe_step1_results.npz"

PCA_DIM = 30
CAUSAL_WINDOW = 8
EVAL_START_IDX = 15
LOGREG_C = 0.05
LOGREG_MAX_ITER = 5000
RANDOM_STATE = 42
SPD_EPS = 1e-10
BURST_LEN = 6

# Per-process global cache for Windows spawn ProcessPoolExecutor
_G_STATE: dict = {}


def _init_state(state: dict) -> None:
    global _G_STATE
    _G_STATE = state


def causal_sosfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Strictly causal 4th-order Butterworth IIR filtering initialized at steady state from x[:, 0]."""
    zi_base = sosfilt_zi(sos)  # (n_sections, 2)
    zi = zi_base[:, None, :] * x[:, :1][None, :, :]  # (n_sections, n_ch, 2)
    out, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return out


def emp_cov_epochs(epochs: np.ndarray) -> np.ndarray:
    """Unregularized sample spatial covariance (N, ch, ch) per trial after temporal centering."""
    xc = epochs - epochs.mean(axis=-1, keepdims=True)
    n_t = xc.shape[-1]
    covs = np.einsum("nct,ndt->ncd", xc, xc) / float(n_t)
    return 0.5 * (covs + np.swapaxes(covs, -1, -2))


def oas_from_emp_cov(emp_covs: np.ndarray, n_samples: int) -> np.ndarray:
    """Exact sklearn-equivalent OAS shrinkage applied directly to sample covariance matrices."""
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


def pack_dual_filter_block(
    ep_sf_f: np.ndarray,
    ep_sf_h: np.ndarray,
    ep_ci_f: np.ndarray,
    ep_ci_h: np.ndarray,
    y: np.ndarray,
    extra: dict | None = None,
) -> dict:
    """Pack both sosfiltfilt (zp) and strictly causal sosfilt (ca) covariances."""
    n_t = ep_ci_f.shape[-1]
    emp_ci_f = emp_cov_epochs(ep_ci_f)
    emp_ci_h = emp_cov_epochs(ep_ci_h)
    cov_zp_f = oas_from_emp_cov(emp_cov_epochs(ep_sf_f), n_t)
    cov_zp_h = oas_from_emp_cov(emp_cov_epochs(ep_sf_h), n_t)
    cov_ca_f = oas_from_emp_cov(emp_ci_f, n_t)
    cov_ca_h = oas_from_emp_cov(emp_ci_h, n_t)
    blk = {
        "zp": {
            "cov_f": cov_zp_f,
            "cov_h": cov_zp_h,
            "log_f": b2a.sym_logm(cov_zp_f),
            "log_h": b2a.sym_logm(cov_zp_h),
            "y": y,
        },
        "ca": {
            "cov_f": cov_ca_f,
            "cov_h": cov_ca_h,
            "log_f": b2a.sym_logm(cov_ca_f),
            "log_h": b2a.sym_logm(cov_ca_h),
            "emp_f": emp_ci_f,
            "emp_h": emp_ci_h,
            "n_t": n_t,
            "y": y,
        },
    }
    if extra:
        blk.update(extra)
    return blk


# ==============================================================================
# Dataset Dual-Filter Workers (read disk once -> both sosfiltfilt and sosfilt)
# ==============================================================================
def worker_b2a(sid: int) -> dict:
    sos_filters = {
        name: butter(4, [lo, hi], btype="bandpass", fs=b2a.SFREQ, output="sos")
        for name, (lo, hi) in b2a.BANDS
    }
    s0 = int(round((b2a.CUE_OFFSET_SEC + b2a.POST_CUE_TMIN) * b2a.SFREQ))
    s1 = int(round((b2a.CUE_OFFSET_SEC + b2a.POST_CUE_TMAX) * b2a.SFREQ)) + 1
    out = {"sid": sid}
    for sess_name in ("T", "E"):
        mat = sio.loadmat(b2a.DATA_DIR / f"A{sid:02d}{sess_name}.mat", simplify_cells=True)
        buf = {k: [] for k in ("zp_f", "zp_h", "ca_f", "ca_h")}
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
                sos = sos_filters[bname]
                f_zp = sosfiltfilt(sos, X_car, axis=-1)
                f_ca = causal_sosfilt(sos, X_car)
                buf[f"zp_{tag}"].append(np.stack([f_zp[:, i + s0 : i + s1] for i in idx0], axis=0))
                buf[f"ca_{tag}"].append(np.stack([f_ca[:, i + s0 : i + s1] for i in idx0], axis=0))
        out[sess_name] = pack_dual_filter_block(
            np.concatenate(buf["zp_f"], 0),
            np.concatenate(buf["zp_h"], 0),
            np.concatenate(buf["ca_f"], 0),
            np.concatenate(buf["ca_h"], 0),
            np.concatenate(ys, 0),
        )
    return out


def worker_chang(subj_dir: Path) -> dict:
    sid = subj_dir.name
    mi_files = sorted(subj_dir.glob(f"{sid}_MI[0-9]*.set"))
    sos_filters = {
        name: butter(4, [lo, hi], btype="bandpass", fs=ch25.ORIG_SFREQ, output="sos")
        for name, (lo, hi) in ch25.BANDS
    }
    s0 = int(round(ch25.TMIN * ch25.SFREQ))
    s1 = int(round(ch25.TMAX * ch25.SFREQ)) + 1
    buf = {k: [] for k in ("zp_f", "zp_h", "ca_f", "ca_h")}
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
            sos = sos_filters[bname]
            f_zp = sosfiltfilt(sos, raw_core, axis=-1)[:, :: ch25.DECIM]
            f_ca = causal_sosfilt(sos, raw_core)[:, :: ch25.DECIM]
            buf[f"zp_{tag}"].append(np.stack([f_zp[:, s + s0 : s + s1] for s in lats], axis=0))
            buf[f"ca_{tag}"].append(np.stack([f_ca[:, s + s0 : s + s1] for s in lats], axis=0))
    return pack_dual_filter_block(
        np.concatenate(buf["zp_f"], 0),
        np.concatenate(buf["zp_h"], 0),
        np.concatenate(buf["ca_f"], 0),
        np.concatenate(buf["ca_h"], 0),
        np.concatenate(ys, 0),
        extra={"sid": sid},
    )


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
    sos_filters = {
        name: butter(4, [lo, hi], btype="bandpass", fs=ky18.SFREQ, output="sos")
        for name, (lo, hi) in ky18.BANDS
    }
    ep = {}
    for bname, tag in (("full", "f"), ("high_beta", "h")):
        sos = sos_filters[bname]
        f_zp = sosfiltfilt(sos, data, axis=-1)
        f_ca = causal_sosfilt(sos, data)
        ep[f"zp_{tag}"] = np.stack([f_zp[:, s : s + ky18.EPOCH_SAMPLES] for s in onsets_arr], axis=0)
        ep[f"ca_{tag}"] = np.stack([f_ca[:, s : s + ky18.EPOCH_SAMPLES] for s in onsets_arr], axis=0)
    m_subj = re.search(r"Subject([A-Za-z]+)", mat_path.name)
    subj_id = m_subj.group(1) if m_subj else mat_path.stem
    return pack_dual_filter_block(
        ep["zp_f"], ep["zp_h"], ep["ca_f"], ep["ca_h"], y_all,
        extra={"name": mat_path.stem, "subj": subj_id},
    )


def worker_eegbci(sid: int) -> dict:
    mne.set_log_level("WARNING")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sub_str = f"S{sid:03d}"
    sos_filters = {
        name: butter(4, [lo, hi], btype="bandpass", fs=eeg_ab.TARGET_SFREQ, output="sos")
        for name, (lo, hi) in eeg_ab.BANDS
    }
    buf = {k: [] for k in ("zp_f", "zp_h", "ca_f", "ca_h")}
    ys = []
    s0 = int(round(eeg_ab.TMIN * eeg_ab.TARGET_SFREQ))
    s1 = int(round(eeg_ab.TMAX * eeg_ab.TARGET_SFREQ)) + 1
    for run in eeg_ab.MI_RUNS:
        edf_path = eeg_ab.DATA_DIR / sub_str / f"{sub_str}R{run:02d}.edf"
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)
        if abs(raw.info["sfreq"] - eeg_ab.TARGET_SFREQ) > 1e-3:
            raw.resample(eeg_ab.TARGET_SFREQ, verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        raw.pick([ch for ch in eeg_ab.CORE_CHANNELS if ch in raw.ch_names])
        events, ev_all = mne.events_from_annotations(raw, verbose=False)
        mi_id = {k: v for k, v in ev_all.items() if k in ("T1", "T2")}
        data_cont = raw.get_data()
        t1_code = mi_id["T1"]
        vev = events[np.isin(events[:, 2], [mi_id["T1"], mi_id["T2"]])]
        vev = vev[(vev[:, 0] + s0 >= 0) & (vev[:, 0] + s1 <= data_cont.shape[1])]
        ys.append((vev[:, 2] != t1_code).astype(np.int64))
        for bname, tag in (("full", "f"), ("high_beta", "h")):
            sos = sos_filters[bname]
            f_zp = sosfiltfilt(sos, data_cont, axis=-1)
            f_ca = causal_sosfilt(sos, data_cont)
            buf[f"zp_{tag}"].append(np.stack([f_zp[:, ev[0] + s0 : ev[0] + s1] for ev in vev], axis=0))
            buf[f"ca_{tag}"].append(np.stack([f_ca[:, ev[0] + s0 : ev[0] + s1] for ev in vev], axis=0))
    return pack_dual_filter_block(
        np.concatenate(buf["zp_f"], 0),
        np.concatenate(buf["zp_h"], 0),
        np.concatenate(buf["ca_f"], 0),
        np.concatenate(buf["ca_h"], 0),
        np.concatenate(ys, 0),
        extra={"sid": sid},
    )


def subset_block(blk: dict, mask: np.ndarray) -> dict:
    out = {
        "cov_f": blk["cov_f"][mask],
        "cov_h": blk["cov_h"][mask],
        "log_f": blk["log_f"][mask],
        "log_h": blk["log_h"][mask],
        "y": blk["y"][mask],
    }
    if "emp_f" in blk:
        out["emp_f"] = blk["emp_f"][mask]
        out["emp_h"] = blk["emp_h"][mask]
        out["n_t"] = blk["n_t"]
    return out


# ==============================================================================
# Missing Baselines & Exact Fold Evaluator (newton-cg, max_iter=5000)
# ==============================================================================
def z_coral_srccolor_unshrunk(Z_fit: np.ndarray, Z_apply: np.ndarray, src_sqrt_z: np.ndarray) -> np.ndarray:
    """Classic Sun et al. CORAL in z-space: unshrunk cov + 1e-5*I -> recolor with src_sqrt_z."""
    mean_z = Z_fit.mean(axis=0, keepdims=True)
    zc = Z_fit - mean_z
    n, d = zc.shape
    emp = (zc.T @ zc) / max(n - 1, 1) + 1e-5 * np.eye(d)
    return (Z_apply - mean_z) @ (b2a.sym_inv_sqrtm(emp) @ src_sqrt_z)


def z_coral_srccolor_oas(Z_fit: np.ndarray, Z_apply: np.ndarray, src_sqrt_z: np.ndarray) -> np.ndarray:
    """OAS-regularized CORAL in z-space WITH source covariance re-coloring."""
    mean_z = Z_fit.mean(axis=0, keepdims=True)
    cov_z, _ = b2a.exact_oas(Z_fit - mean_z, assume_centered=True)
    return (Z_apply - mean_z) @ (b2a.sym_inv_sqrtm(cov_z) @ src_sqrt_z)


def fit_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(
        solver="newton-cg", max_iter=LOGREG_MAX_ITER, C=LOGREG_C, random_state=RANDOM_STATE
    ).fit(X, y)


def eval_fold_all_methods(
    train_blocks: list[dict],
    test_blk: dict,
    include_online_ea: bool = False,
) -> dict[str, float]:
    y_tr = np.concatenate([b["y"] for b in train_blocks], axis=0)
    y_te = test_blk["y"][EVAL_START_IDX:]
    res = {}

    # 1. Baseline (Uncal / K=0)
    cov_tr_f = np.concatenate([b["cov_f"] for b in train_blocks], axis=0)
    cov_tr_h = np.concatenate([b["cov_h"] for b in train_blocks], axis=0)
    m_tr_f = b2a.sym_expm(np.concatenate([b["log_f"] for b in train_blocks], axis=0).mean(axis=0))
    m_tr_h = b2a.sym_expm(np.concatenate([b["log_h"] for b in train_blocks], axis=0).mean(axis=0))
    tk0_f = b2a.project_tangent(cov_tr_f, m_tr_f)
    tk0_h = b2a.project_tangent(cov_tr_h, m_tr_h)
    pca_k0_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(tk0_f)
    pca_k0_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(tk0_h)
    X_tr_k0 = np.concatenate([pca_k0_f.transform(tk0_f), pca_k0_h.transform(tk0_h)], axis=1)
    X_te_k0 = np.concatenate([
        pca_k0_f.transform(b2a.project_tangent(test_blk["cov_f"][EVAL_START_IDX:], m_tr_f)),
        pca_k0_h.transform(b2a.project_tangent(test_blk["cov_h"][EVAL_START_IDX:], m_tr_h)),
    ], axis=1)
    sc_k0 = StandardScaler().fit(X_tr_k0)
    clf_k0 = fit_lr(sc_k0.transform(X_tr_k0), y_tr)
    res["Uncal (K=0)"] = float(np.mean(clf_k0.predict(sc_k0.transform(X_te_k0)) == y_te))

    # 2. Source block 1st-order recentering (full tangent)
    rec_blocks = [
        (
            b2a.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(axis=0))),
            b2a.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(axis=0))),
        )
        for b in train_blocks
    ]
    tf_all = np.concatenate([r[0] for r in rec_blocks], axis=0)
    th_all = np.concatenate([r[1] for r in rec_blocks], axis=0)

    # Direct full tangent (no PCA-30, no CORAL)
    T_tr_direct = np.concatenate([tf_all, th_all], axis=1)
    sc_dir = StandardScaler().fit(T_tr_direct)
    clf_dir = fit_lr(sc_dir.transform(T_tr_direct), y_tr)

    # Causal RPA dispersion scaling per block (Rodrigues et al. 2019)
    rpa_disp_src_f = float(np.sqrt(np.mean(np.sum(tf_all ** 2, axis=1))))
    rpa_disp_src_h = float(np.sqrt(np.mean(np.sum(th_all ** 2, axis=1))))
    rpa_tr_blocks = []
    for rf, rh in rec_blocks:
        df = max(float(np.sqrt(np.mean(np.sum(rf ** 2, axis=1)))), 1e-8)
        dh = max(float(np.sqrt(np.mean(np.sum(rh ** 2, axis=1)))), 1e-8)
        rpa_tr_blocks.append((rf * (rpa_disp_src_f / df), rh * (rpa_disp_src_h / dh)))
    pca_rpa_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate([r[0] for r in rpa_tr_blocks]))
    pca_rpa_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate([r[1] for r in rpa_tr_blocks]))
    X_tr_rpa = np.concatenate([
        np.concatenate([pca_rpa_f.transform(rf), pca_rpa_h.transform(rh)], axis=1)
        for rf, rh in rpa_tr_blocks
    ], axis=0)
    sc_rpa = StandardScaler().fit(X_tr_rpa)
    clf_rpa = fit_lr(sc_rpa.transform(X_tr_rpa), y_tr)

    # PCA-30 shared subspace
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
    src_sqrt_z_f = uni.sym_sqrtm(src_cov_z_f)
    src_sqrt_z_h = uni.sym_sqrtm(src_cov_z_h)

    wf_inv, sf_cov_v, sf_sqrt_v = uni.fit_within_operator(zf_all, y_tr)
    wh_inv, sh_cov_v, sh_sqrt_v = uni.fit_within_operator(zh_all, y_tr)

    pca_methods = [
        "G1 (1st-order)",
        "OAS-CORAL (Orig)",
        "z-CORAL (Unshrunk-Src)",
        "z-CORAL (OAS-Src)",
        "Adapt-CORAL",
        "Within-CORAL(0.7)",
        "Within-Adapt",
    ]
    X_tr = {m: [] for m in pca_methods}
    for zf, zh in zip(zf_list, zh_list):
        uf, uh = sc_f.transform(zf), sc_h.transform(zh)
        vf, vh = zf @ wf_inv, zh @ wh_inv
        X_tr["G1 (1st-order)"].append(np.concatenate([uf, uh], axis=1))
        X_tr["OAS-CORAL (Orig)"].append(np.concatenate([b2a.coral_fit_transform(zf, zf), b2a.coral_fit_transform(zh, zh)], 1))
        X_tr["z-CORAL (Unshrunk-Src)"].append(np.concatenate([
            z_coral_srccolor_unshrunk(zf, zf, src_sqrt_z_f),
            z_coral_srccolor_unshrunk(zh, zh, src_sqrt_z_h),
        ], 1))
        X_tr["z-CORAL (OAS-Src)"].append(np.concatenate([
            z_coral_srccolor_oas(zf, zf, src_sqrt_z_f),
            z_coral_srccolor_oas(zh, zh, src_sqrt_z_h),
        ], 1))
        X_tr["Adapt-CORAL"].append(np.concatenate([uni.u_adapt_transform(uf, uf), uni.u_adapt_transform(uh, uh)], 1))
        X_tr["Within-CORAL(0.7)"].append(np.concatenate([
            uni.within_fixed_transform(vf, vf, 0.7),
            uni.within_fixed_transform(vh, vh, 0.7),
        ], 1))
        X_tr["Within-Adapt"].append(np.concatenate([
            uni.within_adapt_srccolor(vf, vf, sf_cov_v, sf_sqrt_v),
            uni.within_adapt_srccolor(vh, vh, sh_cov_v, sh_sqrt_v),
        ], 1))

    clfs = {m: fit_lr(np.concatenate(X_tr[m], 0), y_tr) for m in pca_methods}

    # Test loop (trials[15:])
    n_te = len(test_blk["y"])
    n_eval = n_te - EVAL_START_IDX
    X_te = {m: np.empty((n_eval, PCA_DIM * 2), dtype=np.float64) for m in pca_methods}
    X_te_dir = np.empty((n_eval, tf_all.shape[1] * 2), dtype=np.float64)
    X_te_rpa = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)
    cov_f, cov_h = test_blk["cov_f"], test_blk["cov_h"]
    log_f, log_h = test_blk["log_f"], test_blk["log_h"]

    for out_idx, t in enumerate(range(EVAL_START_IDX, n_te)):
        start = max(0, t - CAUSAL_WINDOW)
        mf = b2a.sym_expm(log_f[start:t].mean(axis=0))
        mh = b2a.sym_expm(log_h[start:t].mean(axis=0))
        tf_win = b2a.project_tangent(cov_f[start : t + 1], mf)
        th_win = b2a.project_tangent(cov_h[start : t + 1], mh)
        X_te_dir[out_idx] = np.concatenate([tf_win[-1], th_win[-1]])

        df_w = max(float(np.sqrt(np.mean(np.sum(tf_win[:-1] ** 2, axis=1)))), 1e-8)
        dh_w = max(float(np.sqrt(np.mean(np.sum(th_win[:-1] ** 2, axis=1)))), 1e-8)
        X_te_rpa[out_idx] = np.concatenate([
            pca_rpa_f.transform(tf_win[-1:] * (rpa_disp_src_f / df_w))[0],
            pca_rpa_h.transform(th_win[-1:] * (rpa_disp_src_h / dh_w))[0],
        ])

        zf = pca_f.transform(tf_win)
        zh = pca_h.transform(th_win)
        uf, uh = sc_f.transform(zf), sc_h.transform(zh)
        vf, vh = zf @ wf_inv, zh @ wh_inv

        X_te["G1 (1st-order)"][out_idx] = np.concatenate([uf[-1], uh[-1]])
        X_te["OAS-CORAL (Orig)"][out_idx] = np.concatenate([
            b2a.coral_fit_transform(zf[:-1], zf[-1:])[0],
            b2a.coral_fit_transform(zh[:-1], zh[-1:])[0],
        ])
        X_te["z-CORAL (Unshrunk-Src)"][out_idx] = np.concatenate([
            z_coral_srccolor_unshrunk(zf[:-1], zf[-1:], src_sqrt_z_f)[0],
            z_coral_srccolor_unshrunk(zh[:-1], zh[-1:], src_sqrt_z_h)[0],
        ])
        X_te["z-CORAL (OAS-Src)"][out_idx] = np.concatenate([
            z_coral_srccolor_oas(zf[:-1], zf[-1:], src_sqrt_z_f)[0],
            z_coral_srccolor_oas(zh[:-1], zh[-1:], src_sqrt_z_h)[0],
        ])
        X_te["Adapt-CORAL"][out_idx] = np.concatenate([
            uni.u_adapt_transform(uf[:-1], uf[-1:])[0],
            uni.u_adapt_transform(uh[:-1], uh[-1:])[0],
        ])
        X_te["Within-CORAL(0.7)"][out_idx] = np.concatenate([
            uni.within_fixed_transform(vf[:-1], vf[-1:], 0.7)[0],
            uni.within_fixed_transform(vh[:-1], vh[-1:], 0.7)[0],
        ])
        X_te["Within-Adapt"][out_idx] = np.concatenate([
            uni.within_adapt_srccolor(vf[:-1], vf[-1:], sf_cov_v, sf_sqrt_v)[0],
            uni.within_adapt_srccolor(vh[:-1], vh[-1:], sh_cov_v, sh_sqrt_v)[0],
        ])

    for m in pca_methods:
        res[m] = float(np.mean(clfs[m].predict(X_te[m]) == y_te))
    res["Direct Tang-LR"] = float(np.mean(clf_dir.predict(sc_dir.transform(X_te_dir)) == y_te))
    res["Causal RPA (W=8)"] = float(np.mean(clf_rpa.predict(sc_rpa.transform(X_te_rpa)) == y_te))

    # Online EA (W=8) + OAS Tangent LR (CPU linear EA comparator!)
    if include_online_ea and "emp_f" in test_blk:
        n_t = int(test_blk["n_t"])
        n_ch = test_blk["emp_f"].shape[-1]
        eye_ch = np.eye(n_ch, dtype=np.float64)
        ea_tr_f, ea_tr_h = [], []
        for b in train_blocks:
            rf = b2a.sym_inv_sqrtm(b["emp_f"].mean(axis=0) + 1e-6 * eye_ch)
            rh = b2a.sym_inv_sqrtm(b["emp_h"].mean(axis=0) + 1e-6 * eye_ch)
            ea_tr_f.append(oas_from_emp_cov(rf @ b["emp_f"] @ rf, int(b["n_t"])))
            ea_tr_h.append(oas_from_emp_cov(rh @ b["emp_h"] @ rh, int(b["n_t"])))
        ea_cov_tr_f = np.concatenate(ea_tr_f, 0)
        ea_cov_tr_h = np.concatenate(ea_tr_h, 0)
        m_ea_f = b2a.sym_expm(b2a.sym_logm(ea_cov_tr_f).mean(axis=0))
        m_ea_h = b2a.sym_expm(b2a.sym_logm(ea_cov_tr_h).mean(axis=0))
        tea_f = b2a.project_tangent(ea_cov_tr_f, m_ea_f)
        tea_h = b2a.project_tangent(ea_cov_tr_h, m_ea_h)
        pca_eaf = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(tea_f)
        pca_eah = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(tea_h)
        X_tr_ea = np.concatenate([pca_eaf.transform(tea_f), pca_eah.transform(tea_h)], 1)
        sc_ea = StandardScaler().fit(X_tr_ea)
        clf_ea = fit_lr(sc_ea.transform(X_tr_ea), y_tr)

        te_emp_f, te_emp_h = test_blk["emp_f"], test_blk["emp_h"]
        ea_te_cov_f = np.empty((n_eval, n_ch, n_ch), dtype=np.float64)
        ea_te_cov_h = np.empty((n_eval, n_ch, n_ch), dtype=np.float64)
        for out_idx, t in enumerate(range(EVAL_START_IDX, n_te)):
            s = max(0, t - CAUSAL_WINDOW)
            rf = b2a.sym_inv_sqrtm(te_emp_f[s:t].mean(axis=0) + 1e-6 * eye_ch)
            rh = b2a.sym_inv_sqrtm(te_emp_h[s:t].mean(axis=0) + 1e-6 * eye_ch)
            ea_te_cov_f[out_idx] = rf @ te_emp_f[t] @ rf
            ea_te_cov_h[out_idx] = rh @ te_emp_h[t] @ rh
        ea_te_oas_f = oas_from_emp_cov(ea_te_cov_f, n_t)
        ea_te_oas_h = oas_from_emp_cov(ea_te_cov_h, n_t)
        X_te_ea = np.concatenate([
            pca_eaf.transform(b2a.project_tangent(ea_te_oas_f, m_ea_f)),
            pca_eah.transform(b2a.project_tangent(ea_te_oas_h, m_ea_h)),
        ], 1)
        res["Online EA + TangLR"] = float(np.mean(clf_ea.predict(sc_ea.transform(X_te_ea)) == y_te))

    return res


def reorder_bursty_stream(blk: dict, burst_len: int = BURST_LEN, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    y = blk["y"]
    classes = np.unique(y)
    pools = {c: list(rng.permutation(np.where(y == c)[0])) for c in classes}
    order = []
    cls_cycle = list(classes)
    c_ptr = 0
    while any(len(pools[c]) > 0 for c in classes):
        c = cls_cycle[c_ptr % len(cls_cycle)]
        c_ptr += 1
        if not pools[c]:
            continue
        take = min(burst_len, len(pools[c]))
        order.extend(pools[c][:take])
        pools[c] = pools[c][take:]
    return subset_block(blk, np.asarray(order, dtype=np.int64))


def reorder_random_stream(blk: dict, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    return subset_block(blk, rng.permutation(len(blk["y"])))


# Top-level picklable worker functions for Windows spawn
def _task_loso_or_pair(task_spec: tuple) -> dict[str, float]:
    kind, idx, inc_ea = task_spec
    data = _G_STATE.get("blocks")
    if kind == "loso":
        tr = [data[j] for j in range(len(data)) if j != idx]
        te = data[idx]
    elif kind == "b2a_cs":
        tr = [_G_STATE["tr_blocks"][idx]]
        te = _G_STATE["te_blocks"][idx]
    elif kind == "b2a_loso":
        tr = [_G_STATE["all_blocks"][j][s] for j in range(9) if j != idx for s in ("T", "E")]
        te = _G_STATE["all_blocks"][idx]["E"]
    elif kind == "kaya_loso":
        subjs = _G_STATE["subjs"]
        tr = [data[j] for j in range(len(data)) if subjs[j] != subjs[idx]]
        te = data[idx]
    elif kind == "kaya_cs":
        tr_i, te_i = idx
        tr = [data[tr_i]]
        te = data[te_i]
    elif kind == "chang_rand":
        tr = [data[j] for j in range(len(data)) if j != idx]
        te = reorder_random_stream(data[idx], seed=100 + idx)
    elif kind == "burst_loso":
        tr = [data[j] for j in range(len(data)) if j != idx]
        te = reorder_bursty_stream(data[idx], burst_len=BURST_LEN, seed=42 + idx)
    elif kind == "burst_b2a":
        tr = [_G_STATE["all_blocks"][j][s] for j in range(9) if j != idx for s in ("T", "E")]
        te = reorder_bursty_stream(_G_STATE["all_blocks"][idx]["E"], burst_len=BURST_LEN, seed=42 + idx)
    else:
        raise ValueError(kind)
    return eval_fold_all_methods(tr, te, include_online_ea=inc_ea)


def p_str(a: np.ndarray, b: np.ndarray) -> str:
    if np.allclose(a, b):
        return "p=1.000"
    return f"p={float(wilcoxon(a, b)[1]):.4g}"


def print_block_summary(title: str, rows: list[dict[str, float]], key_order: list[str]) -> dict[str, np.ndarray]:
    arrs = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in key_order if k in rows[0]}
    print(f"\n[{title}] (N={len(rows)})")
    print("  " + " | ".join(f"{k}: {v.mean()*100:5.2f}%" for k, v in arrs.items()))
    wa = arrs["Within-Adapt"]
    comps = [k for k in ["Uncal (K=0)", "G1 (1st-order)", "OAS-CORAL (Orig)", "z-CORAL (Unshrunk-Src)", "z-CORAL (OAS-Src)", "Causal RPA (W=8)", "Adapt-CORAL", "Within-CORAL(0.7)", "Online EA + TangLR"] if k in arrs]
    print("  WA vs -> " + " | ".join(f"{c}: {(wa-arrs[c]).mean()*100:+.2f}% ({p_str(wa, arrs[c])})" for c in comps))
    return arrs


def subject_aggregate_kaya(rows: list[dict[str, float]], subj_ids: list[str], key_order: list[str]) -> dict[str, np.ndarray]:
    uniq = sorted(set(subj_ids))
    agg = []
    for s in uniq:
        idxs = [i for i, x in enumerate(subj_ids) if x == s]
        agg.append({k: float(np.mean([rows[i][k] for i in idxs])) for k in key_order if k in rows[0]})
    return print_block_summary(f"Kaya SUBJECT-LEVEL (N_subj={len(uniq)})", agg, key_order)


def run_pool(state: dict, specs: list[tuple]) -> list[dict[str, float]]:
    with ProcessPoolExecutor(initializer=_init_state, initargs=(state,)) as ex:
        return list(ex.map(_task_loso_or_pair, specs))


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 96)
    print("STEP-1 PROBE: Strict Causal sosfilt, Missing Baselines, Kaya Subject-Level, Bursty Stress")
    print("=" * 96)

    with ProcessPoolExecutor() as ex:
        b2a_subs = list(ex.map(worker_b2a, list(range(1, 10))))
    print(f"Loaded BCI-IV-2a (9 subj) in {time.perf_counter()-t_start:.1f}s")

    t0 = time.perf_counter()
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor() as ex:
        ch_subs = list(ex.map(worker_chang, ch_dirs))
    print(f"Loaded Chang2025 ({len(ch_subs)} subj: {ch_subs[0]['sid']}..{ch_subs[-1]['sid']}) in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    mat_files = sorted(ky18.DATA_DIR.glob("*.mat"))
    with ProcessPoolExecutor() as ex:
        ky_sess = list(ex.map(worker_kaya, mat_files))
    print(f"Loaded Kaya2018 ({len(ky_sess)} sess) in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    with ProcessPoolExecutor() as ex:
        eeg_subs = list(ex.map(worker_eegbci, list(range(1, 110))))
    print(f"Loaded EEGBCI (109 subj) in {time.perf_counter()-t0:.1f}s")

    all_methods = [
        "Uncal (K=0)", "G1 (1st-order)", "OAS-CORAL (Orig)",
        "z-CORAL (Unshrunk-Src)", "z-CORAL (OAS-Src)",
        "Causal RPA (W=8)", "Direct Tang-LR",
        "Adapt-CORAL", "Within-CORAL(0.7)", "Within-Adapt", "Online EA + TangLR",
    ]
    store = {}

    for mode, inc_ea in [("zp", False), ("ca", True)]:
        print("\n" + "#" * 96)
        print(f"FILTER MODE = {mode.upper()} ({'Zero-Phase sosfiltfilt' if mode=='zp' else 'STRICT CAUSAL sosfilt'}) | solver=newton-cg max_iter=5000")
        print("#" * 96)

        # BCI-IV-2a
        for proto_tag, cls_set, is_cs in [
            ("2a_bin_cs", (1, 2), True),
            ("2a_bin_loso", (1, 2), False),
            ("2a_4cls_cs", (1, 2, 3, 4), True),
            ("2a_4cls_loso", (1, 2, 3, 4), False),
        ]:
            if is_cs:
                tr_b = [subset_block(b2a_subs[i]["T"][mode], np.isin(b2a_subs[i]["T"][mode]["y"], cls_set)) for i in range(9)]
                te_b = [subset_block(b2a_subs[i]["E"][mode], np.isin(b2a_subs[i]["E"][mode]["y"], cls_set)) for i in range(9)]
                rows = run_pool({"tr_blocks": tr_b, "te_blocks": te_b}, [("b2a_cs", i, inc_ea) for i in range(9)])
            else:
                all_b = [{s: subset_block(b2a_subs[i][s][mode], np.isin(b2a_subs[i][s][mode]["y"], cls_set)) for s in ("T", "E")} for i in range(9)]
                rows = run_pool({"all_blocks": all_b}, [("b2a_loso", i, inc_ea) for i in range(9)])
            arrs = print_block_summary(f"{mode}:{proto_tag}", rows, all_methods)
            for k, v in arrs.items():
                store[f"{mode}__{proto_tag}__{k}"] = v

        # Chang2025
        for proto_tag, cls_set in [("chang_bin", (1, 2)), ("chang_3cls", (1, 2, 3))]:
            ch_filt = [subset_block(s[mode], np.isin(s[mode]["y"], cls_set)) for s in ch_subs]
            rows = run_pool({"blocks": ch_filt}, [("loso", i, inc_ea) for i in range(len(ch_filt))])
            arrs = print_block_summary(f"{mode}:{proto_tag}", rows, all_methods)
            for k, v in arrs.items():
                store[f"{mode}__{proto_tag}__{k}"] = v

        # Kaya2018
        ky_bin = [subset_block(s[mode], np.isin(s[mode]["y"], (1, 2))) for s in ky_sess]
        ky_bin_subj = [s["subj"] for s in ky_sess]
        rows = run_pool({"blocks": ky_bin, "subjs": ky_bin_subj}, [("kaya_loso", i, inc_ea) for i in range(len(ky_bin))])
        arrs = print_block_summary(f"{mode}:kaya_bin_loso (17 sess)", rows, all_methods)
        agg = subject_aggregate_kaya(rows, ky_bin_subj, all_methods)
        for k, v in arrs.items():
            store[f"{mode}__kaya_bin_loso_sess__{k}"] = v
        for k, v in agg.items():
            store[f"{mode}__kaya_bin_loso_subj__{k}"] = v

        cs_targets, cs_subjs = [], []
        for s_name in sorted(set(ky_bin_subj)):
            idxs = [i for i, x in enumerate(ky_bin_subj) if x == s_name]
            if len(idxs) >= 2:
                for ti in idxs[1:]:
                    cs_targets.append((idxs[0], ti))
                    cs_subjs.append(s_name)
        rows = run_pool({"blocks": ky_bin}, [("kaya_cs", pair, inc_ea) for pair in cs_targets])
        arrs = print_block_summary(f"{mode}:kaya_bin_cs (10 sess)", rows, all_methods)
        agg = subject_aggregate_kaya(rows, cs_subjs, all_methods)
        for k, v in arrs.items():
            store[f"{mode}__kaya_bin_cs_sess__{k}"] = v
        for k, v in agg.items():
            store[f"{mode}__kaya_bin_cs_subj__{k}"] = v

        ky3_idx = [i for i, s in enumerate(ky_sess) if np.all(np.isin([1, 2, 3], s[mode]["y"]))]
        ky3 = [subset_block(ky_sess[i][mode], np.isin(ky_sess[i][mode]["y"], (1, 2, 3))) for i in ky3_idx]
        ky3_subj = [ky_sess[i]["subj"] for i in ky3_idx]
        rows = run_pool({"blocks": ky3, "subjs": ky3_subj}, [("kaya_loso", i, inc_ea) for i in range(len(ky3))])
        arrs = print_block_summary(f"{mode}:kaya_3cls_loso (16 sess)", rows, all_methods)
        agg = subject_aggregate_kaya(rows, ky3_subj, all_methods)
        for k, v in arrs.items():
            store[f"{mode}__kaya_3cls_loso_sess__{k}"] = v
        for k, v in agg.items():
            store[f"{mode}__kaya_3cls_loso_subj__{k}"] = v

        # EEGBCI N=109
        eeg_blocks = [s[mode] for s in eeg_subs]
        rows = run_pool({"blocks": eeg_blocks}, [("loso", i, inc_ea) for i in range(109)])
        arrs = print_block_summary(f"{mode}:eegbci_bin_loso", rows, all_methods)
        for k, v in arrs.items():
            store[f"{mode}__eegbci_bin_loso__{k}"] = v

    # STAGE 3: Stress Tests under STRICT CAUSAL sosfilt
    print("\n" + "#" * 96)
    print("STRESS TESTS (STRICT CAUSAL sosfilt): Chang Random Shuffle + Bursty Block-6 Imbalance")
    print("#" * 96)
    for proto_tag, cls_set in [("chang_bin_rand", (1, 2)), ("chang_3cls_rand", (1, 2, 3))]:
        ch_filt = [subset_block(s["ca"], np.isin(s["ca"]["y"], cls_set)) for s in ch_subs]
        rows = run_pool({"blocks": ch_filt}, [("chang_rand", i, False) for i in range(len(ch_filt))])
        arrs = print_block_summary(f"ca:{proto_tag}", rows, all_methods)
        for k, v in arrs.items():
            store[f"stress__{proto_tag}__{k}"] = v

    for tag, cls_set in [("2a_bin_burst6", (1, 2)), ("2a_4cls_burst6", (1, 2, 3, 4))]:
        all_b = [{s: subset_block(b2a_subs[i][s]["ca"], np.isin(b2a_subs[i][s]["ca"]["y"], cls_set)) for s in ("T", "E")} for i in range(9)]
        rows = run_pool({"all_blocks": all_b}, [("burst_b2a", i, False) for i in range(9)])
        arrs = print_block_summary(f"ca:{tag}", rows, all_methods)
        for k, v in arrs.items():
            store[f"stress__{tag}__{k}"] = v

    eeg_ca = [s["ca"] for s in eeg_subs]
    rows = run_pool({"blocks": eeg_ca}, [("burst_loso", i, False) for i in range(109)])
    arrs = print_block_summary("ca:eegbci_burst6", rows, all_methods)
    for k, v in arrs.items():
        store[f"stress__eegbci_burst6__{k}"] = v

    np.savez_compressed(OUTPUT_NPZ, **store)
    print(f"\nSaved all Step-1 probe arrays to {OUTPUT_NPZ} in {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
