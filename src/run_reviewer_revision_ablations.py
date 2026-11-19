"""
Reviewer Revision Ablation Suite (Major Comments 1-5 Verification)
==================================================================
Runs the five strictly causal CPU ablations requested in peer review:
  1. Major 4b: Chang2025 12-channel-only CAR vs 59-channel whole-cap CAR (Bin & 3-Cls LOSO).
  2. Major 2:  Cross-session (CS) W=8 causal sliding-window training vs single-block training
               (BCI-IV-2a Bin/4-Cls CS, Kaya2018 Bin CS).
  3. Major 3:  Cached historical tangent vectors v_j(M_j) vs full-window re-projection at M_t
               (EEGBCI, BCI-IV-2a 4-Cls, Chang2025 3-Cls) + CPU timing comparison.
  4. Major 5:  Cold-start sensitivity across tau_1 in {1, 2, 4}, t=1 damping in {0, 1/3, 1},
               ramp floor 1-t/W (On/Off), and gate clip bounds [0.10,0.95]/[0.25,0.92]/[0.40,0.85]
               on [0:15] burn-in, [15:] post-burn-in, and [0:] full stream.
  5. Major 1:  Real-stream natural discrepancy stratification (within-subject top-25% high-delta_t
               vs bottom-75% stationary trials) on EEGBCI, 2a 4-Cls, and Chang 3-Cls, plus
               observed natural delta_t quantiles vs synthetic drift delta_t.

Saves all arrays to results/reviewer_revision_ablations.npz.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import numpy as np
import scipy.io as sio
from scipy.signal import butter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import kaya2018_causal_coral_repro as ky18
import run_p1_audit_updates as p1
import run_step2_riemannian_master as step2
import unified_within_adapt_coral_benchmark as uni

OUT_NPZ = config.RESULTS_DIR / "reviewer_revision_ablations.npz"

_STATE: dict = {}


def _init(state: dict) -> None:
    global _STATE
    _STATE = state


def exact_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    return p1.exact_wilcoxon_clean(a, b)


# ==============================================================================
# 1. Chang2025 12-channel-only CAR loader
# ==============================================================================
def worker_chang_car12(subj_dir: Path) -> dict:
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
        raw_core = raw[pick_idx]
        # Perform CAR strictly within the 12 selected sensorimotor channels
        raw_core = raw_core - raw_core.mean(axis=0, keepdims=True)
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
            f_ca = step2.causal_sosfilt(sos_filters[bname], raw_core)[:, :: ch25.DECIM]
            buf[tag].append(np.stack([f_ca[:, s + s0 : s + s1] for s in lats], axis=0))
    return step2.pack_ca_block(np.concatenate(buf["f"], 0), np.concatenate(buf["h"], 0), np.concatenate(ys, 0), {"sid": sid})


# ==============================================================================
# 2. CS Sliding-W=8 Training vs Single-Block Global Training
# ==============================================================================
def within_adapt_clip(
    V_fit: np.ndarray,
    V_apply: np.ndarray,
    src_cov_v: np.ndarray,
    src_sqrt_v: np.ndarray,
    scale: float = 1.0,
    g_min: float = 0.25,
    g_max: float = 0.92,
) -> tuple[np.ndarray, float, float]:
    mean_v = V_fit.mean(axis=0, keepdims=True)
    cov_v, _ = b2a.exact_oas(V_fit - mean_v, assume_centered=True)
    denom = max(float(np.linalg.norm(src_cov_v, ord="fro")), 1e-8)
    dist = float(np.linalg.norm(cov_v - src_cov_v, ord="fro")) / denom
    gamma = float(np.clip(np.exp(-scale * dist), g_min, g_max))
    cov_tilde = (1.0 - gamma) * cov_v + gamma * src_cov_v
    return (V_apply - mean_v) @ (b2a.sym_inv_sqrtm(cov_tilde) @ src_sqrt_v), dist, gamma


def stream_extract_features(
    blk: dict,
    m_src_log_f: np.ndarray,
    m_src_log_h: np.ndarray,
    pca_f: PCA,
    pca_h: PCA,
    sc_f: StandardScaler,
    sc_h: StandardScaler,
    wf_inv: np.ndarray,
    wh_inv: np.ndarray,
    sf_cov_v: np.ndarray,
    sf_sqrt_v: np.ndarray,
    sh_cov_v: np.ndarray,
    sh_sqrt_v: np.ndarray,
    tau1: float = 2.0,
    t1_alpha: float = 1.0 / 3.0,
    use_ramp: bool = True,
    g_min: float = 0.25,
    g_max: float = 0.92,
) -> tuple[np.ndarray, np.ndarray]:
    n_tr = len(blk["y"])
    G1_feat = np.empty((n_tr, step2.PCA_DIM * 2), dtype=np.float64)
    WA_feat = np.empty((n_tr, step2.PCA_DIM * 2), dtype=np.float64)
    cov_f, cov_h = blk["cov_f"], blk["cov_h"]
    log_f, log_h = blk["log_f"], blk["log_h"]
    W = step2.CAUSAL_WINDOW

    for t in range(n_tr):
        if t == 0:
            mf, mh = b2a.sym_expm(m_src_log_f), b2a.sym_expm(m_src_log_h)
            zf0 = pca_f.transform(b2a.project_tangent(cov_f[0:1], mf))
            zh0 = pca_h.transform(b2a.project_tangent(cov_h[0:1], mh))
            G1_feat[0] = np.concatenate([sc_f.transform(zf0)[0], sc_h.transform(zh0)[0]])
            WA_feat[0] = np.concatenate([(zf0 @ wf_inv)[0], (zh0 @ wh_inv)[0]])
            continue
        if t < W:
            mf = b2a.sym_expm((log_f[:t].sum(0) + tau1 * m_src_log_f) / (t + tau1))
            mh = b2a.sym_expm((log_h[:t].sum(0) + tau1 * m_src_log_h) / (t + tau1))
            start = 0
        else:
            mf = b2a.sym_expm(log_f[t - W : t].mean(0))
            mh = b2a.sym_expm(log_h[t - W : t].mean(0))
            start = t - W

        zf = pca_f.transform(b2a.project_tangent(cov_f[start : t + 1], mf))
        zh = pca_h.transform(b2a.project_tangent(cov_h[start : t + 1], mh))
        vf, vh = zf @ wf_inv, zh @ wh_inv
        G1_feat[t] = np.concatenate([sc_f.transform(zf)[-1], sc_h.transform(zh)[-1]])
        if t < 2:
            WA_feat[t] = np.concatenate([vf[-1] - t1_alpha * vf[:-1].mean(0), vh[-1] - t1_alpha * vh[:-1].mean(0)])
        elif t < W:
            g_floor = (1.0 - (t / float(W))) if use_ramp else 0.0
            parts = []
            for v_win, s_cov, s_sqrt in ((vf, sf_cov_v, sf_sqrt_v), (vh, sh_cov_v, sh_sqrt_v)):
                mv = v_win[:-1].mean(0, keepdims=True)
                cv, _ = b2a.exact_oas(v_win[:-1] - mv, assume_centered=True)
                dist = float(np.linalg.norm(cv - s_cov, ord="fro")) / max(float(np.linalg.norm(s_cov, ord="fro")), 1e-8)
                g = max(g_floor, float(np.clip(np.exp(-1.0 * dist), g_min, g_max)))
                mat = b2a.sym_inv_sqrtm((1.0 - g) * cv + g * s_cov) @ s_sqrt
                parts.append(((v_win[-1:] - mv) @ mat)[0])
            WA_feat[t] = np.concatenate(parts)
        else:
            af, _, _ = within_adapt_clip(vf[:-1], vf[-1:], sf_cov_v, sf_sqrt_v, 1.0, g_min, g_max)
            ah, _, _ = within_adapt_clip(vh[:-1], vh[-1:], sh_cov_v, sh_sqrt_v, 1.0, g_min, g_max)
            WA_feat[t] = np.concatenate([af[0], ah[0]])
    return G1_feat, WA_feat


def eval_cs_slide_fold(tr_blk: dict, te_blk: dict) -> dict[str, float]:
    y_tr = tr_blk["y"]
    y_te = te_blk["y"][step2.EVAL_START_IDX :]
    m_src_log_f = tr_blk["log_f"].mean(axis=0)
    m_src_log_h = tr_blk["log_h"].mean(axis=0)
    tf_rec = b2a.project_tangent(tr_blk["cov_f"], b2a.sym_expm(m_src_log_f))
    th_rec = b2a.project_tangent(tr_blk["cov_h"], b2a.sym_expm(m_src_log_h))

    pca_f = PCA(n_components=step2.PCA_DIM, random_state=step2.RANDOM_STATE).fit(tf_rec)
    pca_h = PCA(n_components=step2.PCA_DIM, random_state=step2.RANDOM_STATE).fit(th_rec)
    zf, zh = pca_f.transform(tf_rec), pca_h.transform(th_rec)
    sc_f, sc_h = StandardScaler().fit(zf), StandardScaler().fit(zh)
    wf_inv, sf_cov_v, sf_sqrt_v = uni.fit_within_operator(zf, y_tr)
    wh_inv, sh_cov_v, sh_sqrt_v = uni.fit_within_operator(zh, y_tr)

    # Global single-block baseline (identical to Table III)
    X_tr_g1_glob = np.concatenate([sc_f.transform(zf), sc_h.transform(zh)], 1)
    vf, vh = zf @ wf_inv, zh @ wh_inv
    X_tr_wa_glob = np.concatenate([
        step2.within_adapt_s(vf, vf, sf_cov_v, sf_sqrt_v, 1.0),
        step2.within_adapt_s(vh, vh, sh_cov_v, sh_sqrt_v, 1.0),
    ], 1)
    clf_g1_glob = step2.fit_lr(X_tr_g1_glob, y_tr)
    clf_wa_glob = step2.fit_lr(X_tr_wa_glob, y_tr)

    # Matched causal sliding-window feature extraction on training session [0:] & [W:]
    X_tr_g1_sl, X_tr_wa_sl = stream_extract_features(
        tr_blk, m_src_log_f, m_src_log_h, pca_f, pca_h, sc_f, sc_h,
        wf_inv, wh_inv, sf_cov_v, sf_sqrt_v, sh_cov_v, sh_sqrt_v,
    )
    clf_g1_sl = step2.fit_lr(X_tr_g1_sl, y_tr)
    clf_wa_sl = step2.fit_lr(X_tr_wa_sl, y_tr)
    clf_wa_sl8 = step2.fit_lr(X_tr_wa_sl[step2.CAUSAL_WINDOW :], y_tr[step2.CAUSAL_WINDOW :])

    # Test stream
    X_te_g1, X_te_wa = stream_extract_features(
        te_blk, m_src_log_f, m_src_log_h, pca_f, pca_h, sc_f, sc_h,
        wf_inv, wh_inv, sf_cov_v, sf_sqrt_v, sh_cov_v, sh_sqrt_v,
    )
    X_te_g1 = X_te_g1[step2.EVAL_START_IDX :]
    X_te_wa = X_te_wa[step2.EVAL_START_IDX :]

    return {
        "G1_glob": float(np.mean(clf_g1_glob.predict(X_te_g1) == y_te)),
        "G1_slide": float(np.mean(clf_g1_sl.predict(X_te_g1) == y_te)),
        "WA_glob": float(np.mean(clf_wa_glob.predict(X_te_wa) == y_te)),
        "WA_slide": float(np.mean(clf_wa_sl.predict(X_te_wa) == y_te)),
        "WA_slide8": float(np.mean(clf_wa_sl8.predict(X_te_wa) == y_te)),
    }


# ==============================================================================
# 3-5. Worker on LOSO Folds: Cached History vs Re-Proj, Cold-Start Ablation, High-Delta Strat
# ==============================================================================
def _worker_loso_diagnostics(idx: int) -> dict[str, float]:
    blocks = _STATE["blocks"]
    tr = [blocks[j] for j in range(len(blocks)) if j != idx]
    te = blocks[idx]
    y_tr = np.concatenate([b["y"] for b in tr], axis=0)
    y_te_all = te["y"]
    y_te = y_te_all[step2.EVAL_START_IDX :]
    n_te = len(y_te_all)

    m_src_log_f = np.concatenate([b["log_f"] for b in tr], axis=0).mean(axis=0)
    m_src_log_h = np.concatenate([b["log_h"] for b in tr], axis=0).mean(axis=0)
    rec_blocks = [
        (
            b2a.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(0))),
            b2a.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(0))),
        )
        for b in tr
    ]
    tf_all = np.concatenate([r[0] for r in rec_blocks], 0)
    th_all = np.concatenate([r[1] for r in rec_blocks], 0)
    pca_f = PCA(n_components=step2.PCA_DIM, random_state=step2.RANDOM_STATE).fit(tf_all)
    pca_h = PCA(n_components=step2.PCA_DIM, random_state=step2.RANDOM_STATE).fit(th_all)
    zf_l, zh_l = [pca_f.transform(r[0]) for r in rec_blocks], [pca_h.transform(r[1]) for r in rec_blocks]
    zf_all, zh_all = np.concatenate(zf_l, 0), np.concatenate(zh_l, 0)
    sc_f, sc_h = StandardScaler().fit(zf_all), StandardScaler().fit(zh_all)
    sz_f, _ = b2a.exact_oas(zf_all - zf_all.mean(0, keepdims=True), True)
    sz_h, _ = b2a.exact_oas(zh_all - zh_all.mean(0, keepdims=True), True)
    szf_s, szh_s = uni.sym_sqrtm(sz_f), uni.sym_sqrtm(sz_h)
    wfi, sfc, sfs = uni.fit_within_operator(zf_all, y_tr)
    whi, shc, shs = uni.fit_within_operator(zh_all, y_tr)

    X_tr_g1 = np.concatenate([np.concatenate([sc_f.transform(z), sc_h.transform(zh)], 1) for z, zh in zip(zf_l, zh_l)], 0)
    X_tr_orig = np.concatenate([np.concatenate([b2a.coral_fit_transform(z, z), b2a.coral_fit_transform(zh, zh)], 1) for z, zh in zip(zf_l, zh_l)], 0)
    X_tr_zc = np.concatenate([np.concatenate([step2.z_coral_oas_src(z, z, szf_s), step2.z_coral_oas_src(zh, zh, szh_s)], 1) for z, zh in zip(zf_l, zh_l)], 0)
    X_tr_wa = np.concatenate([np.concatenate([step2.within_adapt_s(z @ wfi, z @ wfi, sfc, sfs, 1.0), step2.within_adapt_s(zh @ whi, zh @ whi, shc, shs, 1.0)], 1) for z, zh in zip(zf_l, zh_l)], 0)

    clf_g1 = step2.fit_lr(X_tr_g1, y_tr)
    clf_orig = step2.fit_lr(X_tr_orig, y_tr)
    clf_zc = step2.fit_lr(X_tr_zc, y_tr)
    clf_wa = step2.fit_lr(X_tr_wa, y_tr)

    out = {}

    # A. Exact re-projection vs Cached past v_j(M_j) / z_j(M_j) & Natural delta_t tracking
    cov_f, cov_h = te["cov_f"], te["cov_h"]
    log_f, log_h = te["log_f"], te["log_h"]
    W = step2.CAUSAL_WINDOW

    z_cur_f = np.empty((n_te, step2.PCA_DIM), dtype=np.float64)
    z_cur_h = np.empty((n_te, step2.PCA_DIM), dtype=np.float64)
    v_cur_f = np.empty((n_te, step2.PCA_DIM), dtype=np.float64)
    v_cur_h = np.empty((n_te, step2.PCA_DIM), dtype=np.float64)

    n_eval = n_te - step2.EVAL_START_IDX
    X_g1_te = np.empty((n_eval, 60), dtype=np.float64)
    X_orig_exact = np.empty((n_eval, 60), dtype=np.float64)
    X_zc_exact = np.empty((n_eval, 60), dtype=np.float64)
    X_wa_exact = np.empty((n_eval, 60), dtype=np.float64)
    X_orig_cache = np.empty((n_eval, 60), dtype=np.float64)
    X_wa_cache = np.empty((n_eval, 60), dtype=np.float64)
    deltas = np.empty(n_eval, dtype=np.float64)

    for t in range(n_te):
        if t == 0:
            mf, mh = b2a.sym_expm(m_src_log_f), b2a.sym_expm(m_src_log_h)
            zf_t = pca_f.transform(b2a.project_tangent(cov_f[0:1], mf))[0]
            zh_t = pca_h.transform(b2a.project_tangent(cov_h[0:1], mh))[0]
            z_cur_f[0], z_cur_h[0] = zf_t, zh_t
            v_cur_f[0], v_cur_h[0] = zf_t @ wfi, zh_t @ whi
            continue
        if t < W:
            mf = b2a.sym_expm((log_f[:t].sum(0) + 2.0 * m_src_log_f) / (t + 2.0))
            mh = b2a.sym_expm((log_h[:t].sum(0) + 2.0 * m_src_log_h) / (t + 2.0))
            start = 0
        else:
            mf = b2a.sym_expm(log_f[t - W : t].mean(0))
            mh = b2a.sym_expm(log_h[t - W : t].mean(0))
            start = t - W

        zf_win = pca_f.transform(b2a.project_tangent(cov_f[start : t + 1], mf))
        zh_win = pca_h.transform(b2a.project_tangent(cov_h[start : t + 1], mh))
        vf_win, vh_win = zf_win @ wfi, zh_win @ whi
        z_cur_f[t], z_cur_h[t] = zf_win[-1], zh_win[-1]
        v_cur_f[t], v_cur_h[t] = vf_win[-1], vh_win[-1]

        if t >= step2.EVAL_START_IDX:
            oi = t - step2.EVAL_START_IDX
            X_g1_te[oi] = np.concatenate([sc_f.transform(zf_win)[-1], sc_h.transform(zh_win)[-1]])
            X_orig_exact[oi] = np.concatenate([b2a.coral_fit_transform(zf_win[:-1], zf_win[-1:])[0], b2a.coral_fit_transform(zh_win[:-1], zh_win[-1:])[0]])
            X_zc_exact[oi] = np.concatenate([step2.z_coral_oas_src(zf_win[:-1], zf_win[-1:], szf_s)[0], step2.z_coral_oas_src(zh_win[:-1], zh_win[-1:], szh_s)[0]])
            af, df, _ = within_adapt_clip(vf_win[:-1], vf_win[-1:], sfc, sfs, 1.0, 0.25, 0.92)
            ah, dh, _ = within_adapt_clip(vh_win[:-1], vh_win[-1:], shc, shs, 1.0, 0.25, 0.92)
            X_wa_exact[oi] = np.concatenate([af[0], ah[0]])
            deltas[oi] = max(df, dh)

            # Cached historical vectors computed at past M_j (j in [t-W, t-1]) + current at M_t
            X_orig_cache[oi] = np.concatenate([
                b2a.coral_fit_transform(z_cur_f[t - W : t], zf_win[-1:])[0],
                b2a.coral_fit_transform(z_cur_h[t - W : t], zh_win[-1:])[0],
            ])
            X_wa_cache[oi] = np.concatenate([
                step2.within_adapt_s(v_cur_f[t - W : t], vf_win[-1:], sfc, sfs, 1.0)[0],
                step2.within_adapt_s(v_cur_h[t - W : t], vh_win[-1:], shc, shs, 1.0)[0],
            ])

    pred_g1 = clf_g1.predict(X_g1_te) == y_te
    pred_orig = clf_orig.predict(X_orig_exact) == y_te
    pred_zc = clf_zc.predict(X_zc_exact) == y_te
    pred_wa = clf_wa.predict(X_wa_exact) == y_te
    pred_orig_c = clf_orig.predict(X_orig_cache) == y_te
    pred_wa_c = clf_wa.predict(X_wa_cache) == y_te

    out["reproj__Orig"] = float(np.mean(pred_orig))
    out["cache__Orig"] = float(np.mean(pred_orig_c))
    out["reproj__WA"] = float(np.mean(pred_wa))
    out["cache__WA"] = float(np.mean(pred_wa_c))

    # High-delta_t stratification (within-subject Top-25% vs Bottom-75% causal discrepancy)
    q75 = float(np.percentile(deltas, 75.0))
    hi = deltas >= q75
    lo = ~hi
    for tag, arr in (("G1", pred_g1), ("Orig", pred_orig), ("zCORAL", pred_zc), ("WA", pred_wa)):
        out[f"hi_delta__{tag}"] = float(np.mean(arr[hi]))
        out[f"lo_delta__{tag}"] = float(np.mean(arr[lo]))
    out["delta_p50"] = float(np.median(deltas))
    out["delta_p75"] = q75
    out["delta_p95"] = float(np.percentile(deltas, 95.0))

    # B. Cold-start pure early-trial sensitivity (t < W, post-W fixed; only evaluate t < 15 + [0:])
    cs_configs = [
        ("default", 2.0, 1.0 / 3.0, True),
        ("tau1_1", 1.0, 1.0 / 3.0, True),
        ("tau1_4", 4.0, 1.0 / 3.0, True),
        ("t1_zero", 2.0, 0.0, True),
        ("t1_full", 2.0, 1.0, True),
        ("no_ramp", 2.0, 1.0 / 3.0, False),
    ]
    for name, tau1, alpha, ramp in cs_configs:
        _, wa_stream = stream_extract_features(
            te, m_src_log_f, m_src_log_h, pca_f, pca_h, sc_f, sc_h,
            wfi, whi, sfc, sfs, shc, shs,
            tau1=tau1, t1_alpha=alpha, use_ramp=ramp, g_min=0.25, g_max=0.92,
        )
        ok = clf_wa.predict(wa_stream) == y_te_all
        out[f"cs_{name}__burn"] = float(np.mean(ok[: step2.EVAL_START_IDX]))
        out[f"cs_{name}__all"] = float(np.mean(ok))

    # C. Clip bounds sensitivity [0.10, 0.95] vs [0.25, 0.92] vs [0.40, 0.85] (including matched train)
    for cname, gmin, gmax in (("clip_wide", 0.10, 0.95), ("clip_tight", 0.40, 0.85)):
        Xtr_c = np.concatenate([
            np.concatenate([
                within_adapt_clip(z @ wfi, z @ wfi, sfc, sfs, 1.0, gmin, gmax)[0],
                within_adapt_clip(zh @ whi, zh @ whi, shc, shs, 1.0, gmin, gmax)[0],
            ], 1)
            for z, zh in zip(zf_l, zh_l)
        ], 0)
        clf_c = step2.fit_lr(Xtr_c, y_tr)
        _, wa_stream_c = stream_extract_features(
            te, m_src_log_f, m_src_log_h, pca_f, pca_h, sc_f, sc_h,
            wfi, whi, sfc, sfs, shc, shs, g_min=gmin, g_max=gmax,
        )
        ok_c = clf_c.predict(wa_stream_c) == y_te_all
        out[f"cs_{cname}__burn"] = float(np.mean(ok_c[: step2.EVAL_START_IDX]))
        out[f"cs_{cname}__all"] = float(np.mean(ok_c))
        out[f"cs_{cname}__post15"] = float(np.mean(ok_c[step2.EVAL_START_IDX :]))

    return out


def benchmark_reproj_vs_cache_cpu(n_ch: int = 22) -> dict[str, float]:
    rng = np.random.default_rng(42)
    W = step2.CAUSAL_WINDOW
    d_tan = n_ch * (n_ch + 1) // 2
    pca_f = rng.normal(size=(d_tan, step2.PCA_DIM))
    wfi = np.eye(step2.PCA_DIM)
    sfc, sfs = np.eye(step2.PCA_DIM), np.eye(step2.PCA_DIM)

    covs = np.stack([np.eye(n_ch) + 0.05 * (A + A.T) for A in rng.normal(size=(W + 1, n_ch, n_ch))], 0)
    mf = np.eye(n_ch) + 0.02 * np.eye(n_ch)
    v_hist = rng.normal(size=(W, step2.PCA_DIM))

    t_reproj, t_cache = [], []
    for it in range(550):
        t0 = time.perf_counter()
        vf = (b2a.project_tangent(covs, mf) @ pca_f) @ wfi
        _ = within_adapt_clip(vf[:-1], vf[-1:], sfc, sfs, 1.0, 0.25, 0.92)[0]
        dt_r = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        vt = (b2a.project_tangent(covs[-1:], mf) @ pca_f) @ wfi
        _ = within_adapt_clip(v_hist, vt, sfc, sfs, 1.0, 0.25, 0.92)[0]
        dt_c = (time.perf_counter() - t1) * 1000.0

        if it >= 50:
            t_reproj.append(dt_r)
            t_cache.append(dt_c)
    return {
        f"lat_{n_ch}ch_reproj_geom_ms": float(np.mean(t_reproj)),
        f"lat_{n_ch}ch_cache_geom_ms": float(np.mean(t_cache)),
    }


def main() -> None:
    t0 = time.perf_counter()
    store: dict[str, np.ndarray] = {}
    master = np.load(config.RESULTS_DIR / "step2_unified_causal_master.npz")

    print("=" * 92)
    print("REVIEWER REVISION ABLATIONS (Major 1-5 Verification)")
    print("=" * 92)

    # --------------------------------------------------------------------------
    # 1. Chang2025 12-ch-only CAR vs 59-ch CAR
    # --------------------------------------------------------------------------
    print("\n[1/5] Chang2025 12-channel-only CAR vs 59-channel whole-cap CAR...")
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor() as ex:
        ch_car12 = list(ex.map(worker_chang_car12, ch_dirs))
    for tag, cset in [("chang_bin", (1, 2)), ("chang_3cls", (1, 2, 3))]:
        cf = [step2.subset_block(s, np.isin(s["y"], cset)) for s in ch_car12]
        with ProcessPoolExecutor(initializer=step2._init_state, initargs=({"blocks": cf},)) as ex:
            rows = list(ex.map(step2._worker_main_task, [("loso", i, False) for i in range(len(cf))]))
        for m in ("Uncal", "G1", "Orig", "z-CORAL (OAS-Src)", "Within-Adapt"):
            arr = np.asarray([r[m] for r in rows], dtype=np.float64)
            store[f"car12__{tag}__{m}"] = arr
        wa12 = store[f"car12__{tag}__Within-Adapt"]
        g1_12 = store[f"car12__{tag}__G1"]
        unc12 = store[f"car12__{tag}__Uncal"]
        wa59 = master[f"{tag}__Within-Adapt"]
        p_unc, _, _ = exact_p(wa12, unc12)
        p_g1, _, _ = exact_p(wa12, g1_12)
        p_59, _, _ = exact_p(wa12, wa59)
        print(
            f"  {tag:10s} (12-ch CAR): Uncal={unc12.mean()*100:.2f}% | G1={g1_12.mean()*100:.2f}% | "
            f"WA={wa12.mean()*100:.2f}% (vs Uncal p={p_unc:.4g}, vs G1 p={p_g1:.4g}) | "
            f"59-ch WA={wa59.mean()*100:.2f}% (12 vs 59 p={p_59:.4g})"
        )

    # --------------------------------------------------------------------------
    # 2. CS Sliding-W=8 Training vs Global Single-Block Training
    # --------------------------------------------------------------------------
    print("\n[2/5] Cross-Session (CS) Sliding-W=8 Train vs Global Single-Block Train...")
    with ProcessPoolExecutor() as ex:
        b2a_subs = list(ex.map(step2.worker_b2a, range(1, 10)))
    mat_files = sorted(ky18.DATA_DIR.glob("*.mat"))
    with ProcessPoolExecutor() as ex:
        ky_sess = list(ex.map(step2.worker_kaya, mat_files))

    for tag, cset in [("2a_bin_cs", (1, 2)), ("2a_4cls_cs", (1, 2, 3, 4))]:
        rows = [
            eval_cs_slide_fold(
                step2.subset_block(b2a_subs[i]["T"], np.isin(b2a_subs[i]["T"]["y"], cset)),
                step2.subset_block(b2a_subs[i]["E"], np.isin(b2a_subs[i]["E"]["y"], cset)),
            )
            for i in range(9)
        ]
        for k in rows[0]:
            store[f"cs_slide__{tag}__{k}"] = np.asarray([r[k] for r in rows], dtype=np.float64)
        wg, ws, ws8 = store[f"cs_slide__{tag}__WA_glob"], store[f"cs_slide__{tag}__WA_slide"], store[f"cs_slide__{tag}__WA_slide8"]
        g1g, g1s = store[f"cs_slide__{tag}__G1_glob"], store[f"cs_slide__{tag}__G1_slide"]
        p_ws, _, _ = exact_p(ws, wg)
        p_ws8, _, _ = exact_p(ws8, wg)
        print(
            f"  {tag:10s}: G1 glob={g1g.mean()*100:.2f}% vs slide={g1s.mean()*100:.2f}% | "
            f"WA glob={wg.mean()*100:.2f}% vs slide[0:]={ws.mean()*100:.2f}% (p={p_ws:.4g}) vs slide[8:]={ws8.mean()*100:.2f}% (p={p_ws8:.4g})"
        )

    ky_bin = [step2.subset_block(s, np.isin(s["y"], (1, 2))) for s in ky_sess]
    ky_subj = [s["subj"] for s in ky_sess]
    cs_pairs, cs_subjs = [], []
    for sn in sorted(set(ky_subj)):
        idxs = [i for i, x in enumerate(ky_subj) if x == sn]
        if len(idxs) >= 2:
            for ti in idxs[1:]:
                cs_pairs.append((idxs[0], ti))
                cs_subjs.append(sn)
    ky_rows = [eval_cs_slide_fold(ky_bin[tr_i], ky_bin[te_i]) for tr_i, te_i in cs_pairs]
    uniq_ky = sorted(set(cs_subjs))
    for k in ky_rows[0]:
        agg = [float(np.mean([ky_rows[i][k] for i, sn in enumerate(cs_subjs) if sn == s])) for s in uniq_ky]
        store[f"cs_slide__kaya_bin_cs__{k}"] = np.asarray(agg, dtype=np.float64)
    wg, ws, ws8 = store["cs_slide__kaya_bin_cs__WA_glob"], store["cs_slide__kaya_bin_cs__WA_slide"], store["cs_slide__kaya_bin_cs__WA_slide8"]
    g1g, g1s = store["cs_slide__kaya_bin_cs__G1_glob"], store["cs_slide__kaya_bin_cs__G1_slide"]
    p_ws, _, _ = exact_p(ws, wg)
    print(
        f"  kaya_bin_cs (N=5): G1 glob={g1g.mean()*100:.2f}% vs slide={g1s.mean()*100:.2f}% | "
        f"WA glob={wg.mean()*100:.2f}% vs slide[0:]={ws.mean()*100:.2f}% (p={p_ws:.4g}) vs slide[8:]={ws8.mean()*100:.2f}%"
    )

    # --------------------------------------------------------------------------
    # 3-5. Diagnostics on EEGBCI (N=109), BCI-IV-2a 4-Cls (N=9), Chang 3-Cls (N=28)
    # --------------------------------------------------------------------------
    print("\n[3-5/5] Cached-History vs Re-Proj, Cold-Start Sensitivity, & Natural High-Delta Stratification...")
    with ProcessPoolExecutor() as ex:
        ch_subs = list(ex.map(step2.worker_chang, ch_dirs))
    with ProcessPoolExecutor() as ex:
        eeg_subs = list(ex.map(step2.worker_eegbci, range(1, 110)))

    b2a_4_all = [[step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2, 3, 4))) for s in ("T", "E")] for i in range(9)]
    b2a_4_te = [step2.subset_block(b2a_subs[i]["E"], np.isin(b2a_subs[i]["E"]["y"], (1, 2, 3, 4))) for i in range(9)]
    # Flatten 2a LOSO helper using exact step2 convention: train=all other T&E, test=E
    ch3_blocks = [step2.subset_block(s, np.isin(s["y"], (1, 2, 3))) for s in ch_subs]

    cohort_runs: list[tuple[str, list[dict]]] = [
        ("eegbci", eeg_subs),
        ("chang_3cls", ch3_blocks),
    ]
    diag_all: dict[str, list[dict[str, float]]] = {}
    for tag, blks in cohort_runs:
        with ProcessPoolExecutor(initializer=_init, initargs=({"blocks": blks},)) as ex:
            diag_all[tag] = list(ex.map(_worker_loso_diagnostics, range(len(blks))))

    # 2a 4-Cls LOSO separately (since train uses T+E of others, test uses E)
    rows_2a4 = []
    for idx in range(9):
        tr_b = [b for j in range(9) if j != idx for b in b2a_4_all[j]]
        te_b = b2a_4_te[idx]
        # Reuse worker logic cleanly by placing train blocks + target at the end
        _init({"blocks": tr_b + [te_b]})
        rows_2a4.append(_worker_loso_diagnostics(len(tr_b)))
    diag_all["2a_4cls"] = rows_2a4

    for tag, rows in diag_all.items():
        for k in rows[0]:
            store[f"diag_{tag}__{k}"] = np.asarray([r[k] for r in rows], dtype=np.float64)

        print(f"\n  === Cohort: {tag} (N={len(rows)}) ===")
        # Cached vs Reproj
        r_wa, c_wa = store[f"diag_{tag}__reproj__WA"], store[f"diag_{tag}__cache__WA"]
        r_or, c_or = store[f"diag_{tag}__reproj__Orig"], store[f"diag_{tag}__cache__Orig"]
        pw, _, _ = exact_p(r_wa, c_wa)
        po, _, _ = exact_p(r_or, c_or)
        print(
            f"    ReProj vs Cached-Past-V: WA {r_wa.mean()*100:.2f}% vs {c_wa.mean()*100:.2f}% (p={pw:.4g}) | "
            f"Orig {r_or.mean()*100:.2f}% vs {c_or.mean()*100:.2f}% (p={po:.4g})"
        )
        # High-delta stratification
        p_hg1, _, _ = exact_p(store[f"diag_{tag}__hi_delta__WA"], store[f"diag_{tag}__hi_delta__G1"])
        p_hzc, _, _ = exact_p(store[f"diag_{tag}__hi_delta__WA"], store[f"diag_{tag}__hi_delta__zCORAL"])
        print(
            f"    Natural Top-25% High-delta_t (med={store[f'diag_{tag}__delta_p50'].mean():.3f}, "
            f"p75={store[f'diag_{tag}__delta_p75'].mean():.3f}, p95={store[f'diag_{tag}__delta_p95'].mean():.3f}): "
            f"G1={store[f'diag_{tag}__hi_delta__G1'].mean()*100:.2f}% | "
            f"Orig={store[f'diag_{tag}__hi_delta__Orig'].mean()*100:.2f}% | "
            f"zC={store[f'diag_{tag}__hi_delta__zCORAL'].mean()*100:.2f}% | "
            f"WA={store[f'diag_{tag}__hi_delta__WA'].mean()*100:.2f}% (vs G1 p={p_hg1:.4g}, vs zC p={p_hzc:.4g})"
        )
        # Cold start sensitivity
        def_b, def_a = store[f"diag_{tag}__cs_default__burn"], store[f"diag_{tag}__cs_default__all"]
        parts_b, parts_a = [f"def={def_b.mean()*100:.2f}%"], [f"def={def_a.mean()*100:.2f}%"]
        for cname in ("tau1_1", "tau1_4", "t1_zero", "t1_full", "no_ramp", "clip_wide", "clip_tight"):
            cb = store[f"diag_{tag}__cs_{cname}__burn"]
            ca = store[f"diag_{tag}__cs_{cname}__all"]
            pb, _, _ = exact_p(cb, def_b)
            pa, _, _ = exact_p(ca, def_a)
            parts_b.append(f"{cname}={cb.mean()*100:.2f}%(p={pb:.3g})")
            parts_a.append(f"{cname}={ca.mean()*100:.2f}%(p={pa:.3g})")
        print("    Burn-in [0:15]: " + " | ".join(parts_b))
        print("    Full    [0:]  : " + " | ".join(parts_a))

    for nch in (13, 22):
        lat = benchmark_reproj_vs_cache_cpu(nch)
        for k, v in lat.items():
            store[k] = np.asarray([v])
        print(f"  Geom step CPU time ({nch}ch, 1 band): Full-Window ReProj={lat[f'lat_{nch}ch_reproj_geom_ms']:.3f} ms vs Single-Trial Cache={lat[f'lat_{nch}ch_cache_geom_ms']:.3f} ms")

    np.savez_compressed(OUT_NPZ, **store)
    print(f"\nSaved {OUT_NPZ} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
