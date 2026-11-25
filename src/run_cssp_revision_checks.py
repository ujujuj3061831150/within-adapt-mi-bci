"""Supplementary CSSP manuscript verification & confounder-isolation suite.

Reproduces:
1. Exact trial counts & source-block sizes across all four public cohorts;
2. Component isolation controls (`z-Adapt + StandardScaler`, `z-CORAL + StandardScaler`,
   static `G1 + Sigma_W^-1/2 + StandardScaler`, and `gamma == 1.0` online centering ablation)
   across stationary protocols, synthetic stressors, and Chang2025 12-ch CAR;
3. PCA tangent dimension sensitivity (D in {15, 30, 60, 120}) and 80%-variance dimensions;
4. End-to-end single-trial CPU latency on real recorded EEGBCI (S001) and BCI-IV-2a (A01) epochs.

Usage:
    python src/run_cssp_revision_checks.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import kaya2018_causal_coral_repro as ky18
import run_p1_audit_updates as p1a
import run_reviewer_revision_ablations as rra
import run_step2_riemannian_master as step2
import unified_within_adapt_coral_benchmark as uni
from config import RESULTS_DIR

OUT_JSON = RESULTS_DIR / "cssp_revision_checks.json"


def exact_p(a: np.ndarray, b: np.ndarray) -> float:
    pv, _, _ = p1a.exact_wilcoxon_clean(a, b)
    return float(pv)


def eval_m2_fold(train_blocks: list[dict], test_blk: dict) -> dict[str, float]:
    """Compare G1, z-Adapt (raw/+StandardScaler), G1+Sigma_W^-1/2, WA(gamma=1), and Within-Adapt."""
    y_tr = np.concatenate([b["y"] for b in train_blocks], 0)
    y_te = test_blk["y"][15:]
    rec = [
        (
            b2a.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(0))),
            b2a.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(0))),
        )
        for b in train_blocks
    ]
    tf_all = np.concatenate([r[0] for r in rec], 0)
    th_all = np.concatenate([r[1] for r in rec], 0)
    pf = PCA(n_components=30, random_state=42).fit(tf_all)
    ph = PCA(n_components=30, random_state=42).fit(th_all)
    zf_l = [pf.transform(r[0]) for r in rec]
    zh_l = [ph.transform(r[1]) for r in rec]
    zf_all, zh_all = np.concatenate(zf_l, 0), np.concatenate(zh_l, 0)

    scf, sch = StandardScaler().fit(zf_all), StandardScaler().fit(zh_all)
    szf, _ = b2a.exact_oas(zf_all - zf_all.mean(0, keepdims=True), True)
    szh, _ = b2a.exact_oas(zh_all - zh_all.mean(0, keepdims=True), True)
    szfs, szhs = uni.sym_sqrtm(szf), uni.sym_sqrtm(szh)

    wfi, sfc, sfs = uni.fit_within_operator(zf_all, y_tr)
    whi, shc, shs = uni.fit_within_operator(zh_all, y_tr)

    vf_all, vh_all = zf_all @ wfi, zh_all @ whi
    sc_v_static = StandardScaler().fit(np.concatenate([vf_all, vh_all], 1))

    Xtr_g1 = np.concatenate([np.concatenate([scf.transform(zf), sch.transform(zh)], 1) for zf, zh in zip(zf_l, zh_l)], 0)
    Xtr_za = np.concatenate([np.concatenate([step2.within_adapt_s(zf, zf, szf, szfs, 1.0), step2.within_adapt_s(zh, zh, szh, szhs, 1.0)], 1) for zf, zh in zip(zf_l, zh_l)], 0)
    sc_za = StandardScaler().fit(Xtr_za)

    Xtr_wstat = sc_v_static.transform(np.concatenate([vf_all, vh_all], 1))
    Xtr_g1w = np.concatenate([
        np.concatenate([zf @ wfi - (zf @ wfi).mean(0, keepdims=True), zh @ whi - (zh @ whi).mean(0, keepdims=True)], 1)
        for zf, zh in zip(zf_l, zh_l)
    ], 0)
    Xtr_wa = np.concatenate([
        np.concatenate([step2.within_adapt_s(zf @ wfi, zf @ wfi, sfc, sfs, 1.0), step2.within_adapt_s(zh @ whi, zh @ whi, shc, shs, 1.0)], 1)
        for zf, zh in zip(zf_l, zh_l)
    ], 0)
    sc_wa = StandardScaler().fit(Xtr_wa)

    clf_g1 = step2.fit_lr(Xtr_g1, y_tr)
    clf_za = step2.fit_lr(Xtr_za, y_tr)
    clf_za_sc = step2.fit_lr(sc_za.transform(Xtr_za), y_tr)
    clf_wstat = step2.fit_lr(Xtr_wstat, y_tr)
    clf_g1w = step2.fit_lr(Xtr_g1w, y_tr)
    clf_wa = step2.fit_lr(Xtr_wa, y_tr)
    clf_wa_sc = step2.fit_lr(sc_wa.transform(Xtr_wa), y_tr)

    n_te = len(test_blk["y"])
    n_eval = n_te - 15
    Xg1, Xza, Xwstat, Xg1w, Xwa = [np.empty((n_eval, 60), dtype=np.float64) for _ in range(5)]
    for oi, t in enumerate(range(15, n_te)):
        mf = b2a.sym_expm(test_blk["log_f"][t - 8 : t].mean(0))
        mh = b2a.sym_expm(test_blk["log_h"][t - 8 : t].mean(0))
        zf = pf.transform(b2a.project_tangent(test_blk["cov_f"][t - 8 : t + 1], mf))
        zh = ph.transform(b2a.project_tangent(test_blk["cov_h"][t - 8 : t + 1], mh))
        vf, vh = zf @ wfi, zh @ whi
        Xg1[oi] = np.concatenate([scf.transform(zf[-1:])[0], sch.transform(zh[-1:])[0]])
        Xza[oi] = np.concatenate([step2.within_adapt_s(zf[:-1], zf[-1:], szf, szfs, 1.0)[0], step2.within_adapt_s(zh[:-1], zh[-1:], szh, szhs, 1.0)[0]])
        Xwstat[oi] = sc_v_static.transform(np.concatenate([vf[-1:], vh[-1:]], 1))[0]
        Xg1w[oi] = np.concatenate([(vf[-1] - vf[:-1].mean(0)), (vh[-1] - vh[:-1].mean(0))])
        Xwa[oi] = np.concatenate([step2.within_adapt_s(vf[:-1], vf[-1:], sfc, sfs, 1.0)[0], step2.within_adapt_s(vh[:-1], vh[-1:], shc, shs, 1.0)[0]])

    return {
        "G1": float(np.mean(clf_g1.predict(Xg1) == y_te)),
        "zAdapt": float(np.mean(clf_za.predict(Xza) == y_te)),
        "zAdapt_SS": float(np.mean(clf_za_sc.predict(sc_za.transform(Xza)) == y_te)),
        "G1_W_static": float(np.mean(clf_wstat.predict(Xwstat) == y_te)),
        "WA_gamma1": float(np.mean(clf_g1w.predict(Xg1w) == y_te)),
        "WA": float(np.mean(clf_wa.predict(Xwa) == y_te)),
        "WA_SS": float(np.mean(clf_wa_sc.predict(sc_wa.transform(Xwa)) == y_te)),
    }


def _m2_worker(args):
    kind, idx = args
    blocks = step2._G_STATE["blocks"]
    if kind == "b2a_cs":
        tr, te = [blocks[idx]["T"]], blocks[idx]["E"]
    elif kind == "b2a_loso":
        tr, te = [blocks[j][s] for j in range(9) if j != idx for s in ("T", "E")], blocks[idx]["E"]
    elif kind == "loso":
        tr, te = [blocks[j] for j in range(len(blocks)) if j != idx], blocks[idx]
    else:
        raise ValueError(kind)
    return eval_m2_fold(tr, te)


def _m2_stress_worker(args):
    idx, is_b2a = args
    blocks = step2._G_STATE["blocks"]
    if is_b2a:
        tr = [blocks[j][s] for j in range(len(blocks)) if j != idx for s in ("T", "E")]
        te_raw = blocks[idx]["E"]
    else:
        tr = [blocks[j] for j in range(len(blocks)) if j != idx]
        te_raw = blocks[idx]

    out = {}
    n_te = len(te_raw["y"])
    n_ch = te_raw["log_f"].shape[1]
    for mode in ("disp_3x", "disp_0.25x", "aniso_ch_drift"):
        log_f, log_h = te_raw["log_f"].copy(), te_raw["log_h"].copy()
        if mode == "disp_3x":
            mf0, mh0 = log_f.mean(0), log_h.mean(0)
            log_f, log_h = mf0 + 3.0 * (log_f - mf0), mh0 + 3.0 * (log_h - mh0)
        elif mode == "disp_0.25x":
            mf0, mh0 = log_f.mean(0), log_h.mean(0)
            log_f, log_h = mf0 + 0.25 * (log_f - mf0), mh0 + 0.25 * (log_h - mh0)
        elif mode == "aniso_ch_drift":
            rng = np.random.default_rng(1000 + idx)
            for t in range(n_te):
                d_vec = np.zeros(n_ch, dtype=np.float64)
                d_vec[:3] = rng.normal(0.0, 1.8, size=3)
                D = np.diag(d_vec)
                log_f[t] = log_f[t] + D
                log_h[t] = log_h[t] + D
        cov_f = np.stack([b2a.sym_expm(l) for l in log_f], 0)
        cov_h = np.stack([b2a.sym_expm(l) for l in log_h], 0)
        te_mod = dict(te_raw)
        te_mod["log_f"], te_mod["log_h"] = log_f, log_h
        te_mod["cov_f"], te_mod["cov_h"] = cov_f, cov_h
        m2_res = eval_m2_fold(tr, te_mod)

        y_tr = np.concatenate([b["y"] for b in tr], 0)
        y_te = te_mod["y"][15:]
        m_src_f = b2a.sym_expm(np.concatenate([b["log_f"] for b in tr], 0).mean(0))
        m_src_h = b2a.sym_expm(np.concatenate([b["log_h"] for b in tr], 0).mean(0))
        tk0_f = b2a.project_tangent(np.concatenate([b["cov_f"] for b in tr], 0), m_src_f)
        tk0_h = b2a.project_tangent(np.concatenate([b["cov_h"] for b in tr], 0), m_src_h)
        pk0_f = PCA(n_components=30, random_state=42).fit(tk0_f)
        pk0_h = PCA(n_components=30, random_state=42).fit(tk0_h)
        Xtr_k0 = np.concatenate([pk0_f.transform(tk0_f), pk0_h.transform(tk0_h)], 1)
        sck0 = StandardScaler().fit(Xtr_k0)
        clfk0 = step2.fit_lr(sck0.transform(Xtr_k0), y_tr)
        Xte_k0 = np.concatenate([pk0_f.transform(b2a.project_tangent(cov_f[15:], m_src_f)), pk0_h.transform(b2a.project_tangent(cov_h[15:], m_src_h))], 1)
        m2_res["Uncal"] = float(np.mean(clfk0.predict(sck0.transform(Xte_k0)) == y_te))
        out[mode] = m2_res
    return out


def _pca_d_worker(idx):
    blocks = step2._G_STATE["blocks"]
    is_cs = step2._G_STATE["is_cs"]
    if is_cs:
        tr, te = [blocks[idx]["T"]], blocks[idx]["E"]
    else:
        tr, te = [blocks[j][s] for j in range(9) if j != idx for s in ("T", "E")], blocks[idx]["E"]
    y_tr = np.concatenate([b["y"] for b in tr], 0)
    y_te = te["y"][15:]
    rec = [
        (b2a.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(0))),
         b2a.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(0))))
        for b in tr
    ]
    tf_all = np.concatenate([r[0] for r in rec], 0)
    th_all = np.concatenate([r[1] for r in rec], 0)

    pf_full = PCA(n_components=None, random_state=42).fit(tf_all)
    ph_full = PCA(n_components=None, random_state=42).fit(th_all)
    d80_f = int(np.searchsorted(np.cumsum(pf_full.explained_variance_ratio_), 0.80) + 1)
    d80_h = int(np.searchsorted(np.cumsum(ph_full.explained_variance_ratio_), 0.80) + 1)

    res = {"d80_f": float(d80_f), "d80_h": float(d80_h)}
    for d_dim in (15, 30, 60, 120):
        pf = PCA(n_components=d_dim, random_state=42).fit(tf_all)
        ph = PCA(n_components=d_dim, random_state=42).fit(th_all)
        zf_l = [pf.transform(r[0]) for r in rec]
        zh_l = [ph.transform(r[1]) for r in rec]
        zf_all, zh_all = np.concatenate(zf_l, 0), np.concatenate(zh_l, 0)
        scf, sch = StandardScaler().fit(zf_all), StandardScaler().fit(zh_all)
        szf, _ = b2a.exact_oas(zf_all - zf_all.mean(0, keepdims=True), True)
        szh, _ = b2a.exact_oas(zh_all - zh_all.mean(0, keepdims=True), True)
        szfs, szhs = uni.sym_sqrtm(szf), uni.sym_sqrtm(szh)
        wfi, sfc, sfs = uni.fit_within_operator(zf_all, y_tr)
        whi, shc, shs = uni.fit_within_operator(zh_all, y_tr)

        cg1 = step2.fit_lr(np.concatenate([np.concatenate([scf.transform(zf), sch.transform(zh)], 1) for zf, zh in zip(zf_l, zh_l)], 0), y_tr)
        czc = step2.fit_lr(np.concatenate([np.concatenate([step2.z_coral_oas_src(zf, zf, szfs), step2.z_coral_oas_src(zh, zh, szhs)], 1) for zf, zh in zip(zf_l, zh_l)], 0), y_tr)
        cwa = step2.fit_lr(np.concatenate([np.concatenate([step2.within_adapt_s(zf @ wfi, zf @ wfi, sfc, sfs, 1.0), step2.within_adapt_s(zh @ whi, zh @ whi, shc, shs, 1.0)], 1) for zf, zh in zip(zf_l, zh_l)], 0), y_tr)

        n_te = len(te["y"])
        Xg1, Xzc, Xwa = [np.empty((n_te - 15, d_dim * 2), dtype=np.float64) for _ in range(3)]
        for oi, t in enumerate(range(15, n_te)):
            mf = b2a.sym_expm(te["log_f"][t - 8 : t].mean(0))
            mh = b2a.sym_expm(te["log_h"][t - 8 : t].mean(0))
            zf = pf.transform(b2a.project_tangent(te["cov_f"][t - 8 : t + 1], mf))
            zh = ph.transform(b2a.project_tangent(te["cov_h"][t - 8 : t + 1], mh))
            vf, vh = zf @ wfi, zh @ whi
            Xg1[oi] = np.concatenate([scf.transform(zf[-1:])[0], sch.transform(zh[-1:])[0]])
            Xzc[oi] = np.concatenate([step2.z_coral_oas_src(zf[:-1], zf[-1:], szfs)[0], step2.z_coral_oas_src(zh[:-1], zh[-1:], szhs)[0]])
            Xwa[oi] = np.concatenate([step2.within_adapt_s(vf[:-1], vf[-1:], sfc, sfs, 1.0)[0], step2.within_adapt_s(vh[:-1], vh[-1:], shc, shs, 1.0)[0]])

        res[f"D{d_dim}_G1"] = float(np.mean(cg1.predict(Xg1) == y_te))
        res[f"D{d_dim}_zCORAL"] = float(np.mean(czc.predict(Xzc) == y_te))
        res[f"D{d_dim}_WA"] = float(np.mean(cwa.predict(Xwa) == y_te))
    return res


def benchmark_real_latency() -> dict:
    """Measure end-to-end single-trial CPU latency on real EEGBCI S001 and BCI-IV-2a A01 epochs."""
    import mne, warnings
    import eegbci_reviewer_ablations as eeg_ab

    mne.set_log_level("WARNING")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(42)

    raw_eeg = mne.io.read_raw_edf(eeg_ab.DATA_DIR / "S001" / "S001R04.edf", preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw_eeg)
    raw_eeg.set_eeg_reference("average", projection=False, verbose=False)
    raw_eeg.pick([ch for ch in eeg_ab.CORE_CHANNELS if ch in raw_eeg.ch_names])
    events, ev_all = mne.events_from_annotations(raw_eeg, verbose=False)
    vev = events[np.isin(events[:, 2], [ev_all["T1"], ev_all["T2"]])]
    s0, s1 = int(round(eeg_ab.TMIN * 160.0)), int(round(eeg_ab.TMAX * 160.0)) + 1
    d_eeg = raw_eeg.get_data() * 1e6
    eps_eeg = [d_eeg[:, ev[0] + s0 : ev[0] + s1] for ev in vev if ev[0] + s1 <= d_eeg.shape[1]]

    mat = sio.loadmat(b2a.DATA_DIR / "A01T.mat", simplify_cells=True)
    eps_b2a = []
    sb0 = int(round((b2a.CUE_OFFSET_SEC + b2a.POST_CUE_TMIN) * 250.0))
    sb1 = int(round((b2a.CUE_OFFSET_SEC + b2a.POST_CUE_TMAX) * 250.0)) + 1
    for run_dict in mat["data"]:
        tr = np.asarray(run_dict["trial"], dtype=np.int64)
        if tr.size == 0:
            continue
        X_eeg = np.asarray(run_dict["X"], dtype=np.float64)[:, :22].T
        for i in tr - 1:
            eps_b2a.append(X_eeg[:, i + sb0 : i + sb1])

    out = {}
    for name, eps, n_ch, sfreq in [("EEGBCI_13ch_real", eps_eeg, 13, 160.0), ("BCIC2a_22ch_real", eps_b2a, 22, 250.0)]:
        sos_f = butter(4, [8.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
        sos_h = butter(4, [21.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
        zi_f = np.repeat(sosfilt_zi(sos_f)[:, None, :], n_ch, axis=1)
        zi_h = np.repeat(sosfilt_zi(sos_h)[:, None, :], n_ch, axis=1)
        d_tan = n_ch * (n_ch + 1) // 2
        pca_f, pca_h = rng.normal(size=(d_tan, 30)), rng.normal(size=(d_tan, 30))
        wfi, whi = np.eye(30), np.eye(30)
        sf_c, sh_c, sf_s, sh_s = np.eye(30), np.eye(30), np.eye(30), np.eye(30)
        coef = rng.normal(size=(60,))
        cov_buf_f = [0.5 * (c + c.T) for c in (np.eye(n_ch) + 0.05 * rng.normal(size=(n_ch, n_ch)) for _ in range(9))]
        cov_buf_h = [c.copy() for c in cov_buf_f]
        log_buf_f = [b2a.sym_logm(c[None])[0] for c in cov_buf_f]
        log_buf_h = [b2a.sym_logm(c[None])[0] for c in cov_buf_h]

        lats = []
        for it in range(550):
            raw_ep = eps[it % len(eps)]
            t0 = time.perf_counter()
            x_car = raw_ep - raw_ep.mean(axis=0, keepdims=True)
            ep_f, zi_f = sosfilt(sos_f, x_car, axis=-1, zi=zi_f)
            ep_h, zi_h = sosfilt(sos_h, x_car, axis=-1, zi=zi_h)
            cf, _ = b2a.exact_oas(ep_f.T, assume_centered=False)
            ch, _ = b2a.exact_oas(ep_h.T, assume_centered=False)
            lf, lh = b2a.sym_logm(cf[None])[0], b2a.sym_logm(ch[None])[0]
            cov_arr_f = np.stack(cov_buf_f[-8:] + [cf], axis=0)
            cov_arr_h = np.stack(cov_buf_h[-8:] + [ch], axis=0)
            mf, mh = b2a.sym_expm(np.mean(log_buf_f[-8:], axis=0)), b2a.sym_expm(np.mean(log_buf_h[-8:], axis=0))
            vf = (b2a.project_tangent(cov_arr_f, mf) @ pca_f) @ wfi
            vh = (b2a.project_tangent(cov_arr_h, mh) @ pca_h) @ whi
            out_f = step2.within_adapt_s(vf[:-1], vf[-1:], sf_c, sf_s, 1.0)[0]
            out_h = step2.within_adapt_s(vh[:-1], vh[-1:], sh_c, sh_s, 1.0)[0]
            _ = float(np.dot(np.concatenate([out_f, out_h]), coef))
            dt = (time.perf_counter() - t0) * 1000.0
            cov_buf_f.append(cf); cov_buf_f.pop(0)
            cov_buf_h.append(ch); cov_buf_h.pop(0)
            log_buf_f.append(lf); log_buf_f.pop(0)
            log_buf_h.append(lh); log_buf_h.pop(0)
            if it >= 50:
                lats.append(dt)
        a = np.asarray(lats)
        out[name] = {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "n_real_trials": len(eps)}
    return out


def eval_zcoral_ss_2a4(b2a_4: list[dict]) -> dict:
    """Verify z-CORAL (OAS-Src) + StandardScaler vs Within-Adapt on BCI-IV-2a 4-class CS and LOSO."""
    out = {}
    for proto in ("cs", "loso"):
        zc_raw, zc_ss, wa_raw = [], [], []
        for i in range(9):
            tr = [b2a_4[i]["T"]] if proto == "cs" else [b2a_4[j][s] for j in range(9) if j != i for s in ("T", "E")]
            te = b2a_4[i]["E"]
            y_tr, y_te = np.concatenate([b["y"] for b in tr], 0), te["y"][15:]
            rec = [(b2a.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(0))),
                    b2a.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(0)))) for b in tr]
            pf = PCA(n_components=30, random_state=42).fit(np.concatenate([r[0] for r in rec], 0))
            ph = PCA(n_components=30, random_state=42).fit(np.concatenate([r[1] for r in rec], 0))
            zf_l = [pf.transform(r[0]) for r in rec]
            zh_l = [ph.transform(r[1]) for r in rec]
            zf_all, zh_all = np.concatenate(zf_l, 0), np.concatenate(zh_l, 0)
            szf, _ = b2a.exact_oas(zf_all - zf_all.mean(0, keepdims=True), True)
            szh, _ = b2a.exact_oas(zh_all - zh_all.mean(0, keepdims=True), True)
            szfs, szhs = uni.sym_sqrtm(szf), uni.sym_sqrtm(szh)
            wfi, sfc, sfs = uni.fit_within_operator(zf_all, y_tr)
            whi, shc, shs = uni.fit_within_operator(zh_all, y_tr)
            Xtr_zc = np.concatenate([np.concatenate([step2.z_coral_oas_src(zf, zf, szfs), step2.z_coral_oas_src(zh, zh, szhs)], 1) for zf, zh in zip(zf_l, zh_l)], 0)
            sc_zc = StandardScaler().fit(Xtr_zc)
            Xtr_wa = np.concatenate([np.concatenate([step2.within_adapt_s(zf @ wfi, zf @ wfi, sfc, sfs, 1.0), step2.within_adapt_s(zh @ whi, zh @ whi, shc, shs, 1.0)], 1) for zf, zh in zip(zf_l, zh_l)], 0)
            czc = step2.fit_lr(Xtr_zc, y_tr)
            czc_ss = step2.fit_lr(sc_zc.transform(Xtr_zc), y_tr)
            cwa = step2.fit_lr(Xtr_wa, y_tr)
            n_te = len(te["y"])
            Xzc, Xwa = np.empty((n_te - 15, 60)), np.empty((n_te - 15, 60))
            for oi, t in enumerate(range(15, n_te)):
                mf = b2a.sym_expm(te["log_f"][t - 8 : t].mean(0))
                mh = b2a.sym_expm(te["log_h"][t - 8 : t].mean(0))
                zf = pf.transform(b2a.project_tangent(te["cov_f"][t - 8 : t + 1], mf))
                zh = ph.transform(b2a.project_tangent(te["cov_h"][t - 8 : t + 1], mh))
                vf, vh = zf @ wfi, zh @ whi
                Xzc[oi] = np.concatenate([step2.z_coral_oas_src(zf[:-1], zf[-1:], szfs)[0], step2.z_coral_oas_src(zh[:-1], zh[-1:], szhs)[0]])
                Xwa[oi] = np.concatenate([step2.within_adapt_s(vf[:-1], vf[-1:], sfc, sfs, 1.0)[0], step2.within_adapt_s(vh[:-1], vh[-1:], shc, shs, 1.0)[0]])
            zc_raw.append(np.mean(czc.predict(Xzc) == y_te))
            zc_ss.append(np.mean(czc_ss.predict(sc_zc.transform(Xzc)) == y_te))
            wa_raw.append(np.mean(cwa.predict(Xwa) == y_te))
        zc_raw_a, zc_ss_a, wa_raw_a = np.asarray(zc_raw) * 100.0, np.asarray(zc_ss) * 100.0, np.asarray(wa_raw) * 100.0
        out[f"2a_4cls_{proto}"] = {
            "zCORAL": float(zc_raw_a.mean()), "p_vs_WA": exact_p(wa_raw_a, zc_raw_a),
            "zCORAL_SS": float(zc_ss_a.mean()), "p_SS_vs_WA": exact_p(wa_raw_a, zc_ss_a),
            "WA": float(wa_raw_a.mean()),
        }
    return out


def main() -> None:
    results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        b2a_subs = list(ex.map(step2.worker_b2a, range(1, 10)))
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor(max_workers=8) as ex:
        ch_subs = list(ex.map(step2.worker_chang, ch_dirs))
    mat_files = sorted(ky18.DATA_DIR.glob("*.mat"))
    with ProcessPoolExecutor(max_workers=8) as ex:
        ky_sess = list(ex.map(step2.worker_kaya, mat_files))
    with ProcessPoolExecutor(max_workers=8) as ex:
        eeg_subs = list(ex.map(step2.worker_eegbci, range(1, 110)))

    b2a_bin = [{s: step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2))) for s in ("T", "E")} for i in range(9)]
    b2a_4 = [{s: step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2, 3, 4))) for s in ("T", "E")} for i in range(9)]
    ch_3 = [step2.subset_block(s, np.isin(s["y"], (1, 2, 3))) for s in ch_subs]
    ky_bin = [step2.subset_block(s, np.isin(s["y"], (1, 2))) for s in ky_sess]
    ky3 = [step2.subset_block(s, np.isin(s["y"], (1, 2, 3))) for s in ky_sess if np.all(np.isin([1, 2, 3], s["y"]))]

    ky_cs_post = 0
    ky_bin_subj = [s["subj"] for s in ky_sess]
    for sn in sorted(set(ky_bin_subj)):
        idxs = [i for i, x in enumerate(ky_bin_subj) if x == sn]
        if len(idxs) >= 2:
            for ti in idxs[1:]:
                ky_cs_post += len(ky_bin[ti]["y"]) - 15

    results["trial_counts"] = {
        "eegbci_total": int(sum(len(s["y"]) for s in eeg_subs)),
        "eegbci_post15": int(sum(len(s["y"]) - 15 for s in eeg_subs)),
        "kaya_bin_sessions": len(ky_bin), "kaya_bin_total": int(sum(len(s["y"]) for s in ky_bin)), "kaya_bin_post15": int(sum(len(s["y"]) - 15 for s in ky_bin)),
        "kaya_3cls_sessions": len(ky3), "kaya_3cls_total": int(sum(len(s["y"]) for s in ky3)), "kaya_3cls_post15": int(sum(len(s["y"]) - 15 for s in ky3)),
        "kaya_cs_post15": int(ky_cs_post),
    }
    results["source_block_trial_counts"] = {
        "eegbci_3runs": [int(min(len(s["y"]) for s in eeg_subs)), int(max(len(s["y"]) for s in eeg_subs))],
        "bci2a_session_4cls": 288,
        "bci2a_session_bin": 144,
        "chang_bin": [int(min(sum(np.isin(s["y"], (1, 2))) for s in ch_subs)), int(max(sum(np.isin(s["y"], (1, 2))) for s in ch_subs))],
        "chang_3cls": [int(min(sum(np.isin(s["y"], (1, 2, 3))) for s in ch_subs)), int(max(sum(np.isin(s["y"], (1, 2, 3))) for s in ch_subs))],
        "kaya_session_bin": [int(min(len(s["y"]) for s in ky_bin)), int(max(len(s["y"]) for s in ky_bin))],
        "kaya_session_3cls": [int(min(len(s["y"]) for s in ky3)), int(max(len(s["y"]) for s in ky3))],
    }

    m2_summary = {}
    for tag, blocks, kind, n_sub in [
        ("2a_bin_cs", b2a_bin, "b2a_cs", 9),
        ("2a_4cls_cs", b2a_4, "b2a_cs", 9),
        ("2a_4cls_loso", b2a_4, "b2a_loso", 9),
        ("chang_3cls", ch_3, "loso", 28),
        ("eegbci_bin_loso", eeg_subs, "loso", 109),
    ]:
        with ProcessPoolExecutor(max_workers=8, initializer=step2._init_state, initargs=({"blocks": blocks},)) as ex:
            rows = list(ex.map(_m2_worker, [(kind, i) for i in range(n_sub)]))
        arrs = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        m2_summary[tag] = {
            k: {"mean": float(v.mean() * 100), "p_vs_WA": exact_p(arrs["WA"], v), "p_vs_WA_SS": exact_p(arrs["WA_SS"], v)}
            for k, v in arrs.items()
        }

    for tag, blocks, is_b2a in [("2a_4cls", b2a_4, True), ("chang_3cls", ch_3, False)]:
        with ProcessPoolExecutor(max_workers=8, initializer=step2._init_state, initargs=({"blocks": blocks},)) as ex:
            rows = list(ex.map(_m2_stress_worker, [(i, is_b2a) for i in range(len(blocks))]))
        for mode in ("disp_3x", "disp_0.25x", "aniso_ch_drift"):
            arrs = {k: np.asarray([r[mode][k] for r in rows]) for k in rows[0][mode]}
            m2_summary[f"{tag}__{mode}"] = {
                k: {"mean": float(v.mean() * 100), "p_vs_WA": exact_p(arrs["WA"], v)}
                for k, v in arrs.items()
            }
    results["m2"] = m2_summary

    with ProcessPoolExecutor(max_workers=8) as ex:
        ch_car12 = list(ex.map(rra.worker_chang_car12, ch_dirs))
    ch3_car12 = [step2.subset_block(s, np.isin(s["y"], (1, 2, 3))) for s in ch_car12]
    with ProcessPoolExecutor(max_workers=8, initializer=step2._init_state, initargs=({"blocks": ch3_car12},)) as ex:
        rows_c12_full = list(ex.map(step2._worker_main_task, [("loso", i, False) for i in range(28)]))
        rows_c12_m2 = list(ex.map(_m2_worker, [("loso", i) for i in range(28)]))
    c12_res = {}
    for k in ["Uncal", "G1", "Causal RPA (W=8)", "z-CORAL (OAS-Src)", "Within-CORAL(0.7)", "Within-Adapt"]:
        arr = np.asarray([r[k] for r in rows_c12_full])
        wa = np.asarray([r["Within-Adapt"] for r in rows_c12_full])
        c12_res[k] = {"mean": float(arr.mean() * 100), "p_vs_WA": exact_p(wa, arr)}
    for k in ["WA_gamma1", "G1_W_static", "zAdapt_SS", "WA_SS"]:
        arr = np.asarray([r[k] for r in rows_c12_m2])
        wa = np.asarray([r["WA"] for r in rows_c12_m2])
        c12_res[k] = {"mean": float(arr.mean() * 100), "p_vs_WA": exact_p(wa, arr)}
    results["chang3_car12"] = c12_res
    results["z_coral_ss_2a4"] = eval_zcoral_ss_2a4(b2a_4)

    pca_res = {}
    for tag, is_cs in [("2a_4cls_cs", True), ("2a_4cls_loso", False)]:
        with ProcessPoolExecutor(max_workers=8, initializer=step2._init_state, initargs=({"blocks": b2a_4, "is_cs": is_cs},)) as ex:
            rows = list(ex.map(_pca_d_worker, range(9)))
        pca_res[tag] = {"d80_f_mean": float(np.mean([r["d80_f"] for r in rows])), "d80_h_mean": float(np.mean([r["d80_h"] for r in rows]))}
        for d_dim in (15, 30, 60, 120):
            wa = np.asarray([r[f"D{d_dim}_WA"] for r in rows])
            g1 = np.asarray([r[f"D{d_dim}_G1"] for r in rows])
            zc = np.asarray([r[f"D{d_dim}_zCORAL"] for r in rows])
            pca_res[tag][f"D{d_dim}"] = {
                "G1": float(g1.mean() * 100), "zCORAL": float(zc.mean() * 100), "WA": float(wa.mean() * 100),
                "p_WA_vs_zCORAL": exact_p(wa, zc), "p_WA_vs_G1": exact_p(wa, g1),
            }
    results["m5_pca"] = pca_res
    results["m6_real_latency"] = benchmark_real_latency()

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved CSSP revision checks ->", OUT_JSON)


if __name__ == "__main__":
    main()
