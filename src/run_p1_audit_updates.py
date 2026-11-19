"""
P0/P1 Audit Closure Script
==========================
Computes:
  1. Exact zero-excluded Wilcoxon p-values & 10-protocol Holm-Bonferroni for Table IV & Fig. 3.
  2. Exact zero-excluded Wilcoxon p-values for cold-start [0:], DL contrasts, EEGBCI s-gate & band ablations.
  3. Exact single-process 109-fold [0:] CPU runtime and dense single-trial CPU latency for Uncal (Fixed Tangent).
  4. Unified causal uV 2a_4cls_burst6 stress test (so all stress tests come from unified pipeline).
Saves to results/p1_audit_updates.npz.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import bcic2a_causal_coral_repro as b2a
import probe_step1_audit_and_causal as ps
import run_step2_riemannian_master as step2

OUT_NPZ = config.RESULTS_DIR / "p1_audit_updates.npz"


def exact_wilcoxon_clean(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> tuple[float, float, int]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    nz = diff[np.abs(diff) > tol]
    n_zero = int(len(diff) - len(nz))
    if len(nz) == 0:
        return 1.0, 0.0, n_zero
    res = wilcoxon(nz, method="exact" if len(nz) <= 25 else "auto")
    import scipy.stats
    ranks = scipy.stats.rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    rb = (w_plus - w_minus) / (len(nz) * (len(nz) + 1) / 2.0)
    return float(res.pvalue), float(rb), n_zero


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m_len = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m_len, dtype=np.float64)
    prev = 0.0
    for r, idx in enumerate(order):
        val = min(1.0, max(prev, pvals[idx] * (m_len - r)))
        adj[idx] = val
        prev = val
    return adj.tolist()


def benchmark_uncal_latency_and_batch(eeg_subs: list[dict]) -> dict[str, np.ndarray]:
    out = {}
    rng = np.random.default_rng(42)
    for n_ch, n_t, sfreq, tag in [(13, 481, 160.0, "13"), (22, 751, 250.0, "22")]:
        d_tan = n_ch * (n_ch + 1) // 2
        sos_f = butter(4, [8.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
        sos_h = butter(4, [21.0, 30.0], btype="bandpass", fs=sfreq, output="sos")
        zi_f = np.repeat(sosfilt_zi(sos_f)[:, None, :], n_ch, axis=1)
        zi_h = np.repeat(sosfilt_zi(sos_h)[:, None, :], n_ch, axis=1)
        A = rng.normal(size=(n_ch, n_ch))
        m_f = A @ A.T / n_ch + 0.5 * np.eye(n_ch)
        m_h = A @ A.T / n_ch + 0.5 * np.eye(n_ch)
        pca_f = np.linalg.qr(rng.normal(size=(d_tan, 30)))[0]
        pca_h = np.linalg.qr(rng.normal(size=(d_tan, 30)))[0]
        sc_mu = rng.normal(size=(60,))
        sc_sd = np.abs(rng.normal(size=(60,))) + 0.5
        coef = rng.normal(size=(60,))
        inter = 0.0
        raw_ep = rng.normal(size=(n_ch, n_t)) * 10.0
        lats = []
        for it in range(550):
            t0 = time.perf_counter()
            x_car = raw_ep - raw_ep.mean(0, keepdims=True)
            xf, zi_f = sosfilt(sos_f, x_car, axis=-1, zi=zi_f)
            xh, zi_h = sosfilt(sos_h, x_car, axis=-1, zi=zi_h)
            cf, _ = b2a.exact_oas(xf.T, assume_centered=False)
            ch_, _ = b2a.exact_oas(xh.T, assume_centered=False)
            tf = b2a.project_tangent(cf[None], m_f) @ pca_f
            th = b2a.project_tangent(ch_[None], m_h) @ pca_h
            x_feat = (np.concatenate([tf[0], th[0]]) - sc_mu) / sc_sd
            _ = float(np.dot(x_feat, coef) + inter)
            dt = (time.perf_counter() - t0) * 1000.0
            if it >= 50:
                lats.append(dt)
        a = np.asarray(lats)
        out[f"uncal_lat_{tag}__mean"] = np.asarray([a.mean()])
        out[f"uncal_lat_{tag}__std"] = np.asarray([a.std()])
        out[f"uncal_lat_{tag}__p95"] = np.asarray([np.percentile(a, 95)])
        print(f"  Uncal CPU Latency [{tag}ch]: {a.mean():.2f}+-{a.std():.2f} ms (P95={np.percentile(a,95):.2f})")

    # 109-fold single-process [0:] batch runtime for Uncal
    t_s = time.perf_counter()
    for i in range(109):
        tr = [eeg_subs[j] for j in range(109) if j != i]
        te = eeg_subs[i]
        y_tr = np.concatenate([b["y"] for b in tr], 0)
        cf_tr = np.concatenate([b["cov_f"] for b in tr], 0)
        ch_tr = np.concatenate([b["cov_h"] for b in tr], 0)
        mf = b2a.sym_expm(np.concatenate([b["log_f"] for b in tr], 0).mean(0))
        mh = b2a.sym_expm(np.concatenate([b["log_h"] for b in tr], 0).mean(0))
        tk_f = b2a.project_tangent(cf_tr, mf)
        tk_h = b2a.project_tangent(ch_tr, mh)
        pf = PCA(n_components=30, random_state=42).fit(tk_f)
        ph = PCA(n_components=30, random_state=42).fit(tk_h)
        X_tr = np.concatenate([pf.transform(tk_f), ph.transform(tk_h)], 1)
        sc = StandardScaler().fit(X_tr)
        clf = step2.fit_lr(sc.transform(X_tr), y_tr)
        X_te = np.concatenate([pf.transform(b2a.project_tangent(te["cov_f"], mf)),
                               ph.transform(b2a.project_tangent(te["cov_h"], mh))], 1)
        _ = clf.predict(sc.transform(X_te))
    dt_batch = time.perf_counter() - t_s
    out["uncal_1proc_full0_time"] = np.asarray([dt_batch])
    print(f"  Uncal 109-fold Single-Proc [0:] Runtime: {dt_batch:.2f} s")
    return out


def _burst_2a4_worker(idx: int) -> dict[str, float]:
    all_b = step2._G_STATE["all_blocks"]
    tr = [all_b[j][s] for j in range(9) if j != idx for s in ("T", "E")]
    te = ps.reorder_bursty_stream(all_b[idx]["E"], burst_len=6, seed=42 + idx)
    return step2.eval_fold_main(tr, te, return_probs=False)


def main() -> None:
    t0 = time.perf_counter()
    m = np.load(config.RESULTS_DIR / "step2_unified_causal_master.npz")
    dl = np.load(config.RESULTS_DIR / "step2_dl_and_latency.npz")
    s4 = np.load(config.RESULTS_DIR / "step4_fixes.npz")

    store: dict[str, np.ndarray] = {}

    protos = [
        "eegbci_bin_loso", "2a_bin_cs", "2a_bin_loso", "2a_4cls_cs", "2a_4cls_loso",
        "chang_bin", "chang_3cls", "kaya_bin_loso", "kaya_bin_cs", "kaya_3cls_loso",
    ]

    # 1. Zero-excluded exact p-values & recomputed Holm vs Uncal + cold-start [0:] vs Uncal
    raw_vs_unc = []
    for p in protos:
        pr = float(s4[f"stat__{p}__vs_Uncal__p"][0])
        raw_vs_unc.append(pr)
        p_cs_unc, rb_cs_unc, nz_cs = exact_wilcoxon_clean(m[f"{p}__CS__Within-Adapt__all"], m[f"{p}__CS__Uncal__all"])
        store[f"clean__{p}__raw_p"] = np.asarray([pr])
        store[f"clean__{p}__CS_all_vs_Uncal__p"] = np.asarray([p_cs_unc])
        store[f"clean__{p}__CS_all_vs_Uncal__rb"] = np.asarray([rb_cs_unc])
        store[f"clean__{p}__CS_all_vs_Uncal__nz"] = np.asarray([nz_cs])

    holm_vs_unc = holm_bonferroni(raw_vs_unc)
    print("[1] Zero-Excluded Exact Uncal Raw + Holm & Cold-Start [0:] p-values:")
    for p, pr, ph in zip(protos, raw_vs_unc, holm_vs_unc):
        store[f"clean__{p}__holm_p"] = np.asarray([ph])
        print(f"  {p:16s}: raw_p={pr:.4g} | holm_p={ph:.4g} | CS[0:] vs Uncal p={store[f'clean__{p}__CS_all_vs_Uncal__p'][0]:.4g}")

    # 2. Zero-excluded DL contrasts (Static EEGNet, Online EA+EEGNet, ShallowConvNet)
    print("\n[2] Zero-Excluded DL Contrasts:")
    for p in ["eegbci_bin_loso", "2a_bin_loso", "2a_4cls_loso", "chang_bin", "chang_3cls"]:
        wa = m[f"{p}__Within-Adapt"]
        for dkey in ["Static_EEGNet", "Online_EA_EEGNet"]:
            arr = dl[f"{p}__{dkey}__acc15"]
            pv, rb, nz = exact_wilcoxon_clean(wa, arr)
            store[f"dl_clean__{p}__vs_{dkey}__p"] = np.asarray([pv])
            store[f"dl_clean__{p}__vs_{dkey}__rb"] = np.asarray([rb])
            print(f"  {p:16s} vs {dkey:18s}: WA={wa.mean()*100:.2f}%, DL={arr.mean()*100:.2f}%, diff={(wa.mean()-arr.mean())*100:+.2f}%, p={pv:.4g} (nz={nz})")
        if p == "eegbci_bin_loso":
            sh = dl[f"{p}__Static_ShallowConvNet__acc15"]
            pv, rb, nz = exact_wilcoxon_clean(wa, sh)
            store[f"dl_clean__{p}__vs_Static_ShallowConvNet__p"] = np.asarray([pv])
            print(f"  {p:16s} vs Static_ShallowConvNet: p={pv:.4g} (nz={nz})")

    # Also full [0:] EEGBCI & 2a_4cls EA+EEGNet
    for p in ["eegbci_bin_loso", "2a_4cls_loso"]:
        wa0 = m[f"{p}__CS__Within-Adapt__all"]
        ea0 = dl[f"{p}__Online_EA_EEGNet__acc0"]
        pv, _, nz = exact_wilcoxon_clean(wa0, ea0)
        store[f"dl_clean__{p}__CS0_vs_EA__p"] = np.asarray([pv])
        print(f"  {p:16s} [0:] WA={wa0.mean()*100:.2f}% vs EA={ea0.mean()*100:.2f}% (p={pv:.4g}, nz={nz})")

    # 3. EEGBCI internal band & gate scale zero-excluded p-values
    print("\n[3] EEGBCI Internal Ablations (Zero-Excluded Exact Wilcoxon):")
    wa_eeg = m["eegbci_bin_loso__Within-Adapt"]
    for abk in ["full_only", "hb_only", "disjoint_mu_beta", "s_0.5", "s_2.0"]:
        pv, rb, nz = exact_wilcoxon_clean(wa_eeg, m[f"eeg_ab__{abk}"])
        store[f"eeg_ab_clean__vs_{abk}__p"] = np.asarray([pv])
        print(f"  WA vs {abk:18s}: {m[f'eeg_ab__{abk}'].mean()*100:.2f}% | p={pv:.4g} (nz={nz})")

    # 4. Unified 2a_4cls_burst6 verification
    with ProcessPoolExecutor(max_workers=8) as ex:
        b2a_subs = list(ex.map(step2.worker_b2a, range(1, 10)))
    all_b4 = [{s: step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2, 3, 4))) for s in ("T", "E")} for i in range(9)]
    with ProcessPoolExecutor(max_workers=8, initializer=step2._init_state, initargs=({"all_blocks": all_b4},)) as ex:
        brows = list(ex.map(_burst_2a4_worker, range(9)))
    print("\n[4] Unified 2a_4cls_burst6:")
    for mn in ["Uncal", "G1", "Orig", "z-CORAL (OAS-Src)", "Within-CORAL(0.7)", "Within-Adapt"]:
        arr = np.asarray([r[mn] for r in brows])
        store[f"b2a4_burst6__{mn}"] = arr
        print(f"  {mn:20s}: {arr.mean()*100:.2f}%")
    p_b_g1, _, _ = exact_wilcoxon_clean(store["b2a4_burst6__Within-Adapt"], store["b2a4_burst6__G1"])
    p_b_zc, _, _ = exact_wilcoxon_clean(store["b2a4_burst6__Within-Adapt"], store["b2a4_burst6__z-CORAL (OAS-Src)"])
    store["b2a4_burst6__vs_G1__p"] = np.asarray([p_b_g1])
    store["b2a4_burst6__vs_zCORAL__p"] = np.asarray([p_b_zc])
    print(f"  WA vs G1 p={p_b_g1:.4f} | WA vs z-CORAL p={p_b_zc:.4f}")

    # 5. Uncal exact batch runtime and single-trial latency
    print("\n[5] Uncal Exact 109-fold Runtime & Dense Latency:")
    with ProcessPoolExecutor(max_workers=8) as ex:
        eeg_subs = list(ex.map(step2.worker_eegbci, range(1, 110)))
    store.update(benchmark_uncal_latency_and_batch(eeg_subs))

    np.savez_compressed(OUT_NPZ, **store)
    print(f"\nSaved P0/P1 audit updates to {OUT_NPZ} in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
