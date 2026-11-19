"""
Controlled Physical Boundary Stress Test: Dispersion Shift & Asymmetric Channel Drift
====================================================================================
Evaluates G1, Causal RPA (W=8), z-CORAL (OAS-Src), Within-CORAL(0.7), and Within-Adapt
on BCI-IV-2a 4-Cls (N=9) and Chang2025 3-Cls (N=28) under three controlled physical
perturbations applied strictly to the held-out test stream:
  - disp_3x        : Log-Euclidean dispersion amplified 3.0x around target session mean
  - disp_0.25x     : Log-Euclidean dispersion attenuated 0.25x around target session mean
  - aniso_ch_drift : Asymmetric non-stationary variance drift on 3 channels (std=1.8)
Saves deterministic per-subject arrays and exact zero-excluded Wilcoxon p-values to
results/dispersion_drift_stress.npz.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import run_p1_audit_updates as p1a
import run_step2_riemannian_master as step2
import unified_within_adapt_coral_benchmark as uni

OUT_NPZ = config.RESULTS_DIR / "dispersion_drift_stress.npz"
MODES = ("disp_3x", "disp_0.25x", "aniso_ch_drift")


def _stress_fold_worker(args: tuple) -> dict[str, dict[str, float]]:
    tag, idx, is_b2a = args
    blocks = step2._G_STATE["blocks"]
    n_sub = len(blocks)
    if is_b2a:
        tr = [blocks[j][s] for j in range(n_sub) if j != idx for s in ("T", "E")]
        te = blocks[idx]["E"]
    else:
        tr = [blocks[j] for j in range(n_sub) if j != idx]
        te = blocks[idx]

    y_tr = np.concatenate([b["y"] for b in tr], 0)
    y_te = te["y"][15:]
    rec = [
        (
            b2a.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(0))),
            b2a.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(0))),
        )
        for b in tr
    ]
    tf_all = np.concatenate([r[0] for r in rec], 0)
    th_all = np.concatenate([r[1] for r in rec], 0)

    rpa_sf = max(float(np.sqrt(np.mean(np.sum(tf_all ** 2, 1)))), 1e-8)
    rpa_sh = max(float(np.sqrt(np.mean(np.sum(th_all ** 2, 1)))), 1e-8)
    rpa_tr = [
        (
            rf * (rpa_sf / max(float(np.sqrt(np.mean(np.sum(rf ** 2, 1)))), 1e-8)),
            rh * (rpa_sh / max(float(np.sqrt(np.mean(np.sum(rh ** 2, 1)))), 1e-8)),
        )
        for rf, rh in rec
    ]
    p_rf = PCA(n_components=30, random_state=42).fit(np.concatenate([r[0] for r in rpa_tr], 0))
    p_rh = PCA(n_components=30, random_state=42).fit(np.concatenate([r[1] for r in rpa_tr], 0))
    Xrpa = np.concatenate([np.concatenate([p_rf.transform(rf), p_rh.transform(rh)], 1) for rf, rh in rpa_tr], 0)
    scr = StandardScaler().fit(Xrpa)
    crpa = step2.fit_lr(scr.transform(Xrpa), y_tr)

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

    cg1 = step2.fit_lr(np.concatenate([np.concatenate([scf.transform(zf), sch.transform(zh)], 1) for zf, zh in zip(zf_l, zh_l)], 0), y_tr)
    czc = step2.fit_lr(np.concatenate([np.concatenate([step2.z_coral_oas_src(zf, zf, szfs), step2.z_coral_oas_src(zh, zh, szhs)], 1) for zf, zh in zip(zf_l, zh_l)], 0), y_tr)
    cwc = step2.fit_lr(np.concatenate([np.concatenate([uni.within_fixed_transform(zf @ wfi, zf @ wfi, 0.7), uni.within_fixed_transform(zh @ whi, zh @ whi, 0.7)], 1) for zf, zh in zip(zf_l, zh_l)], 0), y_tr)
    cwa = step2.fit_lr(np.concatenate([np.concatenate([step2.within_adapt_s(zf @ wfi, zf @ wfi, sfc, sfs, 1.0), step2.within_adapt_s(zh @ whi, zh @ whi, shc, shs, 1.0)], 1) for zf, zh in zip(zf_l, zh_l)], 0), y_tr)

    fold_out = {}
    n_te = len(te["y"])
    n_ch = te["log_f"].shape[1]
    for mode in MODES:
        log_f, log_h = te["log_f"].copy(), te["log_h"].copy()
        if mode == "disp_3x":
            mf0, mh0 = log_f.mean(0), log_h.mean(0)
            log_f = mf0 + 3.0 * (log_f - mf0)
            log_h = mh0 + 3.0 * (log_h - mh0)
        elif mode == "disp_0.25x":
            mf0, mh0 = log_f.mean(0), log_h.mean(0)
            log_f = mf0 + 0.25 * (log_f - mf0)
            log_h = mh0 + 0.25 * (log_h - mh0)
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

        Xg1, Xrpa, Xzc, Xwc, Xwa = [np.empty((n_te - 15, 60), dtype=np.float64) for _ in range(5)]
        for oi, t in enumerate(range(15, n_te)):
            mf = b2a.sym_expm(log_f[t - 8 : t].mean(0))
            mh = b2a.sym_expm(log_h[t - 8 : t].mean(0))
            tfw = b2a.project_tangent(cov_f[t - 8 : t + 1], mf)
            thw = b2a.project_tangent(cov_h[t - 8 : t + 1], mh)
            dfw = max(float(np.sqrt(np.mean(np.sum(tfw[:-1] ** 2, 1)))), 1e-8)
            dhw = max(float(np.sqrt(np.mean(np.sum(thw[:-1] ** 2, 1)))), 1e-8)
            Xrpa[oi] = np.concatenate([p_rf.transform(tfw[-1:] * (rpa_sf / dfw))[0], p_rh.transform(thw[-1:] * (rpa_sh / dhw))[0]])
            zf, zh = pf.transform(tfw), ph.transform(thw)
            vf, vh = zf @ wfi, zh @ whi
            Xg1[oi] = np.concatenate([scf.transform(zf[-1:])[0], sch.transform(zh[-1:])[0]])
            Xzc[oi] = np.concatenate([step2.z_coral_oas_src(zf[:-1], zf[-1:], szfs)[0], step2.z_coral_oas_src(zh[:-1], zh[-1:], szhs)[0]])
            Xwc[oi] = np.concatenate([uni.within_fixed_transform(vf[:-1], vf[-1:], 0.7)[0], uni.within_fixed_transform(vh[:-1], vh[-1:], 0.7)[0]])
            Xwa[oi] = np.concatenate([step2.within_adapt_s(vf[:-1], vf[-1:], sfc, sfs, 1.0)[0], step2.within_adapt_s(vh[:-1], vh[-1:], shc, shs, 1.0)[0]])

        fold_out[mode] = {
            "G1": float(np.mean(cg1.predict(Xg1) == y_te)),
            "RPA": float(np.mean(crpa.predict(scr.transform(Xrpa)) == y_te)),
            "zCORAL": float(np.mean(czc.predict(Xzc) == y_te)),
            "WC07": float(np.mean(cwc.predict(Xwc) == y_te)),
            "WA": float(np.mean(cwa.predict(Xwa) == y_te)),
        }
    return fold_out


def main() -> None:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        b2a_subs = list(ex.map(step2.worker_b2a, range(1, 10)))
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor(max_workers=8) as ex:
        ch_subs = list(ex.map(step2.worker_chang, ch_dirs))

    b2a_4 = [{s: step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2, 3, 4))) for s in ("T", "E")} for i in range(9)]
    ch_3 = [step2.subset_block(s, np.isin(s["y"], (1, 2, 3))) for s in ch_subs]

    store = {}
    for tag, blocks, is_b2a in [("2a_4cls", b2a_4, True), ("chang_3cls", ch_3, False)]:
        with ProcessPoolExecutor(max_workers=8, initializer=step2._init_state, initargs=({"blocks": blocks},)) as ex:
            rows = list(ex.map(_stress_fold_worker, [(tag, i, is_b2a) for i in range(len(blocks))]))
        for mode in MODES:
            for mname in ("G1", "RPA", "zCORAL", "WC07", "WA"):
                store[f"{tag}__{mode}__{mname}"] = np.asarray([r[mode][mname] for r in rows], dtype=np.float64)
            wa = store[f"{tag}__{mode}__WA"]
            p_str = []
            for comp in ("G1", "RPA", "zCORAL", "WC07"):
                pv, rb, _ = p1a.exact_wilcoxon_clean(wa, store[f"{tag}__{mode}__{comp}"])
                store[f"{tag}__{mode}__vs_{comp}__p"] = np.asarray([pv])
                store[f"{tag}__{mode}__vs_{comp}__rb"] = np.asarray([rb])
                p_str.append(f"vs {comp}={pv:.4f}")
            print(
                f"[{tag:10s} | {mode:15s}] "
                + " | ".join(f"{k}={store[f'{tag}__{mode}__{k}'].mean()*100:.2f}%" for k in ("G1", "RPA", "zCORAL", "WC07", "WA"))
                + " || " + ", ".join(p_str)
            )

    np.savez_compressed(OUT_NPZ, **store)
    print(f"\nSaved controlled stress results to {OUT_NPZ} in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
