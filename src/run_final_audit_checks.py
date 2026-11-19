"""
Final Pre-Submission Audit Checks (C1 TOST/CI, C2 z-Adapt Matched Ablation, M1 Causal MDM, M2 Holm)
=====================================================================================================
Computes:
  1. C2: Exact matched ablation removing ONLY within-class whitening Sigma_W^{-1/2}:
         z-Adapt (OAS-Src) runs exact Eqs. (6)-(8) (s=1.0, gamma in [0.25, 0.92], src_cov_z prior)
         in PCA-30 z-space vs Within-Adapt in v-space on EEGBCI, 2a Bin/4Cls CS, 2a 4Cls LOSO, Chang 3-Cls.
  2. M1: Causal Recentered Dual-Band Log-Euclidean MDM baseline across ALL 10 protocols (trials[15:]).
  3. C1: Paired difference (Within-Adapt - G1) Mean, 95% t-CI, and exact Wilcoxon TOST at +-1.5% and +-2.0%.
  4. M2: Holm-Bonferroni correction across the BCI-IV-2a contrasts and all 10 protocols vs z-CORAL (OAS-Src).

Saves results to results/final_audit_checks.npz.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import numpy as np
import scipy.stats as stats
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import kaya2018_causal_coral_repro as ky18
import run_p1_audit_updates as p1
import run_reviewer_revision_ablations as rev
import run_step2_riemannian_master as step2
import unified_within_adapt_coral_benchmark as uni

OUT_NPZ = config.RESULTS_DIR / "final_audit_checks.npz"
_STATE: dict = {}


def _init(state: dict) -> None:
    global _STATE
    _STATE = state


def _sym_logm_single(mat: np.ndarray) -> np.ndarray:
    return b2a.sym_logm(mat[None])[0]


def causal_recentered_mdm_fold(train_blocks: list[dict], test_blk: dict) -> float:
    """Dual-Band Causal W=8 Log-Euclidean Recentered MDM (Minimum Distance to Mean)."""
    y_tr = np.concatenate([b["y"] for b in train_blocks], axis=0)
    classes = np.unique(y_tr)

    # Recenter each source training block by its own Log-Euclidean block mean
    log_rec_f, log_rec_h = [], []
    for b in train_blocks:
        mf_inv_s = b2a.sym_inv_sqrtm(b2a.sym_expm(b["log_f"].mean(0)))
        mh_inv_s = b2a.sym_inv_sqrtm(b2a.sym_expm(b["log_h"].mean(0)))
        log_rec_f.append(b2a.sym_logm(mf_inv_s @ b["cov_f"] @ mf_inv_s))
        log_rec_h.append(b2a.sym_logm(mh_inv_s @ b["cov_h"] @ mh_inv_s))
    L_tr_f = np.concatenate(log_rec_f, axis=0)
    L_tr_h = np.concatenate(log_rec_h, axis=0)

    cents_f = np.stack([L_tr_f[y_tr == c].mean(axis=0) for c in classes], axis=0)
    cents_h = np.stack([L_tr_h[y_tr == c].mean(axis=0) for c in classes], axis=0)

    # Test time strictly causal W=8 recentering on trials[15:]
    n_te = len(test_blk["y"])
    W = step2.CAUSAL_WINDOW
    cov_f, cov_h = test_blk["cov_f"], test_blk["cov_h"]
    log_f, log_h = test_blk["log_f"], test_blk["log_h"]
    preds = np.empty(n_te - step2.EVAL_START_IDX, dtype=y_tr.dtype)

    for oi, t in enumerate(range(step2.EVAL_START_IDX, n_te)):
        mf_inv_s = b2a.sym_inv_sqrtm(b2a.sym_expm(log_f[t - W : t].mean(0)))
        mh_inv_s = b2a.sym_inv_sqrtm(b2a.sym_expm(log_h[t - W : t].mean(0)))
        lf_t = _sym_logm_single(mf_inv_s @ cov_f[t] @ mf_inv_s)
        lh_t = _sym_logm_single(mh_inv_s @ cov_h[t] @ mh_inv_s)
        df = np.sum((cents_f - lf_t[None]) ** 2, axis=(1, 2))
        dh = np.sum((cents_h - lh_t[None]) ** 2, axis=(1, 2))
        preds[oi] = classes[int(np.argmin(df + dh))]

    return float(np.mean(preds == test_blk["y"][step2.EVAL_START_IDX :]))


def _worker_mdm_task(spec: tuple) -> float:
    kind, idx = spec
    data = _STATE.get("blocks")
    if kind == "loso":
        tr, te = [data[j] for j in range(len(data)) if j != idx], data[idx]
    elif kind == "b2a_cs":
        tr, te = [_STATE["tr_blocks"][idx]], _STATE["te_blocks"][idx]
    elif kind == "b2a_loso":
        tr = [_STATE["all_blocks"][j][s] for j in range(9) if j != idx for s in ("T", "E")]
        te = _STATE["all_blocks"][idx]["E"]
    elif kind == "kaya_loso":
        subjs = _STATE["subjs"]
        tr, te = [data[j] for j in range(len(data)) if subjs[j] != subjs[idx]], data[idx]
    elif kind == "kaya_cs":
        tr_i, te_i = idx
        tr, te = [data[tr_i]], data[te_i]
    else:
        raise ValueError(kind)
    return causal_recentered_mdm_fold(tr, te)


def _eval_z_adapt_fold(train_blocks: list[dict], test_blk: dict) -> dict[str, float]:
    """Matched ablation: runs EXACT Eqs. (6)-(8) in PCA-30 z-space (no Sigma_W^{-1/2}) vs v-space."""
    y_tr = np.concatenate([b["y"] for b in train_blocks], axis=0)
    y_te = test_blk["y"][step2.EVAL_START_IDX :]

    rec_blocks = [
        (
            b2a.project_tangent(b["cov_f"], b2a.sym_expm(b["log_f"].mean(0))),
            b2a.project_tangent(b["cov_h"], b2a.sym_expm(b["log_h"].mean(0))),
        )
        for b in train_blocks
    ]
    pca_f = PCA(n_components=step2.PCA_DIM, random_state=step2.RANDOM_STATE).fit(np.concatenate([r[0] for r in rec_blocks], 0))
    pca_h = PCA(n_components=step2.PCA_DIM, random_state=step2.RANDOM_STATE).fit(np.concatenate([r[1] for r in rec_blocks], 0))
    zf_l, zh_l = [pca_f.transform(r[0]) for r in rec_blocks], [pca_h.transform(r[1]) for r in rec_blocks]
    zf_all, zh_all = np.concatenate(zf_l, 0), np.concatenate(zh_l, 0)

    # z-space source covariance & sqrt (NO Sigma_W whitening)
    sz_f, _ = b2a.exact_oas(zf_all - zf_all.mean(0, keepdims=True), assume_centered=True)
    sz_h, _ = b2a.exact_oas(zh_all - zh_all.mean(0, keepdims=True), assume_centered=True)
    szf_s, szh_s = uni.sym_sqrtm(sz_f), uni.sym_sqrtm(sz_h)

    # v-space operators (exact Within-Adapt)
    wfi, sfc, sfs = uni.fit_within_operator(zf_all, y_tr)
    whi, shc, shs = uni.fit_within_operator(zh_all, y_tr)

    X_tr_za = np.concatenate([
        np.concatenate([
            rev.within_adapt_clip(z, z, sz_f, szf_s, 1.0, 0.25, 0.92)[0],
            rev.within_adapt_clip(zh, zh, sz_h, szh_s, 1.0, 0.25, 0.92)[0],
        ], 1)
        for z, zh in zip(zf_l, zh_l)
    ], 0)
    X_tr_wa = np.concatenate([
        np.concatenate([
            rev.within_adapt_clip(z @ wfi, z @ wfi, sfc, sfs, 1.0, 0.25, 0.92)[0],
            rev.within_adapt_clip(zh @ whi, zh @ whi, shc, shs, 1.0, 0.25, 0.92)[0],
        ], 1)
        for z, zh in zip(zf_l, zh_l)
    ], 0)

    clf_za = step2.fit_lr(X_tr_za, y_tr)
    clf_wa = step2.fit_lr(X_tr_wa, y_tr)

    n_te = len(test_blk["y"])
    n_eval = n_te - step2.EVAL_START_IDX
    W = step2.CAUSAL_WINDOW
    X_te_za = np.empty((n_eval, 60), dtype=np.float64)
    X_te_wa = np.empty((n_eval, 60), dtype=np.float64)
    cov_f, cov_h = test_blk["cov_f"], test_blk["cov_h"]
    log_f, log_h = test_blk["log_f"], test_blk["log_h"]

    for oi, t in enumerate(range(step2.EVAL_START_IDX, n_te)):
        mf = b2a.sym_expm(log_f[t - W : t].mean(0))
        mh = b2a.sym_expm(log_h[t - W : t].mean(0))
        zf = pca_f.transform(b2a.project_tangent(cov_f[t - W : t + 1], mf))
        zh = pca_h.transform(b2a.project_tangent(cov_h[t - W : t + 1], mh))
        vf, vh = zf @ wfi, zh @ whi
        X_te_za[oi] = np.concatenate([
            rev.within_adapt_clip(zf[:-1], zf[-1:], sz_f, szf_s, 1.0, 0.25, 0.92)[0][0],
            rev.within_adapt_clip(zh[:-1], zh[-1:], sz_h, szh_s, 1.0, 0.25, 0.92)[0][0],
        ])
        X_te_wa[oi] = np.concatenate([
            rev.within_adapt_clip(vf[:-1], vf[-1:], sfc, sfs, 1.0, 0.25, 0.92)[0][0],
            rev.within_adapt_clip(vh[:-1], vh[-1:], shc, shs, 1.0, 0.25, 0.92)[0][0],
        ])

    return {
        "z_Adapt": float(np.mean(clf_za.predict(X_te_za) == y_te)),
        "Within_Adapt": float(np.mean(clf_wa.predict(X_te_wa) == y_te)),
    }


def _worker_z_adapt_task(spec: tuple) -> dict[str, float]:
    kind, idx = spec
    data = _STATE.get("blocks")
    if kind == "loso":
        tr, te = [data[j] for j in range(len(data)) if j != idx], data[idx]
    elif kind == "b2a_cs":
        tr, te = [_STATE["tr_blocks"][idx]], _STATE["te_blocks"][idx]
    elif kind == "b2a_loso":
        tr = [_STATE["all_blocks"][j][s] for j in range(9) if j != idx for s in ("T", "E")]
        te = _STATE["all_blocks"][idx]["E"]
    else:
        raise ValueError(kind)
    return _eval_z_adapt_fold(tr, te)


def tost_wilcoxon(diff: np.ndarray, bound: float) -> tuple[float, float, float, float]:
    """Paired 95% t-CI + one-sided Wilcoxon TOST p-value within [-bound, +bound]."""
    n = len(diff)
    m = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    ci_lo, ci_hi = m - t_crit * se, m + t_crit * se

    meth = "exact" if n <= 25 else "auto"
    d_lo = diff + bound
    d_lo = d_lo[np.abs(d_lo) > 1e-12]
    p1 = float(wilcoxon(d_lo, alternative="greater", method=meth if len(d_lo) <= 25 else "auto").pvalue) if len(d_lo) else 1.0

    d_hi = diff - bound
    d_hi = d_hi[np.abs(d_hi) > 1e-12]
    p2 = float(wilcoxon(d_hi, alternative="less", method=meth if len(d_hi) <= 25 else "auto").pvalue) if len(d_hi) else 1.0
    return m, ci_lo, ci_hi, max(p1, p2)


def main() -> None:
    t0 = time.perf_counter()
    store: dict[str, np.ndarray] = {}
    master = np.load(config.RESULTS_DIR / "step2_unified_causal_master.npz")

    print("=" * 94)
    print("FINAL PRE-SUBMISSION AUDIT CHECKS (C1 TOST/CI, C2 z-Adapt Ablation, M1 Causal MDM, M2 Holm)")
    print("=" * 94)

    # --------------------------------------------------------------------------
    # C1 & M2 from cached master results (instant)
    # --------------------------------------------------------------------------
    protos = [
        ("eegbci_bin_loso", "EEGBCI Bin LOSO (109)"),
        ("2a_bin_cs", "BCI-IV-2a Bin CS (9)"),
        ("2a_bin_loso", "BCI-IV-2a Bin LOSO (9)"),
        ("2a_4cls_cs", "BCI-IV-2a 4-Cls CS (9)"),
        ("2a_4cls_loso", "BCI-IV-2a 4-Cls LOSO (9)"),
        ("chang_bin", "Chang2025 Bin LOSO (28)"),
        ("chang_3cls", "Chang2025 3-Cls LOSO (28)"),
        ("kaya_bin_loso", "Kaya2018 Bin LOSO (7)"),
        ("kaya_bin_cs", "Kaya2018 Bin CS (5)"),
        ("kaya_3cls_loso", "Kaya2018 3-Cls LOSO (7)"),
    ]
    print("\n[C1] Paired Difference (Within-Adapt - G1) 95% CI & Wilcoxon TOST (±1.5% and ±2.0%):")
    for tag, label in protos:
        diff = master[f"{tag}__Within-Adapt"] - master[f"{tag}__G1"]
        m, lo15, hi15, p_t15 = tost_wilcoxon(diff, 0.015)
        _, _, _, p_t20 = tost_wilcoxon(diff, 0.020)
        store[f"c1__{tag}__diff"] = diff
        store[f"c1__{tag}__tost15"] = np.asarray([p_t15])
        store[f"c1__{tag}__tost20"] = np.asarray([p_t20])
        print(f"  {label:28s}: Delta={m*100:+5.2f}% [95% CI: {lo15*100:+5.2f}%, {hi15*100:+5.2f}%] | TOST(±1.5%) p={p_t15:.4g} | TOST(±2.0%) p={p_t20:.4g}")

    print("\n[M2] Holm-Bonferroni on BCI-IV-2a & Full 10-Protocol contrasts vs z-CORAL (OAS-Src):")
    p_zc_all = []
    for tag, _ in protos:
        pr, _, _ = p1.exact_wilcoxon_clean(master[f"{tag}__Within-Adapt"], master[f"{tag}__z-CORAL (OAS-Src)"])
        p_zc_all.append(pr)
    h10_zc = p1.holm_bonferroni(p_zc_all)
    b2a_tags = ["2a_bin_cs", "2a_4cls_cs", "2a_4cls_loso"]
    p_b2a = [p1.exact_wilcoxon_clean(master[f"{t}__Within-Adapt"], master[f"{t}__z-CORAL (OAS-Src)"])[0] for t in b2a_tags]
    h3_b2a = p1.holm_bonferroni(p_b2a)
    for t, pr, ph in zip(b2a_tags, p_b2a, h3_b2a):
        store[f"m2_b2a_{t}_raw_p"] = np.asarray([pr])
        store[f"m2_b2a_{t}_holm3_p"] = np.asarray([ph])
        print(f"  {t:14s}: raw p={pr:.4f} -> 3-contrast Holm p={ph:.4f}")
    for (tag, label), pr, ph in zip(protos, p_zc_all, h10_zc):
        store[f"m2_10_{tag}_zc_holm"] = np.asarray([ph])
        if pr < 0.05 or ph < 0.10:
            print(f"  10-proto vs z-CORAL [{label:24s}]: raw p={pr:.4g} -> 10-proto Holm p={ph:.4g}")

    # Load datasets once for C2 and M1
    with ProcessPoolExecutor() as ex:
        b2a_subs = list(ex.map(step2.worker_b2a, range(1, 10)))
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor() as ex:
        ch_subs = list(ex.map(step2.worker_chang, ch_dirs))
    mat_files = sorted(ky18.DATA_DIR.glob("*.mat"))
    with ProcessPoolExecutor() as ex:
        ky_sess = list(ex.map(step2.worker_kaya, mat_files))
    with ProcessPoolExecutor() as ex:
        eeg_subs = list(ex.map(step2.worker_eegbci, range(1, 110)))

    # --------------------------------------------------------------------------
    # C2: Exact matched z-Adapt (OAS-Src) vs Within-Adapt
    # --------------------------------------------------------------------------
    print("\n[C2] Matched Ablation: z-Adapt (OAS-Src, NO Sigma_W) vs Within-Adapt (WITH Sigma_W):")
    c2_specs = [
        ("eegbci_bin_loso", "loso", {"blocks": eeg_subs}, 109),
        (
            "2a_bin_cs",
            "b2a_cs",
            {
                "tr_blocks": [step2.subset_block(b2a_subs[i]["T"], np.isin(b2a_subs[i]["T"]["y"], (1, 2))) for i in range(9)],
                "te_blocks": [step2.subset_block(b2a_subs[i]["E"], np.isin(b2a_subs[i]["E"]["y"], (1, 2))) for i in range(9)],
            },
            9,
        ),
        (
            "2a_4cls_cs",
            "b2a_cs",
            {
                "tr_blocks": [step2.subset_block(b2a_subs[i]["T"], np.isin(b2a_subs[i]["T"]["y"], (1, 2, 3, 4))) for i in range(9)],
                "te_blocks": [step2.subset_block(b2a_subs[i]["E"], np.isin(b2a_subs[i]["E"]["y"], (1, 2, 3, 4))) for i in range(9)],
            },
            9,
        ),
        (
            "2a_4cls_loso",
            "b2a_loso",
            {"all_blocks": [{s: step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2, 3, 4))) for s in ("T", "E")} for i in range(9)]},
            9,
        ),
        (
            "chang_3cls",
            "loso",
            {"blocks": [step2.subset_block(s, np.isin(s["y"], (1, 2, 3))) for s in ch_subs]},
            len(ch_subs),
        ),
    ]
    for tag, kind, st, n_f in c2_specs:
        with ProcessPoolExecutor(initializer=_init, initargs=(st,)) as ex:
            rows = list(ex.map(_worker_z_adapt_task, [(kind, i) for i in range(n_f)]))
        za = np.asarray([r["z_Adapt"] for r in rows], dtype=np.float64)
        wa = np.asarray([r["Within_Adapt"] for r in rows], dtype=np.float64)
        zc_raw = master[f"{tag}__z-CORAL (OAS-Src)"]
        g1_raw = master[f"{tag}__G1"]
        p_w_za, _, _ = p1.exact_wilcoxon_clean(wa, za)
        p_za_zc, _, _ = p1.exact_wilcoxon_clean(za, zc_raw)
        store[f"c2__{tag}__z_Adapt"] = za
        store[f"c2__{tag}__Within_Adapt"] = wa
        print(
            f"  {tag:16s}: z-CORAL={zc_raw.mean()*100:5.2f}% | z-Adapt={za.mean()*100:5.2f}% | "
            f"Within-Adapt={wa.mean()*100:5.2f}% | G1={g1_raw.mean()*100:5.2f}% | "
            f"WA vs z-Adapt: {(wa-za).mean()*100:+5.2f}% (p={p_w_za:.4g}) | z-Adapt vs z-CORAL p={p_za_zc:.4g}"
        )

    # --------------------------------------------------------------------------
    # M1: Causal Recentered Dual-Band Log-Euclidean MDM across ALL 10 protocols
    # --------------------------------------------------------------------------
    print("\n[M1] Dual-Band Causal Recentered MDM Baseline (trials[15:]) across 10 protocols:")
    mdm_jobs = [
        ("eegbci_bin_loso", "loso", {"blocks": eeg_subs}, range(109), None),
        (
            "2a_bin_cs",
            "b2a_cs",
            {
                "tr_blocks": [step2.subset_block(b2a_subs[i]["T"], np.isin(b2a_subs[i]["T"]["y"], (1, 2))) for i in range(9)],
                "te_blocks": [step2.subset_block(b2a_subs[i]["E"], np.isin(b2a_subs[i]["E"]["y"], (1, 2))) for i in range(9)],
            },
            range(9),
            None,
        ),
        (
            "2a_bin_loso",
            "b2a_loso",
            {"all_blocks": [{s: step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2))) for s in ("T", "E")} for i in range(9)]},
            range(9),
            None,
        ),
        (
            "2a_4cls_cs",
            "b2a_cs",
            {
                "tr_blocks": [step2.subset_block(b2a_subs[i]["T"], np.isin(b2a_subs[i]["T"]["y"], (1, 2, 3, 4))) for i in range(9)],
                "te_blocks": [step2.subset_block(b2a_subs[i]["E"], np.isin(b2a_subs[i]["E"]["y"], (1, 2, 3, 4))) for i in range(9)],
            },
            range(9),
            None,
        ),
        (
            "2a_4cls_loso",
            "b2a_loso",
            {"all_blocks": [{s: step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2, 3, 4))) for s in ("T", "E")} for i in range(9)]},
            range(9),
            None,
        ),
        ("chang_bin", "loso", {"blocks": [step2.subset_block(s, np.isin(s["y"], (1, 2))) for s in ch_subs]}, range(len(ch_subs)), None),
        ("chang_3cls", "loso", {"blocks": [step2.subset_block(s, np.isin(s["y"], (1, 2, 3))) for s in ch_subs]}, range(len(ch_subs)), None),
    ]

    ky_bin = [step2.subset_block(s, np.isin(s["y"], (1, 2))) for s in ky_sess]
    ky_bin_subj = [s["subj"] for s in ky_sess]
    mdm_jobs.append(("kaya_bin_loso", "kaya_loso", {"blocks": ky_bin, "subjs": ky_bin_subj}, range(len(ky_bin)), ky_bin_subj))

    cs_pairs, cs_subjs = [], []
    for sn in sorted(set(ky_bin_subj)):
        idxs = [i for i, x in enumerate(ky_bin_subj) if x == sn]
        if len(idxs) >= 2:
            for ti in idxs[1:]:
                cs_pairs.append((idxs[0], ti))
                cs_subjs.append(sn)
    mdm_jobs.append(("kaya_bin_cs", "kaya_cs", {"blocks": ky_bin}, cs_pairs, cs_subjs))

    ky3_idx = [i for i, s in enumerate(ky_sess) if np.all(np.isin([1, 2, 3], s["y"]))]
    ky3 = [step2.subset_block(ky_sess[i], np.isin(ky_sess[i]["y"], (1, 2, 3))) for i in ky3_idx]
    ky3_subj = [ky_sess[i]["subj"] for i in ky3_idx]
    mdm_jobs.append(("kaya_3cls_loso", "kaya_loso", {"blocks": ky3, "subjs": ky3_subj}, range(len(ky3)), ky3_subj))

    mdm_pvals = []
    mdm_tags = []
    for tag, kind, st, items, subjs in mdm_jobs:
        with ProcessPoolExecutor(initializer=_init, initargs=(st,)) as ex:
            vals = list(ex.map(_worker_mdm_task, [(kind, it) for it in items]))
        if subjs is not None:
            uniq = sorted(set(subjs))
            arr = np.asarray([float(np.mean([vals[i] for i, s in enumerate(subjs) if s == u])) for u in uniq], dtype=np.float64)
        else:
            arr = np.asarray(vals, dtype=np.float64)
        store[f"mdm__{tag}"] = arr
        wa = master[f"{tag}__Within-Adapt"]
        p_m, _, _ = p1.exact_wilcoxon_clean(wa, arr)
        mdm_pvals.append(p_m)
        mdm_tags.append(tag)

    mdm_holm = p1.holm_bonferroni(mdm_pvals)
    for tag, pr, ph in zip(mdm_tags, mdm_pvals, mdm_holm):
        arr = store[f"mdm__{tag}"]
        wa = master[f"{tag}__Within-Adapt"]
        print(f"  {tag:16s}: MDM={arr.mean()*100:5.2f}% | WA={wa.mean()*100:5.2f}% (Delta={(wa-arr).mean()*100:+6.2f}%, raw p={pr:.4g}, Holm10 p={ph:.4g})")

    np.savez_compressed(OUT_NPZ, **store)
    print(f"\nSaved {OUT_NPZ} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
