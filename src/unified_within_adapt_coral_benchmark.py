"""
Unified 4-Dataset Cross-Validation of Adaptive & Within-Class CORAL Fixes
=========================================================================
Evaluates 6 methods with max_iter=3000 (zero convergence warnings) across ALL
4 independent MI datasets (153 subjects total, 10 protocols, trials[15:]):
  1. Baseline (K=0): Global training Riemannian mean + PCA-30 + StandardScaler
  2. G1 (1st-order W=8): Causal W=8 Riemannian mean + PCA-30 + StandardScaler
  3. OAS-CORAL (Orig): Original W=8 OAS whitening to I_30 (no post-scaler)
  4. Adapt-CORAL (s=0.8): Full mean + u-space adaptive shrinkage toward I_30
  5. Within-CORAL (g=0.7): Source within-class whitened space + g=0.7 prior
  6. Within-Adapt (SrcColor): Source within-class whitened space + adaptive
     alignment back to source total covariance Sigma_v^{(S)}

Datasets & Protocols:
  - EEGBCI (N=109): Binary L/R 109-fold LOSO
  - BCI-IV-2a (N=9): Binary CS, Binary LOSO, 4-Class CS, 4-Class LOSO
  - Kaya2018 (N=7, 17 sess): Binary Subject-LOSO, Binary CS, 3-Class Subject-LOSO
  - Chang2025 (N=28): Binary LOSO, 3-Class LOSO
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import eegbci_reviewer_ablations as eeg_ab
import kaya2018_causal_coral_repro as ky18

OUTPUT_NPZ = config.RESULTS_DIR / "unified_4dataset_results.npz"

EVAL_START_IDX = 15
CAUSAL_WINDOW = 8
PCA_DIM = 30
LOGREG_C = 0.05
LOGREG_MAX_ITER = 3000
RANDOM_STATE = 42
SPD_EPS = 1e-10

METHODS = [
    "Baseline (K=0)",
    "G1 (1st-order)",
    "OAS-CORAL (Orig)",
    "Adapt-CORAL",
    "Within-CORAL(0.7)",
    "Within-Adapt",
]


def sym_sqrtm(mat: np.ndarray) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, SPD_EPS)
    return (vecs * np.sqrt(vals)) @ vecs.T


def u_adapt_transform(U_fit: np.ndarray, U_apply: np.ndarray, scale: float = 0.8) -> np.ndarray:
    mean_u = U_fit.mean(axis=0, keepdims=True)
    cov_u, _ = b2a.exact_oas(U_fit - mean_u, assume_centered=True)
    d = U_fit.shape[1]
    eye = np.eye(d, dtype=np.float64)
    dist = np.linalg.norm(cov_u - eye, ord="fro") / np.sqrt(d)
    gamma = float(np.clip(np.exp(-scale * dist), 0.05, 0.95))
    cov_tilde = (1.0 - gamma) * cov_u + gamma * eye
    return (U_apply - mean_u) @ b2a.sym_inv_sqrtm(cov_tilde)


def within_fixed_transform(V_fit: np.ndarray, V_apply: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    mean_v = V_fit.mean(axis=0, keepdims=True)
    cov_v, _ = b2a.exact_oas(V_fit - mean_v, assume_centered=True)
    d = V_fit.shape[1]
    cov_tilde = (1.0 - gamma) * cov_v + gamma * np.eye(d, dtype=np.float64)
    return (V_apply - mean_v) @ b2a.sym_inv_sqrtm(cov_tilde)


def within_adapt_srccolor(
    V_fit: np.ndarray,
    V_apply: np.ndarray,
    src_cov_v: np.ndarray,
    src_sqrt_v: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    mean_v = V_fit.mean(axis=0, keepdims=True)
    cov_v, _ = b2a.exact_oas(V_fit - mean_v, assume_centered=True)
    denom = max(float(np.linalg.norm(src_cov_v, ord="fro")), 1e-8)
    dist = float(np.linalg.norm(cov_v - src_cov_v, ord="fro")) / denom
    gamma = float(np.clip(np.exp(-scale * dist), 0.25, 0.92))
    cov_tilde = (1.0 - gamma) * cov_v + gamma * src_cov_v
    mat = b2a.sym_inv_sqrtm(cov_tilde) @ src_sqrt_v
    return (V_apply - mean_v) @ mat


def fit_within_operator(Z_all: np.ndarray, y_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(y_tr)
    res = np.concatenate([Z_all[y_tr == c] - Z_all[y_tr == c].mean(axis=0, keepdims=True) for c in classes], axis=0)
    w_cov, _ = b2a.exact_oas(res, assume_centered=True)
    w_inv_sqrt = b2a.sym_inv_sqrtm(w_cov)
    V_all = Z_all @ w_inv_sqrt
    src_cov_v, _ = b2a.exact_oas(V_all - V_all.mean(axis=0, keepdims=True), assume_centered=True)
    src_sqrt_v = sym_sqrtm(src_cov_v)
    return w_inv_sqrt, src_cov_v, src_sqrt_v


def eval_single_fold(
    train_blocks: list[dict],
    test_blk: dict,
    proj_fn,
    rec_blocks: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, float]:
    """Unified leak-free fold evaluation across all 6 methods with shared PCA & window projections."""
    y_tr = np.concatenate([b["y"] for b in train_blocks], axis=0)
    y_te = test_blk["y"][EVAL_START_IDX:]

    # 1. Baseline (K=0)
    cov_tr_f = np.concatenate([b["cov_f"] for b in train_blocks], axis=0)
    cov_tr_h = np.concatenate([b["cov_h"] for b in train_blocks], axis=0)
    m_tr_f = b2a.sym_expm(np.concatenate([b["log_f"] for b in train_blocks], axis=0).mean(axis=0))
    m_tr_h = b2a.sym_expm(np.concatenate([b["log_h"] for b in train_blocks], axis=0).mean(axis=0))
    v_k0_f = proj_fn(cov_tr_f, m_tr_f)
    v_k0_h = proj_fn(cov_tr_h, m_tr_h)
    pca_k0_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_k0_f)
    pca_k0_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_k0_h)
    X_tr_k0 = np.concatenate([pca_k0_f.transform(v_k0_f), pca_k0_h.transform(v_k0_h)], axis=1)
    X_te_k0 = np.concatenate(
        [
            pca_k0_f.transform(proj_fn(test_blk["cov_f"][EVAL_START_IDX:], m_tr_f)),
            pca_k0_h.transform(proj_fn(test_blk["cov_h"][EVAL_START_IDX:], m_tr_h)),
        ],
        axis=1,
    )
    clf_k0 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=LOGREG_MAX_ITER, C=LOGREG_C, random_state=RANDOM_STATE)),
    ]).fit(X_tr_k0, y_tr)
    res_accs = {"Baseline (K=0)": float(np.mean(clf_k0.predict(X_te_k0) == y_te))}

    # 2. Training-side recentering & PCA(30)
    if rec_blocks is None:
        rec_blocks = [
            (
                proj_fn(b["cov_f"], b2a.sym_expm(b["log_f"].mean(axis=0))),
                proj_fn(b["cov_h"], b2a.sym_expm(b["log_h"].mean(axis=0))),
            )
            for b in train_blocks
        ]

    pca_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate([r[0] for r in rec_blocks]))
    pca_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate([r[1] for r in rec_blocks]))

    zf_list = [pca_f.transform(r[0]) for r in rec_blocks]
    zh_list = [pca_h.transform(r[1]) for r in rec_blocks]
    zf_all = np.concatenate(zf_list, axis=0)
    zh_all = np.concatenate(zh_list, axis=0)

    sc_f = StandardScaler().fit(zf_all)
    sc_h = StandardScaler().fit(zh_all)
    wf_inv, sf_cov_v, sf_sqrt_v = fit_within_operator(zf_all, y_tr)
    wh_inv, sh_cov_v, sh_sqrt_v = fit_within_operator(zh_all, y_tr)

    adapt_methods = METHODS[1:]
    X_tr = {m: [] for m in adapt_methods}
    for zf, zh in zip(zf_list, zh_list):
        uf, uh = sc_f.transform(zf), sc_h.transform(zh)
        vf, vh = zf @ wf_inv, zh @ wh_inv
        X_tr["G1 (1st-order)"].append(np.concatenate([uf, uh], axis=1))
        X_tr["OAS-CORAL (Orig)"].append(
            np.concatenate([b2a.coral_fit_transform(zf, zf), b2a.coral_fit_transform(zh, zh)], axis=1)
        )
        X_tr["Adapt-CORAL"].append(np.concatenate([u_adapt_transform(uf, uf), u_adapt_transform(uh, uh)], axis=1))
        X_tr["Within-CORAL(0.7)"].append(
            np.concatenate([within_fixed_transform(vf, vf, 0.7), within_fixed_transform(vh, vh, 0.7)], axis=1)
        )
        X_tr["Within-Adapt"].append(
            np.concatenate(
                [
                    within_adapt_srccolor(vf, vf, sf_cov_v, sf_sqrt_v),
                    within_adapt_srccolor(vh, vh, sh_cov_v, sh_sqrt_v),
                ],
                axis=1,
            )
        )

    clfs = {
        m: LogisticRegression(max_iter=LOGREG_MAX_ITER, C=LOGREG_C, random_state=RANDOM_STATE).fit(
            np.concatenate(X_tr[m], axis=0), y_tr
        )
        for m in adapt_methods
    }

    # 3. Single shared causal W=8 pass on test session trials[15:]
    n_te = len(test_blk["y"])
    n_eval = n_te - EVAL_START_IDX
    X_te = {m: np.empty((n_eval, PCA_DIM * 2), dtype=np.float64) for m in adapt_methods}
    cov_f, cov_h = test_blk["cov_f"], test_blk["cov_h"]
    log_f, log_h = test_blk["log_f"], test_blk["log_h"]

    for out_idx, t in enumerate(range(EVAL_START_IDX, n_te)):
        start = max(0, t - CAUSAL_WINDOW)
        mf = b2a.sym_expm(log_f[start:t].mean(axis=0))
        mh = b2a.sym_expm(log_h[start:t].mean(axis=0))
        zf = pca_f.transform(proj_fn(cov_f[start : t + 1], mf))
        zh = pca_h.transform(proj_fn(cov_h[start : t + 1], mh))
        uf, uh = sc_f.transform(zf), sc_h.transform(zh)
        vf, vh = zf @ wf_inv, zh @ wh_inv

        X_te["G1 (1st-order)"][out_idx] = np.concatenate([uf[-1], uh[-1]])
        X_te["OAS-CORAL (Orig)"][out_idx] = np.concatenate(
            [b2a.coral_fit_transform(zf[:-1], zf[-1:])[0], b2a.coral_fit_transform(zh[:-1], zh[-1:])[0]]
        )
        X_te["Adapt-CORAL"][out_idx] = np.concatenate(
            [u_adapt_transform(uf[:-1], uf[-1:])[0], u_adapt_transform(uh[:-1], uh[-1:])[0]]
        )
        X_te["Within-CORAL(0.7)"][out_idx] = np.concatenate(
            [within_fixed_transform(vf[:-1], vf[-1:], 0.7)[0], within_fixed_transform(vh[:-1], vh[-1:], 0.7)[0]]
        )
        X_te["Within-Adapt"][out_idx] = np.concatenate(
            [
                within_adapt_srccolor(vf[:-1], vf[-1:], sf_cov_v, sf_sqrt_v)[0],
                within_adapt_srccolor(vh[:-1], vh[-1:], sh_cov_v, sh_sqrt_v)[0],
            ]
        )

    for m in adapt_methods:
        res_accs[m] = float(np.mean(clfs[m].predict(X_te[m]) == y_te))
    return res_accs


def summarize_protocol(title: str, rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    arrs = {m: np.asarray([r[m] for r in rows], dtype=np.float64) for m in METHODS}
    print(f"\n[{title}] (N={len(rows)})")
    print("  " + " | ".join(f"{m}: {arrs[m].mean()*100:5.2f}%±{arrs[m].std(ddof=1)*100:4.2f}%" for m in METHODS))
    # Paired p-value vs OAS-CORAL (Orig) and G1 for the two unified candidates
    for cand in ["Adapt-CORAL", "Within-CORAL(0.7)", "Within-Adapt"]:
        p_g1 = float(wilcoxon(arrs[cand], arrs["G1 (1st-order)"])[1]) if not np.allclose(arrs[cand], arrs["G1 (1st-order)"]) else 1.0
        p_orig = float(wilcoxon(arrs[cand], arrs["OAS-CORAL (Orig)"])[1]) if not np.allclose(arrs[cand], arrs["OAS-CORAL (Orig)"]) else 1.0
        p_k0 = float(wilcoxon(arrs[cand], arrs["Baseline (K=0)"])[1]) if not np.allclose(arrs[cand], arrs["Baseline (K=0)"]) else 1.0
        print(
            f"    {cand:<18}: vs K=0 {((arrs[cand]-arrs['Baseline (K=0)']).mean()*100):+5.2f}% (p={p_k0:.4g}) | "
            f"vs G1 {((arrs[cand]-arrs['G1 (1st-order)']).mean()*100):+5.2f}% (p={p_g1:.4g}) | "
            f"vs Orig {((arrs[cand]-arrs['OAS-CORAL (Orig)']).mean()*100):+5.2f}% (p={p_orig:.4g})"
        )
    return arrs


def main() -> None:
    t0 = time.perf_counter()
    saved_results: dict[str, np.ndarray] = {}
    max_workers = min(6, os.cpu_count() or 4)

    # 1. BCI-IV-2a (N=9)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        b2a_subs = list(ex.map(b2a.load_subject_worker, list(range(1, 10))))

    for cname, cls_tup in [("2a_bin", (1, 2)), ("2a_4cls", (1, 2, 3, 4))]:
        filt = [{"T": b2a.filter_session(s["T"], cls_tup), "E": b2a.filter_session(s["E"], cls_tup)} for s in b2a_subs]
        # Precompute recentered training blocks
        rec_T = [
            (b2a.project_tangent(s["T"]["cov_f"], b2a.sym_expm(s["T"]["log_f"].mean(0))),
             b2a.project_tangent(s["T"]["cov_h"], b2a.sym_expm(s["T"]["log_h"].mean(0))))
            for s in filt
        ]
        rec_E = [
            (b2a.project_tangent(s["E"]["cov_f"], b2a.sym_expm(s["E"]["log_f"].mean(0))),
             b2a.project_tangent(s["E"]["cov_h"], b2a.sym_expm(s["E"]["log_h"].mean(0))))
            for s in filt
        ]
        cs_rows = [eval_single_fold([filt[i]["T"]], filt[i]["E"], b2a.project_tangent, [rec_T[i]]) for i in range(9)]
        loso_rows = [
            eval_single_fold(
                [filt[j][k] for j in range(9) if j != i for k in ("T", "E")],
                filt[i]["E"],
                b2a.project_tangent,
                [r for j in range(9) if j != i for r in (rec_T[j], rec_E[j])],
            )
            for i in range(9)
        ]
        for tag, rws in [(f"{cname}_cs", cs_rows), (f"{cname}_loso", loso_rows)]:
            arrs = summarize_protocol(tag, rws)
            for m, v in arrs.items():
                saved_results[f"{tag}__{m}"] = v

    # 2. Chang2025 (N=28)
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ch_subs = list(ex.map(ch25.load_subject_worker, ch_dirs))
    for cname, cls_tup in [("chang_bin", (1, 2)), ("chang_3cls", (1, 2, 3))]:
        filt = [ch25.filter_subject(s, cls_tup) for s in ch_subs]
        recs = [
            (ch25.project_tangent(s["cov_f"], ch25.sym_expm(s["log_f"].mean(0))),
             ch25.project_tangent(s["cov_h"], ch25.sym_expm(s["log_h"].mean(0))))
            for s in filt
        ]
        rows = [
            eval_single_fold(
                [filt[j] for j in range(len(filt)) if j != i],
                filt[i],
                ch25.project_tangent,
                [recs[j] for j in range(len(filt)) if j != i],
            )
            for i in range(len(filt))
        ]
        arrs = summarize_protocol(cname, rows)
        for m, v in arrs.items():
            saved_results[f"{cname}__{m}"] = v

    # 3. Kaya2018 (17 sessions)
    mat_files = sorted(ky18.DATA_DIR.glob("*.mat"))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ky_sess = list(ex.map(ky18.load_session_worker, mat_files))

    bin_ky = [ky18.filter_session(s, (1, 2)) for s in ky_sess]
    rec_ky_bin = [
        (ky18.project_tangent(s["cov_f"], ky18.sym_expm(s["log_f"].mean(0))),
         ky18.project_tangent(s["cov_h"], ky18.sym_expm(s["log_h"].mean(0))))
        for s in bin_ky
    ]
    ky_bin_loso = [
        eval_single_fold(
            [bin_ky[j] for j in range(len(bin_ky)) if bin_ky[j]["subj"] != bin_ky[i]["subj"]],
            bin_ky[i],
            ky18.project_tangent,
            [rec_ky_bin[j] for j in range(len(bin_ky)) if bin_ky[j]["subj"] != bin_ky[i]["subj"]],
        )
        for i in range(len(bin_ky))
    ]
    arrs = summarize_protocol("kaya_bin_loso", ky_bin_loso)
    for m, v in arrs.items():
        saved_results[f"kaya_bin_loso__{m}"] = v

    ky_cs_rows = []
    for subj in sorted(set(s["subj"] for s in bin_ky)):
        idxs = [i for i, s in enumerate(bin_ky) if s["subj"] == subj]
        if len(idxs) >= 2:
            tr_i = idxs[0]
            for te_i in idxs[1:]:
                ky_cs_rows.append(eval_single_fold([bin_ky[tr_i]], bin_ky[te_i], ky18.project_tangent, [rec_ky_bin[tr_i]]))
    arrs = summarize_protocol("kaya_bin_cs", ky_cs_rows)
    for m, v in arrs.items():
        saved_results[f"kaya_bin_cs__{m}"] = v

    tri_idx = [i for i, s in enumerate(ky_sess) if np.all(np.isin([1, 2, 3], s["y"]))]
    tri_ky = [ky18.filter_session(ky_sess[i], (1, 2, 3)) for i in tri_idx]
    rec_ky_tri = [
        (ky18.project_tangent(s["cov_f"], ky18.sym_expm(s["log_f"].mean(0))),
         ky18.project_tangent(s["cov_h"], ky18.sym_expm(s["log_h"].mean(0))))
        for s in tri_ky
    ]
    ky_3_loso = [
        eval_single_fold(
            [tri_ky[j] for j in range(len(tri_ky)) if tri_ky[j]["subj"] != tri_ky[i]["subj"]],
            tri_ky[i],
            ky18.project_tangent,
            [rec_ky_tri[j] for j in range(len(tri_ky)) if tri_ky[j]["subj"] != tri_ky[i]["subj"]],
        )
        for i in range(len(tri_ky))
    ]
    arrs = summarize_protocol("kaya_3cls_loso", ky_3_loso)
    for m, v in arrs.items():
        saved_results[f"kaya_3cls_loso__{m}"] = v

    # 4. EEGBCI (N=109)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        eeg_res = list(ex.map(eeg_ab.load_subject_worker, list(range(1, 110))))
    eeg_subs = []
    for r in eeg_res:
        if len(r) != 2:
            _, cf, ch, lf, lh, y_sub, _ = r
            eeg_subs.append({"cov_f": cf, "cov_h": ch, "log_f": lf, "log_h": lh, "y": y_sub})
    rec_eeg = [
        (eeg_ab.project_tangent(s["cov_f"], eeg_ab.sym_expm(s["log_f"].mean(0))),
         eeg_ab.project_tangent(s["cov_h"], eeg_ab.sym_expm(s["log_h"].mean(0))))
        for s in eeg_subs
    ]
    eeg_rows = [
        eval_single_fold(
            [eeg_subs[j] for j in range(len(eeg_subs)) if j != i],
            eeg_subs[i],
            eeg_ab.project_tangent,
            [rec_eeg[j] for j in range(len(eeg_subs)) if j != i],
        )
        for i in range(len(eeg_subs))
    ]
    arrs = summarize_protocol("eegbci_bin_loso", eeg_rows)
    for m, v in arrs.items():
        saved_results[f"eegbci_bin_loso__{m}"] = v

    np.savez(OUTPUT_NPZ, **saved_results)
    print(f"\nSaved unified 4-dataset results to: {OUTPUT_NPZ}")
    print(f"Total Unified 4-Dataset Runtime: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
