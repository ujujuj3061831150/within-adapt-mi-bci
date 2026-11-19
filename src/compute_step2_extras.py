"""
Compute remaining unified uV + causal sosfilt extras:
  1. Global 109-subject EEGBCI Haufe forward activation patterns (pattern_mat_full, pattern_mat_hb)
  2. Unified uV EEGBCI Block-6 Bursty stream + Chang2025 random-shuffled stream
  3. Exact parameter count breakdown for Baseline vs Within-Adapt (13-ch config)
Saves to results/step2_extras.npz.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import chang2025_causal_coral_repro as ch25
import eegbci_reviewer_ablations as eeg_ab
import probe_step1_audit_and_causal as ps
import run_step2_riemannian_master as step2
import unified_within_adapt_coral_benchmark as uni

OUT_NPZ = config.RESULTS_DIR / "step2_extras.npz"


def unvectorize_tangent(vec: np.ndarray, n_ch: int) -> np.ndarray:
    mat = np.zeros((n_ch, n_ch), dtype=np.float64)
    rows, cols = np.triu_indices(n_ch)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for idx, (r, c) in enumerate(zip(rows, cols)):
        val = vec[idx] if r == c else vec[idx] * inv_sqrt2
        mat[r, c] = val
        mat[c, r] = val
    return mat


def _burst_eeg_worker(idx: int) -> dict[str, float]:
    eeg_subs = step2._G_STATE["blocks"]
    tr = [eeg_subs[j] for j in range(len(eeg_subs)) if j != idx]
    te = ps.reorder_bursty_stream(eeg_subs[idx], burst_len=6, seed=42 + idx)
    return step2.eval_fold_main(tr, te, return_probs=False)


def _rand_chang_worker(idx: int) -> dict[str, float]:
    blocks = step2._G_STATE["blocks"]
    tr = [blocks[j] for j in range(len(blocks)) if j != idx]
    te = ps.reorder_random_stream(blocks[idx], seed=100 + idx)
    return step2.eval_fold_main(tr, te, return_probs=False)


def main() -> None:
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as ex:
        eeg_subs = list(ex.map(step2.worker_eegbci, range(1, 110)))
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor() as ex:
        ch_subs = list(ex.map(step2.worker_chang, ch_dirs))
    print(f"Loaded causal uV EEGBCI & Chang in {time.perf_counter()-t0:.1f}s")

    store = {}

    # 1. Global 109-subject Haufe patterns under unified uV + causal sosfilt
    rec_eeg = [
        (
            eeg_ab.project_tangent(s["cov_f"], eeg_ab.sym_expm(s["log_f"].mean(0))),
            eeg_ab.project_tangent(s["cov_h"], eeg_ab.sym_expm(s["log_h"].mean(0))),
        )
        for s in eeg_subs
    ]
    y_all = np.concatenate([s["y"] for s in eeg_subs])
    pca_f = PCA(n_components=30, random_state=42).fit(np.concatenate([r[0] for r in rec_eeg]))
    pca_h = PCA(n_components=30, random_state=42).fit(np.concatenate([r[1] for r in rec_eeg]))
    zf_l = [pca_f.transform(r[0]) for r in rec_eeg]
    zh_l = [pca_h.transform(r[1]) for r in rec_eeg]
    wf_inv, sf_cov_v, sf_sqrt_v = uni.fit_within_operator(np.concatenate(zf_l), y_all)
    wh_inv, sh_cov_v, sh_sqrt_v = uni.fit_within_operator(np.concatenate(zh_l), y_all)

    X_blocks = [
        np.concatenate(
            [
                step2.within_adapt_s(zf @ wf_inv, zf @ wf_inv, sf_cov_v, sf_sqrt_v, 1.0),
                step2.within_adapt_s(zh @ wh_inv, zh @ wh_inv, sh_cov_v, sh_sqrt_v, 1.0),
            ],
            axis=1,
        )
        for zf, zh in zip(zf_l, zh_l)
    ]
    X_all = np.concatenate(X_blocks, axis=0)
    clf_all = LogisticRegression(solver="newton-cg", max_iter=5000, C=0.05, random_state=42).fit(X_all, y_all)
    w_coef = clf_all.coef_[0]  # (60,) for T2 (right hand) vs T1 (left hand)
    cov_X = np.cov(X_all.T)
    pat_60 = cov_X @ w_coef
    map_f = pca_f.components_.T @ wf_inv @ sf_sqrt_v  # (91, 30)
    map_h = pca_h.components_.T @ wh_inv @ sh_sqrt_v  # (91, 30)
    store["pattern_mat_full"] = unvectorize_tangent(map_f @ pat_60[:30], 13)
    store["pattern_mat_hb"] = unvectorize_tangent(map_h @ pat_60[30:], 13)

    # Exact parameter breakdown (13-ch dual-band):
    # Uncal (K=0): 2 * (13*13 [M_src] + 91 [pca_mean] + 30*91 [W_pca]) + 2*60 [StandardScaler] + 61 [LR]
    # = 2*(169 + 91 + 2730) + 120 + 61 = 5980 + 181 = 6161? Wait: let's check exact stored scalars!
    # In Within-Adapt: per band: 91 [pca_mean] + 30*91 [W_pca] + 30*30 [Sigma_W^{-1/2}] + 30*30 [Sigma_v,src] + 30*30 [Sigma_v,src^{1/2}]
    # = 91 + 2730 + 900 + 900 + 900 = 5521 per band -> 11,042 + 61 LR = 11,103! (Wait: where did earlier 5,823 & 7,503 come from? Let's print exact formula & check!)
    n_pca = 2 * (91 + 30 * 91)  # 5642
    n_k0_m = 2 * (13 * 13)  # 338
    n_k0_sc = 2 * 60  # 120
    n_lr = 60 + 1  # 61
    # Note: if storing Sigma_v,src^{1/2} only (since Sigma_v,src = (Sigma_v,src^{1/2})^2) + Sigma_W^{-1/2}, or symmetric upper triangles:
    print(f"Exact 13-ch param check: PCA(mean+comp)={n_pca}, K=0(M_src upper=2*91=182, full=338)+SC={n_k0_sc}+LR={n_lr} -> {n_pca+182+n_k0_sc+n_lr} (or {n_pca+n_k0_sc+n_lr})")
    print(f"Within-Adapt full matrices: PCA={n_pca} + 2*3*(30x30)={2*3*900} + LR={n_lr} = {n_pca + 5400 + n_lr} (symmetric packed upper: {n_pca + 2*3*465 + n_lr}, minimal Sigma_W^-1/2 + Sigma_v,src^1/2: {n_pca + 2*2*900 + n_lr})")

    # 2. Bursty Block-6 on EEGBCI (uV, causal sosfilt) & Chang random shuffle (with uV+sosfilt)
    with ProcessPoolExecutor(initializer=step2._init_state, initargs=({"blocks": eeg_subs},)) as ex:
        b_rows = list(ex.map(_burst_eeg_worker, range(109)))
    for m in ["Uncal", "G1", "Orig", "z-CORAL (OAS-Src)", "Within-CORAL(0.7)", "Within-Adapt"]:
        arr = np.asarray([r[m] for r in b_rows])
        store[f"eeg_burst6__{m}"] = arr
    print(
        f"[eegbci_burst6 uV+sosfilt] (N=109): "
        + " | ".join(f"{m}: {store[f'eeg_burst6__{m}'].mean()*100:.2f}%" for m in ["Uncal", "G1", "Orig", "z-CORAL (OAS-Src)", "Within-CORAL(0.7)", "Within-Adapt"])
    )

    for tag, cset in [("chang_bin_rand", (1, 2)), ("chang_3cls_rand", (1, 2, 3))]:
        cf = [step2.subset_block(s, np.isin(s["y"], cset)) for s in ch_subs]
        with ProcessPoolExecutor(initializer=step2._init_state, initargs=({"blocks": cf},)) as ex:
            r_rows = list(ex.map(_rand_chang_worker, range(len(cf))))
        for m in ["Uncal", "G1", "Orig", "z-CORAL (Unshrunk-Src)", "z-CORAL (OAS-Src)", "Within-CORAL(0.7)", "Within-Adapt"]:
            store[f"{tag}__{m}"] = np.asarray([r[m] for r in r_rows])
        print(
            f"[{tag} uV+sosfilt] (N=28): "
            + " | ".join(f"{m}: {store[f'{tag}__{m}'].mean()*100:.2f}%" for m in ["Uncal", "G1", "Orig", "z-CORAL (OAS-Src)", "Within-Adapt"])
        )

    np.savez_compressed(OUT_NPZ, **store)
    print(f"Saved extras to {OUT_NPZ} in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
