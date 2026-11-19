"""
Kaya et al. (2018) Large MI Dataset (MNE-kaya2018-data) Cross-Dataset Reproduction
================================================================================
Tests our Dual-Band (Full 8-30 Hz + High-Beta 21-30 Hz) + Source PCA-30 +
Strictly Causal W=8 OAS-CORAL method on the 17 continuous CLA-3St-LRHand sessions
across 7 subjects (Subject A, B, C, D, E, F, J).

Key verified dataset properties (KAYA2018_DIR / data/MNE-kaya2018-data):
  - 17 continuous .mat recordings (~56 min each, fs=200 Hz, ~600-960 trials/session)
  - 19 standard 10-20 EEG channels (Fp1..Pz excluding A1/A2 earlobes and X5 sync)
  - Chronological markers: 1=Left Hand, 2=Right Hand, 3=Passive (SubjectF-150916 has no class 3)
  - Trial window: [0.0, 1.0]s post cue onset (200 samples; minimum ITI across all 17 files is 2.17s)

Protocols evaluated on target session trials[15:]:
  1. Binary (Left vs Right) - 7-Fold Strict Subject-LOSO (17 sessions / 7 subjects)
  2. Binary (Left vs Right) - Cross-Session Within-Subject (Session 1 -> Sessions 2,3 on B,C,E,F,J)
  3. 3-Class (Left/Right/Passive) - 7-Fold Strict Subject-LOSO (16 3-class sessions)
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# Global Configuration
# ==============================================================================
DATA_DIR = config.KAYA2018_DIR
SFREQ = 200.0
EPOCH_SAMPLES = 200  # 1.0 s cue window at 200 Hz (safely < min ITI 2.17s)

# 19 cortical 10-20 EEG channels (exclude A1, A2 earlobe refs and X5 sync)
EEG19_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]

BANDS = [
    ("full", (8.0, 30.0)),
    ("high_beta", (21.0, 30.0)),
]

EVAL_START_IDX = 15
CAUSAL_WINDOW = 8
PCA_DIM = 30
LOGREG_C = 0.05
RANDOM_STATE = 42
SPD_EPS = 1e-10


# ==============================================================================
# Exact Vectorized OAS & SPD Manifold Primitives
# ==============================================================================
def exact_oas(X: np.ndarray, assume_centered: bool = True) -> tuple[np.ndarray, float]:
    """Exact Oracle Approximating Shrinkage matching sklearn.covariance.OAS."""
    n_samples, n_features = X.shape
    X_c = X if assume_centered else (X - X.mean(axis=0, keepdims=True))
    emp_cov = (X_c.T @ X_c) / n_samples
    emp_cov = 0.5 * (emp_cov + emp_cov.T)

    alpha = float(np.mean(emp_cov ** 2))
    mu = float(np.trace(emp_cov)) / n_features
    mu_sq = mu ** 2
    num = alpha + mu_sq
    den = (n_samples + 1) * (alpha - mu_sq / n_features)
    shrinkage = 1.0 if den == 0.0 else min(num / den, 1.0)

    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov, float(shrinkage)


def compute_trial_oas_covariances(X: np.ndarray) -> np.ndarray:
    """Compute regularized OAS spatial covariance matrices (n_trials, 19, 19)."""
    n_trials, n_ch, _ = X.shape
    covs = np.empty((n_trials, n_ch, n_ch), dtype=np.float64)
    for i in range(n_trials):
        cov, _ = exact_oas(X[i].T, assume_centered=False)
        covs[i] = cov
    return covs


def sym_logm(mats: np.ndarray) -> np.ndarray:
    """Batched symmetric matrix logarithm via eigendecomposition."""
    sym = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, SPD_EPS)
    log_eigs = np.log(eigvals)
    return np.einsum("...ij,...j,...kj->...ik", eigvecs, log_eigs, eigvecs)


def sym_expm(mat: np.ndarray) -> np.ndarray:
    """Symmetric matrix exponential via eigendecomposition."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    exp_eigs = np.exp(eigvals)
    return (eigvecs * exp_eigs) @ eigvecs.T


def sym_inv_sqrtm(mat: np.ndarray) -> np.ndarray:
    """Symmetric matrix inverse square root M^{-1/2} via eigendecomposition."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, SPD_EPS)
    inv_sqrt_eigs = 1.0 / np.sqrt(eigvals)
    return (eigvecs * inv_sqrt_eigs) @ eigvecs.T


def vectorize_tangent(tangent_mats: np.ndarray) -> np.ndarray:
    """Extract upper triangle (N, 19, 19) -> (N, 190) with sqrt(2) off-diagonal weights."""
    n_ch = tangent_mats.shape[-1]
    row_idx, col_idx = np.triu_indices(n_ch)
    weights = np.where(row_idx == col_idx, 1.0, np.sqrt(2.0))
    return tangent_mats[:, row_idx, col_idx] * weights[None, :]


def project_tangent(covs: np.ndarray, ref_mat: np.ndarray) -> np.ndarray:
    """Map SPD matrices to tangent space at ref_mat."""
    ref_inv_sqrt = sym_inv_sqrtm(ref_mat)
    return vectorize_tangent(sym_logm(ref_inv_sqrt @ covs @ ref_inv_sqrt))


def coral_fit_transform(Z_fit: np.ndarray, Z_apply: np.ndarray) -> np.ndarray:
    """Full 2nd-order OAS-CORAL covariance whitening."""
    mean_sub = Z_fit.mean(axis=0, keepdims=True)
    cov_sub, _ = exact_oas(Z_fit - mean_sub, assume_centered=True)
    inv_sqrt = sym_inv_sqrtm(cov_sub)
    return (Z_apply - mean_sub) @ inv_sqrt


# ==============================================================================
# Session Loader Worker (Parallelizable)
# ==============================================================================
def load_session_worker(mat_path: Path) -> dict:
    """Load one continuous Kaya2018 CLA session -> CAR 19-ch OAS covs & chronological labels."""
    o = sio.loadmat(mat_path, simplify_cells=True)["o"]
    chnames = list(o["chnames"])
    pick_idx = [chnames.index(ch) for ch in EEG19_CHANNELS]

    data = np.asarray(o["data"], dtype=np.float64)[:, pick_idx].T  # (19, T)
    data = data - data.mean(axis=0, keepdims=True)  # 19-channel CAR

    mk = np.asarray(o["marker"], dtype=np.int64)
    # Extract chronological onset indices where marker enters 1, 2, or 3
    onsets: list[int] = []
    labels: list[int] = []
    n_pts = data.shape[1]
    prev = 0
    for idx, val in enumerate(mk):
        if val in (1, 2, 3) and prev != val:
            if idx + EPOCH_SAMPLES <= n_pts:
                onsets.append(idx)
                labels.append(int(val))
        prev = int(val)

    onsets_arr = np.asarray(onsets, dtype=np.int64)
    y_all = np.asarray(labels, dtype=np.int64)

    band_covs: dict[str, np.ndarray] = {}
    band_logs: dict[str, np.ndarray] = {}
    for band_name, (lo, hi) in BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=SFREQ, output="sos")
        filt = sosfiltfilt(sos, data, axis=-1)
        epochs = np.stack([filt[:, s : s + EPOCH_SAMPLES] for s in onsets_arr], axis=0)
        covs = compute_trial_oas_covariances(epochs)
        band_covs[band_name] = covs
        band_logs[band_name] = sym_logm(covs)

    m_subj = re.search(r"Subject([A-Za-z]+)", mat_path.name)
    subj_id = m_subj.group(1) if m_subj else mat_path.stem
    return {
        "name": mat_path.stem,
        "subj": subj_id,
        "cov_f": band_covs["full"],
        "cov_h": band_covs["high_beta"],
        "log_f": band_logs["full"],
        "log_h": band_logs["high_beta"],
        "y": y_all,
    }


def filter_session(sess: dict, allowed_classes: tuple[int, ...]) -> dict:
    mask = np.isin(sess["y"], allowed_classes)
    return {
        "name": sess["name"],
        "subj": sess["subj"],
        "cov_f": sess["cov_f"][mask],
        "cov_h": sess["cov_h"][mask],
        "log_f": sess["log_f"][mask],
        "log_h": sess["log_h"][mask],
        "y": sess["y"][mask],
    }


# ==============================================================================
# Core Evaluation Routine
# ==============================================================================
def evaluate_fold(train_blocks: list[dict], test_sess: dict) -> dict[str, float]:
    """Evaluate Baseline (K=0), G1 (1st-order W=8), B1 (Full W=8 CORAL), Ours (Dual W=8 CORAL)."""
    y_tr = np.concatenate([b["y"] for b in train_blocks], axis=0)
    y_te_eval = test_sess["y"][EVAL_START_IDX:]

    cov_te_f_eval = test_sess["cov_f"][EVAL_START_IDX:]
    cov_te_h_eval = test_sess["cov_h"][EVAL_START_IDX:]

    # 1. Baseline (K=0)
    cov_tr_f_all = np.concatenate([b["cov_f"] for b in train_blocks], axis=0)
    cov_tr_h_all = np.concatenate([b["cov_h"] for b in train_blocks], axis=0)
    log_tr_f_all = np.concatenate([b["log_f"] for b in train_blocks], axis=0)
    log_tr_h_all = np.concatenate([b["log_h"] for b in train_blocks], axis=0)

    m_tr_f = sym_expm(log_tr_f_all.mean(axis=0))
    m_tr_h = sym_expm(log_tr_h_all.mean(axis=0))

    v_tr_k0_f = project_tangent(cov_tr_f_all, m_tr_f)
    v_tr_k0_h = project_tangent(cov_tr_h_all, m_tr_h)
    pca_k0_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_tr_k0_f)
    pca_k0_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_tr_k0_h)

    X_tr_k0 = np.concatenate([pca_k0_f.transform(v_tr_k0_f), pca_k0_h.transform(v_tr_k0_h)], axis=1)
    X_te_k0 = np.concatenate(
        [
            pca_k0_f.transform(project_tangent(cov_te_f_eval, m_tr_f)),
            pca_k0_h.transform(project_tangent(cov_te_h_eval, m_tr_h)),
        ],
        axis=1,
    )
    clf_k0 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE)),
    ]).fit(X_tr_k0, y_tr)
    acc_k0 = float(np.mean(clf_k0.predict(X_te_k0) == y_te_eval))

    # 2. Per-session recentering + PCA(30)
    rec_blocks_f: list[np.ndarray] = []
    rec_blocks_h: list[np.ndarray] = []
    for blk in train_blocks:
        m_f = sym_expm(blk["log_f"].mean(axis=0))
        m_h = sym_expm(blk["log_h"].mean(axis=0))
        rec_blocks_f.append(project_tangent(blk["cov_f"], m_f))
        rec_blocks_h.append(project_tangent(blk["cov_h"], m_h))

    pca_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate(rec_blocks_f, axis=0))
    pca_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(np.concatenate(rec_blocks_h, axis=0))

    tr_g1_blocks, tr_b1_blocks, tr_ours_blocks = [], [], []
    for rf, rh in zip(rec_blocks_f, rec_blocks_h):
        zf = pca_f.transform(rf)
        zh = pca_h.transform(rh)
        zf_c = coral_fit_transform(zf, zf)
        zh_c = coral_fit_transform(zh, zh)
        tr_g1_blocks.append(np.concatenate([zf, zh], axis=1))
        tr_b1_blocks.append(zf_c)
        tr_ours_blocks.append(np.concatenate([zf_c, zh_c], axis=1))

    clf_g1 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE)),
    ]).fit(np.concatenate(tr_g1_blocks, axis=0), y_tr)
    clf_b1 = LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE).fit(
        np.concatenate(tr_b1_blocks, axis=0), y_tr
    )
    clf_ours = LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE).fit(
        np.concatenate(tr_ours_blocks, axis=0), y_tr
    )

    # 3. Strictly causal W=8 on test session trials[15:]
    n_te = len(test_sess["y"])
    n_eval = n_te - EVAL_START_IDX
    X_te_g1 = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)
    X_te_b1 = np.empty((n_eval, PCA_DIM), dtype=np.float64)
    X_te_ours = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)

    cov_te_f, cov_te_h = test_sess["cov_f"], test_sess["cov_h"]
    log_te_f, log_te_h = test_sess["log_f"], test_sess["log_h"]

    for out_idx, t in enumerate(range(EVAL_START_IDX, n_te)):
        start = max(0, t - CAUSAL_WINDOW)
        m_f = sym_expm(log_te_f[start:t].mean(axis=0))
        m_h = sym_expm(log_te_h[start:t].mean(axis=0))
        zf = pca_f.transform(project_tangent(cov_te_f[start : t + 1], m_f))
        zh = pca_h.transform(project_tangent(cov_te_h[start : t + 1], m_h))

        X_te_g1[out_idx, :PCA_DIM] = zf[-1]
        X_te_g1[out_idx, PCA_DIM:] = zh[-1]

        zf_c = coral_fit_transform(zf[:-1], zf[-1:])[0]
        zh_c = coral_fit_transform(zh[:-1], zh[-1:])[0]
        X_te_b1[out_idx] = zf_c
        X_te_ours[out_idx, :PCA_DIM] = zf_c
        X_te_ours[out_idx, PCA_DIM:] = zh_c

    return {
        "Baseline (K=0)": acc_k0,
        "G1 (1st-order W=8)": float(np.mean(clf_g1.predict(X_te_g1) == y_te_eval)),
        "B1 (Full W=8 CORAL)": float(np.mean(clf_b1.predict(X_te_b1) == y_te_eval)),
        "Ours (Dual W=8 CORAL)": float(np.mean(clf_ours.predict(X_te_ours) == y_te_eval)),
    }


def paired_wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if np.all(np.abs(diff) < 1e-12):
        return 1.0
    _, p = wilcoxon(a, b)
    return float(p)


def print_report(title: str, labels: list[str], subjs: list[str], rows: list[dict[str, float]]) -> None:
    models = ["Baseline (K=0)", "G1 (1st-order W=8)", "B1 (Full W=8 CORAL)", "Ours (Dual W=8 CORAL)"]
    arrs = {m: np.asarray([r[m] for r in rows], dtype=np.float64) for m in models}
    ours = arrs["Ours (Dual W=8 CORAL)"]

    print("\n" + "=" * 118)
    print(f"{title} (Sessions={len(rows)})")
    print("=" * 118)
    print(f"{'Session':<34} | " + " | ".join(f"{m:>18}" for m in models))
    print("-" * 118)
    for lbl, r in zip(labels, rows):
        vals = " | ".join(f"{r[m]*100:17.2f}%" for m in models)
        print(f"{lbl:<34} | {vals}")
    print("-" * 118)
    mean_row = " | ".join(f"{arrs[m].mean()*100:5.2f}% ± {arrs[m].std(ddof=1)*100:5.2f}%  " for m in models)
    print(f"{'Session Mean ± SD':<34} | {mean_row}")

    # Subject-aggregated mean (important since subjects have 1..3 sessions)
    unique_subjs = sorted(set(subjs))
    if len(unique_subjs) < len(subjs):
        subj_arrs = {
            m: np.asarray([np.mean([arrs[m][i] for i, s in enumerate(subjs) if s == us]) for us in unique_subjs])
            for m in models
        }
        smean_row = " | ".join(
            f"{subj_arrs[m].mean()*100:5.2f}% ± {subj_arrs[m].std(ddof=1)*100:5.2f}%  " for m in models
        )
        print(f"{f'Subject Mean (N={len(unique_subjs)})':<34} | {smean_row}")

    print("-" * 118)
    for comp in ["Baseline (K=0)", "G1 (1st-order W=8)", "B1 (Full W=8 CORAL)"]:
        delta = (ours.mean() - arrs[comp].mean()) * 100.0
        p = paired_wilcoxon_p(ours, arrs[comp])
        w = int(np.sum(ours > arrs[comp]))
        t = int(np.sum(np.abs(ours - arrs[comp]) < 1e-12))
        l = len(ours) - w - t
        print(f"  Session-level Ours vs {comp:<20}: {delta:+6.2f}% | Wilcoxon p = {p:.4f} | W/T/L = {w}/{t}/{l}")


def main() -> None:
    t0 = time.perf_counter()
    mat_files = sorted(DATA_DIR.glob("*.mat"))
    print("=" * 118)
    print(f"Kaya et al. (2018) Cross-Dataset Reproduction | Dir: {DATA_DIR} ({len(mat_files)} files)")
    print("Config: 19 EEG chans (exclude A1, A2, X5), CAR, Cue+[0.0, 1.0]s (200 Hz), OAS (19x19 -> 190d), PCA-30, W=8")
    print("=" * 118)

    t_load = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 4)) as executor:
        sessions = list(executor.map(load_session_worker, mat_files))
    print(f"Loaded {len(sessions)} sessions in {time.perf_counter() - t_load:.2f} s.")

    # 1. Binary (Left=1 vs Right=2) - Strict 7-Fold Subject LOSO
    bin_sess = [filter_session(s, (1, 2)) for s in sessions]
    bin_loso_rows: list[dict[str, float]] = []
    for te_sess in bin_sess:
        tr_blocks = [s for s in bin_sess if s["subj"] != te_sess["subj"]]
        bin_loso_rows.append(evaluate_fold(tr_blocks, te_sess))
    print_report(
        "1. BINARY (Left vs Right) - STRICT 7-FOLD SUBJECT-LOSO (Held-out Subject Session[15:])",
        [s["name"].replace("-3St-LRHand", "") for s in bin_sess],
        [s["subj"] for s in bin_sess],
        bin_loso_rows,
    )

    # 2. Binary (Left=1 vs Right=2) - Within-Subject Cross-Session (Session 1 -> Sessions 2,3)
    cs_labels, cs_subjs, cs_rows = [], [], []
    for subj in sorted(set(s["subj"] for s in bin_sess)):
        s_list = [s for s in bin_sess if s["subj"] == subj]
        if len(s_list) < 2:
            continue
        tr_sess = s_list[0]
        for te_sess in s_list[1:]:
            cs_labels.append(f"{subj}: S1 -> {te_sess['name'].split('-')[2]}")
            cs_subjs.append(subj)
            cs_rows.append(evaluate_fold([tr_sess], te_sess))
    print_report(
        "2. BINARY (Left vs Right) - WITHIN-SUBJECT CROSS-SESSION (Session 1 -> Sessions 2,3[15:])",
        cs_labels,
        cs_subjs,
        cs_rows,
    )

    # 3. 3-Class (1=Left, 2=Right, 3=Passive) - Strict Subject LOSO on 16 3-class sessions
    tri_sess = [filter_session(s, (1, 2, 3)) for s in sessions if np.all(np.isin([1, 2, 3], s["y"]))]
    tri_loso_rows: list[dict[str, float]] = []
    for te_sess in tri_sess:
        tr_blocks = [s for s in tri_sess if s["subj"] != te_sess["subj"]]
        tri_loso_rows.append(evaluate_fold(tr_blocks, te_sess))
    print_report(
        "3. 3-CLASS (Left/Right/Passive) - STRICT SUBJECT-LOSO (16 Sessions, Held-out[15:])",
        [s["name"].replace("-3St-LRHand", "") for s in tri_sess],
        [s["subj"] for s in tri_sess],
        tri_loso_rows,
    )

    print("\n" + "=" * 118)
    print(f"Total Runtime (Load + All 3 Protocols): {time.perf_counter() - t0:.2f} s")
    print("=" * 118)


if __name__ == "__main__":
    main()
