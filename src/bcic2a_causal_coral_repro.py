"""
BCI Competition IV-2a (BNCI2014_001) Cross-Dataset Reproduction
================================================================
Tests whether our EEGBCI core finding:
    Dual-Band (Full 8-30 Hz + High-Beta 21-30 Hz)
    + Source-Domain PCA-30 Subspace
    + Strictly Causal W=8 OAS-CORAL
generalizes to the 9-subject, 22-channel, 250 Hz BCI Competition IV-2a dataset
(BCIC2A_DIR / data/MNE-bnci-data/~bci/database/001-2014).

Evaluates 4 protocols on Session E trials[15:]:
  1. Binary (Left vs Right Hand) Cross-Session (Session T -> Session E[15:])
  2. Binary (Left vs Right Hand) 9-fold LOSO (8 subj T+E -> held-out Session E[15:])
  3. 4-Class (Left/Right/Feet/Tongue) Cross-Session (Session T -> Session E[15:])
  4. 4-Class (Left/Right/Feet/Tongue) 9-fold LOSO (8 subj T+E -> held-out Session E[15:])

Models compared under every protocol (all use PCA-30 + LogReg C=0.05):
  - Baseline (K=0): Global training Riemannian mean M_train + StandardScaler
  - G1 (1st-order W=8): Causal W=8 1st-order recentering + StandardScaler
  - B1 (Single-band Full + W=8 CORAL): Full 8-30 Hz only (30d, no post-scaler)
  - Ours (Dual-band Full+HighBeta + W=8 CORAL): 60d, no post-scaler
"""

from __future__ import annotations

import os
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
DATA_DIR = config.BCIC2A_DIR
N_SUBJECTS = 9
SFREQ = 250.0
N_EEG_CHANS = 22  # Channels 0..21 are EEG; 22..24 are 3 EOG channels (strictly dropped)

# In BNCI2014_001 .mat files, `trial` marks fixation start (t=0s); cue onset is at t=2.0s.
CUE_OFFSET_SEC = 2.0
POST_CUE_TMIN = 0.5
POST_CUE_TMAX = 3.5

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
    """Compute regularized OAS spatial covariance matrices (n_trials, 22, 22)."""
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
    """Extract upper triangle (N, 22, 22) -> (N, 253) with sqrt(2) off-diagonal weights."""
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
# Subject Loader Worker (22 EEG Channels, CAR, 0.5~3.5s Post-Cue)
# ==============================================================================
def load_session_arrays(mat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one BNCI2014_001 .mat file -> (covs_full, covs_hb, log_full, log_hb, y)."""
    mat = sio.loadmat(mat_path, simplify_cells=True)
    sos_filters = {
        name: butter(4, [lo, hi], btype="bandpass", fs=SFREQ, output="sos")
        for name, (lo, hi) in BANDS
    }

    start_offset = int(round((CUE_OFFSET_SEC + POST_CUE_TMIN) * SFREQ))
    stop_offset = int(round((CUE_OFFSET_SEC + POST_CUE_TMAX) * SFREQ)) + 1

    band_epochs: dict[str, list[np.ndarray]] = {name: [] for name in sos_filters}
    y_parts: list[np.ndarray] = []

    for run_dict in mat["data"]:
        trials_idx = np.asarray(run_dict["trial"], dtype=np.int64)
        if trials_idx.size == 0:
            continue

        # Strict EOG removal: take first 22 EEG channels & apply full-22-channel CAR
        X_eeg = np.asarray(run_dict["X"], dtype=np.float64)[:, :N_EEG_CHANS]
        X_car = (X_eeg - X_eeg.mean(axis=1, keepdims=True)).T  # (22, T)

        y_run = np.asarray(run_dict["y"], dtype=np.int64)
        # Convert 1-based MATLAB indices to 0-based Python indices
        idx_0 = trials_idx - 1
        y_parts.append(y_run)

        for band_name, sos in sos_filters.items():
            filtered = sosfiltfilt(sos, X_car, axis=-1)
            epochs = np.stack(
                [filtered[:, tr + start_offset : tr + stop_offset] for tr in idx_0],
                axis=0,
            )
            band_epochs[band_name].append(epochs)

    covs_full = compute_trial_oas_covariances(np.concatenate(band_epochs["full"], axis=0))
    covs_hb = compute_trial_oas_covariances(np.concatenate(band_epochs["high_beta"], axis=0))
    log_full = sym_logm(covs_full)
    log_hb = sym_logm(covs_hb)
    y_all = np.concatenate(y_parts, axis=0)
    return covs_full, covs_hb, log_full, log_hb, y_all


def load_subject_worker(subject_id: int) -> dict:
    """Load both Session T and Session E for one subject (A01..A09)."""
    path_t = DATA_DIR / f"A{subject_id:02d}T.mat"
    path_e = DATA_DIR / f"A{subject_id:02d}E.mat"
    cf_t, ch_t, lf_t, lh_t, y_t = load_session_arrays(path_t)
    cf_e, ch_e, lf_e, lh_e, y_e = load_session_arrays(path_e)
    return {
        "sid": subject_id,
        "T": {"cov_f": cf_t, "cov_h": ch_t, "log_f": lf_t, "log_h": lh_t, "y": y_t},
        "E": {"cov_f": cf_e, "cov_h": ch_e, "log_f": lf_e, "log_h": lh_e, "y": y_e},
    }


def filter_session(sess: dict[str, np.ndarray], allowed_classes: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Chronologically filter session trials to allowed classes."""
    mask = np.isin(sess["y"], allowed_classes)
    return {k: v[mask] for k, v in sess.items()}


# ==============================================================================
# Core Evaluation Routine (4 Models on Given Train Blocks -> Test Session[15:])
# ==============================================================================
def evaluate_fold(
    train_blocks: list[dict[str, np.ndarray]],
    test_sess: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Evaluate Baseline (K=0), G1 (1st-order W=8), B1 (Full W=8 CORAL), and
    Ours (Dual-band W=8 CORAL) on test_sess['...'][15:].
    """
    y_tr = np.concatenate([b["y"] for b in train_blocks], axis=0)
    y_te_eval = test_sess["y"][EVAL_START_IDX:]

    cov_te_f_eval = test_sess["cov_f"][EVAL_START_IDX:]
    cov_te_h_eval = test_sess["cov_h"][EVAL_START_IDX:]

    # --------------------------------------------------------------------------
    # 1. Baseline (K=0): Global Training Mean + PCA(30) + StandardScaler
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # 2. Training-domain per-block 1st-order recentering + shared PCA(30)
    # --------------------------------------------------------------------------
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
        zf_coral = coral_fit_transform(zf, zf)
        zh_coral = coral_fit_transform(zh, zh)

        tr_g1_blocks.append(np.concatenate([zf, zh], axis=1))
        tr_b1_blocks.append(zf_coral)
        tr_ours_blocks.append(np.concatenate([zf_coral, zh_coral], axis=1))

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

    # --------------------------------------------------------------------------
    # 3. Strictly causal W=8 test-time alignment on Session E trials[15:]
    # --------------------------------------------------------------------------
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


def run_protocol(
    subjects_data: list[dict],
    allowed_classes: tuple[int, ...],
    mode: str,
) -> dict[str, np.ndarray]:
    """Run Cross-Session or 9-fold LOSO across all 9 subjects."""
    model_names = [
        "Baseline (K=0)",
        "G1 (1st-order W=8)",
        "B1 (Full W=8 CORAL)",
        "Ours (Dual W=8 CORAL)",
    ]
    scores: dict[str, list[float]] = {m: [] for m in model_names}

    filtered = [
        {
            "sid": s["sid"],
            "T": filter_session(s["T"], allowed_classes),
            "E": filter_session(s["E"], allowed_classes),
        }
        for s in subjects_data
    ]

    for te_idx in range(len(filtered)):
        test_sess = filtered[te_idx]["E"]
        if mode == "cross_session":
            train_blocks = [filtered[te_idx]["T"]]
        elif mode == "loso":
            train_blocks = []
            for tr_idx, sub in enumerate(filtered):
                if tr_idx != te_idx:
                    train_blocks.append(sub["T"])
                    train_blocks.append(sub["E"])
        else:
            raise ValueError(mode)

        res = evaluate_fold(train_blocks, test_sess)
        for m in model_names:
            scores[m].append(res[m])

    return {m: np.asarray(scores[m], dtype=np.float64) for m in model_names}


# ==============================================================================
# Reporting
# ==============================================================================
def paired_wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if np.all(np.abs(diff) < 1e-12):
        return 1.0
    _, p = wilcoxon(a, b)
    return float(p)


def print_protocol_report(title: str, results: dict[str, np.ndarray], chance_pct: float) -> None:
    models = list(results.keys())
    ours = results["Ours (Dual W=8 CORAL)"]

    print("\n" + "=" * 110)
    print(f"{title} (Chance = {chance_pct:.1f}%, Eval = Session E trials[15:], N={len(ours)})")
    print("=" * 110)
    header = "Subj  | " + " | ".join(f"{m:>20}" for m in models)
    print(header)
    print("-" * 110)
    for i in range(len(ours)):
        row_vals = " | ".join(f"{results[m][i] * 100:19.2f}%" for m in models)
        print(f"A{i + 1:02d}   | {row_vals}")
    print("-" * 110)
    mean_row = " | ".join(
        f"{results[m].mean() * 100:6.2f}% ± {results[m].std(ddof=1) * 100:5.2f}%     " for m in models
    )
    print(f"Mean  | {mean_row}")
    print("-" * 110)
    for comp in ["Baseline (K=0)", "G1 (1st-order W=8)", "B1 (Full W=8 CORAL)"]:
        delta = (ours.mean() - results[comp].mean()) * 100.0
        p = paired_wilcoxon_p(ours, results[comp])
        wins = int(np.sum(ours > results[comp]))
        ties = int(np.sum(np.abs(ours - results[comp]) < 1e-12))
        losses = len(ours) - wins - ties
        print(
            f"  Ours vs {comp:<20}: {delta:+6.2f}% | Wilcoxon p = {p:.4f} | W/T/L = {wins}/{ties}/{losses}"
        )


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 110)
    print(f"BCI Competition IV-2a Reproduction | Dir: {DATA_DIR}")
    print("Config: 22 EEG chans (3 EOG removed), CAR, Cue+[0.5, 3.5]s (250 Hz), OAS (22x22 -> 253d), PCA-30, W=8")
    print("=" * 110)

    t_load = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 4)) as executor:
        subjects_data = list(executor.map(load_subject_worker, list(range(1, N_SUBJECTS + 1))))

    print(
        f"Loaded {len(subjects_data)} subjects in {time.perf_counter() - t_load:.2f} s | "
        f"T trials/subj={len(subjects_data[0]['T']['y'])}, E trials/subj={len(subjects_data[0]['E']['y'])}"
    )

    # 1. Binary (Left=1 vs Right=2)
    bin_cs = run_protocol(subjects_data, (1, 2), "cross_session")
    bin_loso = run_protocol(subjects_data, (1, 2), "loso")

    # 2. 4-Class (Left=1, Right=2, Feet=3, Tongue=4)
    quad_cs = run_protocol(subjects_data, (1, 2, 3, 4), "cross_session")
    quad_loso = run_protocol(subjects_data, (1, 2, 3, 4), "loso")

    print_protocol_report("1. BINARY (Left vs Right) - CROSS-SESSION (Session T -> Session E[15:])", bin_cs, 50.0)
    print_protocol_report("2. BINARY (Left vs Right) - 9-FOLD LOSO (8 Subj T+E -> Held-out E[15:])", bin_loso, 50.0)
    print_protocol_report("3. 4-CLASS (L/R/F/T) - CROSS-SESSION (Session T -> Session E[15:])", quad_cs, 25.0)
    print_protocol_report("4. 4-CLASS (L/R/F/T) - 9-FOLD LOSO (8 Subj T+E -> Held-out E[15:])", quad_loso, 25.0)

    print("\n" + "=" * 110)
    print(f"Total Runtime (Load + All 4 Protocols): {time.perf_counter() - t0:.2f} s")
    print("=" * 110)


if __name__ == "__main__":
    main()
