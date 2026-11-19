"""
EEGBCI Full-Dataset (N=109) Reviewer-Grade Three-in-One Ablation Audit
=====================================================================
Runs three targeted, leak-free ablations on N=109 subjects (trials[15:]):
  1. Mechanism Ablation (W=8, Full+High-Beta, PCA-30):
       G1: 1st-order M_sub only (+ global StandardScaler)
       G2: 1st-order M_sub + Diagonal (per-dim mean/std) scaling
       G3: 1st-order M_sub + Full OAS-CORAL whitening
  2. Band Necessity Ablation (W=8 + OAS-CORAL):
       B1: Single-band Full (8-30 Hz, 30d)
       B2: Single-band High-Beta (21-30 Hz, 30d)
       B3: Dual-band Full + High-Beta (60d, identical to G3)
  3. Cross-Run Drift Tracking (Transition vs Steady phases of R08 & R12):
       Compares Frozen K=15, Causal W=20, Causal W=8, and Oracle K=All on:
       - Transition: first 7 trials entering R08 and R12
       - Steady: remaining trials in the second half of R08 and R12
  4. Fair 80%-Coverage Rejection Check:
       Compares Baseline (K=0) vs Causal W=8 at 100% and 80% coverage (max P(y|z)).

Dataset:
    PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)
    Path: EEGBCI_DIR / data/MNE-eegbci-data/files/eegmmidb/1.0.0
"""

from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# Global Configuration
# ==============================================================================
DATA_DIR = config.EEGBCI_DIR
N_SUBJECTS = 109
MI_RUNS = [4, 8, 12]
TARGET_SFREQ = 160.0
TMIN, TMAX = 0.5, 3.5

CORE_CHANNELS = [
    "FC3", "FCz", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CPz", "CP4",
]

BANDS = [
    ("full", (8.0, 30.0)),
    ("high_beta", (21.0, 30.0)),
]

EVAL_START_IDX = 15
MIN_TOTAL_TRIALS = 21
TRANSITION_TRIALS_PER_RUN = 7

PCA_DIM = 30
LOGREG_C = 0.05
RANDOM_STATE = 42
SPD_EPS = 1e-10
DIAG_EPS = 1e-6


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
    """Compute regularized OAS spatial covariance matrices (n_trials, 13, 13)."""
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
    """Extract upper triangle (N, 13, 13) -> (N, 91) with sqrt(2) off-diagonal weights."""
    n_ch = tangent_mats.shape[-1]
    row_idx, col_idx = np.triu_indices(n_ch)
    weights = np.where(row_idx == col_idx, 1.0, np.sqrt(2.0))
    return tangent_mats[:, row_idx, col_idx] * weights[None, :]


def project_tangent(covs: np.ndarray, ref_mat: np.ndarray) -> np.ndarray:
    """Map SPD matrices to tangent space at ref_mat."""
    ref_inv_sqrt = sym_inv_sqrtm(ref_mat)
    return vectorize_tangent(sym_logm(ref_inv_sqrt @ covs @ ref_inv_sqrt))


def diag_fit_transform(Z_fit: np.ndarray, Z_apply: np.ndarray) -> np.ndarray:
    """Diagonal (per-dimension zero-mean + std scaling) alignment without off-diagonal rotation."""
    mean_sub = Z_fit.mean(axis=0, keepdims=True)
    std_sub = Z_fit.std(axis=0, ddof=1, keepdims=True) + DIAG_EPS
    return (Z_apply - mean_sub) / std_sub


def coral_fit_transform(Z_fit: np.ndarray, Z_apply: np.ndarray) -> np.ndarray:
    """Full 2nd-order OAS-CORAL covariance whitening."""
    mean_sub = Z_fit.mean(axis=0, keepdims=True)
    cov_sub, _ = exact_oas(Z_fit - mean_sub, assume_centered=True)
    inv_sqrt = sym_inv_sqrtm(cov_sub)
    return (Z_apply - mean_sub) @ inv_sqrt


# ==============================================================================
# Subject Loading Worker (Parallelizable, tracks exact per-run trial indices)
# ==============================================================================
def load_subject_worker(
    subject_id: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[int, str]:
    """Load subject chronologically, precompute OAS covariances, logm(C_i), and Transition mask."""
    mne.set_log_level("WARNING")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sub_str = f"S{subject_id:03d}"

    try:
        sos_filters = {
            name: butter(4, [lo, hi], btype="bandpass", fs=TARGET_SFREQ, output="sos")
            for name, (lo, hi) in BANDS
        }
        band_epochs: dict[str, list[np.ndarray]] = {name: [] for name in sos_filters}
        y_parts: list[np.ndarray] = []
        is_transition_parts: list[np.ndarray] = []

        for run_pos, run in enumerate(MI_RUNS):
            edf_path = DATA_DIR / sub_str / f"{sub_str}R{run:02d}.edf"
            if not edf_path.exists():
                return (subject_id, f"Missing file {edf_path}")

            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1e-3:
                raw.resample(TARGET_SFREQ, verbose=False)

            raw.set_eeg_reference("average", projection=False, verbose=False)
            matched = [ch for ch in CORE_CHANNELS if ch in raw.ch_names]
            if len(matched) != len(CORE_CHANNELS):
                return (subject_id, "Missing core channels")
            raw.pick(matched)

            events, event_id_all = mne.events_from_annotations(raw, verbose=False)
            mi_event_id = {k: v for k, v in event_id_all.items() if k in ("T1", "T2")}
            if len(mi_event_id) < 2:
                return (subject_id, f"Missing T1/T2 events: {event_id_all}")

            data_cont = raw.get_data()
            t1_code = mi_event_id["T1"]
            start_offset = int(round(TMIN * TARGET_SFREQ))
            stop_offset = int(round(TMAX * TARGET_SFREQ)) + 1

            valid_events = events[np.isin(events[:, 2], [mi_event_id["T1"], mi_event_id["T2"]])]
            valid_events = valid_events[
                (valid_events[:, 0] + start_offset >= 0)
                & (valid_events[:, 0] + stop_offset <= data_cont.shape[1])
            ]
            if len(valid_events) == 0:
                return (subject_id, f"No valid trials in R{run:02d}")

            n_run = len(valid_events)
            y_parts.append((valid_events[:, 2] != t1_code).astype(np.int64))
            # Transition = first 7 trials entering subsequent runs (R08, R12)
            within_run_idx = np.arange(n_run)
            trans_mask_run = (run_pos > 0) & (within_run_idx < TRANSITION_TRIALS_PER_RUN)
            is_transition_parts.append(trans_mask_run)

            for band_name, sos in sos_filters.items():
                filtered = sosfiltfilt(sos, data_cont, axis=-1)
                trials = np.stack(
                    [filtered[:, ev[0] + start_offset : ev[0] + stop_offset] for ev in valid_events],
                    axis=0,
                )
                band_epochs[band_name].append(trials)

        y_sub = np.concatenate(y_parts, axis=0)
        trans_mask_sub = np.concatenate(is_transition_parts, axis=0)
        if len(y_sub) < MIN_TOTAL_TRIALS:
            return (subject_id, f"Too few trials ({len(y_sub)} < {MIN_TOTAL_TRIALS})")

        covs_full = compute_trial_oas_covariances(np.concatenate(band_epochs["full"], axis=0))
        covs_hb = compute_trial_oas_covariances(np.concatenate(band_epochs["high_beta"], axis=0))
        log_covs_full = sym_logm(covs_full)
        log_covs_hb = sym_logm(covs_hb)
        return (subject_id, covs_full, covs_hb, log_covs_full, log_covs_hb, y_sub, trans_mask_sub)
    except Exception as exc:
        return (subject_id, str(exc))


def build_causal_w8_eval_features(
    covs_full: np.ndarray,
    covs_hb: np.ndarray,
    log_covs_full: np.ndarray,
    log_covs_hb: np.ndarray,
    pca_full: PCA,
    pca_hb: PCA,
) -> dict[str, np.ndarray]:
    """
    Single pass over trials[15:] with W=8 computing aligned eval features simultaneously for:
      - G1: 1st-order M_sub only (60d)
      - G2: 1st-order M_sub + Diagonal scaling (60d)
      - G3/B3: 1st-order M_sub + Full OAS-CORAL (60d)
      - B1: Full-band CORAL only (first 30d of G3)
      - B2: High-Beta CORAL only (last 30d of G3)
    """
    n_trials = len(covs_full)
    n_eval = n_trials - EVAL_START_IDX
    X_g1 = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)
    X_g2 = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)
    X_g3 = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)

    for idx_out, t in enumerate(range(EVAL_START_IDX, n_trials)):
        start = max(0, t - 8)
        m_full = sym_expm(log_covs_full[start:t].mean(axis=0))
        m_hb = sym_expm(log_covs_hb[start:t].mean(axis=0))

        z_full = pca_full.transform(project_tangent(covs_full[start : t + 1], m_full))
        z_hb = pca_hb.transform(project_tangent(covs_hb[start : t + 1], m_hb))

        # G1: 1st-order M_sub + PCA(30) only
        X_g1[idx_out, :PCA_DIM] = z_full[-1]
        X_g1[idx_out, PCA_DIM:] = z_hb[-1]

        # G2: Diagonal scaling on history [:-1]
        X_g2[idx_out, :PCA_DIM] = diag_fit_transform(z_full[:-1], z_full[-1:])[0]
        X_g2[idx_out, PCA_DIM:] = diag_fit_transform(z_hb[:-1], z_hb[-1:])[0]

        # G3 / B3: Full OAS-CORAL on history [:-1]
        X_g3[idx_out, :PCA_DIM] = coral_fit_transform(z_full[:-1], z_full[-1:])[0]
        X_g3[idx_out, PCA_DIM:] = coral_fit_transform(z_hb[:-1], z_hb[-1:])[0]

    return {
        "G1": X_g1,
        "G2": X_g2,
        "G3": X_g3,
        "B1": X_g3[:, :PCA_DIM],
        "B2": X_g3[:, PCA_DIM:],
    }


def build_causal_w20_eval_features(
    covs_full: np.ndarray,
    covs_hb: np.ndarray,
    log_covs_full: np.ndarray,
    log_covs_hb: np.ndarray,
    pca_full: PCA,
    pca_hb: PCA,
) -> np.ndarray:
    """Causal W=20 + Full OAS-CORAL features on trials[15:]."""
    n_trials = len(covs_full)
    n_eval = n_trials - EVAL_START_IDX
    X_w20 = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)

    for idx_out, t in enumerate(range(EVAL_START_IDX, n_trials)):
        start = max(0, t - 20)
        m_full = sym_expm(log_covs_full[start:t].mean(axis=0))
        m_hb = sym_expm(log_covs_hb[start:t].mean(axis=0))

        z_full = pca_full.transform(project_tangent(covs_full[start : t + 1], m_full))
        z_hb = pca_hb.transform(project_tangent(covs_hb[start : t + 1], m_hb))

        X_w20[idx_out, :PCA_DIM] = coral_fit_transform(z_full[:-1], z_full[-1:])[0]
        X_w20[idx_out, PCA_DIM:] = coral_fit_transform(z_hb[:-1], z_hb[-1:])[0]

    return X_w20


def selective_accuracy(preds: np.ndarray, y_true: np.ndarray, max_probs: np.ndarray, keep_ratio: float = 0.80) -> float:
    """Accuracy on top `keep_ratio` highest-probability trials for one subject."""
    n_keep = max(1, int(round(keep_ratio * len(y_true))))
    keep_idx = np.argsort(max_probs)[-n_keep:]
    return float(np.mean(preds[keep_idx] == y_true[keep_idx]))


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """Paired two-sided Wilcoxon signed-rank test."""
    if np.allclose(x - y, 0.0):
        return 1.0
    _, p_val = wilcoxon(x, y, alternative="two-sided")
    return float(p_val)


def main() -> None:
    t0 = time.perf_counter()
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"EEGBCI dataset directory not found: {DATA_DIR}")

    print(
        f"=== EEGBCI Reviewer-Grade Three-in-One Ablation Audit (N_SUBJECTS={N_SUBJECTS}, "
        f"Eval=trials[{EVAL_START_IDX}:]) ===",
        flush=True,
    )

    n_workers = min(6, os.cpu_count() or 4)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for res in pool.map(load_subject_worker, range(1, N_SUBJECTS + 1)):
            results.append(res)

    covs_full_list: list[np.ndarray] = []
    covs_hb_list: list[np.ndarray] = []
    log_full_list: list[np.ndarray] = []
    log_hb_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    trans_mask_list: list[np.ndarray] = []
    valid_sids: list[int] = []

    for res in results:
        if len(res) == 2:
            sid, msg = res
            print(f"[WARN] Skipping S{sid:03d}: {msg}", flush=True)
        else:
            sid, cf, ch, lcf, lch, y, tmask = res  # type: ignore[misc]
            valid_sids.append(sid)
            covs_full_list.append(cf)
            covs_hb_list.append(ch)
            log_full_list.append(lcf)
            log_hb_list.append(lch)
            y_list.append(y)
            trans_mask_list.append(tmask)

    n_sub = len(valid_sids)
    if n_sub < 2:
        raise RuntimeError("Need at least 2 valid subjects for LOSO evaluation.")

    rec_full_list = [project_tangent(c, sym_expm(lc.mean(axis=0))) for c, lc in zip(covs_full_list, log_full_list)]
    rec_hb_list = [project_tangent(c, sym_expm(lc.mean(axis=0))) for c, lc in zip(covs_hb_list, log_hb_list)]
    sum_log_full = np.array([lc.sum(axis=0) for lc in log_full_list])
    sum_log_hb = np.array([lc.sum(axis=0) for lc in log_hb_list])
    trial_counts = np.array([len(y) for y in y_list])

    t_load = time.perf_counter() - t0
    print(
        f"Loaded {n_sub} subjects (eval trials={sum(len(y) - EVAL_START_IDX for y in y_list)}) "
        f"in {t_load:.2f} s. Running {n_sub}-fold LOSO...",
        flush=True,
    )

    # Storage containers
    acc_k0, acc_k0_80 = [], []
    acc_g1, acc_g2, acc_g3, acc_g3_80 = [], [], [], []
    acc_b1, acc_b2 = [], []
    acc_f15, acc_w20, acc_all = [], [], []

    # Transition vs Steady storage (per-subject arrays, NaN if a subject lacks either phase)
    phase_models = ["Frozen K=15", "Causal W=20", "Causal W=8", "Oracle K=All"]
    trans_accs: dict[str, list[float]] = {m: [] for m in phase_models}
    steady_accs: dict[str, list[float]] = {m: [] for m in phase_models}

    logo = LeaveOneGroupOut()
    dummy = np.zeros(n_sub)
    groups = np.arange(n_sub)

    for train_sids, test_arr in logo.split(dummy, dummy, groups):
        te_idx = int(test_arr[0])
        y_tr = np.concatenate([y_list[i] for i in train_sids], axis=0)
        y_te_eval = y_list[te_idx][EVAL_START_IDX:]
        tmask_eval = trans_mask_list[te_idx][EVAL_START_IDX:]
        smask_eval = ~tmask_eval

        covs_te_full_eval = covs_full_list[te_idx][EVAL_START_IDX:]
        covs_te_hb_eval = covs_hb_list[te_idx][EVAL_START_IDX:]

        # ------------------------------------------------------------------
        # 0. Baseline (K = 0) + 80% coverage check
        # ------------------------------------------------------------------
        n_tr_total = int(trial_counts[train_sids].sum())
        m_tr_full = sym_expm(sum_log_full[train_sids].sum(axis=0) / n_tr_total)
        m_tr_hb = sym_expm(sum_log_hb[train_sids].sum(axis=0) / n_tr_total)

        covs_tr_full = np.concatenate([covs_full_list[i] for i in train_sids], axis=0)
        covs_tr_hb = np.concatenate([covs_hb_list[i] for i in train_sids], axis=0)

        v_tr_k0_full = project_tangent(covs_tr_full, m_tr_full)
        v_tr_k0_hb = project_tangent(covs_tr_hb, m_tr_hb)
        pca_k0_full = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_tr_k0_full)
        pca_k0_hb = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_tr_k0_hb)

        X_tr_k0 = np.concatenate(
            [pca_k0_full.transform(v_tr_k0_full), pca_k0_hb.transform(v_tr_k0_hb)], axis=1
        )
        X_te_k0 = np.concatenate(
            [
                pca_k0_full.transform(project_tangent(covs_te_full_eval, m_tr_full)),
                pca_k0_hb.transform(project_tangent(covs_te_hb_eval, m_tr_hb)),
            ],
            axis=1,
        )
        clf_k0 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE)),
        ])
        clf_k0.fit(X_tr_k0, y_tr)
        probs_k0 = clf_k0.predict_proba(X_te_k0)
        preds_k0 = clf_k0.classes_[np.argmax(probs_k0, axis=1)]
        acc_k0.append(float(np.mean(preds_k0 == y_te_eval)))
        acc_k0_80.append(selective_accuracy(preds_k0, y_te_eval, np.max(probs_k0, axis=1), 0.80))

        # ------------------------------------------------------------------
        # Shared PCA(30) on 1st-order recentered training subjects
        # ------------------------------------------------------------------
        v_tr_rec_full = np.concatenate([rec_full_list[i] for i in train_sids], axis=0)
        v_tr_rec_hb = np.concatenate([rec_hb_list[i] for i in train_sids], axis=0)
        pca_c_full = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_tr_rec_full)
        pca_c_hb = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_tr_rec_hb)

        # Build training matrices for G1 (1st-order), G2 (diag scaling), G3/B3 (CORAL)
        tr_g1_blocks, tr_g2_blocks, tr_g3_blocks = [], [], []
        for s_i in train_sids:
            zf = pca_c_full.transform(rec_full_list[s_i])
            zh = pca_c_hb.transform(rec_hb_list[s_i])
            tr_g1_blocks.append(np.concatenate([zf, zh], axis=1))
            tr_g2_blocks.append(
                np.concatenate([diag_fit_transform(zf, zf), diag_fit_transform(zh, zh)], axis=1)
            )
            tr_g3_blocks.append(
                np.concatenate([coral_fit_transform(zf, zf), coral_fit_transform(zh, zh)], axis=1)
            )

        X_tr_g1 = np.concatenate(tr_g1_blocks, axis=0)
        X_tr_g2 = np.concatenate(tr_g2_blocks, axis=0)
        X_tr_g3 = np.concatenate(tr_g3_blocks, axis=0)

        clf_g1 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE)),
        ]).fit(X_tr_g1, y_tr)
        clf_g2 = LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE).fit(X_tr_g2, y_tr)
        clf_g3 = LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE).fit(X_tr_g3, y_tr)

        # Band ablations B1 (Full only) & B2 (High-Beta only)
        clf_b1 = LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE).fit(
            X_tr_g3[:, :PCA_DIM], y_tr
        )
        clf_b2 = LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE).fit(
            X_tr_g3[:, PCA_DIM:], y_tr
        )

        # ------------------------------------------------------------------
        # Evaluate W=8 variants (G1, G2, G3/B3, B1, B2)
        # ------------------------------------------------------------------
        feats_w8 = build_causal_w8_eval_features(
            covs_full_list[te_idx], covs_hb_list[te_idx],
            log_full_list[te_idx], log_hb_list[te_idx],
            pca_c_full, pca_c_hb,
        )
        acc_g1.append(float(clf_g1.score(feats_w8["G1"], y_te_eval)))
        acc_g2.append(float(clf_g2.score(feats_w8["G2"], y_te_eval)))

        probs_g3 = clf_g3.predict_proba(feats_w8["G3"])
        preds_g3 = clf_g3.classes_[np.argmax(probs_g3, axis=1)]
        acc_g3.append(float(np.mean(preds_g3 == y_te_eval)))
        acc_g3_80.append(selective_accuracy(preds_g3, y_te_eval, np.max(probs_g3, axis=1), 0.80))

        acc_b1.append(float(clf_b1.score(feats_w8["B1"], y_te_eval)))
        acc_b2.append(float(clf_b2.score(feats_w8["B2"], y_te_eval)))

        # ------------------------------------------------------------------
        # Experiment 3: Frozen K=15, Causal W=20, Causal W=8, Oracle K=All
        # ------------------------------------------------------------------
        # Frozen K=15
        m15_f = sym_expm(log_full_list[te_idx][:15].mean(axis=0))
        m15_h = sym_expm(log_hb_list[te_idx][:15].mean(axis=0))
        zf15_w = pca_c_full.transform(project_tangent(covs_full_list[te_idx][:15], m15_f))
        zh15_w = pca_c_hb.transform(project_tangent(covs_hb_list[te_idx][:15], m15_h))
        zf15_e = pca_c_full.transform(project_tangent(covs_te_full_eval, m15_f))
        zh15_e = pca_c_hb.transform(project_tangent(covs_te_hb_eval, m15_h))
        X_te_f15 = np.concatenate(
            [coral_fit_transform(zf15_w, zf15_e), coral_fit_transform(zh15_w, zh15_e)], axis=1
        )
        preds_f15 = clf_g3.predict(X_te_f15)
        acc_f15.append(float(np.mean(preds_f15 == y_te_eval)))

        # Causal W=20
        X_te_w20 = build_causal_w20_eval_features(
            covs_full_list[te_idx], covs_hb_list[te_idx],
            log_full_list[te_idx], log_hb_list[te_idx],
            pca_c_full, pca_c_hb,
        )
        preds_w20 = clf_g3.predict(X_te_w20)
        acc_w20.append(float(np.mean(preds_w20 == y_te_eval)))

        # Oracle K=All
        zf_all = pca_c_full.transform(rec_full_list[te_idx])
        zh_all = pca_c_hb.transform(rec_hb_list[te_idx])
        X_te_all = np.concatenate(
            [
                coral_fit_transform(zf_all, zf_all[EVAL_START_IDX:]),
                coral_fit_transform(zh_all, zh_all[EVAL_START_IDX:]),
            ],
            axis=1,
        )
        preds_all = clf_g3.predict(X_te_all)
        acc_all.append(float(np.mean(preds_all == y_te_eval)))

        # Record phase-specific accuracies
        pred_map = {
            "Frozen K=15": preds_f15,
            "Causal W=20": preds_w20,
            "Causal W=8": preds_g3,
            "Oracle K=All": preds_all,
        }
        for m_name, preds_m in pred_map.items():
            trans_accs[m_name].append(float(np.mean(preds_m[tmask_eval] == y_te_eval[tmask_eval])) if np.any(tmask_eval) else np.nan)
            steady_accs[m_name].append(float(np.mean(preds_m[smask_eval] == y_te_eval[smask_eval])) if np.any(smask_eval) else np.nan)

    # Convert to numpy arrays
    arr_k0, arr_k0_80 = np.array(acc_k0), np.array(acc_k0_80)
    arr_g1, arr_g2, arr_g3, arr_g3_80 = np.array(acc_g1), np.array(acc_g2), np.array(acc_g3), np.array(acc_g3_80)
    arr_b1, arr_b2 = np.array(acc_b1), np.array(acc_b2)
    arr_f15, arr_w20, arr_all = np.array(acc_f15), np.array(acc_w20), np.array(acc_all)

    # ==========================================================================
    # Print Experiment 1: Mechanism Ablation
    # ==========================================================================
    print("\n" + "=" * 86)
    print(f"EXPERIMENT 1: Alignment Mechanism Ablation (N={n_sub}, W=8, Full+HighBeta PCA-30)")
    print("-" * 86)
    for label, arr in [
        ("G1 (1st-order M_sub only + Global Scaler)", arr_g1),
        ("G2 (1st-order M_sub + Diagonal Scaling)  ", arr_g2),
        ("G3 (1st-order M_sub + Full OAS-CORAL)    ", arr_g3),
    ]:
        print(f"  {label}: {arr.mean()*100:5.2f}% ± {arr.std(ddof=1)*100:4.2f}%")
    print("-" * 86)
    print(f"  G2 vs G1 (Effect of Diagonal Scaling):   delta = {(arr_g2.mean()-arr_g1.mean())*100:+5.2f}%, p = {safe_wilcoxon(arr_g2, arr_g1):.4e}")
    print(f"  G3 vs G1 (Total 2nd-order CORAL Gain):   delta = {(arr_g3.mean()-arr_g1.mean())*100:+5.2f}%, p = {safe_wilcoxon(arr_g3, arr_g1):.4e}")
    print(f"  G3 vs G2 (Pure Off-Diag Rotation Gain):  delta = {(arr_g3.mean()-arr_g2.mean())*100:+5.2f}%, p = {safe_wilcoxon(arr_g3, arr_g2):.4e}")
    print("=" * 86)

    # ==========================================================================
    # Print Experiment 2: Band Necessity Ablation
    # ==========================================================================
    print("\n" + "=" * 86)
    print(f"EXPERIMENT 2: Band Necessity Ablation (N={n_sub}, W=8 + Full OAS-CORAL)")
    print("-" * 86)
    for label, arr in [
        ("B1 (Single-band Full 8-30 Hz, 30d)       ", arr_b1),
        ("B2 (Single-band High-Beta 21-30 Hz, 30d) ", arr_b2),
        ("B3 (Dual-band Full + High-Beta, 60d)     ", arr_g3),
    ]:
        print(f"  {label}: {arr.mean()*100:5.2f}% ± {arr.std(ddof=1)*100:4.2f}%")
    print("-" * 86)
    print(f"  B3 vs B1 (Incremental Value of HighBeta): delta = {(arr_g3.mean()-arr_b1.mean())*100:+5.2f}%, p = {safe_wilcoxon(arr_g3, arr_b1):.4e}")
    print(f"  B3 vs B2 (Dual-band vs HighBeta alone):   delta = {(arr_g3.mean()-arr_b2.mean())*100:+5.2f}%, p = {safe_wilcoxon(arr_g3, arr_b2):.4e}")
    print("=" * 86)

    # ==========================================================================
    # Print Experiment 3: Cross-Run Drift Tracking (Transition vs Steady)
    # ==========================================================================
    valid_phase_mask = np.ones(n_sub, dtype=bool)
    for m in phase_models:
        valid_phase_mask &= ~np.isnan(trans_accs[m]) & ~np.isnan(steady_accs[m])

    print("\n" + "=" * 92)
    print(
        f"EXPERIMENT 3: Cross-Run Drift Tracking (Transition: first 7 trials of R08/R12 vs Steady: rest, "
        f"n={int(valid_phase_mask.sum())})"
    )
    print("-" * 92)
    print(
        f"{'Mode':<16} | {'Overall':>14} | {'Transition (R08/12[0:7])':>24} | "
        f"{'Steady (R08/12[7:])':>20} | {'Trans - Steady':>14}"
    )
    print("-" * 92)
    overall_map = {
        "Frozen K=15": arr_f15,
        "Causal W=20": arr_w20,
        "Causal W=8": arr_g3,
        "Oracle K=All": arr_all,
    }
    t_arrays = {m: np.array(trans_accs[m])[valid_phase_mask] for m in phase_models}
    s_arrays = {m: np.array(steady_accs[m])[valid_phase_mask] for m in phase_models}

    for m in phase_models:
        ov = overall_map[m]
        ta = t_arrays[m]
        sa = s_arrays[m]
        diff_ts = (ta.mean() - sa.mean()) * 100
        print(
            f"{m:<16} | {ov.mean()*100:5.2f}% ± {ov.std(ddof=1)*100:4.2f}% | "
            f"{ta.mean()*100:10.2f}% ± {ta.std(ddof=1)*100:4.2f}%      | "
            f"{sa.mean()*100:7.2f}% ± {sa.std(ddof=1)*100:4.2f}%   | {diff_ts:+13.2f}%"
        )
    print("-" * 92)
    print(
        f"  Transition Phase: Causal W=8 vs Frozen K=15 -> "
        f"delta = {(t_arrays['Causal W=8'].mean()-t_arrays['Frozen K=15'].mean())*100:+.2f}%, "
        f"p = {safe_wilcoxon(t_arrays['Causal W=8'], t_arrays['Frozen K=15']):.4e}"
    )
    print(
        f"  Transition Phase: Causal W=8 vs Causal W=20 -> "
        f"delta = {(t_arrays['Causal W=8'].mean()-t_arrays['Causal W=20'].mean())*100:+.2f}%, "
        f"p = {safe_wilcoxon(t_arrays['Causal W=8'], t_arrays['Causal W=20']):.4e}"
    )
    print(
        f"  Steady Phase:     Causal W=8 vs Frozen K=15 -> "
        f"delta = {(s_arrays['Causal W=8'].mean()-s_arrays['Frozen K=15'].mean())*100:+.2f}%, "
        f"p = {safe_wilcoxon(s_arrays['Causal W=8'], s_arrays['Frozen K=15']):.4e}"
    )
    print(
        f"  Steady Phase:     Causal W=8 vs Causal W=20 -> "
        f"delta = {(s_arrays['Causal W=8'].mean()-s_arrays['Causal W=20'].mean())*100:+.2f}%, "
        f"p = {safe_wilcoxon(s_arrays['Causal W=8'], s_arrays['Causal W=20']):.4e}"
    )
    print("=" * 92)

    # ==========================================================================
    # Print Experiment 4: Fair 80% Confidence Rejection Check
    # ==========================================================================
    print("\n" + "=" * 86)
    print("EXPERIMENT 4: Fair 80%-Coverage Confidence Rejection (Baseline K=0 vs Causal W=8)")
    print("-" * 86)
    print(
        f"  Baseline (K=0):   100% Cov = {arr_k0.mean()*100:5.2f}% ± {arr_k0.std(ddof=1)*100:4.2f}% -> "
        f"80% Cov = {arr_k0_80.mean()*100:5.2f}% ± {arr_k0_80.std(ddof=1)*100:4.2f}% "
        f"(gain: {(arr_k0_80.mean()-arr_k0.mean())*100:+.2f}%)"
    )
    print(
        f"  Causal W=8 (G3):  100% Cov = {arr_g3.mean()*100:5.2f}% ± {arr_g3.std(ddof=1)*100:4.2f}% -> "
        f"80% Cov = {arr_g3_80.mean()*100:5.2f}% ± {arr_g3_80.std(ddof=1)*100:4.2f}% "
        f"(gain: {(arr_g3_80.mean()-arr_g3.mean())*100:+.2f}%)"
    )
    print(
        f"  80%-Cov Comparison (Causal W=8 vs Baseline K=0): "
        f"delta = {(arr_g3_80.mean()-arr_k0_80.mean())*100:+.2f}%, p = {safe_wilcoxon(arr_g3_80, arr_k0_80):.4e}"
    )
    print("=" * 86)
    print(f"\nTotal elapsed time: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
