"""
Chang et al. (2025) Multi-Paradigm Upper-Limb MI Dataset (MNE-chang2025-data) Reproduction
=========================================================================================
Tests our Dual-Band (Full 8-30 Hz + High-Beta 21-30 Hz) + Source PCA-30 +
Strictly Causal W=8 OAS-CORAL method on all 28 subjects (S201~S230) in
CHANG2025_DIR / data/MNE-chang2025-data.

Key verified dataset properties:
  - 28 subjects (S201~S230), 3~6 MI runs per subject (MI1..MI4 / MI6)
  - 59 EEG channels, original fs=1000 Hz -> full-59ch CAR -> 4th-order bandpass -> downsample 250 Hz ->
    12 overlapping core sensorimotor channels (FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP4; CPz not recorded)
  - Event markers: 1=Left Hand, 2=Right Hand, 3=Both Hands; min tail >= 4.047 s
  - Window: post-cue [0.5, 3.5] s (identical to EEGBCI and BCI-IV-2a)

Protocols evaluated on held-out subject trials[15:] (28-fold LOSO):
  1. Binary (Left Hand vs Right Hand: 1 vs 2)
  2. 3-Class (Left / Right / Both Hands: 1 vs 2 vs 3)
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
DATA_DIR = config.CHANG2025_DIR
ORIG_SFREQ = 1000.0
DECIM = 4
SFREQ = ORIG_SFREQ / DECIM  # 250.0 Hz
TMIN, TMAX = 0.5, 3.5

CORE12_CHANNELS = [
    "FC3", "FCz", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP4",
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
    """Compute regularized OAS spatial covariance matrices (n_trials, 12, 12)."""
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
    """Extract upper triangle (N, 12, 12) -> (N, 78) with sqrt(2) off-diagonal weights."""
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
# Subject Loader Worker (Parallelizable)
# ==============================================================================
def load_subject_worker(subj_dir: Path) -> dict:
    """Load pure MI runs for one Chang2025 subject -> 12-ch OAS covariances & labels."""
    sid = subj_dir.name
    mi_files = sorted(subj_dir.glob(f"{sid}_MI[0-9]*.set"))

    sos_filters = {
        name: butter(4, [lo, hi], btype="bandpass", fs=ORIG_SFREQ, output="sos")
        for name, (lo, hi) in BANDS
    }
    start_250 = int(round(TMIN * SFREQ))
    stop_250 = int(round(TMAX * SFREQ)) + 1

    band_epochs: dict[str, list[np.ndarray]] = {name: [] for name in sos_filters}
    y_parts: list[np.ndarray] = []

    for set_path in mi_files:
        eeg = sio.loadmat(set_path, simplify_cells=True)["EEG"]
        n_ch = int(eeg["nbchan"])
        n_pts = int(eeg["pnts"])
        ch_names = [c["labels"] for c in eeg["chanlocs"]]
        pick_idx = [ch_names.index(c) for c in CORE12_CHANNELS]

        raw = np.fromfile(set_path.with_suffix(".fdt"), dtype=np.float32).reshape(
            (n_ch, n_pts), order="F"
        ).astype(np.float64)
        raw = raw - raw.mean(axis=0, keepdims=True)  # Full 59-channel CAR
        raw_core = raw[pick_idx]

        events = eeg["event"]
        if isinstance(events, dict):
            events = [events]
        ev_lat = []
        ev_type = []
        for ev in events:
            t_val = int(ev["type"])
            if t_val in (1, 2, 3):
                s_250 = int(round((float(ev["latency"]) - 1.0) / DECIM))
                if s_250 + stop_250 <= (n_pts // DECIM):
                    ev_lat.append(s_250)
                    ev_type.append(t_val)

        y_parts.append(np.asarray(ev_type, dtype=np.int64))
        for band_name, sos in sos_filters.items():
            filt_250 = sosfiltfilt(sos, raw_core, axis=-1)[:, ::DECIM]
            epochs = np.stack(
                [filt_250[:, s + start_250 : s + stop_250] for s in ev_lat],
                axis=0,
            )
            band_epochs[band_name].append(epochs)

    covs_full = compute_trial_oas_covariances(np.concatenate(band_epochs["full"], axis=0))
    covs_hb = compute_trial_oas_covariances(np.concatenate(band_epochs["high_beta"], axis=0))
    return {
        "sid": sid,
        "cov_f": covs_full,
        "cov_h": covs_hb,
        "log_f": sym_logm(covs_full),
        "log_h": sym_logm(covs_hb),
        "y": np.concatenate(y_parts, axis=0),
    }


def filter_subject(sub: dict, allowed_classes: tuple[int, ...]) -> dict:
    mask = np.isin(sub["y"], allowed_classes)
    return {
        "sid": sub["sid"],
        "cov_f": sub["cov_f"][mask],
        "cov_h": sub["cov_h"][mask],
        "log_f": sub["log_f"][mask],
        "log_h": sub["log_h"][mask],
        "y": sub["y"][mask],
    }


# ==============================================================================
# Core LOSO Evaluation Routine
# ==============================================================================
def run_loso(subjects: list[dict], allowed_classes: tuple[int, ...], title: str) -> None:
    filtered = [filter_subject(s, allowed_classes) for s in subjects]
    models = ["Baseline (K=0)", "G1 (1st-order W=8)", "B1 (Full W=8 CORAL)", "Ours (Dual W=8 CORAL)"]
    scores: dict[str, list[float]] = {m: [] for m in models}

    rec_f_all = []
    rec_h_all = []
    for s in filtered:
        mf = sym_expm(s["log_f"].mean(axis=0))
        mh = sym_expm(s["log_h"].mean(axis=0))
        rec_f_all.append(project_tangent(s["cov_f"], mf))
        rec_h_all.append(project_tangent(s["cov_h"], mh))

    n_sub = len(filtered)
    for te_idx in range(n_sub):
        tr_idx = [i for i in range(n_sub) if i != te_idx]
        y_tr = np.concatenate([filtered[i]["y"] for i in tr_idx], axis=0)
        te_sub = filtered[te_idx]
        y_te_eval = te_sub["y"][EVAL_START_IDX:]

        cov_tr_f = np.concatenate([filtered[i]["cov_f"] for i in tr_idx], axis=0)
        cov_tr_h = np.concatenate([filtered[i]["cov_h"] for i in tr_idx], axis=0)
        log_tr_f = np.concatenate([filtered[i]["log_f"] for i in tr_idx], axis=0)
        log_tr_h = np.concatenate([filtered[i]["log_h"] for i in tr_idx], axis=0)

        m_tr_f = sym_expm(log_tr_f.mean(axis=0))
        m_tr_h = sym_expm(log_tr_h.mean(axis=0))
        v_tr_k0_f = project_tangent(cov_tr_f, m_tr_f)
        v_tr_k0_h = project_tangent(cov_tr_h, m_tr_h)
        pca_k0_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_tr_k0_f)
        pca_k0_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(v_tr_k0_h)

        X_tr_k0 = np.concatenate([pca_k0_f.transform(v_tr_k0_f), pca_k0_h.transform(v_tr_k0_h)], axis=1)
        X_te_k0 = np.concatenate(
            [
                pca_k0_f.transform(project_tangent(te_sub["cov_f"][EVAL_START_IDX:], m_tr_f)),
                pca_k0_h.transform(project_tangent(te_sub["cov_h"][EVAL_START_IDX:], m_tr_h)),
            ],
            axis=1,
        )
        clf_k0 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE)),
        ]).fit(X_tr_k0, y_tr)
        scores["Baseline (K=0)"].append(float(np.mean(clf_k0.predict(X_te_k0) == y_te_eval)))

        pca_f = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(
            np.concatenate([rec_f_all[i] for i in tr_idx], axis=0)
        )
        pca_h = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit(
            np.concatenate([rec_h_all[i] for i in tr_idx], axis=0)
        )

        tr_g1, tr_b1, tr_ours = [], [], []
        for i in tr_idx:
            zf = pca_f.transform(rec_f_all[i])
            zh = pca_h.transform(rec_h_all[i])
            zf_c = coral_fit_transform(zf, zf)
            zh_c = coral_fit_transform(zh, zh)
            tr_g1.append(np.concatenate([zf, zh], axis=1))
            tr_b1.append(zf_c)
            tr_ours.append(np.concatenate([zf_c, zh_c], axis=1))

        clf_g1 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE)),
        ]).fit(np.concatenate(tr_g1, axis=0), y_tr)
        clf_b1 = LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE).fit(
            np.concatenate(tr_b1, axis=0), y_tr
        )
        clf_ours = LogisticRegression(max_iter=1000, C=LOGREG_C, random_state=RANDOM_STATE).fit(
            np.concatenate(tr_ours, axis=0), y_tr
        )

        n_te = len(te_sub["y"])
        n_eval = n_te - EVAL_START_IDX
        X_te_g1 = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)
        X_te_b1 = np.empty((n_eval, PCA_DIM), dtype=np.float64)
        X_te_ours = np.empty((n_eval, PCA_DIM * 2), dtype=np.float64)

        for out_idx, t in enumerate(range(EVAL_START_IDX, n_te)):
            start = max(0, t - CAUSAL_WINDOW)
            mf = sym_expm(te_sub["log_f"][start:t].mean(axis=0))
            mh = sym_expm(te_sub["log_h"][start:t].mean(axis=0))
            zf = pca_f.transform(project_tangent(te_sub["cov_f"][start : t + 1], mf))
            zh = pca_h.transform(project_tangent(te_sub["cov_h"][start : t + 1], mh))

            X_te_g1[out_idx, :PCA_DIM] = zf[-1]
            X_te_g1[out_idx, PCA_DIM:] = zh[-1]

            zf_c = coral_fit_transform(zf[:-1], zf[-1:])[0]
            zh_c = coral_fit_transform(zh[:-1], zh[-1:])[0]
            X_te_b1[out_idx] = zf_c
            X_te_ours[out_idx, :PCA_DIM] = zf_c
            X_te_ours[out_idx, PCA_DIM:] = zh_c

        scores["G1 (1st-order W=8)"].append(float(np.mean(clf_g1.predict(X_te_g1) == y_te_eval)))
        scores["B1 (Full W=8 CORAL)"].append(float(np.mean(clf_b1.predict(X_te_b1) == y_te_eval)))
        scores["Ours (Dual W=8 CORAL)"].append(float(np.mean(clf_ours.predict(X_te_ours) == y_te_eval)))

    arrs = {m: np.asarray(scores[m], dtype=np.float64) for m in models}
    ours = arrs["Ours (Dual W=8 CORAL)"]
    n_eval_tot = int(sum(len(s["y"]) - EVAL_START_IDX for s in filtered))
    print("\n" + "=" * 108)
    print(f"{title} (N={n_sub} subjects, total eval trials={n_eval_tot})")
    print("=" * 108)
    print(f"{'Model':<26} | {'Mean ± SD':>20} | {'vs Ours Delta':>14} | {'Wilcoxon p':>12} | {'Ours W/T/L':>12}")
    print("-" * 108)
    for m in models:
        a = arrs[m]
        mean_sd = f"{a.mean()*100:6.2f}% ± {a.std(ddof=1)*100:5.2f}%"
        if m == "Ours (Dual W=8 CORAL)":
            print(f"{m:<26} | {mean_sd:>20} | {'—':>14} | {'—':>12} | {'—':>12}")
        else:
            delta = (ours.mean() - a.mean()) * 100.0
            diff = ours - a
            p = 1.0 if np.all(np.abs(diff) < 1e-12) else float(wilcoxon(ours, a)[1])
            w = int(np.sum(ours > a))
            t = int(np.sum(np.abs(diff) < 1e-12))
            l = n_sub - w - t
            p_str = f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"
            print(f"{m:<26} | {mean_sd:>20} | {delta:+13.2f}% | {p_str:>12} | {f'{w}/{t}/{l}':>12}")

    low_mask = arrs["Baseline (K=0)"] < np.median(arrs["Baseline (K=0)"])
    print("-" * 108)
    print(
        f"Lower-half baseline subgroup (n={int(low_mask.sum())}): "
        + " | ".join(f"{m}: {arrs[m][low_mask].mean()*100:.2f}%" for m in models)
    )


def main() -> None:
    t0 = time.perf_counter()
    subj_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    print("=" * 108)
    print(f"Chang et al. (2025) Cross-Dataset Reproduction | Dir: {DATA_DIR} ({len(subj_dirs)} subjects)")
    print("Config: 59-ch CAR -> 12 core motor ch -> 250 Hz, [0.5, 3.5]s, OAS (12x12 -> 78d), PCA-30, W=8")
    print("=" * 108)

    t_load = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 4)) as executor:
        subjects = list(executor.map(load_subject_worker, subj_dirs))
    print(f"Loaded {len(subjects)} subjects in {time.perf_counter() - t_load:.2f} s.")

    run_loso(subjects, (1, 2), "1. BINARY (Left vs Right Hand) - 28-FOLD LOSO (trials[15:])")
    run_loso(subjects, (1, 2, 3), "2. 3-CLASS (Left / Right / Both Hands) - 28-FOLD LOSO (trials[15:])")

    print("\n" + "=" * 108)
    print(f"Total Runtime (Load + Both Protocols): {time.perf_counter() - t0:.2f} s")
    print("=" * 108)


if __name__ == "__main__":
    main()
