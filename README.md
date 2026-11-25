# Discriminant-Preserving Causal Riemannian Alignment (`Within-Adapt`)

Official reproducibility codebase for:

> **Discriminant-Preserving Causal Riemannian Alignment for Zero-Calibration Online Motor Imagery Decoding**  
> Hongjie Chen and Fang Duan\*  
> College of Information Science and Engineering, Huaqiao University, Xiamen 361021, China  
> *(Submitted to Circuits, Systems, and Signal Processing)*

---

## Overview

Zero-calibration online motor imagery brain--computer interfaces (MI-BCIs) require strictly causal filtering (`sosfilt`, zero future lookahead) and rapid spatial covariance adaptation from trial `t = 0` without target labels. Under unified microvolt ($\mu$V) scaling and causal preprocessing across **153 subjects (4 public cohorts, 10 LOSO/cross-session protocols)**:

- **Stationary First-Order Sufficiency**: First-order causal Log-Euclidean sliding recentering (`G1`, $W=8$) accounts for most stationary cross-subject transfer gains and achieves $\pm 2.0\text{ pp}$ Wilcoxon TOST equivalence with `Within-Adapt` on 8 of 10 stationary benchmarks ($p \ge 0.0781$).
- **Why Ungated CORAL Fails**: Class-agnostic and ungated source-recolored correlation alignment (`Orig`, `z-CORAL`) compresses $(K-1)$-dimensional between-class discriminant variance below unity ($\mathrm{SNR}_k/(1+\mathrm{SNR}_k)<1$) when PCA compresses anisotropic high-channel tangent spaces (22-channel BCI-IV-2a, top $K-1$ $v$-space eigenvalue $\lambda_{\max} = 1.47$).
- **Variance-Safe Second-Order Safeguard (`Within-Adapt`)**: Combining causal nested Full (8--30 Hz) + High-Beta (21--30 Hz) OAS covariances, common-basepoint tangent re-projection, source within-class whitening ($\Sigma_W^{-1/2}$, normalizing nuisance noise to $I_D$ without external post-alignment scaling), and discrepancy-gated source-prior shrinkage ($\gamma_t \in [0.25, 0.92]$):
  - Eliminates ungated source-recolored CORAL negative transfer on BCI-IV-2a (`+3.79 pp` binary CS, `p = 0.0078`; `+6.02 pp` 4-class CS, `p = 0.0117`; `+3.58 pp` 4-class LOSO, `p = 0.0117`; 3-contrast Holm `p = 0.0234`).
  - Outperforms `G1` under minimalist 12-channel CAR referencing (`+1.38 pp`, `p = 0.0059`), dispersion mismatch (`+2.12` to `+2.65 pp`, `p <= 0.0273`), and per-trial three-channel contact-gain jitter (`+2.52 pp`, `p = 0.0025`), while online $\gamma_t < 1$ outperforms fixed $\gamma \equiv 1.0$ centering across Chang2025 3-class attenuation, jitter, and 12-channel CAR (`p <= 0.0151`).
  - Lifts 109-subject PhysioNet EEGBCI accuracy from `61.90%` to `69.53%` (Holm `p = 1.27e-8`) and outperforms 3-seed causal `Online EA + EEGNet-8,2` on 4-class BCI-IV-2a LOSO (`55.43%` vs. `43.93%`, `p = 0.0039`).
  - Executes in **1.00 ms/trial** dense (**0.95 ms** on real EEGBCI trials) on CPU (13 ch) and **14.39 s** for full 109-fold LOSO fitting on a single CPU core (`74x` faster per seed than GPU EEGNet retraining).

---

## Repository Layout

```text
within-adapt-mi-bci/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── verify_tables.py            # Prints primary benchmark tables & statistical tests from results/*.npz
├── generate_figures.py         # Reproduces manuscript Figs. 2--6 from results/*.npz
├── figures/                    # Final vector PDF and 600-DPI PNG manuscript figures (Figs. 1--6)
├── results/                    # Precomputed per-subject/fold .npz & CSSP check JSON archives
└── src/
    ├── config.py               # Environment-overridable dataset and output paths
    ├── bcic2a_causal_coral_repro.py
    ├── chang2025_causal_coral_repro.py
    ├── eegbci_reviewer_ablations.py
    ├── kaya2018_causal_coral_repro.py
    ├── unified_within_adapt_coral_benchmark.py
    ├── dl_loaders_causal.py
    ├── probe_step1_audit_and_causal.py
    ├── run_step2_riemannian_master.py
    ├── run_step2_dl_fair_and_latency.py
    ├── compute_step2_extras.py
    ├── run_step4_fixes.py
    ├── run_p1_audit_updates.py
    ├── run_gamma_ablation.py
    ├── run_dispersion_drift_stress.py
    ├── run_reviewer_revision_ablations.py
    ├── run_final_audit_checks.py
    └── run_cssp_revision_checks.py
```

---

## Quick Verification (No Raw EEG Required)

All per-subject and per-fold outputs reported in the manuscript are bundled in `results/*.npz`. After installing dependencies:

```bash
pip install -r requirements.txt
```

1. **Verify primary benchmark tables and statistical tests**:
   ```bash
   python verify_tables.py
   ```
2. **Regenerate Manuscript Figures 2--6** (vector PDF + 600-DPI PNG exported to `figures_reproduced/`):
   ```bash
   python generate_figures.py
   ```
   *(Figure 1 in `figures/fig1_within_adapt_overview.pdf` is the architecture diagram.)*

---

## Full End-to-End Reproduction from Raw Public Datasets

### 1. Download Public Datasets

Place the four public cohorts under `./data/` (or set `WITHIN_ADAPT_DATA_ROOT` / dataset-specific env vars defined in `src/config.py`):

| Cohort | Source Reference | Default Path under `WITHIN_ADAPT_DATA_ROOT` |
| :--- | :--- | :--- |
| **PhysioNet EEGBCI** (109 subjects) | Schalk et al. (2004); [PhysioNet eegmmidb](https://physionet.org/content/eegmmidb/1.0.0/) | `MNE-eegbci-data/files/eegmmidb/1.0.0/` |
| **BCI Competition IV-2a** (9 subjects) | Tangermann et al. (2012), [doi:10.3389/fnins.2012.00055](https://doi.org/10.3389/fnins.2012.00055) | `MNE-bnci-data/~bci/database/001-2014/` |
| **Kaya et al. (2018)** (17 sessions, `CLA`) | Kaya et al. (2018), *Scientific Data* 5:180211 | `MNE-kaya2018-data/` |
| **Chang et al. (2025)** (28 subjects) | Chang et al. (2025), [doi:10.1038/s41597-025-06147-6](https://doi.org/10.1038/s41597-025-06147-6) | `MNE-chang2025-data/` |

### 2. Run Full Pipeline Scripts in Order

```bash
export WITHIN_ADAPT_DATA_ROOT=/path/to/bci_data   # optional if placed in ./data

python src/run_step2_riemannian_master.py        # Main benchmark tables, EEGBCI regimes, cold start, W-sweep, latency
python src/run_step2_dl_fair_and_latency.py      # 3-seed causal EEGNet / EA+EEGNet & latency (GPU recommended)
python src/compute_step2_extras.py               # EEGBCI burst-6, Chang random shuffle, Haufe matrices
python src/probe_step1_audit_and_causal.py       # Auxiliary causal baseline comparisons
python src/run_step4_fixes.py                    # Clean Wilcoxon stats, n=106 control, eigenspace, Haufe, refits
python src/run_p1_audit_updates.py               # Zero-excluded Holm/DL stats, Uncal CPU runtime, 2a burst-6
python src/run_gamma_ablation.py                 # Fixed-gamma Within-CORAL sweep (Fig. 4b in manuscript)
python src/run_dispersion_drift_stress.py        # 3x/0.25x dispersion & 3-channel contact-gain jitter tests (Fig. 6a)
python src/run_reviewer_revision_ablations.py    # 12-ch CAR, Eq. (5) re-projection, cold-start sensitivity (Fig. 6b,c)
python src/run_final_audit_checks.py             # TOST equivalence, z-Adapt ablation, Causal Recentered MDM
python src/run_cssp_revision_checks.py           # Component isolations, PCA D-sweep, real-trial latency, Kaya counts
```

Then regenerate figures into `figures_reproduced/` and verify all tables:

```bash
python verify_tables.py
python generate_figures.py
```

---

## Citation

If you find this codebase useful in your research, please cite:

```bibtex
@article{chen2026withinadapt,
  title   = {Discriminant-Preserving Causal Riemannian Alignment for Zero-Calibration Online Motor Imagery Decoding},
  author  = {Chen, Hongjie and Duan, Fang},
  journal = {Circuits, Systems, and Signal Processing},
  year    = {2026},
  url     = {https://github.com/ujujuj3061831150/within-adapt-mi-bci},
  note    = {Submitted}
}
```

## Contact & License

- **Hongjie Chen**: `chenhj976@gmail.com`  
- **Fang Duan** (Corresponding author): `nkfetsh@gmail.com`  

This codebase is released under the [MIT License](LICENSE).
