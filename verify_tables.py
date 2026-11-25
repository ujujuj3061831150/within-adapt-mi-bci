"""Verify Tables 2--4 and primary statistical claims from `results/*.npz`.

Usage:
    python verify_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def exact_wilcoxon(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> tuple[float, float]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    nz = d[np.abs(d) > tol]
    if len(nz) == 0:
        return 1.0, 0.0
    meth = "exact" if len(nz) <= 25 else "auto"
    p = float(wilcoxon(nz, method=meth).pvalue)
    ranks = rankdata(np.abs(nz))
    rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / (len(nz) * (len(nz) + 1) / 2.0))
    return p, rb


def holm(pvals: list[float]) -> list[float]:
    arr = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(arr)
    adj = np.empty_like(arr)
    prev = 0.0
    for rk, idx in enumerate(order):
        val = min(1.0, max(prev, arr[idx] * (len(arr) - rk)))
        adj[idx] = val
        prev = val
    return adj.tolist()


def main() -> None:
    m = np.load(RESULTS_DIR / "step2_unified_causal_master.npz")
    dl = np.load(RESULTS_DIR / "step2_dl_and_latency.npz")
    s4 = np.load(RESULTS_DIR / "step4_fixes.npz")
    p1 = np.load(RESULTS_DIR / "p1_audit_updates.npz")
    dd = np.load(RESULTS_DIR / "dispersion_drift_stress.npz")
    rr = np.load(RESULTS_DIR / "reviewer_revision_ablations.npz")
    fa = np.load(RESULTS_DIR / "final_audit_checks.npz")
    ex = np.load(RESULTS_DIR / "step2_extras.npz")

    print("=== TABLE 2: EEGBCI (N=109, Post-Burn-In) & Deep Baselines ===")
    wa_eeg = m["eegbci_bin_loso__Within-Adapt"]
    unc_eeg = m["eegbci_bin_loso__Uncal"]
    st_dl = dl["eegbci_bin_loso__Static_EEGNet__acc15"]
    ea_dl = dl["eegbci_bin_loso__Online_EA_EEGNet__acc15"]
    for name, arr, tm, lat in [
        ("Static EEGNet-8,2", st_dl, f"{dl['eegbci_bin_loso__Static_EEGNet__time_seed0'][0]:.2f} s/seed GPU", "---"),
        ("Uncal (fixed tangent)", unc_eeg, f"{p1['uncal_1proc_full0_time'][0]:.2f} s 1-CPU", f"{p1['uncal_lat_13__mean'][0]:.2f} ms CPU"),
        ("Online EA + EEGNet-8,2", ea_dl, f"{dl['eegbci_bin_loso__Online_EA_EEGNet__time_seed0'][0]:.2f} s/seed GPU",
         f"{s4['dl_1th_cpu_13__mean'][0]:.2f} ms 1T CPU / {dl['lat_dl_13__gpu_mean'][0]:.2f} ms GPU"),
        ("Within-Adapt (WA)", wa_eeg,
         f"{s4['wa_1proc_full0_time'][0]:.2f} s full / {dl['wa_1proc_109_time'][0]:.2f} s post-15",
         f"{s4['dense_wa_13__mean'][0]:.2f} ms CPU"),
    ]:
        pv = exact_wilcoxon(wa_eeg, arr)[0] if name != "Within-Adapt (WA)" else float("nan")
        print(f"  {name:24s}: {arr.mean()*100:5.2f} +- {arr.std(ddof=1)*100:5.2f}% | fit={tm} | inf={lat} | p vs WA={pv:.4g}")

    print("\n=== TABLE 3: Cross-Dataset Generalization (10 Protocols x 10 Methods) ===")
    protos = [
        ("eegbci_bin_loso", "EEGBCI 2-cls LOSO"),
        ("2a_bin_loso", "BCI-IV-2a 2-cls LOSO"),
        ("2a_4cls_loso", "BCI-IV-2a 4-cls LOSO"),
        ("chang_bin", "Chang2025 2-cls LOSO"),
        ("chang_3cls", "Chang2025 3-cls LOSO"),
        ("kaya_bin_loso", "Kaya2018 2-cls LOSO"),
        ("kaya_3cls_loso", "Kaya2018 3-cls LOSO"),
        ("2a_bin_cs", "BCI-IV-2a 2-cls CS"),
        ("2a_4cls_cs", "BCI-IV-2a 4-cls CS"),
        ("kaya_bin_cs", "Kaya2018 2-cls CS"),
    ]
    methods = [
        "Uncal", "G1", "Causal RPA (W=8)", "Direct Tang-LR", "Online EA + TangLR",
        "Orig", "z-CORAL (OAS-Src)", "Adapt-CORAL", "Within-CORAL(0.7)", "Within-Adapt",
    ]
    raw_unc = [exact_wilcoxon(m[f"{t}__Within-Adapt"], m[f"{t}__Uncal"])[0] for t, _ in protos]
    holm_unc = holm(raw_unc)
    for idx, (tag, lbl) in enumerate(protos):
        wa = m[f"{tag}__Within-Adapt"]
        un = m[f"{tag}__Uncal"]
        pu, ru = exact_wilcoxon(wa, un)
        pg, _ = exact_wilcoxon(wa, m[f"{tag}__G1"])
        pr, _ = exact_wilcoxon(wa, m[f"{tag}__Causal RPA (W=8)"])
        pz, _ = exact_wilcoxon(wa, m[f"{tag}__z-CORAL (OAS-Src)"])
        scores = " ".join(f"{m[f'{tag}__{k}'].mean()*100:5.2f}" for k in methods)
        print(f"  {lbl:21s} (n={len(wa):3d}): [{scores}] | Gain={(wa-un).mean()*100:+6.2f} pp raw={pu:.4g} holm={holm_unc[idx]:.4g} r={ru:+.3f} | p_G1={pg:.4f} p_RPA={pr:.4f} p_zC={pz:.4f}")

    p_2a_zc = [exact_wilcoxon(m[f"{t}__Within-Adapt"], m[f"{t}__z-CORAL (OAS-Src)"])[0] for t in ("2a_bin_cs", "2a_4cls_cs", "2a_4cls_loso")]
    print("  BCI-IV-2a 3-contrast vs z-CORAL Holm p:", [round(x, 4) for x in holm(p_2a_zc)])

    print("\n=== TABLE 4: Multi-Class W-Sweep & Dense CPU Latency ===")
    for prefix, wins in [("wsweep_2a4", [8, 12, 16, 24, 48]), ("wsweep_ch3", [8, 12, 16, 24])]:
        for meth in ["G1", "Orig", "z-CORAL (OAS-Src)", "Within-Adapt"]:
            vals = " | ".join(f"W={w}:{m[f'{prefix}__{meth}__W{w}'].mean()*100:5.2f}%" for w in wins)
            print(f"  {prefix} {meth:18s}: {vals}")
    print(f"  Dense CPU 13ch: {s4['dense_wa_13__mean'][0]:.2f} +- {s4['dense_wa_13__std'][0]:.2f} ms (P95={s4['dense_wa_13__p95'][0]:.2f} ms)")
    print(f"  Dense CPU 22ch: {s4['dense_wa_22__mean'][0]:.2f} +- {s4['dense_wa_22__std'][0]:.2f} ms (P95={s4['dense_wa_22__p95'][0]:.2f} ms)")

    print("\n=== KEY STRESSORS, ABLATIONS & INTERPRETABILITY CHECKS ===")
    wa12 = rr["car12__chang_3cls__Within-Adapt"]
    g112 = rr["car12__chang_3cls__G1"]
    zc12 = rr["car12__chang_3cls__z-CORAL (OAS-Src)"]
    print(f"  12-ch CAR Chang 3-cls : WA={wa12.mean()*100:.2f}% vs G1={g112.mean()*100:.2f}% (p={exact_wilcoxon(wa12,g112)[0]:.4f}) | z-CORAL={zc12.mean()*100:.2f}% (p={exact_wilcoxon(wa12,zc12)[0]:.4f})")
    for k in ("2a_4cls__disp_3x", "2a_4cls__disp_0.25x", "chang_3cls__disp_0.25x", "chang_3cls__aniso_ch_drift"):
        wa_s, g1_s, zc_s = dd[f"{k}__WA"], dd[f"{k}__G1"], dd[f"{k}__zCORAL"]
        print(f"  Stress {k:26s}: WA={wa_s.mean()*100:.2f}% vs G1={g1_s.mean()*100:.2f}% (p={dd[f'{k}__vs_G1__p'][0]:.4g}) | z-CORAL={zc_s.mean()*100:.2f}% (p={dd[f'{k}__vs_zCORAL__p'][0]:.4g})")
    print(f"  Reproj vs Cache (2a 4-cls) : WA={rr['diag_2a_4cls__reproj__WA'].mean()*100:.2f}% vs {rr['diag_2a_4cls__cache__WA'].mean()*100:.2f}% (p={exact_wilcoxon(rr['diag_2a_4cls__reproj__WA'], rr['diag_2a_4cls__cache__WA'])[0]:.4f})")
    print(f"  z-Adapt vs WA (2a 4-cls CS): WA={fa['c2__2a_4cls_cs__Within_Adapt'].mean()*100:.2f}% vs z-Adapt={fa['c2__2a_4cls_cs__z_Adapt'].mean()*100:.2f}% (p={exact_wilcoxon(fa['c2__2a_4cls_cs__Within_Adapt'], fa['c2__2a_4cls_cs__z_Adapt'])[0]:.4f})")
    print(f"  2a 4-cls EA+EEGNet LOSO    : WA={m['2a_4cls_loso__Within-Adapt'].mean()*100:.2f}% vs EA+EEGNet={dl['2a_4cls_loso__Online_EA_EEGNet__acc15'].mean()*100:.2f}% (p={exact_wilcoxon(m['2a_4cls_loso__Within-Adapt'], dl['2a_4cls_loso__Online_EA_EEGNet__acc15'])[0]:.4f})")
    print(f"  EEGBCI Haufe focal ratios  : Full={s4['focal_ratio_full'][0]:.2f}x, HB={s4['focal_ratio_hb'][0]:.2f}x")
    print(f"  EEGBCI burst-6             : WA={ex['eeg_burst6__Within-Adapt'].mean()*100:.2f}% vs G1={ex['eeg_burst6__G1'].mean()*100:.2f}%")

    cssp_json = RESULTS_DIR / "cssp_revision_checks.json"
    if cssp_json.exists():
        cj = json.loads(cssp_json.read_text(encoding="utf-8"))
        print("\n=== CSSP SUPPLEMENTARY CHECKS (Table 4 controls, PCA D-sweep, real latency) ===")
        for tag in ("2a_4cls_cs", "2a_4cls_loso"):
            m2t = cj["m2"][tag]
            zc = cj["z_coral_ss_2a4"][tag]
            print(f"  Confounders {tag:12s}: WA={m2t['WA']['mean']:.2f}% | zAd+SS={m2t['zAdapt_SS']['mean']:.2f}%(p={m2t['zAdapt_SS']['p_vs_WA']:.4f}) | gamma=1:{m2t['WA_gamma1']['mean']:.2f}% | zCORAL+SS={zc['zCORAL_SS']:.2f}%(p={zc['p_SS_vs_WA']:.4f})")
        for tag in ("2a_4cls__disp_3x", "2a_4cls__disp_0.25x", "2a_4cls__aniso_ch_drift", "chang_3cls__disp_3x", "chang_3cls__disp_0.25x", "chang_3cls__aniso_ch_drift"):
            r = cj["m2"][tag]
            print(f"  Stress {tag:26s}: Uncal={r['Uncal']['mean']:.2f}% | gamma=1:{r['WA_gamma1']['mean']:.2f}% | zAd+SS={r['zAdapt_SS']['mean']:.2f}%(p={r['zAdapt_SS']['p_vs_WA']:.4f}) | WA={r['WA']['mean']:.2f}%")
        c12 = cj["chang3_car12"]
        print(f"  Stress chang3_car12              : gamma=1:{c12['WA_gamma1']['mean']:.2f}%(p={c12['WA_gamma1']['p_vs_WA']:.4f}) | zAd+SS={c12['zAdapt_SS']['mean']:.2f}%(p={c12['zAdapt_SS']['p_vs_WA']:.4f}) | WA={c12['Within-Adapt']['mean']:.2f}%")
        for tag in ("2a_4cls_cs", "2a_4cls_loso"):
            pd = cj["m5_pca"][tag]
            print(f"  PCA D-sweep {tag:12s}: " + " | ".join(f"D{d}:WA={pd[f'D{d}']['WA']:.2f}%/zC={pd[f'D{d}']['zCORAL']:.2f}%" for d in (15, 30, 60, 120)))
        rl = cj["m6_real_latency"]
        print(f"  Real CPU latency                 : 13ch={rl['EEGBCI_13ch_real']['mean']:.2f}+-{rl['EEGBCI_13ch_real']['std']:.2f} ms (P95={rl['EEGBCI_13ch_real']['p95']:.2f}) | 22ch={rl['BCIC2a_22ch_real']['mean']:.2f}+-{rl['BCIC2a_22ch_real']['std']:.2f} ms (P95={rl['BCIC2a_22ch_real']['p95']:.2f})")


if __name__ == "__main__":
    main()
