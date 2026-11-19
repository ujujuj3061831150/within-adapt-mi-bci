"""
Plan B: Fixed-Shrinkage Within-CORAL(gamma) Sweep vs. Adaptive Within-Adapt
==========================================================================
Tests gamma in {0.0, 0.1, 0.3, 0.5, 0.7, 0.9} on BCI-IV-2a (Bin CS, 4-Cls CS,
4-Cls LOSO) and Chang2025 (3-Cls LOSO) under the exact unified uV + causal sosfilt
+ newton-cg protocol, computing zero-excluded exact Wilcoxon p-values against Within-Adapt.
Saves to results/gamma_ablation.npz.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import config

import numpy as np
from sklearn.decomposition import PCA

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import run_p1_audit_updates as p1a
import run_step2_riemannian_master as step2
import unified_within_adapt_coral_benchmark as uni

OUT_NPZ = config.RESULTS_DIR / "gamma_ablation.npz"
GAMMAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def _gamma_worker(args: tuple) -> dict[str, float]:
    tag, idx = args
    blocks = step2._G_STATE["blocks"]
    if tag == "2a_bin_cs":
        tr = [blocks[idx]["T"]]
        te = blocks[idx]["E"]
    elif tag == "2a_4cls_cs":
        tr = [blocks[idx]["T"]]
        te = blocks[idx]["E"]
    elif tag == "2a_4cls_loso":
        tr = [blocks[j][s] for j in range(9) if j != idx for s in ("T", "E")]
        te = blocks[idx]["E"]
    else:
        tr = [blocks[j] for j in range(len(blocks)) if j != idx]
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
    pf = PCA(n_components=30, random_state=42).fit(np.concatenate([r[0] for r in rec], 0))
    ph = PCA(n_components=30, random_state=42).fit(np.concatenate([r[1] for r in rec], 0))
    zf_l = [pf.transform(r[0]) for r in rec]
    zh_l = [ph.transform(r[1]) for r in rec]
    zf_all, zh_all = np.concatenate(zf_l, 0), np.concatenate(zh_l, 0)
    wfi, sfc, sfs = uni.fit_within_operator(zf_all, y_tr)
    whi, shc, shs = uni.fit_within_operator(zh_all, y_tr)

    out = {}
    n_te = len(te["y"])
    for g in GAMMAS:
        Xtr = np.concatenate(
            [
                np.concatenate(
                    [
                        uni.within_fixed_transform(zf @ wfi, zf @ wfi, g),
                        uni.within_fixed_transform(zh @ whi, zh @ whi, g),
                    ],
                    1,
                )
                for zf, zh in zip(zf_l, zh_l)
            ],
            0,
        )
        clf = step2.fit_lr(Xtr, y_tr)
        Xte = np.empty((n_te - 15, 60), dtype=np.float64)
        for oi, t in enumerate(range(15, n_te)):
            mf = b2a.sym_expm(te["log_f"][t - 8 : t].mean(0))
            mh = b2a.sym_expm(te["log_h"][t - 8 : t].mean(0))
            vf = pf.transform(b2a.project_tangent(te["cov_f"][t - 8 : t + 1], mf)) @ wfi
            vh = ph.transform(b2a.project_tangent(te["cov_h"][t - 8 : t + 1], mh)) @ whi
            Xte[oi, :30] = uni.within_fixed_transform(vf[:-1], vf[-1:], g)[0]
            Xte[oi, 30:] = uni.within_fixed_transform(vh[:-1], vh[-1:], g)[0]
        out[f"g_{g}"] = float(np.mean(clf.predict(Xte) == y_te))

    # Reference Within-Adapt
    Xtr_wa = np.concatenate(
        [
            np.concatenate(
                [
                    step2.within_adapt_s(zf @ wfi, zf @ wfi, sfc, sfs, 1.0),
                    step2.within_adapt_s(zh @ whi, zh @ whi, shc, shs, 1.0),
                ],
                1,
            )
            for zf, zh in zip(zf_l, zh_l)
        ],
        0,
    )
    clf_wa = step2.fit_lr(Xtr_wa, y_tr)
    Xte_wa = np.empty((n_te - 15, 60), dtype=np.float64)
    for oi, t in enumerate(range(15, n_te)):
        mf = b2a.sym_expm(te["log_f"][t - 8 : t].mean(0))
        mh = b2a.sym_expm(te["log_h"][t - 8 : t].mean(0))
        vf = pf.transform(b2a.project_tangent(te["cov_f"][t - 8 : t + 1], mf)) @ wfi
        vh = ph.transform(b2a.project_tangent(te["cov_h"][t - 8 : t + 1], mh)) @ whi
        Xte_wa[oi, :30] = step2.within_adapt_s(vf[:-1], vf[-1:], sfc, sfs, 1.0)[0]
        Xte_wa[oi, 30:] = step2.within_adapt_s(vh[:-1], vh[-1:], shc, shs, 1.0)[0]
    out["WA"] = float(np.mean(clf_wa.predict(Xte_wa) == y_te))
    return out


def main() -> None:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        b2a_subs = list(ex.map(step2.worker_b2a, range(1, 10)))
    ch_dirs = sorted([d for d in ch25.DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("S")])
    with ProcessPoolExecutor(max_workers=8) as ex:
        ch_subs = list(ex.map(step2.worker_chang, ch_dirs))

    b2a_bin = [{s: step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2))) for s in ("T", "E")} for i in range(9)]
    b2a_4 = [{s: step2.subset_block(b2a_subs[i][s], np.isin(b2a_subs[i][s]["y"], (1, 2, 3, 4))) for s in ("T", "E")} for i in range(9)]
    ch_3 = [step2.subset_block(s, np.isin(s["y"], (1, 2, 3))) for s in ch_subs]

    store = {}
    for tag, blocks, n_sub in [
        ("2a_bin_cs", b2a_bin, 9),
        ("2a_4cls_cs", b2a_4, 9),
        ("2a_4cls_loso", b2a_4, 9),
        ("chang_3cls", ch_3, len(ch_3)),
    ]:
        with ProcessPoolExecutor(max_workers=8, initializer=step2._init_state, initargs=({"blocks": blocks},)) as ex:
            rows = list(ex.map(_gamma_worker, [(tag, i) for i in range(n_sub)]))
        wa = np.asarray([r["WA"] for r in rows], dtype=np.float64)
        store[f"{tag}__WA"] = wa
        parts = []
        for g in GAMMAS:
            arr = np.asarray([r[f"g_{g}"] for r in rows], dtype=np.float64)
            pv, rb, nz = p1a.exact_wilcoxon_clean(wa, arr)
            store[f"{tag}__g_{g}"] = arr
            store[f"{tag}__vs_g_{g}__p"] = np.asarray([pv])
            parts.append(f"g={g}:{arr.mean()*100:.2f}%(p={pv:.4f})")
        print(f"[{tag:12s}] WA={wa.mean()*100:.2f}% | " + " | ".join(parts))

    np.savez_compressed(OUT_NPZ, **store)
    print(f"\nSaved gamma ablation to {OUT_NPZ} in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
