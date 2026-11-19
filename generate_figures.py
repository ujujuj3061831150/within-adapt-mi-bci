"""Standalone Matplotlib/MNE script reproducing Figures 2--6 from `results/*.npz`.

No external styling package is required. By default, outputs vector PDF and
600-DPI PNG files into `figures_reproduced/` so pre-packaged `figures/` remain
intact. Pass `--out-dir figures` to overwrite `figures/` directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import mne.channels.layout
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "figures_reproduced"

# Unified Okabe-Ito colorblind-safe palette (consistent across ALL figures)
COL = {
    "uncal": "#6E6E6E",      # dark neutral gray
    "static": "#A6A6A6",     # light neutral gray
    "orig": "#E69F00",       # orange / ochre
    "zcoral": "#D55E00",     # vermilion
    "g1": "#009E73",         # bluish green
    "rpa": "#56B4E9",        # sky blue
    "ea_dl": "#CC79A7",      # reddish purple
    "wa": "#0072B2",         # deep Okabe-Ito blue (ours)
}
INK = "#1A1A1A"
REF_GRAY = "#8C8C8C"

PROTOCOLS = [
    ("EEGBCI", "eegbci_bin_loso", 109, 2),
    ("BCI-IV-2a", "2a_bin_loso", 9, 2),
    ("BCI-IV-2a", "2a_4cls_loso", 9, 4),
    ("Chang2025", "chang_bin", 28, 2),
    ("Chang2025", "chang_3cls", 28, 3),
    ("Kaya2018", "kaya_bin_loso", 7, 2),
    ("Kaya2018", "kaya_3cls_loso", 7, 3),
    ("BCI-IV-2a", "2a_bin_cs", 9, 2),
    ("BCI-IV-2a", "2a_4cls_cs", 9, 4),
    ("Kaya2018", "kaya_bin_cs", 5, 2),
]

CORE_CHANNELS = [
    "FC3", "FCz", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CPz", "CP4",
]


def apply_elsevier_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (3.5, 2.625),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "figure.constrained_layout.use": False,
        "axes.unicode_minus": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def sem(arr: np.ndarray) -> float:
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    draws = np.random.default_rng(seed).integers(0, len(arr), (n_boot, len(arr)))
    return tuple(float(x) for x in np.percentile(arr[draws].mean(axis=1), [2.5, 97.5]))


def exact_wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    return float(wilcoxon(d, method="exact").pvalue)


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_bracket(ax, x0: float, x1: float, y: float, label: str, h: float = 1.6) -> None:
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color=INK, lw=0.7, clip_on=False)
    ax.annotate(label, ((x0 + x1) / 2.0, y), xytext=(0, 1.5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.3, color=INK)


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2.5)


def place_labels_aligned(fig, items: list[tuple], fontsize: float = 9.5) -> None:
    """Place bold (a), (b), ... labels aligned on shared figure coordinates."""
    fig.canvas.draw()
    for letter, x_fig, ax in items:
        y_fig = ax.get_position().y1 + 0.016
        fig.text(x_fig, y_fig, f"({letter})", fontsize=fontsize, fontweight="bold",
                 ha="left", va="bottom", color=INK)


def audit_and_export(fig, name: str, size_inches: tuple[float, float]) -> None:
    fig.set_size_inches(*size_inches)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    print(f"Saved {pdf_path.name} & {png_path.name} -> {OUT_DIR}")
    plt.close(fig)


# ==============================================================================
# FIG. 2: EEGBCI Main Benchmark (N=109), Operational Conditions, Runtime/Latency
# ==============================================================================
def build_fig2(m, dl, stats, pvals) -> None:
    apply_elsevier_style()
    size = (7.48, 4.85)
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 2, left=0.175, right=0.975, top=0.91, bottom=0.10,
                          height_ratios=[1.65, 0.90], width_ratios=[1.58, 1.0],
                          hspace=0.52, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    for ax in (ax_a, ax_b, ax_c, ax_d):
        clean_axis(ax)

    key = "eegbci_bin_loso"
    methods = [
        ("Uncal", m[f"{key}__Uncal"] * 100.0, COL["uncal"], "o"),
        ("Orig", m[f"{key}__Orig"] * 100.0, COL["orig"], "^"),
        ("G1", m[f"{key}__G1"] * 100.0, COL["g1"], "s"),
        ("Static\nEEGNet", dl[f"{key}__Static_EEGNet__acc15"] * 100.0, COL["static"], "o"),
        ("EA +\nEEGNet", dl[f"{key}__Online_EA_EEGNet__acc15"] * 100.0, COL["ea_dl"], "P"),
        ("Within-\nAdapt", m[f"{key}__Within-Adapt"] * 100.0, COL["wa"], "D"),
    ]
    rng = np.random.default_rng(16)
    ax_a.axhline(50.0, color="#C8C8C8", ls="--", lw=0.8, zorder=0)
    ax_a.axhline(12.0, color="#E0E0E0", ls="-", lw=0.6, zorder=0)

    bp = ax_a.boxplot(
        [vals for _, vals, _, _ in methods],
        positions=np.arange(len(methods)),
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=INK, lw=1.0),
        whiskerprops=dict(color="#555555", lw=0.7),
        capprops=dict(color="#555555", lw=0.7),
        boxprops=dict(lw=0.7),
        zorder=2,
    )
    for patch, (_, _, col, _) in zip(bp["boxes"], methods):
        patch.set_facecolor(col)
        patch.set_alpha(0.26)
        patch.set_edgecolor(col)

    for i, (_, vals, col, mk) in enumerate(methods):
        jit = rng.uniform(-0.17, 0.17, len(vals))
        ax_a.scatter(i + jit, vals, s=7.0, color=col, marker=mk, alpha=0.45,
                     edgecolors="none", zorder=3)
        ax_a.scatter([i], [vals.mean()], s=22, marker="D", facecolor="white",
                     edgecolor=INK, lw=0.9, zorder=5)
        ax_a.text(i, 6.2, f"{vals.mean():.1f}%", ha="center", va="center",
                  fontsize=7.0, fontweight="bold" if i == 5 else "normal", color=INK)

    ax_a.text(-0.56, 6.2, "Mean:", ha="right", va="center", fontsize=6.9, color="#555555", style="italic")
    add_bracket(ax_a, 4, 5, 104.0, sig_stars(float(pvals[f"dl_clean__{key}__vs_Online_EA_EEGNet__p"][0])))
    add_bracket(ax_a, 3, 5, 111.0, sig_stars(float(pvals[f"dl_clean__{key}__vs_Static_EEGNet__p"][0])))
    add_bracket(ax_a, 0, 5, 118.0, sig_stars(float(pvals[f"clean__{key}__raw_p"][0])))
    ax_a.set(
        xlim=(-1.25, 5.55), ylim=(1.0, 125.0),
        yticks=[20, 40, 60, 80, 100],
        xticks=range(6), xticklabels=[x[0] for x in methods],
        ylabel="LOSO accuracy (%)", title="EEGBCI benchmark (n = 109)",
    )
    ax_a.tick_params(axis="x", length=0, pad=3)

    # Panel (b): Operating conditions
    offsets = [-0.24, -0.08, 0.08, 0.24]
    op_meta = [
        ("Uncal", COL["uncal"], "o"),
        ("Orig", COL["orig"], "^"),
        ("G1", COL["g1"], "s"),
        ("Within-Adapt", COL["wa"], "D"),
    ]
    for off, (mname, col, mk) in zip(offsets, op_meta):
        vals = [m[f"eeg_reg__{mname}__{r}"] * 100.0 for r in ("low", "trans", "steady", "rej80")]
        vals = [v[np.isfinite(v)] for v in vals]
        ax_b.errorbar(
            np.arange(4) + off, [v.mean() for v in vals], yerr=[sem(v) for v in vals],
            fmt=mk, color=col, mec=INK if mname == "Within-Adapt" else col,
            mfc="white" if mname == "Orig" else col, mew=0.65, lw=0.95, ms=4.2,
            capsize=2.0, label=mname, zorder=4 if mname == "Within-Adapt" else 3,
        )
    ax_b.set(
        xlim=(-0.48, 3.48), ylim=(50.0, 83.0), yticks=[50, 60, 70, 80],
        xticks=range(4), xticklabels=["Low\nbaseline", "Run\nswitch", "Steady", "80%\nconf."],
        ylabel="Accuracy (% ± SEM)", title="Operating conditions",
    )
    ax_b.tick_params(axis="x", length=0, pad=3)
    ax_b.legend(loc="upper left", ncol=2, frameon=False, columnspacing=0.8,
                handletextpad=0.35, borderaxespad=0.2)

    # Panel (c): Offline 109-fold fitting time (log scale lollipop)
    batch = [
        ("Within-Adapt (CPU)", float(stats["wa_1proc_full0_time"][0]), COL["wa"], "D"),
        ("EA + EEGNet (GPU)", float(dl[f"{key}__Online_EA_EEGNet__time_seed0"][0]), COL["ea_dl"], "P"),
        ("Static EEGNet (GPU)", float(dl[f"{key}__Static_EEGNet__time_seed0"][0]), COL["static"], "o"),
    ]
    for y, (lbl, val, col, mk) in enumerate(batch):
        ax_c.plot([8.0, val], [y, y], color=col, lw=1.4, alpha=0.65, zorder=2)
        ax_c.scatter([val], [y], s=32, color=col, marker=mk, edgecolors=INK, linewidths=0.55, zorder=4)
        txt = f"{val:.1f} s" if y == 0 else f"{val:,.0f} s"
        ax_c.annotate(txt, (val, y), xytext=(5, 0), textcoords="offset points",
                      va="center", fontsize=7.4, fontweight="bold" if y == 0 else "normal", color=INK)
    ax_c.annotate("74× faster/seed", xy=(25.0, 0.08), xytext=(150.0, 0.55),
                  arrowprops=dict(arrowstyle="->", color=COL["wa"], lw=0.85, connectionstyle="arc3,rad=-0.15"),
                  fontsize=7.3, fontweight="bold", color=COL["wa"], ha="center")
    ax_c.set(
        xscale="log", xlim=(8.0, 5200.0), xticks=[10, 100, 1000],
        xticklabels=["10", "100", "1,000"],
        ylim=(2.55, -0.55), yticks=range(3), yticklabels=[b[0] for b in batch],
        xlabel="Wall-clock time (s, log scale)", title="Offline 109-fold fitting",
    )
    ax_c.minorticks_off()

    # Panel (d): Single-trial inference latency (Mean ± SD + P95 + 2.0 ms budget)
    lats = [
        ("WA · CPU", float(stats["dense_wa_13__mean"][0]), float(stats["dense_wa_13__std"][0]),
         float(stats["dense_wa_13__p95"][0]), COL["wa"], "D"),
        ("EA · 1T CPU", float(stats["dl_1th_cpu_13__mean"][0]), float(stats["dl_1th_cpu_13__std"][0]),
         None, COL["ea_dl"], "P"),
        ("EA · GPU", float(dl["lat_dl_13__gpu_mean"][0]), float(dl["lat_dl_13__gpu_std"][0]),
         float(dl["lat_dl_13__gpu_p95"][0]), COL["ea_dl"], "s"),
    ]
    ax_d.axvline(2.0, color="#B22222", ls=":", lw=0.85, zorder=1)
    ax_d.text(1.96, -0.45, "2 ms budget", ha="right", va="top", fontsize=6.7, color="#B22222")
    for y, (lbl, mu, sd, p95, col, mk) in enumerate(lats):
        ax_d.errorbar(mu, y, xerr=sd, fmt=mk, color=col, mec=INK, mew=0.5,
                      lw=1.1, capsize=2.4, ms=4.6, zorder=4)
        if p95 is not None:
            ax_d.scatter([p95], [y], s=20, marker="|", color=col, linewidths=1.2, zorder=4)
        ax_d.annotate(f"{mu:.2f} ± {sd:.2f}", (max(mu + sd, p95 or 0), y),
                      xytext=(4, 0), textcoords="offset points", va="center",
                      fontsize=7.2, color=INK)
    ax_d.set(
        xlim=(0.0, 2.15), xticks=[0.0, 0.5, 1.0, 1.5, 2.0],
        ylim=(2.55, -0.75), yticks=range(3), yticklabels=[x[0] for x in lats],
        xlabel="Latency (ms; bar = P95)", title="Single-trial inference",
    )

    place_labels_aligned(fig, [
        ("a", 0.015, ax_a), ("b", 0.595, ax_b),
        ("c", 0.015, ax_c), ("d", 0.595, ax_d),
    ])
    audit_and_export(fig, "fig2_eegbci_main_benchmark", size)


# ==============================================================================
# FIG. 3: Cross-Dataset Generalization (Individual-aware + Forest Gain Plot)
# ==============================================================================
def build_fig3(m, s4, pvals) -> None:
    apply_elsevier_style()
    size = (7.48, 5.25)
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 1, left=0.075, right=0.975, top=0.885, bottom=0.09,
                          height_ratios=[1.12, 1.10], hspace=0.45)

    top = gs[0].subgridspec(1, 2, width_ratios=[7.0, 3.0], wspace=0.09)
    ax_loso = fig.add_subplot(top[0])
    ax_cs = fig.add_subplot(top[1])

    cmp_methods = [
        ("Uncal", COL["uncal"], "o"),
        ("G1", COL["g1"], "s"),
        ("z-CORAL (OAS-Src)", COL["zcoral"], "v"),
        ("Within-Adapt", COL["wa"], "D"),
    ]
    rng = np.random.default_rng(7)

    for ax, group, title in [(ax_loso, PROTOCOLS[:7], "Cross-subject (LOSO)"),
                             (ax_cs, PROTOCOLS[7:], "Cross-session (CS)")]:
        clean_axis(ax)
        for i, (disp, key, n, ncls) in enumerate(group):
            chance = 100.0 / ncls
            ax.hlines(chance, i - 0.42, i + 0.42, color="#999999", ls=":", lw=0.85, zorder=0)
            xpos = i + np.array([-0.27, -0.09, 0.09, 0.27])
            data = [m[f"{key}__{meth}"] * 100.0 for meth, _, _ in cmp_methods]

            if n >= 10:
                # Slim boxplot + translucent points for n >= 10
                bp = ax.boxplot(
                    data, positions=xpos, widths=0.14, patch_artist=True,
                    manage_ticks=False, showfliers=False, zorder=2,
                    boxprops=dict(lw=0.55), whiskerprops=dict(lw=0.55, color="#555555"),
                    capprops=dict(lw=0.55, color="#555555"),
                    medianprops=dict(color=INK, lw=0.8),
                )
                for patch, (_, col, _) in zip(bp["boxes"], cmp_methods):
                    patch.set_facecolor(col)
                    patch.set_alpha(0.32)
                    patch.set_edgecolor(col)
                for x, vals, (_, col, mk) in zip(xpos, data, cmp_methods):
                    jit = rng.uniform(-0.04, 0.04, len(vals))
                    ax.scatter(x + jit, vals, s=3.5 if n > 50 else 5.0, color=col,
                               marker=mk, alpha=0.20 if n > 50 else 0.45, edgecolors="none", zorder=1.5)
                    ax.scatter([x], [vals.mean()], s=12, marker="D", facecolor="white",
                               edgecolor=INK, lw=0.7, zorder=4)
            else:
                # Small n (n=5,7,9): NO boxplot! Direct stripplot + paired mean bar
                stacked = np.stack(data)
                ax.plot(xpos, stacked, color="#666666", lw=0.45, alpha=0.28, zorder=1)
                for x, vals, (_, col, mk) in zip(xpos, data, cmp_methods):
                    ax.scatter(np.full(n, x), vals, s=11.0, color=col, marker=mk,
                               alpha=0.82, edgecolors=INK, linewidths=0.25, zorder=3)
                    ax.hlines(vals.mean(), x - 0.065, x + 0.065, color=INK, lw=1.4, zorder=4)

            if key in ("2a_4cls_loso", "2a_bin_cs", "2a_4cls_cs"):
                p_zc = float(s4[f"stat__{key}__vs_zCORAL__p"][0])
                add_bracket(ax, xpos[2], xpos[3], 103.5, sig_stars(p_zc), h=1.8)

        ax.set(
            xlim=(-0.52, len(group) - 0.48), ylim=(10.0, 114.0),
            yticks=[25, 50, 75, 100],
            xticks=range(len(group)),
            xticklabels=[f"{d}\n{k}-cls (n={n})" for d, _, n, k in group],
            title=title,
        )
        ax.tick_params(axis="x", length=0, labelsize=6.9, pad=3)
    ax_loso.set_ylabel("Accuracy (%)")
    ax_cs.tick_params(axis="y", labelleft=False)

    leg_handles = [
        Line2D([], [], color=col, marker=mk, ls="none", ms=5.0, mec=INK, mew=0.4, label=lbl)
        for lbl, col, mk in [("Uncal", COL["uncal"], "o"), ("G1", COL["g1"], "s"),
                             ("z-CORAL (OAS-Src)", COL["zcoral"], "v"), ("Within-Adapt", COL["wa"], "D")]
    ] + [Line2D([], [], color="#999999", ls=":", lw=1.0, label="Chance (1/K)")]
    fig.legend(handles=leg_handles, loc="upper center", bbox_to_anchor=(0.525, 0.985),
               ncol=5, frameon=False, columnspacing=1.1, handletextpad=0.35)

    # Panel (b): Forest plot of paired gain (Within-Adapt - Uncal) across 10 protocols
    bot = gs[1].subgridspec(1, 3, width_ratios=[1.25, 2.15, 1.35], wspace=0.06)
    ax_lbl = fig.add_subplot(bot[0])
    ax_for = fig.add_subplot(bot[1])
    ax_txt = fig.add_subplot(bot[2])

    rows = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
    for ax in (ax_lbl, ax_for, ax_txt):
        ax.set_ylim(10.6, -0.6)
        ax.set_yticks([])
    ax_lbl.set_axis_off()
    ax_txt.set_axis_off()
    ax_lbl.set_xlim(0, 1)
    ax_txt.set_xlim(0, 1)
    clean_axis(ax_for)
    ax_for.spines["left"].set_visible(False)
    ax_for.axvline(0.0, color=REF_GRAY, ls="--", lw=0.85, zorder=0)
    ax_for.axhline(7.0, color="#CCCCCC", lw=0.75, zorder=0)
    ax_lbl.axhline(7.0, color="#CCCCCC", lw=0.75, zorder=0)
    ax_txt.axhline(7.0, color="#CCCCCC", lw=0.75, zorder=0)

    ax_lbl.text(0.0, -0.95, "Protocol (Split)", fontweight="bold", fontsize=7.6)
    ax_lbl.text(0.98, -0.95, "n", ha="right", fontweight="bold", fontsize=7.6)
    ax_for.set_title("Paired Gain: Within-Adapt − Uncal (95% CI)", fontsize=8.2, pad=7)
    ax_txt.text(0.04, -0.95, "Mean Δ [95% CI] (pp)", fontweight="bold", fontsize=7.6)

    sig_styles = {
        "holm": (COL["wa"], "s", COL["wa"], "Holm P < 0.05"),
        "raw": (COL["g1"], "D", "white", "Raw P < 0.05 only"),
        "ns": ("#666666", "o", "white", "n.s. (P ≥ 0.05)"),
    }
    for y, (disp, key, n, ncls) in zip(rows, PROTOCOLS):
        split = "CS" if "cs" in key else "LOSO"
        ax_lbl.text(0.0, y, f"{disp} · {ncls}-cls ({split})", va="center", fontsize=7.3, color=INK)
        ax_lbl.text(0.98, y, str(n), ha="right", va="center", fontsize=7.3, color=INK)

        delta = (m[f"{key}__Within-Adapt"] - m[f"{key}__Uncal"]) * 100.0
        mu = float(delta.mean())
        lo, hi = bootstrap_ci(delta)
        ph = float(pvals[f"clean__{key}__holm_p"][0])
        pr = float(pvals[f"clean__{key}__raw_p"][0])
        st = "holm" if ph < 0.05 else ("raw" if pr < 0.05 else "ns")
        col, mk, mfc, _ = sig_styles[st]

        ax_for.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], fmt=mk, color=col,
                        mfc=mfc, mec=col, mew=1.0, ms=4.5, lw=1.0, capsize=2.0, zorder=3)
        star = " *" if st == "holm" else (" †" if st == "raw" else "")
        ax_txt.text(0.04, y, f"{mu:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}",
                    va="center", fontsize=7.2, color=INK)

    ax_for.set(xlim=(-4.5, 17.5), xticks=[-4, 0, 4, 8, 12, 16],
               xlabel="Accuracy difference (pp)")
    fleg = [
        Line2D([], [], color=c, marker=mk, mfc=mfc, mew=1.0, ls="none", ms=4.5, label=lbl)
        for c, mk, mfc, lbl in sig_styles.values()
    ]
    ax_for.legend(handles=fleg, loc="lower right", frameon=True, facecolor="white",
                  edgecolor="#DDDDDD", fontsize=6.7, handletextpad=0.3, borderpad=0.35)

    place_labels_aligned(fig, [("a", 0.015, ax_loso), ("b", 0.015, ax_for)])
    audit_and_export(fig, "fig3_cross_dataset_generalization", size)


# ==============================================================================
# FIG. 4: Cold-Start Gain (Dot+CI, no small-n boxplots), DL Comparison & W-Sweep
# ==============================================================================
def build_fig4(m, dl, pvals) -> None:
    apply_elsevier_style()
    size = (7.48, 5.45)
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 1, left=0.08, right=0.975, top=0.905, bottom=0.145,
                          height_ratios=[1.05, 1.05], hspace=0.46)

    # Panel (a): Stripplot + Mean ± 95% CI uniformly across all 10 protocols (P1 compliant!)
    top = gs[0].subgridspec(1, 2, width_ratios=[7.0, 3.0], wspace=0.09)
    ax_a1 = fig.add_subplot(top[0])
    ax_a2 = fig.add_subplot(top[1])
    rng = np.random.default_rng(27)
    for ax, group, title in [(ax_a1, PROTOCOLS[:7], "Cross-subject (LOSO)"),
                             (ax_a2, PROTOCOLS[7:], "Cross-session (CS)")]:
        clean_axis(ax)
        ax.axhline(0.0, color=REF_GRAY, ls="--", lw=0.8, zorder=0)
        for i, (_, key, n, _) in enumerate(group):
            for off, period, col, mk in [(-0.17, "burn", "#56B4E9", "o"), (0.17, "all", COL["wa"], "D")]:
                v = 100.0 * (m[f"{key}__CS__Within-Adapt__{period}"] - m[f"{key}__CS__Uncal__{period}"])
                x = i + off
                jit = rng.uniform(-0.06, 0.06, len(v))
                ax.scatter(x + jit, v, s=6.5 if n > 20 else 10.0, color=col, marker=mk,
                           alpha=0.35 if n > 20 else 0.72, edgecolors="none", zorder=2)
                mu = float(v.mean())
                lo, hi = bootstrap_ci(v)
                ax.errorbar(x, mu, yerr=[[mu - lo], [hi - mu]], fmt=mk, color=INK,
                            mfc=col, mec=INK, mew=0.65, ms=4.2, lw=1.0, capsize=2.0, zorder=4)
        ax.set(
            ylim=(-45.0, 52.0), yticks=[-40, -20, 0, 20, 40],
            xlim=(-0.50, len(group) - 0.50), xticks=range(len(group)),
            xticklabels=[f"{d.replace('BCI-IV-', '').replace('2025', '').replace('2018', '')}\n{k}-cls (n={n})"
                         for d, _, n, k in group],
            title=title,
        )
        ax.tick_params(axis="x", length=0, labelsize=6.9, pad=3)
    ax_a1.set_ylabel("Gain over Uncal (pp)")
    ax_a2.tick_params(axis="y", labelleft=False)
    a_leg = [
        Line2D([], [], color=INK, mfc="#56B4E9", marker="o", ls="none", ms=4.6, label="First 15 trials [0:15] (mean ± 95% CI)"),
        Line2D([], [], color=INK, mfc=COL["wa"], marker="D", ls="none", ms=4.6, label="Full stream [0:] (mean ± 95% CI)"),
    ]
    fig.legend(handles=a_leg, loc="upper center", bbox_to_anchor=(0.525, 0.985),
               ncol=2, frameon=False, columnspacing=1.2)

    # Bottom row: (b) DL Baselines paired gain + (c) W-Sweep without muddy overlapping ribbons
    bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.34)
    ax_b = fig.add_subplot(bot[0])
    clean_axis(ax_b)
    ax_b.axhline(0.0, color=REF_GRAY, ls="--", lw=0.8, zorder=0)
    dl_keys = [PROTOCOLS[i] for i in range(5)]
    for i, (_, key, n, _) in enumerate(dl_keys):
        wa_v = m[f"{key}__Within-Adapt"]
        for off, base, col, mk in [
            (-0.17, "Static_EEGNet", COL["static"], "o"),
            (0.17, "Online_EA_EEGNet", COL["ea_dl"], "P"),
        ]:
            delta = 100.0 * (wa_v - dl[f"{key}__{base}__acc15"])
            jit = rng.uniform(-0.07, 0.07, len(delta))
            ax_b.scatter(i + off + jit, delta, s=7.0 if n > 20 else 10.5, color=col,
                         marker=mk, alpha=0.45 if n > 20 else 0.78, edgecolors="none", zorder=2)
            ax_b.hlines(delta.mean(), i + off - 0.11, i + off + 0.11, color=INK, lw=1.4, zorder=4)
        p_ea = float(pvals[f"dl_clean__{key}__vs_Online_EA_EEGNet__p"][0])
        ax_b.text(i + 0.17, 48.0, sig_stars(p_ea), ha="center", va="bottom", fontsize=7.4, color=INK)
    ax_b.set(
        xlim=(-0.48, 4.48), ylim=(-32.0, 54.0), yticks=[-20, 0, 20, 40],
        xticks=range(5), xticklabels=["EEGBCI\n2-cls", "2a\n2-cls", "2a\n4-cls", "Chang\n2-cls", "Chang\n3-cls"],
        ylabel="Within-Adapt gain (pp)", title="vs. Causal deep baselines",
    )
    ax_b.tick_params(axis="x", length=0, labelsize=7.0, pad=3)
    b_leg = [
        Line2D([], [], color=COL["static"], marker="o", ls="none", ms=4.5, label="Static EEGNet"),
        Line2D([], [], color=COL["ea_dl"], marker="P", ls="none", ms=4.8, label="EA + EEGNet"),
    ]
    ax_b.legend(handles=b_leg, loc="upper center", bbox_to_anchor=(0.50, -0.21),
                ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)

    sw = bot[1].subgridspec(1, 2, wspace=0.28)
    ax_c1, ax_c2 = fig.add_subplot(sw[0]), fig.add_subplot(sw[1])
    sw_styles = [
        ("Within-Adapt", "Within-Adapt", COL["wa"], "-", "D"),
        ("G1", "G1", COL["g1"], "--", "s"),
        ("Orig", "Orig", COL["orig"], "-.", "^"),
        ("z-CORAL (OAS-Src)", "z-CORAL", COL["zcoral"], ":", "v"),
    ]
    for ax, prefix, wins, title, ylim, yticks in [
        (ax_c1, "wsweep_2a4", [8, 12, 16, 24, 48], "IV-2a · 4-cls (n=9)", (46.0, 63.5), [48, 52, 56, 60]),
        (ax_c2, "wsweep_ch3", [8, 12, 16, 24], "Chang · 3-cls (n=28)", (45.0, 54.0), [46, 48, 50, 52, 54]),
    ]:
        clean_axis(ax)
        ax.axvline(8, color=REF_GRAY, ls=":", lw=0.8)
        for key, lbl, col, ls, mk in sw_styles:
            vals = [100.0 * m[f"{prefix}__{key}__W{w}"] for w in wins]
            mu = np.array([v.mean() for v in vals])
            err = np.array([sem(v) for v in vals])
            # Only shade Within-Adapt ± SEM to prevent unreadable multi-ribbon overlap
            if key == "Within-Adapt":
                ax.fill_between(wins, mu - err, mu + err, color=col, alpha=0.14, lw=0)
            ax.plot(wins, mu, color=col, ls=ls, marker=mk, mfc="white" if key != "Within-Adapt" else col,
                    mec=col, mew=0.9, lw=1.25, ms=3.8, label=lbl, zorder=4 if key == "Within-Adapt" else 3)
        ax.set(
            xlim=(6, max(wins) + 2), ylim=ylim, yticks=yticks,
            xticks=[8, 16, 24, 48] if len(wins) == 5 else wins,
            title=title, xlabel="W (trials)",
        )
    ax_c1.set_ylabel("Accuracy (% ± SEM band)")
    h_sw, l_sw = ax_c1.get_legend_handles_labels()
    fig.legend(h_sw, l_sw, loc="upper center", bbox_to_anchor=(0.74, 0.068),
               ncol=2, frameon=False, columnspacing=1.0, handletextpad=0.4)

    place_labels_aligned(fig, [("a", 0.015, ax_a1), ("b", 0.015, ax_b), ("c", 0.485, ax_c1)])
    audit_and_export(fig, "fig4_coldstart_dl_and_wsweep", size)


# ==============================================================================
# FIG. 5: Subspace Anisotropy, Adaptive Gate Sweep & Grayscale-Safe Haufe Maps
# ==============================================================================
def build_fig5(stats, gamma) -> None:
    apply_elsevier_style()
    mne.set_log_level("ERROR")
    size = (7.48, 4.75)
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 1, left=0.09, right=0.92, top=0.91, bottom=0.06,
                          height_ratios=[1.0, 1.38], hspace=0.45)

    top = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.35)
    ax_a = fig.add_subplot(top[0])
    ax_b = fig.add_subplot(top[1])
    clean_axis(ax_a)
    clean_axis(ax_b)

    # Panel (a): Subspace variance with shaded anisotropic collapse zone
    ax_a.axhspan(9.0, 15.8, xmin=0.0, xmax=0.46, color="#FDE8E1", alpha=0.55, zorder=0)
    ax_a.text(49.5, 15.1, "Anisotropic regime\n(CORAL collapses)", fontsize=6.6,
              color=COL["zcoral"], va="top", fontweight="bold")
    diag_pts = [
        ("eegbci_bin", "EEGBCI (n=109)", COL["wa"], "D", (6, 4)),
        ("chang_3cls", "Chang 3-cls (n=28)", COL["g1"], "s", (6, 4)),
        ("2a_4cls", "IV-2a 4-cls (n=9)", COL["zcoral"], "^", (7, -10)),
    ]
    for key, lbl, col, mk, off in diag_pts:
        pca = float(stats[f"diag__{key}__pca30_var"][0]) * 100.0
        keep = float(stats[f"diag__{key}__k1_frac"][0]) * 100.0
        eig = float(stats[f"diag__{key}__eig_top"][0])
        ax_a.scatter([pca], [keep], s=36, color=col, marker=mk, edgecolors=INK, linewidths=0.55, zorder=4)
        ax_a.annotate(f"{lbl}\n$\\lambda_{{\\max}} = {eig:.3f}$", (pca, keep), xytext=off,
                      textcoords="offset points", fontsize=6.8, va="center", color=INK)
    ax_a.set(xlim=(48, 88), ylim=(1.0, 15.5), xticks=[50, 60, 70, 80],
             yticks=[3, 6, 9, 12, 15], xlabel="PCA-30 retained variance (%)",
             ylabel="Top (K−1) v-space share (%)", title="Subspace anisotropy")

    # Panel (b): Fixed gamma vs adaptive Within-Adapt
    levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    ax_b.axhline(0.0, color=REF_GRAY, ls="--", lw=0.85, zorder=1)
    ax_b.text(0.02, 0.08, "Adaptive Within-Adapt (0 pp)", fontsize=6.7, color="#555555")
    g_specs = [
        ("2a_bin_cs", "2a 2-cls CS", COL["rpa"], "o", "--"),
        ("2a_4cls_cs", "2a 4-cls CS", COL["zcoral"], "^", "-."),
        ("2a_4cls_loso", "2a 4-cls LOSO", COL["wa"], "s", "-"),
        ("chang_3cls", "Chang 3-cls", COL["g1"], "D", ":"),
    ]
    for key, lbl, col, mk, ls in g_specs:
        base = gamma[f"{key}__WA"].mean()
        deltas = [100.0 * (gamma[f"{key}__g_{g}"].mean() - base) for g in levels]
        ax_b.plot(levels, deltas, color=col, ls=ls, marker=mk, mfc="white", mec=col,
                  mew=0.9, ms=3.8, lw=1.15, label=lbl, zorder=3)
    ax_b.set(xticks=levels, ylim=(-1.60, 0.55), yticks=[-1.5, -1.0, -0.5, 0.0, 0.5],
             xlabel="Fixed shrinkage γ", ylabel="Accuracy Δ vs. adaptive (pp)",
             title="Fixed γ vs. adaptive gate")
    ax_b.legend(loc="lower right", ncol=2, frameon=True, facecolor="white",
                edgecolor="#DDDDDD", fontsize=6.6, columnspacing=0.7, handletextpad=0.35)

    # Panels (c, d): Standard RdBu_r diverging topomaps + explicit sensor dots + C3/Cz/C4 labels
    info = mne.create_info(CORE_CHANNELS, 160.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False)
    coords = mne.channels.layout._find_topomap_coords(info, picks="eeg")
    vals = [np.diag(stats[f"causal_haufe_{k}"]) for k in ("full", "hb")]
    lim = float(np.max(np.abs(vals)))
    norm = mpl.colors.Normalize(-lim, lim)
    cmap = plt.get_cmap("RdBu_r")

    bot = gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.10)
    xg = np.linspace(-0.105, 0.105, 225)
    X, Y = np.meshgrid(xg, xg)
    K = np.exp(-0.5 * (cdist(np.column_stack([X.ravel(), Y.ravel()]), coords) / 0.017) ** 2)
    denom = K.sum(axis=1) + 0.15

    topo_axes = []
    for i, (v, title) in enumerate(zip(vals, ["Full (8–30 Hz)", "High-Beta (21–30 Hz)"])):
        ax = fig.add_subplot(bot[i])
        topo_axes.append(ax)
        orig_im, _ = mne.viz.plot_topomap(np.zeros(len(v)), info, axes=ax, sensors=False,
                                          show=False, cmap=cmap, vlim=(-1, 1), contours=0)
        clip = orig_im.get_clip_path()
        orig_im.remove()

        z = ((K @ v) / denom).reshape(X.shape)
        fill = ax.contourf(X, Y, z, levels=np.linspace(-lim, lim, 96), norm=norm,
                           cmap=cmap, zorder=1, antialiased=False)
        fill.set_edgecolor("face")
        fill.set_linewidth(0.25)
        fill.set_clip_path(clip)

        # Signed contours: dashed for negative (ERD), solid for positive
        neg_lvls = np.linspace(-lim * 0.80, -lim * 0.20, 4)
        pos_lvls = np.linspace(lim * 0.20, lim * 0.80, 4)
        c_neg = ax.contour(X, Y, z, levels=neg_lvls, colors="#1F2937", linewidths=0.55, linestyles="--", zorder=2)
        c_pos = ax.contour(X, Y, z, levels=pos_lvls, colors="#1F2937", linewidths=0.55, linestyles="-", zorder=2)
        c_neg.set_clip_path(clip)
        c_pos.set_clip_path(clip)

        # Plot 13 core sensors and label C3 / Cz / C4 for unambiguous grayscale polarity reading
        ax.scatter(coords[:, 0], coords[:, 1], s=7.0, color="#222222", zorder=3)
        for ch_name, dx, dy in [("C3", -0.008, -0.013), ("Cz", 0.0, 0.010), ("C4", 0.008, -0.013)]:
            idx = CORE_CHANNELS.index(ch_name)
            ax.text(coords[idx, 0] + dx, coords[idx, 1] + dy, ch_name,
                    fontsize=6.4, fontweight="bold", ha="center", va="center", color="#111111", zorder=4)
        ax.set_title(title, pad=3, fontsize=8.3)

    cax = fig.add_subplot(bot[2])
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.14 * pos.height, pos.width * 0.85, pos.height * 0.72])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-0.06, 0.0, 0.06])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params(labelsize=7.2, length=2.2, pad=2)
    cb.set_label("Haufe pattern (a.u.; dashed < 0)", labelpad=3, fontsize=7.6)

    place_labels_aligned(fig, [
        ("a", 0.018, ax_a), ("b", 0.465, ax_b),
        ("c", 0.018, topo_axes[0]), ("d", 0.465, topo_axes[1]),
    ])
    audit_and_export(fig, "fig5_neuro_interpretability", size)


# ==============================================================================
# NEW FIG. 6: Operational Stressors (Dispersion/Drift/12-ch CAR) & Ablations
# ==============================================================================
def build_fig6(m, s4, dds, rra) -> None:
    apply_elsevier_style()
    size = (7.48, 5.15)
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 1, left=0.085, right=0.975, top=0.865, bottom=0.09,
                          height_ratios=[1.12, 1.0], hspace=0.48)

    # Panel (a): Controlled perturbation stress tests (Gain relative to G1, pp)
    ax_a = fig.add_subplot(gs[0])
    clean_axis(ax_a)
    ax_a.axhline(0.0, color=REF_GRAY, ls="--", lw=0.85, zorder=1)
    ax_a.axvline(2.5, color="#DDDDDD", lw=0.8, zorder=0)

    stress_specs = [
        ("2a_4cls__disp_3x", "IV-2a 4-cls\n3× disp. (n=9)"),
        ("2a_4cls__disp_0.25x", "IV-2a 4-cls\n0.25× disp. (n=9)"),
        ("2a_4cls__aniso_ch_drift", "IV-2a 4-cls\nCh. drift (n=9)"),
        ("chang_3cls__disp_3x", "Chang 3-cls\n3× disp. (n=28)"),
        ("chang_3cls__disp_0.25x", "Chang 3-cls\n0.25× disp. (n=28)"),
        ("chang_3cls__aniso_ch_drift", "Chang 3-cls\nCh. drift (n=28)"),
    ]
    stress_methods = [
        ("zCORAL", "z-CORAL (OAS-Src)", COL["zcoral"], "v", -0.22),
        ("RPA", "Causal RPA", COL["rpa"], "s", 0.0),
        ("WA", "Within-Adapt (Ours)", COL["wa"], "D", 0.22),
    ]
    rng = np.random.default_rng(99)
    for i, (pref, _) in enumerate(stress_specs):
        g1 = dds[f"{pref}__G1"] * 100.0
        for mkey, mlbl, col, mk, off in stress_methods:
            vals = dds[f"{pref}__{mkey}"] * 100.0 - g1
            x = i + off
            jit = rng.uniform(-0.045, 0.045, len(vals))
            ax_a.scatter(x + jit, vals, s=5.5 if len(vals) > 20 else 8.0,
                         color=col, marker=mk, alpha=0.35 if len(vals) > 20 else 0.55,
                         edgecolors="none", zorder=2)
            mu = float(vals.mean())
            err = sem(vals)
            ax_a.errorbar(x, mu, yerr=err, fmt=mk, color=INK, mfc=col, mec=INK,
                          mew=0.65, ms=4.5, lw=1.0, capsize=2.0, zorder=4,
                          label=mlbl if i == 0 else None)
            if mkey == "WA":
                p_g1 = float(dds[f"{pref}__vs_G1__p"][0])
                if p_g1 < 0.05:
                    y_star = max(14.5, float(vals.max()) + 1.0)
                    ax_a.text(x, y_star, f"{sig_stars(p_g1)}\n(+{mu:.2f})", ha="center", va="bottom",
                              fontsize=6.8, fontweight="bold", color=COL["wa"], linespacing=0.95)

    ax_a.set(
        xlim=(-0.52, 5.52), ylim=(-11.5, 22.8), yticks=[-10, -5, 0, 5, 10, 15, 20],
        xticks=range(6), xticklabels=[s[1] for s in stress_specs],
        ylabel="Gain over G1 (pp; ± SEM)",
        title="Controlled covariance & channel-drift stress tests (vs. G1 = 0 pp)",
    )
    ax_a.tick_params(axis="x", length=0, labelsize=7.1, pad=3)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.50, 1.27), ncol=3,
                frameon=False, columnspacing=1.2, handletextpad=0.35)

    # Bottom row: (b) 12-ch minimalist CAR vs 59-ch CAR & (c) Internal ablations
    bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.32)
    ax_b = fig.add_subplot(bot[0])
    ax_c = fig.add_subplot(bot[1])
    clean_axis(ax_b)
    clean_axis(ax_c)

    # Panel (b): Chang 3-class 59-ch CAR vs 12-ch CAR
    car_methods = [
        ("Uncal", "Uncal", COL["uncal"], "o"),
        ("G1", "G1", COL["g1"], "s"),
        ("z-CORAL (OAS-Src)", "z-CORAL", COL["zcoral"], "v"),
        ("Within-Adapt", "Within-Adapt", COL["wa"], "D"),
    ]
    for i, (mkey, mlbl, col, mk) in enumerate(car_methods):
        v59 = m[f"chang_3cls__{mkey}"] * 100.0
        v12 = rra[f"car12__chang_3cls__{mkey}"] * 100.0
        for off, arr, mfc, lbl_tag in [(-0.14, v59, "white", "59-ch CAR"), (0.14, v12, col, "12-ch CAR")]:
            mu, err = float(arr.mean()), sem(arr)
            ax_b.errorbar(i + off, mu, yerr=err, fmt=mk, color=col, mfc=mfc, mec=col,
                          mew=1.0, ms=4.8, lw=1.1, capsize=2.2, zorder=4)
        ax_b.plot([i - 0.14, i + 0.14], [v59.mean(), v12.mean()], color=col, lw=0.8, ls="-", zorder=2)

    p_car = exact_wilcoxon_p(rra["car12__chang_3cls__Within-Adapt"], rra["car12__chang_3cls__G1"])
    add_bracket(ax_b, 1.14, 3.14, 54.0, f"{sig_stars(p_car)} (+1.38 pp)", h=0.8)
    ax_b.set(
        xlim=(-0.48, 3.48), ylim=(42.5, 56.5), yticks=[44, 48, 52, 56],
        xticks=range(4), xticklabels=[c[1] for c in car_methods],
        ylabel="Accuracy (% ± SEM)", title="Wearable 12-ch CAR (Chang 3-cls, n=28)",
    )
    ax_b.tick_params(axis="x", length=0, pad=3)
    b_leg = [
        Line2D([], [], color=INK, mfc="white", marker="s", ls="none", ms=4.5, label="59-ch CAR"),
        Line2D([], [], color=INK, mfc=INK, marker="s", ls="none", ms=4.5, label="12-ch only CAR"),
    ]
    ax_b.legend(handles=b_leg, loc="lower right", frameon=True, facecolor="white",
                edgecolor="#DDDDDD", fontsize=6.7, handletextpad=0.35)

    # Panel (c): Band ablation (IV-2a 4-cls) + Re-projection vs Cached tangent
    items = [
        ("HB-only\n(21–30 Hz)", s4["mc_band__2a_4cls__WA__hb_only"] * 100.0, COL["static"], "o"),
        ("Mu+Beta\n(disjoint)", s4["mc_band__2a_4cls__WA__disjoint_mu_beta"] * 100.0, COL["orig"], "^"),
        ("Full-only\n(8–30 Hz)", s4["mc_band__2a_4cls__WA__full_only"] * 100.0, COL["g1"], "s"),
        ("Dual-Nested\n(Ours)", s4["mc_band__2a_4cls__WA__dual_nested"] * 100.0, COL["wa"], "D"),
        ("Cached\ntangent", rra["diag_2a_4cls__cache__WA"] * 100.0, COL["zcoral"], "v"),
        ("Re-proj.\nEq. (5)", rra["diag_2a_4cls__reproj__WA"] * 100.0, COL["wa"], "D"),
    ]
    ax_c.axvline(3.5, color="#DDDDDD", lw=0.8, zorder=0)
    for i, (_, arr, col, mk) in enumerate(items):
        jit = rng.uniform(-0.12, 0.12, len(arr))
        ax_c.scatter(i + jit, arr, s=9.5, color=col, marker=mk, alpha=0.55,
                     edgecolors=INK, linewidths=0.25, zorder=2)
        ax_c.hlines(arr.mean(), i - 0.22, i + 0.22, color=INK, lw=1.4, zorder=4)
        ax_c.text(i, 21.5, f"{arr.mean():.1f}", ha="center", va="center", fontsize=6.8, color=INK)

    add_bracket(ax_c, 0, 3, 79.0, "**", h=1.8)
    add_bracket(ax_c, 1, 3, 85.0, "*", h=1.8)
    add_bracket(ax_c, 4, 5, 79.0, "** (+4.60 pp)", h=1.8)
    ax_c.set(
        xlim=(-0.55, 5.55), ylim=(18.0, 91.5), yticks=[25, 45, 65, 85],
        xticks=range(6), xticklabels=[it[0] for it in items],
        ylabel="Accuracy (%)", title="Ablations: IV-2a 4-cls (n = 9)",
    )
    ax_c.tick_params(axis="x", length=0, labelsize=6.5, pad=2.5)

    place_labels_aligned(fig, [
        ("a", 0.015, ax_a), ("b", 0.015, ax_b), ("c", 0.495, ax_c),
    ])
    audit_and_export(fig, "fig6_stress_and_ablations", size)


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Reproduce manuscript Figures 2--6 from results/*.npz")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures_reproduced", help="Output directory")
    args = parser.parse_args()
    OUT_DIR = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)

    with np.load(RESULTS_DIR / "step2_unified_causal_master.npz") as m, \
         np.load(RESULTS_DIR / "step2_dl_and_latency.npz") as dl, \
         np.load(RESULTS_DIR / "step4_fixes.npz") as s4, \
         np.load(RESULTS_DIR / "p1_audit_updates.npz") as pvals, \
         np.load(RESULTS_DIR / "gamma_ablation.npz") as gamma, \
         np.load(RESULTS_DIR / "dispersion_drift_stress.npz") as dds, \
         np.load(RESULTS_DIR / "reviewer_revision_ablations.npz") as rra:
        build_fig2(m, dl, s4, pvals)
        build_fig3(m, s4, pvals)
        build_fig4(m, dl, pvals)
        build_fig5(s4, gamma)
        build_fig6(m, s4, dds, rra)
    print("\nAll manuscript figures (Figs. 2--6) exported to", OUT_DIR)


if __name__ == "__main__":
    main()
