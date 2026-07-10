"""Core transient/sustained metrics on the CONTINUOUS kernel-width axis.

Continuum re-render of the §2 transient-vs-sustained core-metrics figure
(`transient_vs_sustained.py`). That figure split cells into transient (narrow
kernel) vs sustained (broad kernel) classes and compared their properties with
boxplots. The spectrum result ([[tf_transient_sustained_state_jul2026]] /
docs 2026-07 transient-sustained-spectrum) showed kernel width is a GRADED axis,
not two classes, so here every metric is trended against the continuous width
`interp_fwhm` directly:

  * per-metric panel = decile-binned mean +/- bootstrap CI + scatter + monotone
    trend + Spearman (via `continuum_common.binned_trend`);
  * metrics: TF selectivity (`c1_r_log2`), baseline rate (`base_hz`), then the
    three downstream-coupling OUTCOMES (`change_on`, `hit_ramp`, `fa_ramp`);
  * a width-distribution histogram (median marked).

Stats txt reports, per metric, the global Spearman(rho, p) AND
`segmented_vs_linear` ΔBIC (negative => a straight/graded line beats a two-piece
threshold fit, echoing "graded not stepped"), plus per-region Spearman for the
three outcomes.

Cache-only (kernel_width_continuous.csv + registry c1_r); no session reloads.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import spearmanr

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import (  # noqa: E402
    load_width_metrics, binned_trend, WIDTH, OUTCOMES, REPO, _cmap,
)
from visdetect.analysis.spectrum_stats import segmented_vs_linear  # noqa: E402

OUT = Path(REPO) / "FIGURES/tf_glm_bg046/core_metrics_continuum"
XLABEL = "kernel width interp_fwhm (s)"

# (column, y-axis label, trend color) — order per task: selectivity, rate,
# then the three downstream OUTCOMES.
METRICS = [
    ("c1_r_log2", "TF selectivity (C1 log2)", "#6a51a3"),
    ("base_hz", "baseline rate (Hz)", "#0868ac"),
    ("change_on", "Change_ON response (Hz)", "#238b45"),
    ("hit_ramp", "Hit motor ramp (Hz)", "#d94801"),
    ("fa_ramp", "FA motor ramp (Hz)", "#ce1256"),
]
OUTCOME_COLS = {c for c, _ in OUTCOMES}


def main():
    d = load_width_metrics()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    w = d[WIDTH].to_numpy(float)
    w_med = float(np.nanmedian(w))
    n = int(np.isfinite(w).sum())

    lines = [
        "Core transient/sustained metrics vs CONTINUOUS kernel width (interp_fwhm)",
        f"n cells = {len(d)} (finite width = {n}) | "
        f"DMS={int((d.region=='DMS').sum())} VMS={int((d.region=='VMS').sum())}",
        f"width median = {w_med:.4f} s   "
        f"[IQR {np.nanpercentile(w,25):.4f}-{np.nanpercentile(w,75):.4f}]",
        "",
        "ΔBIC = segmented_vs_linear delta_bic = bic_linear - bic_segmented;",
        "  NEGATIVE => a straight/graded line beats a 2-piece threshold fit",
        "  (echoes the SPECTRUM result: width is graded, not two classes).",
        "",
    ]

    fig = plt.figure(figsize=(18, 10.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.32)
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    for (col, ylab, color), (r, c) in zip(METRICS, positions):
        ax = fig.add_subplot(gs[r, c])
        res = binned_trend(ax, w, d[col].to_numpy(float), color=color)
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(ylab)
        ax.set_title(ylab, fontsize=10.5)
        seg = segmented_vs_linear(w, d[col].to_numpy(float))
        # ΔBIC annotation (below the Spearman that binned_trend already draws)
        ax.text(0.03, 0.80, f"ΔBIC={seg['delta_bic']:+.1f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=8, color="0.35")
        tag = " [OUTCOME]" if col in OUTCOME_COLS else ""
        lines.append(
            f"[{col:10s}] Spearman rho={res['rho']:+.3f} p={res['p']:.2e}"
            f"   segmented ΔBIC={seg['delta_bic']:+7.1f} (bp={seg['breakpoint']:.3f}"
            f", slope {seg['slope_lo']:+.2g}->{seg['slope_hi']:+.2g}){tag}"
        )

    # Panel F: width distribution (median marked), bars tinted by the width gradient
    axh = fig.add_subplot(gs[1, 2])
    cmap = _cmap()
    wf = w[np.isfinite(w)]
    counts, edges = np.histogram(wf, bins=np.linspace(0, np.nanpercentile(wf, 99.5), 30))
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm = (centers - centers.min()) / (centers.max() - centers.min() + 1e-12)
    axh.bar(centers, counts, width=np.diff(edges), color=cmap(norm),
            edgecolor="white", linewidth=0.3, align="center")
    axh.axvline(w_med, color="k", ls="--", lw=1.6)
    axh.text(w_med, axh.get_ylim()[1] * 0.96, f" median\n {w_med:.3f}s",
             ha="left", va="top", fontsize=8)
    axh.set_xlabel(XLABEL)
    axh.set_ylabel("cell count")
    axh.set_title("width distribution (graded, no gap)", fontsize=10.5)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)

    # per-region Spearman for the three OUTCOMES
    lines.append("")
    lines.append("per-region Spearman (width vs outcome):")
    for col, lab in OUTCOMES:
        parts = []
        for reg in ("DMS", "VMS"):
            sub = d[d.region == reg]
            xx = sub[WIDTH].to_numpy(float)
            yy = sub[col].to_numpy(float)
            mm = np.isfinite(xx) & np.isfinite(yy)
            if mm.sum() > 2:
                rho, p = spearmanr(xx[mm], yy[mm])
                parts.append(f"{reg} rho={rho:+.3f} p={p:.2e} (n={int(mm.sum())})")
            else:
                parts.append(f"{reg} n/a")
        lines.append(f"  [{col:10s}] " + " | ".join(parts))

    fig.suptitle(
        "Core transient/sustained metrics on the continuous width axis "
        "(binned deciles + trend)",
        fontsize=13, y=1.005,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"core_metrics_continuum.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)

    (OUT / "core_metrics_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/core_metrics_continuum.png (+.pdf, +_stats.txt)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
