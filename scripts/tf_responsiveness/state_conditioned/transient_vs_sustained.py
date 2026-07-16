"""Functional differences between TRANSIENT (fast/sensory) and SUSTAINED
TF-responsive cells.

The latency test ([[tf_kernel_latency_outcome_coupling_jul2026]]) used kernel
PEAK LATENCY. This asks the distinct 'fast/sensory vs sustained' question using
kernel DURATION = `kernel_fwhm` (full-width-at-half-max of the GLM TF kernel):
  transient = narrow kernel (fwhm <= 0.05 s = one 50 ms bin);
  sustained = broad kernel  (fwhm >= 0.15 s = 3+ bins).
Literature prior (synthesis-phase3-celltypes): fast/transient, tightly stimulus-
locked responses = FSI-like; sustained/integrating = SPN-like.

Questions:
  1. Is width a continuum or two classes? (floor-dominated → describe fractions)
  2. Is peak LATENCY independent of WIDTH? (Spearman) — are 'early' and 'transient'
     the same subset, or separate axes?
  3. Do transient vs sustained cells differ in: TF selectivity (c1_r), firing rate
     (base_hz), REGION (DMS/VMS), and downstream coupling (change / motor ramps)?

Registry + cached per-cell metrics only (no session reloads). Non-parametric
(Mann-Whitney U / Spearman). Population = responsive cells in QC-pass, pre-
breakdown sessions (good_dates), same as the rep-cells / latency figures.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import spearmanr, mannwhitneyu

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO, _registry, good_dates  # noqa: E402

MICE = [("BG_046", "DMS", "#2c7fb8"), ("BG_039", "DMS", "#41ab5d"),
        ("BG_031", "VMS", "#ef6548")]
NARROW = 0.05     # <= one 50 ms bin  → transient
BROAD = 0.15      # >= 3 bins         → sustained
TCOL, SCOL = "#e6550d", "#3182bd"   # transient / sustained
OUT = Path(str(REPO)) / "FIGURES/tf_glm_bg046/transient_vs_sustained"
METRICS_CACHE = Path(str(REPO)) / ("FIGURES/tf_glm_bg046/"
                     "latency_outcome_coupling/latency_outcome_metrics.csv")
OUTCOMES = [("change_on", "Change_ON resp"), ("hit_ramp", "Hit ramp"), ("fa_ramp", "FA ramp")]


def load_cells():
    frames = []
    for subj, region, _ in MICE:
        r = _registry(subj)
        r = r[r.resp & r.session_date.isin(good_dates(subj))].copy()
        r["subject"] = subj
        r["region"] = region
        frames.append(r[["subject", "region", "session", "unit", "kernel_peak_t",
                         "kernel_fwhm", "c1_r_log2", "n_spikes"]])
    cells = pd.concat(frames, ignore_index=True)
    if METRICS_CACHE.exists():
        m = pd.read_csv(METRICS_CACHE)[["subject", "session", "unit", "base_hz",
                                        "change_on", "hit_ramp", "fa_ramp"]]
        cells = cells.merge(m, on=["subject", "session", "unit"], how="left")
    cells["class"] = np.where(cells.kernel_fwhm <= NARROW, "transient",
                              np.where(cells.kernel_fwhm >= BROAD, "sustained", "intermediate"))
    return cells


def _mwu(a, b):
    a = pd.Series(a).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(b).replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan
    u, p = mannwhitneyu(a, b)
    return float(a.median()), float(b.median()), float(p)


def main():
    cells = load_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    n = len(cells)
    ntr = (cells["class"] == "transient").sum()
    nsu = (cells["class"] == "sustained").sum()
    lines = [f"n responsive (good sessions) = {n}",
             f"transient (fwhm<={NARROW}) = {ntr} ({100*ntr/n:.0f}%) | "
             f"intermediate = {(cells['class']=='intermediate').sum()} | "
             f"sustained (fwhm>={BROAD}) = {nsu} ({100*nsu/n:.0f}%)"]
    # peak latency independent of width?
    rho_pw, p_pw = spearmanr(cells.kernel_peak_t, cells.kernel_fwhm)
    lines.append(f"peak_t vs fwhm: Spearman rho={rho_pw:+.3f} p={p_pw:.2e} "
                 "(are 'early' and 'transient' the same axis?)")

    fig = plt.figure(figsize=(18, 10.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.32)

    # A: fwhm distribution by region
    axa = fig.add_subplot(gs[0, 0])
    bins = np.arange(0, 0.7, 0.05)
    for subj, region, c in MICE:
        axa.hist(cells.loc[cells.subject == subj, "kernel_fwhm"], bins=bins, histtype="step",
                 lw=2, color=c, label=f"{subj} ({region})", density=True)
    axa.axvline(NARROW, color=TCOL, ls="--", lw=1.2); axa.axvline(BROAD, color=SCOL, ls="--", lw=1.2)
    axa.set_xlabel("kernel FWHM (s)"); axa.set_ylabel("density")
    axa.set_title("TF-kernel DURATION is floor-dominated\n(most cells transient)", fontsize=10.5)
    axa.legend(frameon=False, fontsize=8)

    # B: peak_t vs fwhm (are the two axes independent?)
    axb = fig.add_subplot(gs[0, 1])
    for subj, region, c in MICE:
        d = cells[cells.subject == subj]
        axb.scatter(d.kernel_peak_t, d.kernel_fwhm + np.random.default_rng(1).normal(0, 0.008, len(d)),
                    s=14, color=c, alpha=0.5, edgecolors="none")
    axb.axhline(NARROW, color=TCOL, ls="--", lw=1); axb.axhline(BROAD, color=SCOL, ls="--", lw=1)
    axb.set_xlabel("kernel peak latency (s)"); axb.set_ylabel("kernel FWHM (s) [jittered]")
    axb.set_title(f"latency vs width  ρ={rho_pw:+.2f} (p={p_pw:.0e})", fontsize=10.5)

    # C: transient vs sustained — selectivity & firing rate
    axc = fig.add_subplot(gs[0, 2])
    props = [("c1_r_log2", "TF selectivity (C1)"), ("base_hz", "baseline rate (Hz)")]
    for pi, (col, lab) in enumerate(props):
        for si, cls in enumerate(("transient", "sustained")):
            v = cells.loc[cells["class"] == cls, col].replace([np.inf, -np.inf], np.nan).dropna()
            if not len(v):
                continue
            # z-score within property so both fit one axis
            allv = cells[col].replace([np.inf, -np.inf], np.nan).dropna()
            z = (v - allv.mean()) / (allv.std() + 1e-9)
            xc = pi + (si - 0.5) * 0.4
            jit = (np.random.default_rng(si).random(len(z)) - 0.5) * 0.16
            axc.scatter(np.full(len(z), xc) + jit, z, s=8, alpha=0.35,
                        color=(TCOL if cls == "transient" else SCOL), edgecolors="none")
            axc.hlines(np.median(z), xc - 0.18, xc + 0.18, color="k", lw=2)
        mt, ms, pp = _mwu(cells.loc[cells["class"] == "transient", col],
                          cells.loc[cells["class"] == "sustained", col])
        axc.text(pi, 3.0, f"p={pp:.1e}", ha="center", fontsize=8)
        lines.append(f"[{col}] transient med={mt:.3g} vs sustained med={ms:.3g}  MWU p={pp:.2e}")
    axc.set_xticks([0, 1]); axc.set_xticklabels([p[1] for p in props], fontsize=9)
    axc.set_ylabel("z (within property)"); axc.set_ylim(-3.3, 3.6)
    axc.axhline(0, color="0.7", lw=0.8, ls=":")
    axc.set_title("transient vs sustained: cell properties", fontsize=10.5)
    from matplotlib.lines import Line2D
    axc.legend(handles=[Line2D([0], [0], marker="o", ls="", color=TCOL, label="transient"),
                        Line2D([0], [0], marker="o", ls="", color=SCOL, label="sustained")],
               frameon=False, fontsize=8, loc="lower right")

    # D: region composition of transient vs sustained
    axd = fig.add_subplot(gs[1, 0])
    ct = pd.crosstab(cells["class"], cells["region"], normalize="index").reindex(
        ["transient", "intermediate", "sustained"])
    bottom = np.zeros(len(ct))
    for reg, c in (("DMS", "#3474ae"), ("VMS", "#ef6548")):
        if reg in ct:
            axd.bar(ct.index, ct[reg], bottom=bottom, color=c, label=reg)
            bottom += ct[reg].values
    axd.set_ylabel("fraction"); axd.set_title("region composition by class", fontsize=10.5)
    axd.legend(frameon=False, fontsize=9)
    axd.tick_params(axis="x", labelrotation=15)
    # chi-square transient vs sustained x region
    from scipy.stats import chi2_contingency
    sub = cells[cells["class"].isin(["transient", "sustained"])]
    tab = pd.crosstab(sub["class"], sub["region"])
    try:
        chi2, pchi, *_ = chi2_contingency(tab)
        lines.append(f"[region x class] chi2={chi2:.1f} p={pchi:.2e}  (DMS/VMS enrichment)")
    except Exception:
        pchi = np.nan
    axd.text(0.5, 0.02, f"χ² p={pchi:.1e}", transform=axd.transAxes, ha="center", fontsize=8)

    # E: transient vs sustained — outcome coupling
    axe = fig.add_subplot(gs[1, 1])
    if "change_on" in cells:
        for oi, (col, lab) in enumerate(OUTCOMES):
            for si, cls in enumerate(("transient", "sustained")):
                v = cells.loc[cells["class"] == cls, col].replace([np.inf, -np.inf], np.nan).dropna()
                xc = oi + (si - 0.5) * 0.4
                jit = (np.random.default_rng(si + 3).random(len(v)) - 0.5) * 0.16
                axe.scatter(np.full(len(v), xc) + jit, v, s=8, alpha=0.3,
                            color=(TCOL if cls == "transient" else SCOL), edgecolors="none")
                axe.hlines(np.median(v), xc - 0.18, xc + 0.18, color="k", lw=2)
            mt, ms, pp = _mwu(cells.loc[cells["class"] == "transient", col],
                              cells.loc[cells["class"] == "sustained", col])
            axe.text(oi, axe.get_ylim()[1] * 0.9 if False else 30, f"p={pp:.1e}", ha="center", fontsize=8)
            lines.append(f"[{col}] transient med={mt:.3g} vs sustained med={ms:.3g}  MWU p={pp:.2e}")
        axe.axhline(0, color="0.7", lw=0.8, ls=":")
        axe.set_xticks(range(len(OUTCOMES))); axe.set_xticklabels([o[1] for o in OUTCOMES], fontsize=9)
        axe.set_ylabel("Δ firing (Hz)"); axe.set_ylim(-12, 34)
        axe.set_title("transient vs sustained: outcome coupling", fontsize=10.5)

    # F: stats text
    axf = fig.add_subplot(gs[1, 2]); axf.axis("off")
    axf.text(0.0, 1.0, "\n".join(lines), transform=axf.transAxes, va="top", ha="left",
             fontsize=8.2, family="monospace")

    for ax in (axb, axc, axd, axe):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Transient (fast/sensory) vs sustained TF-responsive cells — functional comparison\n"
                 "kernel-width classes: is 'fast vs sustained' a real functional split?",
                 fontsize=13, y=1.005)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"transient_vs_sustained.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)
    (OUT / "transient_vs_sustained_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/transient_vs_sustained.png (+.pdf)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
