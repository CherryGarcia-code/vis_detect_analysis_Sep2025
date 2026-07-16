"""Consolidation: the kernel-WIDTH -> outcome-coupling effect is NOT a threshold
artifact. kernel_fwhm is a coarse 50ms-grid index (~60% at the floor), so show:
  (top) the CONTINUOUS monotonic relationship — Spearman(kernel_fwhm, outcome),
        pooled + per region (DMS/VMS), for Change_ON / Hit ramp / FA ramp;
  (bottom) threshold ROBUSTNESS — the sustained-minus-transient gap is stable
        across split definitions (current, median-split, exclude-floor, strict).
Reads load_cells() (registry + metrics cache) — no session loading.
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
from transient_vs_sustained import load_cells, NARROW, BROAD                  # noqa: E402
from representative_cells import REPO                                         # noqa: E402

OUTCOMES = [("change_on", "Change_ON response"), ("hit_ramp", "Hit motor ramp"),
            ("fa_ramp", "FA motor ramp")]
REGCOL = {"DMS": "#3474ae", "VMS": "#ef6548"}
OUT = Path(str(REPO)) / "FIGURES/tf_glm_bg046/robustness_width_coupling"


def _clean(df, col):
    return df[["kernel_fwhm", col, "region", "subject"]].replace([np.inf, -np.inf], np.nan).dropna()


def _mwu(a, b):
    a = pd.Series(a).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(b).replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan, len(a), len(b), np.nan
    return float(b.median() - a.median()), len(a), len(b), float(mannwhitneyu(a, b).pvalue)


SPLITS = [
    ("current\n(<=0.05 / >=0.15)", lambda f: ("transient" if f <= NARROW else ("sustained" if f >= BROAD else None))),
    ("median split", None),  # filled at runtime
    ("exclude floor\n(0<..<=0.05 / >=0.15)", lambda f: ("transient" if 0 < f <= NARROW else ("sustained" if f >= BROAD else None))),
    ("strict\n(==0 / >=0.20)", lambda f: ("transient" if f == 0 else ("sustained" if f >= 0.20 else None))),
]


def main():
    df = load_cells()
    df = df.replace([np.inf, -np.inf], np.nan)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})
    lines = []
    med = df["kernel_fwhm"].median()
    SPLITS[1] = (f"median split\n(< {med:.2f} / >= {med:.2f})",
                 lambda f: "transient" if f < med else "sustained")

    fig = plt.figure(figsize=(17, 9.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.28, height_ratios=[1, 0.9])
    rng = np.random.default_rng(0)

    # ── top: continuous monotonic relationship ─────────────────────────
    for j, (col, lab) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[0, j])
        d = _clean(df, col)
        x = d.kernel_fwhm.values + rng.normal(0, 0.008, len(d))   # jitter the discrete grid
        for reg in ("DMS", "VMS"):
            m = d.region == reg
            ax.scatter(x[m], d[col][m], s=10, alpha=0.35, color=REGCOL[reg], edgecolors="none")
        # binned median trend
        edges = np.arange(0, 0.5, 0.05)
        cx, cy = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (d.kernel_fwhm >= lo) & (d.kernel_fwhm < hi)
            if sel.sum() >= 8:
                cx.append((lo + hi) / 2); cy.append(d[col][sel].median())
        ax.plot(cx, cy, "-o", color="k", lw=2, ms=5, zorder=5)
        rho, p = spearmanr(d.kernel_fwhm, d[col])
        rd = spearmanr(d[d.region == "DMS"].kernel_fwhm, d[d.region == "DMS"][col])[0]
        rv = spearmanr(d[d.region == "VMS"].kernel_fwhm, d[d.region == "VMS"][col])[0]
        ax.axvline(NARROW, color="0.6", ls="--", lw=0.8); ax.axvline(BROAD, color="0.6", ls="--", lw=0.8)
        ax.set_title(f"{lab}\nSpearman ρ={rho:+.2f} (p={p:.0e})  ·  DMS {rd:+.2f} / VMS {rv:+.2f}",
                     fontsize=11)
        ax.set_xlabel("GLM TF-kernel FWHM (s)", fontsize=12)
        if j == 0:
            ax.set_ylabel("Δ firing (Hz)", fontsize=12)
        ax.set_xlim(-0.03, 0.48)
        ax.axhline(0, color="0.8", lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lines.append(f"[{col}] continuous Spearman(fwhm) pooled ρ={rho:+.3f} p={p:.2e} | DMS {rd:+.3f} VMS {rv:+.3f}")

    # ── bottom: threshold robustness (gap stable across definitions) ────
    for j, (col, lab) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[1, j])
        names, gaps, ps = [], [], []
        for nm, fn in SPLITS:
            cl = df["kernel_fwhm"].map(fn)
            tr = df.loc[cl == "transient", col]; su = df.loc[cl == "sustained", col]
            gap, nt, ns, p = _mwu(tr, su)
            names.append(nm); gaps.append(gap); ps.append(p)
            lines.append(f"   [{col}] split '{nm.replace(chr(10),' ')}': gap(sus-tra)={gap:+.2f}Hz "
                         f"nt={nt} ns={ns} MWU p={p:.2e}")
        xs = np.arange(len(names))
        bars = ax.bar(xs, gaps, color="#5aa469", alpha=0.9)
        for xi, g, p in zip(xs, gaps, ps):
            star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))
            ax.text(xi, g + 0.05, f"{g:+.1f}\n{star}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7.5)
        ax.axhline(0, color="0.6", lw=0.8)
        if j == 0:
            ax.set_ylabel("sustained − transient\nΔ firing gap (Hz)", fontsize=11)
        ax.set_title(f"{lab}: gap stable across split defs", fontsize=10.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("The kernel-WIDTH → outcome-coupling effect is monotonic and split-definition-robust\n"
                 "(continuous Spearman, per region; sustained−transient gap stable across 4 threshold definitions) — "
                 "not a coarse-threshold artifact",
                 fontsize=13, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"robustness_width_coupling.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (OUT / "robustness_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/robustness_width_coupling.png")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
