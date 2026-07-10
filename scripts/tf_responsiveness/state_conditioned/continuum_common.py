# scripts/tf_responsiveness/state_conditioned/continuum_common.py
"""Shared helpers for the continuum re-renders of the transient/sustained figures:
a width+metrics loader, the decile-binned-trend panel, and a width-bin family for
PSTHs. Pure functions are unit-tested; plotting wrappers are thin.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO, _registry, good_dates   # noqa: E402

REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}
MICE = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
WIDTH = "interp_fwhm"
OUTCOMES = [("change_on", "Change_ON response"), ("hit_ramp", "Hit motor ramp"),
            ("fa_ramp", "FA motor ramp")]
CACHE = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"


def _cmap():
    # matplotlib.cm.get_cmap was deprecated in 3.7 and is removed in 3.11; use
    # the colormaps registry so this keeps working on the installed 3.10.x.
    import matplotlib
    return matplotlib.colormaps["viridis"]


WIDTH_CMAP = None  # lazily set in plotting code via _cmap() to avoid import at load


def load_width_metrics() -> pd.DataFrame:
    """One row per responsive cell: continuous width + coupling metrics (from
    kernel_width_continuous.csv) joined to registry TF selectivity c1_r_log2."""
    d = pd.read_csv(CACHE, dtype={"session": str})
    # registry c1_r_log2 keyed by (subject, session, unit)
    regs = []
    for subj, _ in MICE:
        r = _registry(subj)[["session", "unit", "c1_r_log2"]].copy()
        r["subject"] = subj
        regs.append(r)
    reg = pd.concat(regs, ignore_index=True)
    reg["unit"] = reg["unit"].astype(int)
    d["unit"] = d["unit"].astype(int)
    d = d.merge(reg, on=["subject", "session", "unit"], how="left")
    d["region"] = d["subject"].map(REGION)
    return d


def decile_stats(x, y, n_bins=10, n_boot=1000, seed=42) -> dict:
    """Equal-count width bins; per-bin mean of y + bootstrap CI; global Spearman."""
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    x, y = x[order], y[order]
    rng = np.random.default_rng(seed)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    centers, mean, lo, hi, npb = [], [], [], [], []
    for b in range(n_bins):
        sel = (x >= edges[b]) & (x < edges[b + 1])
        yb = y[sel]
        if yb.size == 0:
            continue
        centers.append(float(np.median(x[sel])))
        mean.append(float(np.mean(yb)))
        boots = np.array([np.mean(rng.choice(yb, yb.size)) for _ in range(n_boot)])
        lo.append(float(np.percentile(boots, 2.5)))
        hi.append(float(np.percentile(boots, 97.5)))
        npb.append(int(yb.size))
    rho, p = spearmanr(x, y) if x.size > 2 else (np.nan, np.nan)
    return {"centers": np.array(centers), "mean": np.array(mean),
            "ci_lo": np.array(lo), "ci_hi": np.array(hi),
            "rho": float(rho), "p": float(p), "n_per_bin": np.array(npb)}


def width_bin_assign(width, n=5):
    """Assign each cell to one of n equal-count width bins (0..n-1) + return edges."""
    width = np.asarray(width, float)
    finite = width[np.isfinite(width)]
    edges = np.quantile(finite, np.linspace(0, 1, n + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, width, side="right") - 1, 0, n - 1)
    idx = np.where(np.isfinite(width), idx, -1)
    return idx.astype(int), edges


def binned_trend(ax, x, y, *, n_bins=10, color="#238b45", scatter=True, label=None):
    """Scatter + decile mean±bootstrap-CI + monotonic trend + Spearman annotation."""
    d = decile_stats(x, y, n_bins=n_bins)
    x = np.asarray(x, float); y = np.asarray(y, float)
    if scatter:
        ax.scatter(x, y, s=6, alpha=0.18, color="0.5", edgecolors="none", zorder=1)
    ax.fill_between(d["centers"], d["ci_lo"], d["ci_hi"], color=color, alpha=0.25, zorder=2)
    ax.plot(d["centers"], d["mean"], "o-", color=color, lw=2, ms=5,
            label=label, zorder=3)
    ax.text(0.03, 0.95, f"ρ={d['rho']:+.2f}\np={d['p']:.1e}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return d
