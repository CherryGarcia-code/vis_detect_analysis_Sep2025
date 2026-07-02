"""DEMO (methods illustration, NOT a talk figure): how the BASELINE (z-reference) WINDOW
choice changes the Baseline_ON up/down sign-split.

Reads the existing BG_046 event cache (event_psth_cache_BG_046.npz), whose Baseline_ON
per-unit z-traces are referenced to the canonical (-1.75,-1.25) s window, and RE-REFERENCES
each unit's trace to several candidate baseline windows, side by side.

WHY this is exact (no re-loading sessions): the cached z is an affine transform of the raw
rate, z = (r - mu_canon)/sigma_canon. Re-CENTERING that trace to a new window W (subtracting
its mean over W) is identical to centering the raw rate on W -- so the pre-onset divergence
shown here is the true effect of the reference-MEAN choice. (Full re-z-scoring would ALSO
divide by each unit's SD within W; that rescales amplitude/height but does not change the
qualitative pre-onset behaviour. We hold the SD at the canonical value so the panels share a
y-scale and isolate the centering effect.)

Same units, same held-out sign rule (sign from ODD trials in 0-1 s; EVEN half plotted) in
every panel -- only the reference window differs.

Usage: py scripts/talk_substrate/demo_baseline_window.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402
from visdetect.analysis.utils import bootstrap_ci  # noqa: E402

setup_style()
EVENT = "Baseline_ON"
SIGN_WIN = (0.0, 1.0)        # post-onset window that DEFINES up/down (held-out, odd trials)
UP, DOWN = "#d73027", "#4575b4"
WINDOWS = [
    ("(-1.75, -1.25) s\ncanonical (far ITI ref)", (-1.75, -1.25)),
    ("(-1.0, -0.3) s\nlate-ITI ref", (-1.0, -0.3)),
    ("(-0.3, -0.05) s\nimmediately pre-onset", (-0.3, -0.05)),
]


def wmask(bc, w):
    return (bc >= w[0]) & (bc <= w[1])


def recenter(traces, bc, w):
    """Subtract each unit's mean over window w (exact re-referencing of the cached z-trace)."""
    base = np.nanmean(traces[:, wmask(bc, w)], axis=1, keepdims=True)
    return traces - base


def main():
    cache = E.load_event_cache("BG_046")
    bc = E.bc(cache, EVENT)
    odd = E.mat(cache, EVENT, "all", "odd")     # (n_units, n_bins), z to canonical window, smoothed
    even = E.mat(cache, EVENT, "all", "even")
    sgnmask = wmask(bc, SIGN_WIN)

    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5.2 * len(WINDOWS), 4.7), sharey=True)
    for i, (lab, W) in enumerate(WINDOWS):
        ax = axes[i]
        odd_rc = recenter(odd, bc, W)
        even_rc = recenter(even, bc, W)
        sgn = np.nanmean(odd_rc[:, sgnmask], axis=1)                 # held-out sign under window W
        finite = np.isfinite(even_rc).all(1) & np.isfinite(sgn)
        for grp, col, name in [("up", UP, "Up"), ("down", DOWN, "Down")]:
            m = finite & ((sgn > 0) if grp == "up" else (sgn < 0))
            M = even_rc[m]
            nU = M.shape[0]
            if nU == 0:
                continue
            mean = M.mean(0)
            lo, hi = bootstrap_ci(M, n_bootstrap=1000, ci_level=0.95, axis=0, seed=42)
            ax.plot(bc, mean, color=col, lw=1.9, label=f"{name}-modulated (n={nU})")
            ax.fill_between(bc, lo, hi, color=col, alpha=0.2)
        ax.axvspan(W[0], W[1], color="0.80", alpha=0.7, zorder=0, label="z-baseline window")
        ax.axvspan(SIGN_WIN[0], SIGN_WIN[1], color="#ffe08a", alpha=0.30, zorder=0)
        ax.axvline(0, color="k", lw=1.0)
        ax.axhline(0, color="0.6", lw=0.7, ls=":")
        ax.set_title(lab, fontsize=9.5)
        ax.set_xlabel("time from baseline onset (s)")
        if i == 0:
            ax.set_ylabel("z (canonical baseline SD)\nre-referenced mean")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("BG_046 Baseline_ON up/down split — effect of the z-baseline (reference) WINDOW choice",
                 fontsize=12, y=1.03)
    fig.text(0.5, -0.08,
             "Same units, same held-out sign rule (sign from ODD trials in the yellow 0-1 s window; EVEN "
             "half plotted). Grey band = the reference window each panel is centred on. As the reference "
             "moves toward t=0 the pre-onset divergence collapses and the split anchors at onset — i.e. the "
             "'split before onset' in the canonical panel is the far reference window, not a pre-onset neural "
             "signal. (SD held at canonical so panels share a y-scale; full re-z-scoring would also rescale "
             "height.) Bands = bootstrap 95% CI across units.",
             ha="center", fontsize=8, color="#555555", wrap=True)
    out = C.FIG_DIR / "demo_baseline_window.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[demo] wrote {out}")


if __name__ == "__main__":
    main()
