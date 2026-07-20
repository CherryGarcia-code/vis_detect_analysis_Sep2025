"""Fig5e: fraction of significantly active units above baseline vs time from lick,
grouped by CELL CLASS {transient, sustained, non-TF}.

Faithful within-striatum port of Khilkevich & Lohse (Nature 2024) Fig 5e. At each
time bin the fraction of units whose |z of the mean lick-PETH| > 2.576 (P<0.01),
minus the pre-lick baseline fraction measured in [-2, -1.8] s, with a
bootstrap-OVER-NEURONS 95% CI (5,000x). The paper's brain-area grouping is
replaced by the project's transient->sustained width axis + the non-TF
(TF-non-responsive) reference population. PER-REGION ALWAYS: pooled + DMS + VMS.

Cache-only: reads data/cache/preparatory_fig5/prep_<lick>.npz built by
build_prep_cache.py. No session reload.

Usage:  py fig5e_fraction_active.py [--lick hit|fa]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import prep_common as C  # noqa: E402
from visdetect.analysis.preparatory import (  # noqa: E402
    active_mask, bootstrap_fraction_ci, population_onset)

FIGROOT = C.REPO / "FIGURES/preparatory_fig5"
REGIONS = [("pooled", None), ("DMS", "DMS"), ("VMS", "VMS")]
GROUPS = ["transient", "sustained", "non-TF"]  # the 3 panel-e lines
N_BOOT = 5000


def _group_mask(cls: np.ndarray, resp: np.ndarray, group: str) -> np.ndarray:
    """Boolean row selector for a panel-e group. non-TF == TF-non-responsive."""
    if group == "non-TF":
        return ~resp
    return cls == group


def main(lick: str = "hit") -> None:
    path = C.REPO / f"data/cache/preparatory_fig5/prep_{lick}.npz"
    if not path.exists():
        raise SystemExit(f"cache missing: {path} — run build_prep_cache.py --lick {lick}")

    D = np.load(path, allow_pickle=True)
    t = np.asarray(D["t"], float)
    z = np.asarray(D["z"], float)
    cls = D["cls"].astype(str)
    resp = np.asarray(D["resp"], bool)
    region = D["region"].astype(str)

    A = active_mask(z)                                        # |z|>2.576, nUnits x nBins
    base_mask = (t >= C.BASE_FRAC_WIN[0]) & (t <= C.BASE_FRAC_WIN[1])
    lick_lbl = lick.upper()

    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass

    for rname, rval in REGIONS:
        rmask = np.ones(len(cls), bool) if rval is None else (region == rval)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        rows = []
        printed = []
        for group in GROUPS:
            sel = rmask & _group_mask(cls, resp, group)
            n = int(sel.sum())
            color = C.CLASS_COLORS[group]
            if n == 0:
                rows.append({"region": rname, "group": group, "n_units": 0,
                             "onset_s": np.nan, "peak_frac": np.nan, "t_peak": np.nan})
                printed.append(f"{group}:n=0")
                continue
            mean, lo, hi = bootstrap_fraction_ci(A[sel], baseline_bins=base_mask, n=N_BOOT)
            ax.fill_between(t, lo, hi, color=color, alpha=0.20, lw=0)
            ax.plot(t, mean, color=color, lw=2.4, label=f"{group} (n={n})")
            onset = population_onset(t, mean, lo)
            if np.any(np.isfinite(mean)):
                pk = int(np.nanargmax(mean))
                peak_frac, t_peak = float(mean[pk]), float(t[pk])
            else:
                peak_frac, t_peak = np.nan, np.nan
            rows.append({"region": rname, "group": group, "n_units": n,
                         "onset_s": onset, "peak_frac": peak_frac, "t_peak": t_peak})
            printed.append(f"{group}:n={n} onset={onset if np.isnan(onset) else round(onset,3)}")

        ax.axvline(0, color="k", lw=1.0, ls="--")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_xlabel(f"time from {lick_lbl} lick (s)")
        ax.set_ylabel("fraction active above baseline")
        ax.set_title(f"Fig5e  fraction of active units — {rname} ({lick_lbl})",
                     fontsize=15, loc="left")
        ax.legend(frameon=False, loc="upper left", fontsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        outdir = FIGROOT / rname
        outdir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"fig5e_{lick}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(rows).to_csv(outdir / f"fig5e_{lick}_stats.csv", index=False)
        print(f"[{rname}] " + " | ".join(printed), flush=True)

    print(f"wrote {FIGROOT}/{{pooled,DMS,VMS}}/fig5e_{lick}.{{png,pdf}} (+_stats.csv)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lick", choices=["hit", "fa"], default="hit")
    a = ap.parse_args()
    main(lick=a.lick)
