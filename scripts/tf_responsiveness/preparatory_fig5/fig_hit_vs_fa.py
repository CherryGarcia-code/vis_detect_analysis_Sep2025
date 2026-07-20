"""Hit vs FA comparison: fraction of active units above baseline vs time from lick,
overlaying the HIT (decision) lick and the FA (impulsive early) lick on the same
axes, per cell class {transient, sustained, non-TF} and per region.

Shows whether a class is recruited earlier/later or with different dynamics
depending on the OUTCOME (reported detection vs impulsive lick), and whether that
differs by region (DMS vs VMS). Grid: rows = class, cols = region; each cell has
HIT (solid) and FA (dashed) fraction-active + bootstrap-over-neurons 95% CI, with
per-lick onset markers.

Cache-only: reads prep_hit.npz + prep_fa.npz. No session reload.
Usage:  py fig_hit_vs_fa.py
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
from matplotlib.lines import Line2D

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import prep_common as C  # noqa: E402
from visdetect.analysis.preparatory import active_mask, bootstrap_fraction_ci, population_onset  # noqa: E402

FIGROOT = C.REPO / "FIGURES/preparatory_fig5/hit_vs_fa"
REGIONS = [("pooled", None), ("DMS", "DMS"), ("VMS", "VMS")]
CLASSES = ["transient", "sustained", "non-TF"]
N_BOOT = 2000
LICKS = [("hit", "-", "HIT (decision)"), ("fa", "--", "FA (impulsive)")]


def _load(lick):
    D = np.load(C.REPO / f"data/cache/preparatory_fig5/prep_{lick}.npz", allow_pickle=True)
    return {"t": np.asarray(D["t"], float), "A": active_mask(np.asarray(D["z"], float)),
            "cls": D["cls"].astype(str), "resp": np.asarray(D["resp"], bool),
            "region": D["region"].astype(str)}


def _mask(d, region, group):
    rm = np.ones(len(d["cls"]), bool) if region is None else (d["region"] == region)
    gm = (~d["resp"]) if group == "non-TF" else (d["cls"] == group)
    return rm & gm


def main():
    data = {lk: _load(lk) for lk, _, _ in LICKS}
    t = data["hit"]["t"]
    base_mask = (t >= C.BASE_FRAC_WIN[0]) & (t <= C.BASE_FRAC_WIN[1])
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    FIGROOT.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16.5, 13.0))
    gs = gridspec.GridSpec(len(CLASSES), len(REGIONS), hspace=0.30, wspace=0.20)
    rows = []
    for ri, group in enumerate(CLASSES):
        color = C.CLASS_COLORS[group]
        for ci, (rname, rval) in enumerate(REGIONS):
            ax = fig.add_subplot(gs[ri, ci])
            ax.axvspan(C.BASE_FRAC_WIN[0], C.BASE_FRAC_WIN[1], color="0.9", zorder=0)
            for lk, ls, _lab in LICKS:
                d = data[lk]
                sel = _mask(d, rval, group)
                n = int(sel.sum())
                if n == 0:
                    continue
                mean, lo, hi = bootstrap_fraction_ci(d["A"][sel], baseline_bins=base_mask, n=N_BOOT)
                onset = population_onset(t, mean, lo)
                ax.fill_between(t, lo, hi, color=color, alpha=0.12, lw=0)
                ax.plot(t, mean, color=color, lw=2.2, ls=ls)
                if np.isfinite(onset):
                    ax.plot([onset], [0.02], marker="^" if ls == "-" else "v", color=color,
                            ms=9, mec="k", mew=0.5, clip_on=False, zorder=6)
                rows.append({"region": rname, "group": group, "lick": lk, "n_units": n,
                             "onset_s": onset, "peak_frac": float(np.nanmax(mean)),
                             "t_peak": float(t[np.nanargmax(mean)])})
            ax.axvline(0, color="k", lw=0.9, ls=":")
            ax.axhline(0, color="0.85", lw=0.8)
            ax.set_xlim(float(t[0]), float(t[-1]))
            ax.set_ylim(-0.05, 1.0)
            if ri == 0:
                ax.set_title(rname, fontsize=15, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{group}\nfraction active", fontsize=12.5)
            if ri == len(CLASSES) - 1:
                ax.set_xlabel("time from lick (s)")
            # onset annotation
            hh = [r for r in rows if r["region"] == rname and r["group"] == group and r["lick"] == "hit"]
            ff = [r for r in rows if r["region"] == rname and r["group"] == group and r["lick"] == "fa"]
            txt = ""
            if hh and np.isfinite(hh[0]["onset_s"]):
                txt += f"HIT onset {hh[0]['onset_s']:+.2f}s\n"
            if ff and np.isfinite(ff[0]["onset_s"]):
                txt += f"FA  onset {ff[0]['onset_s']:+.2f}s"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                    fontsize=9.5, color="0.25")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    handles = [Line2D([0], [0], color="0.3", lw=2.2, ls=ls, label=lab) for _, ls, lab in LICKS]
    handles += [Line2D([0], [0], marker="^", color="0.3", lw=0, mec="k", label="HIT onset"),
                Line2D([0], [0], marker="v", color="0.3", lw=0, mec="k", label="FA onset")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Preparatory recruitment: HIT (decision) vs FA (impulsive) lick — by cell class x region",
                 fontsize=15, y=1.005)
    for ext in ("png", "pdf"):
        fig.savefig(FIGROOT / f"fig_hit_vs_fa.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(FIGROOT / "fig_hit_vs_fa_stats.csv", index=False)

    # brief console summary of onset (hit vs fa) per class/region
    print("onset (s from lick), HIT vs FA:", flush=True)
    for rname, _ in REGIONS:
        for group in CLASSES:
            h = df[(df.region == rname) & (df.group == group) & (df.lick == "hit")]
            f = df[(df.region == rname) & (df.group == group) & (df.lick == "fa")]
            hv = h.onset_s.iloc[0] if len(h) else np.nan
            fv = f.onset_s.iloc[0] if len(f) else np.nan
            print(f"  [{rname:6s}] {group:10s} HIT={hv:+.3f}  FA={fv:+.3f}  (HIT-FA={hv-fv:+.3f})", flush=True)
    print(f"wrote {FIGROOT}/fig_hit_vs_fa.png (+pdf, +_stats.csv)", flush=True)


if __name__ == "__main__":
    main()
