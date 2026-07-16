"""Neuronal-activity view of the TF-cell / lick-responsiveness story: FA-lick-
aligned population activity of TF-responsive cells, sorted by each cell's pre-lick
ramp, with the canonical lick-responsive label marked. Shows WHY sustained cells
are 89% lick-responsive — they carry the strong, consistent pre-lick motor ramp —
while transient cells' ramp is weaker. Reuses the FA traces already cached by
heatmap_transient_sustained (peth_traces.npz) + the lick_acquisition per-cell
labels — no session reload.
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
from matplotlib.colors import TwoSlopeNorm, ListedColormap

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from transient_vs_sustained import TCOL, SCOL                                # noqa: E402

FIG = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046")
NPZ = FIG / "heatmap_transient_sustained" / "peth_traces.npz"
LICK = FIG / "lick_acquisition" / "lick_acquisition_cells.csv"
OUT = FIG / "fa_lick_activity"
PRE = (-0.3, -0.15)     # canonical pre-lick window (== FA-lick responsiveness window)


def main():
    D = np.load(NPZ, allow_pickle=True)
    cls = D["meta_cls"].astype(str)
    subj, sess, unit = D["meta_subject"].astype(str), D["meta_session"].astype(str), D["meta_unit"].astype(int)
    t = D["t_fa"]
    M = D["mat_fa"]

    lk = pd.read_csv(LICK)
    lmap = {(str(r.subject), str(r.session), int(r.unit)): bool(r.lick_sig) for r in lk.itertuples()}
    lick_sig = np.array([lmap.get((subj[i], sess[i], int(unit[i])), np.nan) for i in range(len(cls))])

    pre = (t >= PRE[0]) & (t <= PRE[1])
    ramp = np.nanmean(M[:, pre], axis=1)
    keep = np.isin(cls, ["transient", "sustained"]) & np.isfinite(ramp) & np.isfinite(M).any(1) & np.isfinite(lick_sig)
    idx = np.where(keep)[0]
    idx = idx[np.argsort(ramp[idx])[::-1]]     # all cells sorted by pre-lick ramp, descending
    Msort = M[idx]
    cls_s = cls[idx]
    lick_s = lick_sig[idx].astype(bool)
    n = len(idx)
    n_tr, n_su = int((cls_s == "transient").sum()), int((cls_s == "sustained").sum())

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 12})

    rr = np.nanmean(M[:, pre], axis=1)     # pre-lick ramp per cell (all)
    rr_s = np.nanmean(Msort[:, pre], axis=1)

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, width_ratios=[0.045, 1, 0.62], height_ratios=[1, 3.0],
                           wspace=0.14, hspace=0.30)

    # ── grand-average PSTH (top-middle), class means ──────────────────
    axp = fig.add_subplot(gs[0, 1])
    for c, col in (("transient", TCOL), ("sustained", SCOL)):
        m = cls_s == c
        mean = np.nanmean(Msort[m], 0)
        # 95% CI (1.96 * SEM) — project convention: never shade a bare SEM.
        ci = 1.96 * np.nanstd(Msort[m], 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Msort[m]), 0).clip(1))
        pctl = 100 * lick_s[m].mean()
        axp.plot(t, mean, color=col, lw=2.6, label=f"{c}  ({pctl:.0f}% lick-resp, n={m.sum()})")
        axp.fill_between(t, mean - ci, mean + ci, color=col, alpha=0.2)
    axp.axvspan(PRE[0], PRE[1], color="0.85", zorder=0, label="pre-lick window")
    axp.axvline(0, color="k", lw=1.1, ls="--")
    axp.axhline(0, color="0.8", lw=0.8)
    axp.set_xlim(t[0], t[-1]); axp.set_ylabel("z-score (pop mean)", fontsize=13)
    axp.set_title("FA-lick population response — sustained carry the pre-lick ramp",
                  fontsize=14, fontweight="bold")
    axp.legend(frameon=False, fontsize=10.5, loc="upper left")
    axp.tick_params(labelsize=11)
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)

    # ── pre-lick ramp DISTRIBUTION by class (top-right) ───────────────
    axd = fig.add_subplot(gs[0, 2])
    lo, hi = np.nanpercentile(rr[np.isfinite(rr)], [1, 99])
    bins = np.linspace(lo, hi, 22)
    for c, col in (("transient", TCOL), ("sustained", SCOL)):
        v = rr_s[cls_s == c]
        axd.hist(v, bins=bins, density=True, color=col, alpha=0.5, label=f"{c} (med {np.median(v):+.2f})")
        axd.axvline(np.median(v), color=col, lw=2.2, ls="--")
    axd.axvline(0, color="0.6", lw=0.9, ls=":")
    axd.set_xlabel("pre-lick ramp (z)", fontsize=12); axd.set_ylabel("density", fontsize=12)
    axd.set_title("sustained ramps are larger", fontsize=13, fontweight="bold")
    axd.legend(frameon=False, fontsize=9.5, loc="upper right")
    axd.tick_params(labelsize=10)
    for sp in ("top", "right"):
        axd.spines[sp].set_visible(False)

    # ── class strip (left of heatmap) ─────────────────────────────────
    axc = fig.add_subplot(gs[1, 0])
    strip = np.array([[0.0] if c == "transient" else [1.0] for c in cls_s])
    axc.imshow(strip, aspect="auto", cmap=ListedColormap([TCOL, SCOL]),
               extent=[0, 1, n, 0], interpolation="nearest")
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_ylabel("cells, sorted by pre-lick ramp  →", fontsize=12)

    # ── heatmap (spans cols 1–2) ──────────────────────────────────────
    axh = fig.add_subplot(gs[1, 1:])
    im = axh.imshow(Msort, aspect="auto", cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=3),
                    extent=[t[0], t[-1], n, 0], interpolation="nearest")
    axh.axvspan(PRE[0], PRE[1], color="k", alpha=0.10, zorder=1)
    axh.axvline(0, color="k", lw=1.1, ls="--")
    axh.set_xlabel("t from FA lick (s)", fontsize=13); axh.set_yticks([])
    axh.tick_params(labelsize=11)
    # transient/sustained block labels on the colour strip
    cb = fig.colorbar(im, ax=axh, fraction=0.024, pad=0.012, ticks=[-1, 0, 1, 2, 3])
    cb.set_label("z-score", fontsize=11); cb.ax.tick_params(labelsize=9)

    fig.suptitle(f"FA-lick-aligned activity of TF-responsive cells (n={n}: {n_tr} transient, {n_su} sustained) — "
                 "sustained cells cluster at the strong pre-lick-ramp end (the basis of their 89% lick-responsiveness)",
                 fontsize=13, y=0.98)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fa_lick_activity.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fa_lick_activity.png  [n={n}, transient={n_tr}, sustained={n_su}]")


if __name__ == "__main__":
    main()
