"""Across-learning change-size scaling, split HIT vs MISS (addresses the hit+miss pooling
confound). big-small scaling computed WITHIN hits and WITHIN misses, per stage, per animal.

If the big-small scaling persists on MISS trials (no response/movement), it is a SENSORY/
evidence signal; if it is hit-only, it is decision/response-contingent. Reads the per-subject
per-unit-session crossed caches (ws_learning_crossed_<SUBJECT>.csv, concatenated). Bands =
bootstrap 95% CI over unit-sessions. DESCRIPTIVE, NOT N1.

Usage: py scripts/talk_substrate/ws_learning_hitmiss.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402
from visdetect.analysis.config import OUTCOME_COLORS  # noqa: E402
from visdetect.analysis.utils import bootstrap_ci  # noqa: E402

setup_style()
STAGES = ["Naive", "Learning", "Expert"]
ANIMALS = [("BG_046", "Striatum DMS"), ("BG_039", "Striatum DMS"),
           ("BG_031", "Striatum VMS"), ("BG_038", "Cortex M1/S1 ref")]
def _load_all():
    frames = []
    for s, _r in ANIMALS:
        p = C.CACHE_DIR / f"ws_learning_crossed_{s}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mean_ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    lo, hi = bootstrap_ci(x, n_bootstrap=1000, ci_level=0.95, axis=0, seed=42)
    return float(np.mean(x)), float(lo), float(hi), len(x)


def pts(ax, df, scaling_col, color, label):
    xs, ys, lo, hi, ns = [], [], [], [], []
    for i, s in enumerate(STAGES):
        d = df[df.stage == s][scaling_col]
        mu, l, h, n = mean_ci(d)
        if np.isfinite(mu):
            xs.append(i); ys.append(mu); lo.append(mu - l); hi.append(h - mu); ns.append(n)
    if xs:
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=color, capsize=3, lw=1.8,
                    label=f"{label}")
    return ns


def main():
    df = _load_all()
    df["hit_scaling"] = df["hit_big"] - df["hit_small"]
    df["miss_scaling"] = df["miss_big"] - df["miss_small"]

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.24)
    rows = []
    for ai, (subj, region) in enumerate(ANIMALS):
        ax = fig.add_subplot(gs[ai // 2, ai % 2])
        sub = df[df.subject == subj]
        nh = pts(ax, sub, "hit_scaling", OUTCOME_COLORS["Hit"], "HIT trials")
        nm = pts(ax, sub, "miss_scaling", OUTCOME_COLORS["Miss"], "MISS trials")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xticks(range(3)); ax.set_xticklabels(STAGES)
        ax.set_ylabel("big − small change response (z)")
        ax.set_title(f"{subj} ({region})", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for s in STAGES:
            d = df[(df.subject == subj) & (df.stage == s)]
            hmu = mean_ci(d["hit_scaling"]); mmu = mean_ci(d["miss_scaling"])
            rows.append(dict(subject=subj, region=region, stage=s,
                             hit_scaling=round(hmu[0], 4) if np.isfinite(hmu[0]) else None,
                             hit_ci=f"[{hmu[1]:.3f},{hmu[2]:.3f}]" if np.isfinite(hmu[0]) else None,
                             hit_n=hmu[3],
                             miss_scaling=round(mmu[0], 4) if np.isfinite(mmu[0]) else None,
                             miss_ci=f"[{mmu[1]:.3f},{mmu[2]:.3f}]" if np.isfinite(mmu[0]) else None,
                             miss_n=mmu[3]))

    fig.suptitle("Change-size (evidence) scaling ACROSS LEARNING — HIT vs MISS split",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.02,
             "big−small change response computed WITHIN hits and WITHIN misses (per-unit, 0–1 s "
             "post-change, shared baseline). Scaling on MISS trials (no response) = sensory/evidence; "
             "hit-only = decision-contingent. Bands = bootstrap 95% CI over unit-sessions; naive thin.",
             ha="center", fontsize=8, color="#555555", wrap=True)
    out = C.FIG_DIR.parent / "ws_learning_hitmiss.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(C.FIG_DIR.parent / "ws_learning_hitmiss.csv", index=False)
    print(f"[fig] wrote {out}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
