"""Across-learning change-size scaling x STATE (BG_046 only): does the big-small evidence
scaling differ by behavioural state (Impulsive / StimSens / Disengaged), and does that change
Learning -> Expert? State joined per trial (trial_idx). big-small within each state x stage.

CAVEAT (on figure): states are defined from behaviour (partly circular), naive is thin, and
state composition is itself stage-dependent (early sessions skew disengaged/impulsive). So this
is suggestive, DESCRIPTIVE — not a clean state-independent claim. Bands = bootstrap 95% CI.

Usage: py scripts/talk_substrate/ws_learning_state.py
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
from visdetect.analysis.config import STATE_LABEL_COLORS  # noqa: E402
from visdetect.analysis.utils import bootstrap_ci  # noqa: E402

C.setup_talk_style()
STAGES = ["Naive", "Learning", "Expert"]
STATES = ["Impulsive", "StimSens", "Disengaged"]
CSV = C.CACHE_DIR / "ws_learning_crossed_BG_046.csv"


def mean_ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    lo, hi = bootstrap_ci(x, n_bootstrap=1000, ci_level=0.95, axis=0, seed=42)
    return float(np.mean(x)), float(lo), float(hi), len(x)


def main():
    df = pd.read_csv(CSV)
    df = df[df.subject == "BG_046"].copy()
    for st in STATES:
        df[f"{st}_scaling"] = df[f"{st}_big"] - df[f"{st}_small"]

    fig = plt.figure(figsize=(13, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    rows = []
    for st in STATES:
        xs, ys, lo, hi = [], [], [], []
        for i, stg in enumerate(STAGES):
            d = df[df.stage == stg][f"{st}_scaling"]
            mu, l, h, n = mean_ci(d)
            rows.append(dict(state=st, stage=stg, scaling=round(mu, 4) if np.isfinite(mu) else None,
                             ci_lo=round(l, 4) if np.isfinite(mu) else None,
                             ci_hi=round(h, 4) if np.isfinite(mu) else None, n=n))
            if np.isfinite(mu):
                xs.append(i); ys.append(mu); lo.append(mu - l); hi.append(h - mu)
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=STATE_LABEL_COLORS[st],
                        capsize=3, lw=1.9, label=st)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels(STAGES)
    ax.set_ylabel("big − small change response (z)")
    ax.set_title("BG_046: change-size scaling across learning, by STATE", fontsize=C.FS["title"])
    ax.legend(frameon=False, fontsize=C.FS["legend"])

    # n per state x stage (the power/composition guard)
    ax2 = fig.add_subplot(gs[1])
    sdf = pd.DataFrame(rows)
    width = 0.25
    for j, st in enumerate(STATES):
        ns = [sdf[(sdf.state == st) & (sdf.stage == s)]["n"].iloc[0] for s in STAGES]
        ax2.bar(np.arange(3) + (j - 1) * width, ns, width,
                color=STATE_LABEL_COLORS[st], label=st)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(STAGES)
    ax2.set_ylabel("unit-sessions (with ≥3 trials)")
    ax2.set_title("Power / state composition (guard)", fontsize=C.FS["title"])
    ax2.legend(frameon=False, fontsize=C.FS["legend"])

    fig.suptitle("BG_046: change-size scaling ACROSS LEARNING x behavioural STATE (descriptive)",
                 fontsize=C.FS["suptitle"], y=1.0)
    fig.text(0.5, -0.04,
             "big−small within each state x stage (per-unit, 0–1 s post-change, shared baseline). "
             "CAVEAT: states are behaviourally defined (partly circular); naive thin; state composition "
             "is stage-dependent (right panel) — suggestive only. Colours = STATE_LABEL_COLORS.",
             ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.FIG_DIR.parent / "BG_046" / "ws_learning_state.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    sdf.to_csv(C.FIG_DIR.parent / "BG_046" / "ws_learning_state.csv", index=False)
    print(f"[fig] wrote {out}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
