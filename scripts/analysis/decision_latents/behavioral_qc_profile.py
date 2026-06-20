"""B8 Phase-0.5: behavioral data-QUALITY profiler — check the distributions of
everything the analysis counts on, BEFORE scoring, and set thresholds from the
distributions instead of guessing.

Motivation (user directive, 2026-06-20): don't run scores on underpowered units
and patch the junk downstream (e.g. an 18-trial session×mood cell whose 11
go-trials gave a negative-slope logistic fit → an absurd clamped threshold that
spiked F1D). Solid foundation first.

The analysis UNIT is the (session × mood) CELL, not the session — slicing a
healthy session by mood can still leave a thin cell. So we profile both levels.

Outputs (presentation-ready):
  fig_b8_QC_distributions.png   — histogram of every dependency with the current
                                  threshold drawn, + a gate-yield summary panel.
  behavioral_qc_cell_table.csv  — per-cell QC metrics (the basis for an `usable` flag).

Run:
  WT=$(pwd); PYTHONPATH="$WT/src" py scripts/analysis/decision_latents/behavioral_qc_profile.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from visdetect.suite.plotting import setup_style
from visdetect.analysis.config import ROOT, SUBJECT, STATE_LABEL_COLORS, CHANGE_SIZES
from visdetect.analysis.decision_latents import _logistic

setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents")
TRIAL_CACHE = os.path.join(CACHE_DIR, "decision_latents_trialtable.csv")
os.makedirs(FIG_DIR, exist_ok=True)

# Current thresholds the pipeline uses (to be JUSTIFIED or REVISED by this profile)
MIN_CELL_TRIALS = 20     # descriptive_cell_table
MIN_GO_TRIALS = 8        # sharpness_scores logistic gate
MIN_DISTINCT_CS = 2      # sharpness_scores
MIN_RT_PER_CS = 3        # rt_cv per change size
MIN_SESSION_TRIALS = 50  # enumerate_valid_sessions
MOOD_ORDER = ("Impulsive", "StimSens", "Disengaged")


def _fit_slope(go):
    if len(go) < 2 or go["change_size"].nunique() < 2:
        return np.nan
    x = np.log2(go["change_size"].values); y = go["lick"].values.astype(float)
    try:
        (a, b), _ = curve_fit(_logistic, x, y, p0=[0.0, 1.0],
                              bounds=([-20.0, -20.0], [20.0, 20.0]), maxfev=5000)
        return float(b)
    except Exception:
        return np.nan


def build_cell_qc(trials):
    rows = []
    for (sname, mood), cell in trials.groupby(["session_name", "state_label"]):
        go = cell[cell["change_size"] > 1.0]
        catch = cell[np.isclose(cell["change_size"], 1.0)]
        per_cs = [int((go["change_size"] == cs).sum()) for cs in CHANGE_SIZES]
        b = _fit_slope(go)
        rows.append({
            "session_name": sname, "state_label": mood,
            "n_trials": len(cell), "n_go": len(go), "n_catch": len(catch),
            "n_distinct_cs": int(go["change_size"].nunique()),
            "catch_lick_rate": float(catch["lick"].mean()) if len(catch) else np.nan,
            "slope_b": b,
            "min_go_per_cs": int(min(per_cs)) if per_cs else 0,
            "n_fa": int((cell["outcome"] == "fa").sum()),
            "mean_state_conf": float(cell["state_confidence"].mean()),
            "session_idx": int(cell["session_idx"].iloc[0]),
        })
    return pd.DataFrame(rows)


def _hist(ax, vals, thr=None, thr_label=None, bins=20, color="#888888", logy=False):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    if thr is not None:
        ax.axvline(thr, color="#cc2a36", ls="--", lw=1.6)
        n_below = int((vals < thr).sum())
        ax.text(0.97, 0.95, f"{thr_label}\n{n_below}/{vals.size} below",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#cc2a36")
    if logy:
        ax.set_yscale("log")


def fig_qc(trials, cellqc):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))

    _hist(axes[0, 0], cellqc["n_trials"], MIN_CELL_TRIALS, f"min={MIN_CELL_TRIALS}", bins=30, logy=True)
    axes[0, 0].set_title("CELL total trials"); axes[0, 0].set_xlabel("n trials in (session×mood)")

    _hist(axes[0, 1], cellqc["n_go"], MIN_GO_TRIALS, f"min={MIN_GO_TRIALS}", bins=30, logy=True)
    axes[0, 1].set_title("CELL go-trials (psychometric support)"); axes[0, 1].set_xlabel("n go-trials")

    vc = cellqc["n_distinct_cs"].value_counts().sort_index()
    axes[0, 2].bar(vc.index, vc.values, color="#888888", edgecolor="white")
    axes[0, 2].axvline(MIN_DISTINCT_CS - 0.5, color="#cc2a36", ls="--", lw=1.6)
    axes[0, 2].set_title("CELL distinct go change-sizes"); axes[0, 2].set_xlabel("# distinct Δ (of 5)")
    axes[0, 2].text(0.97, 0.95, f"min={MIN_DISTINCT_CS}", transform=axes[0, 2].transAxes,
                    ha="right", va="top", fontsize=7.5, color="#cc2a36")

    # catch lick rate (guessing floor) by mood
    for mood in MOOD_ORDER:
        sub = cellqc[cellqc["state_label"] == mood]["catch_lick_rate"]
        sub = sub[np.isfinite(sub)]
        if len(sub):
            axes[1, 0].hist(sub, bins=np.linspace(0, 1, 21), alpha=0.55,
                            color=STATE_LABEL_COLORS.get(mood, "#999999"), label=mood)
    axes[1, 0].set_title("CELL catch lick-rate (guessing floor)")
    axes[1, 0].set_xlabel("P(lick | catch)"); axes[1, 0].legend(frameon=False, fontsize=7)

    # fitted slope sign — negative slope = degenerate fit
    _hist(axes[1, 1], cellqc["slope_b"], 0.0, "b=0 (sign)", bins=40)
    n_neg = int((cellqc["slope_b"] < 0).sum())
    n_fit = int(np.isfinite(cellqc["slope_b"]).sum())
    axes[1, 1].set_title(f"CELL fitted slope b ({n_neg}/{n_fit} NEGATIVE = degenerate)")
    axes[1, 1].set_xlabel("logistic slope b")

    _hist(axes[1, 2], cellqc["min_go_per_cs"], MIN_RT_PER_CS, f"min={MIN_RT_PER_CS}", bins=30, logy=True)
    axes[1, 2].set_title("CELL min go-trials per change-size (RT-CV support)")
    axes[1, 2].set_xlabel("min trials across the 5 Δ")

    _hist(axes[2, 0], cellqc["n_fa"], bins=30, logy=True)
    axes[2, 0].set_title("CELL FA-trial count (FA-hazard support)"); axes[2, 0].set_xlabel("n FA trials")

    _hist(axes[2, 1], cellqc["mean_state_conf"], bins=20)
    axes[2, 1].set_title("CELL mean state-label confidence"); axes[2, 1].set_xlabel("mean confidence")

    # session-level total trials
    sess_tot = trials.groupby("session_name").size()
    _hist(axes[2, 2], sess_tot.values, MIN_SESSION_TRIALS, f"min={MIN_SESSION_TRIALS}", bins=30)
    axes[2, 2].set_title("SESSION total labeled trials"); axes[2, 2].set_xlabel("n trials in session")

    fig.suptitle("B8 behavioral-QC distributions — every dependency, with current threshold (red) and #units below\n"
                 "(set thresholds from these distributions; unit = session×mood CELL unless 'SESSION')",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(FIG_DIR, "fig_b8_QC_distributions.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p


def report(cellqc):
    n = len(cellqc)
    print(f"\n=== Cell-level QC gate yields (n={n} session×mood cells) ===")
    gates = {
        f"n_trials >= {MIN_CELL_TRIALS}": cellqc["n_trials"] >= MIN_CELL_TRIALS,
        f"n_go >= {MIN_GO_TRIALS}": cellqc["n_go"] >= MIN_GO_TRIALS,
        f"distinct_cs >= {MIN_DISTINCT_CS}": cellqc["n_distinct_cs"] >= MIN_DISTINCT_CS,
        "slope_b > 0 (monotonic)": cellqc["slope_b"] > 0,
        f"min_go_per_cs >= {MIN_RT_PER_CS}": cellqc["min_go_per_cs"] >= MIN_RT_PER_CS,
    }
    passing = pd.Series(True, index=cellqc.index)
    for name, g in gates.items():
        g = g.fillna(False)
        passing &= g
        print(f"  {name:<28}: pass {int(g.sum()):>3}/{n}  (drop {int((~g).sum())})")
    print(f"  {'ALL gates (usable cell)':<28}: pass {int(passing.sum()):>3}/{n}  (drop {int((~passing).sum())})")
    print("\n  By mood (usable cells):")
    for mood in MOOD_ORDER:
        m = cellqc["state_label"] == mood
        print(f"    {mood:<11}: {int((passing & m).sum()):>3}/{int(m.sum())} usable")
    # show the cells that fail (the foundation we'd otherwise build on)
    bad = cellqc[~passing].sort_values("session_idx")
    if len(bad):
        print("\n  Failing cells (excluded foundation):")
        for _, r in bad.iterrows():
            why = []
            if r["n_trials"] < MIN_CELL_TRIALS: why.append(f"nTot={int(r['n_trials'])}")
            if r["n_go"] < MIN_GO_TRIALS: why.append(f"nGo={int(r['n_go'])}")
            if not (r["slope_b"] > 0): why.append(f"b={r['slope_b']:.2f}")
            if r["min_go_per_cs"] < MIN_RT_PER_CS: why.append(f"minCS={int(r['min_go_per_cs'])}")
            print(f"    sidx={int(r['session_idx']):>3} {r['state_label']:<11} {r['session_name']}: {', '.join(why)}")
    return passing


if __name__ == "__main__":
    if not os.path.exists(TRIAL_CACHE):
        raise SystemExit(f"missing {TRIAL_CACHE}; run run_decision_latents_by_state.py first.")
    trials = pd.read_csv(TRIAL_CACHE)
    print(f"loaded {len(trials)} trials, {trials['session_name'].nunique()} sessions, "
          f"moods={sorted(trials['state_label'].unique())}")
    cellqc = build_cell_qc(trials)
    cellqc["usable"] = report(cellqc)
    out_csv = os.path.join(CACHE_DIR, "behavioral_qc_cell_table.csv")
    cellqc.to_csv(out_csv, index=False)
    print("\nfigure:", fig_qc(trials, cellqc))
    print("cell QC table:", out_csv)
