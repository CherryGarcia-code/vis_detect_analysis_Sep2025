"""B8 exploratory: the honest discrete psychometric + multi-criterion threshold.

Motivated by a figure-review question on F1B: that panel shows a *continuous*
"change size at 50% detection", but there are only 5 discrete change sizes, and
50% is a soft criterion when the lower asymptote (catch/guessing lick rate) is
already high. This script makes two presentation-ready companions to F1B:

  F1C  fig_b8_F1C_psychometric_discrete.png
       The raw discretized psychometric: mean P(detect) at each ACTUAL change
       size (catch 1.0 included as the guessing/FA anchor), per mood, with
       per-session spaghetti + bootstrap CI across sessions, and the 0.5/0.6/0.7
       criterion lines drawn so you can SEE where each criterion bites.

  F1D  fig_b8_F1D_threshold_multicriterion.png
       F1B redone at three detection criteria (50/60/70%): per-session detection
       threshold over the learning axis. Annotates the fraction of session-cells
       whose threshold pins at the 1.0 floor — which rises with criterion for the
       liberal Impulsive mood, exactly because its curve starts above 0.6.

Reads the cached raw trial table (no re-load of sessions). Run:
  WT=$(pwd); PYTHONPATH="$WT/src" py scripts/analysis/decision_latents/explore_psychometric_discrete.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from visdetect.suite.plotting import setup_style
from visdetect.analysis.config import ROOT, SUBJECT, STATE_LABEL_COLORS
from visdetect.analysis.decision_latents import _logistic

setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents")
TRIAL_CACHE = os.path.join(CACHE_DIR, "decision_latents_trialtable.csv")
os.makedirs(FIG_DIR, exist_ok=True)

MOODS = ("Impulsive", "StimSens")
CRITERIA = (0.5, 0.6, 0.7)
# Match the deliverable's cell-inclusion gate (decision_latents.descriptive_cell_table)
# so F1D's 50% panel reconciles with F1B: drop cells with < 20 total trials. A
# looser gate lets in tiny noisy cells whose logistic fit can go non-monotonic
# (negative slope) and emit an absurd clamped threshold (~6.6) that spikes the
# binned-mean trend — exactly the F1D-vs-F1B discrepancy this guards against.
MIN_CELL_TRIALS = 20
# All change sizes the mouse actually saw, catch first (the guessing anchor).
ALL_CS = [1.0, 1.25, 1.35, 1.5, 2.0, 4.0]
ALL_CS_LABELS = ["1.0\n(catch)", "1.25", "1.35", "1.5", "2", "4"]
GO_CS = [c for c in ALL_CS if c > 1.0]
RNG = np.random.default_rng(42)


def save_fig(fig, name):
    p = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def _bootstrap_ci(vals, n_boot=1000):
    """Percentile bootstrap 95% CI of the mean over sessions (seed=42)."""
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    if vals.size == 1:
        return (vals[0], vals[0], vals[0])
    boot = np.array([RNG.choice(vals, vals.size, replace=True).mean()
                     for _ in range(n_boot)])
    return (float(vals.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def _fit_ab(sub_go):
    """Refit the 2-param logistic on log2(change_size) vs lick for one
    (session x mood) GO subset. Returns (a, b) or (nan, nan). Same gate/bounds
    as decision_latents.sharpness_scores so thresholds are comparable to F1B."""
    if len(sub_go) < 8 or sub_go["change_size"].nunique() < 2:
        return (np.nan, np.nan)
    x = np.log2(sub_go["change_size"].values)
    y = sub_go["lick"].values.astype(float)
    try:
        (a, b), _ = curve_fit(_logistic, x, y, p0=[0.0, 1.0],
                              bounds=([-20.0, -20.0], [20.0, 20.0]), maxfev=5000)
        return (float(a), float(b))
    except Exception:
        return (np.nan, np.nan)


def _threshold_at(a, b, p):
    """Change size at detection probability p from a 2-param logistic.
    P = 1/(1+exp(-(a+b*x))) = p  ->  x_p = (logit(p) - a)/b ;  thr = 2**x_p,
    clamped to [1.0, 8.0]. NaN unless the slope is positive (>= 1e-3): a
    detection threshold is only defined for an INCREASING psychometric;
    near-flat or negative-slope fits (small-sample noise) have no threshold."""
    if not (np.isfinite(a) and np.isfinite(b)) or b < 1e-3:
        return np.nan
    logit = np.log(p / (1.0 - p))
    x_p = (logit - a) / b
    return float(np.clip(2.0 ** x_p, 1.0, 8.0))


# ── F1C: the honest discrete psychometric ──────────────────────────────────
def fig_F1C_discrete(trials):
    # per-session mean P(lick) at each change size, per mood
    cell = (trials.groupby(["session_name", "state_label", "change_size"])["lick"]
            .mean().reset_index())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xpos = {cs: i for i, cs in enumerate(ALL_CS)}
    for mood in MOODS:
        c = STATE_LABEL_COLORS[mood]
        m = cell[cell["state_label"] == mood]
        # faint per-session spaghetti
        for sname, g in m.groupby("session_name"):
            g = g[g["change_size"].isin(ALL_CS)]
            xs = [xpos[cs] for cs in g["change_size"]]
            ax.plot(xs, g["lick"].values, "-", color=c, alpha=0.10, lw=0.8)
        # mean + bootstrap CI across sessions at each change size
        means, los, his, xs = [], [], [], []
        for cs in ALL_CS:
            vals = m[m["change_size"] == cs]["lick"].values
            mu, lo, hi = _bootstrap_ci(vals)
            if np.isfinite(mu):
                means.append(mu); los.append(lo); his.append(hi); xs.append(xpos[cs])
        ax.plot(xs, means, "o-", color=c, lw=2.4, ms=7, label=mood, zorder=5)
        ax.fill_between(xs, los, his, color=c, alpha=0.22, zorder=4)
    # criterion lines
    for p in CRITERIA:
        ax.axhline(p, color="0.55", ls=":", lw=1.0)
        ax.text(len(ALL_CS) - 0.95, p + 0.012, f"{int(p*100)}%", color="0.4",
                fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(len(ALL_CS)))
    ax.set_xticklabels(ALL_CS_LABELS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("actual change size (Δ TF ratio) — discrete, equal-spaced")
    ax.set_ylabel("P(detect)  [lick rate]")
    ax.set_title("F1C  The real (discrete) psychometric behind F1B\n"
                 "catch=guessing floor; mean±bootstrap CI over sessions, faint=per-session")
    ax.legend(frameon=False, loc="lower right")
    return save_fig(fig, "fig_b8_F1C_psychometric_discrete")


# ── F1D: F1B at three detection criteria ───────────────────────────────────
def fig_F1D_multicriterion(trials):
    # refit (a,b) once per (session x mood) on its GO trials; reuse session_idx
    # as the learning axis. Iterate FULL cells so the n_total >= MIN_CELL_TRIALS
    # gate matches the deliverable (descriptive_cell_table) and F1D reconciles
    # with F1B at 50%.
    rows = []
    for (sname, mood), cell in trials.groupby(["session_name", "state_label"]):
        if mood not in MOODS or len(cell) < MIN_CELL_TRIALS:
            continue
        a, b = _fit_ab(cell[cell["change_size"] > 1.0])
        sidx = float(cell["session_idx"].iloc[0])
        row = {"session_idx": sidx, "state_label": mood}
        for p in CRITERIA:
            row[f"thr_{p}"] = _threshold_at(a, b, p)
        rows.append(row)
    cells = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(CRITERIA), figsize=(15, 4.4), sharey=True)
    for ax, p in zip(axes, CRITERIA):
        col = f"thr_{p}"
        thr_all = cells[col].values.astype(float)
        floor_notes = []
        for mood in MOODS:
            c = STATE_LABEL_COLORS[mood]
            sub = cells[(cells["state_label"] == mood) & np.isfinite(cells[col])]
            if sub.empty:
                continue
            ax.scatter(sub["session_idx"], sub[col], color=c, s=22, alpha=0.5, label=mood)
            xv = sub["session_idx"].values.astype(float)
            yv = sub[col].values.astype(float)
            if xv.size >= 2 and np.ptp(xv) > 0:
                edges = np.linspace(xv.min(), xv.max(), 6)
                bc = 0.5 * (edges[:-1] + edges[1:])
                bm = [np.nanmean(yv[(xv >= edges[i]) & (xv <= edges[i + 1])])
                      if np.any((xv >= edges[i]) & (xv <= edges[i + 1])) else np.nan
                      for i in range(len(edges) - 1)]
                ax.plot(bc, bm, "-", color=c, lw=2.2)
            frac_floor = float(np.mean(np.isclose(yv, 1.0))) if yv.size else np.nan
            floor_notes.append(f"{mood}: {frac_floor*100:.0f}% at floor")
        if np.any(np.isfinite(thr_all)):
            top = min(2.0, float(np.nanpercentile(thr_all, 95)) + 0.1)
        else:
            top = 2.0
        ax.set_ylim(0.95, max(top, 1.0))
        ax.set_xlabel("session index (chronological, learning →)")
        ax.set_title(f"{int(p*100)}% detection criterion")
        ax.text(0.02, 0.98, "\n".join(floor_notes), transform=ax.transAxes,
                fontsize=8, va="top", ha="left", color="0.3")
        if p == CRITERIA[0]:
            ax.set_ylabel("change size at criterion\n(lower = sharper)")
            ax.legend(frameon=False, loc="upper right")
    fig.suptitle("F1D  Detection threshold over learning at 50 / 60 / 70% criteria\n"
                 "(higher criterion = harder test; Impulsive pins at floor as criterion rises)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_fig(fig, "fig_b8_F1D_threshold_multicriterion")


if __name__ == "__main__":
    if not os.path.exists(TRIAL_CACHE):
        raise SystemExit(f"missing trial-table cache: {TRIAL_CACHE}\n"
                         "run run_decision_latents_by_state.py first to build it.")
    trials = pd.read_csv(TRIAL_CACHE)
    print(f"loaded {len(trials)} trials, {trials['session_name'].nunique()} sessions")
    print("F1C:", fig_F1C_discrete(trials))
    print("F1D:", fig_F1D_multicriterion(trials))
