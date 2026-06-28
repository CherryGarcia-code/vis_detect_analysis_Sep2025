"""B8 Phase 2 (Task 0.7): naive-session label-reliability protocol (fix g).

Plain English: the mood labeler was calibrated on good-behavior sessions. The
earliest naive sessions are 'out of distribution', so before we trust their
moods we look at: how much of each session is each mood, and how confident the
labeler is on each trial. The watch zone is the low-d' naive sessions, where the
mouse barely knows the task and the labeler may be guessing.

This is a JUDGMENT checkpoint, not an automated pass/fail. For each session we
compute a boolean `naive_reliable` = (>=80% of the session's trials have
state_confidence > 0.7). Sessions that FAIL this rule are flagged; downstream
they drop to a coarse, no-mood treatment (their moods are too shaky to trust).

Deliverables:
  * FIGURES/decision_latents/BG_046/fig_b8_P2_label_reliability.png
      (i) mean label-confidence vs session d', with unreliable sessions flagged
      (ii) mood composition per session in chronological order, unreliable
           sessions annotated
  * data/cache/decision_latents/b8p2_label_reliability.csv
  * printed list of flagged (unreliable) sessions

Coverage note: dl.enumerate_valid_sessions() already returns 45 tagged sessions
(full coverage), so the labeler does NOT need to be re-run here; we just confirm
coverage and proceed.
"""
import os
import sys
import gc

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Console is cp1252; allow d' / arrows in prints without crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from visdetect.suite.plotting import setup_style          # styling only (NOT save_figure)
from visdetect.suite.loader import load_session
from visdetect.analysis import decision_latents as dl
from visdetect.analysis.config import (
    ROOT, SUBJECT, parse_session_date, STATE_LABEL_COLORS,
)

setup_style()

FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Confidence-gate parameters (Task 0.7 fix g). A session is "naive_reliable" iff
# >= RELIABLE_TRIAL_FRAC of its trials have state_confidence > CONF_THRESH.
CONF_THRESH = 0.7
RELIABLE_TRIAL_FRAC = 0.80
MOODS = ["Impulsive", "StimSens", "Disengaged", "Abort"]


def save_fig(fig, name):
    """Local saver (deliberately NOT suite.plotting.save_figure): writes a
    presentation-ready PNG into FIGURES/decision_latents/<subject>/."""
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def session_reliability_row(session_name, labels):
    """One reliability row for a session, given its per-trial state labels.

    Parameters
    ----------
    session_name : str   DDMMYYYY id.
    labels : pd.DataFrame  indexed by trial_idx with columns
        ['state_label', 'state_confidence'] (as returned by dl.load_state_labels).

    Returns a dict with mood proportions, mean confidence, high-confidence trial
    fraction, and the `naive_reliable` boolean (>= RELIABLE_TRIAL_FRAC of trials
    with state_confidence > CONF_THRESH). d' is filled in by the caller (needs the
    session object).
    """
    conf = pd.to_numeric(labels["state_confidence"], errors="coerce")
    n_trials = int(len(labels))
    frac_high = float((conf > CONF_THRESH).mean()) if n_trials else 0.0
    props = labels["state_label"].value_counts(normalize=True)
    row = {
        "session": str(session_name),
        "n_trials": n_trials,
        "mean_conf": float(conf.mean()) if n_trials else np.nan,
        "frac_conf_gt_0.7": frac_high,
        "naive_reliable": bool(frac_high >= RELIABLE_TRIAL_FRAC),
    }
    row.update({m: float(props.get(m, 0.0)) for m in MOODS})
    return row


def compute_table():
    """Build the per-session reliability table over all tagged sessions."""
    sessions = dl.enumerate_valid_sessions(subject=SUBJECT)   # chronological
    print(f"Tag coverage: {len(sessions)} tagged sessions "
          f"(enumerate_valid_sessions). No labeler re-run needed.")
    rows = []
    for sname in sessions:
        labels = dl.load_state_labels(sname, subject=SUBJECT)
        row = session_reliability_row(sname, labels)
        # d' needs the session object (Tier-2 covariate); load only for this.
        sess = load_session(sname)
        row["dprime"] = dl.session_dprime(sess)
        rows.append(row)
        del sess
        gc.collect()
    tab = pd.DataFrame(rows)
    # Session ids are DDMMYYYY: a plain string sort is by day-of-month, NOT
    # chronological. Sort by canonical (year, month, day) so x-axis runs
    # naive -> expert.
    tab = tab.sort_values(
        "session", key=lambda s: s.map(parse_session_date)
    ).reset_index(drop=True)
    return tab


def make_figure(tab):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    # ── Panel A: mean label-confidence vs session d' ──────────────────────────
    reliable = tab["naive_reliable"].values
    axes[0].scatter(
        tab.loc[reliable, "dprime"], tab.loc[reliable, "mean_conf"],
        s=45, color="#4d4d4d", label="reliable", zorder=3,
    )
    axes[0].scatter(
        tab.loc[~reliable, "dprime"], tab.loc[~reliable, "mean_conf"],
        s=70, facecolor="none", edgecolor="#d7301f", linewidth=1.8,
        label="UNRELIABLE (drops to no-mood)", zorder=4,
    )
    # Label the flagged sessions so reviewers can find them on the x-axis.
    for _, r in tab.loc[~reliable].iterrows():
        if np.isfinite(r["dprime"]) and np.isfinite(r["mean_conf"]):
            axes[0].annotate(
                r["session"], (r["dprime"], r["mean_conf"]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=6, color="#d7301f",
            )
    axes[0].axhline(CONF_THRESH, ls="--", lw=0.8, color="#999999")
    axes[0].set_xlabel("session d-prime (sensitivity)")
    axes[0].set_ylabel("mean label confidence")
    axes[0].set_title(
        "How sure is the mood labeler, by performance?\n"
        "(low-d' naive sessions = the watch zone)"
    )
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")

    # ── Panel B: mood composition per session (chronological) ─────────────────
    x = np.arange(len(tab))
    bottom = np.zeros(len(tab))
    for m in MOODS:
        c = STATE_LABEL_COLORS.get(m, "#999999")   # Abort -> grey via palette
        axes[1].bar(x, tab[m], bottom=bottom, label=m, color=c, width=0.85)
        bottom += tab[m].values
    # Flag unreliable sessions: red tick + red x-tick-label.
    for xi, (_, r) in zip(x, tab.iterrows()):
        if not r["naive_reliable"]:
            axes[1].plot([xi], [1.02], marker="v", color="#d7301f",
                         markersize=6, clip_on=False)
    axes[1].set_ylim(0, 1.06)
    axes[1].set_xticks(x)
    xtl = axes[1].set_xticklabels(tab["session"], rotation=90, fontsize=5)
    for xi, lab in zip(x, xtl):
        if not tab["naive_reliable"].iloc[xi]:
            lab.set_color("#d7301f")
            lab.set_fontweight("bold")
    axes[1].set_xlabel("session (chronological: naive -> expert)")
    axes[1].set_ylabel("fraction of trials")
    axes[1].set_title(
        "What moods make up each session?\n"
        "(red triangle / red label = unreliable, mood dropped downstream)"
    )
    axes[1].legend(frameon=False, fontsize=7, ncol=4, loc="lower center",
                   bbox_to_anchor=(0.5, -0.55))

    fig.suptitle(
        "B8 Phase 2 label-reliability check: which naive sessions' moods can we trust?",
        fontsize=11, y=1.04,
    )
    return fig


def main():
    tab = compute_table()

    csv_path = os.path.join(CACHE_DIR, "b8p2_label_reliability.csv")
    tab.to_csv(csv_path, index=False)

    fig = make_figure(tab)
    fig_path = save_fig(fig, "fig_b8_P2_label_reliability")

    # ── Report ────────────────────────────────────────────────────────────────
    cols = ["session", "dprime", "mean_conf", "frac_conf_gt_0.7",
            "naive_reliable"] + MOODS
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\nPer-session reliability summary (chronological):")
    print(tab[cols].to_string(index=False))

    flagged = tab.loc[~tab["naive_reliable"], "session"].tolist()
    print(f"\nFlagged UNRELIABLE sessions "
          f"(<{int(RELIABLE_TRIAL_FRAC * 100)}% of trials with "
          f"confidence > {CONF_THRESH}): {len(flagged)}")
    if flagged:
        for s in flagged:
            r = tab.loc[tab["session"] == s].iloc[0]
            print(f"  {s}  d'={r['dprime']:.2f}  mean_conf={r['mean_conf']:.3f}  "
                  f"frac>{CONF_THRESH}={r['frac_conf_gt_0.7']:.2f}")
    else:
        print("  (none -- every tagged session passes the confidence gate)")

    print(f"\nFigure: {fig_path}")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
