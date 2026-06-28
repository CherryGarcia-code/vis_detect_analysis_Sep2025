"""B8 Phase-2 — comprehension_flag rule SENSITIVITY diagnostic (Task 0.6, fix f).

Plain English: the "two impulsivities" problem — a mouse can lick early because it
genuinely can't yet tell the grating changed (PRE-comprehension impulsivity) or
because it understands the rule but is trigger-happy (POST-comprehension). We split
the chronology into "pre" and "post" with a single boundary session. This script
asks: HOW MUCH does that boundary (and the downstream pre/post trial counts) depend
on WHICH behavioural criterion we use to draw it?

We compare two rules for the pre→post boundary on the SAME chronology:
  * rule="dprime"        — first session with d′ ≥ 0.5 (the default; spec §7).
  * rule="easy_hitrate"  — first session whose EASY-change hit-rate (P(lick) on the
                           largest change-size, Δ=4.0, go trials) ≥ 0.5.

It overlays both boundaries on the d′ / easy-hit-rate trajectory, ANNOTATES which
sessions flip pre↔post between the two rules, and reports the ±1-session
sensitivity of the downstream pre/post TRIAL counts (i.e. if we'd drawn the
boundary one session earlier or later, how much would the pre/post split move).

RECOMMENDATION (recorded here so it boots in a fresh chat): default to
**rule="dprime"**. On the real BG_046 chronology (45 sessions, 16692 trials)
the d′ rule draws the boundary at session_idx 2 (25062025) and the easy-hit-rate
rule at session_idx 1 (24062025) — they AGREE TO WITHIN ONE SESSION (a single
session, 24062025, flips pre↔post between rules), so the comprehension boundary
is ROBUST to the rule choice. ±1-session sensitivity of the default d′ split is
small (pre-fraction 0.2% / 0.3% / 3.6% at −1 / base / +1; a ~3.5 pp swing), so
the downstream pre/post counts barely move. rule="dprime" therefore stands as the
Task-4.2 default. See the printed report + ``fig_b8_P2_comprehension_boundary.png``.

Worktree run recipe:
  WT=$(pwd); PYTHONPATH="$WT/src" \
    py scripts/analysis/decision_latents/_comprehension_flag_explore.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows consoles default to cp1252, which can't encode the non-ASCII glyphs
# (d-prime ', pre<->post, ->) we print below. Force UTF-8 so the report prints
# cleanly instead of crashing with UnicodeEncodeError. matplotlib handles Unicode
# in figure text independently, so titles/captions are unaffected.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass  # older Python / non-reconfigurable stream: prints may still fail, harmless

from visdetect.suite.plotting import setup_style
from visdetect.analysis.config import ROOT, SUBJECT, parse_session_date
from visdetect.analysis import decision_latents as dl

setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents")
TRIAL_CACHE = os.path.join(CACHE_DIR, "decision_latents_trialtable.csv")
os.makedirs(FIG_DIR, exist_ok=True)

THRESHOLD = 0.5          # both rules use the low "knows-the-rule" bar (spec §7)
EASY_CHANGE_SIZE = 4.0   # largest go-trial change ratio = the easiest detection


def save_fig(fig, name):
    """Write a presentation-ready PNG to FIGURES/decision_latents/<SUBJECT>/
    (NOT suite.plotting.save_figure — keep new work out of analysis_suite)."""
    p = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def load_session_trajectory():
    """Return a per-session DataFrame (chronological) with d′, easy-change
    hit-rate, total trials, and the within-session trial count.

    Prefers the cached trial table (whose ``session_dprime`` column is exactly
    ``dl.session_dprime`` per session); falls back to reloading sessions if the
    cache is absent. The easy-change hit-rate is P(lick) over go trials at the
    largest change-size (Δ=4.0)."""
    if os.path.exists(TRIAL_CACHE):
        trials = pd.read_csv(TRIAL_CACHE)
        trials["session_name"] = trials["session_name"].astype(str).str.zfill(8)
        rows = []
        for sname, g in trials.groupby("session_name"):
            easy = g[np.isclose(g["change_size"], EASY_CHANGE_SIZE)]
            hr = float(easy["lick"].mean()) if len(easy) else np.nan
            rows.append({
                "session_name": sname,
                "dprime": float(g["session_dprime"].iloc[0]),
                "easy_hitrate": hr,
                "n_trials": int(len(g)),
                "n_easy": int(len(easy)),
            })
        df = pd.DataFrame(rows)
    else:                      # fallback: reload sessions (slower)
        import gc
        from visdetect.suite.loader import load_session
        rows = []
        for sname in dl.enumerate_valid_sessions():
            sess = load_session(sname)
            sd = dl.session_dprime(sess)
            labels = dl.load_state_labels(sname)
            tab = dl.build_trial_table(sess, labels, sname)
            easy = tab[np.isclose(tab["change_size"], EASY_CHANGE_SIZE)]
            rows.append({
                "session_name": sname.zfill(8), "dprime": sd,
                "easy_hitrate": float(easy["lick"].mean()) if len(easy) else np.nan,
                "n_trials": int(len(tab)), "n_easy": int(len(easy)),
            })
            del sess
            gc.collect()
        df = pd.DataFrame(rows)

    df = df.sort_values("session_name", key=lambda s: s.map(parse_session_date))
    df = df.reset_index(drop=True)
    df["session_idx"] = np.arange(len(df))
    return df


def _boundary_idx(flags, df):
    """Chronological index of the first 'post' session (the boundary), or len(df)
    if no session ever comprehends."""
    for i, sname in enumerate(df["session_name"]):
        if flags.get(sname) == "post":
            return i
    return len(df)


def _split_counts(df, boundary_idx):
    """(pre_trials, post_trials, pre_sessions, post_sessions) for a boundary at the
    given chronological index (sessions [0, boundary_idx) = pre, rest = post)."""
    pre = df.iloc[:boundary_idx]
    post = df.iloc[boundary_idx:]
    return (int(pre["n_trials"].sum()), int(post["n_trials"].sum()),
            len(pre), len(post))


def sensitivity_report(df):
    """Compute both rules' boundaries, the sessions that flip between them, and the
    ±1-session sensitivity of the downstream pre/post trial counts. Returns a dict
    of everything the figure + the printed report need."""
    dprime_by = dict(zip(df["session_name"], df["dprime"]))
    hr_by = dict(zip(df["session_name"], df["easy_hitrate"]))

    flags_dp = dl.assign_comprehension_flags(dprime_by, threshold=THRESHOLD, rule="dprime")
    flags_hr = dl.assign_comprehension_flags(dprime_by, threshold=THRESHOLD,
                                             rule="easy_hitrate", hitrate_by_session=hr_by)
    bidx_dp = _boundary_idx(flags_dp, df)
    bidx_hr = _boundary_idx(flags_hr, df)

    # sessions whose pre/post label DIFFERS between the two rules
    flipped = [s for s in df["session_name"] if flags_dp.get(s) != flags_hr.get(s)]

    # ±1-session sensitivity of the DEFAULT (d′) boundary's downstream counts
    n_sess = len(df)
    base = _split_counts(df, bidx_dp)
    minus1 = _split_counts(df, max(0, bidx_dp - 1))
    plus1 = _split_counts(df, min(n_sess, bidx_dp + 1))
    total_trials = int(df["n_trials"].sum())

    def _pct_pre(counts):
        return 100.0 * counts[0] / total_trials if total_trials else np.nan

    return {
        "flags_dp": flags_dp, "flags_hr": flags_hr,
        "bidx_dp": bidx_dp, "bidx_hr": bidx_hr,
        "boundary_session_dp": df["session_name"].iloc[bidx_dp] if bidx_dp < n_sess else None,
        "boundary_session_hr": df["session_name"].iloc[bidx_hr] if bidx_hr < n_sess else None,
        "flipped_sessions": flipped, "n_flipped": len(flipped),
        "base": base, "minus1": minus1, "plus1": plus1,
        "pct_pre_base": _pct_pre(base), "pct_pre_minus1": _pct_pre(minus1),
        "pct_pre_plus1": _pct_pre(plus1),
        "total_trials": total_trials,
    }


def fig_boundary(df, rep):
    """Overlay both rules' pre→post boundary on the chronological d′ / easy-hit-rate
    trajectory; mark the threshold and any sessions that FLIP between the rules."""
    x = df["session_idx"].values
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(x, df["dprime"], "o-", color="#2b6cb0", lw=1.8, ms=4,
            label="d′ (sensitivity)")
    ax.plot(x, df["easy_hitrate"], "s-", color="#d97706", lw=1.8, ms=4,
            label="easy-change hit-rate  P(lick | Δ=4)")
    ax.axhline(THRESHOLD, ls=":", color="#666666", lw=1.2)
    ax.text(x[-1], THRESHOLD + 0.02, f"threshold = {THRESHOLD}", ha="right",
            va="bottom", fontsize=8, color="#666666")

    # the two boundaries (first 'post' session)
    if rep["bidx_dp"] < len(df):
        ax.axvline(rep["bidx_dp"], color="#2b6cb0", ls="--", lw=2.0)
        ax.text(rep["bidx_dp"], 1.02, "d′ boundary", color="#2b6cb0",
                rotation=90, va="bottom", ha="right", fontsize=8,
                transform=ax.get_xaxis_transform())
    if rep["bidx_hr"] < len(df):
        ax.axvline(rep["bidx_hr"], color="#d97706", ls="--", lw=2.0)
        ax.text(rep["bidx_hr"], 1.02, "easy-hit-rate boundary", color="#d97706",
                rotation=90, va="bottom", ha="left", fontsize=8,
                transform=ax.get_xaxis_transform())

    # shade the chronological span of sessions that FLIP pre↔post between rules
    flip_idx = [int(df.index[df["session_name"] == s][0]) for s in rep["flipped_sessions"]]
    for fi in flip_idx:
        ax.axvspan(fi - 0.5, fi + 0.5, color="#e53e3e", alpha=0.12, zorder=0)
    if flip_idx:
        ax.scatter([df["session_idx"].iloc[i] for i in flip_idx],
                   [df["dprime"].iloc[i] for i in flip_idx],
                   facecolors="none", edgecolors="#e53e3e", s=120, lw=2.0,
                   zorder=5, label=f"flips between rules (n={len(flip_idx)})")

    ax.set_xlabel("session index (chronological, learning →)")
    ax.set_ylabel("d′  /  easy-change hit-rate")
    ax.set_ylim(min(-0.5, float(np.nanmin(df["dprime"])) - 0.1), 1.1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # plain-language sensitivity caption
    b = rep["base"]; m = rep["minus1"]; p = rep["plus1"]
    cap = (
        "Two rules for the pre→post 'comprehension' boundary. The default d′ rule "
        f"draws it at session {rep['bidx_dp']} "
        f"({rep['boundary_session_dp']}); the easy-hit-rate rule at session "
        f"{rep['bidx_hr']} ({rep['boundary_session_hr']}). "
        f"{rep['n_flipped']} session(s) flip pre↔post between the rules (red).  "
        f"±1-session sensitivity of the d′ split: pre-trials = "
        f"{m[0]} / {b[0]} / {p[0]} (−1 / base / +1), "
        f"i.e. pre fraction {rep['pct_pre_minus1']:.0f}% / {rep['pct_pre_base']:.0f}% "
        f"/ {rep['pct_pre_plus1']:.0f}%."
    )
    fig.suptitle("B8 Phase-2 — how the comprehension pre/post boundary depends on the "
                 "rule (d′ vs easy-change hit-rate)", fontsize=12)
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5, wrap=True,
             color="#333333")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return save_fig(fig, "fig_b8_P2_comprehension_boundary")


if __name__ == "__main__":
    df = load_session_trajectory()
    print(f"loaded {len(df)} sessions (chronological), "
          f"total {int(df['n_trials'].sum())} trials")
    rep = sensitivity_report(df)

    print("\n=== comprehension_flag rule comparison ===")
    print(f"  rule='dprime'      boundary: session_idx={rep['bidx_dp']:>2} "
          f"({rep['boundary_session_dp']})")
    print(f"  rule='easy_hitrate' boundary: session_idx={rep['bidx_hr']:>2} "
          f"({rep['boundary_session_hr']})")
    print(f"  sessions flipping pre↔post between rules: {rep['n_flipped']}"
          + (f"  → {rep['flipped_sessions']}" if rep["flipped_sessions"] else ""))

    print("\n=== ±1-session sensitivity of the DEFAULT (d′) downstream split ===")
    for tag, c, pct in [("−1 session", rep["minus1"], rep["pct_pre_minus1"]),
                        ("  base    ", rep["base"], rep["pct_pre_base"]),
                        ("+1 session", rep["plus1"], rep["pct_pre_plus1"])]:
        print(f"  {tag}: pre={c[0]:>5} trials ({c[2]} sess) | "
              f"post={c[1]:>5} trials ({c[3]} sess) | pre-fraction {pct:.1f}%")
    swing = abs(rep["pct_pre_plus1"] - rep["pct_pre_minus1"])
    print(f"  → moving the boundary ±1 session swings the pre-fraction by "
          f"{swing:.1f} percentage points.")

    print("\nfigure:", fig_boundary(df, rep))

    # explicit recommendation for the controller
    print("\n=== RECOMMENDATION ===")
    if rep["n_flipped"] == 0:
        print("  The two rules agree on the boundary (0 flips) → use rule='dprime' "
              "(default; the conventional sensitivity criterion).")
    else:
        print(f"  The two rules disagree on {rep['n_flipped']} session(s). "
              "Default remains rule='dprime' unless the easy-hit-rate boundary is "
              "clearly better-justified by the trajectory above — inspect the figure.")
