"""Manual SESSION-level sorter — hand-label whole sessions into behavioural groups.

Mirrors the proven trial-level state labeler (scripts/state_labeling/run_state_labeler.py):
human paints labels on a matplotlib view -> autosave CSV -> a separate script fits an
interpretable tree and validates it with Cohen's kappa.

DESIGN (see docs/superpowers/specs/2026-07-31-session-grouping-and-learning-axis.md):
  * BLINDED. Sessions are shown in a fixed RANDOM order with stage / date / session id /
    d' HIDDEN. If you can see "this is session 40" your labels inherit the d'-based staging
    we are trying to replace, and "groups track learning" becomes a tautology.
  * TEST-RETEST. ~15% of sessions are silently presented TWICE. That yields your own
    self-consistency kappa — if your labels are not reliable, no rule fitted to them can be.
  * ALL tagged sessions are queued, including manifest-`Excluded` ones: those are exactly
    the disengaged/bad days this exercise is meant to characterise, and queueing them avoids
    re-importing the d' gate.

Reads the per-session state-tag CSVs (data/cache/state_tags/<SUBJ>/<session>.csv), which
already carry trial_idx / outcome / lick_valence / color / state_label / state_confidence —
so NO pkl loading is needed and navigation is instant.

Keys:  1-5 assign a group | u unsure/skip | <- -> prev/next | n note (console) | q quit+save
Autosaves (atomic upsert) after every label, so a crash loses nothing; re-running resumes
at the first unlabelled presentation.

Run: py scripts/session_sorting/run_session_sorter.py --subject BG_046
Out: data/cache/session_sorting/manual_session_labels.csv
     data/cache/session_sorting/<SUBJ>_presentation_queue.csv
"""
import os
import sys
import argparse
from datetime import datetime, timezone

# TkAgg BEFORE pyplot (re-asserted after the visdetect imports below — qc.py / tf_pulse.py
# call matplotlib.use("Agg") at import time and would otherwise clobber the interactive backend).
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from visdetect.analysis.config import STATE_LABEL_COLORS, canonical_session_id  # noqa: E402

# Re-assert the interactive backend AFTER visdetect imports, then fail fast.
matplotlib.use("TkAgg", force=True)
plt.switch_backend("TkAgg")
if "tkagg" not in matplotlib.get_backend().lower():
    raise SystemExit(f"Interactive backend did not stick (got {matplotlib.get_backend()!r}). "
                     "Cannot run the GUI.")

# ── Label taxonomy (agreed 2026-07-31). Changing this AFTER labelling starts
# invalidates existing labels — bump the file name if you must change it. ──────
GROUPS = [
    "Balanced",              # sustained engagement, both states well represented
    "Impulsive-dominated",   # pervasive early / anticipatory licking
    "Disengaged-dominated",  # long non-responding stretches, low participation
    "Deteriorating",         # starts fine then falls apart (a DYNAMICS category)
    "Low-yield",             # too few trials / too fragmented to judge
]
GROUP_COLORS = {
    "Balanced": "#41ab5d", "Impulsive-dominated": "#ef6548",
    "Disengaged-dominated": "#3474ae", "Deteriorating": "#d9a441",
    "Low-yield": "#969696", "Unsure": "#cccccc",
}
ROLL_W = 31          # rolling window (trials) — matches STATE_LABEL_W_DEFAULT
REPEAT_FRAC = 0.15   # fraction of sessions silently shown twice (test-retest)

TAG_DIR = os.path.join(_ROOT, "data", "cache", "state_tags")
OUT_DIR = os.path.join(_ROOT, "data", "cache", "session_sorting")
LABELS_PATH = os.path.join(OUT_DIR, "manual_session_labels.csv")
LABEL_COLUMNS = ["subject", "session_name", "presentation_idx", "is_repeat",
                 "group", "confidence", "labeler", "timestamp", "notes"]


# Columns a real per-session tag file must have (the directory also holds
# roll-up files like `_tag_summary.csv` which are NOT sessions — skip them).
REQUIRED_TAG_COLS = {"trial_idx", "outcome", "is_go", "is_hit", "is_fa",
                     "color", "state_label"}


def list_tagged_sessions(subject):
    """Session ids with a valid per-session tag CSV (excludes `_`-prefixed roll-ups)."""
    tdir = os.path.join(TAG_DIR, subject)
    out = []
    for f in sorted(os.listdir(tdir)):
        if not f.endswith(".csv") or f.startswith("_"):
            continue
        cols = set(pd.read_csv(os.path.join(tdir, f), nrows=0).columns)
        if REQUIRED_TAG_COLS <= cols:
            out.append(f[:-4])
        else:
            print(f"  skip {f}: missing {sorted(REQUIRED_TAG_COLS - cols)}")
    return out


# ── Queue ─────────────────────────────────────────────────────────────
def build_queue(subject, seed=42):
    """Fixed blinded presentation order (persisted so resume/repeats are stable)."""
    qpath = os.path.join(OUT_DIR, f"{subject}_presentation_queue.csv")
    if os.path.exists(qpath):
        return pd.read_csv(qpath, dtype={"session_name": str})
    sessions = list_tagged_sessions(subject)
    if not sessions:
        raise SystemExit(f"No per-session state-tag CSVs in {os.path.join(TAG_DIR, subject)}")
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(sessions))
    n_rep = max(1, int(round(REPEAT_FRAC * len(sessions))))
    repeats = list(rng.choice(sessions, size=n_rep, replace=False))
    rows = [{"session_name": s, "is_repeat": False} for s in order]
    # insert repeats in the back half so they are well separated from the original
    for s in repeats:
        pos = int(rng.integers(len(rows) // 2, len(rows) + 1))
        rows.insert(pos, {"session_name": s, "is_repeat": True})
    q = pd.DataFrame(rows)
    q["presentation_idx"] = range(len(q))
    os.makedirs(OUT_DIR, exist_ok=True)
    q.to_csv(qpath, index=False)
    print(f"Built presentation queue: {len(sessions)} sessions + {n_rep} repeats "
          f"= {len(q)} presentations -> {qpath}")
    return q


# ── Label IO (atomic upsert, keyed on presentation_idx) ───────────────
def load_labels():
    if os.path.exists(LABELS_PATH) and os.path.getsize(LABELS_PATH) > 0:
        return pd.read_csv(LABELS_PATH, dtype={"session_name": str, "notes": str}).fillna(
            {"notes": ""})
    return pd.DataFrame(columns=LABEL_COLUMNS)


def save_label(rec):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_labels()
    key = (df["subject"] == rec["subject"]) & (df["presentation_idx"] == rec["presentation_idx"]) \
        if len(df) else pd.Series([], dtype=bool)
    if len(df) and key.any():
        for k, v in rec.items():
            df.loc[key, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df = df.reindex(columns=LABEL_COLUMNS)
    tmp = LABELS_PATH + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, LABELS_PATH)          # atomic
    return df


# ── Per-session view ──────────────────────────────────────────────────
def load_tags(subject, session_name):
    return pd.read_csv(os.path.join(TAG_DIR, subject, f"{session_name}.csv"),
                       dtype={"session_name": str})


def draw(fig, axes, tags, pres_i, n_pres, current_label):
    ax_r, ax_s, ax_c, ax_t = axes
    for a in axes:
        a.clear()
    n = len(tags)
    x = np.arange(n)

    # 1) outcome raster (per-trial colour already in the tag file)
    ax_r.bar(x, 1.0, width=1.0, color=tags["color"].values, linewidth=0)
    ax_r.set_xlim(-0.5, n - 0.5); ax_r.set_ylim(0, 1)
    ax_r.set_yticks([]); ax_r.set_xticklabels([])
    ax_r.set_ylabel("outcome", fontsize=9)

    # 2) state strip (dim low-confidence trials)
    cols = tags["state_label"].map(STATE_LABEL_COLORS).fillna("#eeeeee").values
    alphas = np.where(tags.get("state_gated", pd.Series(1, index=tags.index)).values == -1, 0.30, 1.0)
    ax_s.bar(x, 1.0, width=1.0, color=cols, alpha=None, linewidth=0)
    for xi, a in zip(x[alphas < 1], alphas[alphas < 1]):
        ax_s.bar([xi], [1.0], width=1.0, color="white", alpha=0.65, linewidth=0)
    ax_s.set_xlim(-0.5, n - 0.5); ax_s.set_ylim(0, 1)
    ax_s.set_yticks([]); ax_s.set_xticklabels([])
    ax_s.set_ylabel("state", fontsize=9)

    # 3) rolling behaviour: hit rate on go trials + early-lick fraction
    is_go = tags["is_go"].astype(bool).values
    is_hit = tags["is_hit"].astype(bool).values
    is_fa = tags["is_fa"].astype(bool).values
    r = lambda v: pd.Series(v.astype(float)).rolling(ROLL_W, center=True, min_periods=5).mean().values
    go_sum = pd.Series(is_go.astype(float)).rolling(ROLL_W, center=True, min_periods=5).sum().values
    hit_sum = pd.Series((is_hit & is_go).astype(float)).rolling(ROLL_W, center=True, min_periods=5).sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        hit_rate = np.where(go_sum > 0, hit_sum / go_sum, np.nan)
    ax_c.plot(x, hit_rate, color="#238b45", lw=1.6, label="hit rate (go)")
    ax_c.plot(x, r(is_fa), color="#ef6548", lw=1.6, label="early-lick frac")
    ax_c.set_xlim(-0.5, n - 0.5); ax_c.set_ylim(0, 1)
    ax_c.set_xlabel("trial", fontsize=9)
    ax_c.set_ylabel(f"rolling ({ROLL_W})", fontsize=9)
    ax_c.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)

    # 4) blinded info + key help + running occupancy
    occ = tags["state_label"].value_counts(normalize=True)
    occ_txt = "   ".join(f"{k[:4]} {occ.get(k, 0.0):.2f}"
                         for k in ["StimSens", "Impulsive", "Disengaged", "Abort"])
    ax_t.axis("off")
    lab = current_label or "— unlabelled —"
    ax_t.text(0.0, 0.78, f"presentation {pres_i + 1} / {n_pres}      n_trials = {n}",
              fontsize=11, fontweight="bold", transform=ax_t.transAxes)
    ax_t.text(0.0, 0.50, f"state occupancy:  {occ_txt}", fontsize=9,
              color="0.3", transform=ax_t.transAxes)
    keyhelp = "  |  ".join(f"[{i+1}] {g}" for i, g in enumerate(GROUPS))
    ax_t.text(0.0, 0.24, keyhelp + "  |  [u] unsure", fontsize=8.5,
              color="0.25", transform=ax_t.transAxes)
    ax_t.text(0.0, 0.02, "← → navigate   n note   q quit+save", fontsize=8,
              color="0.45", transform=ax_t.transAxes)
    ax_t.text(1.0, 0.78, lab, fontsize=12, fontweight="bold", ha="right",
              color=GROUP_COLORS.get(current_label, "0.6"), transform=ax_t.transAxes)

    fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--labeler", default=os.environ.get("USERNAME", "BG"))
    ap.add_argument("--start-idx", type=int, default=None,
                    help="jump to a presentation index (default: first unlabelled)")
    args = ap.parse_args()

    queue = build_queue(args.subject)
    labels = load_labels()
    done = set(labels.loc[labels["subject"] == args.subject, "presentation_idx"].astype(int)) \
        if len(labels) else set()
    start = args.start_idx
    if start is None:
        todo = [i for i in queue["presentation_idx"] if int(i) not in done]
        start = int(todo[0]) if todo else 0
    print(f"{args.subject}: {len(done)}/{len(queue)} presentations labelled. Starting at {start}.")

    # kill default matplotlib keymaps so our single-key bindings are unambiguous
    for p in list(plt.rcParams):
        if p.startswith("keymap."):
            plt.rcParams[p] = []

    fig = plt.figure(figsize=(15, 8.5))
    # state strip gets real estate: it is what the eye actually judges
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.5, 1.3, 2.4, 1.1], hspace=0.22,
                           left=0.06, right=0.98, top=0.96, bottom=0.07)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    state = {"i": int(start), "tags": None}

    def current_label_for(pidx):
        d = load_labels()
        m = (d["subject"] == args.subject) & (d["presentation_idx"] == pidx)
        return str(d.loc[m, "group"].iloc[0]) if len(d) and m.any() else None

    def show():
        row = queue.iloc[state["i"]]
        state["tags"] = load_tags(args.subject, str(row["session_name"]))
        draw(fig, axes, state["tags"], state["i"], len(queue),
             current_label_for(int(row["presentation_idx"])))

    def assign(group):
        row = queue.iloc[state["i"]]
        save_label({
            "subject": args.subject, "session_name": str(row["session_name"]),
            "presentation_idx": int(row["presentation_idx"]),
            "is_repeat": bool(row["is_repeat"]), "group": group, "confidence": "high",
            "labeler": args.labeler,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": "",
        })
        print(f"  [{state['i'] + 1}/{len(queue)}] -> {group}")
        if state["i"] < len(queue) - 1:
            state["i"] += 1
        show()

    def on_key(event):
        k = (event.key or "").lower()
        if k in {str(i + 1) for i in range(len(GROUPS))}:
            assign(GROUPS[int(k) - 1])
        elif k == "u":
            assign("Unsure")
        elif k in ("right", "down"):
            state["i"] = min(state["i"] + 1, len(queue) - 1); show()
        elif k in ("left", "up"):
            state["i"] = max(state["i"] - 1, 0); show()
        elif k == "n":
            note = input("note> ").strip()
            row = queue.iloc[state["i"]]
            d = load_labels()
            m = (d["subject"] == args.subject) & (d["presentation_idx"] == int(row["presentation_idx"]))
            if len(d) and m.any():
                rec = d.loc[m].iloc[0].to_dict(); rec["notes"] = note; save_label(rec)
                print("  note saved")
            else:
                print("  (label the session first, then add a note)")
            show()
        elif k == "q":
            print(f"\nSaved -> {LABELS_PATH}")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()


if __name__ == "__main__":
    main()
