"""B8 Phase 2 — Task 0.8: Expert-anchor data inventory (fix h, part 1).

Plain English: the Phase-2 generative fit is seeded "expert-first" — it learns
the dials on the mouse's most expert sessions, then walks backwards through
learning. That backward seed needs a handful of genuinely expert anchor
sessions. This script inventories every usable session and decides which ones
qualify as an "expert anchor": the mouse saw the change clearly (d' > 0.7) AND
both of the two engaged moods (Impulsive, StimSens) carried enough trials
(n >= 20) to fit anything.

The expert-subset SIZE drives the Task 0.9 contingency:
  * >= 3 expert anchors  -> Task 0.9 runs in `expert` (backward-seeded) mode.
  * <  3 expert anchors  -> Task 0.9 falls back to `pooled` mode.

Deliverables (both under the canonical decision-latents cache / FIGURES dirs):
  * data/cache/decision_latents/b8p2_expert_anchor_inventory.csv  — one row per
    session: dprime, n_imp, n_stim, usable_gen cell count, is_expert_anchor.
  * FIGURES/decision_latents/<SUBJECT>/fig_b8_P2_expert_anchor_inventory.png —
    d' x per-mood-n scatter, expert subset highlighted (presentation-ready).

Worktree run recipe:
  WT=$(pwd); PYTHONPATH="$WT/src" py scripts/analysis/decision_latents/_expert_anchor_inventory.py
"""
from __future__ import annotations
import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# cp1252 console guard — non-ASCII (d', mu, ...) must not crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style          # styling only
from visdetect.analysis.config import ROOT, SUBJECT, STATE_LABEL_COLORS
from visdetect.analysis import decision_latents as dl

setup_style()

FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
CSV_PATH = os.path.join(CACHE_DIR, "b8p2_expert_anchor_inventory.csv")
FIG_NAME = "fig_b8_P2_expert_anchor_inventory"

# Expert-anchor thresholds (fix h, part 1; brief Task 0.8).
EXPERT_DPRIME_MIN = 0.7    # the change was seen clearly enough to be "expert"
EXPERT_PER_MOOD_N_MIN = 20  # both engaged moods need a populated cell to fit on
ANCHOR_MOODS = ("Impulsive", "StimSens")  # the two ENGAGED moods (Disengaged excluded)


def save_fig(fig, name):
    """Write to top-level FIGURES/, NOT analysis_suite (per repo convention)."""
    p = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def build_inventory():
    """One row per valid session: d', per-mood n_trials, per-mood usable_gen cell
    count, and the is_expert_anchor verdict. Built the SAME way the Phase-1
    pipeline scores cells (build_trial_table -> compute_cell_qc per mood-slice),
    so the inventory is self-contained and re-runnable from raw sessions."""
    sessions = dl.enumerate_valid_sessions()              # chronological order
    rows = []
    for sname in sessions:
        sess = load_session(sname)
        dprime = dl.session_dprime(sess)
        labels = dl.load_state_labels(sname)
        tab = dl.build_trial_table(sess, labels, sname)
        del sess
        gc.collect()

        rec = {"session_name": sname, "dprime": dprime}
        usable_gen_total = 0
        per_mood_n = {}
        for mood in ANCHOR_MOODS:
            cell = tab[tab["state_label"] == mood]
            n = int(len(cell))
            per_mood_n[mood] = n
            if n > 0:
                qc = dl.compute_cell_qc(cell)
                ug = bool(qc["usable_generative"])
            else:
                ug = False
            rec[f"n_{mood.lower()[:4]}"] = n     # n_impu / n_stim
            rec[f"usable_gen_{mood.lower()[:4]}"] = int(ug)
            usable_gen_total += int(ug)
        rec["usable_gen_cells"] = usable_gen_total
        # expert anchor: d' > 0.7 AND BOTH engaged moods have n >= 20 trials
        both_populated = all(per_mood_n[m] >= EXPERT_PER_MOOD_N_MIN for m in ANCHOR_MOODS)
        rec["is_expert_anchor"] = bool(
            np.isfinite(dprime) and dprime > EXPERT_DPRIME_MIN and both_populated)
        rows.append(rec)

    df = pd.DataFrame(rows)
    # stable, readable column order
    cols = ["session_name", "dprime", "n_impu", "n_stim",
            "usable_gen_impu", "usable_gen_stim", "usable_gen_cells",
            "is_expert_anchor"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def fig_inventory(df):
    """d' (x) x per-mood min-n (y) scatter; expert anchors highlighted.

    y = min(n_impu, n_stim) — the LIMITING engaged-mood cell, because an anchor
    needs BOTH moods populated. Dashed guides mark the d' > 0.7 and n >= 20 gates;
    points in the top-right quadrant that clear both ARE the expert anchors.
    Color = the limiting mood, ringed black if it is an expert anchor."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    d = df["dprime"].values.astype(float)
    n_imp = df["n_impu"].values.astype(float)
    n_stim = df["n_stim"].values.astype(float)
    min_n = np.minimum(n_imp, n_stim)
    # color each point by which engaged mood is the limiting (smaller-n) one
    limiting_is_imp = n_imp <= n_stim
    colors = np.where(limiting_is_imp,
                      STATE_LABEL_COLORS["Impulsive"], STATE_LABEL_COLORS["StimSens"])
    is_anchor = df["is_expert_anchor"].values.astype(bool)

    # axis extents (computed up-front so the shaded zone + annotations are placed correctly)
    finite_d = d[np.isfinite(d)]
    xmax = max(np.nanmax(finite_d) * 1.05, EXPERT_DPRIME_MIN + 0.3) if finite_d.size else 2.0
    xmin = min(np.nanmin(finite_d) - 0.1, 0.0) if finite_d.size else 0.0
    ymax = max(np.nanmax(min_n) * 1.08, EXPERT_PER_MOOD_N_MIN + 10) if min_n.size else 100

    # shade the expert quadrant (drawn first, behind everything)
    ax.fill_betweenx([EXPERT_PER_MOOD_N_MIN, ymax], EXPERT_DPRIME_MIN, xmax,
                     color="#ffe9a8", alpha=0.35, zorder=0)

    # gate guides
    ax.axvline(EXPERT_DPRIME_MIN, ls="--", color="#888888", lw=1.2)
    ax.axhline(EXPERT_PER_MOOD_N_MIN, ls="--", color="#888888", lw=1.2)
    ax.text(EXPERT_DPRIME_MIN + 0.02, ymax, f"d' > {EXPERT_DPRIME_MIN}",
            rotation=90, va="top", ha="left", fontsize=8, color="#888888")

    # non-anchor points (faint), then anchors (ringed) on top
    for sel, is_anc in [(~is_anchor, False), (is_anchor, True)]:
        if not np.any(sel):
            continue
        kw = (dict(s=130, alpha=0.95, edgecolors="k", linewidths=1.8, zorder=3) if is_anc
              else dict(s=55, alpha=0.55, edgecolors="none", zorder=2))
        ax.scatter(d[sel], min_n[sel], c=colors[sel], **kw)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("session sensitivity  d'  (how clearly the change was seen)")
    ax.set_ylabel("trials in the THINNER engaged mood\nmin(n Impulsive, n StimSens)")

    n_anchor = int(is_anchor.sum())
    ax.set_title(
        "Expert-anchor inventory for the backward-seeded fit\n"
        f"a session is an expert anchor if d' > {EXPERT_DPRIME_MIN} AND both engaged "
        f"moods have >= {EXPERT_PER_MOOD_N_MIN} trials  ->  {n_anchor} qualify")
    # legend proxies for the two limiting-mood colors + anchor ring
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", label="limited by Impulsive",
               markerfacecolor=STATE_LABEL_COLORS["Impulsive"], markersize=9),
        Line2D([0], [0], marker="o", color="w", label="limited by StimSens",
               markerfacecolor=STATE_LABEL_COLORS["StimSens"], markersize=9),
        Line2D([0], [0], marker="o", color="w", label="expert anchor",
               markerfacecolor="#cccccc", markeredgecolor="k", markeredgewidth=1.8,
               markersize=11),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.text(0.5, -0.02,
             "Each dot is one session. The y-axis is the SMALLER of the two engaged-mood "
             "trial counts, because\nan anchor needs BOTH moods populated. Ringed dots in "
             "the shaded zone are the expert anchors that\nseed the Phase-2 backward fit "
             "(>= 3 -> expert mode; < 3 -> pooled fallback).",
             ha="center", va="top", fontsize=8, color="#555555")
    fig.tight_layout()
    return save_fig(fig, FIG_NAME)


def main():
    df = build_inventory()
    df.to_csv(CSV_PATH, index=False)
    fig_path = fig_inventory(df)

    anchors = df[df["is_expert_anchor"]]
    anchor_ids = list(anchors["session_name"].astype(str))
    n_anchor = len(anchor_ids)

    # ── full per-session inventory summary ──────────────────────────────────
    print("=" * 78)
    print("B8 Phase 2 — Task 0.8: Expert-anchor data inventory")
    print("=" * 78)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        show = df.copy()
        show["dprime"] = show["dprime"].map(lambda x: f"{x:.3f}")
        print(show.to_string(index=False))
    print("-" * 78)
    print(f"Total valid sessions inventoried : {len(df)}")
    print(f"Expert-anchor threshold          : d' > {EXPERT_DPRIME_MIN} AND "
          f"both engaged moods n >= {EXPERT_PER_MOOD_N_MIN}")
    print(f"EXPERT-ANCHOR COUNT              : {n_anchor}")
    print(f"Expert-anchor session ids        : {anchor_ids}")
    mode = "expert" if n_anchor >= 3 else "pooled/fallback"
    print(f"Task 0.9 contingency             : {n_anchor} anchors -> '{mode}' mode "
          f"({'>=' if n_anchor >= 3 else '<'} 3)")
    print("-" * 78)
    print(f"CSV    : {CSV_PATH}")
    print(f"Figure : {fig_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
