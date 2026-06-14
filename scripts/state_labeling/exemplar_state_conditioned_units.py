"""State-conditioned single-unit activity — BG_046 Expert session 17092025.

Demonstrates how the activity of well-chosen *responsive* striatal units differs
across the behavioral states defined by the new outcome-statistics state labeler
(Impulsive / StimSens / Abort), replacing the old GLM-HMM.

Layout: PSTH small-multiples grid (3 rows x 3 units). Each panel overlays the
three state PSTHs (mean +/- SEM, raw Hz) for one unit. Three alignment blocks:

  Row A  Sensory     align Change_ON (Hit+Miss)  -> top Change_ON-responsive units
  Row B  Motor (FA)  align FA lick   (FA)         -> top peri-FA-lick modulated units
  Row C  Motor (Hit) align Hit lick  (Hit)        -> top peri-Hit-lick modulated units

Why this design / grounding:
  - State names predict the neural contrast: a "StimSens" (engaged) regime should
    show a stronger sensory response at Change_ON than the "Impulsive" or "Abort"
    regimes; lick-locked motor units let us ask whether the *motor* response is
    state-invariant while the *sensory* response is state-dependent
    (sensory/decision selected downstream — synthesis-batch03-rodent-perception;
    state = different cue->action mapping, not different actions —
    synthesis-phase3-behavioral-state / Calhoun GLM-HMM).
  - Alignment respects EVENT_VALID_OUTCOMES (Change_ON only Hit/Miss; FA only fa;
    Hit only hit) — fa/abort trials never saw a change.
  - Per-unit PSTHs are shown in raw Hz: each state is a different trial subset of
    the *same* unit, so within-unit raw rates are valid (no cross-unit averaging;
    CLAUDE.md normalization rules).

State tags: data/cache/state_tags/BG_046/17092025.csv (drop-in hmm_downstream aliases).
Output: figures/state_labeler/BG_046/state_conditioned_exemplar_units_17092025.png
"""

import os
import sys
import gc

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from visdetect.suite.config import STATE_LABEL_COLORS, CACHE_DIR
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.utils import get_good_cluster_ids, build_population_tensor, smooth_psth
from visdetect.analysis.constants import DEFAULT_SIGMA_MS

setup_style()

# ----------------------------------------------------------------------------
SESSION = "17092025"
SUBJECT = "BG_046"
STATES = ["Impulsive", "StimSens", "Abort"]   # Disengaged absent this session
BIN = 0.01
SIGMA_MS = DEFAULT_SIGMA_MS
N_UNITS_PER_BLOCK = 3

# State tags live in data/cache/state_tags; resolve against the repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_TAG_CSV = os.path.join(
    _REPO, "data", "cache", "state_tags", SUBJECT, f"{SESSION}.csv"
)
RESP_CACHE = os.path.join(
    _REPO, "analysis_suite", "cache", "responsiveness_all_sessions.csv"
)

# Block configuration: (row_label, event, outcome_filter, window, base_win, resp_win)
BLOCKS = [
    ("Sensory  —  aligned to Change onset",
     "Change_ON", {"Hit", "Miss"}, (-0.5, 1.0), (-0.4, -0.05), (0.0, 0.25)),
    ("Motor  —  aligned to early (FA) lick",
     "FA", {"FA"}, (-1.0, 0.6), (-1.0, -0.6), (-0.15, 0.15)),
    ("Motor  —  aligned to rewarded (Hit) lick",
     "Hit", {"Hit"}, (-1.0, 0.6), (-1.0, -0.6), (-0.15, 0.15)),
]


def per_state_psth(tensor, valid_trials, state_of_trial, unit_col):
    """Return {state: (mean_hz, sem_hz, n)} for one unit column, smoothed."""
    labels = np.array([state_of_trial.get(int(ti), None) for ti in valid_trials])
    out = {}
    for st in STATES:
        mask = labels == st
        n = int(mask.sum())
        if n == 0:
            out[st] = (None, None, 0)
            continue
        trials = tensor[mask, :, unit_col]                 # (n, bins) Hz
        mean = smooth_psth(trials.mean(axis=0), BIN, SIGMA_MS)
        sem = smooth_psth(trials.std(axis=0) / np.sqrt(max(n, 1)), BIN, SIGMA_MS)
        out[st] = (mean, sem, n)
    return out


def main():
    print(f"[exemplar] loading {SUBJECT} {SESSION} ...")
    sess = load_session(SESSION)
    good_ids = list(get_good_cluster_ids(sess))
    print(f"  good units: {len(good_ids)}")

    # state assignment: trial_idx -> state_label (argmax assignment, ungated)
    tags = pd.read_csv(STATE_TAG_CSV)
    state_of_trial = dict(zip(tags["trial_idx"].astype(int), tags["state_label"]))

    # responsiveness cache (Change_ON) for Block A unit ranking
    resp = pd.read_csv(RESP_CACHE)
    resp = resp[resp["session_name"].astype(str) == SESSION].copy()

    block_results = []   # list of dicts: {row_label, window, event, units:[{cid, metric, psth}]}
    used_units = set()   # keep all 9 exemplars distinct across blocks

    for row_label, event, ofilter, window, base_win, resp_win in BLOCKS:
        print(f"  block [{event}] building tensor ...")
        tensor, bc, valid_trials = build_population_tensor(
            sess, good_ids, event_name=event, window=window,
            bin_size=BIN, outcome_filter=ofilter,
        )
        assert tensor.shape[2] == len(good_ids)

        # ---- unit selection ----
        if event == "Change_ON":
            # rank by |d'| from responsiveness screen, restricted to good units
            cand = resp[resp["cluster_id"].isin(good_ids)].copy()
            cand["abs_dp"] = cand["dprime"].abs()
            cand = cand.sort_values("abs_dp", ascending=False)
            ranked_ids = cand["cluster_id"].tolist()
            metric_of = dict(zip(cand["cluster_id"], cand["dprime"]))
            metric_name = "d'"
        else:
            # rank by |peri-lick modulation| computed from this tensor
            base_mask = (bc >= base_win[0]) & (bc < base_win[1])
            resp_mask = (bc >= resp_win[0]) & (bc < resp_win[1])
            per_unit = tensor.mean(axis=0)                  # (bins, units)
            delta = per_unit[resp_mask, :].mean(axis=0) - per_unit[base_mask, :].mean(axis=0)
            order = np.argsort(-np.abs(delta))
            ranked_ids = [good_ids[c] for c in order]
            metric_of = {good_ids[c]: float(delta[c]) for c in range(len(good_ids))}
            metric_name = "ΔHz"

        # take the top-ranked units not already shown in an earlier block
        chosen_ids = []
        for cid in ranked_ids:
            if cid in used_units:
                continue
            chosen_ids.append(cid)
            if len(chosen_ids) == N_UNITS_PER_BLOCK:
                break
        used_units.update(chosen_ids)

        units = []
        for cid in chosen_ids:
            col = good_ids.index(cid)
            units.append({
                "cid": cid,
                "metric": metric_of.get(cid, np.nan),
                "metric_name": metric_name,
                "psth": per_state_psth(tensor, valid_trials, state_of_trial, col),
            })
        block_results.append({
            "row_label": row_label, "event": event, "bc": bc, "units": units,
        })
        del tensor
        gc.collect()

    del sess
    gc.collect()

    # ---------------------------------------------------------------- plot ---
    nrows, ncols = len(BLOCKS), N_UNITS_PER_BLOCK
    fig = plt.figure(figsize=(13, 11))
    gs = gridspec.GridSpec(
        nrows, ncols, hspace=0.62, wspace=0.28,
        left=0.09, right=0.97, top=0.88, bottom=0.07,
    )

    for r, block in enumerate(block_results):
        bc = block["bc"]
        for c, unit in enumerate(block["units"]):
            ax = fig.add_subplot(gs[r, c])
            ymax = 0.0
            for st in STATES:
                mean, sem, n = unit["psth"][st]
                if mean is None:
                    continue
                color = STATE_LABEL_COLORS[st]
                ax.plot(bc, mean, color=color, lw=1.8, zorder=3,
                        label=f"{st} (n={n})")
                ax.fill_between(bc, mean - sem, mean + sem, color=color,
                                alpha=0.18, lw=0, zorder=2)
                ymax = max(ymax, np.nanmax(mean + sem))

            ax.axvline(0, color="k", ls="--", lw=0.9, alpha=0.6, zorder=1)
            ax.set_xlim(bc[0], bc[-1])
            ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            mname = unit["metric_name"]
            ax.set_title(f"Unit {unit['cid']}  ({mname}={unit['metric']:+.2f})",
                         fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Firing rate (Hz)")
            if r == nrows - 1:
                ax.set_xlabel(f"Time from {block['event'].replace('_',' ')} (s)")
            # per-panel state legend (small, since n differs per panel only by row)
            ax.legend(frameon=False, fontsize=7.5, loc="upper left",
                      handlelength=1.0, borderaxespad=0.2)

        # row band label on the left margin + centered descriptor above the row
        y_top = gs[r, 0].get_position(fig).y1
        fig.text(0.012, y_top + 0.030, f"{chr(65 + r)}",
                 fontsize=14, fontweight="bold", va="bottom")
        fig.text(0.53, y_top + 0.034, block["row_label"],
                 ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color="#333333")

    fig.suptitle(
        f"State-conditioned single-unit activity  —  {SUBJECT}  {SESSION} (Expert)\n"
        "behavioral-state labeler:  Impulsive / StimSens / Abort",
        fontsize=13, fontweight="bold", y=0.985,
    )

    save_figure(fig, f"state_conditioned_exemplar_units_{SESSION}",
                f"state_labeler/{SUBJECT}")
    plt.close(fig)
    print("[exemplar] done.")


if __name__ == "__main__":
    main()
