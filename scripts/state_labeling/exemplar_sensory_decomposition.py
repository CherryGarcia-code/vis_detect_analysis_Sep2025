"""Sensory change response by state, decomposed by change size — BG_046 17092025 (Expert).

Refines the Change_ON block of the exemplar state figure in response to two points:
  (1) The change-aligned set is Hit/Miss only (trials that reached the change
      period). Here we restrict to *Hit* trials and split by change-size group
      (Small 1.25-1.5x vs Big 2.0-4.0x). Hit-only removes the Hit/Miss-composition
      confound (states differ in hit:miss ratio; Miss trials lack the response
      lick) and is well-powered per state; per-state Hit-vs-Miss is n-limited in an
      Expert session (misses are rare) and is left to a population analysis.
      Catch trials (change_size=1.0, no real change) are excluded from both groups.
  (2) Activity is shown in raw Hz: each state line is a trial subset of the SAME
      unit (no cross-unit averaging), so raw rates are unbiased and let baseline
      (tonic-arousal) state differences remain visible. The early window (0-250 ms,
      pre-lick) is shaded as the sensory-clean readout, since on Hit trials the
      response lick contaminates the later post-change window with motor activity.

State tags: data/cache/state_tags/BG_046/17092025.csv (outcome-statistics labeler).
Alignment respects EVENT_VALID_OUTCOMES. Output:
  figures/state_labeler/BG_046/sensory_decomposition_by_changesize_17092025.png
"""

import os
import gc

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import STATE_LABEL_COLORS
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.utils import build_population_tensor, smooth_psth
from visdetect.analysis.constants import DEFAULT_SIGMA_MS

setup_style()

SESSION = "17092025"
SUBJECT = "BG_046"
STATES = ["Impulsive", "StimSens", "Abort"]
HERO_UNITS = [413, 273, 561]          # top Change_ON-responsive units (by |d'|)
BIN = 0.01
SIGMA_MS = DEFAULT_SIGMA_MS
WINDOW = (-0.5, 1.0)
SENSORY_WIN = (0.0, 0.25)             # early, pre-lick sensory window (shaded)

# change-size groups (go trials only; catch=1.0 excluded)
SIZE_GROUPS = [
    ("Small change\n(1.25-1.5x)", {1.25, 1.35, 1.5}),
    ("Big change\n(2.0-4.0x)",    {2.0, 4.0}),
]

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_TAG_CSV = os.path.join(_REPO, "data", "cache", "state_tags", SUBJECT, f"{SESSION}.csv")


def main():
    print(f"[sensory-decomp] loading {SUBJECT} {SESSION} ...")
    sess = load_session(SESSION)
    tags = pd.read_csv(STATE_TAG_CSV)
    state_of_trial = dict(zip(tags["trial_idx"].astype(int), tags["state_label"]))
    change_size_of = {i: float(getattr(t, "change_size", np.nan))
                      for i, t in enumerate(sess.trials)}

    # Hit trials aligned to Change_ON, for the three hero units
    tensor, bc, valid_trials = build_population_tensor(
        sess, HERO_UNITS, event_name="Change_ON", window=WINDOW,
        bin_size=BIN, outcome_filter={"Hit"},
    )
    assert tensor.shape[2] == len(HERO_UNITS)
    states_arr = np.array([state_of_trial.get(int(ti)) for ti in valid_trials])
    sizes_arr = np.array([change_size_of.get(int(ti), np.nan) for ti in valid_trials])
    del sess
    gc.collect()

    # ---------------------------------------------------------------- plot ---
    nrows, ncols = len(HERO_UNITS), len(SIZE_GROUPS)
    fig = plt.figure(figsize=(9.5, 11))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.45, wspace=0.22,
                           left=0.10, right=0.97, top=0.88, bottom=0.07)

    for ui, cid in enumerate(HERO_UNITS):
        col_unit = ui  # tensor column == HERO_UNITS index
        for si, (size_label, size_set) in enumerate(SIZE_GROUPS):
            ax = fig.add_subplot(gs[ui, si])
            size_mask = np.array([s in size_set for s in sizes_arr])
            ymax = 0.0
            for st in STATES:
                mask = size_mask & (states_arr == st)
                n = int(mask.sum())
                if n == 0:
                    continue
                trials = tensor[mask, :, col_unit]
                mean = smooth_psth(trials.mean(axis=0), BIN, SIGMA_MS)
                sem = smooth_psth(trials.std(axis=0) / np.sqrt(n), BIN, SIGMA_MS)
                color = STATE_LABEL_COLORS[st]
                ax.plot(bc, mean, color=color, lw=1.8, zorder=3, label=f"{st} (n={n})")
                ax.fill_between(bc, mean - sem, mean + sem, color=color,
                                alpha=0.18, lw=0, zorder=2)
                ymax = max(ymax, np.nanmax(mean + sem))

            ax.axvspan(*SENSORY_WIN, color="0.85", alpha=0.5, zorder=0)
            ax.axvline(0, color="k", ls="--", lw=0.9, alpha=0.6, zorder=1)
            ax.set_xlim(bc[0], bc[-1])
            ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, fontsize=7.5, loc="upper left",
                      handlelength=1.0, borderaxespad=0.2)
            if si == 0:
                ax.set_ylabel(f"Unit {cid}\nFiring rate (Hz)", fontsize=10)
            if ui == 0:
                ax.set_title(size_label, fontsize=11, fontweight="bold")
            if ui == nrows - 1:
                ax.set_xlabel("Time from change onset (s)")

    fig.suptitle(
        f"Sensory change response by state, split by change size — "
        f"{SUBJECT} {SESSION} (Expert)\n"
        "Hit trials only (outcome-controlled);  shaded = early pre-lick sensory window",
        fontsize=12.5, fontweight="bold", y=0.975,
    )
    save_figure(fig, f"sensory_decomposition_by_changesize_{SESSION}",
                f"state_labeler/{SUBJECT}")
    plt.close(fig)
    print("[sensory-decomp] done.")


if __name__ == "__main__":
    main()
