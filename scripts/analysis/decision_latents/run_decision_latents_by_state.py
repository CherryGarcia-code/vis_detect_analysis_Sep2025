"""B8 Fig: decision-latents by state (Step 1, descriptive).

Plain English: measures three behavioural 'dials' — sharpness (can it tell the
change happened), itchiness (is it trigger-happy), timing (does it expect the
change now) — split by mood (Impulsive vs StimSens), across learning, and
saves them as figures + a per-trial table.

Worktree run recipe:
  WT=$(pwd); PYTHONPATH="$WT/src" py scripts/analysis/decision_latents/run_decision_latents_by_state.py
"""
import os, sys, gc, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style          # styling only
from visdetect.analysis.config import ROOT, SUBJECT, STATE_LABEL_COLORS  # canonical new-labeler mood palette
from visdetect.analysis import decision_latents as dl
setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents")
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)
CACHE = os.path.join(CACHE_DIR, "decision_latents_by_state.csv")        # deliverable: per-trial LATENT table
TRIAL_CACHE = os.path.join(CACHE_DIR, "decision_latents_trialtable.csv")  # raw per-trial table (build() cache)

def save_fig(fig, name):                       # writes to top-level FIGURES/, not analysis_suite/
    p = os.path.join(FIG_DIR, f"{name}.png"); fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig); return p

def build(force=False):
    if os.path.exists(TRIAL_CACHE) and not force:
        return pd.read_csv(TRIAL_CACHE)
    sessions = dl.enumerate_valid_sessions()
    dprime = {}
    parts = []
    for sname in sessions:
        sess = load_session(sname)
        dprime[sname] = dl.session_dprime(sess)
        labels = dl.load_state_labels(sname)
        parts.append((sname, dl.build_trial_table(sess, labels, sname)))
        del sess; gc.collect()
    flags = dl.assign_comprehension_flags(dprime)
    frames = []
    for sname, tab in parts:
        tab["session_dprime"] = dprime[sname]; tab["comprehension_flag"] = flags[sname]
        frames.append(tab)
    all_trials = pd.concat(frames, ignore_index=True)
    all_trials.to_csv(TRIAL_CACHE, index=False)
    return all_trials

def fig_sharpness(cells):                 # F1
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for mood, c in [(m, STATE_LABEL_COLORS[m]) for m in ("Impulsive", "StimSens")]:
        sub = cells[cells["state_label"] == mood].sort_values("session_dprime")
        ax[0].plot(sub["session_dprime"], sub["psy_slope"], "o-", color=c, label=mood)
    ax[0].set_xlabel("session d′ (learning →)"); ax[0].set_ylabel("psychometric slope")
    ax[0].set_ylim(-1, 21)                  # slope is bounded to [-20, 20] in sharpness_scores
    ax[0].set_title("F1  Sharpness rises with learning\n(steeper = tells the change apart better)")
    ax[0].legend(frameon=False)
    # F2: RT variability split small vs big change size (averaging across ALL sizes
    # washed out the signal). Variability shrinks with learning, more/earlier for
    # big changes. Non-mood colors here (panel is not mood-split).
    SMALL_COLOR, BIG_COLOR = "#444444", "#1b9e8a"   # dark grey / teal
    small_cols = [c for c in ["rt_cv_cs1.25", "rt_cv_cs1.35", "rt_cv_cs1.5"] if c in cells.columns]
    big_cols = [c for c in ["rt_cv_cs2.0", "rt_cv_cs4.0"] if c in cells.columns]
    if small_cols or big_cols:
        srt = cells.sort_values("session_dprime")
        x = srt["session_dprime"]
        if small_cols:
            ax[1].scatter(x, np.nanmean(srt[small_cols].values, axis=1),
                          color=SMALL_COLOR, s=24, label="small Δ (1.25–1.5)")
        if big_cols:
            ax[1].scatter(x, np.nanmean(srt[big_cols].values, axis=1),
                          color=BIG_COLOR, s=24, label="big Δ (2,4)")
        ax[1].set_title("F2  RT variability shrinks with learning\n(more/earlier for big changes)")
        ax[1].set_xlabel("session d′"); ax[1].set_ylabel("RT CV")
        ax[1].legend(frameon=False)
    return save_fig(fig, "fig_b8_F1_F2_sharpness")

def fig_itchiness(cells):                 # F3
    fig, ax = plt.subplots(figsize=(7, 4))
    for mood, c in [(m, STATE_LABEL_COLORS[m]) for m in ("Impulsive", "StimSens")]:
        sub = cells[cells["state_label"] == mood]
        ax.scatter(sub["criterion_c"], sub["fa_rate"], color=c, label=mood)
    ax.set_xlabel("criterion c  (low = trigger-happy)"); ax.set_ylabel("FA rate")
    ax.set_title("F3  Itchiness separates the moods\n(Impulsive = liberal criterion, more early licks)")
    ax.legend(frameon=False)
    return save_fig(fig, "fig_b8_F3_itchiness")

def fig_timing(all_trials):               # F4
    fig, ax = plt.subplots(figsize=(8, 4))
    cc, ch, _ = dl.change_onset_hazard(all_trials)
    lc, lh, _ = dl.lick_hazard(all_trials)

    def _norm_to_peak(centers, hz):
        # Normalize each hazard to its own max over the plotted (x<12) window so
        # both peak at 1.0 — the visual point is peak ALIGNMENT, not amplitude.
        # Clip to x<12 first so a single late low-at-risk bin can't dominate.
        win = np.asarray(centers) < 12.0
        peak = hz[win].max() if win.any() else hz.max()
        return hz / max(peak, 1e-9)

    ax.plot(cc, _norm_to_peak(cc, ch), label="change-onset hazard (when the change comes)")
    ax.plot(lc, _norm_to_peak(lc, lh), label="lick hazard (when it licks)")
    ax.set_xlim(0, 12); ax.set_ylim(0, 1.05)
    ax.set_xlabel("time from baseline on (s)"); ax.set_ylabel("hazard (norm. to own peak)")
    ax.set_title("F4  Temporal expectation\n(does licking line up with when the change actually comes?)")
    ax.legend(frameon=False)
    return save_fig(fig, "fig_b8_F4_timing")

def fig_bias_not_gain(cells):             # F5
    fig, ax = plt.subplots(figsize=(7, 4))
    for mood, c in [(m, STATE_LABEL_COLORS[m]) for m in ("Impulsive", "StimSens")]:
        sub = cells[cells["state_label"] == mood]
        ax.scatter(sub["psy_slope"], sub["dprime"], color=c, label=mood)
    ax.set_xlabel("psychometric slope"); ax.set_ylabel("d′ (true sensitivity)")
    ax.set_xlim(-1, 21)                     # slope is bounded to [-20, 20]; leave y (d′) auto
    ax.set_title("F5  Bias-not-gain test\n(Impulsive looks eager but d′ should NOT be higher)")
    ax.legend(frameon=False)
    return save_fig(fig, "fig_b8_F5_bias_not_gain")

if __name__ == "__main__":
    force = "--force" in sys.argv
    all_trials = build(force=force)
    cells = dl.descriptive_cell_table(all_trials)
    lat = dl.descriptive_latent_table(all_trials, cells)
    cells.to_csv(os.path.join(CACHE_DIR, "decision_latents_cell_scores.csv"), index=False)
    lat.to_csv(CACHE, index=False)
    fig_sharpness(cells); fig_itchiness(cells); fig_timing(all_trials); fig_bias_not_gain(cells)
    # F-summary: which dial moves with learning / which separates moods
    # Defensive: select only score columns that actually exist (all-cells-underpowered edge).
    cols = [c for c in ["psy_slope", "dprime", "criterion_c",
                        "fa_rate", "lick_hazard_peak_time"] if c in cells.columns]
    summ = cells.groupby("state_label")[cols].mean()
    summ.to_csv(os.path.join(FIG_DIR, "decision_latents_stats.csv"))
    print(summ)
