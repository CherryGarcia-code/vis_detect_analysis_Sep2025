"""B8 Fig: decision-latents by state (Step 1, descriptive).

Plain English: measures three behavioural 'dials' — sharpness (can it tell the
change happened), itchiness (is it trigger-happy), timing (does it expect the
change now) — split by mood (Impulsive vs StimSens), across learning, and
saves them as figures + a per-trial table.

Worktree run recipe:
  WT=$(pwd); PYTHONPATH="$WT/src" py scripts/analysis/decision_latents/run_decision_latents_by_state.py
"""
import os, sys, gc, warnings, numpy as np, pandas as pd, matplotlib
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
    sessions = dl.enumerate_valid_sessions()              # CHRONOLOGICAL order
    # Key the index by zero-padded 8-digit id: writing the trial table to CSV
    # makes pandas re-read all-digit session names as int (dropping the leading
    # zero, e.g. 01072025 → 1072025), so a raw-string map would silently miss.
    sidx = {s.zfill(8): i for i, s in enumerate(sessions)}  # chronological index = learning axis (F1/F2)
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
        tab["session_idx"] = sidx[sname.zfill(8)]         # chronological index (NOT d′) is the learning axis
        frames.append(tab)
    all_trials = pd.concat(frames, ignore_index=True)
    all_trials.to_csv(TRIAL_CACHE, index=False)
    return all_trials

# 5 canonical go-trial change sizes (psychometric x-axis)
CHANGE_SIZES = [1.25, 1.35, 1.5, 2.0, 4.0]


def fig_sharpness(cells, all_trials):     # F1
    """Two panels: (A) psychometric curve SHAPE by mood, (B) detection threshold
    falling over chronological training."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # Panel A — psychometric curves: P(lick on GO trials) vs change_size, one
    # line per mood, averaged across all that mood's go trials.
    go = all_trials[all_trials["change_size"] > 1.0]
    for mood, c in [(m, STATE_LABEL_COLORS[m]) for m in ("Impulsive", "StimSens")]:
        sub = go[go["state_label"] == mood]
        if sub.empty:
            continue
        p = sub.groupby("change_size")["lick"].mean()
        # plot only the canonical change sizes, in order
        xs = [cs for cs in CHANGE_SIZES if cs in p.index]
        ys = [p.loc[cs] for cs in xs]
        ax[0].plot(xs, ys, "o-", color=c, label=mood)
    ax[0].set_xlabel("change size (Δ TF ratio)"); ax[0].set_ylabel("P(detect) on GO trials")
    ax[0].set_xticks(CHANGE_SIZES); ax[0].set_xticklabels([str(cs) for cs in CHANGE_SIZES])
    ax[0].set_ylim(0, 1)
    ax[0].set_title("F1A  Psychometric curves by mood\n(shape: small Δ shallow, big Δ steep)")
    ax[0].legend(frameon=False)
    # Panel B — per-cell detection threshold vs chronological session index, one
    # series per engaged mood, with a light binned-mean trend. Lower = sharper.
    for mood, c in [(m, STATE_LABEL_COLORS[m]) for m in ("Impulsive", "StimSens")]:
        sub = cells[(cells["state_label"] == mood) & np.isfinite(cells["psy_threshold"])]
        if sub.empty:
            continue
        ax[1].scatter(sub["session_idx"], sub["psy_threshold"], color=c, s=22,
                      alpha=0.5, label=mood)
        # light binned-mean trend (~5 equal-width chronological bins)
        xv = sub["session_idx"].values.astype(float); yv = sub["psy_threshold"].values.astype(float)
        if xv.size >= 2 and np.ptp(xv) > 0:
            edges = np.linspace(xv.min(), xv.max(), 6)
            bc = 0.5 * (edges[:-1] + edges[1:])
            bm = [np.nanmean(yv[(xv >= edges[i]) & (xv <= edges[i + 1])])
                  if np.any((xv >= edges[i]) & (xv <= edges[i + 1])) else np.nan
                  for i in range(len(edges) - 1)]
            ax[1].plot(bc, bm, "-", color=c, lw=2.2)
    ax[1].set_xlabel("session index (chronological, learning →)")
    ax[1].set_ylabel("change size at 50% detection\n(lower = sharper)")
    ax[1].set_title("F1B  Sensitivity improves with training\n(threshold falls)")
    ax[1].legend(frameon=False)
    return save_fig(fig, "fig_b8_F1_sharpness")


def fig_rt_variability(cells):            # F2
    """RT-variability vs chronological training, engaged moods only, binned mean
    ± SEM for small vs big change sizes."""
    SMALL_COLOR, BIG_COLOR = "#444444", "#1b9e8a"   # dark grey / teal
    fig, ax = plt.subplots(figsize=(7, 4))
    # engaged moods only (exclude Disengaged)
    eng = cells[cells["state_label"].isin(["Impulsive", "StimSens"])].copy()
    small_cols = [c for c in ["rt_cv_cs1.25", "rt_cv_cs1.35", "rt_cv_cs1.5"] if c in eng.columns]
    big_cols = [c for c in ["rt_cv_cs2.0", "rt_cv_cs4.0"] if c in eng.columns]
    with warnings.catch_warnings():               # all-NaN rows (cell w/ no Hit RTs) → harmless empty-slice warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        eng["rt_cv_small"] = (np.nanmean(eng[small_cols].values, axis=1)
                              if small_cols else np.nan)
        eng["rt_cv_big"] = (np.nanmean(eng[big_cols].values, axis=1)
                            if big_cols else np.nan)

    def _binned(ycol, color, label):
        sub = eng[np.isfinite(eng["session_idx"]) & np.isfinite(eng[ycol])]
        if sub.empty:
            return
        xv = sub["session_idx"].values.astype(float); yv = sub[ycol].values.astype(float)
        # optional faint raw scatter behind the trend
        ax.scatter(xv, yv, color=color, s=12, alpha=0.18)
        if xv.size < 2 or np.ptp(xv) == 0:
            return
        edges = np.linspace(xv.min(), xv.max(), 6)   # ~5 equal-width chronological bins
        bc, bm, be = [], [], []
        for i in range(len(edges) - 1):
            m = (xv >= edges[i]) & (xv <= edges[i + 1])
            yy = yv[m]; yy = yy[np.isfinite(yy)]
            if yy.size == 0:
                continue
            bc.append(0.5 * (edges[i] + edges[i + 1]))
            bm.append(np.mean(yy))
            be.append(np.std(yy) / np.sqrt(yy.size) if yy.size > 1 else 0.0)
        bc, bm, be = np.array(bc), np.array(bm), np.array(be)
        ax.plot(bc, bm, "-", color=color, lw=2.4, label=label)
        ax.fill_between(bc, bm - be, bm + be, color=color, alpha=0.18)

    _binned("rt_cv_small", SMALL_COLOR, "small Δ (1.25–1.5)")
    _binned("rt_cv_big", BIG_COLOR, "big Δ (2,4)")
    ax.set_xlabel("session index (chronological, learning →)"); ax.set_ylabel("Hit RT CV")
    ax.set_title("F2  RT-variability shrinks with training\n— faster & further for big Δ")
    ax.legend(frameon=False)
    return save_fig(fig, "fig_b8_F2_rt_variability")

def fig_itchiness(cells):                 # F3
    fig, ax = plt.subplots(figsize=(7, 4))
    for mood, c in [(m, STATE_LABEL_COLORS[m]) for m in ("Impulsive", "StimSens")]:
        sub = cells[cells["state_label"] == mood]
        ax.scatter(sub["criterion_c"], sub["fa_rate"], color=c, label=mood)
    ax.set_xlabel("criterion c  (low = trigger-happy)"); ax.set_ylabel("FA rate")
    ax.set_title("F3  Itchiness separates the moods\n(Impulsive = liberal criterion, more early licks)\n"
                 "(largely confirms the labeler definition — Impulsive is defined via early/inappropriate licks)")
    ax.legend(frameon=False)
    return save_fig(fig, "fig_b8_F3_itchiness")

def fig_timing(all_trials):               # F4
    fig, ax = plt.subplots(figsize=(8, 4))
    cc, ch, _ = dl.change_onset_hazard(all_trials)
    fc, fh, _ = dl.fa_lick_hazard(all_trials)     # anticipatory/early-lick hazard (primary)
    lc, lh, _ = dl.lick_hazard(all_trials)        # all licks (faint reference)

    def _norm_to_peak(centers, hz):
        # Normalize each hazard to its own max over the plotted (x<12) window so
        # both reach 1.0 — the visual point is peak ALIGNMENT, not amplitude.
        # Clip to x<12 first so a single late low-at-risk bin can't dominate.
        win = np.asarray(centers) < 12.0
        peak = hz[win].max() if win.any() else hz.max()
        return hz / max(peak, 1e-9)

    # faint all-lick hazard for reference (drawn first, behind the primaries)
    ax.plot(lc, _norm_to_peak(lc, lh), color="#999999", lw=1.0, alpha=0.6,
            label="all licks (reference)")
    ax.plot(cc, _norm_to_peak(cc, ch), color="#444444", lw=2.0,
            label="change-onset hazard (when the change comes)")
    ax.plot(fc, _norm_to_peak(fc, fh), color=STATE_LABEL_COLORS["Impulsive"], lw=2.0,
            label="early (FA) licks (anticipatory)")
    # earliest possible change (real-data fact: changes NEVER occur before 6 s)
    ax.axvline(6.0, ls="--", color="k", lw=1.0)
    ax.text(6.05, 1.0, "earliest possible change", rotation=90,
            va="top", ha="left", fontsize=8, color="k")
    ax.set_xlim(0, 12); ax.set_ylim(0, 1.05)
    ax.set_xlabel("time from baseline on (s)"); ax.set_ylabel("hazard (norm. to own peak)")
    ax.set_title("F4  Temporal expectation — anticipatory (early) licks are timed toward\n"
                 "the expected change even before it can occur (<6 s)")
    ax.legend(frameon=False, fontsize=8)
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
    # Chronological session index = learning axis for F1/F2 (rebuilt here so it
    # survives the trial-table CSV round-trip in build()). Key on zfill(8) because
    # the CSV round-trip drops leading zeros (01072025 → 1072025); zero-pad both sides.
    sidx = {s.zfill(8): i for i, s in enumerate(dl.enumerate_valid_sessions())}
    cells["session_idx"] = cells["session_name"].astype(str).str.zfill(8).map(sidx)
    lat = dl.descriptive_latent_table(all_trials, cells)
    cells.to_csv(os.path.join(CACHE_DIR, "decision_latents_cell_scores.csv"), index=False)
    lat.to_csv(CACHE, index=False)
    fig_sharpness(cells, all_trials); fig_rt_variability(cells)
    fig_itchiness(cells); fig_timing(all_trials); fig_bias_not_gain(cells)
    # F-summary: which dial moves with learning / which separates moods
    # Defensive: select only score columns that actually exist (all-cells-underpowered edge).
    cols = [c for c in ["psy_slope", "dprime", "criterion_c",
                        "fa_rate", "lick_hazard_peak_time"] if c in cells.columns]
    summ = cells.groupby("state_label")[cols].mean()
    summ.to_csv(os.path.join(FIG_DIR, "decision_latents_stats.csv"))
    print(summ)
