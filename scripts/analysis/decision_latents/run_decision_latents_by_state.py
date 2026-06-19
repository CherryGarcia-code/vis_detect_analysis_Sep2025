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

# 5 canonical go-trial change sizes (psychometric x-axis). Equal-spaced x: we
# plot against integer positions 0..4 with these strings as ticklabels so the
# crowded small Δ (1.25/1.35/1.5) get the same visual spacing as the big ones.
CHANGE_SIZES = [1.25, 1.35, 1.5, 2.0, 4.0]
CHANGE_SIZE_LABELS = ["1.25", "1.35", "1.5", "2", "4"]


def _plot_psychometric_curves(ax, go_subset):
    """Draw the two mood curves of P(lick on GO trials) vs change_size on ``ax``,
    using EQUAL-SPACED integer x positions (0..4) so small Δ stop crowding.
    Returns the number of go trials actually plotted."""
    n_total = 0
    for mood in ("Impulsive", "StimSens"):
        c = STATE_LABEL_COLORS[mood]
        sub = go_subset[go_subset["state_label"] == mood]
        if sub.empty:
            continue
        p = sub.groupby("change_size")["lick"].mean()
        xs = [i for i, cs in enumerate(CHANGE_SIZES) if cs in p.index]   # equal-spaced positions
        ys = [p.loc[CHANGE_SIZES[i]] for i in xs]
        if xs:
            ax.plot(xs, ys, "o-", color=c, label=mood)
            n_total += len(sub)
    ax.set_xticks(range(len(CHANGE_SIZES)))
    ax.set_xticklabels(CHANGE_SIZE_LABELS)
    ax.set_ylim(0, 1)
    return n_total


def fig_F1A_curves_by_thirds(all_trials):
    """F1A (variant 1): psychometric curves by mood, split into chronological
    thirds of session_idx (early / mid / late). Equal-spaced change-size x."""
    go = all_trials[all_trials["change_size"] > 1.0].copy()
    si = go["session_idx"].values.astype(float)
    lo, hi = np.nanmin(si), np.nanmax(si)
    edges = np.linspace(lo, hi + 1e-9, 4)        # 3 equal chronological parts
    names = ["early", "mid", "late"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for k, ax in enumerate(axes):
        mask = (go["session_idx"] >= edges[k]) & (go["session_idx"] < edges[k + 1])
        if k == 2:                                # last bin is closed on the right
            mask = (go["session_idx"] >= edges[k]) & (go["session_idx"] <= edges[k + 1])
        sub = go[mask]
        n = _plot_psychometric_curves(ax, sub)
        ax.set_xlabel("change size (Δ TF ratio)")
        ax.set_title(f"{names[k]} (n={n} trials)")
        if k == 0:
            ax.set_ylabel("P(detect) on GO trials"); ax.legend(frameon=False)
    fig.suptitle("F1A  Psychometric curves by mood — chronological thirds\n"
                 "(equal-spaced change-size axis; P(detect) on GO trials)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_fig(fig, "fig_b8_F1A_curves_by_thirds")


def fig_F1A_curves_by_comprehension(all_trials):
    """F1A (variant 2): psychometric curves by mood, split by comprehension_flag
    (pre vs post). Equal-spaced change-size x."""
    go = all_trials[all_trials["change_size"] > 1.0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, flag in zip(axes, ["pre", "post"]):
        sub = go[go["comprehension_flag"] == flag]
        n = _plot_psychometric_curves(ax, sub)
        ax.set_xlabel("change size (Δ TF ratio)")
        ax.set_title(f"{flag} (n={n} trials)")
        if flag == "pre":
            ax.set_ylabel("P(detect) on GO trials"); ax.legend(frameon=False)
    fig.suptitle("F1A  Psychometric curves by mood — comprehension split\n"
                 "(equal-spaced change-size axis; P(detect) on GO trials)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_fig(fig, "fig_b8_F1A_curves_by_comprehension")


def fig_F1B_threshold(cells):
    """F1B: per-cell detection threshold vs chronological session index, one
    series per engaged mood, with a light binned-mean trend. Lower = sharper.
    Robust y-limit: clip the top to the 95th percentile (+0.1) so a few outliers
    don't flatten the trajectory."""
    fig, ax = plt.subplots(figsize=(7, 4))
    thr_all = cells["psy_threshold"].values.astype(float)
    for mood in ("Impulsive", "StimSens"):
        c = STATE_LABEL_COLORS[mood]
        sub = cells[(cells["state_label"] == mood) & np.isfinite(cells["psy_threshold"])]
        if sub.empty:
            continue
        ax.scatter(sub["session_idx"], sub["psy_threshold"], color=c, s=22,
                   alpha=0.5, label=mood)
        xv = sub["session_idx"].values.astype(float); yv = sub["psy_threshold"].values.astype(float)
        if xv.size >= 2 and np.ptp(xv) > 0:
            edges = np.linspace(xv.min(), xv.max(), 6)
            bc = 0.5 * (edges[:-1] + edges[1:])
            bm = [np.nanmean(yv[(xv >= edges[i]) & (xv <= edges[i + 1])])
                  if np.any((xv >= edges[i]) & (xv <= edges[i + 1])) else np.nan
                  for i in range(len(edges) - 1)]
            ax.plot(bc, bm, "-", color=c, lw=2.2)
    # robust y-limit so outliers don't flatten the trajectory
    if np.any(np.isfinite(thr_all)):
        top = min(2.0, float(np.nanpercentile(thr_all, 95)) + 0.1)
    else:
        top = 2.0
    ax.set_ylim(0.95, max(top, 1.0))
    ax.set_xlabel("session index (chronological, learning →)")
    ax.set_ylabel("change size at 50% detection\n(lower = sharper)")
    ax.set_title("F1B  Sensitivity improves with training\n(threshold falls; y-axis clipped to 95th pct)")
    ax.legend(frameon=False)
    return save_fig(fig, "fig_b8_F1B_threshold")


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
    # robust ylim: clip the top to the 97th percentile of all plotted CVs (+0.05)
    # so the lone ~1.0 outlier doesn't dominate the panel.
    all_cv = np.concatenate([eng["rt_cv_small"].values, eng["rt_cv_big"].values])
    all_cv = all_cv[np.isfinite(all_cv)]
    top = min(0.75, float(np.nanpercentile(all_cv, 97)) + 0.05) if all_cv.size else 0.75
    ax.set_ylim(0, max(top, 0.1))
    # honest title: the across-session training shrinkage is weak/absent in BG_046,
    # but big-Δ licks ARE less variable than small-Δ (the real, present effect).
    ax.set_title("F2  RT-variability: big-Δ licks are less variable than small-Δ\n"
                 "(across-session training decline weak/absent in BG_046)")
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
    """F5 redesign: direct per-cell d′-by-mood (the GAIN axis) and criterion-c-by-
    mood (the BIAS axis), engaged moods only. Strip/jitter of per-cell values with
    a mean ± 95% CI overlay per mood. The old d′-vs-psy_slope scatter is dropped
    because psy_slope is a poor x (flat for Impulsive)."""
    moods = ("Impulsive", "StimSens")
    eng = cells[cells["state_label"].isin(moods)]
    rng = np.random.default_rng(42)

    def _strip(ax, col):
        for i, mood in enumerate(moods):
            c = STATE_LABEL_COLORS[mood]
            y = eng.loc[eng["state_label"] == mood, col].values.astype(float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            x = i + rng.uniform(-0.12, 0.12, size=y.size)   # horizontal jitter
            ax.scatter(x, y, color=c, s=24, alpha=0.55, zorder=2)
            # mean ± 95% CI (1.96·SEM); fall back to just the mean for n==1
            m = float(np.mean(y))
            ax.scatter(i, m, color=c, s=90, marker="_", linewidths=2.5, zorder=3)
            if y.size > 1:
                ci = 1.96 * np.std(y, ddof=1) / np.sqrt(y.size)
                ax.errorbar(i, m, yerr=ci, color=c, capsize=5, lw=2.0, zorder=3)
        ax.set_xticks(range(len(moods)))
        ax.set_xticklabels(moods)
        ax.set_xlim(-0.5, len(moods) - 0.5)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    _strip(ax[0], "dprime")
    ax[0].set_ylabel("d′ (sensitivity)")
    ax[0].set_title("the GAIN axis — should NOT be higher for Impulsive")
    _strip(ax[1], "criterion_c")
    ax[1].set_ylabel("criterion c (low = liberal/trigger-happy)")
    ax[1].set_title("the BIAS axis")
    # small caption: criterion is partly definitional (Impulsive ≙ early/liberal licks)
    fig.text(0.5, 0.005,
             "note: criterion c is partly definitional — the labeler defines Impulsive "
             "via early/inappropriate (liberal) licks (spec §7).",
             ha="center", va="bottom", fontsize=8, color="#555555")
    fig.suptitle("F5 Bias-not-gain: moods differ on bias (criterion), not gain (d′)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
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
    # F1 split into F1A (two stage-split variants) + F1B; F5 redesigned.
    fig_F1A_curves_by_thirds(all_trials)
    fig_F1A_curves_by_comprehension(all_trials)
    fig_F1B_threshold(cells)
    fig_rt_variability(cells)
    fig_itchiness(cells); fig_timing(all_trials); fig_bias_not_gain(cells)

    # F4 diagnostic: after the censoring fix the FA-hazard should NO LONGER track
    # the raw FA-time density. Print corr(hazard, density) over 0–6 s and the
    # hazard mean(0–3s) vs mean(3–6s).
    fc, fh, _ = dl.fa_lick_hazard(all_trials)
    fa_t = all_trials.loc[all_trials["outcome"] == "fa", "decision_time"].values.astype(float)
    fa_t = fa_t[np.isfinite(fa_t)]
    edges = np.arange(0.0, float(fc.max()) + 0.05 + 1e-9, 0.05)
    dens, _ = np.histogram(fa_t, bins=edges)
    dens = dens.astype(float)
    n = min(len(fc), len(dens))
    fcn, fhn, dnn = fc[:n], fh[:n], dens[:n]

    def _corr(hi):
        w = fcn <= hi
        if np.std(fhn[w]) > 0 and np.std(dnn[w]) > 0:
            return float(np.corrcoef(fhn[w], dnn[w])[0, 1])
        return float("nan")

    corr06 = _corr(6.0)
    haz_0_3 = float(np.nanmean(fhn[(fcn >= 0.0) & (fcn < 3.0)]))
    haz_3_6 = float(np.nanmean(fhn[(fcn >= 3.0) & (fcn < 6.0)]))
    print(f"[F4 diagnostic] corr(FA-hazard, FA-density) 0-6s = {corr06:.3f}")
    print(f"[F4 diagnostic] FA-hazard mean 0-3s = {haz_0_3:.5f}  |  3-6s = {haz_3_6:.5f}")
    # NOTE: changes never occur before ~6 s, so non-FA trials only START leaving the
    # FA at-risk set after 6 s — the censoring fix is a no-op in 0–6 s (corr stays
    # high there). The fix's effect appears once the post-6 s window is included:
    print(f"[F4 diagnostic] corr 0-12s = {_corr(12.0):.3f}  |  full = {_corr(float(fcn.max()) + 1):.3f}"
          f"  (buggy@decision was ~0.92 / ~0.73 — fix decorrelates post-6s)")
    # F-summary: which dial moves with learning / which separates moods
    # Defensive: select only score columns that actually exist (all-cells-underpowered edge).
    cols = [c for c in ["psy_slope", "dprime", "criterion_c",
                        "fa_rate", "lick_hazard_peak_time"] if c in cells.columns]
    summ = cells.groupby("state_label")[cols].mean()
    summ.to_csv(os.path.join(FIG_DIR, "decision_latents_stats.csv"))
    print(summ)
