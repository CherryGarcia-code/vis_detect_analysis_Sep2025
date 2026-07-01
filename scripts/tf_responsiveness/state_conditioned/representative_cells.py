"""Representative TF-responsive cells (one per subject) — response to the TF
PULSE (fast vs slow) and to each OUTCOME (Hit/Miss @ Change_ON, FA @ FA-event),
faceted by behavioural STATE, with 95% bootstrap CI and trial-count MATCHING
across states (subsample to the min per outcome — the clean way given c1_r's
trial-count sensitivity). Plus a population across-learning panel. QC-passing
sessions only (drop the Excluded/breakdown sessions). Uses project conventions
throughout: align.get_event_times_by_trial + EVENT_VALID_OUTCOMES, DEFAULT_BIN_SIZE
/ DEFAULT_SIGMA_MS, TF_PULSE_WINDOW, EVENT_RESPONSIVENESS_WINDOWS.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_bg046/src")
from visdetect.core.session import load_session  # noqa: E402
from visdetect.analysis.align import get_event_times_by_trial, align_spikes_to_events  # noqa: E402
from visdetect.analysis.constants import (DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS,  # noqa: E402
    TF_PULSE_WINDOW, EVENT_RESPONSIVENESS_WINDOWS)
from visdetect.analysis.tf_glm import (TFGLMConfig, assemble_design,  # noqa: E402
    pulse_times_from_tf)
from visdetect.analysis.tf_glm_data import session_trial_regressors  # noqa: E402

REPO = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
STATES = ["StimSens", "Impulsive", "Disengaged"]
STATE_COLORS = {"StimSens": "#6baed6", "Impulsive": "#ef6548", "Disengaged": "#3474ae"}
# representative session per subject (QC-pass, all 3 states, strong TF cell)
REPR = {"BG_046": ("10092025", "DMS"), "BG_039": ("16042025", "DMS"), "BG_031": ("190325", "VMS")}
OUTCOME_EVENT = [("hit", "Change_ON"), ("miss", "Change_ON"), ("fa", "FA")]
DISP_WIN = (-0.5, 1.0)          # align.py default display window
BIN = DEFAULT_BIN_SIZE
SIG = DEFAULT_SIGMA_MS / 1000.0 / BIN   # smoothing sigma in bins
MIN_TRIALS = 12
SEED = 42
OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/representative_cells")
try:
    from visdetect.viz.plotting import set_style
    set_style("talk")
except Exception:
    pass


def _spikes(session, uid):
    for c in session.clusters:
        if int(c.cluster_id) == int(uid):
            return np.asarray(c.spike_times, float).ravel()
    return np.zeros(0)


def _smooth(a):
    return gaussian_filter1d(a, SIG) if SIG > 0 else a


def _boot_ci(mat, n=1000, seed=SEED):
    """mat: (n_events, n_bins) firing rate in Hz (align_spikes_to_events already
    divides by bin_size). Returns smoothed (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    m = _smooth(mat.mean(0))
    if len(mat) < 3:
        return m, m, m
    boots = np.stack([_smooth(mat[rng.integers(0, len(mat), len(mat))].mean(0)) for _ in range(n)])
    return m, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)


def outcome_state_peths(spikes, session, lab, outcome, event):
    """{state -> (t, mean, lo, hi, n)} for one outcome/event, N-matched across states."""
    et = np.array(get_event_times_by_trial(session, event), float)  # per-trial, NaN if invalid
    # trials of this outcome with a valid event time, grouped by state
    by_state = {st: [] for st in STATES}
    for i, t in enumerate(session.trials):
        oc = str(getattr(t, "trialoutcome", "") or "").lower()
        if oc == outcome and i < et.size and np.isfinite(et[i]):
            stt = lab.get(i)
            if stt in by_state:
                by_state[stt].append(et[i])
    have = {st: np.array(v) for st, v in by_state.items() if len(v) >= MIN_TRIALS}
    if not have:
        return {}, None
    nmatch = min(len(v) for v in have.values())          # trial-count MATCH across states
    rng = np.random.default_rng(SEED)
    out, t_axis = {}, None
    for st, times in have.items():
        sel = times[rng.choice(len(times), nmatch, replace=False)] if len(times) > nmatch else times
        binned, t_axis = align_spikes_to_events(spikes, sel.tolist(), window=DISP_WIN, bin_size=BIN)
        binned = np.asarray(binned, float)
        m, lo, hi = _boot_ci(binned)
        out[st] = (t_axis, m, lo, hi, nmatch)
    return out, t_axis


def pulse_peths(spikes, session):
    """fast/slow TF-pulse PETH (Hz, baseline-subtracted) + CI95."""
    cfg = TFGLMConfig(include_movement=False, include_phase=False, include_tiled_baseline=True,
                      standardize_design=True, fast_fit=True, tf_encoding="log2", min_pulses_per_label=20)
    trials, units = session_trial_regressors(session, cfg)
    d = assemble_design(trials, cfg)
    fast, slow = pulse_times_from_tf(d, cfg)
    res = {}
    for name, p in (("fast", fast), ("slow", slow)):
        binned, t = align_spikes_to_events(spikes, p.tolist(), window=TF_PULSE_WINDOW, bin_size=BIN)
        binned = np.asarray(binned, float)
        m, lo, hi = _boot_ci(binned)
        base = m[t < 0].mean()
        res[name] = (t, m - base, lo - base, hi - base)
    return res


def _registry(subj):
    return pd.read_csv(f"{REPO}/data/cache/tf_responsive/{subj.lower().replace('_','')}_tf_responsive.csv",
                       dtype={"session": str, "session_date": str})


def pick_unit(subj, date):
    """Best exemplar: responsive cell with the highest C1 AMONG the higher-firing
    half (clean PETHs need firing rate, not just C1)."""
    reg = _registry(subj)
    reg = reg[reg.resp_log2.astype(str).str.lower().isin(["true", "1", "1.0"]) & (reg.session_date == date)]
    hi = reg[reg.n_spikes >= reg.n_spikes.median()]
    r = (hi if len(hi) else reg).sort_values("c1_r_log2", ascending=False).iloc[0]
    return r["session"], int(r["unit"]), float(r["c1_r_log2"])


def learning_curve(subj):
    """responsive fraction per learning stage over QC-pass sessions (fraction is
    trial-count-robust, unlike c1_r). Returns [(stage, frac, n_units, n_sess)]."""
    man = pd.read_csv(f"{REPO}/data/{subj}_staging_manifest.csv", dtype={"session_name": str})
    qc = man.loc[~man.qc_fail.astype(bool)]
    stage = dict(zip(qc.session_name, qc.stage.astype(str)))
    reg = _registry(subj)
    reg = reg[reg.session_date.isin(qc.session_name)].copy()
    reg["stage"] = reg.session_date.map(stage)
    reg["resp"] = reg.resp_log2.astype(str).str.lower().isin(["true", "1", "1.0"])
    order = [s for s in ["Naive", "Learning", "Expert"] if s in set(reg.stage)]
    out = []
    for st in order:
        g = reg[reg.stage == st]
        out.append((st, 100 * g.resp.mean(), len(g), g.session_date.nunique()))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(23, 12))
    gs = gridspec.GridSpec(3, 5, hspace=0.5, wspace=0.36)
    for row, (subj, (date, region)) in enumerate(REPR.items()):
        sess, uid, c1 = pick_unit(subj, date)
        s = load_session(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
        spk = _spikes(s, uid)
        lab = pd.read_csv(f"{REPO}/data/cache/state_tags/{subj}/{date}.csv").set_index("trial_idx")["state_label"].to_dict()
        # --- pulse panel ---
        axp = fig.add_subplot(gs[row, 0])
        pp = pulse_peths(spk, s)
        for nm, col in (("fast", "#d6322a"), ("slow", "#2b6fb3")):
            t, m, lo, hi = pp[nm]
            axp.plot(t, m, color=col, lw=2, label=nm)
            axp.fill_between(t, lo, hi, color=col, alpha=0.2)
        axp.axvline(0, color="0.7", lw=0.8); axp.axhline(0, color="0.7", lw=0.8)
        axp.set_title(f"{subj} ({region})  u{uid}\nTF pulse  (C1={c1:.2f})", fontsize=11)
        axp.set_ylabel("Δ firing (Hz)"); axp.set_xlabel("t from pulse (s)")
        axp.legend(frameon=False, fontsize=8)
        # --- outcome x state panels ---
        for col, (oc, ev) in enumerate(OUTCOME_EVENT, start=1):
            ax = fig.add_subplot(gs[row, col])
            peths, _ = outcome_state_peths(spk, s, lab, oc, ev)
            for st in STATES:
                if st in peths:
                    t, m, lo, hi, n = peths[st]
                    ax.plot(t, m, color=STATE_COLORS[st], lw=1.8, label=f"{st} (n={n})")
                    ax.fill_between(t, lo, hi, color=STATE_COLORS[st], alpha=0.18)
            # shade the canonical responsiveness (post) window for this event
            pw = EVENT_RESPONSIVENESS_WINDOWS.get(ev, (None, (0, 0)))[1]
            if pw and DISP_WIN[0] <= pw[1] and pw[0] <= DISP_WIN[1]:
                ax.axvspan(max(pw[0], DISP_WIN[0]), min(pw[1], DISP_WIN[1]), color="0.9", zorder=0)
            ax.axvline(0, color="0.7", lw=0.8)
            title = {"hit": "Hit @ Change_ON", "miss": "Miss @ Change_ON", "fa": "FA @ early-lick"}[oc]
            ax.set_title(title + "  (N-matched)", fontsize=10)
            ax.set_xlabel(f"t from {ev} (s)")
            if peths:
                ax.legend(frameon=False, fontsize=7.5)
        # --- population across-learning (col 4): responsive fraction per stage ---
        axl = fig.add_subplot(gs[row, 4])
        lc = learning_curve(subj)
        axl.bar(range(len(lc)), [f for _, f, _, _ in lc], color="#5aa469", width=0.62)
        for i, (st, f, nu, ns) in enumerate(lc):
            axl.text(i, f, f"{f:.1f}%\n{ns}s", ha="center", va="bottom", fontsize=8)
        axl.set_xticks(range(len(lc))); axl.set_xticklabels([s for s, *_ in lc], fontsize=9)
        axl.set_ylabel("% TF-responsive"); axl.set_ylim(0, None)
        axl.set_title("Across learning\n(resp. fraction, QC-pass)", fontsize=10)
        for sp in ("top", "right"): axl.spines[sp].set_visible(False)
    fig.suptitle("Representative TF-responsive cells — pulse response + outcome × behavioural state "
                 "(95% CI, trial-count matched; QC-pass sessions)", fontsize=13, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"representative_cells.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/representative_cells.png (+.pdf)")


if __name__ == "__main__":
    main()
