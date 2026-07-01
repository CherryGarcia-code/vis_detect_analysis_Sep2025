"""Representative TF-responsive cells (best exemplar per subject, QC-pass
sessions) — response to the TF PULSE (fast vs slow) and to the OUTCOMES
(Hit vs Miss @ Change_ON, FA @ early-lick), all with 95% bootstrap CI. No state
faceting. Plus a 'recruitment' panel = TF-responsive fraction across CHRONOLOGICAL
5-session bins (regardless of learning stage). Project conventions throughout:
align.get_event_times_by_trial + EVENT_VALID_OUTCOMES, DEFAULT_BIN_SIZE/SIGMA_MS,
TF_PULSE_WINDOW, EVENT_RESPONSIVENESS_WINDOWS.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
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
from visdetect.analysis import config  # noqa: E402
from visdetect.analysis.align import get_event_times_by_trial, align_spikes_to_events  # noqa: E402
from visdetect.analysis.constants import (DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS,  # noqa: E402
    TF_PULSE_WINDOW, EVENT_RESPONSIVENESS_WINDOWS)
from visdetect.analysis.tf_glm import (TFGLMConfig, assemble_design,  # noqa: E402
    pulse_times_from_tf)
from visdetect.analysis.tf_glm_data import session_trial_regressors  # noqa: E402

REPO = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
SUBJECTS = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
DISP_WIN = (-0.5, 1.0)
BIN = DEFAULT_BIN_SIZE
SIG = DEFAULT_SIGMA_MS / 1000.0 / BIN
BINSIZE = 5             # sessions per recruitment bin
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
    """mat: (n_events, n_bins) in Hz (align_spikes_to_events already /bin_size)."""
    rng = np.random.default_rng(seed)
    m = _smooth(mat.mean(0))
    if len(mat) < 3:
        return m, m, m
    boots = np.stack([_smooth(mat[rng.integers(0, len(mat), len(mat))].mean(0)) for _ in range(n)])
    return m, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)


def _registry(subj):
    r = pd.read_csv(f"{REPO}/data/cache/tf_responsive/{subj.lower().replace('_','')}_tf_responsive.csv",
                    dtype={"session": str, "session_date": str})
    r["resp"] = r.resp_log2.astype(str).str.lower().isin(["true", "1", "1.0"])
    return r


def _qc_dates(subj):
    man = pd.read_csv(f"{REPO}/data/{subj}_staging_manifest.csv", dtype={"session_name": str})
    return set(man.loc[~man.qc_fail.astype(bool), "session_name"])


def pick_unit(subj):
    """Best exemplar across QC-pass sessions: highest C1 among the higher-firing
    responsive cells (clean PETHs need firing rate)."""
    reg = _registry(subj)
    reg = reg[reg.resp & reg.session_date.isin(_qc_dates(subj))]
    hi = reg[reg.n_spikes >= reg.n_spikes.quantile(0.6)]
    r = (hi if len(hi) else reg).sort_values("c1_r_log2", ascending=False).iloc[0]
    return r["session"], int(r["unit"]), float(r["c1_r_log2"])


def outcome_peth(spikes, session, outcome, event):
    """single PETH (Hz) + CI95 for trials of `outcome`, aligned to `event`."""
    et = np.array(get_event_times_by_trial(session, event), float)
    times = [et[i] for i, t in enumerate(session.trials)
             if str(getattr(t, "trialoutcome", "") or "").lower() == outcome
             and i < et.size and np.isfinite(et[i])]
    if len(times) < 5:
        return None
    binned, t = align_spikes_to_events(spikes, times, window=DISP_WIN, bin_size=BIN)
    m, lo, hi = _boot_ci(np.asarray(binned, float))
    return t, m, lo, hi, len(times)


def pulse_peths(spikes, session):
    cfg = TFGLMConfig(include_movement=False, include_phase=False, include_tiled_baseline=True,
                      standardize_design=True, fast_fit=True, tf_encoding="log2", min_pulses_per_label=20)
    trials, _ = session_trial_regressors(session, cfg)
    d = assemble_design(trials, cfg)
    fast, slow = pulse_times_from_tf(d, cfg)
    res = {}
    for nm, p in (("fast", fast), ("slow", slow)):
        binned, t = align_spikes_to_events(spikes, p.tolist(), window=TF_PULSE_WINDOW, bin_size=BIN)
        m, lo, hi = _boot_ci(np.asarray(binned, float))
        base = m[t < 0].mean()
        res[nm] = (t, m - base, lo - base, hi - base)
    return res


def recruit_bins(subj, binsize=BINSIZE):
    """TF-responsive fraction per CHRONOLOGICAL bin of `binsize` QC-pass sessions."""
    reg = _registry(subj)
    reg = reg[reg.session_date.isin(_qc_dates(subj))]
    per = (reg.groupby("session_date").agg(n=("resp", "size"), r=("resp", "sum")).reset_index())
    per["d"] = per.session_date.map(config.parse_session_date)
    per = per.dropna(subset=["d"]).sort_values("d").reset_index(drop=True)
    out = []
    for b in range(0, len(per), binsize):
        chunk = per.iloc[b:b + binsize]
        frac = 100 * chunk.r.sum() / chunk.n.sum()
        out.append((b // binsize, frac, int(chunk.r.sum()), int(chunk.n.sum()), len(chunk)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.48, wspace=0.34)
    for row, (subj, region) in enumerate(SUBJECTS):
        sess, uid, c1 = pick_unit(subj)
        s = load_session(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
        spk = _spikes(s, uid)
        # --- pulse ---
        axp = fig.add_subplot(gs[row, 0]); pp = pulse_peths(spk, s)
        for nm, col in (("fast", "#d6322a"), ("slow", "#2b6fb3")):
            t, m, lo, hi = pp[nm]
            axp.plot(t, m, color=col, lw=2, label=nm); axp.fill_between(t, lo, hi, color=col, alpha=0.2)
        axp.axvline(0, color="0.7", lw=0.8); axp.axhline(0, color="0.7", lw=0.8)
        axp.set_title(f"{subj} ({region})  u{uid}\nTF pulse  (C1={c1:.2f})", fontsize=11)
        axp.set_ylabel("Δ firing (Hz)"); axp.set_xlabel("t from pulse (s)"); axp.legend(frameon=False, fontsize=8)
        # --- Hit vs Miss @ Change_ON ---
        axc = fig.add_subplot(gs[row, 1])
        for oc, col, ls in (("hit", "#1a7f37", "-"), ("miss", "0.45", "--")):
            r = outcome_peth(spk, s, oc, "Change_ON")
            if r:
                t, m, lo, hi, n = r
                axc.plot(t, m, color=col, lw=2, ls=ls, label=f"{oc} (n={n})")
                axc.fill_between(t, lo, hi, color=col, alpha=0.18)
        pw = EVENT_RESPONSIVENESS_WINDOWS["Change_ON"][1]
        axc.axvspan(pw[0], pw[1], color="0.9", zorder=0); axc.axvline(0, color="0.7", lw=0.8)
        axc.set_title("Hit vs Miss @ Change_ON", fontsize=10); axc.set_xlabel("t from Change_ON (s)")
        axc.legend(frameon=False, fontsize=8)
        # --- FA @ early-lick ---
        axf = fig.add_subplot(gs[row, 2])
        r = outcome_peth(spk, s, "fa", "FA")
        if r:
            t, m, lo, hi, n = r
            axf.plot(t, m, color="#8856a7", lw=2, label=f"FA (n={n})")
            axf.fill_between(t, lo, hi, color="#8856a7", alpha=0.2)
        fw = EVENT_RESPONSIVENESS_WINDOWS["FA"][1]
        axf.axvspan(fw[0], fw[1], color="0.9", zorder=0); axf.axvline(0, color="0.7", lw=0.8)
        axf.set_title("FA @ early-lick", fontsize=10); axf.set_xlabel("t from FA (s)")
        axf.legend(frameon=False, fontsize=8)
        # --- recruitment: 5-session chronological bins ---
        axr = fig.add_subplot(gs[row, 3]); rb = recruit_bins(subj)
        axr.plot([b for b, *_ in rb], [f for _, f, *_ in rb], "-o", color="#5aa469", ms=6, lw=1.5)
        for b, f, nr, nu, ns in rb:
            axr.text(b, f, f"{f:.1f}", ha="center", va="bottom", fontsize=7)
        axr.set_xticks([b for b, *_ in rb])
        axr.set_xticklabels([f"{b*BINSIZE+1}-{b*BINSIZE+ns}" for b, *_, ns in rb], fontsize=7, rotation=30)
        axr.set_ylabel("% TF-responsive"); axr.set_ylim(0, None)
        axr.set_title(f"Recruitment\n({BINSIZE}-session bins, chronological)", fontsize=10)
        axr.set_xlabel("session bin")
        for ax in (axc, axf, axr):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Representative TF-responsive cells — TF-pulse + outcome responses (95% CI) · "
                 "recruitment across 5-session bins (QC-pass sessions)", fontsize=13, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"representative_cells.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/representative_cells.png (+.pdf)")
    for subj, _ in SUBJECTS:
        sess, uid, c1 = pick_unit(subj)
        print(f"  {subj}: exemplar {sess} u{uid} C1={c1:.2f}")


if __name__ == "__main__":
    main()
