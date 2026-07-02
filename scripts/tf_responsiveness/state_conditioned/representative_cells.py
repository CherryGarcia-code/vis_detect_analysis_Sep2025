"""Representative TF-responsive cells (best exemplars, QC-pass & pre-breakdown
sessions) — TF-pulse (fast vs slow) + outcomes (Hit vs Miss @ Change_ON, FA @
early-lick), 95% bootstrap CI, no state faceting. Plus 'recruitment' = TF-
responsive fraction across CHRONOLOGICAL 5-session bins.

Rows: BG_046, BG_039, BG_031 (early-responding u239), BG_031 (late-responding) —
so the early- vs late-kernel profiles can be compared in the outcome plots.

Fixes vs the first pass:
  * dates parsed with a ROBUST DDMMYYYY/DDMMYY parser (config.parse_session_date
    mis-parses 6-digit dates -> BG_031's March sessions were mis-ordered);
  * sessions with >=50% Disengaged trials (the engagement BREAKDOWN, e.g. BG_039
    June) are DROPPED even though they pass manifest QC;
  * TF-pulse display window widened so the late fast/slow split is visible.
Conventions: align.get_event_times_by_trial + EVENT_VALID_OUTCOMES,
DEFAULT_BIN_SIZE/SIGMA_MS, EVENT_RESPONSIVENESS_WINDOWS.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import re
import datetime
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
    EVENT_RESPONSIVENESS_WINDOWS)
from visdetect.analysis.tf_glm import (TFGLMConfig, assemble_design,  # noqa: E402
    pulse_times_from_tf)
from visdetect.analysis.tf_glm_data import session_trial_regressors  # noqa: E402

REPO = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
# (subject, region, late?) — late=True forces a high-kernel-peak-t exemplar
ROWS = [("BG_046", "DMS", False), ("BG_039", "DMS", False),
        ("BG_031", "VMS", False), ("BG_031", "VMS", True)]
DISP_WIN = (-0.5, 1.0)
PULSE_DISP = (-0.4, 0.8)        # widened so the late fast/slow split shows
DISENG_MAX = 50.0              # drop sessions with >= this % Disengaged (breakdown)
BIN = DEFAULT_BIN_SIZE
SIG = DEFAULT_SIGMA_MS / 1000.0 / BIN
BINSIZE = 5
SEED = 42
OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/representative_cells")
try:
    from visdetect.viz.plotting import set_style
    set_style("talk")
except Exception:
    pass


def _pdate(d):
    """Robust DDMMYYYY / DDMMYY -> datetime.date (config.parse_session_date
    mis-parses the 6-digit form)."""
    m = re.match(r"(\d{2})(\d{2})(\d{2,4})", str(d))
    if not m:
        return None
    dd, mm, yy = m.groups()
    year = int(yy) if len(yy) == 4 else 2000 + int(yy)
    try:
        return datetime.date(year, int(mm), int(dd))
    except Exception:
        return None


def _spikes(session, uid):
    for c in session.clusters:
        if int(c.cluster_id) == int(uid):
            return np.asarray(c.spike_times, float).ravel()
    return np.zeros(0)


def _smooth(a):
    return gaussian_filter1d(a, SIG) if SIG > 0 else a


def _boot_ci(mat, n=1000, seed=SEED):
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


def good_dates(subj, max_diseng=DISENG_MAX):
    """QC-pass (manifest qc_fail=False) AND < max_diseng % Disengaged trials
    (drops the engagement-breakdown sessions that still pass QC)."""
    man = pd.read_csv(f"{REPO}/data/{subj}_staging_manifest.csv", dtype={"session_name": str})
    qc = man.loc[~man.qc_fail.astype(bool), "session_name"]
    keep = set()
    for d in qc:
        sf = Path(f"{REPO}/data/cache/state_tags/{subj}/{d}.csv")
        if sf.exists():
            dis = 100 * (pd.read_csv(sf).state_label == "Disengaged").mean()
            if dis < max_diseng:
                keep.add(d)
        else:
            keep.add(d)
    return keep


def pick_unit(subj, late=False):
    reg = _registry(subj)
    reg = reg[reg.resp & reg.session_date.isin(good_dates(subj))]
    reg = reg[reg.n_spikes >= reg.n_spikes.quantile(0.6)]
    if late:
        reg = reg[reg.kernel_peak_t.between(0.30, 0.60)]
    r = reg.sort_values("c1_r_log2", ascending=False).iloc[0]
    return r["session"], int(r["unit"]), float(r["c1_r_log2"]), float(r["kernel_peak_t"])


def outcome_peth(spikes, session, outcome, event):
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
        binned, t = align_spikes_to_events(spikes, p.tolist(), window=PULSE_DISP, bin_size=BIN)
        m, lo, hi = _boot_ci(np.asarray(binned, float))
        base = m[t < 0].mean()
        res[nm] = (t, m - base, lo - base, hi - base)
    return res


def recruit_bins(subj, binsize=BINSIZE):
    reg = _registry(subj)
    reg = reg[reg.session_date.isin(good_dates(subj))]
    per = reg.groupby("session_date").agg(n=("resp", "size"), r=("resp", "sum")).reset_index()
    per["d"] = per.session_date.map(_pdate)
    per = per.dropna(subset=["d"]).sort_values("d").reset_index(drop=True)
    out = []
    for b in range(0, len(per), binsize):
        chunk = per.iloc[b:b + binsize]
        out.append((b // binsize, 100 * chunk.r.sum() / chunk.n.sum(), len(chunk)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(17, 19.5))
    gs = gridspec.GridSpec(4, 3, hspace=0.72, wspace=0.30)
    for row, (subj, region, late) in enumerate(ROWS):
        sess, uid, c1, kt = pick_unit(subj, late=late)
        s = load_session(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
        spk = _spikes(s, uid)
        tag = "LATE" if kt >= 0.30 else "EARLY"
        # --- pulse ---
        axp = fig.add_subplot(gs[row, 0]); pp = pulse_peths(spk, s)
        for nm, col in (("fast", "#d6322a"), ("slow", "#2b6fb3")):
            t, m, lo, hi = pp[nm]
            axp.plot(t, m, color=col, lw=2, label=nm); axp.fill_between(t, lo, hi, color=col, alpha=0.2)
        axp.axvline(0, color="0.7", lw=0.8); axp.axhline(0, color="0.7", lw=0.8)
        axp.set_title(f"{subj} ({region})  u{uid}  [{tag}]\nTF pulse  C1={c1:.2f}, peak={kt:.2f}s",
                      fontsize=14, fontweight="bold")
        axp.set_ylabel("Δ firing (Hz)", fontsize=15); axp.set_xlabel("t from pulse (s)", fontsize=15)
        axp.legend(frameon=False, fontsize=12)
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
        axc.set_title("Hit vs Miss @ Change_ON", fontsize=14, fontweight="bold")
        axc.set_xlabel("t from Change_ON (s)", fontsize=15)
        axc.legend(frameon=False, fontsize=12)
        # --- FA @ early-lick ---
        axf = fig.add_subplot(gs[row, 2])
        r = outcome_peth(spk, s, "fa", "FA")
        if r:
            t, m, lo, hi, n = r
            axf.plot(t, m, color="#8856a7", lw=2, label=f"FA (n={n})")
            axf.fill_between(t, lo, hi, color="#8856a7", alpha=0.2)
        fw = EVENT_RESPONSIVENESS_WINDOWS["FA"][1]
        axf.axvspan(fw[0], fw[1], color="0.9", zorder=0); axf.axvline(0, color="0.7", lw=0.8)
        axf.set_title("FA @ early-lick", fontsize=14, fontweight="bold")
        axf.set_xlabel("t from FA (s)", fontsize=15)
        axf.legend(frameon=False, fontsize=12)
        for ax in (axp, axc, axf):
            ax.tick_params(labelsize=12)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Representative TF-responsive cells — TF-pulse + outcome responses (95% CI)\n"
                 "(QC-pass, pre-breakdown sessions; early- vs late-responding BG_031 exemplars)",
                 fontsize=16, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"representative_cells.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/representative_cells.png (+.pdf)")
    for subj, region, late in ROWS:
        sess, uid, c1, kt = pick_unit(subj, late=late)
        print(f"  {subj} {'LATE' if late else 'best'}: {sess} u{uid} C1={c1:.2f} peak={kt:.2f}s")


if __name__ == "__main__":
    main()
