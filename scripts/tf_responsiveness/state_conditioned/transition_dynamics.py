"""(b) State-TRANSITION dynamics: at a behavioural engaged<->Disengaged transition,
does the striatal population task-state offset LEAD, coincide with, or LAG the
behavioural state-label flip?

Per session (good/stable population):
  - per-trial baseline population vector = each unit's firing rate in a 0.5 s window
    from Baseline_ON (pre-stimulus, tonic), z-scored per unit across trials.
  - task-state CD = mean(engaged) - mean(Disengaged) of those z-vectors, unit-norm
    (engaged = StimSens+Impulsive). Built EXCLUDING the ±WIN trials around each
    transition (so a pre-flip shift isn't circularly induced by the CD training).
  - per-trial projection onto the CD = how "engaged-like" the population baseline is.
Find transitions: a run of >=RUN engaged trials immediately followed by >=RUN
Disengaged (and vice versa). Align the projection to the flip (trial 0 = first
trial of the NEW state), sign-flipped so every transition goes engaged->Disengaged.
Pool across transitions/sessions/subjects.

LEAD TEST: is the projection at the last engaged trials (t=-1,-2) already shifted
toward Disengaged relative to the stable pre-window (t=-6..-4)? A shift BEFORE the
label flips = neural leads behaviour.

CAVEATS: the CD separates the states by construction, so only the TIMING (lead/lag)
is informative, not the mere existence of a shift. State labels are behavioural
([[state_labeler_circularity_caveat]]). Baseline (pre-stimulus) projection is not
lick-leakage-prone but could reflect movement/arousal. EXPLORATORY — priors are a
comprehensively null static state effect.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO, _spikes, load_session, get_event_times_by_trial, good_dates  # noqa: E402
from population_geometry import _state_by_trial, ENGAGED                       # noqa: E402

BASE_WIN = 0.5        # s from Baseline_ON for the per-trial tonic baseline
RUN = 4               # min run length each side of a transition
WIN = 6               # trials aligned each side of the flip
PRE = (-6, -4)        # stable pre-window for the lead test
SUBJECTS = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/transition_dynamics")


def _count(spk, a, b):
    return np.searchsorted(spk, b) - np.searchsorted(spk, a)


def _runs_transitions(states):
    """Yield (t0, direction) where a run of >=RUN of one class is immediately
    followed by a run of >=RUN of the other. direction=+1 eng->dis, -1 dis->eng.
    t0 = index of the first trial of the NEW state."""
    out = []
    n = len(states)
    i = 0
    # compress into runs of eng('E')/dis('D'); ignore None (unlabeled) by breaking runs
    for i in range(RUN, n - RUN + 1):
        a = states[i - RUN:i]
        b = states[i:i + RUN]
        if None in a or None in b:
            continue
        if len(set(a)) == 1 and len(set(b)) == 1 and a[0] != b[0]:
            out.append((i, +1 if a[0] == "E" else -1))
    return out


def session_transitions(subj, sess):
    lab = _state_by_trial(subj, sess)
    if lab is None:
        return None
    s = load_session(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
    uids = list(getattr(s, "good_and_stable_ids", None) or getattr(s, "good_cluster_ids", None) or [])
    n = len(s.trials)
    if len(uids) < 8 or n < 30:
        del s; gc.collect(); return None
    etb = np.asarray(get_event_times_by_trial(s, "Baseline_ON"), float)
    rates = np.full((n, len(uids)), np.nan)
    for j, u in enumerate(uids):
        spk = np.sort(_spikes(s, u))
        if spk.size == 0:
            continue
        for i in range(n):
            if i < etb.size and np.isfinite(etb[i]):
                rates[i, j] = _count(spk, etb[i], etb[i] + BASE_WIN) / BASE_WIN
    del s; gc.collect()
    # z-score per unit across trials
    mu = np.nanmean(rates, 0); sd = np.nanstd(rates, 0)
    z = (rates - mu) / (sd + 1e-9)
    states = ["E" if lab.get(i) in ENGAGED else ("D" if lab.get(i) == "Disengaged" else None)
              for i in range(n)]
    trans = _runs_transitions(states)
    if not trans:
        return None
    # trials to EXCLUDE from CD training (±WIN around each transition)
    excl = set()
    for t0, _ in trans:
        excl.update(range(t0 - WIN, t0 + WIN))
    train = np.array([i for i in range(n) if i not in excl and states[i] is not None
                      and np.isfinite(z[i]).all()])
    eng = np.array([i for i in train if states[i] == "E"])
    dis = np.array([i for i in train if states[i] == "D"])
    if len(eng) < 10 or len(dis) < 10:
        return None
    cd = np.nanmean(z[eng], 0) - np.nanmean(z[dis], 0)
    cd = cd / (np.linalg.norm(cd) + 1e-9)
    proj = np.nansum(z * cd[None, :], axis=1)     # per-trial projection
    # aligned windows (sign so all go eng->dis: eng-side high, dis-side low)
    aligned = []
    for t0, direction in trans:
        seg = proj[t0 + PRE[0]:t0 + WIN]           # length 2*WIN? -6..+5 -> 12
        if len(seg) != (WIN - PRE[0]):
            continue
        if direction == -1:                        # dis->eng: flip so eng-side is 'pre'
            seg = seg[::-1]
        aligned.append(seg)
    if not aligned:
        return None
    return dict(subject=subj, session=sess, aligned=np.array(aligned), n_trans=len(aligned))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    ap.parse_args()
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    rows = []
    for subj, region in SUBJECTS:
        gd = good_dates(subj)
        pkls = sorted(Path(f"{REPO}/data/pkls/{subj}").glob("*.pkl"))
        sessions = [p.stem for p in pkls if p.stem.replace(f"{subj}_", "", 1) in gd]
        for sess in sessions:
            r = session_transitions(subj, sess)
            if r is not None:
                r["region"] = region
                rows.append(r)
                print(f"  {subj}/{sess}: {r['n_trans']} transitions", flush=True)
    if not rows:
        print("no transitions found"); return
    OUT.mkdir(parents=True, exist_ok=True)

    lags = np.arange(PRE[0], WIN)     # -6..+5
    allseg = np.vstack([r["aligned"] for r in rows])
    n_trans = len(allseg)
    n_sess = len(rows)
    lines = [f"total transitions={n_trans} over {n_sess} sessions "
             f"({sum(r['n_trans'] for r in rows if r['region']=='DMS')} DMS / "
             f"{sum(r['n_trans'] for r in rows if r['region']=='VMS')} VMS)"]

    # per-transition center (subtract each transition's own pre-window mean) so we
    # test the SHAPE/timing, then pool
    pre_mask = (lags >= PRE[0]) & (lags <= PRE[1])
    centered = allseg - np.nanmean(allseg[:, pre_mask], axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    # panel 1: mean projection trajectory
    m = np.nanmean(allseg, 0); sem = np.nanstd(allseg, 0) / np.sqrt(np.sum(np.isfinite(allseg), 0))
    ax1.axvspan(-0.5, WIN - 0.5, color="#3474ae", alpha=0.06)
    ax1.axvline(0, color="k", lw=1.2, ls="--", label="behavioural flip")
    ax1.plot(lags, m, "-o", color="#6a51a3", lw=2.2, ms=5)
    ax1.fill_between(lags, m - sem, m + sem, color="#6a51a3", alpha=0.2)
    ax1.set_xlabel("trial relative to behavioural flip (0 = first new-state trial)")
    ax1.set_ylabel("task-state projection\n(engaged-like  →  ↑)")
    ax1.set_title(f"engaged→Disengaged transition (n={n_trans})", fontsize=12, fontweight="bold")
    ax1.legend(frameon=False, fontsize=10)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # panel 2: LEAD test — projection per trial vs the pre-window, per-transition
    ax2.axhline(0, color="0.7", lw=0.8, ls=":")
    ax2.axvline(0, color="k", lw=1.2, ls="--")
    mc = np.nanmean(centered, 0); semc = np.nanstd(centered, 0) / np.sqrt(np.sum(np.isfinite(centered), 0))
    ax2.plot(lags, mc, "-o", color="#238b45", lw=2.2, ms=5)
    ax2.fill_between(lags, mc - semc, mc + semc, color="#238b45", alpha=0.2)
    # test each pre-flip trial (-1,-2,-3) vs 0 (Wilcoxon across transitions)
    for lag in (-3, -2, -1):
        col = list(lags).index(lag)
        v = centered[:, col]; v = v[np.isfinite(v)]
        p = wilcoxon(v).pvalue if len(v) >= 6 else np.nan
        star = "*" if (p == p and p < 0.05) else ""
        ax2.text(lag, mc[col] + 0.05, f"p={p:.2f}{star}", ha="center", fontsize=7.5)
        lines.append(f"LEAD test trial {lag}: centered proj median={np.median(v):+.3f} Wilcoxon-vs-0 p={p:.3f}")
    ax2.set_xlabel("trial relative to flip")
    ax2.set_ylabel("Δ projection vs stable pre-window\n(−6..−4)")
    ax2.set_title("does the neural offset shift BEFORE the flip? (lead test)", fontsize=12, fontweight="bold")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.suptitle("State-TRANSITION dynamics — does the striatal task-state offset lead the behavioural flip?  "
                 "[EXPLORATORY; CD cross-validated off-transition]", fontsize=12.5, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"transition_dynamics.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (OUT / "transition_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/transition_dynamics.png")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
