"""Batch TF responsiveness across sessions (corrected pulses).

For each session computes three per-unit responsiveness fractions and writes one
summary row:
  - baseline-pulse common-mode  (subtle SENSORY probe, motor-free): post-pulse
    |z|>=3 of the all-pulse-triggered mean vs a far pre-pulse baseline.
  - miss-change short-latency    (motor-FREE sensory, no lick): |z|>=3 in
    0.03-0.15 s of the Change_ON response on MISS trials.
  - hit-change late              (MOTOR/decision): |z|>=3 in 0.25-0.55 s of the
    Change_ON response on HIT trials.

Loads pkls by ABSOLUTE path (default: the primary repo's data/pkls) so the
worktree's *fixed* code can process the full dataset without copying. Run with
PYTHONPATH pointing at the worktree src so the off-by-one fix is in effect.

Usage:
    cd /e/python_analysis/git_repos/vd_tf_phase0
    PYTHONPATH=src py scripts/tf_responsiveness/batch_responsiveness.py \
        --subject BG_046                       # all QC-manifest sessions
    PYTHONPATH=src py scripts/tf_responsiveness/batch_responsiveness.py \
        --subject BG_031 --sessions 240325,02042025,200325,210325,250325
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.core.session import load_session as core_load
from visdetect.analysis.utils import get_good_cluster_ids
from visdetect.analysis.tf_pulse import TFRespPulseConfig, _collect_pulses, _smooth_binned_activity
from visdetect.analysis.align import get_event_times_by_trial

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "data" / "cache" / "tf_selectivity"
_FIGS = _ROOT / "figures" / "tf_responsiveness"
DEFAULT_DATA = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/data/pkls"

# baseline-pulse (sensory) params
PW0, PW1, PDT, PSIG = -0.5, 0.5, 0.004, 17.0
P_BASE, P_POST = (-0.5, -0.3), (0.02, 0.25)
# change-response params
CW0, CW1, CDT, CSIG = -0.4, 0.7, 0.004, 15.0
C_BASE, C_SL, C_MO = (-0.35, -0.05), (0.03, 0.15), (0.25, 0.55)


def _trig(st, ev, tv, dt, sig):
    """Event-triggered mean rate (Hz). Pools all peri-event spikes and smooths
    ONCE -- identical to averaging per-event smoothed traces (smoothing is
    linear), but fast enough to use every pulse (no subsampling, which would
    inflate the peak-z via undersampling noise)."""
    if ev.size == 0:
        return np.zeros(tv.size)
    lo, hi = float(tv[0]), float(tv[-1] + dt)
    i0 = np.searchsorted(st, ev + lo)
    i1 = np.searchsorted(st, ev + hi)
    rel = np.concatenate([st[i0[k]:i1[k]] - ev[k] for k in range(ev.size)]) if ev.size else np.array([])
    return _smooth_binned_activity(rel, tv, (sig / 1000.0) / dt) / ev.size / dt


def _z(tr, tv, base):
    m = (tv >= base[0]) & (tv < base[1])
    sd = tr[m].std()
    return (tr - tr[m].mean()) / (sd if sd > 1e-6 else 1.0)


def _peakz(z, tv, win):
    m = (tv >= win[0]) & (tv < win[1])
    seg = z[m]
    return float(seg[np.argmax(np.abs(seg))]) if seg.size else np.nan


def analyse_session(path, max_pulses, seed=0):
    sess = core_load(path)
    cids = get_good_cluster_ids(sess)
    by = {int(c.cluster_id): np.sort(np.asarray(c.spike_times, float).ravel())
          for c in sess.clusters}
    # baseline pulses (subsample for speed; mean responsiveness is well estimated)
    fast, slow = _collect_pulses(sess, TFRespPulseConfig(trace_pre=PW0, dt=PDT,
                                                         post_window=(0.0, PW1)))
    allp = np.concatenate([fast, slow])  # use every pulse (fast _trig makes this cheap)
    ptv = np.arange(PW0, PW1, PDT)
    # change events split by outcome
    ch = np.array(get_event_times_by_trial(sess, "Change_ON"), float)
    outs = [str(getattr(t, "trialoutcome", "")).lower() for t in sess.trials]
    hit = np.array([ch[i] for i in range(len(ch)) if np.isfinite(ch[i]) and outs[i] == "hit"])
    miss = np.array([ch[i] for i in range(len(ch)) if np.isfinite(ch[i]) and outs[i] == "miss"])
    ctv = np.arange(CW0, CW1, CDT)

    n = 0
    cm = []      # baseline-pulse common-mode post-peak z
    msl = []     # miss short-lat z
    hlate = []   # hit late z
    for cid in cids:
        st = by.get(int(cid))
        if st is None or st.size == 0:
            continue
        n += 1
        zc = _z(_trig(st, allp, ptv, PDT, PSIG), ptv, P_BASE)
        cm.append(_peakz(zc, ptv, P_POST))
        if miss.size >= 10:
            msl.append(_peakz(_z(_trig(st, miss, ctv, CDT, CSIG), ctv, C_BASE), ctv, C_SL))
        if hit.size >= 10:
            hlate.append(_peakz(_z(_trig(st, hit, ctv, CDT, CSIG), ctv, C_BASE), ctv, C_MO))
    cm = np.abs(np.array(cm)); msl = np.abs(np.array(msl)); hlate = np.abs(np.array(hlate))
    frac = lambda a, t: (float(np.mean(a >= t)) if a.size else np.nan)
    return dict(n_units=n, n_fast=int(fast.size), n_slow=int(slow.size),
                n_hit=int(hit.size), n_miss=int(miss.size),
                cm_resp_frac=frac(cm, 3), cm_resp_frac5=frac(cm, 5),
                miss_sl_frac=frac(msl, 3), hit_late_frac=frac(hlate, 3))


def bg046_sessions():
    from visdetect.analysis.config import load_staging_manifest
    m = load_staging_manifest(qc_only=True)
    out = []
    for _, r in m.iterrows():
        out.append((str(int(r["session_name"])).zfill(8), str(r.get("stage", "?"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--sessions", default="", help="comma list; default=BG_046 QC manifest")
    ap.add_argument("--data-root", default=DEFAULT_DATA)
    ap.add_argument("--max-pulses", type=int, default=4000)
    args = ap.parse_args()

    if args.sessions:
        sess_list = [(s, "?") for s in args.sessions.split(",")]
    elif args.subject == "BG_046":
        sess_list = bg046_sessions()
    else:
        raise SystemExit("provide --sessions for non-BG_046 subjects")

    rows = []
    for i, (sn, stage) in enumerate(sess_list, 1):
        path = f"{args.data_root}/{args.subject}/{args.subject}_{sn}.pkl"
        if not Path(path).exists():
            print(f"[{i}/{len(sess_list)}] {sn}: MISSING pkl, skip"); continue
        t0 = time.time()
        try:
            res = analyse_session(path, args.max_pulses)
        except Exception as e:
            print(f"[{i}/{len(sess_list)}] {sn}: ERROR {type(e).__name__}: {e}"); continue
        res.update(subject=args.subject, session=sn, stage=stage)
        rows.append(res)
        print(f"[{i}/{len(sess_list)}] {sn} ({stage}): {res['n_units']}u  "
              f"baseline-pulse resp={res['cm_resp_frac']*100:.1f}%  "
              f"miss-shortlat={res['miss_sl_frac']*100:.1f}%  "
              f"hit-late(motor)={res['hit_late_frac']*100:.1f}%  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    _OUT.mkdir(parents=True, exist_ok=True)
    out_csv = _OUT / f"batch_responsiveness_{args.subject}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    if len(df):
        print("\nSUMMARY (median across sessions):")
        print(f"  baseline-pulse responsive: {df['cm_resp_frac'].median()*100:.1f}%  "
              f"(expect ~5% under null)")
        print(f"  miss short-lat sensory:    {df['miss_sl_frac'].median()*100:.1f}%")
        print(f"  hit late (motor):          {df['hit_late_frac'].median()*100:.1f}%")
        # figure
        df2 = df.sort_values("session").reset_index(drop=True)
        x = np.arange(len(df2))
        fig, ax = plt.subplots(figsize=(max(8, len(df2) * 0.4), 4.5))
        ax.plot(x, df2["cm_resp_frac"] * 100, "o-", color="seagreen", label="baseline-pulse (sensory, motor-free)")
        ax.plot(x, df2["miss_sl_frac"] * 100, "s-", color="darkorange", label="miss short-lat change (sensory, motor-free)")
        ax.plot(x, df2["hit_late_frac"] * 100, "^--", color="purple", alpha=0.6, label="hit late change (motor)")
        ax.axhline(5, color="0.6", ls=":", lw=1, label="5% chance")
        ax.set_xticks(x); ax.set_xticklabels(df2["session"], rotation=90, fontsize=7)
        ax.set_ylabel("% units responsive (|z|>=3)")
        ax.set_title(f"{args.subject}: responsiveness across sessions (corrected pulses)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _FIGS.mkdir(parents=True, exist_ok=True)
        out_png = _FIGS / f"batch_responsiveness_{args.subject}.png"
        fig.savefig(out_png, dpi=140); print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
