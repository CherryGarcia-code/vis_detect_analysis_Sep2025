"""Deferred Stage-3 control: LICK-TIME SHUFFLE null.

Align TF-responsive cells to RANDOM times (same count as real hit-licks, drawn
uniformly across the task span) instead of the real lick, z-scored to the SAME
2 s pre-change baseline. If the pre-lick preparatory ramp is genuinely lick-locked,
the shuffled fraction-active is flat (~baseline) — the observed ramp cannot be a
slow-drift / arousal artifact that merely happens to align. Responsive cells only
(they carry the transient/sustained headline); LOCAL ProcessPool.

Usage:  py licktime_shuffle_control.py [--workers N]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import gc
import sys
import zlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_common as C
from build_prep_cache import unit_lick_ztrace
from visdetect.core.session import load_session
from visdetect.analysis.align import get_event_times_by_trial
from visdetect.analysis.preparatory import active_mask


def _process(task):
    subj, sess, recs = task
    pkl = C.REPO / "data/pkls" / subj / f"{sess}.pkl"
    if not pkl.exists():
        return {"rows": [], "err": f"MISSING {pkl}"}
    try:
        s = load_session(str(pkl))
        change = np.asarray(get_event_times_by_trial(s, "Change_ON"), float)
        licks = np.asarray(get_event_times_by_trial(s, "Hit"), float)
        rt = licks - change
        licks = np.where(np.isfinite(rt) & (rt >= C.MIN_RT), licks, np.nan)
        change_t = change[np.isfinite(change)]
        lick_t = licks[np.isfinite(licks)]
        if len(lick_t) < C.MIN_LICKS or len(change_t) < 1:
            del s
            return {"rows": [], "err": None}
        rng = np.random.default_rng(zlib.crc32(str(sess).encode()))
        lo, hi = float(np.min(change_t)) - 2.0, float(np.max(change_t)) + 2.0
        rand_t = np.sort(rng.uniform(lo, hi, len(lick_t)))     # random times, same count
        rows = []
        for r in recs:
            uid = int(r["unit"])
            spk = C.spikes_for(s, uid)
            z, t, n = unit_lick_ztrace(spk, list(rand_t), list(change_t),
                                       lick_win=C.LICK_WIN, base_win=C.BASE_WIN,
                                       bin_s=C.BIN, sigma_bins=C.SIG_BINS)
            if z is None:
                continue
            rows.append({"cls": C.class_from_fwhm(float(r["kernel_fwhm"])), "z": z, "t": t})
        del s
        gc.collect()
        return {"rows": rows, "err": None}
    except Exception as e:
        import traceback
        return {"rows": [], "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def main(n_workers=10):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = []
    for subj, _ in C.MICE:
        r = C.load_registry(subj)
        r = r[r.resp & r.session_date.isin(C.good_dates(subj))]
        for sess, g in r.groupby("session"):
            tasks.append((subj, sess, g[["unit", "kernel_fwhm"]].to_dict("records")))
    print(f"START licktime-shuffle | {len(tasks)} sessions | {n_workers} workers", flush=True)
    rows, errs = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for fut in as_completed([ex.submit(_process, t) for t in tasks]):
            res = fut.result()
            rows += res["rows"]
            if res["err"]:
                errs.append(res["err"])
    if not rows:
        print("NO ROWS", flush=True)
        return
    t = np.asarray(rows[0]["t"], float)
    cls = np.array([r["cls"] for r in rows])
    Z = np.array([r["z"] for r in rows])
    A = active_mask(Z)
    bmask = (t >= C.BASE_FRAC_WIN[0]) & (t <= C.BASE_FRAC_WIN[1])
    print(f"SHUFFLED lick-time control | {len(rows)} responsive cells | {len(errs)} errors", flush=True)
    print("(compare to REAL peaks: transient ~0.64, sustained ~0.88; shuffled should be ~0)", flush=True)
    for c in ("transient", "sustained"):
        m = cls == c
        if not m.any():
            continue
        f = np.nanmean(A[m], 0)
        f = f - np.nanmean(f[bmask])
        print(f"  {c}: n={int(m.sum())}  SHUFFLED peak_frac={np.nanmax(f):+.3f} @ {t[np.nanargmax(f)]:+.2f}s  "
              f"mean|frac|={np.nanmean(np.abs(f)):.3f}", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    main(n_workers=a.workers)
