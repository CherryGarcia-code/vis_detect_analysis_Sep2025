"""Stage 1: per-unit MEAN lick-aligned PETH z-scored to the 2 s pre-CHANGE baseline
(Khilkevich & Lohse Fig 5). TF-responsive + non-TF, all 3 mice. LOCAL ProcessPool.
Usage: py build_prep_cache.py [--lick hit|fa] [--workers N]"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import gc
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_common as C
from visdetect.core.session import load_session
from visdetect.analysis.align import get_event_times_by_trial, align_spikes_to_events
from visdetect.analysis.preparatory import baseline_mean_sd, zscore_trace


def unit_lick_ztrace(spikes, lick_times, change_times, *, lick_win, base_win, bin_s, sigma_bins):
    """Return (z, t, n_licks). z = (smoothed mean lick-PETH - mu_bl)/sd_bl, with
    (mu_bl, sd_bl) from pooled pre-change baseline bins (unsmoothed)."""
    lick_times = [x for x in lick_times if np.isfinite(x)]
    change_times = [x for x in change_times if np.isfinite(x)]
    if len(lick_times) < 1 or len(change_times) < 1 or len(spikes) == 0:
        return None, None, len(lick_times)
    b_binned, _bt = align_spikes_to_events(spikes, change_times, window=base_win, bin_size=bin_s)
    mu, sd = baseline_mean_sd(b_binned)
    l_binned, t = align_spikes_to_events(spikes, lick_times, window=lick_win, bin_size=bin_s)
    m = np.nanmean(np.asarray(l_binned, float), axis=0)
    if sigma_bins > 0:
        m = gaussian_filter1d(m, sigma_bins)
    return zscore_trace(m, mu, sd), np.asarray(t, float), len(lick_times)


def _select(subj, resp):
    r = C.load_registry(subj)
    r = r[(r.resp == resp) & r.session_date.isin(C.good_dates(subj))]
    return r[["session", "unit", "kernel_fwhm"]]


def _process_session(task):
    subj, sess, recs, lick, resp = task
    pkl = C.REPO / "data/pkls" / subj / f"{sess}.pkl"
    if not pkl.exists():
        return {"rows": [], "err": f"MISSING {pkl}", "dropped": 0}
    try:
        s = load_session(str(pkl))
        change = np.asarray(get_event_times_by_trial(s, "Change_ON"), float)  # hit+miss (valid)
        lick_ev = "Hit" if lick == "hit" else "FA"
        licks = np.asarray(get_event_times_by_trial(s, lick_ev), float)        # finite = matching trials
        if lick == "hit":  # >=MIN_RT s from change (paper Fig 6 rule)
            rt = licks - change
            licks = np.where(np.isfinite(rt) & (rt >= C.MIN_RT), licks, np.nan)
        change_t = change[np.isfinite(change)]
        lick_t = licks[np.isfinite(licks)]
        rows, dropped = [], 0
        for r in recs:
            uid = int(r["unit"])
            spk = C.spikes_for(s, uid)
            z, t, n = unit_lick_ztrace(spk, list(lick_t), list(change_t),
                                       lick_win=C.LICK_WIN, base_win=C.BASE_WIN,
                                       bin_s=C.BIN, sigma_bins=C.SIG_BINS)
            if z is None or n < C.MIN_LICKS:
                dropped += 1
                continue
            rows.append({"subject": subj, "session": sess, "unit": uid, "resp": bool(resp),
                         "kernel_fwhm": float(r["kernel_fwhm"]), "z": z, "t": t, "n": n})
        del s
        gc.collect()
        return {"rows": rows, "err": None, "dropped": dropped}
    except Exception as e:
        import traceback
        return {"rows": [], "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}", "dropped": 0}


def main(lick="hit", n_workers=10):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    width = C.load_width()
    wmap = {(str(r.subject), str(r.session), int(r.unit)): float(r.interp_fwhm)
            for r in width.itertuples()}
    tasks = []
    for subj, _ in C.MICE:
        for resp in (True, False):
            sel = _select(subj, resp)
            for sess, g in sel.groupby("session"):
                tasks.append((subj, sess, g[["unit", "kernel_fwhm"]].to_dict("records"), lick, resp))
    n_workers = max(1, min(n_workers, len(tasks)))
    print(f"START prep cache lick={lick} | {len(tasks)} session-jobs | {n_workers} workers", flush=True)
    rows, errs, dropped = [], [], 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_session, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            rows += res["rows"]
            dropped += res["dropped"]
            if res["err"]:
                errs.append(res["err"])
            tag = ("ERR " + res["err"].splitlines()[0]) if res["err"] else (str(len(res["rows"])) + " cells")
            print(f"  [{i+1}/{len(tasks)}] {tag}", flush=True)
    t_axis = next((r["t"] for r in rows if r["t"] is not None), np.zeros(0))
    L = len(t_axis)
    Z = np.full((len(rows), L), np.nan)
    for i, r in enumerate(rows):
        if r["z"] is not None and len(r["z"]) == L:
            Z[i] = r["z"]
    subjects = np.array([r["subject"] for r in rows])
    sessions = np.array([r["session"] for r in rows])
    units = np.array([r["unit"] for r in rows])
    resp_flag = np.array([r["resp"] for r in rows])  # from REGISTRY, carried through (not width membership)
    fwhm = np.array([r["kernel_fwhm"] for r in rows])
    interp = np.array([wmap.get((s, ss, int(u)), np.nan)
                       for s, ss, u in zip(subjects, sessions, units)])
    cls = np.array([(C.class_from_fwhm(f) if rp else "non-TF") for f, rp in zip(fwhm, resp_flag)])
    out = {
        "meta_subject": subjects, "meta_session": sessions, "meta_unit": units,
        "region": np.array([C.REGION[s] for s in subjects]),
        "resp": resp_flag, "cls": cls, "interp_fwhm": interp, "kernel_fwhm": fwhm,
        "n_licks": np.array([r["n"] for r in rows]),
        "z": Z, "t": t_axis,
    }
    outp = C.REPO / f"data/cache/preparatory_fig5/prep_{lick}.npz"
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **out)
    n_resp = int(resp_flag.sum())
    n_non = int((~resp_flag).sum())
    n_width = int(np.isfinite(interp).sum())
    print(f"wrote {outp} | {len(rows)} cells (resp {n_resp} / non-TF {n_non}; "
          f"{n_width} resp with interp_fwhm) | dropped {dropped} (<{C.MIN_LICKS} licks) | "
          f"{len(errs)} session errors", flush=True)
    print(f"per region: {pd.Series(out['region']).value_counts().to_dict()}", flush=True)
    if errs:
        print("FIRST ERRORS:", flush=True)
        for e in errs[:5]:
            print("  " + e.splitlines()[0], flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lick", choices=["hit", "fa"], default="hit")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    main(lick=a.lick, n_workers=a.workers)
