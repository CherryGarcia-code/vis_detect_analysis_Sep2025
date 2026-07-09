"""Component 0: rebuild per-cell z-scored PETH traces (pulse / Change_ON / FA) for
ALL 520 responsive cells — including the ~106 intermediate-width cells the cached
peth_traces.npz drops (they are the MIDDLE of the width continuum). Reuses the
heatmap trace logic (session_trial_regressors -> design -> pulse_times -> _ztrace)
but with NO transient/sustained class filter. Parallelised across sessions
(ProcessPool, BLAS pinned 1/worker); deterministic per-session pulse-subsample seed.
LOCAL ONLY (reads data/pkls/, never X:). Usage: py rebuild_peth_traces_all.py [--workers N]
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
import pandas as pd
from scipy.ndimage import gaussian_filter1d

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import REPO, MICE                                       # noqa: E402
from representative_cells import (_registry, good_dates, _spikes, load_session,  # noqa: E402
                                  get_event_times_by_trial)
from heatmap_transient_sustained import ALIGN, BIN, SIG, PULSE_CAP, MIN_EV, _cfg  # noqa: E402
from visdetect.analysis.align import align_spikes_to_events                   # noqa: E402
from visdetect.analysis.tf_glm import assemble_design, pulse_times_from_tf    # noqa: E402
from visdetect.analysis.tf_glm_data import session_trial_regressors           # noqa: E402

OUT_NPZ = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_all.npz"
NARROW, BROAD = 0.05, 0.15  # for the reference cls label only (grid kernel_fwhm)


def _ztrace(spk, times, win, base):
    if len(times) < MIN_EV:
        return None, None
    binned, t = align_spikes_to_events(spk, list(times), window=win, bin_size=BIN)
    binned = np.asarray(binned, float)
    bmask = (t >= base[0]) & (t < base[1])
    bvals = binned[:, bmask].ravel()
    mu, sd = bvals.mean(), bvals.std()
    if not np.isfinite(sd) or sd < 1e-6:
        sd = max(bvals.mean(), 1.0)
    z = gaussian_filter1d(binned.mean(0), SIG) if SIG > 0 else binned.mean(0)
    return (z - mu) / sd, t


def _outcome_times(session, event, outcome):
    et = np.asarray(get_event_times_by_trial(session, event), float)
    return [et[i] for i, tr in enumerate(session.trials)
            if str(getattr(tr, "trialoutcome", "") or "").lower() == outcome
            and i < et.size and np.isfinite(et[i])]


def _responsive_all(subj):
    r = _registry(subj)
    r = r[r.resp & r.session_date.isin(good_dates(subj))]
    return r[["session", "unit", "kernel_fwhm"]]


def _process_session(task):
    subj, sess, recs = task
    pkl = Path(REPO) / "data/pkls" / subj / f"{sess}.pkl"
    if not pkl.exists():
        return {"rows": [], "err": f"MISSING {pkl}"}
    try:
        s = load_session(str(pkl))
        cfg = _cfg()
        trials, _ = session_trial_regressors(s, cfg)
        d = assemble_design(trials, cfg)
        fast, _slow = pulse_times_from_tf(d, cfg)
        fast = np.asarray(fast, float)
        # deterministic per-session seed: zlib.crc32 is stable across processes/runs,
        # unlike str hash() (randomised by PYTHONHASHSEED) — so the pulse subsample
        # is reproducible.
        rng = np.random.default_rng(zlib.crc32(str(sess).encode()))
        if fast.size > PULSE_CAP:
            fast = np.sort(rng.choice(fast, PULSE_CAP, replace=False))
        ev = {"pulse": fast,
              "change": _outcome_times(s, "Change_ON", "hit"),
              "fa": _outcome_times(s, "FA", "fa")}
        rows = []
        for r in recs:
            uid = int(r["unit"])
            spk = np.sort(_spikes(s, uid))
            tr = {}
            for k, (win, base) in ALIGN.items():
                z, t = _ztrace(spk, ev[k], win, base)
                tr[k] = (z, t)
            fw = float(r["kernel_fwhm"])
            cls = "transient" if fw <= NARROW else ("sustained" if fw >= BROAD else "intermediate")
            rows.append({"subject": subj, "session": sess, "unit": uid, "cls": cls, "tr": tr})
        del s; gc.collect()
        return {"rows": rows, "err": None}
    except Exception as e:
        import traceback
        return {"rows": [], "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def main(n_workers=10):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = []
    for subj, _ in MICE:
        r = _responsive_all(subj)
        for sess, g in r.groupby("session"):
            tasks.append((subj, sess, g[["unit", "kernel_fwhm"]].to_dict("records")))
    n_workers = max(1, min(n_workers, len(tasks)))
    print(f"START rebuild | {len(tasks)} sessions | {sum(len(t[2]) for t in tasks)} cells "
          f"| {n_workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_process_session, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            res = fut.result(); results.append(res); done += 1
            print(f"  [{done}/{len(tasks)}] {'ERR '+res['err'].splitlines()[0] if res['err'] else str(len(res['rows']))+' cells'}", flush=True)

    # Determine each alignment's time axis (first non-None), then assemble padded mats.
    all_rows = [row for res in results for row in res["rows"]]
    tax = {k: None for k in ALIGN}
    for row in all_rows:
        for k in ALIGN:
            z, t = row["tr"][k]
            if t is not None and tax[k] is None:
                tax[k] = np.asarray(t, float)
    out = {"meta_subject": np.array([r["subject"] for r in all_rows]),
           "meta_session": np.array([r["session"] for r in all_rows]),
           "meta_unit": np.array([r["unit"] for r in all_rows]),
           "meta_cls": np.array([r["cls"] for r in all_rows])}
    for k in ALIGN:
        L = len(tax[k]) if tax[k] is not None else 0
        M = np.full((len(all_rows), L), np.nan)
        for i, row in enumerate(all_rows):
            z, t = row["tr"][k]
            if z is not None and len(z) == L:
                M[i] = z
        out[f"mat_{k}"] = M
        out[f"t_{k}"] = tax[k] if tax[k] is not None else np.zeros(0)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **out)
    errs = [r["err"] for r in results if r["err"]]
    print(f"wrote {OUT_NPZ} | {len(all_rows)} cells | {len(errs)} session errors", flush=True)
    print(f"cls counts: {pd.Series(out['meta_cls']).value_counts().to_dict()}", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10)
    main(n_workers=ap.parse_args().workers)
