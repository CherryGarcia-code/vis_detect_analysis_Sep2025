"""Model-free width cross-check, re-run with ALL fast/slow pulses (was capped at 600).

`kernel_width_continuous.csv` carries `pulse_fwhm` — a MODEL-FREE width taken from the
fast-minus-slow pulse-PETH contrast, meant as an independent check on the GLM-derived
`interp_fwhm`. It came out weak (Spearman ~0.11) and that was recorded as "inherent".
It probably was not: it used PULSE_CAP=600, i.e. ~1.5% of the ~41k pulses/session, and
the per-pulse response sits ~20x below the spiking noise. This re-runs it with ALL pulses.

If the correlation with interp_fwhm now rises, the width axis gains a SECOND independent,
model-free confirmation (the first being that the raw fast-pulse PETH's own FWHM now
tracks interp_fwhm at rho=+0.43 after the same cap was removed).

Only the pulse PETHs are recomputed — the GLM kernels are cached, so no refit is needed.
The pulse PETH is VECTORISED here (visdetect's tf_pulse_peth loops in Python over pulses,
which is fine for 600 and hopeless for 41k).

Usage:  py scripts/tf_responsiveness/state_conditioned/recompute_pulse_fwhm_allpulses.py [--workers N]
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

_HERE = str(Path(__file__).resolve().parent)
_CB = str(Path(_HERE).parents[0] / "cluster_bg")
for _p in (_HERE, _CB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from representative_cells import REPO, _registry, good_dates          # noqa: E402
from tf_glm_bg_task import _cfg                                       # noqa: E402
from visdetect.core.session import load_session                      # noqa: E402
from visdetect.analysis.tf_glm import (                              # noqa: E402
    assemble_design, count_vector, pulse_times_from_tf, _lag_offsets,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors  # noqa: E402
from visdetect.analysis.kernel_width import interpolated_fwhm, temporal_spread  # noqa: E402

MICE = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
WIDTH_CSV = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"
OUT_CSV = Path(REPO) / "data/cache/tf_glm_bg046/pulse_fwhm_allpulses.csv"


def _peth_vec(v, bin_edges, trial_index, pulse_times, win, bin_s):
    """Vectorised event-triggered average of a per-bin signal (the Python-loop version
    in visdetect is O(n_pulses) and dies on 41k pulses). Lags are clipped to the pulse's
    OWN trial, exactly as tf_pulse_peth does."""
    v = np.asarray(v, float)
    be = np.asarray(bin_edges, float)
    ti = np.asarray(trial_index)
    offs = _lag_offsets(win, bin_s)
    tax = offs * bin_s
    p = np.asarray(pulse_times, float)
    if be.size == 0 or p.size == 0:
        return tax, np.full(offs.size, np.nan)
    idx = np.searchsorted(be, p, side="right") - 1
    ok0 = (idx >= 0) & (idx < v.size)
    idx = idx[ok0]
    cols = idx[:, None] + offs[None, :]                       # (n_pulses, n_lags)
    inb = (cols >= 0) & (cols < v.size)
    colc = np.clip(cols, 0, v.size - 1)
    same = ti[colc] == ti[idx][:, None]                       # stay inside the trial
    good = inb & same
    rows = np.where(good, v[colc], np.nan)
    with np.errstate(invalid="ignore"):
        return tax, np.nanmean(rows, axis=0)


def _session(task):
    subj, sess, recs = task
    pkl = Path(REPO) / "data/pkls" / subj / f"{sess}.pkl"
    if not pkl.exists():
        return []
    try:
        s = load_session(str(pkl))
        cfg = _cfg("log2")
        trials, units = session_trial_regressors(s, cfg)
        d = assemble_design(trials, cfg)
        fast, slow = pulse_times_from_tf(d, cfg)
        fast, slow = np.asarray(fast, float), np.asarray(slow, float)   # ALL of them
        ti, win, bs = d.trial_index, cfg.pulse_eval_win, cfg.bin_s
        rows = []
        for r in recs:
            uid = int(r["unit"])
            if uid not in units:
                continue
            y = count_vector(trials, units[uid], d)
            tax, a_fast = _peth_vec(y, d.bin_edges, ti, fast, win, bs)
            _, a_slow = _peth_vec(y, d.bin_edges, ti, slow, win, bs)
            contrast = (a_fast - a_slow) / bs
            # de-mean on the pre-pulse quarter, exactly as the original _pulse_width did
            contrast = contrast - np.median(contrast[:max(1, len(contrast) // 4)])
            tax = np.asarray(tax, float)
            rows.append(dict(subject=subj, session=sess, unit=uid,
                             n_fast=int(fast.size), n_slow=int(slow.size),
                             pulse_fwhm_all=interpolated_fwhm(contrast, tax),
                             pulse_spread_all=temporal_spread(contrast, tax)))
        del s
        gc.collect()
        return rows
    except Exception as e:                                    # a bad session must not kill the pool
        print(f"  {subj}/{sess} FAILED: {type(e).__name__}: {e}", flush=True)
        return []


def main(workers=14):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = []
    for subj, _ in MICE:
        r = _registry(subj)
        r = r[r.resp & r.session_date.isin(good_dates(subj))]
        for sess, g in r.groupby("session"):
            tasks.append((subj, str(sess), g[["unit"]].to_dict("records")))
    print(f"START pulse_fwhm (ALL pulses) | {len(tasks)} sessions | {workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_session, t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            rr = f.result()
            rows.extend(rr)
            print(f"  [{i+1}/{len(tasks)}] +{len(rr)} cells", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # ── the comparison: does the model-free width now track the GLM width? ──────
    from scipy.stats import spearmanr
    w = pd.read_csv(WIDTH_CSV, dtype={"session": str})
    m = w.merge(df, on=["subject", "session", "unit"], how="inner")
    print(f"\njoined {len(m)}/{len(w)} cells", flush=True)
    for col, lab in [("pulse_fwhm", "OLD pulse_fwhm (600-pulse cap)"),
                     ("pulse_fwhm_all", "NEW pulse_fwhm (ALL pulses)")]:
        if col not in m.columns:
            continue
        mm = m[np.isfinite(m[col]) & np.isfinite(m.interp_fwhm)]
        rho, p = spearmanr(mm[col], mm.interp_fwhm)
        print(f"  Spearman({lab:32s}, interp_fwhm) = {rho:+.3f}  p={p:.2e}  (n={len(mm)})",
              flush=True)
    print(f"\nwrote {OUT_CSV}", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    main(workers=ap.parse_args().workers)
