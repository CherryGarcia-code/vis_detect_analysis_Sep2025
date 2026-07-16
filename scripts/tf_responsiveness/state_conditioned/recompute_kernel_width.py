# scripts/tf_responsiveness/state_conditioned/recompute_kernel_width.py
"""Component A: recompute a CONTINUOUS TF-kernel width per responsive cell.

The registry only stored the 50 ms-grid `kernel_fwhm` (the raw kernel was never
cached — verified). This refits the full BG GLM LOCALLY from Session pkls (the
exact config the registry used), extracts the raw FIR kernel, and computes sub-bin
continuous width (interpolated FWHM + temporal spread). A validation gate asserts the
recomputed grid-FWHM reproduces the registry value before the continuous width is
trusted; the raw kernel vectors are saved (the missing cache).
The model-free fast-minus-slow pulse width lives in `recompute_pulse_fwhm_allpulses.py`
(all pulses + the canonical leakage guard) — NOT here; see the note by OUT_CSV.

Parallelised across SESSIONS with a ProcessPool (BLAS pinned to 1 thread/worker):
each per-cell 10-fold Poisson-GLM fit is heavy, so a serial run is ~3 h; session-
level parallelism over ~24 sessions cuts that to ~20-25 min. Each session is
independent and every per-cell fit is deterministic (fixed seed + folds), so the
parallel result is identical to serial (row order is re-sorted for reproducibility).

LOCAL ONLY — reads data/pkls/, never X:.
Usage: py recompute_kernel_width.py [--workers N]
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
from pathlib import Path

# Pin BLAS to a single thread BEFORE numpy is imported. On Windows the ProcessPool
# uses spawn, so every worker re-runs this module top-level -> each worker inherits
# 1-thread BLAS and N workers do not oversubscribe the 20 cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# BG cluster task lives in a sibling dir; add it for _cfg reuse.
_CLUSTER_BG = str(Path(_HERE).parents[0] / "cluster_bg")
if _CLUSTER_BG not in sys.path:
    sys.path.insert(0, _CLUSTER_BG)

from representative_cells import REPO, _registry, good_dates          # noqa: E402
from tf_glm_bg_task import _cfg                                        # noqa: E402
from visdetect.core.session import load_session                       # noqa: E402
from visdetect.analysis.tf_glm import (                               # noqa: E402
    assemble_design, fit_poisson_cv, make_trial_folds, _tf_kernel,
    _lag_offsets, count_vector,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors   # noqa: E402
from visdetect.analysis.kernel_width import (                         # noqa: E402
    grid_fwhm, interpolated_fwhm, temporal_spread, peak_lag,
)

MICE = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
OUT_CSV = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"
METRICS = Path(REPO) / "FIGURES/tf_glm_bg046/latency_outcome_coupling/latency_outcome_metrics.csv"
DEFAULT_WORKERS = 10

# ⚠️ REMOVED (Jul 2026): the `pulse_fwhm` / `pulse_spread` columns and their PULSE_CAP=600
# computation. That cap used ~600 of the ~41k pulses/session (~1.5%), leaving the model-free
# width NOISE — it correlated with the GLM width at rho=+0.045 (p=0.31, i.e. with nothing) and
# with its own all-pulse recomputation at only +0.048. Any statistic resting on it rested on
# nothing, and a column named `pulse_fwhm` sitting in the shipped CSV was a live footgun for the
# next script that grabbed it assuming it was the model-free width.
# THE REPLACEMENT: `pulse_fwhm_all` in data/cache/tf_glm_bg046/pulse_fwhm_allpulses.csv —
# ALL pulses AND the canonical TFRespPulseConfig leakage guard applied (rho=+0.218 vs the GLM
# width). Built by `py recompute_pulse_fwhm_allpulses.py`; joined where needed (see
# spectrum_vs_classes.py). Do not reintroduce a capped pulse width here.


def _responsive(subj):
    r = _registry(subj)
    r = r[r.resp & r.session_date.isin(good_dates(subj))]
    return r[["session", "session_date", "unit", "n_spikes", "kernel_fwhm", "kernel_peak_t"]]


def _process_session(task):
    """Worker: fit every responsive cell in one session, return rows + kernel vectors.

    Module-level and picklable (ProcessPool spawn). One bad cell is skipped (does
    not kill the session); a failed session returns an error string (does not kill
    the pool)."""
    subj, region, sess, recs = task
    pkl = Path(REPO) / "data/pkls" / subj / f"{sess}.pkl"
    if not pkl.exists():
        return {"subj": subj, "sess": sess, "rows": [], "kvecs": {},
                "err": f"MISSING pkl {pkl}", "n_req": len(recs), "n_skip_cell": 0}
    try:
        s = load_session(str(pkl))
        cfg = _cfg("log2")
        trials, units = session_trial_regressors(s, cfg)
        d = assemble_design(trials, cfg)
        folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
        lags = _lag_offsets(cfg.kern["tf"], cfg.bin_s) * cfg.bin_s
        rows, kvecs, n_skip = [], {}, 0
        for r in recs:
            uid = int(r["unit"])
            if uid not in units:
                n_skip += 1
                continue
            try:
                y = count_vector(trials, units[uid], d)
                full = fit_poisson_cv(d.X, y, cfg, folds)
                K = _tf_kernel(full, d, cfg)
                if K is None or not np.any(np.isfinite(K)):
                    n_skip += 1
                    continue
                kvecs[f"{sess}_u{uid}"] = np.asarray(K, float)  # session-scoped: cluster ids recur across sessions
                rows.append(dict(
                    subject=subj, session=sess, unit=uid, n_spikes=int(r["n_spikes"]),
                    kernel_fwhm_registry=float(r["kernel_fwhm"]),
                    grid_fwhm=grid_fwhm(K, lags),
                    interp_fwhm=interpolated_fwhm(K, lags),
                    temporal_spread=temporal_spread(K, lags),
                    kernel_peak_t_recompute=peak_lag(K, lags),
                    kernel_peak_t_registry=float(r["kernel_peak_t"]),
                ))
            except Exception as e:  # one bad cell must not kill the session
                n_skip += 1
                print(f"    {subj}/{sess} u{uid} FAILED: {type(e).__name__}: {e}", flush=True)
        del s
        gc.collect()
        return {"subj": subj, "sess": sess, "rows": rows, "kvecs": kvecs,
                "err": None, "n_req": len(recs), "n_skip_cell": n_skip}
    except Exception as e:  # a bad session must not kill the pool
        import traceback
        return {"subj": subj, "sess": sess, "rows": [], "kvecs": {},
                "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                "n_req": len(recs), "n_skip_cell": 0}


def main(n_workers=DEFAULT_WORKERS):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    metrics = pd.read_csv(METRICS)[["subject", "session", "unit", "base_hz",
                                    "change_on", "hit_ramp", "fa_ramp"]] if METRICS.exists() else None

    # Build the flat session task list across all mice.
    tasks = []
    for subj, region in MICE:
        resp = _responsive(subj)
        for sess, g in resp.groupby("session"):
            recs = g[["unit", "n_spikes", "kernel_fwhm", "kernel_peak_t"]].to_dict("records")
            tasks.append((subj, region, sess, recs))
    n_workers = max(1, min(n_workers, len(tasks)))
    print(f"START recompute | {len(tasks)} sessions | {sum(len(t[3]) for t in tasks)} responsive cells "
          f"| {n_workers} workers (BLAS=1/worker)", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_process_session, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            tag = f" ERR {res['err'].splitlines()[0]}" if res["err"] else ""
            print(f"  [{done}/{len(tasks)}] {res['subj']}/{res['sess']}: "
                  f"{len(res['rows'])} cells (skip {res['n_skip_cell']}/{res['n_req']}){tag}", flush=True)

    # Aggregate rows + per-subject kernel vectors.
    rows, kvecs_by_subj, errors = [], {}, []
    for res in results:
        rows.extend(res["rows"])
        if res["err"]:
            errors.append((res["subj"], res["sess"], res["err"]))
        kvecs_by_subj.setdefault(res["subj"], {}).update(res["kvecs"])

    cfg0 = _cfg("log2")
    lags = _lag_offsets(cfg0.kern["tf"], cfg0.bin_s) * cfg0.bin_s
    for subj, kv in kvecs_by_subj.items():
        if kv:
            npz = Path(REPO) / f"data/cache/tf_glm_bg046/kernel_vectors_{subj}.npz"
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, lags=lags, units=np.array(list(kv.keys())), **kv)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["subject", "session", "unit"]).reset_index(drop=True)
    if metrics is not None and len(df):
        df = df.merge(metrics, on=["subject", "session", "unit"], how="left")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    if errors:
        print(f"\n{len(errors)} SESSION ERROR(S):", flush=True)
        for subj, sess, err in errors:
            print(f"  {subj}/{sess}: {err.splitlines()[0]}", flush=True)

    # Completeness (the fidelity gate below would pass at 100% even on a partial run).
    n_req_total = sum(r["n_req"] for r in results)
    n_skip_total = sum(r["n_skip_cell"] for r in results)
    n_vec_total = sum(len(kv) for kv in kvecs_by_subj.values())
    print(f"COMPLETENESS: {len(df)} CSV rows + {n_vec_total} kernel vectors from "
          f"{n_req_total} requested cells ({n_skip_total} skipped, {len(errors)} session errors)",
          flush=True)

    # ── VALIDATION GATE ────────────────────────────────────────────────
    ok = np.isclose(df.grid_fwhm, df.kernel_fwhm_registry, atol=1e-9) if len(df) else np.array([])
    frac = float(ok.mean()) if len(df) else 0.0
    print(f"\nVALIDATION: grid_fwhm reproduces registry kernel_fwhm for "
          f"{int(ok.sum())}/{len(df)} cells ({100*frac:.1f}%)", flush=True)
    if frac < 0.95:
        bad = df.loc[~ok, ["subject", "session", "unit", "grid_fwhm", "kernel_fwhm_registry"]].head(20)
        print("MISMATCHES (investigate config before trusting continuous width):", flush=True)
        print(bad.to_string(index=False), flush=True)
        raise SystemExit("VALIDATION FAILED: grid FWHM does not reproduce the registry")
    print(f"wrote {OUT_CSV}  (n={len(df)} responsive cells)", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    a = ap.parse_args()
    main(n_workers=a.workers)
