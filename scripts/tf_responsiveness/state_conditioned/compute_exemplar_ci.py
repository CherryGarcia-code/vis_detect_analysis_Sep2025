"""Per-cell 95% CI bands for the 6 exemplar GLM TF kernels (trial bootstrap).

The exemplar figure plots single-cell GLM TF kernels; a single fitted kernel is a
point estimate. This script puts a 95% CI on each by a TRIAL BOOTSTRAP of the exact
GLM refit that produced the cached kernels: resample the cell's trials with
replacement, refit the ridge-Poisson at the point-estimate lambda (fast_fit selects
one lambda per unit), read off the TF FIR coefficient block, repeat B=200 times, and
take per-lag 2.5/50/97.5 percentiles. Each bootstrap kernel is sign-aligned (fixed
sign from the point estimate) and display-smoothed with the SAME sigma the figure
uses, so the band is a valid envelope around the plotted line.

Validated (prototype): a single full-data fit at the point-estimate lambda reproduces
the cached fold-mean kernel to corr 1.0000, so the 1-fit-per-bootstrap is faithful
and ~10x cheaper than replicating the 10-fold CV per resample.

Parallelised across (cell, bootstrap-chunk) with a ProcessPool, BLAS pinned to 1
thread/worker (env set before numpy import; Windows spawn re-runs this top-level so
every worker inherits it). LOCAL ONLY (reads data/pkls/, never X:). Writes
data/cache/tf_glm_bg046/exemplar_kernel_ci.npz.

Usage:  py scripts/tf_responsiveness/state_conditioned/compute_exemplar_ci.py [--workers N]
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
import zlib
from pathlib import Path

# Pin BLAS to 1 thread BEFORE numpy import (spawn re-runs this in every worker).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402
from scipy.ndimage import gaussian_filter1d                            # noqa: E402

_HERE = str(Path(__file__).resolve().parent)
_CB = str(Path(_HERE).parents[0] / "cluster_bg")
for _p in (_HERE, _CB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from continuum_common import load_width_metrics, REPO                  # noqa: E402
from exemplar_kernels_continuum import (                              # noqa: E402
    select_exemplars, _load_kernels, DISPLAY_SMOOTH,
)
from tf_glm_bg_task import _cfg                                        # noqa: E402
from visdetect.core.session import load_session                       # noqa: E402
from visdetect.analysis.tf_glm import (                               # noqa: E402
    assemble_design, count_vector, make_trial_folds, _select_lambda_once, _fit_one,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors   # noqa: E402

B_TOTAL = 200
N_CHUNKS = 3
OUT = Path(REPO) / "data/cache/tf_glm_bg046/exemplar_kernel_ci.npz"


def _boot_chunk(task):
    """Worker: n_boot trial-bootstrap refits of ONE cell; return sign-aligned,
    smoothed TF kernels (n_boot, n_lags). Module-level + picklable (spawn)."""
    subj, sess, uid, seed, n_boot, sign = task
    pkl = Path(REPO) / "data/pkls" / subj / f"{sess}.pkl"
    s = load_session(str(pkl))
    cfg = _cfg("log2")
    trials, units = session_trial_regressors(s, cfg)
    d = assemble_design(trials, cfg)
    tf_sl = d.col_groups["tf"]
    y = count_vector(trials, units[int(uid)], d)
    folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
    lam = _select_lambda_once(d.X, y, folds, cfg)     # point-estimate lambda (deterministic)
    tids = np.unique(d.trial_index)
    rows_by = {int(t): np.where(d.trial_index == t)[0] for t in tids}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_boot)):
        samp = rng.choice(tids, size=tids.size, replace=True)     # resample TRIALS
        rows = np.concatenate([rows_by[int(t)] for t in samp])
        m = _fit_one(d.X[rows], y[rows], lam)
        out.append(gaussian_filter1d(m.coef_[tf_sl] * sign, DISPLAY_SMOOTH))
    del s
    gc.collect()
    return (f"{sess}_u{int(uid)}", np.asarray(out, float))


def _chunk_sizes(total, n):
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def main(workers=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    d = load_width_metrics()
    kmap, lags = _load_kernels()
    sus, tra = select_exemplars(d, kmap)
    cells = pd.concat([sus, tra], ignore_index=True)

    tasks = []
    for _, r in cells.iterrows():
        Kc = np.asarray(kmap[(str(r.session), int(r.unit))], float)
        sign = float(np.sign(Kc[int(np.argmax(np.abs(Kc)))]) or 1.0)  # fixed point-estimate sign
        for cix, nb in enumerate(_chunk_sizes(B_TOTAL, N_CHUNKS)):
            seed = zlib.crc32(f"{r.session}_u{int(r.unit)}_c{cix}".encode()) & 0xffffffff
            tasks.append((str(r.subject), str(r.session), int(r.unit), seed, nb, sign))

    nw = workers or min(len(tasks), max(1, (os.cpu_count() or 4) - 2))
    print(f"START exemplar CI | {len(cells)} cells x {B_TOTAL} boot ({N_CHUNKS} chunks) "
          f"= {len(tasks)} tasks | {nw} workers (BLAS=1/worker)", flush=True)

    acc = {}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(_boot_chunk, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            bkey, arr = fut.result()
            acc.setdefault(bkey, []).append(arr)
            print(f"  [{i+1}/{len(tasks)}] {bkey}: +{arr.shape[0]} boots", flush=True)

    save = {"lags": np.asarray(lags, float)}
    for bkey, chunks in acc.items():
        allK = np.vstack(chunks)
        lo, med, hi = np.percentile(allK, [2.5, 50, 97.5], axis=0)
        save[f"{bkey}_lo"], save[f"{bkey}_med"], save[f"{bkey}_hi"] = lo, med, hi
        ip = int(np.argmax(np.abs(med)))
        print(f"  {bkey}: n={allK.shape[0]} peak lag {lags[ip]:.2f}s "
              f"CI=[{lo[ip]:+.4f},{hi[ip]:+.4f}] excl0={not (lo[ip] <= 0 <= hi[ip])}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **save)
    print(f"wrote {OUT}  ({len(acc)} cells, B={B_TOTAL})", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(workers=a.workers)
