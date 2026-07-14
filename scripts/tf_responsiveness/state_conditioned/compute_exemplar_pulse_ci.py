"""95% CI band for the RAW fast-pulse PETH of the 6 exemplar cells (trial bootstrap).

The exemplar figure draws each cell's GLM kernel with a bootstrap 95% CI, but the grey
raw pulse-triggered average was drawn as if it were exact. It is not — and now that the
grey trace is used as INDEPENDENT (model-free) corroboration of the kernel, it needs its
own error band.

RESAMPLING UNIT = THE TRIAL, not the pulse. There are ~56 fast pulses per trial and they
are NOT independent (same trial, same brain state), so a pulse-level bootstrap would
badly understate the CI. Resampling trials (and taking all pulses inside the resampled
trials) respects that correlation AND matches the resampling unit used for the kernel's
CI (compute_exemplar_ci.py), so the two bands on each panel are directly comparable.

Procedure per cell: bin spikes around ALL fast pulses -> fix the baseline mean/SD from
the full data (they are normalisation constants, not the estimand) -> resample trials
with replacement B=200x -> mean the pulses of the resampled trials -> smooth (SIG) ->
z-score -> per-lag 2.5/97.5 percentiles. Sign-aligned by the GLM KERNEL (never by the
trace's own post-window sign, which would be circular).

Writes data/cache/tf_glm_bg046/exemplar_pulse_ci.npz. LOCAL ONLY (reads data/pkls/).

Usage:  py scripts/tf_responsiveness/state_conditioned/compute_exemplar_pulse_ci.py
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from scipy.ndimage import gaussian_filter1d                           # noqa: E402

_HERE = str(Path(__file__).resolve().parent)
_CB = str(Path(_HERE).parents[0] / "cluster_bg")
for _p in (_HERE, _CB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from continuum_common import load_width_metrics, REPO                 # noqa: E402
from exemplar_kernels_continuum import select_exemplars, _load_kernels  # noqa: E402
from heatmap_transient_sustained import ALIGN, BIN, SIG, _cfg          # noqa: E402
from representative_cells import _spikes                              # noqa: E402
from visdetect.core.session import load_session                       # noqa: E402
from visdetect.analysis.tf_glm import assemble_design, pulse_times_from_tf  # noqa: E402
from visdetect.analysis.tf_glm_data import session_trial_regressors   # noqa: E402
from visdetect.analysis.align import align_spikes_to_events           # noqa: E402

B_BOOT = 200
# Longer window than the shared cache's ALIGN["pulse"] = (-0.4, 0.8): the GLM kernel runs
# to lag 1.45 s, so a 0.8 s PETH would leave the grey trace stopping short of the coloured
# one. Recomputing 6 cells over the full range costs minutes (a full-cache rebuild would
# cost ~1 h), so the exemplar figure gets its grey trace + CI from HERE, not from the cache.
PULSE_WIN = (-0.4, 1.45)
OUT = Path(REPO) / "data/cache/tf_glm_bg046/exemplar_pulse_ci.npz"


def _boot_cell(task):
    subj, sess, uid, sign, seed = task
    s = load_session(str(Path(REPO) / "data/pkls" / subj / f"{sess}.pkl"))
    cfg = _cfg()          # heatmap_transient_sustained._cfg takes no args (log2 encoding)
    trials, units = session_trial_regressors(s, cfg)
    d = assemble_design(trials, cfg)
    fast, _ = pulse_times_from_tf(d, cfg)
    fast = np.asarray(fast, float)

    # the trial each pulse belongs to (searchsorted on the stitched, gapped bin edges)
    bi = np.searchsorted(np.asarray(d.bin_edges, float), fast, side="right") - 1
    bi = np.clip(bi, 0, len(d.trial_index) - 1)
    ptrial = np.asarray(d.trial_index, int)[bi]

    _, base = ALIGN["pulse"]          # same baseline window; LONGER display window
    win = PULSE_WIN
    binned, t = align_spikes_to_events(_spikes(s, int(uid)), list(fast), window=win, bin_size=BIN)
    binned = np.asarray(binned, float)
    t = np.asarray(t, float)

    # ⚠️ CENSOR LAGS THAT RUN PAST THE CHANGE EVENT. Fast pulses live in the BASELINE
    # period, and align_spikes_to_events does NOT clip at trial boundaries — so a long
    # window around a pulse late in the baseline runs straight into the change stimulus
    # and the lick, and their (large) responses contaminate the long lags. Unmasked, that
    # made even TRANSIENT cells look like they had a sustained raw response. For each
    # pulse, keep only lags that stay BEFORE that trial's change (or, on trials with no
    # change, before the trial ends); everything later becomes NaN and is dropped from the
    # per-lag mean. n therefore falls with lag — that is honest, not a bug.
    tr_end = {int(k): float(np.max(np.asarray(d.bin_edges, float)[np.asarray(d.trial_index) == k]))
              for k in np.unique(d.trial_index)}
    cut = np.array([
        float(trials[k].change_time) if np.isfinite(getattr(trials[k], "change_time", np.nan))
        else tr_end[int(k)]
        for k in range(len(trials))], float)
    max_lag = cut[ptrial] - fast                      # seconds of usable lag per pulse
    binned = np.where(t[None, :] < max_lag[:, None], binned, np.nan)
    n_at_lag = np.sum(np.isfinite(binned), axis=0)

    # baseline normalisation constants — from the FULL data, held fixed across bootstraps
    bmask = (t >= base[0]) & (t < base[1])
    bvals = binned[:, bmask].ravel()
    mu, sd = float(np.nanmean(bvals)), float(np.nanstd(bvals))
    if not np.isfinite(sd) or sd < 1e-6:
        sd = max(mu, 1.0)

    def _z(rows):                       # NaN-aware: lags past the change are censored
        return (gaussian_filter1d(np.nanmean(binned[rows], axis=0), SIG) - mu) / sd * sign

    tids = np.unique(ptrial)
    rows_by = {int(x): np.where(ptrial == x)[0] for x in tids}
    rng = np.random.default_rng(seed)
    boots = np.empty((B_BOOT, binned.shape[1]), float)
    for b in range(B_BOOT):
        samp = rng.choice(tids, size=tids.size, replace=True)      # resample TRIALS
        rows = np.concatenate([rows_by[int(x)] for x in samp])
        boots[b] = _z(rows)
    point = _z(np.arange(binned.shape[0]))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    del s
    gc.collect()
    return (f"{sess}_u{int(uid)}", np.asarray(t, float), point, lo, hi,
            int(fast.size), int(tids.size), n_at_lag)


def main(workers=6):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    d = load_width_metrics()
    kmap, lags = _load_kernels()
    sus, tra = select_exemplars(d, kmap)
    cells = pd.concat([sus, tra], ignore_index=True)

    tasks = []
    for _, r in cells.iterrows():
        K = np.asarray(kmap[(str(r.session), int(r.unit))], float)
        sign = float(np.sign(K[int(np.argmax(np.abs(K)))]) or 1.0)   # KERNEL's sign
        seed = zlib.crc32(f"pulseci_{r.session}_u{int(r.unit)}".encode()) & 0xffffffff
        tasks.append((str(r.subject), str(r.session), int(r.unit), sign, seed))

    print(f"START exemplar pulse CI | {len(tasks)} cells x {B_BOOT} trial-bootstraps "
          f"| {workers} workers", flush=True)
    save = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
        for fut in as_completed([ex.submit(_boot_cell, t) for t in tasks]):
            key, t, point, lo, hi, npulse, ntrial, n_at_lag = fut.result()
            save[f"{key}_lo"], save[f"{key}_hi"] = lo, hi
            save[f"{key}_point"] = point
            save[f"{key}_n"] = n_at_lag
            save["t_pulse"] = t
            print(f"  {key}: {npulse} pulses / {ntrial} trials | pulses surviving the "
                  f"change-censor: {n_at_lag[0]} at lag {t[0]:+.2f}s -> {n_at_lag[-1]} at "
                  f"lag {t[-1]:+.2f}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **save)
    print(f"wrote {OUT}  ({len(save)//3} cells, B={B_BOOT}, trial bootstrap)", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    main(workers=ap.parse_args().workers)
