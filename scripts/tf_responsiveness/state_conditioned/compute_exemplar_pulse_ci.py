"""95% CI band for the model-free fast-MINUS-slow pulse contrast of the 6 exemplar cells
(trial bootstrap).

The exemplar figure draws each cell's GLM kernel with a bootstrap 95% CI, overlaid with a
grey model-free pulse-triggered trace. That trace is the FAST-minus-SLOW contrast (see
below), which is the correct model-free analog of the kernel and cancels within-trial
firing background; it is drawn with its own bootstrap 95% CI (it is far noisier than the
kernel, which is why the GLM is the better estimator).

RESAMPLING UNIT = THE TRIAL, not the pulse. There are ~56 fast pulses per trial and they
are NOT independent (same trial, same brain state), so a pulse-level bootstrap would
badly understate the CI. Resampling trials (and taking all pulses inside the resampled
trials) respects that correlation AND matches the resampling unit used for the kernel's
CI (compute_exemplar_ci.py), so the two bands on each panel are directly comparable.

COMPLETE-CASE pulse selection (NOT per-lag censoring). Fast pulses live in the trial
BASELINE and align_spikes_to_events does not clip at trial boundaries, so a long window
around a late-baseline pulse runs into the change and the lick. The earlier fix censored
each lag past that pulse's change — but that changes the SAMPLE with lag (pulses dropped
at long lags come from ~45-50% LOWER-firing epochs), so a single global baseline mean
drifts upward with lag and fakes a sustained ramp (composition bias, up to ~1x the peak;
caught by adversarial review). Instead we keep only pulses whose ENTIRE display window
stays before that trial's change (hit/miss) or outcome lick (fa/abort/ref) AND >=
min_after_baseline s after Baseline_ON. The surviving sample is then CONSTANT across all
lags, so the global baseline mean/SD is valid and no censoring (hence no composition bias)
is needed. This also fixes the FA lick never being cut (on fa/abort trials the cut is now
the actual lick, not the trial end = next Baseline_ON).

Procedure per cell: select complete-case pulses -> bin spikes around them -> fix the
baseline mean/SD from that (fixed) sample -> resample trials with replacement B=200x ->
mean the pulses of the resampled trials -> smooth (SIG) -> z-score -> per-lag 2.5/97.5
percentiles. Sign-aligned by the GLM KERNEL (never by the trace's own post-window sign,
which would be circular).

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
# canonical TF-pulse eligibility guard (tf_pulse.py:49-53) — same as rebuild_peth_traces_all
from visdetect.analysis.tf_pulse import TFRespPulseConfig, _outcome_time_for_trial  # noqa: E402

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
    fast, slow = pulse_times_from_tf(d, cfg)
    fast, slow = np.asarray(fast, float), np.asarray(slow, float)

    be = np.asarray(d.bin_edges, float)
    tidx = np.asarray(d.trial_index, int)

    def _ptrial(pl):                    # trial each pulse belongs to (gapped bin edges)
        bi = np.clip(np.searchsorted(be, pl, side="right") - 1, 0, len(tidx) - 1)
        return tidx[bi]
    pf, ps = _ptrial(fast), _ptrial(slow)

    # COMPLETE-CASE selection on BOTH fast and slow (replaces the composition-biased per-lag
    # censor). cut[k] = the first contaminating event on trial k: the change on hit/miss, or
    # the outcome lick on fa/abort/ref (change_time is NaN there). Keep a pulse only if its
    # WHOLE window ends before that cut AND it starts >= min_after_baseline s after
    # Baseline_ON — so the sample is identical at every lag (no composition drift) and the FA
    # lick is genuinely excluded (the old tr_end = next Baseline_ON, past the lick).
    pcfg = TFRespPulseConfig()
    strials = getattr(s, "trials", []) or []
    tr_end = {int(k): float(np.max(be[tidx == k])) for k in np.unique(tidx)}
    cut = np.full(len(trials), np.nan)
    t0arr = np.full(len(trials), np.nan)
    for k in range(len(trials)):
        t0arr[k] = float(trials[k].t_start)
        ct = float(getattr(trials[k], "change_time", np.nan))
        if np.isfinite(ct):
            cut[k] = ct
        else:
            t_out = _outcome_time_for_trial(strials[k], t0arr[k]) if k < len(strials) else None
            cut[k] = float(t_out) if t_out is not None else tr_end[int(k)]

    def _keep(pl, ptr):
        return (pl + PULSE_WIN[1] <= cut[ptr]) & (pl >= t0arr[ptr] + pcfg.min_after_baseline)
    kf, ks = _keep(fast, pf), _keep(slow, ps)
    fast, pf = fast[kf], pf[kf]
    slow, ps = slow[ks], ps[ks]

    # The model-free analog of the GLM TF kernel is the FAST-minus-SLOW contrast (a fast
    # pulse is a TF increase, slow is a decrease; the kernel is the per-unit-TF response).
    # fast and slow pulses share the same within-trial firing background (identical trial-
    # phase timing, KS p=0.55), so the contrast CANCELS that background — which the fast-only
    # PETH cannot, and which the GLM removes via its baseline/trial-start regressors. This is
    # the same background-cancelling contrast used for pulse_fwhm.
    spk = _spikes(s, int(uid))
    bf, t = align_spikes_to_events(spk, list(fast), window=PULSE_WIN, bin_size=BIN)
    bs_, _ = align_spikes_to_events(spk, list(slow), window=PULSE_WIN, bin_size=BIN)
    bf, bs_, t = np.asarray(bf, float), np.asarray(bs_, float), np.asarray(t, float)
    _, base = ALIGN["pulse"]
    bmask = (t >= base[0]) & (t < base[1])

    def _contrast(rf, rs):              # fixed sample, plain mean; Hz; baseline-subtracted
        mf = np.mean(bf[rf], axis=0) if len(rf) else np.zeros(t.size)
        ms = np.mean(bs_[rs], axis=0) if len(rs) else np.zeros(t.size)
        c = gaussian_filter1d((mf - ms) / BIN, SIG)
        return (c - np.mean(c[bmask])) * sign

    if fast.size == 0 or slow.size == 0:   # no complete-case pulse survived (should not happen)
        nanrow = np.full(t.size, np.nan)
        return (f"{sess}_u{int(uid)}", t, nanrow, nanrow, nanrow, 0, 0, np.zeros(t.size, int))

    tids = np.unique(np.concatenate([pf, ps]))
    rf_by = {int(x): np.where(pf == x)[0] for x in tids}   # empty array if trial has no fast
    rs_by = {int(x): np.where(ps == x)[0] for x in tids}
    rng = np.random.default_rng(seed)
    boots = np.empty((B_BOOT, t.size), float)
    for b in range(B_BOOT):
        samp = rng.choice(tids, size=tids.size, replace=True)      # resample TRIALS
        rf = np.concatenate([rf_by[int(x)] for x in samp])
        rs = np.concatenate([rs_by[int(x)] for x in samp])
        boots[b] = _contrast(rf, rs)
    point = _contrast(np.arange(fast.size), np.arange(slow.size))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    n_at_lag = np.full(t.size, int(fast.size + slow.size), int)    # CONSTANT sample per lag
    del s
    gc.collect()
    return (f"{sess}_u{int(uid)}", t, point, lo, hi,
            int(fast.size + slow.size), int(tids.size), n_at_lag)


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
            print(f"  {key}: {npulse} complete-case pulses / {ntrial} trials "
                  f"(constant sample across all lags: {n_at_lag[0]} = {n_at_lag[-1]})",
                  flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **save)
    print(f"wrote {OUT}  ({len(save)//3} cells, B={B_BOOT}, trial bootstrap)", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    main(workers=ap.parse_args().workers)
