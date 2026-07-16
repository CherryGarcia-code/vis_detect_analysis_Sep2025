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
# The project's canonical TF-pulse eligibility guard (tf_pulse.py:49-53): >=1 s after
# Baseline_ON, >=1 s before the change, >=2 s before an fa/abort/ref lick. The GLM does not
# need it (it regresses change/lick out); a raw PETH plot has no such protection and does.
from visdetect.analysis.tf_pulse import (                                        # noqa: E402
    TFRespPulseConfig, _outcome_time_for_trial,
)
from visdetect.analysis.align import align_spikes_to_events                   # noqa: E402
from visdetect.analysis.tf_glm import assemble_design, pulse_times_from_tf    # noqa: E402
from visdetect.analysis.tf_glm_data import session_trial_regressors           # noqa: E402

OUT_NPZ = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_all.npz"
NARROW, BROAD = 0.05, 0.15  # for the reference cls label only (grid kernel_fwhm)


def _ztrace(spk, times, win, base, max_lag=None):
    """PETH z-scored to a pre-event baseline.

    `max_lag` (per event, seconds) CENSORS lags that run past that event's valid horizon.
    Needed for the fast TF pulse: pulses live in the trial BASELINE and
    align_spikes_to_events does NOT clip at trial boundaries, so a +0.8 s window around a
    pulse late in the baseline runs into the CHANGE and the LICK. Those (large) responses
    contaminate the LATE lags — and because their size scales with the cell's change/lick
    coupling, which itself correlates with kernel width, the contamination can FAKE a
    width->duration relationship. Censored bins become NaN and drop out of the mean, so n
    falls with lag (honest)."""
    if len(times) < MIN_EV:
        return None, None
    binned, t = align_spikes_to_events(spk, list(times), window=win, bin_size=BIN)
    binned = np.asarray(binned, float)
    t = np.asarray(t, float)
    if max_lag is not None:
        binned = np.where(t[None, :] < np.asarray(max_lag, float)[:, None], binned, np.nan)
    bmask = (t >= base[0]) & (t < base[1])
    bvals = binned[:, bmask].ravel()
    mu, sd = float(np.nanmean(bvals)), float(np.nanstd(bvals))
    if not np.isfinite(sd) or sd < 1e-6:
        sd = max(mu, 1.0)
    m = np.nanmean(binned, axis=0)
    z = gaussian_filter1d(m, SIG) if SIG > 0 else m
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
        # PULSE_CAP is now None (use ALL ~41k fast pulses): the old 600-pulse subsample
        # left the raw pulse PETH noise-dominated. Guard kept so a cap can be restored.
        if PULSE_CAP is not None and fast.size > PULSE_CAP:
            fast = np.sort(rng.choice(fast, PULSE_CAP, replace=False))

        # ── COMPLETE-CASE PULSE SELECTION ──────────────────────────────────────
        # Fast pulses sit in the trial BASELINE and align_spikes_to_events does NOT clip at
        # trial boundaries, so a +0.8 s window around a pulse late in the baseline runs into
        # the CHANGE and/or the LICK. Their (large) responses contaminate the late lags, and
        # because their size scales with the cell's change/lick coupling — which correlates
        # with kernel width — they can FAKE a width->duration relationship.
        #
        # ⚠️ Do NOT fix this by per-lag censoring (NaN-ing offending lags): that changes the
        # SAMPLE COMPOSITION with lag (pulses dropped at long lags have ~45-50% LOWER
        # baseline firing than survivors), so a single global mu then drifts upward with lag
        # — trading one artifact for another. (Caught by adversarial review.)
        #
        # ⚠️ And "clip to the trial" protects nothing: the design's t_end is the NEXT trial's
        # Baseline_ON (tf_glm_data.py:293, 481-482), a median ~4.2 s PAST the first lick.
        #
        # THE FIX IS THE PROJECT'S OWN LEAKAGE GUARD. TFRespPulseConfig (tf_pulse.py:49-53)
        # already defines which TF pulses are eligible, and `_collect_pulses` applies it:
        #     min_after_baseline           = 1.0 s   (>= 1 s AFTER Baseline_ON)
        #     min_before_change            = 1.0 s   (>= 1 s BEFORE the change)
        #     min_before_outcome_fa_abort  = 2.0 s   (>= 2 s BEFORE an fa/abort/ref lick)
        # The GLM's pulse_times_from_tf applies NONE of these — it only thresholds the TF
        # vector — which is the root cause of the change/lick contamination. Apply the
        # canonical guard here, reusing the project's own `_outcome_time_for_trial` so this
        # cannot drift from the reference implementation.
        # (Threshold stays the GLM's 0.5 s.d., per Khilkevich Methods p17 — the guards are a
        # separate concern from the pulse-detection threshold.)
        pcfg = TFRespPulseConfig()
        be = np.asarray(d.bin_edges, float)
        tix = np.asarray(d.trial_index)
        bi = np.clip(np.searchsorted(be, fast, side="right") - 1, 0, tix.size - 1)
        ptr = tix[bi]
        keep = np.zeros(fast.size, bool)
        strials = getattr(s, "trials", []) or []
        for k in range(len(trials)):
            sel = np.where(ptr == k)[0]
            if not sel.size:
                continue
            t0 = float(trials[k].t_start)
            ct = float(getattr(trials[k], "change_time", np.nan))
            t_out = (_outcome_time_for_trial(strials[k], t0) if k < len(strials) else None)
            ok = fast[sel] >= (t0 + pcfg.min_after_baseline)
            if np.isfinite(ct):
                ok &= fast[sel] <= (ct - pcfg.min_before_change)
            if t_out is not None:
                ok &= fast[sel] <= (float(t_out) - pcfg.min_before_outcome_fa_abort)
            keep[sel] = ok
        n_all, n_clean = int(fast.size), int(keep.sum())
        fast = fast[keep]

        ev = {"pulse": fast,
              "change": _outcome_times(s, "Change_ON", "hit"),
              "fa": _outcome_times(s, "FA", "fa")}

        rows = []
        for r in recs:
            uid = int(r["unit"])
            spk = np.sort(_spikes(s, uid))
            tr = {}
            for k, (win, base) in ALIGN.items():
                z, t = _ztrace(spk, ev[k], win, base)      # no per-lag censoring needed now
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
