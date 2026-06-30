"""Unified event-PSTH cache for the talk-substrate figures (BG_046 striatum).

Loads each pkl ONCE, builds a population tensor per event, z-scores per unit to a
SHARED pre-event baseline (CLAUDE.md golden rule), then stores per-unit mean
z-traces for every condition slice, with full / odd / even halves (the odd/even
split lets downstream figures define a unit's modulation SIGN on held-out trials,
avoiding double-dipping).

Reuses canonical machinery (no reinventing):
  - get_trial_dataframe()  -> per-trial outcome / change_size / rt / is_go / response_time
  - classify_fa_type()     -> early/late FA via FA_RT_SPLIT
  - SMALL_/BIG_CHANGE_SIZES, EVENT_RESPONSIVENESS_WINDOWS (baseline+response windows)
  - build_population_tensor / compute_zscore_normalized / smooth_psth

Trial-type correctness:
  - "Hit" event = true detection licks only: outcome 'hit' AND change_size>1
    (catch-trial licks, change_size~=1 = SDT false alarms, are EXCLUDED).
  - Change_ON hit/miss restricted to go trials (change_size>1).
  - FA = behavioural early lick (change_size-agnostic), split early/late by latency.

Output: data/cache/talk_substrate/event_psth_cache.npz
  unit_meta: session_8 (str), cluster_id (int), celltype (display str)  [length N]
  bc__{event}                         -> bin centres (s)
  {event}__{cond}__{half}             -> (N, n_bins) per-unit smoothed mean z (NaN if no trials)
  {event}__{cond}__ntr                -> (N,) full-half trial count for that cond (per session)

Usage: py scripts/talk_substrate/build_event_cache.py [--limit N]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gc
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

from visdetect.analysis import config as cfg                       # noqa: E402
from visdetect.suite.loader import load_session, list_pkl_sessions  # noqa: E402
from visdetect.analysis.behavior import get_trial_dataframe, classify_fa_type  # noqa: E402
from visdetect.analysis.constants import (                         # noqa: E402
    DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS, EVENT_RESPONSIVENESS_WINDOWS,
    SMALL_CHANGE_SIZES, BIG_CHANGE_SIZES,
)
from visdetect.analysis.utils import (                             # noqa: E402
    get_good_cluster_ids, build_population_tensor,
    compute_zscore_normalized, smooth_psth,
)

BIN = DEFAULT_BIN_SIZE
CACHE = C.CACHE_DIR / f"event_psth_cache_{C.SUBJECT}.npz"   # subject-scoped (multi-animal)

# Per-event: plot window + (baseline, sign) windows taken from the canonical
# EVENT_RESPONSIVENESS_WINDOWS, plus the condition slices to store.
_W = EVENT_RESPONSIVENESS_WINDOWS
_DAW = cfg.DEFAULT_ANALYSIS_WINDOW   # canonical (-1.0, 1.5)
STATES = ["Impulsive", "StimSens", "Disengaged"]
_STATE_CONDS = [f"state_{s}" for s in STATES]


def _win(event: str, post_lick: bool = False):
    """Plot window DERIVED from canonical constants: start at DEFAULT_ANALYSIS_WINDOW[0],
    extended earlier only if the canonical baseline lies before it (lick-aligned baseline
    is -1.75..-1.25). Post-extent = DEFAULT_ANALYSIS_WINDOW[1], or 0.75 s after a lick
    (the one deliberate display choice; no canonical post-lick window exists)."""
    base0 = _W[event][0][0]
    pre = min(_DAW[0], round(base0 - 0.25, 3))
    post = 0.75 if post_lick else _DAW[1]
    return (pre, post)


EVENTS = {
    "Baseline_ON": dict(window=_win("Baseline_ON"), baseline=_W["Baseline_ON"][0],
                        sign=_W["Baseline_ON"][1], conds=["all", "hit", "miss", "fa"]),
    "Change_ON":   dict(window=_win("Change_ON"), baseline=_W["Change_ON"][0],
                        sign=_W["Change_ON"][1],
                        conds=["all", "hit", "miss", "small", "big"] + _STATE_CONDS),
    "Hit":         dict(window=_win("Hit", post_lick=True), baseline=_W["Hit"][0],
                        sign=_W["Hit"][1], conds=["all", "small", "big"] + _STATE_CONDS),
    "FA":          dict(window=_win("FA", post_lick=True), baseline=_W["FA"][0],
                        sign=_W["FA"][1], conds=["all", "early", "late"]),
}
HALVES = ["full", "odd", "even"]

LATENTS = cfg.ROOT + "/data/cache/decision_latents/decision_latents_by_state.csv"


def state_by_trial():
    """{session_8: {trial_idx: state_label}} — join key is trial_idx (verified)."""
    lat = pd.read_csv(LATENTS)
    lat["s8"] = lat["session_name"].map(C.canon)
    out = {}
    for s8, g in lat.groupby("s8"):
        out[s8] = dict(zip(g["trial_idx"].astype(int), g["state_label"].astype(str)))
    return out


IS_046 = (C.SUBJECT == "BG_046")


def active_conds(ev: str):
    """Conditions to store for this subject — state conds only for BG_046 (latents exist there)."""
    return [c for c in EVENTS[ev]["conds"] if (IS_046 or not c.startswith("state_"))]


def cond_mask(cond: str, sub) -> np.ndarray:
    """Boolean mask over the valid-trial rows (a get_trial_dataframe slice, in tensor order)."""
    o = sub["outcome"].values
    cs = sub["change_size"].values.astype(float)
    rt = sub["rt"].values.astype(float)
    isgo = sub["is_go"].values
    if cond == "all":
        return np.ones(len(sub), bool)
    if cond == "hit":
        return (o == "hit") & isgo
    if cond == "miss":
        return (o == "miss") & isgo
    if cond == "fa":
        return o == "fa"
    if cond == "small":
        return isgo & (cs < 2.0)            # SMALL_CHANGE_SIZES = {1.25,1.35,1.5}
    if cond == "big":
        return isgo & (cs >= 2.0)           # BIG_CHANGE_SIZES = {2.0,4.0}
    if cond == "early":
        return np.array([classify_fa_type(r) == "early" for r in rt])
    if cond == "late":
        return np.array([classify_fa_type(r) == "late" for r in rt])
    if cond.startswith("state_"):
        st = cond.split("_", 1)[1]
        return isgo & (sub["state"].values == st)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ct_lookup, sessions_8 = C.celltype_and_sessions(C.SUBJECT)
    sbt = state_by_trial() if IS_046 else {}
    if args.limit:
        sessions_8 = sessions_8[: args.limit]
    print(f"[cache] {C.SUBJECT}: {len(sessions_8)} sessions"
          + (f" (LIMIT {args.limit})" if args.limit else ""))

    meta_sess, meta_cid, meta_ct = [], [], []
    # per (event,cond,half) -> list of (n_units_session, nbins) blocks; ntr similarly
    traces = {f"{ev}__{c}__{h}": [] for ev in EVENTS for c in active_conds(ev) for h in HALVES}
    ntr = {f"{ev}__{c}__ntr": [] for ev in EVENTS for c in active_conds(ev)}
    bc_by_event = {}
    nbins = {}

    for si, s8 in enumerate(sessions_8, 1):
        try:
            sess = load_session(s8)
        except Exception as e:  # noqa: BLE001
            print(f"  [{si}/{len(sessions_8)}] {s8}: load failed ({e}); skip")
            continue
        cids = get_good_cluster_ids(sess)
        nU = len(cids)
        tdf = get_trial_dataframe(sess)
        if nU == 0 or tdf.empty or "trial_idx" not in tdf.columns:
            print(f"  [{si}/{len(sessions_8)}] {s8}: no trials/units; skip")
            del sess; gc.collect(); continue
        tdf = tdf.set_index("trial_idx")
        ct = [ct_lookup.get((s8, int(c)), C.UNKNOWN) for c in cids]
        smap = sbt.get(s8, {})  # trial_idx -> state_label
        meta_sess.extend([s8] * nU)
        meta_cid.extend(int(c) for c in cids)
        meta_ct.extend(ct)

        line = [f"  [{si}/{len(sessions_8)}] {s8}: {nU}u"]
        for ev, spec in EVENTS.items():
            nb = nbins.get(ev)
            try:
                tensor, bc, valid = build_population_tensor(
                    sess, list(cids), event_name=ev,
                    window=spec["window"], bin_size=BIN)
            except ValueError:
                tensor = None
            if tensor is not None and tensor.shape[0] > 0:
                bc_by_event[ev] = bc
                nb = len(bc); nbins[ev] = nb
                z = compute_zscore_normalized(tensor, bc, spec["baseline"])  # (T,bins,U)
                # Guard: drop units with a degenerate (near-zero-variance) baseline whose z
                # explodes — sparse SPN units with ~0 spikes in the baseline window get std~0,
                # and the canonical z-score floors std at 1e-6, so one post-event spike -> huge z.
                # Real responses stay |z| < ~20; cap at 50 and NaN the offenders (per event).
                _bad = np.nanmax(np.abs(z), axis=(0, 1)) > 50.0
                if _bad.any():
                    z[:, :, _bad] = np.nan
                sub = tdf.loc[valid].reset_index()  # attrs aligned to tensor trial order
                sub["state"] = [smap.get(int(p), "") for p in valid]
            # for every cond/half, append an (nU, nb) block (NaN if unavailable)
            for cond in active_conds(ev):
                m = cond_mask(cond, sub) if (tensor is not None and tensor.shape[0] > 0) \
                    else np.zeros(0, bool)
                idx = np.where(m)[0]
                ntr[f"{ev}__{cond}__ntr"].append(np.full(nU, len(idx), dtype=int))
                halves = {"full": idx, "odd": idx[::2], "even": idx[1::2]}
                for h in HALVES:
                    sel = halves[h]
                    if nb is None:
                        block = np.zeros((nU, 1)) * np.nan  # placeholder, fixed later
                    elif len(sel) == 0:
                        block = np.full((nU, nb), np.nan)
                    else:
                        um = np.nanmean(z[sel], axis=0).T  # (U, bins)
                        block = smooth_psth(um, BIN, sigma_ms=DEFAULT_SIGMA_MS)
                    traces[f"{ev}__{cond}__{h}"].append(block)
            line.append(f"{ev}:{0 if tensor is None else tensor.shape[0]}tr")
        print(" | ".join(line))
        del sess
        gc.collect()

    # finalise: fix any placeholder (1-wide NaN) blocks to the right nbins
    out = {"unit_meta_session": np.array(meta_sess),
           "unit_meta_cluster_id": np.array(meta_cid, dtype=int),
           "unit_meta_celltype": np.array(meta_ct)}
    for ev in EVENTS:
        nb = nbins.get(ev, 1)
        out[f"bc__{ev}"] = bc_by_event.get(ev, np.array([]))
        for cond in active_conds(ev):
            for h in HALVES:
                fixed = []
                for b in traces[f"{ev}__{cond}__{h}"]:
                    if b.shape[1] != nb:
                        b = np.full((b.shape[0], nb), np.nan)
                    fixed.append(b)
                out[f"{ev}__{cond}__{h}"] = (np.vstack(fixed) if fixed
                                             else np.zeros((0, nb)))
            out[f"{ev}__{cond}__ntr"] = (np.concatenate(ntr[f"{ev}__{cond}__ntr"])
                                         if ntr[f"{ev}__{cond}__ntr"] else np.zeros(0, int))

    np.savez_compressed(CACHE, **out)
    n_units = len(meta_sess)
    print(f"[cache] wrote {CACHE}  ({n_units} units, {len(sessions_8)} sessions)")
    # quick sanity
    ctarr = np.array(meta_ct)
    print(f"[cache] celltype: " + ", ".join(
        f"{k}={int((ctarr == k).sum())}" for k in (C.NARROW, C.BROAD, C.UNKNOWN)))


if __name__ == "__main__":
    main()
