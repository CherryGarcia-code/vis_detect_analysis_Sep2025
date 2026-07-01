"""Shared extraction pass for the state x neural exploratory analyses.

Loads each qualifying Expert session ONCE and caches everything the four
exploratory analyses need (sensory vs motor, pre-change ramp, decoding,
response->RT), so the analyses run fast off disk.

Per session -> analysis_suite/cache/state_neural_explore/{sid8}.npz with:
  unit_ids                (n_units,)
  # sensory (Change_ON, go-Hit) per-unit early-window z, per state
  change_evoked_Impulsive, change_evoked_StimSens   (n_units,)
  # motor per-unit peri-lick z, per state
  fa_evoked_Impulsive,  fa_evoked_StimSens          (n_units,)   (FA lick)
  hl_evoked_Impulsive,  hl_evoked_StimSens          (n_units,)   (Hit lick)
  n_fa_Impulsive, n_fa_StimSens, n_hl_Impulsive, n_hl_StimSens   scalars
  # per-trial Change_ON data (Hit+Miss, ALL change sizes incl catch) for decoding/RT
  trial_z                 (n_trials, n_units)  early-window z per trial
  trial_state             (n_trials,)  str
  trial_outcome           (n_trials,)  str  hit/miss
  trial_csize             (n_trials,)  float
  trial_is_go             (n_trials,)  bool
  trial_rt                (n_trials,)  float (change->lick, nan for miss)

Pre-change ramp (#2) reuses analysis_suite/cache/state_gain_traces.npz.
"""

import os
import gc

import numpy as np
import pandas as pd

from visdetect.suite.loader import load_session, load_staging_manifest
from visdetect.analysis.utils import build_population_tensor
from visdetect.analysis.align import get_event_times_by_trial

SUBJECT = "BG_046"
STATES = ["Impulsive", "StimSens"]
BIN = 0.01
GO_SET = {1.25, 1.35, 1.5, 2.0, 4.0}
MIN_UNITS = 8
MIN_TRIALS_STATE = 8

CHG_WIN = (-0.5, 1.0); CHG_BASE = (-0.4, -0.05); CHG_EARLY = (0.0, 0.25)
LICK_WIN = (-1.0, 0.6); LICK_BASE = (-1.0, -0.6); LICK_PERI = (-0.15, 0.15)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG_DIR = os.path.join(_REPO, "data", "cache", "state_tags", SUBJECT)
RESP_CACHE = os.path.join(_REPO, "data", "cache", "state_labeling", "responsiveness_all_sessions.csv")
OUT_DIR = os.path.join(_REPO, "data", "cache", "state_labeling", "state_neural_explore")


def _evoked_by_state(tensor, bc, valid_trials, state_of, bm, bs, peri_win, states_wanted):
    """Per-unit peri-event z by state + trial counts. Returns ({state:array}, {state:n})."""
    peri = (bc >= peri_win[0]) & (bc < peri_win[1])
    vt = np.array([int(t) for t in valid_trials])
    st = np.array([state_of.get(t) for t in vt])
    out, ns = {}, {}
    for s in states_wanted:
        m = st == s
        ns[s] = int(m.sum())
        if m.sum() == 0:
            out[s] = np.full(tensor.shape[2], np.nan)
            continue
        mt = tensor[m].mean(axis=0)                       # (bins, units)
        z = (mt - bm[None, :]) / bs[None, :]
        out[s] = z[peri, :].mean(axis=0)
    return out, ns


def process(sname, resp_all):
    sid8 = str(sname).zfill(8)
    resp = resp_all[(resp_all["session_name"].astype(str).str.zfill(8) == sid8)
                    & (resp_all["is_responsive"])]
    uids = [int(c) for c in resp["cluster_id"].tolist()]
    if len(uids) < MIN_UNITS:
        return None
    tag_csv = os.path.join(TAG_DIR, f"{sid8}.csv")
    if not os.path.exists(tag_csv):
        return None
    try:
        sess = load_session(sid8)
    except FileNotFoundError:
        return None
    present = {c.cluster_id for c in sess.clusters}
    uids = [u for u in uids if u in present]
    if len(uids) < MIN_UNITS:
        del sess; gc.collect(); return None

    tags = pd.read_csv(tag_csv)
    state_of = dict(zip(tags["trial_idx"].astype(int), tags["state_label"]))
    csize = {i: float(getattr(t, "change_size", np.nan)) for i, t in enumerate(sess.trials)}
    oc = {i: (getattr(t, "trialoutcome", "") or "").lower() for i, t in enumerate(sess.trials)}
    hit_t = np.array(get_event_times_by_trial(sess, "Hit"), float)
    chg_t = np.array(get_event_times_by_trial(sess, "Change_ON"), float)

    # ---- Change_ON tensor (Hit+Miss, all sizes incl catch) ----
    tC, bcC, vtC = build_population_tensor(
        sess, uids, event_name="Change_ON", window=CHG_WIN, bin_size=BIN,
        outcome_filter={"Hit", "Miss"})
    vtC = np.array([int(t) for t in vtC])
    base_bins = (bcC >= CHG_BASE[0]) & (bcC < CHG_BASE[1])
    early_bins = (bcC >= CHG_EARLY[0]) & (bcC < CHG_EARLY[1])
    nU = len(uids)
    sz = np.array([csize.get(t, np.nan) for t in vtC])
    go = np.array([s in GO_SET for s in sz])
    # require state coverage on go-Hit for the sensory comparison
    stC = np.array([state_of.get(t) for t in vtC])
    ocC = np.array([oc.get(t) for t in vtC])
    hit_go = go & (ocC == "hit")
    if any(int((hit_go & (stC == s)).sum()) < MIN_TRIALS_STATE for s in STATES):
        del sess, tC; gc.collect(); return None

    # shared per-unit baseline (all go trials, Change_ON)
    bm = np.array([tC[go][:, base_bins, j].ravel().mean() for j in range(nU)])
    bs = np.array([max(tC[go][:, base_bins, j].ravel().std(), 1e-6) for j in range(nU)])

    # per-trial early-window z (n_trials, n_units) for ALL Change_ON trials
    trial_z = ((tC[:, early_bins, :].mean(axis=1) - bm[None, :]) / bs[None, :])

    # sensory per-unit evoked by state (go-Hit only)
    change_ev = {}
    for s in STATES:
        m = hit_go & (stC == s)
        mt = tC[m].mean(axis=0)
        z = (mt - bm[None, :]) / bs[None, :]
        change_ev[s] = z[early_bins, :].mean(axis=0)

    # trial metadata
    trial_rt = np.array([(hit_t[t] - chg_t[t]) if (ocC[i] == "hit"
                         and np.isfinite(hit_t[t]) and np.isfinite(chg_t[t])) else np.nan
                         for i, t in enumerate(vtC)])
    del tC; gc.collect()

    # ---- FA lick tensor ----
    fa_ev, n_fa = {s: np.full(nU, np.nan) for s in STATES}, {s: 0 for s in STATES}
    try:
        tF, bcF, vtF = build_population_tensor(
            sess, uids, event_name="FA", window=LICK_WIN, bin_size=BIN, outcome_filter={"FA"})
        lb = (bcF >= LICK_BASE[0]) & (bcF < LICK_BASE[1])
        fbm = np.array([tF[:, lb, j].ravel().mean() for j in range(nU)])
        fbs = np.array([max(tF[:, lb, j].ravel().std(), 1e-6) for j in range(nU)])
        fa_ev, n_fa = _evoked_by_state(tF, bcF, vtF, state_of, fbm, fbs, LICK_PERI, STATES)
        del tF; gc.collect()
    except Exception:
        pass

    # ---- Hit lick tensor ----
    hl_ev, n_hl = {s: np.full(nU, np.nan) for s in STATES}, {s: 0 for s in STATES}
    try:
        tH, bcH, vtH = build_population_tensor(
            sess, uids, event_name="Hit", window=LICK_WIN, bin_size=BIN, outcome_filter={"Hit"})
        lb = (bcH >= LICK_BASE[0]) & (bcH < LICK_BASE[1])
        hbm = np.array([tH[:, lb, j].ravel().mean() for j in range(nU)])
        hbs = np.array([max(tH[:, lb, j].ravel().std(), 1e-6) for j in range(nU)])
        hl_ev, n_hl = _evoked_by_state(tH, bcH, vtH, state_of, hbm, hbs, LICK_PERI, STATES)
        del tH; gc.collect()
    except Exception:
        pass

    del sess; gc.collect()
    return dict(
        sid8=sid8, unit_ids=np.array(uids),
        change_evoked_Impulsive=change_ev["Impulsive"], change_evoked_StimSens=change_ev["StimSens"],
        fa_evoked_Impulsive=fa_ev["Impulsive"], fa_evoked_StimSens=fa_ev["StimSens"],
        hl_evoked_Impulsive=hl_ev["Impulsive"], hl_evoked_StimSens=hl_ev["StimSens"],
        n_fa_Impulsive=n_fa["Impulsive"], n_fa_StimSens=n_fa["StimSens"],
        n_hl_Impulsive=n_hl["Impulsive"], n_hl_StimSens=n_hl["StimSens"],
        trial_z=trial_z.astype(np.float32),
        trial_state=stC.astype("U12"), trial_outcome=ocC.astype("U8"),
        trial_csize=sz, trial_is_go=go, trial_rt=trial_rt,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    resp_all = pd.read_csv(RESP_CACHE)
    man = load_staging_manifest(qc_only=False)
    sess_list = [str(s) for s in man.loc[man["stage"] == "Expert", "session_name"]]
    n_ok = 0
    for sname in sess_list:
        r = process(sname, resp_all)
        if r is None:
            continue
        np.savez(os.path.join(OUT_DIR, f"{r['sid8']}.npz"), **r)
        n_ok += 1
        print(f"  {r['sid8']}: {len(r['unit_ids'])} units, {r['trial_z'].shape[0]} trials, "
              f"FA(Imp/SS)={r['n_fa_Impulsive']}/{r['n_fa_StimSens']}, "
              f"Hit(Imp/SS)={r['n_hl_Impulsive']}/{r['n_hl_StimSens']}")
    print(f"[extract] cached {n_ok} sessions -> {OUT_DIR}")


if __name__ == "__main__":
    main()
