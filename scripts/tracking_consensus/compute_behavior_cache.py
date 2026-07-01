"""Behavioral-response cache for the high-confidence, learning-spanning consensus neurons.

For each selected consensus neuron (learning-spanning, DANT-composite-trusted,
ISI-validated) we compute, per agreed session, four families of behavioural signal
so the render step can show how each *identified same neuron* changes across learning:

  1. task-event PSTHs  : Baseline_ON, Change_ON (hit / miss), Hit-lick, FA-lick
  2. decision selectivity: Change_ON Hit-vs-Miss AUROC; big- vs small-change tuning
  3. behavioural-state   : Change_ON evoked response split by state
                           (Impulsive / StimSens / Disengaged) from the state labeller
  4. choice / RT         : per-trial Change_ON response -> AUROC(hit vs miss) and
                           Spearman(response, reaction time) on hit go-trials

One pass over the needed session pkls (all LOCAL). Reuses the per-trial Change_ON
response for BOTH choice/RT and the state split, so state modulation costs no extra
tensor builds.

Output: data/cache/tracking_consensus/BG_046/behavior_cache.pkl
        + behavior_cohort.csv (the selected neurons)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "pipelines" / "tracking"))
from visdetect.analysis.config import canonical_session_id, session_date_key  # noqa: E402
from visdetect.analysis.utils import build_population_tensor, compute_auroc  # noqa: E402
from visdetect.analysis.tracking_qc import extract_unit_psths  # noqa: E402
from visdetect.analysis.align import get_event_times_by_trial  # noqa: E402
from visdetect.analysis.constants import DEFAULT_BIN_SIZE  # noqa: E402
from visdetect.core.session import load_session  # noqa: E402
import _subject_paths as sjp  # noqa: E402

SUBJECT = "BG_046"
CACHE = ROOT / "data/cache/tracking_consensus/BG_046"
STATE_DIR = ROOT / "data/cache/state_tags/BG_046"
PKL_DIR = sjp.pkl_dir(SUBJECT)

STATES = ["Impulsive", "StimSens", "Disengaged"]
CHANGE_RESP_WIN = (0.0, 0.3)     # s after Change_ON
BIG_SIZES = {2.0, 4.0}
SMALL_SIZES = {1.25, 1.35, 1.5}


def select_behavior_uids(cohort: pd.DataFrame, n_top: int = 8):
    """High-confidence learning-spanning set + the strict Naive->Expert exemplar."""
    ls = cohort[(cohort["learning_to_expert"]) & (cohort["dant_composite"] == "trusted")
                & (cohort["n_agree"] >= 4) & (cohort["matched_isi_pctile"] >= 0.7)].copy()
    ls = ls.sort_values("matched_isi_pctile", ascending=False)
    uids = list(ls["um_uid"].head(n_top))
    # add the unique strict Naive->Expert track (marquee across-learning cell)
    for u in cohort[cohort["naive_to_expert"]]["um_uid"]:
        if u not in uids:
            uids.append(int(u))
    return uids


def _go_indices(session):
    out = []
    for i, t in enumerate(session.trials):
        cs = getattr(t, "change_size", None)
        if cs is not None and float(cs) > 1.0:
            out.append(i)
    return out


def _psth_dict(session, ks):
    d = extract_unit_psths(session, ks)
    out = {}
    for k, (psth, centers, n) in d.items():
        out[k] = {"psth": None if psth is None else np.asarray(psth, np.float32),
                  "centers": None if centers is None else np.asarray(centers, np.float32),
                  "n": int(n)}
    return out


def _fa_psth(session, ks):
    try:
        tens, centers, valid = build_population_tensor(
            session, [ks], event_name="FA", window=(-2.0, 1.0),
            bin_size=DEFAULT_BIN_SIZE, outcome_filter={"fa"})
        if tens.shape[0] == 0:
            return {"psth": None, "centers": None, "n": 0}
        from visdetect.analysis.utils import smooth_psth
        mean_rate = tens[:, :, 0].mean(axis=0)
        sm = smooth_psth(mean_rate, bin_size=DEFAULT_BIN_SIZE, sigma_ms=25.0)
        return {"psth": np.asarray(sm, np.float32), "centers": np.asarray(centers, np.float32),
                "n": int(len(valid))}
    except (ValueError, Exception):
        return {"psth": None, "centers": None, "n": 0}


def _change_trialwise(session, ks, go_idx, state_of_trial):
    """Per-trial Change_ON response -> choice AUROC, RT Spearman, state-split means."""
    res = {"choice_auroc": np.nan, "n_hit": 0, "n_miss": 0,
           "rt_spearman": np.nan, "rt_p": np.nan, "n_rt": 0,
           "state_resp": {s: {"mean": np.nan, "n": 0} for s in STATES},
           "big_resp": np.nan, "small_resp": np.nan}
    try:
        tens, centers, valid = build_population_tensor(
            session, [ks], event_name="Change_ON", window=(-0.5, 0.5),
            bin_size=DEFAULT_BIN_SIZE, outcome_filter={"hit", "miss"},
            trial_indices=go_idx)
    except ValueError:
        return res
    if tens.shape[0] == 0:
        return res
    centers = np.asarray(centers)
    mask = (centers >= CHANGE_RESP_WIN[0]) & (centers < CHANGE_RESP_WIN[1])
    resp = tens[:, mask, 0].mean(axis=1)                     # per-trial scalar (Hz)
    oc = np.array([str(getattr(session.trials[i], "trialoutcome", "")).lower() for i in valid])
    cs = np.array([float(getattr(session.trials[i], "change_size", np.nan)) for i in valid])
    hit_m, miss_m = oc == "hit", oc == "miss"
    res["n_hit"], res["n_miss"] = int(hit_m.sum()), int(miss_m.sum())
    if hit_m.sum() >= 5 and miss_m.sum() >= 5:
        res["choice_auroc"] = float(compute_auroc(resp[hit_m], resp[miss_m]))
    # change-size tuning (hit trials): big vs small evoked response
    big_m = hit_m & np.isin(np.round(cs, 2), list(BIG_SIZES))
    small_m = hit_m & np.isin(np.round(cs, 2), list(SMALL_SIZES))
    if big_m.sum() >= 3:
        res["big_resp"] = float(resp[big_m].mean())
    if small_m.sum() >= 3:
        res["small_resp"] = float(resp[small_m].mean())
    # RT coding on hits
    rts = np.asarray(get_event_times_by_trial(session, "Hit"), dtype=float)  # per-trial RT (s), NaN else
    rt_valid = np.array([rts[i] if i < len(rts) else np.nan for i in valid])
    hm = hit_m & np.isfinite(rt_valid)
    if hm.sum() >= 6 and np.ptp(rt_valid[hm]) > 0:
        rho, p = spearmanr(resp[hm], rt_valid[hm])
        res["rt_spearman"], res["rt_p"], res["n_rt"] = float(rho), float(p), int(hm.sum())
    # state-split of the evoked response (reuse resp)
    if state_of_trial:
        st_valid = np.array([state_of_trial.get(i, None) for i in valid], dtype=object)
        for s in STATES:
            m = st_valid == s
            if m.sum() >= 3:
                res["state_resp"][s] = {"mean": float(resp[m].mean()), "n": int(m.sum())}
    return res


def _baseline_by_state(session, ks, state_of_trial):
    """Per-trial baseline-period firing (ALL trials, so Impulsive/FA trials count),
    split by behavioural state. Complements the change-response split which excludes
    FA-heavy Impulsive trials."""
    res = {s: {"mean": np.nan, "n": 0} for s in STATES}
    if not state_of_trial:
        return res
    try:
        tens, centers, valid = build_population_tensor(
            session, [ks], event_name="Baseline_ON", window=(0.0, 0.8),
            bin_size=DEFAULT_BIN_SIZE)
    except ValueError:
        return res
    if tens.shape[0] == 0:
        return res
    rate = tens[:, :, 0].mean(axis=1)
    st = np.array([state_of_trial.get(i, None) for i in valid], dtype=object)
    for s in STATES:
        m = st == s
        if m.sum() >= 3:
            res[s] = {"mean": float(rate[m].mean()), "n": int(m.sum())}
    return res


def _state_map(session_key):
    f = STATE_DIR / f"{session_key}.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    df = df[df["state_gated"] != -1]   # gated trials (state_gated encodes the state id, -1 = ungated)
    return {int(t): str(lbl) for t, lbl in zip(df["trial_idx"], df["state_label"])}


def main():
    cohort = pd.read_csv(CACHE / "consensus_cohort.csv")
    members = pd.read_csv(CACHE / "consensus_members.csv", dtype={"session_key": str})
    members["session_key"] = members["session_key"].map(canonical_session_id)
    members["um_uid"] = members["um_uid"].astype(int)
    members["ks_unit_id"] = members["ks_unit_id"].astype(int)

    stage_of = {canonical_session_id(s): st for s, st in
                zip(pd.read_csv(ROOT / f"data/{SUBJECT}_staging_manifest.csv",
                                dtype={"session_name": str})["session_name"],
                    pd.read_csv(ROOT / f"data/{SUBJECT}_staging_manifest.csv")["stage"])}

    uids = select_behavior_uids(cohort)
    print(f"selected {len(uids)} behavioural-cohort neurons: {uids}")
    sel = cohort[cohort["um_uid"].isin(uids)].copy()
    sel.to_csv(CACHE / "behavior_cohort.csv", index=False)

    mem = members[members["um_uid"].isin(uids)]
    by_sess = {}
    for _, r in mem.iterrows():
        by_sess.setdefault(r["session_key"], []).append((int(r["um_uid"]), int(r["ks_unit_id"])))

    cache = {}
    sessions = sorted(by_sess, key=session_date_key)
    for i, sk in enumerate(sessions, 1):
        pkl = sjp.session_pkl(SUBJECT, sk, PKL_DIR)
        if pkl is None:
            print(f"  [{i}/{len(sessions)}] {sk}: no pkl -> skip"); continue
        S = load_session(str(pkl))
        stage = stage_of.get(sk, "Unknown")
        state_of_trial = _state_map(sk)
        go_idx = _go_indices(S)
        for um_uid, ks in by_sess[sk]:
            feats = {"stage": stage, "ks_unit_id": ks,
                     "psths": _psth_dict(S, ks), "fa": _fa_psth(S, ks)}
            feats.update(_change_trialwise(S, ks, go_idx, state_of_trial))
            feats["baseline_state"] = _baseline_by_state(S, ks, state_of_trial)
            cache[(um_uid, sk)] = feats
        print(f"  [{i}/{len(sessions)}] {sk} ({stage}): {len(by_sess[sk])} units, "
              f"states={'yes' if state_of_trial else 'no'}", flush=True)
        del S

    with open(CACHE / "behavior_cache.pkl", "wb") as f:
        pickle.dump(cache, f)
    print(f"\nwrote behavior_cache.pkl ({len(cache)} (uid,session) entries) + behavior_cohort.csv")


if __name__ == "__main__":
    main()
