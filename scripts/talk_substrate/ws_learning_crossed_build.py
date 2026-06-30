"""Builder for the across-learning SPLITS: per unit-session change-size scaling (big-small)
computed WITHIN each outcome (hit / miss) and WITHIN each behavioural state (BG_046 only),
tagged by learning stage. Addresses (a) the hit+miss pooling confound and (b) state x stage.

Why a dedicated pass: the event cache crosses size and outcome/state only marginally, not
jointly. Here we re-load each session once and compute per-unit mean z (0-1 s post-change,
shared -0.4..-0.05 s baseline) per trial, then average within (outcome x size) and
(state x size). Output: one tidy per-unit-session CSV the figures read.

Scaling within a group = mean_z(group & big) - mean_z(group & small). Cell type = COMMON
cutoff (FIX A). State (BG_046) joined by trial_idx (verified). NOT N1.

Usage: py scripts/talk_substrate/ws_learning_crossed_build.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gc
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.loader import load_session                     # noqa: E402
from visdetect.analysis.config import canonical_session_id as canon  # noqa: E402
from visdetect.analysis.behavior import get_trial_dataframe         # noqa: E402
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS  # noqa: E402
from visdetect.analysis.utils import (                              # noqa: E402
    get_good_cluster_ids, build_population_tensor, compute_zscore_normalized,
)

WINDOW = (-0.5, 1.5)
BASELINE = EVENT_RESPONSIVENESS_WINDOWS["Change_ON"][0]   # (-0.4, -0.05) shared baseline
SCWIN = (0.0, 1.0)                                        # scaling measurement window
STATES = ["Impulsive", "StimSens", "Disengaged"]
OUT = C.CACHE_DIR / f"ws_learning_crossed_{C.SUBJECT}.csv"   # per-subject (run via VISDETECT_SUBJECT)
LATENTS = C.cfg.ROOT + "/data/cache/decision_latents/decision_latents_by_state.csv"


def stage_map(subj):
    m = pd.read_csv(f"data/{subj}_staging_manifest.csv", dtype={"session_name": str})
    return dict(zip(m["session_name"].map(canon), m["stage"].astype(str)))


def state_by_trial():
    lat = pd.read_csv(LATENTS)
    lat["s8"] = lat["session_name"].map(canon)
    return {s8: dict(zip(g["trial_idx"].astype(int), g["state_label"].astype(str)))
            for s8, g in lat.groupby("s8")}


def gmean(arr, mask):
    """Mean over selected trials, NaN if <3 trials."""
    if mask.sum() < 3:
        return np.nan
    return float(np.nanmean(arr[mask]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    # SINGLE subject = the active env subject. cfg.PKL_DIR / RAW_WF_DIR are import-time fixed to
    # VISDETECT_SUBJECT, so load_session() only resolves THIS subject's pkls — run once per
    # subject via the env var (do NOT loop subjects in one process; that loaded BG_046's pkls).
    subj = C.SUBJECT
    thr, _ = C.common_t2p_cutoff()
    sbt = state_by_trial() if subj == "BG_046" else {}
    sessions = sorted(pd.unique(E.load_event_cache(subj)["unit_meta_session"].astype(str)))
    if args.limit:
        sessions = sessions[: args.limit]
    smap = stage_map(subj)
    t2p = {(str(r.session_8), int(r.cluster_id)): float(r.t2p_ms)
           for r in C.load_t2p(subj).itertuples()}
    is046 = subj == "BG_046"
    rows = []
    print(f"[build] {subj}: {len(sessions)} sessions")
    for si, s8 in enumerate(sessions, 1):
        stage = smap.get(canon(s8), "NA")
        if stage not in ("Naive", "Learning", "Expert"):
            continue
        try:
            sess = load_session(s8)
        except Exception as e:  # noqa: BLE001
            print(f"  {subj} {s8}: load failed ({e})"); continue
        cids = get_good_cluster_ids(sess)
        tdf = get_trial_dataframe(sess)
        if len(cids) == 0 or tdf.empty or "trial_idx" not in tdf.columns:
            del sess; gc.collect(); continue
        tdf = tdf.set_index("trial_idx")
        try:
            tensor, bc, valid = build_population_tensor(
                sess, list(cids), event_name="Change_ON", window=WINDOW, bin_size=0.025)
        except ValueError:
            del sess; gc.collect(); continue
        z = compute_zscore_normalized(tensor, bc, BASELINE)         # (T, bins, U)
        z[:, :, np.nanmax(np.abs(z), axis=(0, 1)) > 50.0] = np.nan  # degenerate-baseline cap
        win = (bc >= SCWIN[0]) & (bc <= SCWIN[1])
        pertrial = np.nanmean(z[:, win, :], axis=1)                 # (T, U) per-trial mean z
        sub = tdf.loc[valid]
        o = sub["outcome"].values
        cs = sub["change_size"].values.astype(float)
        isgo = sub["is_go"].values
        big = isgo & (cs >= 2.0)
        small = isgo & (cs < 2.0)
        states = (np.array([sbt.get(s8, {}).get(int(p), "") for p in valid])
                  if is046 else np.array([""] * len(valid)))
        for u, cid in enumerate(cids):
            a = pertrial[:, u]
            ct = C.normalize_celltype("FSI" if t2p.get((str(s8), int(cid)), np.nan) < thr
                                      else "SPN") if (str(s8), int(cid)) in t2p else C.UNKNOWN
            row = dict(subject=subj, session=s8, stage=stage, cluster_id=int(cid), celltype=ct,
                       hit_big=gmean(a, (o == "hit") & big), hit_small=gmean(a, (o == "hit") & small),
                       miss_big=gmean(a, (o == "miss") & big), miss_small=gmean(a, (o == "miss") & small),
                       n_hit_big=int(((o == "hit") & big).sum()), n_hit_small=int(((o == "hit") & small).sum()),
                       n_miss_big=int(((o == "miss") & big).sum()), n_miss_small=int(((o == "miss") & small).sum()))
            if is046:
                for st in STATES:
                    row[f"{st}_big"] = gmean(a, (states == st) & big)
                    row[f"{st}_small"] = gmean(a, (states == st) & small)
            rows.append(row)
        print(f"  {subj} [{si}/{len(sessions)}] {s8} ({stage}): {len(cids)}u")
        del sess; gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"[build] wrote {OUT} ({len(df)} unit-sessions)")


if __name__ == "__main__":
    main()
