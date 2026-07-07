"""Held-out log-ISI validation for the UM x DANT consensus cohort.

The held-out ISI fingerprint is the INDEPENDENT axis that convinces a skeptic the
consensus tracks are the same neuron: neither tracker uses spike-timing statistics
to match, so cross-session agreement of the ISI shape is corroborating evidence.

We split each unit's spikes into even/odd partitions (``partitioned_isi_hists``)
and use the ODD (holdout) log-ISI histogram only -- statistically independent of
the curation ISI feature. Then:
  * matched     = cross-session pairs WITHIN a consensus track (same neuron)
  * non-matched = within-session pairs across DIFFERENT tracks (different neurons,
                  recorded simultaneously -> the correct null)
AUC(matched vs non-matched) and the per-track matched correlation quantify it.

Loads each of the ~41 cohort session pkls exactly once (all LOCAL; no X: compute).

Outputs (data/cache/tracking_consensus/BG_046/):
  isi_holdout.pkl        {(session_key, ks_unit_id): holdout_hist(50,)}
  nonmatched_corrs.npy   non-matched holdout-ISI corr distribution (for figure panel)
  isi_validation.json    cohort AUC, n_matched, n_nonmatched, null summary
  + augments consensus_cohort.csv with matched_isi_corr, n_isi_sessions, matched_pctile
"""
from __future__ import annotations

import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "pipelines" / "tracking"))
from visdetect.analysis.config import canonical_session_id  # noqa: E402
from visdetect.analysis.track_curation import partitioned_isi_hists  # noqa: E402
from visdetect.core.session import load_session  # noqa: E402
import _subject_paths as sjp  # noqa: E402

SUBJECT = "BG_046"
OUT_DIR = ROOT / "data/cache/tracking_consensus/BG_046"
MEMBERS = OUT_DIR / "consensus_members.csv"
COHORT = OUT_DIR / "consensus_cohort.csv"
PKL_DIR = sjp.pkl_dir(SUBJECT)


def _valid(h):
    h = np.asarray(h, dtype=float)
    return h if (h.size and np.all(np.isfinite(h))) else None


def _corr(a, b):
    a, b = _valid(a), _valid(b)
    if a is None or b is None or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def build_holdout_cache(members: pd.DataFrame) -> dict:
    """One pass over cohort session pkls -> {(session_key, ks_unit_id): holdout_hist}."""
    holdout = {}
    sessions = sorted(members["session_key"].unique())
    for i, sess in enumerate(sessions, 1):
        pkl = sjp.session_pkl(SUBJECT, sess, PKL_DIR)
        if pkl is None:
            print(f"  [{i}/{len(sessions)}] {sess}: NO pkl -> skip", flush=True)
            continue
        S = load_session(str(pkl))
        by_id = {int(c.cluster_id): c for c in S.clusters}
        need = members.loc[members["session_key"] == sess, "ks_unit_id"].astype(int).unique()
        got = 0
        for kid in need:
            c = by_id.get(int(kid))
            if c is None:
                continue
            _, hold = partitioned_isi_hists(np.asarray(c.spike_times, dtype=float))
            holdout[(sess, int(kid))] = np.asarray(hold, dtype=np.float32)
            got += 1
        print(f"  [{i}/{len(sessions)}] {sess}: {got}/{len(need)} units", flush=True)
        del S
    return holdout


def main():
    members = pd.read_csv(MEMBERS, dtype={"session_key": str})
    members["session_key"] = members["session_key"].map(canonical_session_id)
    members["ks_unit_id"] = members["ks_unit_id"].astype(int)
    members["um_uid"] = members["um_uid"].astype(int)

    cache_path = OUT_DIR / "isi_holdout.pkl"
    reuse = "--rebuild" not in sys.argv and cache_path.exists()
    if reuse:
        with open(cache_path, "rb") as f:
            holdout = pickle.load(f)
        # rebuild only if the cohort membership changed
        needed = set(zip(members["session_key"], members["ks_unit_id"]))
        if needed.issubset(set(holdout.keys())):
            print(f"reusing cached {len(holdout)} holdout hists (pass --rebuild to force)")
        else:
            print("cohort membership changed -> rebuilding holdout cache")
            reuse = False
    if not reuse:
        holdout = build_holdout_cache(members)
        with open(cache_path, "wb") as f:
            pickle.dump(holdout, f)
        print(f"cached {len(holdout)} holdout hists")

    # per-node holdout hist keyed by track
    node_hist = {}   # (um_uid, session_key) -> hist
    track_sessions = {}   # um_uid -> [(session, hist)]
    for _, r in members.iterrows():
        h = holdout.get((r["session_key"], r["ks_unit_id"]))
        if h is None:
            continue
        node_hist[(r["um_uid"], r["session_key"])] = h
        track_sessions.setdefault(r["um_uid"], []).append((r["session_key"], h))

    # matched: within-track cross-session pairs
    matched = []
    per_track = {}
    for uid, sh in track_sessions.items():
        cs = [_corr(a, b) for (_, a), (_, b) in combinations(sh, 2)]
        cs = [c for c in cs if np.isfinite(c)]
        matched += cs
        if cs:
            per_track[uid] = (float(np.mean(cs)), len(sh))

    # non-matched: within-session pairs across different tracks
    bysess = {}
    for (uid, sess), h in node_hist.items():
        bysess.setdefault(sess, []).append((uid, h))
    nonmatched = []
    for sess, items in bysess.items():
        for (u1, h1), (u2, h2) in combinations(items, 2):
            if u1 == u2:
                continue
            c = _corr(h1, h2)
            if np.isfinite(c):
                nonmatched.append(c)
    matched = np.array(matched, dtype=float)
    nonmatched = np.array(nonmatched, dtype=float)
    np.save(OUT_DIR / "nonmatched_corrs.npy", nonmatched)

    # AUC = P(matched corr > non-matched corr) via Mann-Whitney U / (n*m)
    def _auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        allv = np.concatenate([pos, neg])
        ranks = pd.Series(allv).rank().to_numpy()
        r_pos = ranks[:len(pos)].sum()
        u = r_pos - len(pos) * (len(pos) + 1) / 2
        return float(u / (len(pos) * len(neg)))

    auc = _auc(matched, nonmatched)

    # cross-check against library implementation
    lib_auc = None
    try:
        from visdetect.analysis.track_curation import held_out_isi_auc_by_tier
        cohort = pd.read_csv(COHORT, dtype={"agreed_sessions": str})
        tdf = pd.DataFrame({
            "curated_uid": cohort["um_uid"].astype(int),
            "kept_sessions": cohort["agreed_sessions"],
            "confidence_tier": "consensus",
        })
        hd = {(int(u), s): h for (u, s), h in node_hist.items()}
        res = held_out_isi_auc_by_tier(tdf, hd)
        lib_auc = res.get("consensus", {})
    except Exception as e:  # pragma: no cover
        lib_auc = f"lib check failed: {e}"

    summary = {
        "auc_matched_vs_nonmatched": round(auc, 4),
        "n_matched_pairs": int(len(matched)),
        "n_nonmatched_pairs": int(len(nonmatched)),
        "matched_corr_mean": round(float(np.mean(matched)), 4),
        "matched_corr_median": round(float(np.median(matched)), 4),
        "nonmatched_corr_mean": round(float(np.mean(nonmatched)), 4),
        "nonmatched_corr_p95": round(float(np.percentile(nonmatched, 95)), 4),
        "library_cross_check": lib_auc,
    }
    with open(OUT_DIR / "isi_validation.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nISI VALIDATION:", json.dumps(summary, indent=2))

    # augment cohort csv with per-track matched corr + population percentile
    cohort = pd.read_csv(COHORT)
    def _pctile(uid):
        v = per_track.get(int(uid))
        if v is None or len(nonmatched) == 0:
            return np.nan
        return round(float((nonmatched < v[0]).mean()), 4)
    cohort["matched_isi_corr"] = cohort["um_uid"].map(
        lambda u: round(per_track[int(u)][0], 4) if int(u) in per_track else np.nan)
    cohort["n_isi_sessions"] = cohort["um_uid"].map(
        lambda u: per_track[int(u)][1] if int(u) in per_track else 0)
    cohort["matched_isi_pctile"] = cohort["um_uid"].map(_pctile)
    cohort.to_csv(COHORT, index=False)
    print(f"\naugmented {COHORT} with matched_isi_corr / n_isi_sessions / matched_isi_pctile")
    print(cohort[["um_uid", "dant_uid", "n_agree", "jaccard", "matched_isi_corr",
                  "matched_isi_pctile", "dant_composite"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
