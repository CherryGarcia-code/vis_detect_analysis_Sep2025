"""Build the UM x DANT *consensus cohort* for BG_046.

A **consensus track** = a neuron that BOTH trackers -- UnitMatch (UM) and DANT --
*independently* place together across >= 2 sessions. Concretely: on the set of
(session, ks_unit_id) nodes both trackers observed, we take the mutual-best
cluster-id correspondence between UM's ``global_uid`` and DANT's ``dant_uid``.
The agreed sessions are those where UM says "global_uid G" AND DANT says
"dant_uid D" for the *same* physical unit. Two independent algorithms agreeing
is the strongest "same neuron across learning" evidence for a skeptic.

All inputs are LOCAL (no X: / Samba compute):
  UM     : data/cache/um_ref/unit_index.csv          (session, ks_unit_id, global_uid)
  DANT   : data/cache/dant/BG_046/dant_registry.csv  (session, ks_unit_id, dant_uid; -1 = untracked)
  stages : data/BG_046_staging_manifest.csv
  UM tier   : FIGURES/tracking_qc/curation/curated_tracks.csv
  DANT tier : FIGURES/tracking_dant/BG_046/curation/curated_tracks.csv
  DANT comp : FIGURES/tracking_dant/BG_046/curation/composite_retier.csv

Output: data/cache/tracking_consensus/BG_046/consensus_cohort.csv  (one row per consensus track)

CRITICAL: sessions are joined via ``canonical_session_id`` (8-digit DDMMYYYY string).
UM stores 7-or-8 digit tokens, DANT stores zfill-8. A raw string == join silently
drops the 14 single-digit-day sessions (leading-zero footgun). Never join on raw tokens.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# repo root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from visdetect.analysis.config import canonical_session_id, session_date_key  # noqa: E402

SUBJECT = "BG_046"
UM_REG = ROOT / "data/cache/um_ref/unit_index.csv"
DANT_REG = ROOT / "data/cache/dant/BG_046/dant_registry.csv"
STAGING = ROOT / f"data/{SUBJECT}_staging_manifest.csv"
UM_TIER = ROOT / "FIGURES/tracking_qc/curation/curated_tracks.csv"
DANT_TIER = ROOT / "FIGURES/tracking_dant/BG_046/curation/curated_tracks.csv"
DANT_COMP = ROOT / "FIGURES/tracking_dant/BG_046/curation/composite_retier.csv"
OUT_DIR = ROOT / "data/cache/tracking_consensus/BG_046"
OUT_CSV = OUT_DIR / "consensus_cohort.csv"
OUT_MEMBERS = OUT_DIR / "consensus_members.csv"

STAGE_RANK = {"Naive": 0, "Learning": 1, "Expert": 2}


# ---------------------------------------------------------------- loaders
def _load_registry(path: Path, uid_col: str, drop_untracked: bool) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df[["session", "ks_unit_id", uid_col]].copy()
    df["session_key"] = df["session"].map(canonical_session_id)
    df["ks_unit_id"] = df["ks_unit_id"].astype(int)
    df[uid_col] = df[uid_col].astype(int)
    if drop_untracked:
        df = df[df[uid_col] >= 0]
    return df.reset_index(drop=True)


def _stage_map() -> dict:
    st = pd.read_csv(STAGING, dtype={"session_name": str})
    return {canonical_session_id(s): stg for s, stg in zip(st["session_name"], st["stage"])}


def _tier_map(path: Path, uid_col: str = "curated_uid", tier_col: str = "confidence_tier") -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    return {int(u): t for u, t in zip(df[uid_col], df[tier_col])}


def _dant_composite_map() -> dict:
    if not DANT_COMP.exists():
        return {}
    df = pd.read_csv(DANT_COMP, dtype=str)
    return {int(u): v for u, v in zip(df["curated_uid"], df["composite_verdict"])}


# ---------------------------------------------------------------- core join
def build_consensus(um: pd.DataFrame, dant: pd.DataFrame, min_agree: int = 2) -> pd.DataFrame:
    """Mutual-best UM<->DANT correspondence on shared (session, ks_unit_id) nodes."""
    shared = um.merge(dant, on=["session_key", "ks_unit_id"], how="inner",
                      suffixes=("_um", "_dant"))
    if shared.empty:
        raise RuntimeError("no shared (session, ks_unit_id) nodes between UM and DANT registries")

    # count agreed sessions per (global_uid, dant_uid) pair
    gd = (shared.groupby(["global_uid", "dant_uid"])["session_key"]
          .agg(n_agree="nunique", sessions=lambda s: sorted(set(s), key=session_date_key))
          .reset_index())

    # co-observed set sizes (denominators for purity / jaccard), on shared nodes only
    g_size = shared.groupby("global_uid")["session_key"].nunique()
    d_size = shared.groupby("dant_uid")["session_key"].nunique()

    # mutual-best partner (idxmax picks first on ties -- rare, noted)
    best_d = gd.loc[gd.groupby("global_uid")["n_agree"].idxmax(), ["global_uid", "dant_uid"]]
    best_d = dict(zip(best_d["global_uid"], best_d["dant_uid"]))
    best_g = gd.loc[gd.groupby("dant_uid")["n_agree"].idxmax(), ["dant_uid", "global_uid"]]
    best_g = dict(zip(best_g["dant_uid"], best_g["global_uid"]))

    rows = []
    mutual = set()
    for _, r in gd.iterrows():
        G, D, n = int(r["global_uid"]), int(r["dant_uid"]), int(r["n_agree"])
        if n < min_agree:
            continue
        if best_d.get(G) != D or best_g.get(D) != G:
            continue  # not mutual-best
        gs, ds = int(g_size[G]), int(d_size[D])
        union = gs + ds - n
        mutual.add((G, D))
        rows.append({
            "um_uid": G, "dant_uid": D, "n_agree": n,
            "agreed_sessions": ";".join(r["sessions"]),
            "purity_um": round(n / gs, 4) if gs else np.nan,
            "purity_dant": round(n / ds, 4) if ds else np.nan,
            "jaccard": round(n / union, 4) if union else np.nan,
            "um_co_obs_n": gs, "dant_co_obs_n": ds,
        })
    out = pd.DataFrame(rows)
    # per-node membership of the agreed (mutual-best) tracks -- one row per (track, session)
    key = list(zip(shared["global_uid"], shared["dant_uid"]))
    members = shared[[k in mutual for k in key]][
        ["global_uid", "dant_uid", "session_key", "ks_unit_id"]].copy()
    members = members.rename(columns={"global_uid": "um_uid"}).sort_values(
        ["um_uid", "session_key"]).reset_index(drop=True)
    if out.empty:
        return out, members
    out = out.sort_values(["n_agree", "jaccard"], ascending=[False, False]).reset_index(drop=True)
    return out, members


def annotate(cohort: pd.DataFrame) -> pd.DataFrame:
    stage_of = _stage_map()
    um_tier = _tier_map(UM_TIER)
    dant_tier = _tier_map(DANT_TIER)
    dant_comp = _dant_composite_map()

    def _stages(sess_str):
        return [stage_of.get(s, "Unknown") for s in sess_str.split(";")]

    recs = []
    for _, r in cohort.iterrows():
        stages = _stages(r["agreed_sessions"])
        sset = set(stages)
        has_naive = "Naive" in sset
        has_learn = "Learning" in sset
        has_exp = "Expert" in sset
        recs.append({
            **r.to_dict(),
            "stages": ";".join(stages),
            "n_stages": len({s for s in stages if s in STAGE_RANK}),
            "has_naive": has_naive, "has_learning": has_learn, "has_expert": has_exp,
            "naive_to_expert": bool(has_naive and has_exp),
            "learning_to_expert": bool((has_naive or has_learn) and has_exp),
            "um_tier": um_tier.get(int(r["um_uid"]), "n/a"),
            "dant_tier": dant_tier.get(int(r["dant_uid"]), "n/a"),
            "dant_composite": dant_comp.get(int(r["dant_uid"]), "n/a"),
        })
    return pd.DataFrame(recs)


def main():
    um = _load_registry(UM_REG, "global_uid", drop_untracked=False)
    dant = _load_registry(DANT_REG, "dant_uid", drop_untracked=True)
    print(f"UM rows {len(um)} ({um['global_uid'].nunique()} global_uids); "
          f"DANT tracked rows {len(dant)} ({dant['dant_uid'].nunique()} dant_uids)")

    n_shared_nodes = pd.merge(um[["session_key", "ks_unit_id"]],
                              dant[["session_key", "ks_unit_id"]],
                              on=["session_key", "ks_unit_id"]).shape[0]
    print(f"shared (session,ks_unit) nodes: {n_shared_nodes}")

    cohort, members = build_consensus(um, dant, min_agree=2)
    cohort = annotate(cohort)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUT_CSV, index=False)
    members.to_csv(OUT_MEMBERS, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(cohort)} consensus tracks, span>=2)")
    print(f"wrote {OUT_MEMBERS}  ({len(members)} member nodes)")
    print(f"  span>=3: {(cohort['n_agree']>=3).sum()};  span>=5: {(cohort['n_agree']>=5).sum()}")
    print(f"  naive_to_expert: {cohort['naive_to_expert'].sum()};  "
          f"learning_to_expert: {cohort['learning_to_expert'].sum()}")
    print("\nTop 12 by agreed span:")
    cols = ["um_uid", "dant_uid", "n_agree", "jaccard", "purity_um", "purity_dant",
            "n_stages", "naive_to_expert", "um_tier", "dant_tier", "dant_composite"]
    print(cohort[cols].head(12).to_string(index=False))

    # ---- validation vs the known worked example (DANT 631 == UM 942) ----
    row = cohort[(cohort["um_uid"] == 942) & (cohort["dant_uid"] == 631)]
    print("\n[validate] UM 942 <-> DANT 631 (expected ~13 agreed sessions):")
    if row.empty:
        print("  !! NOT FOUND as a mutual-best consensus pair -- investigate")
    else:
        print(row[["um_uid", "dant_uid", "n_agree", "agreed_sessions"]].to_string(index=False))
    return cohort


if __name__ == "__main__":
    main()
