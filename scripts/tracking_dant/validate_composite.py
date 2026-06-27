#!/usr/bin/env python3
"""Validate the composite-verdict re-tiering on the INDEPENDENT held-out ISI axis.

composite_retier.py found 411 "hidden gems" (curation=review but composite=trusted).
Composite-trusted is a looser bar than the link-by-link sweep, so the question is whether
these extra tracks are REAL. This computes held-out ISI AUC for three groups:
  link_trusted  — curation tier == trusted (the 155 already reported)
  hidden_gem    — curation review BUT composite verdict trusted (the 411)
  other         — everything else
If hidden_gem AUC ~= link_trusted AUC, the under-reporting is real and validated.

Reuses cd.collect_holdout_isi + tc.held_out_isi_auc_by_tier. One session pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import curate_dant as cd            # noqa: E402
import inclusive_trusted as it      # noqa: E402

WT = cd.WORKTREE_ROOT
PRIMARY = cd.PRIMARY_DEFAULT
CUR = WT / "FIGURES" / "tracking_dant" / "BG_046" / "curation"
REG = WT / "data" / "cache" / "dant" / "BG_046" / "dant_registry_curation.csv"
PKL = PRIMARY / "data" / "pkls" / "BG_046"


def main() -> int:
    subj = "BG_046"
    sjp, tc, _ = cd._import_pipeline(subj)
    tracks = pd.read_csv(CUR / "curated_tracks.csv")
    comp = pd.read_csv(CUR / "composite_retier.csv")[["curated_uid", "composite_verdict"]]
    df = tracks.merge(comp, on="curated_uid", how="inner")

    def group(r):
        if r["confidence_tier"] == "trusted":
            return "link_trusted"
        if r["confidence_tier"] == "review" and r["composite_verdict"] == "trusted":
            return "hidden_gem"
        return "other"
    df["confidence_tier"] = df.apply(group, axis=1)   # rename so held_out_isi_auc_by_tier groups on it

    reg = pd.read_csv(REG, dtype={"session": str})
    kept_pairs = it.kept_pairs_from(df.rename(columns={}), reg, sjp.session_date_key)
    print(f"kept_pairs={len(kept_pairs)}", flush=True)
    holdout = cd.collect_holdout_isi(kept_pairs, subj, PKL)
    res = tc.held_out_isi_auc_by_tier(df, holdout)

    counts = df["confidence_tier"].value_counts().to_dict()
    print("\n=== held-out ISI AUC by group (independent axis) ===", flush=True)
    for grp in ["link_trusted", "hidden_gem", "other"]:
        r = res.get(grp, {})
        print(f"  {grp:13s} n={counts.get(grp,0):4d}  AUC={r.get('auc', float('nan')):.3f}  "
              f"(matched={r.get('n_matched',0)}, nonmatched={r.get('n_nonmatched',0)})", flush=True)
    with open(CUR / "composite_validation.json", "w") as f:
        json.dump({"counts": counts, "auc": res}, f, indent=2)
    print(f"wrote {CUR / 'composite_validation.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
