#!/usr/bin/env python3
"""Held-out-ISI AUC by confidence tier for a curated-track table (spec sec 8.2).

Usage:
    py scripts/pipelines/tracking/validate_curation.py --subject BG_049
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # for _subject_paths


def _early_subject(default: str = "BG_046") -> str:
    for i, a in enumerate(sys.argv):
        if a == "--subject" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--subject="):
            return a.split("=", 1)[1]
    return default


os.environ["VISDETECT_SUBJECT"] = _early_subject()

import _subject_paths as sjp                                    # noqa: E402
from visdetect.analysis import track_curation as tc             # noqa: E402
from visdetect.core.session import load_session                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--tracks", type=Path, default=None)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--liberal-col", default="global_uid")
    ap.add_argument("--pkl-dir", type=Path, default=None)
    args = ap.parse_args()
    subj = args.subject
    out_dir = sjp.curation_out_dir(subj)
    if args.tracks is None: args.tracks = out_dir / "curated_tracks.csv"
    if args.registry is None: args.registry = sjp.um_registry(subj)
    if args.pkl_dir is None: args.pkl_dir = sjp.pkl_dir(subj)

    tracks = pd.read_csv(args.tracks)
    reg = pd.read_csv(args.registry)
    reg["session"] = reg["session"].astype(str)
    reg["uid"] = reg[args.liberal_col].astype(int)
    # (uid, session) -> ks_unit_id, restricted to kept sessions of each curated_uid
    kept_pairs: Dict[Tuple[int, str], int] = {}
    for _, row in tracks.iterrows():
        uid = int(row["curated_uid"])
        for s in [s for s in str(row["kept_sessions"]).split(";") if s]:
            m = reg[(reg["uid"] == uid) & (reg["session"] == s)]
            if len(m):
                kept_pairs[(uid, s)] = int(m.iloc[0]["ks_unit_id"])

    # Build held-out ISI hist per (uid, session) — load each session once.
    holdout: Dict[Tuple[int, str], np.ndarray] = {}
    for sess in sorted({s for (_, s) in kept_pairs}):
        pkl = sjp.session_pkl(subj, sess, args.pkl_dir)
        if pkl is None:
            continue
        S = load_session(str(pkl))
        cmap = {c.cluster_id: c for c in S.clusters}
        for (uid, s), kid in kept_pairs.items():
            if s != sess or kid not in cmap:
                continue
            _, hold = tc.partitioned_isi_hists(np.asarray(cmap[kid].spike_times))
            holdout[(uid, s)] = hold
        del S

    result = tc.held_out_isi_auc_by_tier(tracks, holdout)
    print(json.dumps(result, indent=2))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "curation_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_dir / 'curation_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
