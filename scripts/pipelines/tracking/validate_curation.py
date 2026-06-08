#!/usr/bin/env python3
"""Held-out-ISI AUC by confidence tier for a curated-track table (spec sec 8.2).

Usage:
    py scripts/pipelines/tracking/validate_curation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis import track_curation as tc                 # noqa: E402
from visdetect.core.session import load_session                     # noqa: E402

UM_ROOT = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
               "BG_046/unit_match/output/all42")
DEFAULT_REGISTRY = UM_ROOT / "batch0" / "unit_index.csv"
DEFAULT_TRACKS = REPO_ROOT / "FIGURES" / "tracking_qc" / "curation" / "curated_tracks.csv"
DEFAULT_PKL_DIR = REPO_ROOT / "data" / "pkls" / "BG_046"
OUT_DIR = REPO_ROOT / "FIGURES" / "tracking_qc" / "curation"


def _session_pkl(pkl_dir: Path, sess: str):
    for s in (sess, str(sess).zfill(8)):
        p = pkl_dir / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--liberal-col", default="batch_uid_liberal")
    ap.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR)
    args = ap.parse_args()

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
        pkl = _session_pkl(args.pkl_dir, sess)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "curation_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUT_DIR / 'curation_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
