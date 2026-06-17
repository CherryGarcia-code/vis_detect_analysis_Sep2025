# scripts/anatomy/localize_units.py
"""Localize units: peak channel -> channel atlas row -> per-unit CCF/region.

Usage:
    py scripts/anatomy/localize_units.py --subject BG_046
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from visdetect.anatomy.peak_channel import unit_peak_channel

UNIT_COLS = ["session_name", "cluster_id", "peak_channel", "shank", "depth_um",
             "ccf_ap", "ccf_ml", "ccf_dv", "region_acronym", "region_name",
             "region_coarse", "region_confidence", "loc_method"]


def localize_subject_units(subject, atlas_csv, sig_csv, raw_wf_root,
                           units_by_session: Dict[str, List[int]]) -> pd.DataFrame:
    from build_channel_atlas import resolve_session_dir, session_token
    atlas = pd.read_csv(atlas_csv)
    # dtype=str + zfill(8): session tokens like "01072025" otherwise read back as
    # int 1072025 (leading zero dropped) and the join silently misses.
    sig_df = pd.read_csv(sig_csv, dtype={"session_name": str})
    sig = {str(k).zfill(8): v
           for k, v in zip(sig_df["session_name"], sig_df["chanmap_signature"])}
    rows = []
    for sess, cluster_ids in units_by_session.items():
        token = session_token(sess).zfill(8)
        signature = sig.get(token)
        if signature is None:
            continue
        sess_dir = resolve_session_dir(raw_wf_root, token) or str(sess)
        chans = atlas[atlas["chanmap_signature"] == signature].set_index("channel")
        for cid in cluster_ids:
            pc = unit_peak_channel(raw_wf_root, sess_dir, cid)
            if pc is None or pc not in chans.index:
                continue
            a = chans.loc[pc]
            rows.append({
                "session_name": int(token), "cluster_id": int(cid), "peak_channel": int(pc),
                "shank": int(a["shank"]), "depth_um": float(a["y_um"]),
                "ccf_ap": float(a["ccf_ap"]), "ccf_ml": float(a["ccf_ml"]),
                "ccf_dv": float(a["ccf_dv"]),
                "region_acronym": a["region_acronym"], "region_name": a["region_name"],
                "region_coarse": a["region_coarse"],
                "region_confidence": float(a["region_confidence"]),
                "loc_method": a["loc_method"],
            })
    return pd.DataFrame(rows, columns=UNIT_COLS)


def append_unit_anatomy(df: pd.DataFrame, out_csv) -> None:
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        combined = pd.concat([prev, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["session_name", "cluster_id"], keep="last")
    else:
        combined = df
    combined.to_csv(out_csv, index=False)


def _units_by_session_for_subject(subject) -> Dict[str, List[int]]:
    """Per-session good_and_stable cluster ids from the subject's PKLs, keyed by
    numeric date token. Loads pkls by explicit path (avoids the SUBJECT-env-scoped
    suite.loader.load_session, which resolves only the active subject)."""
    import glob
    from visdetect.core.session import load_session   # path-based loader
    from visdetect.suite.config import ROOT
    from build_channel_atlas import session_token
    out: Dict[str, List[int]] = {}
    pkl_dir = os.path.join(ROOT, "data", "pkls", subject)
    for path in sorted(glob.glob(os.path.join(pkl_dir, f"{subject}_*.pkl"))):
        token = session_token(os.path.basename(path)[:-4])
        if not token.isdigit():
            continue   # skip variants/backups (e.g. *_preconsolidate, *_b)
        sess = load_session(path)
        ids = sess.good_and_stable_ids or [c.cluster_id for c in sess.clusters]
        out[token] = [int(i) for i in ids]
        del sess
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--raw-wf-root", default=None)
    ap.add_argument("--anatomy-dir", default="data/anatomy")
    args = ap.parse_args()
    raw_root = args.raw_wf_root or os.path.join("data", "unit_match", "input", args.subject)
    atlas_csv = os.path.join(args.anatomy_dir, f"{args.subject}_channel_atlas.csv")
    sig_csv = os.path.join(args.anatomy_dir, f"{args.subject}_session_signatures.csv")
    units = _units_by_session_for_subject(args.subject)
    df = localize_subject_units(args.subject, atlas_csv, sig_csv, raw_root, units)
    append_unit_anatomy(df, os.path.join(args.anatomy_dir, "unit_anatomy.csv"))
    print(f"{args.subject}: localized {len(df)} units -> {args.anatomy_dir}/unit_anatomy.csv")


if __name__ == "__main__":
    main()
