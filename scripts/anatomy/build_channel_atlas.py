# scripts/anatomy/build_channel_atlas.py
"""Build a subject's per-channel CCF/region atlas from its track artifact.

Usage:
    py scripts/anatomy/build_channel_atlas.py --subject BG_046
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from visdetect.anatomy.atlas import AllenAtlas
from visdetect.anatomy.channel_geometry import chanmap_signature
from visdetect.anatomy.localize import build_channel_atlas
from visdetect.anatomy.orientation import validate_shank_order
from visdetect.anatomy.tracks import load_track_artifact
from visdetect.analysis.tracking_qc import load_channel_positions


def session_token(name) -> str:
    """Numeric date token (DDMMYYYY) from a session dir / pkl name.

    'BG_031_01042025' -> '01042025'; '01072025' -> '01072025'.
    """
    m = re.search(r"(\d{6,8})$", str(name))
    return m.group(1) if m else str(name)


def resolve_session_dir(raw_wf_root, token) -> Optional[str]:
    """Actual session subdir under raw_wf_root matching a date token, handling
    bare ('01072025') and subject-prefixed ('BG_031_01042025') dir names."""
    root = str(raw_wf_root)
    cands = {str(token), str(token).zfill(8)}
    if not os.path.isdir(root):
        return None
    for d in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, d)):
            continue
        if d in cands or session_token(d) in cands:
            return d
    return None


def build_subject_atlas(subject, artifact_path, raw_wf_root, session_names: List[str],
                        atlas: AllenAtlas, out_dir) -> pd.DataFrame:
    art = load_track_artifact(artifact_path)
    validate_shank_order(art)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sig_rows, atlas_by_sig = [], {}
    for name in session_names:
        token = session_token(name)
        sess_dir = resolve_session_dir(raw_wf_root, token)
        if sess_dir is None:
            print(f"  {name}: no session directory found, skipping")
            continue
        pos = load_channel_positions(raw_wf_root, sess_dir)
        if pos is None:
            print(f"  {name}: no channel_positions in {sess_dir}, skipping")
            continue
        sig = chanmap_signature(pos)
        sig_rows.append({"session_name": token, "chanmap_signature": sig})
        if sig not in atlas_by_sig:
            atlas_by_sig[sig] = build_channel_atlas(subject, art, pos, sig, atlas)

    atlas_df = (pd.concat(atlas_by_sig.values(), ignore_index=True)
                if atlas_by_sig else pd.DataFrame())
    atlas_df.to_csv(out_dir / f"{subject}_channel_atlas.csv", index=False)
    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty:
        sig_df["session_name"] = sig_df["session_name"].astype(str)
    sig_df.to_csv(out_dir / f"{subject}_session_signatures.csv", index=False)
    print(f"{subject}: {len(atlas_by_sig)} unique chanmap(s), "
          f"{len(sig_rows)} sessions -> {out_dir}")
    return atlas_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--raw-wf-root", default=None,
                    help="defaults to data/unit_match/input/<subject>")
    ap.add_argument("--artifact", default=None,
                    help="defaults to data/anatomy/<subject>_shank_tracks.json")
    ap.add_argument("--out-dir", default="data/anatomy")
    args = ap.parse_args()

    raw_root = args.raw_wf_root or os.path.join("data", "unit_match", "input", args.subject)
    artifact = args.artifact or os.path.join("data", "anatomy", f"{args.subject}_shank_tracks.json")
    if not os.path.isdir(raw_root):
        print(f"{args.subject}: raw-wf root not found: {raw_root}")
        raise SystemExit(1)
    sessions = sorted(d for d in os.listdir(raw_root)
                      if os.path.isdir(os.path.join(raw_root, d)))
    atlas = AllenAtlas()  # real Allen atlas (downloads/caches on first use)
    build_subject_atlas(args.subject, artifact, raw_root, sessions, atlas, args.out_dir)


if __name__ == "__main__":
    main()
