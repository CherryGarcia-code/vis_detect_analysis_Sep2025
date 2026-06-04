"""Regenerate the GLT from a canonical long registry, then rebuild+validate the unit table.

Registry-agnostic: --registry-long defaults to UM 3.2.9 all42/unit_index.csv;
swap to a DeepUM-derived long registry later via the same flag.

Usage:
    py scripts/pipelines/tracking/regen_glt_from_um.py --workers 6
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from visdetect.analysis.tracking_registry import (  # noqa: E402
    load_canonical_long, find_cluster_collisions, resolve_collisions, long_to_cellregistry,
)
from visdetect.analysis.track_verdict import load_kept_map  # noqa: E402

DEFAULT_LONG = ("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
                "BG_046/unit_match/output/all42/unit_index.csv")
TRIMMED = REPO_ROOT / "FIGURES" / "tracking_qc" / "verdicts_trimmed.csv"
WIDE_OUT = REPO_ROOT / "data" / "unit_match" / "output" / "BG_046_um329_CellRegistry.csv"
GLT_OUT = REPO_ROOT / "table_output" / "Grand_Longitudinal_Table.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry-long", default=DEFAULT_LONG)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    long_df = load_canonical_long(args.registry_long)
    print(f"loaded {len(long_df)} unit-session rows, {long_df['global_uid'].nunique()} UIDs")

    coll = find_cluster_collisions(long_df)
    print(f"cluster collisions (one cluster -> >1 UID): {len(coll)} rows")

    kept = load_kept_map(TRIMMED) if TRIMMED.exists() else {}
    resolved = resolve_collisions(long_df, kept)
    print(f"after collision resolution: {len(resolved)} rows")

    if resolved.empty:
        raise ValueError(
            "resolved registry is empty (all rows ambiguous/dropped) — check the "
            "collision policy / kept-sessions inputs before the expensive GLT step")

    wide = long_to_cellregistry(resolved)
    WIDE_OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(WIDE_OUT)
    print(f"wrote wide CellRegistry: {WIDE_OUT}  ({wide.shape[0]} UIDs x {wide.shape[1]} sessions)")

    # Regenerate the GLT via the existing producer.
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "analysis" / "build_longitudinal_table.py"),
           "--registry", str(WIDE_OUT), "--output", str(GLT_OUT), "--workers", str(args.workers)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # build_longitudinal_table.py can exit 0 even if it fails to find/write a
    # registry; guard so we don't rebuild the unit table on a stale/absent GLT.
    if not GLT_OUT.exists() or GLT_OUT.stat().st_size == 0:
        raise RuntimeError(
            f"GLT not produced at {GLT_OUT} after build_longitudinal_table.py")

    # Rebuild + validate the unit table.
    from visdetect.suite.loader import build_unit_table
    df = build_unit_table(qc_only=True, validate=True)
    print(f"unit table rebuilt + validated: {len(df)} rows; "
          f"track_verdict counts:\n{df['track_verdict'].value_counts()}")


if __name__ == "__main__":
    main()
