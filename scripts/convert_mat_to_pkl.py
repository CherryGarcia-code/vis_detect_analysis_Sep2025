"""Convert a legacy MATLAB session .mat file into a normalized .pkl Session.

Usage:
  python scripts/convert_mat_to_pkl.py data/BG_046_15082025.mat [--out data/BG_046_15082025.pkl]

This uses the existing helper that parses MATLAB structures and the shared
session_io.save_session to write a pickle compatible with the rest of the repo.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from visdetect.core.io import load_mat_file_to_session
from visdetect.core.legacy_io import save_session, session_summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert MATLAB .mat session to .pkl Session")
    p.add_argument("mat_path", type=Path, help="Path to source .mat file")
    p.add_argument("--out", type=Path, default=None, help="Output .pkl path (default: data/<stem>.pkl)")
    args = p.parse_args(argv)

    mat_path: Path = args.mat_path
    if not mat_path.exists():
        raise SystemExit(f"Input .mat not found: {mat_path}")

    out: Path
    if args.out is None:
        out = REPO / "data" / (mat_path.stem + ".pkl")
    else:
        out = args.out
        if not out.suffix:
            out = out.with_suffix(".pkl")

    print(f"[convert] Loading {mat_path} ...")
    session = load_mat_file_to_session(str(mat_path))
    print("[convert] Saving to", out)
    save_session(session, str(out))
    summ = session_summary(session)
    print("[summary]", summ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
