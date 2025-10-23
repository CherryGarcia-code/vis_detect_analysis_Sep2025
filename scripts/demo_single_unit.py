"""Demo script: compute QC table and raster+PSTH for a session.

Usage: python scripts/demo_single_unit.py path/to/session.pkl --out-dir png_output/demo_single_unit --n 2
"""
import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'src'))

from src.su_analysis import demo_for_session


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Demo single-unit QC and raster/PSTH")
    p.add_argument("session", help="Path to session pickle")
    p.add_argument("--out-dir", default=str(repo_root / "png_output" / "demo_single_unit"))
    # allow optional positional overrides to avoid ambiguous arg parsing with conda
    p.add_argument("n", nargs='?', type=int, default=None, help="(positional) Number of example clusters to plot")
    p.add_argument("out_dir_pos", nargs='?', default=None, help="(positional) output directory")
    p.add_argument("--n", dest="n_flag", type=int, default=None, help="Number of example clusters to plot")
    p.add_argument("--event", type=str, default="Change_ON", help="Event to align to")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # allow environment variable overrides to avoid command-line arg parsing issues
    import os
    env_n = os.environ.get('DEMO_N')
    env_out = os.environ.get('DEMO_OUTDIR')
    if env_n is not None:
        try:
            args.n = int(env_n)
        except Exception:
            pass
    if env_out is not None:
        args.out_dir = env_out
    # positional overrides
    if getattr(args, 'n', None) is None and getattr(args, 'n_flag', None) is not None:
        args.n = args.n_flag
    if getattr(args, 'out_dir_pos', None) is not None:
        args.out_dir = args.out_dir_pos
    res = demo_for_session(args.session, out_dir=args.out_dir, n_examples=args.n, event_name=args.event)
    print("Wrote:")
    print("QC:", res["qc_csv"])
    for p in res["pngs"]:
        print("PNG:", p)


if __name__ == "__main__":
    main()
