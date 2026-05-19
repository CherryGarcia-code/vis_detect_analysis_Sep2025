"""Batch run TF pulse grid plotting for all pickles in a directory.

Usage:
    python scripts/batch_processing/batch_plot_tf_grids.py --pkl-dir pkls --out-dir png_output/tf_pulse_grids_v2 --profile striatal_strict
"""
import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

from visdetect.core.session import load_session

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir", default="pkls", type=Path)
    ap.add_argument("--out-dir", default="png_output/tf_pulse_grids_v2", type=Path)
    ap.add_argument("--profile", default="striatal_strict")
    ap.add_argument("--which", default="both")
    args = ap.parse_args()

    pkls = sorted(list(args.pkl_dir.glob("*.pkl")))
    if not pkls:
        print(f"No .pkl files found in {args.pkl_dir}")
        return

    print(f"Found {len(pkls)} sessions. Starting batch processing...")
    
    # Ensure output dir exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pkl in tqdm(pkls, desc="Sessions"):
        cmd = [
            sys.executable, "scripts/analysis/plot_tf_pulse_grid.py",
            "--file", str(pkl),
            "--out", str(args.out_dir),
            "--which", args.which,
            "--profile", args.profile,
            "--sort"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\nError processing {pkl.name}:")
            print(e.stderr)
            continue

if __name__ == "__main__":
    main()
