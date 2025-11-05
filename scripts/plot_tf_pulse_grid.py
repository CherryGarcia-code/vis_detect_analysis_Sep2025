"""Plot a grid of TF pulse per-unit z-scored mean traces for visual inspection.

Usage:
  python scripts/plot_tf_pulse_grid.py --file data/BG_031_260325.pkl --out png_output/tf_pulse_grids --which slow --cols 10
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.session_io import load_session
from src.tf_pulse import TFRespPulseConfig, plot_tf_pulse_grid


def main(argv=None):
    ap = argparse.ArgumentParser(description="TF pulse z-trace grid per session")
    ap.add_argument("--file", required=True, help="Path to session pickle file")
    ap.add_argument("--out", default="png_output/tf_pulse_grids", help="Output root for grid image")
    ap.add_argument("--which", choices=["fast", "slow", "both"], default="slow", help="Which traces to plot")
    ap.add_argument("--cols", type=int, default=10, help="Number of columns in the grid")
    ap.add_argument("--kept-only", action="store_true", help="Plot only kept/good clusters")
    args = ap.parse_args(argv)

    sess = load_session(args.file)
    ident = f"{getattr(sess,'subject','unknown')}_{getattr(sess,'session_name','unknown')}"
    cfg = TFRespPulseConfig(kept_only=args.kept_only)
    out_png = Path(args.out) / ident / f"tf_pulse_grid_{args.which}.png"
    p = plot_tf_pulse_grid(sess, str(out_png), cfg=cfg, selection_csv=None, n_cols=args.cols, which=args.which)
    print(f"Wrote grid: {p}")


if __name__ == "__main__":
    raise SystemExit(main())
