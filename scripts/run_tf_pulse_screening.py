"""Batch runner for TF pulse responsiveness screening.

Discovers session pickle files and runs TF pulse screening, writing per-session
CSVs and optional aggregate PNGs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback
import pandas as pd

# Ensure repository root (parent of scripts/) is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.session_io import load_session
from src.tf_pulse import TFRespPulseConfig, run_tf_pulse_screening


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run TF pulse responsiveness screening across sessions")
    ap.add_argument("--input-glob", default="data/*.pkl", help="Glob pattern for session pickle files")
    ap.add_argument("--out-root", default="table_output/tf_pulse", help="Root folder for per-session CSV outputs")
    ap.add_argument("--png-root", default="png_output/tf_pulse", help="Root folder for per-session PNG outputs")
    ap.add_argument("--selection-csv", default=None, help="Path to unit selection CSV (optional; defaults to unit_qc path per session)")
    ap.add_argument("--kept-only", action="store_true", help="Restrict to kept/good clusters only")
    ap.add_argument("--fast-thresh-log2", type=float, default=0.25, help="Log2(TF) threshold for fast pulses")
    ap.add_argument("--slow-thresh-log2", type=float, default=-0.25, help="Log2(TF) threshold for slow pulses")
    ap.add_argument("--pre-window", nargs=2, type=float, default=[-0.4, 0.0], help="Pre window [start, end] (s)")
    ap.add_argument("--post-window", nargs=2, type=float, default=[0.0, 0.5], help="Post window [start, end] (s)")
    ap.add_argument("--dt", type=float, default=0.001, help="Smoothing grid dt (s)")
    ap.add_argument("--sigma-ms", type=float, default=13.3, help="Gaussian sigma (ms) for smoothing")
    ap.add_argument("--z-thresh", type=float, default=3.0, help="Z threshold for responsiveness")
    grid_group = ap.add_mutually_exclusive_group()
    grid_group.add_argument("--grid", dest="grid", action="store_true", help="Generate per-session 'both' grid (default)")
    grid_group.add_argument("--no-grid", dest="grid", action="store_false", help="Skip grid generation to save time")
    ap.set_defaults(grid=True)
    args = ap.parse_args(argv)

    cfg = TFRespPulseConfig(
        fast_thresh_log2=args.fast_thresh_log2,
        slow_thresh_log2=args.slow_thresh_log2,
        pre_window=(args.pre_window[0], args.pre_window[1]),
        post_window=(args.post_window[0], args.post_window[1]),
        dt=args.dt,
        sigma_ms=args.sigma_ms,
        z_thresh=args.z_thresh,
        kept_only=args.kept_only,
    )

    files = sorted([str(p) for p in Path().glob(args.input_glob)])
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.png_root).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for f in files:
        try:
            sess = load_session(f)
            ident = f"{getattr(sess,'subject','unknown')}_{getattr(sess,'session_name','unknown')}"
            paths = run_tf_pulse_screening(sess, out_root=args.out_root, png_root=args.png_root, cfg=cfg, selection_csv=args.selection_csv, generate_grid=args.grid)
            manifest_rows.append({"file": f, "session": ident, "status": "OK", **paths})
            print(f"[OK] {ident} -> {paths.get('csv','')} ")
        except Exception as e:
            tb = traceback.format_exc()
            manifest_rows.append({"file": f, "session": Path(f).stem, "status": f"ERROR: {e}", "traceback": tb})
            print(f"[ERROR] {f}: {e}")

    man_df = pd.DataFrame(manifest_rows)
    man_path = Path(args.out_root) / "tf_pulse_manifest.csv"
    man_df.to_csv(man_path, index=False)
    print(f"Wrote manifest: {man_path}")


if __name__ == "__main__":
    sys.exit(main())
