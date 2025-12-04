"""Run TF pulse screening for an explicit list of session pickle files.

This small helper mirrors the batch runner but accepts explicit file paths
so callers can pick arbitrary subsets without relying on glob patterns.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback
import pandas as pd

# Ensure repository root is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visdetect.core.session import load_session
from visdetect.analysis.su_analysis import selection_csv_default_path
from visdetect.analysis.tf_pulse import TFRespPulseConfig, run_tf_pulse_screening


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run TF pulse screening for listed session files")
    ap.add_argument("files", nargs="+", help="Explicit session pickle files to process")
    ap.add_argument("--out-root", default="group_meeting_27112025/table_output/tf_pulse", help="Root folder for per-session CSV outputs")
    ap.add_argument("--png-root", default="group_meeting_27112025/png_output/tf_pulse", help="Root folder for per-session PNG outputs")
    ap.add_argument("--selection-csv", default=None, help="Explicit path to unit selection CSV (overrides per-session default)")
    ap.add_argument("--profiles-root", default="group_meeting_27112025/table_output/unit_qc", help="Root where per-session unit_selection.csv files are stored")
    ap.add_argument("--no-grid", dest="grid", action="store_false", help="Skip grid generation to save time")
    ap.set_defaults(grid=True)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", help="Skip sessions that already have tf_pulse_units.csv in out-root")
    ap.add_argument("--fast-thresh-log2", type=float, default=0.25)
    ap.add_argument("--slow-thresh-log2", type=float, default=-0.25)
    ap.add_argument("--z-thresh", type=float, default=3.0)
    args = ap.parse_args(argv)

    cfg = TFRespPulseConfig(
        fast_thresh_log2=args.fast_thresh_log2,
        slow_thresh_log2=args.slow_thresh_log2,
        pre_window=( -0.4, 0.0),
        post_window=(0.0, 0.5),
        dt=0.001,
        sigma_ms=13.3,
        z_thresh=args.z_thresh,
    )

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.png_root).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for f in args.files:
        fpath = str(Path(f))
        stem = Path(fpath).stem
        out_dir_this = Path(args.out_root) / stem
        csv_check = out_dir_this / "tf_pulse_units.csv"
        if args.skip_existing and csv_check.exists():
            manifest_rows.append({"file": fpath, "session": stem, "status": "SKIPPED"})
            print(f"[SKIP] {stem} (tf_pulse_units.csv exists)")
            continue
        try:
            sess = load_session(fpath)
            ident = f"{getattr(sess,'subject','unknown')}_{getattr(sess,'session_name','unknown')}"
            sel_csv = args.selection_csv
            if cfg.kept_only and sel_csv is None:
                sel_csv = str(selection_csv_default_path(sess, root=args.profiles_root))
            paths = run_tf_pulse_screening(
                sess,
                out_root=args.out_root,
                png_root=args.png_root,
                cfg=cfg,
                selection_csv=sel_csv,
                generate_grid=args.grid,
                show_progress=True,
            )
            manifest_rows.append({"file": fpath, "session": ident, "status": "OK", **paths})
            print(f"[OK] {ident} -> {paths.get('csv','')}")
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
