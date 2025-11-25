#!/usr/bin/env python
"""Batch-run unit selection across all .pkl sessions in data/.

Usage (examples):

# Default: Baseline_ON, window [-0.5, 1.0], bin=0.02
python scripts/run_unit_selection_batch.py

# Customize thresholds and event
python scripts/run_unit_selection_batch.py \
  --event Baseline_ON \
  --min-total-spikes 400 \
  --min-mean-rate 0.05 \
  --max-isi-frac 0.2 \
  --min-median-spt 0.1

Outputs per session go into table_output/unit_qc/<subject_session>/
Writes unit_metrics.csv, unit_selection.csv, and quick validation plots.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from visdetect.core.legacy_io import load_session, session_summary
from src import qc


def _progress(iterable, *, desc: str = "", unit: str = "item"):
    total = len(iterable) if hasattr(iterable, "__len__") else None
    if total is None or total == 0:
        for item in iterable:
            yield item
        return
    width = 30
    print(f"{desc} (0/{total} {unit})")
    for idx, item in enumerate(iterable, 1):
        pct = idx / total
        filled = int(width * pct)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r{desc} [{bar}] {idx}/{total} {unit}", end="", flush=True)
        yield item
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Batch unit selection across data/*.pkl")
    p.add_argument("--data-dir", default="data", help="Folder containing *.pkl sessions")
    p.add_argument("--out-root", default="table_output/unit_qc", help="Root folder for outputs")
    p.add_argument("--event", default="Baseline_ON", help="Alignment event name")
    p.add_argument("--window", nargs=2, type=float, default=[-0.5, 1.0], help="Window [s] around event")
    p.add_argument("--bin-size", type=float, default=0.02, help="Bin size [s]")
    p.add_argument("--profile", default=None, help="QC profile name (see config/qc_profiles.yml)")
    p.add_argument("--profiles-yaml", default=None, help="Path to qc_profiles.yml (optional)")
    p.add_argument("--min-total-spikes", type=int, default=None)
    p.add_argument("--min-mean-rate", type=float, default=None)
    p.add_argument("--max-isi-frac", type=float, default=None)
    p.add_argument("--min-median-spt", type=float, default=None, help="Min median spikes per trial in window")
    p.add_argument("--max-median-spt", type=float, default=None, help="Max median spikes per trial in window (optional)")
    p.add_argument("--no-require-good", action="store_true", help="Do not require 'good' cluster label")
    p.add_argument("--only-new", action="store_true", help="Skip sessions that already have unit_selection.csv")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pkls = sorted(data_dir.glob("*.pkl"))
    if not pkls:
        print("No .pkl files found in", data_dir)
        return 0

    print(f"Found {len(pkls)} pkl files in {data_dir}")
    for pkl in _progress(pkls, desc="Processing sessions", unit="session"):
        try:
            sess = load_session(str(pkl))
            summ = session_summary(sess)
            subj = summ.get("subject") or "unknown"
            sname = summ.get("session_name") or pkl.stem
            outdir = out_root / f"{subj}_{sname}"
            if args.only_new and (outdir / "unit_selection.csv").exists():
                print(f"[skip] {pkl.name} -> already exists: {outdir}")
                continue
            print(f"[run] {pkl.name} -> {outdir}")
            # Build optional overrides dict (only include non-None)
            overrides = {
                k: v
                for k, v in dict(
                    require_good_cluster=not args.no_require_good,
                    min_total_spikes=args.min_total_spikes,
                    min_mean_rate_hz=args.min_mean_rate,
                    max_isi_viol_frac=args.max_isi_frac,
                    min_median_spikes_per_trial=args.min_median_spt,
                    max_median_spikes_per_trial=args.max_median_spt,
                ).items()
                if v is not None
            }

            res = qc.run_unit_selection(
                sess,
                outdir=str(outdir),
                event_name=args.event,
                window=(float(args.window[0]), float(args.window[1])),
                bin_size=float(args.bin_size),
                profile=args.profile,
                profiles_path=args.profiles_yaml,
                params=overrides,
                make_plots=True,
            )
            print(f"  kept {res['n_kept']} / {res['n_total']}")
        except Exception as e:
            print(f"[error] {pkl.name}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
