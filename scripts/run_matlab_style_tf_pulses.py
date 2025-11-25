#!/usr/bin/env python
"""Run MATLAB-style TF pulse responsiveness analysis on session pickles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.matlab_ports.tf_pulse import (  # noqa: E402
    TFPulseConfig,
    compute_tf_pulse_responsiveness,
)
from visdetect.core.legacy_io import load_session, session_summary  # noqa: E402
from visdetect.analysis.su_analysis import load_kept_ids, selection_csv_default_path  # noqa: E402


def _ensure_out_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_psth(result, cfg: TFPulseConfig, out_png: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.time_axis, result.fast_mean, color="#D9534F", label="Fast pulses")
    ax.fill_between(
        result.time_axis,
        result.fast_mean - result.fast_sem,
        result.fast_mean + result.fast_sem,
        color="#D9534F",
        alpha=0.25,
        linewidth=0,
    )
    ax.plot(result.time_axis, result.slow_mean, color="#3778BF", label="Slow pulses")
    ax.fill_between(
        result.time_axis,
        result.slow_mean - result.slow_sem,
        result.slow_mean + result.slow_sem,
        color="#3778BF",
        alpha=0.25,
        linewidth=0,
    )
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from TF pulse (s)")
    ax.set_ylabel("Spike count/bin (smoothed)")
    ax.set_title("MATLAB-style TF pulse PSTH")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def analyze_session(session_path: str, out_dir: Path, cfg: TFPulseConfig, profiles_root: str | None = None, kept_only: bool = True):
    session = load_session(session_path)
    summary = session_summary(session)
    good_ids = None
    if kept_only:
        try:
            root = profiles_root or "table_output/unit_qc"
            sel_csv = selection_csv_default_path(session, root=root)
            good_ids = load_kept_ids(session, selection_csv=str(sel_csv))
        except Exception:
            good_ids = None
    result = compute_tf_pulse_responsiveness(session, cfg, good_ids=good_ids)

    stub = Path(session_path).stem
    csv_path = out_dir / f"{stub}_matlab_tf.csv"
    png_path = out_dir / f"{stub}_matlab_tf_psth.png"
    meta_path = out_dir / f"{stub}_matlab_tf_meta.json"

    result.table.to_csv(csv_path, index=False)
    _plot_psth(result, cfg, png_path)

    meta = {
        "session": session_path,
        "subject": summary.get("subject"),
        "session_name": summary.get("session_name"),
        "n_trials": summary.get("n_trials"),
        "n_clusters": summary.get("n_clusters"),
        "n_fast_events": result.n_fast_events,
        "n_slow_events": result.n_slow_events,
        "csv": str(csv_path),
        "psth_png": str(png_path),
    }
    pd.Series(meta).to_json(meta_path, indent=2)
    return meta


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument(
        "--out-dir",
        default="table_output/matlab_ports/tf",
        help="Directory for CSVs and plots",
    )
    parser.add_argument("--fast-threshold", type=float, default=1.2)
    parser.add_argument("--slow-threshold", type=float, default=0.8)
    parser.add_argument("--min-events", type=int, default=8)
    parser.add_argument("--profiles-root", default="table_output/unit_qc")
    kept_group = parser.add_mutually_exclusive_group()
    kept_group.add_argument("--kept-only", dest="kept_only", action="store_true")
    kept_group.add_argument("--no-kept-only", dest="kept_only", action="store_false")
    parser.set_defaults(kept_only=True)
    args = parser.parse_args(argv)

    cfg = TFPulseConfig(
        fast_threshold=args.fast_threshold,
        slow_threshold=args.slow_threshold,
        min_events=args.min_events,
    )
    out_dir = _ensure_out_dir(args.out_dir)

    metas = []
    for session_path in args.sessions:
        meta = analyze_session(session_path, out_dir, cfg, profiles_root=args.profiles_root, kept_only=args.kept_only)
        metas.append(meta)
        print(f"Processed {session_path} -> {meta['csv']}")

    summary_csv = out_dir / "matlab_tf_summary.csv"
    pd.DataFrame(metas).to_csv(summary_csv, index=False)
    print(f"Wrote summary table to {summary_csv}")


if __name__ == "__main__":
    main()
