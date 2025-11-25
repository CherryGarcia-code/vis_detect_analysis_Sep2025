#!/usr/bin/env python
"""Run MATLAB-style lick responsiveness analysis on session pickles.

Usage:
    python scripts/run_matlab_style_licks.py \
        --sessions pkls/BG_046_30062025.pkl pkls/BG_046_01072025.pkl \
        --out-dir table_output/matlab_ports/lick
"""

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

from visdetect.analysis.matlab_ports.lick import (
    MatlabLickConfig,
    compute_fa_lick_responsiveness,
)
from visdetect.analysis.su_analysis import load_kept_ids, selection_csv_default_path
from visdetect.core.legacy_io import session_summary, load_session


def _ensure_out_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_mean_psth(result, cfg: MatlabLickConfig, out_png: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.time_axis, result.psth_mean, color="#0055A4", label="Mean PSTH")
    ax.fill_between(
        result.time_axis,
        result.psth_mean - result.psth_sem,
        result.psth_mean + result.psth_sem,
        color="#0055A4",
        alpha=0.25,
        linewidth=0,
    )
    ax.axvspan(*cfg.baseline_window, color="#A5A5A5", alpha=0.2, label="Baseline window")
    ax.axvspan(*cfg.post_window, color="#E4572E", alpha=0.2, label="Post window")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Time from FA lick (s)")
    ax.set_ylabel("Spike count/bin (smoothed)")
    ax.set_title("MATLAB-style lick PSTH")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)



def analyze_session(
    session_path: str,
    out_dir: Path,
    cfg: MatlabLickConfig,
    profiles_root: str | None = None,
    profile_name: str | None = None,
    kept_only: bool = True,
) -> Path:
    session = load_session(session_path)
    summary = session_summary(session)
    good_ids = None
    if kept_only:
        try:
            # We assume unit_selection.csv already reflects the chosen profile
            root = profiles_root or "table_output/unit_qc"
            sel_csv = selection_csv_default_path(session, root=root)
            good_ids = load_kept_ids(session, selection_csv=str(sel_csv))
        except Exception as exc:
            print(
                f"[warn] kept_only requested but could not load selection for {session_path}: {exc}.\n"
                "       Falling back to session.good_cluster_ids.",
                file=sys.stderr,
            )
            good_ids = None

    result = compute_fa_lick_responsiveness(session, cfg, good_ids=good_ids)

    session_stub = Path(session_path).stem
    csv_path = out_dir / f"{session_stub}_matlab_lick.csv"
    png_path = out_dir / f"{session_stub}_matlab_lick_psth.png"

    result.table.to_csv(csv_path, index=False)
    _plot_mean_psth(result, cfg, png_path)

    meta = {
        "session_path": session_path,
        "csv": str(csv_path),
        "psth_png": str(png_path),
        "subject": summary.get("subject"),
        "session_name": summary.get("session_name"),
        "n_trials": summary.get("n_trials"),
        "n_clusters": summary.get("n_clusters"),
        "n_events_used": int(result.table["n_events"].max()) if not result.table.empty else 0,
        "n_sig_units": int(result.table["is_significant"].sum()) if not result.table.empty else 0,
    }
    return csv_path, png_path, meta



def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", required=True, help="Paths to session .pkl files")
    parser.add_argument(
        "--out-dir",
        default="table_output/matlab_ports/lick",
        help="Directory to store per-session outputs",
    )
    parser.add_argument(
        "--profiles-root",
        default="table_output/unit_qc",
        help="Root directory where unit selection CSVs are stored",
    )
    parser.add_argument(
        "--profile-name",
        default="striatal_strict",
        help="QC profile name to select kept_only units",
    )
    parser.add_argument(
        "--no-kept-only",
        action="store_true",
        help="Do not enforce kept_only selection; use session.good_cluster_ids",
    )
    args = parser.parse_args(argv)

    cfg = MatlabLickConfig()
    out_dir = _ensure_out_dir(args.out_dir)

    metas = []
    for session_path in args.sessions:
        csv_path, png_path, meta = analyze_session(
            session_path,
            out_dir,
            cfg,
            profiles_root=args.profiles_root,
            profile_name=args.profile_name,
            kept_only=(not args.no_kept_only),
        )
        metas.append(meta)
        print(f"Processed {session_path} -> {csv_path}")

    summary_csv = out_dir / "matlab_lick_summary.csv"
    pd.DataFrame(metas).to_csv(summary_csv, index=False)
    print(f"Wrote summary table to {summary_csv}")


if __name__ == "__main__":
    main()
