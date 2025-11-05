"""Summarize lick-aligned decoding timecourses across sessions.

Scans table_output/lick_decoding/<session>/<A>_vs_<B>/lick_cd_timecourse.csv and computes:
- window-averaged effect in a specified post-event window (default 0–0.2 s)
- onset time: earliest post-0 time where -log10(p) exceeds threshold for at least N consecutive bins
- optional area-above-threshold in the window (AAT)

Writes a tidy CSV and overview plots (effect vs comparison, onset vs comparison).
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _onset_time(df: pd.DataFrame, p_thresh: float, min_consec: int) -> float:
    if df.empty:
        return np.nan
    t = df["time"].to_numpy(dtype=float)
    p = df["p_value"].to_numpy(dtype=float)
    mask_post0 = t >= 0
    if not np.any(mask_post0):
        return np.nan
    t = t[mask_post0]
    p = p[mask_post0]
    sig = p < p_thresh
    if not np.any(sig):
        return np.nan
    run = 0
    for i, s in enumerate(sig):
        run = run + 1 if s else 0
        if run >= min_consec:
            idx = i - min_consec + 1
            return float(t[idx])
    return np.nan


def _window_effect(df: pd.DataFrame, w0: float, w1: float) -> float:
    if df.empty:
        return np.nan
    t = df["time"].to_numpy(dtype=float)
    e = df["effect"].to_numpy(dtype=float)
    mask = (t >= w0) & (t <= w1)
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(e[mask]))


def _aat(df: pd.DataFrame, w0: float, w1: float, p_thresh: float) -> float:
    if df.empty:
        return 0.0
    t = df["time"].to_numpy(dtype=float)
    p = df["p_value"].to_numpy(dtype=float)
    mask = (t >= w0) & (t <= w1)
    if not np.any(mask):
        return 0.0
    import math
    x = t[mask]
    y = -np.log10(np.clip(p[mask], 1e-12, 1.0))
    thr = -math.log10(p_thresh)
    y_pos = np.maximum(0.0, y - thr)
    if len(x) < 2:
        return float(y_pos.sum())
    dx = np.diff(x)
    step = float(np.nanmean(dx)) if dx.size else 0.0
    return float(y_pos.sum() * step)


DEFAULT_COMPARISON_ORDER = [
    "Hit_vs_FA_early",
    "Hit_vs_FA_late",
    "FA_early_vs_FA_late",
]


def summarize(lick_root: Path, out_csv: Path, png_dir: Path, w0: float, w1: float, p_thresh: float, min_consec: int):
    rows: List[Dict] = []

    for session_dir in sorted([p for p in lick_root.iterdir() if p.is_dir()]):
        # look for <A>_vs_<B>/lick_cd_timecourse.csv
        for pair_dir in session_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            csv_path = pair_dir / "lick_cd_timecourse.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            rows.append(
                {
                    "session": session_dir.name,
                    "comparison": pair_dir.name,
                    "effect_window": _window_effect(df, w0, w1),
                    "onset": _onset_time(df, p_thresh=p_thresh, min_consec=min_consec),
                    "aat": _aat(df, w0, w1, p_thresh=p_thresh),
                }
            )

    if not rows:
        print("No lick decoding CSVs found.")
        return

    out_df = pd.DataFrame(rows).sort_values(["comparison", "session"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    png_dir.mkdir(parents=True, exist_ok=True)

    # Determine categorical order: keep present comparisons in predefined order
    comps_present = [c for c in DEFAULT_COMPARISON_ORDER if c in set(out_df["comparison"].unique())]
    xpos = {c: i for i, c in enumerate(comps_present)}

    def _pretty_label(name: str) -> str:
        return name.replace("_vs_", " vs ")

    def _agg_plot(y_col: str, ylabel: str, fname: str):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        means = []
        sems = []
        for c in comps_present:
            vals = out_df.loc[out_df["comparison"] == c, y_col].astype(float).dropna().values
            if vals.size == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                means.append(float(np.nanmean(vals)))
                sems.append(float(np.nanstd(vals) / np.sqrt(max(1, len(vals)))))
            xs = np.full((len(vals),), xpos[c])
            ax.scatter(xs, vals, color="#7f7f7f", s=16, alpha=0.7)
        ax.errorbar([xpos[c] for c in comps_present], means, yerr=sems, color="#1f77b4", marker="o", linewidth=2)
        ax.set_xticks([xpos[c] for c in comps_present], [_pretty_label(c) for c in comps_present])
        ax.set_xlabel("Comparison (categorical)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs comparison")
        fig.tight_layout()
        fig.savefig(png_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _agg_plot("effect_window", f"Mean effect {w0:g}–{w1:g}s", "effect_vs_comparison.png")
    _agg_plot("onset", "Onset time (s)", "onset_vs_comparison.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lick-root", default="table_output/lick_decoding")
    p.add_argument("--out-root", default="table_output/lick_decoding_summary")
    p.add_argument("--png-root", default="png_output/lick_decoding_summary")
    p.add_argument("--effect-window", nargs=2, type=float, default=[0.0, 0.2])
    p.add_argument("--p-thresh", type=float, default=0.05)
    p.add_argument("--min-consec", type=int, default=2)
    args = p.parse_args()

    lick_root = Path(args.lick_root)
    out_root = Path(args.out_root)
    png_root = Path(args.png_root)

    out_csv = out_root / "lick_decoding_summary.csv"
    summarize(
        lick_root=lick_root,
        out_csv=out_csv,
        png_dir=png_root,
        w0=float(args.effect_window[0]),
        w1=float(args.effect_window[1]),
        p_thresh=float(args.p_thresh),
        min_consec=int(args.min_consec),
    )
    print(f"Wrote summary CSV to {out_csv}")


if __name__ == "__main__":
    main()
