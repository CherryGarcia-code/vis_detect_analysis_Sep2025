"""Summarize per-size decoding timecourses across sessions.

Scans table_output/decoding/<session>/size_*/decoding_timecourse.csv and computes:
- window-averaged effect in a specified post-event window (default 0–0.2 s)
- onset time: earliest post-0 time where -log10(p) exceeds threshold for at least N consecutive bins
- optional area-above-threshold in the window (AAT)

Writes a tidy CSV and overview plots (effect vs size, onset vs size).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_size_folder(name: str) -> float | None:
    # Expect 'size_<value>' with 'p' as decimal point
    m = re.match(r"^size_([0-9]+p[0-9]+|[0-9]+)$", name)
    if not m:
        return None
    s = m.group(1).replace("p", ".")
    try:
        return float(s)
    except Exception:
        return None


def _onset_time(df: pd.DataFrame, p_thresh: float, min_consec: int) -> float:
    # df columns: time, effect, p_value
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
    # find first run of length >= min_consec
    run = 0
    for i, s in enumerate(sig):
        run = run + 1 if s else 0
        if run >= min_consec:
            # onset at start of this run
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
    # area above threshold of -log10(p); approximate with rectangle rule
    import math

    x = t[mask]
    y = -np.log10(np.clip(p[mask], 1e-12, 1.0))
    thr = -math.log10(p_thresh)
    y_pos = np.maximum(0.0, y - thr)
    if len(x) < 2:
        return float(y_pos.sum())
    dx = np.diff(x)
    # assume uniform binning; use mean dx
    step = float(np.nanmean(dx)) if dx.size else 0.0
    return float(y_pos.sum() * step)


DEFAULT_SIZE_ORDER = [1.0, 1.25, 1.35, 1.5, 2.0, 4.0]


def summarize(decoding_root: Path, out_csv: Path, png_dir: Path, w0: float, w1: float, p_thresh: float, min_consec: int):
    rows: List[Dict] = []
    # Iterate sessions: include only directories under decoding_root
    for session_dir in sorted([p for p in decoding_root.iterdir() if p.is_dir()]):
        # session_dir: table_output/decoding/<session>
        sizes = []
        for sub in session_dir.iterdir():
            if not sub.is_dir():
                continue
            size_val = _parse_size_folder(sub.name)
            if size_val is None:
                continue
            csv_path = sub / "decoding_timecourse.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            rows.append(
                {
                    "session": session_dir.name,
                    "size": float(size_val),
                    "effect_window": _window_effect(df, w0, w1),
                    "onset": _onset_time(df, p_thresh=p_thresh, min_consec=min_consec),
                    "aat": _aat(df, w0, w1, p_thresh=p_thresh),
                }
            )
            sizes.append(size_val)

    if not rows:
        print("No per-size decoding CSVs found.")
        return

    out_df = pd.DataFrame(rows).sort_values(["size", "session"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # Plots: effect vs size and onset vs size (scatter per session + mean±SEM)
    png_dir.mkdir(parents=True, exist_ok=True)

    # Determine categorical order: only include sizes present, keep fixed order
    sizes_present = [s for s in DEFAULT_SIZE_ORDER if s in set(out_df["size"].unique())]
    xpos = {s: i for i, s in enumerate(sizes_present)}

    def _agg_plot(y_col: str, ylabel: str, fname: str):
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
        sizes = sizes_present
        means = []
        sems = []
        for s in sizes:
            vals = out_df.loc[out_df["size"] == s, y_col].astype(float).dropna().values
            if vals.size == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                means.append(float(np.nanmean(vals)))
                sems.append(float(np.nanstd(vals) / np.sqrt(max(1, len(vals)))))
            # scatter per session
            xs = np.full((len(vals),), xpos[s])
            ax.scatter(xs, vals, color="#7f7f7f", s=16, alpha=0.7)
        ax.errorbar([xpos[s] for s in sizes], means, yerr=sems, color="#1f77b4", marker="o", linewidth=2)
        # Category ticks and labels
        tick_labels = ["1.0 (reference)" if abs(s - 1.0) < 1e-6 else f"{s:g}" for s in sizes]
        ax.set_xticks([xpos[s] for s in sizes], tick_labels)
        ax.set_xlabel("Change size (categorical)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs size")
        fig.tight_layout()
        fig.savefig(png_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _agg_plot("effect_window", f"Mean effect {w0:g}–{w1:g}s", "effect_vs_size.png")
    _agg_plot("onset", "Onset time (s)", "onset_vs_size.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoding-root", default="table_output/decoding")
    p.add_argument("--out-root", default="table_output/decoding_summary")
    p.add_argument("--png-root", default="png_output/decoding_summary")
    p.add_argument("--effect-window", nargs=2, type=float, default=[0.0, 0.2])
    p.add_argument("--p-thresh", type=float, default=0.05)
    p.add_argument("--min-consec", type=int, default=2)
    args = p.parse_args()

    decoding_root = Path(args.decoding_root)
    out_root = Path(args.out_root)
    png_root = Path(args.png_root)

    out_csv = out_root / "decoding_per_size_summary.csv"
    summarize(
        decoding_root=decoding_root,
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
