"""Change-size tuning analysis for single units.

Computes per-unit tuning to Change_ON size on Hit trials. Produces:
- unit_tuning_by_size.csv: long table per (unit, size) with base/resp FR and delta
- unit_tuning.csv: per-unit summary with slope (delta vs size), R^2, p-values
  from one-way Kruskal-Wallis across sizes and from a monotonic trend test.

Optionally generates per-unit tuning plots.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr

from visdetect.analysis import align as align_mod
from visdetect.analysis.su_analysis import load_kept_ids


@dataclass
class TuningConfig:
    event_name: str = "Change_ON"
    base_win: Tuple[float, float] = (-0.2, 0.0)
    resp_win: Tuple[float, float] = (0.0, 0.2)
    bin_size: float = 0.01
    kept_only: bool = True
    min_trials_per_size: int = 4
    max_sizes: int = 12  # for plotting; compute uses all
    smooth_sigma: Optional[float] = None


def _fr_per_trial(spike_times: np.ndarray, event_times: List[float], window: Tuple[float, float], bin_size: float) -> np.ndarray:
    M, _ = align_mod.align_spikes_to_events(spike_times, event_times, window=window, bin_size=bin_size)
    if M.size == 0:
        return np.zeros((0,), dtype=float)
    return np.nanmean(M, axis=1)


def _collect_hit_trials_with_size(session) -> pd.DataFrame:
    """Return DataFrame with columns: trial_idx, size, event_time for Hit trials with a valid size and event time."""
    trials = getattr(session, "trials", []) or []
    ets = align_mod.get_event_times_by_trial(session, "Change_ON")
    rows = []
    for i, t in enumerate(trials):
        if getattr(t, "trialoutcome", None) != "Hit":
            continue
        size = getattr(t, "change_size", None)
        if size is None:
            continue
        try:
            et = float(ets[i])
        except Exception:
            et = np.nan
        if np.isnan(et):
            continue
        try:
            size_f = float(size)
        except Exception:
            continue
        rows.append((i, size_f, et))
    return pd.DataFrame(rows, columns=["trial_idx", "size", "event_time"]) if rows else pd.DataFrame(columns=["trial_idx", "size", "event_time"])  # type: ignore


def compute_change_size_tuning_table(session, cfg: Optional[TuningConfig] = None, selection_csv: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute tuning tables.

    Returns (by_size, summary):
      - by_size: columns [cluster_id, size, n_trials, base_mean, resp_mean, delta_mean]
      - summary: one row per cluster with stats: n_sizes, n_trials, slope, r2, p_kw, p_trend
    """
    if cfg is None:
        cfg = TuningConfig()

    hits = _collect_hit_trials_with_size(session)
    if hits.empty:
        # Return empty tables
        return (
            pd.DataFrame(columns=["cluster_id", "size", "n_trials", "base_mean", "resp_mean", "delta_mean"]),
            pd.DataFrame(columns=["cluster_id", "n_sizes", "n_trials", "slope", "r2", "p_kw", "p_trend"]),
        )

    # Select clusters
    cluster_ids = [int(c.cluster_id) for c in session.clusters]
    if cfg.kept_only:
        kept = set(load_kept_ids(session, selection_csv))
        if kept:
            cluster_ids = [cid for cid in cluster_ids if cid in kept]

    sizes_sorted = np.sort(hits["size"].unique())

    by_size_rows = []
    summary_rows = []
    for cid in cluster_ids:
        c = next((x for x in session.clusters if int(x.cluster_id) == int(cid)), None)
        if c is None:
            continue
        st = np.asarray(c.spike_times).flatten()

        # Per-size stats
        deltas = []
        size_list = []
        n_total = 0
        groups_for_kw: List[np.ndarray] = []
        for size in sizes_sorted:
            ets = hits.loc[hits["size"] == size, "event_time"].tolist()
            if len(ets) < cfg.min_trials_per_size:
                continue
            base = _fr_per_trial(st, ets, cfg.base_win, cfg.bin_size)
            resp = _fr_per_trial(st, ets, cfg.resp_win, cfg.bin_size)
            if base.size == 0 or resp.size == 0:
                continue
            diff = resp - base
            by_size_rows.append(
                {
                    "cluster_id": cid,
                    "size": float(size),
                    "n_trials": int(len(ets)),
                    "base_mean": float(np.nanmean(base)),
                    "resp_mean": float(np.nanmean(resp)),
                    "delta_mean": float(np.nanmean(diff)),
                }
            )
            deltas.append(np.nanmean(diff))
            groups_for_kw.append(diff)
            size_list.append(float(size))
            n_total += len(ets)

        if len(size_list) >= 2:
            # Linear trend via Spearman between size and delta_mean (robust monotonic test)
            rho, p_trend = spearmanr(size_list, deltas)
            # Simple least-squares slope and R^2
            x = np.array(size_list, dtype=float)
            y = np.array(deltas, dtype=float)
            x_mean = x.mean()
            y_mean = y.mean()
            denom = np.sum((x - x_mean) ** 2)
            if denom > 0:
                slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
            else:
                slope = 0.0
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum((y - (x_mean + slope * (x - x_mean) + y_mean - slope * x_mean)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            # Kruskal across per-trial diffs
            try:
                if len(groups_for_kw) >= 2 and all(len(g) >= cfg.min_trials_per_size for g in groups_for_kw):
                    _, p_kw = kruskal(*groups_for_kw)
                else:
                    p_kw = np.nan
            except Exception:
                p_kw = np.nan
        else:
            slope, r2, p_kw, p_trend = (np.nan, np.nan, np.nan, np.nan)

        summary_rows.append(
            {
                "cluster_id": cid,
                "n_sizes": int(len(size_list)),
                "n_trials": int(n_total),
                "slope": float(slope),
                "r2": float(r2),
                "p_kw": float(p_kw) if p_kw == p_kw else np.nan,
                "p_trend": float(p_trend) if p_trend == p_trend else np.nan,
            }
        )

    by_size_df = pd.DataFrame(by_size_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not by_size_df.empty:
        by_size_df = by_size_df.sort_values(["cluster_id", "size"]).reset_index(drop=True)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["cluster_id"]).reset_index(drop=True)
    return by_size_df, summary_df


def plot_unit_tuning(by_size_df: pd.DataFrame, cid: int, out_path: Optional[str] = None):
    """Plot delta_mean vs size on a categorical x-axis with equidistant ticks.

    Size order: [1.0 (reference), 1.25, 1.35, 1.5, 2.0, 4.0], filtered to sizes present.
    """
    df = by_size_df.loc[by_size_df["cluster_id"] == cid]
    if df.empty:
        return None
    # Categorical order
    ORDER = [1.0, 1.25, 1.35, 1.5, 2.0, 4.0]
    sizes_present = [s for s in ORDER if s in set(df["size"].astype(float).unique())]
    if not sizes_present:
        return None
    pos = {s: i for i, s in enumerate(sizes_present)}
    # Map to positions in specified order
    df_sorted = df.copy()
    df_sorted = df_sorted[df_sorted["size"].isin(sizes_present)]
    df_sorted["_x"] = df_sorted["size"].map(pos)
    df_sorted = df_sorted.sort_values("_x")

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.8))
    ax.plot(df_sorted["_x"].values, df_sorted["delta_mean"].values, marker="o", color="#1f77b4")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    # Ticks and labels
    tick_labels = ["1.0 (reference)" if abs(s - 1.0) < 1e-6 else f"{s:g}" for s in sizes_present]
    ax.set_xticks([pos[s] for s in sizes_present], tick_labels)
    ax.set_xlabel("Change size (categorical)")
    ax.set_ylabel("ΔFR (Hz): resp − base")
    ax.set_title(f"Unit {cid} tuning (Hit trials)")
    fig.tight_layout()
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def run_tuning_for_session(session, out_dir: str, png_dir: Optional[str] = None, cfg: Optional[TuningConfig] = None, selection_csv: Optional[str] = None) -> Dict[str, str]:
    """Compute tuning tables and save; optionally generate per-unit tuning plots.

    Returns dict of output paths.
    """
    if cfg is None:
        cfg = TuningConfig()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    by_size, summary = compute_change_size_tuning_table(session, cfg=cfg, selection_csv=selection_csv)
    paths: Dict[str, str] = {}
    by_csv = outp / "unit_tuning_by_size.csv"
    sm_csv = outp / "unit_tuning.csv"
    by_size.to_csv(by_csv, index=False)
    summary.to_csv(sm_csv, index=False)
    paths["by_size_csv"] = str(by_csv)
    paths["summary_csv"] = str(sm_csv)

    if png_dir is not None and not by_size.empty:
        outpng = Path(png_dir)
        outpng.mkdir(parents=True, exist_ok=True)
        for cid in sorted(by_size["cluster_id"].unique().tolist()):
            plot_unit_tuning(by_size, cid, out_path=str(outpng / f"cluster_{cid}_tuning.png"))
    return paths
