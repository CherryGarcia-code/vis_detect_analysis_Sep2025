"""Population decoding (Hit vs Miss) using coding direction.

Builds a trials x bins x units population tensor aligned to an event, filters
to specified outcomes, and applies time_resolved_cd to compute projections,
effects, and permutation p-values across time.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Set
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import align as align_mod
from src.su_analysis import load_kept_ids
from src.coding_direction import time_resolved_cd


@dataclass
class DecodingConfig:
    event_name: str = "Change_ON"
    window: Tuple[float, float] = (-0.5, 0.5)
    bin_size: float = 0.02
    kept_only: bool = True
    outcomes: Sequence[str] = ("Hit", "Miss")
    method: str = "shrinkage"  # or 'ridge'
    reg: float = 1.0
    n_splits: int = 5
    n_permutations: int = 200
    random_state: int = 0
    # Optional unit filtering by responsiveness
    responsive_only: bool = False
    responsiveness_csv: Optional[str] = None
    responsiveness_outcome: Optional[str] = None  # e.g., 'All', 'Hit', 'Miss', or None for any
    responsiveness_p_thresh: float = 0.05
    # Trial filters
    size_filter: Optional[Tuple[float, float]] = None  # (min, max) inclusive
    min_trials_per_class: int = 8


def _responsive_unit_ids(csv_path: Optional[str], outcome: Optional[str], p_thresh: float) -> Optional[Set[int]]:
    if not csv_path:
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    # outcome handling: if outcome is None -> accept any outcome row; if 'All' present, prefer that match
    if outcome:
        mask = df["outcome"].astype(str).str.lower() == str(outcome).strip().lower()
        dff = df.loc[mask]
        if dff.empty:
            dff = df  # fallback to any outcome
    else:
        dff = df
    try:
        dff = dff.loc[(dff["p_value"].astype(float) < p_thresh) & (dff["is_responsive"].astype(bool))]
    except Exception:
        dff = dff.loc[dff.get("is_responsive", False) == True]
    if dff.empty:
        return set()
    return set(dff["cluster_id"].astype(int).unique().tolist())


def build_population_tensor(session, cfg: DecodingConfig, selection_csv: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pop, y, bin_centers).

    pop: trials x bins x units for trials matching cfg.outcomes
    y: binary labels (1 for cfg.outcomes[0], 0 for cfg.outcomes[1])
    bin_centers: time axis
    """
    # Determine cluster set
    cluster_ids = [int(c.cluster_id) for c in session.clusters]
    if cfg.kept_only:
        kept = set(load_kept_ids(session, selection_csv))
        if kept:
            cluster_ids = [cid for cid in cluster_ids if cid in kept]
    # Optionally restrict to responsive units
    if cfg.responsive_only:
        resp_set = _responsive_unit_ids(cfg.responsiveness_csv, cfg.responsiveness_outcome, cfg.responsiveness_p_thresh)
        if resp_set is not None:
            cluster_ids = [cid for cid in cluster_ids if cid in resp_set]
    if not cluster_ids:
        raise ValueError("No clusters available for decoding")

    trials = getattr(session, "trials", []) or []
    ets_by_trial = align_mod.get_event_times_by_trial(session, cfg.event_name)
    labels = []
    sel_event_times = []
    sel_trial_indices = []
    # Collect trials for the two outcomes in order
    positive, negative = cfg.outcomes[0], cfg.outcomes[1]
    for i, t in enumerate(trials):
        out = getattr(t, "trialoutcome", None)
        if out not in cfg.outcomes:
            continue
        try:
            et = float(ets_by_trial[i])
        except Exception:
            et = np.nan
        if np.isnan(et):
            continue
        # Optional size filter
        if cfg.size_filter is not None:
            sz = getattr(t, "change_size", None)
            try:
                if sz is None or not (cfg.size_filter[0] <= float(sz) <= cfg.size_filter[1]):
                    continue
            except Exception:
                continue
        labels.append(1 if out == positive else 0)
        sel_event_times.append(et)
        sel_trial_indices.append(i)
    if len(sel_event_times) < 2 * cfg.min_trials_per_class or len(np.unique(labels)) < 2:
        raise ValueError("Not enough trials or only one outcome present for decoding")

    # Consistent bins
    _, bin_centers = align_mod.align_spikes_to_events(np.array([]), sel_event_times, window=cfg.window, bin_size=cfg.bin_size)

    # Build trials x bins x units tensor
    units = []
    for c in session.clusters:
        cid = int(c.cluster_id)
        if cid not in cluster_ids:
            continue
        M, _ = align_mod.align_spikes_to_events(c.spike_times, sel_event_times, window=cfg.window, bin_size=cfg.bin_size)
        # M: trials x bins
        units.append(M)
    if not units:
        raise ValueError("No units after alignment")
    # Stack into tensor: trials x bins x units
    # Ensure all unit matrices have same shape (robust to empty)
    min_trials = min(u.shape[0] for u in units)
    if min_trials == 0:
        raise ValueError("Aligned matrices are empty")
    # Truncate to min_trials to align across units (should be equal normally)
    units_trunc = [u[:min_trials, :] for u in units]
    pop = np.stack(units_trunc, axis=2)
    y = np.array(labels[:min_trials], dtype=int)
    return pop, y, bin_centers


def run_time_resolved_decoding(session, out_dir: str, cfg: Optional[DecodingConfig] = None, selection_csv: Optional[str] = None) -> Dict[str, str]:
    """Run time-resolved CD decoding and save CSV and plot.

    Outputs under out_dir: decoding_timecourse.csv and decoding_timecourse.png.
    """
    if cfg is None:
        cfg = DecodingConfig()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    pop, y, bc = build_population_tensor(session, cfg, selection_csv=selection_csv)
    res = time_resolved_cd(
        pop,
        cond_mask=(y == 1),
        method=cfg.method,
        reg=cfg.reg,
        n_splits=cfg.n_splits,
        n_permutations=cfg.n_permutations,
        random_state=cfg.random_state,
    )
    # Save CSV
    df = pd.DataFrame({
        "time": bc,
        "effect": res["effect"],
        "p_value": res["pvals"],
    })
    csv_path = outp / "decoding_timecourse.csv"
    df.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    # Primary: CD effect
    line_eff, = ax.plot(
        bc,
        res["effect"],
        color="#2ca02c",
        label=f"{cfg.outcomes[0]} − {cfg.outcomes[1]}",
    )
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    # Secondary axis for -log10 p
    ax2 = ax.twinx()
    line_p, = ax2.plot(
        bc,
        -np.log10(np.clip(res["pvals"], 1e-12, 1.0)),
        color="#d62728",
        alpha=0.9,
        label="-log10(p)",
    )
    ax2.axhline(-np.log10(0.05), color="#7f7f7f", linestyle=":", linewidth=0.8)
    # Axis labels and colors to match series
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CD effect", color="#2ca02c")
    ax.tick_params(axis="y", colors="#2ca02c")
    ax.spines["left"].set_edgecolor("#2ca02c")
    ax2.set_ylabel("-log10(p)", color="#d62728")
    ax2.tick_params(axis="y", colors="#d62728")
    ax2.spines["right"].set_edgecolor("#d62728")
    ax.set_title(f"Decoding {cfg.outcomes[0]} vs {cfg.outcomes[1]} (CD)")
    # Explicit combined legend with the two line handles only
    ax.legend([line_eff, line_p], [line_eff.get_label(), line_p.get_label()], fontsize="small", loc="upper left")
    fig.tight_layout()
    png_path = outp / "decoding_timecourse.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"csv": str(csv_path), "png": str(png_path)}
