"""First-lick aligned population decoding (coding direction).

Align trials to the first left-lick NI event (default 'Lick_L'),
label trials into categories:
  - 'Hit' (trialoutcome == 'Hit')
  - 'FA_early' (trialoutcome == 'FA' and lick latency from baseline <= threshold)
  - 'FA_late' (trialoutcome == 'FA' and lick latency from baseline > threshold)

Runs time-resolved coding-direction decoding pairwise between present classes
and writes per-comparison CSVs and plots.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
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
class LickCDConfig:
    event_name: str = "Lick_L"  # NI key for first left lick per trial
    baseline_key: str = "Baseline_ON"
    window: Tuple[float, float] = (-0.5, 0.5)
    bin_size: float = 0.02
    kept_only: bool = True
    fa_early_threshold: float = 3.0  # seconds from baseline
    min_trials_per_class: int = 8
    method: str = "shrinkage"
    reg: float = 1.0
    n_splits: int = 5
    n_permutations: int = 200
    random_state: int = 0
    # Optional gating by lick responsiveness
    responsive_only: bool = False
    responsiveness_outcome: str = "All"  # which outcome row to use from responsiveness CSV


def _labels_by_trial(session, cfg: LickCDConfig) -> Tuple[List[float], List[str]]:
    """Return per-trial lick times and class labels.

    Returns (lick_times_by_trial, labels) where labels in {'Hit','FA_early','FA_late', None}
    """
    n_trials = len(getattr(session, "trials", []) or [])
    if n_trials == 0:
        return [], []
    lick_by_trial = align_mod.get_event_times_by_trial(session, cfg.event_name)
    base_by_trial = align_mod.get_event_times_by_trial(session, cfg.baseline_key)
    labels: List[Optional[str]] = []
    lick_times: List[float] = []
    trials = getattr(session, "trials", []) or []
    for i in range(n_trials):
        try:
            lt = float(lick_by_trial[i])
        except Exception:
            lt = np.nan
        if not np.isfinite(lt):
            labels.append(None)
            lick_times.append(np.nan)
            continue
        out = getattr(trials[i], "trialoutcome", None)
        # compute latency from baseline
        bt = base_by_trial[i] if i < len(base_by_trial) else np.nan
        try:
            bt = float(bt)
        except Exception:
            bt = np.nan
        latency = float(lt - bt) if (np.isfinite(lt) and np.isfinite(bt)) else np.nan
        lab: Optional[str]
        if out == "Hit":
            lab = "Hit"
        elif out == "FA":
            if np.isfinite(latency) and latency <= cfg.fa_early_threshold:
                lab = "FA_early"
            else:
                lab = "FA_late"
        else:
            lab = None
        labels.append(lab)
        lick_times.append(lt)
    return lick_times, [l if l is not None else "" for l in labels]


def _build_population(session, cfg: LickCDConfig, selection_csv: Optional[str], responsiveness_csv: Optional[str], classes: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pop, y, bin_centers) for two classes.
    pop: trials x bins x units, y in {0,1} with y=1 for classes[0].
    """
    lick_times, labels = _labels_by_trial(session, cfg)
    # Select trials in the two classes
    cls_a, cls_b = classes
    idxs = [i for i, lab in enumerate(labels) if lab in (cls_a, cls_b) and np.isfinite(lick_times[i])]
    if len(idxs) < 2 * cfg.min_trials_per_class:
        raise ValueError("Insufficient trials across classes")
    ets = [lick_times[i] for i in idxs]
    y = np.array([1 if labels[i] == cls_a else 0 for i in idxs], dtype=int)

    # Determine cluster set
    cluster_ids = [int(c.cluster_id) for c in session.clusters]
    if cfg.kept_only:
        kept = set(load_kept_ids(session, selection_csv))
        if kept:
            cluster_ids = [cid for cid in cluster_ids if cid in kept]
    # Filter to lick-responsive units if requested and CSV provided
    if cfg.responsive_only and responsiveness_csv:
        try:
            df_resp = pd.read_csv(responsiveness_csv)
            col_out = df_resp.get("outcome")
            if col_out is None:
                mask = df_resp["is_responsive"].astype(bool).values
                resp_ids = set(df_resp.loc[mask, "cluster_id"].astype(int).tolist())
            else:
                resp_ids = set(
                    df_resp.loc[(df_resp["outcome"] == cfg.responsiveness_outcome) & (df_resp["is_responsive"].astype(bool)), "cluster_id"].astype(int).tolist()
                )
            if resp_ids:
                cluster_ids = [cid for cid in cluster_ids if cid in resp_ids]
        except Exception:
            pass
    if not cluster_ids:
        raise ValueError("No clusters available")

    # Bins
    _, bc = align_mod.align_spikes_to_events(np.array([]), ets, window=cfg.window, bin_size=cfg.bin_size)

    units = []
    min_trials = None
    for c in session.clusters:
        cid = int(c.cluster_id)
        if cid not in cluster_ids:
            continue
        M, _ = align_mod.align_spikes_to_events(c.spike_times, ets, window=cfg.window, bin_size=cfg.bin_size)
        units.append(M)
        if min_trials is None:
            min_trials = M.shape[0]
        else:
            min_trials = min(min_trials, M.shape[0])
    if not units or min_trials is None or min_trials == 0:
        raise ValueError("No aligned unit matrices")
    units_trunc = [u[:min_trials, :] for u in units]
    pop = np.stack(units_trunc, axis=2)
    y = y[:min_trials]
    return pop, y, bc


def _plot_timecourse(bc, effect, pvals, title: str, out_png: Path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    line_eff, = ax.plot(bc, effect, color="#2ca02c", label="Class1 − Class0")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax2 = ax.twinx()
    import numpy as _np
    line_p, = ax2.plot(bc, -_np.log10(_np.clip(pvals, 1e-12, 1.0)), color="#d62728", alpha=0.9, label="-log10(p)")
    ax2.axhline(-_np.log10(0.05), color="#7f7f7f", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CD effect", color="#2ca02c"); ax.tick_params(axis="y", colors="#2ca02c"); ax.spines["left"].set_edgecolor("#2ca02c")
    ax2.set_ylabel("-log10(p)", color="#d62728"); ax2.tick_params(axis="y", colors="#d62728"); ax2.spines["right"].set_edgecolor("#d62728")
    ax.set_title(title)
    ax.legend([line_eff, line_p], [line_eff.get_label(), line_p.get_label()], fontsize="small", loc="upper left")
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def run_lick_decoding(session, out_root: str, png_root: str, cfg: Optional[LickCDConfig] = None, selection_csv: Optional[str] = None, responsiveness_csv: Optional[str] = None) -> Dict[str, List[str]]:
    if cfg is None:
        cfg = LickCDConfig()
    outp_root = Path(out_root) / f"{getattr(session,'subject','unknown')}_{getattr(session,'session_name','unknown')}"
    pngp_root = Path(png_root) / outp_root.name
    outp_root.mkdir(parents=True, exist_ok=True)

    # Determine which classes are present
    lick_times, labels = _labels_by_trial(session, cfg)
    classes_present = sorted({lab for lab in labels if lab})
    # Only keep among our target set
    target = ["Hit", "FA_early", "FA_late"]
    classes_present = [c for c in target if c in classes_present]
    if len(classes_present) < 2:
        return {"skipped": ["<2 classes present>"]}

    results: Dict[str, List[str]] = {}
    pairs = [(a, b) for i, a in enumerate(classes_present) for b in classes_present[i+1:]]
    for a, b in pairs:
        try:
            pop, y, bc = _build_population(session, cfg, selection_csv, responsiveness_csv, (a, b))
            res = time_resolved_cd(
                pop,
                cond_mask=(y == 1),
                method=cfg.method,
                reg=cfg.reg,
                n_splits=cfg.n_splits,
                n_permutations=cfg.n_permutations,
                random_state=cfg.random_state,
            )
            df = pd.DataFrame({"time": bc, "effect": res["effect"], "p_value": res["pvals"]})
            out_dir = outp_root / f"{a}_vs_{b}"
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "lick_cd_timecourse.csv"
            df.to_csv(csv_path, index=False)
            png_dir = pngp_root / f"{a}_vs_{b}"
            png = png_dir / "lick_cd_timecourse.png"
            _plot_timecourse(bc, res["effect"], res["pvals"], f"Lick CD: {a} vs {b}", png)
            results[f"{a}_vs_{b}"] = [str(csv_path), str(png)]
        except Exception as e:
            results[f"{a}_vs_{b}"] = [f"ERROR: {e}"]
    return results
