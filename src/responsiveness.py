"""Responsive neuron screening utilities.

Computes per-unit responsiveness to an event (default Change_ON) using
paired baseline vs response windows on a per-trial basis. Supports
per-outcome analysis and restriction to kept units from unit selection.

Outputs: a tidy DataFrame with per-unit statistics and optional figures.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from src import align as align_mod
from src import su_analysis as su


@dataclass
class RespConfig:
    event_name: str = "Change_ON"
    base_win: Tuple[float, float] = (-0.2, 0.0)
    resp_win: Tuple[float, float] = (0.0, 0.2)
    bin_size: float = 0.01
    per_outcome: bool = True
    kept_only: bool = True
    min_trials: int = 5
    n_perm: int = 500
    smooth_sigma: Optional[float] = None


def _fr_per_trial(spike_times: np.ndarray, event_times: List[float], window: Tuple[float, float], bin_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-trial mean FR within window using binned alignment.

    Uses align_spikes_to_events (counts/bin) and averages across bins in the window.
    Returns (fr_trials, bin_centers) where fr_trials has shape (n_trials,).
    """
    M, bin_centers = align_mod.align_spikes_to_events(spike_times, event_times, window=window, bin_size=bin_size)
    if M.size == 0:
        return np.zeros(0, dtype=float), bin_centers
    # M already in Hz (counts/bin_size); average across time bins
    fr = np.nanmean(M, axis=1)  # mean FR per trial over the window
    return fr, bin_centers


def _paired_perm_p(diff: np.ndarray, n_perm: int = 1000, rng: Optional[np.random.Generator] = None) -> float:
    """Two-sided sign-flip permutation test on paired differences.

    diff: per-trial differences (resp - base). Under H0, sign of each diff is symmetric.
    Returns p-value.
    """
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return 1.0
    if rng is None:
        rng = np.random.default_rng()
    obs = float(np.nanmean(x))
    if n_perm <= 0:
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * x).mean(axis=1)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return p


def _effect_sizes(base_fr: np.ndarray, resp_fr: np.ndarray) -> Dict[str, float]:
    base = np.asarray(base_fr, dtype=float)
    resp = np.asarray(resp_fr, dtype=float)
    mask = np.isfinite(base) & np.isfinite(resp)
    base = base[mask]
    resp = resp[mask]
    if base.size == 0 or resp.size == 0:
        return {"delta_fr": np.nan, "dprime": np.nan, "auc": np.nan}
    diff = resp - base
    delta = float(np.nanmean(diff))
    sd = float(np.nanstd(diff, ddof=1))
    dprime = delta / sd if sd > 0 else np.nan
    try:
        # Treat as independent for AUC approximation
        u, _ = mannwhitneyu(resp, base, alternative="two-sided")
        auc = float(u / (resp.size * base.size)) if resp.size > 0 and base.size > 0 else np.nan
    except Exception:
        auc = np.nan
    return {"delta_fr": delta, "dprime": dprime, "auc": auc}


def compute_responsiveness_table(
    session,
    cfg: RespConfig,
    selection_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Compute per-unit responsiveness table for a session.

    Returns a DataFrame with rows per unit (and per outcome if cfg.per_outcome).
    Columns include cluster_id, outcome, n_trials, delta_fr, dprime, auc, p_value, is_responsive.
    """
    # Determine cluster set
    kept_ids: Optional[List[int]] = None
    if cfg.kept_only:
        kept_ids = su.load_kept_ids(session, selection_csv)
    cluster_ids = [int(c.cluster_id) for c in session.clusters if (kept_ids is None or int(c.cluster_id) in kept_ids)]

    # Build event times by trial and outcome labels
    ets_all = align_mod.get_event_times_by_trial(session, cfg.event_name)
    trials = getattr(session, "trials", []) or []
    outcomes = [getattr(t, "trialoutcome", None) for t in trials]

    # Prepare outcome groupings
    def _indices_for_outcome(name: Optional[str]) -> List[int]:
        if name is None:
            return [i for i, et in enumerate(ets_all) if np.isfinite(et)]
        else:
            return [i for i, t in enumerate(trials) if getattr(t, "trialoutcome", None) == name and np.isfinite(ets_all[i])]

    outcome_levels: List[Optional[str]]
    if cfg.per_outcome:
        unique = sorted({o for o in outcomes if o is not None})
        outcome_levels = unique
    else:
        outcome_levels = [None]  # pooled

    rows = []
    rng = np.random.default_rng(12345)
    for cid in cluster_ids:
        # Find cluster object
        c = next((x for x in session.clusters if int(x.cluster_id) == int(cid)), None)
        if c is None:
            continue
        st = np.asarray(c.spike_times).flatten()
        # Evaluate each outcome level
        for out in outcome_levels:
            idxs = _indices_for_outcome(out)
            if len(idxs) < cfg.min_trials:
                rows.append({
                    "cluster_id": cid,
                    "outcome": out or "All",
                    "n_trials": len(idxs),
                    "delta_fr": np.nan,
                    "dprime": np.nan,
                    "auc": np.nan,
                    "p_value": np.nan,
                    "is_responsive": False,
                })
                continue
            ets_sel = [float(ets_all[i]) for i in idxs]
            base_fr, _ = _fr_per_trial(st, ets_sel, cfg.base_win, cfg.bin_size)
            resp_fr, _ = _fr_per_trial(st, ets_sel, cfg.resp_win, cfg.bin_size)
            # Paired test via sign-flip permutation on differences
            diff = resp_fr - base_fr
            p = _paired_perm_p(diff, n_perm=cfg.n_perm, rng=rng)
            eff = _effect_sizes(base_fr, resp_fr)
            is_resp = bool((p < 0.05) and np.isfinite(eff["dprime"]))
            rows.append({
                "cluster_id": cid,
                "outcome": out or "All",
                "n_trials": len(idxs),
                **eff,
                "p_value": float(p),
                "is_responsive": is_resp,
            })

    df = pd.DataFrame(rows)
    # Sort for readability
    df = df.sort_values(["cluster_id", "outcome"]).reset_index(drop=True)
    return df


def run_responsiveness(
    session,
    out_dir: str,
    cfg: Optional[RespConfig] = None,
    selection_csv: Optional[str] = None,
    make_plots: bool = True,
) -> Dict[str, str]:
    """Compute responsiveness and save CSV and optional plots.

    Returns paths dict.
    """
    if cfg is None:
        cfg = RespConfig()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    df = compute_responsiveness_table(session, cfg, selection_csv)
    csv_path = outp / "unit_responsive.csv"
    df.to_csv(csv_path, index=False)

    paths: Dict[str, str] = {"csv": str(csv_path)}

    if make_plots:
        # Distribution of delta FR (pooled outcomes)
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.hist(df["delta_fr"].dropna().values, bins=40, color="#4C78A8", alpha=0.8)
            ax.set_xlabel("ΔFR (Hz)")
            ax.set_ylabel("Units")
            ax.set_title("Distribution of ΔFR")
            fig.tight_layout()
            p = outp / "delta_fr_hist.png"
            fig.savefig(p, dpi=140, bbox_inches="tight")
            plt.close(fig)
            paths["delta_hist"] = str(p)
        except Exception:
            pass

        # Volcano: effect vs significance (use d' vs -log10 p)
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            x = df["dprime"].astype(float).values
            pvals = df["p_value"].astype(float).values
            y = -np.log10(np.clip(pvals, 1e-12, 1.0))
            colors = np.where(df["is_responsive"].astype(bool).values, "#d62728", "#7f7f7f")
            ax.scatter(x, y, s=16, c=colors, alpha=0.8, edgecolors="none")
            ax.axhline(-np.log10(0.05), color="k", linestyle="--", linewidth=0.8)
            ax.set_xlabel("d'")
            ax.set_ylabel("-log10(p)")
            ax.set_title("Responsiveness volcano")
            fig.tight_layout()
            p = outp / "volcano_responsiveness.png"
            fig.savefig(p, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths["volcano"] = str(p)
        except Exception:
            pass

    return paths
