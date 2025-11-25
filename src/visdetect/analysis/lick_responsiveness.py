"""Lick-aligned responsiveness analysis.

Defines lick-responsive units using a paired pre- vs post-lick comparison with
trial-wise post windows truncated at the next lick onset minus a small buffer.

Outputs a per-unit table analogous to src/responsiveness.py but tailored to lick events.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis.su_analysis import load_kept_ids
from visdetect.analysis import align as align_mod
from visdetect.utils.progress import Progress, progress_iter


@dataclass
class LickRespConfig:
    event_name: str = "Lick_L"  # first lick per trial
    base_win: Tuple[float, float] = (-0.2, 0.0)  # relative to first lick
    post_end: float = 0.2  # nominal max post window length (s)
    min_post: float = 0.05  # minimum post window length required to keep a trial
    buffer: float = 0.03  # subtract from next-lick latency to avoid preparatory activity
    truncate_by_next_lick: bool = True
    kept_only: bool = True
    min_trials: int = 5
    n_perm: int = 500


def _paired_perm_p(diff: np.ndarray, n_perm: int = 1000, rng: Optional[np.random.Generator] = None) -> float:
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
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def _effect_stats(base_fr: np.ndarray, post_fr: np.ndarray) -> Dict[str, float]:
    base = np.asarray(base_fr, dtype=float)
    post = np.asarray(post_fr, dtype=float)
    mask = np.isfinite(base) & np.isfinite(post)
    base = base[mask]
    post = post[mask]
    if base.size == 0 or post.size == 0:
        return {"delta_fr": np.nan}
    diff = post - base
    return {"delta_fr": float(np.nanmean(diff))}


def _first_and_next_lick_times(session, event_name: str) -> Tuple[List[float], np.ndarray]:
    """Return per-trial first lick times and the full sorted lick time array.

    For next-lick lookup we use the global lick time vector from ni_events.
    """
    lick_by_trial = align_mod.get_event_times_by_trial(session, event_name)
    ni_events = getattr(session, "ni_events", {}) or {}
    raw = ni_events.get(event_name, [])
    if isinstance(raw, dict) and "rise_t" in raw:
        all_licks = np.array(raw["rise_t"], dtype=float).flatten()
    else:
        all_licks = np.array(raw, dtype=float).flatten()
    all_licks = all_licks[np.isfinite(all_licks)]
    all_licks.sort()
    return lick_by_trial, all_licks


def _fr_in_window(st_sorted: np.ndarray, start: float, end: float) -> float:
    if end <= start:
        return np.nan
    # spike rate within [start, end]
    cnt = np.sum((st_sorted >= start) & (st_sorted < end))
    return float(cnt) / float(end - start)


def compute_lick_responsiveness_table(session, cfg: LickRespConfig, selection_csv: Optional[str] = None, show_progress: bool = False) -> pd.DataFrame:
    kept_ids = load_kept_ids(session, selection_csv) if cfg.kept_only else None
    cluster_ids = [int(c.cluster_id) for c in session.clusters if (kept_ids is None or int(c.cluster_id) in kept_ids)]

    lick_by_trial, all_licks = _first_and_next_lick_times(session, cfg.event_name)
    trials = getattr(session, "trials", []) or []
    outcomes = [getattr(t, "trialoutcome", None) for t in trials]

    # indices with a valid first lick time
    valid_trial_idxs = [i for i, t in enumerate(lick_by_trial) if np.isfinite(t)]
    rows = []
    rng = np.random.default_rng(12345)

    prog = Progress("Lick responsiveness (clusters)", total=len(cluster_ids)) if show_progress else None
    if prog:
        prog.start()
    for idx_c, cid in enumerate(cluster_ids, 1):
        c = next((x for x in session.clusters if int(x.cluster_id) == int(cid)), None)
        if c is None:
            continue
        st = np.asarray(c.spike_times, dtype=float).flatten()
        st.sort()

        # pooled and per-outcome rows
        outcome_levels: List[Optional[str]] = [None]  # pooled
        unique_out = sorted({o for o in outcomes if o is not None})
        outcome_levels.extend(unique_out)

        for out in outcome_levels:
            if out is None:
                idxs = valid_trial_idxs
            else:
                idxs = [i for i in valid_trial_idxs if outcomes[i] == out]
            if len(idxs) < cfg.min_trials:
                rows.append({
                    "cluster_id": cid,
                    "outcome": out or "All",
                    "n_trials": len(idxs),
                    "delta_fr": np.nan,
                    "p_value": np.nan,
                    "is_responsive": False,
                })
                continue

            base_vals: List[float] = []
            post_vals: List[float] = []
            for i in idxs:
                t0 = float(lick_by_trial[i])
                # Determine post window end per trial
                end_rel = cfg.post_end
                if cfg.truncate_by_next_lick and all_licks.size > 0:
                    j = int(np.searchsorted(all_licks, t0, side="right"))
                    if j < all_licks.size:
                        next_dt = float(all_licks[j] - t0) - cfg.buffer
                        if np.isfinite(next_dt):
                            end_rel = max(0.0, min(cfg.post_end, next_dt))
                # Skip trials with too-short post window
                if end_rel < cfg.min_post:
                    continue
                # Compute FRs
                base_start = t0 + cfg.base_win[0]
                base_end = t0 + cfg.base_win[1]
                post_start = t0 + 0.0
                post_end = t0 + end_rel
                base_vals.append(_fr_in_window(st, base_start, base_end))
                post_vals.append(_fr_in_window(st, post_start, post_end))

            base_arr = np.array(base_vals, dtype=float)
            post_arr = np.array(post_vals, dtype=float)
            mask = np.isfinite(base_arr) & np.isfinite(post_arr)
            base_arr = base_arr[mask]
            post_arr = post_arr[mask]
            if base_arr.size < cfg.min_trials or post_arr.size < cfg.min_trials:
                rows.append({
                    "cluster_id": cid,
                    "outcome": out or "All",
                    "n_trials": int(min(base_arr.size, post_arr.size)),
                    "delta_fr": np.nan,
                    "p_value": np.nan,
                    "is_responsive": False,
                })
                continue

            diff = post_arr - base_arr
            p = _paired_perm_p(diff, n_perm=cfg.n_perm, rng=rng)
            eff = _effect_stats(base_arr, post_arr)
            is_resp = bool((p < 0.05) and np.isfinite(eff["delta_fr"]))
            rows.append({
                "cluster_id": cid,
                "outcome": out or "All",
                "n_trials": int(min(base_arr.size, post_arr.size)),
                **eff,
                "p_value": float(p),
                "is_responsive": is_resp,
            })
        # progress update per cluster
        if prog:
            prog.update(idx_c)

    df = pd.DataFrame(rows).sort_values(["cluster_id", "outcome"]).reset_index(drop=True)
    if prog:
        prog.close()
    return df


def run_lick_responsiveness(session, out_dir: str, cfg: Optional[LickRespConfig] = None, selection_csv: Optional[str] = None, make_plots: bool = True, show_progress: bool = False) -> Dict[str, str]:
    if cfg is None:
        cfg = LickRespConfig()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    df = compute_lick_responsiveness_table(session, cfg, selection_csv, show_progress=show_progress)
    csv_path = outp / "unit_lick_responsive.csv"
    df.to_csv(csv_path, index=False)

    paths: Dict[str, str] = {"csv": str(csv_path)}

    if make_plots:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.hist(df["delta_fr"].dropna().values, bins=40, color="#4C78A8", alpha=0.85)
            ax.set_xlabel("ΔFR post−pre (Hz)")
            ax.set_ylabel("Units")
            ax.set_title("Lick responsiveness: ΔFR distribution")
            fig.tight_layout()
            p = outp / "lick_delta_fr_hist.png"
            fig.savefig(p, dpi=140, bbox_inches="tight")
            plt.close(fig)
            paths["delta_hist"] = str(p)
        except Exception:
            pass
    return paths
