"""MATLAB-faithful lick responsiveness analysis.

This module mirrors the legacy `find_lick_responses_BG.m` workflow used in the lab
so that we can generate directly comparable metrics from Python. The key design
choices are copied from that script:

* Only first-lick false alarms (FA trials) are considered.
* Licks must occur at least 3 s after `Baseline_ON` to count (late baseline).
* Spike matrices are computed with 1 ms bins over [-2 s, +0.75 s].
* Responsiveness is quantified via an unpaired two-sample t-test comparing
    the preparatory window [-1.75, -1.25] s to the pre-movement window [-0.3, -0.15] s.
* Only "good" clusters (per `session.good_cluster_ids`) are analyzed, matching the MATLAB script.
* PSTHs are smoothed with a Gaussian (window 50 samples) before plotting.

Outputs are returned as a tidy pandas DataFrame and optional PSTH data for
further visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import stats

from visdetect.core.legacy_io import Session, load_session
from visdetect.utils.progress import Progress


@dataclass
class MatlabLickConfig:
    pre_event_window: float = 2.0
    post_event_window: float = 0.75
    bin_size: float = 0.001
    min_fa_delay: float = 3.0
    baseline_window: Tuple[float, float] = (-1.75, -1.25)
    post_window: Tuple[float, float] = (-0.3, -0.15)
    smooth_bins: int = 50
    min_events: int = 5

    @property
    def time_edges(self) -> np.ndarray:
        return np.arange(
            -self.pre_event_window,
            self.post_event_window + self.bin_size,
            self.bin_size,
        )

    @property
    def time_centers(self) -> np.ndarray:
        edges = self.time_edges
        return (edges[:-1] + edges[1:]) * 0.5


@dataclass
class MatlabLickResult:
    table: pd.DataFrame
    psth_mean: np.ndarray
    psth_sem: np.ndarray
    time_axis: np.ndarray


@dataclass
class MatlabLickUnitTrace:
    cluster_id: int
    z_trace: np.ndarray
    sem_trace: np.ndarray
    z_peak: float


class MatlabLickAnalyzer:
    def __init__(self, cfg: Optional[MatlabLickConfig] = None, good_ids: Optional[Iterable[int]] = None):
        self.cfg = cfg or MatlabLickConfig()
        self.good_ids = set(int(x) for x in good_ids) if good_ids is not None else None

    # ------------------------------------------------------------------
    def run_session(self, session: Session) -> MatlabLickResult:
        event_times = self._fa_lick_times(session)
        if len(event_times) < self.cfg.min_events:
            cols = [
                "cluster_id",
                "n_events",
                "baseline_mean",
                "post_mean",
                "delta_mean",
                "p_value",
                "is_significant",
            ]
            return MatlabLickResult(
                table=pd.DataFrame(columns=cols),
                psth_mean=np.empty((0, self.cfg.time_centers.size)),
                psth_sem=np.empty((0, self.cfg.time_centers.size)),
                time_axis=self.cfg.time_centers,
            )

        psth_by_cluster: Dict[int, np.ndarray] = {}
        rows: List[Dict[str, float]] = []
        edges = self.cfg.time_edges

        for cl in session.clusters:
            if self.good_ids is not None and int(cl.cluster_id) not in self.good_ids:
                continue
            spikes = np.asarray(cl.spike_times, dtype=float)
            spikes = spikes[np.isfinite(spikes)]
            if spikes.size == 0:
                continue
            matrix = self._build_psth_matrix(spikes, event_times, edges)
            if matrix is None or matrix.shape[0] < self.cfg.min_events:
                continue
            psth_by_cluster[cl.cluster_id] = self._smooth_trials(matrix)
            row = self._compute_stats(matrix, cl.cluster_id)
            rows.append(row)

        table = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)

        if psth_by_cluster:
            stack = np.stack(list(psth_by_cluster.values()))
            mean_psth = stack.mean(axis=0)
            sem_psth = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
        else:
            mean_psth = np.zeros(self.cfg.time_centers.size)
            sem_psth = np.zeros(self.cfg.time_centers.size)

        return MatlabLickResult(
            table=table,
            psth_mean=mean_psth,
            psth_sem=sem_psth,
            time_axis=self.cfg.time_centers,
        )

    # ------------------------------------------------------------------
    def _fa_lick_times(self, session: Session, show_progress: bool = False) -> List[float]:
        # Be robust to different ni_events formats: sometimes events are stored
        # as a dict with keys like 'rise_t' or 'times' rather than a plain list.
        raw_baseline = session.ni_events.get("Baseline_ON", []) if getattr(session, "ni_events", None) is not None else []
        if isinstance(raw_baseline, dict):
            if "rise_t" in raw_baseline:
                baseline = np.asarray(raw_baseline.get("rise_t", []), dtype=float).flatten()
            elif "times" in raw_baseline:
                baseline = np.asarray(raw_baseline.get("times", []), dtype=float).flatten()
            else:
                baseline = np.asarray([], dtype=float)
        else:
            baseline = np.asarray(raw_baseline, dtype=float).flatten()
        trials = session.trials
        n = min(len(trials), baseline.size) if baseline.size > 0 else 0
        events: List[float] = []
        if show_progress and n:
            print(f"Scanning FA trials for lick events: 0/{n} (0.0%)", flush=True)
            step = max(1, n // 20)
        else:
            step = 0
        for i in range(n):
            trial = trials[i]
            outcome = (trial.trialoutcome or "").lower()
            if outcome != "fa" or n == 0:
                continue
            rt = trial.reactiontimes.get("FA") if trial.reactiontimes else None
            if rt is None or not np.isfinite(rt):
                continue
            delay = float(rt)
            if delay < self.cfg.min_fa_delay:
                continue
            events.append(float(baseline[i] + delay))
            if show_progress and step and ((i + 1) == n or i == 0 or ((i + 1) % step) == 0):
                pct = 100.0 * (i + 1) / n
                print(f"Scanning FA trials for lick events: {i+1}/{n} ({pct:4.1f}%)", flush=True)
        return events

    def _build_psth_matrix(
        self,
        spikes: np.ndarray,
        events: Sequence[float],
        bin_edges: np.ndarray,
    ) -> Optional[np.ndarray]:
        window = (bin_edges[0], bin_edges[-1])
        duration = window[1] - window[0]
        if events is None or len(events) == 0 or duration <= 0:
            return None
        n_bins = bin_edges.size - 1
        matrix = np.zeros((len(events), n_bins), dtype=float)
        for r, et in enumerate(events):
            rel = spikes - float(et)
            mask = (rel >= window[0]) & (rel < window[1])
            if not np.any(mask):
                continue
            counts, _ = np.histogram(rel[mask], bins=bin_edges)
            matrix[r] = counts
        return matrix

    def _window_indices(self, window: Tuple[float, float]) -> Tuple[int, int]:
        start = int(round((window[0] + self.cfg.pre_event_window) / self.cfg.bin_size))
        end = int(round((window[1] + self.cfg.pre_event_window) / self.cfg.bin_size))
        return max(start, 0), max(end, 0)

    def _smooth_trials(self, matrix: np.ndarray) -> np.ndarray:
        if self.cfg.smooth_bins <= 1:
            return matrix.mean(axis=0)
        sigma = self.cfg.smooth_bins / 6.0
        smoothed = ndimage.gaussian_filter1d(matrix, sigma=sigma, axis=1, mode="nearest")
        return smoothed.mean(axis=0)

    def _compute_stats(self, matrix: np.ndarray, cluster_id: int) -> Dict[str, float]:
        base_idx = self._window_indices(self.cfg.baseline_window)
        post_idx = self._window_indices(self.cfg.post_window)
        base = matrix[:, base_idx[0]:base_idx[1]].mean(axis=1)
        post = matrix[:, post_idx[0]:post_idx[1]].mean(axis=1)
        if base.size < self.cfg.min_events or post.size < self.cfg.min_events:
            return {
                "cluster_id": cluster_id,
                "n_events": int(matrix.shape[0]),
                "baseline_mean": np.nan,
                "post_mean": np.nan,
                "delta_mean": np.nan,
                "p_value": np.nan,
                "is_significant": False,
            }
        _, p_val = stats.ttest_ind(post, base, equal_var=False, nan_policy="omit")
        delta = float(np.nanmean(post - base))
        return {
            "cluster_id": cluster_id,
            "n_events": int(matrix.shape[0]),
            "baseline_mean": float(np.nanmean(base)),
            "post_mean": float(np.nanmean(post)),
            "delta_mean": delta,
            "p_value": float(p_val),
            "is_significant": bool(p_val < 0.05),
        }

    def collect_unit_traces(self, session: Session, show_progress: bool = False) -> Tuple[np.ndarray, List[MatlabLickUnitTrace]]:
        event_times = self._fa_lick_times(session, show_progress=show_progress)
        if len(event_times) < self.cfg.min_events:
            return self.cfg.time_centers, []
        t_vec = self.cfg.time_centers
        entries: List[MatlabLickUnitTrace] = []
        clusters: List = []
        for cl in session.clusters:
            cid = int(cl.cluster_id)
            if self.good_ids is not None and cid not in self.good_ids:
                continue
            clusters.append(cl)
        total = len(clusters)
        desc = "Computing lick PSTHs"
        if show_progress and total:
            print(f"{desc}: 0/{total} (0.0%)", flush=True) 
            step = max(1, total // 20)
        else:
            step = 0
        for idx, cl in enumerate(clusters, 1):
            spikes = np.asarray(cl.spike_times, dtype=float)
            spikes = spikes[np.isfinite(spikes)]
            if spikes.size == 0:
                if show_progress and step and (idx == total or idx == 1 or idx % step == 0):
                    pct = 100.0 * idx / total
                    print(f"{desc}: {idx}/{total} ({pct:4.1f}%)", flush=True)
                continue
            matrix = self._build_psth_matrix(spikes, event_times, self.cfg.time_edges)
            if matrix is None or matrix.shape[0] < self.cfg.min_events:
                if show_progress and step and (idx == total or idx == 1 or idx % step == 0):
                    pct = 100.0 * idx / total
                    print(f"{desc}: {idx}/{total} ({pct:4.1f}%)", flush=True)
                continue
            sigma = self.cfg.smooth_bins / 6.0 
            smooth_trials = ndimage.gaussian_filter1d(matrix, sigma=sigma, axis=1, mode="nearest")
            mean_trace = np.nanmean(smooth_trials, axis=0)
            if smooth_trials.shape[0] > 1:
                sem_trace = np.nanstd(smooth_trials, axis=0, ddof=1) / np.sqrt(smooth_trials.shape[0])
            else:
                sem_trace = np.zeros_like(mean_trace)
            z_trace, peak = self._zscore_mean_trace(mean_trace, t_vec)
            entries.append(
                MatlabLickUnitTrace(
                    cluster_id=int(cl.cluster_id),
                    z_trace=z_trace,
                    sem_trace=sem_trace,
                    z_peak=peak,
                )
            )
            if show_progress and step and (idx == total or idx == 1 or idx % step == 0):
                pct = 100.0 * idx / total
                print(f"{desc}: {idx}/{total} ({pct:4.1f}%)", flush=True)
        return t_vec, entries

    def _zscore_mean_trace(self, mean_trace: np.ndarray, t_vec: np.ndarray) -> Tuple[np.ndarray, float]:
        if mean_trace.size == 0:
            return mean_trace, float("nan")
        mask = (t_vec >= self.cfg.baseline_window[0]) & (t_vec < self.cfg.baseline_window[1])
        mu = float(np.nanmean(mean_trace[mask])) if np.any(mask) else 0.0
        sd = float(np.nanstd(mean_trace[mask])) if np.any(mask) else 0.0
        if not np.isfinite(sd) or sd <= 0:
            z = mean_trace * 0.0
            return z, float("nan")
        z = (mean_trace - mu) / sd
        post_mask = (t_vec >= self.cfg.post_window[0]) & (t_vec < self.cfg.post_window[1])
        peak = float(np.nanmax(z[post_mask])) if np.any(post_mask) else float("nan")
        return z, peak


# ---------------------------------------------------------------------------
def compute_fa_lick_responsiveness(
    session_or_path: Session | str | Path,
    cfg: Optional[MatlabLickConfig] = None,
    good_ids: Optional[Iterable[int]] = None,
) -> MatlabLickResult:
    if isinstance(session_or_path, (str, Path)):
        session = load_session(str(session_or_path))
    else:
        session = session_or_path
    ids = good_ids
    if ids is None and hasattr(session, "good_cluster_ids"):
        ids = session.good_cluster_ids
    analyzer = MatlabLickAnalyzer(cfg, good_ids=ids)
    return analyzer.run_session(session)


def collect_fa_lick_traces(
    session_or_path: Session | str | Path,
    cfg: Optional[MatlabLickConfig] = None,
    good_ids: Optional[Iterable[int]] = None,
    show_progress: bool = False,
) -> Tuple[np.ndarray, List[MatlabLickUnitTrace]]:
    if isinstance(session_or_path, (str, Path)):
        session = load_session(str(session_or_path))
    else:
        session = session_or_path
    analyzer = MatlabLickAnalyzer(cfg, good_ids=good_ids)
    return analyzer.collect_unit_traces(session, show_progress=show_progress)


if __name__ == "__main__":
    import argparse
    import sys
    from visdetect.analysis.su_analysis import load_kept_ids, selection_csv_default_path

    parser = argparse.ArgumentParser(description="MATLAB-style FA-lick responsiveness (library CLI)")
    parser.add_argument("--sessions", nargs="+", required=True, help="Session .pkl paths")
    parser.add_argument("--profiles-root", default="table_output/unit_qc", help="Root of unit selection CSVs")
    parser.add_argument("--profile-name", default="striatal_strict", help="QC profile to select kept_only units")
    parser.add_argument("--no-kept-only", action="store_true", help="Disable kept_only gating")
    args = parser.parse_args()

    cfg = MatlabLickConfig()
    for sess in args.sessions:
        kept_ids = None
        if not args.no_kept_only:
            try:
                sel_csv = selection_csv_default_path(load_session(sess), root=args.profiles_root)
                kept_ids = load_kept_ids(load_session(sess), selection_csv=str(sel_csv))
            except Exception as exc:
                print(f"[warn] Could not load kept_only for {sess}: {exc}. Using session.good_cluster_ids.", file=sys.stderr)
                kept_ids = None
        res = compute_fa_lick_responsiveness(sess, cfg=cfg, good_ids=kept_ids)
        n_sig = int(res.table["is_significant"].sum()) if not res.table.empty else 0
        n_ev = int(res.table["n_events"].max()) if not res.table.empty else 0
        print(f"{sess}: units={len(res.table)} sig={n_sig} n_events={n_ev}")
