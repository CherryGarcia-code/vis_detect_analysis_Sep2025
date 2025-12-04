"""Archived MATLAB-port TF pulse analyzer (moved from src/visdetect/utils/matlab_ports/tf_pulse.py).

Kept for provenance; the project now standardizes on
`src/visdetect/analysis/tf_pulse.py` as the canonical TF-pulse implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage, stats

from visdetect.core.legacy_io import Session, load_session


@dataclass
class TFPulseConfig:
    pre_event_window: float = 0.5
    post_event_window: float = 1.0
    bin_size: float = 0.001
    smooth_bins: int = 50
    post_mean_window: float = 0.5  # seconds after pulse for rate comparison
    frame_interval: float = 1 / 60.0
    pad_baseline_start: float = 1.0  # drop first second of baseline
    pad_before_action: float = 0.2  # drop final 200 ms before FA/abort
    fast_threshold: float = 1.2  # TF (Hz) considered "fast"
    slow_threshold: float = 0.8  # TF (Hz) considered "slow"
    min_event_gap: float = 0.05
    min_events: int = 8

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

    @property
    def post_window_indices(self) -> Tuple[int, int]:
        start = int(round(self.pre_event_window / self.bin_size))
        end = int(round((self.pre_event_window + self.post_mean_window) / self.bin_size))
        return start, end


@dataclass
class TFPulseResult:
    table: pd.DataFrame
    time_axis: np.ndarray
    fast_mean: np.ndarray
    fast_sem: np.ndarray
    slow_mean: np.ndarray
    slow_sem: np.ndarray
    n_fast_events: int
    n_slow_events: int


class TFPulseAnalyzer:
    def __init__(self, cfg: Optional[TFPulseConfig] = None, good_ids: Optional[Iterable[int]] = None):
        self.cfg = cfg or TFPulseConfig()
        self.good_ids = set(int(x) for x in good_ids) if good_ids is not None else None

    # ------------------------------------------------------------------
    def run_session(self, session: Session) -> TFPulseResult:
        fast_events, slow_events = self._collect_events(session)
        edges = self.cfg.time_edges

        cluster_rows: List[Dict[str, float]] = []
        fast_psths: List[np.ndarray] = []
        slow_psths: List[np.ndarray] = []

        for cl in session.clusters:
            cid = int(cl.cluster_id)
            if self.good_ids is not None and cid not in self.good_ids:
                continue
            spikes = np.asarray(cl.spike_times, dtype=float)
            spikes = spikes[np.isfinite(spikes)]
            if spikes.size == 0:
                continue
            fast_matrix = self._build_psth_matrix(spikes, fast_events, edges)
            slow_matrix = self._build_psth_matrix(spikes, slow_events, edges)
            if fast_matrix is None or slow_matrix is None:
                continue
            if fast_matrix.shape[0] < self.cfg.min_events or slow_matrix.shape[0] < self.cfg.min_events:
                continue

            row = self._compute_stats(cid, fast_matrix, slow_matrix)
            cluster_rows.append(row)
            fast_psths.append(self._smooth_trials(fast_matrix))
            slow_psths.append(self._smooth_trials(slow_matrix))

        table = pd.DataFrame(cluster_rows).sort_values("cluster_id").reset_index(drop=True)

        if fast_psths:
            fast_stack = np.stack(fast_psths)
            fast_mean = fast_stack.mean(axis=0)
            if fast_stack.shape[0] > 1:
                fast_sem = fast_stack.std(axis=0, ddof=1) / np.sqrt(fast_stack.shape[0])
            else:
                fast_sem = np.zeros_like(fast_mean)
        else:
            fast_mean = np.zeros(self.cfg.time_centers.size)
            fast_sem = np.zeros_like(fast_mean)

        if slow_psths:
            slow_stack = np.stack(slow_psths)
            slow_mean = slow_stack.mean(axis=0)
            if slow_stack.shape[0] > 1:
                slow_sem = slow_stack.std(axis=0, ddof=1) / np.sqrt(slow_stack.shape[0])
            else:
                slow_sem = np.zeros_like(slow_mean)
        else:
            slow_mean = np.zeros(self.cfg.time_centers.size)
            slow_sem = np.zeros_like(slow_mean)

        return TFPulseResult(
            table=table,
            time_axis=self.cfg.time_centers,
            fast_mean=fast_mean,
            fast_sem=fast_sem,
            slow_mean=slow_mean,
            slow_sem=slow_sem,
            n_fast_events=len(fast_events),
            n_slow_events=len(slow_events),
        )

    # ------------------------------------------------------------------
    def _collect_events(self, session: Session) -> Tuple[np.ndarray, np.ndarray]:
        baseline_on = np.asarray(session.ni_events.get("Baseline_ON", []), dtype=float)
        change_on = np.asarray(session.ni_events.get("Change_ON", []), dtype=float)
        n_trials = min(len(session.trials), baseline_on.size)
        fast_times: List[float] = []
        slow_times: List[float] = []

        for idx in range(n_trials):
            trial = session.trials[idx]
            tf_vec = self._clean_tf_vector(trial.baseline_values)
            if tf_vec is None:
                continue
            base_t = baseline_on[idx]
            change_t = change_on[idx] if idx < change_on.size else np.nan
            ev_fast, ev_slow = self._extract_trial_events(trial, tf_vec, base_t, change_t)
            fast_times.extend(ev_fast)
            slow_times.extend(ev_slow)

        return np.array(fast_times, dtype=float), np.array(slow_times, dtype=float)

    def _clean_tf_vector(self, vec: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return arr

    def _extract_trial_events(
        self,
        trial,
        tf_vec: np.ndarray,
        base_time: float,
        change_time: float,
    ) -> Tuple[List[float], List[float]]:
        cfg = self.cfg
        times_rel = np.arange(tf_vec.size) * cfg.frame_interval
        rel_stop = change_time - base_time - cfg.post_event_window if np.isfinite(change_time) else np.inf

        outcome = (trial.trialoutcome or "").lower()
        rt = None
        if outcome == "fa":
            rt = trial.reactiontimes.get("FA") if trial.reactiontimes else None
        elif outcome == "abort":
            rt = trial.reactiontimes.get("abort") if trial.reactiontimes else None
        if rt is not None and np.isfinite(rt):
            rel_stop = min(rel_stop, rt - cfg.pad_before_action)
        rel_stop = max(rel_stop, cfg.pad_baseline_start)

        mask = (times_rel >= cfg.pad_baseline_start) & (times_rel <= rel_stop)
        if not np.any(mask):
            return [], []
        times_rel = times_rel[mask]
        tf_rel = tf_vec[mask]

        fast_indices = self._segment_starts(tf_rel >= cfg.fast_threshold)
        slow_indices = self._segment_starts(tf_rel <= cfg.slow_threshold)

        fast_times = self._index_to_times(times_rel, fast_indices, base_time)
        slow_times = self._index_to_times(times_rel, slow_indices, base_time)

        fast_times = self._apply_gap(fast_times, cfg.min_event_gap)
        slow_times = self._apply_gap(slow_times, cfg.min_event_gap)
        return fast_times, slow_times

    def _segment_starts(self, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return np.array([], dtype=int)
        starts = mask & np.concatenate(([True], ~mask[:-1]))
        return np.flatnonzero(starts)

    def _index_to_times(self, times_rel: np.ndarray, indices: np.ndarray, base_time: float) -> List[float]:
        if indices.size == 0:
            return []
        subset = times_rel[indices]
        return [float(base_time + t) for t in subset]

    def _apply_gap(self, times: List[float], min_gap: float) -> List[float]:
        if not times:
            return []
        kept: List[float] = []
        last = -np.inf
        for t in times:
            if t - last >= min_gap:
                kept.append(t)
                last = t
        return kept

    def _build_psth_matrix(
        self,
        spikes: np.ndarray,
        events: np.ndarray,
        bin_edges: np.ndarray,
    ) -> Optional[np.ndarray]:
        if events.size == 0:
            return None
        window = (bin_edges[0], bin_edges[-1])
        n_bins = bin_edges.size - 1
        matrix = np.zeros((events.size, n_bins), dtype=float)
        for i, et in enumerate(events):
            rel = spikes - float(et)
            mask = (rel >= window[0]) & (rel < window[1])
            if not np.any(mask):
                continue
            counts, _ = np.histogram(rel[mask], bins=bin_edges)
            matrix[i] = counts
        return matrix

    def _smooth_trials(self, matrix: np.ndarray) -> np.ndarray:
        if self.cfg.smooth_bins <= 1:
            return matrix.mean(axis=0)
        sigma = self.cfg.smooth_bins / 6.0
        smoothed = ndimage.gaussian_filter1d(matrix, sigma=sigma, axis=1, mode="nearest")
        return smoothed.mean(axis=0)

    def _compute_stats(self, cid: int, fast_matrix: np.ndarray, slow_matrix: np.ndarray) -> Dict[str, float]:
        start, end = self.cfg.post_window_indices
        fast_rates = fast_matrix[:, start:end].mean(axis=1)
        slow_rates = slow_matrix[:, start:end].mean(axis=1)
        _, p_val = stats.ttest_ind(fast_rates, slow_rates, equal_var=False, nan_policy="omit")
        fast_mean_rate = float(np.nanmean(fast_rates))
        slow_mean_rate = float(np.nanmean(slow_rates))
        return {
            "cluster_id": cid,
            "n_fast_events": int(fast_matrix.shape[0]),
            "n_slow_events": int(slow_matrix.shape[0]),
            "fast_mean": fast_mean_rate,
            "slow_mean": slow_mean_rate,
            "delta_fast_minus_slow": float(fast_mean_rate - slow_mean_rate),
            "p_value": float(p_val),
            "is_tf_responsive": bool(p_val < 0.05),
        }


# ---------------------------------------------------------------------------
def compute_tf_pulse_responsiveness(
    session_or_path: Session | str | Path,
    cfg: Optional[TFPulseConfig] = None,
    good_ids: Optional[Iterable[int]] = None,
) -> TFPulseResult:
    if isinstance(session_or_path, (str, Path)):
        session = load_session(str(session_or_path))
    else:
        session = session_or_path
    ids = good_ids
    if ids is None and hasattr(session, "good_cluster_ids"):
        ids = session.good_cluster_ids
    analyzer = TFPulseAnalyzer(cfg, good_ids=ids)
    return analyzer.run_session(session)
