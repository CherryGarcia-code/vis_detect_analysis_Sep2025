"""Synthetic session generator for quick progress bar testing.

Creates a lightweight Session-like object using the existing dataclasses
from `visdetect.session` so analysis functions can run without large data.

Usage:
    from visdetect.utils.synthetic import make_synthetic_session
    sess = make_synthetic_session(n_trials=40, n_clusters=25)

The synthetic data mimics baseline / change events and includes spike trains
with modest firing rates plus occasional post-change bursts.
"""
from __future__ import annotations

import numpy as np
from typing import List
from visdetect.core.session import Session, Trial, Cluster


def _rand_spike_train(t_start: float, t_end: float, rate_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Poisson spikes between t_start and t_end at given rate."""
    dur = max(0.0, t_end - t_start)
    expected = dur * rate_hz
    n = rng.poisson(expected)
    if n <= 0:
        return np.empty(0, dtype=float)
    return np.sort(rng.uniform(t_start, t_end, size=n).astype(float))


def make_synthetic_session(n_trials: int = 50, n_clusters: int = 20, seed: int = 0) -> Session:
    rng = np.random.default_rng(seed)
    trials: List[Trial] = []
    baseline_on: List[float] = []
    change_on: List[float] = []

    # Trial timing parameters
    base_interval = 3.5  # seconds per trial start spacing
    min_change_delay = 2.0
    for i in range(n_trials):
        t0 = i * base_interval
        baseline_on.append(t0)
        # variable change time after baseline
        change_t_rel = min_change_delay + rng.uniform(0.3, 0.8)
        change_on.append(t0 + change_t_rel)
        # outcome assignment
        outcome = rng.choice(["Hit", "Miss", "FA"], p=[0.5, 0.3, 0.2])
        rts = {}
        if outcome == "Hit":
            rts["RT"] = rng.uniform(0.25, 0.55)
        elif outcome == "Miss":
            rts["Miss"] = rng.uniform(0.6, 1.0)
        elif outcome == "FA":
            rts["FA"] = rng.uniform(3.2, 4.0)  # long baseline lick latency
        baseline_vec = (rng.random(60) * 40.0) + 1.0  # synthetic TF vector
        trials.append(
            Trial(
                trialoutcome=outcome,
                reactiontimes=rts,
                change_size=rng.choice([1.25, 1.35, 1.5, 2.0]),
                orientation=None,
                ITI=rng.uniform(1.0, 2.0),
                change_time=change_t_rel,
                baseline_values=baseline_vec,
            )
        )

    clusters: List[Cluster] = []
    # Build spikes: baseline low rate + optional post-change / lick bursts
    session_end = change_on[-1] + 5.0
    for cid in range(n_clusters):
        spikes_all = []
        base_rate = rng.uniform(2.0, 6.0)  # Hz
        burst_rate = base_rate * rng.uniform(2.0, 4.0)
        for t_base, t_change in zip(baseline_on, change_on):
            # baseline activity
            spikes_all.append(_rand_spike_train(t_base, t_change, base_rate, rng))
            # possibility of post-change burst if cluster is pseudo TF-responsive
            if rng.random() < 0.35:  # 35% chance of responsiveness episode
                burst_end = t_change + rng.uniform(0.25, 0.35)
                spikes_all.append(_rand_spike_train(t_change, burst_end, burst_rate, rng))
            # occasional false alarm lick-aligned burst for FA trials
            # approximate absolute lick time (baseline + FA latency)
        spikes_concat = np.concatenate(spikes_all) if spikes_all else np.empty(0, dtype=float)
        clusters.append(Cluster(cluster_id=cid, spike_times=np.sort(spikes_concat), quality=None))

    ni_events = {
        "Baseline_ON": np.array(baseline_on, dtype=float),
        "Change_ON": np.array(change_on, dtype=float),
        "session_name": "SYNTHETIC",
    }
    sess = Session(
        trials=trials,
        clusters=clusters,
        subject="SYN",
        session_name="SIM",
        good_cluster_ids=[c.cluster_id for c in clusters],
        ni_events=ni_events,
    )
    return sess


__all__ = ["make_synthetic_session"]