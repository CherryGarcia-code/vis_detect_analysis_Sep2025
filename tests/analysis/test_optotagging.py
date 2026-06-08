# tests/analysis/test_optotagging.py
import numpy as np
import pytest
from visdetect.core.session import Session, Cluster
from visdetect.analysis import optotagging as ot


# ── shared synthetic builders ────────────────────────────────────────
def _pulses(n=501, spacing=1.0, t0=10.0):
    return t0 + np.arange(n) * spacing


def _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.2, base_rate=5.0,
                     collision=True, respond=True, seed=0):
    """Baseline Poisson + a locked post-pulse spike.

    If collision=True, the locked spike is SUPPRESSED on any pulse that already
    has a spontaneous spike within (latency+1 ms) before the pulse (true antidromic).
    If respond=False, never add the locked spike (pure baseline unit).
    """
    rng = np.random.default_rng(seed)
    t_end = pulses[-1] + 2.0
    n_base = rng.poisson(base_rate * t_end)
    spikes = list(rng.uniform(0, t_end, size=n_base))
    spikes_arr = np.sort(np.asarray(spikes))
    cw = (latency_ms + ot.COLLISION_REFRACTORY_MS) / 1000.0
    add = []
    for p in pulses:
        if not respond:
            continue
        j0 = np.searchsorted(spikes_arr, p - cw)
        j1 = np.searchsorted(spikes_arr, p)
        has_pre = (j1 - j0) > 0
        if collision and has_pre:
            continue  # antidromic spike collides → absent
        add.append(p + latency_ms / 1000.0 + rng.normal(0, jitter_ms / 1000.0))
    return np.sort(np.concatenate([spikes_arr, np.asarray(add)]))


def test_baseline_rate_recovers_poisson_rate():
    pulses = _pulses(n=200)
    sp = _antidromic_unit(pulses, base_rate=8.0, respond=False, seed=1)
    lam = ot.baseline_rate_hz(sp, pulses)
    assert 6.0 < lam < 10.0  # ~8 Hz


def test_estimate_response_window_finds_peak():
    pulses = _pulses(n=300)
    sp = _antidromic_unit(pulses, latency_ms=4.0, jitter_ms=0.1,
                          base_rate=3.0, collision=False, seed=2)
    rw = ot.estimate_response_window(sp, pulses)
    assert abs(rw.peak_latency_ms - 4.0) < 0.6
    assert rw.window_ms[0] < rw.peak_latency_ms < rw.window_ms[1]
    assert 1.0 < rw.baseline_rate_hz < 5.0
