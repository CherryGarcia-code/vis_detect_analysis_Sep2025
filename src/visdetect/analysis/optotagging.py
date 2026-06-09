
"""Optotagging analysis: antidromic identification of D1/D2 SPNs.

Protocol: Post-session, 501 laser pulses to GPe (D2 tagging) followed by
501 pulses to SNr (D1 tagging), separated by a pause (>5 s).

Statistical test: SALT (Stimulus-Associated spike Latency Test,
Kvitsiani et al. 2013). Compares the distribution of first-spike latencies
after real pulses vs a jittered baseline using a two-sample
Kolmogorov–Smirnov-style test on spike-count histograms.

Functions
---------
split_laser_blocks        Separate the two stimulation blocks (GPe, SNr).
salt_test                 Gold-standard SALT significance test.
baseline_rate_hz          Pooled pre-pulse baseline firing rate.
estimate_response_window  Find the antidromic response latency/window (ResponseWindow).
excess_reliability        Baseline-corrected response reliability.
excess_jitter             First-spike-latency jitter (ms) within the response window.
poisson_excess_test       Upper-tail Poisson p-value for response-window excess.
collision_test            Offline antidromic confirmation via collision suppression (CollisionResult).
OptoTagger                Per-session analysis class.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from visdetect.core.session import Session, Cluster
from scipy.stats import poisson as _poisson, fisher_exact as _fisher_exact  # _poisson: poisson_excess_test (Task 2); _fisher_exact: collision_test (Task 5)


# ── Constants ──────────────────────────────────────────────────────────
LASER_KEY = "Laser"
RESPONSE_WINDOW_MS = (0.0, 10.0)   # post-pulse window for spike detection
BASELINE_WINDOW_MS = (-50.0, -5.0) # pre-pulse baseline for rate estimation (-5 guard)
SALT_N_JITTER = 500                # number of jittered baselines for SALT
SALT_ALPHA = 0.01                  # significance threshold
SALT_BIN_MS = 0.5                  # histogram bin width for SALT test
MIN_GAP_S = 5.0                    # minimum gap between GPe/SNr blocks
EXPECTED_PULSES_PER_BLOCK = 501
# Thresholds for "responsive" (applied after SALT p < alpha)
MAX_LATENCY_MS = 8.0
MAX_JITTER_MS = 3.5
MIN_RELIABILITY = 0.1

# ── New constants (antidromic redesign) ────────────────────────────────
SALT_BASELINE_WINDOW_MS = (-250.0, -5.0)  # canonical-SALT baseline period
RESPONSE_SEARCH_MS = (1.0, 10.0)          # antidromic latency search range
RESP_PSTH_BIN_MS = 0.1                    # fine PSTH bin for peak finding
RESP_HALFWIDTH_MS = 0.75                  # response-window half-width about the peak
COLLISION_REFRACTORY_MS = 1.0             # added to latency for the collision window
MIN_COLLISION_EXPECTED = 10               # min collision-eligible pulses to test
MIN_COLLISION_FREE = 30                   # min collision-free pulses to test
MAX_SALT_BASELINE_WINDOWS = 50            # cap baseline windows (cost bound)
# Tier thresholds
CANDIDATE_SALT_ALPHA = 0.05
CANDIDATE_POISSON_ALPHA = 0.01
CANDIDATE_MIN_EXCESS_REL = 0.02
STRICT_SALT_ALPHA = 0.01
STRICT_MAX_JITTER_MS = 1.0


@dataclass
class OptoMetrics:
    """Optotagging metrics for a single unit on a single fiber target."""
    cluster_id: int
    fiber: str                   # "GPe" or "SNr"
    is_responsive: bool
    latency_ms: float            # mean first-spike latency
    jitter_ms: float             # std of first-spike latency
    reliability: float           # fraction of pulses with ≥1 spike
    salt_p: float                # SALT p-value
    n_pulses: int
    first_spike_latencies: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


@dataclass
class ResponseWindow:
    peak_latency_ms: float
    window_ms: Tuple[float, float]
    baseline_rate_hz: float
    n_resp_spikes: int


@dataclass
class CollisionResult:
    status: str               # 'pass' | 'fail' | 'untestable'
    suppression_index: float
    p_free: float
    p_expected: float
    n_free: int
    n_expected: int
    fisher_p: float


def _count_in_window(spikes: np.ndarray, pulses: np.ndarray,
                     window_ms: Tuple[float, float]) -> int:
    a, b = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    tot = 0
    for p in pulses:
        i0 = np.searchsorted(spikes, p + a)
        i1 = np.searchsorted(spikes, p + b)
        tot += i1 - i0
    return int(tot)


def baseline_rate_hz(spike_times, pulse_times,
                     baseline_window_ms: Tuple[float, float] = BASELINE_WINDOW_MS) -> float:
    """Mean firing rate (Hz) in the pre-pulse baseline window, pooled over pulses.

    Parameters
    ----------
    spike_times : array-like of sorted spike times (seconds).
    pulse_times : array-like of laser pulse onsets (seconds).
    baseline_window_ms : (start, end) in ms relative to each pulse (both negative).

    Returns
    -------
    float : pooled baseline rate; 0.0 if spikes/pulses are empty or the window has
            non-positive duration.
    """
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    if len(spikes) == 0 or len(pulses) == 0:
        return 0.0
    dur = (baseline_window_ms[1] - baseline_window_ms[0]) / 1000.0
    total = _count_in_window(spikes, pulses, baseline_window_ms)
    return total / (len(pulses) * dur) if dur > 0 else 0.0


def estimate_response_window(spike_times, pulse_times,
                             search_ms: Tuple[float, float] = RESPONSE_SEARCH_MS,
                             bin_ms: float = RESP_PSTH_BIN_MS,
                             baseline_window_ms: Tuple[float, float] = BASELINE_WINDOW_MS,
                             half_width_ms: float = RESP_HALFWIDTH_MS) -> ResponseWindow:
    """Estimate the antidromic response window from the baseline-subtracted PSTH.

    Builds a fine-binned post-pulse PSTH over ``search_ms``, subtracts the expected
    baseline counts, takes the peak bin as the response latency, and returns a window
    of +/- ``half_width_ms`` around it (clipped to ``search_ms``). Always returns a
    window even for noise units (peak of a near-flat curve) -- downstream significance
    tests are responsible for rejecting non-responders.

    Returns
    -------
    ResponseWindow : peak_latency_ms, window_ms, baseline_rate_hz, n_resp_spikes.
    """
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    lam_b = baseline_rate_hz(spikes, pulses, baseline_window_ms)
    s0, s1 = search_ms[0] / 1000.0, search_ms[1] / 1000.0
    n_bins = max(1, int(round((s1 - s0) * 1000.0 / bin_ms)))
    edges = np.linspace(s0, s1, n_bins + 1)
    counts = np.zeros(n_bins)
    for p in pulses:
        i0 = np.searchsorted(spikes, p + s0)
        i1 = np.searchsorted(spikes, p + s1)
        if i1 > i0:
            counts += np.histogram(spikes[i0:i1] - p, bins=edges)[0]
    bin_s = (s1 - s0) / n_bins
    expected = lam_b * bin_s * len(pulses)
    peak_bin = int(np.argmax(counts - expected))
    peak_lat = float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0 * 1000.0)
    w0 = float(max(search_ms[0], peak_lat - half_width_ms))
    w1 = float(min(search_ms[1], peak_lat + half_width_ms))
    n_resp = _count_in_window(spikes, pulses, (w0, w1))
    return ResponseWindow(peak_lat, (w0, w1), lam_b, n_resp)


def excess_reliability(spike_times, pulse_times,
                       window_ms: Tuple[float, float],
                       baseline_rate_hz_val: float) -> float:
    """Baseline-corrected fraction of pulses with a response-window spike.

    Parameters
    ----------
    spike_times : array-like
        Sorted spike times (seconds).
    pulse_times : array-like
        Laser pulse onset times (seconds).
    window_ms : (start, end)
        Response window in ms relative to each pulse.
    baseline_rate_hz_val : float
        Pre-pulse baseline firing rate (Hz); used to estimate the fraction of
        pulses that would contain >=1 spike by chance.

    Returns
    -------
    float
        max(0, p_resp - p_base), where p_resp = fraction of pulses with >=1
        spike in the window and p_base = 1 - exp(-baseline_rate * |window|) is
        the baseline-expected hit rate. Returns 0.0 if no pulses.
    """
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    if len(pulses) == 0:
        return 0.0
    a, b = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    hits = 0
    for p in pulses:
        i0 = np.searchsorted(spikes, p + a)
        i1 = np.searchsorted(spikes, p + b)
        if i1 > i0:
            hits += 1
    p_resp = hits / len(pulses)
    win_dur = b - a
    p_base = 1.0 - np.exp(-baseline_rate_hz_val * win_dur)
    return float(max(0.0, p_resp - p_base))


def excess_jitter(spike_times, pulse_times,
                  window_ms: Tuple[float, float]) -> float:
    """Std of first-spike latencies (ms) within the window, over responding pulses.

    For each pulse, takes the first spike found inside ``window_ms`` (if any),
    computes its latency in milliseconds, and returns the standard deviation
    across all pulses that had at least one spike there.

    Parameters
    ----------
    spike_times : array-like
        Sorted spike times (seconds).
    pulse_times : array-like
        Laser pulse onset times (seconds).
    window_ms : (start, end)
        Response window in ms relative to each pulse.

    Returns
    -------
    float
        Std (ms) of first-spike latencies across responding pulses; returns
        ``nan`` if fewer than 2 pulses produced a spike in the window.
    """
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    a, b = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    lat = []
    for p in pulses:
        i0 = np.searchsorted(spikes, p + a)
        i1 = np.searchsorted(spikes, p + b)
        if i1 > i0:
            lat.append((spikes[i0] - p) * 1000.0)
    if len(lat) < 2:
        return float("nan")
    return float(np.std(lat))


def poisson_excess_test(spike_times, pulse_times,
                        window_ms: Tuple[float, float],
                        baseline_rate_hz_val: float) -> float:
    """Upper-tail Poisson p-value that response-window spikes exceed baseline.

    Parameters
    ----------
    spike_times : array-like
        Sorted spike times (seconds).
    pulse_times : array-like
        Laser pulse onset times (seconds).
    window_ms : (start, end)
        Response window in ms relative to each pulse.
    baseline_rate_hz_val : float
        Pre-pulse baseline firing rate (Hz).

    Returns
    -------
    float
        P(X >= k_obs) under Poisson(lam), where k_obs is the pooled spike count in
        ``window_ms`` across all pulses and lam = baseline_rate_hz_val * |window| *
        n_pulses. Returns 1.0 if no pulses; if lam <= 0, returns 0.0 when k_obs > 0
        else 1.0. Note: because the window is chosen post-hoc (peak bin), this test is
        the sensitive leg of the candidate tier, not a strict gate.
    """
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    if len(pulses) == 0:
        return 1.0
    k_obs = _count_in_window(spikes, pulses, window_ms)
    win_dur = (window_ms[1] - window_ms[0]) / 1000.0
    lam = baseline_rate_hz_val * win_dur * len(pulses)
    if lam <= 0:
        return 0.0 if k_obs > 0 else 1.0
    return float(_poisson.sf(k_obs - 1, lam))  # P(X >= k_obs)


def collision_test(spike_times, pulse_times, peak_latency_ms: float,
                   window_ms: Tuple[float, float],
                   refractory_ms: float = COLLISION_REFRACTORY_MS,
                   min_expected: int = MIN_COLLISION_EXPECTED,
                   min_free: int = MIN_COLLISION_FREE,
                   alpha: float = 0.05) -> CollisionResult:
    """Offline collision test for antidromic confirmation.

    Partitions pulses into collision-expected (a spontaneous spike fell within
    ``peak_latency_ms + refractory_ms`` ms before the pulse) and collision-free,
    then tests whether the response proportion is significantly higher on
    collision-free pulses using a one-sided Fisher exact test.

    A true antidromic spike will be annihilated by a spontaneous spike that
    collides with it head-on, so the response rate should be substantially
    suppressed on collision-expected pulses.

    Parameters
    ----------
    spike_times : array-like
        Sorted spike times (seconds).
    pulse_times : array-like
        Laser pulse onset times (seconds).
    peak_latency_ms : float
        Estimated antidromic latency (ms); used to compute the collision window
        ``(peak_latency_ms + refractory_ms) / 1000`` s before each pulse.
    window_ms : (start, end)
        Response window in ms relative to each pulse (from ``estimate_response_window``).
    refractory_ms : float
        Refractory period added to the peak latency for the collision window.
    min_expected : int
        Minimum collision-expected pulses required; returns ``"untestable"`` if fewer.
    min_free : int
        Minimum collision-free pulses required; returns ``"untestable"`` if fewer.
    alpha : float
        Significance threshold for the Fisher exact test.

    Returns
    -------
    CollisionResult
        status: ``'pass'`` if collision significantly suppresses the response,
        ``'fail'`` if not, ``'untestable'`` if too few eligible pulses in either group.
        suppression_index = (p_free - p_expected) / p_free.
        p_free / p_expected: response proportions in each group.
        n_free / n_expected: pulse counts in each group.
        fisher_p: one-sided Fisher exact p-value (``nan`` when untestable).
    """
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    cw = (peak_latency_ms + refractory_ms) / 1000.0
    a, b = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    resp_free = n_free = resp_exp = n_exp = 0
    for p in pulses:
        j0 = np.searchsorted(spikes, p - cw)
        j1 = np.searchsorted(spikes, p)
        has_pre = (j1 - j0) > 0
        i0 = np.searchsorted(spikes, p + a)
        i1 = np.searchsorted(spikes, p + b)
        has_resp = (i1 - i0) > 0
        if has_pre:
            n_exp += 1
            resp_exp += int(has_resp)
        else:
            n_free += 1
            resp_free += int(has_resp)
    p_free = resp_free / n_free if n_free > 0 else float("nan")
    p_exp = resp_exp / n_exp if n_exp > 0 else float("nan")
    supp = ((p_free - p_exp) / p_free
            if (n_free > 0 and n_exp > 0 and p_free > 0) else float("nan"))
    if n_exp < min_expected or n_free < min_free:
        return CollisionResult("untestable", supp, p_free, p_exp,
                               n_free, n_exp, float("nan"))
    table = [[resp_free, n_free - resp_free], [resp_exp, n_exp - resp_exp]]
    _, fp = _fisher_exact(table, alternative="greater")
    status = "pass" if (fp < alpha and p_free > p_exp) else "fail"
    return CollisionResult(status, supp, p_free, p_exp, n_free, n_exp, float(fp))


# ── Helper: split laser blocks ────────────────────────────────────────
def split_laser_blocks(
    pulse_times: np.ndarray,
    min_gap_s: float = MIN_GAP_S,
    expected_n: int = EXPECTED_PULSES_PER_BLOCK,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Split a flat array of laser pulse times into GPe (block 1) and SNr (block 2).

    Strategy:
      1. Find inter-pulse intervals > *min_gap_s* to identify block boundaries.
      2. Keep only blocks of size close to *expected_n* (within ±20 %).
      3. Return the first two valid blocks as (GPe, SNr).

    Returns (None, None) if fewer than two valid blocks are found.
    """
    pulse_times = np.sort(pulse_times.flatten())
    if len(pulse_times) < 2:
        return None, None

    ipis = np.diff(pulse_times)
    gap_idx = np.where(ipis > min_gap_s)[0]

    # Build block boundaries
    starts = np.concatenate([[0], gap_idx + 1]).astype(int)
    ends = np.concatenate([gap_idx + 1, [len(pulse_times)]]).astype(int)

    lo = int(expected_n * 0.8)
    hi = int(expected_n * 1.2)
    valid_blocks = []
    for s, e in zip(starts, ends):
        n = e - s
        if lo <= n <= hi:
            valid_blocks.append(pulse_times[s:e])

    if len(valid_blocks) >= 2:
        return valid_blocks[0], valid_blocks[1]
    elif len(valid_blocks) == 1:
        return valid_blocks[0], None
    return None, None


# ── SALT test ──────────────────────────────────────────────────────────
def _salt_test_jsd_uniform(
    spike_times: np.ndarray,
    pulse_times: np.ndarray,
    response_window_ms: Tuple[float, float] = RESPONSE_WINDOW_MS,
    baseline_window_ms: Tuple[float, float] = BASELINE_WINDOW_MS,
    n_jitter: int = SALT_N_JITTER,
    bin_ms: float = SALT_BIN_MS,
) -> float:
    """Original SALT implementation retained for reference (JSD-to-uniform null, RNG-based).

    For each laser pulse, build a histogram of spike times in the response
    window (fine bins of *bin_ms*).  Then build *n_jitter* baseline
    histograms by drawing random windows from the baseline period.
    The test statistic is JSD(real_hist, uniform); the p-value is the
    fraction of jittered JSD values that equal or exceed the real one.

    References: Kvitsiani et al., Nature 2013.

    Parameters
    ----------
    spike_times : sorted spike times (seconds).
    pulse_times : laser onset times (seconds).
    response_window_ms : (start, end) in ms relative to pulse.
    baseline_window_ms : (start, end) in ms for jitter baseline.
    n_jitter : number of jittered comparisons.
    bin_ms : bin width for latency histograms.

    Returns
    -------
    p_value : float in [0, 1].
    """
    spike_times = np.asarray(spike_times, dtype=float).ravel()
    pulse_times = np.asarray(pulse_times, dtype=float).ravel()
    n_pulses = len(pulse_times)
    if n_pulses == 0 or len(spike_times) == 0:
        return 1.0

    win_s = response_window_ms[0] / 1000.0
    win_e = response_window_ms[1] / 1000.0
    base_s = baseline_window_ms[0] / 1000.0
    base_e = baseline_window_ms[1] / 1000.0
    win_dur = win_e - win_s
    n_bins = max(1, int(round(win_dur * 1000.0 / bin_ms)))
    uniform = np.ones(n_bins) / n_bins

    jitter_range = base_e - base_s - win_dur
    if jitter_range <= 0:
        return 1.0

    # ── Vectorized: assign each spike a bin index relative to each pulse ──
    # For each pulse, find spikes in [pulse + win_s, pulse + win_e]
    # and digitize them into bins.

    def _build_hist(offsets_s: np.ndarray) -> np.ndarray:
        """Build average histogram across all pulses with given offsets."""
        hist = np.zeros(n_bins, dtype=float)
        for pi in range(n_pulses):
            t0 = pulse_times[pi] + offsets_s[pi]
            i0 = np.searchsorted(spike_times, t0)
            i1 = np.searchsorted(spike_times, t0 + win_dur)
            if i1 > i0:
                rel = spike_times[i0:i1] - t0
                # Digitize: bin index for each spike
                bin_idx = np.minimum(
                    (rel * (n_bins / win_dur)).astype(int), n_bins - 1
                )
                for b in bin_idx:
                    hist[b] += 1.0
        return hist / n_pulses

    # Real histogram (offset = win_s for all pulses)
    real_offsets = np.full(n_pulses, win_s)
    real_hist = _build_hist(real_offsets)
    if real_hist.sum() == 0:
        return 1.0
    real_hist_norm = real_hist / real_hist.sum()
    real_js = _jensen_shannon(real_hist_norm, uniform)

    # Null distribution
    rng = np.random.default_rng(42)
    n_exceed = 0
    for ji in range(n_jitter):
        offsets = rng.uniform(base_s, base_s + jitter_range, size=n_pulses)
        jit_hist = _build_hist(offsets)
        jit_total = jit_hist.sum()
        if jit_total == 0:
            continue
        jit_hist_norm = jit_hist / jit_total
        if _jensen_shannon(jit_hist_norm, uniform) >= real_js:
            n_exceed += 1

    p_value = float(n_exceed) / n_jitter
    return max(p_value, 1.0 / (n_jitter + 1))


def salt_test(spike_times, pulse_times,
              response_window_ms: Tuple[float, float] = RESPONSE_WINDOW_MS,
              baseline_window_ms: Tuple[float, float] = SALT_BASELINE_WINDOW_MS,
              n_jitter: int = SALT_N_JITTER,   # accepted for back-compat; ignored
              bin_ms: float = SALT_BIN_MS,
              max_windows: int = MAX_SALT_BASELINE_WINDOWS) -> float:
    """Canonical SALT (Kvitsiani et al. 2013).

    Latency distributions (with a 'no-spike' category) are built for the test window
    and for many equal-width baseline windows. The null is the distribution of
    baseline-vs-baseline JS divergences; the statistic is the mean test-vs-baseline
    JS divergence; p = (1 + #{null >= stat}) / (1 + n_null). Deterministic.
    ``n_jitter`` is accepted for backward compatibility but ignored.
    """
    spikes = np.asarray(spike_times, float).ravel()
    pulses = np.asarray(pulse_times, float).ravel()
    if len(spikes) == 0 or len(pulses) == 0:
        return 1.0
    win_dur = (response_window_ms[1] - response_window_ms[0]) / 1000.0
    test_off = response_window_ms[0] / 1000.0
    b0, b1 = baseline_window_ms[0] / 1000.0, baseline_window_ms[1] / 1000.0
    if win_dur <= 0:
        return 1.0
    n_base_full = int((b1 - b0) // win_dur)
    if n_base_full < 2:
        return 1.0
    offsets = b0 + np.arange(n_base_full) * win_dur
    if n_base_full > max_windows:
        offsets = offsets[np.linspace(0, n_base_full - 1, max_windows).astype(int)]
    n_bins = max(1, int(round(win_dur * 1000.0 / bin_ms)))

    def _dist(offset: float) -> np.ndarray:
        hist = np.zeros(n_bins + 1)  # last entry = 'no spike'
        for p in pulses:
            t0 = p + offset
            i0 = np.searchsorted(spikes, t0)
            i1 = np.searchsorted(spikes, t0 + win_dur)
            if i1 > i0:
                rel = spikes[i0] - t0
                bi = min(int(rel / win_dur * n_bins), n_bins - 1)
                hist[bi] += 1
            else:
                hist[-1] += 1
        s = hist.sum()
        return hist / s if s > 0 else hist

    test_d = _dist(test_off)
    base_d = [_dist(o) for o in offsets]
    null = [_jensen_shannon(base_d[i], base_d[j])
            for i in range(len(base_d)) for j in range(i + 1, len(base_d))]
    if not null:
        return 1.0
    stat = float(np.mean([_jensen_shannon(test_d, bd) for bd in base_d]))
    null = np.asarray(null)
    return float((1 + np.sum(null >= stat)) / (1 + len(null)))


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability vectors."""
    m = 0.5 * (p + q)
    # Avoid log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / (m + 1e-30)), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(q / (m + 1e-30)), 0.0)
    return float(0.5 * kl_pm.sum() + 0.5 * kl_qm.sum())


# ── First-spike latency extraction ────────────────────────────────────
def _first_spike_latencies(
    spike_times: np.ndarray,
    pulse_times: np.ndarray,
    window_ms: Tuple[float, float] = RESPONSE_WINDOW_MS,
) -> Tuple[np.ndarray, int, float]:
    """Return (latencies_ms, hit_count, reliability)."""
    win_s = window_ms[0] / 1000.0
    win_e = window_ms[1] / 1000.0
    latencies = []
    hit_count = 0
    for p in pulse_times:
        i0 = np.searchsorted(spike_times, p + win_s)
        i1 = np.searchsorted(spike_times, p + win_e)
        if i1 > i0:
            latencies.append((spike_times[i0] - p) * 1000.0)
            hit_count += 1
    reliability = hit_count / len(pulse_times) if len(pulse_times) > 0 else 0.0
    return np.array(latencies, dtype=float), hit_count, reliability


# ── OptoTagger class ──────────────────────────────────────────────────
class OptoTagger:
    """Per-session optotagging analysis.

    Splits laser pulses into GPe (block 1, D2 tagging) and SNr (block 2,
    D1 tagging), then analyses each unit with the SALT test plus
    latency/jitter/reliability criteria.
    """

    def __init__(
        self,
        session: Session,
        laser_key: str = LASER_KEY,
        response_window_ms: Tuple[float, float] = RESPONSE_WINDOW_MS,
        baseline_window_ms: Tuple[float, float] = BASELINE_WINDOW_MS,
        salt_alpha: float = SALT_ALPHA,
        salt_n_jitter: int = SALT_N_JITTER,
    ):
        self.session = session
        self.laser_key = laser_key
        self.response_window_ms = response_window_ms
        self.baseline_window_ms = baseline_window_ms
        self.salt_alpha = salt_alpha
        self.salt_n_jitter = salt_n_jitter

        # Resolve laser key
        ni = session.ni_events or {}
        if laser_key not in ni:
            resolved = None
            for k in ni:
                if "laser" in k.lower() or "opto" in k.lower():
                    resolved = k
                    break
            if resolved is None:
                raise ValueError(
                    f"Laser key '{laser_key}' not found in ni_events. "
                    f"Available: {list(ni.keys())}"
                )
            self.laser_key = resolved

        events = ni[self.laser_key]
        if isinstance(events, dict) and "rise_t" in events:
            all_pulses = np.asarray(events["rise_t"], dtype=float).flatten()
        else:
            all_pulses = np.asarray(events, dtype=float).flatten()

        self.gpe_pulses, self.snr_pulses = split_laser_blocks(all_pulses)
        n_gpe = len(self.gpe_pulses) if self.gpe_pulses is not None else 0
        n_snr = len(self.snr_pulses) if self.snr_pulses is not None else 0
        print(f"  Laser blocks: GPe={n_gpe} pulses, SNr={n_snr} pulses "
              f"(total raw={len(all_pulses)})")

    # ── Single-unit analysis ──────────────────────────────────────────
    def analyze_unit(
        self, cluster: Cluster, pulse_times: np.ndarray, fiber: str
    ) -> OptoMetrics:
        """Analyse one unit against one block of laser pulses."""
        spikes = np.asarray(cluster.spike_times, dtype=float).ravel()
        n_pulses = len(pulse_times)

        if len(spikes) == 0 or n_pulses == 0:
            return OptoMetrics(
                cluster_id=cluster.cluster_id, fiber=fiber,
                is_responsive=False, latency_ms=np.nan,
                jitter_ms=np.nan, reliability=0.0, salt_p=1.0,
                n_pulses=n_pulses,
            )

        latencies, hit_count, reliability = _first_spike_latencies(
            spikes, pulse_times, self.response_window_ms
        )

        if hit_count == 0:
            return OptoMetrics(
                cluster_id=cluster.cluster_id, fiber=fiber,
                is_responsive=False, latency_ms=np.nan,
                jitter_ms=np.nan, reliability=reliability, salt_p=1.0,
                n_pulses=n_pulses,
            )

        latency_mean = float(np.mean(latencies))
        jitter = float(np.std(latencies))

        # SALT test
        p_val = salt_test(
            spikes, pulse_times,
            response_window_ms=self.response_window_ms,
            baseline_window_ms=self.baseline_window_ms,
            n_jitter=self.salt_n_jitter,
        )

        is_responsive = (
            p_val < self.salt_alpha
            and latency_mean < MAX_LATENCY_MS
            and jitter < MAX_JITTER_MS
            and reliability >= MIN_RELIABILITY
        )

        return OptoMetrics(
            cluster_id=cluster.cluster_id, fiber=fiber,
            is_responsive=is_responsive, latency_ms=latency_mean,
            jitter_ms=jitter, reliability=reliability, salt_p=p_val,
            n_pulses=n_pulses, first_spike_latencies=latencies,
        )

    # ── Analyze all units for both fibers ─────────────────────────────
    def analyze_all(
        self, cluster_ids: Optional[List[int]] = None
    ) -> List[OptoMetrics]:
        """Run SALT + metrics for every (unit, fiber) combination.

        Parameters
        ----------
        cluster_ids : restrict to these cluster IDs.  If *None*, uses
                      good_and_stable_ids → good_cluster_ids → all.
        """
        if cluster_ids is None:
            if getattr(self.session, "good_and_stable_ids", None):
                cluster_ids = list(self.session.good_and_stable_ids)
            elif getattr(self.session, "good_cluster_ids", None):
                cluster_ids = list(self.session.good_cluster_ids)
            else:
                cluster_ids = [c.cluster_id for c in self.session.clusters]

        results: List[OptoMetrics] = []
        fibers = []
        if self.gpe_pulses is not None:
            fibers.append(("GPe", self.gpe_pulses))
        if self.snr_pulses is not None:
            fibers.append(("SNr", self.snr_pulses))

        for c in self.session.clusters:
            if c.cluster_id not in cluster_ids:
                continue
            for fiber_name, pulses in fibers:
                m = self.analyze_unit(c, pulses, fiber_name)
                results.append(m)
        return results

