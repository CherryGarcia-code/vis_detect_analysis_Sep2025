"""Tracking-free anatomical population field — instrument primitives.

Cross-session correspondence comes from fixed anatomy on a MATCH-FREE
registered depth axis (the amplitude-depth activity landscape), never from
single-unit tracking. See docs/superpowers/specs/2026-07-07-tracking-free-
population-field-design.md.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# New constants (flagged for user confirmation — Global Constraints).
DEPTH_BIN_UM: float = 60.0          # analysis grid depth bin
REG_MAX_LAG_UM: float = 300.0
# Registration uses a FINE, SMOOTHED fingerprint decoupled from the coarse grid
# (matches the proven diagnose_intersession_drift recipe: 10 um bins, sigma=2).
# A coarse/spiky profile makes the cross-session correlation argmax jitter, which
# consecutive-pair chaining then accumulates into large spurious shifts.
REG_BIN_UM: float = 10.0
REG_SMOOTH_BINS: int = 2


def depth_bin_edges(channel_positions: np.ndarray,
                    depth_bin_um: float = DEPTH_BIN_UM) -> np.ndarray:
    """Monotonic y-edges (µm) covering the active depth band at ``depth_bin_um``."""
    y = np.asarray(channel_positions, float)[:, 1]
    lo = np.floor(y.min() / depth_bin_um) * depth_bin_um
    hi = np.ceil(y.max() / depth_bin_um) * depth_bin_um
    return np.arange(lo, hi + depth_bin_um, depth_bin_um)


def registration_y_edges(channel_positions: np.ndarray,
                         bin_um: float = REG_BIN_UM) -> np.ndarray:
    """Fine depth-bin edges for the REGISTRATION fingerprint, matching the proven
    diagnose_intersession_drift construction: pad ONE bin below y.min and TWO above
    y.max (``arange(y.min-bin, y.max+2*bin, bin)``).

    The padding/offset is load-bearing, NOT cosmetic: a tight ``[floor(min),
    ceil(max)]`` axis (as ``depth_bin_edges`` gives) aliases against the ~15 um
    NP2.0 row pitch, and on low-correlation yield-transition session pairs a
    spurious far-lag correlation peak then beats lag 0 — empirically railing our
    registration at 270-630 um where this construction gets 0 (== diagnose).
    """
    y = np.asarray(channel_positions, float)[:, 1]
    return np.arange(y.min() - bin_um, y.max() + 2.0 * bin_um, bin_um)


def robust_unit_depth(mean_waveform: np.ndarray,
                      channel_positions: np.ndarray,
                      ptp_frac: float = 0.5) -> float:
    """Amplitude(ptp)-weighted centroid of channel depth over the FOOTPRINT only.

    Only channels whose ptp is >= ``ptp_frac`` * the peak ptp contribute; the
    hundreds of near-zero-ptp noise channels (which would otherwise drag the
    centroid toward the probe centre — a 119 µm bias measured on real BG_046) are
    excluded. NaN if no amplitude.
    """
    ptp = np.asarray(mean_waveform.max(axis=0) - mean_waveform.min(axis=0), float)
    maxptp = ptp.max() if ptp.size else 0.0
    if not np.isfinite(maxptp) or maxptp <= 0:
        return float("nan")
    y = np.asarray(channel_positions, float)[:, 1]
    mask = ptp >= ptp_frac * maxptp
    w = ptp[mask]
    return float((w * y[mask]).sum() / w.sum())


def amplitude_depth_fingerprint(unit_waveforms: List[np.ndarray],
                                channel_positions: np.ndarray,
                                y_edges: np.ndarray) -> np.ndarray:
    """Pool every channel's ptp of every unit into its depth bin (whole-probe)."""
    y = np.asarray(channel_positions, float)[:, 1]
    n_bins = len(y_edges) - 1
    chan_bin = np.clip(np.searchsorted(y_edges, y) - 1, 0, n_bins - 1)
    profile = np.zeros(n_bins, float)
    for mw in unit_waveforms:
        ptp = mw.max(axis=0) - mw.min(axis=0)       # (n_chan,)
        np.add.at(profile, chan_bin, ptp)
    return profile


def smooth_fingerprint(profile: np.ndarray,
                       n_bins: int = REG_SMOOTH_BINS) -> np.ndarray:
    """Gaussian-smooth an amplitude-depth fingerprint (sigma = ``n_bins`` bins).

    Turns the spiky per-unit profile into a smooth density so cross-session
    correlation is stable — not jittered by which exact units are present, the
    failure that (unsmoothed, on the coarse grid) railed registration. Lifted from
    scripts/pipelines/tracking/diagnose_intersession_drift.py::smooth.
    """
    profile = np.asarray(profile, float)
    if n_bins <= 0:
        return profile
    k = np.exp(-0.5 * (np.arange(-3 * n_bins, 3 * n_bins + 1) / n_bins) ** 2)
    k /= k.sum()
    return np.convolve(profile, k, mode="same")


def estimate_shift_bins(ref: np.ndarray, mov: np.ndarray,
                        max_lag_bins: int) -> Tuple[int, float]:
    """Rigid bin shift aligning ``mov`` onto ``ref`` + peak normalized corr.

    Lifted from scripts/pipelines/tracking/diagnose_intersession_drift.py::estimate_shift.
    """
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    denom = np.sqrt((ref ** 2).sum() * (mov ** 2).sum())
    if denom < 1e-9:
        return 0, 0.0
    best_lag, best_c = 0, -np.inf
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        shifted = np.roll(mov, lag)
        if lag > 0:
            shifted[:lag] = 0
        elif lag < 0:
            shifted[lag:] = 0
        c = float((ref * shifted).sum() / denom)
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag, best_c


from visdetect.analysis.tracking_qc import (          # noqa: E402
    load_raw_mean_waveform, load_channel_positions,
)


def session_fingerprint_from_root(raw_wf_root, session_name: str,
                                  unit_ids: List[int],
                                  y_edges: np.ndarray) -> np.ndarray:
    """Whole-probe amplitude-depth fingerprint for one session's good+stable units."""
    pos = load_channel_positions(raw_wf_root, session_name)
    if pos is None:
        return np.zeros(len(y_edges) - 1, float)
    wfs = []
    for uid in unit_ids:
        mw = load_raw_mean_waveform(raw_wf_root, session_name, int(uid))
        if mw is not None:
            wfs.append(mw)
    return amplitude_depth_fingerprint(wfs, pos, y_edges)


def session_shift_um(fingerprints: Dict[str, np.ndarray], ref_session: str,
                     depth_bin_um: float = DEPTH_BIN_UM,
                     max_lag_um: float = REG_MAX_LAG_UM
                     ) -> Dict[str, Tuple[float, float]]:
    """Per-session rigid registration shift (µm) + corr vs the reference session.

    Positive shift_um ⇒ that session's landscape sits deeper than the reference.
    """
    ref = fingerprints[ref_session]
    max_lag_bins = int(round(max_lag_um / depth_bin_um))
    out: Dict[str, Tuple[float, float]] = {}
    for sess, mov in fingerprints.items():
        lag, corr = estimate_shift_bins(ref, mov, max_lag_bins)
        out[sess] = (-lag * depth_bin_um, corr)   # deeper session -> positive shift
    return out


def session_shift_um_chained(fingerprints: Dict[str, np.ndarray],
                             sessions_chronological: List[str],
                             depth_bin_um: float = DEPTH_BIN_UM,
                             max_lag_um: float = REG_MAX_LAG_UM
                             ) -> Dict[str, Tuple[float, float]]:
    """Per-session registration shift (µm) via CONSECUTIVE-pair chaining, anchored
    at the LATEST session (shift 0), walking backward to the earliest.

    Adjacent sessions have similar unit yield, so their amplitude-depth fingerprints
    match in SHAPE. A single distant reference does not: its shape drifts as yield
    grows (few→many units), and raw correlation then misreads that shape mismatch as
    a position shift (the failure that railed ``session_shift_um`` at ±300 µm on real
    BG_046 data, where the true whole-probe drift is ~0). Each session's shift is the
    cumulative sum of consecutive rigid steps back to the anchor; the returned corr is
    the CONSECUTIVE-pair correlation for that session's link (1.0 for the anchor) — a
    per-session confidence to gate on.

    ``sessions_chronological`` must be ordered earliest→latest.
    Positive shift_um ⇒ that session's landscape sits deeper than the latest anchor.
    """
    order = list(sessions_chronological)
    if not order:
        return {}
    max_lag_bins = int(round(max_lag_um / depth_bin_um))
    anchor = order[-1]
    out: Dict[str, Tuple[float, float]] = {anchor: (0.0, 1.0)}
    cum = 0.0
    for i in range(len(order) - 2, -1, -1):
        earlier, later = order[i], order[i + 1]
        lag, corr = estimate_shift_bins(fingerprints[later], fingerprints[earlier],
                                        max_lag_bins)
        cum += -lag * depth_bin_um     # deeper 'earlier' vs 'later' -> +; accumulate to anchor
        out[earlier] = (cum, corr)
    return out


def registered_depth(raw_depth_um: float, shift_um: float) -> float:
    """Depth on the common registered axis: subtract the session's rigid shift."""
    return float(raw_depth_um) - float(shift_um)


def n_field_bins(y_edges: np.ndarray, n_shanks: int = 4) -> int:
    return int(n_shanks * (len(y_edges) - 1))


def unit_field_index(registered_depth_um: float, shank: int,
                     y_edges: np.ndarray, n_shanks: int = 4) -> int:
    """Flattened shank×depth bin index; depth clipped into the grid range."""
    n_depth = len(y_edges) - 1
    depth_bin = int(np.clip(np.searchsorted(y_edges, registered_depth_um) - 1,
                            0, n_depth - 1))
    s = int(np.clip(shank, 0, n_shanks - 1))
    return s * n_depth + depth_bin


from visdetect.analysis.utils import build_population_tensor as _build_population_tensor  # noqa: E402
from visdetect.analysis.constants import DEFAULT_BIN_SIZE                                  # noqa: E402


def build_field_tensor(session, unit_ids: List[int], unit_bin_index: np.ndarray,
                       n_bins_anat: int, event_name: str = "Change_ON",
                       window: Tuple[float, float] = (-1.0, 1.5),
                       bin_size: float = DEFAULT_BIN_SIZE,
                       outcome_filter: Optional[set] = None):
    """Aggregate the per-unit tensor into a (trials × time × anatomical-bin) field.

    Each field bin = SUM of member units' Hz (the local MUA-analog). Units whose
    bin index is outside [0, n_bins_anat) are dropped. Pass -1 (not NaN) for
    off-grid / no-depth units — an int-cast NaN is garbage.
    """
    per_unit, bin_centers, valid = _build_population_tensor(
        session, list(unit_ids), event_name=event_name, window=window,
        bin_size=bin_size, outcome_filter=outcome_filter)
    field = np.zeros((per_unit.shape[0], per_unit.shape[1], n_bins_anat), float)
    idx = np.asarray(unit_bin_index, int)
    for u in range(per_unit.shape[2]):
        b = idx[u]
        if 0 <= b < n_bins_anat:
            field[:, :, b] += per_unit[:, :, u]
    return field, bin_centers, valid


from visdetect.analysis.tracking_qc import extract_peak_channel      # noqa: E402


def fingerprint_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def peak_vs_centroid_depth(mean_waveform: np.ndarray,
                           channel_positions: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(channel_positions, float)[:, 1]
    peak_chan = extract_peak_channel(mean_waveform)
    return float(y[peak_chan]), robust_unit_depth(mean_waveform, channel_positions)


def audit_shift_vs_um_offset(match_free_um: Dict[str, float],
                             um_offset_um: Dict[str, float]) -> Dict[str, float]:
    """Compare match-free registration to the UM-anchored offset on shared sessions."""
    shared = [s for s in match_free_um if s in um_offset_um
              and np.isfinite(um_offset_um[s]) and np.isfinite(match_free_um[s])]
    if not shared:
        return {"n": 0, "median_abs_diff_um": float("nan"),
                "max_abs_diff_um": float("nan")}
    diffs = np.array([abs(match_free_um[s] - um_offset_um[s]) for s in shared])
    return {"n": int(len(shared)),
            "median_abs_diff_um": float(np.median(diffs)),
            "max_abs_diff_um": float(diffs.max())}


from collections import Counter                                      # noqa: E402


def select_dominant_signature(sig_by_session: Dict[str, str]
                              ) -> Tuple[str, List[str]]:
    """Spec §3 rule: pick the chanmap signature covering the most sessions."""
    counts = Counter(sig_by_session.values())
    chosen = max(counts, key=lambda s: (counts[s], s))
    sessions = [k for k, v in sig_by_session.items() if v == chosen]
    return chosen, sessions
