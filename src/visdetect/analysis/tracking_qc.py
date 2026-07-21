"""Per-UID tracking QC: metrics, badge logic, and extraction primitives.

This module is library code (no I/O orchestration). The
`scripts/pipelines/tracking/build_qc_sheets.py` driver wires it up.

See docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md
"""

from __future__ import annotations

import gc
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from visdetect.analysis.utils import build_population_tensor, smooth_psth
from visdetect.analysis.constants import DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS

# ─── Badge thresholds (tweakable; documented in spec §7) ──────────────
ISI_PASS: float = 0.75
ISI_WARN: float = 0.65

DEPTH_PASS_UM: float = 25.0
DEPTH_WARN_UM: float = 40.0

WAVE_PASS_R: float = 0.95
WAVE_WARN_R: float = 0.90

FR_CV_PASS: float = 0.35
FR_CV_WARN: float = 0.60

# ISI peak-agreement (cross-session bimodality detector).
# Fraction of sessions whose ISI peak bin is within +/- 2 of the modal peak bin.
# Low agreement -> per-session ISI peaks land in different places -> likely
# UnitMatch matched two biologically distinct units with similar templates.
ISI_PEAK_AGREE_PASS: float = 0.85
ISI_PEAK_AGREE_WARN: float = 0.65
ISI_PEAK_AGREE_TOL_BINS: int = 2

# Functional response stability (cross-session Baseline_ON PSTH shape correlation).
# Median pairwise Pearson r across all session pairs. Uses only the stimulus-
# locked baseline response — robust to learning-driven magnitude changes (Pearson r
# normalizes by std) but sensitive to genuinely different stimulus tuning (which
# would indicate UM matched two distinct neurons with similar waveforms).
#
# Calibrated to BG_046 cohort distribution (May 2026): striatal Baseline_ON
# PSTHs are weakly modulated, so absolute correlations are systematically low.
# Gold-standard UID 942 scores ~0.62; the metric still rank-orders correctly
# but absolute scale needs cohort-appropriate thresholds.
FUNC_RESP_PASS: float = 0.40
FUNC_RESP_WARN: float = 0.15
FUNC_RESP_MIN_PSTH_STD: float = 0.5   # Hz; below this, PSTHs are too flat to discriminate

# ISI histogram cross-session correlation (richer than badge_isi_peak which only
# looks at argmax bin). Captures full ISI distribution shape — handles bursting
# cells (with consistent bimodal ISIs) correctly. Calibrated to BG_046 cohort
# distribution (May 2026): gold-standard UIDs ~0.97-0.99, anti-drift suspect
# ~0.74, known matching-failures 0.58-0.61.
ISI_HIST_CORR_PASS: float = 0.85
ISI_HIST_CORR_WARN: float = 0.65

# ISI hist-corr auto-pass: a UID whose set-wide ISI shape correlation is
# exceptionally consistent (>= 0.95, top ~25% of BG_046 cohort) is promoted
# to trusted regardless of marginal failures on other badges. Hard biophysical
# signals (wave or depth FAIL) still block — same philosophy as the
# skip-able-trim hard-outlier rule.
ISI_HIST_CORR_AUTOPASS: float = 0.95

# ─── Change-size pools for Change_ON heatmaps ─────────────────────────
# Change-size pools for Change_ON heatmaps.
# Deliberately differs from visdetect.analysis.constants.{BIG,SMALL}_CHANGE_SIZES:
# 1.5× is excluded here because the spec treats it as ambiguous mid (spec §4).
BIG_POOL: Set[float] = {2.0, 4.0}
SMALL_POOL: Set[float] = {1.25, 1.35}

# ─── Footprint extraction ─────────────────────────────────────────────
# How many channels above/below the peak to include in the footprint snippet.
FOOTPRINT_HALFWIDTH_CHANS: int = 8


# ─── Cross-session metric functions ───────────────────────────────────

def depth_std_um(depths_um: np.ndarray) -> float:
    """Std of peak-channel depth across sessions, in microns.

    NaN values are ignored. Returns NaN if fewer than 2 finite values.
    """
    arr = np.asarray(depths_um, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=0))


def waveform_corr(waveforms: np.ndarray) -> float:
    """Mean pairwise Pearson r of L2-normalized peak-channel waveforms.

    Parameters
    ----------
    waveforms : ndarray, shape (n_sessions, n_samples)
        Per-session mean waveform on the peak channel.

    Returns
    -------
    float
        Mean over the (n*(n-1)/2) cross-session pairwise correlations.
        NaN if fewer than 2 sessions or if normalization fails.
    """
    arr = np.asarray(waveforms, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return float("nan")

    # L2-normalize per row; drop rows that are all-zero
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    keep = norms.flatten() > 1e-12
    if keep.sum() < 2:
        return float("nan")
    normed = arr[keep] / norms[keep]

    # Pearson r of normalized vectors == cosine == dot product after mean removal
    # We want Pearson, not cosine — subtract row mean first
    normed = normed - normed.mean(axis=1, keepdims=True)
    # Renormalize after mean-subtraction
    norms2 = np.linalg.norm(normed, axis=1, keepdims=True)
    norms2[norms2 < 1e-12] = 1.0
    normed = normed / norms2

    n = normed.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(float(np.dot(normed[i], normed[j])))
    return float(np.mean(pairs))


def fr_cv(rates_hz: np.ndarray) -> float:
    """Coefficient of variation (std/mean) of baseline firing rate.

    NaNs are dropped. Returns NaN for empty / zero-mean / single-session inputs.
    """
    arr = np.asarray(rates_hz, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) < 1e-9:
        return float("nan")
    return float(np.std(arr, ddof=0) / mean)


def isi_peak_agreement(per_session_isi_hists: Sequence[np.ndarray]) -> float:
    """Fraction of sessions whose ISI peak bin agrees with the modal peak bin.

    Bimodal cross-session ISI overlays (a fingerprint of UnitMatch matching
    two distinct neurons with similar waveforms) produce low agreement scores.

    Parameters
    ----------
    per_session_isi_hists : sequence of (50,) ndarrays
        Each is the normalised log-ISI histogram for one session. NaN-only
        histograms are ignored (sessions with too few spikes for an ISI hist).

    Returns
    -------
    float in [0, 1], or NaN if fewer than 2 valid sessions.
        1.0 = all sessions peak at the same (+/- ISI_PEAK_AGREE_TOL_BINS) bin.
    """
    from collections import Counter
    peaks: List[int] = []
    for h in per_session_isi_hists:
        if h is None:
            continue
        h_arr = np.asarray(h, dtype=float)
        if h_arr.size == 0 or np.all(np.isnan(h_arr)):
            continue
        peaks.append(int(np.argmax(h_arr)))
    if len(peaks) < 2:
        return float("nan")
    mode_peak = Counter(peaks).most_common(1)[0][0]
    peaks_arr = np.asarray(peaks)
    fraction = float(np.mean(np.abs(peaks_arr - mode_peak) <= ISI_PEAK_AGREE_TOL_BINS))
    return fraction


def baseline_psth_corr(per_session_psths: Sequence[Optional[np.ndarray]]) -> float:
    """Median pairwise Pearson r of per-session Baseline_ON PSTHs.

    Pearson r is invariant to per-session magnitude scaling, so this catches
    SHAPE changes (e.g., a different neuron with different stimulus tuning)
    rather than magnitude changes (e.g., learning-driven gain changes for the
    same neuron).

    Parameters
    ----------
    per_session_psths : sequence of (n_bins,) ndarrays, or None
        One PSTH per session. None entries (sessions with no trials for this
        condition) are dropped.

    Returns
    -------
    float
        Median over all (n*(n-1)/2) cross-session pairwise correlations. NaN
        if fewer than 2 valid sessions, or if all valid PSTHs are flat.
    """
    arrs: List[np.ndarray] = []
    for p in per_session_psths:
        if p is None:
            continue
        a = np.asarray(p, dtype=float)
        if a.size == 0 or np.all(np.isnan(a)) or float(np.std(a)) < 1e-12:
            continue
        arrs.append(a)
    if len(arrs) < 2:
        return float("nan")
    # Pad / truncate to common length (PSTHs should already share length, but
    # be defensive in case of off-by-one in edge cases).
    min_len = min(a.size for a in arrs)
    stack = np.stack([a[:min_len] for a in arrs])
    # Modulation gate: if PSTHs are essentially flat, the correlation is
    # meaningless. Return NaN to signal "can't discriminate" (which
    # badge_func_resp interprets as pass-by-default).
    per_session_std = np.std(stack, axis=1)
    if float(np.median(per_session_std)) < FUNC_RESP_MIN_PSTH_STD:
        return float("nan")
    # Pearson r via mean-subtract + L2-normalize
    centered = stack - stack.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    unit = centered / norms
    n = unit.shape[0]
    pairs = [float(np.dot(unit[i], unit[j])) for i in range(n) for j in range(i + 1, n)]
    return float(np.median(pairs))


def baseline_isi_hist_corr(per_session_isi_hists: Sequence[np.ndarray]) -> float:
    """Median pairwise Pearson r of per-session log-ISI histograms.

    Captures full ISI distribution shape — handles bursting cells (with
    consistent bimodal ISIs) correctly, unlike isi_peak_agreement which looks
    only at the argmax bin. Architecturally mirrors waveform_corr.

    Parameters
    ----------
    per_session_isi_hists : sequence of (n_bins,) ndarrays, or None
        Per-session log-ISI histograms. None / NaN-only / flat (std < 1e-12)
        hists are dropped.

    Returns
    -------
    float
        Median over the n*(n-1)/2 pairwise Pearson r values. NaN if fewer than
        2 valid sessions remain after dropping.
    """
    arrs: List[np.ndarray] = []
    for h in per_session_isi_hists:
        if h is None:
            continue
        a = np.asarray(h, dtype=float)
        if a.size == 0 or np.all(np.isnan(a)) or float(np.std(a)) < 1e-12:
            continue
        arrs.append(a)
    if len(arrs) < 2:
        return float("nan")
    min_len = min(a.size for a in arrs)
    stack = np.stack([a[:min_len] for a in arrs])
    # Pearson r via mean-subtract + L2-normalize → pairwise dot products
    centered = stack - stack.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    unit = centered / norms
    n = unit.shape[0]
    pairs = [float(np.dot(unit[i], unit[j])) for i in range(n) for j in range(i + 1, n)]
    return float(np.median(pairs))


# ─── Cross-session probe drift correction ─────────────────────────────
# Anchored on UM consecutive-pair matches with prob > DRIFT_ANCHOR_PROB.
# The probe physically drifts a small amount between chronic sessions; for some
# tracked units, what looks like "depth instability across sessions" is actually
# the probe moving, not the unit moving. We estimate per-session Z-offset from
# high-confidence (prob>0.95) consecutive-pair matches, then subtract it from
# each session's peak depth to get a drift-corrected depth.
DRIFT_ANCHOR_PROB: float = 0.95
DRIFT_MIN_ANCHORS: int = 5


def estimate_session_drift(unit_index_df: pd.DataFrame,
                           prob_matrix: np.ndarray,
                           raw_wf_root,
                           sessions_chronological: List[str],
                           anchor_prob: float = DRIFT_ANCHOR_PROB,
                           min_anchors: int = DRIFT_MIN_ANCHORS,
                           ) -> Dict[str, float]:
    """Estimate per-session Z-offset relative to the first session.

    For each consecutive session pair (i -> i+1), the drift is the median
    Z-difference of high-confidence UM matches (prob > anchor_prob). Drifts
    are cumulative: session i's offset is the sum of all i'-> i'+1 drifts
    for i' < i, anchored at session 0 -> 0.

    Parameters
    ----------
    unit_index_df : pd.DataFrame
        UM unit_index.csv loaded; columns include 'session', 'ks_unit_id'.
        Row order must align with `prob_matrix`.
    prob_matrix : ndarray, shape (n_um_units, n_um_units)
        UM output_prob_matrix.npy; index aligned with unit_index_df row order.
    raw_wf_root : Path
        Root of unit_match input data (e.g. data/unit_match/input/BG_046).
        Used to load channel_positions per session.
    sessions_chronological : list of str
        Session names in chronological order. The first is the anchor (offset=0).
    anchor_prob : float
        Min match prob to count a pair as an anchor.
    min_anchors : int
        If a session pair has fewer than this many anchor matches, drift for
        that pair is NaN and the cumulative chain breaks (remaining sessions
        get NaN).

    Returns
    -------
    dict[session_name -> Z-offset in microns relative to first session]
        Offset NaN for sessions where the cumulative chain breaks.
    """
    # Normalize session-name format: UM unit_index sometimes stores 7-char names
    # (e.g. '1072025' for July 1) while the staging manifest pads to 8 ('01072025').
    # All comparisons happen in zfill(8) space; lookups back to the raw UM rows
    # use a per-session original-string remember.
    def _norm(s: str) -> str:
        return str(s).zfill(8)

    idx = unit_index_df.copy().reset_index(drop=True)
    idx["session_raw"] = idx["session"].astype(str)
    idx["session"] = idx["session_raw"].map(_norm)

    sessions_norm = [_norm(s) for s in sessions_chronological]
    # Map normalized name -> the raw UM/file-system name (some UM rows store
    # '1072025'; load_channel_positions / load_raw_mean_waveform tolerate both,
    # but we keep the raw form for clarity in cache keys).
    raw_by_norm: Dict[str, str] = {}
    for raw, norm in zip(idx["session_raw"].values, idx["session"].values):
        raw_by_norm.setdefault(str(norm), str(raw))
    # Sessions in `sessions_chronological` that don't appear in UM at all
    # get a fallback raw form equal to their input string.
    for raw, norm in zip(sessions_chronological, sessions_norm):
        raw_by_norm.setdefault(str(norm), str(raw))

    # Pre-load channel_positions per session (cheap; one .npy per session).
    chan_pos_by_session: Dict[str, Optional[np.ndarray]] = {}
    for snorm in sessions_norm:
        chan_pos_by_session[snorm] = load_channel_positions(
            raw_wf_root, raw_by_norm[snorm]
        )

    # Per-(session, ks_id) peak-channel cache. Sentinel -1 = file missing.
    peak_chan_cache: Dict[Tuple[str, int], int] = {}

    def _peak_z(session_norm: str, ks_id: int) -> Optional[float]:
        key = (session_norm, int(ks_id))
        if key not in peak_chan_cache:
            mw = load_raw_mean_waveform(
                raw_wf_root, raw_by_norm[session_norm], int(ks_id)
            )
            if mw is None:
                peak_chan_cache[key] = -1
            else:
                peak_chan_cache[key] = extract_peak_channel(mw)
        pc = peak_chan_cache[key]
        if pc < 0:
            return None
        chan_pos = chan_pos_by_session.get(session_norm)
        if chan_pos is None or pc >= chan_pos.shape[0]:
            return None
        return float(chan_pos[pc, 1])

    # Pre-index UM rows by session (normalized) for fast sub-matrix slicing.
    rows_by_session: Dict[str, np.ndarray] = {
        s: np.where(idx["session"].values == s)[0]
        for s in sessions_norm
    }
    ks_by_session: Dict[str, np.ndarray] = {
        s: idx["ks_unit_id"].values[rows_by_session[s]]
        for s in sessions_norm
    }

    # Output keyed by ORIGINAL input strings so the caller can look up offsets
    # by whatever session-name form they used (which is what's also stored in
    # SessionRecord.session_name).
    sess_to_offset: Dict[str, float] = {sessions_chronological[0]: 0.0}
    # Also publish the normalized form for robust downstream lookups.
    sess_to_offset[sessions_norm[0]] = 0.0
    running_offset = 0.0
    n_sess = len(sessions_chronological)
    # Chain policy: if a pair has too few anchors, we mark THAT session's offset
    # as NaN (so depth_std_um_corrected drops it) but continue the chain at the
    # previous running_offset for downstream sessions. This loses correction
    # for the gap session itself but preserves coverage for everything that
    # follows — a defensible trade for chronic recordings where ~all gap sessions
    # are isolated bad days rather than systematic chain failures.
    for i in range(n_sess - 1):
        s_a_orig = sessions_chronological[i]
        s_b_orig = sessions_chronological[i + 1]
        s_a = sessions_norm[i]
        s_b = sessions_norm[i + 1]
        rows_a = rows_by_session.get(s_a, np.array([], dtype=int))
        rows_b = rows_by_session.get(s_b, np.array([], dtype=int))
        if rows_a.size == 0 or rows_b.size == 0:
            sess_to_offset[s_b_orig] = float("nan")
            sess_to_offset[s_b] = float("nan")
            print(f"  drift {s_a}->{s_b}: session missing from UM index", flush=True)
            continue

        # Vectorized: pull the (rows_a x rows_b) sub-block of the prob matrix
        # in one call, then locate cells exceeding anchor_prob.
        sub = prob_matrix[np.ix_(rows_a, rows_b)]
        hit_a, hit_b = np.where(sub > anchor_prob)
        if hit_a.size == 0:
            sess_to_offset[s_b_orig] = float("nan")
            sess_to_offset[s_b] = float("nan")
            print(f"  drift {s_a}->{s_b}: 0 anchors (this session uncorrectable)",
                  flush=True)
            continue

        ks_a_arr = ks_by_session[s_a]
        ks_b_arr = ks_by_session[s_b]
        deltas: List[float] = []
        for ka_idx, kb_idx in zip(hit_a, hit_b):
            z_a = _peak_z(s_a, int(ks_a_arr[ka_idx]))
            z_b = _peak_z(s_b, int(ks_b_arr[kb_idx]))
            if z_a is None or z_b is None:
                continue
            deltas.append(z_b - z_a)
        if len(deltas) >= min_anchors:
            pair_drift = float(np.median(deltas))
            running_offset += pair_drift
            sess_to_offset[s_b_orig] = running_offset
            sess_to_offset[s_b] = running_offset
            print(f"  drift {s_a}->{s_b}: {len(deltas)} anchors, "
                  f"d={pair_drift:+.2f} um, cum={running_offset:+.2f} um",
                  flush=True)
        else:
            sess_to_offset[s_b_orig] = float("nan")
            sess_to_offset[s_b] = float("nan")
            print(f"  drift {s_a}->{s_b}: only {len(deltas)} anchors "
                  f"(< {min_anchors}); this session uncorrectable",
                  flush=True)
    return sess_to_offset


def depth_std_um_corrected(uid: "UIDIntermediate",
                            drift_offsets: Dict[str, float]) -> float:
    """Depth std after subtracting per-session drift offsets.

    The lookup is tolerant of session-name padding differences ('1072025' vs
    '01072025') so the caller doesn't need to normalize.

    Returns NaN if fewer than 2 sessions have a finite drift correction available.
    """
    corrected: List[float] = []
    for r in uid.sessions:
        sname = str(r.session_name)
        offset = drift_offsets.get(sname, float("nan"))
        if not np.isfinite(offset):
            # Try zfill(8) fallback for padding-mismatch sessions
            offset = drift_offsets.get(sname.zfill(8), float("nan"))
        if not np.isfinite(offset) or not np.isfinite(r.peak_depth_um):
            continue
        corrected.append(r.peak_depth_um - offset)
    if len(corrected) < 2:
        return float("nan")
    return float(np.std(np.asarray(corrected), ddof=0))


# ─── Badge / verdict logic ────────────────────────────────────────────

def _badge_threshold(value: float, pass_thr: float, warn_thr: float,
                     direction: str) -> str:
    """Apply pass/warn/fail thresholds.

    direction='high' : pass if value >= pass_thr, warn between, fail below.
    direction='low'  : pass if value <= pass_thr, warn between, fail above.
    NaN always returns 'fail'.
    """
    if not np.isfinite(value):
        return "fail"
    if direction == "high":
        if value >= pass_thr:
            return "pass"
        if value >= warn_thr:
            return "warn"
        return "fail"
    elif direction == "low":
        if value <= pass_thr:
            return "pass"
        if value <= warn_thr:
            return "warn"
        return "fail"
    raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")


def badge_isi(median_corr: float) -> str:
    return _badge_threshold(median_corr, ISI_PASS, ISI_WARN, direction="high")


def badge_depth(std_um: float) -> str:
    return _badge_threshold(std_um, DEPTH_PASS_UM, DEPTH_WARN_UM, direction="low")


def badge_waveform(mean_pairwise_r: float) -> str:
    return _badge_threshold(mean_pairwise_r, WAVE_PASS_R, WAVE_WARN_R, direction="high")


def badge_fr(cv: float) -> str:
    return _badge_threshold(cv, FR_CV_PASS, FR_CV_WARN, direction="low")


def badge_isi_peak(agreement: float) -> str:
    return _badge_threshold(agreement, ISI_PEAK_AGREE_PASS,
                            ISI_PEAK_AGREE_WARN, direction="high")


def badge_func_resp(median_r: float) -> str:
    """Functional response stability badge.

    NOTE: NaN -> "pass" (not "fail"), unlike other badges. NaN here means the
    Baseline_ON PSTH is too flat to discriminate (quiet cell), not that anything
    is wrong with the tracking. Absence of stimulus modulation is not evidence
    of matching failure.
    """
    if not np.isfinite(median_r):
        return "pass"
    return _badge_threshold(median_r, FUNC_RESP_PASS, FUNC_RESP_WARN, direction="high")


def badge_isi_hist_corr(r: float) -> str:
    """ISI histogram cross-session correlation badge.

    NaN → "fail" (standard pattern for ISI metrics; distinct from badge_func_resp
    which is lenient on NaN). NaN here means we couldn't compute the metric,
    which is itself a signal that something is wrong with the unit.
    """
    return _badge_threshold(r, ISI_HIST_CORR_PASS, ISI_HIST_CORR_WARN,
                            direction="high")


def composite_verdict(badges: Sequence[str]) -> str:
    """Spec §7 composite logic.

    trusted = all pass
    review  = ≤1 warn AND no fails
    suspect = any fail OR ≥2 warns
    """
    n_fail = sum(1 for b in badges if b == "fail")
    n_warn = sum(1 for b in badges if b == "warn")
    if n_fail >= 1 or n_warn >= 2:
        return "suspect"
    if n_warn == 1:
        return "review"
    return "trusted"


def apply_isi_autopass(verdict: str,
                       isi_hist_corr: float,
                       wave_badge: str,
                       depth_badge: str,
                       threshold: float = ISI_HIST_CORR_AUTOPASS) -> str:
    """Promote verdict to 'trusted' when ISI shape correlation is exceptionally
    strong AND no hard biophysical badge fails.

    Hard biophysical signals (wave_badge or depth_badge == 'fail') block the
    promotion — they suggest a physically different unit at the recording
    position, which ISI alone cannot overrule.

    Parameters
    ----------
    verdict : str
        Current composite verdict ('trusted', 'review', 'suspect').
    isi_hist_corr : float
        Set-wide median pairwise Pearson r of per-session log-ISI hists.
        NaN values fail the threshold check (no promotion).
    wave_badge, depth_badge : str
        Individual badge levels ('pass', 'warn', 'fail').
    threshold : float
        Promotion threshold (default ISI_HIST_CORR_AUTOPASS).

    Returns
    -------
    str
        'trusted' if promotion conditions are met, else unchanged `verdict`.
    """
    if (np.isfinite(isi_hist_corr)
            and isi_hist_corr >= threshold
            and wave_badge != "fail"
            and depth_badge != "fail"):
        return "trusted"
    return verdict


def load_isi_scores(csv_path) -> Dict[int, float]:
    """Read the median ISI corr per global_uid from validate_long_tracks output.

    Missing UIDs are returned as NaN via a defaultdict.
    """
    df = pd.read_csv(csv_path)
    scores = defaultdict(lambda: float("nan"))
    for _, row in df.iterrows():
        scores[int(row["global_uid"])] = float(row["median"])
    return scores


# ─── ISI histogram ────────────────────────────────────────────────────
# Matches the binning used by validate_long_tracks.py (1 ms .. 10 s, log).
_ISI_BIN_EDGES = np.logspace(-3, 1, 51)
_ISI_CENTERS = 0.5 * (_ISI_BIN_EDGES[:-1] + _ISI_BIN_EDGES[1:])


def isi_log_histogram(spike_times: np.ndarray, n_bins: int = 50
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised log-ISI histogram, 1 ms .. 10 s, 50 bins by default.

    Returns
    -------
    h : ndarray, shape (n_bins,)
        Probability mass per bin (sums to 1).  All-NaN if too few spikes.
    centers : ndarray, shape (n_bins,)
        Bin centres (s).
    """
    if n_bins != 50:
        edges = np.logspace(-3, 1, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        edges = _ISI_BIN_EDGES
        centers = _ISI_CENTERS

    if spike_times is None or len(spike_times) < 20:
        return np.full(n_bins, np.nan), centers
    isis = np.diff(np.sort(spike_times))
    isis = isis[(isis > 0) & (isis < 10)]
    if len(isis) < 10:
        return np.full(n_bins, np.nan), centers
    h, _ = np.histogram(isis, bins=edges)
    if h.sum() == 0:
        return np.full(n_bins, np.nan), centers
    return h.astype(float) / h.sum(), centers


# ─── Waveform / footprint extraction ──────────────────────────────────

def extract_peak_channel(mean_waveform: np.ndarray) -> int:
    """Index of the channel with the largest peak-to-peak amplitude.

    Parameters
    ----------
    mean_waveform : ndarray, shape (n_samples, n_channels)
    """
    ptp = mean_waveform.max(axis=0) - mean_waveform.min(axis=0)
    return int(np.argmax(ptp))


def extract_footprint(mean_waveform: np.ndarray, peak_chan: int,
                      halfwidth: int = FOOTPRINT_HALFWIDTH_CHANS
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Footprint snippet: (n_samples, 2*halfwidth+1) clipped at probe edges.

    Returns
    -------
    snippet : ndarray, shape (n_samples, n_channels_kept)
    channel_indices : ndarray, shape (n_channels_kept,)
    """
    n_ch = mean_waveform.shape[1]
    lo = max(0, peak_chan - halfwidth)
    hi = min(n_ch, peak_chan + halfwidth + 1)
    channels = np.arange(lo, hi)
    snippet = mean_waveform[:, lo:hi]
    return snippet, channels


def load_raw_mean_waveform(raw_wf_root, session_name: str, ks_unit_id: int
                            ) -> Optional[np.ndarray]:
    """Load Unit{kid}_RawSpikes.npy and return mean across CV halves.

    Parameters
    ----------
    raw_wf_root : str or Path
        e.g. ``data/unit_match/input/BG_046``
    session_name : str
        DDMMYYYY (8-digit) — matches the unit-match input layout.
    ks_unit_id : int

    Returns
    -------
    mean_waveform : ndarray, shape (n_samples, n_channels), or None if file missing.
    """
    candidates = [session_name, session_name.zfill(8)]
    for cand in candidates:
        path = os.path.join(str(raw_wf_root), cand, "RawWaveforms",
                            f"Unit{ks_unit_id}_RawSpikes.npy")
        if os.path.exists(path):
            raw = np.load(path)   # (n_samples, n_channels, n_cv)
            if raw.ndim == 3:
                return raw.mean(axis=-1).astype(np.float32)
            elif raw.ndim == 2:
                return raw.astype(np.float32)
            return None
    return None


def load_channel_positions(raw_wf_root, session_name: str) -> Optional[np.ndarray]:
    """Load channel_positions.npy for a session.  Shape (n_channels, 2) [x_um, y_um]."""
    for cand in (session_name, session_name.zfill(8)):
        path = os.path.join(str(raw_wf_root), cand, "channel_positions.npy")
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
    return None


# Spec §5 / §4: PSTH conditions per UID per session.
# Keys are stable IDs used as dict keys in the intermediate record.
PSTH_CONDITIONS: Dict[str, Dict] = {
    "baseline_on":        {"event": "Baseline_ON", "outcomes": None,           "sizes": None,       "window": (-2.0, 1.5)},
    "change_on_big_hit":  {"event": "Change_ON",   "outcomes": {"hit"},        "sizes": BIG_POOL,   "window": (-0.5, 0.5)},
    "change_on_big_miss": {"event": "Change_ON",   "outcomes": {"miss"},       "sizes": BIG_POOL,   "window": (-0.5, 0.5)},
    "change_on_sm_hit":   {"event": "Change_ON",   "outcomes": {"hit"},        "sizes": SMALL_POOL, "window": (-0.5, 0.5)},
    "change_on_sm_miss":  {"event": "Change_ON",   "outcomes": {"miss"},       "sizes": SMALL_POOL, "window": (-0.5, 0.5)},
    "hit_lick":           {"event": "Hit",         "outcomes": {"hit"},        "sizes": None,       "window": (-2.0, 1.0)},
    # FA (early/anticipatory) lick, motor-aligned. Window matches hit_lick and spans
    # the canonical FA baseline (-1.75,-1.25) in EVENT_RESPONSIVENESS_WINDOWS so a
    # baseline-subtracted FA PSTH can be taken without re-deriving the window per script.
    "fa_lick":            {"event": "FA",          "outcomes": {"fa"},         "sizes": None,       "window": (-2.0, 1.0)},
}


def _trial_indices_for_sizes(session, sizes: Optional[Set[float]]) -> Optional[List[int]]:
    """Return trial indices whose change_size is in `sizes`, or None for no filter."""
    if sizes is None:
        return None
    out = []
    for i, t in enumerate(session.trials):
        cs = getattr(t, "change_size", None)
        if cs is None:
            continue
        # Match within tolerance because change sizes are floats
        for sz in sizes:
            if abs(float(cs) - sz) < 1e-3:
                out.append(i)
                break
    return out


def extract_unit_psths(session, ks_unit_id: int,
                       restrict_trials: Optional[Set[int]] = None,
                       with_sem: bool = False,
                        ) -> Dict[str, Tuple]:
    """Build PSTHs for all spec conditions for one (session, unit).

    Returns
    -------
    dict[condition_key] -> (psth_smoothed_hz, bin_centers, n_trials)
        psth shape: (n_bins,)
        bin_centers shape: (n_bins,)
        n_trials: int — number of trials averaged
        If no trials match, value is (None, None, 0).

    with_sem : if True, each value is a 4-tuple
        (psth_smoothed_hz, bin_centers, n_trials, sem_smoothed_hz) where
        sem is the across-trial standard error (same smoothing as the mean;
        all-zeros when <2 trials). Empty conditions become (None, None, 0, None).
        Default (False) preserves the 3-tuple contract used by every other
        caller (track_curation, compute_behavior_cache, tests).
    """
    empty = (None, None, 0, None) if with_sem else (None, None, 0)
    out: Dict[str, Tuple] = {}
    for key, cfg in PSTH_CONDITIONS.items():
        trial_idx = _trial_indices_for_sizes(session, cfg["sizes"])
        if restrict_trials is not None:
            allowed = set(int(t) for t in restrict_trials)
            if trial_idx is None:
                trial_idx = sorted(allowed)
            else:
                trial_idx = [i for i in trial_idx if i in allowed]
        if trial_idx is not None and len(trial_idx) == 0:
            out[key] = empty
            continue
        try:
            tensor, centers, valid = build_population_tensor(
                session,
                cluster_ids=[ks_unit_id],
                event_name=cfg["event"],
                window=cfg["window"],
                bin_size=DEFAULT_BIN_SIZE,
                outcome_filter=cfg["outcomes"],
                trial_indices=trial_idx,
            )
        except ValueError:
            out[key] = empty
            continue
        # tensor: (n_trials, n_bins, 1) — collapse units, mean over trials, smooth
        rates = tensor[:, :, 0]                       # (n_trials, n_bins)
        mean_rate = rates.mean(axis=0)
        smoothed = smooth_psth(mean_rate, bin_size=DEFAULT_BIN_SIZE,
                                sigma_ms=DEFAULT_SIGMA_MS)
        if with_sem:
            n_tr = rates.shape[0]
            if n_tr >= 2:
                sem_rate = rates.std(axis=0, ddof=1) / np.sqrt(n_tr)
                sem_sm = smooth_psth(sem_rate, bin_size=DEFAULT_BIN_SIZE,
                                     sigma_ms=DEFAULT_SIGMA_MS)
            else:
                sem_sm = np.zeros_like(smoothed)
            out[key] = (smoothed, centers, len(valid), sem_sm)
        else:
            out[key] = (smoothed, centers, len(valid))
    return out


@dataclass
class SessionRecord:
    """Per-session extracted data for one UID."""
    session_name: str
    ks_unit_id: int
    stage: str
    peak_chan: int
    peak_depth_um: float
    amplitude: float
    baseline_fr_hz: float
    waveform_peak: np.ndarray             # (n_samples,)
    footprint: np.ndarray                 # (n_samples, n_channels_kept)
    footprint_channels: np.ndarray        # (n_channels_kept,)
    isi_hist: np.ndarray                  # (50,)
    isi_centers: np.ndarray               # (50,)
    psths: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = field(default_factory=dict)


@dataclass
class UIDIntermediate:
    """Everything needed to render one UID's QC sheet."""
    global_uid: int
    span: int
    has_naive_to_expert: bool
    suspect_known: bool
    sessions: List[SessionRecord] = field(default_factory=list)


def _compute_baseline_fr(cluster, session) -> float:
    """Per-session baseline firing rate proxy: spikes per second over the cluster's own active span.

    Implementation note: this uses the cluster's last spike time as the denominator
    (not the session-wide recording duration), so units that go silent near the end
    of a session will have an inflated rate. Adequate for QC trending; precise
    per-trial pre-stimulus rates can be computed in a v2.
    """
    if cluster.spike_times is None or len(cluster.spike_times) == 0:
        return float("nan")
    duration = float(cluster.spike_times[-1])
    if duration < 1.0:
        return float("nan")
    return len(cluster.spike_times) / duration


def extract_session_records(session, ks_unit_ids: Sequence[int], session_name: str,
                             stage: str, raw_wf_root, channel_positions: Optional[np.ndarray]
                             ) -> Dict[int, SessionRecord]:
    """Extract per-UID SessionRecord for every (uid, ks_id) in this session.

    Returns a dict keyed by ks_unit_id.  Caller maps ks_id -> global_uid.
    """
    out: Dict[int, SessionRecord] = {}
    cluster_map = {c.cluster_id: c for c in session.clusters}
    for kid in ks_unit_ids:
        cluster = cluster_map.get(int(kid))
        if cluster is None:
            continue

        # Waveform / footprint
        mean_wf = load_raw_mean_waveform(raw_wf_root, session_name, int(kid))
        if mean_wf is None:
            # Cluster exists but no raw waveform file — skip
            continue
        peak_chan = extract_peak_channel(mean_wf)
        peak_wave = mean_wf[:, peak_chan]
        footprint, fp_chans = extract_footprint(mean_wf, peak_chan)

        # Depth & amplitude
        if channel_positions is not None and peak_chan < channel_positions.shape[0]:
            depth_um = float(channel_positions[peak_chan, 1])
        else:
            depth_um = float("nan")
        amplitude = float(peak_wave.max() - peak_wave.min())

        # FR / ISI
        baseline_fr = _compute_baseline_fr(cluster, session)
        spike_times = np.asarray(cluster.spike_times)
        isi_h, isi_c = isi_log_histogram(spike_times)

        # PSTHs (with_sem=True so QC sheets can draw per-session CI95 bands)
        psths = extract_unit_psths(session, int(kid), with_sem=True)

        out[int(kid)] = SessionRecord(
            session_name=session_name,
            ks_unit_id=int(kid),
            stage=stage,
            peak_chan=peak_chan,
            peak_depth_um=depth_um,
            amplitude=amplitude,
            baseline_fr_hz=baseline_fr,
            waveform_peak=peak_wave.astype(np.float32),
            footprint=footprint.astype(np.float32),
            footprint_channels=fp_chans,
            isi_hist=isi_h.astype(np.float32),
            isi_centers=isi_c.astype(np.float32),
            psths=psths,
        )
    return out


# ─── Task 9: cohort selection + cache I/O ─────────────────────────────
KNOWN_SUSPECTS: Set[int] = {779, 873, 872}


def select_long_tracks(unit_index_csv, isi_stats_csv,
                       min_span: int = 10) -> pd.DataFrame:
    """Long-track cohort: UIDs with span >= min_span.

    Span is taken from isi_stats_csv (authoritative). UIDs not present there
    fall back to counting unique sessions in unit_index.

    Returns
    -------
    DataFrame with columns: global_uid, span, sessions, suspect_known
    """
    ui = pd.read_csv(unit_index_csv)
    span_by_uid = ui.groupby("global_uid")["session"].nunique().to_dict()

    if Path(isi_stats_csv).exists():
        stats = pd.read_csv(isi_stats_csv)
        for _, r in stats.iterrows():
            span_by_uid[int(r["global_uid"])] = int(r["span"])

    rows = []
    for uid, span in span_by_uid.items():
        if span < min_span:
            continue
        sessions = ui.loc[ui["global_uid"] == uid, "session"].astype(str).tolist()
        rows.append({
            "global_uid": int(uid),
            "span": int(span),
            "sessions": sessions,
            "suspect_known": int(uid) in KNOWN_SUSPECTS,
        })
    return pd.DataFrame(rows).sort_values("global_uid").reset_index(drop=True)


def annotate_naive_to_expert(cohort: pd.DataFrame, manifest: pd.DataFrame
                              ) -> pd.DataFrame:
    """Add has_naive_to_expert column based on manifest stage assignments.

    A UID is N→E if it spans (any of first 8 sessions) and (any of last 8 sessions).
    Uses chronological order from manifest.session_idx.
    """
    chrono = manifest.sort_values("session_idx").reset_index(drop=True)
    first_eight = set(chrono["session_name"].astype(str).str.zfill(8).head(8))
    last_eight  = set(chrono["session_name"].astype(str).str.zfill(8).tail(8))

    flags = []
    for _, row in cohort.iterrows():
        sess = set(str(s).zfill(8) for s in row["sessions"])
        flags.append(bool(sess & first_eight) and bool(sess & last_eight))
    cohort = cohort.copy()
    cohort["has_naive_to_expert"] = flags
    return cohort


def save_cache(intermediates: Dict[int, UIDIntermediate], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(intermediates, f)


def load_cache(path) -> Optional[Dict[int, UIDIntermediate]]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)



def load_um_pair_scores(um_output_root, uid_to_sessions: Dict[int, List[str]],
                         uid_to_ks: Dict[int, Dict[str, int]]
                         ) -> Dict[int, np.ndarray]:
    """Read batch0/output_prob_matrix.npy + batch0/unit_index.csv, then
    return per-UID arrays of consecutive-session match probabilities.

    Parameters
    ----------
    um_output_root : Path
        e.g. ``X:/.../unit_match/output/all42``
    uid_to_sessions : dict[uid -> chronological list of session names (strings)]
    uid_to_ks : dict[uid -> dict[session_name -> ks_unit_id]]

    Returns
    -------
    dict[uid] -> ndarray of shape (n_sessions_for_uid - 1,)
        Empty array if matrix or rows are missing.
    """
    root = Path(um_output_root)
    matrix_path = root / "batch0" / "output_prob_matrix.npy"
    index_path  = root / "batch0" / "unit_index.csv"
    if not matrix_path.exists() or not index_path.exists():
        return {uid: np.array([]) for uid in uid_to_sessions}

    mat = np.load(matrix_path)
    idx = pd.read_csv(index_path)
    idx["session"] = idx["session"].astype(str)
    lookup: Dict[Tuple[str, int], int] = {}
    for i, row in idx.iterrows():
        lookup[(str(row["session"]), int(row["ks_unit_id"]))] = i

    out = {}
    for uid, sess_list in uid_to_sessions.items():
        ks_map = uid_to_ks.get(uid, {})
        rows = []
        for s in sess_list:
            kid = ks_map.get(s)
            if kid is None:
                rows.append(None)
                continue
            rows.append(lookup.get((s, int(kid))))
        scores = []
        for a, b in zip(rows[:-1], rows[1:]):
            if a is None or b is None:
                scores.append(np.nan)
                continue
            scores.append(float(mat[a, b]))
        out[uid] = np.array(scores, dtype=float)
    return out


# ----- Per-session outlier detection & longest contiguous good-run -----

# Session-outlier thresholds (used by find_stable_subset).
SESSION_ISI_PEAK_TOL_BINS: int = 2
SESSION_FR_MAD_THRESH: float = 3.0
SESSION_WAVE_CORR_THRESH: float = 0.70
SESSION_DEPTH_DEVIATION_UM: float = 30.0


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to std for normal data."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    med = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - med)))


def session_outlier_flags(uid: "UIDIntermediate") -> Dict[str, List[bool]]:
    """For each session in this UID, flag whether it's an outlier on each criterion.

    A session is "bad" if it's an outlier on >=2 criteria, OR if its ISI peak
    bin diverges by >SESSION_ISI_PEAK_TOL_BINS from the modal peak across
    sessions (the strongest single signal of matching failure).

    Returns
    -------
    dict with keys 'isi_peak', 'fr', 'wave', 'depth', 'unknown_stage',
    'is_hard_outlier', 'is_soft_outlier', 'is_outlier' — each a list of
    bools aligned with uid.sessions.

    The classifications:
    - 'isi_peak', 'fr', 'wave', 'depth', 'unknown_stage' are atomic flags
      for each criterion.
    - 'is_outlier' is the existing composite rule: True iff isi_peak OR
      (sum of atomic flags) >= 2 OR unknown_stage. Independent of the
      hard/soft split below.
    - 'is_hard_outlier' = wave OR depth (anatomical/physical mismatch
      signals; never skipped by find_stable_subset).
    - 'is_soft_outlier' = is_outlier AND NOT is_hard_outlier (data-quality
      or transient signals; eligible for skip).
    """
    n = len(uid.sessions)
    out = {
        "isi_peak":         [False] * n,
        "fr":               [False] * n,
        "wave":             [False] * n,
        "depth":            [False] * n,
        "unknown_stage":    [False] * n,
        "is_hard_outlier":  [False] * n,   # NEW: wave OR depth flag set
        "is_soft_outlier":  [False] * n,   # NEW: is_outlier AND NOT is_hard_outlier
        "is_outlier":       [False] * n,
    }
    if n == 0:
        return out

    # ISI peak deviation
    peaks: List[Optional[int]] = []
    for r in uid.sessions:
        h = np.asarray(r.isi_hist, dtype=float)
        if h.size == 0 or np.all(np.isnan(h)):
            peaks.append(None)
        else:
            peaks.append(int(np.argmax(h)))
    valid_peaks = [p for p in peaks if p is not None]
    if len(valid_peaks) >= 2:
        from collections import Counter
        mode_peak = Counter(valid_peaks).most_common(1)[0][0]
        for i, p in enumerate(peaks):
            if p is not None and abs(p - mode_peak) > SESSION_ISI_PEAK_TOL_BINS:
                out["isi_peak"][i] = True

    # FR MAD-based outlier
    rates = np.array([r.baseline_fr_hz for r in uid.sessions], dtype=float)
    finite = np.isfinite(rates)
    if finite.sum() >= 3:
        med = float(np.median(rates[finite]))
        mad = _mad(rates)
        if np.isfinite(mad) and mad > 1e-6:
            for i, rate in enumerate(rates):
                if np.isfinite(rate) and abs(rate - med) / mad > SESSION_FR_MAD_THRESH:
                    out["fr"][i] = True

    # Waveform correlation to median waveform (per session)
    waves = [r.waveform_peak for r in uid.sessions if r.waveform_peak is not None]
    if len(waves) >= 2:
        min_len = min(w.size for w in waves)
        stack = np.stack([w[:min_len] for w in waves]).astype(float)
        median_wave = np.median(stack, axis=0)
        mw_centered = median_wave - median_wave.mean()
        mw_norm = np.linalg.norm(mw_centered)
        if mw_norm > 1e-9:
            mw_unit = mw_centered / mw_norm
            for i, r in enumerate(uid.sessions):
                if r.waveform_peak is None:
                    continue
                w = np.asarray(r.waveform_peak, dtype=float)[:min_len]
                w_centered = w - w.mean()
                w_norm = np.linalg.norm(w_centered)
                if w_norm < 1e-9:
                    continue
                corr = float(np.dot(mw_centered, w_centered) / (mw_norm * w_norm))
                if corr < SESSION_WAVE_CORR_THRESH:
                    out["wave"][i] = True

    # Depth deviation
    depths = np.array([r.peak_depth_um for r in uid.sessions], dtype=float)
    finite_d = np.isfinite(depths)
    if finite_d.sum() >= 2:
        med_d = float(np.median(depths[finite_d]))
        for i, d in enumerate(depths):
            if np.isfinite(d) and abs(d - med_d) > SESSION_DEPTH_DEVIATION_UM:
                out["depth"][i] = True

    # Unknown-stage flag: session whose stage could not be resolved in the manifest.
    # Such sessions are unconditionally treated as outliers to prevent them from
    # anchoring the "good run" window used by find_stable_subset.
    for i, rec in enumerate(uid.sessions):
        if rec.stage == "Unknown":
            out["unknown_stage"][i] = True

    # Composite outlier rule
    for i in range(n):
        strikes = sum([out["isi_peak"][i], out["fr"][i], out["wave"][i], out["depth"][i]])
        # ISI peak divergence alone is sufficient (strongest single signal);
        # otherwise need >=2 criteria; unknown-stage always forces outlier.
        out["is_outlier"][i] = (
            out["isi_peak"][i]
            or strikes >= 2
            or out["unknown_stage"][i]
        )
        # Hard vs soft classification (used by skip-able trimming).
        # Hard = wave or depth outlier — strongly suggests a different physical
        # unit at this probe position; never skip across these. Soft = any
        # other outlier type (unknown_stage, fr, isi_peak) — data-quality or
        # transient issues; cell identity may be intact, eligible for skip.
        out["is_hard_outlier"][i] = out["wave"][i] or out["depth"][i]
        out["is_soft_outlier"][i] = out["is_outlier"][i] and not out["is_hard_outlier"][i]

    return out


def _longest_good_run_contiguous(is_outlier: Sequence[bool]) -> Tuple[int, int]:
    """Return (start_idx, end_idx_exclusive) of the longest contiguous run of
    non-outlier sessions. (0, 0) if no good sessions.

    Internal helper: used as the fallback inside `longest_good_run` when the
    skip-able algorithm cannot find a span whose kept set passes the
    consistency gate."""
    best_start, best_end = 0, 0
    cur_start = None
    arr = list(is_outlier) + [True]  # sentinel
    for i, bad in enumerate(arr):
        if not bad:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                length = i - cur_start
                if length > (best_end - best_start):
                    best_start, best_end = cur_start, i
                cur_start = None
    return best_start, best_end


def longest_good_run(
    is_outlier: Sequence[bool],
    is_hard_outlier: Sequence[bool],
    isi_hists: Sequence[Optional[np.ndarray]],
    *,
    threshold: float = ISI_HIST_CORR_PASS,
) -> Dict[str, List[int]]:
    """Skip-able trim: largest set of non-outlier sessions inside any
    hard-outlier-free span whose set-wide ISI hist correlation passes
    `threshold`.

    Algorithm:
      1. Find all maximal contiguous spans containing NO hard outliers.
      2. For each span, candidate kept_set = sessions in the span that are
         NOT outliers of any kind (soft or hard).
      3. Compute set-wide baseline_isi_hist_corr on kept_set's hists.
      4. If correlation >= threshold (or fewer than 2 kept — gate
         trivially satisfied for size 1; size 0 disqualifies), the span
         qualifies. The skipped_set = soft outliers inside the span.
      5. Pick the span with the LARGEST kept_set (ties → longest span,
         then earliest start).
      6. If NO span qualifies, fall back to longest contiguous all-good
         run (no skipping).

    Returns
    -------
    Dict[str, List[int]] with keys 'kept_indices' (sorted) and
    'skipped_indices' (sorted). Indices outside the chosen span (or
    hard outliers anywhere) are NOT returned by this function — the
    caller computes 'dropped' as the complement.
    """
    n = len(is_outlier)
    if n == 0:
        return {"kept_indices": [], "skipped_indices": []}

    # Step 1: maximal hard-outlier-free spans
    spans: List[Tuple[int, int]] = []  # [(start, end_exclusive), ...]
    cur_start: Optional[int] = None
    for i in range(n):
        if is_hard_outlier[i]:
            if cur_start is not None:
                spans.append((cur_start, i))
                cur_start = None
        else:
            if cur_start is None:
                cur_start = i
    if cur_start is not None:
        spans.append((cur_start, n))

    # Step 2-4: evaluate each span
    best_kept: List[int] = []
    best_skipped: List[int] = []
    best_key: Optional[Tuple[int, int, int]] = None  # (len(kept), span_len, -start)
    for (s, e) in spans:
        kept = [i for i in range(s, e) if not is_outlier[i]]
        skipped = [i for i in range(s, e) if is_outlier[i]]
        if not kept:
            continue
        if len(kept) >= 2:
            kept_hists = [isi_hists[i] for i in kept]
            corr = baseline_isi_hist_corr(kept_hists)
            if not (np.isfinite(corr) and corr >= threshold):
                continue
        # Step 5 tie-breaking via tuple comparison: larger kept_set wins;
        # ties broken by longer span; ties broken by earliest start
        # (negate s so smaller-start gives larger key component).
        span_len = e - s
        candidate_key = (len(kept), span_len, -s)
        if best_key is None or candidate_key > best_key:
            best_kept = kept
            best_skipped = skipped
            best_key = candidate_key

    if best_kept:
        return {"kept_indices": best_kept, "skipped_indices": best_skipped}

    # Step 6: fallback to contiguous-all-good
    # Union is_outlier with is_hard_outlier so the fallback honors the
    # "hard outliers always break runs" invariant (spec §3.2 step 1).
    # Hard outliers can have is_outlier=False (wave-only/depth-only has
    # strikes=1, fails the composite rule), so passing is_outlier alone
    # would let them leak into kept_indices.
    effective_outlier = [a or b for a, b in zip(is_outlier, is_hard_outlier)]
    start, end = _longest_good_run_contiguous(effective_outlier)
    return {"kept_indices": list(range(start, end)), "skipped_indices": []}


def find_stable_subset(uid: "UIDIntermediate") -> Dict[str, object]:
    """Identify a stable kept subset of sessions for this UID, allowing
    skip-over of soft outliers when cross-gap ISI fingerprint consistency
    holds.

    Returns
    -------
    dict with keys:
        outlier_flags    : Dict[str, List[bool]]  (from session_outlier_flags)
        kept_indices     : List[int]              (GOOD sessions in kept span)
        skipped_indices  : List[int]              (soft outliers inside span)
        dropped_indices  : List[int]              (outside span, or hard
                                                    outliers anywhere)
        trimmed_span     : int                    (len of kept_indices)

    Invariants:
        kept ∪ skipped ∪ dropped == range(len(uid.sessions))
        the three sets are pairwise disjoint
    """
    flags = session_outlier_flags(uid)
    isi_hists = [r.isi_hist for r in uid.sessions]
    run = longest_good_run(
        flags["is_outlier"], flags["is_hard_outlier"], isi_hists,
    )
    kept = run["kept_indices"]
    skipped = run["skipped_indices"]
    accounted = set(kept) | set(skipped)
    dropped = [i for i in range(len(uid.sessions)) if i not in accounted]
    return {
        "outlier_flags": flags,
        "kept_indices": kept,
        "skipped_indices": skipped,
        "dropped_indices": dropped,
        "trimmed_span": len(kept),
    }
