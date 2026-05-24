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

DEPTH_PASS_UM: float = 15.0
DEPTH_WARN_UM: float = 30.0

WAVE_PASS_R: float = 0.95
WAVE_WARN_R: float = 0.90

FR_CV_PASS: float = 0.35
FR_CV_WARN: float = 0.60

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
    "baseline_on":        {"event": "Baseline_ON", "outcomes": None,           "sizes": None,       "window": (-0.5, 1.5)},
    "change_on_big_hit":  {"event": "Change_ON",   "outcomes": {"hit"},        "sizes": BIG_POOL,   "window": (-0.5, 0.5)},
    "change_on_big_miss": {"event": "Change_ON",   "outcomes": {"miss"},       "sizes": BIG_POOL,   "window": (-0.5, 0.5)},
    "change_on_sm_hit":   {"event": "Change_ON",   "outcomes": {"hit"},        "sizes": SMALL_POOL, "window": (-0.5, 0.5)},
    "change_on_sm_miss":  {"event": "Change_ON",   "outcomes": {"miss"},       "sizes": SMALL_POOL, "window": (-0.5, 0.5)},
    "hit_lick":           {"event": "Hit",         "outcomes": {"hit"},        "sizes": None,       "window": (-1.0, 1.0)},
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


def extract_unit_psths(session, ks_unit_id: int
                        ) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """Build PSTHs for all spec conditions for one (session, unit).

    Returns
    -------
    dict[condition_key] -> (psth_smoothed_hz, bin_centers, n_trials)
        psth shape: (n_bins,)
        bin_centers shape: (n_bins,)
        n_trials: int — number of trials averaged
        If no trials match, value is (None, None, 0).
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    for key, cfg in PSTH_CONDITIONS.items():
        trial_idx = _trial_indices_for_sizes(session, cfg["sizes"])
        if trial_idx is not None and len(trial_idx) == 0:
            out[key] = (None, None, 0)
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
            out[key] = (None, None, 0)
            continue
        # tensor: (n_trials, n_bins, 1) — collapse units, mean over trials, smooth
        mean_rate = tensor[:, :, 0].mean(axis=0)
        smoothed = smooth_psth(mean_rate, bin_size=DEFAULT_BIN_SIZE,
                                sigma_ms=DEFAULT_SIGMA_MS)
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

        # PSTHs
        psths = extract_unit_psths(session, int(kid))

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
