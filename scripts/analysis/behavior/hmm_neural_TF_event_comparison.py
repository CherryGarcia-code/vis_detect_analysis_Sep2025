"""Per-session multi-event PSTH comparison: TF-responsive vs non-TF neurons.

For each session, produces a figure with:
  - **Rows** = neuron groups (default 3: TF-excited, TF-suppressed, Non-TF;
    configurable via ``--polarity-mode``)
  - **Columns** = event types:
      1. Baseline (aligned to Baseline_ON — stimulus onset)
      2. Hit (big + small Δ overlaid; aligned to change onset or lick,
         controlled by ``--hit-align``)
      3. FA (early — RT ≤ 3.0 s; aligned to FA lick)
      4. FA (late — RT > 3.0 s; aligned to FA lick)
      5. Miss (aligned to stimulus change onset)
      6. TF pulse response (fast or slow, depending on ``--tf-type``)

Each subplot shows the population-average, z-scored PSTH (smoothed) with SEM
shading across units in that group.

Supports parallel session processing via ``ProcessPoolExecutor`` with ``tqdm``
progress bars.  Optional ``--facet-state`` splits trials by HMM state and
generates per-state figures (including state-filtered TF pulse responses).

Usage
-----
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        --data-dir  data/hmm/BG_046 \\
        --pkl-dir   data/pkls/BG_046 \\
        --manifest  data/BG_046_staging_manifest_v2.csv \\
        --tf-dir    FIGURES/tf \\
        --out       FIGURES/behavior/BG_046/hmm/tf_event_comparison \\
        --n-workers 4

    # Align Hit subplot to lick time instead of change onset:
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        ... --hit-align lick

    # Facet by HMM state (one figure per state + all-trials):
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        ... --facet-state

    # 2-row simple mode (TF-responsive vs Non-TF):
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        ... --polarity-mode simple

    # 5-row full mode (TF-fast-exc, TF-fast-sup, TF-slow-exc, TF-slow-sup, Non-TF):
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        ... --polarity-mode full

    # Shared y-axis across event columns:
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        ... --shared-yaxis

    # Replot from cached PSTHs (skip session loading + spike alignment):
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        ... --replot-only

    # Classify by lick responsiveness instead of TF:
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        ... --classify-by lick --lick-dir FIGURES/lick/BG_046

    # Classify by lick using Hit + FA licks (not just FA):
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        ... --classify-by lick --lick-dir FIGURES/lick/BG_046 --lick-events both

    # Add lick-classified rows below TF rows in the same figure:
    python scripts/analysis/behavior/hmm_neural_TF_event_comparison.py \\
        ... --add-lick-rows --lick-dir FIGURES/lick/BG_046
"""

import argparse
import pickle
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # noqa: D103 — minimal fallback
        return it

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from visdetect.core.session import load_session
from visdetect.analysis.align import (
    align_spikes_to_events,
    get_event_times_by_trial,
)
from visdetect.analysis.config import load_staging_manifest
from visdetect.analysis.hmm_downstream import (
    load_hmm_results,
    smooth_psth,
)
from visdetect.analysis.constants import (
    BIG_CHANGE_SIZES,
    SMALL_CHANGE_SIZES,
    FA_RT_SPLIT,
    TF_PULSE_PRE_WINDOW,
    TF_PULSE_POST_WINDOW,
    TF_PULSE_WINDOW,
    TF_FAST_THRESH_LOG2,
    TF_SLOW_THRESH_LOG2,
    TF_SAMPLE_PERIOD,
)
from visdetect.viz.plotting import set_style, despine
import matplotlib.colors as mcolors

# Base columns for the figure (label, internal key).
# Hit big/small are OVERLAID on each Hit subplot.
# Both change-aligned and lick-aligned Hit columns are shown by default.
# TF pulse column(s) are appended dynamically by _get_event_columns().
_EVENT_COLUMNS_BASE = [
    ("Baseline", "baseline"),
    ("Hit (@ Δ)", "hit_change"),
    ("Hit (@ lick)", "hit_lick"),
    ("FA (early)", "fa_early"),
    ("FA (late)", "fa_late"),
    ("Miss", "miss"),
]


def _get_event_columns(tf_type: str) -> List[Tuple[str, str]]:
    """Return the column list for the figure, with TF column(s) appended.

    When ``tf_type='both'``, two TF columns (fast + slow) are shown
    side-by-side.  Otherwise a single TF pulse column is shown.
    """
    cols = list(_EVENT_COLUMNS_BASE)
    if tf_type == "both":
        cols.append(("TF fast", "tf_pulse_fast"))
        cols.append(("TF slow", "tf_pulse_slow"))
    else:
        cols.append(("TF pulse", "tf_pulse"))
    return cols


# Map hit column keys to their (big, small) sub-keys in the psths dict.
HIT_COL_SUBKEYS = {
    "hit_change": ("hit_big_change", "hit_small_change"),
    "hit_lick":   ("hit_big_lick",   "hit_small_lick"),
}

# Alignment reference labels shown beneath column titles.
EVENT_ALIGN_LABELS = {
    "baseline": "@ stim onset",
    "hit_change": "@ change onset",
    "hit_lick": "@ lick",
    "fa_early": "@ FA lick",
    "fa_late": "@ FA lick",
    "miss": "@ change onset",
    "tf_pulse": "@ TF pulse",
    "tf_pulse_fast": "@ TF fast pulse",
    "tf_pulse_slow": "@ TF slow pulse",
}

# Baseline lick-contamination filter: exclude Baseline_ON events where any
# lick occurs within this many seconds after baseline onset.
BASELINE_LICK_EXCLUSION_WINDOW = 1.0
LICK_NI_KEYS = ["Lick_L", "lick_L"]  # NI event keys for raw lick times


# =====================================================================
# Dataclass for per-session config
# =====================================================================

@dataclass
class SessionTask:
    """Everything needed to process one session (picklable for parallel)."""
    pkl_path: str
    session_name: str
    tf_csv_path: Optional[str]
    tf_dir: str                  # root TF screening directory (e.g. FIGURES/tf)
    assignments_rows: list   # list-of-dicts for this session
    out_dir: str
    z_thresh_tf: float = 3.0
    min_fr: float = 1.0
    tf_type: str = "fast"    # fast / slow / both (for TF pulse column(s))
    polarity_mode: str = "split"   # split / full / simple
    shared_yaxis: bool = False
    sigma_ms: float = 25.0
    bin_size: float = 0.025
    window_behavioral: Tuple[float, float] = (-0.5, 1.0)
    window_tf: Tuple[float, float] = TF_PULSE_WINDOW
    min_units: int = 3
    normalize: str = "zscore"
    replot_only: bool = False    # skip computation, re-draw from cached PSTHs
    shared_overlay_yaxis: bool = False  # enforce shared y-axis across overlay row columns
    # Classification mode
    classify_by: str = "tf"      # "tf" or "lick"
    lick_csv_path: Optional[str] = None   # path to lick_responsiveness.csv
    lick_dir: str = ""           # root lick directory (e.g. FIGURES/lick/BG_046)
    lick_events: str = "fa"      # "fa", "hit", or "both"
    add_lick_rows: bool = False   # append lick-classified rows below TF rows
    rt_shift_s: float = 0.0       # shift lick-aligned RTs by this many seconds (negative = earlier)
    # State faceting
    facet_state: bool = False
    state_labels: List[str] = field(default_factory=list)


# =====================================================================
# Unit classification
# =====================================================================

def _classify_units(
    session,
    tf_csv_path: Optional[str],
    z_thresh: float,
    min_fr: float,
    polarity_mode: str,
) -> Dict[str, List[int]]:
    """Classify quality units into neuron groups.

    Returns
    -------
    dict mapping group label → list of cluster IDs.
    Groups depend on *polarity_mode*:
      - ``'split'``: TF-excited, TF-suppressed, Non-TF
      - ``'full'``:  TF-fast-exc, TF-fast-sup, TF-slow-exc, TF-slow-sup, Non-TF
      - ``'simple'``: TF-responsive, Non-TF
    """
    quality_ids = set(
        session.good_and_stable_ids
        or session.good_cluster_ids
        or [c.cluster_id for c in session.clusters]
    )

    # Filter by minimum firing rate
    fr_ok: set = set()
    for c in session.clusters:
        cid = c.cluster_id
        if cid not in quality_ids:
            continue
        st = c.spike_times
        if st is None or len(st) == 0:
            continue
        dur = float(st[-1] - st[0])
        if dur < 1e-6:
            continue
        if len(st) / dur >= min_fr:
            fr_ok.add(cid)

    # Load TF responsiveness CSV if available
    tf_responsive: Dict[str, set] = defaultdict(set)  # group_key → set of cids
    if tf_csv_path is not None and Path(tf_csv_path).exists():
        tf_df = pd.read_csv(tf_csv_path)
        for _, row in tf_df.iterrows():
            cid = int(row["cluster_id"])
            if cid not in fr_ok:
                continue
            z_max_f = abs(row.get("z_max_fast", 0.0))
            z_min_f = abs(row.get("z_min_fast", 0.0))
            z_max_s = abs(row.get("z_max_slow", 0.0))
            z_min_s = abs(row.get("z_min_slow", 0.0))

            fast_resp = z_max_f >= z_thresh or z_min_f >= z_thresh
            slow_resp = z_max_s >= z_thresh or z_min_s >= z_thresh

            if polarity_mode == "simple":
                if fast_resp or slow_resp:
                    tf_responsive["TF-responsive"].add(cid)
            elif polarity_mode == "full":
                if fast_resp:
                    # Polarity: excited if peak > |trough|, else suppressed
                    if row.get("z_max_fast", 0.0) >= abs(row.get("z_min_fast", 0.0)):
                        tf_responsive["TF-fast-exc"].add(cid)
                    else:
                        tf_responsive["TF-fast-sup"].add(cid)
                if slow_resp:
                    if row.get("z_max_slow", 0.0) >= abs(row.get("z_min_slow", 0.0)):
                        tf_responsive["TF-slow-exc"].add(cid)
                    else:
                        tf_responsive["TF-slow-sup"].add(cid)
            else:  # "split"
                any_resp = fast_resp or slow_resp
                if any_resp:
                    # Determine polarity from whichever response is larger
                    peak = max(z_max_f, z_max_s)
                    trough = max(z_min_f, z_min_s)
                    if peak >= trough:
                        tf_responsive["TF-excited"].add(cid)
                    else:
                        tf_responsive["TF-suppressed"].add(cid)

    # Non-TF: quality units that are NOT in any TF group
    all_tf = set()
    for s in tf_responsive.values():
        all_tf |= s
    non_tf = fr_ok - all_tf

    # Build ordered result
    groups: Dict[str, List[int]] = {}
    if polarity_mode == "simple":
        groups["TF-responsive"] = sorted(tf_responsive.get("TF-responsive", []))
        groups["Non-TF"] = sorted(non_tf)
    elif polarity_mode == "full":
        for key in ["TF-fast-exc", "TF-fast-sup", "TF-slow-exc", "TF-slow-sup"]:
            groups[key] = sorted(tf_responsive.get(key, []))
        groups["Non-TF"] = sorted(non_tf)
    else:  # "split"
        groups["TF-excited"] = sorted(tf_responsive.get("TF-excited", []))
        groups["TF-suppressed"] = sorted(tf_responsive.get("TF-suppressed", []))
        groups["Non-TF"] = sorted(non_tf)

    return groups


# =====================================================================
# Lick-responsiveness classification
# =====================================================================

def _get_lick_classification_events(
    session,
    mode: str = "fa",
) -> List[float]:
    """Extract lick event times used for classifying lick responsiveness.

    Parameters
    ----------
    mode : ``'fa'`` | ``'hit'`` | ``'both'``
        - ``fa``: First-lick on late FA trials (FA RT >= 3 s), aligned to
          Baseline_ON + FA RT.  Matches the existing MatlabLickAnalyzer.
        - ``hit``: Hit-trial lick times (change_size > 1), aligned to
          Change_ON + RT.
        - ``both``: Pool FA + Hit lick events together.
    """
    trials = getattr(session, "trials", []) or []
    ni = getattr(session, "ni_events", {}) or {}
    baseline_arr = _to_float_array(ni.get("Baseline_ON", []))
    change_arr = np.array(
        get_event_times_by_trial(session, "Change_ON"), dtype=float,
    )

    fa_events: List[float] = []
    hit_events: List[float] = []

    for i, trial in enumerate(trials):
        outcome = (trial.trialoutcome or "").lower()
        rt_dict = trial.reactiontimes or {}

        if outcome == "fa" and mode in ("fa", "both"):
            fa_rt = rt_dict.get("FA")
            if (
                fa_rt is not None
                and np.isfinite(fa_rt)
                and float(fa_rt) >= 3.0
                and i < len(baseline_arr)
                and np.isfinite(baseline_arr[i])
            ):
                fa_events.append(float(baseline_arr[i] + fa_rt))

        if outcome == "hit" and mode in ("hit", "both"):
            cs = getattr(trial, "change_size", None) or getattr(
                trial, "changescale", None
            )
            if cs is not None and float(cs) > 1.0:
                rt = rt_dict.get("RT")
                if (
                    rt is not None
                    and np.isfinite(rt)
                    and i < len(change_arr)
                    and np.isfinite(change_arr[i])
                ):
                    hit_events.append(float(change_arr[i] + rt))

    if mode == "fa":
        return fa_events
    elif mode == "hit":
        return hit_events
    else:  # both
        return fa_events + hit_events


def _classify_units_by_lick(
    session,
    lick_csv_path: Optional[str],
    min_fr: float,
    lick_events_mode: str = "fa",
) -> Dict[str, List[int]]:
    """Classify quality units into lick-responsiveness groups.

    Returns dict mapping group label → list of cluster IDs:
      - ``Lick-excited`` : significant (p < 0.05) and delta_mean > 0
      - ``Lick-inhibited``: significant (p < 0.05) and delta_mean < 0
      - ``Non-lick``      : not significant

    When ``lick_events_mode='fa'`` and ``lick_csv_path`` exists, reads
    pre-computed results from CSV.  Otherwise computes on-the-fly.
    """
    # --- Quality / FR filter (same as TF classifier) ---
    quality_ids = set(
        session.good_and_stable_ids
        or session.good_cluster_ids
        or [c.cluster_id for c in session.clusters]
    )
    fr_ok: set = set()
    for c in session.clusters:
        cid = c.cluster_id
        if cid not in quality_ids:
            continue
        st = c.spike_times
        if st is None or len(st) == 0:
            continue
        dur = float(st[-1] - st[0])
        if dur < 1e-6:
            continue
        if len(st) / dur >= min_fr:
            fr_ok.add(cid)

    groups: Dict[str, List[int]] = {
        "Lick-excited": [],
        "Lick-inhibited": [],
        "Non-lick": [],
    }

    # --- Try CSV (only valid for "fa" mode) ---
    if (
        lick_events_mode == "fa"
        and lick_csv_path is not None
        and Path(lick_csv_path).exists()
    ):
        df = pd.read_csv(lick_csv_path)
        classified = set()
        for _, row in df.iterrows():
            cid = int(row["cluster_id"])
            if cid not in fr_ok:
                continue
            sig = bool(row.get("is_significant", False))
            delta = float(row.get("delta_mean", 0.0))
            if sig and delta > 0:
                groups["Lick-excited"].append(cid)
            elif sig and delta < 0:
                groups["Lick-inhibited"].append(cid)
            else:
                groups["Non-lick"].append(cid)
            classified.add(cid)
        # Units in fr_ok but absent from CSV → Non-lick
        for cid in sorted(fr_ok - classified):
            groups["Non-lick"].append(cid)
        for k in groups:
            groups[k] = sorted(groups[k])
        return groups

    # --- On-the-fly computation (hit / both / fa-without-csv) ---
    from visdetect.analysis.lick import MatlabLickConfig, MatlabLickAnalyzer

    lick_events = _get_lick_classification_events(session, mode=lick_events_mode)
    if len(lick_events) < 5:
        # Not enough events — everything is Non-lick
        groups["Non-lick"] = sorted(fr_ok)
        return groups

    cfg = MatlabLickConfig()
    analyzer = MatlabLickAnalyzer(cfg)
    edges = cfg.time_edges

    for c in session.clusters:
        cid = c.cluster_id
        if cid not in fr_ok:
            continue
        spikes = np.asarray(c.spike_times, dtype=float)
        spikes = spikes[np.isfinite(spikes)]
        if spikes.size == 0:
            groups["Non-lick"].append(cid)
            continue
        matrix = analyzer._build_psth_matrix(spikes, lick_events, edges)
        if matrix is None or matrix.shape[0] < cfg.min_events:
            groups["Non-lick"].append(cid)
            continue
        stats = analyzer._compute_stats(matrix, cid)
        if stats["is_significant"] and stats["delta_mean"] > 0:
            groups["Lick-excited"].append(cid)
        elif stats["is_significant"] and stats["delta_mean"] < 0:
            groups["Lick-inhibited"].append(cid)
        else:
            groups["Non-lick"].append(cid)

    for k in groups:
        groups[k] = sorted(groups[k])
    return groups


# =====================================================================
# Event extraction helpers
# =====================================================================

def _get_behavioral_event_times(
    session,
    assignments_df: pd.DataFrame,
    session_name: str,
    trial_indices: Optional[set] = None,
    rt_shift_s: float = 0.0,
) -> Dict[str, List[float]]:
    """Extract per-event-type global times for behavioural events.

    Parameters
    ----------
    trial_indices : optional set of 0-based trial indices.
        When provided, only trials whose index is in this set are
        included.  Used for HMM state faceting.
    rt_shift_s : float
        Additive shift (in seconds) applied to lick-aligned events
        (FA early/late, and Hit @ lick).  Negative values move the
        alignment earlier in time (e.g. ``-0.2`` shifts 200 ms before
        the detected lick).  Default 0.

    Returns dict with keys:
        baseline, hit_big_change, hit_small_change, hit_big_lick,
        hit_small_lick, fa_early, fa_late, miss.
    Each value is a list of absolute event times (seconds).

    Hit events are returned in **both** change-aligned and lick-aligned
    forms so both columns can be plotted side-by-side.

    **Baseline lick filter:** Baseline_ON events are excluded when any
    raw lick (NI ``Lick_L`` / ``lick_L``) occurs within
    ``BASELINE_LICK_EXCLUSION_WINDOW`` seconds (default 1.0 s) of
    baseline onset, to avoid contamination from preparatory motor activity.
    """
    trials = getattr(session, "trials", []) or []
    ni = getattr(session, "ni_events", {}) or {}

    # Get Baseline_ON and Change_ON per trial
    baseline_arr = _to_float_array(ni.get("Baseline_ON", []))
    change_arr = np.array(get_event_times_by_trial(session, "Change_ON"), dtype=float)

    # Raw lick event times (for baseline contamination filter)
    _lick_parts: List[np.ndarray] = []
    for _lk in LICK_NI_KEYS:
        if _lk in ni:
            _lick_parts.append(_to_float_array(ni[_lk]))
    all_lick_times = (
        np.sort(np.concatenate(_lick_parts)) if _lick_parts else np.array([])
    )

    n = len(trials)
    result: Dict[str, List[float]] = {
        "baseline": [],
        "hit_big_change": [], "hit_small_change": [],
        "hit_big_lick": [], "hit_small_lick": [],
        "fa_early": [], "fa_late": [], "miss": [],
    }

    for i in range(n):
        # State-faceting filter: skip trials not assigned to the target state
        if trial_indices is not None and i not in trial_indices:
            continue
        trial = trials[i]
        outcome = (getattr(trial, "trialoutcome", "") or "").lower()
        cs = getattr(trial, "change_size", None)
        try:
            cs = float(cs)
        except (TypeError, ValueError):
            cs = None

        # Baseline_ON event — exclude if any lick occurs within
        # BASELINE_LICK_EXCLUSION_WINDOW seconds after baseline onset.
        if i < len(baseline_arr) and np.isfinite(baseline_arr[i]):
            btime = float(baseline_arr[i])
            lick_contaminated = False
            if all_lick_times.size > 0:
                # Binary search for efficiency
                idx = np.searchsorted(all_lick_times, btime)
                # Check licks from idx onwards that fall within the window
                while idx < all_lick_times.size:
                    lt = all_lick_times[idx]
                    if lt > btime + BASELINE_LICK_EXCLUSION_WINDOW:
                        break
                    if lt >= btime:
                        lick_contaminated = True
                        break
                    idx += 1
            if not lick_contaminated:
                result["baseline"].append(btime)

        if outcome == "hit" and cs is not None and cs > 1.0:
            # Change-aligned hit
            if i < len(change_arr) and np.isfinite(change_arr[i]):
                if cs in BIG_CHANGE_SIZES:
                    result["hit_big_change"].append(float(change_arr[i]))
                elif cs in SMALL_CHANGE_SIZES:
                    result["hit_small_change"].append(float(change_arr[i]))
            # Lick-aligned hit
            rts = getattr(trial, "reactiontimes", {}) or {}
            rt_val = rts.get("RT", None)
            if rt_val is not None:
                try:
                    rt_val = float(rt_val)
                except (TypeError, ValueError):
                    rt_val = None
            if (
                rt_val is not None
                and np.isfinite(rt_val)
                and i < len(change_arr)
                and np.isfinite(change_arr[i])
            ):
                t_event = float(change_arr[i] + rt_val + rt_shift_s)
                if cs in BIG_CHANGE_SIZES:
                    result["hit_big_lick"].append(t_event)
                elif cs in SMALL_CHANGE_SIZES:
                    result["hit_small_lick"].append(t_event)

        elif outcome == "miss" and cs is not None and cs > 1.0:
            # Align to Change_ON
            if i < len(change_arr) and np.isfinite(change_arr[i]):
                result["miss"].append(float(change_arr[i]))

        elif outcome == "fa":
            # FA: align to lick time = baseline + RT
            rts = getattr(trial, "reactiontimes", {}) or {}
            rt_val = rts.get("FA", None)
            if rt_val is None:
                continue
            try:
                rt_val = float(rt_val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(rt_val):
                continue
            if i < len(baseline_arr) and np.isfinite(baseline_arr[i]):
                t_event = float(baseline_arr[i] + rt_val + rt_shift_s)
                if rt_val <= FA_RT_SPLIT:
                    result["fa_early"].append(t_event)
                else:
                    result["fa_late"].append(t_event)

    return result


def _load_tf_pulse_times(
    tf_dir: Path,
    session_name: str,
    tf_type: str = "fast",
    subject: str = "BG_046",
) -> List[float]:
    """Load pre-computed TF pulse times from the saved CSV.

    The TF pipeline saves ``tf_pulse_times.csv`` (with ``fast_times`` and
    ``slow_times`` columns) inside each session's TF output directory.
    This avoids re-extracting pulse times from scratch.

    Falls back to ``visdetect.analysis.tf_pulse._collect_pulses`` when
    no CSV is found (requires the session object, handled by caller).
    """
    col = "fast_times" if tf_type == "fast" else "slow_times"

    # Try common naming patterns for TF output directory
    for d in [tf_dir / f"{subject}_{session_name}", tf_dir / session_name]:
        csv_path = d / "tf_pulse_times.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if col in df.columns:
                return df[col].dropna().astype(float).tolist()
    return []  # caller should fall back


def _collect_tf_pulse_times_fallback(session, tf_type: str = "fast") -> List[float]:
    """Compute TF pulse times from session data using the canonical function.

    Uses ``visdetect.analysis.tf_pulse._collect_pulses`` so the logic
    is not duplicated.
    """
    try:
        from visdetect.analysis.tf_pulse import _collect_pulses, TFRespPulseConfig
        cfg = TFRespPulseConfig()
        fast_times, slow_times = _collect_pulses(session, cfg)
        if tf_type == "fast":
            return fast_times.tolist()
        else:
            return slow_times.tolist()
    except Exception as exc:
        print(f"    WARN: TF pulse fallback failed: {exc}")
        return []


def _to_float_array(x) -> np.ndarray:
    """Robustly convert NI event data to a 1-D float array."""
    if isinstance(x, dict):
        if "rise_t" in x:
            x = x["rise_t"]
        elif "times" in x:
            x = x["times"]
        else:
            return np.array([], dtype=float)
    return np.asarray(x, dtype=float).flatten()


def _filter_tf_pulses_by_trials(
    tf_pulse_times: List[float],
    baseline_arr: np.ndarray,
    trial_indices: set,
) -> List[float]:
    """Filter TF pulse times to only those falling within specified trials.

    A TF pulse is considered to belong to trial *i* when it falls in the
    interval ``[Baseline_ON[i], Baseline_ON[i+1])`` (last trial extends to
    infinity).  Only pulses inside trials whose index is in *trial_indices*
    are retained.

    This allows per-state TF pulse PSTHs when ``--facet-state`` is used.
    """
    if len(tf_pulse_times) == 0 or len(baseline_arr) == 0:
        return tf_pulse_times  # nothing to filter

    # Build sorted array of trial boundaries
    valid_mask = np.isfinite(baseline_arr)
    if not valid_mask.any():
        return tf_pulse_times

    # Assemble trial start times (sorted by index)
    n_trials = len(baseline_arr)
    trial_starts = np.full(n_trials, np.nan)
    trial_starts[valid_mask] = baseline_arr[valid_mask]

    # Build accepted intervals from trial_indices
    # Each trial spans [baseline[i], baseline[i+1]), last trial to +inf
    tf_arr = np.array(tf_pulse_times, dtype=float)
    keep = np.zeros(len(tf_arr), dtype=bool)

    sorted_indices = sorted(trial_indices)
    for ti in sorted_indices:
        if ti >= n_trials or np.isnan(trial_starts[ti]):
            continue
        t_start = trial_starts[ti]
        # Find next trial start for upper bound
        t_end = np.inf
        for j in range(ti + 1, n_trials):
            if np.isfinite(trial_starts[j]):
                t_end = trial_starts[j]
                break
        mask = (tf_arr >= t_start) & (tf_arr < t_end)
        keep |= mask

    return tf_arr[keep].tolist()


# =====================================================================
# Population PSTH computation
# =====================================================================

def _compute_population_psth(
    session,
    cluster_ids: List[int],
    event_times: List[float],
    window: Tuple[float, float],
    bin_size: float,
    sigma_ms: float,
    normalize: str = "zscore",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute population-average PSTH for a set of units at given events.

    Returns (mean_psth, sem_psth, bin_centers) or None if not enough data.
    Each unit contributes one mean PSTH (z-scored by default).
    """
    if len(event_times) < 3 or len(cluster_ids) < 1:
        return None

    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bc = (bins[:-1] + bins[1:]) / 2.0

    unit_psths = []
    for cid in cluster_ids:
        spike_times = None
        for c in session.clusters:
            if c.cluster_id == cid:
                spike_times = c.spike_times
                break
        if spike_times is None or len(spike_times) == 0:
            continue

        fr_matrix, _ = align_spikes_to_events(
            spike_times, event_times, window=window, bin_size=bin_size,
        )
        if fr_matrix.shape[0] == 0:
            continue

        mean_fr = fr_matrix.mean(axis=0)
        smoothed = smooth_psth(mean_fr, bin_size=bin_size, sigma_ms=sigma_ms)

        # Normalize
        if normalize in ("zscore", "baseline-subtract"):
            bl_mask = bc < 0.0
            if bl_mask.any():
                mu = smoothed[bl_mask].mean()
                if normalize == "zscore":
                    sd = smoothed[bl_mask].std()
                    if sd > 1e-6:
                        smoothed = (smoothed - mu) / sd
                    else:
                        # Baseline too flat to z-score — skip this unit
                        # to avoid mixing raw-Hz values into the z-scored
                        # population average.
                        continue
                else:
                    smoothed = smoothed - mu

        unit_psths.append(smoothed)

    if len(unit_psths) < 1:
        return None

    mat = np.array(unit_psths)
    mean_psth = mat.mean(axis=0)
    sem_psth = mat.std(axis=0) / np.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.zeros_like(mean_psth)
    return mean_psth, sem_psth, bc


# =====================================================================
# PSTH cache helpers (for --replot-only)
# =====================================================================

def _build_psth_cache(
    session,
    groups: Dict[str, List[int]],
    event_times: Dict[str, List[float]],
    tf_pulse_times: Dict[str, List[float]],
    task: SessionTask,
    state_label: Optional[str] = None,
) -> dict:
    """Pre-compute every PSTH needed for a figure and return a cache dict.

    Uses a **pre-compute–then-aggregate** strategy: each unit's normalised
    PSTH is computed *once* per event type, then shared across all groups
    that contain that unit.  With ``--add-lick-rows`` this avoids redundant
    ``align_spikes_to_events`` calls (≈2× faster).

    The cache contains all data that ``_generate_session_figure`` needs, so
    figures can be regenerated without loading the session pkl.

    Parameters
    ----------
    tf_pulse_times : dict mapping TF column key (``'tf_pulse'``,
        ``'tf_pulse_fast'``, ``'tf_pulse_slow'``) to list of pulse times.
    """
    row_labels = [k for k in groups if len(groups[k]) >= task.min_units]

    # -- 1. Collect every unique cluster ID across all groups ---------------
    all_cids: set = set()
    for cids in groups.values():
        all_cids.update(cids)

    # Build spike-time lookup (avoids O(n) scan per cid later)
    spike_lookup: Dict[int, np.ndarray] = {}
    for c in session.clusters:
        if c.cluster_id in all_cids:
            st = c.spike_times
            if st is not None and len(st) > 0:
                spike_lookup[c.cluster_id] = np.asarray(st, dtype=float)

    # -- 2. Define event types → (event_list, window) ----------------------
    event_columns = _get_event_columns(task.tf_type)
    event_specs: Dict[str, Tuple[List[float], Tuple[float, float]]] = {}
    for _, col_key in event_columns:
        if col_key in HIT_COL_SUBKEYS:
            big_key, small_key = HIT_COL_SUBKEYS[col_key]
            event_specs[big_key] = (
                event_times.get(big_key, []), task.window_behavioral)
            event_specs[small_key] = (
                event_times.get(small_key, []), task.window_behavioral)
        elif col_key.startswith("tf_pulse"):
            event_specs[col_key] = (
                tf_pulse_times.get(col_key, []), task.window_tf)
        else:
            event_specs[col_key] = (
                event_times.get(col_key, []), task.window_behavioral)

    # Pre-compute bin-centres for each distinct window
    bc_by_window: Dict[Tuple[float, float], np.ndarray] = {}
    for _, window in event_specs.values():
        if window not in bc_by_window:
            bins = np.arange(window[0], window[1] + task.bin_size, task.bin_size)
            bc_by_window[window] = (bins[:-1] + bins[1:]) / 2.0

    # -- 3. Pre-compute per-unit normalised PSTHs --------------------------
    #   Key: (cluster_id, event_key)  →  smoothed 1-D array
    unit_psth: Dict[Tuple[int, str], np.ndarray] = {}

    for evt_key, (evts, window) in event_specs.items():
        if len(evts) < 3:
            continue
        bc = bc_by_window[window]
        bl_mask = bc < 0.0

        for cid in all_cids:
            if cid not in spike_lookup:
                continue
            fr_matrix, _ = align_spikes_to_events(
                spike_lookup[cid], evts, window=window,
                bin_size=task.bin_size,
            )
            if fr_matrix.shape[0] == 0:
                continue
            mean_fr = fr_matrix.mean(axis=0)
            smoothed = smooth_psth(
                mean_fr, bin_size=task.bin_size, sigma_ms=task.sigma_ms,
            )
            # Normalise
            if task.normalize in ("zscore", "baseline-subtract"):
                if bl_mask.any():
                    mu = smoothed[bl_mask].mean()
                    if task.normalize == "zscore":
                        sd = smoothed[bl_mask].std()
                        if sd > 1e-6:
                            smoothed = (smoothed - mu) / sd
                        else:
                            continue  # skip flat-baseline units
                    else:
                        smoothed = smoothed - mu
            unit_psth[(cid, evt_key)] = smoothed

    # -- 4. Aggregate per-unit PSTHs by group ------------------------------
    psths: dict = {}
    for grp_label in row_labels:
        cids = groups[grp_label]
        for _, col_key in event_columns:
            sub_keys = (list(HIT_COL_SUBKEYS[col_key])
                        if col_key in HIT_COL_SUBKEYS else [col_key])
            for sk in sub_keys:
                window = event_specs.get(
                    sk, ([], task.window_behavioral))[1]
                bc = bc_by_window.get(window)
                traces = [unit_psth[(c, sk)]
                          for c in cids if (c, sk) in unit_psth]
                if not traces or bc is None:
                    psths[(grp_label, sk)] = None
                    continue
                mat = np.array(traces)
                mean_p = mat.mean(axis=0)
                sem_p = (mat.std(axis=0) / np.sqrt(mat.shape[0])
                         if mat.shape[0] > 1
                         else np.zeros_like(mean_p))
                psths[(grp_label, sk)] = (mean_p, sem_p, bc)

    # -- 5. Event counts ---------------------------------------------------
    event_counts: Dict[str, int] = {}
    for key in ("baseline",
                "hit_big_change", "hit_small_change",
                "hit_big_lick", "hit_small_lick",
                "fa_early", "fa_late", "miss"):
        event_counts[key] = len(event_times.get(key, []))
    for tf_key, tf_times in tf_pulse_times.items():
        event_counts[tf_key] = len(tf_times)

    return {
        "groups": groups,
        "event_counts": event_counts,
        "psths": psths,
        "state_label": state_label,
    }


def _cache_path(out_dir: str, session_name: str,
                state_label: Optional[str] = None) -> Path:
    """Return the path to a PSTH cache pickle for one figure."""
    suffix = f"_{state_label.lower().replace(' ', '_')}" if state_label else ""
    return Path(out_dir) / session_name / f"psth_cache{suffix}.pkl"


def _save_psth_cache(cache_data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(cache_data, fh, protocol=4)


def _load_psth_cache(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# =====================================================================
# Per-session figure generation
# =====================================================================

ROW_PALETTE = {
    # TF groups
    "TF-excited": "#d62728",
    "TF-suppressed": "#1f77b4",
    "TF-responsive": "#9467bd",
    "TF-fast-exc": "#e41a1c",
    "TF-fast-sup": "#377eb8",
    "TF-slow-exc": "#ff7f00",
    "TF-slow-sup": "#4daf4a",
    "Non-TF": "#7f7f7f",
    # Lick groups
    "Lick-excited": "#ff7f0e",
    "Lick-inhibited": "#2ca02c",
    "Non-lick": "#bcbd22",
}


def _generate_session_figure(
    psth_cache: dict,
    task: SessionTask,
    state_label: Optional[str] = None,
) -> str:
    """Create and save the multi-panel figure for one session.

    Parameters
    ----------
    psth_cache : dict produced by ``_build_psth_cache`` (or loaded from disk).
        Keys: ``"groups"``, ``"event_counts"``, ``"psths"``.
    state_label : optional label for the HMM state slice (e.g. "engaged").
        When set, appended to the title and filename.

    Returns the path to the saved figure.
    """
    groups = psth_cache["groups"]
    event_counts = psth_cache["event_counts"]
    psths = psth_cache["psths"]

    row_labels = [k for k in groups if len(groups[k]) >= task.min_units]
    if not row_labels:
        return ""

    event_columns = _get_event_columns(task.tf_type)
    n_grp_rows = len(row_labels)
    n_cols = len(event_columns)
    # Extra overlay row at the bottom (all groups on one set of axes)
    n_rows = n_grp_rows + 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharey="row" if task.shared_yaxis else False,
    )

    for ri, grp_label in enumerate(row_labels):
        cids = groups[grp_label]
        color = ROW_PALETTE.get(grp_label, "#333333")

        for ci, (col_title, col_key) in enumerate(event_columns):
            ax = axes[ri, ci]

            if col_key in HIT_COL_SUBKEYS:
                # ---- Overlay big and small Δ on the same subplot ----
                big_key, small_key = HIT_COL_SUBKEYS[col_key]
                # Lighter shade for small Δ
                rgb = mcolors.to_rgb(color)
                light_rgb = tuple(min(1.0, c * 0.55 + 0.45) for c in rgb)

                result_big = psths.get((grp_label, big_key))
                result_small = psths.get((grp_label, small_key))

                has_any = False
                if result_big is not None:
                    mean_p, sem_p, bc = result_big
                    ax.fill_between(bc, mean_p - sem_p, mean_p + sem_p,
                                    alpha=0.25, color=color)
                    ax.plot(bc, mean_p, color=color, linewidth=1.5,
                            label=f"Big (n={event_counts.get(big_key, 0)})")
                    has_any = True
                if result_small is not None:
                    mean_p, sem_p, bc = result_small
                    ax.fill_between(bc, mean_p - sem_p, mean_p + sem_p,
                                    alpha=0.15, color=light_rgb)
                    ax.plot(bc, mean_p, color=light_rgb, linewidth=1.5,
                            linestyle="--",
                            label=f"Small (n={event_counts.get(small_key, 0)})")
                    has_any = True
                if has_any:
                    ax.axvline(0, color="k", linewidth=0.6, linestyle="--",
                               alpha=0.4)
                    ax.legend(fontsize=6, loc="upper right")
                    ax.text(
                        0.98, 0.78,
                        f"{len(cids)} units",
                        transform=ax.transAxes, fontsize=7,
                        ha="right", va="top", color="0.4",
                    )
                else:
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10, color="0.6")

            else:
                # ---- Standard single-trace column ----
                n_evts = event_counts.get(col_key, 0)
                result = psths.get((grp_label, col_key))

                if result is not None:
                    mean_p, sem_p, bc = result
                    ax.fill_between(bc, mean_p - sem_p, mean_p + sem_p,
                                    alpha=0.25, color=color)
                    ax.plot(bc, mean_p, color=color, linewidth=1.5)
                    ax.axvline(0, color="k", linewidth=0.6, linestyle="--",
                               alpha=0.4)
                    ax.text(
                        0.98, 0.95,
                        f"n={n_evts} evts\n{len(cids)} units",
                        transform=ax.transAxes, fontsize=7,
                        ha="right", va="top", color="0.4",
                    )
                else:
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10, color="0.6")

            # Labels
            if ri == 0:
                align_note = EVENT_ALIGN_LABELS.get(col_key, "")
                ax.set_title(f"{col_title}\n{align_note}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{grp_label}\n({len(cids)} units)", fontsize=9)

            despine(ax)

    # ---- Overlay row: all groups on the same axes ----
    overlay_ri = n_grp_rows
    for ci, (col_title, col_key) in enumerate(event_columns):
        ax = axes[overlay_ri, ci]

        if col_key in HIT_COL_SUBKEYS:
            big_key, small_key = HIT_COL_SUBKEYS[col_key]
            for grp_label_ov in row_labels:
                color_ov = ROW_PALETTE.get(grp_label_ov, "#333333")
                rgb = mcolors.to_rgb(color_ov)
                light_rgb = tuple(min(1.0, c * 0.55 + 0.45) for c in rgb)

                res_big = psths.get((grp_label_ov, big_key))
                res_small = psths.get((grp_label_ov, small_key))
                if res_big is not None:
                    mean_p, sem_p, bc = res_big
                    ax.fill_between(bc, mean_p - sem_p, mean_p + sem_p,
                                    alpha=0.15, color=color_ov)
                    ax.plot(bc, mean_p, color=color_ov, linewidth=1.5,
                            label=f"{grp_label_ov} big")
                if res_small is not None:
                    mean_p, sem_p, bc = res_small
                    ax.fill_between(bc, mean_p - sem_p, mean_p + sem_p,
                                    alpha=0.10, color=light_rgb)
                    ax.plot(bc, mean_p, color=light_rgb, linewidth=1.2,
                            linestyle="--",
                            label=f"{grp_label_ov} sm")
            ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.4)
            ax.legend(fontsize=5, loc="upper right", ncol=1)
        else:
            for grp_label_ov in row_labels:
                color_ov = ROW_PALETTE.get(grp_label_ov, "#333333")
                result_ov = psths.get((grp_label_ov, col_key))
                if result_ov is not None:
                    mean_p, sem_p, bc = result_ov
                    ax.fill_between(bc, mean_p - sem_p, mean_p + sem_p,
                                    alpha=0.15, color=color_ov)
                    ax.plot(bc, mean_p, color=color_ov, linewidth=1.5,
                            label=grp_label_ov)
            ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.4)
            ax.legend(fontsize=6, loc="upper right")

        if ci == 0:
            ax.set_ylabel("Overlay\n(all groups)", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        despine(ax)

    # Optionally enforce shared y-axis across the overlay row
    if task.shared_overlay_yaxis:
        overlay_ylims = [axes[overlay_ri, ci].get_ylim() for ci in range(n_cols)]
        y_lo = min(yl[0] for yl in overlay_ylims)
        y_hi = max(yl[1] for yl in overlay_ylims)
        for ci in range(n_cols):
            axes[overlay_ri, ci].set_ylim(y_lo, y_hi)

    # Title — include state label when faceting
    title_parts = [f"TF Event Comparison \u2014 {task.session_name}"]
    if state_label is not None:
        title_parts.append(f"[{state_label}]")
    fig.suptitle(
        " ".join(title_parts),
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = Path(task.out_dir) / task.session_name
    out_path.mkdir(parents=True, exist_ok=True)
    suffix = f"_{state_label.lower().replace(' ', '_')}" if state_label else ""
    fig_path = out_path / f"tf_event_comparison{suffix}.png"
    fig.savefig(str(fig_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


# =====================================================================
# Top-level session worker (for parallel dispatch)
# =====================================================================

def _process_single_session(task: SessionTask) -> dict:
    """Process one session end-to-end.  Designed to be called in a worker process.

    Returns a result dict with keys: session_name, status, message, fig_path,
    group_counts.
    """
    result = {
        "session_name": task.session_name,
        "status": "ok",
        "message": "",
        "fig_path": "",
        "group_counts": {},
    }
    try:
        session = load_session(task.pkl_path)

        # 1. Classify units (TF or Lick based on --classify-by)
        if task.classify_by == "lick":
            groups = _classify_units_by_lick(
                session,
                lick_csv_path=task.lick_csv_path,
                min_fr=task.min_fr,
                lick_events_mode=task.lick_events,
            )
        else:
            groups = _classify_units(
                session,
                tf_csv_path=task.tf_csv_path,
                z_thresh=task.z_thresh_tf,
                min_fr=task.min_fr,
                polarity_mode=task.polarity_mode,
            )

        # Optionally append lick-classified rows below TF rows
        if task.add_lick_rows and task.classify_by == "tf":
            lick_groups = _classify_units_by_lick(
                session,
                lick_csv_path=task.lick_csv_path,
                min_fr=task.min_fr,
                lick_events_mode=task.lick_events,
            )
            groups.update(lick_groups)

        result["group_counts"] = {k: len(v) for k, v in groups.items()}

        # Check we have at least one group with enough units
        has_data = any(len(v) >= task.min_units for v in groups.values())
        if not has_data:
            result["status"] = "skip"
            result["message"] = "No neuron group has enough units"
            return result

        # 2. Build assignments DataFrame subset for this session
        adf = pd.DataFrame(task.assignments_rows)

        # 3. Extract behavioral event times (all trials)
        event_times = _get_behavioral_event_times(
            session, adf, task.session_name,
            rt_shift_s=task.rt_shift_s,
        )

        # 4. Load TF pulse event times (prefer saved CSV, fall back to compute)
        #    Build a dict keyed by TF column key (tf_pulse / tf_pulse_fast / tf_pulse_slow).
        tf_pulse_times: Dict[str, List[float]] = {}
        if task.tf_type == "both":
            for _tf_key, _tf_t in [("tf_pulse_fast", "fast"), ("tf_pulse_slow", "slow")]:
                times = _load_tf_pulse_times(Path(task.tf_dir), task.session_name, tf_type=_tf_t)
                if not times:
                    times = _collect_tf_pulse_times_fallback(session, tf_type=_tf_t)
                tf_pulse_times[_tf_key] = times
        else:
            times = _load_tf_pulse_times(Path(task.tf_dir), task.session_name, tf_type=task.tf_type)
            if not times:
                times = _collect_tf_pulse_times_fallback(session, tf_type=task.tf_type)
            tf_pulse_times["tf_pulse"] = times

        # Pre-compute Baseline_ON array for TF pulse state filtering
        ni = getattr(session, "ni_events", {}) or {}
        baseline_arr = _to_float_array(ni.get("Baseline_ON", []))

        # 5. Generate figure(s) — build cache, save, and plot
        fig_paths: List[str] = []

        if task.facet_state and adf.shape[0] > 0 and "hmm_state" in adf.columns:
            # ---- State-faceted mode ----
            # "All trials" figure first
            cache_all = _build_psth_cache(
                session, groups, event_times, tf_pulse_times, task,
                state_label="all",
            )
            _save_psth_cache(cache_all,
                             _cache_path(task.out_dir, task.session_name, "all"))
            fp = _generate_session_figure(cache_all, task, state_label="all")
            if fp:
                fig_paths.append(fp)

            # Per-state figures
            states_present = sorted(adf["hmm_state"].unique())
            for sidx in states_present:
                label = (
                    task.state_labels[int(sidx)]
                    if int(sidx) < len(task.state_labels)
                    else f"State_{sidx}"
                )
                trial_set = set(
                    adf.loc[adf["hmm_state"] == sidx, "trial_idx"]
                    .astype(int).values
                )
                state_events = _get_behavioral_event_times(
                    session, adf, task.session_name,
                    trial_indices=trial_set,
                    rt_shift_s=task.rt_shift_s,
                )
                # Filter TF pulses to this state's trial windows
                state_tf_pulses = {
                    k: _filter_tf_pulses_by_trials(v, baseline_arr, trial_set)
                    for k, v in tf_pulse_times.items()
                }
                cache_state = _build_psth_cache(
                    session, groups, state_events, state_tf_pulses, task,
                    state_label=label,
                )
                _save_psth_cache(cache_state,
                                 _cache_path(task.out_dir, task.session_name, label))
                fp = _generate_session_figure(cache_state, task, state_label=label)
                if fp:
                    fig_paths.append(fp)
        else:
            # ---- Standard mode (all trials) ----
            cache = _build_psth_cache(
                session, groups, event_times, tf_pulse_times, task,
            )
            _save_psth_cache(cache,
                             _cache_path(task.out_dir, task.session_name))
            fp = _generate_session_figure(cache, task)
            if fp:
                fig_paths.append(fp)

        result["fig_path"] = "; ".join(fig_paths)
        if not fig_paths:
            result["status"] = "skip"
            result["message"] = "No groups with enough units for figure"
        else:
            result["message"] = (
                f"Saved figure with {sum(1 for v in groups.values() if len(v) >= task.min_units)} rows"
            )

    except Exception as exc:
        result["status"] = "error"
        result["message"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return result


def _replot_single_session(task: SessionTask) -> dict:
    """Re-generate figures from cached PSTH data (no session pkl needed).

    Looks for ``psth_cache*.pkl`` files in the session's output directory
    and regenerates figures from them.
    """
    result = {
        "session_name": task.session_name,
        "status": "ok",
        "message": "",
        "fig_path": "",
        "group_counts": {},
    }
    try:
        set_style(context="talk")
        out_path = Path(task.out_dir) / task.session_name
        cache_files = sorted(out_path.glob("psth_cache*.pkl"))
        if not cache_files:
            result["status"] = "skip"
            result["message"] = "No cached PSTH data found — run without --replot-only first"
            return result

        fig_paths: List[str] = []
        for cf in cache_files:
            cache = _load_psth_cache(cf)
            state_label = cache.get("state_label")
            result["group_counts"] = {
                k: len(v) for k, v in cache["groups"].items()
            }
            fp = _generate_session_figure(cache, task, state_label=state_label)
            if fp:
                fig_paths.append(fp)

        result["fig_path"] = "; ".join(fig_paths)
        if fig_paths:
            result["message"] = f"Replotted {len(fig_paths)} figure(s) from cache"
        else:
            result["status"] = "skip"
            result["message"] = "Cache found but no plottable groups"

    except Exception as exc:
        result["status"] = "error"
        result["message"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return result


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-session multi-event PSTH comparison: TF-responsive vs Non-TF neurons.",
    )
    parser.add_argument("--data-dir", required=True,
                        help="HMM results directory (model + assignments).")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of HMM states (default: highest-K on disk).")
    parser.add_argument("--pkl-dir", required=True,
                        help="Directory with session pkl files.")
    parser.add_argument("--manifest", default=None,
                        help="Staging manifest CSV (optional filter).")
    parser.add_argument("--session", default=None,
                        help="Single session name (optional).")
    parser.add_argument("--tf-dir", required=True,
                        help="Root TF screening directory (e.g. FIGURES/tf).")
    parser.add_argument("--out", default="FIGURES/behavior/hmm/tf_event_comparison",
                        help="Output directory.")
    parser.add_argument("--exclude-qc-fail", action="store_true",
                        help="DEPRECATED: SESSION_FILTER handles QC.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Bypass SESSION_FILTER.")
    parser.add_argument("--z-thresh-tf", type=float, default=3.0,
                        help="Z-score threshold for TF responsiveness.")
    parser.add_argument("--min-fr", type=float, default=1.0,
                        help="Minimum firing rate (Hz).")
    parser.add_argument("--min-units", type=int, default=3,
                        help="Minimum units in a group to include row in figure.")
    parser.add_argument("--tf-type", default="fast",
                        choices=["fast", "slow", "both"],
                        help="TF pulse type for the pulse-response column(s). "
                             "'both' shows fast and slow in separate columns.")

    # Polarity / layout
    parser.add_argument("--polarity-mode", default="split",
                        choices=["split", "full", "simple"],
                        help="Neuron grouping: "
                             "'split' = TF-excited, TF-suppressed, Non-TF (default); "
                             "'full' = TF-fast-exc/sup × TF-slow-exc/sup + Non-TF; "
                             "'simple' = TF-responsive, Non-TF.")
    parser.add_argument("--shared-yaxis", action="store_true",
                        help="Share y-axis across event columns within each row.")
    parser.add_argument("--shared-overlay-yaxis", action="store_true",
                        help="Enforce shared y-axis across all columns in the "
                             "overlay (bottom) row.  Default: each overlay "
                             "column has its own independent y-axis.")
    parser.add_argument("--facet-state", action="store_true",
                        help="Generate per-HMM-state figures in addition to "
                             "the all-trials figure.")
    parser.add_argument("--rt-shift-ms", type=float, default=0.0,
                        help="Shift lick-aligned event times by this many ms "
                             "(negative = earlier, e.g. -200 shifts 200 ms "
                             "before detected lick). Affects FA early/late "
                             "and Hit (@ lick) column. Default: 0.")

    # PSTH parameters
    parser.add_argument("--sigma-ms", type=float, default=25.0,
                        help="Gaussian smoothing sigma (ms).")
    parser.add_argument("--bin-size", type=float, default=0.025,
                        help="Bin size in seconds.")
    parser.add_argument("--window-pre", type=float, default=0.5,
                        help="Pre-event window for behavioral events (s).")
    parser.add_argument("--window-post", type=float, default=1.0,
                        help="Post-event window for behavioral events (s).")
    parser.add_argument("--normalize", default="zscore",
                        choices=["zscore", "baseline-subtract", "none"],
                        help="PSTH normalization mode (default: zscore).")

    # Parallelism
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (default 1 = serial).")
    parser.add_argument("--replot-only", action="store_true",
                        help="Skip computation; regenerate figures from cached "
                             "PSTH data (psth_cache*.pkl).  Much faster for "
                             "layout/styling iterations.")

    # Classification mode
    parser.add_argument("--classify-by", default="tf",
                        choices=["tf", "lick"],
                        help="Classify neurons by TF responsiveness (default) "
                             "or lick responsiveness.")
    parser.add_argument("--lick-dir", default=None,
                        help="Root lick analysis directory containing per-session "
                             "lick_responsiveness.csv files "
                             "(e.g. FIGURES/lick/BG_046).")
    parser.add_argument("--lick-events", default="fa",
                        choices=["fa", "hit", "both"],
                        help="Which lick events to use for lick classification: "
                             "'fa' = late FA licks (default, uses CSV); "
                             "'hit' = Hit-trial licks (change>1); "
                             "'both' = pool FA + Hit licks.")
    parser.add_argument("--add-lick-rows", action="store_true",
                        help="When --classify-by=tf, also append lick-classified "
                             "rows (Lick-excited, Lick-inhibited, Non-lick) "
                             "below the TF rows in the same figure.")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style(context="talk")

    # Load HMM results
    model, assignments_df, state_labels = load_hmm_results(
        Path(args.data_dir), K=args.K,
    )
    K = model.n_states
    print(f"Loaded K={K} model, labels={state_labels}")

    # Determine sessions
    pkl_dir = Path(args.pkl_dir)
    tf_dir = Path(args.tf_dir)
    lick_dir = Path(args.lick_dir) if args.lick_dir else None

    if args.session:
        session_names = [args.session]
    elif args.manifest or True:  # Always load manifest for session filtering
        manifest = load_staging_manifest(
            manifest_path=args.manifest,
            apply_filter=not getattr(args, 'no_filter', False),
        )
        session_names = manifest["session_name"].tolist()
    else:
        session_names = assignments_df["session_name"].unique().tolist()

    # Build tasks
    tasks: List[SessionTask] = []
    subject = "BG_046"  # default; could be parameterised

    window_behavioral = (-args.window_pre, args.window_post)
    window_tf = TF_PULSE_WINDOW

    for sname in session_names:
        # Find pkl
        candidates = list(pkl_dir.glob(f"*{sname}*.pkl"))
        if not candidates:
            print(f"  SKIP {sname}: pkl not found")
            continue

        # Find TF CSV
        tf_csv: Optional[str] = None
        for d in [tf_dir / f"{subject}_{sname}", tf_dir / sname]:
            p = d / "tf_pulse_grid_both.csv"
            if p.exists():
                tf_csv = str(p)
                break
        if tf_csv is None:
            print(f"  WARN {sname}: no tf_pulse_grid_both.csv — "
                  f"all units will be Non-TF")

        # Find Lick CSV
        lick_csv: Optional[str] = None
        if lick_dir is not None:
            for d in [lick_dir / f"{subject}_{sname}", lick_dir / sname]:
                p = d / "lick_responsiveness.csv"
                if p.exists():
                    lick_csv = str(p)
                    break
            if lick_csv is None and (
                args.classify_by == "lick" or args.add_lick_rows
            ):
                if args.lick_events == "fa":
                    print(f"  WARN {sname}: no lick_responsiveness.csv — "
                          f"will compute on-the-fly")

        # Slice assignments for this session
        sdf = assignments_df[assignments_df["session_name"] == sname]
        rows = sdf.to_dict("records")

        tasks.append(SessionTask(
            pkl_path=str(candidates[0]),
            session_name=sname,
            tf_csv_path=tf_csv,
            tf_dir=str(tf_dir),
            assignments_rows=rows,
            out_dir=str(out_dir),
            z_thresh_tf=args.z_thresh_tf,
            min_fr=args.min_fr,
            tf_type=args.tf_type,
            polarity_mode=args.polarity_mode,
            shared_yaxis=args.shared_yaxis,
            sigma_ms=args.sigma_ms,
            bin_size=args.bin_size,
            window_behavioral=window_behavioral,
            window_tf=window_tf,
            min_units=args.min_units,
            normalize=args.normalize,
            shared_overlay_yaxis=args.shared_overlay_yaxis,
            rt_shift_s=args.rt_shift_ms / 1000.0,
            replot_only=args.replot_only,
            classify_by=args.classify_by,
            lick_csv_path=lick_csv,
            lick_dir=str(lick_dir) if lick_dir else "",
            lick_events=args.lick_events,
            add_lick_rows=args.add_lick_rows,
            facet_state=args.facet_state,
            state_labels=state_labels,
        ))

    facet_str = f", facet-state={args.facet_state}" if args.facet_state else ""
    replot_str = " [REPLOT-ONLY]" if args.replot_only else ""
    print(f"\nProcessing {len(tasks)} sessions "
          f"(workers={args.n_workers}, polarity={args.polarity_mode}{facet_str}){replot_str}\n")

    # Choose worker function
    worker_fn = _replot_single_session if args.replot_only else _process_single_session

    # ---- Execute ----
    results: List[dict] = []

    if args.n_workers <= 1:
        # Serial
        for task in tqdm(tasks, desc="Sessions", unit="sess"):
            r = worker_fn(task)
            results.append(r)
            _print_result(r)
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(worker_fn, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Sessions", unit="sess"):
                task = futures[future]
                try:
                    r = future.result()
                except Exception as exc:
                    r = {
                        "session_name": task.session_name,
                        "status": "error",
                        "message": str(exc),
                        "fig_path": "",
                        "group_counts": {},
                    }
                results.append(r)
                _print_result(r)

    # ---- Summary ----
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skip")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"\n{'=' * 60}")
    print(f"DONE: {n_ok} figures saved, {n_skip} skipped, {n_err} errors")
    print(f"Output: {out_dir}")

    # Save summary CSV
    summary_df = pd.DataFrame([
        {
            "session_name": r["session_name"],
            "status": r["status"],
            "message": r["message"],
            "fig_path": r["fig_path"],
            **{f"n_{k}": v for k, v in r.get("group_counts", {}).items()},
        }
        for r in results
    ])
    summary_path = out_dir / "processing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")


def _print_result(r: dict) -> None:
    """Print a one-line status for a processed session."""
    tag = {"ok": "OK", "skip": "SKIP", "error": "ERR"}.get(r["status"], "???")
    counts = r.get("group_counts", {})
    counts_str = ", ".join(f"{k}={v}" for k, v in counts.items()) if counts else ""
    msg = r.get("message", "")
    print(f"  [{tag}] {r['session_name']}: {msg}  ({counts_str})")


if __name__ == "__main__":
    main()
