"""State-conditioned neural analysis using HMM behavioral states.

Produces two analysis levels:

  **Per-session** (in session subdirectories):
    - State-conditioned PSTHs for responsive units (Gaussian-smoothed)
    - Population modulation index histograms
    - State-transition-triggered PSTHs

  **Pooled across sessions** (in pooled/ subdirectory):
    - Grand-average PSTHs per state for each unit that appears in
      multiple sessions, or aggregated population PSTHs
    - Population modulation index aggregated across all sessions

Key features:
  - ``--K`` flag selects which fitted model to use (default: highest K)
  - Units filtered by minimum firing rate (default >=1 Hz)
  - Units ranked by visual (or TF) responsiveness
  - Event-appropriate windows: different baseline/response windows for
    Change_ON, FA, Hit, Baseline_ON
  - Trial-type filtering: FA/abort trials excluded when aligning to
    Change_ON (change never occurred on those trials)
  - ``--unit-select tf-fast|tf-slow|tf-any`` uses TF pulse screening
    to select units instead of visual responsiveness
  - ``--normalize zscore|baseline-subtract|none`` controls PSTH
    normalization.  Per-unit plots default to raw Hz; pooled plots
    default to z-score so every unit contributes equally regardless
    of absolute firing rate.
  - PSTHs are Gaussian-smoothed (default sigma=25 ms)
  - Pooled analysis combines trials across sessions for each state

Usage
-----
    # Default (visually responsive units, best-K model):
    python scripts/analysis/behavior/hmm_neural_states.py \\
        --data-dir  data/hmm/BG_046 \\
        --pkl-dir   data/pkls/BG_046 \\
        --manifest  data/BG_046_staging_manifest_v2.csv \\
        --out       FIGURES/behavior/BG_046/hmm/neural \\
        --exclude-qc-fail

    # Specific K (e.g. K=3 model):
    python scripts/analysis/behavior/hmm_neural_states.py \\
        --data-dir data/hmm/BG_046 --pkl-dir data/pkls/BG_046 --K 3 \\
        --out FIGURES/behavior/BG_046/hmm/neural_K3

    # TF-responsive cells by state:
    python scripts/analysis/behavior/hmm_neural_states.py \\
        --data-dir data/hmm/BG_046 --pkl-dir data/pkls/BG_046 \\
        --unit-select tf-any --tf-dir FIGURES/tf \\
        --out FIGURES/behavior/BG_046/hmm/neural_tf

    # FA-aligned analysis:
    python scripts/analysis/behavior/hmm_neural_states.py \\
        --data-dir data/hmm/BG_046 --pkl-dir data/pkls/BG_046 \\
        --event FA --window-pre 2.0 --window-post 0.5 \\
        --out FIGURES/behavior/BG_046/hmm/neural_fa
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from visdetect.analysis.config import load_staging_manifest
from visdetect.core.session import load_session
from visdetect.analysis.hmm_downstream import (
    compute_population_state_modulation,
    compute_state_conditioned_psth,
    compute_transition_triggered_psth,
    find_state_transitions,
    get_state_trial_indices,
    load_hmm_results,
    smooth_psth,
)
from visdetect.viz.plotting import set_style, despine
from visdetect.analysis.constants import (
    EVENT_RESPONSIVENESS_WINDOWS,
    EVENT_VALID_OUTCOMES,
)


def _state_palette(K):
    base = ["#7570b3", "#1b9e77", "#d95f02", "#e7298a", "#66a61e", "#e6ab02"]
    return base[:K]


# =====================================================================
# Unit selection helpers
# =====================================================================

def compute_unit_firing_rate(session, cluster_id: int) -> float:
    """Overall mean firing rate (Hz) for a cluster across the full recording."""
    for c in session.clusters:
        if c.cluster_id == cluster_id:
            if c.spike_times is None or len(c.spike_times) == 0:
                return 0.0
            duration = float(c.spike_times[-1] - c.spike_times[0])
            return len(c.spike_times) / max(duration, 1e-6)
    return 0.0


def compute_visual_responsiveness(
    session,
    cluster_id: int,
    event_name: str = "Change_ON",
    baseline_window: Tuple[float, float] | None = None,
    response_window: Tuple[float, float] | None = None,
    bin_size: float = 0.025,
) -> float:
    """Compute a responsiveness index: (response - baseline) / (response + baseline).

    Returns a value in [-1, 1]; higher = more visually driven.
    Returns 0 if insufficient data.

    Windows are chosen automatically per event type (see
    ``EVENT_RESPONSIVENESS_WINDOWS``) but can be overridden explicitly.
    When *event_name* is ``'Change_ON'``, only Hit and Miss trials are
    used (FA/abort trials never saw the actual change).
    """
    from visdetect.analysis.align import (
        get_event_times_by_trial,
        align_spikes_to_events,
    )
    try:
        # Resolve windows: use per-event defaults unless overridden
        if baseline_window is None or response_window is None:
            defaults = EVENT_RESPONSIVENESS_WINDOWS.get(
                event_name, ((-0.4, -0.05), (0.03, 0.25))
            )
            if baseline_window is None:
                baseline_window = defaults[0]
            if response_window is None:
                response_window = defaults[1]

        event_times = get_event_times_by_trial(session, event_name)

        # Trial-type filter
        valid_outcomes = EVENT_VALID_OUTCOMES.get(event_name, None)
        if valid_outcomes is not None and hasattr(session, "trials"):
            valid = []
            for idx, t in enumerate(event_times):
                if np.isnan(t):
                    continue
                if idx < len(session.trials):
                    oc = getattr(session.trials[idx], "trialoutcome", "").lower()
                    if oc in valid_outcomes:
                        valid.append(t)
                else:
                    valid.append(t)
        else:
            valid = [t for t in event_times if not np.isnan(t)]

        if len(valid) < 10:
            return 0.0
        spike_times = None
        for c in session.clusters:
            if c.cluster_id == cluster_id:
                spike_times = c.spike_times
                break
        if spike_times is None or len(spike_times) == 0:
            return 0.0

        full_window = (baseline_window[0], response_window[1])
        fr_matrix, bc = align_spikes_to_events(
            spike_times, valid, window=full_window, bin_size=bin_size,
        )
        if fr_matrix.shape[0] == 0:
            return 0.0

        mean_fr = fr_matrix.mean(axis=0)
        bl_mask = (bc >= baseline_window[0]) & (bc <= baseline_window[1])
        resp_mask = (bc >= response_window[0]) & (bc <= response_window[1])
        bl_rate = mean_fr[bl_mask].mean() if bl_mask.any() else 0.0
        resp_rate = mean_fr[resp_mask].mean() if resp_mask.any() else 0.0

        denom = resp_rate + bl_rate
        if denom < 0.5:  # essentially silent
            return 0.0
        return (resp_rate - bl_rate) / denom
    except Exception:
        return 0.0


def select_units(
    session,
    min_fr: float = 1.0,
    max_units: int = 20,
    event_name: str = "Change_ON",
    rank_by_responsiveness: bool = True,
) -> List[int]:
    """Select quality units: filter by firing rate, rank by responsiveness.

    Returns up to max_units cluster IDs, sorted by visual responsiveness
    (most responsive first).
    """
    # Start with quality-curated list
    cluster_ids = (
        session.good_and_stable_ids
        or session.good_cluster_ids
        or [c.cluster_id for c in session.clusters]
    )
    if not cluster_ids:
        return []

    # Filter by minimum firing rate
    candidates = []
    for cid in cluster_ids:
        fr = compute_unit_firing_rate(session, cid)
        if fr >= min_fr:
            candidates.append((cid, fr))

    if not candidates:
        return []

    if rank_by_responsiveness and len(candidates) > 0:
        # Score responsiveness and sort
        scored = []
        for cid, fr in candidates:
            resp = compute_visual_responsiveness(session, cid, event_name=event_name)
            scored.append((cid, resp, fr))
        # Sort: most responsive first
        scored.sort(key=lambda x: -abs(x[1]))
        return [cid for cid, _, _ in scored[:max_units]]
    else:
        return [cid for cid, _ in candidates[:max_units]]


def select_tf_responsive_units(
    session,
    tf_dir: Path,
    tf_type: str = "any",
    min_fr: float = 1.0,
    max_units: int = 20,
    z_thresh: float = 3.0,
) -> List[int]:
    """Select units identified as TF-responsive from pre-computed screening.

    Parameters
    ----------
    session : Session object.
    tf_dir : root TF output directory (e.g. ``FIGURES/tf``).
        Expects a subdirectory named ``BG_046_{session_name}`` containing
        ``tf_pulse_grid_both.csv``.
    tf_type : ``'fast'``, ``'slow'``, or ``'any'`` (either/both).
    min_fr : minimum firing rate to keep.
    max_units : max units to return.
    z_thresh : absolute z-score threshold for responsiveness.

    Returns
    -------
    list of cluster IDs, sorted by max |z| descending.
    """
    sname = session.session_name or ""
    subject = getattr(session, "subject", "BG_046")

    # Try common naming patterns for TF output directory
    candidates_dirs = [
        tf_dir / f"{subject}_{sname}",
        tf_dir / sname,
    ]
    csv_path = None
    for d in candidates_dirs:
        p = d / "tf_pulse_grid_both.csv"
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        print(f"    TF: no tf_pulse_grid_both.csv found for {sname}")
        return []

    tf_df = pd.read_csv(csv_path)

    # Classify responsiveness using z-score columns
    fast_resp = (
        (tf_df["z_max_fast"].abs() >= z_thresh) |
        (tf_df["z_min_fast"].abs() >= z_thresh)
    )
    slow_resp = (
        (tf_df["z_max_slow"].abs() >= z_thresh) |
        (tf_df["z_min_slow"].abs() >= z_thresh)
    )

    if tf_type == "fast":
        mask = fast_resp
    elif tf_type == "slow":
        mask = slow_resp
    else:  # "any"
        mask = fast_resp | slow_resp

    responsive = tf_df[mask].copy()
    if responsive.empty:
        print(f"    TF: no {tf_type}-responsive units found for {sname}")
        return []

    # Rank by max absolute z-score
    responsive["max_abs_z"] = responsive[
        ["z_max_fast", "z_min_fast", "z_max_slow", "z_min_slow"]
    ].abs().max(axis=1)
    responsive = responsive.sort_values("max_abs_z", ascending=False)

    # Intersect with quality-curated cluster IDs and firing rate filter
    quality_ids = set(
        session.good_and_stable_ids
        or session.good_cluster_ids
        or [c.cluster_id for c in session.clusters]
    )
    result = []
    for _, row in responsive.iterrows():
        cid = int(row["cluster_id"])
        if cid not in quality_ids:
            continue
        fr = compute_unit_firing_rate(session, cid)
        if fr >= min_fr:
            result.append(cid)
        if len(result) >= max_units:
            break

    print(f"    TF: {len(result)} {tf_type}-responsive units selected "
          f"(z>={z_thresh}, FR>={min_fr} Hz)")
    return result


# =====================================================================
# Normalization helper
# =====================================================================

def _normalize_psth(
    psth: np.ndarray,
    bin_centers: np.ndarray,
    mode: str = "none",
    baseline_end: float = 0.0,
) -> np.ndarray:
    """Normalize a PSTH (1-D array) relative to its pre-event baseline.

    Parameters
    ----------
    psth : 1-D firing-rate array (Hz).
    bin_centers : corresponding time axis.
    mode : ``'none'`` (raw Hz), ``'zscore'``, or ``'baseline-subtract'``.
    baseline_end : right edge of baseline window (default 0 = event onset).

    Returns
    -------
    Normalized 1-D array (same length).
    """
    if mode == "none":
        return psth
    bl_mask = bin_centers < baseline_end
    if not bl_mask.any():
        return psth  # can't normalise without baseline bins
    bl = psth[bl_mask]
    mu = bl.mean()
    if mode == "baseline-subtract":
        return psth - mu
    # zscore
    sigma = bl.std()
    if sigma < 1e-6:  # silent baseline → just subtract mean
        return psth - mu
    return (psth - mu) / sigma


def _ylabel_for_mode(mode: str) -> str:
    if mode == "zscore":
        return "Firing rate (z-score)"
    elif mode == "baseline-subtract":
        return "\u0394 Firing rate (Hz)"
    return "Firing rate (Hz)"


# =====================================================================
# Plotting helpers (improved)
# =====================================================================

def plot_state_psths_overlay(
    session,
    assignments_df: pd.DataFrame,
    cluster_id: int,
    state_labels: list,
    n_states: int,
    out_dir: Path,
    event_name: str = "Change_ON",
    window=(-0.5, 1.0),
    bin_size: float = 0.025,
    sigma_ms: float = 25.0,
    min_trials: int = 15,
    session_label: str = "",
    valid_outcomes: set | None = None,
    normalize: str = "none",
):
    """Overlay state PSTHs for one unit — only states with enough trials.

    normalize : 'none' (raw Hz), 'zscore', or 'baseline-subtract'.
    """
    palette = _state_palette(n_states)
    sname = session_label or session.session_name or "unknown"

    # Resolve outcome filter from event name if not explicitly given
    if valid_outcomes is None:
        valid_outcomes = EVENT_VALID_OUTCOMES.get(event_name, None)

    results = {}
    for k in range(n_states):
        result = compute_state_conditioned_psth(
            session, assignments_df, k, cluster_id,
            event_name=event_name, window=window, bin_size=bin_size,
            valid_outcomes=valid_outcomes,
        )
        results[k] = result

    # Which states have enough trials?
    active_states = [k for k in range(n_states) if results[k]["n_trials"] >= min_trials]
    if len(active_states) < 1:
        return False  # nothing to plot

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in active_states:
        r = results[k]
        psth = smooth_psth(r["psth"], bin_size=bin_size, sigma_ms=sigma_ms)
        sem = smooth_psth(r["sem"], bin_size=bin_size, sigma_ms=sigma_ms)
        bc = r["bin_centers"]
        psth = _normalize_psth(psth, bc, mode=normalize)
        # SEM doesn't shift with baseline-subtract but scales with zscore
        if normalize == "zscore":
            bl_mask = bc < 0.0
            bl_std = smooth_psth(r["psth"], bin_size=bin_size, sigma_ms=sigma_ms)[bl_mask].std()
            if bl_std > 1e-6:
                sem = sem / bl_std
        ax.fill_between(bc, psth - sem, psth + sem, alpha=0.2, color=palette[k])
        ax.plot(bc, psth, color=palette[k], linewidth=1.8,
                label=f"{state_labels[k]} (n={r['n_trials']})")

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(_ylabel_for_mode(normalize))
    ax.set_title(f"Cluster {cluster_id} — {sname}")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_xlim(window)
    despine(ax)
    plt.tight_layout()

    sess_dir = out_dir / sname
    sess_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(sess_dir / f"unit_{cluster_id}_state_psth.png", dpi=150)
    plt.close(fig)
    return True


def plot_modulation_histogram(
    mi_df: pd.DataFrame,
    state_labels: list,
    state_a: int,
    state_b: int,
    label: str,
    out_dir: Path,
):
    """Histogram of modulation indices across all units."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = mi_df["modulation_index"].dropna().values
    if len(vals) == 0:
        plt.close(fig)
        return
    ax.hist(vals, bins=30, color="steelblue", edgecolor="k", alpha=0.7)
    ax.axvline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel(f"MI  ({state_labels[state_a]} vs {state_labels[state_b]})")
    ax.set_ylabel("Number of units")
    ax.set_title(f"Population Modulation — {label}")
    med = np.median(vals)
    ax.axvline(med, color="orange", linewidth=1.5, linestyle="-",
               label=f"Median={med:.3f}")
    ax.legend()
    despine(ax)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = label.replace("/", "_").replace("\\", "_")
    fig.savefig(
        out_dir / f"modulation_{state_labels[state_a]}_vs_{state_labels[state_b]}_{safe_name}.png",
        dpi=150,
    )
    plt.close(fig)


def plot_transition_psth(
    session,
    assignments_df: pd.DataFrame,
    cluster_id: int,
    from_state: int,
    to_state: int,
    state_labels: list,
    out_dir: Path,
    event_name: str = "Change_ON",
    window=(-0.5, 1.0),
    bin_size: float = 0.025,
    sigma_ms: float = 25.0,
    min_transitions: int = 3,
    normalize: str = "none",
):
    """Plot pre- vs post-transition PSTHs for one unit."""
    sname = session.session_name or "unknown"
    result = compute_transition_triggered_psth(
        session, assignments_df, cluster_id, from_state, to_state,
        event_name=event_name, window=window, bin_size=bin_size,
    )

    if result["n_transitions"] < min_transitions:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    bc = result["bin_centers"]

    pre = smooth_psth(result["pre_psth"], bin_size=bin_size, sigma_ms=sigma_ms)
    pre_sem = smooth_psth(result["pre_sem"], bin_size=bin_size, sigma_ms=sigma_ms)
    post = smooth_psth(result["post_psth"], bin_size=bin_size, sigma_ms=sigma_ms)
    post_sem = smooth_psth(result["post_sem"], bin_size=bin_size, sigma_ms=sigma_ms)

    # Normalize both traces to the same reference (pre-trace baseline)
    pre = _normalize_psth(pre, bc, mode=normalize)
    post = _normalize_psth(post, bc, mode=normalize)
    if normalize == "zscore":
        raw_pre = smooth_psth(result["pre_psth"], bin_size=bin_size, sigma_ms=sigma_ms)
        bl_std = raw_pre[bc < 0.0].std() if (bc < 0.0).any() else 1.0
        if bl_std > 1e-6:
            pre_sem = pre_sem / bl_std
            post_sem = post_sem / bl_std

    ax.fill_between(bc, pre - pre_sem, pre + pre_sem, alpha=0.2, color="tab:blue")
    ax.plot(bc, pre, color="tab:blue", linewidth=1.8,
            label=f"Pre ({state_labels[from_state]})")
    ax.fill_between(bc, post - post_sem, post + post_sem, alpha=0.2, color="tab:red")
    ax.plot(bc, post, color="tab:red", linewidth=1.8,
            label=f"Post ({state_labels[to_state]})")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(_ylabel_for_mode(normalize))
    ax.set_title(f"Cluster {cluster_id} — {state_labels[from_state]}→{state_labels[to_state]}\n"
                 f"{sname} ({result['n_transitions']} transitions)")
    ax.legend(fontsize=9)
    despine(ax)
    plt.tight_layout()

    sess_dir = out_dir / sname
    sess_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        sess_dir / f"unit_{cluster_id}_transition_{from_state}_to_{to_state}.png",
        dpi=120,
    )
    plt.close(fig)
    return True


# =====================================================================
# Per-session processing
# =====================================================================

def process_session(
    session,
    assignments_df: pd.DataFrame,
    state_labels: list,
    n_states: int,
    out_dir: Path,
    max_units: int = 20,
    min_fr: float = 1.0,
    min_trials: int = 15,
    min_transitions: int = 3,
    event_name: str = "Change_ON",
    window=(-0.5, 1.0),
    bin_size: float = 0.025,
    sigma_ms: float = 25.0,
    unit_select: str = "responsive",
    tf_dir: Path | None = None,
    z_thresh_tf: float = 3.0,
    normalize: str = "none",
):
    """Run per-session neural analyses.

    Parameters
    ----------
    unit_select : ``'responsive'`` (default), ``'tf-fast'``,
        ``'tf-slow'``, or ``'tf-any'``.
    tf_dir : root TF screening directory (needed for tf-* modes).
    z_thresh_tf : z-score threshold for TF responsive classification.
    normalize : ``'none'`` (raw Hz), ``'zscore'``, or
        ``'baseline-subtract'``.
    """
    sname = session.session_name or "unknown"
    sdf = assignments_df[assignments_df["session_name"] == sname]
    print(f"\n  Session: {sname} ({len(sdf)} trials)")

    # Report state trial counts
    state_counts = {}
    for k in range(n_states):
        n = int((sdf["hmm_state"] == k).sum())
        state_counts[k] = n
    active = {k: n for k, n in state_counts.items() if n >= min_trials}
    print(f"    States with >={min_trials} trials: "
          + ", ".join(f"{state_labels[k]}={n}" for k, n in active.items()))

    if len(active) < 2:
        print(f"    SKIP: need >=2 active states for comparison (have {len(active)})")
        return

    # Select units by quality and responsiveness
    if unit_select.startswith("tf-"):
        tf_type = unit_select.replace("tf-", "")  # "fast", "slow", "any"
        if tf_dir is None:
            print(f"    SKIP: --tf-dir required for unit-select={unit_select}")
            return
        subset = select_tf_responsive_units(
            session, tf_dir, tf_type=tf_type, min_fr=min_fr,
            max_units=max_units, z_thresh=z_thresh_tf,
        )
    else:
        subset = select_units(
            session, min_fr=min_fr, max_units=max_units, event_name=event_name,
        )
    if not subset:
        print(f"    No units selected (mode={unit_select}, FR>={min_fr} Hz).")
        return
    print(f"    {len(subset)} units selected (mode={unit_select})")

    # 1. State-conditioned PSTHs
    n_plotted = 0
    for cid in subset:
        try:
            ok = plot_state_psths_overlay(
                session, assignments_df, cid, state_labels, n_states,
                out_dir, event_name=event_name, window=window,
                bin_size=bin_size, sigma_ms=sigma_ms, min_trials=min_trials,
                normalize=normalize,
            )
            if ok:
                n_plotted += 1
        except Exception as exc:
            print(f"      Unit {cid} PSTH failed: {exc}")
    print(f"    Plotted {n_plotted} state-conditioned PSTHs")

    # 2. Modulation index — pick best pair of active states
    #    Prefer an Engaged vs Disengaged pair; fall back to any two active
    state_a, state_b = _pick_comparison_pair(active, state_labels)
    if state_a is not None and state_b is not None:
        all_cluster_ids = (
            session.good_and_stable_ids
            or session.good_cluster_ids
            or [c.cluster_id for c in session.clusters]
        )
        mi_df = compute_population_state_modulation(
            session, assignments_df, state_a, state_b,
            cluster_ids=all_cluster_ids,
            event_name=event_name, response_window=(0.0, 0.3),
        )
        if len(mi_df) > 0:
            sess_dir = out_dir / sname
            sess_dir.mkdir(parents=True, exist_ok=True)
            mi_df.to_csv(sess_dir / "modulation_index.csv", index=False)
            plot_modulation_histogram(
                mi_df, state_labels, state_a, state_b, sname, sess_dir,
            )
            print(f"    Modulation {state_labels[state_a]} vs {state_labels[state_b]}: "
                  f"median MI={mi_df['modulation_index'].median():.3f} ({len(mi_df)} units)")

    # 3. Transition PSTHs
    states_seq = sdf.sort_values("trial_idx")["hmm_state"].values
    trans = [(states_seq[i], states_seq[i + 1])
             for i in range(len(states_seq) - 1)
             if states_seq[i] != states_seq[i + 1]]
    if trans:
        mc = Counter(trans).most_common(1)[0]
        from_s, to_s = mc[0]
        n_trans = mc[1]
        if n_trans >= min_transitions:
            print(f"    Transition PSTHs: {state_labels[from_s]}→{state_labels[to_s]} "
                  f"({n_trans} transitions)")
            n_trans_plotted = 0
            for cid in subset[:10]:
                try:
                    ok = plot_transition_psth(
                        session, assignments_df, cid, from_s, to_s,
                        state_labels, out_dir,
                        event_name=event_name, window=window,
                        bin_size=bin_size, sigma_ms=sigma_ms,
                        min_transitions=min_transitions,
                        normalize=normalize,
                    )
                    if ok:
                        n_trans_plotted += 1
                except Exception as exc:
                    print(f"      Unit {cid} transition PSTH failed: {exc}")
            print(f"    Plotted {n_trans_plotted} transition PSTHs")
        else:
            print(f"    Skipping transitions: only {n_trans} found "
                  f"(need >={min_transitions})")


def _pick_comparison_pair(active_states: dict, state_labels: list):
    """Pick best pair of active states for modulation comparison.

    Prefers Engaged vs Disengaged; falls back to Engaged vs Biased,
    then any two active states.
    """
    active_keys = list(active_states.keys())
    if len(active_keys) < 2:
        return None, None

    engaged = [k for k in active_keys if "Engaged" in state_labels[k]]
    disengaged = [k for k in active_keys if "Disengaged" in state_labels[k]]
    biased = [k for k in active_keys if "Biased" in state_labels[k]]

    if engaged and disengaged:
        return engaged[0], disengaged[0]
    if engaged and biased:
        return engaged[0], biased[0]
    # Fall back to any two
    return active_keys[0], active_keys[1]


# =====================================================================
# Pooled cross-session analysis
# =====================================================================

def pooled_population_psth(
    sessions: list,
    assignments_df: pd.DataFrame,
    state_labels: list,
    n_states: int,
    out_dir: Path,
    min_fr: float = 1.0,
    max_units_per_session: int = 30,
    min_trials: int = 15,
    event_name: str = "Change_ON",
    window=(-0.5, 1.0),
    bin_size: float = 0.025,
    sigma_ms: float = 25.0,
    normalize: str = "zscore",
    unit_select: str = "responsive",
    tf_dir: Path | None = None,
    z_thresh_tf: float = 3.0,
):
    """Compute grand-average population PSTH per state, pooled across sessions.

    Each unit’s mean PSTH is normalised (default: z-scored against its own
    pre-event baseline) before averaging across units, so every neuron
    contributes equally regardless of absolute firing rate.
    """
    from visdetect.analysis.align import (
        get_event_times_by_trial,
        align_spikes_to_events,
    )

    palette = _state_palette(n_states)
    pooled_dir = out_dir / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)

    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bc = (bins[:-1] + bins[1:]) / 2.0
    n_bins = len(bc)

    # Collect per-state firing rate vectors: list of 1-D arrays (each = 1 trial-unit combo)
    state_fr_lists = {k: [] for k in range(n_states)}
    state_trial_counts = {k: 0 for k in range(n_states)}
    n_units_total = 0

    for session in sessions:
        sname = session.session_name or ""
        if unit_select.startswith("tf-"):
            tf_type = unit_select.replace("tf-", "")
            if tf_dir is None:
                print(f"    SKIP {sname}: --tf-dir required for unit-select={unit_select}")
                continue
            subset = select_tf_responsive_units(
                session, tf_dir, tf_type=tf_type, min_fr=min_fr,
                max_units=max_units_per_session, z_thresh=z_thresh_tf,
            )
        else:
            subset = select_units(
                session, min_fr=min_fr, max_units=max_units_per_session,
                event_name=event_name, rank_by_responsiveness=True,
            )
        if not subset:
            continue
        n_units_total += len(subset)

        try:
            all_event_times = get_event_times_by_trial(session, event_name)
        except Exception:
            continue

        for cid in subset:
            spike_times = None
            for c in session.clusters:
                if c.cluster_id == cid:
                    spike_times = c.spike_times
                    break
            if spike_times is None or len(spike_times) == 0:
                continue

            for k in range(n_states):
                trial_idx = get_state_trial_indices(assignments_df, sname, k)
                # Apply trial-type filter
                valid_outcomes = EVENT_VALID_OUTCOMES.get(event_name, None)
                trials = getattr(session, "trials", []) or []
                valid_events = []
                for ti in trial_idx:
                    if ti >= len(all_event_times) or np.isnan(all_event_times[ti]):
                        continue
                    if valid_outcomes is not None and ti < len(trials):
                        oc = getattr(trials[ti], "trialoutcome", "").lower()
                        if oc not in valid_outcomes:
                            continue
                    valid_events.append(all_event_times[ti])
                if len(valid_events) < 5:
                    continue

                fr_matrix, _ = align_spikes_to_events(
                    spike_times, valid_events, window=window, bin_size=bin_size,
                )
                if fr_matrix.shape[0] > 0:
                    # Store the mean PSTH for this unit in this state
                    unit_psth = fr_matrix.mean(axis=0)
                    # Normalize per-unit before pooling
                    unit_psth = _normalize_psth(
                        smooth_psth(unit_psth, bin_size=bin_size, sigma_ms=sigma_ms),
                        bc, mode=normalize,
                    )
                    state_fr_lists[k].append(unit_psth)
                    state_trial_counts[k] += fr_matrix.shape[0]

    # Plot grand-average population PSTH per state
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(n_states):
        if len(state_fr_lists[k]) < 3:
            continue
        mat = np.array(state_fr_lists[k])  # (n_unit_sessions, n_bins)
        # Already normalised per-unit; just average and compute SEM
        mean_psth = mat.mean(axis=0)
        sem = mat.std(axis=0) / np.sqrt(mat.shape[0])
        n_unit_sess = mat.shape[0]
        ax.fill_between(bc, mean_psth - sem, mean_psth + sem,
                        alpha=0.2, color=palette[k])
        ax.plot(bc, mean_psth, color=palette[k], linewidth=2,
                label=f"{state_labels[k]} ({n_unit_sess} unit-sessions, "
                      f"{state_trial_counts[k]} trials)")

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(_ylabel_for_mode(normalize))
    ax.set_title(f"Grand-Average Population PSTH by HMM State\n"
                 f"({len(sessions)} sessions, {n_units_total} total unit-selections)")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_xlim(window)
    despine(ax)
    plt.tight_layout()
    fig.savefig(pooled_dir / "population_psth_by_state.png", dpi=150)
    plt.close(fig)
    print(f"  Pooled population PSTH saved: {pooled_dir / 'population_psth_by_state.png'}")

    # Summary stats
    for k in range(n_states):
        n = len(state_fr_lists[k])
        print(f"    {state_labels[k]}: {n} unit-sessions, "
              f"{state_trial_counts[k]} total trials")

    return state_fr_lists, state_trial_counts


def pooled_modulation_index(
    sessions: list,
    assignments_df: pd.DataFrame,
    state_labels: list,
    n_states: int,
    out_dir: Path,
    event_name: str = "Change_ON",
):
    """Compute modulation index across ALL sessions, pooled."""
    pooled_dir = out_dir / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)

    # Find best comparison pair from the full dataset
    counts = assignments_df["hmm_state"].value_counts()
    active = {k: int(counts.get(k, 0)) for k in range(n_states) if counts.get(k, 0) >= 50}
    state_a, state_b = _pick_comparison_pair(active, state_labels)
    if state_a is None or state_b is None:
        print("  Pooled MI: insufficient states with enough trials")
        return

    all_mi_rows = []
    for session in sessions:
        sname = session.session_name or ""
        sdf = assignments_df[assignments_df["session_name"] == sname]
        na = int((sdf["hmm_state"] == state_a).sum())
        nb = int((sdf["hmm_state"] == state_b).sum())
        if na < 10 or nb < 10:
            continue

        cluster_ids = (
            session.good_and_stable_ids
            or session.good_cluster_ids
            or [c.cluster_id for c in session.clusters]
        )
        mi_df = compute_population_state_modulation(
            session, assignments_df, state_a, state_b,
            cluster_ids=cluster_ids,
            event_name=event_name, response_window=(0.0, 0.3),
        )
        if len(mi_df) > 0:
            mi_df["session_name"] = sname
            all_mi_rows.append(mi_df)

    if not all_mi_rows:
        print("  No sessions had enough trials in both states for pooled MI")
        return

    pooled_mi = pd.concat(all_mi_rows, ignore_index=True)
    pooled_mi.to_csv(pooled_dir / "pooled_modulation_index.csv", index=False)
    print(f"  Pooled MI: {len(pooled_mi)} unit-sessions across "
          f"{len(all_mi_rows)} sessions")
    print(f"    Median MI = {pooled_mi['modulation_index'].median():.3f}")

    plot_modulation_histogram(
        pooled_mi, state_labels, state_a, state_b,
        f"Pooled ({len(all_mi_rows)} sessions)", pooled_dir,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="State-conditioned neural analysis (improved)."
    )
    parser.add_argument("--data-dir", required=True,
                        help="HMM results directory (model + assignments).")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of states to load (e.g. 3). "
                             "Default: highest-K model found on disk.")
    parser.add_argument("--pkl-dir", required=True,
                        help="Directory with session pkl files.")
    parser.add_argument("--manifest", default=None,
                        help="Staging manifest CSV.")
    parser.add_argument("--session", default=None,
                        help="Single session name to process (optional).")
    parser.add_argument("--out", default="FIGURES/behavior/hmm/neural",
                        help="Output directory for neural plots.")
    parser.add_argument("--max-units", type=int, default=20,
                        help="Max units to plot per session.")
    parser.add_argument("--min-fr", type=float, default=1.0,
                        help="Minimum firing rate (Hz) to include a unit.")
    parser.add_argument("--min-trials", type=int, default=15,
                        help="Minimum trials per state to plot.")
    parser.add_argument("--min-transitions", type=int, default=3,
                        help="Minimum transition count for transition PSTHs.")
    parser.add_argument("--sigma-ms", type=float, default=25.0,
                        help="Gaussian smoothing sigma (ms).")
    parser.add_argument("--event", default="Change_ON",
                        help="NI event to align to.")
    parser.add_argument("--window-pre", type=float, default=0.5)
    parser.add_argument("--window-post", type=float, default=1.0)
    parser.add_argument("--bin-size", type=float, default=0.025,
                        help="Bin size in seconds (default 25 ms).")
    parser.add_argument("--exclude-qc-fail", action="store_true",
                        help="DEPRECATED: SESSION_FILTER handles QC.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Bypass SESSION_FILTER.")
    parser.add_argument("--skip-pooled", action="store_true",
                        help="Skip the pooled cross-session analysis.")
    parser.add_argument("--skip-per-session", action="store_true",
                        help="Skip per-session analysis (only do pooled).")
    parser.add_argument("--unit-select", default="responsive",
                        choices=["responsive", "tf-fast", "tf-slow", "tf-any"],
                        help="Unit selection strategy: "
                             "'responsive' = rank by visual responsiveness (default), "
                             "'tf-fast' = TF fast-pulse responsive, "
                             "'tf-slow' = TF slow-pulse responsive, "
                             "'tf-any' = any TF responsive.")
    parser.add_argument("--tf-dir", default=None,
                        help="Root TF screening directory (FIGURES/tf). "
                             "Required when --unit-select is tf-*.")
    parser.add_argument("--z-thresh-tf", type=float, default=3.0,
                        help="Z-score threshold for TF responsiveness (default 3.0).")
    parser.add_argument("--normalize", default="none",
                        choices=["none", "zscore", "baseline-subtract"],
                        help="PSTH normalization: 'none' = raw Hz (default for "
                             "per-unit), 'zscore' = z-score to pre-event "
                             "baseline, 'baseline-subtract' = subtract baseline "
                             "mean.  Pooled plots always use zscore unless "
                             "--normalize none is set explicitly.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style(context="talk")

    # Load HMM results
    model, assignments_df, state_labels = load_hmm_results(
        Path(args.data_dir), K=args.K,
    )
    K = model.n_states
    # Validate TF args
    tf_dir = Path(args.tf_dir) if args.tf_dir else None
    if args.unit_select.startswith("tf-") and tf_dir is None:
        parser.error("--tf-dir is required when --unit-select is tf-*")

    print(f"Loaded K={K} model, labels={state_labels}")
    if args.unit_select != "responsive":
        print(f"Unit selection: {args.unit_select} (z>={args.z_thresh_tf})")

    window = (-args.window_pre, args.window_post)
    pkl_dir = Path(args.pkl_dir)

    # Determine which sessions to process
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

    # Load all sessions
    loaded_sessions = []
    for sname in session_names:
        candidates = list(pkl_dir.glob(f"*{sname}*.pkl"))
        if not candidates:
            print(f"  SKIP {sname}: pkl not found")
            continue
        try:
            session = load_session(str(candidates[0]))
            loaded_sessions.append(session)
        except Exception as exc:
            print(f"  SKIP {sname}: {exc}")
            continue

    print(f"\nLoaded {len(loaded_sessions)} sessions")

    # ---- Per-session analysis ----
    if not args.skip_per_session:
        print("\n" + "=" * 60)
        print("PER-SESSION ANALYSIS")
        print("=" * 60)
        for session in loaded_sessions:
            process_session(
                session, assignments_df, state_labels, K, out_dir,
                max_units=args.max_units,
                min_fr=args.min_fr,
                min_trials=args.min_trials,
                min_transitions=args.min_transitions,
                event_name=args.event,
                window=window,
                bin_size=args.bin_size,
                sigma_ms=args.sigma_ms,
                unit_select=args.unit_select,
                tf_dir=tf_dir,
                z_thresh_tf=args.z_thresh_tf,
                normalize=args.normalize,
            )

    # ---- Pooled cross-session analysis ----
    if not args.skip_pooled and len(loaded_sessions) > 1:
        print("\n" + "=" * 60)
        print("POOLED CROSS-SESSION ANALYSIS")
        print("=" * 60)
        pooled_population_psth(
            loaded_sessions, assignments_df, state_labels, K, out_dir,
            min_fr=args.min_fr,
            max_units_per_session=args.max_units + 10,
            min_trials=5,  # lower threshold since pooling
            event_name=args.event,
            window=window,
            bin_size=args.bin_size,
            sigma_ms=args.sigma_ms,
            normalize=args.normalize if args.normalize != "none" else "zscore",
            unit_select=args.unit_select,
            tf_dir=tf_dir,
            z_thresh_tf=args.z_thresh_tf,
        )
        pooled_modulation_index(
            loaded_sessions, assignments_df, state_labels, K, out_dir,
            event_name=args.event,
        )

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
