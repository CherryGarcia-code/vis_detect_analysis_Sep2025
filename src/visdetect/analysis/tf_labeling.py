"""TF manual labeling infrastructure — label I/O, queue logic, data loading.

Provides the data layer for the interactive TF labeling GUI:
  - Label record model and CSV persistence (append-safe, crash-safe)
  - Smart priority queue for ordering units to review
  - Data loading helpers that merge NPZ traces with algorithmic classification

The label file lives at ``data/labels/tf_manual_labels.csv`` and grows
incrementally as the reviewer works through the queue.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Compute ROOT locally to avoid importing visdetect.analysis.config, which
# triggers visdetect.core.__init__ → qc.py → matplotlib.use("Agg") and
# kills any interactive backend the caller may have set.
ROOT = str(Path(__file__).resolve().parents[3])

# ── Paths ──────────────────────────────────────────────────────────────
LABELS_DIR = os.path.join(ROOT, "data", "labels")
LABELS_PATH = os.path.join(LABELS_DIR, "tf_manual_labels.csv")
CLASSIFICATION_CSV = os.path.join(ROOT, "data", "cache", "tf_labeling",
                                  "tf_cell_classification.csv")
TF_TRACES_DIR = os.path.join(ROOT, "data", "cache", "tf_traces", "BG_046")
RASTER_CACHE_DIR = os.path.join(ROOT, "data", "cache", "tf_raster_cache")

# ── Tier constants (match g_tf_cell_classifier.py) ─────────────────────
TIER_SPLITTER = "Tier 1 (Splitter)"
TIER_UNILATERAL = "Tier 2 (Unilateral)"
TIER_OMNI = "Tier 3 (Omni)"
TIER_NONE = "Non-responsive"

VALID_TIERS = [TIER_SPLITTER, TIER_UNILATERAL, TIER_OMNI, TIER_NONE]

VALID_SUB_TYPES = {
    TIER_SPLITTER: ["Fast+/Slow-", "Slow+/Fast-"],
    TIER_UNILATERAL: ["Fast+", "Fast-", "Slow+", "Slow-"],
    TIER_OMNI: ["Both+", "Both-"],
    TIER_NONE: ["None"],
}

TIER_COLORS = {
    TIER_SPLITTER: "#8E24AA",
    TIER_UNILATERAL: "#FB8C00",
    TIER_OMNI: "#43A047",
    TIER_NONE: "#BDBDBD",
}

LABEL_COLUMNS = [
    "session_name", "cluster_id",
    "manual_tier", "manual_sub_type", "confidence", "notes",
    "algo_tier", "algo_sub_type", "reviewer", "timestamp",
]


# ── Label record ───────────────────────────────────────────────────────

@dataclass
class LabelRecord:
    session_name: int
    cluster_id: int
    manual_tier: str
    manual_sub_type: str
    confidence: str = "high"       # high / medium / low
    notes: str = ""
    algo_tier: str = ""
    algo_sub_type: str = ""
    reviewer: str = "BG"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(
                timespec="seconds")


# ── Label I/O ──────────────────────────────────────────────────────────

def load_labels(path: Optional[str] = None) -> pd.DataFrame:
    """Load existing manual labels, or return empty DataFrame."""
    path = path or LABELS_PATH
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Ensure expected columns
        for col in LABEL_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=LABEL_COLUMNS)


def save_label(record: LabelRecord, path: Optional[str] = None) -> None:
    """Append or update a single label (crash-safe via atomic write)."""
    path = path or LABELS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = load_labels(path)
    row = asdict(record)

    # Check if this unit already has a label — update in place
    mask = ((df["session_name"].astype(int) == int(record.session_name)) &
            (df["cluster_id"].astype(int) == int(record.cluster_id)))

    if mask.any():
        idx = df.index[mask][0]
        for col, val in row.items():
            df.at[idx, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Atomic write: write to temp, then rename
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def get_label_stats(path: Optional[str] = None) -> Dict:
    """Return summary statistics of current labels."""
    df = load_labels(path)
    n = len(df)
    if n == 0:
        return {"total": 0, "by_tier": {}, "by_confidence": {}}
    return {
        "total": n,
        "by_tier": df["manual_tier"].value_counts().to_dict(),
        "by_confidence": df["confidence"].value_counts().to_dict(),
    }


# ── Priority queue ─────────────────────────────────────────────────────

def _compute_priority(row: pd.Series) -> float:
    """Score a unit for review priority (higher = review sooner).

    Priority ranking:
      1. Borderline cases (near decision boundaries)
      2. Algorithmically classified responsive units (verify true positives)
      3. High-z non-responsive (potential false negatives)
      4. Low-z non-responsive (likely true negatives)
    """
    tier = row.get("tier", TIER_NONE)
    z = abs(row.get("z_abs_max_npz", 0.0))
    trend = abs(row.get("trend_ratio", 0.0))

    # Rescued splitters (check if mirror score is just above threshold)
    mirror = row.get("mirror_score", 0.0)
    if not np.isfinite(mirror):
        mirror = 0.0

    # Permutation p-values (lower = more significant)
    p_fast = row.get("p_peak_fast", 1.0)
    p_slow = row.get("p_peak_slow", 1.0)
    if not np.isfinite(p_fast):
        p_fast = 1.0
    if not np.isfinite(p_slow):
        p_slow = 1.0
    min_p = min(p_fast, p_slow)

    # Base priority by tier
    if tier == TIER_NONE:
        if z >= 1.5:
            # High-z non-responsive: potential false negatives
            base = 300 + z * 20
        else:
            # Low-z: true negatives, low priority
            base = z * 10
    else:
        # All responsive tiers: verify these first
        base = 500 + z * 10

    # Boost borderline cases (p-values near alpha thresholds)
    if 0.005 < min_p < 0.05:
        base += 200  # near the strict alpha boundary
    elif 0.05 < min_p < 0.15:
        base += 100  # near the conjunction alpha boundary

    # Boost trend-excluded (might be wrong)
    if row.get("sub_type", "") == "Trend-excluded":
        base += 250

    # Boost if trend_ratio is borderline (0.3-0.7)
    if 0.3 < trend < 0.7:
        base += 50

    return float(base)


def get_labeling_queue(
    classification_csv: Optional[str] = None,
    labels_path: Optional[str] = None,
    include_labeled: bool = False,
) -> pd.DataFrame:
    """Return units ordered by review priority.

    Parameters
    ----------
    classification_csv : str, optional
        Path to algorithmic classification CSV.
    labels_path : str, optional
        Path to manual labels CSV. Already-labeled units are excluded
        unless *include_labeled* is True.
    include_labeled : bool
        If True, include already-labeled units (for re-review).

    Returns
    -------
    pd.DataFrame
        Classification rows sorted by descending priority, with a
        ``priority`` column and ``is_labeled`` flag.
    """
    csv_path = classification_csv or CLASSIFICATION_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Classification CSV not found: {csv_path}\n"
            "Run g_tf_cell_classifier.py first.")

    df = pd.read_csv(csv_path)
    df["priority"] = df.apply(_compute_priority, axis=1)

    # Mark already-labeled units
    labels = load_labels(labels_path)
    if not labels.empty:
        labeled_keys = set(
            zip(labels["session_name"].astype(int),
                labels["cluster_id"].astype(int)))
        df["is_labeled"] = df.apply(
            lambda r: (int(r["session_name"]), int(r["cluster_id"])) in labeled_keys,
            axis=1)
    else:
        df["is_labeled"] = False

    if not include_labeled:
        df = df[~df["is_labeled"]].copy()

    df = df.sort_values("priority", ascending=False).reset_index(drop=True)
    return df


# ── Data loading for GUI ───────────────────────────────────────────────

def load_unit_traces(session_name: int, cluster_id: int) -> Optional[Dict]:
    """Load z-scored traces for a single unit from the NPZ cache.

    Returns dict with keys: t_vec, fast_z, slow_z, fast_z_sem, slow_z_sem,
    z_max_fast, z_min_fast, z_max_slow, z_min_slow.
    Returns None if not found.
    """
    sname_padded = str(int(session_name)).zfill(8)
    npz_path = os.path.join(TF_TRACES_DIR,
                            f"BG_046_{sname_padded}_traces.npz")
    if not os.path.exists(npz_path):
        return None
    try:
        npz = np.load(npz_path, allow_pickle=False)
        cids = npz["cluster_ids"].astype(int)
        idx = np.where(cids == int(cluster_id))[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        return {
            "t_vec": npz["t_vec"],
            "fast_z": npz["fast_z"][i],
            "slow_z": npz["slow_z"][i],
            "fast_z_sem": npz["fast_z_sem"][i],
            "slow_z_sem": npz["slow_z_sem"][i],
            "z_max_fast": float(npz["z_max_fast"][i]),
            "z_min_fast": float(npz["z_min_fast"][i]),
            "z_max_slow": float(npz["z_max_slow"][i]),
            "z_min_slow": float(npz["z_min_slow"][i]),
        }
    except Exception:
        return None


def load_unit_rasters(session_name: int, cluster_id: int) -> Optional[Dict]:
    """Load pre-cached raster data for a single unit.

    Returns dict with keys: fast_raster (list of arrays), slow_raster,
    n_fast_pulses, n_slow_pulses, t_range.
    Returns None if not cached yet.
    """
    sname = str(int(session_name)).zfill(8)
    cache_path = os.path.join(RASTER_CACHE_DIR,
                              f"{sname}_{cluster_id}_raster.npz")
    if not os.path.exists(cache_path):
        return None
    try:
        npz = np.load(cache_path, allow_pickle=True)
        return {
            "fast_raster": npz["fast_raster"],
            "slow_raster": npz["slow_raster"],
            "n_fast_pulses": int(npz["n_fast_pulses"]),
            "n_slow_pulses": int(npz["n_slow_pulses"]),
            "t_range": npz["t_range"],
        }
    except Exception:
        return None
