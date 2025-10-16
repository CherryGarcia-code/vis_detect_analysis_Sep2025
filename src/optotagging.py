"""Optotagging analysis helpers

This module contains functions to detect and summarize optotagged units
from session objects produced by your loader.

The implementations are intentionally lightweight stubs to be filled in
with project-specific logic (waveform comparison, latency distributions,
laser-evoked PSTHs, etc.).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def detect_optotagged_units(
    session: Any,
    stim_event: str = "Laser_ON",
    win: Tuple[float, float] = (0.0, 0.02),
    threshold: float = 5.0,
) -> Dict[int, Dict[str, Any]]:
    """Detect putative optotagged units in a session.

    Args:
        session: session object loaded with `load_session`.
        stim_event: event name for laser onset used when computing PSTHs.
        win: time window (s) after stimulus to consider evoked activity.
        threshold: simple threshold on evoked spike-rate increase (a.u.).

    Returns:
        A dict keyed by cluster id with summary metrics.
    """
    # Placeholder: real implementation should compute trial-aligned PSTHs
    # and calculate per-cluster metrics (latency, z-score, reliability).
    results = {}
    clusters = getattr(session, "good_cluster_ids", getattr(session, "clusters", []))
    for cid in clusters:
        results[int(cid)] = {
            "cluster_id": int(cid),
            "is_optotagged": False,
            "evoked_rate": np.nan,
            "latency_ms": np.nan,
            "notes": "stub - implement compute_psth and detection rules",
        }
    return results


def summarize_optotagging(detections: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Convert detections dict to a tidy DataFrame for reporting."""
    rows = []
    for cid, info in detections.items():
        r = info.copy()
        rows.append(r)
    df = pd.DataFrame(rows)
    if "cluster_id" in df.columns:
        df = df.set_index("cluster_id")
    return df
