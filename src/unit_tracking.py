"""Utilities to attempt tracking the same neurons across sessions.

This module provides helper functions and data structures to compute pairwise
unit similarity (waveforms, firing statistics) and to propose matches across
sessions. Implementations are stubs and will need project-specific thresholds
and gating logic.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_unit_similarity(sess_a: Any, sess_b: Any, cluster_id_a: int, cluster_id_b: int) -> Dict[str, float]:
    """Compute similarity metrics between two clusters across sessions.

    Returns a dict with waveform_corr, isi_ks, firing_rate_ratio, and a
    composite score.
    """
    # Placeholder: extract waveforms and compute correlation; compute ISI stats
    return {
        'waveform_corr': 0.0,
        'isi_ks': 1.0,
        'firing_rate_ratio': 1.0,
        'composite_score': 0.0
    }


def propose_matches(sess_a: Any, sess_b: Any, top_k: int = 5) -> pd.DataFrame:
    """Return a DataFrame listing top-k candidate matches between sessions.

    Columns: cluster_a, cluster_b, composite_score, waveform_corr, isi_ks, firing_rate_ratio
    """
    # Placeholder: iterate over possible cluster pairs and compute similarity
    rows = []
    clusters_a = getattr(sess_a, 'good_cluster_ids', [])
    clusters_b = getattr(sess_b, 'good_cluster_ids', [])
    for a in clusters_a:
        for b in clusters_b:
            rows.append({'cluster_a': int(a), 'cluster_b': int(b), 'composite_score': 0.0,
                         'waveform_corr': 0.0, 'isi_ks': 1.0, 'firing_rate_ratio': 1.0})
    df = pd.DataFrame(rows)
    return df.sort_values('composite_score', ascending=False).head(top_k)
