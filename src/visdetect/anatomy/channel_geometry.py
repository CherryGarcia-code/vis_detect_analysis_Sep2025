# src/visdetect/anatomy/channel_geometry.py
"""Shank assignment + chanmap signature for NP2.0 four-shank probes."""
from __future__ import annotations

import hashlib

import numpy as np


def assign_shanks(channel_positions: np.ndarray, n_shanks: int = 4,
                  gap_um: float = 120.0) -> np.ndarray:
    """Per-channel probe shank index (0..n_shanks-1), ordered by ascending x.

    Shanks are detected as clusters of x separated by gaps > gap_um (NP2.0 shank
    pitch ~250 um, within-shank column spacing ~32 um).
    """
    x = np.asarray(channel_positions, float)[:, 0]
    order = np.argsort(np.unique(x))
    ux = np.unique(x)
    # group unique x values into shanks by gaps
    group_of_ux = np.zeros(len(ux), dtype=int)
    g = 0
    for i in range(1, len(ux)):
        if ux[i] - ux[i - 1] > gap_um:
            g += 1
        group_of_ux[i] = g
    n_found = g + 1
    if n_found != n_shanks:
        raise ValueError(f"expected {n_shanks} shanks, found {n_found} (gap_um={gap_um})")
    ux_to_group = {v: int(group_of_ux[i]) for i, v in enumerate(ux)}
    return np.array([ux_to_group[v] for v in x], dtype=int)


def chanmap_signature(channel_positions: np.ndarray) -> str:
    """Order-independent hex hash of the (x,y) site set, rounded to 1 um."""
    pos = np.round(np.asarray(channel_positions, float), 1)
    rows = sorted(map(tuple, pos.tolist()))
    h = hashlib.sha1(repr(rows).encode("utf-8")).hexdigest()
    return h[:16]
