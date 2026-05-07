"""SpikeGLX metadata parsing utilities.

Canonical implementations for reading .meta files, extracting sample rates,
and finding CatGT sync-edge output files.  Consolidates logic previously
duplicated in split_by_shank.py, SGLXMetaToCoords.py, and
validate_metadata_duration.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def read_meta(meta_path: Path) -> dict:
    """Parse a SpikeGLX ``.meta`` file into a key-value dict.

    Parameters
    ----------
    meta_path : Path
        Path to the ``.ap.meta`` or ``.lf.meta`` file.

    Returns
    -------
    dict
        Keys are meta-field names (leading ``~`` stripped), values are strings.
    """
    meta: dict = {}
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            meta[key.lstrip("~")] = value
    return meta


def get_sample_rate(meta: dict) -> float:
    """Return the AP-band sample rate from parsed metadata.

    Parameters
    ----------
    meta : dict
        Output of :func:`read_meta`.

    Returns
    -------
    float
        Sample rate in Hz (typically 30000.0).
    """
    return float(meta["imSampRate"])


def find_sync_edge_file(probe_dir: Path) -> Optional[Path]:
    """Find the CatGT sync-edge text file in a probe directory.

    CatGT writes sync edge times to files matching the pattern
    ``*tcat.imec*.ap.xd_*.txt``.

    Parameters
    ----------
    probe_dir : Path
        Path to a probe directory (e.g., ``.../Kilosort&Phy/BG_039_..._imec0``).

    Returns
    -------
    Path or None
        Path to the sync-edge file, or ``None`` if not found.
    """
    matches = sorted(probe_dir.glob("*tcat.imec*.ap.xd_*.txt"))
    return matches[0] if matches else None


def write_ni_sync_txt(rise_times: np.ndarray, output_path: Path) -> None:
    """Write NI-DAQ sync pulse rise times to a text file.

    Each rise time is written on its own line with 6 decimal places,
    matching the format expected by TPrime's ``-tostream`` argument.

    Parameters
    ----------
    rise_times : np.ndarray
        1-D array of sync pulse rise times in seconds.
    output_path : Path
        Destination file path (e.g., ``Nidaq/NI_Sync.txt``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(output_path), rise_times, fmt="%.6f")
