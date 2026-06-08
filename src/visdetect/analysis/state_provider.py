"""Pluggable behavioural-state interface for track curation.

The curation pipeline consumes a per-session trial->state table; it never
imports any state model. The HMM is one provider; a hand/ethogram labeler can
write the same CSV later. See
docs/superpowers/specs/2026-06-07-track-curation-design.md sec 4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

# ── Canonical 3-state vocabulary ──────────────────────────────────────
DISENGAGED = "disengaged"
IMPULSIVE = "impulsive"
IN_ZONE = "in_zone"
CANONICAL_STATES: Tuple[str, str, str] = (DISENGAGED, IMPULSIVE, IN_ZONE)


def state_table_path(session_name: str, states_dir) -> Path:
    return Path(states_dir) / f"{str(session_name).zfill(8)}_states.csv"


def write_state_table(session_name: str,
                      rows: Sequence[Tuple[int, str, float]],
                      states_dir) -> Path:
    """Write a per-session state table. rows = (trial_idx, state_label, confidence).

    trial_idx MUST index into session.trials (raw trial order) — the same space
    build_population_tensor / extract_unit_psths use. NOT the HMM valid-trial
    ordering. See spec sec 4.2 (index-space contract).
    """
    for _, label, _ in rows:
        if label not in CANONICAL_STATES:
            raise ValueError(
                f"state_label {label!r} not in canonical {CANONICAL_STATES}")
    states_dir = Path(states_dir)
    states_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["trial_idx", "state_label", "confidence"])
    df["trial_idx"] = df["trial_idx"].astype(int)
    out = state_table_path(session_name, states_dir)
    df.to_csv(out, index=False)
    return out


def load_state_table(session_name: str, states_dir
                     ) -> Dict[int, Tuple[str, float]]:
    """Return {raw trial_idx -> (state_label, confidence)}; {} if no file."""
    path = state_table_path(session_name, states_dir)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(r["trial_idx"]): (str(r["state_label"]), float(r["confidence"]))
            for _, r in df.iterrows()}


def in_zone_trial_indices(session_name: str, states_dir,
                          min_confidence: float = 0.0) -> List[int]:
    """Sorted raw trial indices labeled in_zone with confidence >= floor."""
    table = load_state_table(session_name, states_dir)
    return sorted(t for t, (lab, conf) in table.items()
                  if lab == IN_ZONE and conf >= min_confidence)
