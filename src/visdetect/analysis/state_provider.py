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


import re
from typing import List, Optional

from visdetect.core.session import Session

_HMM_CANONICAL = {
    "Stimulus_sensitive": IN_ZONE,
    "StimSens": IN_ZONE,          # behavioral-state labeler vocabulary
    "Impulsive": IMPULSIVE,
    "Disengaged": DISENGAGED,
}


def canonical_from_hmm_label(label: str) -> Optional[str]:
    """Map an HMM label to the canonical vocabulary; None if not one of the three.

    Strips a trailing rank suffix ('_1', '_2') produced by
    hmm.auto_label_states_explicit for duplicate states. 'Intermediate_*' has no
    canonical equivalent -> None (trial gets no state -> excluded from in_zone).
    """
    base = re.sub(r"_\d+$", "", str(label))
    return _HMM_CANONICAL.get(base)


def rows_from_decoded_df(df) -> List[Tuple[int, str, float]]:
    """Convert a decode_session DataFrame to state-table rows.

    Requires columns 'trial_idx' (raw index), 'hmm_state_label', 'p_state_max'.
    Rows whose label has no canonical mapping are dropped.
    """
    rows: List[Tuple[int, str, float]] = []
    for _, r in df.iterrows():
        canon = canonical_from_hmm_label(r["hmm_state_label"])
        if canon is None:
            continue
        rows.append((int(r["trial_idx"]), canon, float(r["p_state_max"])))
    return rows


def rows_from_tag_df(df, *, label_col: str = "state_label",
                     gate_col: str = "state_gated",
                     conf_col: str = "state_confidence",
                     use_gating: bool = True,
                     ungated_value: int = -1) -> List[Tuple[int, str, float]]:
    """Convert a behavioral-state-labeler tag table to canonical state-table rows.

    The labeler (scripts/state_labeling/, data/cache/state_tags/<subject>/<sess>.csv)
    emits its own vocabulary (StimSens/Impulsive/Disengaged/Abort/...). This maps
    `label_col` -> the canonical 3-state set via canonical_from_hmm_label; rows with
    no canonical equivalent (e.g. Abort) are dropped.

    Gating: `gate_col` is the GATED state index, with `ungated_value` (-1) marking
    trials that did not pass the labeler's confidence gate. When use_gating, those
    are dropped. Confidence comes from `conf_col` (1.0 if absent/NaN).
    """
    rows: List[Tuple[int, str, float]] = []
    has_gate = gate_col in df.columns
    has_conf = conf_col in df.columns
    for _, r in df.iterrows():
        if use_gating and has_gate and int(r[gate_col]) == ungated_value:
            continue
        canon = canonical_from_hmm_label(r[label_col])
        if canon is None:
            continue
        conf = float(r[conf_col]) if has_conf and pd.notna(r[conf_col]) else 1.0
        rows.append((int(r["trial_idx"]), canon, conf))
    return rows


class UniformInZoneStateProvider:
    """Bootstrap provider: labels EVERY valid trial 'in_zone' (confidence 1.0).

    Temporary — lets the curation pipeline run end-to-end before the final
    state-identification method exists. Equivalent to all-trials fingerprinting.
    """

    def write(self, session: Session, session_name: str, states_dir) -> Path:
        from visdetect.analysis.behavior import get_trial_dataframe
        df = get_trial_dataframe(session)
        rows = [(int(i), IN_ZONE, 1.0) for i in df["trial_idx"].tolist()]
        return write_state_table(session_name, rows, states_dir)


class HMMStateProvider:
    """Provider wrapping a fitted GLM-HMM via hmm.decode_session."""

    def __init__(self, model, state_labels: List[str]):
        self.model = model
        self.state_labels = state_labels

    def write(self, session: Session, session_name: str, states_dir) -> Path:
        from visdetect.analysis.hmm import decode_session
        df = decode_session(self.model, session, state_labels=self.state_labels)
        if "p_state_max" not in df.columns:
            pcols = [c for c in df.columns if c.startswith("p_state_")]
            df = df.copy()
            df["p_state_max"] = df[pcols].max(axis=1) if pcols else 1.0
        rows = rows_from_decoded_df(df)
        return write_state_table(session_name, rows, states_dir)
