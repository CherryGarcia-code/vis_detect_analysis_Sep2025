"""B8 — behavioral decision-latents decomposed by state (Phase 1: descriptive).

Plain English: for every trial we want three numbers — 'sharpness' (how clearly
the mouse can tell the grating changed), 'itchiness' (how trigger-happy it is
before any real change), and 'timing' (how strongly it expects the change now).
Phase 1 measures these directly from behaviour, split by the mouse's mood
(Impulsive vs StimSens), across learning. No model fitting here.
"""
from __future__ import annotations
import os
import pandas as pd

MAIN_MOODS = ("Impulsive", "StimSens")
SEPARATE_MOODS = ("Disengaged",)
EXCLUDED_MOODS = ("Abort",)
_DEFAULT_TAG_DIR = os.path.join("data", "cache", "state_tags")

def load_state_labels(session_name, subject="BG_046", tag_dir=None):
    base = os.path.join(tag_dir or _DEFAULT_TAG_DIR, subject)
    candidates = [str(session_name)]
    try:
        candidates.append(str(int(session_name)).zfill(8))  # leading-zero form
    except (TypeError, ValueError):
        pass
    for cand in candidates:
        path = os.path.join(base, f"{cand}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[df["trial_idx"].notna()].copy()
            df["trial_idx"] = df["trial_idx"].astype(int)
            return df.set_index("trial_idx")[["state_label", "state_confidence"]]
    raise FileNotFoundError(f"No state-tag file for {session_name} under {base}")
