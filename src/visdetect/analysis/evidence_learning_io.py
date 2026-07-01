"""B10 evidence_learning — multi-subject data loaders (library).

Reusable across the B10 scripts (behavioral / neural / state). Kept in the
library (not scripts/) because it is reusable logic and must be importable by
tests. Spans BG_046 + BG_039 (DMS) and BG_031 (VMS) — NOT the single-subject
suite loader.

Id handling: everything joins on ``config.session_date_key`` (a (yyyy, mm, dd)
tuple) — the only key that reconciles the registry's subject-prefixed / ``_v2``
session ids, the manifest's zero-dropped ids, and the pkl filenames. pkls are
glob-resolved (some sessions are ``_v2`` re-recordings).
"""
from __future__ import annotations
import os
import gc
import glob
import pandas as pd

from visdetect.core.session import load_session as _core_load
from visdetect.analysis.config import session_date_key

SUBJECTS = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}
STAGES = ("Naive", "Learning", "Expert")
DATA_DIR = "data"
CACHE_DIR = os.path.join("data", "cache", "evidence_learning")
FIG_DIR = os.path.join("FIGURES", "evidence_learning")


def _reg_filename(subject):
    return subject.replace("_", "").lower()          # BG_046 -> bg046


def load_manifest(subject, data_dir=DATA_DIR):
    m = pd.read_csv(os.path.join(data_dir, f"{subject}_staging_manifest.csv"))
    m["skey"] = m["session_name"].map(session_date_key)
    return m


def _pkl_index(subject, data_dir=DATA_DIR):
    """{session_date_key: pkl_path} for a subject (handles _v2 / mixed naming)."""
    out = {}
    for p in glob.glob(os.path.join(data_dir, "pkls", subject, f"{subject}_*.pkl")):
        k = _safe_key(os.path.basename(p)[len(subject) + 1:-4])
        if k is not None:
            out[k] = p
    return out


def subject_sessions(subject, stages=STAGES, data_dir=DATA_DIR):
    """Yield (skey, session_name, stage, Session) for QC-pass sessions in
    ``stages`` (chronological by manifest order). skey = session_date_key tuple;
    session_name = the raw manifest id (for state-tag / display)."""
    man = load_manifest(subject, data_dir)
    man = man[man["stage"].isin(set(stages))]
    idx = _pkl_index(subject, data_dir)
    for _, r in man.iterrows():
        path = idx.get(r["skey"])
        if path is None:
            continue
        sess = _core_load(path)
        yield r["skey"], str(r["session_name"]), r["stage"], sess
        del sess
        gc.collect()


def tf_responsive_units(subject, data_dir=DATA_DIR):
    """{session_date_key: {cluster_id: sign}} for TF-responsive units.

    sign = +1 if c1_r_log2 >= 0 (fast-TF-preferring) else -1. NOTE:
    ``region_bank_confirmed`` is False across the whole registry (provisional
    anatomy), so it is NOT gated on here; pooling is by SUBJECT (DMS: 046+039;
    VMS: 031), and per-unit region labels are treated as provisional."""
    reg = pd.read_csv(os.path.join(data_dir, "cache", "tf_responsive",
                                   f"{_reg_filename(subject)}_tf_responsive.csv"))
    reg = reg[reg["resp_log2"] == True].copy()
    reg["skey"] = reg["session"].map(session_date_key)
    out = {}
    for skey, g in reg.groupby("skey"):
        out[skey] = {int(u): (1 if c >= 0 else -1)
                     for u, c in zip(g["unit"], g["c1_r_log2"])}
    return out


def _safe_key(stem):
    """session_date_key or None (skips non-session files like _tag_summary.csv)."""
    try:
        return session_date_key(stem)
    except (ValueError, TypeError):
        return None


def _state_tag_index(subject, data_dir=DATA_DIR):
    out = {}
    for p in glob.glob(os.path.join(data_dir, "cache", "state_tags", subject, "*.csv")):
        k = _safe_key(os.path.splitext(os.path.basename(p))[0])
        if k is not None:
            out[k] = p
    return out


def load_state_labels_by_key(subject, skey, data_dir=DATA_DIR):
    """DataFrame indexed by trial_idx with [state_label, state_confidence], or
    None if no state-tag file matches skey."""
    path = _state_tag_index(subject, data_dir).get(skey)
    if path is None:
        return None
    df = pd.read_csv(path)
    df = df[df["trial_idx"].notna()].copy()
    df["trial_idx"] = df["trial_idx"].astype(int)
    return df.set_index("trial_idx")[["state_label", "state_confidence"]]
