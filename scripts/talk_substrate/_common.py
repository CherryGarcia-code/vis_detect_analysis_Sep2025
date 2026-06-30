"""Shared, talk-specific glue for the descriptive lab-talk figures (subject BG_046).

This is *presentation substrate* work (one concept per figure, plain-English
captions) — NOT the N1 urgency-ramp decode. Heavy lifting (tensors, z-scoring,
alignment) comes from ``visdetect.*``; this module only holds talk-specific glue:
output/cache paths, the cell-type label normaliser (two on-disk vocabularies +
the backslash footgun in ``CELLTYPE_COLORS``), colours, and a thin figure saver
that writes to ``FIGURES/talk_substrate/<subject>/`` (NOT analysis_suite/).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis import config as cfg                      # noqa: E402
from visdetect.analysis.config import (                           # noqa: E402
    canonical_session_id, CELLTYPE_COLORS, STATE_LABEL_COLORS, OUTCOME_COLORS,
)

SUBJECT = cfg.SUBJECT  # "BG_046"

FIG_DIR = REPO_ROOT / "FIGURES" / "talk_substrate" / SUBJECT
CACHE_DIR = REPO_ROOT / "data" / "cache" / "talk_substrate"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# GMM cell-type stats (threshold line for Fig A); produced by
# scripts/analysis/build_waveform_celltype_labels.py
WAVEFORM_STATS_PATH = REPO_ROOT / "FIGURES" / "qc" / "waveform_celltype_stats.csv"

# ── Cell-type display vocabulary ──────────────────────────────────────────────
# Canonical *display* labels for the broad/narrow waveform axis. We keep the
# region-agnostic gloss in the label: in striatum narrow≈FSI / broad≈SPN, but the
# split is purely spike-width (t2p), so callers add the region caveat in captions.
NARROW = "Narrow (FSI)"
BROAD = "Broad (MSN/Proj)"
UNKNOWN = "Unknown"
CELLTYPE_ORDER = [NARROW, BROAD]

# Two producers write different strings (see celltype-region discovery finding):
#   build_waveform_celltype_labels.py      -> {"FSI","SPN","Unclassified"}
#   concat_sort/regen_waveform_labels.py   -> {"Narrow (FSI)","Broad (MSN/Proj)"}
_NARROW_RAW = {"fsi", "narrow", "narrow (fsi)"}
_BROAD_RAW = {"spn", "msn", "broad", "broad (msn/proj)", "broad (msn\\proj)"}


def normalize_celltype(raw) -> str:
    """Map any on-disk celltype string to {NARROW, BROAD, UNKNOWN}.

    Tolerates both producer vocabularies and the backslash variant of the
    broad key. NaN / "Unclassified" / unrecognised -> UNKNOWN.
    """
    if raw is None:
        return UNKNOWN
    s = str(raw).strip().lower()
    if s in _NARROW_RAW:
        return NARROW
    if s in _BROAD_RAW:
        return BROAD
    return UNKNOWN


def celltype_color(display_label: str) -> str:
    """Colour for a *display* celltype label, robust to the backslash key in
    ``CELLTYPE_COLORS`` ("Broad (MSN\\Proj)")."""
    for k, v in CELLTYPE_COLORS.items():
        kl = k.lower()
        if display_label == NARROW and "narrow" in kl:
            return v
        if display_label == BROAD and "broad" in kl:
            return v
    return {NARROW: "#e74c3c", BROAD: "#3498db", UNKNOWN: "#9e9e9e"}.get(
        display_label, "#9e9e9e")


def canon(session) -> str:
    """Canonical 8-digit session id (single source of truth join key)."""
    return canonical_session_id(session)


def save_talk_figure(fig, name: str, dpi: int = 300) -> Path:
    """Save a figure to FIGURES/talk_substrate/<subject>/<name>.png and close it."""
    import matplotlib.pyplot as plt
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def stats_csv_path(name: str) -> Path:
    """Companion stats CSV path next to the figure."""
    return FIG_DIR / f"{name}_stats.csv"


def load_celltype_lookup(subject: str = SUBJECT) -> dict:
    """(session_8, cluster_id) -> display cell-type, from the on-disk waveform labels.

    Keyed via canonical_session_id — correct for BG_046 (8-digit ids). For subjects with
    6-digit DDMMYY ids use build_celltype_labels_from_rawwf (raw-token keyed) instead."""
    import pandas as pd
    path = (cfg.WAVEFORM_LABELS_PATH if subject == SUBJECT
            else os.path.join(cfg.ROOT, "data", subject, "waveform_celltype_labels.csv"))
    lab = pd.read_csv(path)
    lab["session_8"] = lab["session_date"].map(canon)
    lab["cluster_id"] = lab["cluster_id"].astype(int)
    lab["ct"] = lab["celltype"].map(normalize_celltype)
    return {(r.session_8, r.cluster_id): r.ct for r in lab.itertuples()}


def build_celltype_labels_from_rawwf(subject: str = SUBJECT, write_csv: bool = True):
    """Compute FSI/SPN cell types from RawWaveforms for `subject`, keyed by the RAW pkl token.

    Robust to (a) subject-prefixed RawWaveforms dirs (BG_039_<date>) vs bare-token dirs
    (BG_046's <date>), and (b) mixed 6-digit DDMMYY / 8-digit DDMMYYYY tokens — it bypasses
    canonical_session_id (which mangles 6-digit ids) and keys on the raw pkl token so the
    lookup matches load_session(token). Per-subject 2-component GMM on trough-to-peak.

    Returns (lut, info): lut = {(raw_token, cluster_id): display_celltype}; info = GMM stats.
    Also writes data/<subject>/waveform_celltype_labels.csv (session_date = raw token).
    """
    import glob
    import pandas as pd
    from visdetect.analysis.tracking_qc import load_raw_mean_waveform, extract_peak_channel
    from visdetect.analysis.waveform_celltype import compute_waveform_features, classify_celltype

    # Derive from the SUBJECT ARG (NOT cfg.RAW_WF_DIR, which is fixed to the import-time env
    # subject — that caused other subjects to read BG_046's RawWaveforms when looped in-process).
    rawroot = os.path.join(cfg.ROOT, "data", "unit_match", "input", subject)
    rows = []
    for rw in sorted(glob.glob(os.path.join(rawroot, "*", "RawWaveforms"))):
        sess_dir = os.path.basename(os.path.dirname(rw))         # 'BG_039_02062025' or '01072025'
        token = sess_dir[len(subject) + 1:] if sess_dir.startswith(subject + "_") else sess_dir
        for f in glob.glob(os.path.join(rw, "Unit*_RawSpikes.npy")):
            try:
                kid = int(os.path.basename(f)[len("Unit"):].split("_")[0])
            except ValueError:
                continue
            mean_wf = load_raw_mean_waveform(rawroot, sess_dir, kid)
            if mean_wf is None or mean_wf.ndim != 2:
                continue
            pc = extract_peak_channel(mean_wf)
            rows.append({"session_date": token, "cluster_id": int(kid),
                         "t2p_ms": compute_waveform_features(mean_wf[:, pc])["t2p_ms"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}, {}
    labels, info = classify_celltype(df["t2p_ms"].values)   # {FSI, SPN, Unclassified}
    df["celltype"] = labels
    if write_csv:
        out_csv = os.path.join(cfg.ROOT, "data", subject, "waveform_celltype_labels.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df[["session_date", "cluster_id", "celltype"]].to_csv(out_csv, index=False)
        # subject-scoped per-unit t2p cache (raw token) for WS2 label-validity
        df[["session_date", "cluster_id", "t2p_ms"]].rename(
            columns={"session_date": "session_8"}).to_csv(
            CACHE_DIR / f"waveform_t2p_{subject}.csv", index=False)
    lut = {(str(r.session_date), int(r.cluster_id)): normalize_celltype(r.celltype)
           for r in df.itertuples()}
    return lut, info


ALL_SUBJECTS = ["BG_046", "BG_039", "BG_031", "BG_038"]
_T2P_FILES = {"BG_046": "bg046_waveform_t2p.csv", "BG_039": "waveform_t2p_BG_039.csv",
              "BG_031": "waveform_t2p_BG_031.csv", "BG_038": "waveform_t2p_BG_038.csv"}


def load_t2p(subject: str):
    """Per-unit trough-to-peak cache (session_8, cluster_id, t2p_ms). session_8 read as str
    to preserve 6-digit DDMMYY tokens."""
    import pandas as pd
    return pd.read_csv(CACHE_DIR / _T2P_FILES[subject], dtype={"session_8": str})


# Per-subject ISI-feature CSV (BG_046 keeps the legacy bg046_ name; others are *_BG_xxx).
# Single source so callers don't re-derive the path and trip the BG_046 naming footgun.
_ISI_FILES = {"BG_046": "bg046_isi_features.csv", "BG_039": "isi_features_BG_039.csv",
              "BG_031": "isi_features_BG_031.csv", "BG_038": "isi_features_BG_038.csv"}


def isi_features_path(subject: str = SUBJECT):
    """Path to the per-subject ISI-feature CSV (handles BG_046's legacy bg046_ name)."""
    return CACHE_DIR / _ISI_FILES[subject]


# Recording-site label per subject — single source for figure titles/captions (FIX E:
# BG_038 is a cortical M1/S1 REFERENCE probe, not the MOs source region).
REGION_LABEL = {
    "BG_046": "striatum DMS (CP)", "BG_039": "striatum DMS (CP)",
    "BG_031": "striatum VMS (CP)", "BG_038": "cortex M1/S1 (reference)",
}


def region_label(subject: str = SUBJECT) -> str:
    """Recording-site label for a subject (for figure titles); '' if unknown."""
    return REGION_LABEL.get(subject, "")


def common_t2p_cutoff(subjects=tuple(ALL_SUBJECTS)):
    """ONE narrow/broad width cutoff for ALL figures: a 2-component GMM on the POOLED t2p
    across `subjects` (canonical classify_celltype; threshold = mean of component means).
    Recomputed live (not hardcoded). Returns (cutoff_ms, info)."""
    import numpy as np
    from visdetect.analysis.waveform_celltype import classify_celltype
    t = np.concatenate([load_t2p(s)["t2p_ms"].values for s in subjects])
    _, info = classify_celltype(t)
    return float(info["threshold_ms"]), info


def common_celltype(cache, subjects, thr):
    """Narrow/broad masks for a cache's units under the COMMON width cutoff `thr`, by joining
    per-unit t2p on (session, cluster_id). Returns (narrow_mask, broad_mask, t2p_array)."""
    import numpy as np
    lut = {}
    for s in subjects:
        for r in load_t2p(s).itertuples():
            lut[(str(r.session_8), int(r.cluster_id))] = float(r.t2p_ms)
    sess = cache["unit_meta_session"].astype(str)
    cid = cache["unit_meta_cluster_id"].astype(int)
    t2p = np.array([lut.get((sess[i], int(cid[i])), np.nan) for i in range(len(sess))])
    return (np.isfinite(t2p) & (t2p < thr)), (np.isfinite(t2p) & (t2p >= thr)), t2p


def celltype_and_sessions(subject: str = SUBJECT):
    """(lut, sessions) keyed consistently with load_session, for ANY subject.

    BG_046: canonical 8-digit ids (load_celltype_lookup). Other subjects: raw pkl tokens
    (build_celltype_labels_from_rawwf) so 6-digit DDMMYY ids resolve and aren't mangled."""
    from visdetect.suite.loader import list_pkl_sessions
    if subject == "BG_046":
        lut = load_celltype_lookup(subject)
        return lut, sorted({s for (s, _c) in lut.keys()})
    lut, info = build_celltype_labels_from_rawwf(subject)
    if info:
        print(f"[celltype] {subject}: GMM t2p threshold="
              f"{info.get('threshold_ms', float('nan')):.3f} ms, {len(lut)} labelled units")
    have = set(list_pkl_sessions(subject))
    sessions = [s for s in sorted({t for (t, _c) in lut.keys()}) if s in have]
    return lut, sessions
