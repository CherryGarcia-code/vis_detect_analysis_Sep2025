"""I/O + config for the Fig-5 e-h preparatory-activity port. THIS-repo paths only
(no vd_tf_bg046 sibling hardcode). Pure math lives in visdetect.analysis.preparatory."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
import pandas as pd

from visdetect.analysis.constants import DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS

REPO = Path(__file__).resolve().parents[3]
MICE = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}

CLASS_COLORS = {"transient": "#3182bd", "sustained": "#e6550d", "non-TF": "#969696"}
WIDTH_CMAP = "viridis"

BIN = DEFAULT_BIN_SIZE                        # 0.025 s
SIG_BINS = DEFAULT_SIGMA_MS / 1000.0 / BIN    # 25 ms sigma in bins
LICK_WIN = (-2.0, 1.5)                         # around lick onset
BASE_WIN = (-2.0, 0.0)                         # 2 s pre-CHANGE (paper z-baseline)
BASE_FRAC_WIN = (-2.0, -1.8)                   # pre-lick baseline-fraction window (paper)
MIN_LICKS = 10
MIN_RT = 0.4
NARROW, BROAD = 0.05, 0.15                     # grid-fwhm class cut (project convention)
DISENG_MAX = 50.0


def load_registry(subj: str) -> pd.DataFrame:
    r = pd.read_csv(
        REPO / f"data/cache/tf_responsive/{subj.lower().replace('_','')}_tf_responsive.csv",
        dtype={"session": str, "session_date": str})
    r["resp"] = r.resp_log2.astype(str).str.lower().isin(["true", "1", "1.0"])
    return r


def good_dates(subj: str, max_diseng: float = DISENG_MAX) -> set:
    man = pd.read_csv(REPO / f"data/{subj}_staging_manifest.csv", dtype={"session_name": str})
    qc = man.loc[~man.qc_fail.astype(bool), "session_name"]
    keep = set()
    for d in qc:
        sf = REPO / f"data/cache/state_tags/{subj}/{d}.csv"
        if sf.exists():
            if 100 * (pd.read_csv(sf).state_label == "Disengaged").mean() < max_diseng:
                keep.add(d)
        else:
            keep.add(d)
    return keep


def spikes_for(session, uid: int) -> np.ndarray:
    for c in session.clusters:
        if int(c.cluster_id) == int(uid):
            return np.sort(np.asarray(c.spike_times, float).ravel())
    return np.zeros(0)


def load_width() -> pd.DataFrame:
    return pd.read_csv(REPO / "data/cache/tf_glm_bg046/kernel_width_continuous.csv",
                       dtype={"session": str})[["subject", "session", "unit", "interp_fwhm"]]


def class_from_fwhm(fwhm: float) -> str:
    return "transient" if fwhm <= NARROW else ("sustained" if fwhm >= BROAD else "intermediate")
