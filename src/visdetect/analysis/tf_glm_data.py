"""Adapters that turn a data source into tf_glm TrialRegressors + spike times.

Two sources:
  - load_khilkevich_session(): the paper's npx_converted parquet/csv sessions
    (full 19-regressor positive control).
  - session_trial_regressors(): a visdetect Session (BG_046/BG_039 reduced set).

The npx_converted schema was verified against a real session
(MoHa_20260212_dmdmTemporalExpectation / 1074894 / ML_1074894_M2_S01) on
2026-06-19. The COL_* constants below reflect the REAL column names, not the
brief's expected defaults (which differed). See the per-file notes inline.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from visdetect.analysis.tf_glm import TFGLMConfig, TrialRegressors, trial_bin_edges

# ── Column-name constants — verified against the real npx_converted schema ──
# neural.parquet / clusters.csv:
COL_UNIT = "cluster_id"            # unit id (both neural.parquet & clusters.csv)
COL_SPIKE_T = "spike_time"         # spike time in seconds (neural.parquet)
COL_REGION = "brain_region"        # fine CCF acronym (NOT "region")
COL_REGION_COMB = "brain_region_comb"  # coarser grouped region label

# trials.parquet columns (consumed by the Task-9 regressor builder, not the
# loader itself). The paper stores TF as Stim1TF/Stim2TF rather than a single
# "change_size"; the change ratio is Stim2TF / Stim1TF. Event times live in
# *_rise columns. Behaviour is in `trialoutcome`.
COL_TRIAL_IDX = "trial_idx"
COL_STIM1_TF = "Stim1TF"
COL_STIM2_TF = "Stim2TF"
COL_CHANGE_SIZE = "Stim2TF"        # ratio = Stim2TF / Stim1TF (no scalar col)
COL_BASELINE_ON = "Baseline_ON_rise"
COL_CHANGE_ON = "Change_ON_rise"
COL_OUTCOME = "trialoutcome"


@dataclass
class KhilSession:
    units: Dict[int, np.ndarray]
    regions: Dict[int, str]
    trials: pd.DataFrame
    licks: np.ndarray
    baseline_on: np.ndarray
    change_on: np.ndarray
    valve: np.ndarray
    airpuff: np.ndarray
    stim: pd.DataFrame
    movement: dict
    running: np.ndarray


def _read(d: Path, name_parquet: str, name_csv: str) -> pd.DataFrame:
    """Read a parquet file, falling back to the CSV mirror.

    The npx_converted sessions ship both a `.parquet` and a `.csv` for the big
    tables (trials, neural). Some tables (stim, running) ship CSV only. If
    pyarrow is missing, the parquet read raises and we fall back to CSV.
    """
    p = d / name_parquet
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    return pd.read_csv(d / name_csv)


def load_khilkevich_session(session_dir) -> KhilSession:
    d = Path(session_dir)
    neural = _read(d, "neural.parquet", "spikes.csv")
    trials = _read(d, "trials.parquet", "trials.csv")
    clusters = pd.read_csv(d / "clusters.csv")

    # Per-unit sorted spike times.
    units: Dict[int, np.ndarray] = {}
    for uid, g in neural.groupby(COL_UNIT):
        units[int(uid)] = np.sort(g[COL_SPIKE_T].to_numpy(float))

    # Per-unit region label (fine CCF acronym). clusters.csv is the canonical
    # one-row-per-unit table; fall back to neural.parquet if a column is absent.
    regions: Dict[int, str] = {}
    region_src = clusters if COL_REGION in clusters.columns else neural
    if COL_REGION in region_src.columns and COL_UNIT in region_src.columns:
        for uid, rg in (region_src[[COL_UNIT, COL_REGION]]
                        .drop_duplicates(subset=[COL_UNIT])
                        .itertuples(index=False)):
            regions[int(uid)] = str(rg)

    def _daq(channel_csv: str) -> np.ndarray:
        """Event onset times from a per-channel daq CSV.

        The daq_*.csv files have columns [rise_t, fall_t, duration]; the onset
        is `rise_t` (column 0). Catch / non-triggered trials carry NaN rows,
        which we drop so the returned array is a clean 1-D vector of times.
        """
        fp = d / channel_csv
        if fp.exists():
            col = pd.read_csv(fp)
            vals = col.iloc[:, 0].to_numpy(float)
            return vals[~np.isnan(vals)]
        return np.zeros(0)

    licks = _daq("daq_Lick_L.csv")
    baseline_on = _daq("daq_Baseline_ON.csv")
    change_on = _daq("daq_Change_ON.csv")
    valve = _daq("daq_Valve_L.csv")
    airpuff = _daq("daq_Air_puff.csv")

    # Per-frame stimulus (trial_idx, frame_idx, TF, tag, vbl, frame_time).
    stim = _read(d, "stim.parquet", "stim.csv")

    # Running wheel: running.csv has columns [time, speed]; keep both columns.
    running = (pd.read_csv(d / "running.csv").to_numpy(float)
               if (d / "running.csv").exists() else np.zeros((0, 2)))

    # Movement bundle: {'licks': arr, 'running': {time,speed},
    #                   'video': {pupil_area, mouth_me, whisker_me}}.
    movement: dict = {}
    mp = d / "movement.pkl"
    if mp.exists():
        import pickle
        with open(mp, "rb") as f:
            movement = pickle.load(f)

    return KhilSession(units=units, regions=regions, trials=trials, licks=licks,
                       baseline_on=baseline_on, change_on=change_on, valve=valve,
                       airpuff=airpuff, stim=stim, movement=movement,
                       running=running)
