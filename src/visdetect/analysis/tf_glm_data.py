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
    regions: Dict[int, str]            # fine CCF acronym (VISp1, VISp5, CP, ...)
    trials: pd.DataFrame
    licks: np.ndarray
    baseline_on: np.ndarray
    change_on: np.ndarray
    valve: np.ndarray
    airpuff: np.ndarray
    stim: pd.DataFrame
    movement: dict
    running: np.ndarray
    regions_coarse: Optional[Dict[int, str]] = None  # grouped label (VISp, CP)
    session_dir: Optional[str] = None  # source dir (for daq_Eye_cam.csv movement ts)


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

    # Coarse (grouped) region label, e.g. VISp1/VISp5 -> VISp, used for region
    # selection in the regressor builder (matches the survey's coarse labels).
    regions_coarse: Dict[int, str] = {}
    coarse_src = clusters if COL_REGION_COMB in clusters.columns else neural
    if COL_REGION_COMB in coarse_src.columns and COL_UNIT in coarse_src.columns:
        for uid, rg in (coarse_src[[COL_UNIT, COL_REGION_COMB]]
                        .drop_duplicates(subset=[COL_UNIT])
                        .itertuples(index=False)):
            regions_coarse[int(uid)] = str(rg)

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
                       running=running, regions_coarse=regions_coarse,
                       session_dir=str(d))


# ── Task 9: per-trial regressor builder for the Khilkevich npx_converted data ──

# Trial-event / per-trial column names (verified against the real schema).
COL_TF_COL = "TF"                 # per-frame linear TF in stim.csv
COL_STIM_FRAME_TIME = "frame_time"  # NEURAL-clock per-frame time (NOT vbl, which
                                  # is the raw VBL hardware clock ~1.6e9)
COL_VALVE = "Valve_L_rise"        # per-trial reward (valve open) time
COL_AIRPUFF = "Air_puff_rise"     # per-trial air-puff onset (NaN when none)
COL_LICK_TIMES = "lick_times"     # per-trial array of lick times (object column)

# Snap-to grid for the change ratio (matches tf_glm.CHANGE_SIZES).
_CHANGE_SIZES = (1.0, 1.25, 1.35, 1.5, 2.0, 4.0)


def _resample_to_bins(times, values, bin_edges, bin_s, fill=0.0):
    """Mean of a (times, values) signal within each 50-ms bin; empties -> fill.

    ``bin_edges`` are the LEFT edges of this trial's bins. A sample at time t
    falls in bin floor((t - edges[0]) / bin_s). Samples outside the edge span
    are dropped. Used for the per-frame TF signal and the wheel speed.
    """
    edges = np.asarray(bin_edges, float)
    if edges.size == 0:
        return np.zeros(0)
    t = np.asarray(times, float)
    v = np.asarray(values, float)
    out = np.full(edges.size, float(fill))
    if t.size == 0 or not np.isfinite(edges[0]):
        return out
    # Compute float bin indices first; only cast the finite, in-window samples
    # (casting NaN/inf to int is undefined and raises a RuntimeWarning).
    fidx = (t - edges[0]) / bin_s + 1e-9
    finite = np.isfinite(fidx) & np.isfinite(v)
    idx = np.full(t.size, -1, dtype=int)
    idx[finite] = np.floor(fidx[finite]).astype(int)
    acc = np.zeros(edges.size)
    cnt = np.zeros(edges.size)
    ok = finite & (idx >= 0) & (idx < edges.size)
    np.add.at(acc, idx[ok], v[ok])
    np.add.at(cnt, idx[ok], 1.0)
    nz = cnt > 0
    out[nz] = acc[nz] / cnt[nz]
    return out


def _snap_change_size(ratio: float) -> float:
    """Snap a raw Stim2TF/Stim1TF ratio to the nearest canonical change size."""
    if not np.isfinite(ratio):
        return 1.0
    arr = np.asarray(_CHANGE_SIZES, float)
    return float(arr[int(np.argmin(np.abs(arr - ratio)))])


def _eyecam_movement_signals(ks: "KhilSession", session_dir):
    """(eye_t, motion_energy, pupil) on the NEURAL clock for the full model.

    movement.pkl holds video arrays (mouth_me / whisker_me / pupil_area, length
    ~305025). Their TIME VECTOR is the eye-camera frame rise times in
    daq_Eye_cam.csv (~305028 rows), which are already on the neural/spike clock
    (same clock as Baseline_ON_rise etc.). We TRUNCATE both to their common
    length so index i of each movement array maps to eye-cam rise-time i.

    motion_energy = mean(mouth_me, whisker_me) (face/whisker movement that
    accompanies licking); pupil = pupil_area. Returns three equal-length 1-D
    arrays, or empties if the inputs are missing.
    """
    video = (ks.movement or {}).get("video", {})
    mouth = np.asarray(video.get("mouth_me", np.zeros(0)), float).ravel()
    whisk = np.asarray(video.get("whisker_me", np.zeros(0)), float).ravel()
    pupil = np.asarray(video.get("pupil_area", np.zeros(0)), float).ravel()
    if mouth.size == 0 and whisk.size == 0 and pupil.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    # Eye-camera frame rise times (neural clock) from daq_Eye_cam.csv (col 0).
    eye_t = np.zeros(0)
    if session_dir is not None:
        fp = Path(session_dir) / "daq_Eye_cam.csv"
        if fp.exists():
            eye_t = pd.read_csv(fp).iloc[:, 0].to_numpy(float)
    if eye_t.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    # motion-energy = mean of available face channels (mouth + whisker).
    me_parts = [a for a in (mouth, whisk) if a.size]
    if me_parts:
        m = min(a.size for a in me_parts)
        motion = np.mean(np.vstack([a[:m] for a in me_parts]), axis=0)
    else:
        motion = np.zeros(0)

    # Truncate eye-cam times and each movement signal to their common length so
    # index i of the movement array maps to eye-cam rise-time i.
    n = min(x.size for x in (eye_t, motion if motion.size else eye_t,
                             pupil if pupil.size else eye_t))
    eye_t = eye_t[:n]
    motion = motion[:n] if motion.size else np.zeros(n)
    pupil = pupil[:n] if pupil.size else np.zeros(n)
    return eye_t, motion, pupil


def khilkevich_trial_regressors(
    ks: KhilSession, cfg: TFGLMConfig, region: Optional[str] = None
) -> Tuple[List[TrialRegressors], Dict[int, np.ndarray]]:
    """Build per-trial TrialRegressors + the requested region's spike trains.

    Adapts the Khilkevich npx_converted schema (verified on real sessions) to
    the tf_glm TrialRegressors container. Alignment is driven by the per-trial
    ``ks.trials`` DataFrame (NOT the NaN-dropped flat daq arrays), so trial
    indexing is preserved on catch/abort trials.

    Per trial i:
      - t_start = Baseline_ON_rise[i]; t_end = next sorted baseline onset
        (last trial: t_start + 20 s).
      - change_time = Change_ON_rise[i] (NaN on catch/abort -> no change col).
      - change_size = round(Stim2TF/Stim1TF, 2) snapped to CHANGE_SIZES.
      - tf_bins: per-frame linear TF from ks.stim (frame_time = neural clock)
        resampled onto THIS trial's bins. Linear TF is passed through;
        pulse_times_from_tf log2's it internally.
      - lick_times: per-trial lick_times array if present, else flat ks.licks
        filtered to [t_start, t_end).
      - reward_time: first Valve_L_rise in [t_start, t_end) (per-trial col,
        falls back to the flat ks.valve array).
      - abort_time: change_time if trialoutcome == 'abort' else NaN.
      - wheel_bins: |speed| from ks.running resampled onto the trial's bins.
      - phase_bins = None (PHASE IS ABSENT from the export; run include_phase=False).

    When ``cfg.include_movement`` (the full Khilkevich-faithful model), three
    further regressors are populated (else left None/NaN so the reduced model is
    unaffected):
      - motion_bins: mean(mouth_me, whisker_me) face/whisker motion-energy from
        movement.pkl, on the daq_Eye_cam.csv (neural-clock) frame times,
        resampled onto THIS trial's bins.
      - pupil_bins: pupil_area from movement.pkl on the same eye-cam clock.
      - airpuff_time: per-trial Air_puff_rise (NaN when no puff).

    The returned spike dict holds only units whose region matches ``region``
    (matched on fine OR coarse label; None -> all units).
    """
    bs = cfg.bin_s
    tr = ks.trials

    bon = tr[COL_BASELINE_ON].to_numpy(float)
    n = bon.size
    # Trial ends = next sorted baseline onset; last trial gets a fixed window.
    order = np.argsort(np.where(np.isfinite(bon), bon, np.inf))
    ends = np.full(n, np.nan)
    for k in range(n):
        i = order[k]
        if k + 1 < n:
            ends[i] = bon[order[k + 1]]
        else:
            ends[i] = bon[i] + 20.0 if np.isfinite(bon[i]) else np.nan

    chg = (tr[COL_CHANGE_ON].to_numpy(float) if COL_CHANGE_ON in tr.columns
           else np.full(n, np.nan))
    s1 = (tr[COL_STIM1_TF].to_numpy(float) if COL_STIM1_TF in tr.columns
          else np.ones(n))
    s2 = (tr[COL_STIM2_TF].to_numpy(float) if COL_STIM2_TF in tr.columns
          else np.ones(n))
    valve = (tr[COL_VALVE].to_numpy(float) if COL_VALVE in tr.columns
             else np.full(n, np.nan))
    airpuff_col = (tr[COL_AIRPUFF].to_numpy(float) if COL_AIRPUFF in tr.columns
                   else np.full(n, np.nan))
    outcome = (tr[COL_OUTCOME].astype(str).to_numpy() if COL_OUTCOME in tr.columns
               else np.array([""] * n))
    has_lick_col = COL_LICK_TIMES in tr.columns
    lick_col = tr[COL_LICK_TIMES].to_numpy(object) if has_lick_col else None

    # Per-frame stimulus signal (neural-clock frame_time + linear TF).
    stim = ks.stim
    s_t = (stim[COL_STIM_FRAME_TIME].to_numpy(float)
           if COL_STIM_FRAME_TIME in stim.columns else stim.iloc[:, 0].to_numpy(float))
    s_tf = (stim[COL_TF_COL].to_numpy(float)
            if COL_TF_COL in stim.columns else stim.iloc[:, 1].to_numpy(float))

    # Running wheel: col 0 = time (neural clock), col 1 = signed speed.
    if ks.running.size:
        run_t = ks.running[:, 0]
        run_v = np.abs(ks.running[:, 1])
    else:
        run_t = np.zeros(0)
        run_v = np.zeros(0)

    # Movement-controlled regressors (full Khilkevich model only): face/whisker
    # motion-energy + pupil on the eye-cam (neural) clock. Empty unless asked.
    if cfg.include_movement:
        eye_t, motion_sig, pupil_sig = _eyecam_movement_signals(ks, ks.session_dir)
    else:
        eye_t = motion_sig = pupil_sig = np.zeros(0)

    trials_regs: List[TrialRegressors] = []
    for i in range(n):
        t0, t1 = float(bon[i]), float(ends[i])
        if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
            # Degenerate trial: emit an empty regressor so trial indexing holds.
            t1 = t0 + bs if np.isfinite(t0) else 0.0
        edges = trial_bin_edges(t0, t1, bs)
        tf_bins = _resample_to_bins(s_t, s_tf, edges, bs, fill=0.0)
        wheel_bins = _resample_to_bins(run_t, run_v, edges, bs, fill=0.0)

        if has_lick_col and lick_col[i] is not None and np.ndim(lick_col[i]) > 0:
            lk = np.asarray(lick_col[i], float)
            lk = lk[(lk >= t0) & (lk < t1)]
        else:
            lk = ks.licks[(ks.licks >= t0) & (ks.licks < t1)]

        ratio = round(s2[i] / s1[i], 2) if (np.isfinite(s1[i]) and s1[i] != 0
                                            and np.isfinite(s2[i])) else 1.0
        change_size = _snap_change_size(ratio)
        ct = chg[i] if (i < chg.size and np.isfinite(chg[i])) else np.nan

        rew = valve[i] if (i < valve.size and np.isfinite(valve[i])) else np.nan
        if not np.isfinite(rew) and ks.valve.size:
            inwin = ks.valve[(ks.valve >= t0) & (ks.valve < t1)]
            rew = float(inwin[0]) if inwin.size else np.nan

        abort_time = ct if str(outcome[i]).lower() == "abort" else np.nan

        # Movement regressors (full model only): resample motion-energy + pupil
        # onto THIS trial's per-trial bins (never the global concatenated edges);
        # air-puff is a per-trial event time.
        if cfg.include_movement and eye_t.size:
            motion_bins = _resample_to_bins(eye_t, motion_sig, edges, bs, fill=0.0)
            pupil_bins = _resample_to_bins(eye_t, pupil_sig, edges, bs, fill=0.0)
        else:
            motion_bins = None
            pupil_bins = None
        airpuff_time = (float(airpuff_col[i]) if (cfg.include_movement
                        and i < airpuff_col.size and np.isfinite(airpuff_col[i]))
                        else np.nan)

        trials_regs.append(TrialRegressors(
            t_start=t0, t_end=t1, change_time=ct, change_size=change_size,
            tf_bins=tf_bins, lick_times=lk, reward_time=rew,
            abort_time=abort_time, wheel_bins=wheel_bins, phase_bins=None,
            motion_bins=motion_bins, pupil_bins=pupil_bins,
            airpuff_time=airpuff_time))

    # Region selection: match fine OR coarse label; None -> all units.
    coarse = ks.regions_coarse or {}
    if region is None:
        unit_ids = list(ks.units.keys())
    else:
        unit_ids = [u for u in ks.units
                    if ks.regions.get(u, "") == region
                    or coarse.get(u, "") == region]
    return trials_regs, {u: ks.units[u] for u in unit_ids}


# ── Task 10: per-trial regressor builder for a visdetect Session (BG_046/039) ──

# Outcomes whose Change_ON is a real change event (the stimulus actually changed
# or was withheld on a catch trial). On FA/abort the change stimulus was never
# presented, so its "change time" is scientifically meaningless -> NaN.
# (Mirrors CLAUDE.md's EVENT_VALID_OUTCOMES for Change_ON, plus catch trials
# where the mouse withheld: hit/miss/ref are the change-reached outcomes.)
_CHANGE_REACHED_OUTCOMES = {"hit", "miss", "ref"}

# baseline_values (St1TrialVector) is logged ~3x per 50-ms TF update; stride 3
# recovers the genuine 50-ms TF grid from the baseline start.
_BASELINE_STRIDE = 3


def session_trial_regressors(
    session, cfg: TFGLMConfig
) -> Tuple[List[TrialRegressors], Dict[int, np.ndarray]]:
    """Build per-trial TrialRegressors + good-unit spike trains from a Session.

    Reduced regressor set on the NEURAL clock (no phase, no motion-energy/pupil),
    matching ``khilkevich_trial_regressors`` so the BG result is apples-to-apples
    with the Khilkevich positive control.

    Per trial i (1:1 with ``session.trials``):
      - ``t_start`` = ``ni_events['Baseline_ON'][i]``; ``t_end`` = next
        Baseline_ON onset (last trial: ``t_start + 20 s``).
      - ``change_time`` = ``ni_events['Change_ON'][i]`` BUT only when the trial's
        outcome reached the change (hit/miss/ref); NaN on FA/abort (the change
        stimulus was never presented) -> no change regressor for that trial.
      - ``change_size`` = ``float(trial.change_size)`` (already the canonical
        ratio in {1.0, 1.25, 1.35, 1.5, 2.0, 4.0}).
      - ``tf_bins``: ``trial.baseline_values`` (St1TrialVector) decimated by
        ``stride 3`` -> 50-ms linear-TF grid from baseline start, placed onto the
        trial's per-trial 50-ms bin edges and ZEROED at/after change onset.
      - ``lick_times``: ``ni_events['Piezo_1']`` in [t_start, t_end).
      - ``reward_time``: first finite ``ni_events['Valve_L'][i]`` (per-trial
        array; NaN if none).
      - ``abort_time``: ``change_time`` if outcome == 'abort' else NaN.
        (NB: change_time is NaN'd for abort above, so abort uses the raw
        ``Change_ON[i]`` time directly.)
      - ``wheel_bins``: ``ni_events['Rot_enc_A']`` tick DENSITY (ticks/bin)
        resampled onto the trial's per-trial edges.
      - ``phase_bins`` = None (run ``include_phase=False``).

    Units: ``session.good_and_stable_ids or session.good_cluster_ids`` (the
    project's QC pool), mapped to spike trains from ``session.clusters``. The
    <500-spike skip happens in the RUN LOOP, not here.
    """
    bs = cfg.bin_s
    ni = session.ni_events or {}

    bon = np.asarray(ni.get("Baseline_ON", np.zeros(0)), float).ravel()
    con = np.asarray(ni.get("Change_ON", np.zeros(0)), float).ravel()
    # Lick channel: BG_046 logs licks on the piezo channel (Piezo_1); other
    # subjects (e.g. BG_039 cortex) log them on Lick_L / Lick_R instead. Pool
    # whichever channels are present so the lick regressor is never empty (the
    # lick control is essential -- this is the "lick-controlled GLM").
    licks = _collect_lick_times(ni)
    valve = np.asarray(ni.get("Valve_L", np.zeros(0)), float).ravel()
    rot = np.asarray(ni.get("Rot_enc_A", np.zeros(0)), float).ravel()
    rot = np.sort(rot[np.isfinite(rot)])

    n = len(session.trials)
    # Trial ends = next sorted baseline onset; last trial gets a fixed window.
    order = np.argsort(np.where(np.isfinite(bon), bon, np.inf)) if bon.size else np.zeros(0, int)
    ends = np.full(n, np.nan)
    for k in range(order.size):
        i = order[k]
        if i >= n:
            continue
        if k + 1 < order.size and order[k + 1] < n:
            ends[i] = bon[order[k + 1]]
        else:
            ends[i] = bon[i] + 20.0 if np.isfinite(bon[i]) else np.nan

    trials_regs: List[TrialRegressors] = []
    for i, trial in enumerate(session.trials):
        t0 = float(bon[i]) if i < bon.size else np.nan
        t1 = float(ends[i]) if i < ends.size else np.nan
        if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
            t1 = t0 + bs if np.isfinite(t0) else 0.0
        edges = trial_bin_edges(t0, t1, bs)

        outcome = str(trial.trialoutcome or "").lower()
        raw_change = float(con[i]) if (i < con.size and np.isfinite(con[i])) else np.nan
        change_time = raw_change if outcome in _CHANGE_REACHED_OUTCOMES else np.nan

        change_size = (float(trial.change_size)
                       if trial.change_size is not None else 1.0)

        # TF signal: decimate baseline_values to the 50-ms grid, build a
        # per-frame (time, value) signal anchored at baseline start, resample
        # onto this trial's edges, and zero it at/after change onset.
        tf_bins = np.zeros(edges.size, float)
        bv = (np.asarray(trial.baseline_values, float).ravel()
              if trial.baseline_values is not None else np.zeros(0))
        if bv.size and edges.size:
            tf_vals = bv[::_BASELINE_STRIDE]
            tf_times = t0 + np.arange(tf_vals.size) * bs
            tf_bins = _resample_to_bins(tf_times, tf_vals, edges, bs, fill=0.0)
            # Zero TF after change onset (baseline ends at the change).
            if np.isfinite(raw_change):
                tf_bins[edges >= raw_change] = 0.0

        # Wheel: rotary-encoder A tick DENSITY (ticks per bin), resampled by
        # counting ticks per bin (use value=1 per tick so the bin mean -> mean
        # density; equivalently a histogram, but reuse the shared resampler with
        # a count accumulator).
        wheel_bins = _tick_density_to_bins(rot, edges, bs)

        lk = licks[(licks >= t0) & (licks < t1)] if licks.size else np.zeros(0)

        rew = float(valve[i]) if (i < valve.size and np.isfinite(valve[i])) else np.nan
        abort_time = raw_change if outcome == "abort" else np.nan

        trials_regs.append(TrialRegressors(
            t_start=t0, t_end=t1, change_time=change_time,
            change_size=change_size, tf_bins=tf_bins, lick_times=lk,
            reward_time=rew, abort_time=abort_time, wheel_bins=wheel_bins,
            phase_bins=None))

    # Unit selection: prefer good_and_stable_ids, fall back to good_cluster_ids,
    # else every cluster. Map to spike trains from session.clusters.
    spike_map = {int(c.cluster_id): np.asarray(c.spike_times, float).ravel()
                 for c in session.clusters}
    sel = session.good_and_stable_ids or session.good_cluster_ids
    if sel:
        unit_ids = [int(u) for u in sel if int(u) in spike_map]
    else:
        unit_ids = list(spike_map.keys())
    units = {u: spike_map[u] for u in unit_ids}
    return trials_regs, units


# Lick channels, in priority order. BG_046 uses the piezo channel; BG_039 and
# other subjects use the optical Lick_L/Lick_R channels. Pool all present (a
# lick is a lick regardless of which spout/channel detected it).
_LICK_CHANNELS = ("Piezo_1", "Lick_L", "Lick_R", "Piezo_2")


def _collect_lick_times(ni: dict) -> np.ndarray:
    """Sorted, finite lick times pooled across whichever lick channels exist."""
    parts = []
    for ch in _LICK_CHANNELS:
        v = ni.get(ch)
        if v is None:
            continue
        a = np.asarray(v, float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            parts.append(a)
    if not parts:
        return np.zeros(0)
    return np.sort(np.concatenate(parts))


def _tick_density_to_bins(tick_times: np.ndarray, bin_edges: np.ndarray,
                          bin_s: float) -> np.ndarray:
    """Tick count per 50-ms bin for a monotonic array of event times.

    The rotary-encoder A channel stores tick TIMESTAMPS (not a speed signal);
    wheel speed is proportional to tick density. Bin i = [edges[i], edges[i]+bs).
    """
    edges = np.asarray(bin_edges, float)
    if edges.size == 0 or tick_times.size == 0 or not np.isfinite(edges[0]):
        return np.zeros(edges.size, float)
    full = np.append(edges, edges[-1] + bin_s)
    counts, _ = np.histogram(tick_times, bins=full)
    return counts.astype(float)
