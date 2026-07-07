"""B9 — state-matched baseline TF-encoding across learning.

Thin glue around the registry TF-GLM (``visdetect.analysis.tf_glm`` +
``tf_glm_data.session_trial_regressors``, reused UNCHANGED). B9's only change vs
the registry run is WHICH trials are fed in (filtered by state label + stage);
the readout is the registry's own ``c1_r`` (= ``c1_r_log2``).

See docs/superpowers/specs/2026-07-01-B9-state-matched-baseline-tf-encoding-learning-design.md
and the matching plan.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Iterable
import numpy as np
import pandas as pd

from visdetect.analysis.config import canonical_session_id
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, make_trial_folds,
    fit_poisson_cv, identify_tf_responsive_pulse, pulse_times_from_tf,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = _REPO / "data" / "cache" / "tf_responsive" / "bg039_tf_responsive.csv"
DEFAULT_STATES_DIR = _REPO / "data" / "cache" / "state_tags"
DEFAULT_MANIFEST = _REPO / "data" / "BG_039_staging_manifest.csv"
PKL_ROOT = _REPO / "data" / "pkls"
STATE_CONF_THRESH = 0.8


def registry_path(subject: str) -> Path:
    """data/cache/tf_responsive/bg{nnn}_tf_responsive.csv (BG_031 -> bg031)."""
    return _REPO / "data" / "cache" / "tf_responsive" / f"{subject.lower().replace('_', '')}_tf_responsive.csv"


def manifest_path(subject: str) -> Path:
    return _REPO / "data" / f"{subject}_staging_manifest.csv"


def robust_date(k):
    """Parse a (canonical) session key to a date, handling 8-digit DDMMYYYY and the
    6-digit DDMMYY case that canonical_session_id zero-pads to 00DDMMYY."""
    import datetime
    k = str(k).split("_")[0]
    if len(k) == 8 and k.isdigit():
        dd, mm, yyyy = int(k[:2]), int(k[2:4]), int(k[4:])
        if dd == 0:                              # 00120325 <- 6-digit DDMMYY
            s6 = k[2:]; dd, mm, yy = int(s6[:2]), int(s6[2:4]), int(s6[4:]); yyyy = 2000 + yy
        try:
            return datetime.date(yyyy, mm, dd)
        except Exception:
            return None
    return None


def date_stage_map(subject: str) -> dict:
    """Assign EVERY registry session a learning stage by DATE, using the manifest's
    Naive/Learning/Expert date ranges — so QC-'Excluded' sessions (dropped by the
    behavioural filter) are still staged for state-conditioned / recruitment work."""
    reg = load_registry(registry_path(subject))
    man = pd.read_csv(manifest_path(subject)); man["sess_key"] = man["session_name"].map(canonical_session_id)
    man["date"] = [robust_date(k) for k in man["sess_key"]]
    naive = [d for d, s in zip(man.date, man.stage.astype(str)) if s == "Naive" and d]
    expert = [d for d, s in zip(man.date, man.stage.astype(str)) if s == "Expert" and d]
    naive_end = max(naive) if naive else None
    expert_start = min(expert) if expert else None
    out = {}
    for k in reg.sess_key.unique():
        d = robust_date(k)
        if d is None:
            out[k] = "?"
        elif naive_end and d <= naive_end:
            out[k] = "Naive"
        elif expert_start and d >= expert_start:
            out[k] = "Expert"
        else:
            out[k] = "Learning"
    return out

_ENC_COLS = ["unit", "c1_r", "c2_p", "is_responsive_rerun", "r_red_mean",
             "n_folds_used", "kernel_peak_t", "kernel_fwhm", "n_spikes", "n_trials"]


def load_registry(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or DEFAULT_REGISTRY)
    df["sess_key"] = df["session_date"].map(canonical_session_id)
    return df


def load_state_tags(subject: str, session_date: str, states_dir=None) -> pd.DataFrame:
    """Load a session's state tags. Robust to the id footgun: the on-disk file is
    named with the ORIGINAL id (e.g. 6-digit ``120325.csv``), while the canonical
    key is ``00120325`` — try the raw id first, then the canonical form.
    """
    d = Path(states_dir or DEFAULT_STATES_DIR) / subject
    key = canonical_session_id(session_date)
    cand = [d / f"{session_date}.csv", d / f"{key}.csv"]
    fp = next((c for c in cand if c.exists()), None)
    if fp is None:
        raise FileNotFoundError(f"No state tags for {subject}/{session_date} (tried {[c.name for c in cand]})")
    df = pd.read_csv(fp)
    df["sess_key"] = key
    return df


def session_stage_map(manifest_path=None) -> dict:
    m = pd.read_csv(manifest_path or DEFAULT_MANIFEST)
    return {canonical_session_id(s): str(stg)
            for s, stg in zip(m["session_name"], m["stage"])}


def b9_cfg() -> TFGLMConfig:
    """The registry's exact _cfg('log2') — DO NOT change these values."""
    return TFGLMConfig(
        include_movement=False, include_phase=False,
        include_tiled_baseline=True, standardize_design=True,
        fast_fit=True, responsive_criterion="c2",
        tf_encoding="log2", min_pulses_per_label=20,
    )


def state_trial_indices(tags: pd.DataFrame, state: str,
                        conf_thresh: float = STATE_CONF_THRESH) -> List[int]:
    keep = (tags["state_label"] == state) & (tags["state_confidence"] >= conf_thresh)
    return sorted(int(i) for i in tags.loc[keep, "trial_idx"].tolist())


def state_conditioned_encoding(session, subset_idx: List[int], cfg: TFGLMConfig,
                               min_spikes: int = 500,
                               unit_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Run the registry TF-GLM on the given trial subset; return per-unit c1_r.

    Wiring is identical to the registry run except that ``trials_regs`` is
    filtered to ``subset_idx`` (0-based indices into ``session.trials``) before
    ``assemble_design``. ``unit_ids`` (optional) restricts the units fit (used to
    bound compute — e.g. responsive + a non-responsive sample).
    """
    trials_regs, units = session_trial_regressors(session, cfg)
    if unit_ids is not None:
        keep = {int(u) for u in unit_ids}
        units = {u: s for u, s in units.items() if int(u) in keep}
    subset_idx = [i for i in subset_idx if 0 <= i < len(trials_regs)]
    if len(subset_idx) < cfg.n_folds:
        return pd.DataFrame(columns=_ENC_COLS)
    sub = [trials_regs[i] for i in subset_idx]
    design = assemble_design(sub, cfg)
    if design.bin_edges.size == 0:
        return pd.DataFrame(columns=_ENC_COLS)
    fold_ids = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    tf_cols = design.col_groups["tf"]
    rows = []
    for uid, spikes in units.items():
        y = count_vector(sub, spikes, design)
        ns = float(y.sum())
        if ns < min_spikes:
            continue
        full = fit_poisson_cv(design.X, y, cfg, fold_ids)
        Xr = design.X.copy(); Xr[:, tf_cols] = 0.0
        red = fit_poisson_cv(Xr, y, cfg, fold_ids)
        r = identify_tf_responsive_pulse(design, y, full, red, cfg)
        rows.append({"unit": int(uid), "c1_r": r["c1_r"], "c2_p": r["c2_p"],
                     "is_responsive_rerun": r["is_responsive"], "r_red_mean": r["r_red_mean"],
                     "n_folds_used": r["n_folds_used"], "kernel_peak_t": r["kernel_peak_t"],
                     "kernel_fwhm": r["kernel_fwhm"], "n_spikes": ns, "n_trials": len(subset_idx)})
    return pd.DataFrame(rows, columns=_ENC_COLS)
