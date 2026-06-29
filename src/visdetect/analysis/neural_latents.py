"""N1 neural-latent correspondence: join the B8 per-trial timing latent to
per-trial striatal tensors and test the urgency-ramp hypothesis. Library only
(no plotting / no __main__). See spec 2026-06-29-N1-...-design.md."""
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from visdetect.analysis.config import ROOT, canonical_session_id, parse_session_date
from visdetect.analysis.utils import (
    build_population_tensor, compute_zscore_normalized, get_good_cluster_ids)

DEFAULT_LATENT_CSV = os.path.join(
    ROOT, "data", "cache", "decision_latents", "decision_latents_by_state.csv")

# columns copied verbatim from the latent table into the per-trial y frame
_Y_COLS = ["trial_idx", "outcome", "change_size", "decision_time",
           "change_time_planned", "change_reached", "state_label",
           "timing_urgency_at_decision", "itchiness_caution", "sharpness_drift",
           "evidence_integral_at_decision", "expected_change_time"]

@dataclass
class JoinResult:
    z: np.ndarray            # (n_kept, n_bins, n_units), per-unit shared-baseline z
    bin_centers: np.ndarray  # (n_bins,) seconds rel. Baseline_ON
    y: pd.DataFrame          # one row per kept trial, in z row order
    unit_ids: list           # cluster ids, positional with z's unit axis
    kept_trials: list        # original session trial indices, in z row order

def load_latent_table(path=None):
    df = pd.read_csv(path or DEFAULT_LATENT_CSV, dtype={"session_name": str})
    df["sess_canon"] = df["session_name"].map(canonical_session_id)
    return df

def fitted_expert_sessions(df):
    fitted = df.loc[df["sharpness_drift"].notna(), "sess_canon"].unique()
    return sorted(fitted, key=parse_session_date)

def join_session(session, latent_rows, *, window, bin_size=0.025,
                 baseline_window=(-1.3, -0.3), min_rate_hz=1.0, verify=True):
    good_ids = get_good_cluster_ids(session, min_rate_hz=min_rate_hz)
    tensor, bin_centers, valid_trials = build_population_tensor(
        session, cluster_ids=good_ids, event_name="Baseline_ON",
        window=window, bin_size=bin_size)
    lut = {int(getattr(r, "trial_idx")): r for r in latent_rows.itertuples(index=False)}
    keep = [r for r, ti in enumerate(valid_trials) if int(ti) in lut]
    if not keep:
        raise ValueError(f"join_session: no overlap between tensor trials and "
                         f"latent trial_idx (n_valid={len(valid_trials)}, "
                         f"n_latent={len(lut)})")
    kept_trials = [int(valid_trials[r]) for r in keep]
    z = compute_zscore_normalized(tensor[keep], bin_centers, baseline_window)
    y = pd.DataFrame([{c: getattr(lut[ti], c) for c in _Y_COLS} for ti in kept_trials])
    if verify:
        _verify_join(session, kept_trials, lut)
    return JoinResult(z=z, bin_centers=bin_centers, y=y,
                      unit_ids=list(good_ids), kept_trials=kept_trials)

def _verify_join(session, kept_trials, lut):
    """Triple-check (outcome / change_size / change_time) that trial_idx indexes
    the SAME trial the latent row describes. Fails loud on any mismatch."""
    base = np.asarray(session.ni_events["Baseline_ON"]).ravel()
    assert len(base) >= len(session.trials), (
        f"Baseline_ON ({len(base)}) shorter than trials ({len(session.trials)})")
    for ti in kept_trials:
        tr, lr = session.trials[ti], lut[ti]
        assert (getattr(tr, "trialoutcome", "") or "").lower() == str(lr.outcome).lower(), \
            f"outcome mismatch at trial_idx={ti}"
        assert np.isclose(float(tr.change_size), float(lr.change_size)), \
            f"change_size mismatch at trial_idx={ti}"
        if np.isfinite(float(lr.change_time_planned)):
            assert np.isclose(float(tr.change_time), float(lr.change_time_planned), atol=1e-6), \
                f"change_time mismatch at trial_idx={ti}"
