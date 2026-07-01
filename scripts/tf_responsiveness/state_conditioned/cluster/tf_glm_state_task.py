"""Cluster array-task worker — RIGOROUS state-conditioned TF-GLM re-fit.

Tests whether TF encoding is state-modulated by RE-FITTING the pulse-criterion
GLM on trial subsets, per cell:
  engaged  = trials in StimSens/Impulsive states (Disengaged dropped)
  matched  = a random subset of ALL trials with the SAME trial count as engaged
             (the POWER control: same n, random states)

If engaged C1 >> matched C1, the sharper TF response is genuine state-gating,
not just "fewer, cleaner trials". Session-wide C1/C2 come from the registry
(joined downstream), so we don't refit the full session here.

Targets a subset of cells (responsive + any with a hint of TF, c1>=floor) read
from the registry, so this is far cheaper than the full sweep. Per (subject,
session, chunk) it builds the engaged + matched designs ONCE, then fits each
target unit on both. Resumable per-unit CSV.

Usage (cluster):
  python tf_glm_state_task.py --targets targets_state.csv --task-id $SLURM_ARRAY_TASK_ID \
     --data-root <pkls> --state-root <state_tags> --out-dir <results_state>
"""
from __future__ import annotations
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
for _up in _HERE.parents:
    if (_up / "src" / "visdetect").is_dir():
        if str(_up / "src") not in sys.path:
            sys.path.insert(0, str(_up / "src"))
        break

from visdetect.core.session import load_session  # noqa: E402
from visdetect.analysis.tf_glm import (  # noqa: E402
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive_pulse, pulse_times_from_tf)
from visdetect.analysis.tf_glm_data import session_trial_regressors  # noqa: E402

MIN_SPIKES = 500
ENGAGED = {"StimSens", "Impulsive"}
SEED = 42
OUT_COLS = ["task_id", "subject", "session", "unit", "n_spikes",
            "n_eng_trials", "n_all_trials",
            "eng_c1", "eng_c2_p", "eng_resp", "eng_r_full", "eng_r_red",
            "mat_c1", "mat_c2_p", "mat_resp", "mat_r_full", "mat_r_red", "fit_s"]


def _cfg():
    return TFGLMConfig(include_movement=False, include_phase=False, include_tiled_baseline=True,
                       standardize_design=True, fast_fit=True, responsive_criterion="c2",
                       tf_encoding="log2", min_pulses_per_label=20)


def _tf_cols(d):
    sl = d.col_groups.get("tf")
    return np.arange(*sl.indices(d.X.shape[1]), dtype=int) if sl is not None else np.empty(0, int)


def _done_units(f):
    if not f.exists():
        return set()
    try:
        return {int(u) for u in pd.read_csv(f)["unit"]}
    except Exception:
        return set()


def _append(f, row):
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row])[OUT_COLS].to_csv(f, mode="a", header=not f.exists(), index=False)


def _state_by_trial(state_root, subject, session):
    date = session.replace(f"{subject}_", "", 1)
    f = Path(state_root) / subject / f"{date}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).set_index("trial_idx")["state_label"].to_dict()


def _fit_condition(trials_sub, units, uid, cfg):
    """Fit full+reduced on a trial subset; return identify_tf_responsive_pulse dict
    (or None if the subset can't yield a valid TF design)."""
    d = assemble_design(trials_sub, cfg)
    tf = _tf_cols(d)
    if tf.size == 0 or d.X.shape[0] < 50:
        return None
    folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
    Xr = d.X.copy(); Xr[:, tf] = 0.0
    y = count_vector(trials_sub, units[uid], d)
    if y.sum() < MIN_SPIKES // 2:
        return None
    red = fit_poisson_cv(Xr, y, cfg, folds)
    full = fit_poisson_cv(d.X, y, cfg, folds)
    return identify_tf_responsive_pulse(d, y, full, red, cfg)


def run_task(targets, task_id, data_root, state_root, out_dir):
    tg = pd.read_csv(targets, dtype={"session": str, "subject": str, "pkl_rel": str})
    sel = tg[tg["task_id"] == task_id]
    if not len(sel):
        print(f"[task {task_id}] no such task_id", flush=True); return 1
    r = sel.iloc[0]
    subject, session = str(r["subject"]), str(r["session"])
    pkl = Path(data_root) / str(r["pkl_rel"])
    unit_ids = [int(u) for u in str(r["unit_ids"]).split(";") if u != ""]
    out_csv = Path(out_dir) / f"task_{task_id}.csv"
    done = _done_units(out_csv)
    todo = [u for u in unit_ids if u not in done]
    print(f"[task {task_id}] {subject}/{session} | {len(unit_ids)} target units "
          f"({len(done)} cached, {len(todo)} to fit)", flush=True)
    if not todo:
        return 0
    if not pkl.exists():
        print(f"[task {task_id}] FATAL: pkl not found {pkl}", flush=True); return 2
    lab = _state_by_trial(state_root, subject, session)
    if lab is None:
        print(f"[task {task_id}] FATAL: no state tags for {session}", flush=True); return 2

    cfg = _cfg()
    sess = load_session(str(pkl))
    trials, units = session_trial_regressors(sess, cfg)
    n_all = len(trials)
    eng_idx = [i for i in range(n_all) if lab.get(i) in ENGAGED]
    if len(eng_idx) < 30:
        print(f"[task {task_id}] FATAL: only {len(eng_idx)} engaged trials; skip", flush=True)
        return 2
    rng = np.random.default_rng(SEED)
    mat_idx = sorted(rng.choice(n_all, len(eng_idx), replace=False).tolist())
    trials_eng = [trials[i] for i in eng_idx]
    trials_mat = [trials[i] for i in mat_idx]
    print(f"[task {task_id}] trials: all={n_all} engaged={len(eng_idx)} matched={len(mat_idx)}",
          flush=True)

    for k, uid in enumerate(todo):
        try:
            if uid not in units:
                continue
            ns = float(np.sum(np.isfinite(units[uid])))
            if ns < MIN_SPIKES:
                continue
            t0 = time.time()
            oe = _fit_condition(trials_eng, units, uid, cfg)
            om = _fit_condition(trials_mat, units, uid, cfg)
            if oe is None or om is None:
                print(f"  [task {task_id}] u{uid}: degenerate subset; skip", flush=True)
                continue
            rec = dict(task_id=task_id, subject=subject, session=session, unit=int(uid),
                       n_spikes=ns, n_eng_trials=len(eng_idx), n_all_trials=n_all,
                       eng_c1=float(oe["c1_r"]), eng_c2_p=float(oe["c2_p"]),
                       eng_resp=bool(oe["is_responsive"]), eng_r_full=float(oe["r_full_mean"]),
                       eng_r_red=float(oe["r_red_mean"]),
                       mat_c1=float(om["c1_r"]), mat_c2_p=float(om["c2_p"]),
                       mat_resp=bool(om["is_responsive"]), mat_r_full=float(om["r_full_mean"]),
                       mat_r_red=float(om["r_red_mean"]), fit_s=round(time.time() - t0, 1))
            _append(out_csv, rec)
            print(f"  [task {task_id}] u{uid} ({k+1}/{len(todo)}): eng_C1={rec['eng_c1']:.3f}"
                  f"(p={rec['eng_c2_p']:.1e}) vs mat_C1={rec['mat_c1']:.3f} "
                  f"[{rec['fit_s']}s]", flush=True)
        except Exception as e:
            print(f"  [task {task_id}] u{uid} FAILED: {e}\n{traceback.format_exc()}", flush=True)
            continue
    print(f"[task {task_id}] done.", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--state-root", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    return run_task(a.targets, a.task_id, a.data_root, a.state_root, a.out_dir)


if __name__ == "__main__":
    sys.exit(main())
