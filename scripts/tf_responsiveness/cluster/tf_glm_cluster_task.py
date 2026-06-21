"""Cluster array-task worker — brain-wide Khilkevich-Lohse TF-GLM replication.

One array task = one row of a targets CSV: ``(session_rel, region, unit_ids)``.
For each unit it assembles ONE movement-inclusive design (the corrected
``log2(TF)/0.25``-octave encoding, commit 37419eb) and fits FOUR
column-masked ridge-Poisson variants on the SAME 10-fold CV split::

    full_move      = all regressors (incl. motion-energy + pupil + air-puff)
    reduced_move   = full_move with the TF kernel zeroed
    full_nomove    = full_move with the movement regressors zeroed
    reduced_nomove = full_move with TF AND movement zeroed

C2 (the paired one-sided t-test that the TF kernel improves held-out
prediction) is evaluated TWICE: against the movement-controlled nuisance set
(full_move vs reduced_move) and against the no-movement set (full_nomove vs
reduced_nomove). Recording both lets the master table report the
movement-controlled TF-responsive fraction per region AND test whether
no-movement-flagged cells SURVIVE movement control (genuine TF) or COLLAPSE
(movement confound) -- the decisive arbiter.

Per-unit rows are appended to ``<out-dir>/task_<id>.csv`` immediately, so a
re-queued/pre-empted task resumes where it left off (units already in the file
are skipped).

Usage (from the sbatch)::

    PYTHONPATH=<stage>/code/src python tf_glm_cluster_task.py \
        --targets <stage>/targets.csv --task-id $SLURM_ARRAY_TASK_ID \
        --data-root /ceph/.../npx_converted --out-dir <stage>/results
"""
from __future__ import annotations
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Self-bootstrap the repo `src` dir onto sys.path if PYTHONPATH was not set
# (walk up from this file until a `src/visdetect` package is found).
_HERE = Path(__file__).resolve()
for _up in _HERE.parents:
    if (_up / "src" / "visdetect").is_dir():
        if str(_up / "src") not in sys.path:
            sys.path.insert(0, str(_up / "src"))
        break

from visdetect.analysis.tf_glm import (  # noqa: E402
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive,
)
from visdetect.analysis.tf_glm_data import (  # noqa: E402
    load_khilkevich_session, khilkevich_trial_regressors,
)

MIN_SPIKES = 500  # per-unit spike-count gate (paper-power floor), applied at fit

OUT_COLS = [
    "task_id", "session_rel", "region", "unit", "n_spikes",
    # movement-controlled TF test (the faithful paper full model)
    "c2_p_move", "resp_move", "r_full_move", "r_red_move", "c1_r_move",
    # no-movement TF test (movement regressors ablated)
    "c2_p_nomove", "resp_nomove", "r_full_nomove", "r_red_nomove", "c1_r_nomove",
    # TF kernel shape (from the movement-controlled full model)
    "kernel_peak_t", "kernel_fwhm", "n_folds_used", "fit_s",
]


def _cols(design, key):
    """Integer column indices of a col_group slice (empty array if absent)."""
    sl = design.col_groups.get(key)
    if sl is None:
        return np.empty(0, dtype=int)
    return np.arange(*sl.indices(design.X.shape[1]), dtype=int)


def _move_cols(design):
    """Union of motion_energy + pupil + airpuff column indices."""
    idx = np.concatenate([_cols(design, k)
                          for k in ("motion_energy", "pupil", "airpuff")])
    return np.unique(idx).astype(int)


def _done_units(out_csv):
    if not out_csv.exists():
        return set()
    try:
        d = pd.read_csv(out_csv)
        return {int(u) for u in d["unit"]}
    except Exception:
        return set()


def _append(out_csv, row):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = not out_csv.exists()
    pd.DataFrame([row])[OUT_COLS].to_csv(
        out_csv, mode="a", header=header, index=False)


def run_task(targets_csv, task_id, data_root, out_dir,
             both_models=True, verbose=True):
    tg = pd.read_csv(targets_csv)
    sel = tg[tg["task_id"] == task_id]
    if not len(sel):
        print(f"[task {task_id}] no such task_id in {targets_csv}", flush=True)
        return 1
    r = sel.iloc[0]
    session_rel = str(r["session_rel"])
    region = str(r["region"])
    unit_ids = [int(u) for u in str(r["unit_ids"]).split(";") if u != ""]
    session_dir = str(Path(data_root) / session_rel)
    out_csv = Path(out_dir) / f"task_{task_id}.csv"

    done = _done_units(out_csv)
    todo = [u for u in unit_ids if u not in done]
    print(f"[task {task_id}] {session_rel} | {region} | {len(unit_ids)} units "
          f"({len(done)} cached, {len(todo)} to fit) | both_models={both_models}",
          flush=True)
    if not todo:
        return 0

    cfg = TFGLMConfig(include_movement=True, include_phase=False,
                      fast_fit=True, responsive_criterion="c2")
    ks = load_khilkevich_session(session_dir)
    trials, units = khilkevich_trial_regressors(ks, cfg, region=region)
    design = assemble_design(trials, cfg)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)

    X = design.X
    tf_cols = _cols(design, "tf")
    mv_cols = _move_cols(design)
    if tf_cols.size == 0:
        print(f"[task {task_id}] FATAL: no TF columns in design; aborting",
              flush=True)
        return 2
    if both_models and mv_cols.size == 0:
        print(f"[task {task_id}] WARN: no movement columns found; "
              f"falling back to movement-only model", flush=True)
        both_models = False

    def zeroed(*cols):
        Z = X.copy()
        for c in cols:
            if len(c):
                Z[:, c] = 0.0
        return Z

    X_red_move = zeroed(tf_cols)              # full_move = X (everything)
    if both_models:
        X_full_nomove = zeroed(mv_cols)       # - movement
        X_red_nomove = zeroed(tf_cols, mv_cols)  # - TF - movement

    for k, uid in enumerate(todo):
        try:
            if uid not in units:
                print(f"  [task {task_id}] unit {uid} absent from region "
                      f"{region}; skip", flush=True)
                continue
            y = count_vector(trials, units[uid], design)
            ns = float(y.sum())
            if ns < MIN_SPIKES:
                print(f"  [task {task_id}] unit {uid}: {int(ns)} < {MIN_SPIKES} "
                      f"spk; skip", flush=True)
                continue
            t0 = time.time()
            fit_fm = fit_poisson_cv(X, y, cfg, folds)
            fit_rm = fit_poisson_cv(X_red_move, y, cfg, folds)
            out_m = identify_tf_responsive(design, y, fit_fm, fit_rm, cfg)
            rec = dict(
                task_id=task_id, session_rel=session_rel, region=region,
                unit=uid, n_spikes=ns,
                c2_p_move=float(out_m["c2_p"]),
                resp_move=bool(out_m["is_responsive"]),
                r_full_move=float(out_m["r_full_mean"]),
                r_red_move=float(out_m["r_red_mean"]),
                c1_r_move=float(out_m["c1_r"]),
                kernel_peak_t=float(out_m["kernel_peak_t"]),
                kernel_fwhm=float(out_m["kernel_fwhm"]),
                n_folds_used=int(out_m["n_folds_used"]),
            )
            if both_models:
                fit_fn = fit_poisson_cv(X_full_nomove, y, cfg, folds)
                fit_rn = fit_poisson_cv(X_red_nomove, y, cfg, folds)
                out_n = identify_tf_responsive(design, y, fit_fn, fit_rn, cfg)
                rec.update(
                    c2_p_nomove=float(out_n["c2_p"]),
                    resp_nomove=bool(out_n["is_responsive"]),
                    r_full_nomove=float(out_n["r_full_mean"]),
                    r_red_nomove=float(out_n["r_red_mean"]),
                    c1_r_nomove=float(out_n["c1_r"]),
                )
            else:
                rec.update(c2_p_nomove=np.nan, resp_nomove=False,
                           r_full_nomove=np.nan, r_red_nomove=np.nan,
                           c1_r_nomove=np.nan)
            rec["fit_s"] = round(time.time() - t0, 1)
            _append(out_csv, rec)
            if verbose:
                print(f"  [task {task_id}] unit {uid} ({k+1}/{len(todo)}): "
                      f"{int(ns)}spk | move resp={rec['resp_move']} "
                      f"p={rec['c2_p_move']:.1e} | nomove resp={rec['resp_nomove']} "
                      f"p={rec['c2_p_nomove'] if both_models else float('nan'):.1e} "
                      f"[{rec['fit_s']}s]", flush=True)
        except Exception as e:  # one bad unit must not kill the whole task
            print(f"  [task {task_id}] unit {uid} FAILED: {e}\n"
                  f"{traceback.format_exc()}", flush=True)
            continue
    print(f"[task {task_id}] done.", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--data-root", required=True,
                   help="npx_converted root (ceph on cluster, X: on Windows)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-both-models", action="store_true",
                   help="fit ONLY the movement-controlled model (half the fits; "
                        "skips the no-movement survival comparison)")
    a = p.parse_args(argv)
    return run_task(a.targets, a.task_id, a.data_root, a.out_dir,
                    both_models=not a.no_both_models)


if __name__ == "__main__":
    sys.exit(main())
