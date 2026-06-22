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
    # FAITHFUL full model (movement + tiled baseline + phase + standardized),
    # log2(TF)/0.25 octave encoding = the authors' published encoding.
    "c2_p_log2", "resp_log2", "r_full_log2", "r_red_log2", "c1_r_log2",
    # Same full model with a LINEAR-Hz TF encoding = control for whether any
    # responsiveness is an encoding artifact.
    "c2_p_lin", "resp_lin", "r_full_lin", "r_red_lin", "c1_r_lin",
    # TF kernel shape (from the log2 full model)
    "kernel_peak_t", "kernel_fwhm", "n_folds_used", "fit_s",
]


def _cols(design, key):
    """Integer column indices of a col_group slice (empty array if absent)."""
    sl = design.col_groups.get(key)
    if sl is None:
        return np.empty(0, dtype=int)
    return np.arange(*sl.indices(design.X.shape[1]), dtype=int)


def _faithful_cfg(tf_encoding):
    """The authors' full model: movement controls + 80x200ms tiled baseline +
    12-bin phase + whole-design standardization (glmnet standardize=true),
    fast_fit, C2 criterion. tf_encoding 'log2' (faithful) or 'linear' (control)."""
    return TFGLMConfig(
        include_movement=True, include_phase=True, include_tiled_baseline=True,
        standardize_design=True, fast_fit=True, responsive_criterion="c2",
        tf_encoding=tf_encoding)


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
             with_linear=True, verbose=True):
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
          f"({len(done)} cached, {len(todo)} to fit) | with_linear={with_linear}",
          flush=True)
    if not todo:
        return 0

    # Build the faithful design ONCE per (session, region) in both encodings.
    # The non-TF columns are identical (and identically standardized) between
    # them, so the reduced (TF-zeroed) model is shared across encodings.
    cfg_log = _faithful_cfg("log2")
    ks = load_khilkevich_session(session_dir)
    trials, units = khilkevich_trial_regressors(ks, cfg_log, region=region)
    d_log = assemble_design(trials, cfg_log)
    folds = make_trial_folds(d_log.trial_index, cfg_log.n_folds, cfg_log.seed)
    tf_cols = _cols(d_log, "tf")
    if tf_cols.size == 0:
        print(f"[task {task_id}] FATAL: no TF columns in design; aborting", flush=True)
        return 2
    d_lin = assemble_design(trials, _faithful_cfg("linear")) if with_linear else None
    X_red = d_log.X.copy(); X_red[:, tf_cols] = 0.0   # TF ablated (encoding-free)
    print(f"[task {task_id}] design X={d_log.X.shape} "
          f"(tiled_baseline={_cols(d_log,'tiled_baseline').size}, "
          f"phase={_cols(d_log,'phase').size}, movement="
          f"{_cols(d_log,'motion_energy').size+_cols(d_log,'pupil').size})", flush=True)

    for k, uid in enumerate(todo):
        try:
            if uid not in units:
                print(f"  [task {task_id}] unit {uid} absent from region "
                      f"{region}; skip", flush=True)
                continue
            y = count_vector(trials, units[uid], d_log)
            ns = float(y.sum())
            if ns < MIN_SPIKES:
                print(f"  [task {task_id}] unit {uid}: {int(ns)} < {MIN_SPIKES} "
                      f"spk; skip", flush=True)
                continue
            t0 = time.time()
            red = fit_poisson_cv(X_red, y, cfg_log, folds)
            full_log = fit_poisson_cv(d_log.X, y, cfg_log, folds)
            o_log = identify_tf_responsive(d_log, y, full_log, red, cfg_log)
            rec = dict(
                task_id=task_id, session_rel=session_rel, region=region,
                unit=uid, n_spikes=ns,
                c2_p_log2=float(o_log["c2_p"]), resp_log2=bool(o_log["is_responsive"]),
                r_full_log2=float(o_log["r_full_mean"]),
                r_red_log2=float(o_log["r_red_mean"]), c1_r_log2=float(o_log["c1_r"]),
                kernel_peak_t=float(o_log["kernel_peak_t"]),
                kernel_fwhm=float(o_log["kernel_fwhm"]),
                n_folds_used=int(o_log["n_folds_used"]),
            )
            if with_linear:
                full_lin = fit_poisson_cv(d_lin.X, y, cfg_log, folds)
                o_lin = identify_tf_responsive(d_lin, y, full_lin, red, cfg_log)
                rec.update(
                    c2_p_lin=float(o_lin["c2_p"]), resp_lin=bool(o_lin["is_responsive"]),
                    r_full_lin=float(o_lin["r_full_mean"]),
                    r_red_lin=float(o_lin["r_red_mean"]), c1_r_lin=float(o_lin["c1_r"]),
                )
            else:
                rec.update(c2_p_lin=np.nan, resp_lin=False, r_full_lin=np.nan,
                           r_red_lin=np.nan, c1_r_lin=np.nan)
            rec["fit_s"] = round(time.time() - t0, 1)
            _append(out_csv, rec)
            if verbose:
                print(f"  [task {task_id}] unit {uid} ({k+1}/{len(todo)}): "
                      f"{int(ns)}spk | log2 resp={rec['resp_log2']} "
                      f"dR={rec['r_full_log2']-rec['r_red_log2']:+.4f} "
                      f"p={rec['c2_p_log2']:.1e} | lin resp={rec['resp_lin']} "
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
    p.add_argument("--no-linear", action="store_true",
                   help="skip the linear-encoding control (one fewer fit/unit)")
    a = p.parse_args(argv)
    return run_task(a.targets, a.task_id, a.data_root, a.out_dir,
                    with_linear=not a.no_linear)


if __name__ == "__main__":
    sys.exit(main())
