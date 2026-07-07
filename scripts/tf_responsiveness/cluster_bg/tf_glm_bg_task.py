"""Cluster array-task worker — BG-mouse TF-GLM (pkl path).

The sibling ``cluster/tf_glm_cluster_task.py`` runs the authors' (MoHa) parquet
data; THIS worker runs our own mice from ``Session`` pkls. One array task =
one ``(subject, session, chunk)`` row of a targets CSV. The worker loads the
session pkl, builds the BG regressor design (``session_trial_regressors`` —
baseline-TF from St1TrialVector, lick + wheel + reward controls, NO movement
or phase, exactly as validated locally on BG_046), and for the units in its
chunk fits the corrected pulse-criterion GLM:

    reduced  = full design with the TF kernel zeroed
    full     = all regressors (log2-TF encoding, the authors' faithful encoding)
    full_lin = same with a linear-Hz TF encoding (encoding-artifact control)

C1/C2 (``identify_tf_responsive_pulse``: fast-minus-slow pulse-PETH corr > 0.2
AND the 10-fold residual t-test) decide TF-responsiveness. Per-unit rows are
appended to ``task_<id>.csv`` immediately (resumable: a re-queued task skips
units already in the file). The fast/slow pulse PETHs (actual + full-model
prediction, Hz) are saved to ``task_<id>_peth.npz`` for the visualization.

Unit chunking is by STRIDE (``units[chunk_idx::n_chunks]``) so the targets CSV
needs no per-session unit list — the worker self-partitions after loading the
session's ``good_and_stable_ids`` (sorted by spike count, slowest first).

Usage (from the sbatch)::

    PYTHONPATH=<stage>/code/src python tf_glm_bg_task.py \
        --targets <stage>/targets_bg.csv --task-id $SLURM_ARRAY_TASK_ID \
        --data-root <stage>/bg_pkls --out-dir <stage>/results_bg
"""
from __future__ import annotations
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Self-bootstrap repo `src` onto sys.path if PYTHONPATH was not set.
_HERE = Path(__file__).resolve()
for _up in _HERE.parents:
    if (_up / "src" / "visdetect").is_dir():
        if str(_up / "src") not in sys.path:
            sys.path.insert(0, str(_up / "src"))
        break

from visdetect.core.session import load_session  # noqa: E402
from visdetect.analysis.tf_glm import (  # noqa: E402
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive_pulse, pulse_times_from_tf,
    tf_pulse_peth,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors  # noqa: E402

MIN_SPIKES = 500  # per-unit spike-count gate (paper-power floor), applied at fit

OUT_COLS = [
    "task_id", "subject", "session", "unit", "n_spikes",
    # FAITHFUL log2(TF)/0.25-octave full model (tiled baseline + standardized).
    "c1_r_log2", "c2_p_log2", "resp_log2", "r_full_log2", "r_red_log2",
    # Linear-Hz TF encoding = encoding-artifact control (same reduced model).
    "c1_r_lin", "c2_p_lin", "resp_lin", "r_full_lin", "r_red_lin",
    "kernel_peak_t", "kernel_fwhm", "n_folds_used", "fit_s",
]


def _cfg(tf_encoding):
    """BG faithful design minus the regressors our mice lack (movement, phase).

    Matches the locally validated config: tiled baseline (80x200ms ~= the
    authors' 'time since baseline start'), whole-design standardization,
    fast_fit, 0.5-s.d. pulses, min_pulses_per_label=20."""
    return TFGLMConfig(
        include_movement=False, include_phase=False, include_tiled_baseline=True,
        standardize_design=True, fast_fit=True, responsive_criterion="c2",
        tf_encoding=tf_encoding, min_pulses_per_label=20)


def _tf_cols(design):
    sl = design.col_groups.get("tf")
    if sl is None:
        return np.empty(0, dtype=int)
    return np.arange(*sl.indices(design.X.shape[1]), dtype=int)


def _done_units(out_csv):
    if not out_csv.exists():
        return set()
    try:
        return {int(u) for u in pd.read_csv(out_csv)["unit"]}
    except Exception:
        return set()


def _append(out_csv, row):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = not out_csv.exists()
    pd.DataFrame([row])[OUT_COLS].to_csv(out_csv, mode="a", header=header, index=False)


def _load_peth(npz_path):
    """Existing per-unit PETHs (resume): {unit:int -> 4xnlags}, plus t_axis."""
    if not npz_path.exists():
        return {}, None
    try:
        z = np.load(npz_path, allow_pickle=True)
        t_axis = z["t_axis"]
        peth = {int(u): z[f"u{int(u)}"] for u in z["units"] if f"u{int(u)}" in z}
        return peth, (t_axis if t_axis.size else None)
    except Exception:
        return {}, None


def _save_peth(npz_path, subject, session, t_axis, peth):
    np.savez_compressed(
        npz_path,
        t_axis=(t_axis if t_axis is not None else np.zeros(0)),
        subject=str(subject), session=str(session),
        units=np.array(sorted(peth.keys())),
        **{f"u{u}": v for u, v in peth.items()})


def run_task(targets_csv, task_id, data_root, out_dir, with_linear=True, verbose=True):
    tg = pd.read_csv(targets_csv, dtype={"session": str, "subject": str,
                                         "pkl_rel": str})
    sel = tg[tg["task_id"] == task_id]
    if not len(sel):
        print(f"[task {task_id}] no such task_id in {targets_csv}", flush=True)
        return 1
    r = sel.iloc[0]
    subject = str(r["subject"]); session = str(r["session"])
    pkl_rel = str(r["pkl_rel"])
    chunk_idx = int(r["chunk_idx"]); n_chunks = int(r["n_chunks"])
    pkl_path = Path(data_root) / pkl_rel
    out_csv = Path(out_dir) / f"task_{task_id}.csv"
    out_npz = Path(out_dir) / f"task_{task_id}_peth.npz"

    if not pkl_path.exists():
        print(f"[task {task_id}] FATAL: pkl not found: {pkl_path}", flush=True)
        return 2

    cfg_log = _cfg("log2")
    sess = load_session(str(pkl_path))
    trials, units = session_trial_regressors(sess, cfg_log)
    # self-partition: slowest (highest-spike) units first, strided across chunks
    spk = {u: float(np.sum(np.isfinite(units[u]))) for u in units}
    ordered = sorted(units, key=lambda u: spk[u], reverse=True)
    my_units = ordered[chunk_idx::n_chunks]

    done = _done_units(out_csv)
    todo = [u for u in my_units if u not in done]
    print(f"[task {task_id}] {subject}/{session} chunk {chunk_idx}/{n_chunks} | "
          f"{len(my_units)} units ({len(done)} cached, {len(todo)} to fit) | "
          f"with_linear={with_linear}", flush=True)
    if not todo:
        return 0

    d_log = assemble_design(trials, cfg_log)
    tf_cols = _tf_cols(d_log)
    if tf_cols.size == 0:
        print(f"[task {task_id}] FATAL: no TF columns in design; aborting", flush=True)
        return 2
    folds = make_trial_folds(d_log.trial_index, cfg_log.n_folds, cfg_log.seed)
    X_red = d_log.X.copy(); X_red[:, tf_cols] = 0.0
    d_lin = assemble_design(trials, _cfg("linear")) if with_linear else None
    fast, slow = pulse_times_from_tf(d_log, cfg_log)
    ti, win, bs = d_log.trial_index, cfg_log.pulse_eval_win, cfg_log.bin_s
    print(f"[task {task_id}] design X={d_log.X.shape} (tf_cols={tf_cols.size}) | "
          f"trials={np.unique(ti).size} | fast/slow pulses={fast.size}/{slow.size}",
          flush=True)

    peth, t_axis = _load_peth(out_npz)
    for k, uid in enumerate(todo):
        try:
            y = count_vector(trials, units[uid], d_log)
            ns = float(y.sum())
            if ns < MIN_SPIKES:
                print(f"  [task {task_id}] unit {uid}: {int(ns)} < {MIN_SPIKES} spk; "
                      f"skip", flush=True)
                continue
            t0 = time.time()
            red = fit_poisson_cv(X_red, y, cfg_log, folds)
            full_log = fit_poisson_cv(d_log.X, y, cfg_log, folds)
            o_log = identify_tf_responsive_pulse(d_log, y, full_log, red, cfg_log)
            rec = dict(
                task_id=task_id, subject=subject, session=session, unit=int(uid),
                n_spikes=ns,
                c1_r_log2=float(o_log["c1_r"]), c2_p_log2=float(o_log["c2_p"]),
                resp_log2=bool(o_log["is_responsive"]),
                r_full_log2=float(o_log["r_full_mean"]),
                r_red_log2=float(o_log["r_red_mean"]),
                kernel_peak_t=float(o_log["kernel_peak_t"]),
                kernel_fwhm=float(o_log["kernel_fwhm"]),
                n_folds_used=int(o_log["n_folds_used"]))
            # fast/slow pulse PETHs (actual + full-model pred), Hz, for the viz
            tax, a_fast = tf_pulse_peth(y, d_log.bin_edges, fast, win, bs, trial_index=ti)
            _, a_slow = tf_pulse_peth(y, d_log.bin_edges, slow, win, bs, trial_index=ti)
            _, p_fast = tf_pulse_peth(full_log.pred, d_log.bin_edges, fast, win, bs, trial_index=ti)
            _, p_slow = tf_pulse_peth(full_log.pred, d_log.bin_edges, slow, win, bs, trial_index=ti)
            t_axis = tax
            peth[int(uid)] = np.vstack([a_fast / bs, a_slow / bs, p_fast / bs, p_slow / bs])
            if with_linear:
                full_lin = fit_poisson_cv(d_lin.X, y, cfg_log, folds)
                o_lin = identify_tf_responsive_pulse(d_lin, y, full_lin, red, cfg_log)
                rec.update(
                    c1_r_lin=float(o_lin["c1_r"]), c2_p_lin=float(o_lin["c2_p"]),
                    resp_lin=bool(o_lin["is_responsive"]),
                    r_full_lin=float(o_lin["r_full_mean"]),
                    r_red_lin=float(o_lin["r_red_mean"]))
            else:
                rec.update(c1_r_lin=np.nan, c2_p_lin=np.nan, resp_lin=False,
                           r_full_lin=np.nan, r_red_lin=np.nan)
            rec["fit_s"] = round(time.time() - t0, 1)
            _append(out_csv, rec)
            _save_peth(out_npz, subject, session, t_axis, peth)  # checkpoint
            if verbose:
                print(f"  [task {task_id}] unit {uid} ({k+1}/{len(todo)}): {int(ns)}spk "
                      f"| log2 resp={rec['resp_log2']} C1={rec['c1_r_log2']:.3f} "
                      f"p={rec['c2_p_log2']:.1e} | lin resp={rec['resp_lin']} "
                      f"[{rec['fit_s']}s]", flush=True)
        except Exception as e:  # one bad unit must not kill the task
            print(f"  [task {task_id}] unit {uid} FAILED: {e}\n"
                  f"{traceback.format_exc()}", flush=True)
            continue
    print(f"[task {task_id}] done.", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--data-root", required=True, help="pkl root (ceph on cluster)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-linear", action="store_true",
                   help="skip the linear-encoding control (one fewer fit/unit)")
    a = p.parse_args(argv)
    return run_task(a.targets, a.task_id, a.data_root, a.out_dir,
                    with_linear=not a.no_linear)


if __name__ == "__main__":
    sys.exit(main())
