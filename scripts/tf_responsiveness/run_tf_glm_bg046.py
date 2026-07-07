"""Run the corrected (pulse-criterion) TF-GLM on BG_046 sessions, LOCALLY.

BG_046 pkls are on E: (local), so this never touches the ceph SMB gateway. The
faithful design minus the regressors BG_046 lacks: include_movement=False (no
processed video), include_phase=False (grating phase not in the data; the author
likely randomizes phi0 per trial, so it can't be reconstructed). TF comes from
trial.baseline_values (St1TrialVector). Pulses = +/-0.5 s.d. (paper Methods p17),
criterion = full-vs-reduced fast-slow pulse-PETH (identify_tf_responsive_pulse).

Parallelism: one worker per session (each worker builds its own design, so no
big-array IPC), BLAS pinned to 1 thread/worker. Each session writes:
  <out>/bg046_<session>.csv        per-unit C1/C2/resp + spike count
  <out>/bg046_<session>_peth.npz   per-unit fast/slow pulse PETHs (actual +
                                   full-model predicted) for the visualization

Usage:
  py run_tf_glm_bg046.py --sessions BG_046_01092025,BG_046_02092025 --workers 4
  py run_tf_glm_bg046.py --expert 8 --workers 8     # first 8 Sept(Expert) sessions
"""
from __future__ import annotations
import os
# Pin BLAS BEFORE numpy import so each worker process is single-threaded.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import argparse
import time
import glob
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_bg046/src")
from visdetect.core.session import load_session
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive_pulse, pulse_times_from_tf,
    tf_pulse_peth,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors

PKL_DIR = Path("E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/data/pkls/BG_046")
MIN_SPIKES = 500
OUT_COLS = ["session", "unit", "n_spikes", "c1_r", "c2_p", "is_responsive",
            "r_full", "r_red", "n_folds_used", "kernel_peak_t", "fit_s"]


def _cfg():
    return TFGLMConfig(include_movement=False, include_phase=False,
                       include_tiled_baseline=True, standardize_design=True,
                       fast_fit=True, tf_encoding="log2", responsive_criterion="c2",
                       min_pulses_per_label=20)


def process_session(session_name, out_dir, max_units=None, verbose=True):
    """Fit every good/stable unit in one session; save CSV + PETH npz."""
    out_csv = Path(out_dir) / f"bg046_{session_name}.csv"
    out_npz = Path(out_dir) / f"bg046_{session_name}_peth.npz"
    if out_csv.exists() and out_npz.exists():
        return f"{session_name}: cached"
    try:
        cfg = _cfg()
        pkl = PKL_DIR / f"{session_name}.pkl"
        sess = load_session(str(pkl))
        trials, units = session_trial_regressors(sess, cfg)
        d = assemble_design(trials, cfg)
        folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
        fast, slow = pulse_times_from_tf(d, cfg)
        tfcols = np.arange(*d.col_groups["tf"].indices(d.X.shape[1]))
        Xr = d.X.copy(); Xr[:, tfcols] = 0.0
        ti, win, bs = d.trial_index, cfg.pulse_eval_win, cfg.bin_s

        spk = {u: float(np.sum(np.isfinite(units[u]))) for u in units}
        uids = [u for u in sorted(units, key=lambda u: spk[u], reverse=True)
                if spk[u] >= MIN_SPIKES]
        if max_units:
            uids = uids[:max_units]

        rows, peth = [], {}
        # shared pulse-PETH time axis (same for every unit)
        t_axis = None
        for k, u in enumerate(uids):
            y = count_vector(trials, units[u], d)
            if y.sum() < MIN_SPIKES:
                continue
            t0 = time.time()
            full = fit_poisson_cv(d.X, y, cfg, folds)
            red = fit_poisson_cv(Xr, y, cfg, folds)
            o = identify_tf_responsive_pulse(d, y, full, red, cfg)
            # actual + full-model-predicted fast/slow pulse PETHs (Hz)
            tax, a_fast = tf_pulse_peth(y, d.bin_edges, fast, win, bs, trial_index=ti)
            _, a_slow = tf_pulse_peth(y, d.bin_edges, slow, win, bs, trial_index=ti)
            _, p_fast = tf_pulse_peth(full.pred, d.bin_edges, fast, win, bs, trial_index=ti)
            _, p_slow = tf_pulse_peth(full.pred, d.bin_edges, slow, win, bs, trial_index=ti)
            t_axis = tax
            peth[str(int(u))] = np.vstack([a_fast / bs, a_slow / bs,
                                           p_fast / bs, p_slow / bs])  # -> Hz
            rows.append(dict(
                session=session_name, unit=int(u), n_spikes=float(y.sum()),
                c1_r=float(o["c1_r"]), c2_p=float(o["c2_p"]),
                is_responsive=bool(o["is_responsive"]),
                r_full=float(o["r_full_mean"]), r_red=float(o["r_red_mean"]),
                n_folds_used=int(o["n_folds_used"]),
                kernel_peak_t=float(o["kernel_peak_t"]),
                fit_s=round(time.time() - t0, 1)))
            if verbose and (k % 10 == 0):
                print(f"  [{session_name}] {k+1}/{len(uids)} units", flush=True)

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=OUT_COLS).to_csv(out_csv, index=False)
        np.savez_compressed(out_npz, t_axis=(t_axis if t_axis is not None else np.zeros(0)),
                            units=np.array(list(peth.keys())),
                            **{f"u{k}": v for k, v in peth.items()})
        n_resp = int(sum(r["is_responsive"] for r in rows))
        return f"{session_name}: {len(rows)} units, {n_resp} TF-responsive"
    except Exception as e:
        return f"{session_name}: FAILED {e}\n{traceback.format_exc()}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", default=None, help="comma-sep session names")
    p.add_argument("--expert", type=int, default=None,
                   help="use first N Sept-2025 (Expert) sessions")
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default="E:/python_analysis/git_repos/vd_tf_bg046/"
                   "data/cache/tf_glm_bg046")
    a = p.parse_args(argv)

    if a.sessions:
        sess = [s.strip() for s in a.sessions.split(",") if s.strip()]
    elif a.expert:
        sept = sorted(Path(x).stem for x in glob.glob(str(PKL_DIR / "*092025*.pkl")))
        sess = sept[:a.expert]
    else:
        raise SystemExit("pass --sessions or --expert N")
    print(f"Running {len(sess)} sessions on {a.workers} workers -> {a.out_dir}")
    print("sessions:", sess, flush=True)

    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_session, s, a.out_dir, a.max_units): s for s in sess}
        for fut in cf.as_completed(futs):
            print(" ", fut.result().split("\n")[0], flush=True)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
