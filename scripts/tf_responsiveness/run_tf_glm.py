"""Run the validated TF-encoding GLM on visdetect Session data (BG_046/BG_039).

Applies the Khilkevich-Lohse per-unit Poisson encoding GLM (TF FIR +
trial_start + time_in_base + 6 change sizes + lick_prep/lick_exec + reward +
abort + wheel; NO phase, NO motion-energy/pupil) to a directory of visdetect
.pkl sessions and reports the per-session and overall fraction of TF-responsive
neurons by the authors' C2 criterion (paired full-vs-reduced TF-ablation test,
one-sided t-test p<0.01).

This is the scientific payoff: does BG_046 medial striatum (DMS) encode
moment-to-moment baseline TF with the faithful method (an earlier wrong method
gave ~0%)? BG_039 (cortex/M2) is the in-house positive control and should land
HIGH (like VISp ~27%).

Outputs:
  - <out CSV>  (per-unit table: session, unit, n_spikes, c1_r, c2_p,
                r_full_mean, r_red_mean, is_responsive, kernel metrics)

Example
-------
PYTHONPATH=src py scripts/tf_responsiveness/run_tf_glm.py \
    --pkl-dir "E:/.../data/pkls/BG_046" --limit 3 --max-units 30 \
    --label BG_046_DMS --out data/cache/tf_glm/bg046_dms.csv
"""
from __future__ import annotations
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `PYTHONPATH=src` invocation; also self-bootstrap the repo src dir.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from visdetect.core.session import load_session
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors

MIN_SPIKES = 500  # paper-power floor; skip low-spike units


def _list_pkls(pkl_dir: Path, limit=None):
    pkls = sorted(p for p in pkl_dir.glob("*.pkl"))
    if limit:
        pkls = pkls[:limit]
    return pkls


def run_session(pkl_path: Path, cfg: TFGLMConfig, max_units=None,
                max_trials=None, verbose=True):
    """Fit full + reduced GLM per good unit in one session; return per-unit rows.

    ``max_trials`` subsamples the first N trials before assembling the design
    (a RUNTIME knob, not a scientific one -- mirrors run_tf_glm_khilkevich.py).
    BG_046 sessions run ~630 trials -> X ~150k rows, making each nested-CV
    Poisson fit slow (~6 min/unit). Capping trials shrinks the design
    proportionally while leaving thousands of 50-ms TF bins (>> any pulse floor),
    so the per-region TF-responsive fraction is unaffected in spirit. The same
    cap was used for the validated Khilkevich positive control.
    """
    sname = pkl_path.stem
    session = load_session(str(pkl_path))
    trials, units = session_trial_regressors(session, cfg)
    if max_trials:
        trials = trials[:max_trials]
    design = assemble_design(trials, cfg)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)

    # Reduced design: zero the TF FIR block (SAME folds as full).
    Xr = design.X.copy()
    Xr[:, design.col_groups["tf"]] = 0.0

    if verbose:
        print(f"[{sname}] {len(trials)} trials, {len(units)} good units, "
              f"X={design.X.shape}", flush=True)

    # Rank units by total spikes (highest first); keep >= MIN_SPIKES, cap.
    spk_total = {uid: float(units[uid].size) for uid in units}
    uids = [u for u in sorted(units, key=lambda u: spk_total[u], reverse=True)
            if spk_total[u] >= MIN_SPIKES]
    n_qualifying = len(uids)
    if max_units:
        uids = uids[:max_units]
    if verbose:
        print(f"  [{sname}] {n_qualifying} units >= {MIN_SPIKES} spk; "
              f"fitting {len(uids)}", flush=True)

    rows = []
    for k, uid in enumerate(uids):
        y = count_vector(trials, units[uid], design)
        if y.sum() < MIN_SPIKES:
            continue
        t0 = time.time()
        full = fit_poisson_cv(design.X, y, cfg, folds)
        red = fit_poisson_cv(Xr, y, cfg, folds)
        out = identify_tf_responsive(design, y, full, red, cfg)
        rows.append({
            "session": sname,
            "unit": int(uid),
            "n_spikes": float(y.sum()),
            "c1_r": out["c1_r"],
            "c2_p": out["c2_p"],
            "r_full_mean": out["r_full_mean"],
            "r_red_mean": out["r_red_mean"],
            "is_responsive": bool(out["is_responsive"]),
            "kernel_peak_t": out["kernel_peak_t"],
            "kernel_fwhm": out["kernel_fwhm"],
        })
        if verbose:
            print(f"  [{sname}] unit {uid} ({k+1}/{len(uids)}): "
                  f"{int(y.sum())} spk, c1_r={out['c1_r']:.2f} "
                  f"c2_p={out['c2_p']:.1e} resp={out['is_responsive']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    del session
    gc.collect()
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl-dir", required=True, help="directory of .pkl sessions")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N sessions (chronological by name)")
    p.add_argument("--max-units", type=int, default=None,
                   help="cap units per session (highest-spike first)")
    p.add_argument("--max-trials", type=int, default=None,
                   help="subsample first N trials (runtime knob; shrinks design)")
    p.add_argument("--label", required=True,
                   help="region label for reporting, e.g. BG_046_DMS")
    p.add_argument("--out", required=True, help="output per-unit CSV path")
    p.add_argument("--include-phase", action="store_true",
                   help="(phase absent from BG pkls; leave off)")
    p.add_argument("--no-fast-fit", action="store_true",
                   help="use full nested-CV lambda search (much slower)")
    a = p.parse_args(argv)

    cfg = TFGLMConfig(include_phase=a.include_phase, fast_fit=not a.no_fast_fit)
    pkl_dir = Path(a.pkl_dir)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pkls = _list_pkls(pkl_dir, a.limit)
    print(f"=== {a.label}: {len(pkls)} session(s) from {pkl_dir} "
          f"(criterion={cfg.responsive_criterion}) ===", flush=True)

    from visdetect.analysis.lick_channels import NoLickChannelError

    all_rows = []
    for pkl in pkls:
        try:
            rows = run_session(pkl, cfg, a.max_units, a.max_trials)
        except NoLickChannelError as exc:
            # Skip, don't abort: this driver globs a whole directory, so one
            # unresolvable session must not discard every session after it.
            print(f"  SKIP {pkl}: no usable NI lick channel ({exc})", flush=True)
            continue
        all_rows.extend(rows)
        if rows:
            df_s = pd.DataFrame(rows)
            n = len(df_s)
            frac = 100.0 * df_s["is_responsive"].mean()
            print(f"--- {pkl.stem}: {n} units, TF-responsive {frac:.1f}% "
                  f"(C2) ---", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)

    n = len(df)
    if n:
        frac = 100.0 * df["is_responsive"].mean()
        n_resp = int(df["is_responsive"].sum())
        print(f"\n=== {a.label} OVERALL: {n_resp}/{n} units TF-responsive "
              f"= {frac:.1f}% (C2) ===", flush=True)
    else:
        print(f"\n=== {a.label} OVERALL: 0 units fitted ===", flush=True)
    print(f"Saved per-unit table: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
