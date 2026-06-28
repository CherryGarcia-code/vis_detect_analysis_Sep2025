"""Decisive movement-control test on the authors' CP units.

Re-fits EACH CP unit already scored by the REDUCED model
(``data/cache/tf_glm/khilkevich_diagnostic.csv``) with the FULL
movement-controlled Khilkevich-faithful GLM
(``TFGLMConfig(include_movement=True, include_phase=False, fast_fit=True)``):
motion-energy + pupil (continuous FIR) + air-puff (event FIR) added as nuisance
controls on top of the reduced regressor set. The TF-ablation C2 test is run
INSIDE the full model (full design vs the same design with the TF block zeroed),
on the SAME trial-blocked folds and the SAME ``fast_fit`` λ-selection scheme.

The question: when face/whisker motion-energy + pupil are regressed out, does
the reduced model's TF-responsive flag survive? If the responsive fraction
COLLAPSES the reduced flags were a movement confound; if it SURVIVES and stays
in the paper's 5-45% range the TF signal is genuine.

Responsive criterion = C2 alone (``c2_p < 0.01``), matching the brief's
definition of "TF-responsive" and the cfg default (``responsive_criterion='c2'``).

FULL trials (no --max-trials). Units < 500 spikes are skipped (paper-power
floor); any unit whose fit fails is dropped and counted.

Output:
  data/cache/tf_glm/khilkevich_fullmodel_CP.csv  (per-unit; full-model c2_p +
                                                   joined reduced c2_p)

Run:
  PYTHONPATH=src py scripts/tf_responsiveness/run_tf_glm_fullmodel_CP.py
"""
from __future__ import annotations
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive,
)
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors,
)

ROOT = Path("X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted")
DIAG = _REPO / "data" / "cache" / "tf_glm" / "khilkevich_diagnostic.csv"
OUT = _REPO / "data" / "cache" / "tf_glm" / "khilkevich_fullmodel_CP.csv"

REGION = "CP"
MIN_SPIKES = 500
C2_THRESH = 0.01

# Map the diagnostic session name -> npx_converted session dir (same dirs the
# diagnostic used; see run_tf_glm_diagnostic.SESSIONS).
SESSION_DIRS = {
    "ML_1116764_S02_M2_V1": ROOT / "1116764" / "ML_1116764_S02_M2_V1",
    "ML_1116764_S03_M2_V1": ROOT / "1116764" / "ML_1116764_S03_M2_V1",
}

OUT_COLS = [
    "region", "session", "unit", "n_spikes",
    "c1_r", "c2_p", "r_full_mean", "r_red_mean", "is_responsive",
    "c2_p_reduced", "is_responsive_reduced",
]


def main():
    diag = pd.read_csv(DIAG)
    cp = diag[diag["region"] == REGION].copy()
    cp["unit"] = cp["unit"].astype(int)
    # Reduced-model responsive flag = C2 alone (the brief's definition), NOT the
    # csv's is_responsive column (which may use the paper's c1_and_c2 conjunction).
    reduced_c2 = {(r["session"], int(r["unit"])): float(r["c2_p"])
                  for _, r in cp.iterrows()}

    cfg = TFGLMConfig(include_movement=True, include_phase=False, fast_fit=True)
    print(f"Full model cfg: include_movement={cfg.include_movement} "
          f"include_phase={cfg.include_phase} fast_fit={cfg.fast_fit} "
          f"responsive_criterion={cfg.responsive_criterion!r}", flush=True)

    rows = []
    failed = []
    for sess_name, sub in cp.groupby("session"):
        sdir = SESSION_DIRS[sess_name]
        print(f"\n[{sess_name}] loading {sdir}", flush=True)
        ks = load_khilkevich_session(sdir)
        trials, units = khilkevich_trial_regressors(ks, cfg, region=REGION)
        design = assemble_design(trials, cfg)
        folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
        Xr = design.X.copy()
        Xr[:, design.col_groups["tf"]] = 0.0
        print(f"[{sess_name}] {len(trials)} trials, FULL X={design.X.shape}, "
              f"col_groups={list(design.col_groups)}", flush=True)

        for _, drow in sub.iterrows():
            uid = int(drow["unit"])
            if uid not in units:
                print(f"  [{REGION}|{sess_name}] unit {uid} NOT in region units, "
                      f"dropping", flush=True)
                failed.append((sess_name, uid, "missing_unit"))
                continue
            y = count_vector(trials, units[uid], design)
            if y.sum() < MIN_SPIKES:
                print(f"  [{REGION}|{sess_name}] unit {uid}: {int(y.sum())} spk "
                      f"< {MIN_SPIKES}, skipped", flush=True)
                failed.append((sess_name, uid, "low_spikes"))
                continue
            t0 = time.time()
            try:
                full = fit_poisson_cv(design.X, y, cfg, folds)
                red = fit_poisson_cv(Xr, y, cfg, folds)
                out = identify_tf_responsive(design, y, full, red, cfg)
            except Exception as e:  # noqa: BLE001
                print(f"  [{REGION}|{sess_name}] unit {uid} FIT FAILED: {e}",
                      flush=True)
                traceback.print_exc()
                failed.append((sess_name, uid, f"fit_error:{type(e).__name__}"))
                continue
            c2_red = reduced_c2.get((sess_name, uid), np.nan)
            row = {
                "region": REGION, "session": sess_name, "unit": uid,
                "n_spikes": float(y.sum()),
                "c1_r": float(out["c1_r"]), "c2_p": float(out["c2_p"]),
                "r_full_mean": float(out["r_full_mean"]),
                "r_red_mean": float(out["r_red_mean"]),
                "is_responsive": bool(out["is_responsive"]),
                "c2_p_reduced": float(c2_red),
                "is_responsive_reduced": bool(np.isfinite(c2_red)
                                              and c2_red < C2_THRESH),
            }
            rows.append(row)
            print(f"  [{REGION}|{sess_name}] unit {uid}: {int(y.sum())}spk "
                  f"FULL c2_p={out['c2_p']:.3e} (reduced {c2_red:.3e}) "
                  f"c1_r={out['c1_r']:.3f} resp={out['is_responsive']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

        del ks
        import gc
        gc.collect()

    df = pd.DataFrame(rows)[OUT_COLS] if rows else pd.DataFrame(columns=OUT_COLS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # ── Summary ─────────────────────────────────────────────────────────────
    n = len(df)
    n_full_resp = int((df["c2_p"] < C2_THRESH).sum()) if n else 0
    n_red_resp = int(df["is_responsive_reduced"].sum()) if n else 0
    survivors = df[(df["is_responsive_reduced"]) & (df["c2_p"] < C2_THRESH)] if n else df
    n_survive = len(survivors)

    print("\n================ DECISIVE MOVEMENT-CONTROL TEST (CP) ================")
    print(f"Units fit (full model): {n}   (dropped {len(failed)}: {failed})")
    if n:
        print(f"FULL-model  TF-responsive (c2_p<0.01): {n_full_resp}/{n} "
              f"= {100.0*n_full_resp/n:.1f}%")
        print(f"REDUCED     TF-responsive (c2_p<0.01): {n_red_resp}/{n} "
              f"= {100.0*n_red_resp/n:.1f}%")
        print(f"Reduced-flagged units: {n_red_resp}; SURVIVING under movement "
              f"control: {n_survive}")
        if n_red_resp:
            print("\nReduced-flagged units, full-model outcome:")
            rf = df[df["is_responsive_reduced"]]
            for _, r in rf.iterrows():
                tag = "SURVIVES" if r["c2_p"] < C2_THRESH else "collapses"
                print(f"  {r['session']} u{int(r['unit'])}: reduced c2_p="
                      f"{r['c2_p_reduced']:.3e} -> full c2_p={r['c2_p']:.3e}  "
                      f"[{tag}]")
    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
