"""Decisive test: re-fit the diagnostic's VISp units with the FULL
movement-controlled GLM and compare to the reduced-model TF-responsive calls.

Background
----------
``run_tf_glm_diagnostic.py`` fit a REDUCED Poisson encoding GLM (TF FIR +
trial_start + time_in_base + 6 change sizes + lick_prep/lick_exec + reward +
abort + wheel; NO motion-energy/pupil/airpuff) on the Khilkevich-Lohse VISp
units and called a unit TF-responsive when C2 (the paired full-vs-TF-ablated
held-out predictive-correlation test) had ``c2_p < 0.01``. Those reduced flags
could be a MOVEMENT CONFOUND: face/whisker motion and pupil co-vary with the TF
pulses (mice move/lick more on speed changes), so a "TF kernel" can carry
movement variance the reduced model has no nuisance regressor for.

This script re-fits EACH VISp unit already in
``data/cache/tf_glm/khilkevich_diagnostic.csv`` with the FULL
movement-controlled model (``TFGLMConfig(include_movement=True,
include_phase=False, fast_fit=True)``), against a TF-ablated reduced model that
KEEPS the movement regressors (motion-energy + pupil + airpuff). Same trials,
same folds, same identification rule (default ``responsive_criterion="c2"``) as
the diagnostic. The only change is that motion-energy/pupil/airpuff are now
controlled for in BOTH the full and the reduced model, so C2 isolates the TF
kernel's UNIQUE contribution ABOVE movement.

Decisive read-out
------------------
- If the responsive fraction COLLAPSES under movement control, the reduced flags
  were a movement confound.
- If it largely SURVIVES and stays in the paper's 5-45% range, the TF signal is
  genuine.

Outputs
-------
  data/cache/tf_glm/khilkevich_fullmodel_VISp.csv
    per-unit: session, unit, n_spikes, FULL-model c2_p/c1_r/r_full_mean/
    r_red_mean/is_responsive, and the matched REDUCED-model c2_p (joined from
    khilkevich_diagnostic.csv).

Run
---
PYTHONPATH=src py scripts/tf_responsiveness/run_tf_glm_fullmodel_visp.py
"""
from __future__ import annotations
import sys
import time
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

# Same Khilkevich session dirs + region as the diagnostic (single source).
ROOT = Path("X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted")
SESSIONS = {
    "ML_1116764_S02_M2_V1": "1116764/ML_1116764_S02_M2_V1",
    "ML_1116764_S03_M2_V1": "1116764/ML_1116764_S03_M2_V1",
}
REGION = "VISp"
MIN_SPIKES = 500
C2_THRESH = 0.01

DIAG_CSV = _REPO / "data" / "cache" / "tf_glm" / "khilkevich_diagnostic.csv"
OUT_CSV = _REPO / "data" / "cache" / "tf_glm" / "khilkevich_fullmodel_VISp.csv"

OUT_COLS = [
    "region", "session", "unit", "n_spikes",
    "c1_r", "c2_p", "r_full_mean", "r_red_mean", "is_responsive",
    "reduced_c2_p", "reduced_is_responsive",
]


def _append_row(row):
    """Checkpoint one per-unit result immediately (resume-safe)."""
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()
    pd.DataFrame([row])[OUT_COLS].to_csv(
        OUT_CSV, mode="a", header=write_header, index=False)


def _done_units():
    if not OUT_CSV.exists():
        return set()
    d = pd.read_csv(OUT_CSV)
    return set(zip(d["session"].astype(str), d["unit"].astype(int)))


def main():
    diag = pd.read_csv(DIAG_CSV)
    vis = diag[diag["region"] == REGION].copy()
    vis["session"] = vis["session"].astype(str)
    vis["unit"] = vis["unit"].astype(int)
    # reduced-model c2_p, keyed by (session, unit), for the join.
    red_c2 = {(r["session"], int(r["unit"])): float(r["c2_p"])
              for _, r in vis.iterrows()}
    print(f"Diagnostic VISp units: {len(vis)} "
          f"(reduced-flagged c2_p<{C2_THRESH}: "
          f"{int((vis['c2_p'] < C2_THRESH).sum())})", flush=True)

    cfg = TFGLMConfig(include_movement=True, include_phase=False, fast_fit=True)
    done = _done_units()
    n_fail = 0

    for sess_name, rel in SESSIONS.items():
        want = sorted(vis[vis["session"] == sess_name]["unit"].tolist())
        todo = [u for u in want if (sess_name, int(u)) not in done]
        if not todo:
            print(f"[{sess_name}] all {len(want)} VISp units already cached, "
                  f"skipping", flush=True)
            continue

        ks = load_khilkevich_session(ROOT / rel)
        trials, units = khilkevich_trial_regressors(ks, cfg, region=REGION)
        design = assemble_design(trials, cfg)
        folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
        Xr = design.X.copy()
        Xr[:, design.col_groups["tf"]] = 0.0  # TF-ablated; movement KEPT

        has_move = all(g in design.col_groups
                       for g in ("motion_energy", "pupil", "airpuff"))
        print(f"[{sess_name}] {len(trials)} trials, X={design.X.shape}, "
              f"movement_cols={has_move}, "
              f"col_groups={list(design.col_groups.keys())}", flush=True)
        if not has_move:
            print(f"  WARNING: movement regressors absent from design for "
                  f"{sess_name} -- full model would equal reduced!", flush=True)

        print(f"[{sess_name}] fitting {len(todo)}/{len(want)} units "
              f"({len(want) - len(todo)} cached)", flush=True)

        for k, uid in enumerate(todo):
            if int(uid) not in units:
                print(f"  [{sess_name}] unit {uid}: NOT in region {REGION} "
                      f"units -- DROPPED", flush=True)
                n_fail += 1
                continue
            try:
                y = count_vector(trials, units[int(uid)], design)
                if y.sum() < MIN_SPIKES:
                    print(f"  [{sess_name}] unit {uid}: {int(y.sum())} spk "
                          f"< {MIN_SPIKES} -- DROPPED", flush=True)
                    n_fail += 1
                    continue
                t0 = time.time()
                full = fit_poisson_cv(design.X, y, cfg, folds)
                red = fit_poisson_cv(Xr, y, cfg, folds)
                out = identify_tf_responsive(design, y, full, red, cfg)
                rc2 = red_c2.get((sess_name, int(uid)), np.nan)
                row = {
                    "region": REGION, "session": sess_name, "unit": int(uid),
                    "n_spikes": float(y.sum()),
                    "c1_r": float(out["c1_r"]), "c2_p": float(out["c2_p"]),
                    "r_full_mean": float(out["r_full_mean"]),
                    "r_red_mean": float(out["r_red_mean"]),
                    "is_responsive": bool(out["is_responsive"]),
                    "reduced_c2_p": float(rc2),
                    "reduced_is_responsive": bool(np.isfinite(rc2)
                                                  and rc2 < C2_THRESH),
                }
                _append_row(row)
                print(f"  [{sess_name}] unit {uid} ({k+1}/{len(todo)}): "
                      f"{int(y.sum())}spk FULL c2_p={out['c2_p']:.2e} "
                      f"c1_r={out['c1_r']:.3f} resp={out['is_responsive']} | "
                      f"reduced c2_p={rc2:.2e} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [{sess_name}] unit {uid}: FAILED ({type(e).__name__}: "
                      f"{e}) -- DROPPED", flush=True)
                n_fail += 1
                continue

        del ks, trials, units, design, Xr
        import gc
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.read_csv(OUT_CSV)
    df = df[df["region"] == REGION].copy()
    n = len(df)
    full_resp = df["c2_p"] < C2_THRESH
    red_resp = df["reduced_c2_p"] < C2_THRESH
    n_full = int(full_resp.sum())
    n_red = int(red_resp.sum())
    n_survive = int((red_resp & full_resp).sum())

    print("\n================= DECISIVE TEST: VISp =================")
    print(f"n units fit (full model): {n}  (dropped/failed: {n_fail})")
    print(f"FULL movement-controlled TF-responsive (c2_p<{C2_THRESH}): "
          f"{n_full}/{n} = {100.0*n_full/n:.1f}%")
    print(f"REDUCED-model TF-responsive (same units, from diagnostic): "
          f"{n_red}/{n} = {100.0*n_red/n:.1f}%")
    print(f"Reduced-flagged that SURVIVE movement control: "
          f"{n_survive}/{n_red}")
    print(f"\nPer-unit CSV: {OUT_CSV}")
    print("Reduced-flagged units (session, unit, reduced_c2_p -> full_c2_p, "
          "survives):")
    for _, r in df[red_resp].sort_values("reduced_c2_p").iterrows():
        surv = (r["c2_p"] < C2_THRESH)
        print(f"  {r['session']} u{int(r['unit'])}: "
              f"reduced {r['reduced_c2_p']:.2e} -> full {r['c2_p']:.2e}  "
              f"{'SURVIVES' if surv else 'collapses'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
