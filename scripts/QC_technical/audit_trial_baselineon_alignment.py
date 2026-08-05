"""QC: does each pkl's TRIAL TABLE match its ephys Baseline_ON events?

Found 2026-08-03 while investigating same-day re-recordings. Some recording days were
RESTARTED after a problem, producing two ephys files; the converter attached the FULL
day's behavioural table to EACH of them. Signature: n_trials >> n(Baseline_ON), and the
behavioural duration implied by the trial table exceeds the ephys length.

Worked example (BG_031 19052025): both pkls carry the same 569 trials, but the plain file
has 231 Baseline_ON events and the `_b` file 339 — and 231 + 339 = 570 ~= 569. The day was
split across two recordings; neither pkl is internally aligned.

CONSEQUENCE — scope carefully:
  * BEHAVIOUR-only analyses (trial outcomes, RTs, change_size, state tags, session sorting)
    read the TRIAL TABLE only and are UNAFFECTED.
  * NEURAL analyses that align to ni_events (Baseline_ON / Change_ON, PETHs, population
    tensors) are INVALID on affected sessions — trial i does not correspond to event i.

Emits `neural_safe` so downstream neural code can filter. As of the QC1 repair this is
the MEASURED verdict from `solve_alignment` (trial<->event pairing actually verified against
outcome/change-presence and change-time residuals), NOT a count heuristic:
  neural_safe = a solve_alignment() Alignment was found for the pkl
The old count proxy is RETAINED alongside it as `count_safe` (downstream code and the QC1
spec reference the distinction):
  count_safe  = |n_baseline_on - n_trials| <= TOL_BENIGN
A small count excess was previously ASSUMED benign; the measured check tests it directly, so
`count_safe` and `neural_safe` can disagree in either direction. Trust `neural_safe`.

Run: py scripts/QC_technical/audit_trial_baselineon_alignment.py [--subjects BG_046 ...]
Out: data/cache/qc_alignment/trial_vs_baselineon_audit.csv
"""
import os
import sys
import glob
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd

from visdetect.core.session import load_session
from visdetect.core.run_alignment import solve_alignment

TOL_BENIGN = 9          # |diff| <= this is treated as a benign end-of-session artifact

# Columns produced by the MEASURED alignment solver (as opposed to the count proxy).
MEASURED_COLUMNS = ("agreement", "median_resid_s", "resid_n", "runner_up_resid_s", "aligned")

OUT_DIR = os.path.join(_ROOT, "data", "cache", "qc_alignment")
OUT_CSV = os.path.join(OUT_DIR, "trial_vs_baselineon_audit.csv")


def audit_pkl(path):
    s = load_session(path)
    try:
        trials = s.trials or []
        n = len(trials)
        bon = np.asarray((s.ni_events or {}).get("Baseline_ON", []), dtype=float).ravel()
        spikes = [float(np.max(c.spike_times)) for c in (s.clusters or [])[:200]
                  if getattr(c, "spike_times", None) is not None and len(c.spike_times)]
        a = solve_alignment(s.trials, s.ni_events)
        measured = {
            "agreement": a.agreement if a else float("nan"),
            "median_resid_s": a.resid_s if a else float("nan"),
            "resid_n": a.resid_n if a else 0,
            "runner_up_resid_s": a.runner_up_resid_s if a else float("nan"),
            "aligned": a is not None,
            "trial_start": a.trial_start if a else -1,
            "event_offset": a.event_offset if a else -1,
        }
        return {"n_trials": n, "n_baseline_on": int(len(bon)), "diff": int(len(bon) - n),
                "ephys_s": round(max(spikes), 1) if spikes else np.nan,
                "bon_last": round(float(bon.max()), 1) if len(bon) else np.nan,
                **measured}
    finally:
        del s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    pkl_root = os.path.join(_ROOT, "data", "pkls")
    subjects = args.subjects or sorted(
        d for d in os.listdir(pkl_root) if os.path.isdir(os.path.join(pkl_root, d)))

    rows = []
    for subj in subjects:
        for p in sorted(glob.glob(os.path.join(pkl_root, subj, f"{subj}_*.pkl"))):
            rec = {"subject": subj, "file": os.path.basename(p)}
            try:
                rec.update(audit_pkl(p))
            except Exception as exc:                       # a load failure is itself a finding
                rec.update({"n_trials": -1, "n_baseline_on": -1, "diff": np.nan,
                            "ephys_s": np.nan, "bon_last": np.nan,
                            "error": f"{type(exc).__name__}: {exc}"})
            rows.append(rec)
            print(f"  {rec['subject']} {rec['file']}: trials={rec['n_trials']} "
                  f"bon={rec['n_baseline_on']} diff={rec['diff']}")

    df = pd.DataFrame(rows)
    df["match"] = df["diff"] == 0
    df["count_safe"] = df["diff"].abs() <= TOL_BENIGN     # old proxy, retained
    df["neural_safe"] = df["aligned"].fillna(False).astype(bool)   # measured
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\naudited {len(df)} pkls | exact match {int(df['match'].sum())} | "
          f"count_safe {int(df['count_safe'].sum())} | "
          f"neural_safe (measured) {int(df['neural_safe'].sum())} | "
          f"NOT safe {int((~df['neural_safe']).sum())}")

    # The scientific payoff: sessions the measured check and the count proxy disagree on.
    disagree = df[df["count_safe"].fillna(False).astype(bool) != df["neural_safe"]]
    if len(disagree):
        print(f"\nDISAGREEMENT count_safe vs neural_safe ({len(disagree)}):")
        print(disagree[["subject", "file", "n_trials", "n_baseline_on", "diff",
                        "count_safe", "neural_safe", "agreement", "median_resid_s"]]
              .to_string(index=False))
    bad = df[~df["neural_safe"]].sort_values("diff")
    if len(bad):
        print("\nNEURAL-UNSAFE (do NOT use for ni_events-aligned analyses):")
        print(bad[["subject", "file", "n_trials", "n_baseline_on", "diff", "ephys_s"]]
              .to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
