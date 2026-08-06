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

*** WHAT neural_safe=True DOES AND DOES NOT MEAN — READ BEFORE CONSUMING ***
It means: a VERIFIED pairing exists AT THE RECORDED OFFSETS, i.e. trial
`trial_start + k` corresponds to event `event_offset + k`, for k in
[0, n_trials_matched).
It does NOT mean trial i corresponds to event i. In the 2026-08-05 run, 12 of the
212 aligned rows pair at a NON-ZERO offset (9 with trial_start>0, 3 with
event_offset>0). And even at offset 0, `n_trials_matched` can be far smaller than
`n_trials`: BG_038_08082025 has 2046 trials vs 850 events at trial_start=0,
event_offset=0 -- a verified pairing in which 1196 trials (58%) have NO ephys event.
=> Consumers MUST map through `visdetect.core.run_alignment.build_trial_event_index`
   (which yields -1 for trials with no event) and MUST NOT index ni_events arrays
   with the raw trial index. Filtering on `neural_safe==True` and then doing
   `bon[i]` produces silently wrong PETHs.

The old count proxy is RETAINED alongside it as `count_safe` (downstream code and the QC1
spec reference the distinction):
  count_safe  = |n_baseline_on - n_trials| <= TOL_BENIGN
A small count excess was previously ASSUMED benign; the measured check tests it directly, so
`count_safe` and `neural_safe` can disagree in either direction. Trust `neural_safe`.

Every row carries EVIDENCE, including failures. On a rejected row (`aligned=False`) the
measured columns describe the BEST REJECTED CANDIDATE (from `best_candidate`) -- they are
the reason it failed, never a usable pairing -- and `reject_reason` names the exit.

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
from visdetect.core.run_alignment import best_candidate, rejection_reason, solve_alignment

TOL_BENIGN = 9          # |diff| <= this is treated as a benign end-of-session artifact

# Columns produced by the MEASURED alignment solver (as opposed to the count proxy),
# with the value each takes when a pkl cannot even be loaded. audit_pkl fills all of
# them on every row, so a failed row still carries its evidence.
MEASURED_DEFAULTS = {
    "agreement": np.nan,
    "median_resid_s": np.nan,
    "resid_n": 0,
    "runner_up_resid_s": np.nan,
    "aligned": False,
    "trial_start": -1,
    "event_offset": -1,
    "n_trials_matched": 0,     # NOT n_trials: trials outside the matched block have no event
    "reject_reason": "",
}
MEASURED_COLUMNS = tuple(MEASURED_DEFAULTS)

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
        # On a REJECTED pkl, report the best rejected candidate so the row carries the
        # evidence for WHY it failed (agreement 0.55 = badly misaligned is a different
        # finding from 0.998 = one anomalous trial). ev is a hypothesis, not a pairing.
        ev = a if a is not None else best_candidate(s.trials, s.ni_events)
        measured = dict(MEASURED_DEFAULTS)
        if ev is not None:
            measured.update({
                "agreement": ev.agreement,
                "median_resid_s": ev.resid_s,
                "resid_n": ev.resid_n,
                "runner_up_resid_s": ev.runner_up_resid_s,
                "trial_start": ev.trial_start,
                "event_offset": ev.event_offset,
                "n_trials_matched": ev.n_trials_matched,
            })
        measured["aligned"] = a is not None
        measured["reject_reason"] = (
            "" if a is not None else rejection_reason(s.trials, s.ni_events, best=ev)
        )
        return {"n_trials": n, "n_baseline_on": int(len(bon)), "diff": int(len(bon) - n),
                "ephys_s": round(max(spikes), 1) if spikes else np.nan,
                "bon_last": round(float(bon.max()), 1) if len(bon) else np.nan,
                **measured}
    finally:
        del s


def derive_verdicts(df):
    """Add `match`, `count_safe` (old count proxy) and `neural_safe` (MEASURED).

    `neural_safe` comes from `aligned` alone -- never from `diff`. A row with diff==0
    but aligned=False is NOT neural-safe; that divergence is the whole point of the
    measured check.
    """
    df["match"] = df["diff"] == 0
    df["count_safe"] = df["diff"].abs() <= TOL_BENIGN     # old proxy, retained
    df["neural_safe"] = df["aligned"].fillna(False).astype(bool)   # measured
    return df


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
                # Measured defaults MUST be included: if EVERY pkl fails, the DataFrame
                # would otherwise have no 'aligned' column and derive_verdicts would
                # KeyError instead of writing a CSV documenting the failures.
                rec.update({"n_trials": -1, "n_baseline_on": -1, "diff": np.nan,
                            "ephys_s": np.nan, "bon_last": np.nan,
                            **MEASURED_DEFAULTS, "reject_reason": "load_error",
                            "error": f"{type(exc).__name__}: {exc}"})
            rows.append(rec)
            print(f"  {rec['subject']} {rec['file']}: trials={rec['n_trials']} "
                  f"bon={rec['n_baseline_on']} diff={rec['diff']}")

    df = derive_verdicts(pd.DataFrame(rows))
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
        print("  (agreement/median_resid_s describe the BEST REJECTED candidate)")
        print(bad[["subject", "file", "n_trials", "n_baseline_on", "diff", "ephys_s",
                   "reject_reason", "agreement", "median_resid_s", "resid_n"]]
              .to_string(index=False))
        print("\nreject_reason tally:")
        print(bad["reject_reason"].value_counts().to_string())
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
