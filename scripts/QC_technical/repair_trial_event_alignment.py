"""QC1: write a verified trial->event index map into each pkl.

Never truncates the trial table: trials with no ephys event get -1, so
behaviour-only analyses keep every trial while neural code hard-skips them.
Always backs up before mutating.

Run: py scripts/QC_technical/repair_trial_event_alignment.py --subjects BG_046
Out: data/cache/qc_alignment/alignment_repair_report.csv
"""
import argparse
import gc
import glob
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd

from visdetect.core.run_alignment import build_trial_event_index, solve_alignment
from visdetect.core.session import load_session

OUT_DIR = os.path.join(_ROOT, "data", "cache", "qc_alignment")
OUT_CSV = os.path.join(OUT_DIR, "alignment_repair_report.csv")


def backup_pkl(path: str) -> str:
    """Copy the pkl into <dir>/qc1_backup/ with a UTC stamp. Raises on failure."""
    d = os.path.join(os.path.dirname(path), "qc1_backup")
    os.makedirs(d, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(d, f"{os.path.basename(path)}.bak_{stamp}")
    shutil.copy2(path, dest)
    if not os.path.exists(dest):
        raise IOError(f"backup failed for {path}")
    return dest


def repair_session(path: str, dry_run: bool = False) -> dict:
    s = load_session(path)
    try:
        n_tr = len(s.trials or [])
        a = solve_alignment(s.trials, s.ni_events)
        row = {
            "file": os.path.basename(path),
            "n_trials": n_tr,
            "solved": a is not None,
            "trial_start": a.trial_start if a else -1,
            "n_matched": a.n_trials_matched if a else 0,
            "event_offset": a.event_offset if a else -1,
            "agreement": a.agreement if a else float("nan"),
            "resid_s": a.resid_s if a else float("nan"),
            "resid_n": a.resid_n if a else 0,
            "runner_up_agreement": a.runner_up_agreement if a else float("nan"),
            "runner_up_resid_s": a.runner_up_resid_s if a else float("nan"),
            "n_no_ephys": 0,
        }
        idx = build_trial_event_index(n_tr, a)
        row["n_no_ephys"] = int((idx == -1).sum())
        if dry_run:
            return row

        outcomes_before = [getattr(t, "trialoutcome", None) for t in (s.trials or [])]
        backup_pkl(path)
        s.trial_event_index = idx
        with open(path, "wb") as f:
            pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)

        # behaviour must be byte-identical, and the map must have landed.
        # Use explicit raises, NOT assert: `py -O` strips asserts, and this is
        # the integrity gate on an irreplaceable-data mutation.
        chk = load_session(path)
        try:
            outcomes_after = [getattr(t, "trialoutcome", None) for t in (chk.trials or [])]
            if outcomes_after != outcomes_before:
                raise RuntimeError(
                    f"{path}: REPAIR CORRUPTED BEHAVIOUR — trial outcomes changed "
                    f"({len(outcomes_before)} before, {len(outcomes_after)} after). "
                    f"Restore from the backup in qc1_backup/."
                )
            written = getattr(chk, "trial_event_index", None)
            if written is None or not np.array_equal(np.asarray(written, dtype=int), idx):
                raise RuntimeError(
                    f"{path}: trial_event_index did not round-trip through the write. "
                    f"Restore from the backup in qc1_backup/."
                )
        finally:
            del chk
        return row
    finally:
        del s
        gc.collect()


# Reference alignments, measured by hand against the X: raw source (spec §2).
# These are the SAME triples asserted by tests/test_run_alignment_realdata.py.
_GATE_CASES = [
    ("BG_046", "BG_046_19082025.pkl",   0,   0,   587),
    ("BG_046", "BG_046_20082025.pkl",   0,   228, 486),
    ("BG_046", "BG_046_05092025_b.pkl", 281, 0,   248),
]


def verify_realdata_gate() -> None:
    """Abort unless the solver still reproduces the three measured alignments.

    The pytest gate (tests/test_run_alignment_realdata.py) SKIPS when the pkls
    are absent, so a green test run does not prove the gate ever executed. This
    script mutates pkls, so it re-checks in-process and refuses to run
    otherwise. main() only -- repair_session() is called directly by unit tests
    with synthetic temp pkls and must not trigger this.
    """
    for subj, fname, exp_start, exp_off, exp_n in _GATE_CASES:
        path = os.path.join(_ROOT, "data", "pkls", subj, fname)
        if not os.path.exists(path):
            raise SystemExit(
                f"REFUSING TO RUN: reference session missing: {path}\n"
                f"  The real-data gate cannot be verified, so the repair is not safe to run."
            )
        s = load_session(path)
        try:
            a = solve_alignment(s.trials, s.ni_events)
        finally:
            del s
            gc.collect()
        got = None if a is None else (a.trial_start, a.event_offset, a.n_trials_matched)
        if got != (exp_start, exp_off, exp_n):
            raise SystemExit(
                f"REFUSING TO RUN: solver no longer reproduces {fname}.\n"
                f"  expected (trial_start, event_offset, n_matched) = "
                f"{(exp_start, exp_off, exp_n)}\n  got = {got}\n"
                f"  Fix the solver before repairing any pkl."
            )
    print("real-data gate OK: 3/3 reference alignments reproduced")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=["BG_046"])
    ap.add_argument("--files", nargs="*", default=None, help="explicit pkl basenames")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    verify_realdata_gate()      # refuses to proceed if the solver regressed

    rows = []
    for subj in args.subjects:
        for p in sorted(glob.glob(os.path.join(_ROOT, "data", "pkls", subj, f"{subj}_*.pkl"))):
            if args.files and os.path.basename(p) not in args.files:
                continue
            rec = {"subject": subj}
            rec.update(repair_session(p, dry_run=args.dry_run))
            rows.append(rec)
            print(f"  {subj} {rec['file']}: solved={rec['solved']} "
                  f"start={rec['trial_start']} off={rec['event_offset']} "
                  f"agr={rec['agreement']:.4f} resid={rec['resid_s']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.dry_run:
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved: {OUT_CSV}")
    print(f"solved {int(df['solved'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
