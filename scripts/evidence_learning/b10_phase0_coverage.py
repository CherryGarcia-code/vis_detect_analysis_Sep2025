"""B10 Phase 0 — coverage / usable gate per subject x stage.

Plain English: before measuring anything, count how many impulsive (FA) licks
are usable for a kernel (enough pre-lick history + a matched no-lick control)
and how many TF-responsive cells exist, and flag which cells are worth analysing.

Run: py scripts/evidence_learning/b10_phase0_coverage.py
Out: data/cache/evidence_learning/b10_coverage.csv
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import pandas as pd
from visdetect.analysis import psychophysical_kernel as pk
from visdetect.analysis.evidence_learning_io import (
    SUBJECTS, CACHE_DIR, subject_sessions, tf_responsive_units)

MIN_FA = 30          # usable threshold (spec §6, formalized)


def coverage_row(subject, stage, skey, session, tf_by_key=None):
    eps = pk.fa_kernel_epochs(session)
    wh = pk.withhold_epochs(session, eps)
    n_ok = sum(1 for w in wh if w is not None)
    n_tf = len((tf_by_key or {}).get(skey, {}))
    return {"subject": subject, "stage": stage, "skey": str(skey),
            "n_fa_usable": len(eps), "n_withhold_ok": n_ok, "n_tf_units": n_tf,
            "usable": len(eps) >= MIN_FA and n_ok >= MIN_FA}


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = []
    for subject in SUBJECTS:
        tf = tf_responsive_units(subject)
        for skey, sname, stage, sess in subject_sessions(subject):
            rows.append(coverage_row(subject, stage, skey, sess, tf))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CACHE_DIR, "b10_coverage.csv"), index=False)
    print(df.groupby(["subject", "stage"])[
        ["n_fa_usable", "n_withhold_ok", "n_tf_units"]].sum())


if __name__ == "__main__":
    main()
