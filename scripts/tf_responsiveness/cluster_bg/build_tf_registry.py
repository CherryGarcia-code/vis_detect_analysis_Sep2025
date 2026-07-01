"""Build a clean per-subject TF-responsive registry from a cluster results dir,
for downstream use. Writes the tidy per-unit cell-identity table (drops internal
task_id / r_full / r_red), adds `session_date` (subject-stripped) + `region` +
`region_bank_confirmed=False` (pooling gate). CSV-only (fast; no PETH npz).

Usage:
  py build_tf_registry.py --results <results_bg_SUBJ> --subject BG_039 \
     --region DMS --out data/cache/tf_responsive/bg039_tf_responsive.csv
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_master(results_dir):
    csvs = [f for f in glob.glob(str(Path(results_dir) / "task_*.csv")) if "_peth" not in f]
    if not csvs:
        raise SystemExit(f"no task_*.csv in {results_dir}")
    m = pd.concat([pd.read_csv(f, dtype={"session": str, "subject": str}) for f in csvs],
                  ignore_index=True)
    for c in ("resp_log2", "resp_lin"):
        if c in m.columns:
            m[c] = m[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    return m.drop_duplicates(subset=["session", "unit"], keep="last")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--region", required=True, help="e.g. DMS / VMS")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    m = load_master(a.results)
    m["session_date"] = [s.replace(f"{subj}_", "", 1) for s, subj in zip(m.session, m.subject)]
    m["region"] = a.region
    m["region_bank_confirmed"] = False
    cols = ["subject", "session", "session_date", "region", "region_bank_confirmed",
            "unit", "n_spikes", "resp_log2", "c1_r_log2", "c2_p_log2",
            "kernel_peak_t", "kernel_fwhm", "resp_lin", "c1_r_lin"]
    reg = (m[[c for c in cols if c in m.columns]]
           .sort_values(["session", "unit"]).reset_index(drop=True))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    print(f"  {len(reg)} units | {reg.session.nunique()} sessions | "
          f"{int(reg.resp_log2.sum())} TF-responsive ({100*reg.resp_log2.mean():.1f}%) "
          f"| region={a.region}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
