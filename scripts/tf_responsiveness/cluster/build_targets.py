"""Enumerate (session, region, unit-chunk) array-task targets for the cluster
TF-GLM replication, from the Khilkevich ``npx_converted`` ``clusters.csv`` files.

Cheap: reads only ``clusters.csv`` per session (no spike loading). The per-unit
``>=500``-spike gate is applied later by the worker at fit time, so a chunk may
contain a few sub-threshold units that the worker simply skips.

Modes
-----
  --sessions a/b,c/d     explicit ``session_rel`` list (relative to --scan-root)
  (no --sessions)        walk every animal/session dir under --scan-root

Region filtering keys off the coarse ``brain_region_comb`` label (so ``VISp``
covers VISp1/VISp5..., ``CP`` covers CP), matching how
``khilkevich_trial_regressors`` resolves ``region=``.

Writes a targets CSV with columns
``task_id, session_rel, region, n_units, unit_ids`` (``unit_ids`` ';'-joined).
One row per array task; prints the ``sbatch --array=1-N`` range to use.

Examples
--------
Decisive VISp+CP arbiter (the two dual-probe 1116764 sessions)::

    py build_targets.py \
        --scan-root "X:/.../npx_converted" \
        --sessions 1116764/ML_1116764_S02_M2_V1,1116764/ML_1116764_S03_M2_V1 \
        --regions VISp,CP --chunk 15 --out targets_decisive.csv

Brain-wide sweep, all coarse regions::

    py build_targets.py --scan-root "X:/.../npx_converted" \
        --regions all --chunk 15 --out targets_brainwide.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

REGION_COL = "brain_region_comb"
UNIT_COL = "cluster_id"


def session_region_units(session_dir: Path, regions):
    """{coarse_region: [unit_ids]} for one session, filtered to `regions`."""
    cl = session_dir / "clusters.csv"
    if not cl.exists():
        return {}
    d = pd.read_csv(cl)
    if REGION_COL not in d.columns or UNIT_COL not in d.columns:
        return {}
    out = {}
    for reg, g in d.groupby(REGION_COL):
        reg = str(reg)
        if regions is not None and reg not in regions:
            continue
        out[reg] = sorted(int(u) for u in g[UNIT_COL].unique())
    return out


def iter_sessions(scan_root: Path, animals=None, sessions=None):
    """Yield (session_rel, session_dir). Explicit `sessions` overrides the walk."""
    root = Path(scan_root)
    if sessions:
        for s in sessions:
            yield s, root / s
        return
    for animal_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if animals is not None and animal_dir.name not in animals:
            continue
        for sess_dir in sorted(p for p in animal_dir.iterdir() if p.is_dir()):
            if (sess_dir / "clusters.csv").exists():
                yield f"{animal_dir.name}/{sess_dir.name}", sess_dir


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan-root", required=True)
    p.add_argument("--sessions", default=None,
                   help="comma-sep session_rel list (e.g. 1116764/ML_..._S02_M2_V1)")
    p.add_argument("--animals", default=None,
                   help="comma-sep animal-id whitelist (walk mode only)")
    p.add_argument("--regions", default="VISp,CP",
                   help="comma-sep coarse regions, or 'all'")
    p.add_argument("--chunk", type=int, default=15,
                   help="units per array task (bounds per-task runtime)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    regions = (None if a.regions.strip().lower() == "all"
               else {r.strip() for r in a.regions.split(",") if r.strip()})
    sessions = ([s.strip() for s in a.sessions.split(",") if s.strip()]
                if a.sessions else None)
    animals = ({x.strip() for x in a.animals.split(",") if x.strip()}
               if a.animals else None)

    rows, tid = [], 1
    for session_rel, sess_dir in iter_sessions(Path(a.scan_root), animals, sessions):
        for reg, uids in sorted(session_region_units(sess_dir, regions).items()):
            for ch in chunks(uids, a.chunk):
                rows.append(dict(task_id=tid, session_rel=session_rel,
                                 region=reg, n_units=len(ch),
                                 unit_ids=";".join(str(u) for u in ch)))
                tid += 1

    df = pd.DataFrame(rows, columns=["task_id", "session_rel", "region",
                                     "n_units", "unit_ids"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    n_units = int(df["n_units"].sum()) if len(df) else 0
    n_sess = df["session_rel"].nunique() if len(df) else 0
    regs = sorted(df["region"].unique()) if len(df) else []
    print(f"Wrote {a.out}")
    print(f"  {len(df)} tasks | {n_units} units | {n_sess} sessions | regions={regs}")
    if len(df):
        per_reg = df.groupby("region")["n_units"].sum().to_dict()
        print(f"  units/region: {per_reg}")
        print(f"  -> sbatch array range:  --array=1-{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
