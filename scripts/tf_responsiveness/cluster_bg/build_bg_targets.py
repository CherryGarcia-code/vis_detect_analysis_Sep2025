"""Enumerate (subject, session, chunk) array-task targets for the BG-mouse
TF-GLM sweep. Cheap: globs pkl FILENAMES only (no loading) — the worker
self-partitions a session's units by stride after loading it.

Each session becomes ``--chunks`` array tasks; task t fits ``units[t::chunks]``
(units sorted slowest-first inside the worker), bounding per-task runtime to
roughly ``n_units/chunks`` fits. Sessions with fewer units than ``--chunks``
simply leave the high-index tasks empty (they exit immediately).

Session id is the pkl STEM kept verbatim as a string — never int-cast — so the
leading-zero day, 6-digit dates, and suffixed names (BG_012's
``..._prot4_lickEndsTrial``) all survive intact. Downstream JOINS to
manifest/anatomy/celltype tables must still canonicalize via
``config.canonical_session_id`` (out of scope for this standalone sweep).

Writes ``task_id, subject, session, pkl_rel, chunk_idx, n_chunks`` and prints
the ``sbatch --array=1-N`` range.

Examples
--------
All subjects, 8 chunks/session::

    py build_bg_targets.py --pkl-root "X:/.../tf_glm_cluster/bg_pkls" \
        --chunks 8 --out targets_bg.csv

Just the striatal mice::

    py build_bg_targets.py --pkl-root <root> --subjects BG_046,BG_031,BG_039 \
        --chunks 8 --out targets_bg_striatum.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl-root", required=True,
                   help="root holding <SUBJECT>/<session>.pkl")
    p.add_argument("--subjects", default=None,
                   help="comma-sep subject whitelist (default: all under root)")
    p.add_argument("--chunks", type=int, default=8,
                   help="array tasks per session (units strided across them)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    root = Path(a.pkl_root)
    subjects = ({s.strip() for s in a.subjects.split(",") if s.strip()}
                if a.subjects else None)

    rows, tid = [], 1
    subj_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    for sd in subj_dirs:
        subj = sd.name
        if subjects is not None and subj not in subjects:
            continue
        for pkl in sorted(sd.glob("*.pkl")):
            session = pkl.stem  # verbatim string id
            pkl_rel = f"{subj}/{pkl.name}"
            for ci in range(a.chunks):
                rows.append(dict(task_id=tid, subject=subj, session=session,
                                 pkl_rel=pkl_rel, chunk_idx=ci, n_chunks=a.chunks))
                tid += 1

    df = pd.DataFrame(rows, columns=["task_id", "subject", "session", "pkl_rel",
                                     "chunk_idx", "n_chunks"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    n_sess = df["session"].nunique() if len(df) else 0
    per_subj = (df.groupby("subject")["session"].nunique().to_dict()
                if len(df) else {})
    print(f"Wrote {a.out}")
    print(f"  {len(df)} tasks | {n_sess} sessions | {a.chunks} chunks/session")
    print(f"  sessions/subject: {per_subj}")
    if len(df):
        print(f"  -> sbatch array range:  --array=1-{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
