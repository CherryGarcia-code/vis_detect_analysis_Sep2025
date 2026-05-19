#!/usr/bin/env python3
"""Run UnitMatch across all BG_046 sessions in large overlapping batches.

The archived UnitMatch result used 10-session batches stitched into a 42-column
registry (UMparam n_sessions=10) on the outdated UnitMatchPy 2.41 -- five
batches, four stitch boundaries, fragmenting long tracks.

All 42 sessions in ONE batch does not fit in RAM: UnitMatchPy's
extract_metric_scores builds (n_timepoints, N, N) arrays -- ~7.6 GB each at
N=6679 units, several live at once -> >60 GB peak.

Compromise: a few LARGE overlapping batches (default 28 sessions, 14 overlap
=> 2 batches for 42 sessions, ONE boundary), then reconcile the per-batch
unique IDs via union-find over the shared (session, ks_unit_id) keys. The big
overlap makes reconciliation well-constrained.

MUST run under the unitmatch_env interpreter via conda run (MKL DLLs):
    C:/Users/Ben/anaconda3/Scripts/conda.exe run -n unitmatch_env \
        --no-capture-output python -u \
        scripts/pipelines/tracking/run_unitmatch_all.py

Output (data/unit_match/output/BG_046_all42/):
    cell_registry.csv  - global UID x session wide table (ks cluster ids)
    unit_index.csv     - (session, ks_unit_id) -> global_uid + batch info
    run_summary.json   - params + track-span stats (intermediate algorithm)
    batch{i}/          - per-batch unit_index + prob matrix
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as ov
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
DEFAULT_OUT = REPO_ROOT / "data" / "unit_match" / "output" / "BG_046_all42"

# old batched run (UnitMatchPy 2.41, 10-session batches) for side-by-side
OLD_BATCHED = {2: 17.4, 3: 8.3, 5: 3.4, 10: 1.0, 15: 0.4, 20: 0.1}


def parse_session_date(name: str) -> datetime:
    s = str(name)
    if len(s) == 7:
        s = "0" + s
    return datetime.strptime(s, "%d%m%Y")


def make_batches(n: int, batch_size: int, overlap: int):
    """Overlapping [start, end) index ranges covering range(n)."""
    if n <= batch_size:
        return [(0, n)]
    step = batch_size - overlap
    starts = list(range(0, n - overlap, step))
    if starts[-1] + batch_size < n:          # ensure the tail is covered
        starts.append(n - batch_size)
    return [(s, min(s + batch_size, n)) for s in starts]


def span_stats(spans: np.ndarray) -> dict:
    spans = np.asarray(spans)
    out = {"n_tracked_ids": int(len(spans)), "median_span": float(np.median(spans)),
           "mean_span": float(spans.mean()), "max_span": int(spans.max())}
    for thr in (2, 3, 5, 10, 15, 20):
        out[f"ge_{thr}"] = int((spans >= thr).sum())
    return out


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:        # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def run_one_batch(ks_dirs, session_names, batch_label, out_dir):
    """Canonical UnitMatchPy pipeline on one batch. Returns a per-unit DataFrame."""
    print(f"\n--- {batch_label}: {len(ks_dirs)} sessions "
          f"({session_names[0]}..{session_names[-1]}) ---", flush=True)
    param = default_params.get_default_param()
    param["KS_dirs"] = ks_dirs

    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(ks_dirs)
    param = util.get_probe_geometry(channel_pos[0], param)

    waveform, session_id, session_switch, within_session, good_units, param = \
        util.load_good_waveforms(wave_paths, unit_label_paths, param,
                                 good_units_only=True)
    n_units = int(param["n_units"])
    print(f"  n_units={n_units}  ((t,N,N) array ~"
          f"{23 * n_units**2 * 8 / 1e9:.1f} GB)", flush=True)

    clus_info = {"good_units": good_units, "session_switch": session_switch,
                 "session_id": session_id,
                 "original_ids": np.concatenate(good_units)}

    extracted = ov.extract_parameters(waveform, channel_pos, clus_info, param)
    del waveform
    total_score, candidate_pairs, scores_to_include, predictors = \
        ov.extract_metric_scores(extracted, session_switch, within_session,
                                 param, niter=2)
    prior_match = 1 - (param["n_expected_matches"] / n_units ** 2)
    priors = np.array((prior_match, 1 - prior_match))
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels,
                                                 cond, param, add_one=1)
    probability = bf.apply_naive_bayes(parameter_kernels, priors,
                                       predictors, param, cond)
    prob_matrix = probability[:, 1].reshape(n_units, n_units)

    uid_lists = aid.assign_unique_id(prob_matrix, param, clus_info)
    sess_idx = np.asarray(session_id).ravel().astype(int)
    ks_ids = np.asarray(clus_info["original_ids"]).ravel().astype(int)
    df = pd.DataFrame({
        "session": [session_names[s] for s in sess_idx],
        "ks_unit_id": ks_ids,
        "batch_uid": np.asarray(uid_lists[1]).ravel(),        # intermediate
        "batch_uid_liberal": np.asarray(uid_lists[0]).ravel(),
        "batch_uid_conservative": np.asarray(uid_lists[2]).ravel(),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "unit_index.csv", index=False)
    np.save(out_dir / "output_prob_matrix.npy", prob_matrix.astype(np.float32))
    return df


def reconcile(batch_dfs, uid_col="batch_uid"):
    """Union-find across batches: (session, ks_unit_id) keys sharing a
    within-batch UID are the same cell. Returns {(session, ks): global_uid}."""
    uf = UnionFind()
    for bi, df in enumerate(batch_dfs):
        for buid, grp in df.groupby(uid_col):
            keys = list(zip(grp["session"], grp["ks_unit_id"]))
            for k in keys:
                uf.union(keys[0], k)
    roots = {}
    mapping = {}
    for df in batch_dfs:
        for s, k in zip(df["session"], df["ks_unit_id"]):
            r = uf.find((s, k))
            if r not in roots:
                roots[r] = len(roots)
            mapping[(s, k)] = roots[r]
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Batched-overlapping UnitMatch over all sessions")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=28,
                    help="sessions per batch (memory-bounded; 28 ~= 27 GB peak)")
    ap.add_argument("--overlap", type=int, default=14,
                    help="overlapping sessions between consecutive batches")
    ap.add_argument("--n-sessions", type=int, default=None,
                    help="limit to first N sessions (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    sess_dirs = sorted([d for d in args.input.iterdir() if d.is_dir()],
                       key=lambda d: parse_session_date(d.name))
    if args.n_sessions:
        sess_dirs = sess_dirs[:args.n_sessions]
    session_names = [d.name for d in sess_dirs]
    n = len(sess_dirs)
    batches = make_batches(n, args.batch_size, args.overlap)
    print(f"UnitMatch over {n} sessions in {len(batches)} batch(es): "
          f"{[ (session_names[a], session_names[b-1]) for a, b in batches ]}",
          flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    batch_dfs = []
    for bi, (a, b) in enumerate(batches):
        ks = [str(d) for d in sess_dirs[a:b]]
        df = run_one_batch(ks, session_names[a:b], f"batch{bi}",
                           args.out_dir / f"batch{bi}")
        batch_dfs.append(df)

    # ── reconcile per-batch UIDs into global UIDs ───────────────────
    print("\nReconciling batch UIDs (union-find over shared sessions) ...",
          flush=True)
    global_map = reconcile(batch_dfs, "batch_uid")

    # one row per (session, ks_unit_id); batches agree on overlap so dedup
    rows = {}
    for df in batch_dfs:
        for s, k in zip(df["session"], df["ks_unit_id"]):
            rows[(s, k)] = {"session": s, "ks_unit_id": int(k),
                            "global_uid": global_map[(s, k)]}
    unit_index = pd.DataFrame(list(rows.values()))
    unit_index.to_csv(args.out_dir / "unit_index.csv", index=False)

    reg = unit_index.pivot_table(index="global_uid", columns="session",
                                 values="ks_unit_id", aggfunc="first")
    reg = reg.reindex(columns=session_names)
    reg.to_csv(args.out_dir / "cell_registry.csv")

    spans = reg.notna().sum(axis=1).values
    stats = span_stats(spans)
    elapsed = time.time() - t0

    summary = {"n_sessions": n, "n_batches": len(batches),
               "batch_size": args.batch_size, "overlap": args.overlap,
               "n_unit_session_entries": int(len(unit_index)),
               "unitmatchpy_version": "3.2.9",
               "elapsed_min": round(elapsed / 60, 1), "span_stats": stats}
    with open(args.out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── report ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print(f"UNITMATCH 3.2.9 ({len(batches)} overlapping batches, {n} sessions) "
          f"-- {elapsed/60:.1f} min")
    print("=" * 66)
    print(f"  unit-session entries: {len(unit_index)}   "
          f"global tracked IDs: {len(spans)}")
    print(f"  track span: median {stats['median_span']:.0f}, "
          f"mean {stats['mean_span']:.2f}, max {stats['max_span']}")
    print("  span distribution  vs OLD batched run (UnitMatchPy 2.41, 10-batches):")
    for thr in (2, 3, 5, 10, 15, 20):
        c = stats[f"ge_{thr}"]
        pct = 100 * c / len(spans) if len(spans) else 0
        print(f"    >= {thr:2d} sessions: {c:5d}  ({pct:5.1f}%)   "
              f"old: {OLD_BATCHED[thr]:.1f}%")
    print(f"\n  output -> {args.out_dir}")


if __name__ == "__main__":
    main()
