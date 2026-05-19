#!/usr/bin/env python3
"""Quick unit yield comparison: original vs concat-sort pkls.

Counts clusters, good clusters, and those passing the >=1 Hz rate filter
used by the analysis_suite. No heavy QC computation -- just loads and counts.

Usage:
    python scripts/pipelines/concat_sort/quick_yield_summary.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from visdetect.core.session import load_session

SUBJECT = "BG_046"
OLD_DIR = REPO_ROOT / "data" / "pkls" / SUBJECT
NEW_DIR = REPO_ROOT / "data" / "pkls" / f"{SUBJECT}_concat_sort"
MIN_RATE_HZ = 1.0


def count_units(session):
    """Return (total, n_good, n_good_rate1hz) mimicking get_good_cluster_ids logic."""
    total = len(session.clusters)

    # Good clusters (good_and_stable > good_cluster_ids > all)
    good_stable = getattr(session, "good_and_stable_ids", None) or []
    good_ids = session.good_cluster_ids if session.good_cluster_ids else []
    if good_stable:
        candidates = set(good_stable)
    elif good_ids:
        candidates = set(good_ids)
    else:
        candidates = {c.cluster_id for c in session.clusters}

    n_good = len(candidates)

    # Rate filter: >= 1 Hz
    # Compute recording duration from last spike time across all clusters
    max_t = max((c.spike_times[-1] if len(c.spike_times) > 0 else 0) for c in session.clusters)
    rec_dur = max_t if max_t > 0 else 1.0
    passed = 0
    for c in session.clusters:
        if c.cluster_id in candidates:
            rate = len(c.spike_times) / rec_dur if rec_dur > 0 else 0
            if rate >= MIN_RATE_HZ:
                passed += 1

    return total, n_good, passed


def main():
    # Find sessions with both pkls
    new_pkls = sorted(NEW_DIR.glob("*.pkl"))
    rows = []

    for new_pkl in new_pkls:
        name = new_pkl.stem
        old_pkl = OLD_DIR / new_pkl.name
        print(f"  {name} ...", end="", flush=True)

        # Load concat-sort
        ns = load_session(str(new_pkl))
        nt, ng, np_ = count_units(ns)

        # Load original if it exists
        if old_pkl.exists():
            os_ = load_session(str(old_pkl))
            ot, og, op = count_units(os_)
        else:
            ot, og, op = None, None, None

        rows.append({
            "session": name,
            "old_total": ot,
            "old_good": og,
            "old_pass_1hz": op,
            "new_total": nt,
            "new_good": ng,
            "new_pass_1hz": np_,
        })
        print(f"  old={op}  new={np_}")

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "FIGURES" / "concat_sort_qc" / "yield_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Print summary table
    print("\n" + "=" * 85)
    print(f"{'Session':<22} {'Old total':>10} {'Old good':>9} {'Old >=1Hz':>10}"
          f"  {'New total':>10} {'New good':>9} {'New >=1Hz':>10}")
    print("-" * 85)
    for _, r in df.iterrows():
        ot_s = str(r["old_total"]) if pd.notna(r["old_total"]) else "-"
        og_s = str(r["old_good"]) if pd.notna(r["old_good"]) else "-"
        op_s = str(r["old_pass_1hz"]) if pd.notna(r["old_pass_1hz"]) else "-"
        print(f"{r['session']:<22} {ot_s:>10} {og_s:>9} {op_s:>10}"
              f"  {int(r['new_total']):>10} {int(r['new_good']):>9} {int(r['new_pass_1hz']):>10}")

    # Aggregates
    has_both = df.dropna(subset=["old_pass_1hz"])
    if len(has_both) > 0:
        print("-" * 85)
        print(f"{'MEAN':<22} "
              f"{has_both['old_total'].mean():>10.0f} "
              f"{has_both['old_good'].mean():>9.0f} "
              f"{has_both['old_pass_1hz'].mean():>10.1f}"
              f"  {has_both['new_total'].mean():>10.0f} "
              f"{has_both['new_good'].mean():>9.0f} "
              f"{has_both['new_pass_1hz'].mean():>10.1f}")
        print(f"{'MEDIAN':<22} "
              f"{has_both['old_total'].median():>10.0f} "
              f"{has_both['old_good'].median():>9.0f} "
              f"{has_both['old_pass_1hz'].median():>10.1f}"
              f"  {has_both['new_total'].median():>10.0f} "
              f"{has_both['new_good'].median():>9.0f} "
              f"{has_both['new_pass_1hz'].median():>10.1f}")
        print(f"{'SUM':<22} "
              f"{has_both['old_total'].sum():>10.0f} "
              f"{has_both['old_good'].sum():>9.0f} "
              f"{has_both['old_pass_1hz'].sum():>10.0f}"
              f"  {has_both['new_total'].sum():>10.0f} "
              f"{has_both['new_good'].sum():>9.0f} "
              f"{has_both['new_pass_1hz'].sum():>10.0f}")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
