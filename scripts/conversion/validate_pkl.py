"""Validate a newly-generated .pkl against a reference .pkl.

Compares all fields (trials, clusters, spike times, NI events, metadata)
and reports PASS/FAIL per field with details on mismatches.

Usage:
    py scripts/conversion/validate_pkl.py \
        --new data/pkls/BG_046_new/BG_046_01072025.pkl \
        --reference data/pkls/BG_046/BG_046_01072025.pkl

    py scripts/conversion/validate_pkl.py \
        --new-dir data/pkls/BG_046_new \
        --ref-dir data/pkls/BG_046
"""

import argparse
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from visdetect.core.session import load_session


def compare_sessions(new_path: str, ref_path: str, atol: float = 1e-10) -> dict:
    """Compare two Session .pkl files field by field.

    Returns dict mapping field names to (passed: bool, detail: str).
    """
    results = {}
    try:
        new = load_session(new_path)
    except FileNotFoundError:
        return {"load_new": (False, f"File not found: {new_path}")}
    except Exception as e:
        return {"load_new": (False, f"Failed to load new .pkl ({new_path}): {e}")}

    try:
        ref = load_session(ref_path)
    except FileNotFoundError:
        return {"load_ref": (False, f"File not found: {ref_path}")}
    except Exception as e:
        return {"load_ref": (False, f"Failed to load reference .pkl ({ref_path}): {e}")}

    # ── Metadata ─────────────────────────────────────────────────
    for attr in ("subject", "session_name"):
        nv = getattr(new, attr)
        rv = getattr(ref, attr)
        ok = nv == rv
        results[attr] = (ok, f"new={nv!r} ref={rv!r}")

    # ── Trials ───────────────────────────────────────────────────
    n_new, n_ref = len(new.trials), len(ref.trials)
    results["trial_count"] = (n_new == n_ref, f"new={n_new} ref={n_ref}")

    if n_new == n_ref:
        trial_mismatches = []
        for i, (nt, rt) in enumerate(zip(new.trials, ref.trials)):
            issues = _compare_trial(nt, rt, i, atol)
            trial_mismatches.extend(issues)
        if trial_mismatches:
            detail = f"{len(trial_mismatches)} issue(s): " + "; ".join(trial_mismatches[:5])
            if len(trial_mismatches) > 5:
                detail += f" ... and {len(trial_mismatches) - 5} more"
            results["trial_fields"] = (False, detail)
        else:
            results["trial_fields"] = (True, "all trials match")

    # ── Cluster IDs ──────────────────────────────────────────────
    for attr in ("good_cluster_ids", "good_and_stable_ids"):
        nv = sorted(getattr(new, attr) or [])
        rv = sorted(getattr(ref, attr) or [])
        if nv == rv:
            results[attr] = (True, f"n={len(nv)}")
        else:
            only_new = set(nv) - set(rv)
            only_ref = set(rv) - set(nv)
            detail = f"new={len(nv)} ref={len(rv)}"
            if only_new:
                detail += f" only_in_new({len(only_new)})={sorted(only_new)[:10]}"
            if only_ref:
                detail += f" only_in_ref({len(only_ref)})={sorted(only_ref)[:10]}"
            results[attr] = (False, detail)

    # ── Clusters (spike times) ───────────────────────────────────
    new_cids = sorted(c.cluster_id for c in new.clusters)
    ref_cids = sorted(c.cluster_id for c in ref.clusters)
    results["cluster_count"] = (
        len(new_cids) == len(ref_cids),
        f"new={len(new_cids)} ref={len(ref_cids)}",
    )
    results["cluster_ids"] = (
        new_cids == ref_cids,
        f"match={new_cids == ref_cids}",
    )

    if new_cids == ref_cids:
        new_map = {c.cluster_id: c for c in new.clusters}
        ref_map = {c.cluster_id: c for c in ref.clusters}
        spike_issues = []
        for cid in new_cids:
            nc = new_map[cid]
            rc = ref_map[cid]
            n_new_sp = len(nc.spike_times)
            n_ref_sp = len(rc.spike_times)
            if n_new_sp != n_ref_sp:
                spike_issues.append(f"clu {cid}: n_spikes new={n_new_sp} ref={n_ref_sp}")
                continue
            if not np.allclose(nc.spike_times, rc.spike_times, atol=atol):
                max_diff = np.max(np.abs(nc.spike_times - rc.spike_times))
                spike_issues.append(f"clu {cid}: max_diff={max_diff:.2e}")
        if spike_issues:
            detail = f"{len(spike_issues)} cluster(s) differ: " + "; ".join(spike_issues[:5])
            results["spike_times"] = (False, detail)
        else:
            results["spike_times"] = (True, f"all {len(new_cids)} clusters match")

    # ── NI events ────────────────────────────────────────────────
    new_keys = set(new.ni_events.keys()) if new.ni_events else set()
    ref_keys = set(ref.ni_events.keys()) if ref.ni_events else set()
    results["ni_event_keys"] = (new_keys == ref_keys, f"new={sorted(new_keys)} ref={sorted(ref_keys)}")

    if new_keys == ref_keys and new.ni_events:
        ni_issues = []
        for k in sorted(new_keys):
            nv = new.ni_events[k]
            rv = ref.ni_events[k]
            issue = _compare_ni_value(k, nv, rv, atol)
            if issue:
                ni_issues.append(issue)
        if ni_issues:
            results["ni_event_values"] = (False, "; ".join(ni_issues[:5]))
        else:
            results["ni_event_values"] = (True, "all events match")

    return results


def _compare_trial(nt, rt, idx, atol):
    """Compare two Trial objects, return list of issue strings."""
    issues = []
    # Scalar fields
    for attr in ("trialoutcome", "change_size", "orientation", "ITI", "change_time"):
        nv = getattr(nt, attr)
        rv = getattr(rt, attr)
        if nv != rv:
            # Check for float near-equality
            try:
                if abs(float(nv) - float(rv)) < atol:
                    continue
            except (TypeError, ValueError):
                pass
            issues.append(f"trial[{idx}].{attr}: new={nv!r} ref={rv!r}")

    # Reaction times
    nrt = nt.reactiontimes or {}
    rrt = rt.reactiontimes or {}
    if set(nrt.keys()) != set(rrt.keys()):
        issues.append(f"trial[{idx}].reactiontimes keys differ")
    else:
        for k in nrt:
            nv, rv = nrt[k], rrt[k]
            try:
                nv_f, rv_f = float(nv), float(rv)
                if np.isnan(nv_f) and np.isnan(rv_f):
                    continue
                if abs(nv_f - rv_f) > atol:
                    issues.append(f"trial[{idx}].reactiontimes[{k}]: {nv} vs {rv}")
            except (TypeError, ValueError):
                if nv != rv:
                    issues.append(f"trial[{idx}].reactiontimes[{k}]: {nv!r} vs {rv!r}")

    # Baseline values
    nbv = nt.baseline_values
    rbv = rt.baseline_values
    if nbv is not None and rbv is not None:
        nbv = np.asarray(nbv).flatten()
        rbv = np.asarray(rbv).flatten()
        if nbv.shape != rbv.shape:
            issues.append(f"trial[{idx}].baseline_values shape: {nbv.shape} vs {rbv.shape}")
        elif not np.allclose(nbv, rbv, atol=atol):
            max_diff = np.max(np.abs(nbv - rbv))
            issues.append(f"trial[{idx}].baseline_values max_diff={max_diff:.2e}")
    elif (nbv is None) != (rbv is None):
        issues.append(f"trial[{idx}].baseline_values: one is None")

    return issues


def _compare_ni_value(key, nv, rv, atol):
    """Compare a single NI event value, return issue string or None."""
    if isinstance(nv, np.ndarray) and isinstance(rv, np.ndarray):
        if nv.dtype == object or rv.dtype == object:
            # Object arrays (session_name, frame_times_tr) — check string content
            if nv.size == rv.size:
                for i in range(nv.size):
                    if isinstance(nv.flat[i], str) and isinstance(rv.flat[i], str):
                        if nv.flat[i] != rv.flat[i]:
                            return f"{key}[{i}]: {nv.flat[i]!r} vs {rv.flat[i]!r}"
                    elif isinstance(nv.flat[i], dict) and isinstance(rv.flat[i], dict):
                        # frame_times_tr dicts — skip deep comparison for now
                        pass
                return None
            return f"{key}: object array size {nv.size} vs {rv.size}"

        if nv.shape != rv.shape:
            return f"{key}: shape {nv.shape} vs {rv.shape}"
        if not np.allclose(nv, rv, atol=atol, equal_nan=True):
            mask = ~(np.isnan(nv) & np.isnan(rv))
            if mask.any():
                max_diff = np.max(np.abs(nv[mask] - rv[mask]))
                return f"{key}: max_diff={max_diff:.2e}"
        return None

    if isinstance(nv, str) and isinstance(rv, str):
        return None if nv == rv else f"{key}: {nv!r} vs {rv!r}"

    return f"{key}: type mismatch {type(nv).__name__} vs {type(rv).__name__}"


def print_results(results: dict, label: str = ""):
    """Print validation results with PASS/FAIL formatting."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    all_pass = True
    for field, (passed, detail) in results.items():
        status = "PASS" if passed else "FAIL"
        marker = "  " if passed else "!!"
        print(f"  {marker} [{status}] {field}: {detail}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ** ALL CHECKS PASSED **")
    else:
        n_fail = sum(1 for _, (p, _) in results.items() if not p)
        print(f"  ** {n_fail} CHECK(S) FAILED **")
    print()
    return all_pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate new .pkl against reference .pkl"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--new", type=Path,
        help="Path to single new .pkl file",
    )
    group.add_argument(
        "--new-dir", type=Path,
        help="Directory of new .pkl files (batch mode)",
    )
    parser.add_argument(
        "--reference", type=Path,
        help="Path to single reference .pkl file (used with --new)",
    )
    parser.add_argument(
        "--ref-dir", type=Path,
        help="Directory of reference .pkl files (used with --new-dir)",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-10,
        help="Absolute tolerance for float comparison (default: 1e-10)",
    )

    args = parser.parse_args(argv)

    if args.new:
        # Single file mode
        ref = args.reference
        if ref is None:
            parser.error("--reference is required with --new")
        results = compare_sessions(str(args.new), str(ref), atol=args.atol)
        ok = print_results(results, label=args.new.name)
        return 0 if ok else 1

    else:
        # Batch mode
        ref_dir = args.ref_dir
        if ref_dir is None:
            parser.error("--ref-dir is required with --new-dir")

        new_files = sorted(args.new_dir.glob("*.pkl"))
        if not new_files:
            print(f"No .pkl files found in {args.new_dir}")
            return 1

        total_pass = 0
        total_fail = 0
        for new_path in new_files:
            ref_path = ref_dir / new_path.name
            if not ref_path.exists():
                print(f"\n  SKIP: {new_path.name} — no reference file")
                continue
            results = compare_sessions(str(new_path), str(ref_path), atol=args.atol)
            ok = print_results(results, label=new_path.name)
            if ok:
                total_pass += 1
            else:
                total_fail += 1

        print(f"\n{'=' * 60}")
        print(f"  BATCH SUMMARY: {total_pass} passed, {total_fail} failed")
        print(f"{'=' * 60}")
        return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
