#!/usr/bin/env python3
"""
Scan a directory tree and report which folders contain (or look like they contain)
completed Kilosort sessions.

Usage examples:
  python scripts/scan_kilosort_sessions.py "X:\\public\\...\\Processed data" --json scripts/kilosort_report.json

Outputs:
  - JSON or CSV listing folders and flags:
    path, has_rez, has_spike_npy, has_phy_clusters, has_phy_dir, has_chanmap, has_bin, status, notes

This script is conservative and configurable; edit the patterns near the top if your lab uses different filenames.
"""
import argparse
import json
import os
import fnmatch
import csv
from pathlib import Path

SPIKE_NPY_FILES = {
    "spike_times.npy",
    "spike_templates.npy",
    "spike_clusters.npy",
    "spike_samples.npy",
    "templates.npy",
    "spike_index.npy",
}
PHY_CLUSTER_FILES = {
    "cluster_groups.csv",
    "cluster_group.tsv",
    "cluster_info.tsv",
    "cluster_kslabel.tsv",
}
CHANMAP_PATTERNS = ["chanMap*.mat", "chanmap*.mat", "*chanMap*.mat"]
BINARY_SUFFIXES = (".bin", ".dat", ".ns5", ".mda")


def check_folder(folder_path: Path):
    try:
        items = list(folder_path.iterdir())
    except Exception as e:
        return {
            "path": str(folder_path),
            "error": f"cannot list folder: {e}",
            "status": "Error",
        }

    names = [p.name for p in items]
    lower_items = {n.lower() for n in names}

    has_rez = "rez.mat" in lower_items
    has_spike_npy = any(name in lower_items for name in SPIKE_NPY_FILES)
    has_phy = any(name in lower_items for name in PHY_CLUSTER_FILES)

    has_chanmap = any(fnmatch.fnmatch(n.lower(), pat.lower()) for pat in CHANMAP_PATTERNS for n in lower_items)
    has_bin = any(n.lower().endswith(suffix) for n in lower_items for suffix in BINARY_SUFFIXES)

    has_phy_dir = any(p.is_dir() and (p.name.lower().startswith("phy") or "phy" in p.name.lower() or p.name.lower().startswith("sorting")) for p in items)

    notes = []
    if has_rez:
        status = "Complete"
        notes.append("found rez.mat")
    elif has_spike_npy and has_phy:
        status = "Complete"
        notes.append("spike .npy files + phy cluster file")
    elif has_spike_npy and has_bin:
        status = "Likely Complete"
        notes.append("spike .npy files + recording binary")
    elif has_phy_dir and has_chanmap:
        status = "Likely Complete"
        notes.append("phy dir + chanMap")
    else:
        status = "Incomplete"
        if not (has_rez or has_spike_npy or has_phy):
            notes.append("no Kilosort/Phy outputs detected")
        else:
            notes.append("partial outputs found")

    return {
        "path": str(folder_path),
        "has_rez": bool(has_rez),
        "has_spike_npy": bool(has_spike_npy),
        "has_phy_clusters": bool(has_phy),
        "has_phy_dir": bool(has_phy_dir),
        "has_chanmap": bool(has_chanmap),
        "has_bin": bool(has_bin),
        "status": status,
        "notes": "; ".join(notes),
    }


def scan(root_path: Path, max_depth: int = 4, verbose: bool = False):
    results = []
    root_path = root_path.resolve()
    for dirpath, dirnames, filenames in os.walk(root_path):
        # compute depth relative to root
        try:
            rel = Path(dirpath).relative_to(root_path)
            depth = len(rel.parts)
        except Exception:
            depth = 0
        if depth > max_depth:
            continue

        p = Path(dirpath)
        lower_files = {f.lower() for f in filenames}
        interesting = False
        if lower_files & SPIKE_NPY_FILES or "rez.mat" in lower_files or lower_files & PHY_CLUSTER_FILES:
            interesting = True
        if any(f.lower().endswith(suffix) for f in filenames for suffix in BINARY_SUFFIXES):
            interesting = True
        if any(fnmatch.fnmatch(p.name.lower(), "*session*") or fnmatch.fnmatch(p.name.lower(), "*recording*") for f in filenames):
            interesting = True

        if interesting:
            res = check_folder(p)
            results.append(res)
            if verbose:
                print(json.dumps(res, indent=2))
    return results


def save_csv(results, outpath):
    keys = ["path","status","has_rez","has_spike_npy","has_phy_clusters","has_phy_dir","has_chanmap","has_bin","notes"]
    with open(outpath, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in keys})


def save_json(results, outpath):
    with open(outpath, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Scan directory tree for Kilosort sessions")
    ap.add_argument("root", help="Root directory to scan")
    ap.add_argument("--csv", help="Write CSV report path")
    ap.add_argument("--json", help="Write JSON report path")
    ap.add_argument("--max-depth", type=int, default=6, help="Max recursion depth relative to root (default 6)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        # Try to handle Windows-style path when running in POSIX shell
        # Convert X:\... to /mnt/x/... or /x/... heuristically isn't reliable here, so error out with helpful message
        print(f"ERROR: root not found: {root}")
        print("If this is a Windows network path (e.g. X:\\\\share\\... ), ensure the drive is mounted and accessible from this shell.")
        return 2

    results = scan(root, max_depth=args.max_depth, verbose=args.verbose)
    print(f"Found {len(results)} candidate folders with Kilosort/Phy outputs under {root}")

    if args.csv:
        save_csv(results, args.csv)
        print("Wrote CSV:", args.csv)
    if args.json:
        save_json(results, args.json)
        print("Wrote JSON:", args.json)

    summary = {}
    for r in results:
        summary.setdefault(r.get("status", "Unknown"), 0)
        summary[r.get("status", "Unknown")] += 1
    print("Summary:", summary)

if __name__ == "__main__":
    main()
