#!/usr/bin/env python3
"""Curate cluster_group.tsv in UnitMatch input dirs to match extracted units.

UnitMatch's load_good_waveforms loads Unit{ID}_RawSpikes.npy for EVERY row of
cluster_group.tsv labelled 'good'. When that tsv is the raw Kilosort label file
it lists the full KS 'good' set (a superset of the good_and_stable units we
extracted), so UnitMatch tries to load missing files and crashes. This rewrites
each session's cluster_group.tsv so 'good' == exactly the units that have a
RawWaveforms file (mirrors BG_046's curated tsv).

Idempotent. Use this to fix input dirs extracted before the curation step was
added to prep_unitmatch_full_trial_waveforms.py.

Usage:
    py scripts/analysis/fix_unitmatch_cluster_group.py --input data/unit_match/input/BG_039
    py scripts/analysis/fix_unitmatch_cluster_group.py --input <dir1> <dir2> ...
"""
import argparse
from pathlib import Path


def curate_one(session_dir):
    wav_dir = session_dir / "RawWaveforms"
    if not wav_dir.is_dir():
        return None
    ids = sorted(int(p.name[4:].split('_')[0])
                 for p in wav_dir.glob("Unit*_RawSpikes.npy"))
    lines = ["cluster_id\tKSLabel"] + [f"{i}\tgood" for i in ids]
    # write_bytes => LF endings (file is consumed by UnitMatch on Linux/ceph)
    (session_dir / "cluster_group.tsv").write_bytes(("\n".join(lines) + "\n").encode())
    return len(ids)


def main():
    ap = argparse.ArgumentParser(description="Curate cluster_group.tsv to extracted units")
    ap.add_argument("--input", nargs="+", required=True,
                    help="One or more UnitMatch input roots (each holds session subdirs)")
    args = ap.parse_args()

    for root in args.input:
        root = Path(root)
        sess_dirs = sorted(d for d in root.iterdir()
                           if d.is_dir() and (d / "RawWaveforms").is_dir())
        total = 0
        for d in sess_dirs:
            n = curate_one(d)
            total += (n or 0)
        print(f"{root}: curated {len(sess_dirs)} sessions, {total} good units total")


if __name__ == "__main__":
    main()
