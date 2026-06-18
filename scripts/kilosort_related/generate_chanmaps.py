"""Generate *_kilosortChanMap.mat for all sessions of a subject.

Uses SGLXMetaToCoords.MetaToCoords (the official Janelia script, outType=1)
which reads probe geometry from the SpikeGLX metadata file and writes a
Kilosort-compatible .mat with chanMap, xcoords, ycoords, kcoords, connected.

Output is saved alongside each session's tcat.imec0.ap.meta file with the
naming convention: {stem}_kilosortChanMap.mat
(e.g. BG_040_02052025_g0_tcat.imec0.ap_kilosortChanMap.mat)

Usage:
    py scripts/kilosort_related/generate_chanmaps.py BG_040
    py scripts/kilosort_related/generate_chanmaps.py BG_041
    py scripts/kilosort_related/generate_chanmaps.py BG_049
    py scripts/kilosort_related/generate_chanmaps.py BG_040 --force
"""

import argparse
import os
import sys
from pathlib import Path

# Add chanMap_related to path to import SGLXMetaToCoords
CHANMAP_DIR = Path(__file__).parent.parent.parent / "chanMap_related"
sys.path.insert(0, str(CHANMAP_DIR))
from SGLXMetaToCoords import MetaToCoords

PROCESSED_ROOT = "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/{subject}/Processed data"


def find_imec_dirs(proc_root):
    """Return sorted list of imec0 directory paths."""
    results = []
    for sess in sorted(os.listdir(proc_root)):
        ks_phy = os.path.join(proc_root, sess, "Kilosort&Phy")
        if not os.path.isdir(ks_phy):
            continue
        for d in sorted(os.listdir(ks_phy)):
            if "_imec" in d and os.path.isdir(os.path.join(ks_phy, d)):
                results.append(os.path.join(ks_phy, d))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", help="Subject name e.g. BG_040")
    parser.add_argument("--force", action="store_true", help="Overwrite existing chanmaps")
    args = parser.parse_args()

    proc_root = PROCESSED_ROOT.format(subject=args.subject)
    imec_dirs = find_imec_dirs(proc_root)
    if not imec_dirs:
        print(f"No imec dirs found under {proc_root}")
        sys.exit(1)

    print(f"{args.subject}: {len(imec_dirs)} sessions")

    ok = 0
    skipped = 0
    failed = []

    for imec_dir in imec_dirs:
        sess = Path(imec_dir).parent.parent.name
        meta_files = list(Path(imec_dir).glob("*tcat.imec0.ap.meta"))
        if not meta_files:
            print(f"  {sess}: SKIP — no tcat.imec0.ap.meta found")
            failed.append(sess)
            continue

        meta_path = meta_files[0]
        expected_out = meta_path.with_name(meta_path.stem + "_kilosortChanMap.mat")

        if expected_out.exists() and not args.force:
            print(f"  {sess}: already exists, skipping.")
            skipped += 1
            continue

        try:
            MetaToCoords(metaFullPath=meta_path, outType=1, showPlot=False)
            if expected_out.exists():
                print(f"  {sess}: OK -> {expected_out.name}")
                ok += 1
            else:
                print(f"  {sess}: ERROR — MetaToCoords ran but output not found at {expected_out}")
                failed.append(sess)
        except Exception as e:
            print(f"  {sess}: ERROR — {e}")
            failed.append(sess)

    print(f"\nDone: {ok} generated, {skipped} skipped, {len(failed)} failed.")
    if failed:
        print("Failed sessions:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
