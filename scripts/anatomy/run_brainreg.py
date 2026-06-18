# scripts/anatomy/run_brainreg.py
"""Thin unattended wrapper around the brainreg CLI (headless; cluster-friendly).

Usage:
    py scripts/anatomy/run_brainreg.py --image <vol.tif> --out <dir> \
        --voxel 5 5 5 --orientation asr --atlas allen_mouse_25um
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def build_command(image, out, voxel, orientation, atlas):
    return ["brainreg", image, out,
            "-v", str(voxel[0]), str(voxel[1]), str(voxel[2]),
            "--orientation", orientation, "--atlas", atlas]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", nargs=3, type=float, required=True)
    ap.add_argument("--orientation", required=True, help="e.g. 'asr' (BrainGlobe convention)")
    ap.add_argument("--atlas", default="allen_mouse_25um")
    a = ap.parse_args()
    cmd = build_command(a.image, a.out, a.voxel, a.orientation, a.atlas)
    print("running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
