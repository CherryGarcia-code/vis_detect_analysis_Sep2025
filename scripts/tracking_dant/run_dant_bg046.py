"""Run DANT (multi-shank) on the BG_046 input folder. Use DANT_PY from the worktree root."""
import argparse
import os

import numpy as np
import hjson

np.random.seed(42)  # DANT does not seed its motion init / bootstrap

from pyDANT import runDANTMultiShank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=os.path.join(os.path.dirname(__file__), "settings_bg046.json"))
    ap.add_argument("--path-to-data", default=None, help="override settings path_to_data")
    ap.add_argument("--output-folder", default=None, help="override settings output_folder")
    args = ap.parse_args()

    with open(args.settings) as f:
        user_settings = hjson.load(f)
    if args.path_to_data:
        user_settings["path_to_data"] = args.path_to_data
    if args.output_folder:
        user_settings["output_folder"] = args.output_folder
    os.makedirs(user_settings["output_folder"], exist_ok=True)
    print(f"Running DANT multi-shank: {user_settings['path_to_data']} -> {user_settings['output_folder']}")
    runDANTMultiShank(user_settings)


if __name__ == "__main__":
    main()
