"""Copy a local clone of the Bombcell repo's pyBombCell folder into this workspace.

Usage (run locally where the cloned repo is accessible):

python scripts/copy_pybombcell_locally.py --src "C:\path\to\bombcell\pyBombCell"

The script will copy the entire folder into `src/integrations/_pybombcell_local/`.
It will also copy any top-level LICENSE file found adjacent to `pyBombCell` (to preserve
the original GPL-3.0 license text) and create a README describing obligations.

IMPORTANT: Copying GPL-3.0 code into this repository means the GPL applies to the
copied files. Do not change the license header or remove the LICENSE file. Only run
this script if you accept those terms.
"""

import argparse
from pathlib import Path
import shutil
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Copy pyBombCell folder into workspace (local operation)"
    )
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the cloned bombcell repo pyBombCell folder",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path("src") / "integrations" / "_pybombcell_local",
        help="Destination inside this repo",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite destination if it exists"
    )
    return p.parse_args()


def main():
    args = parse_args()
    src = args.src
    dest = args.dest
    if not src.exists() or not src.is_dir():
        print("Source path not found or not a directory:", src)
        sys.exit(2)
    if dest.exists():
        if args.overwrite:
            print("Removing existing destination", dest)
            shutil.rmtree(dest)
        else:
            print("Destination already exists:", dest)
            print("Use --overwrite to replace")
            sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {src} -> {dest} (this may take a moment)")
    shutil.copytree(src, dest)

    # Try to copy adjacent LICENSE or COPYRIGHT files if present in parent folder
    parent = src.parent
    license_candidates = ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"]
    for name in license_candidates:
        p = parent / name
        if p.exists():
            shutil.copy2(p, dest / name)
            print("Copied license:", p.name)
            break

    # Add a short note file reminding about GPL-3.0 obligations
    note = dest / "README_COPYRIGHT.md"
    note.write_text(
        """# Local copy of pyBombCell (Bombcell)

This folder is a local copy of the `pyBombCell` Python code from the Bombcell project.
Bombcell is licensed under GNU GPL-3.0. By copying these files into this repository you
agree to keep the license intact and follow GPL-3.0 terms for these files. Do not remove
the LICENSE file or modify the license header.
"""
    )
    print("Copy complete. Local copy is at", dest)


if __name__ == "__main__":
    main()
