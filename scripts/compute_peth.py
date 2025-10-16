"""Compute PETHs for a session and save them to disk.

Usage: python scripts/compute_peth.py path/to/session.pkl event_name output_dir
"""

import sys
from pathlib import Path
import numpy as np
import json
from src.session_io import load_session
from src.align import compute_peth_for_session


def main(argv):
    if len(argv) < 4:
        print("Usage: compute_peth.py path/to/session.pkl event_name output_dir")
        return 2
    pkl = Path(argv[1])
    event_name = argv[2]
    outdir = Path(argv[3])
    outdir.mkdir(parents=True, exist_ok=True)

    session = load_session(str(pkl))
    peths = compute_peth_for_session(session, event_name)
    # Save summary JSON with shapes
    summary = {
        int(k): {"n_trials": v["n_trials"], "peth_len": len(v["peth"])}
        for k, v in peths.items()
    }
    with (outdir / "peths_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    # Save one example numpy array
    if peths:
        cid, data = next(iter(peths.items()))
        np.save(outdir / f"peth_cluster_{cid}.npy", data["peth"])
    print("Wrote PETHs to", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
