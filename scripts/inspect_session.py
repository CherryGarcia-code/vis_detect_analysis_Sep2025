"""Simple CLI to inspect a session .pkl and produce a JSON summary + a PNG.

Usage: python scripts/inspect_session.py path/to/session.pkl outputs/
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.session_io import load_session, session_summary


def main(argv):
    if len(argv) < 3:
        print("Usage: inspect_session.py path/to/session.pkl output_dir")
        return 2
    pkl = Path(argv[1])
    outdir = Path(argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    session = load_session(str(pkl))
    summary = session_summary(session)
    # Write JSON summary
    with (outdir / "session_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Make a simple histogram of per-cluster spike counts
    counts = [len(c.spike_times) for c in session.clusters]
    plt.figure(figsize=(6, 4))
    plt.hist(counts, bins=50)
    plt.xlabel("Total spikes per cluster")
    plt.ylabel("Count")
    plt.title(f"Spikes per cluster ({summary.get('n_clusters')})")
    plt.tight_layout()
    plt.savefig(outdir / "spike_counts_hist.png", dpi=120)
    print("Wrote", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
