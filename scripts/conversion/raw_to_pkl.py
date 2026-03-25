"""Batch convert raw session data directly to .pkl (no MATLAB dependency).

Reads behavioral JSON, NI-DAQ events, and Kilosort spike data from
raw/processed directories and produces Session .pkl files.

Usage:
    py scripts/conversion/raw_to_pkl.py \
        --raw-root "X:/public/.../BG_046/Raw data" \
        --processed-root "X:/public/.../BG_046/Processed data" \
        --out-dir data/pkls/BG_046

    py scripts/conversion/raw_to_pkl.py \
        --raw-root "X:/public/.../BG_046/Raw data" \
        --processed-root "X:/public/.../BG_046/Processed data" \
        --out-dir data/pkls/BG_046 \
        --session BG_046_01072025

    py scripts/conversion/raw_to_pkl.py --dry-run ...
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from visdetect.core.ingest import build_session_from_raw
from visdetect.core.session import save_session


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def discover_sessions(raw_root: Path) -> list:
    """List session folder names under raw_root."""
    if not raw_root.exists():
        return []
    return sorted(
        d.name
        for d in raw_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert raw session data to .pkl (replaces MATLAB pipeline)"
    )
    parser.add_argument(
        "--raw-root", type=Path, required=True,
        help="Root directory for raw data (contains session subdirectories)",
    )
    parser.add_argument(
        "--processed-root", type=Path, required=True,
        help="Root directory for processed data (Kilosort, NI-DAQ)",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory for .pkl files",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Convert a single session (e.g., BG_046_01072025)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .pkl files",
    )
    parser.add_argument(
        "--keep-all-good", action="store_true",
        help="Store all KS-good clusters (not just stable ones)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List sessions without converting",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Discover sessions
    if args.session:
        sessions = [args.session]
    else:
        sessions = discover_sessions(args.raw_root)

    if not sessions:
        logging.error("No sessions found in %s", args.raw_root)
        return 1

    logging.info("Found %d session(s)", len(sessions))
    logging.info("Raw root:       %s", args.raw_root)
    logging.info("Processed root: %s", args.processed_root)
    logging.info("Output dir:     %s", args.out_dir)

    if args.dry_run:
        logging.info("\n=== DRY RUN ===")
        for sname in sessions:
            pkl_path = args.out_dir / f"{sname}.pkl"
            status = "EXISTS" if pkl_path.exists() else "NEEDS CONVERSION"
            # Check for required subdirectories
            raw_ok = (args.raw_root / sname / "Session").exists()
            proc_ok = (args.processed_root / sname / "Nidaq").exists()
            ks_ok = (args.processed_root / sname / "Kilosort&Phy").exists()
            flags = f"raw:{'Y' if raw_ok else 'N'} nidaq:{'Y' if proc_ok else 'N'} ks:{'Y' if ks_ok else 'N'}"
            logging.info("  %s  %s  [%s]", status, sname, flags)
        return 0

    # Process each session
    logging.info("\n=== STARTING CONVERSION ===\n")
    success_count = 0
    skip_count = 0
    failed = []

    for sname in tqdm(sessions, desc="Converting", unit="session"):
        pkl_path = args.out_dir / f"{sname}.pkl"

        if pkl_path.exists() and not args.force:
            logging.info("SKIP (exists): %s", sname)
            skip_count += 1
            continue

        try:
            session = build_session_from_raw(
                args.raw_root,
                args.processed_root,
                sname,
                keep_all_good=args.keep_all_good,
            )
            save_session(session, str(pkl_path))

            n_trials = len(session.trials)
            n_clusters = len(session.clusters)
            n_stable = len(session.good_and_stable_ids) if session.good_and_stable_ids else 0
            logging.info(
                "OK: %s -> %d trials, %d clusters (%d stable)",
                sname, n_trials, n_clusters, n_stable,
            )
            success_count += 1

            del session
            gc.collect()

        except Exception as e:
            logging.error("FAILED: %s — %s", sname, e)
            failed.append((sname, str(e)))

    # Summary
    logging.info("\n=== SUMMARY ===")
    logging.info("Total:     %d", len(sessions))
    logging.info("Success:   %d", success_count)
    logging.info("Skipped:   %d", skip_count)
    logging.info("Failed:    %d", len(failed))
    if failed:
        logging.error("\nFailed sessions:")
        for sname, err in failed:
            logging.error("  %s: %s", sname, err)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
