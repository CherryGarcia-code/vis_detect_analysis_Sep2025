"""Run TPrime spike-time correction on processed Neuropixels sessions.

Replaces the MATLAB ``Run_TPrime.m`` / ``Run_TPrime_all_sessions_per_subject.m``
workflow entirely in Python.  For each session:

1. Finds probe directories under ``Kilosort&Phy/``
2. Converts ``spike_times.npy`` (samples) -> ``spike_times_sec.npy`` (seconds)
3. Extracts NI-DAQ sync rise times -> ``Nidaq/NI_Sync.txt``
4. Calls TPrime.exe to produce ``spike_times_sec_adj.npy``

Usage:
    py scripts/pipelines/run_tprime.py \
        --processed-root "X:/.../BG_039/Processed data" \
        --tprime-exe "G:/.../TPrime/TPrime.exe"

    py scripts/pipelines/run_tprime.py \
        --processed-root "X:/.../BG_039/Processed data" \
        --session BG_039_02042025 \
        --tprime-exe "G:/.../TPrime/TPrime.exe"

    py scripts/pipelines/run_tprime.py --dry-run ...
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np


from visdetect.core.ingest import load_ni_events
from visdetect.core.spikeglx import (
    find_sync_edge_file,
    get_sample_rate,
    read_meta,
    write_ni_sync_txt,
)

DEFAULT_TPRIME_EXE = (
    r"G:\Postdoc_research\Neuropixels_chronic"
    r"\DMDM_NPX_postprocessing_tools\TPrime\TPrime.exe"
)
DEFAULT_SYNC_PERIOD = 1.0


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def discover_sessions(processed_root: Path) -> list[str]:
    """List session folder names under processed_root."""
    if not processed_root.exists():
        return []
    return sorted(
        d.name
        for d in processed_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def discover_probe_dirs(session_dir: Path) -> list[Path]:
    """Find probe directories under Kilosort&Phy/."""
    ks_dir = session_dir / "Kilosort&Phy"
    if not ks_dir.exists():
        return []
    return sorted(
        d for d in ks_dir.iterdir()
        if d.is_dir() and "_imec" in d.name
    )


def find_meta_file(probe_dir: Path) -> Path | None:
    """Find the .ap.meta file in a probe directory."""
    matches = sorted(probe_dir.glob("*.ap.meta"))
    return matches[0] if matches else None


def find_ks_output_dir(probe_dir: Path) -> Path | None:
    """Find the Kilosort output subdirectory (e.g., kilosort4/) inside a probe dir."""
    # KS4 outputs go into a kilosort4/ subfolder
    ks4_dir = probe_dir / "kilosort4"
    if ks4_dir.exists():
        return ks4_dir
    # Fallback: spike_times.npy directly in probe dir (older KS3 layout)
    if (probe_dir / "spike_times.npy").exists():
        return probe_dir
    return None


def is_session_complete(probe_dirs: list[Path]) -> bool:
    """Check if all probes already have spike_times_sec_adj.npy."""
    if not probe_dirs:
        return False
    for d in probe_dirs:
        ks_dir = find_ks_output_dir(d)
        if ks_dir is None:
            return False
        if not (ks_dir / "spike_times_sec_adj.npy").exists():
            return False
    return True


def process_session(
    session_dir: Path,
    tprime_exe: str,
    sync_period: float,
    dry_run: bool = False,
    timeout: int = 1800,
) -> bool:
    """Run TPrime on a single session.

    Returns True on success, raises on failure.
    """
    session_name = session_dir.name
    probe_dirs = discover_probe_dirs(session_dir)

    if not probe_dirs:
        logging.warning("  No probe dirs found in %s", session_dir)
        return False

    logging.info("  Found %d probe(s): %s",
                 len(probe_dirs),
                 ", ".join(d.name for d in probe_dirs))

    # ── Step 1: Extract NI-DAQ sync rise times ─────────────────────
    ni_sync_path = session_dir / "Nidaq" / "NI_Sync.txt"

    if not ni_sync_path.exists():
        logging.info("  Extracting NI sync rise times...")
        ni_events = load_ni_events(session_dir)

        synch = ni_events.get("Synch")
        if synch is None:
            raise ValueError(f"No 'Synch' field in NIdaq_events for {session_name}")

        rise_times = np.asarray(synch).flatten()
        if rise_times.size == 0:
            raise ValueError(f"Empty sync rise times for {session_name}")

        if dry_run:
            logging.info("  [DRY RUN] Would write NI_Sync.txt (%d pulses)", rise_times.size)
        else:
            write_ni_sync_txt(rise_times, ni_sync_path)
            logging.info("  Wrote NI_Sync.txt (%d pulses)", rise_times.size)
    else:
        logging.info("  NI_Sync.txt already exists, reusing")

    # ── Step 2: Per-probe processing ───────────────────────────────
    for probe_dir in probe_dirs:
        logging.info("  Processing %s...", probe_dir.name)

        # Find Kilosort output directory (kilosort4/ subfolder)
        ks_dir = find_ks_output_dir(probe_dir)
        if ks_dir is None:
            raise FileNotFoundError(
                f"No Kilosort output (kilosort4/ or spike_times.npy) in {probe_dir}"
            )
        logging.debug("    KS output dir: %s", ks_dir)

        # Find metadata and get sample rate (meta is at probe dir level)
        meta_path = find_meta_file(probe_dir)
        if meta_path is None:
            raise FileNotFoundError(f"No .ap.meta file in {probe_dir}")
        meta = read_meta(meta_path)
        sample_rate = get_sample_rate(meta)
        logging.debug("    Sample rate: %.1f Hz", sample_rate)

        # Convert spike_times.npy (samples) -> spike_times_sec.npy (seconds)
        spike_times_path = ks_dir / "spike_times.npy"
        spike_times_sec_path = ks_dir / "spike_times_sec.npy"

        if not spike_times_path.exists():
            raise FileNotFoundError(f"No spike_times.npy in {ks_dir}")

        spike_samples = np.load(str(spike_times_path))
        spike_seconds = spike_samples.astype(np.float64) / sample_rate

        if dry_run:
            logging.info("    [DRY RUN] Would write spike_times_sec.npy (%d spikes)",
                         spike_seconds.size)
        else:
            np.save(str(spike_times_sec_path), spike_seconds)
            logging.info("    Wrote spike_times_sec.npy (%d spikes)", spike_seconds.size)

        # Find CatGT sync edge file (at probe dir level)
        edge_file = find_sync_edge_file(probe_dir)
        if edge_file is None:
            raise FileNotFoundError(
                f"No CatGT sync-edge file (*tcat.imec*.ap.xd_*.txt) in {probe_dir}"
            )
        logging.debug("    Edge file: %s", edge_file.name)

        # Build TPrime command
        adj_path = ks_dir / "spike_times_sec_adj.npy"

        # Use forward slashes in paths (matches MATLAB .bat behavior)
        ni_sync_fwd = str(ni_sync_path).replace("\\", "/")
        edge_fwd = str(edge_file).replace("\\", "/")
        sec_fwd = str(spike_times_sec_path).replace("\\", "/")
        adj_fwd = str(adj_path).replace("\\", "/")

        cmd = (
            f'"{tprime_exe}" -syncperiod={sync_period} '
            f'-tostream="{ni_sync_fwd}" '
            f'-fromstream=0,"{edge_fwd}" '
            f'-events=0,"{sec_fwd},{adj_fwd}"'
        )

        if dry_run:
            logging.info("    [DRY RUN] Would run:\n      %s", cmd)
        else:
            logging.info("    Running TPrime...")
            logging.debug("    CMD: %s", cmd)
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                logging.error("    TPrime stderr: %s", result.stderr)
                raise RuntimeError(
                    f"TPrime failed (exit {result.returncode}) for {probe_dir.name}"
                )

            if result.stdout.strip():
                logging.debug("    TPrime stdout: %s", result.stdout.strip())

            # Verify output
            if not adj_path.exists():
                raise RuntimeError(
                    f"TPrime completed but spike_times_sec_adj.npy not found in {probe_dir}"
                )

            # Report drift
            orig = np.load(str(spike_times_sec_path))
            adj = np.load(str(adj_path))
            if orig.size == adj.size and orig.size > 0:
                drift = adj.astype(np.float64) - orig.astype(np.float64)
                logging.info(
                    "    Done. Drift: mean=%.4f ms, max=%.4f ms",
                    np.mean(drift) * 1000,
                    np.max(np.abs(drift)) * 1000,
                )
            else:
                logging.info("    Done. Output: %d spikes", adj.size)

    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run TPrime spike-time correction (replaces MATLAB pipeline)"
    )
    parser.add_argument(
        "--processed-root", type=Path, required=True,
        help="Root 'Processed data' directory (contains session subdirectories)",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Process a single session (e.g., BG_039_02042025)",
    )
    parser.add_argument(
        "--tprime-exe", type=str, default=DEFAULT_TPRIME_EXE,
        help=f"Path to TPrime.exe (default: {DEFAULT_TPRIME_EXE})",
    )
    parser.add_argument(
        "--sync-period", type=float, default=DEFAULT_SYNC_PERIOD,
        help=f"Sync pulse period in seconds (default: {DEFAULT_SYNC_PERIOD})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if spike_times_sec_adj.npy already exists",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="Per-probe TPrime.exe subprocess timeout in seconds (default: 1800). "
             "Raise for large sessions (tens of millions of spikes) over slow mounts.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if not args.processed_root.exists():
        logging.error("Processed root not found: %s", args.processed_root)
        return 1

    # Discover sessions
    if args.session:
        sessions = [args.session]
    else:
        sessions = discover_sessions(args.processed_root)

    if not sessions:
        logging.error("No sessions found in %s", args.processed_root)
        return 1

    logging.info("Found %d session(s)", len(sessions))
    logging.info("Processed root: %s", args.processed_root)
    logging.info("TPrime exe:     %s", args.tprime_exe)
    logging.info("Sync period:    %.1f s", args.sync_period)

    # Process each session
    success_count = 0
    skip_count = 0
    failed: list[tuple[str, str]] = []

    for sname in sessions:
        session_dir = args.processed_root / sname

        if not session_dir.exists():
            logging.warning("Session directory not found: %s", session_dir)
            failed.append((sname, "directory not found"))
            continue

        probe_dirs = discover_probe_dirs(session_dir)

        # Skip check
        if not args.force and is_session_complete(probe_dirs):
            logging.info("SKIP (complete): %s", sname)
            skip_count += 1
            continue

        logging.info("Processing: %s", sname)

        try:
            process_session(session_dir, args.tprime_exe, args.sync_period,
                            args.dry_run, args.timeout)
            success_count += 1
        except Exception as e:
            logging.error("FAILED: %s - %s", sname, e)
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
