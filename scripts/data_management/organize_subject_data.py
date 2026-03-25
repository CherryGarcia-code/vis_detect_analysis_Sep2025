"""organize_subject_data.py — Safe organization of subject data files.

Scans a subject's directory on the network drive and generates a report of
operations needed to match the BG_046 gold-standard structure:

  1. Copy loose behavioral JSONs → Raw data/{session}/Session/
  2. Copy raw_backup sessions   → Raw data/{session}/ with proper subdirs

Safety guarantees:
  - Copy-only (never deletes or moves files)
  - Dry-run by default (--execute required to perform copies)
  - Verifies file sizes match after copy
  - Skips files that already exist at destination
  - Writes a timestamped log of all operations

Usage:
  py scripts/data_management/organize_subject_data.py BG_031
  py scripts/data_management/organize_subject_data.py BG_038 --execute
  py scripts/data_management/organize_subject_data.py BG_039 --execute --include-raw-backup
"""

import argparse
import datetime
import logging
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ── Constants ───────────────────────────────────────────────────────────
WEPHYS_ROOT = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys")
RAW_BACKUP = WEPHYS_ROOT / "raw_backup"
FSMDATA = WEPHYS_ROOT / "FSMdata"

# JSON triplet suffixes
JSON_TYPES = ("__trials.json", "__session_settings.json", "__computer_settings.json")

# Session dir variant suffixes (stripped during date matching)
VARIANT_SUFFIXES = re.compile(r"(_v2|_b|_c|_passive_watching|_laser|_laser_v2|_2ndlaserSNr)$")


# ── Date parsing ────────────────────────────────────────────────────────
def parse_json_date(filename: str, subject: str) -> str | None:
    """Extract YYYYMMDD date string from a JSON filename.

    Pattern: {subject}_{YYYYMMDD}_{HHMMSS}__{type}.json
    Also handles: {subject}_b_{YYYYMMDD}_... (BG_031_b variant)
    """
    # Standard pattern
    m = re.match(rf"^{re.escape(subject)}(?:_b)?_(\d{{8}})_\d{{6}}__", filename)
    if m:
        return m.group(1)
    return None


def parse_session_dir_date(dirname: str, subject: str) -> str | None:
    """Extract YYYYMMDD date string from a session directory name.

    Handles two naming conventions:
      - 6-digit DDMMYY:   BG_031_050225   → 2025-02-05 → 20250205
      - 8-digit DDMMYYYY: BG_031_01042025 → 2025-04-01 → 20250401

    Strips variant suffixes (_v2, _b, _c, etc.) before parsing.
    """
    # Remove subject prefix
    prefix = subject + "_"
    if not dirname.startswith(prefix):
        return None
    remainder = dirname[len(prefix):]

    # Strip known variant suffixes
    remainder_clean = VARIANT_SUFFIXES.sub("", remainder)

    # Try 8-digit DDMMYYYY first
    m8 = re.match(r"^(\d{2})(\d{2})(\d{4})$", remainder_clean)
    if m8:
        dd, mm, yyyy = m8.groups()
        return f"{yyyy}{mm}{dd}"

    # Try 6-digit DDMMYY
    m6 = re.match(r"^(\d{2})(\d{2})(\d{2})$", remainder_clean)
    if m6:
        dd, mm, yy = m6.groups()
        return f"20{yy}{mm}{dd}"

    return None


def parse_raw_backup_date(dirname: str, subject: str) -> str | None:
    """Extract YYYYMMDD date from a raw_backup directory name.

    Pattern: {subject}_{DDMMYYYY}_g0  or  {subject}_{DDMMYY}_g0
    Also handles suffixes like _laser_g0, _b_g0, _2ndlaserSNr_g0.
    """
    prefix = subject + "_"
    if not dirname.startswith(prefix):
        return None
    remainder = dirname[len(prefix):]

    # Remove _g0 or _g1 suffix
    remainder = re.sub(r"_g\d+$", "", remainder)

    # Strip known variant suffixes
    remainder_clean = VARIANT_SUFFIXES.sub("", remainder)

    # Try 8-digit DDMMYYYY
    m8 = re.match(r"^(\d{2})(\d{2})(\d{4})$", remainder_clean)
    if m8:
        dd, mm, yyyy = m8.groups()
        return f"{yyyy}{mm}{dd}"

    # Try 6-digit DDMMYY
    m6 = re.match(r"^(\d{2})(\d{2})(\d{2})$", remainder_clean)
    if m6:
        dd, mm, yy = m6.groups()
        return f"20{yy}{mm}{dd}"

    return None


# ── Discovery ───────────────────────────────────────────────────────────
def find_loose_jsons(subject_dir: Path, subject: str) -> dict[str, list[Path]]:
    """Find loose behavioral JSONs in a subject's top-level directory.

    Returns: {YYYYMMDD: [path1, path2, ...]} grouped by date.
    """
    by_date = defaultdict(list)
    for f in sorted(subject_dir.iterdir()):
        if not f.is_file():
            continue
        if not any(f.name.endswith(t) for t in JSON_TYPES):
            continue
        date = parse_json_date(f.name, subject)
        if date:
            by_date[date].append(f)
    return dict(by_date)


def find_fsmdata_jsons(subject: str) -> dict[str, list[Path]]:
    """Find behavioral JSONs in the FSMdata/{subject}/ directory.

    Returns: {YYYYMMDD: [path1, path2, ...]} grouped by date.
    """
    fsmdata_dir = FSMDATA / subject
    if not fsmdata_dir.is_dir():
        return {}
    by_date = defaultdict(list)
    for f in sorted(fsmdata_dir.iterdir()):
        if not f.is_file():
            continue
        if not any(f.name.endswith(t) for t in JSON_TYPES):
            continue
        date = parse_json_date(f.name, subject)
        if date:
            by_date[date].append(f)
    return dict(by_date)


def find_session_dirs(raw_data_dir: Path, subject: str) -> dict[str, list[Path]]:
    """Map session directories by date.

    Returns: {YYYYMMDD: [dir1, dir2, ...]}
    Multiple dirs per date when variants exist (e.g. _v2, _b).
    """
    by_date = defaultdict(list)
    if not raw_data_dir.is_dir():
        return {}
    for d in sorted(raw_data_dir.iterdir()):
        if not d.is_dir():
            continue
        date = parse_session_dir_date(d.name, subject)
        if date:
            by_date[date].append(d)
    return dict(by_date)


def find_raw_backup_dirs(subject: str) -> dict[str, list[Path]]:
    """Find raw_backup directories for this subject.

    Returns: {YYYYMMDD: [dir1, dir2, ...]}
    """
    by_date = defaultdict(list)
    if not RAW_BACKUP.is_dir():
        return {}
    for d in sorted(RAW_BACKUP.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith(subject + "_"):
            continue
        # Skip survey probe dirs
        if "SvyPrb" in d.name:
            continue
        date = parse_raw_backup_date(d.name, subject)
        if date:
            by_date[date].append(d)
    return dict(by_date)


# ── Report generation ───────────────────────────────────────────────────
class CopyOp:
    """A planned copy operation."""

    def __init__(self, src: Path, dst: Path, category: str):
        self.src = src
        self.dst = dst
        self.category = category  # "json" or "raw_backup"
        self.size = src.stat().st_size if src.is_file() else 0
        self.status = "pending"  # pending, skipped, done, error

    def __repr__(self):
        size_mb = self.size / (1024 * 1024)
        return f"  {self.src.name}  ({size_mb:.1f} MB) → {self.dst}"


def plan_json_copies(
    subject: str,
    loose_jsons: dict[str, list[Path]],
    fsmdata_jsons: dict[str, list[Path]],
    session_dirs: dict[str, list[Path]],
) -> tuple[list[CopyOp], list[str]]:
    """Plan copy operations for behavioral JSONs → Session/ dirs.

    For each date, prefer loose JSONs from the subject's top-level dir
    (they're the primary source). Fall back to FSMdata if top-level missing.
    Copy into each matching session dir's Session/ subdir.
    """
    ops = []
    anomalies = []

    # Merge sources: prefer top-level, fill gaps from FSMdata
    all_dates = sorted(set(loose_jsons) | set(fsmdata_jsons))

    for date in all_dates:
        jsons = loose_jsons.get(date, []) or fsmdata_jsons.get(date, [])
        if not jsons:
            continue

        if date not in session_dirs:
            anomalies.append(
                f"  Date {date}: {len(jsons)} JSON files but no matching "
                f"session dir in Raw data/"
            )
            continue

        for session_dir in session_dirs[date]:
            session_subdir = session_dir / "Session"
            # Create Session/ if needed (will be created during execute)
            for json_path in jsons:
                dst = session_subdir / json_path.name
                if dst.exists() and dst.stat().st_size == json_path.stat().st_size:
                    continue  # Already there and complete
                ops.append(CopyOp(json_path, dst, "json"))

    # Check for dates with session dirs but no behavioral data
    for date in sorted(session_dirs):
        if date not in loose_jsons and date not in fsmdata_jsons:
            dirs = [d.name for d in session_dirs[date]]
            anomalies.append(
                f"  Date {date}: session dir(s) {dirs} but no behavioral "
                f"JSONs found anywhere"
            )

    return ops, anomalies


def plan_raw_backup_copies(
    subject: str,
    raw_backup_dirs: dict[str, list[Path]],
    session_dirs: dict[str, list[Path]],
    raw_data_dir: Path,
) -> tuple[list[CopyOp], list[str]]:
    """Plan copy operations for raw_backup → Raw data/.

    Only plans copies for sessions NOT already in Raw data/.
    Creates the proper subdirectory structure (EphysNidaq/, Session/, Cameras/).
    """
    ops = []
    anomalies = []

    for date in sorted(raw_backup_dirs):
        if date in session_dirs:
            # Already organized — check if EphysNidaq is populated
            needs_ephys = False
            for session_dir in session_dirs[date]:
                ephys_dir = session_dir / "EphysNidaq"
                if ephys_dir.is_dir() and any(ephys_dir.iterdir()):
                    continue  # Already has data
                else:
                    needs_ephys = True
            if not needs_ephys:
                continue
            # EphysNidaq is empty — plan copy into existing session dir(s)
            for backup_dir in raw_backup_dirs[date]:
                # Match backup to the right session dir by name
                backup_session = re.sub(r"_g\d+$", "", backup_dir.name)
                # Find matching session dir, or use first one for the date
                matched_dirs = [
                    d for d in session_dirs[date]
                    if d.name == backup_session
                    or d.name == backup_session.rstrip("_b").rstrip("_c")
                ]
                dest_session = matched_dirs[0] if matched_dirs else session_dirs[date][0]
                ephys_dir = dest_session / "EphysNidaq"
                if ephys_dir.is_dir() and any(ephys_dir.iterdir()):
                    continue  # This specific dir already has data
                for root, dirs, files in os.walk(backup_dir):
                    rel_root = Path(root).relative_to(backup_dir)
                    for fname in files:
                        src = Path(root) / fname
                        dst = dest_session / "EphysNidaq" / rel_root / fname
                        ops.append(CopyOp(src, dst, "raw_backup"))
            continue

        # Not in Raw data/ — plan the copy
        for backup_dir in raw_backup_dirs[date]:
            # Determine session dir name from the backup dir
            # Strip _g0/_g1 suffix to get session name
            session_name = re.sub(r"_g\d+$", "", backup_dir.name)
            dest_session = raw_data_dir / session_name

            # Walk the backup directory and plan copies
            for root, dirs, files in os.walk(backup_dir):
                rel_root = Path(root).relative_to(backup_dir)
                for fname in files:
                    src = Path(root) / fname
                    # Map into EphysNidaq/ subdirectory
                    dst = dest_session / "EphysNidaq" / rel_root / fname
                    ops.append(CopyOp(src, dst, "raw_backup"))

    return ops, anomalies


# ── Execution ───────────────────────────────────────────────────────────
def execute_copies(ops: list[CopyOp], logger: logging.Logger) -> tuple[int, int, int]:
    """Execute planned copy operations.

    Returns: (copied, skipped, errors)
    """
    copied = skipped = errors = 0

    for op in ops:
        if op.dst.exists():
            dst_size = op.dst.stat().st_size
            if dst_size == op.size:
                op.status = "skipped"
                skipped += 1
                logger.info(f"SKIP (exists, size OK): {op.dst}")
                continue
            else:
                # Partial/corrupt file — remove and re-copy
                logger.warning(
                    f"INCOMPLETE FILE (src={op.size}, dst={dst_size}), "
                    f"re-copying: {op.dst}"
                )
                op.dst.unlink()

        try:
            op.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(op.src), str(op.dst))

            # Verify size
            dst_size = op.dst.stat().st_size
            if dst_size != op.size:
                op.status = "error"
                errors += 1
                logger.error(
                    f"SIZE MISMATCH: {op.src} ({op.size}) → "
                    f"{op.dst} ({dst_size})"
                )
            else:
                op.status = "done"
                copied += 1
                logger.info(f"COPIED: {op.src} → {op.dst}")

        except Exception as e:
            op.status = "error"
            errors += 1
            logger.error(f"ERROR copying {op.src} → {op.dst}: {e}")

    return copied, skipped, errors


def create_empty_subdirs(session_dir: Path, logger: logging.Logger):
    """Create the standard subdirectory structure if missing."""
    for subdir in ("Cameras", "EphysNidaq", "Session"):
        d = session_dir / subdir
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"CREATED DIR: {d}")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Organize subject data to match gold-standard structure."
    )
    parser.add_argument("subject", help="Subject ID (e.g., BG_031)")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform copies (default: dry-run only)",
    )
    parser.add_argument(
        "--include-raw-backup",
        action="store_true",
        help="Include raw_backup → Raw data copies (large files, run overnight)",
    )
    parser.add_argument(
        "--wephys-root",
        type=Path,
        default=WEPHYS_ROOT,
        help=f"Override wEPhys root (default: {WEPHYS_ROOT})",
    )
    args = parser.parse_args()

    subject = args.subject
    wephys = args.wephys_root
    subject_dir = wephys / subject
    raw_data_dir = subject_dir / "Raw data"

    # ── Setup logging ───────────────────────────────────────────────
    log_dir = Path("logs/data_management")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "EXECUTE" if args.execute else "DRY_RUN"
    log_file = log_dir / f"organize_{subject}_{mode_str}_{timestamp}.log"

    logger = logging.getLogger("organize")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(fh)
    # Also log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(f"{'='*70}")
    logger.info(f"Subject:       {subject}")
    logger.info(f"Mode:          {mode_str}")
    logger.info(f"Subject dir:   {subject_dir}")
    logger.info(f"Raw data dir:  {raw_data_dir}")
    logger.info(f"Log file:      {log_file}")
    logger.info(f"{'='*70}")

    # ── Validate paths ──────────────────────────────────────────────
    if not subject_dir.is_dir():
        logger.error(f"Subject directory not found: {subject_dir}")
        sys.exit(1)

    if not raw_data_dir.is_dir():
        logger.warning(f"Raw data/ directory not found — creating: {raw_data_dir}")
        if args.execute:
            raw_data_dir.mkdir(parents=True, exist_ok=True)

    # ── Discovery ───────────────────────────────────────────────────
    logger.info("\n── Discovery ──────────────────────────────────────")

    loose_jsons = find_loose_jsons(subject_dir, subject)
    logger.info(f"Loose JSONs (top-level): {sum(len(v) for v in loose_jsons.values())} "
                f"files across {len(loose_jsons)} dates")

    fsmdata_jsons = find_fsmdata_jsons(subject)
    logger.info(f"FSMdata JSONs:           {sum(len(v) for v in fsmdata_jsons.values())} "
                f"files across {len(fsmdata_jsons)} dates")

    session_dirs = find_session_dirs(raw_data_dir, subject)
    logger.info(f"Session dirs (Raw data): {sum(len(v) for v in session_dirs.values())} "
                f"dirs across {len(session_dirs)} dates")

    raw_backup_dirs = find_raw_backup_dirs(subject)
    logger.info(f"raw_backup dirs:         {sum(len(v) for v in raw_backup_dirs.values())} "
                f"dirs across {len(raw_backup_dirs)} dates")

    # ── Plan: JSON copies ───────────────────────────────────────────
    logger.info("\n── Section A: Behavioral JSON Copies ──────────────")
    json_ops, json_anomalies = plan_json_copies(
        subject, loose_jsons, fsmdata_jsons, session_dirs
    )

    if json_ops:
        total_size = sum(op.size for op in json_ops)
        logger.info(f"Planned: {len(json_ops)} JSON files to copy "
                     f"({total_size / (1024*1024):.1f} MB total)")
        # Show first 20 operations as sample
        for op in json_ops[:20]:
            logger.info(str(op))
        if len(json_ops) > 20:
            logger.info(f"  ... and {len(json_ops) - 20} more")
    else:
        logger.info("No JSON copies needed — all Session/ dirs already populated.")

    # ── Plan: raw_backup copies ─────────────────────────────────────
    logger.info("\n── Section B: raw_backup → Raw data Copies ────────")
    if args.include_raw_backup:
        backup_ops, backup_anomalies = plan_raw_backup_copies(
            subject, raw_backup_dirs, session_dirs, raw_data_dir
        )
        if backup_ops:
            total_size = sum(op.size for op in backup_ops)
            logger.info(
                f"Planned: {len(backup_ops)} files to copy "
                f"({total_size / (1024**3):.1f} GB total)"
            )
            # Group by destination session for readability
            by_session = defaultdict(list)
            for op in backup_ops:
                # Extract session dir name from destination
                parts = op.dst.relative_to(raw_data_dir).parts
                by_session[parts[0]].append(op)
            for sname, sops in sorted(by_session.items()):
                ssize = sum(op.size for op in sops)
                logger.info(f"  {sname}: {len(sops)} files ({ssize / (1024**3):.1f} GB)")
        else:
            logger.info("No raw_backup copies needed.")
    else:
        backup_ops = []
        backup_anomalies = []
        # Still report what WOULD be needed
        missing_dates = sorted(set(raw_backup_dirs) - set(session_dirs))
        if missing_dates:
            logger.info(
                f"Skipped (use --include-raw-backup): {len(missing_dates)} dates "
                f"in raw_backup but not in Raw data/"
            )
            for date in missing_dates:
                dirs = [d.name for d in raw_backup_dirs[date]]
                logger.info(f"  {date}: {dirs}")
        else:
            logger.info("All raw_backup sessions already exist in Raw data/.")

    # ── Anomalies ───────────────────────────────────────────────────
    all_anomalies = json_anomalies + backup_anomalies
    logger.info("\n── Section C: Anomalies ─────────────────────────────")
    if all_anomalies:
        for a in all_anomalies:
            logger.warning(a)
    else:
        logger.info("No anomalies detected.")

    # ── Summary ─────────────────────────────────────────────────────
    all_ops = json_ops + backup_ops
    logger.info(f"\n── Summary ─────────────────────────────────────────")
    logger.info(f"Total planned operations: {len(all_ops)}")
    logger.info(f"  JSON copies:       {len(json_ops)}")
    logger.info(f"  raw_backup copies: {len(backup_ops)}")
    logger.info(f"  Anomalies:         {len(all_anomalies)}")

    if not args.execute:
        logger.info(f"\nDRY RUN — no files were copied.")
        logger.info(f"Re-run with --execute to perform the copies.")
        logger.info(f"Full report saved to: {log_file}")
        return

    # ── Execute ─────────────────────────────────────────────────────
    logger.info(f"\n── Executing Copies ───────────────────────────────")

    # First ensure Session/ subdirs exist for JSON targets
    for op in json_ops:
        op.dst.parent.mkdir(parents=True, exist_ok=True)

    # For raw_backup copies, also create Cameras/ and Session/ subdirs
    if backup_ops:
        created_sessions = set()
        for op in backup_ops:
            parts = op.dst.relative_to(raw_data_dir).parts
            session_path = raw_data_dir / parts[0]
            if session_path not in created_sessions:
                create_empty_subdirs(session_path, logger)
                created_sessions.add(session_path)

    copied, skipped, errors = execute_copies(all_ops, logger)

    logger.info(f"\n── Results ─────────────────────────────────────────")
    logger.info(f"Copied:  {copied}")
    logger.info(f"Skipped: {skipped} (already exist)")
    logger.info(f"Errors:  {errors}")
    logger.info(f"Log:     {log_file}")

    if errors > 0:
        logger.error("ERRORS OCCURRED — check log file for details!")
        sys.exit(1)


if __name__ == "__main__":
    main()
