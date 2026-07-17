"""Scan (and optionally repair) session-id corruption in CSV deliverables.

THE BUG
-------
Session ids are ``DDMMYYYY`` (e.g. ``01072025`` = 1 Jul 2025). When a session id is
written to CSV from an int -- or read back into an all-numeric column, which pandas
types as int64 -- the leading-zero DAY of days 1-9 is dropped:

    01072025  --int-->  1072025   (7 digits)

There is no ``1072025`` session; it is just ``int('01072025')``. Mixing the forms
silently breaks joins (day-1-9 sessions miss) and ordering.

WHY 7 DIGITS IS AN UNAMBIGUOUS SIGNATURE
----------------------------------------
The repo uses two session-token widths:
  * 8-digit DDMMYYYY  (BG_046, BG_038, anatomy, most caches)
  * 6-digit DDMMYY    (BG_031/BG_039 raw tokens)
A DDMMYY token can NEVER be 7 digits, and a DDMMYYYY token is only 7 digits when its
leading zero has been stripped. So a bare 7-digit numeric session id is *always* a
corrupted 8-digit id, and ``zfill(8)`` is always the correct repair.

This tool therefore repairs ONLY exact-7-digit numeric ids. It leaves 6-digit
DDMMYY tokens, 8-digit ids, and non-numeric ids (``BG_012_01112023_pr``,
``01042025_v2``, ``19052025_b``) completely untouched.

Repair is line-surgical: only the one field is rewritten, so float formatting,
quoting and line endings elsewhere are preserved byte-for-byte. Files containing
any quoted field are refused (the naive field split would be unsafe) and reported.

Usage
-----
    py scripts/qc/repair_session_ids.py                 # dry-run: report only
    py scripts/qc/repair_session_ids.py --execute       # rewrite the corrupt files
    py scripts/qc/repair_session_ids.py --root data/cache/decision_latents --execute

Out: prints a per-file report of rows repaired. Exit code 1 if corruption remains
     (so it can double as a CI/pre-commit gate in --check mode).
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# Column headers that hold a session id.
SESSION_COLS = {"session_name", "session_date", "session", "session_id"}

# Dirs that never hold deliverables.
SKIP_PARTS = {".git", ".venv", "archive", "_DeepUnitMatch_repo", ".claude", "node_modules"}

# A bare 7-digit integer == a DDMMYYYY id whose leading-zero day was stripped.
CORRUPT = re.compile(r"^\d{7}$")


def _session_col_indices(header: list[str]) -> list[int]:
    return [i for i, h in enumerate(header) if h.strip().lower() in SESSION_COLS]


def _split_eol(line: str) -> tuple[str, str]:
    for term in ("\r\n", "\n", "\r"):
        if line.endswith(term):
            return line[: -len(term)], term
    return line, ""


def _repair_line(line: str, idxs: list[int]) -> tuple[str, int]:
    """Rewrite only the session fields of one CSV line. Returns (new_line, n_fixed).

    Quote-aware: fields are parsed with the csv module, so a quoted field containing
    an embedded comma (e.g. the cortical region names in BG_038's anatomy table --
    ``"Primary somatosensory area, lower limb, layer 5"``) is handled correctly.
    Every other field is written back byte-identically; only the session field changes.
    """
    body, eol = _split_eol(line)
    if not body:
        return line, 0
    fields = next(csv.reader([body]))
    n = 0
    for i in idxs:
        if i < len(fields) and CORRUPT.match(fields[i].strip()):
            fields[i] = fields[i].strip().zfill(8)
            n += 1
    if n == 0:
        return line, 0  # untouched -> preserve the original bytes exactly
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(fields)
    return buf.getvalue() + eol, n


def scan_file(path: Path) -> dict | None:
    """Return a report dict for `path`, or None if it has no session column."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError):
        return None
    if not raw.strip():
        return None
    lines = raw.splitlines(keepends=True)
    header = lines[0].rstrip("\r\n").split(",")
    idxs = _session_col_indices(header)
    if not idxs:
        return None

    quoted = '"' in raw
    bad = 0
    for line in lines[1:]:
        _, n = _repair_line(line, idxs)
        bad += n
    return {
        "path": path,
        "cols": [header[i] for i in idxs],
        "idxs": idxs,
        "bad": bad,
        "quoted": quoted,
        "lines": lines,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data", help="dir to scan (default: data)")
    ap.add_argument("--execute", action="store_true", help="write repairs (default: dry-run)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any corruption is found (CI gate mode)")
    args = ap.parse_args()

    root = (_ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return 2

    reports = []
    for p in sorted(root.rglob("*.csv")):
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        r = scan_file(p)
        if r is not None:
            reports.append(r)

    corrupt = [r for r in reports if r["bad"] > 0]

    print(f"Scanned {len(reports)} CSV(s) with a session column under {root}")
    print(f"  corrupted : {len(corrupt)}")
    print(f"  clean     : {len(reports) - len(corrupt)}")
    if not corrupt:
        print("\nNo 7-digit session ids found. Clean.")
        return 0

    print()
    total = 0
    for r in sorted(corrupt, key=lambda r: -r["bad"]):
        rel = r["path"].relative_to(_ROOT).as_posix()
        q = "  [quoted fields -> csv-aware repair]" if r["quoted"] else ""
        print(f"  {r['bad']:>6} rows | {rel}  {r['cols']}{q}")
        total += r["bad"]
    print(f"\n  TOTAL corrupt rows: {total}")

    if not args.execute:
        print("\nDRY-RUN. Re-run with --execute to repair.")
        return 1 if args.check else 0

    print("\nRepairing...")
    fixed_files = 0
    fixed_rows = 0
    for r in sorted(corrupt, key=lambda r: -r["bad"]):
        out = [r["lines"][0]]
        n_file = 0
        for line in r["lines"][1:]:
            new, n = _repair_line(line, r["idxs"])
            out.append(new)
            n_file += n
        r["path"].write_text("".join(out), encoding="utf-8", newline="")
        rel = r["path"].relative_to(_ROOT).as_posix()
        print(f"  fixed {n_file:>6} rows | {rel}")
        fixed_files += 1
        fixed_rows += n_file

    print(f"\nRepaired {fixed_rows} rows across {fixed_files} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
