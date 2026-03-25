"""Scan session .meta files and group sessions by IMRO channel map.

Recursively finds *_imec0.ap.meta files under a subject root directory,
extracts the ``imroFile`` field from each, groups sessions by channel map,
and writes per-channel-map CSV summaries.

Usage:
    py scripts/kilosort_related/check_session_chanmaps.py BG_031 --path X:/public/.../BG_031
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_meta_files(subject_root: str) -> List[Tuple[str, str]]:
    """Recursively find all .meta files matching ``*_imec0.ap.meta``.

    Parameters
    ----------
    subject_root : str
        Root directory to search under.

    Returns
    -------
    list of (session_name, meta_file_path) tuples
    """
    meta_files: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(subject_root):
        for f in files:
            if f.endswith('imec0.ap.meta'):
                # Session name is the first part of the file name before '_g'
                match = re.match(r'(.*?)_g', f)
                session_name = match.group(1) if match else f.split('_g')[0]
                meta_files.append((session_name, os.path.join(root, f)))
    return meta_files


def extract_imro_file(meta_path: str) -> str:
    """Extract the ``imroFile`` field from a SpikeGLX .meta file.

    Reads the .meta file line by line and returns the filename component
    of the ``imroFile=`` entry (stripping any directory prefix).

    Parameters
    ----------
    meta_path : str
        Full path to the .meta file.

    Returns
    -------
    str
        The IMRO filename, or empty string if not found or on error.
    """
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('imroFile='):
                    imro_full = line.strip().split('=', 1)[1]
                    return os.path.basename(imro_full.replace('\\', '/'))
    except Exception as e:
        print(f"WARNING: Could not read meta file {meta_path}: {e}")
    return ''


def group_by_imro(results: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group session result rows by their IMRO file.

    Parameters
    ----------
    results : list of dict
        Each dict has 'session' and 'imro_file' keys.

    Returns
    -------
    dict mapping imro_file -> list of result dicts
    """
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in results:
        grouped[row['imro_file']].append(row)
    return grouped


def write_grouped_csvs(
    subject: str,
    grouped_results: Dict[str, List[Dict[str, str]]],
    out_dir: Optional[Path] = None,
) -> None:
    """Write per-IMRO-channel-map CSV summaries.

    Parameters
    ----------
    subject : str
        Subject name (e.g. ``BG_031``), used in output filenames.
    grouped_results : dict
        Mapping from IMRO filename to list of session result dicts.
    out_dir : Path or None
        Output directory.  Defaults to ``data/subject_session_imro_matching/{subject}/``.
    """
    if out_dir is None:
        out_dir = Path('data/subject_session_imro_matching') / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    for imro_file, rows in grouped_results.items():
        if imro_file:
            imro_stem = Path(imro_file).stem
            safe_imro_name = re.sub(r'[^\w\-. ]', '_', imro_stem)
            out_csv = out_dir / f'{subject}_{safe_imro_name}_sessions.csv'
        else:
            out_csv = out_dir / f'{subject}_unknown_imro_sessions.csv'

        with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['session', 'imro_file'])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Wrote summary to {out_csv}")


def main(subject: str, subject_root: str) -> None:
    """Scan meta files and write IMRO-grouped CSV summaries.

    Parameters
    ----------
    subject : str
        Subject name (e.g. ``BG_031``).
    subject_root : str
        Full path to the subject data directory.
    """
    meta_files = find_meta_files(subject_root)
    if not meta_files:
        print(f"No *_imec0.ap.meta files found under {subject_root}")
        return

    results: List[Dict[str, str]] = []
    for session_name, meta_path in meta_files:
        imro_file = extract_imro_file(meta_path)
        results.append({'session': session_name, 'imro_file': imro_file})

    grouped_results = group_by_imro(results)
    write_grouped_csvs(subject, grouped_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Scan session .meta files and group by IMRO channel map.",
    )
    parser.add_argument(
        "subject",
        help="Subject name (e.g. BG_031)",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Full path to the subject data directory",
    )
    args = parser.parse_args()
    main(args.subject, args.path)
