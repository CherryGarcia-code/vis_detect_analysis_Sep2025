"""Build overlapping concatenation windows and prepare KS4 run directories.

Reads the shank-split manifest, creates sliding windows of sessions, and
generates a params.py + channel map for each window × shank so KS4 can be
launched directly on each run directory.

KS4's multi-file mode is used: params.py contains a Python list of binary
paths in dat_path.  KS4 will sort them as one continuous recording and
output spike_datasets.npy to track file membership.

Usage:
    python scripts/pipelines/concat_sort/build_concat_windows.py \
        --shank-manifest data/concat_sort/shank_split/shank_split_manifest.json \
        --window-size 5 --stride 1 \
        --output-dir data/concat_sort/ks4_runs
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Path translation (Windows ↔ Linux) ───────────────────────────────
_PATH_MAPS = [
    ("X:/public/",  "/ceph/mrsic_flogel/public/"),
    ("X:\\public\\", "/ceph/mrsic_flogel/public/"),
]


def _translate_to_local(p: str) -> str:
    """Translate a path to the local platform so file operations work."""
    if platform.system() == "Windows":
        for win_prefix, linux_prefix in _PATH_MAPS:
            if p.startswith(linux_prefix):
                p = win_prefix + p[len(linux_prefix):]
                break
        p = p.replace("/", "\\")
    else:
        for win_prefix, linux_prefix in _PATH_MAPS:
            if p.startswith(win_prefix) or p.upper().startswith(win_prefix.upper()):
                p = linux_prefix + p[len(win_prefix):]
                break
        p = p.replace("\\", "/")
    return p


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(str(date_str).strip(), "%d%m%Y")


def write_params_py(
    run_dir: Path,
    bin_paths: List[str],
    n_channels: int,
    sample_rate: float = 30000.0,
    dtype: str = "int16",
) -> Path:
    """Write a KS4-compatible params.py with multi-file dat_path."""
    # Format dat_path as a Python list literal with forward slashes
    path_strs = ", ".join(f'"{p.replace(chr(92), "/")}"' for p in bin_paths)
    dat_path_line = f"dat_path = [{path_strs}]"

    content = f"""{dat_path_line}
n_channels_dat = {n_channels}
dtype = '{dtype}'
offset = 0
sample_rate = {sample_rate}
hp_filtered = True
"""
    out = run_dir / "params.py"
    out.write_text(content)
    return out


def build_windows(
    shank_manifest: dict,
    window_size: int,
    stride: int,
    output_dir: Path,
) -> dict:
    """Build overlapping windows and prepare KS4 run directories.

    Returns a master run manifest dict.
    """
    # Sort sessions chronologically
    sessions_dict = shank_manifest["sessions"]
    session_names = sorted(
        sessions_dict.keys(),
        key=lambda s: _parse_date(sessions_dict[s]["date"]),
    )

    n_total = len(session_names)
    print(f"Total sessions: {n_total}, window_size={window_size}, stride={stride}")

    if n_total < window_size:
        print(f"WARNING: Only {n_total} sessions, less than window_size={window_size}. "
              f"Creating a single window with all sessions.")

    # Determine which shanks are available (from first session)
    first_sess = sessions_dict[session_names[0]]
    shank_ids = sorted(first_sess["shanks"].keys())
    sample_rate = first_sess["sample_rate"]
    print(f"Shanks: {shank_ids} | Sample rate: {sample_rate} Hz")

    run_manifest = {
        "subject": shank_manifest.get("subject", "unknown"),
        "window_size": window_size,
        "stride": stride,
        "sample_rate": sample_rate,
        "shank_ids": shank_ids,
        "n_sessions": n_total,
        "session_order": session_names,
        "windows": [],
    }

    window_idx = 0
    start = 0
    while start <= n_total - window_size:
        end = start + window_size
        window_sessions = session_names[start:end]

        for shank_id in shank_ids:
            run_dir = output_dir / f"window_{window_idx:03d}" / f"shank_{shank_id}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Collect per-shank binary paths and metadata for this window
            bin_paths = []
            total_samples = 0
            n_channels = None

            for sname in window_sessions:
                sess = sessions_dict[sname]
                shank_info = sess["shanks"][shank_id]
                bin_paths.append(shank_info["path"])
                total_samples += shank_info["n_samples"]
                if n_channels is None:
                    n_channels = shank_info["n_channels"]

            # Write params.py
            write_params_py(run_dir, bin_paths, n_channels, sample_rate)

            # Copy channel map and positions from the first session's shank
            first_sess_data = sessions_dict[window_sessions[0]]
            first_shank = first_sess_data["shanks"][shank_id]
            # Translate path to local platform so the file can be found
            first_shank_dir = Path(_translate_to_local(first_shank["path"])).parent
            first_session_name = window_sessions[0]

            for suffix, dst_name in [
                ("_channel_positions.npy", "channel_positions.npy"),
                ("_channel_map.npy",      "channel_map.npy"),
                ("_chanMap.mat",          "chanMap.mat"),
            ]:
                src = first_shank_dir / f"{first_session_name}_shank{shank_id}{suffix}"
                dst = run_dir / dst_name
                if src.exists():
                    shutil.copy2(str(src), str(dst))
                else:
                    print(f"  WARNING: source not found: {src}")

            total_duration = total_samples / sample_rate

            run_manifest["windows"].append({
                "window_idx": window_idx,
                "shank_id": shank_id,
                "sessions": window_sessions,
                "session_dates": [sessions_dict[s]["date"] for s in window_sessions],
                "run_dir": str(run_dir),
                "bin_paths": bin_paths,
                "n_channels": n_channels,
                "total_samples": total_samples,
                "total_duration_sec": round(total_duration, 2),
                "total_duration_min": round(total_duration / 60, 1),
            })

            print(f"  Window {window_idx} / Shank {shank_id}: "
                  f"{len(window_sessions)} sessions, "
                  f"{n_channels} ch, "
                  f"{total_duration / 60:.1f} min → {run_dir}")

        window_idx += 1
        start += stride

    print(f"\nTotal KS4 runs to launch: {len(run_manifest['windows'])}")
    return run_manifest


def main(argv=None):
    p = argparse.ArgumentParser(description="Build overlapping concatenation windows for KS4")
    p.add_argument("--shank-manifest", type=Path, required=True,
                   help="shank_split_manifest.json from step 2")
    p.add_argument("--window-size", type=int, default=5,
                   help="Number of sessions per concatenation window")
    p.add_argument("--stride", type=int, default=1,
                   help="Stride between windows (1 = maximum overlap)")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "data" / "concat_sort" / "ks4_runs")
    args = p.parse_args(argv)

    with open(args.shank_manifest) as f:
        shank_manifest = json.load(f)

    run_manifest = build_windows(
        shank_manifest, args.window_size, args.stride, args.output_dir
    )

    manifest_path = args.output_dir / "ks4_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"Run manifest → {manifest_path}")

    # Also print a helper: how to run KS4 on each directory
    print("\n── To run KS4 on each window ──")
    print("For each run directory, launch:")
    print("  python -c \"import kilosort; kilosort.run_kilosort(settings={'data_dir': '<run_dir>'})\"")
    print("Or use SpikeInterface / your preferred KS4 launcher.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
