#!/usr/bin/env python
"""Run Kilosort 4 on a single (window × shank) run directory.

Designed to be called from a SLURM array job. Reads ks4_run_manifest.json
to find the run directory for the given 1-based job index.

This script is self-contained — it does NOT depend on the visdetect package.
It only requires: kilosort, numpy, torch (all in the kilosort4 conda env).

Usage:
    python run_kilosort4.py <job_index> --manifest <ks4_run_manifest.json>

    job_index is 1-based (matching SLURM_ARRAY_TASK_ID).

Examples:
    # Single job (e.g. for testing):
    python run_kilosort4.py 1 --manifest /ceph/.../ks4_runs/ks4_run_manifest.json

    # From SLURM:
    python run_kilosort4.py $SLURM_ARRAY_TASK_ID --manifest /ceph/.../ks4_run_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

import numpy as np


# ── Path translation (Windows ↔ Linux network paths) ─────────────────
# Extend this list if your network mount differs.

_PATH_MAPS = [
    ("X:/public/",  "/ceph/mrsic_flogel/public/"),
    ("X:\\public\\", "/ceph/mrsic_flogel/public/"),
]


def translate_path(p: str) -> str:
    """Translate Windows ↔ Linux network paths so manifests generated on one
    platform can be consumed on the other."""
    if platform.system() == "Linux":
        for win_prefix, linux_prefix in _PATH_MAPS:
            if p.startswith(win_prefix) or p.upper().startswith(win_prefix.upper()):
                p = linux_prefix + p[len(win_prefix):]
                break
        p = p.replace("\\", "/")
    else:
        for win_prefix, linux_prefix in _PATH_MAPS:
            if p.startswith(linux_prefix):
                p = win_prefix.replace("/", "\\") + p[len(linux_prefix):]
                break
    return p


# ── Main ──────────────────────────────────────────────────────────────

def run_ks4(
    job_index: int,
    manifest_path: str,
    batch_size: int = 60000,
    nblocks: int = 5,
    drift_smoothing: list[float] | None = None,
    Th_universal: float | None = None,
    Th_learned: float | None = None,
) -> int:
    """Run Kilosort 4 on one (window × shank) run directory.

    Parameters
    ----------
    job_index : int
        0-based index into the manifest's ``windows`` list.
    manifest_path : str
        Path to ks4_run_manifest.json produced by build_concat_windows.py.
    batch_size : int
        KS4 batch size in samples (default 60000 = 2 s at 30 kHz).
    nblocks : int
        Number of blocks for drift correction (default 5).
    drift_smoothing : list of float or None
        Temporal smoothing widths for drift correction, one per nblocks
        dimension (KS4 default: [0.5, 0.5, 0.5]).  Increase for small
        channel counts where per-batch drift estimates are noisy.
    Th_universal : float or None
        Spike detection threshold for universal templates (KS4 default: 9).
        Raise to detect fewer spikes and reduce clustering memory pressure.
    Th_learned : float or None
        Spike detection threshold for learned templates (KS4 default: 8).
        Raise to detect fewer spikes and reduce clustering memory pressure.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    import kilosort

    # ── Load manifest ──
    with open(manifest_path) as f:
        manifest = json.load(f)

    windows = manifest["windows"]
    if job_index < 0 or job_index >= len(windows):
        print(f"Error: job index {job_index} out of range "
              f"(total runs: {len(windows)})")
        return 1

    win = windows[job_index]
    window_idx = win["window_idx"]
    shank_id   = win["shank_id"]
    sessions   = win["sessions"]
    bin_paths  = [translate_path(p) for p in win["bin_paths"]]
    n_channels = win["n_channels"]
    sample_rate = float(manifest.get("sample_rate", 30000))
    run_dir    = Path(translate_path(win["run_dir"]))

    print(f"{'═' * 60}")
    print(f"Job {job_index + 1}/{len(windows)}:  Window {window_idx}  |  Shank {shank_id}")
    print(f"{'═' * 60}")
    print(f"  Sessions ({len(sessions)}): {', '.join(sessions)}")
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {win.get('total_duration_min', '?')} min")
    print(f"  Run dir:  {run_dir}")

    # ── Check completion marker ──
    complete_marker = run_dir / "ks4_complete.txt"
    if complete_marker.exists():
        print("\n  Already completed (ks4_complete.txt found). Skipping.")
        return 0

    # ── Verify binary files exist ──
    for bp in bin_paths:
        if not os.path.exists(bp):
            print(f"\n  ERROR: Binary not found: {bp}")
            return 1
        size_gb = os.path.getsize(bp) / 1e9
        print(f"  OK {bp}  ({size_gb:.2f} GB)")

    # ── GPU check & auto-scale batch_size ──
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU: {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")

        # Auto-scale batch_size to fit GPU VRAM.
        # 96-ch concat windows at batch_size=60000 need ~30-40 GB VRAM.
        # Scale linearly: default 60000 is calibrated for ~48 GB (L40S).
        if batch_size == 60000:  # only auto-scale if user didn't override
            if gpu_mem_gb < 18:
                batch_size = 15000
            elif gpu_mem_gb < 24:
                batch_size = 30000
            elif gpu_mem_gb < 44:
                batch_size = 45000
            # else keep 60000 for 48+ GB GPUs
            print(f"  Auto-scaled batch_size → {batch_size} (for {gpu_mem_gb:.0f} GB VRAM)")
    else:
        print("\n  WARNING: No GPU detected — this will be very slow!")

    # ── KS4 settings ──
    # IMPORTANT: results_dir MUST be set explicitly.  Without it KS4 uses
    # <first_filename>.parent / 'kilosort4' which causes all 4 shanks of a
    # window to overwrite each other's output.
    settings = {
        "data_dir":    str(run_dir),
        "results_dir": str(run_dir),
        "n_chan_bin":   n_channels,
        "fs":          sample_rate,
        "batch_size":  batch_size,
        "nblocks":     nblocks,
    }

    # Override drift smoothing if specified (increase for small channel counts)
    if drift_smoothing is not None:
        settings["drift_smoothing"] = drift_smoothing

    # Override spike detection thresholds if specified (reduces spike count
    # and therefore clustering GPU memory usage).
    if Th_universal is not None:
        settings["Th_universal"] = Th_universal
    if Th_learned is not None:
        settings["Th_learned"] = Th_learned

    # Use chanMap.mat from the run directory (copied there by build_concat_windows.py)
    chanmap_path = run_dir / "chanMap.mat"
    chan_positions_path = run_dir / "channel_positions.npy"
    chan_map_path = run_dir / "channel_map.npy"
    probe = None

    if chanmap_path.exists():
        settings["probe_path"] = str(chanmap_path)
        print(f"  Probe: chanMap.mat")
    elif chan_positions_path.exists():
        # Fallback: build probe from channel_positions.npy + channel_map.npy
        from kilosort.utils import Bunch
        chan_pos = np.load(str(chan_positions_path))
        if chan_map_path.exists():
            chan_map = np.load(str(chan_map_path)).astype(int)
        else:
            chan_map = np.arange(n_channels, dtype=int)
        probe = Bunch()
        probe.NchanTOT = n_channels
        probe.chanMap = chan_map
        probe.xc = chan_pos[:, 0].astype(np.float64)
        probe.yc = chan_pos[:, 1].astype(np.float64)
        probe.kcoords = np.zeros(n_channels, dtype=np.float64)
        print(f"  Probe: built from channel_positions.npy ({n_channels} ch)")
    else:
        # Second fallback: look in the source shank directory
        first_bin = Path(bin_paths[0])
        src_dir = first_bin.parent
        sess_name = src_dir.name
        shank_str = f"shank{shank_id}"
        src_pos = src_dir / f"{sess_name}_{shank_str}_channel_positions.npy"
        src_map = src_dir / f"{sess_name}_{shank_str}_channel_map.npy"
        src_chanmap = src_dir / f"{sess_name}_{shank_str}_chanMap.mat"
        if src_chanmap.exists():
            settings["probe_path"] = str(src_chanmap)
            print(f"  Probe: {src_chanmap.name} (from source dir)")
        elif src_pos.exists():
            from kilosort.utils import Bunch
            chan_pos = np.load(str(src_pos))
            if src_map.exists():
                chan_map = np.load(str(src_map)).astype(int)
            else:
                chan_map = np.arange(n_channels, dtype=int)
            probe = Bunch()
            probe.NchanTOT = n_channels
            probe.chanMap = chan_map
            probe.xc = chan_pos[:, 0].astype(np.float64)
            probe.yc = chan_pos[:, 1].astype(np.float64)
            probe.kcoords = np.zeros(n_channels, dtype=np.float64)
            print(f"  Probe: built from source dir channel_positions.npy ({n_channels} ch)")
        else:
            print(f"\n  ERROR: No probe/chanMap found in run dir or source dir")
            print(f"  Searched: {run_dir}")
            print(f"  Searched: {src_dir}")
            return 1

    print(f"  KS4: n_chan_bin={n_channels}, fs={sample_rate}, "
          f"batch_size={batch_size}, nblocks={nblocks}"
          f", drift_smoothing={drift_smoothing or [0.5, 0.5, 0.5]}"
          f", Th_universal={Th_universal or 9}, Th_learned={Th_learned or 8}")

    # ── Run Kilosort 4 ──
    t0 = datetime.now()
    print(f"\n  Starting Kilosort 4 at {t0.strftime('%H:%M:%S')} ...")

    try:
        # Multi-file mode: pass list of paths so KS4 sorts them as one
        # continuous recording and outputs spike_datasets.npy.
        ks_kwargs = {"settings": settings}
        if probe is not None:
            ks_kwargs["probe"] = probe
        if len(bin_paths) > 1:
            ks_kwargs["filename"] = bin_paths
        else:
            ks_kwargs["filename"] = str(bin_paths[0])
        kilosort.run_kilosort(**ks_kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Fatal GPU / runtime error — no recovery
        err_str = str(e)
        if isinstance(e, RuntimeError) and "out of memory" not in err_str.lower():
            # RuntimeError that is NOT an OOM — might still be fatal
            pass
        print(f"\n  !!! CUDA/Runtime ERROR in Window {window_idx} Shank {shank_id} !!!")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    except Exception as e:
        # Non-GPU error (e.g. matplotlib ValueError during post-sort plotting).
        # Check whether KS4 actually produced output before declaring failure.
        core_outputs = ["spike_times.npy", "spike_clusters.npy", "templates.npy"]
        have_output = all((run_dir / f).exists() for f in core_outputs)
        if have_output:
            print(f"\n  WARNING (non-fatal): {type(e).__name__}: {e}")
            print(f"  Sort output exists — treating as success despite post-sort error.")
            traceback.print_exc()
        else:
            print(f"\n  !!! ERROR in Window {window_idx} Shank {shank_id} !!!")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1

    t1 = datetime.now()
    elapsed = t1 - t0

    # ── Mark complete ──
    with open(complete_marker, "w") as f:
        f.write(f"Completed: {t1}\n")
        f.write(f"Elapsed: {elapsed}\n")
        f.write(f"Window: {window_idx}\n")
        f.write(f"Shank: {shank_id}\n")
        f.write(f"Sessions: {', '.join(sessions)}\n")
        f.write(f"N channels: {n_channels}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Nblocks: {nblocks}\n")
        f.write(f"Drift smoothing: {drift_smoothing or [0.5, 0.5, 0.5]}\n")
        f.write(f"Th_universal: {Th_universal or 9}\n")
        f.write(f"Th_learned: {Th_learned or 8}\n")

    print(f"\n  Finished at {t1.strftime('%H:%M:%S')}  (elapsed {elapsed})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run KS4 on a concat-sort (window × shank) run directory"
    )
    parser.add_argument(
        "job_index", type=int,
        help="1-based job index (use $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to ks4_run_manifest.json from build_concat_windows.py",
    )
    parser.add_argument(
        "--batch-size", type=int, default=60000,
        help="KS4 batch size in samples (default: 60000 = 2s at 30kHz)",
    )
    parser.add_argument(
        "--nblocks", type=int, default=5,
        help="KS4 nblocks for drift correction (default: 5)",
    )
    parser.add_argument(
        "--drift-smoothing", type=float, nargs=3, default=None,
        metavar=("S1", "S2", "S3"),
        help="Drift smoothing widths (3 floats, default: 0.5 0.5 0.5). "
             "Increase for small channel counts (e.g. 3.0 3.0 3.0 for 96 ch).",
    )
    parser.add_argument(
        "--Th-universal", type=float, default=None,
        help="Spike detection threshold for universal templates "
             "(KS4 default: 9). Raise to reduce clustering memory.",
    )
    parser.add_argument(
        "--Th-learned", type=float, default=None,
        help="Spike detection threshold for learned templates "
             "(KS4 default: 8). Raise to reduce clustering memory.",
    )
    args = parser.parse_args()

    # Convert 1-based SLURM index to 0-based Python index
    idx = args.job_index - 1
    sys.exit(run_ks4(
        idx, args.manifest, args.batch_size, args.nblocks,
        drift_smoothing=args.drift_smoothing,
        Th_universal=args.Th_universal,
        Th_learned=args.Th_learned,
    ))
