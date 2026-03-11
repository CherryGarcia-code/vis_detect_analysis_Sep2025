"""Split Neuropixels 2.0 4-shank AP binaries into per-shank files.

For each selected session, reads the CatGT-processed AP binary and its
.meta / IMRO table, groups the 384 AP channels by shank ID (0-3), and
writes four per-shank binary files plus corresponding channel maps.

Usage:
    python scripts/pipelines/concat_sort/split_by_shank.py \
        --sessions-json data/concat_sort/learning_session_selection.json \
        --processed-data-root "X:\\public\\projects\\BeJG_20230130_VisDetect\\wEPhys\\BG_046\\Processed data" \
        --output-dir data/concat_sort/shank_split \
        --chunk-seconds 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import savemat

REPO_ROOT = Path(__file__).resolve().parents[3]

# ── SpikeGLX metadata helpers (adapted from chanMap_related/) ──────────

def read_meta(meta_path: Path) -> dict:
    """Parse SpikeGLX .meta file into key-value dict."""
    meta = {}
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            parts = line.split("=", 1)
            key = parts[0].lstrip("~")
            meta[key] = parts[1]
    return meta


def get_ap_channel_count(meta: dict) -> int:
    """Return the number of AP channels from snsApLfSy."""
    counts = meta["snsApLfSy"].split(",")
    return int(counts[0])


def get_saved_channel_count(meta: dict) -> int:
    return int(meta["nSavedChans"])


def get_sample_rate(meta: dict) -> float:
    return float(meta["imSampRate"])


def parse_imro(meta: dict) -> List[Tuple[int, int, int, int, int]]:
    """Extract IMRO 5-tuples from the ~imroTbl field in the metadata.

    Returns list of (channelID, shankID, bankID, refID, electrodeID).
    """
    imro_text = meta.get("imroTbl", "")
    tuples5 = []
    five_tuples = re.findall(r"\((\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+)\)", imro_text)
    if five_tuples:
        for t in five_tuples:
            nums = [int(x) for x in re.findall(r"\d+", t)]
            if len(nums) == 5:
                tuples5.append(tuple(nums))
    return tuples5


def parse_imro_file(imro_path: str) -> List[Tuple[int, int, int, int, int]]:
    """Parse standalone .imro file."""
    with open(imro_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tuples5 = []
    five_tuples = re.findall(r"\((\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+)\)", text)
    for t in five_tuples:
        nums = [int(x) for x in re.findall(r"\d+", t)]
        if len(nums) == 5:
            tuples5.append(tuple(nums))
    return tuples5


# ── Geometry helpers (consistent with chanMap_related/SGLXMetaToCoords.py) ──

# Probe geometry lookup table from SGLXMetaToCoords.getGeomParams
# [nShank, shankWidth, shankPitch, even_xOff, odd_xOff, horizPitch, vertPitch, rowsPerShank, elecPerShank]
_GEOM_PARAMS = {
    # NP1.0 probes
    "3A":                [1,  70,   0,  27, 11, 32, 20, 480,  960],
    "PRB_1_4_0480_1":    [1,  70,   0,  27, 11, 32, 20, 480,  960],
    "PRB_1_4_0480_1_C":  [1,  70,   0,  27, 11, 32, 20, 480,  960],
    "NP1010":            [1,  70,   0,  27, 11, 32, 20, 480,  960],
    "NP1011":            [1,  70,   0,  27, 11, 32, 20, 480,  960],
    "NP1012":            [1,  70,   0,  27, 11, 32, 20, 480,  960],
    "NP1013":            [1,  70,   0,  27, 11, 32, 20, 480,  960],
    # NP2.0 single-shank
    "PRB2_1_4_0480_1":   [1,  70,   0,  27, 27, 32, 15, 640, 1280],
    "PRB2_1_2_0640_0":   [1,  70,   0,  27, 27, 32, 15, 640, 1280],
    "NP2000":            [1,  70,   0,  27, 27, 32, 15, 640, 1280],
    "NP2003":            [1,  70,   0,  27, 27, 32, 15, 640, 1280],
    "NP2004":            [1,  70,   0,  27, 27, 32, 15, 640, 1280],
    # NP2.0 4-shank
    "PRB2_4_2_0640_0":   [4,  70, 250,  27, 27, 32, 15, 640, 1280],
    "PRB2_4_4_0480_1":   [4,  70, 250,  27, 27, 32, 15, 640, 1280],
    "NP2010":            [4,  70, 250,  27, 27, 32, 15, 640, 1280],
    "NP2013":            [4,  70, 250,  27, 27, 32, 15, 640, 1280],
    "NP2014":            [4,  70, 250,  27, 27, 32, 15, 640, 1280],
}


def get_geom_params(meta: dict) -> List:
    """Return geometry parameter list for the probe, matching SGLXMetaToCoords.getGeomParams."""
    pn = meta.get("imDatPrb_pn", "3A")
    if pn in _GEOM_PARAMS:
        return _GEOM_PARAMS[pn]
    raise ValueError(f"Unsupported probe part number: {pn}")


def geom_map_to_coords(meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse snsGeomMap for per-channel coords — preferred path for modern metadata.

    Returns (shankInd, xCoord, yCoord, connected) arrays, one entry per saved AP channel.
    Matches SGLXMetaToCoords.geomMapToGeom exactly.
    """
    geom_map = meta["snsGeomMap"].split(")")
    n_entry = len(geom_map) - 2  # subtract header and trailing ')'
    shank_ind = np.zeros(n_entry)
    x_coord = np.zeros(n_entry)
    y_coord = np.zeros(n_entry)
    connected = np.zeros(n_entry)
    for i in range(n_entry):
        entry = geom_map[i + 1].lstrip("(")
        parts = entry.split(":")
        shank_ind[i] = int(parts[0])
        x_coord[i] = float(parts[1])
        y_coord[i] = float(parts[2])
        connected[i] = int(parts[3])
    return shank_ind, x_coord, y_coord, connected


def shank_map_to_coords(meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse snsShankMap + probe geometry for per-channel coords — fallback path.

    Returns (shankInd, xCoord, yCoord, connected) arrays.
    Matches SGLXMetaToCoords.shankMapToGeom exactly.
    """
    ap_count = get_ap_channel_count(meta)
    shank_map = meta["snsShankMap"].split(")")
    shank_ind = np.zeros(ap_count)
    col_ind = np.zeros(ap_count)
    row_ind = np.zeros(ap_count)
    connected = np.zeros(ap_count)

    for i in range(ap_count):
        entry = shank_map[i + 1].lstrip("(")
        parts = entry.split(":")
        shank_ind[i] = int(parts[0])
        col_ind[i] = int(parts[1])
        row_ind[i] = int(parts[2])
        connected[i] = int(parts[3])

    geom = get_geom_params(meta)
    # geom = [nShank, shankWidth, shankPitch, even_xOff, odd_xOff, horizPitch, vertPitch, ...]
    odd_rows = (row_ind % 2).astype(bool)
    even_rows = ~odd_rows
    x_coord = col_ind * float(geom[5])  # horizPitch
    x_coord[even_rows] += geom[3]       # even_xOff
    x_coord[odd_rows] += geom[4]        # odd_xOff
    y_coord = row_ind * float(geom[6])  # vertPitch

    return shank_ind, x_coord, y_coord, connected


def get_channel_coords(meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get per-channel coordinates from metadata, following SGLXMetaToCoords priority.

    Prefers snsGeomMap (modern SpikeGLX), falls back to snsShankMap + geometry params.
    Returns (shankInd, xCoord, yCoord, connected) arrays — one entry per saved AP channel.
    x coordinates are shank-local (no shank pitch offset applied).
    """
    if "snsGeomMap" in meta:
        return geom_map_to_coords(meta)
    elif "snsShankMap" in meta:
        return shank_map_to_coords(meta)
    else:
        raise ValueError("Metadata has neither snsGeomMap nor snsShankMap — cannot compute coordinates.")


def group_channels_by_shank(
    tuples5: List[Tuple[int, int, int, int, int]],
) -> Dict[int, List[int]]:
    """Return {shankID: [list of channelIDs]} sorted by channelID."""
    groups: Dict[int, List[int]] = {}
    for ch, shank, bank, refid, elec in tuples5:
        groups.setdefault(shank, []).append(ch)
    for k in groups:
        groups[k].sort()
    return groups


def build_shank_chanmap(
    shank_channels: List[int],
    meta: dict,
    shank_ind_all: np.ndarray,
    x_all: np.ndarray,
    y_all: np.ndarray,
    connected_all: np.ndarray,
    sample_rate: float,
    shank_id: int,
) -> dict:
    """Build a Kilosort-compatible channel map dict for a single shank.

    Channel indices are re-mapped to 0..N-1 for the shank-local binary.
    Coordinates are derived from metadata (via SGLXMetaToCoords-compatible parsing).
    """
    n_ch = len(shank_channels)

    chan_map = np.arange(1, n_ch + 1, dtype=np.float64)   # 1-indexed (matches SGLXMetaToCoords)
    chan_map0 = np.arange(n_ch, dtype=np.float64)          # 0-indexed
    conn = np.ones((n_ch, 1), dtype=np.float64)
    xcoords = np.zeros((n_ch, 1), dtype=np.float64)
    ycoords = np.zeros((n_ch, 1), dtype=np.float64)

    for i, ch in enumerate(shank_channels):
        # x_all already has shank-local x from snsGeomMap/snsShankMap
        xcoords[i, 0] = x_all[ch]
        ycoords[i, 0] = y_all[ch]
        conn[i, 0] = connected_all[ch]

    # kcoords: SGLXMetaToCoords uses shankInd+1, but per-shank file has 1 shank → all 1s
    kcoords = np.ones((n_ch, 1), dtype=np.float64)

    return {
        "chanMap": chan_map.reshape(n_ch, 1),
        "chanMap0ind": chan_map0.reshape(n_ch, 1),
        "connected": conn,
        "xcoords": xcoords,
        "ycoords": ycoords,
        "kcoords": kcoords,
        "fs": np.float64(sample_rate),
        "name": f"shank{shank_id}",
    }


# ── Binary splitting ──────────────────────────────────────────────────

def find_session_ap_bin(processed_root: Path, session_name: str) -> Tuple[Path, Path]:
    """Locate the CatGT AP binary and its .meta for a session.

    Searches under: processed_root / *{session_name}* / Kilosort&Phy / *imec0 /
    Returns (bin_path, meta_path).
    """
    candidates = list(processed_root.glob(f"*{session_name}*"))
    for sess_dir in candidates:
        ks_phy = sess_dir / "Kilosort&Phy"
        if not ks_phy.exists():
            continue
        imec_dirs = list(ks_phy.glob("*imec0"))
        if not imec_dirs:
            continue
        imec_dir = imec_dirs[0]
        # Look for the binary
        bins = list(imec_dir.glob("*.ap.bin"))
        if not bins:
            continue
        bin_path = bins[0]
        meta_path = bin_path.with_suffix("").with_suffix(".meta")
        # Also try .ap.meta directly
        if not meta_path.exists():
            meta_path = Path(str(bin_path).replace(".ap.bin", ".ap.meta"))
        return bin_path, meta_path
    raise FileNotFoundError(
        f"Could not find AP binary for session {session_name} under {processed_root}"
    )


def split_session(
    bin_path: Path,
    meta: dict,
    shank_groups: Dict[int, List[int]],
    output_dir: Path,
    session_name: str,
    sample_rate: float,
    n_saved_chans: int,
    shank_ind_all: np.ndarray,
    x_all: np.ndarray,
    y_all: np.ndarray,
    connected_all: np.ndarray,
    chunk_seconds: float = 30.0,
) -> Dict[int, dict]:
    """Split one session's AP binary into per-shank files.

    Returns {shank_id: {path, n_channels, n_samples, duration_sec, channel_indices}}.
    """
    file_size = os.path.getsize(bin_path)
    bytes_per_sample = 2  # int16
    bytes_per_timepoint = n_saved_chans * bytes_per_sample
    n_samples = file_size // bytes_per_timepoint
    chunk_samples = int(chunk_seconds * sample_rate)

    # Open per-shank output files
    shank_files = {}
    shank_info = {}
    for shank_id, channels in sorted(shank_groups.items()):
        n_ch = len(channels)
        out_path = output_dir / f"{session_name}_shank{shank_id}.ap.bin"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shank_files[shank_id] = open(out_path, "wb")
        shank_info[shank_id] = {
            "path": str(out_path),
            "n_channels": n_ch,
            "n_samples": int(n_samples),
            "duration_sec": round(n_samples / sample_rate, 3),
            "channel_indices": channels,
        }

        # Save channel map .mat (SGLXMetaToCoords-compatible)
        cm = build_shank_chanmap(
            channels, meta, shank_ind_all, x_all, y_all, connected_all,
            sample_rate, shank_id,
        )
        mat_path = output_dir / f"{session_name}_shank{shank_id}_chanMap.mat"
        savemat(str(mat_path), cm)

        # Save channel_positions.npy for KS4
        positions = np.column_stack([cm["xcoords"], cm["ycoords"]])
        np.save(str(output_dir / f"{session_name}_shank{shank_id}_channel_positions.npy"), positions)

        # Save channel_map.npy for KS4
        np.save(str(output_dir / f"{session_name}_shank{shank_id}_channel_map.npy"), cm["chanMap0ind"].flatten())

    # Stream-copy in chunks
    raw = np.memmap(str(bin_path), dtype=np.int16, mode="r")
    raw = raw.reshape(n_samples, n_saved_chans)

    processed = 0
    while processed < n_samples:
        end = min(processed + chunk_samples, n_samples)
        chunk = np.array(raw[processed:end, :])  # read into RAM
        for shank_id, channels in sorted(shank_groups.items()):
            shank_data = chunk[:, channels]
            shank_files[shank_id].write(shank_data.tobytes())
        processed = end
        pct = 100.0 * processed / n_samples
        print(f"    {processed}/{n_samples} samples ({pct:.1f}%)", end="\r")

    print()  # newline after progress

    # Close files
    for f in shank_files.values():
        f.close()

    # Verify sizes
    for shank_id, info in shank_info.items():
        expected_bytes = info["n_channels"] * info["n_samples"] * 2
        actual_bytes = os.path.getsize(info["path"])
        if actual_bytes != expected_bytes:
            print(f"  WARNING: shank {shank_id} size mismatch: expected {expected_bytes}, got {actual_bytes}")
        else:
            print(f"  Shank {shank_id}: {info['n_channels']} ch × {info['n_samples']} samples → {actual_bytes / 1e9:.2f} GB  ✓")

    return shank_info


# ── Skip / completeness check ────────────────────────────────────────

def session_is_complete(output_dir: Path, session_name: str, n_shanks: int = 4) -> bool:
    """Check if a session's shank split output already exists and is complete.

    Requires all per-shank .ap.bin and _chanMap.mat files to exist with non-zero size.
    """
    sess_dir = output_dir / session_name
    if not sess_dir.exists():
        return False
    for shank_id in range(n_shanks):
        bin_path = sess_dir / f"{session_name}_shank{shank_id}.ap.bin"
        mat_path = sess_dir / f"{session_name}_shank{shank_id}_chanMap.mat"
        if not bin_path.exists() or not mat_path.exists():
            return False
        if bin_path.stat().st_size == 0:
            return False
    return True


def _build_session_manifest_entry(
    name: str, sess: dict, bin_path: Path,
    sr: float, n_saved: int, shank_groups: Dict[int, List[int]],
    output_dir: Path,
) -> dict:
    """Build the manifest dict entry for one session (used both when
    processing and when back-filling a skipped session's fragment)."""
    file_size = os.path.getsize(bin_path)
    n_samples = file_size // (n_saved * 2)
    shank_entries = {}
    for shank_id, channels in sorted(shank_groups.items()):
        shank_bin = output_dir / name / f"{name}_shank{shank_id}.ap.bin"
        shank_entries[str(shank_id)] = {
            "path": str(shank_bin),
            "n_channels": len(channels),
            "n_samples": int(n_samples),
            "duration_sec": round(n_samples / sr, 3),
            "channel_indices": channels,
        }
    return {
        "session_name": sess["session_name"],
        "date": sess["date"],
        "source_bin": str(bin_path),
        "sample_rate": sr,
        "n_samples": int(n_samples),
        "shanks": shank_entries,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Split NP2.0 4-shank AP binaries into per-shank files")
    p.add_argument("--sessions-json", type=Path, required=True,
                   help="Path to learning_session_selection.json from step 1")
    p.add_argument("--processed-data-root", type=Path, required=True,
                   help="Root of Processed data (e.g. X:\\...\\BG_046\\Processed data)")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "data" / "concat_sort" / "shank_split")
    p.add_argument("--chunk-seconds", type=float, default=30.0,
                   help="Streaming chunk size in seconds (controls memory usage)")
    p.add_argument("--imro-file", type=Path, default=None,
                   help="Optional standalone .imro file (overrides meta-embedded IMRO)")
    p.add_argument("--session-index", type=int, default=None,
                   help="1-based session index for SLURM array parallelism. "
                        "Processes only this one session.")
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false",
                   help="Re-process sessions even if output already exists")
    p.set_defaults(skip_existing=True)
    p.add_argument("--merge-manifests", action="store_true",
                   help="Merge per-session manifest_fragment.json files into "
                        "the master shank_split_manifest.json and exit.")
    args = p.parse_args(argv)

    with open(args.sessions_json) as f:
        selection = json.load(f)

    sessions = selection["sessions"]
    subject = selection.get("subject", "BG_046")

    # ── Merge mode ────────────────────────────────────────────────────
    if args.merge_manifests:
        fragments = sorted(args.output_dir.glob("*/manifest_fragment.json"))
        manifest = {
            "subject": subject,
            "processed_data_root": str(args.processed_data_root),
            "sessions": {},
        }
        for frag_path in fragments:
            with open(frag_path) as f:
                frag = json.load(f)
            manifest["sessions"].update(frag)
        manifest_path = args.output_dir / "shank_split_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Merged {len(fragments)} session fragments → {manifest_path}")
        return 0

    # ── Determine which sessions to process ───────────────────────────
    if args.session_index is not None:
        idx = args.session_index - 1  # convert to 0-based
        if idx < 0 or idx >= len(sessions):
            print(f"ERROR: --session-index {args.session_index} out of "
                  f"range [1, {len(sessions)}]")
            return 1
        sessions_to_process = [(idx, sessions[idx])]
    else:
        sessions_to_process = list(enumerate(sessions))

    manifest = {
        "subject": subject,
        "processed_data_root": str(args.processed_data_root),
        "sessions": {},
    }

    # Parse IMRO once (all sessions use the same IMRO for BG_046)
    tuples5 = None
    shank_groups = None
    # Metadata-derived coordinates (populated on first session)
    shank_ind_all = None
    x_all = None
    y_all = None
    connected_all = None

    n_total = len(sessions_to_process)
    n_skipped = 0
    n_errors = 0

    for rank, (i, sess) in enumerate(sessions_to_process):
        name = f"{subject}_{sess['session_name']}"
        print(f"\n[{rank + 1}/{n_total}] Processing {name}")

        # Always locate source binary (needed for metadata / skip-fragment)
        try:
            bin_path, meta_path = find_session_ap_bin(
                args.processed_data_root, sess["session_name"])
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            n_errors += 1
            continue

        meta = read_meta(meta_path)
        n_saved = get_saved_channel_count(meta)
        sr = get_sample_rate(meta)
        n_ap = get_ap_channel_count(meta)

        # Parse IMRO for shank grouping — only once since all sessions share it
        if tuples5 is None:
            if args.imro_file and args.imro_file.exists():
                tuples5 = parse_imro_file(str(args.imro_file))
                print(f"  IMRO from file: {args.imro_file} ({len(tuples5)} entries)")
            else:
                tuples5 = parse_imro(meta)
                print(f"  IMRO from meta: {len(tuples5)} entries")
            if not tuples5:
                print("  ERROR: Could not parse IMRO table. Provide --imro-file.")
                return 1
            shank_groups = group_channels_by_shank(tuples5)
            for sid, chs in sorted(shank_groups.items()):
                print(f"    Shank {sid}: {len(chs)} channels")

        # Get coordinates from metadata (SGLXMetaToCoords-compatible)
        if shank_ind_all is None:
            shank_ind_all, x_all, y_all, connected_all = get_channel_coords(meta)
            geom_source = "snsGeomMap" if "snsGeomMap" in meta else "snsShankMap"
            print(f"  Coordinates from {geom_source}: {len(x_all)} channels")
            for sid in sorted(shank_groups.keys()):
                xs = x_all[list(shank_groups[sid])]
                print(f"    Shank {sid}: x ∈ [{xs.min():.1f}, {xs.max():.1f}] µm, "
                      f"y ∈ [{y_all[list(shank_groups[sid])].min():.1f}, "
                      f"{y_all[list(shank_groups[sid])].max():.1f}] µm")

        # ── Skip check (after metadata so we can back-fill fragment) ──
        if args.skip_existing and session_is_complete(args.output_dir, name):
            frag_path = args.output_dir / name / "manifest_fragment.json"
            if not frag_path.exists():
                # Back-fill fragment for sessions completed by earlier runs
                entry = _build_session_manifest_entry(
                    name, sess, bin_path, sr, n_saved, shank_groups,
                    args.output_dir)
                frag_path.parent.mkdir(parents=True, exist_ok=True)
                with open(frag_path, "w") as f:
                    json.dump({name: entry}, f, indent=2)
                print(f"  SKIP (complete; wrote retroactive fragment)")
            else:
                print(f"  SKIP (already complete)")
            n_skipped += 1
            continue

        print(f"  Binary: {bin_path}")
        print(f"  Channels: {n_ap} AP + {n_saved - n_ap} sync  |  SR: {sr} Hz")

        sess_output_dir = args.output_dir / name
        info = split_session(
            bin_path, meta, shank_groups, sess_output_dir, name,
            sr, n_saved, shank_ind_all, x_all, y_all, connected_all,
            args.chunk_seconds,
        )

        sess_entry = {
            "session_name": sess["session_name"],
            "date": sess["date"],
            "source_bin": str(bin_path),
            "sample_rate": sr,
            "n_samples": info[0]["n_samples"] if info else 0,
            "shanks": {str(k): v for k, v in info.items()},
        }
        manifest["sessions"][name] = sess_entry

        # Write per-session manifest fragment (for parallel merge)
        frag_path = sess_output_dir / "manifest_fragment.json"
        with open(frag_path, "w") as f:
            json.dump({name: sess_entry}, f, indent=2)
        print(f"  Fragment → {frag_path}")

    # ── Summary ───────────────────────────────────────────────────────
    n_processed = n_total - n_skipped - n_errors
    print(f"\nDone: {n_processed} processed, {n_skipped} skipped, "
          f"{n_errors} errors")

    # In sequential mode (no --session-index), write the combined manifest
    if args.session_index is None:
        # Incorporate fragments from skipped sessions too
        if n_skipped > 0:
            for frag_path in sorted(
                    args.output_dir.glob("*/manifest_fragment.json")):
                with open(frag_path) as f:
                    frag = json.load(f)
                manifest["sessions"].update(frag)
        if manifest["sessions"]:
            manifest_path = args.output_dir / "shank_split_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Manifest written → {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
