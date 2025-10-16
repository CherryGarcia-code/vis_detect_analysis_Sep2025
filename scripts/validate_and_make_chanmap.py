"""
Validate IMRO -> chanMap configuration and optionally write a Kilosort .mat

Usage examples:
    python scripts/validate_and_make_chanmap.py \
        --imro path/to/IMRO_Tables_BnB_DMS_BG_046.imro \
        --meta path/to/recording.meta \
        --bin path/to/recording.bin \
        --outmat chanMap_BG_046.mat --save-mat

What it checks:
- Parses IMRO entries (channel and bank) robustly.
- Reads .meta to find saved channel count and sample rate (tries common keys).
- Uses bin file size to compute number of samples and checks divisibility.
- Validates channel indices from IMRO exist within expected range.
- Builds chanMap arrays and a conservative x/y geometry (staggered 2-column), or uses the IMRO bank to distribute columns.
- Optionally saves a .mat compatible with Kilosort (fields: chanMap, chanMap0ind, connected, xcoords, ycoords, shankInd, fs, name)

Notes:
- Assumes int16 data in .bin (2 bytes per sample). Use --dtype if different.
- If .meta parsing fails for nchans or sample rate, provide --nchans and --fs manually.

"""

from __future__ import annotations
import argparse
import os
import re
import sys
from typing import List, Tuple

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt


def parse_imro(imro_path: str) -> List[Tuple[int, int, int, int, int]]:
    """Parse imro file and return list of 5-tuples:
    (channelID, shankID, bankID, refID, electrodeID).
    Robust to delimiters and stray chars. Returns an ordered list of tuples.
    """
    with open(imro_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tuples5: List[Tuple[int, int, int, int, int]] = []
    # Prefer explicit 5-number tuples like: (0 1 0 0 288)
    five_tuples = re.findall(r"\((\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+)\)", text)
    if five_tuples:
        for t in five_tuples:
            nums = [int(x) for x in re.findall(r"\d+", t)]
            if len(nums) == 5:
                ch, shank, bank, refid, elec = nums
                tuples5.append((ch, shank, bank, refid, elec))
        return tuples5

    # fallback: try to extract sequences of five ints across the file
    all_nums = [int(x) for x in re.findall(r"\d+", text)]
    for i in range(0, len(all_nums) - 4, 5):
        a, b, c, d, e = all_nums[i : i + 5]
        tuples5.append((a, b, c, d, e))
    return tuples5


def parse_meta(meta_path: str) -> dict:
    """Parse .meta file into key->value dict (strings).
    Also tries to return numeric fields for nchans and sample rate using common keys.
    """
    d = {}
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    # common numeric keys
    hints = {}
    for key in ("nSavedChans", "nChannels", "nChans", "num_channels", "n_channels"):
        if key in d:
            try:
                hints["nSavedChans"] = int(d[key])
                break
            except Exception:
                pass
    for key in (
        "sampleRate",
        "sr",
        "imSampRate",
        "niSampRate",
        "niSampRate",
        "sample_rate",
        "sampling_rate",
    ):
        if key in d:
            try:
                hints["fs"] = float(d[key])
                break
            except Exception:
                pass
    # spikeGLX style: "nSavedChans" etc. Another common is "niSamps" but that's samples count.
    return {**d, **hints}


def compute_n_samples_from_bin(
    bin_path: str, nchans: int, dtype: np.dtype
) -> Tuple[int, int]:
    size = os.path.getsize(bin_path)
    bytes_per_sample = np.dtype(dtype).itemsize
    if nchans <= 0:
        raise ValueError("nchans must be > 0")
    if size % (bytes_per_sample * nchans) != 0:
        return size // (bytes_per_sample * nchans), size
    return size // (bytes_per_sample * nchans), size


def build_chanmap_from_imro(
    tuples5: List[Tuple[int, int, int, int, int]], Nchannels: int, name="chanMap"
) -> dict:
    """Construct arrays for NP2.0 4-shank style probes.
    Expects a list of (channelID, shankID, bankID, refID, electrodeID) tuples.
    """
    chan_map = np.arange(1, Nchannels + 1, dtype=np.int32)  # 1-indexed
    chan_map0 = chan_map - 1
    connected = np.ones(Nchannels, dtype=bool)

    xcoords = np.zeros(Nchannels, dtype=np.float32)
    ycoords = np.zeros(Nchannels, dtype=np.float32)
    shankInd = np.ones(Nchannels, dtype=np.int32)
    kcoords = np.ones(Nchannels, dtype=np.int32)

    # Layout parameters
    inter_shank = 250.0  # lateral spacing between shanks (um)
    col_positions = [43.0, 27.0, 59.0, 11.0]  # per-column x offsets (um)

    # collect observed shank ids to determine base (0- or 1-based)
    shank_ids = sorted(set([t[1] for t in tuples5]))
    shank_min = min(shank_ids) if shank_ids else 0

    for ch, shank, bank, refid, elec in tuples5:
        site = ch + 1  # convert to 1-based index for array placement
        if site < 1 or site > Nchannels:
            continue
        # normalize shank to 1-based indices for shankInd/kcoords
        shank_idx = int(shank - shank_min + 1)
        shankInd[site - 1] = shank_idx
        kcoords[site - 1] = shank_idx

        # electrode id gives row/column within shank for NP2.0
        # row index (0-based) and column index (0..3)
        row = int(elec) // 4
        col_idx = int(elec) % 4
        ycoords[site - 1] = float(row) * 20.0
        xcoords[site - 1] = (
            float(col_positions[col_idx]) + (shank_idx - 1) * inter_shank
        )

    return {
        "chanMap": chan_map,
        "chanMap0ind": chan_map0,
        "connected": connected,
        "xcoords": xcoords,
        "ycoords": ycoords,
        "shankInd": shankInd,
        "kcoords": kcoords,
        "name": name,
    }


def validate(pairs, Nchannels, meta_info, bin_info, dtype):
    problems = []
    channels_in_imro = [p[0] for p in pairs]
    if len(set(channels_in_imro)) != len(channels_in_imro):
        problems.append("Duplicate channel entries in IMRO")
    # check range
    low = min(channels_in_imro) if channels_in_imro else None
    high = max(channels_in_imro) if channels_in_imro else None
    if low is not None and (low < 0 or (high is not None and high >= Nchannels)):
        problems.append(
            f"IMRO channels out of range: found {low}..{high} but expected 0..{Nchannels - 1}"
        )

    # check bin divisibility
    n_samples, size = bin_info
    if size % (np.dtype(dtype).itemsize * Nchannels) != 0:
        problems.append(
            f"Bin file size ({size}) not divisible by bytes_per_sample ({np.dtype(dtype).itemsize}) * nchans ({Nchannels})"
        )
    # compare meta nchans if present
    meta_n = meta_info.get("nSavedChans") or meta_info.get("nChannels")
    if meta_n is not None:
        if int(meta_n) != Nchannels:
            problems.append(
                f"Meta indicates nchans={meta_n} but provided/assumed Nchannels={Nchannels}"
            )
    # fs
    fs_meta = meta_info.get("fs")
    if fs_meta is None:
        # check common keys raw
        for k in ("imSampRate", "sampleRate", "sr"):
            if k in meta_info:
                try:
                    fs_meta = float(meta_info[k])
                    break
                except Exception:
                    pass
    return problems


def plot_chanmap(xcoords, ycoords, connected, title="chanMap"):
    fig, ax = plt.subplots(figsize=(3.5, 8))
    mask = connected
    ax.scatter(xcoords[mask], -ycoords[mask], s=10, c="C0", label="connected")
    ax.scatter(
        xcoords[~mask], -ycoords[~mask], s=10, c="C3", alpha=0.4, label="disabled"
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig, ax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imro", required=True, help="Path to IMRO file (.imro or text)")
    p.add_argument(
        "--meta", required=True, help="Path to .meta file from SpikeGLX or similar"
    )
    p.add_argument("--bin", required=True, help="Path to raw .bin data file")
    p.add_argument(
        "--outmat", default=None, help="Output .mat filename to save chanMap"
    )
    p.add_argument("--save-mat", action="store_true", help="Actually save the .mat")
    p.add_argument(
        "--nchans", type=int, default=None, help="Number of channels (override meta)"
    )
    p.add_argument(
        "--fs", type=float, default=None, help="Sampling rate (override meta)"
    )
    p.add_argument(
        "--dtype",
        default="int16",
        help="Binary dtype (default int16). Use e.g. int16, uint16, int32",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Show a simple scatter plot of the channel layout",
    )
    p.add_argument(
        "--plot-out",
        default=None,
        help="If set, save the plot to this filename (PNG) instead of showing interactively",
    )

    args = p.parse_args()

    pairs = parse_imro(args.imro)
    if not pairs:
        print("No (channel,bank) pairs found in IMRO. Exiting.", file=sys.stderr)
        sys.exit(2)
    print(f"Parsed {len(pairs)} pairs from IMRO (showing first 8): {pairs[:8]}")

    meta = parse_meta(args.meta)
    print("Parsed meta keys:", ", ".join(list(meta.keys())[:10]))

    nchans_meta = None
    if args.nchans is not None:
        Nchannels = args.nchans
        print(f"Using --nchans override: {Nchannels}")
    else:
        if "nSavedChans" in meta:
            Nchannels = int(meta["nSavedChans"])
        elif "nChannels" in meta:
            Nchannels = int(meta["nChannels"])
        elif "nChans" in meta:
            Nchannels = int(meta["nChans"])
        else:
            # fallback assumption: 384 (Neuropixels 1.0) or 385 if sync
            print(
                "Warning: could not detect nchans in .meta; defaulting to 384",
                file=sys.stderr,
            )
            Nchannels = 384
    if args.fs is not None:
        fs = args.fs
    else:
        fs = meta.get("fs", None)
        if fs is None and "imSampRate" in meta:
            try:
                fs = float(meta["imSampRate"])
            except Exception:
                fs = None
    print(f"Assumed Nchannels={Nchannels}, fs={fs}")

    dtype = np.dtype(args.dtype)
    n_samples, size = compute_n_samples_from_bin(args.bin, Nchannels, dtype)
    print(
        f"Bin file size {size} bytes -> {n_samples} samples @ {Nchannels} channels, dtype={dtype}"
    )

    problems = validate(pairs, Nchannels, meta, (n_samples, size), dtype)
    if problems:
        print("Validation issues found:")
        for pr in problems:
            print(" -", pr)
    else:
        print("No validation problems detected (basic checks).")

    chanmap = build_chanmap_from_imro(
        pairs, Nchannels, name=os.path.basename(args.outmat or "chanMap")
    )
    # pack fields for savemat
    matdict = {
        "chanMap": np.asarray(chanmap["chanMap"], dtype=np.int32),
        "chanMap0ind": np.asarray(chanmap["chanMap0ind"], dtype=np.int32),
        "connected": np.asarray(chanmap["connected"], dtype=np.uint8),
        "xcoords": np.asarray(chanmap["xcoords"], dtype=np.float32),
        "ycoords": np.asarray(chanmap["ycoords"], dtype=np.float32),
        "shankInd": np.asarray(chanmap["shankInd"], dtype=np.int32),
        "fs": float(fs) if fs is not None else 30000.0,
        "name": chanmap["name"],
    }

    if args.save_mat:
        out = args.outmat or ("chanMap_from_imro.mat")
        savemat(out, matdict)
        print(f"Wrote {out} with fields: {list(matdict.keys())}")
    else:
        print(
            "Run with --save-mat to write a .mat file. Here is a summary of generated arrays:"
        )
        print("chanMap length:", len(matdict["chanMap"]))
        print("connected count (True):", int(np.sum(matdict["connected"])))
        print(
            "xcoords min/max:",
            np.nanmin(matdict["xcoords"]),
            np.nanmax(matdict["xcoords"]),
        )
        print(
            "ycoords min/max:",
            np.nanmin(matdict["ycoords"]),
            np.nanmax(matdict["ycoords"]),
        )

    if args.plot:
        fig, ax = plot_chanmap(
            matdict["xcoords"],
            matdict["ycoords"],
            matdict["connected"],
            title=os.path.basename(args.outmat or "chanMap"),
        )
        if args.plot_out:
            fig.savefig(args.plot_out, dpi=150)
            print("Wrote plot to", args.plot_out)
        else:
            plt.show()


if __name__ == "__main__":
    main()
