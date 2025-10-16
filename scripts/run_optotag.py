#!/usr/bin/env python
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

# ensure repo root is on sys.path so we can import src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.optotag.laser_detection import pulses_from_ni_events
from src.optotag.align_and_classify import (
    build_units_table,
    align_spikes_to_onsets,
    compute_psth,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pkl", help="session pickle (created by helper)")
    p.add_argument("--outdir", help="output directory", default=None)
    p.add_argument(
        "--split-blocks", action="store_true", help="detect separate pulse blocks"
    )
    p.add_argument(
        "--gap-threshold",
        type=float,
        default=0.5,
        help="gap threshold (s) to split blocks",
    )
    p.add_argument(
        "--window",
        type=float,
        nargs=2,
        default=[-0.1, 0.05],
        help="window (s) around pulse to analyze",
    )
    p.add_argument(
        "--plot-units",
        action="store_true",
        help="save per-unit PSTH and latency histogram plots",
    )
    p.add_argument(
        "--top-n", type=int, default=10, help="top N units to plot by reliability"
    )
    p.add_argument(
        "--waveforms-file",
        type=str,
        default=None,
        help="npz or npy file containing waveforms keyed by cluster id",
    )
    p.add_argument(
        "--wf-sr", type=float, default=30000.0, help="waveform sample rate (Hz)"
    )
    p.add_argument("--latency-ms", type=float, default=6.0)
    p.add_argument("--reliability", type=float, default=0.5)
    args = p.parse_args()

    pkl = Path(args.pkl)
    with open(pkl, "rb") as f:
        sess = pickle.load(f)

    outdir = (
        Path(args.outdir)
        if args.outdir
        else Path("notebooks") / pkl.stem / "optotagging"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    ni_events = (
        sess.get("ni_events")
        if isinstance(sess, dict)
        else getattr(sess, "ni_events", None)
    )
    pulses, sr = pulses_from_ni_events(ni_events)
    # save pulses CSV using csv module to avoid pandas
    pulses_csv = outdir / "pulses.csv"
    import csv

    keys = list(pulses.keys())
    # ensure values are arrays of same length
    n = len(pulses["onset_s"])
    with open(pulses_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            row = [np.asarray(pulses[k])[i] if k in pulses else "" for k in keys]
            writer.writerow(row)

    # get spikes
    if isinstance(sess, dict):
        clusters = sess["clusters"]
        # clusters: list of dicts with cluster_id and spike_times
        spike_times = np.concatenate([c["spike_times"] for c in clusters])
        spike_clusters = np.concatenate(
            [np.full(len(c["spike_times"]), c["cluster_id"]) for c in clusters]
        )
    else:
        clusters = getattr(sess, "clusters")
        spike_times = np.concatenate([c.spike_times for c in clusters])
        spike_clusters = np.concatenate(
            [np.full(len(c.spike_times), c.cluster_id) for c in clusters]
        )

    # convert spike times to seconds if they appear in samples (heuristic: median time > 1e3)
    if np.median(spike_times) > 1e3:
        spike_times_s = spike_times / sr
    else:
        spike_times_s = spike_times

    # optionally load waveforms
    waveforms = None
    if args.waveforms_file:
        wfpath = Path(args.waveforms_file)
        if wfpath.suffix == ".npz":
            data = np.load(wfpath, allow_pickle=True)
            waveforms = {int(k): data[k] for k in data.files}
        else:
            # try to load a dict saved as .npy
            arr = np.load(wfpath, allow_pickle=True)
            if isinstance(arr.item() if hasattr(arr, "item") else arr, dict):
                waveforms = {
                    int(k): np.asarray(v)
                    for k, v in (
                        arr.item().items() if hasattr(arr, "item") else arr.items()
                    )
                }
    # compute units table
    units_rows = build_units_table(
        spike_times_s,
        spike_clusters,
        pulses["onset_s"],
        window=tuple(args.window),
        latency_thresh_ms=args.latency_ms,
        reliability_thresh=args.reliability,
        waveforms=waveforms,
        wf_sample_rate=args.wf_sr,
    )
    units_csv = outdir / "units_optotagging.csv"
    import csv

    if len(units_rows) > 0:
        keys = list(units_rows[0].keys())
        with open(units_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for r in units_rows:
                writer.writerow([r[k] for k in keys])
    else:
        open(units_csv, "w").close()

    # Make a few plots: example raster for top 10 units by reliability
    # top units by reliability
    units_sorted = sorted(
        units_rows, key=lambda r: r.get("reliability", 0.0), reverse=True
    )
    top_units = [r["cluster_id"] for r in units_sorted[: args.top_n]]
    fig, axes = plt.subplots(
        len(top_units), 1, figsize=(6, 2 * len(top_units)), sharex=True
    )
    if len(top_units) == 1:
        axes = [axes]
    for ax, cid in zip(axes, top_units):
        mask = spike_clusters == cid
        st = spike_times_s[mask]
        aligned = align_spikes_to_onsets(st, pulses["onset_s"], tuple(args.window))
        # plot raster
        for i, rel in enumerate(aligned):
            ax.vlines(rel, i + 0.2, i + 0.8, color="k")
        ax.set_ylabel(f"clu {int(cid)}")
    axes[-1].set_xlabel("time (s) relative to pulse")
    plt.tight_layout()
    fig.savefig(outdir / "top_units_raster.png")

    print("Wrote:", pulses_csv, units_csv, outdir / "top_units_raster.png")

    # Per-unit plots: PSTH + latency histogram for opto-classified units and top N units
    if args.plot_units:
        opto_units = [
            r["cluster_id"] for r in units_rows if r.get("classification") == "opto"
        ]
        topn_units = top_units
        to_plot = sorted(set(opto_units) | set(topn_units))
        for cid in to_plot:
            mask = spike_clusters == cid
            st = spike_times_s[mask]
            aligned = align_spikes_to_onsets(st, pulses["onset_s"], tuple(args.window))
            centers, rates = compute_psth(
                aligned, bin_width=0.001, window=tuple(args.window)
            )
            ev_lo, ev_hi = (0.0, 0.01)
            latency_vals = [
                float(rel[(rel >= ev_lo) & (rel <= ev_hi)].min())
                for rel in aligned
                if ((rel >= ev_lo) & (rel <= ev_hi)).any()
            ]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=False)
            ax1.plot(centers, rates, color="C0")
            ax1.set_ylabel("Hz")
            ax1.set_title(f"Unit {int(cid)} PSTH")
            # small raster overlay (first 100 trials)
            for i, rel in enumerate(aligned[:100]):
                ax1.vlines(rel, i * 0.0 + 0.01, i * 0.0 + 0.02, color="k", alpha=0.3)

            if len(latency_vals) > 0:
                ax2.hist([lv * 1000.0 for lv in latency_vals], bins=25, color="C1")
            ax2.set_xlabel("Latency (ms)")
            ax2.set_ylabel("count")
            plt.tight_layout()
            fn = outdir / f"psth_latency_unit_{int(cid)}.png"
            fig.savefig(fn)
            plt.close(fig)

    # If pulse blocks were requested, compute per-block unit tables and save
    if args.split_blocks:
        # re-run pulses detection with block ids
        pulses_blk, _ = pulses_from_ni_events(ni_events, prefer_sample_rate=sr)
        block_ids = pulses_blk.get("block_id", None)
        if block_ids is None:
            print("No block ids detected")
        else:
            import csv

            unique_blocks = np.unique(block_ids)
            for b in unique_blocks:
                mask = block_ids == b
                block_onsets = np.asarray(pulses["onset_s"])[mask]
                rows_b = build_units_table(
                    spike_times_s,
                    spike_clusters,
                    block_onsets,
                    window=tuple(args.window),
                    latency_thresh_ms=args.latency_ms,
                    reliability_thresh=args.reliability,
                )
                csvp = outdir / f"units_block_{int(b)}.csv"
                if len(rows_b) > 0:
                    keys_b = list(rows_b[0].keys())
                    with open(csvp, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(keys_b)
                        for r in rows_b:
                            writer.writerow([r[k] for k in keys_b])
                else:
                    open(csvp, "w").close()


if __name__ == "__main__":
    main()
