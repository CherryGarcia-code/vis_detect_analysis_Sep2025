#!/usr/bin/env python
"""Archived: use the copy in `scripts/archived_lick_scripts/`.

This file was archived because `visdetect.utils.matlab_ports.lick` is the
canonical implementation for lick analyses. A backup of the original
implementation is available at:

    scripts/archived_lick_scripts/plot_lick_for_tf_survivors.py

Running this stub will print a message and exit with code 1.
"""
import sys

if __name__ == "__main__":
        print(
                "This script has been archived. Use scripts/archived_lick_scripts/plot_lick_for_tf_survivors.py instead.",
                file=sys.stderr,
        )
        raise SystemExit(1)
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time
import random
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visdetect.core.session import load_session
from visdetect.analysis.tf_pulse import TFRespPulseConfig, collect_tf_pulse_traces
from visdetect.analysis.lick import MatlabLickConfig, collect_fa_lick_traces
from visdetect.utils.progress import Progress


def _plot_lick_grid(t_vec, entries, out_png: Path, n_cols: int, z_line: float) -> None:
    if not entries:
        print("No matching clusters for lick plot; skipping", out_png)
        return
    n = len(entries)
    n_cols = max(1, n_cols)
    n_rows = int((n + n_cols - 1) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), sharex=True, sharey=True)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, entry in enumerate(entries):
        ax = axes[idx]
        ax.plot(t_vec, entry.z_trace, color="#4e79a7")
        ax.fill_between(t_vec, entry.z_trace - entry.sem_trace, entry.z_trace + entry.sem_trace, color="#4e79a7", alpha=0.25, linewidth=0)
        ax.axvline(0.0, color="k", linestyle="--", lw=0.8)
        ax.axhline(z_line, color="#888", linestyle=":", lw=0.7)
        ax.axhline(-z_line, color="#888", linestyle=":", lw=0.7)
        ax.set_title(f"clu {entry.cluster_id}", fontsize=8)
        if idx % n_cols == 0:
            ax.set_ylabel("z-score")
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("time (s)")

    for j in range(idx + 1, n_rows * n_cols):
        axes[j].axis("off")

    fig.suptitle("FA lick responses (z-scored)", y=0.995)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot FA lick PSTHs for TF survivor clusters")
    ap.add_argument("--file", required=False, default=None, help="Path to session pickle (omit with --synthetic)")
    ap.add_argument("--tf-thresh", type=float, default=3.0, help="|z| threshold for TF survivors")
    ap.add_argument("--cols", type=int, default=10, help="Columns in grid plot")
    ap.add_argument("--kept-only", action="store_true", help="Restrict to kept units")
    ap.add_argument("--out", default="png_output/lick_tf_survivors", help="Output directory root")
    ap.add_argument("--progress", action="store_true", help="Show detailed progress for all stages")
    ap.add_argument("--synthetic", action="store_true", help="Use a synthetic lightweight session for quick testing")
    ap.add_argument("--syn-clusters", type=int, default=8, help="Synthetic: number of clusters")
    ap.add_argument("--syn-trials", type=int, default=120, help="Synthetic: number of trials")
    ap.add_argument(
        "--limit-good",
        type=int,
        default=None,
        help="Debug: limit number of good clusters processed for lick PSTHs",
    )
    args = ap.parse_args(argv)

    if args.synthetic:
        if args.progress:
            print("[Stage 0] Building synthetic session ...", flush=True)
        session = _build_synthetic_session(args.syn_clusters, args.syn_trials)
    else:
        if not args.file:
            print("Error: --file is required unless --synthetic is used", flush=True)
            return 1
        print(f"Loading session from {args.file} ...", flush=True)
        session = _load_with_keepalive(args.file, keepalive_interval=2.0 if args.progress else None)
    ident = f"{getattr(session,'subject','unknown')}_{getattr(session,'session_name','unknown')}"
    out_dir = Path(args.out) / ident
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.progress:
        print("[Stage 1] Collecting TF pulse traces ...", flush=True)
    tf_cfg = TFRespPulseConfig(kept_only=args.kept_only)
    _, tf_entries = collect_tf_pulse_traces(session, cfg=tf_cfg, show_progress=args.progress)
    if args.progress:
        print(f"[Stage 2] Filtering TF survivors from {len(tf_entries)} units ...", flush=True)
    survivor_ids = []
    for e in tf_entries:
        fast_hit = np.isfinite(e.z_max_fast) and e.z_max_fast >= args.tf_thresh
        slow_hit = np.isfinite(e.z_min_slow) and (-e.z_min_slow) >= args.tf_thresh
        if fast_hit or slow_hit:
            survivor_ids.append(int(e.cluster_id))
    survivor_ids = sorted(set(survivor_ids))
    if not survivor_ids:
        print(f"No TF survivors at threshold {args.tf_thresh}; nothing to plot")
        return 0
    if args.progress:
        print(f"[Stage 2] Identified {len(survivor_ids)} TF survivors (|z|>={args.tf_thresh}).")

    lick_cfg = MatlabLickConfig()
    # Prefer good_and_stable_ids, then good_cluster_ids, else all clusters
    if getattr(session, "good_and_stable_ids", None):
        cluster_id_list = list(session.good_and_stable_ids)
    elif getattr(session, "good_cluster_ids", None):
        cluster_id_list = list(session.good_cluster_ids)
    else:
        cluster_id_list = [c.cluster_id for c in session.clusters]

    # Optionally limit number of clusters for debug
    if args.limit_good is not None:
        limited = list(cluster_id_list)[: args.limit_good]
        cluster_id_list = limited
        print(f"Debug limit active: restricting lick PSTHs to {len(limited)} clusters.")

    if args.progress:
        print("[Stage 3] Computing lick-aligned PSTHs ...", flush=True)
    t_vec, lick_entries = collect_fa_lick_traces(
        session,
        cfg=lick_cfg,
        good_ids=cluster_id_list,
        show_progress=args.progress,
    )
    entry_lookup = {entry.cluster_id: entry for entry in lick_entries}
    matched = [entry_lookup[cid] for cid in survivor_ids if cid in entry_lookup]

    if not matched:
        print("TF survivors not found among lick traces; nothing to plot")
        return 0

    survivor_png = out_dir / f"lick_tf_survivors_z{args.tf_thresh:g}.png"
    if args.progress:
        print(f"[Stage 4] Plotting {len(matched)} survivor lick traces ...", flush=True)
    _plot_lick_grid(t_vec, matched, survivor_png, args.cols, args.tf_thresh)
    print(f"Saved lick plot: {survivor_png}")
    return 0


def _load_with_keepalive(path: str | Path, keepalive_interval: float | None = None):
    """Load session with periodic keepalive messages if interval provided."""
    done = threading.Event()
    if keepalive_interval is not None:
        def _keepalive():
            spinner = ['|','/','-','\\']
            idx = 0
            start = time.time()
            while not done.is_set():
                print(f"[loading] {spinner[idx%4]} {time.time()-start:5.1f}s", flush=True)
                idx += 1
                done.wait(keepalive_interval)
        th = threading.Thread(target=_keepalive, daemon=True)
        th.start()
    try:
        session = load_session(str(path))
    finally:
        done.set()
    return session


def _build_synthetic_session(n_clusters: int, n_trials: int):
    """Construct a minimal synthetic session object.
    Provides required attributes for TF and lick analyses while being fast.
    """
    trials = []
    baseline_on_times = []
    change_on_times = []
    for i in range(n_trials):
        t0 = i * 3.0
        baseline_on_times.append(t0)
        change_on_times.append(t0 + 2.0)
        if random.random() < 0.5:
            outcome = 'FA'
            rt_val = 3.3 + random.random()*0.7
            rts = {'FA': rt_val}
        else:
            outcome = 'hit'
            rts = {}
        baseline_values = (np.random.rand(45) * 40.0) + 1.0
        trials.append(SimpleNamespace(
            trialoutcome=outcome,
            reactiontimes=rts,
            baseline_values=baseline_values,
            n_seen=45,
        ))
    clusters = []
    cluster_id_list = []
    for cid in range(n_clusters):
        spikes = []
        for bt in baseline_on_times:
            spikes.extend((bt + np.random.normal(0,0.02,size=5)).tolist())
            if random.random() < 0.4:
                burst_t = bt + 3.5
                spikes.extend((burst_t + np.random.normal(0,0.005,size=15)).tolist())
        clusters.append(SimpleNamespace(cluster_id=cid, spike_times=np.array(sorted(spikes), dtype=float)))
        cluster_id_list.append(cid)
    session = SimpleNamespace(
        trials=trials,
        ni_events={'Baseline_ON': baseline_on_times, 'Change_ON': change_on_times},
        clusters=clusters,
        good_cluster_ids=cluster_id_list,  # keep for compatibility
        subject='SYN',
        session_name='synthetic',
    )
    return session


if __name__ == "__main__":
    raise SystemExit(main())
