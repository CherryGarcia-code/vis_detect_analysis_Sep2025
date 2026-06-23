"""Assemble DANT's multi-shank input folder from visdetect-extracted BG_046 data.

Reads per-session RawWaveforms + pkl spike trains, pools all good units across the 42
sessions, and writes DANT's expected .npy layout. Spike times are converted to ms;
positive-going units are excluded (DANT trough-centering assumes negative spikes).

Run with ANALYSIS_PY from the worktree root.
"""
import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapter  # noqa: E402

from visdetect.core.session import load_session  # noqa: E402
from visdetect.analysis.config import parse_session_date  # noqa: E402

PRIMARY = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
DEFAULT_UM_INPUT = os.path.join(PRIMARY, "data", "unit_match", "input", "BG_046")
DEFAULT_PKL_DIR = os.path.join(PRIMARY, "data", "pkls", "BG_046")
DEFAULT_OUT = "data/cache/dant/BG_046/input"


def _ks_ids_for_session(session_dir):
    """ks unit ids present as RawWaveforms in a session input dir, sorted."""
    rw = os.path.join(session_dir, "RawWaveforms")
    ids = []
    for fn in os.listdir(rw):
        if fn.startswith("Unit") and fn.endswith("_RawSpikes.npy"):
            ids.append(int(fn[len("Unit"):-len("_RawSpikes.npy")]))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--um-input", default=DEFAULT_UM_INPUT)
    ap.add_argument("--pkl-dir", default=DEFAULT_PKL_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--subject", default="BG_046")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "spike_times"), exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    sessions = [d for d in os.listdir(args.um_input)
                if os.path.isdir(os.path.join(args.um_input, d, "RawWaveforms"))]
    sessions = sorted(sessions, key=parse_session_date)
    log(f"{len(sessions)} sessions found; chronological order established.")

    waveforms = []
    session_index = []
    lookup_rows = []
    ref_channel_pos = None
    pooled = 0
    n_excluded_positive = 0
    n_missing_spikes = 0

    for s_idx, sdir in enumerate(sessions, start=1):
        spath = os.path.join(args.um_input, sdir)
        chan_pos = np.load(os.path.join(spath, "channel_positions.npy"))
        if ref_channel_pos is None:
            ref_channel_pos = chan_pos
        elif not np.array_equal(chan_pos, ref_channel_pos):
            raise ValueError(f"channel_positions for session {sdir} differ from session 1 "
                             f"({chan_pos.shape} vs {ref_channel_pos.shape}); pooled geometry ambiguous.")

        pkl_path = os.path.join(args.pkl_dir, f"{args.subject}_{sdir}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"missing pkl for session {sdir}: {pkl_path}")
        sess = load_session(pkl_path)
        spike_map = {int(c.cluster_id): np.asarray(c.spike_times) for c in sess.clusters}

        ks_ids = _ks_ids_for_session(spath)
        n_sess_units = 0
        for ks in ks_ids:
            raw = np.load(os.path.join(spath, "RawWaveforms", f"Unit{ks}_RawSpikes.npy"))
            wave = adapter.collapse_cv(raw)            # (383, 82)
            if adapter.is_positive_going(wave):
                n_excluded_positive += 1
                continue
            if ks not in spike_map:
                n_missing_spikes += 1
                log(f"  [skip] session {sdir} ks {ks}: no spike train in pkl")
                continue
            st_ms = adapter.seconds_to_ms(spike_map[ks])
            np.save(os.path.join(args.out_dir, "spike_times", f"Unit{pooled}.npy"), st_ms)
            waveforms.append(wave)
            session_index.append(s_idx)
            lookup_rows.append({"pooled_index": pooled, "session": sdir,
                                "ks_unit_id": ks, "session_index": s_idx})
            pooled += 1
            n_sess_units += 1

        log(f"  session {sdir} (idx {s_idx}): {n_sess_units} units")
        del sess
        gc.collect()

    waveform_all = np.stack(waveforms, axis=0)          # (n_unit, 383, 82)
    session_index = np.asarray(session_index, dtype=np.int64)
    channel_shanks = adapter.derive_channel_shanks(ref_channel_pos)

    # DANT requires contiguous 1..n_session
    uniq = np.unique(session_index)
    assert uniq.min() == 1 and len(uniq) == uniq.max(), f"session_index not contiguous: {uniq}"

    np.save(os.path.join(args.out_dir, "waveform_all.npy"), waveform_all)
    np.save(os.path.join(args.out_dir, "session_index.npy"), session_index)
    np.save(os.path.join(args.out_dir, "channel_locations.npy"), ref_channel_pos.astype(np.float64))
    np.save(os.path.join(args.out_dir, "channel_shanks.npy"), channel_shanks)
    pd.DataFrame(lookup_rows).to_csv(os.path.join(args.out_dir, "unit_lookup.csv"), index=False)

    log(f"DONE: {pooled} pooled units, {len(uniq)} sessions, waveform_all {waveform_all.shape}, "
        f"{int((channel_shanks.max()+1))} shanks.")
    log(f"Excluded positive-going: {n_excluded_positive}; missing spikes: {n_missing_spikes}.")
    with open(os.path.join(args.out_dir, "build_log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
