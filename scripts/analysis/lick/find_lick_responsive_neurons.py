"""
Find lick-responsive neurons for a session using the logic in lick.py.

Usage:
    python scripts/analysis/lick/find_lick_responsive_neurons.py --session-pkl <session.pkl> --out <output_csv>

The output CSV will contain one row per cluster with columns:
    cluster_id, n_events, baseline_mean, post_mean, delta_mean, p_value, is_significant

Output is saved in the session's figures directory for downstream plotting.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

from visdetect.analysis import lick as lick_mod
from visdetect.core.session import load_session

def main():
    parser = argparse.ArgumentParser(description="Find lick-responsive neurons for a session.")
    parser.add_argument('--session-pkl', required=True, help='Path to session .pkl file')
    parser.add_argument('--out', required=True, help='Output CSV file for lick responsiveness table')
    args = parser.parse_args()

    session = load_session(args.session_pkl)
    res = lick_mod.compute_fa_lick_responsiveness(session)
    res.table.to_csv(args.out, index=False)
    print(f"Wrote lick responsiveness table to {args.out}")

    # Save per-cluster PSTHs, z-traces, and time axis for fast plotting
    # Output .npz file alongside CSV
    import numpy as np
    from pathlib import Path
    out_npz = str(Path(args.out).with_suffix('.npz'))
    # Build arrays: cluster_ids, mean_psth, z_trace, sem_psth (all clusters), time_axis
    # We need to recompute per-cluster PSTHs and z-traces
    analyzer = lick_mod.MatlabLickAnalyzer()
    t_vec, traces = analyzer.collect_unit_traces(session, show_progress=False)
    cluster_ids = np.array([tr.cluster_id for tr in traces], dtype=int)
    z_traces = np.stack([tr.z_trace for tr in traces]) if traces else np.empty((0, len(t_vec)))
    sem_traces = np.stack([tr.sem_trace for tr in traces]) if traces else np.empty((0, len(t_vec)))
    # For mean PSTH, use the un-zscored mean trace (if needed, can be added here)
    np.savez(out_npz,
             cluster_ids=cluster_ids,
             z_traces=z_traces,
             sem_traces=sem_traces,
             time_axis=t_vec)
    print(f"Wrote per-cluster lick traces to {out_npz}")

if __name__ == '__main__':
    main()
