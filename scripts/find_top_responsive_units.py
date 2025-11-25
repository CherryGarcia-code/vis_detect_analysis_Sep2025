"""Find top responsive good units for a session.

Usage (run inside repo conda env):
  python scripts/find_top_responsive_units.py path/to/session.pkl --qc-csv path/to/qc_filtered_typical_good.csv --out e:/.../demo_single_unit_BG_046_15082025/top_units.csv

The script computes pre/post firing rate changes around Change_ON and around behavioral 'Hit'/'Miss' events
and ranks units by (delta_post-pre for Hit) - (delta_post-pre for Miss) and by responsiveness to Change_ON.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'src'))

from visdetect.core.legacy_io import load_session
from visdetect.analysis import align as align_mod
from visdetect.analysis.su_analysis import plot_raster_psth, plot_change_rasters_by_outcome, plot_change_rasters_hit_by_size


def mean_rate_in_window(trials_mat, bin_centers, win):
    # trials_mat: n_trials x n_bins (Hz already)
    if trials_mat.size == 0:
        return np.array([])
    bmask = (bin_centers >= win[0]) & (bin_centers < win[1])
    if not bmask.any():
        return np.zeros(trials_mat.shape[0])
    return trials_mat[:, bmask].mean(axis=1)


def analyze(session_path, qc_csv_path=None, out_csv=None, out_dir=None, top_n=10, bin_size=0.01):
    session = load_session(session_path)
    out_dir = Path(out_dir) if out_dir is not None else Path(session_path).resolve().parents[1] / 'png_output' / 'top_units'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Good clusters: prefer provided qc csv, else use session.good_cluster_ids
    if qc_csv_path is not None and Path(qc_csv_path).exists():
        dfq = pd.read_csv(str(qc_csv_path))
        good_ids = list(dfq['cluster_id'].astype(int).tolist())
    else:
        good_ids = list(session.good_cluster_ids) if getattr(session, 'good_cluster_ids', None) else [c.cluster_id for c in session.clusters]

    # Event times
    hit_times = align_mod.get_event_times(session, 'Hit')
    miss_times = align_mod.get_event_times(session, 'Miss')
    # Change on: filter nan entries
    change_times_raw = session.ni_events.get('Change_ON', [])
    change_times = [float(x) for x in np.asarray(change_times_raw).flatten() if not (isinstance(x, float) and np.isnan(x))]

    results = []

    # windows for pre/post (relative to event zero)
    pre_win = (-0.5, 0.0)
    post_win = (0.0, 0.5)

    for c in session.clusters:
        cid = int(c.cluster_id)
        if cid not in good_ids:
            continue
        st = np.asarray(c.spike_times).flatten()

        # Hit
        trials_hit, bin_centers = align_mod.align_spikes_to_events(st, hit_times, window=(pre_win[0], post_win[1]), bin_size=bin_size)
        n_hit = int(trials_hit.shape[0]) if trials_hit.size else 0
        pre_hit_trials = mean_rate_in_window(trials_hit, bin_centers, pre_win) if n_hit>0 else np.array([])
        post_hit_trials = mean_rate_in_window(trials_hit, bin_centers, post_win) if n_hit>0 else np.array([])
        mean_pre_hit = float(np.nanmean(pre_hit_trials)) if pre_hit_trials.size>0 else np.nan
        mean_post_hit = float(np.nanmean(post_hit_trials)) if post_hit_trials.size>0 else np.nan
        delta_hit = mean_post_hit - mean_pre_hit if (not np.isnan(mean_post_hit) and not np.isnan(mean_pre_hit)) else np.nan

        # Miss
        trials_miss, _ = align_mod.align_spikes_to_events(st, miss_times, window=(pre_win[0], post_win[1]), bin_size=bin_size)
        n_miss = int(trials_miss.shape[0]) if trials_miss.size else 0
        pre_miss_trials = mean_rate_in_window(trials_miss, bin_centers, pre_win) if n_miss>0 else np.array([])
        post_miss_trials = mean_rate_in_window(trials_miss, bin_centers, post_win) if n_miss>0 else np.array([])
        mean_pre_miss = float(np.nanmean(pre_miss_trials)) if pre_miss_trials.size>0 else np.nan
        mean_post_miss = float(np.nanmean(post_miss_trials)) if post_miss_trials.size>0 else np.nan
        delta_miss = mean_post_miss - mean_pre_miss if (not np.isnan(mean_post_miss) and not np.isnan(mean_pre_miss)) else np.nan

        # Change
        trials_change, _ = align_mod.align_spikes_to_events(st, change_times, window=(pre_win[0], post_win[1]), bin_size=bin_size)
        n_change = int(trials_change.shape[0]) if trials_change.size else 0
        pre_change = mean_rate_in_window(trials_change, bin_centers, pre_win) if n_change>0 else np.array([])
        post_change = mean_rate_in_window(trials_change, bin_centers, post_win) if n_change>0 else np.array([])
        mean_pre_change = float(np.nanmean(pre_change)) if pre_change.size>0 else np.nan
        mean_post_change = float(np.nanmean(post_change)) if post_change.size>0 else np.nan
        delta_change = mean_post_change - mean_pre_change if (not np.isnan(mean_post_change) and not np.isnan(mean_pre_change)) else np.nan

        # Basic cluster metrics
        total_spikes = int(st.size)
        session_duration = 1.0
        # try to estimate session duration from ni_events baseline or change
        ni = getattr(session, 'ni_events', {}) or {}
        if 'Baseline_ON' in ni and np.asarray(ni['Baseline_ON']).size>0:
            session_duration = float(np.nanmax(np.asarray(ni['Baseline_ON']).flatten()) + 10.0)
        else:
            if st.size>0:
                session_duration = float(np.nanmax(st))

        mean_fr = float(total_spikes / session_duration) if session_duration>0 else np.nan

        results.append(
            {
                'cluster_id': cid,
                'total_spikes': total_spikes,
                'mean_firing_rate': mean_fr,
                'n_hit': n_hit,
                'mean_pre_hit': mean_pre_hit,
                'mean_post_hit': mean_post_hit,
                'delta_hit': delta_hit,
                'n_miss': n_miss,
                'mean_pre_miss': mean_pre_miss,
                'mean_post_miss': mean_post_miss,
                'delta_miss': delta_miss,
                'n_change': n_change,
                'mean_pre_change': mean_pre_change,
                'mean_post_change': mean_post_change,
                'delta_change': delta_change,
                'score_hit_minus_miss': (delta_hit - delta_miss) if (not np.isnan(delta_hit) and not np.isnan(delta_miss)) else np.nan,
            }
        )

    df = pd.DataFrame(results).sort_values('cluster_id').reset_index(drop=True)

    # Ranking
    df['rank_hit_minus_miss'] = df['score_hit_minus_miss'].abs().rank(method='min', ascending=False)
    df['rank_delta_change'] = df['delta_change'].abs().rank(method='min', ascending=False)

    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    else:
        df.to_csv(str(out_dir / 'top_units_summary.csv'), index=False)

    # Save PSTHs for top N by score_hit_minus_miss
    top_by_hit = df.sort_values('score_hit_minus_miss', key=lambda x: x.abs(), ascending=False).head(top_n)
    for idx, row in top_by_hit.iterrows():
        cid = int(row['cluster_id'])
        # save raster+psth aligned to Hit, Miss, Change
        try:
            png1 = out_dir / f'cluster_{cid}_hit_raster_psth.png'
            plot_raster_psth(session, cid, event_name='Hit', window=(pre_win[0], post_win[1]), bin_size=bin_size, save_path=str(png1))
        except Exception:
            pass
        try:
            png2 = out_dir / f'cluster_{cid}_miss_raster_psth.png'
            plot_raster_psth(session, cid, event_name='Miss', window=(pre_win[0], post_win[1]), bin_size=bin_size, save_path=str(png2))
        except Exception:
            pass
        try:
            png3 = out_dir / f'cluster_{cid}_change_raster_psth.png'
            # align to Change_ON but filter NaNs internally via align_mod
            plot_raster_psth(session, cid, event_name='Change_ON', window=(pre_win[0], post_win[1]), bin_size=bin_size, save_path=str(png3))
        except Exception:
            pass

        # New: Change_ON rasters split by outcome and by hit change size
        try:
            png4 = out_dir / f'cluster_{cid}_change_by_outcome_raster_psth.png'
            plot_change_rasters_by_outcome(session, cid, window=(pre_win[0], post_win[1]), bin_size=bin_size, save_path=str(png4))
        except Exception:
            pass
        try:
            subdir = out_dir / f'cluster_{cid}_hit_by_size'
            plot_change_rasters_hit_by_size(session, cid, window=(pre_win[0], post_win[1]), bin_size=bin_size, save_dir=str(subdir))
        except Exception:
            pass

    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('session', help='path to session pickle')
    p.add_argument('--qc-csv', default=None, help='path to qc_filtered_typical_good.csv (optional)')
    p.add_argument('--out-csv', default=None, help='path to write summary CSV')
    p.add_argument('--out-dir', default=None, help='output directory for PNGs and csv')
    p.add_argument('--top', type=int, default=10, help='top N units to save PSTHs for')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    out_dir = args.out_dir or (Path(args.session).resolve().parents[1] / 'png_output' / ('top_units_' + Path(args.session).stem))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_csv or str(out_dir / 'top_units_summary.csv')
    df = analyze(args.session, qc_csv_path=args.qc_csv, out_csv=out_csv, out_dir=out_dir, top_n=args.top)
    print('Wrote summary to', out_csv)
