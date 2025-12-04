"""Plot state-specific population activity aligned to events across sessions.

Computes mean +/- SEM of z-scored population activity for each session and state
for events: FA-lick, change_time, and Baseline_ON. Outputs heatmaps (sessions x time)
and per-session means CSVs for each event.

Usage:
    python scripts/analysis/plot_state_activity_across_sessions.py --out FIGURES --workers 4
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from visdetect.core.legacy_io import load_session
from visdetect.analysis.lick import MatlabLickConfig


OUT_ROOT = REPO / 'FIGURES'
STATE_DIR = REPO / 'group_meeting_27112025' / 'state_output'
RESP_DIR = REPO / 'group_meeting_27112025' / 'responsiveness_output'
MANIFEST = REPO / 'data' / 'BG_046_sessions_manifest.csv'


def _build_psth_matrix(spikes: np.ndarray, events: list, edges: np.ndarray):
    if events is None or len(events) == 0:
        return None
    n_bins = edges.size - 1
    mat = np.zeros((len(events), n_bins), dtype=float)
    for i, et in enumerate(events):
        rel = spikes - float(et)
        mask = (rel >= edges[0]) & (rel < edges[-1])
        if not np.any(mask):
            continue
        counts, _ = np.histogram(rel[mask], bins=edges)
        mat[i] = counts
    return mat


def _smooth_mean(matrix: np.ndarray, smooth_bins: int):
    from scipy import ndimage
    if matrix is None or matrix.size == 0:
        return None, None
    sigma = smooth_bins / 6.0 if smooth_bins > 1 else 0.0
    if sigma > 0:
        sm = ndimage.gaussian_filter1d(matrix, sigma=sigma, axis=1, mode='nearest')
    else:
        sm = matrix
    mean = np.nanmean(sm, axis=0)
    sem = np.nanstd(sm, axis=0, ddof=1) / np.sqrt(sm.shape[0]) if sm.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem


def _zscore_trace(mean_trace: np.ndarray, t_vec: np.ndarray, baseline_window: tuple):
    mask = (t_vec >= baseline_window[0]) & (t_vec < baseline_window[1])
    if not np.any(mask):
        return np.zeros_like(mean_trace), float('nan')
    mu = float(np.nanmean(mean_trace[mask]))
    sd = float(np.nanstd(mean_trace[mask]))
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(mean_trace), float('nan')
    z = (mean_trace - mu) / sd
    return z, np.nanmax(z[(t_vec >= -0.3) & (t_vec < 0)]) if np.any((t_vec >= -0.3) & (t_vec < 0)) else float('nan')


def process_session(session_name: str, event: str, state: str, cfg: MatlabLickConfig, min_events_override: int | None = None):
    """Compute population mean z-trace for one session, event, and state.
    Returns (session, t_vec, mean, sem) or (session, None, None, None) if insufficient.
    """
    try:
        # load session
        # try pkls then data
        pkl = REPO / 'pkls' / f'{session_name}.pkl'
        if not pkl.exists():
            pkl = REPO / 'data' / f'{session_name}.pkl'
            if not pkl.exists():
                return session_name, None, None, None
        session = load_session(str(pkl))

        # load state labels
        state_csv = STATE_DIR / f'{session_name}_trial_state_labels.csv'
        if not state_csv.exists():
            return session_name, None, None, None
        state_df = pd.read_csv(state_csv)

        # get trial indices for this state (ensure ints)
        if 'trial_id' in state_df.columns:
            sel_trials = state_df.loc[state_df['state'] == state, 'trial_id'].astype(int).values
        else:
            # fallback: use row numbers where state matches
            sel_trials = state_df.index[state_df['state'] == state].astype(int).to_numpy()
        if len(sel_trials) == 0:
            return session_name, None, None, None

        cfg = cfg
        edges = cfg.time_edges
        t_vec = cfg.time_centers

        # Build event times depending on event
        events = []
        trials = session.trials
        # try get baseline array
        raw_baseline = session.ni_events.get('Baseline_ON', []) if getattr(session, 'ni_events', None) is not None else []
        baseline = None
        if isinstance(raw_baseline, dict):
            if 'rise_t' in raw_baseline:
                baseline = np.asarray(raw_baseline.get('rise_t', []), dtype=float).flatten()
            elif 'times' in raw_baseline:
                baseline = np.asarray(raw_baseline.get('times', []), dtype=float).flatten()
            else:
                baseline = np.asarray([], dtype=float)
        else:
            baseline = np.asarray(raw_baseline, dtype=float).flatten()

        for tidx in sel_trials:
            if tidx >= len(trials):
                continue
            tr = trials[int(tidx)]
            if event == 'lick':
                out = (tr.trialoutcome or '').lower()
                if out != 'fa':
                    continue
                rt = tr.reactiontimes.get('FA') if tr.reactiontimes else None
                if rt is None or not np.isfinite(rt):
                    continue
                if tidx < baseline.size:
                    ev = float(baseline[int(tidx)] + float(rt))
                else:
                    continue
                if float(rt) < cfg.min_fa_delay:
                    continue
                events.append(ev)
            elif event == 'change':
                ct = getattr(tr, 'change_time', None)
                if ct is None or not np.isfinite(ct):
                    continue
                events.append(float(ct))
            elif event == 'baseline_on':
                if tidx < baseline.size:
                    events.append(float(baseline[int(tidx)]))

        min_events = cfg.min_events if min_events_override is None else int(min_events_override)
        if len(events) < min_events:
            # insufficient events for this state
            return session_name, None, None, None

        # collect per-cluster traces
        traces = []
        for cl in session.clusters:
            spikes = np.asarray(cl.spike_times, dtype=float)
            spikes = spikes[np.isfinite(spikes)]
            if spikes.size == 0:
                continue
            mat = _build_psth_matrix(spikes, events, edges)
            if mat is None or mat.shape[0] < cfg.min_events:
                continue
            mean, sem = _smooth_mean(mat, cfg.smooth_bins)
            if mean is None:
                continue
            z, peak = _zscore_trace(mean, t_vec, cfg.baseline_window)
            traces.append(z)

        if not traces:
            return session_name, None, None, None
        stack = np.stack(traces)
        mean_pop = np.nanmean(stack, axis=0)
        sem_pop = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else np.zeros_like(mean_pop)
        return session_name, t_vec, mean_pop, sem_pop
    except Exception as e:
        return session_name, None, None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='FIGURES')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--events', nargs='*', default=['lick','change','baseline_on'])
    ap.add_argument('--states', nargs='*', default=['impulsive','balanced','disengaged'])
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # sessions: use manifest order if available
    if MANIFEST.exists():
        mf = pd.read_csv(MANIFEST)
        mf['session'] = mf['subject'].astype(str) + '_' + mf['session_name'].astype(str)
        sessions = mf['session'].tolist()
    else:
        # fallback to responsiveness outputs
        sessions = sorted([p.stem.replace('_unit_responsiveness','') for p in RESP_DIR.glob('*_unit_responsiveness.csv')])

    cfg = MatlabLickConfig()

    # For each event/state, parallelize session processing
    for event in args.events:
        results = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {}
            for sess in sessions:
                for state in args.states:
                    key = (event, state, sess)
                    futures[ex.submit(process_session, sess, event, state, cfg)] = key
            for fut in as_completed(futures):
                sess_name, t_vec, mean, sem = fut.result()
                key = futures[fut]
                # key is (event,state,session)
                _, state, session = key
                if t_vec is None:
                    # store None
                    results.setdefault((event,state), []).append((session, None, None))
                else:
                    results.setdefault((event,state), []).append((session, t_vec, mean, sem))

        # For each state, build heatmap and save CSV
        for state in args.states:
            rows = results.get((event,state), [])
            # keep only sessions with data and preserve original session order
            sess_order = [s for s in sessions]
            rows_map = {r[0]: r[1:] for r in rows if r[1] is not None}
            valid_sessions = [s for s in sess_order if s in rows_map]
            if not valid_sessions:
                print(f'No data for event={event}, state={state}')
                continue
            t_vec = rows_map[valid_sessions[0]][0]
            mat = np.stack([rows_map[s][0] for s in valid_sessions])
            sem_mat = np.stack([rows_map[s][1] for s in valid_sessions])
            # save CSV of means
            df_out = pd.DataFrame(mat, index=valid_sessions, columns=[f't_{x:.3f}' for x in t_vec])
            csvp = out / f'state_activity_{event}_{state}_means.csv'
            df_out.to_csv(csvp)

            # plot heatmap
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(valid_sessions))))
            im = ax.imshow(mat, aspect='auto', cmap='RdYlBu_r', extent=[t_vec[0], t_vec[-1], 0, len(valid_sessions)])
            ax.axvline(0, color='k', linestyle='--', lw=0.8)
            ax.set_yticks(np.arange(len(valid_sessions)) + 0.5)
            ax.set_yticklabels(valid_sessions, fontsize=6)
            ax.set_xlabel('time (s)')
            ax.set_title(f'state={state} event={event} (sessions x time)')
            fig.colorbar(im, ax=ax, label='z-score')
            fig.tight_layout()
            pngp = out / f'state_activity_{event}_{state}_heatmap.png'
            fig.savefig(pngp, dpi=150)
            plt.close(fig)
            print('Wrote', csvp, pngp)


if __name__ == '__main__':
    main()
