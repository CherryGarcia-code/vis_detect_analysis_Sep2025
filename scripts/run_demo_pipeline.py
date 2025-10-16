from pathlib import Path
import sys
import argparse
import logging
from typing import Optional

# Determine repository root (two levels up from this script)
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless operation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Analysis imports (project-local)
from src.session_io import load_session
from src.align import compute_peth_for_session, get_event_times, compute_true_reaction_time
from src.coding_direction import compute_cd_shrinkage, time_resolved_cd


def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description='Run demo single-session pipeline (PETH → pop → CD → baseline stats)')
    p.add_argument('--session-pkl', type=Path, default=repo_root / 'data' / 'BG_031_260325.pkl',
                   help='Path to session pickle (default: data/BG_031_260325.pkl)')
    p.add_argument('--out-dir', type=Path, default=repo_root / 'notebooks', help='Directory to write PNG/CSV outputs')
    p.add_argument('--mode', choices=['quick', 'full'], default='quick', help='Quick (fewer permutations) or full run')
    p.add_argument('--n-permutations', type=int, default=None, help='Override number of permutations for time-resolved CD')
    p.add_argument('--reg', type=float, default=1.0, help='Regularization for shrinkage CD')
    p.add_argument('--overwrite', action='store_true', help='Overwrite output files if they exist')
    # By default the pipeline will prefer canonical PKL-provided good_cluster_ids.
    p.add_argument('--use-good-only', dest='use_good_only', action='store_true', help='Prefer session.good_cluster_ids when present (default)')
    p.add_argument('--no-use-good-only', dest='use_good_only', action='store_false', help='Do not restrict to session.good_cluster_ids; use all clusters')
    p.set_defaults(use_good_only=True)
    p.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return p.parse_args(argv)


def configure_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def safe_save(path: Path, save_fn, overwrite: bool):
    """Helper to save files only if allowed by overwrite flag."""
    if path.exists() and not overwrite:
        logging.info('File exists and --overwrite not set, skipping save: %s', path)
        return False
    try:
        save_fn(path)
        logging.info('Wrote %s', path)
        return True
    except Exception:
        logging.exception('Failed to write %s', path)
        return False


def main(argv: Optional[list] = None):
    args = parse_args(argv)
    configure_logging(args.verbose)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defaults
    bin_size = 0.02
    window = (-0.5, 1.0)
    baseline_window = (-0.5, 0.0)

    logging.info('Loading session from %s', args.session_pkl)
    session = load_session(str(args.session_pkl))

    # Use session_name if available, otherwise fall back to the pickle stem
    session_name = session.session_name or args.session_pkl.stem
    session_out_dir = out_dir / session_name
    session_out_dir.mkdir(parents=True, exist_ok=True)

    # Compute PETHs
    logging.info('Computing Change_ON PETHs (using session.good_cluster_ids by default)...')
    peths = compute_peth_for_session(session, 'Change_ON', window=window, bin_size=bin_size, use_good_only=args.use_good_only)
    cluster_ids = sorted([cid for cid, info in peths.items() if info['trials_matrix'].shape[0] > 0])
    if len(cluster_ids) == 0:
        raise RuntimeError('No clusters with trial matrices found in PETHs')

    # Build population
    example = peths[cluster_ids[0]]
    n_trials_example, n_bins = example['trials_matrix'].shape
    # Ensure n_trials does not exceed the number of behavior trials in session
    n_trials = int(min(n_trials_example, len(session.trials)))
    pop_list = []
    use_ids = []
    for cid in cluster_ids:
        mat = peths[cid]['trials_matrix']
        # If PETH has more trials than behavioral trials, truncate to match
        if mat.shape[0] > n_trials:
            mat = mat[:n_trials, :]
        if mat.shape[0] == n_trials and mat.shape[1] == n_bins:
            pop_list.append(mat)
            use_ids.append(cid)
    pop = np.stack(pop_list, axis=2)
    logging.info('Built pop shape %s', pop.shape)

    # Time-resolved CD (Hit vs Miss)
    # Align trial outcome arrays to the trials used in PETHs (example['trials_matrix'] defines n_trials)
    trial_outcomes_full = [getattr(t, 'trialoutcome', None) for t in session.trials]
    # If session.trials is longer than the PETH trials, truncate to match
    trial_outcomes = trial_outcomes_full[:n_trials]
    cond_hit = np.array([o == 'Hit' for o in trial_outcomes])
    cond_miss = np.array([o == 'Miss' for o in trial_outcomes])
    logging.info('n Hit=%d, n Miss=%d', int(cond_hit.sum()), int(cond_miss.sum()))

    if args.n_permutations is not None:
        n_perm = int(args.n_permutations)
    else:
        n_perm = 20 if args.mode == 'quick' else 500

    # Compute time-resolved CD (Hit vs Miss)
    try:
        res = time_resolved_cd(pop, cond_hit, method='shrinkage', reg=args.reg, n_splits=5,
                               n_permutations=n_perm, random_state=0)
        proj = res.get('proj', None)
        logging.info('Computed time-resolved CD (n_permutations=%d)', n_perm)
    except Exception:
        logging.exception('Failed to compute time-resolved CD')
        proj = None

    # If projections are available, plot Change_ON projections and save group stats
    if proj is not None:
        try:
            bins = np.arange(window[0] + bin_size / 2.0, window[1], bin_size)
            fig_c, ax_c = plt.subplots(figsize=(8, 4))
            A = proj[cond_hit]
            B = proj[cond_miss]
            if A.size > 0 and B.size > 0:
                mean_a = A.mean(axis=0)
                mean_b = B.mean(axis=0)
                sem_a = A.std(axis=0) / np.sqrt(max(1, A.shape[0]))
                sem_b = B.std(axis=0) / np.sqrt(max(1, B.shape[0]))
                ax_c.fill_between(bins, mean_a - sem_a, mean_a + sem_a, color='C0', alpha=0.2)
                ax_c.fill_between(bins, mean_b - sem_b, mean_b + sem_b, color='C1', alpha=0.2)
                ax_c.plot(bins, mean_a, label=f'Hit (n={int(cond_hit.sum())})', color='C0')
                ax_c.plot(bins, mean_b, label=f'Miss (n={int(cond_miss.sum())})', color='C1')
                ax_c.axvline(0, color='k', linestyle='--', linewidth=0.8)
                ax_c.set_xlabel('Time (s) relative to Change_ON')
                ax_c.set_ylabel('Projection onto CD (a.u.)')
                ax_c.legend(loc='upper right', fontsize='small')
                ax_c.set_title('Change_ON CD projections: Hit vs Miss')
                fig_c.tight_layout()

                out_png_change = session_out_dir / 'demo_pipeline_projection_changeON_hit_vs_miss.png'
                safe_save(out_png_change, lambda p: fig_c.savefig(str(p), dpi=150, bbox_inches='tight'), overwrite=args.overwrite)
                plt.close(fig_c)

                # Per-group scalar stats on Change_ON projection (mean across bins)
                # Compute scalar using bins up to and including time 0 only
                bins = np.arange(window[0] + bin_size / 2.0, window[1], bin_size)
                pre0_mask = bins <= 0
                if pre0_mask.any():
                    scalar_change = proj[:, pre0_mask].mean(axis=1)
                else:
                    scalar_change = proj.mean(axis=1)

                # Save per-trial scalar values (pre-0) for aggregation
                try:
                    import pandas as _pd
                    trial_idx = list(range(len(scalar_change)))
                    outcomes = trial_outcomes[:len(scalar_change)]
                    df_trials = _pd.DataFrame({'trial_index': trial_idx, 'outcome': outcomes, 'scalar_pre0': scalar_change})
                    out_trial_csv = session_out_dir / 'demo_pipeline_changeON_scalar_pre0_by_trial.csv'
                    safe_save(out_trial_csv, lambda p: df_trials.to_csv(str(p), index=False), overwrite=args.overwrite)
                except Exception:
                    logging.exception('Failed to save per-trial Change_ON scalars')
                rows_ch = []
                for lbl, mask in (('Hit', cond_hit), ('Miss', cond_miss)):
                    n = int(mask.sum())
                    if n == 0:
                        mean = np.nan
                        sem = np.nan
                    else:
                        vals = scalar_change[mask]
                        mean = float(np.nanmean(vals))
                        sem = float(np.nanstd(vals, ddof=0) / np.sqrt(max(1, n)))
                    rows_ch.append({'group': lbl, 'n': n, 'mean': mean, 'sem': sem})
                stats_change_df = pd.DataFrame(rows_ch).set_index('group')

                # Pairwise p-value between Hit and Miss
                try:
                    vi = scalar_change[cond_hit]
                    vj = scalar_change[cond_miss]
                    if vi.size < 2 or vj.size < 2:
                        p = np.nan
                    else:
                        _, p = stats.ttest_ind(vi, vj, equal_var=False, nan_policy='omit')
                except Exception:
                    p = np.nan
                pvals_change = pd.DataFrame([[np.nan, p], [p, np.nan]], index=['Hit', 'Miss'], columns=['Hit', 'Miss'])

                out_stats_change = session_out_dir / 'demo_pipeline_changeON_group_stats.csv'
                out_pvals_change = session_out_dir / 'demo_pipeline_changeON_pvalues.csv'
                safe_save(out_stats_change, lambda p: stats_change_df.to_csv(str(p)), overwrite=args.overwrite)
                safe_save(out_pvals_change, lambda p: pvals_change.to_csv(str(p)), overwrite=args.overwrite)
                logging.info('Saved Change_ON comparison outputs: %s, %s, %s', out_png_change, out_stats_change, out_pvals_change)
            else:
                logging.info('Insufficient Hit or Miss trials to plot Change_ON projection (A size=%d, B size=%d)', A.size, B.size)
        except Exception:
            logging.exception('Failed to save Change_ON projection or stats')

    # Baseline CD (Hit vs FA)
    logging.info('Computing Baseline_ON PETHs (using session.good_cluster_ids by default)...')

    # We treat the baseline period as starting at Baseline_ON and ending per-trial:
    # - For Hit/Miss trials: end at the Change_ON time for that trial
    # - For FA/abort trials: end at (reaction_time - 1.0s) for that trial
    # We'll align spikes to Baseline_ON with a window that covers the maximum per-trial baseline duration
    # and then compute per-trial means using only bins up to each trial's end.

    # Get per-trial absolute event times
    baseline_on_times = get_event_times(session, 'Baseline_ON')
    change_on_times = get_event_times(session, 'Change_ON')

    # Build per-trial end times (absolute)
    n_trials_available = min(len(session.trials), len(baseline_on_times))
    end_times = np.full((n_trials_available,), np.nan, dtype=float)
    for i in range(n_trials_available):
        trial = session.trials[i]
        outcome = getattr(trial, 'trialoutcome', None)
        t0 = baseline_on_times[i] if i < len(baseline_on_times) else np.nan
        t_change = change_on_times[i] if i < len(change_on_times) else np.nan
        if outcome in ['Hit', 'Miss'] and not np.isnan(t_change):
            end_times[i] = float(t_change)
        elif outcome in ['FA', 'abort']:
            # compute absolute reaction time if available, then subtract 1.0s
            try:
                rt_abs = compute_true_reaction_time(trial, getattr(session, 'ni_events', {}) or {}, i)
            except Exception:
                rt_abs = None
            if rt_abs is not None and not np.isnan(rt_abs):
                end_times[i] = float(rt_abs) - 1.0
            else:
                # fallback: use t0 (zero-length) so trial will be excluded later
                end_times[i] = np.nan
        else:
            # unknown outcome: try to use change time if present
            if not np.isnan(t_change):
                end_times[i] = float(t_change)
            else:
                end_times[i] = np.nan

    # Convert end times to offsets relative to Baseline_ON (start at 0)
    offsets = np.full_like(end_times, np.nan)
    for i in range(n_trials_available):
        if np.isnan(end_times[i]):
            offsets[i] = np.nan
            continue
        t0 = baseline_on_times[i] if i < len(baseline_on_times) else np.nan
        if np.isnan(t0):
            offsets[i] = np.nan
        else:
            offsets[i] = end_times[i] - t0

    # Only keep trials where the baseline period after the initial 1s is positive
    # We drop the first 1.0s after Baseline_ON to avoid stimulus-evoked responses.
    min_offset = bin_size
    durations_after1 = offsets - 1.0
    valid_mask = ~np.isnan(durations_after1) & (durations_after1 > min_offset)
    if not valid_mask.any():
        raise RuntimeError('No valid baseline trials with positive post-1s duration were found')

    max_duration_after1 = float(np.nanmax(durations_after1))
    # Define baseline alignment window starting at 1.0s (relative to Baseline_ON) to the maximum post-1s duration
    baseline_window_dynamic = (1.0, 1.0 + max_duration_after1)
    peths_baseline = compute_peth_for_session(session, 'Baseline_ON', window=baseline_window_dynamic, bin_size=bin_size, use_good_only=args.use_good_only)

    # Build baseline_pop as trials x bins x units for the clusters we used in the Change_ON pop
    baseline_list = []
    for cid in use_ids:
        mat = peths_baseline.get(int(cid), {}).get('trials_matrix', None)
        if mat is None:
            # no baseline trials for this cluster -> fill with zeros for n_trials_available x n_bins
            n_bins = int(np.ceil((baseline_window_dynamic[1] - baseline_window_dynamic[0]) / bin_size))
            mat = np.zeros((n_trials_available, n_bins), dtype=float)
        # If compute_peth_for_session returned more trials than behavioral trials, truncate
        if mat.shape[0] > n_trials_available:
            mat = mat[:n_trials_available, :]
        baseline_list.append(mat)
    baseline_pop = np.stack(baseline_list, axis=2)

    n_baseline_trials, n_bins, n_units = baseline_pop.shape

    # Number of bins up to each trial's end (clamped to available bins)
    n_end_bins = np.zeros((n_baseline_trials,), dtype=int)
    for i in range(n_baseline_trials):
        if i < durations_after1.size and not np.isnan(durations_after1[i]):
            nb = int(np.floor(durations_after1[i] / bin_size))
            nb = max(1, min(nb, n_bins))
            n_end_bins[i] = nb
        else:
            n_end_bins[i] = 0

    # Compute per-trial mean across only the bins up to each trial's end
    X_baseline_all = np.zeros((n_baseline_trials, n_units), dtype=float)
    valid_trial_mask = np.zeros((n_baseline_trials,), dtype=bool)
    for i in range(n_baseline_trials):
        nb = n_end_bins[i]
        if nb <= 0:
            X_baseline_all[i, :] = np.nan
            valid_trial_mask[i] = False
            continue
        vals = baseline_pop[i, :nb, :]
        X_baseline_all[i, :] = vals.mean(axis=0)
        valid_trial_mask[i] = True

    # Align FA / outcome masks to the baseline PETH trial count
    trial_outcomes_baseline = trial_outcomes_full[:n_baseline_trials]
    cond_hit_baseline = np.array([o == 'Hit' for o in trial_outcomes_baseline])
    cond_fa_baseline = np.array([o == 'FA' or o == 'FalseAlarm' for o in trial_outcomes_baseline])
    fa_latencies_full = np.array([np.nan if getattr(t, 'reactiontimes', None) is None else t.reactiontimes.get('FA', np.nan) for t in session.trials], dtype=float)
    fa_latencies_baseline = fa_latencies_full[:n_baseline_trials]
    logging.info('FA latencies finite (baseline window): %d', int(np.isfinite(fa_latencies_baseline).sum()))

    # Select only trials that are valid (had positive baseline duration) and are Hit or FA
    sel_mask = valid_trial_mask & (cond_hit_baseline | cond_fa_baseline)
    X_baseline = X_baseline_all[sel_mask]
    cond_sub = cond_hit_baseline[sel_mask].astype(int)
    if X_baseline.shape[0] < 2 or X_baseline.shape[1] == 0:
        raise RuntimeError('Insufficient baseline data to compute CD')
    cd = compute_cd_shrinkage(X_baseline, cond_sub, reg=float(args.reg))
    proj_baseline = np.tensordot(baseline_pop[sel_mask], cd, axes=([2], [0]))

    # RT-filtered visualization selection
    # RT-filtered visualization selection (use baseline-aligned masks)
    fa_long_mask_baseline = cond_fa_baseline & np.isfinite(fa_latencies_baseline) & (fa_latencies_baseline > 3.0)
    sel_vis = (cond_hit_baseline | fa_long_mask_baseline) & valid_trial_mask
    if sel_vis.sum() > 0:
        proj_baseline_vis = np.tensordot(baseline_pop[sel_vis], cd, axes=([2], [0]))
        cond_sub_vis = cond_hit_baseline[sel_vis].astype(bool)
        logging.info('proj_baseline_vis shape = %s', proj_baseline_vis.shape)

    # Save baseline projection PNG (grouped)
    # Use the dynamic baseline window for plotting (relative to Baseline_ON).
    # Construct bin centers from the actual number of bins used in baseline_pop to avoid length mismatches.
    try:
        bins_baseline = (np.arange(n_bins) * bin_size) + (baseline_window_dynamic[0] + bin_size / 2.0)
    except Exception:
        # Fallback to a safe arange if n_bins is not available
        bins_baseline = np.arange(baseline_window_dynamic[0] + bin_size / 2.0, baseline_window_dynamic[1], bin_size)
    fig, ax = plt.subplots(figsize=(9, 4))
    # groups
    # Use baseline-aligned outcome masks for grouping
    cond_miss_baseline = np.array([o == 'Miss' for o in trial_outcomes_baseline])
    cond_abort_baseline = np.array([isinstance(o, str) and ('abort' in o.lower()) for o in trial_outcomes_baseline])
    fa_long = fa_long_mask_baseline
    fa_short = cond_fa_baseline & np.isfinite(fa_latencies_baseline) & (fa_latencies_baseline <= 3.0)
    groups = {
        'Hit': cond_hit_baseline & valid_trial_mask,
        'FA>3s': fa_long & valid_trial_mask,
        'FA<=3s': fa_short & valid_trial_mask,
        'Abort': cond_abort_baseline & valid_trial_mask,
        'Miss': cond_miss_baseline & valid_trial_mask
    }
    groups = {k: v for k, v in groups.items() if v.sum() > 0}
    logging.info('Groups and counts: %s', {k: int(v.sum()) for k, v in groups.items()})

    outcome_colors = {'FA': 'r', 'Hit': 'g', 'Miss': 'gray', 'abort': 'm'}
    for i, (label, mask) in enumerate(groups.items()):
        if mask.sum() == 0:
            continue
        grp_proj = np.tensordot(baseline_pop[mask], cd, axes=([2], [0]))
        mean = grp_proj.mean(axis=0)
        sem = grp_proj.std(axis=0) / np.sqrt(max(1, grp_proj.shape[0]))
        if label.startswith('FA'):
            base = 'FA'
        elif label.lower().startswith('abort'):
            base = 'abort'
        else:
            base = label
        color = outcome_colors.get(base, f'C{i % 10}')
        ax.fill_between(bins_baseline, mean - sem, mean + sem, color=color, alpha=0.15)
        ax.plot(bins_baseline, mean, label=f"{label} (n={int(mask.sum())})", color=color)

    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (s) relative to Baseline_ON')
    ax.set_ylabel('Projection onto Baseline CD (a.u.)')
    ax.legend(loc='upper right', fontsize='small')
    ax.set_title('Baseline_CD projections: multiple behavioral outcomes')
    fig.tight_layout()

    out_png = session_out_dir / 'demo_pipeline_projection_baseline_groups.png'

    def _save_fig(p: Path):
        fig.savefig(str(p), dpi=150, bbox_inches='tight')

    saved = safe_save(out_png, _save_fig, overwrite=args.overwrite)
    plt.close(fig)

    # Per-group scalar stats and pairwise p-values
    # Compute per-trial scalar projection by projecting the per-trial mean over the actual baseline
    # duration (X_baseline_all) onto the CD vector. This avoids averaging across padded bins.
    scalar_proj = np.full((n_baseline_trials,), np.nan)
    try:
        # X_baseline_all may contain NaNs for invalid trials
        valid_idx = np.where(valid_trial_mask)[0]
        if valid_idx.size > 0:
            scalar_proj[valid_idx] = X_baseline_all[valid_idx].dot(cd)
    except Exception:
        logging.exception('Failed to compute per-trial scalar projections from X_baseline_all')

    # Save per-trial baseline scalar values (pre-0) for aggregation
    try:
        import pandas as _pd
        trial_idx_b = list(range(len(scalar_proj)))
        # trial_outcomes_baseline was already aligned/truncated earlier
        outcomes_b = trial_outcomes_baseline[:len(scalar_proj)]
        df_trials_b = _pd.DataFrame({'trial_index': trial_idx_b, 'outcome': outcomes_b, 'scalar_pre0': scalar_proj})
        out_trial_csv_b = session_out_dir / 'demo_pipeline_baseline_scalar_pre0_by_trial.csv'
        safe_save(out_trial_csv_b, lambda p: df_trials_b.to_csv(str(p), index=False), overwrite=args.overwrite)
    except Exception:
        logging.exception('Failed to save per-trial Baseline scalars')
    rows = []
    for label, mask in groups.items():
        n = int(mask.sum())
        if n == 0:
            mean = np.nan
            sem = np.nan
        else:
            vals = scalar_proj[mask]
            mean = float(np.nanmean(vals))
            sem = float(np.nanstd(vals, ddof=0) / np.sqrt(max(1, n)))
        rows.append({'group': label, 'n': n, 'mean': mean, 'sem': sem})
    stats_df = pd.DataFrame(rows).set_index('group')
    logging.info('\nPer-group summary:\n%s', stats_df)

    labels = list(groups.keys())
    pvals = np.full((len(labels), len(labels)), np.nan)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i >= j:
                continue
            vi = scalar_proj[groups[li]]
            vj = scalar_proj[groups[lj]]
            if vi.size < 2 or vj.size < 2:
                p = np.nan
            else:
                try:
                    _, p = stats.ttest_ind(vi, vj, equal_var=False, nan_policy='omit')
                except Exception:
                    p = np.nan
            pvals[i, j] = p
            pvals[j, i] = p
    pvals_df = pd.DataFrame(pvals, index=labels, columns=labels)
    logging.info('\nPairwise p-values:\n%s', pvals_df)

    stats_out = session_out_dir / 'demo_pipeline_baseline_group_stats.csv'
    pvals_out = session_out_dir / 'demo_pipeline_baseline_group_pvalues.csv'

    def _write_df(p: Path, df: pd.DataFrame):
        df.to_csv(str(p))

    safe_save(stats_out, lambda p: _write_df(p, stats_df), overwrite=args.overwrite)
    safe_save(pvals_out, lambda p: _write_df(p, pvals_df), overwrite=args.overwrite)

    logging.info('All done')


if __name__ == '__main__':
    main()
