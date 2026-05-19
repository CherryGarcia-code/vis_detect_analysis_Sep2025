"""
Compare Neural Activity during Early (<3s) vs Late (>=3s) False Alarms.
Plots Baseline-Subtracted PSTHs to compare shape and amplitude.

Refactored to use `visdetect.analysis.lick` for consistent processing.
Supports parallel processing via ProcessPoolExecutor.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import sys
import concurrent.futures


from visdetect.core.session import load_session
from visdetect.analysis.lick import MatlabLickAnalyzer, MatlabLickConfig
from visdetect.analysis.config import load_staging_manifest

# -------------------------------------------------------------------------
# Custom Analyzer to reuse standard logic but split events
# -------------------------------------------------------------------------
class SplitFaAnalyzer(MatlabLickAnalyzer):
    """
    Wrapper around MatlabLickAnalyzer to handle Early/Late splitting
    while keeping standard binning/smoothing/normalization.
    """
    def get_events_split(self, session, split_time=3.0):
        # We need to manually fetch events because the standard _fa_lick_times 
        # has a hardcoded min_fa_delay check.
        # However, we can use the robust logic from the base class if we 
        # reimplement just the filtering step.
        
        # 1. Reuse the robust time extraction from base class (private method)
        # We can't easily call _fa_lick_times because it filters.
        # So we implement the explicit logic here, fully compliant with the legacy code.
        
        raw_baseline = session.ni_events.get("Baseline_ON", []) if getattr(session, "ni_events", None) is not None else []
        if isinstance(raw_baseline, dict):
            if "rise_t" in raw_baseline: vals = raw_baseline.get("rise_t", [])
            elif "times" in raw_baseline: vals = raw_baseline.get("times", [])
            else: vals = []
            baseline = np.asarray(vals, dtype=float).flatten()
        else:
            baseline = np.asarray(raw_baseline, dtype=float).flatten()
            
        trials = session.trials
        n = min(len(trials), baseline.size)
        
        early, late = [], []
        
        for i in range(n):
            trial = trials[i]
            outcome = (trial.trialoutcome or "").lower()
            if outcome != "fa": continue
                
            rt = trial.reactiontimes.get("FA") if trial.reactiontimes else None
            if rt is None or not np.isfinite(rt): continue
            
            delay = float(rt)
            # Use baseline + delay to get global time
            t_event = float(baseline[i] + delay)
            
            if delay < split_time:
                early.append(t_event)
            else:
                late.append(t_event)
                
        return early, late

    def get_mean_trace(self, session, events, good_ids):
        """
        Calculate population mean trace for a specific set of events.
        Returns: (time_axis, mean_trace_hz) or (None, None)
        """
        if not events or len(events) < self.cfg.min_events:
            return None, None
            
        # Temporarily override the allowed good_ids for this specific calculation
        # The analyzer was initialized with a set, but we might want to subset further 
        # (e.g. only Excited units)
        original_ids = self.good_ids
        if good_ids is not None:
            self.good_ids = set(good_ids)
            
        # 1. Collect all unit PSTHs
        # We use internal methods to get the matrix -> smooth -> mean
        psth_list = []
        
        edges = self.cfg.time_edges
        
        for cl in session.clusters:
            if self.good_ids is not None and int(cl.cluster_id) not in self.good_ids:
                continue
                
            spikes = np.asarray(cl.spike_times, dtype=float)
            spikes = spikes[np.isfinite(spikes)]
            if spikes.size == 0: continue
            
            matrix = self._build_psth_matrix(spikes, events, edges)
            if matrix is None or matrix.shape[0] < self.cfg.min_events:
                continue
            
            # Smooth
            trace = self._smooth_trials(matrix) # Shape: (n_bins,)
            
            # BASELINE SUBTRACTION
            # Get baseline window indices
            base_idx = self._window_indices(self.cfg.baseline_window)
            if base_idx[1] > base_idx[0]:
                base_hz = np.nanmean(trace[base_idx[0]:base_idx[1]])
                trace = trace - base_hz
            
            psth_list.append(trace)
            
        # Restore IDs
        self.good_ids = original_ids
            
        if not psth_list:
            return None, None
            
        # Average across units to get session mean
        stack = np.stack(psth_list)
        session_mean = np.nanmean(stack, axis=0)
        
        return self.cfg.time_centers, session_mean

# -------------------------------------------------------------------------
# Worker Function
# -------------------------------------------------------------------------
def process_single_session_traces(row_tuple, stats_root, cfg_dict):
    """
    Worker function to process a single session.
    Args:
        row_tuple: (index, row_series) or a dict. We'll pass a dict.
        stats_root: Path to stats folder.
        cfg_dict: Dictionary to reconstruct MatlabLickConfig.
    Returns:
        (stage, session_name, session_results)
        where session_results = {ctype: {cond: trace}}
    """
    # Reconstruct Config (Config objects might not pickle well if they have methods, but data classes are fine.
    # To be safe, we passed a dict)
    cfg = MatlabLickConfig(**cfg_dict)
    
    session_name = str(row_tuple['session_name'])
    stage = row_tuple['stage']
    pkl_path = row_tuple['path']
    
    results = {
        'Excited': {'Early': None, 'Late': None},
        'Inhibited': {'Early': None, 'Late': None}
    }
    
    try:
        session = load_session(pkl_path)
    except Exception as e:
        return stage, session_name, None
        
    analyzer = SplitFaAnalyzer(cfg=cfg)
    
    # Get Events
    early, late = analyzer.get_events_split(session, split_time=3.0)
    
    # Load Stats
    stats_path = Path(stats_root) / session_name / 'lick_responsiveness.csv'
    if not stats_path.exists():
        return stage, session_name, None
        
    try:
        stats_df = pd.read_csv(stats_path)
    except:
        return stage, session_name, None

    # Process types
    for ctype in ['Excited', 'Inhibited']:
        if ctype == 'Excited':
            ids = stats_df[(stats_df['is_significant']) & (stats_df['delta_mean'] > 0)]['cluster_id'].values
        else:
            ids = stats_df[(stats_df['is_significant']) & (stats_df['delta_mean'] < 0)]['cluster_id'].values
        
        if len(ids) == 0: continue
        
        _, trace_early = analyzer.get_mean_trace(session, early, ids)
        if trace_early is not None:
            results[ctype]['Early'] = trace_early
            
        _, trace_late = analyzer.get_mean_trace(session, late, ids)
        if trace_late is not None:
            results[ctype]['Late'] = trace_late
            
    return stage, session_name, results


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=None,
                        help='Path to manifest CSV (default: canonical)')
    parser.add_argument('--stats-root', default='FIGURES/lick/BG_046')
    parser.add_argument('--out-dir', default='FIGURES/learning_fa_split')
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument('--no-filter', action='store_true',
                        help='Bypass SESSION_FILTER')
    args = parser.parse_args()
    
    # Configuration
    cfg = MatlabLickConfig(
        pre_event_window=2.0, 
        post_event_window=1.0,
        bin_size=0.01,
        smooth_bins=5,
        baseline_window=(-1.75, -1.25)
    )
    # Convert to dict for passing to workers
    cfg_dict = {
        'pre_event_window': cfg.pre_event_window,
        'post_event_window': cfg.post_event_window,
        'bin_size': cfg.bin_size,
        'smooth_bins': cfg.smooth_bins,
        'baseline_window': cfg.baseline_window
        # min_events etc use defaults or we can explicit pass if needed
        # but MatlabLickConfig defaults are used if not provided
    }
    # Add non-constructor fields if any? No, these are the main ones.
    # Note: min_events default is 5.
    
    # Data Storage
    data = {
        s: {
            t: {'Early': [], 'Late': []} 
            for t in ['Excited', 'Inhibited']
        }
        for s in ['Naive', 'Learning', 'Expert']
    }
    
    manifest = load_staging_manifest(manifest_path=args.manifest,
                                      apply_filter=not args.no_filter)
    
    # Prepare rows for workers
    tasks = []
    for _, row in manifest.iterrows():
        tasks.append(row.to_dict())
            
    print(f"Processing {len(tasks)} sessions with {args.n_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        # submit all
        futures = {executor.submit(process_single_session_traces, task, args.stats_root, cfg_dict): task['session_name'] for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Sessions"):
            try:
                stage, s_name, res = future.result()
                if res is None: continue
                
                # Unpack results into main structure
                for ctype in ['Excited', 'Inhibited']:
                    if res[ctype]['Early'] is not None:
                        data[stage][ctype]['Early'].append(res[ctype]['Early'])
                    if res[ctype]['Late'] is not None:
                        data[stage][ctype]['Late'].append(res[ctype]['Late'])
                        
            except Exception as exc:
                print(f"Worker failed: {exc}")
    
    # Plotting
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    t_vec = cfg.time_centers 
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey='row')
    
    for row_idx, ctype in enumerate(['Excited', 'Inhibited']):
        for col_idx, stage in enumerate(['Naive', 'Learning', 'Expert']):
            ax = axes[row_idx, col_idx]
            
            traces_early = data[stage][ctype]['Early']
            traces_late = data[stage][ctype]['Late']
            
            colors = {'Early': 'tab:orange', 'Late': 'tab:blue'}
            
            for cond, traces in zip(['Early', 'Late'], [traces_early, traces_late]):
                if not traces: continue
                
                stack = np.stack(traces)
                mean = np.nanmean(stack, axis=0)
                sem = np.nanstd(stack, axis=0) / np.sqrt(len(traces))
                
                ax.plot(t_vec, mean, color=colors[cond], label=f"{cond} (n={len(traces)} sess)")
                ax.fill_between(t_vec, mean-sem, mean+sem, color=colors[cond], alpha=0.2)
                
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            # Mark baseline window
            ax.axvspan(-1.75, -1.25, color='gray', alpha=0.1)
            
            if row_idx == 0: ax.set_title(stage)
            if col_idx == 0: ax.set_ylabel(f"{ctype}\nBaseline Subtracted (Hz)")
            if row_idx == 1: ax.set_xlabel("Time from Lick (s)")
            
            # Show legend on every subplot for clarity, or just first? 
            # User said "incomplete" implying it was missing where expected.
            # Let's put it on all of them, or at least consistent.
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize='small')

    plt.suptitle("Early (<3s) vs Late (>=3s) False Alarms")
    plt.tight_layout()
    plt.savefig(out_dir / 'fa_early_late_comparison_v2.png')
    print(f"Plot saved to {out_dir / 'fa_early_late_comparison_v2.png'}")

if __name__ == '__main__':
    # needed for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()
