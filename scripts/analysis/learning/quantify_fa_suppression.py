"""
Quantify FA Suppression: Peak Amplitude & Ramp Slope.

This script quantifies the "suppression" effect observed in Expert mice during Early FAs.
It calculates:
1. Response Amplitude (Hz relative to baseline):
   - Excited units: Max val in window [-0.5, 0.0]
   - Inhibited units: Min val in window [-0.5, 0.0]
2. Ramp Slope (Hz/s in window [-1.0, -0.25])

Comparison: Early (<3s) vs Late (>=3s) FAs.
"""

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import sys
import concurrent.futures
from tqdm import tqdm

# Ensure repo root is in path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from visdetect.core.session import load_session
from visdetect.analysis.lick import MatlabLickAnalyzer, MatlabLickConfig

# -------------------------------------------------------------------------
# Custom Analyzer (Mirrors compare_early_late_fa.py logic)
# -------------------------------------------------------------------------
class SplitFaAnalyzer(MatlabLickAnalyzer):
    def get_events_split(self, session, split_time=3.0):
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
            t_event = float(baseline[i] + delay)
            
            if delay < split_time:
                early.append(t_event)
            else:
                late.append(t_event)
                
        return np.array(early), np.array(late)

    def get_unit_metrics(self, session, events, good_ids):
        """Calculate Peak, Min, and Slope for every unit for a set of events"""
        if len(events) < self.cfg.min_events:
            return pd.DataFrame()

        rows = []
        edges = self.cfg.time_edges
        t_vec = self.cfg.time_centers
        
        # Windows indices
        base_mask = (t_vec >= -1.75) & (t_vec < -1.25)
        peak_mask = (t_vec >= -0.5) & (t_vec <= 0.0)
        ramp_mask = (t_vec >= -1.0) & (t_vec <= -0.25)
        ramp_x = t_vec[ramp_mask]

        if self.good_ids is not None:
             target_ids = self.good_ids.intersection(good_ids) if good_ids is not None else self.good_ids
        else:
             target_ids = set(good_ids) if good_ids is not None else None

        for cl in session.clusters:
            if target_ids is not None and int(cl.cluster_id) not in target_ids:
                continue
                
            spikes = np.asarray(cl.spike_times, dtype=float)
            spikes = spikes[np.isfinite(spikes)]
            if spikes.size == 0: continue

            matrix = self._build_psth_matrix(spikes, events, edges)
            if matrix is None or matrix.shape[0] < self.cfg.min_events:
                continue
                
            unit_trace = self._smooth_trials(matrix) 
            
            # Baseline Subtraction
            baseline_hz = np.nanmean(unit_trace[base_mask])
            norm_trace = unit_trace - baseline_hz
            
            # Metrics
            peak_val = np.nanmax(norm_trace[peak_mask])
            trough_val = np.nanmin(norm_trace[peak_mask])
            
            if np.any(ramp_mask) and len(norm_trace[ramp_mask]) > 1 and np.std(ramp_x) > 0:
                slope, _, _, _, _ = stats.linregress(ramp_x, norm_trace[ramp_mask])
            else:
                slope = np.nan
                
            rows.append({
                'cluster_id': cl.cluster_id,
                'baseline_hz': baseline_hz,
                'peak_val': peak_val,
                'trough_val': trough_val,
                'ramp_slope': slope
            })
            
        return pd.DataFrame(rows)

# -------------------------------------------------------------------------
# Worker
# -------------------------------------------------------------------------
def process_session_metrics(task, stats_root, cfg_dict):
    cfg = MatlabLickConfig(**cfg_dict)
    
    session_name = str(task['session_name'])
    stage = task['stage']
    date = task['date']
    pkl_path = task['path']
    
    try:
        session = load_session(pkl_path)
    except:
        return []
        
    stats_file = Path(stats_root) / session_name / 'lick_responsiveness.csv'
    if not stats_file.exists():
        return []
        
    stats_df = pd.read_csv(stats_file)
    
    # Analyze by Type
    results = []
    
    type_map = {
        'Excited': stats_df[(stats_df['is_significant']) & (stats_df['delta_mean'] > 0)]['cluster_id'].values,
        'Inhibited': stats_df[(stats_df['is_significant']) & (stats_df['delta_mean'] < 0)]['cluster_id'].values
    }
    
    analyzer = SplitFaAnalyzer(cfg=cfg)
    early_ev, late_ev = analyzer.get_events_split(session, split_time=3.0)
    
    for ctype, ids in type_map.items():
        if len(ids) == 0: continue
        
        # Analyze Early
        if len(early_ev) >= 5:
            df = analyzer.get_unit_metrics(session, early_ev, ids)
            if not df.empty:
                # Decide metric based on type
                if ctype == 'Excited':
                    amp = df['peak_val'].mean()
                else:
                    amp = df['trough_val'].mean()
                    
                results.append({
                    'session': session_name, 'date': date, 'stage': stage,
                    'condition': 'Early', 'unit_type': ctype,
                    'n_units': len(df),
                    'response_amp': amp,
                    'ramp_slope': df['ramp_slope'].mean()
                })

        # Analyze Late
        if len(late_ev) >= 5:
            df = analyzer.get_unit_metrics(session, late_ev, ids)
            if not df.empty:
                if ctype == 'Excited':
                    amp = df['peak_val'].mean()
                else:
                    amp = df['trough_val'].mean() # Keep sign (negative)

                results.append({
                    'session': session_name, 'date': date, 'stage': stage,
                    'condition': 'Late', 'unit_type': ctype,
                    'n_units': len(df),
                    'response_amp': amp,
                    'ramp_slope': df['ramp_slope'].mean()
                })
            
    return results

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/BG_046_staging_manifest.csv')
    parser.add_argument('--stats-root', default='FIGURES/lick/BG_046')
    parser.add_argument('--out-dir', default='FIGURES/suppression_quantification')
    parser.add_argument('--n_workers', type=int, default=8)
    args = parser.parse_args()

    cfg = MatlabLickConfig(
        pre_event_window=2.0,   
        post_event_window=1.0,  
        bin_size=0.01,          
        smooth_bins=5           
    )
    
    cfg_dict = {
        'pre_event_window': cfg.pre_event_window,
        'post_event_window': cfg.post_event_window,
        'bin_size': cfg.bin_size,
        'smooth_bins': cfg.smooth_bins,
        'baseline_window': cfg.baseline_window
    }
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = pd.read_csv(args.manifest, dtype={'session_name': str, 'date': str})
    
    tasks = []
    for _, row in manifest.iterrows():
        if row['stage'] in ['Naive', 'Learning', 'Expert']:
            tasks.append(row.to_dict())
            
    print(f"Quantifying metrics for {len(tasks)} sessions (Workers={args.n_workers})...")
    
    all_metrics = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [executor.submit(process_session_metrics, task, args.stats_root, cfg_dict) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                all_metrics.extend(future.result())
            except Exception as e:
                print(f"Error: {e}")

    if not all_metrics:
        print("No metrics calculated.")
        return

    # Plotting
    res_df = pd.DataFrame(all_metrics)
    res_df.to_csv(out_dir / 'suppression_stats_both_types.csv', index=False)
    
    res_df['stage'] = pd.Categorical(res_df['stage'], ['Naive', 'Learning', 'Expert'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for i, ctype in enumerate(['Excited', 'Inhibited']):
        subset = res_df[res_df['unit_type'] == ctype]
        if subset.empty: continue
        
        # Determine Metric Names/Labels
        if ctype == 'Excited':
            amp_label = "Peak Amplitude (Hz)"
        else:
            amp_label = "Trough Depth (Hz)"
            
        # Amp Plot
        ax_amp = axes[i, 0]
        sns.boxplot(data=subset, x='stage', y='response_amp', hue='condition', 
                    palette={'Early': 'tab:orange', 'Late': 'tab:blue'}, showfliers=False, ax=ax_amp)
        sns.stripplot(data=subset, x='stage', y='response_amp', hue='condition', 
                      dodge=True, color='k', alpha=0.5, legend=False, ax=ax_amp)
        ax_amp.set_title(f'{ctype}: {amp_label}')
        ax_amp.set_ylabel(amp_label)
        
        # Slope Plot
        ax_slope = axes[i, 1]
        sns.boxplot(data=subset, x='stage', y='ramp_slope', hue='condition', 
                    palette={'Early': 'tab:orange', 'Late': 'tab:blue'}, showfliers=False, ax=ax_slope)
        sns.stripplot(data=subset, x='stage', y='ramp_slope', hue='condition', 
                      dodge=True, color='k', alpha=0.5, legend=False, ax=ax_slope)
        ax_slope.set_title(f'{ctype}: Ramp Slope')
        ax_slope.set_ylabel('Slope (Hz/s)')

    plt.suptitle("Suppression Metrics: Early vs Late FAs")
    plt.tight_layout()
    plt.savefig(out_dir / 'quantification_both_types.png')
    print("Plots saved to", out_dir)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
