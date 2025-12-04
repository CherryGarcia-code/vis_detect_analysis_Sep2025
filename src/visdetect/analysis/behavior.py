"""
Behavior analysis module for visual detection task.
Calculates performance metrics, psychometrics, and engagement statistics.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Any, Optional, Tuple
from visdetect.core.session import Session, Trial

def calculate_dprime(hit_rate, fa_rate):
    """Calculate d' (sensitivity index) with log-linear correction for 0/1 rates."""
    # Clip rates to avoid infinity
    hit_rate = np.clip(hit_rate, 0.01, 0.99)
    fa_rate = np.clip(fa_rate, 0.01, 0.99)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)

def get_trial_dataframe(session: Session) -> pd.DataFrame:
    """Convert session trials to a pandas DataFrame for easier analysis."""
    rows = []
    for i, t in enumerate(session.trials):
        # Normalize outcome
        outcome = t.trialoutcome.lower() if t.trialoutcome else "unknown"
        
        # Get RT
        rt = np.nan
        if t.reactiontimes:
            # Prioritize 'Hit' or 'FA' keys, or take the first available if generic
            if outcome == 'hit':
                rt = t.reactiontimes.get('Hit', t.reactiontimes.get('hit', np.nan))
            elif outcome == 'fa':
                rt = t.reactiontimes.get('FA', t.reactiontimes.get('fa', np.nan))
            
            # If still nan, try taking any value if dict is not empty
            if np.isnan(rt) and t.reactiontimes:
                try:
                    # Try to find 'RT' key first (common in newer data)
                    if 'RT' in t.reactiontimes:
                        rt = t.reactiontimes['RT']
                    else:
                        rt = float(list(t.reactiontimes.values())[0])
                except:
                    pass

        rows.append({
            'trial_idx': i,
            'outcome': outcome,
            'change_size': t.change_size if t.change_size is not None else 0,
            'orientation': t.orientation if t.orientation is not None else 0,
            'change_time': t.change_time if t.change_time is not None else 0,
            'rt': rt,
            'is_hit': outcome == 'hit',
            'is_miss': outcome == 'miss',
            'is_fa': outcome == 'fa',
            'is_abort': outcome == 'abort',
            'is_go': outcome in ['hit', 'miss'], # Assuming Go trial if it has a change
            'is_catch': outcome in ['fa', 'cr'] # Assuming Catch if no change (CR logic might need refinement based on task structure)
        })
    
    df = pd.DataFrame(rows)
    
    # Calculate Response Time (Time from Trial Start)
    # response_time = rt + change_time
    # Note: For FAs, change_time might be the time the FA happened? Or is it undefined?
    # Usually for FA, there is no change.
    # If change_time is 0 or None for FA, then response_time = rt (which is just time from start?)
    # Let's check the snippet output for FA?
    # The snippet only checked Hits.
    # But for Hits, we definitely want rt + change_time.
    df['response_time'] = df['rt'] + df['change_time']
        
    return df

def classify_fa_type(rt):
    """Classify FA based on RT."""
    if np.isnan(rt):
        return 'unknown'
    return 'early' if rt <= 3.0 else 'late'

def identify_session_state(row):
    """
    Identify behavioral state based on rolling metrics.
    Logic derived from quantile analysis of later sessions (e.g. BG_046).
    
    States:
    - Impulsive: High FA rate
    - Disengaged: High Miss rate 
    - Balanced: Good performance
    """
    # Thresholds
    FA_THRESHOLD = 0.48
    MISS_THRESHOLD = 0.35
    HIT_THRESHOLD = 0.30
    
    if row['rolling_fa_rate'] > FA_THRESHOLD:
        return 'impulsive'
    elif row['rolling_miss_rate'] > MISS_THRESHOLD:
        return 'disengaged'
    else:
        return 'balanced'

def compute_session_performance(session: Session) -> Dict[str, float]:
    """Compute aggregate performance metrics for a session."""
    df = get_trial_dataframe(session)
    if df.empty:
        return {}
        
    n_trials = len(df)
    n_hits = df['is_hit'].sum()
    n_miss = df['is_miss'].sum()
    n_fa = df['is_fa'].sum()
    n_abort = df['is_abort'].sum()
    
    # Hit Rate (excluding change size 1 if needed, but here we do global)
    # User asked for "Hit rate without change size = 1" in manifest.
    # We will calculate both here.
    
    n_go = n_hits + n_miss
    hit_rate = n_hits / n_go if n_go > 0 else 0.0
    miss_rate = n_miss / n_go if n_go > 0 else 0.0
    
    # Hit Rate (no size 1)
    # Assuming change_size=1 is the smallest/hardest or largest/easiest? 
    # Usually 1 is max contrast/size. If user wants to exclude it, maybe they want to see performance on harder trials?
    # Or maybe 1 is a specific condition.
    # Let's assume we filter out rows where change_size == 1.
    df_no_1 = df[df['change_size'] != 1]
    n_hits_no_1 = df_no_1['is_hit'].sum()
    n_miss_no_1 = df_no_1['is_miss'].sum()
    n_go_no_1 = n_hits_no_1 + n_miss_no_1
    hit_rate_no_1 = n_hits_no_1 / n_go_no_1 if n_go_no_1 > 0 else 0.0

    fa_rate_total = n_fa / n_trials if n_trials > 0 else 0.0
    abort_rate = n_abort / n_trials if n_trials > 0 else 0.0
    
    # Fractions of total trials
    fraction_hit = n_hits / n_trials if n_trials > 0 else 0.0
    fraction_miss = n_miss / n_trials if n_trials > 0 else 0.0
    fraction_fa = n_fa / n_trials if n_trials > 0 else 0.0
    fraction_abort = n_abort / n_trials if n_trials > 0 else 0.0
    
    # FA Split
    # FA <= 3s (Early/Impulsive?) vs FA > 3s (Late?)
    # Note: User said "FA>3 seconds - dark red, FA≤3 seconds -light red"
    fas = df[df['is_fa']].copy()
    n_fa_early = fas[fas['rt'] <= 3.0].shape[0]
    n_fa_late = fas[fas['rt'] > 3.0].shape[0]
    
    mean_rt_fa_early = fas[fas['rt'] <= 3.0]['rt'].mean()
    mean_rt_fa_late = fas[fas['rt'] > 3.0]['rt'].mean()
    
    # SEMs
    sem_rt_hit = df[df['is_hit']]['rt'].sem()
    sem_rt_fa_early = fas[fas['rt'] <= 3.0]['rt'].sem()
    sem_rt_fa_late = fas[fas['rt'] > 3.0]['rt'].sem()
    
    # d'
    d_prime = calculate_dprime(hit_rate, fa_rate_total)

    return {
        'n_trials': n_trials,
        'hit_rate': hit_rate,
        'miss_rate': miss_rate,
        'hit_rate_no_size_1': hit_rate_no_1,
        'fa_rate_total': fa_rate_total,
        'abort_rate': abort_rate,
        'fraction_hit': fraction_hit,
        'fraction_miss': fraction_miss,
        'fraction_fa': fraction_fa,
        'fraction_abort': fraction_abort,
        'd_prime': d_prime,
        'n_hits': n_hits,
        'n_miss': n_miss,
        'n_fa': n_fa,
        'n_fa_early': n_fa_early, # <= 3s
        'n_fa_late': n_fa_late,   # > 3s
        'n_abort': n_abort,
        'mean_rt_hit': df[df['is_hit']]['rt'].mean(),
        'median_rt_hit': df[df['is_hit']]['rt'].median(),
        'mean_rt_fa_early': mean_rt_fa_early,
        'mean_rt_fa_late': mean_rt_fa_late,
        'sem_rt_hit': sem_rt_hit,
        'sem_rt_fa_early': sem_rt_fa_early,
        'sem_rt_fa_late': sem_rt_fa_late
    }

def compute_rolling_performance(session: Session, window: int = 30) -> pd.DataFrame:
    """Compute rolling hit rate, FA rate, and RT over the session."""
    df = get_trial_dataframe(session)
    if df.empty:
        return pd.DataFrame()
        
    # Rolling Hit Rate (on Go trials only)
    go_trials = df[df['is_go']].copy()
    go_trials['rolling_hit'] = go_trials['is_hit'].rolling(window=window, min_periods=5).mean()
    
    # Rolling Miss Rate (on Go trials)
    go_trials['rolling_miss'] = go_trials['is_miss'].rolling(window=window, min_periods=5).mean()

    # Rolling FA Rate (on all trials? or just non-aborts?)
    # Let's do rolling FA density over valid trials (non-aborts)
    valid_trials = df[~df['is_abort']].copy()
    valid_trials['rolling_fa'] = valid_trials['is_fa'].rolling(window=window, min_periods=5).mean()
    
    # Map back to original index
    df['rolling_hit_rate'] = np.nan
    df.loc[go_trials.index, 'rolling_hit_rate'] = go_trials['rolling_hit']
    
    df['rolling_miss_rate'] = np.nan
    df.loc[go_trials.index, 'rolling_miss_rate'] = go_trials['rolling_miss']
    
    df['rolling_fa_rate'] = np.nan
    df.loc[valid_trials.index, 'rolling_fa_rate'] = valid_trials['rolling_fa']
    
    # Interpolate
    df['rolling_hit_rate'] = df['rolling_hit_rate'].interpolate(method='linear', limit_direction='both')
    df['rolling_miss_rate'] = df['rolling_miss_rate'].interpolate(method='linear', limit_direction='both')
    df['rolling_fa_rate'] = df['rolling_fa_rate'].interpolate(method='linear', limit_direction='both')
    
    # Identify State
    df['state'] = df.apply(identify_session_state, axis=1)
    
    return df

def compute_psychometric_data(session: Session) -> pd.DataFrame:
    """Compute Hit Rate per Change Size."""
    df = get_trial_dataframe(session)
    if df.empty or 'change_size' not in df.columns:
        return pd.DataFrame()
        
    # Filter for Go trials (Hits + Misses)
    go_df = df[df['is_go']]
    
    if go_df.empty:
        return pd.DataFrame()

    # Group by change size
    stats = go_df.groupby('change_size').agg(
        n_trials=('is_hit', 'count'),
        n_hits=('is_hit', 'sum'),
        mean_rt=('rt', 'mean'),
        sem_rt=('rt', 'sem')
    ).reset_index()
    
    stats['hit_rate'] = stats['n_hits'] / stats['n_trials']
    
    # Calculate error bars for hit rate (binomial proportion confidence interval - Wald or Agresti-Coull)
    # Simple Wald: sqrt(p(1-p)/n)
    stats['sem_hit_rate'] = np.sqrt(stats['hit_rate'] * (1 - stats['hit_rate']) / stats['n_trials'])
    
    return stats
