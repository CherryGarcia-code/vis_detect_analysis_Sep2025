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

        # Determine trial type from change_size, NOT from outcome label.
        # Go trial: change_size > 1 (stimulus changed).  Catch trial: change_size ≈ 1 (no change).
        cs = t.change_size if t.change_size is not None else 1.0
        is_go_trial = (cs - 1.0) > 0.01
        is_catch_trial = not is_go_trial

        rows.append({
            'trial_idx': i,
            'outcome': outcome,
            'change_size': cs,
            'orientation': t.orientation if t.orientation is not None else 0,
            'change_time': t.change_time if t.change_time is not None else 0,
            'rt': rt,
            'is_hit': outcome == 'hit',       # Behavioral label: licked in response window
            'is_miss': outcome == 'miss',     # Behavioral label: withheld lick
            'is_fa': outcome == 'fa',         # Behavioral label: early/anticipatory lick (NOT SDT false alarm)
            'is_abort': outcome == 'abort',
            'is_ref': outcome == 'ref',       # Behavioral label: reflex lick (too fast)
            'is_go': is_go_trial,             # Trial type from change_size (NOT outcome label)
            'is_catch': is_catch_trial,       # Trial type from change_size (NOT outcome label)
        })
    
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Calculate Response Time (Time from Trial Start)
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
    """Compute aggregate performance metrics for a session.

    SDT Classification (uses change_size, NOT behavioral software labels):
      - True Hit:  outcome='hit'  on go trial   (change_size > 1)  — correct detection
      - True Miss: outcome='miss' on go trial   (change_size > 1)  — failed detection
      - True FA:   outcome='hit'  on catch trial (change_size ≈ 1) — licked when no change
      - True CR:   outcome='miss' on catch trial (change_size ≈ 1) — correctly withheld
      - Excluded from SDT: 'fa' (early/anticipatory lick), 'ref' (reflex), 'abort'

    Behavioral label counts (n_hits, n_miss, n_fa, n_abort) and fraction_* columns
    still use the raw software labels so that QC rules remain compatible.
    """
    df = get_trial_dataframe(session)
    if df.empty:
        return {}

    n_trials = len(df)

    # --- Behavioral label counts (raw software labels, for QC & RT analysis) ---
    n_hits_label = int(df['is_hit'].sum())
    n_miss_label = int(df['is_miss'].sum())
    n_fa_label = int(df['is_fa'].sum())       # early/anticipatory licks
    n_abort = int(df['is_abort'].sum())
    n_ref = int(df['is_ref'].sum()) if 'is_ref' in df.columns else 0

    # --- SDT classification (change_size-based) ---
    sdt_hits   = int(((df['is_go'])    & (df['outcome'] == 'hit')).sum())
    sdt_misses = int(((df['is_go'])    & (df['outcome'] == 'miss')).sum())
    sdt_fas    = int(((df['is_catch']) & (df['outcome'] == 'hit')).sum())
    sdt_crs    = int(((df['is_catch']) & (df['outcome'] == 'miss')).sum())

    n_go    = sdt_hits + sdt_misses
    n_catch = sdt_fas  + sdt_crs

    hit_rate  = sdt_hits   / n_go    if n_go    > 0 else 0.0
    miss_rate = sdt_misses / n_go    if n_go    > 0 else 0.0
    fa_rate   = sdt_fas    / n_catch if n_catch > 0 else 0.0

    # hit_rate already excludes catch trials (change_size ≈ 1), so identical
    hit_rate_no_1 = hit_rate

    abort_rate = n_abort / n_trials if n_trials > 0 else 0.0

    # Behavioral label fractions (for QC rules — not SDT)
    fraction_hit   = n_hits_label  / n_trials if n_trials > 0 else 0.0
    fraction_miss  = n_miss_label  / n_trials if n_trials > 0 else 0.0
    fraction_fa    = n_fa_label    / n_trials if n_trials > 0 else 0.0
    fraction_abort = n_abort       / n_trials if n_trials > 0 else 0.0

    # Early-lick (behavioral "FA") split by RT
    fas_early_lick = df[df['is_fa']].copy()
    n_fa_early = int(fas_early_lick[fas_early_lick['rt'] <= 3.0].shape[0])
    n_fa_late  = int(fas_early_lick[fas_early_lick['rt'] > 3.0].shape[0])

    mean_rt_fa_early = fas_early_lick[fas_early_lick['rt'] <= 3.0]['rt'].mean()
    mean_rt_fa_late  = fas_early_lick[fas_early_lick['rt'] > 3.0]['rt'].mean()

    sem_rt_fa_early = fas_early_lick[fas_early_lick['rt'] <= 3.0]['rt'].sem()
    sem_rt_fa_late  = fas_early_lick[fas_early_lick['rt'] > 3.0]['rt'].sem()

    # Hit RTs — only genuine SDT hits (go trials with 'hit' outcome)
    hit_trials = df[(df['is_go']) & (df['outcome'] == 'hit')]
    sem_rt_hit = hit_trials['rt'].sem()

    # d-prime
    d_prime = calculate_dprime(hit_rate, fa_rate)

    return {
        'n_trials': n_trials,
        'hit_rate': hit_rate,                   # SDT hit rate
        'miss_rate': miss_rate,                 # SDT miss rate
        'hit_rate_no_size_1': hit_rate_no_1,    # same as hit_rate (go trials only)
        'fa_rate_total': fa_rate,               # SDT FA rate
        'abort_rate': abort_rate,
        'fraction_hit': fraction_hit,           # behavioral label fraction (for QC)
        'fraction_miss': fraction_miss,
        'fraction_fa': fraction_fa,
        'fraction_abort': fraction_abort,
        'd_prime': d_prime,
        'n_hits': n_hits_label,                 # behavioral "Hit" label count
        'n_miss': n_miss_label,                 # behavioral "Miss" label count
        'n_fa': n_fa_label,                     # behavioral "FA" label count (early licks)
        'n_fa_early': n_fa_early,               # early lick with RT ≤ 3s
        'n_fa_late': n_fa_late,                 # early lick with RT > 3s
        'n_abort': n_abort,
        'n_ref': n_ref,
        'n_sdt_hits': sdt_hits,
        'n_sdt_misses': sdt_misses,
        'n_sdt_fas': sdt_fas,
        'n_sdt_crs': sdt_crs,
        'n_go': n_go,
        'n_catch': n_catch,
        'mean_rt_hit': hit_trials['rt'].mean(),
        'median_rt_hit': hit_trials['rt'].median(),
        'mean_rt_fa_early': mean_rt_fa_early,
        'mean_rt_fa_late': mean_rt_fa_late,
        'sem_rt_hit': sem_rt_hit,
        'sem_rt_fa_early': sem_rt_fa_early,
        'sem_rt_fa_late': sem_rt_fa_late,
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
        
    # SDT-valid go trials: 'hit' or 'miss' outcome on change_size > 1 trials.
    # Excludes early licks ('fa'), reflex ('ref'), and aborts on go trials.
    go_df = df[(df['is_go']) & (df['outcome'].isin(['hit', 'miss']))]

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


def filter_manifest_by_stage(
    manifest_df: pd.DataFrame,
    include_stages: Optional[List[str]] = None,
    exclude_stages: Optional[List[str]] = None,
    merge_naive_learning: bool = False,
    min_trials: Optional[int] = None,
    min_dprime: Optional[float] = None,
    stage_specific_dprime: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Filter and optionally regroup manifest by learning stage.
    
    Provides flexible filtering for comparing different learning phases
    (e.g., Naive vs Expert, Learning + Naive as "Early", etc.). Does NOT
    modify the original dataframe.
    
    Parameters
    ----------
    manifest_df : pd.DataFrame
        Staging manifest with 'stage' column (from stage_sessions.py)
    include_stages : list of str, optional
        Stages to include (e.g., ['Learning', 'Expert']). If None, include all
        non-Excluded stages.
    exclude_stages : list of str, optional
        Stages to explicitly exclude (applied after include_stages).
    merge_naive_learning : bool, default=False
        If True, create a 'stage_group' column where Naive sessions are
        relabeled as 'Learning'. Useful for "Early vs Expert" comparisons.
    min_trials : int, optional
        Minimum total trials (n_go + n_catch) to include.
    min_dprime : float, optional
        Minimum d' threshold applied to all included stages.
    stage_specific_dprime : dict, optional
        Stage-specific d' thresholds (e.g., {'Learning': 1.0, 'Expert': 1.5}).
        Overrides min_dprime for specified stages.
    
    Returns
    -------
    filtered_df : pd.DataFrame
        Filtered manifest with optional 'stage_group' column.
    
    Examples
    --------
    # 1. Compare Learning vs Expert only:
    >>> df = filter_manifest_by_stage(manifest, include_stages=['Learning', 'Expert'])
    
    # 2. Merge Naive into Learning, compare to Expert:
    >>> df = filter_manifest_by_stage(
    ...     manifest,
    ...     include_stages=['Naive', 'Learning', 'Expert'],
    ...     merge_naive_learning=True
    ... )
    >>> # Use df['stage_group'] for comparisons
    
    # 3. "Extremes" comparison (Naive vs Expert, high-quality only):
    >>> df = filter_manifest_by_stage(
    ...     manifest,
    ...     include_stages=['Naive', 'Expert'],
    ...     min_trials=200,
    ...     stage_specific_dprime={'Naive': 0.5, 'Expert': 1.5}
    ... )
    
    # 4. Learning trajectory with quality gate:
    >>> df = filter_manifest_by_stage(
    ...     manifest,
    ...     exclude_stages=['Excluded', 'Disengaged'],
    ...     min_dprime=0.8,
    ...     min_trials=150
    ... )
    
    Notes
    -----
    - Always excludes sessions with qc_fail=True (if that column exists)
    - Filtering is applied in order: qc_fail → include → exclude → trials → dprime
    - stage_group column is only created if merge_naive_learning=True
    """
    df = manifest_df.copy()
    
    # 1. Filter QC failures (if column exists)
    if 'qc_fail' in df.columns:
        df = df[df['qc_fail'] == False].reset_index(drop=True)
    
    # 2. Include stages
    if include_stages is not None:
        df = df[df['stage'].isin(include_stages)].reset_index(drop=True)
    else:
        # Default: include all except 'Excluded'
        df = df[df['stage'] != 'Excluded'].reset_index(drop=True)
    
    # 3. Exclude stages
    if exclude_stages is not None:
        df = df[~df['stage'].isin(exclude_stages)].reset_index(drop=True)
    
    # 4. Minimum trial count filter
    if min_trials is not None:
        if 'n_go' in df.columns and 'n_catch' in df.columns:
            df = df[(df['n_go'] + df['n_catch']) >= min_trials].reset_index(drop=True)
    
    # 5. d' filters
    if 'd_prime' in df.columns:
        # Stage-specific d' thresholds
        if stage_specific_dprime is not None:
            mask = pd.Series(True, index=df.index)
            for stage, thresh in stage_specific_dprime.items():
                stage_mask = df['stage'] == stage
                mask &= ~stage_mask | (df['d_prime'] >= thresh)
            df = df[mask].reset_index(drop=True)
        # Global d' threshold
        elif min_dprime is not None:
            df = df[df['d_prime'] >= min_dprime].reset_index(drop=True)
    
    # 6. Merge Naive → Learning: overwrite 'stage' so all downstream code
    #    (groupby, colors, legends) works without modification.  The
    #    original label is preserved in 'stage_original'.
    if merge_naive_learning:
        df['stage_original'] = df['stage'].copy()
        df.loc[df['stage'] == 'Naive', 'stage'] = 'Learning'
        # Legacy alias kept for any code that reads stage_group
        df['stage_group'] = df['stage']
    
    return df
