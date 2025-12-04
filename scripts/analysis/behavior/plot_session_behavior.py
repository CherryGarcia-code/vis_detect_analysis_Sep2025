"""
Plot single-session behavior analysis.

Generates:
1. Performance Summary (Rolling Hit/Response rates, RTs).
2. Psychometric Curve (Hit Rate vs Change Size).
3. RT Distribution.

Usage:
    python scripts/analysis/behavior/plot_session_behavior.py --session BG_046_17092025 --pkl pkls/BG_046/BG_046_17092025.pkl --out FIGURES/behavior/BG_046_17092025
"""
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Ensure repo root is in path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root / 'src') not in sys.path:
    sys.path.insert(0, str(repo_root / 'src'))

from visdetect.core.session import load_session
from visdetect.analysis.behavior import compute_rolling_performance, compute_psychometric_data, get_trial_dataframe
from visdetect.viz.plotting import set_style, despine

def plot_session_behavior(session, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use 'talk' context as default for now, or make it configurable
    set_style(context='talk')
    
    # 1. Rolling Performance (Engagement)
    df_rolling = compute_rolling_performance(session)
    if not df_rolling.empty:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Background shading for states
        # We need to find segments where state is constant
        # Simplified approach: iterate and fill
        # Or just color the background based on state at each point (might be too busy if it flickers)
        # Let's use fill_between for blocks
        
        # Define colors for states
        state_colors = {'impulsive': '#ffcccc', 'disengaged': '#e6e6fa', 'balanced': 'white'} # Light Red, Lavender, White
        
        # Create a numeric mapping for states to use fill_between logic or just iterate
        # Iterating is safer for variable length segments
        # Optimization: find change points
        df_rolling['state_group'] = (df_rolling['state'] != df_rolling['state'].shift()).cumsum()
        for _, group in df_rolling.groupby('state_group'):
            state = group['state'].iloc[0]
            if state in state_colors and state != 'balanced':
                ax1.axvspan(group['trial_idx'].min(), group['trial_idx'].max(), color=state_colors[state], alpha=0.25, lw=0)

        # Plot Outcomes (Scatter)
        # Hit-green (Exclude change size =1 for this plot; also have option for separating into change sizes)
        # FA>3 seconds - dark red, FA≤3 seconds -light red, abort - dark grey, miss - violet.
        
        # Filter data
        hits = df_rolling[(df_rolling['is_hit']) & (df_rolling['change_size'] != 1)]
        misses = df_rolling[df_rolling['is_miss']]
        aborts = df_rolling[df_rolling['is_abort']]
        
        # Classify FAs
        fas = df_rolling[df_rolling['is_fa']].copy()
        fas['fa_type'] = fas['rt'].apply(lambda x: 'early' if x <= 3.0 else 'late')
        fas_early = fas[fas['fa_type'] == 'early']
        fas_late = fas[fas['fa_type'] == 'late']
        
        # Plot markers at y=1.05 (or just above/below lines) or as a raster at the top?
        # "Add to session dynamics all other outcome rates" -> The user might mean rolling rates of these outcomes.
        # "Show color appropriate shading accordingly throughout the session" -> Done above.
        # "Outcome colors..." -> This implies plotting the outcomes themselves or their rates.
        # Let's plot the Rolling Rates of these outcomes with the specified colors.
        
        # Calculate rolling rates for specific outcomes if not already in df_rolling
        # We have rolling_hit_rate (Go trials), rolling_miss_rate (Go trials), rolling_fa_rate (Valid trials)
        # Let's add rolling abort rate
        df_rolling['rolling_abort'] = df_rolling['is_abort'].rolling(window=30, min_periods=5).mean()
        
        # Plot Lines
        ax1.plot(df_rolling['trial_idx'], df_rolling['rolling_hit_rate'], color='green', label='Hit Rate', linewidth=2)
        ax1.plot(df_rolling['trial_idx'], df_rolling['rolling_miss_rate'], color='purple', label='Miss Rate', linewidth=2)
        ax1.plot(df_rolling['trial_idx'], df_rolling['rolling_fa_rate'], color='red', label='FA Rate', linewidth=2) # Combined FA for line
        ax1.plot(df_rolling['trial_idx'], df_rolling['rolling_abort'], color='darkgrey', label='Abort Rate', linewidth=2)
        
        ax1.set_xlabel('Trial Index')
        ax1.set_ylabel('Rate')
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        # Add title
        plt.title(f"Session Dynamics: {session.session_name}")
        
        despine(ax1)
        plt.tight_layout()
        plt.savefig(out_dir / "session_dynamics.png", dpi=150)
        plt.close()

    # 2. Psychometric Curve
    psy_df = compute_psychometric_data(session)
    if not psy_df.empty and len(psy_df) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # "Performance" = Hits / (Hits + Misses) -> This is what 'hit_rate' in psy_df is (n_hits / n_trials where n_trials is Go trials)
        # "make the spacing of the x axis labels equal (I.e. as ordinal, not numeric continuous numbers)"
        
        x_pos = np.arange(len(psy_df))
        
        # "change line, marker and error bar colors to black."
        ax.errorbar(x_pos, psy_df['hit_rate'], yerr=psy_df['sem_hit_rate'], 
                    fmt='o-', color='black', capsize=5, linewidth=2, markersize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(psy_df['change_size'])
        ax.set_xlabel('Change Size (deg)')
        ax.set_ylabel('Performance (Hit Rate)')
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Psychometric Curve: {session.session_name}")
        
        despine(ax)
        plt.tight_layout()
        plt.savefig(out_dir / "psychometric_curve.png", dpi=150)
        plt.close()

    # 3. RT Distribution
    df_trials = get_trial_dataframe(session)
    if not df_trials.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use 'response_time' (RT + change_time) for Hits as requested
        hits = df_trials[df_trials['is_hit']]['response_time'].dropna()
        
        # For FAs, we might want raw RT or response time?
        # Usually FA RT is defined relative to trial start anyway.
        # If change_time is 0 for FA, then response_time == rt.
        # Let's use response_time for consistency if that's what the user implies by "Hit rt... needs to be correct".
        # But let's stick to the specific request: "Hit rt... without adding baseline... needs to be correct".
        # So for Hits, use response_time.
        
        # Split FAs
        fas = df_trials[df_trials['is_fa']]
        # Use 'rt' for FAs for now unless specified otherwise, as FA doesn't have a "change time" baseline usually.
        # Or does it? If it's a catch trial, maybe there's a virtual change time?
        # Let's stick to 'rt' for FAs to be safe, or 'response_time' if change_time is 0.
        # Actually, if change_time is 0 for FA, response_time = rt.
        fas_early = fas[fas['rt'] <= 3.0]['rt'].dropna()
        fas_late = fas[fas['rt'] > 3.0]['rt'].dropna()
        
        if len(hits) > 0:
            sns.histplot(hits, color='green', label=f'Hits (n={len(hits)})', kde=False, ax=ax, alpha=0.6, bins=20, element="step")
        if len(fas_early) > 0:
            sns.histplot(fas_early, color='lightcoral', label=f'FA <= 3s (n={len(fas_early)})', kde=False, ax=ax, alpha=0.6, bins=20, element="step")
        if len(fas_late) > 0:
            sns.histplot(fas_late, color='darkred', label=f'FA > 3s (n={len(fas_late)})', kde=False, ax=ax, alpha=0.6, bins=20, element="step")
            
        ax.set_xlabel('Response Time (s)') # Changed label to reflect it's from trial start for Hits
        ax.set_ylabel('Count')
        ax.set_title(f"Response Time Distribution: {session.session_name}")
        ax.legend()
        
        despine(ax)
        plt.tight_layout()
        plt.savefig(out_dir / "rt_distribution.png", dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot single session behavior.")
    parser.add_argument('--session', required=True, help='Session name')
    parser.add_argument('--pkl', required=True, help='Path to session pkl')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    try:
        session = load_session(args.pkl)
        plot_session_behavior(session, args.out)
        print(f"Behavior plots saved to {args.out}")
    except Exception as e:
        print(f"Error processing {args.session}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
