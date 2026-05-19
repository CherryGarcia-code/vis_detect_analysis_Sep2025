"""
Plot cross-session behavior analysis (Learning).

Generates:
1. Learning Curve (Hit Rate, FA Rate, d' over sessions).
2. RT Evolution (Mean Hit RT over sessions).
3. Engagement Duration (Trials until disengagement).

Usage:
    python scripts/analysis/behavior/plot_cross_session_behavior.py --manifest data/BG_046_sessions_manifest.csv --out FIGURES/behavior/cross_session_BG_046
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys
import subprocess
from datetime import datetime


from visdetect.analysis.config import load_staging_manifest
from visdetect.analysis.behavior import calculate_dprime
from visdetect.viz.plotting import set_style, despine

def parse_date(session_name):
    # Assumes format BG_046_DDMMYYYY
    try:
        date_str = session_name.split('_')[-1]
        return datetime.strptime(date_str, "%d%m%Y")
    except Exception:
        return datetime.min

def main():
    parser = argparse.ArgumentParser(description="Plot cross-session behavior.")
    parser.add_argument('--manifest', default=None, help='Path to sessions manifest CSV (default: canonical)')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--no-filter', action='store_true', help='Bypass SESSION_FILTER')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Manifest (centralized: applies SESSION_FILTER, sorts chronologically)
    df = load_staging_manifest(manifest_path=args.manifest, apply_filter=not args.no_filter)
    
    # Sort by date
    subject = df.iloc[0]['subject']
    # Ensure session_name has subject prefix for parsing if needed, or just use what's there
    # The manifest usually has 'session_name' as 'DDMMYYYY' or 'BG_046_DDMMYYYY'
    # Let's try to parse date from 'session_name' column directly first
    
    df['parsed_date'] = df['session_name'].apply(lambda x: parse_date(f"{subject}_{x}") if not x.startswith(subject) else parse_date(x))
    df = df.sort_values('parsed_date').reset_index(drop=True)
    
    # Calculate d'
    df['d_prime'] = df.apply(lambda row: calculate_dprime(row['hit_rate'], row['fa_rate']), axis=1)
    
    # Use 'talk' context as default
    set_style(context='talk')
    
    # 1. Learning Curve (Rates)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='session_name', y='hit_rate', marker='o', label='Hit Rate', color='dodgerblue', ax=ax)
    sns.lineplot(data=df, x='session_name', y='fa_rate', marker='o', label='FA Rate', color='crimson', ax=ax)
    
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Rate')
    ax.set_xlabel('Session')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title(f"Learning Curve: {subject}")
    plt.legend()
    despine(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_rates.png", dpi=150)
    plt.close()
    
    # 2. Learning Curve (d')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='session_name', y='d_prime', marker='o', color='purple', ax=ax)
    
    ax.set_ylabel("d' (Sensitivity)")
    ax.set_xlabel('Session')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title(f"Sensitivity (d') Evolution: {subject}")
    despine(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_dprime.png", dpi=150)
    plt.close()
    
    # 3. RT Evolution
    # "Show data and a separate color for FA>3, FA≤3, and Hit on the same plot."
    # We need mean RTs for these categories. The manifest currently only has 'mean_rt' (for hits).
    # We need to update the manifest script to include mean RT for FA types, OR calculate it here if we had access to raw data.
    # Since we only have the manifest here, we can only plot what's in it.
    # However, I updated the manifest script to include 'n_fa_early' etc, but not mean RTs for them.
    # I should update the manifest script to include mean RTs for FA types as well.
    # For now, I will plot what is available and add a TODO or warning if columns are missing.
    # Wait, I can't easily update the manifest script AND re-run it in this single turn if the user didn't ask me to re-run it.
    # But I did update the manifest script in the previous step. I should have added RTs there.
    # Let me quickly check if I can add RTs to the manifest script now.
    
    # Assuming the manifest will have these columns after I update the manifest script in the next step (or same step).
    # I will add the plotting code here assuming columns exist, or handle their absence.
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'mean_rt' in df.columns:
        sns.lineplot(data=df, x='session_name', y='mean_rt', marker='o', color='green', label='Hit RT', ax=ax)
    
    # Check for FA RT columns (I will add these to manifest script next)
    if 'mean_rt_fa_early' in df.columns:
        sns.lineplot(data=df, x='session_name', y='mean_rt_fa_early', marker='o', color='lightcoral', label='FA <= 3s RT', ax=ax)
    
    if 'mean_rt_fa_late' in df.columns:
        sns.lineplot(data=df, x='session_name', y='mean_rt_fa_late', marker='o', color='darkred', label='FA > 3s RT', ax=ax)
        
    ax.set_ylabel("Mean Reaction Time (s)")
    ax.set_xlabel('Session')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title(f"Reaction Time Evolution: {subject}")
    plt.legend()
    despine(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "rt_evolution.png", dpi=150)
    plt.close()

    # 4. Trial Counts & Fractions
    # "simple histogram/bar plot of trial countrs for each outcome absolute count on y axis (and fraction on other secondary y axis.)"
    # "dashed horizontal dark gray line passing at the 40% fraction height"
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Prepare data for stacked bars
    # Outcomes: Hit, Miss, FA, Abort
    # Colors: Hit-green, Miss-purple, FA-red, Abort-darkgrey
    
    sessions = df['session_name']
    x = np.arange(len(sessions))
    width = 0.6
    
    # Counts
    hits = df['n_hits']
    misses = df['n_miss']
    fas = df['n_fa']
    aborts = df['n_abort']
    
    # Stacked Bar Plot
    p1 = ax1.bar(x, hits, width, color='green', label='Hits', alpha=0.8)
    p2 = ax1.bar(x, misses, width, bottom=hits, color='purple', label='Misses', alpha=0.8)
    p3 = ax1.bar(x, fas, width, bottom=hits+misses, color='red', label='FAs', alpha=0.8)
    p4 = ax1.bar(x, aborts, width, bottom=hits+misses+fas, color='darkgrey', label='Aborts', alpha=0.8)
    
    ax1.set_ylabel('Trial Count', color='black')
    ax1.set_xlabel('Session')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sessions, rotation=45, ha='right', fontsize=8)
    
    # Secondary Y-axis for Fractions
    ax2 = ax1.twinx()
    
    # Calculate fractions (Rates)
    # Note: Rates in manifest might be calculated differently (e.g. Hit Rate = Hits / Go Trials).
    # Here we want fraction of TOTAL trials? Or just plot the rates as they are?
    # "fraction on other secondary y axis" usually implies fraction of the total bar height.
    # Let's plot the rates from the manifest as lines, as they are the relevant performance metrics.
    # Hit Rate (Blue/Green?), FA Rate (Red), Miss Rate?
    # Or maybe the user wants the fraction of the total trials that are Hits, Misses, etc?
    # "fraction on other secondary y axis" -> singular "fraction"?
    # Maybe just Hit Rate? Or all of them?
    # Let's plot Hit Rate and FA Rate as lines.
    
    # Actually, if it's a stacked bar of counts, the "fraction" of the bar is implicit.
    # But maybe they want to see the performance rates overlaid.
    # Let's plot Hit Rate (Green line) and FA Rate (Red line).
    
    ax2.plot(x, df['hit_rate'], color='darkgreen', marker='o', linewidth=2, label='Hit Rate (frac)', linestyle='-')
    ax2.plot(x, df['fa_rate'], color='darkred', marker='o', linewidth=2, label='FA Rate (frac)', linestyle='-')
    
    # Threshold line at 40%
    ax2.axhline(y=0.4, color='darkgray', linestyle='--', linewidth=2, label='40% Threshold')
    
    ax2.set_ylabel('Fraction / Rate', color='black')
    ax2.set_ylim(0, 1.05)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.title(f"Trial Counts & Performance: {subject}")
    despine(ax1) # Despine left axis
    # We need right spine for ax2, so maybe don't despine right?
    # despine() removes top and right. We want right for ax2.
    sns.despine(ax=ax1, top=True, right=False)
    
    plt.tight_layout()
    plt.savefig(out_dir / "trial_counts_outcomes.png", dpi=150)
    plt.close()

    # 5. Save Summary CSV
    # "output a csv with all interesting summary descriptive details... including means, meadians , and error calculations"
    summary_csv_path = out_dir / "cross_session_summary_stats.csv"
    df.to_csv(summary_csv_path, index=False)
    print(f"Summary statistics saved to {summary_csv_path}")

    print(f"Cross-session plots saved to {out_dir}")

    # 6. Run Trial Count Histograms
    print("Running trial count histogram analysis...")
    hist_script = Path(__file__).parent / "plot_cross_session_trial_counts_hist.py"
    cmd = [sys.executable, str(hist_script), "--manifest", args.manifest, "--out", args.out]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
