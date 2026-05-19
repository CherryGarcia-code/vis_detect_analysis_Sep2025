"""
Plot histograms of trial counts across sessions for each outcome.

Generates:
1. Histograms of trial counts (X: Trial Count Bins, Y: Number of Sessions) for Hits, Misses, FAs, Aborts.

Usage:
    python scripts/analysis/behavior/plot_cross_session_trial_counts_hist.py --manifest data/BG_046_sessions_manifest.csv --out FIGURES/behavior/cross_session_BG_046
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys


from visdetect.viz.plotting import set_style, despine
from visdetect.analysis.config import load_staging_manifest

def main():
    parser = argparse.ArgumentParser(description="Plot cross-session trial count histograms.")
    parser.add_argument('--manifest', default=None, help='Path to sessions manifest CSV (default: canonical)')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--no-filter', action='store_true', help='Bypass SESSION_FILTER')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Manifest (centralized: applies SESSION_FILTER, sorts chronologically)
    df = load_staging_manifest(manifest_path=args.manifest, apply_filter=not args.no_filter)
    subject = df.iloc[0]['subject'] if 'subject' in df.columns else "Unknown Subject"

    # Use 'talk' context
    set_style(context='talk')

    # Calculate fractions
    if 'n_trials' in df.columns:
        df['frac_hits'] = df['n_hits'] / df['n_trials']
        df['frac_miss'] = df['n_miss'] / df['n_trials']
        df['frac_fa'] = df['n_fa'] / df['n_trials']
        df['frac_abort'] = df['n_abort'] / df['n_trials']
    else:
        print("Warning: 'n_trials' not found in manifest. Fractions will not be plotted.")
        for col in ['frac_hits', 'frac_miss', 'frac_fa', 'frac_abort']:
            df[col] = 0

    outcomes = [
        ('Hits', 'n_hits', 'frac_hits', 'green'),
        ('Misses', 'n_miss', 'frac_miss', 'purple'),
        ('FAs', 'n_fa', 'frac_fa', 'red'),
        ('Aborts', 'n_abort', 'frac_abort', 'gray')
    ]
    
    # 4 rows (outcomes), 2 columns (Counts, Fractions)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    for i, (label, col_count, col_frac, color) in enumerate(outcomes):
        # Plot Count (Left Column)
        ax_count = axes[i, 0]
        if col_count in df.columns:
            sns.histplot(data=df, x=col_count, ax=ax_count, color=color, kde=False, bins=15)
            ax_count.set_title(f"{label} Count Distribution")
            ax_count.set_xlabel("Number of Trials")
            ax_count.set_ylabel("Number of Sessions")
            despine(ax_count)
        else:
            ax_count.text(0.5, 0.5, f"{col_count} missing", ha='center')
            ax_count.axis('off')

        # Plot Fraction (Right Column)
        ax_frac = axes[i, 1]
        if col_frac in df.columns and 'n_trials' in df.columns:
            sns.histplot(data=df, x=col_frac, ax=ax_frac, color=color, kde=False, bins=15)
            ax_frac.set_title(f"{label} Fraction Distribution")
            ax_frac.set_xlabel("Fraction of Total Trials")
            ax_frac.set_ylabel("Number of Sessions")
            ax_frac.set_xlim(0, 1.0) # Fractions are 0-1
            despine(ax_frac)
        else:
            ax_frac.text(0.5, 0.5, "Data missing", ha='center')
            ax_frac.axis('off')

    plt.suptitle(f"Distribution of Trial Counts & Fractions: {subject}", y=1.01)
    plt.tight_layout()
    
    out_path = out_dir / "trial_counts_histogram_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Trial count histograms saved to {out_path}")

if __name__ == "__main__":
    main()
