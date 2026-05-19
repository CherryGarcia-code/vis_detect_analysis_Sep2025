import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path


from visdetect.analysis.config import load_staging_manifest

def plot_learning_curve(df, output_path):
    """Plot d' learning curve from a pre-loaded manifest DataFrame."""
    # Sort by date
    # Format dates as strings with padding (d/m/y -> 0d/0m/y)
    df['date_str'] = df['date'].astype(str).str.zfill(8)
    df['dt'] = pd.to_datetime(df['date_str'], format='%d%m%Y')
    df = df.sort_values('dt')

    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='dt', y='d_prime', marker='o')
    
    # Color points by stage
    sns.scatterplot(data=df, x='dt', y='d_prime', hue='stage', s=100, zorder=5)
    
    plt.axhline(1.5, color='r', linestyle='--', label='Expert Threshold (approx)')
    plt.xticks(rotation=45)
    plt.title("Learning Curve: d' over sessions")
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"Saved learning curve to {output_path}")

if __name__ == "__main__":
    manifest = load_staging_manifest()
    plot_learning_curve(manifest, "FIGURES/learning_curve_BG_046.png")
