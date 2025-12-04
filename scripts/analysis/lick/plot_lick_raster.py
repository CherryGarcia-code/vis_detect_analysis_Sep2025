"""
Plot raster for lick-responsive clusters from .npz traces file.

Usage:
    python scripts/analysis/lick/plot_lick_raster.py --npz <lick_traces.npz> --outdir <FIGURES/session_folder> [--cluster-ids 1 2 3 ...]

If --cluster-ids is omitted, all clusters are plotted.
Each plot is saved as <outdir>/lick_raster_cluster_<id>.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot rasters for lick-responsive clusters.")
    parser.add_argument('--npz', required=True, help='Input .npz file from find_lick_responsive_neurons.py')
    parser.add_argument('--outdir', required=True, help='Output directory for plots')
    parser.add_argument('--cluster-ids', type=int, nargs='*', help='Cluster IDs to plot (default: all)')
    args = parser.parse_args()

    # NOTE: The .npz file does not currently contain trial-by-trial rasters.
    # This script is a placeholder for when that data is available.
    print("Trial-by-trial rasters are not yet saved in the .npz file. Please update the pipeline to include them.")

if __name__ == '__main__':
    main()
