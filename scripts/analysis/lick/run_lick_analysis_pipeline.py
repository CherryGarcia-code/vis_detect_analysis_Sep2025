"""
Run the full lick analysis pipeline for a single session.
Outputs all plots and a summary HTML, PDF, and PNG in FIGURES/lick/<session>/

Usage:
python scripts/analysis/lick/run_lick_analysis_pipeline.py --session BG_046_16092025 --pkl pkls/BG_046_16092025.pkl --out FIGURES/lick/BG_046_16092025"""
import argparse
import os
from pathlib import Path
import subprocess
from datetime import datetime

PLOTS = [
    ("find_lick_responsive_neurons.py", "--session-pkl {pkl} --out {out_dir}/lick_responsiveness.csv"),
    ("plot_lick_responsiveness_summary.py", "--csv {out_dir}/lick_responsiveness.csv --out {out_dir}/lick_responsiveness_summary.png"),
    ("plot_lick_heatmap.py", "--npz {out_dir}/lick_responsiveness.npz --csv {out_dir}/lick_responsiveness.csv --out {out_dir}/lick_heatmap.png"),
    ("plot_lick_mean_psth_posneg.py", "--npz {out_dir}/lick_responsiveness.npz --csv {out_dir}/lick_responsiveness.csv --out {out_dir}/lick_mean_psth_posneg.png")
]

HTML_TEMPLATE = """
<html><head><title>Lick Analysis Summary: {session}</title></head><body>
<h1>Lick Analysis Summary: {session}</h1>
{plots}
</body></html>
"""

def run(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run lick analysis pipeline for a session.")
    parser.add_argument('--session', required=True, help='Session name (e.g. BG_046_17092025)')
    parser.add_argument('--pkl', required=True, help='Path to session pkl file')
    parser.add_argument('--out', required=True, help='Output directory (e.g. FIGURES/lick/BG_046_17092025)')
    parser.add_argument('--stats-only', action='store_true', help='Only run stats generation, skip plotting and summary')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    for script, argstr in PLOTS:
        # If stats-only, skip everything except find_lick_responsive_neurons.py
        if args.stats_only and "find_lick_responsive_neurons.py" not in script:
            continue
            
        script_path = f"scripts/analysis/lick/{script}"
        cmd = f"python {script_path} {argstr.format(pkl=args.pkl, out_dir=args.out, session=args.session)}"
        run(cmd)

    if args.stats_only:
        print(f"Stats generation complete for {args.session}. Skipping plots/summary.")
        return

    # Collect all PNGs for summary
    plot_files = sorted([f for f in out_dir.glob("*.png")])
    # Save a multi-page PDF with all plots arranged in 2x2 panels per page
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg

    pdf_path = out_dir / "lick_analysis_summary.pdf"
    n_per_page = 4  # 2x2 grid
    with PdfPages(pdf_path) as pdf:
        for i in range(0, len(plot_files), n_per_page):
            figs, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs = axs.flatten()
            for j, ax in enumerate(axs):
                idx = i + j
                if idx < len(plot_files):
                    img = mpimg.imread(plot_files[idx])
                    ax.imshow(img)
                    ax.set_title(plot_files[idx].name, fontsize=10)
                    ax.axis('off')
                else:
                    ax.axis('off')
            plt.tight_layout()
            pdf.savefig(figs)
            plt.close(figs)
    print(f"Saved PDF summary to {pdf_path}")

    # Optionally, still generate HTML summary for browser viewing
    plot_tags = "\n".join([f'<h2>{f.name}</h2><img src="{f.name}" style="max-width:800px;"><br>' for f in plot_files])
    html = HTML_TEMPLATE.format(session=args.session, plots=plot_tags)
    html_path = out_dir / f"lick_analysis_summary.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved HTML summary to {html_path}")

if __name__ == "__main__":
    main()
