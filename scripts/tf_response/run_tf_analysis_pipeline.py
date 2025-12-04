"""
Run the full TF analysis pipeline for a single session.
Outputs all plots and a summary HTML, PDF in FIGURES/tf/<session>/

Usage:
    python scripts/tf_response/run_tf_analysis_pipeline.py --session BG_046_17092025 --pkl pkls/BG_046_17092025.pkl --out FIGURES/tf/BG_046_17092025
"""
import argparse
from pathlib import Path
import subprocess

PLOTS = [
    ("plot_tf_pulse_grid.py", "--file {pkl} --out {out_dir} --which both --cols 5"),
    ("find_splitters_from_tf_grid_csv.py", "--csv {out_dir}/tf_pulse_grid_both.csv --out {out_dir}/tf_splitters.csv"),
    ("barplot_top_splitters.py", "--csv {out_dir}/tf_splitters.csv --out {out_dir}/barplot_top_splitters.png"),
    ("heatmap_zscore_diff.py", "--csv {out_dir}/tf_splitters.csv --out {out_dir}/heatmap_zscore_diff.png"),
    ("pairwise_lineplot_splitters.py", "--csv {out_dir}/tf_splitters.csv --out {out_dir}/pairwise_lineplot_splitters.png"),
    ("scatter_split_score_vs_zmax.py", "--csv {out_dir}/tf_splitters.csv --out {out_dir}/scatter_split_score_vs_zmax.png"),
    ("scatter_zmax_fast_vs_slow.py", "--csv {out_dir}/tf_splitters.csv --out {out_dir}/scatter_zmax_fast_vs_slow.png")
]

HTML_TEMPLATE = """
<html><head><title>TF Analysis Summary: {session}</title></head><body>
<h1>TF Analysis Summary: {session}</h1>
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
    parser = argparse.ArgumentParser(description="Run TF analysis pipeline for a session.")
    parser.add_argument('--session', required=True, help='Session name (e.g. BG_046_17092025)')
    parser.add_argument('--pkl', required=True, help='Path to session pkl file')
    parser.add_argument('--out', required=True, help='Output directory (e.g. FIGURES/tf/BG_046_17092025)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)


    # Run plot_tf_pulse_grid.py first, then find the correct tf_pulse_grid_both.csv path
    script, argstr = PLOTS[0]
    script_path = f"scripts/analysis/tf_response/{script}"
    cmd = f"python {script_path} {argstr.format(pkl=args.pkl, out_dir=args.out, session=args.session)}"
    run(cmd)

    # Find tf_pulse_grid_both.csv (search subfolders if needed)
    import glob
    import os
    tf_csv = os.path.join(args.out, "tf_pulse_grid_both.csv")
    if not os.path.exists(tf_csv):
        # Search for it in subfolders
        matches = glob.glob(os.path.join(args.out, "**", "tf_pulse_grid_both.csv"), recursive=True)
        if matches:
            tf_csv = matches[0]
        else:
            raise FileNotFoundError("tf_pulse_grid_both.csv not found in output directory or subfolders.")

    import os
    tf_dir = os.path.dirname(tf_csv)
    # 1. find_splitters_from_tf_grid_csv.py: use found tf_csv and tf_dir/tf_splitters.csv
    script_path = "scripts/analysis/tf_response/find_splitters_from_tf_grid_csv.py"
    splitters_csv = os.path.join(tf_dir, "tf_splitters.csv")
    cmd = f"python {script_path} --csv {tf_csv} --out {splitters_csv}"
    run(cmd)

    # 2. All other scripts: use tf_dir/tf_splitters.csv as input, outputs in tf_dir
    downstream = [
        ("barplot_top_splitters.py", "--csv {splitters} --out {out}"),
        ("heatmap_zscore_diff.py", "--csv {splitters} --out {out}"),
        ("pairwise_lineplot_splitters.py", "--csv {splitters} --out {out}"),
        ("scatter_split_score_vs_zmax.py", "--csv {splitters} --out {out}"),
        ("scatter_zmax_fast_vs_slow.py", "--csv {splitters} --out {out}")
    ]
    out_names = [
        "barplot_top_splitters.png",
        "heatmap_zscore_diff.png",
        "pairwise_lineplot_splitters.png",
        "scatter_split_score_vs_zmax.png",
        "scatter_zmax_fast_vs_slow.png"
    ]
    for (script, argstr), out_name in zip(downstream, out_names):
        script_path = f"scripts/analysis/tf_response/{script}"
        out_path = os.path.join(tf_dir, out_name)
        cmd = f"python {script_path} --csv {splitters_csv} --out {out_path}"
        run(cmd)

    # Collect all PNGs for summary
    plot_files = sorted([f for f in out_dir.glob("*.png")])    # Save a multi-page PDF with all plots arranged in 2x2 panels per page
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg

    pdf_path = out_dir / "tf_analysis_summary.pdf"
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
    html_path = out_dir / f"tf_analysis_summary.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved HTML summary to {html_path}")

if __name__ == "__main__":
    main()
