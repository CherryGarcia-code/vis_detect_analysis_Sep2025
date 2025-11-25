"""Generate a z-threshold sweep of TF pulse grids for a single session.

Creates individual grids for each z in --zs and a combined side-by-side PNG.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visdetect.core.legacy_io import load_session
from visdetect.analysis.tf_pulse import (
    TFRespPulseConfig,
    plot_tf_pulse_grid,
    plot_tf_pulse_high_z_summary,
)


def _progress(iterable, *, desc: str = ""):
    total = len(iterable) if hasattr(iterable, "__len__") else None
    if total is None or total == 0:
        for item in iterable:
            yield item
        return
    width = 30
    print(f"{desc} (0/{total})")
    for idx, item in enumerate(iterable, 1):
        pct = idx / total
        filled = int(width * pct)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r{desc} [{bar}] {idx}/{total}", end="", flush=True)
        yield item
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot TF pulse z-threshold sweep grids for one session")
    ap.add_argument("--file", required=True, help="Path to session pickle file")
    ap.add_argument("--zs", nargs="+", type=float, default=[2.5, 3.0, 3.5], help="Z thresholds to sweep")
    ap.add_argument("--out", default="png_output/tf_pulse_grids", help="Output root folder")
    ap.add_argument("--cols", type=int, default=12, help="Number of columns in each grid")
    ap.add_argument("--kept-only", action="store_true", help="Use only kept/good units")
    ap.add_argument("--high-z", type=float, default=3.0, help="|z| cutoff for the survivor grid + summary")
    args = ap.parse_args(argv)

    sess = load_session(args.file)
    ident = f"{getattr(sess,'subject','unknown')}_{getattr(sess,'session_name','unknown')}"
    out_dir = Path(args.out) / ident
    out_dir.mkdir(parents=True, exist_ok=True)

    # Produce individual grids
    png_paths = []
    for z in _progress(list(args.zs), desc="Generating z grids"):
        cfg = TFRespPulseConfig(kept_only=args.kept_only)
        png = out_dir / f"tf_pulse_grid_both_z{z:g}.png"
        plot_tf_pulse_grid(sess, str(png), cfg=cfg, n_cols=args.cols, which="both", z_line=z)
        png_paths.append(png)

    # Compose a combined side-by-side image
    n = len(png_paths)
    fig, axes = plt.subplots(1, n, figsize=(min(3.5*n, 18), 5), squeeze=False)
    axes = axes[0]
    for ax, p in zip(axes, png_paths):
        img = plt.imread(str(p))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(p.stem.replace('_', ' '), fontsize=10)
    fig.tight_layout()
    combo = out_dir / "tf_pulse_grid_zsweep.png"
    fig.savefig(combo, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote sweep: {combo}")

    # Survivor grid with aligned modulation summary
    survivor_png = out_dir / f"tf_pulse_highz_summary_z{args.high_z:g}.png"
    cfg = TFRespPulseConfig(kept_only=args.kept_only)
    result_path = plot_tf_pulse_high_z_summary(
        sess,
        str(survivor_png),
        cfg=cfg,
        n_cols=args.cols,
        which="both",
        min_abs_z=float(args.high_z),
    )
    if result_path:
        print(f"Wrote high-z summary: {result_path}")
    else:
        print(f"No clusters exceeded |z| ≥ {args.high_z:g}; skipping survivor plot")


if __name__ == "__main__":
    raise SystemExit(main())
