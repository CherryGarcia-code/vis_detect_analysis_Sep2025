from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

REPO = Path(__file__).resolve().parents[1]

# Ensure src importable
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.session_io import load_session
from src import su_analysis as su


def safe_read_png(path: Path):
    try:
        return mpimg.imread(path.as_posix())
    except Exception:
        return None


def draw_image(ax, path: Path, title: str | None = None):
    img = safe_read_png(path)
    ax.axis('off')
    if img is None:
        if title:
            ax.set_title(f"{title} (missing)", fontsize=9)
        return
    ax.imshow(img)
    if title:
        ax.set_title(title, fontsize=9)


def build_unit_comparison(session_pkl: Path, cluster_id: int, out_png: Path,
                           include: List[str], compact_scale: float, raster_line_height: float,
                           window=(-0.5, 1.0), bin_size=0.02) -> Path:
    session = load_session(str(session_pkl))
    key = f"{getattr(session, 'subject', 'unknown')}_{getattr(session, 'session_name', 'unknown')}"

    tmp = REPO / "tmp_demo_qc" / "unit_compare"
    tmp.mkdir(parents=True, exist_ok=True)

    panels: List[tuple[str, Path]] = []

    # 1) Baseline aligned, colored by outcome (grouped)
    if "baseline_outcome" in include:
        p = tmp / f"{key}_c{cluster_id}_baseline_outcome.png"
        su.plot_baseline_raster_psth_by_future_outcome(
            session, cluster_id,
            window=window, bin_size=bin_size,
            sort_trials="outcome", peth_scale="per_outcome",
            smooth_sigma=1.0,
            compact_scale=compact_scale, raster_line_height=raster_line_height,
            save_path=str(p))
        panels.append(("Baseline_ON — by outcome", p))

    # 2) Baseline aligned, chronological order (early→late)
    if "baseline_chrono" in include:
        p = tmp / f"{key}_c{cluster_id}_baseline_chrono.png"
        su.plot_baseline_raster_psth_by_future_outcome(
            session, cluster_id,
            window=window, bin_size=bin_size,
            sort_trials="none", peth_scale="per_outcome",
            smooth_sigma=1.0,
            compact_scale=compact_scale, raster_line_height=raster_line_height,
            save_path=str(p))
        panels.append(("Baseline_ON — chrono", p))

    # 3) Change_ON aligned, Hit vs Miss
    if "change_outcome" in include:
        p = tmp / f"{key}_c{cluster_id}_change_outcome.png"
        su.plot_change_rasters_by_outcome(
            session, cluster_id,
            window=window, bin_size=bin_size,
            smooth_sigma=1.0,
            compact_scale=compact_scale, raster_line_height=raster_line_height,
            save_path=str(p))
        panels.append(("Change_ON — Hit vs Miss", p))

    # Optionally: generic raster aligned to First_Lick if event exists
    if "first_lick" in include:
        try:
            p = tmp / f"{key}_c{cluster_id}_firstlick.png"
            su.plot_raster_psth(
                session, cluster_id,
                event_name="First_Lick", window=window, bin_size=bin_size,
                compact_scale=compact_scale, raster_line_height=raster_line_height,
                save_path=str(p))
            panels.append(("First_Lick — raster/PSTH", p))
        except Exception:
            # silently skip if event not present
            pass

    # Compose into a single row figure
    n = len(panels)
    if n == 0:
        raise SystemExit("No panels requested or produced. Nothing to do.")

    fig_w = max(6.0, 4.0 * n)
    fig_h = 4.0
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]
    for ax, (title, path) in zip(axes, panels):
        draw_image(ax, path, title=title)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png.as_posix(), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main(argv=None):
    p = argparse.ArgumentParser(description="Build a compact per-unit comparison figure across multiple alignments/sorts")
    p.add_argument("--session-pkl", default=None, help="Path to session .pkl; if omitted, use --session-key to find in data/")
    p.add_argument("--session-key", default=None, help="Key like BG_031_01052025 to locate data/<key>.pkl")
    p.add_argument("--cluster-id", type=int, required=True)
    p.add_argument("--out", default="png_output/unit_comparisons/comparison.png")
    p.add_argument("--include", nargs="*", default=["baseline_outcome", "baseline_chrono", "change_outcome"],
                   help="Which panels to include: baseline_outcome baseline_chrono change_outcome first_lick")
    p.add_argument("--compact-scale", type=float, default=0.5)
    p.add_argument("--raster-line-height", type=float, default=0.6)
    p.add_argument("--window", nargs=2, type=float, default=[-0.5, 1.0])
    p.add_argument("--bin-size", type=float, default=0.02)
    args = p.parse_args(argv)

    if args.session_pkl:
        pkl = Path(args.session_pkl)
    else:
        if not args.session_key:
            raise SystemExit("Provide --session-pkl or --session-key")
        pkl = REPO / "data" / f"{args.session_key}.pkl"
    out_png = REPO / args.out

    build_unit_comparison(pkl, args.cluster_id, out_png,
                          include=list(args.include), compact_scale=float(args.compact_scale),
                          raster_line_height=float(args.raster_line_height),
                          window=tuple(map(float, args.window)), bin_size=float(args.bin_size))
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    raise SystemExit(main())
