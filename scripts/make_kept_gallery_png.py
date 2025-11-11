from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def find_selection_csv(profiles_root: Path, session_key: str, prefer_profile: str = "striatal_strict") -> Path | None:
    prefer = profiles_root / f"{session_key}_{prefer_profile}" / "unit_selection.csv"
    exact = profiles_root / session_key / "unit_selection.csv"
    if prefer.exists():
        return prefer
    if exact.exists():
        return exact
    for p in sorted(profiles_root.glob(f"{session_key}*/unit_selection.csv")):
        return p
    return None


def sessions_with_outputs(heat_root: Path, raster_root: Path) -> list[str]:
    sessions = []
    if heat_root.exists():
        for d in heat_root.iterdir():
            if not d.is_dir():
                continue
            if (d / "heatmap_kept_dropped.png").exists():
                sessions.append(d.name)
    return [s for s in sessions if (raster_root / s).exists()]


def safe_read_png(path: Path):
    try:
        return mpimg.imread(path.as_posix())
    except Exception:
        return None


def draw_image(ax, path: Path, title: str | None = None, max_width_px: int | None = None):
    img = safe_read_png(path)
    if img is None:
        ax.axis('off')
        if title:
            ax.set_title(title + " (missing)", fontsize=9)
        return
    ax.imshow(img)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9)


def build_png(sessions: List[str], out_png: Path, per_session: int, profiles_root: Path, heat_root: Path, raster_root: Path):
    # Layout: each session gets a block: [heatmap | population PSTH] on first row, then per-unit rows beneath
    n_sessions = len(sessions)
    per_session_rows = 1 + math.ceil(per_session / 2)  # 1 row for heatmap+pop; then 2 units per row
    n_rows = n_sessions * per_session_rows
    n_cols = 2  # side-by-side columns

    fig_w = 12
    fig_h = max(3.0, n_rows * 2.6)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = [axes]
    axes = axes.reshape((n_rows, n_cols))

    row = 0
    for sess in sessions:
        # Row 0 of block: heatmap + pop PSTH
        hm = heat_root / sess / "heatmap_kept_dropped.png"
        pop = heat_root / sess / "population_psth_by_outcome.png"
        ax_hm = axes[row, 0]
        ax_pop = axes[row, 1]
        draw_image(ax_hm, hm, title=f"{sess} — heatmap (kept vs dropped)")
        draw_image(ax_pop, pop, title=f"{sess} — population PSTH")

        sel = find_selection_csv(profiles_root, sess)
        kept_ids: List[int] = []
        if sel and sel.exists():
            try:
                df = pd.read_csv(sel)
                if "keep" in df.columns and "cluster_id" in df.columns:
                    kept_ids = [int(x) for x in df.loc[df["keep"].astype(bool), "cluster_id"].tolist()]
            except Exception:
                pass
        kept_ids = kept_ids[:per_session]

        # Subsequent rows in this block
        unit_pairs = [kept_ids[i:i+2] for i in range(0, len(kept_ids), 2)]
        for pr in range(math.ceil(per_session / 2)):
            row_idx = row + 1 + pr
            ax_l = axes[row_idx, 0]
            ax_r = axes[row_idx, 1]
            if pr < len(unit_pairs):
                ids = unit_pairs[pr]
            else:
                ids = []
            # Left unit
            if len(ids) >= 1:
                cid = ids[0]
                rp = raster_root / sess / f"cluster_{cid}_raster_psth.png"
                draw_image(ax_l, rp, title=f"{sess} — unit {cid} raster/PSTH")
            else:
                ax_l.axis('off')
            # Right unit
            if len(ids) >= 2:
                cid = ids[1]
                rp = raster_root / sess / f"cluster_{cid}_raster_psth.png"
                draw_image(ax_r, rp, title=f"{sess} — unit {cid} raster/PSTH")
            else:
                ax_r.axis('off')
        # Advance to next block
        row += per_session_rows

    fig.tight_layout(h_pad=1.2, w_pad=0.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png.as_posix(), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main(argv=None):
    p = argparse.ArgumentParser(description="Build a single PNG collage of kept-unit examples across sessions")
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--per-session", type=int, default=4)
    p.add_argument("--profiles-root", default="table_output/unit_qc")
    p.add_argument("--heat-root", default="png_output/heatmaps")
    p.add_argument("--rasters-root", default="png_output/rasters")
    p.add_argument("--out", default="png_output/kept_gallery.png")
    args = p.parse_args(argv)

    profiles_root = REPO / args.profiles_root
    heat_root = REPO / args.heat_root
    rasters_root = REPO / args.rasters_root
    out_png = REPO / args.out

    if args.sessions:
        sessions = list(args.sessions)
    else:
        sessions = sessions_with_outputs(heat_root, rasters_root)
        sessions = sessions[:6]

    if not sessions:
        print("No sessions with outputs found. Run run_generate_plots_for_kept.py first.")
        return 1

    pth = build_png(sessions, out_png, per_session=args.per_session,
                    profiles_root=profiles_root, heat_root=heat_root, raster_root=rasters_root)
    print(f"Wrote gallery PNG: {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
