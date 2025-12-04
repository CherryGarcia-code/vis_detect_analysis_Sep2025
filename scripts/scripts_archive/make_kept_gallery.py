from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import sys

REPO = Path(__file__).resolve().parents[1]


def find_selection_csv(profiles_root: Path, session_key: str, prefer_profile: str = "striatal_strict") -> Path | None:
    prefer = profiles_root / f"{session_key}_{prefer_profile}" / "unit_selection.csv"
    exact = profiles_root / session_key / "unit_selection.csv"
    if prefer.exists():
        return prefer
    if exact.exists():
        return exact
    # fallback: any matching folder
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
    # Ensure rasters exist too
    sessions_ok = []
    for s in sessions:
        if (raster_root / s).exists():
            sessions_ok.append(s)
    return sessions_ok


def build_gallery(sessions: list[str], out_md: Path, per_session: int, profiles_root: Path, heat_root: Path, raster_root: Path):
    lines: list[str] = []
    lines.append(f"# Kept-unit visual gallery\n")
    lines.append(f"This gallery shows per-session population heatmaps/PSTHs and a few example kept-unit rasters/PSTHs.\n\n")

    for sess in sessions:
        lines.append(f"## {sess}\n")
        # Heatmap + population PSTH
        hm = heat_root / sess / "heatmap_kept_dropped.png"
        pop = heat_root / sess / "population_psth_by_outcome.png"
        if hm.exists() or pop.exists():
            lines.append("<div style=\"display:flex; gap:12px; flex-wrap:wrap;\">")
            if hm.exists():
                lines.append(f"<div><div>Heatmap</div><img src=\"{hm.as_posix()}\" width=\"420\"></div>")
            if pop.exists():
                lines.append(f"<div><div>Population PSTH</div><img src=\"{pop.as_posix()}\" width=\"420\"></div>")
            lines.append("</div>\n")

        # Example kept-unit rasters
        sel = find_selection_csv(profiles_root, sess)
        if not sel or not sel.exists():
            lines.append("_No unit_selection.csv found._\n\n")
            continue
        try:
            df = pd.read_csv(sel)
        except Exception:
            lines.append("_Failed to read unit_selection.csv._\n\n")
            continue
        if "keep" not in df.columns or "cluster_id" not in df.columns:
            lines.append("_unit_selection.csv missing required columns._\n\n")
            continue
        kept_ids = [int(x) for x in df.loc[df["keep"].astype(bool), "cluster_id"].tolist()]
        if not kept_ids:
            lines.append("_No kept units._\n\n")
            continue
        kept_ids = kept_ids[:per_session]
        lines.append("<div style=\"display:flex; gap:18px; flex-wrap:wrap;\">")
        for cid in kept_ids:
            rp = raster_root / sess / f"cluster_{cid}_raster_psth.png"
            bp = raster_root / sess / f"cluster_{cid}_baseline_by_outcome.png"
            if not rp.exists() and not bp.exists():
                continue
            lines.append("<div style=\"min-width:420px;\">")
            lines.append(f"<div><strong>Cluster {cid}</strong></div>")
            if rp.exists():
                lines.append(f"<div><img src=\"{rp.as_posix()}\" width=\"420\"></div>")
            if bp.exists():
                lines.append(f"<div><img src=\"{bp.as_posix()}\" width=\"420\"></div>")
            lines.append("</div>")
        lines.append("</div>\n\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def main(argv=None):
    p = argparse.ArgumentParser(description="Build a Markdown gallery of kept-unit examples across sessions")
    p.add_argument("--sessions", nargs="*", default=None, help="Optional explicit session keys (e.g., BG_031_260325)")
    p.add_argument("--per-session", type=int, default=4, help="Number of kept units to show per session")
    p.add_argument("--profiles-root", default="table_output/unit_qc")
    p.add_argument("--heat-root", default="png_output/heatmaps")
    p.add_argument("--rasters-root", default="png_output/rasters")
    p.add_argument("--out", default="png_output/kept_gallery.md")
    args = p.parse_args(argv)

    profiles_root = REPO / args.profiles_root
    heat_root = REPO / args.heat_root
    rasters_root = REPO / args.rasters_root
    out_md = REPO / args.out

    if args.sessions:
        sessions = list(args.sessions)
    else:
        sessions = sessions_with_outputs(heat_root, rasters_root)
        # sample up to 6 sessions if many
        sessions = sessions[:6]

    if not sessions:
        print("No sessions with outputs found. Run run_generate_plots_for_kept.py first.")
        return 1

    md = build_gallery(sessions, out_md, per_session=args.per_session,
                       profiles_root=profiles_root, heat_root=heat_root, raster_root=rasters_root)
    print(f"Wrote gallery: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
