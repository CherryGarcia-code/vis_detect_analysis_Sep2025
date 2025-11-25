"""Batch generator for population heatmaps and kept-unit rasters.

- Loads each *.pkl session in data/ (configurable)
- Finds kept units using unit_selection.csv produced by QC runs (default under table_output/unit_qc/<subject_session>/)
- Generates:
  - Population heatmap around Baseline_ON
  - First-N kept rasters (vanilla)
  - First-N kept baseline-aligned rasters colored by future outcome with multi-line PSTH per outcome

Usage:
  python scripts/run_generate_plots_for_kept.py --data-dir data --out-heatmaps png_output/heatmaps --out-rasters png_output/rasters --max-units 8

Optional:
  --outcome-colored to enable the outcome-colored rasters
  --max-units N to change number of units
  --window -0.5 1.0 --bin-size 0.02 to adjust alignment
  --profiles-root table_output/unit_qc  # where unit_selection.csv are stored
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback

# Ensure repo root on sys.path before importing src
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visdetect.core.legacy_io import load_session
from visdetect.analysis import su_analysis as su


def _session_key(session) -> str:
    subj = getattr(session, "subject", None) or "unknown"
    sname = getattr(session, "session_name", None) or "unknown"
    return f"{subj}_{sname}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="Folder containing *.pkl sessions")
    p.add_argument("--profiles-root", default="table_output/unit_qc", help="Root folder containing unit_selection.csv")
    p.add_argument("--out-heatmaps", default="png_output/heatmaps", help="Root for heatmap PNGs")
    p.add_argument("--out-rasters", default="png_output/rasters", help="Root for raster PNGs")
    p.add_argument("--event", default="Baseline_ON", help="Event for heatmaps/rasters")
    p.add_argument("--window", nargs=2, type=float, default=[-0.5, 1.0], help="Window [s] around event")
    p.add_argument("--bin-size", type=float, default=0.02, help="Bin size [s]")
    p.add_argument("--max-units", type=int, default=8, help="Max kept units per session for rasters")
    p.add_argument("--outcome-colored", action="store_true", help="Also generate baseline rasters colored by future outcome")
    p.add_argument("--prefer-profile", default="striatal_strict", help="Prefer this profile's selection CSV when multiple exist")
    p.add_argument("--sort-trials", default="outcome", choices=["outcome", "future_rt", "none"], help="Sorting for outcome-colored rasters")
    p.add_argument("--peth-scale", default="per_outcome", choices=["shared", "per_outcome"], help="Scaling for outcome PSTH panels")
    p.add_argument("--show-sem", action="store_true", help="Add SEM shading to outcome PSTHs (per trial for single-unit; per cluster for session-level)")
    p.add_argument("--session-pop-outcome", action="store_true", help="Also plot session-level population PSTH by outcome (kept units only)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    heat_root = Path(args.out_heatmaps)
    rast_root = Path(args.out_rasters)

    # Ensure src is importable when run from repo root
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    pkls = sorted(data_dir.glob("*.pkl"))
    results = []
    for pkl in pkls:
        try:
            session = load_session(str(pkl))
            key = _session_key(session)
            print(f"Processing {key} ...")

            # Paths
            heat_dir = heat_root / key
            rast_dir = rast_root / key

            # Locate selection CSV (prefer profile-specific folder if available)
            sel_csv = None
            exact = Path(args.profiles_root) / key / "unit_selection.csv"
            prefer = Path(args.profiles_root) / f"{key}_{args.prefer_profile}" / "unit_selection.csv"
            if prefer.exists():
                sel_csv = str(prefer)
            elif exact.exists():
                sel_csv = str(exact)
            else:
                # fallback: any matching folder
                candidates = list((Path(args.profiles_root)).glob(f"{key}*/unit_selection.csv"))
                if candidates:
                    # choose one containing prefer_profile if possible, else the most recent
                    prefereds = [c for c in candidates if args.prefer_profile in str(c.parent.name)]
                    if prefereds:
                        sel_csv = str(prefereds[0])
                    else:
                        sel_csv = str(sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0])

            # Heatmap (kept vs dropped)
            heat_path = heat_dir / "heatmap_kept_dropped.png"
            su.plot_population_heatmap(
                session,
                event_name=args.event,
                window=tuple(args.window),
                bin_size=args.bin_size,
                selection_csv=sel_csv,
                save_path=str(heat_path),
            )

            # Rasters for first N kept (vanilla)
            su.plot_rasters_for_kept(
                session,
                selection_csv=sel_csv,
                event_name=args.event,
                window=tuple(args.window),
                bin_size=args.bin_size,
                out_dir=str(rast_dir),
                max_units=args.max_units,
            )

            # Outcome-colored baseline rasters
            if args.outcome_colored:
                su.plot_baseline_rasters_for_kept_by_outcome(
                    session,
                    selection_csv=sel_csv,
                    window=tuple(args.window),
                    bin_size=args.bin_size,
                    out_dir=str(rast_dir),
                    max_units=args.max_units,
                    sort_trials=args.sort_trials,
                    peth_scale=args.peth_scale,
                )

            # Session-level population PSTH by outcome
            if args.session_pop_outcome:
                pop_path = heat_dir / "population_psth_by_outcome.png"
                su.plot_session_population_psth_by_outcome(
                    session,
                    event_name=args.event,
                    window=tuple(args.window),
                    bin_size=args.bin_size,
                    selection_csv=sel_csv,
                    kept_only=True,
                    smooth_sigma=1.0,
                    show_sem=args.show_sem,
                    separate_panels=True,
                    save_path=str(pop_path),
                )
            results.append((key, True, None))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            print(f"ERROR processing {pkl.name}: {e}\n{tb}")
            results.append((pkl.name, False, str(e)))

    # Write a small manifest
    manifest = heat_root / "generation_manifest.txt"
    heat_root.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as f:
        for key, ok, err in results:
            f.write(f"{key}\t{'OK' if ok else 'FAIL'}\t{err or ''}\n")
    print(f"Done. Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
