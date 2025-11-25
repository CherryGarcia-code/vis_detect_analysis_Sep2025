"""Batch runner for Hit vs Miss decoding using coding direction.

Outputs per session:
  - table_output/decoding/<session>/decoding_timecourse.csv
  - png_output/decoding/<session>/decoding_timecourse.png

Usage:
  python scripts/run_decoding_hit_miss.py --data-dir data --profiles-root table_output/unit_qc \
    --out-root table_output/decoding --png-root png_output/decoding --kept-only
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import shutil
import traceback

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visdetect.core.legacy_io import load_session
from visdetect.analysis.decoding import DecodingConfig, run_time_resolved_decoding


def _session_key(session) -> str:
    subj = getattr(session, "subject", None) or "unknown"
    sname = getattr(session, "session_name", None) or "unknown"
    return f"{subj}_{sname}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--profiles-root", default="table_output/unit_qc")
    p.add_argument("--out-root", default="table_output/decoding")
    p.add_argument("--png-root", default="png_output/decoding")
    p.add_argument("--kept-only", action="store_true")
    p.add_argument("--prefer-profile", default="striatal_strict")
    p.add_argument("--event", default="Change_ON")
    p.add_argument("--window", nargs=2, type=float, default=[-0.5, 0.5])
    p.add_argument("--bin-size", type=float, default=0.02)
    p.add_argument("--method", default="shrinkage", choices=["shrinkage", "ridge"])
    p.add_argument("--reg", type=float, default=1.0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-permutations", type=int, default=200)
    # Responsiveness-based unit filtering
    p.add_argument("--responsive-only", action="store_true")
    p.add_argument("--resp-root", default="table_output/responsiveness")
    p.add_argument("--resp-outcome", default=None)
    p.add_argument("--p-thresh", type=float, default=0.05)
    # Change size controls
    p.add_argument("--size-min", type=float, default=None)
    p.add_argument("--size-max", type=float, default=None)
    p.add_argument("--per-size", action="store_true", help="Run separate decoding per change size present")
    p.add_argument("--min-trials-per-class", type=int, default=8)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    png_root = Path(args.png_root)

    pkls = sorted(data_dir.glob("*.pkl"))
    results = []
    for pkl in pkls:
        try:
            session = load_session(str(pkl))
            key = _session_key(session)
            print(f"Processing {key} ...")

            # locate selection CSV (prefer profile)
            sel_csv = None
            prefer = Path(args.profiles_root) / f"{key}_{args.prefer_profile}" / "unit_selection.csv"
            exact = Path(args.profiles_root) / key / "unit_selection.csv"
            if prefer.exists():
                sel_csv = str(prefer)
            elif exact.exists():
                sel_csv = str(exact)

            cfg = DecodingConfig(
                event_name=args.event,
                window=tuple(args.window),
                bin_size=args.bin_size,
                kept_only=bool(args.kept_only),
                method=args.method,
                reg=args.reg,
                n_splits=args.n_splits,
                n_permutations=args.n_permutations,
                responsive_only=bool(args.responsive_only),
                responsiveness_csv=None,  # filled below if available
                responsiveness_outcome=args.resp_outcome,
                responsiveness_p_thresh=args.p_thresh,
                size_filter=(args.size_min, args.size_max) if (args.size_min is not None and args.size_max is not None) else None,
                min_trials_per_class=args.min_trials_per_class,
            )

            out_dir = out_root / key
            # Resolve responsiveness CSV if requested
            if args.responsive_only:
                resp_csv = Path(args.resp_root) / key / "unit_responsive.csv"
                if resp_csv.exists():
                    cfg.responsiveness_csv = str(resp_csv)

            # Option A: per-size loop
            if args.per_size:
                # collect sizes present in trials
                sizes = []
                for t in getattr(session, "trials", []) or []:
                    s = getattr(t, "change_size", None)
                    try:
                        if s is not None:
                            sizes.append(float(s))
                    except Exception:
                        continue
                if sizes:
                    levels = sorted(set(sizes))
                else:
                    levels = []
                for s in levels:
                    cfg_s = cfg
                    cfg_s.size_filter = (s, s)
                    out_dir_s = out_dir / f"size_{str(s).replace('.', 'p')}"
                    try:
                        paths = run_time_resolved_decoding(
                            session,
                            out_dir=str(out_dir_s),
                            cfg=cfg_s,
                            selection_csv=sel_csv,
                        )
                        # Mirror PNG
                        png_dir_s = png_root / key / f"size_{str(s).replace('.', 'p')}"
                        png_dir_s.mkdir(parents=True, exist_ok=True)
                        if "png" in paths:
                            src = Path(paths["png"])  # table_output/decoding/<session>/size_*/...
                            dst = png_dir_s / src.name
                            shutil.copyfile(src, dst)
                    except Exception as e:
                        # Per-size decoding might fail due to insufficient trials; continue
                        print(f"Skipping size {s}: {e}")
            else:
                # Single run with optional size range
                paths = run_time_resolved_decoding(
                    session,
                    out_dir=str(out_dir),
                    cfg=cfg,
                    selection_csv=sel_csv,
                )
                # Mirror PNG to png_root
                png_dir = png_root / key
                png_dir.mkdir(parents=True, exist_ok=True)
                if "png" in paths:
                    src = Path(paths["png"])  # table_output/decoding/<session>/...
                    dst = png_dir / src.name
                    try:
                        shutil.copyfile(src, dst)
                    except Exception:
                        pass
            results.append((key, True, None))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            print(f"ERROR processing {pkl.name}: {e}\n{tb}")
            results.append((pkl.name, False, str(e)))

    # manifest
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "decoding_manifest.txt"
    with manifest.open("w") as f:
        for key, ok, err in results:
            f.write(f"{key}\t{'OK' if ok else 'FAIL'}\t{err or ''}\n")
    print(f"Done. Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
