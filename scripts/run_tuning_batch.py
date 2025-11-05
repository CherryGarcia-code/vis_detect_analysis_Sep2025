"""Batch runner for change-size tuning analysis across sessions.

Produces per-session CSVs:
  - table_output/tuning/<session>/unit_tuning_by_size.csv
  - table_output/tuning/<session>/unit_tuning.csv
And per-unit tuning plots under png_output/tuning/<session>/.

Usage:
  python scripts/run_tuning_batch.py --data-dir data --profiles-root table_output/unit_qc \
    --out-root table_output/tuning --png-root png_output/tuning --kept-only
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.session_io import load_session
from src.su_analysis import load_kept_ids
from src.tuning import TuningConfig, run_tuning_for_session


def _session_key(session) -> str:
    subj = getattr(session, "subject", None) or "unknown"
    sname = getattr(session, "session_name", None) or "unknown"
    return f"{subj}_{sname}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--profiles-root", default="table_output/unit_qc")
    p.add_argument("--out-root", default="table_output/tuning")
    p.add_argument("--png-root", default="png_output/tuning")
    p.add_argument("--kept-only", action="store_true")
    p.add_argument("--prefer-profile", default="striatal_strict")
    p.add_argument("--base", nargs=2, type=float, default=[-0.2, 0.0])
    p.add_argument("--resp", nargs=2, type=float, default=[0.0, 0.2])
    p.add_argument("--bin-size", type=float, default=0.01)
    p.add_argument("--min-trials-per-size", type=int, default=4)
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

            cfg = TuningConfig(
                base_win=tuple(args.base),
                resp_win=tuple(args.resp),
                bin_size=args.bin_size,
                kept_only=bool(args.kept_only),
                min_trials_per_size=args.min_trials_per_size,
            )

            out_dir = out_root / key
            png_dir = png_root / key
            paths = run_tuning_for_session(
                session,
                out_dir=str(out_dir),
                png_dir=str(png_dir),
                cfg=cfg,
                selection_csv=sel_csv,
            )
            results.append((key, True, None))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            print(f"ERROR processing {pkl.name}: {e}\n{tb}")
            results.append((pkl.name, False, str(e)))

    # manifest
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "tuning_manifest.txt"
    with manifest.open("w") as f:
        for key, ok, err in results:
            f.write(f"{key}\t{'OK' if ok else 'FAIL'}\t{err or ''}\n")
    print(f"Done. Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
