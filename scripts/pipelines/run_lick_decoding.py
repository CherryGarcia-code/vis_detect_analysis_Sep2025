"""Archived: use the copy in `scripts/archived_lick_scripts/`.

This file was archived because `visdetect.utils.matlab_ports.lick` is the
canonical implementation for lick analyses. A backup of the original
implementation is available at:

    scripts/archived_lick_scripts/run_lick_decoding.py

Running this stub will print a message and exit with code 1.
"""
import sys

if __name__ == "__main__":
        print(
                "This script has been archived. Use scripts/archived_lick_scripts/run_lick_decoding.py instead.",
                file=sys.stderr,
        )
        raise SystemExit(1)
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visdetect.core.session import load_session
from visdetect.analysis.lick_decoding import LickCDConfig, run_lick_decoding


def _session_key(session) -> str:
    subj = getattr(session, "subject", None) or "unknown"
    sname = getattr(session, "session_name", None) or "unknown"
    return f"{subj}_{sname}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--profiles-root", default="table_output/unit_qc")
    p.add_argument("--out-root", default="table_output/lick_decoding")
    p.add_argument("--png-root", default="png_output/lick_decoding")
    kept_group = p.add_mutually_exclusive_group()
    kept_group.add_argument("--kept-only", dest="kept_only", action="store_true")
    kept_group.add_argument("--no-kept-only", dest="kept_only", action="store_false")
    p.set_defaults(kept_only=True)
    p.add_argument("--prefer-profile", default="striatal_strict")
    p.add_argument("--event", default="Lick_L")
    p.add_argument("--window", nargs=2, type=float, default=[-0.5, 0.5])
    p.add_argument("--bin-size", type=float, default=0.02)
    p.add_argument("--fa-early-thresh", type=float, default=3.0)
    p.add_argument("--min-trials-per-class", type=int, default=8)
    p.add_argument("--method", default="shrinkage", choices=["shrinkage", "ridge"])
    p.add_argument("--reg", type=float, default=1.0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-permutations", type=int, default=200)
    p.add_argument("--responsive-only", action="store_true")
    p.add_argument("--resp-root", default="table_output/responsiveness_lick")
    p.add_argument("--resp-outcome", default="All")
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

            # locate selection CSV (prefer consistent default path)
            from visdetect.analysis.su_analysis import selection_csv_default_path
            sel_csv = str(selection_csv_default_path(session, root=args.profiles_root))

            cfg = LickCDConfig(
                event_name=args.event,
                window=tuple(args.window),
                bin_size=args.bin_size,
                kept_only=bool(args.kept_only),
                fa_early_threshold=args.fa_early_thresh,
                min_trials_per_class=args.min_trials_per_class,
                method=args.method,
                reg=args.reg,
                n_splits=args.n_splits,
                n_permutations=args.n_permutations,
                responsive_only=bool(args.responsive_only),
                responsiveness_outcome=args.resp_outcome,
            )
            # locate lick responsiveness CSV if gating requested
            resp_csv = None
            if args.responsive_only:
                resp_csv_path = Path(args.resp_root) / key / "unit_lick_responsive.csv"
                if resp_csv_path.exists():
                    resp_csv = str(resp_csv_path)
            run_lick_decoding(session, out_root=str(out_root), png_root=str(png_root), cfg=cfg, selection_csv=sel_csv, responsiveness_csv=resp_csv)
            results.append((key, True, None))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            print(f"ERROR processing {pkl.name}: {e}\n{tb}")
            results.append((pkl.name, False, str(e)))

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "lick_decoding_manifest.txt"
    with manifest.open("w") as f:
        for key, ok, err in results:
            f.write(f"{key}\t{'OK' if ok else 'FAIL'}\t{err or ''}\n")
    print(f"Done. Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
