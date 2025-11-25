"""Batch runner for responsiveness analysis across sessions.

Generates per-session unit_responsive.csv and quick diagnostic plots.

Usage:
  python scripts/run_responsiveness_batch.py --data-dir data \
      --out-root table_output/responsiveness --png-root png_output/responsiveness \
      --event Change_ON --base -0.2 0.0 --resp 0.0 0.2 --bin-size 0.01 \
      --per-outcome --kept-only
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

from visdetect.core.legacy_io import load_session
from visdetect.analysis.su_analysis import load_kept_ids
from src.responsiveness import RespConfig, run_responsiveness


def _session_key(session) -> str:
    subj = getattr(session, "subject", None) or "unknown"
    sname = getattr(session, "session_name", None) or "unknown"
    return f"{subj}_{sname}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--profiles-root", default="table_output/unit_qc")
    p.add_argument("--out-root", default="table_output/responsiveness")
    p.add_argument("--png-root", default="png_output/responsiveness")
    p.add_argument("--event", default="Change_ON")
    p.add_argument("--base", nargs=2, type=float, default=[-0.2, 0.0])
    p.add_argument("--resp", nargs=2, type=float, default=[0.0, 0.2])
    p.add_argument("--bin-size", type=float, default=0.01)
    p.add_argument("--per-outcome", action="store_true")
    p.add_argument("--kept-only", action="store_true")
    p.add_argument("--prefer-profile", default="striatal_strict")
    p.add_argument("--min-trials", type=int, default=5)
    p.add_argument("--n-perm", type=int, default=500)
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

            cfg = RespConfig(
                event_name=args.event,
                base_win=tuple(args.base),
                resp_win=tuple(args.resp),
                bin_size=args.bin_size,
                per_outcome=bool(args.per_outcome),
                kept_only=bool(args.kept_only),
                min_trials=args.min_trials,
                n_perm=args.n_perm,
            )

            out_dir = out_root / key
            png_dir = png_root / key
            paths = run_responsiveness(
                session,
                out_dir=str(out_dir),
                cfg=cfg,
                selection_csv=sel_csv,
                make_plots=True,
            )
            # Move PNGs (if any) to png_root
            for k in ("delta_hist", "volcano"):
                if k in paths:
                    # Already saved under table_output/responsiveness/<session>
                    # Also copy to png_output/responsiveness/<session>
                    src = Path(paths[k])
                    png_dir.mkdir(parents=True, exist_ok=True)
                    dst = png_dir / src.name
                    try:
                        dst.write_bytes(src.read_bytes())
                    except Exception:
                        pass
            results.append((key, True, None))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            print(f"ERROR processing {pkl.name}: {e}\n{tb}")
            results.append((pkl.name, False, str(e)))

    # manifest
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "responsiveness_manifest.txt"
    with manifest.open("w") as f:
        for key, ok, err in results:
            f.write(f"{key}\t{'OK' if ok else 'FAIL'}\t{err or ''}\n")
    print(f"Done. Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
