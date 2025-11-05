"""Batch runner for lick-aligned responsiveness analysis.

Usage:
  python scripts/run_lick_responsiveness.py --data-dir data \
    --profiles-root table_output/unit_qc --out-root table_output/responsiveness_lick \
    --kept-only --base-win -0.2 0.0 --post-end 0.2 --buffer 0.03 --min-post 0.05
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.session_io import load_session
from src.lick_responsiveness import LickRespConfig, run_lick_responsiveness


def _session_key(session) -> str:
    subj = getattr(session, "subject", None) or "unknown"
    sname = getattr(session, "session_name", None) or "unknown"
    return f"{subj}_{sname}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--profiles-root", default="table_output/unit_qc")
    p.add_argument("--out-root", default="table_output/responsiveness_lick")
    p.add_argument("--kept-only", action="store_true")
    p.add_argument("--prefer-profile", default="striatal_strict")
    p.add_argument("--event", default="Lick_L")
    p.add_argument("--base-win", nargs=2, type=float, default=[-0.2, 0.0])
    p.add_argument("--post-end", type=float, default=0.2)
    p.add_argument("--buffer", type=float, default=0.03)
    p.add_argument("--min-post", type=float, default=0.05)
    p.add_argument("--min-trials", type=int, default=5)
    p.add_argument("--n-perm", type=int, default=500)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)

    pkls = sorted(data_dir.glob("*.pkl"))
    results = []
    for pkl in pkls:
        try:
            session = load_session(str(pkl))
            key = _session_key(session)
            print(f"Processing {key} ...")
            # selection CSV preference
            sel_csv = None
            prefer = Path(args.profiles_root) / f"{key}_{args.prefer_profile}" / "unit_selection.csv"
            exact = Path(args.profiles_root) / key / "unit_selection.csv"
            if prefer.exists():
                sel_csv = str(prefer)
            elif exact.exists():
                sel_csv = str(exact)

            cfg = LickRespConfig(
                event_name=args.event,
                base_win=(args.base_win[0], args.base_win[1]),
                post_end=args.post_end,
                min_post=args.min_post,
                buffer=args.buffer,
                kept_only=bool(args.kept_only),
                min_trials=args.min_trials,
                n_perm=args.n_perm,
            )
            out_dir = out_root / key
            run_lick_responsiveness(session, out_dir=str(out_dir), cfg=cfg, selection_csv=sel_csv, make_plots=True)
            results.append((key, True, None))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            print(f"ERROR processing {pkl.name}: {e}\n{tb}")
            results.append((pkl.name, False, str(e)))

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "lick_responsiveness_manifest.txt"
    with manifest.open("w") as f:
        for key, ok, err in results:
            f.write(f"{key}\t{'OK' if ok else 'FAIL'}\t{err or ''}\n")
    print(f"Done. Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
