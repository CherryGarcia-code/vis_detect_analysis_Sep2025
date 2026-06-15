#!/usr/bin/env python3
"""Write per-session trial->state tables consumed by track curation.

Two modes:
  --provider uniform   bootstrap: every valid trial labeled in_zone (default)
  --provider hmm       use a fitted GLM-HMM (requires --model-path)

Usage:
    py scripts/pipelines/tracking/make_state_tables.py --provider uniform
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.analysis import state_provider as sp                # noqa: E402
from visdetect.core.session import load_session                   # noqa: E402
from visdetect.suite.loader import load_filtered_manifest          # noqa: E402

DEFAULT_PKL_DIR = REPO_ROOT / "data" / "pkls" / "BG_046"
DEFAULT_STATES_DIR = REPO_ROOT / "data" / "cache" / "states" / "BG_046"
DEFAULT_TAGS_DIR = REPO_ROOT / "data" / "cache" / "state_tags" / "BG_046"


def _session_pkl(pkl_dir: Path, sess: str):
    for s in (sess, str(sess).zfill(8)):
        p = pkl_dir / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def _tag_csv(tags_dir: Path, sess: str):
    for s in (sess, str(sess).zfill(8)):
        p = tags_dir / f"{s}.csv"
        if p.exists():
            return p
    return None


def _run_tags(args, manifest) -> int:
    """Convert behavioral-state-labeler tag CSVs -> canonical state tables."""
    import pandas as pd
    n = 0
    for _, mrow in manifest.iterrows():
        sess = str(mrow["session_name"])
        tag = _tag_csv(args.tags_dir, sess)
        if tag is None:
            print(f"  skip {sess}: no tag csv", flush=True); continue
        df = pd.read_csv(tag)
        rows = sp.rows_from_tag_df(df, label_col=args.label_col,
                                   use_gating=not args.no_gating)
        out = sp.write_state_table(sess, rows, args.states_dir)
        n_iz = sum(1 for _, lab, _ in rows if lab == sp.IN_ZONE)
        print(f"  wrote {out.name}: {len(rows)} labeled, {n_iz} in_zone", flush=True)
        n += 1
    print(f"Done: {n} state tables -> {args.states_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["uniform", "hmm", "tags"], default="uniform")
    ap.add_argument("--model-path", type=Path, default=None,
                    help="Fitted GLM-HMM pickle (required for --provider hmm)")
    ap.add_argument("--tags-dir", type=Path, default=DEFAULT_TAGS_DIR,
                    help="Behavioral-state-labeler tag CSVs (for --provider tags)")
    ap.add_argument("--label-col", default="state_label",
                    help="Tag column for the state label (tags mode); "
                         "e.g. state_label (decision tree) or hmm_state_label")
    ap.add_argument("--no-gating", action="store_true",
                    help="tags mode: keep ungated trials (state_gated == -1) too")
    ap.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR)
    ap.add_argument("--states-dir", type=Path, default=DEFAULT_STATES_DIR)
    args = ap.parse_args()

    manifest = load_filtered_manifest(
        include_stages=["Naive", "Learning", "Expert"],
        merge_naive_learning=True, min_trials=150, min_dprime=None)

    if args.provider == "tags":
        return _run_tags(args, manifest)

    if args.provider == "uniform":
        provider = sp.UniformInZoneStateProvider()
    else:
        if args.model_path is None:
            ap.error("--provider hmm requires --model-path")
        import pickle
        from visdetect.analysis.hmm import auto_label_states_explicit
        with open(args.model_path, "rb") as f:
            model = pickle.load(f)
        provider = sp.HMMStateProvider(model, auto_label_states_explicit(model))

    n = 0
    for _, mrow in manifest.iterrows():
        sess = str(mrow["session_name"])
        pkl = _session_pkl(args.pkl_dir, sess)
        if pkl is None:
            print(f"  skip {sess}: no pkl", flush=True); continue
        S = load_session(str(pkl))
        out = provider.write(S, sess, args.states_dir)
        print(f"  wrote {out.name}", flush=True)
        n += 1
        del S
    print(f"Done: {n} state tables -> {args.states_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
