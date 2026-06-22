#!/usr/bin/env python3
"""Write per-session trial->state tables consumed by track curation.

Three modes:
  --provider uniform   bootstrap: every valid trial labeled in_zone (default)
  --provider hmm       use a fitted GLM-HMM (requires --model-path)
  --provider tags      convert behavioral-state-labeler tag CSVs (BG_046)

Multi-subject: pass --subject (default BG_046). For subjects with no staging
manifest (BG_031/038/039/049) the session list is taken from the pkl directory.

Usage:
    py scripts/pipelines/tracking/make_state_tables.py --subject BG_049 --provider uniform
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # for _subject_paths


def _early_subject(default: str = "BG_046") -> str:
    """Read --subject from argv BEFORE importing visdetect, so config-derived
    paths (the staging manifest) resolve for the right subject."""
    for i, a in enumerate(sys.argv):
        if a == "--subject" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--subject="):
            return a.split("=", 1)[1]
    return default


os.environ["VISDETECT_SUBJECT"] = _early_subject()

import pandas as pd                                            # noqa: E402
import _subject_paths as sjp                                   # noqa: E402
from visdetect.analysis import state_provider as sp            # noqa: E402
from visdetect.core.session import load_session                # noqa: E402


def _tag_csv(tags_dir, sess):
    for s in (str(sess), str(sess).zfill(8)):
        p = Path(tags_dir) / f"{s}.csv"
        if p.exists():
            return p
    return None


def _session_tokens(registry_path, subject: str, pkl_dir):
    """Canonical curation session tokens. Prefer the UM registry's `session` column —
    the EXACT tokens curate_tracks/render key on (prefixed BG_049_01092025 for new
    subjects, bare 1072025 for BG_046) — so the state tables we write are found by the
    curation's in_zone lookup. Falls back to pkl stems if the registry is unreadable."""
    try:
        reg = pd.read_csv(registry_path)
        toks = sorted(reg["session"].astype(str).unique().tolist(),
                      key=sjp.session_date_key)
        if toks:
            return toks, "registry"
    except Exception as e:
        print(f"  registry unreadable ({e}); using pkl dir", flush=True)
    toks = [Path(p).name[:-4]
            for p in glob.glob(str(Path(pkl_dir) / f"{subject}_*.pkl"))]
    return sorted(toks, key=sjp.session_date_key), "pkl-dir"


def _run_tags(args, tokens) -> int:
    """Convert behavioral-state-labeler tag CSVs -> canonical state tables."""
    import pandas as pd
    n = 0
    for sess in tokens:
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
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--provider", choices=["uniform", "hmm", "tags"], default="uniform")
    ap.add_argument("--model-path", type=Path, default=None,
                    help="Fitted GLM-HMM pickle (required for --provider hmm)")
    ap.add_argument("--tags-dir", type=Path, default=None,
                    help="Behavioral-state-labeler tag CSVs (for --provider tags)")
    ap.add_argument("--label-col", default="state_label",
                    help="Tag column for the state label (tags mode); "
                         "e.g. state_label (decision tree) or hmm_state_label")
    ap.add_argument("--no-gating", action="store_true",
                    help="tags mode: keep ungated trials (state_gated == -1) too")
    ap.add_argument("--pkl-dir", type=Path, default=None)
    ap.add_argument("--states-dir", type=Path, default=None)
    ap.add_argument("--registry", type=Path, default=None,
                    help="UM registry whose 'session' column gives the canonical "
                         "session tokens curate_tracks keys on (default: the subject's "
                         "all_sessions/all42 unit_index.csv).")
    args = ap.parse_args()
    if args.pkl_dir is None:
        args.pkl_dir = sjp.pkl_dir(args.subject)
    if args.states_dir is None:
        args.states_dir = sjp.states_dir(args.subject)
    if args.tags_dir is None:
        args.tags_dir = sjp.tags_dir(args.subject)
    if args.registry is None:
        args.registry = sjp.um_registry(args.subject)

    tokens, src = _session_tokens(args.registry, args.subject, args.pkl_dir)
    print(f"{args.subject}: {len(tokens)} sessions ({src} source)", flush=True)

    if args.provider == "tags":
        return _run_tags(args, tokens)

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
    for sess in tokens:
        pkl = sjp.session_pkl(args.subject, sess, args.pkl_dir)
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
