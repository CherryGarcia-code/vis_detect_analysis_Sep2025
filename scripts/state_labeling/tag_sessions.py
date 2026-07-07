"""CLI: tag sessions with behavioral states -> per-session CSV cache (+ optional figures).

Subject-aware via the VISDETECT_SUBJECT env var: outputs nest under the active
subject so cross-subject runs never collide, e.g.

    VISDETECT_SUBJECT=BG_031 py scripts/state_labeling/tag_sessions.py --limit 5 --figures

Session source: explicit --sessions if given; else the staging manifest if one
exists for the subject (BG_046); else every pkl on disk (subjects without a
manifest). For subjects with no ground-truth labels the figures are a 2-track
raster+tagger view (no kappa) — a face-validity check, summarised by
_tag_summary.csv (state occupancy + mean outcome composition per tagged state).
"""
import argparse
import gc
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from visdetect.analysis.config import load_staging_manifest, SUBJECT, STAGING_MANIFEST_PATH
from visdetect.analysis.constants import STATE_FEATURE_COLS, STATE_CONFIDENCE_THRESHOLD
from visdetect.suite.loader import load_session, list_pkl_sessions
from visdetect.analysis.state_calibration import CalibrationResult, decode_session_states
from visdetect.analysis.state_labeling import render_tag_figure


def resolve_sessions(args):
    """Session-name list, source chosen by what's available for the subject."""
    if args.sessions:
        sess_args = [str(s) for s in args.sessions]
        if any(s.lower() == "all" for s in sess_args):
            names = list_pkl_sessions()
            print(f"session source: --sessions all => pkls on disk for {SUBJECT} ({len(names)} sessions)")
        else:
            names = sess_args
    elif os.path.exists(STAGING_MANIFEST_PATH):
        manifest = load_staging_manifest(qc_only=True)
        names = [str(r["session_name"]) for _, r in manifest.iterrows()]
        print(f"session source: staging manifest ({len(names)} QC sessions)")
    else:
        names = list_pkl_sessions()
        print(f"session source: pkls on disk for {SUBJECT} ({len(names)} sessions, no manifest)")
    # representative even spread across the chronological list when limiting
    if args.limit and not args.sessions and len(names) > args.limit:
        idx = sorted(set(np.linspace(0, len(names) - 1, args.limit).round().astype(int)))
        names = [names[i] for i in idx]
    elif args.limit:
        names = names[:args.limit]
    return names


def write_summary(per_state_sum, per_state_n, out_path):
    """state occupancy + mean outcome composition per tagged state (face validity)."""
    total = sum(per_state_n.values()) or 1
    rows = []
    for state in sorted(per_state_n):
        n = per_state_n[state]
        mean = per_state_sum[state] / max(n, 1)
        rows.append({"state_label": state, "n_trials": int(n),
                     "occupancy_frac": round(n / total, 4),
                     **{f"mean_{f}": round(float(m), 4) for f, m in zip(STATE_FEATURE_COLS, mean)}})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Tag sessions with behavioral states.")
    ap.add_argument("--model", default="data/state_labels/state_rule.pkl")
    ap.add_argument("--out-dir", default=os.path.join("data", "cache", "state_tags", SUBJECT))
    ap.add_argument("--fig-dir", default=os.path.join("figures", "state_labeler", SUBJECT))
    ap.add_argument("--confidence", type=float, default=STATE_CONFIDENCE_THRESHOLD)
    ap.add_argument("--sessions", nargs="*", default=None, 
                    help="explicit session names; overrides the manifest/pkl source. Use 'all' to include every session on disk")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap to N sessions (evenly spread across the chronological list)")
    ap.add_argument("--figures", action="store_true",
                    help="also write a 2-track raster+tagger figure per session")
    args = ap.parse_args()

    result = CalibrationResult.load(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.figures:
        os.makedirs(args.fig_dir, exist_ok=True)

    sessions = resolve_sessions(args)
    print(f"subject {SUBJECT}: tagging {len(sessions)} session(s) -> {args.out_dir}")

    per_state_sum = defaultdict(lambda: np.zeros(len(STATE_FEATURE_COLS)))
    per_state_n = defaultdict(int)
    tagged_n, skipped = 0, []
    for sn in sessions:
        # Be resilient: one unloadable session must not abort the whole batch.
        try:
            sess = load_session(sn)
            tagged = decode_session_states(result, sess, confidence_threshold=args.confidence)
        except Exception as e:
            skipped.append(sn)
            print(f"SKIP {sn}: {type(e).__name__}: {e}")
            continue
        if tagged.empty:
            skipped.append(sn)
            print(f"SKIP {sn}: no trials")
            del sess; gc.collect()
            continue

        tagged.to_csv(os.path.join(args.out_dir, f"{sn}.csv"), index=False)
        for state, grp in tagged.groupby("state_label"):
            per_state_sum[state] += grp[STATE_FEATURE_COLS].sum().values
            per_state_n[state] += len(grp)
        if args.figures:
            gated = tagged["state_gated"].values
            render_tag_figure(sn, tagged, tagged["state_label"].values, gated,
                              os.path.join(args.fig_dir, f"tag_{sn}.png"))
        tagged_n += 1
        print(f"tagged {sn}: {len(tagged)} trials")
        del sess, tagged
        gc.collect()

    if per_state_n:
        summary = write_summary(per_state_sum, per_state_n,
                                os.path.join(args.out_dir, "_tag_summary.csv"))
        print(f"\nstate occupancy / mean composition ({SUBJECT}):")
        print(summary.to_string(index=False))

    print(f"\nDone: tagged {tagged_n}/{len(sessions)} sessions -> {args.out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
