"""Build the tracking-free population-field instrument for one subject.

Local only (pkls + data/unit_match/input); never computes over X:. Picks the
dominant chanmap signature. Field-tensor caching is deferred to Plan 2 (analysis
layers build tensors on demand via build_field_tensor + registration.csv); this
driver produces match-free registration + the audit gate. The audit's UM-offset
cross-check (audit_shift_vs_um_offset) is also deferred to Plan 2 (it needs the
UnitMatch drift offsets loaded). See
docs/superpowers/plans/2026-07-08-population-field-instrument-plan.md.
"""
import argparse, gc, json, os
import numpy as np
import pandas as pd

from visdetect.analysis import population_field as pf
from visdetect.analysis.config import canonical_session_id, session_date_key, ROOT
from visdetect.analysis.tracking_qc import load_channel_positions, load_raw_mean_waveform
from visdetect.anatomy.channel_geometry import chanmap_signature


def _raw_wf_root(subject):
    return os.path.join(ROOT, "data", "unit_match", "input", subject)


def _session_good_stable_ids(subject, session):
    """good_and_stable ids for a session, or None if its pkl is missing/unreadable."""
    from visdetect.core.session import load_session  # local import (heavy)
    path = os.path.join(ROOT, "data", "pkls", subject, f"{subject}_{session}.pkl")
    if not os.path.exists(path):
        return None
    try:
        sess = load_session(path)
    except Exception as exc:  # unreadable pkl -> skip, don't abort the run
        print(f"WARN: could not load pkl for {session}: {exc}", flush=True)
        return None
    ids = list(sess.good_and_stable_ids or [])
    del sess
    gc.collect()
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--depth-bin-um", type=float, default=pf.DEPTH_BIN_UM)
    args = ap.parse_args()

    root = _raw_wf_root(args.subject)
    sessions = [canonical_session_id(d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]

    # dominant chanmap signature (geometry is per UM-input dir, independent of pkls)
    sig = {}
    for s in sessions:
        pos = load_channel_positions(root, s)
        if pos is not None:
            sig[s] = chanmap_signature(pos)
    if not sig:
        raise SystemExit(f"no channel_positions found under {root}")
    chosen, kept = pf.select_dominant_signature(sig)
    kept = sorted(kept, key=session_date_key)

    # restrict to sessions whose pkl exists (fingerprint needs good_and_stable ids)
    good_ids = {}
    for s in kept:
        ids = _session_good_stable_ids(args.subject, s)
        if ids is None:
            print(f"WARN: no pkl for session {s}; skipping", flush=True)
            continue
        good_ids[s] = ids
    kept = [s for s in kept if s in good_ids]
    if not kept:
        raise SystemExit(f"no sessions with pkls under data/pkls/{args.subject}")

    # common depth grid from the chronologically-first kept (pkl-present) session
    ref = kept[0]
    ref_pos = load_channel_positions(root, ref)
    y_edges = pf.depth_bin_edges(ref_pos, args.depth_bin_um)

    # per-session fingerprints + match-free registration
    fps, n_units = {}, {}
    for s in kept:
        n_units[s] = len(good_ids[s])
        fps[s] = pf.session_fingerprint_from_root(root, s, good_ids[s], y_edges)
    shifts = pf.session_shift_um(fps, ref, args.depth_bin_um, pf.REG_MAX_LAG_UM)

    # audit check (d): per-unit peak-channel vs amplitude-centroid depth agreement
    # (re-loads waveforms; local + one-time, keeps the tested library fn unchanged)
    pc_diffs = []
    for s in kept:
        pos = load_channel_positions(root, s)
        for uid in good_ids[s]:
            mw = load_raw_mean_waveform(root, s, int(uid))
            if mw is None:
                continue
            peak_d, cent_d = pf.peak_vs_centroid_depth(mw, pos)
            if np.isfinite(peak_d) and np.isfinite(cent_d):
                pc_diffs.append(abs(peak_d - cent_d))

    out_dir = os.path.join(ROOT, "data", "cache", "population_field", args.subject)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{"session": s, "shift_um": shifts[s][0], "corr": shifts[s][1],
                   "n_units": n_units[s]} for s in kept]).to_csv(
        os.path.join(out_dir, "registration.csv"), index=False)

    audit = {
        "subject": args.subject, "signature": chosen, "n_sessions": len(kept),
        "max_abs_shift_um": float(np.nanmax([abs(shifts[s][0]) for s in kept])),
        "min_fingerprint_corr": float(np.nanmin([shifts[s][1] for s in kept])),
        "peak_vs_centroid_median_um": float(np.median(pc_diffs)) if pc_diffs else float("nan"),
        "peak_vs_centroid_max_um": float(np.max(pc_diffs)) if pc_diffs else float("nan"),
    }
    with open(os.path.join(out_dir, "audit.json"), "w") as fh:
        json.dump(audit, fh, indent=2)
    print("AUDIT:", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
