"""Build the tracking-free population-field instrument for one subject.

Local only (pkls + data/unit_match/input); never computes over X:. Picks the
dominant chanmap signature. Field-tensor caching is deferred to Plan 2 (analysis
layers); this driver produces match-free registration + the audit gate. See
docs/superpowers/plans/2026-07-08-population-field-instrument-plan.md.
"""
import argparse, json, os
import numpy as np
import pandas as pd

from visdetect.analysis import population_field as pf
from visdetect.analysis.config import (
    canonical_session_id, session_date_key, ROOT,
)
from visdetect.analysis.tracking_qc import load_channel_positions
from visdetect.anatomy.channel_geometry import chanmap_signature


def _raw_wf_root(subject):
    return os.path.join(ROOT, "data", "unit_match", "input", subject)


def _session_good_stable_ids(subject, session):
    from visdetect.core.session import load_session  # local import (heavy)
    path = os.path.join(ROOT, "data", "pkls", subject, f"{subject}_{session}.pkl")
    sess = load_session(path)
    ids = list(sess.good_and_stable_ids or [])
    del sess
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--depth-bin-um", type=float, default=pf.DEPTH_BIN_UM)
    args = ap.parse_args()

    root = _raw_wf_root(args.subject)
    sessions = [canonical_session_id(d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]

    # dominant signature
    sig = {}
    for s in sessions:
        pos = load_channel_positions(root, s)
        if pos is not None:
            sig[s] = chanmap_signature(pos)
    chosen, kept = pf.select_dominant_signature(sig)
    kept = sorted(kept, key=session_date_key)

    # common grid from the reference (chronologically first kept) session
    ref = kept[0]
    ref_pos = load_channel_positions(root, ref)
    y_edges = pf.depth_bin_edges(ref_pos, args.depth_bin_um)

    # fingerprints + registration
    fps, n_units = {}, {}
    for s in kept:
        ids = _session_good_stable_ids(args.subject, s)
        n_units[s] = len(ids)
        fps[s] = pf.session_fingerprint_from_root(root, s, ids, y_edges)
    shifts = pf.session_shift_um(fps, ref, args.depth_bin_um, pf.REG_MAX_LAG_UM)

    out_dir = os.path.join(ROOT, "data", "cache", "population_field", args.subject)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{"session": s, "shift_um": shifts[s][0], "corr": shifts[s][1],
                   "n_units": n_units[s]} for s in kept]).to_csv(
        os.path.join(out_dir, "registration.csv"), index=False)

    audit = {"subject": args.subject, "signature": chosen, "n_sessions": len(kept),
             "max_abs_shift_um": float(np.nanmax([abs(shifts[s][0]) for s in kept])),
             "min_fingerprint_corr": float(np.nanmin([shifts[s][1] for s in kept]))}
    with open(os.path.join(out_dir, "audit.json"), "w") as fh:
        json.dump(audit, fh, indent=2)
    print("AUDIT:", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
