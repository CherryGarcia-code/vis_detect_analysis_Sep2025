"""Build FSI/SPN waveform cell-type labels from current RawWaveforms.

One global GMM(2) over trough-to-peak across all QC-session units.

Usage:
    py scripts/analysis/build_waveform_celltype_labels.py
"""
import glob
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from visdetect.analysis.config import RAW_WF_DIR, WAVEFORM_LABELS_PATH  # noqa: E402
from visdetect.suite.loader import load_staging_manifest                # noqa: E402
from visdetect.analysis.tracking_qc import (                            # noqa: E402
    load_raw_mean_waveform, extract_peak_channel,
)
from visdetect.analysis.waveform_celltype import (                      # noqa: E402
    compute_waveform_features, classify_celltype,
)

STATS_PATH = os.path.join(REPO_ROOT, "FIGURES", "qc", "waveform_celltype_stats.csv")


def session_unit_ids(session_str: str):
    """Kilosort ids with a RawWaveforms file for this session."""
    rw_dir = os.path.join(RAW_WF_DIR, session_str, "RawWaveforms")
    ids = []
    for f in glob.glob(os.path.join(rw_dir, "Unit*_RawSpikes.npy")):
        name = os.path.basename(f)
        try:
            ids.append(int(name.replace("Unit", "").replace("_RawSpikes.npy", "")))
        except ValueError:
            continue
    return sorted(ids)


def main():
    manifest = load_staging_manifest(qc_only=True)
    rows = []
    for sess_int in sorted(manifest["session_name"].astype(int)):
        sess_str = str(sess_int).zfill(8)
        ids = session_unit_ids(sess_str)
        if not ids:
            print(f"  {sess_str}: no RawWaveforms"); continue
        n = 0
        for kid in ids:
            mean_wf = load_raw_mean_waveform(RAW_WF_DIR, sess_str, kid)
            if mean_wf is None:
                continue
            peak_chan = extract_peak_channel(mean_wf)
            feats = compute_waveform_features(mean_wf[:, peak_chan])
            rows.append({"session_date": sess_int, "cluster_id": int(kid),
                         "t2p_ms": feats["t2p_ms"], "half_width_ms": feats["half_width_ms"]})
            n += 1
        print(f"  {sess_str}: {n} units")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No waveforms extracted — check RAW_WF_DIR.")

    labels, info = classify_celltype(df["t2p_ms"].values)
    df["celltype"] = labels
    print(f"GMM: threshold={info['threshold_ms']:.3f} ms, delta_BIC={info['delta_bic']:.1f}, "
          f"n={info['n']}; counts={df['celltype'].value_counts().to_dict()}")

    os.makedirs(os.path.dirname(WAVEFORM_LABELS_PATH), exist_ok=True)
    df[["session_date", "cluster_id", "celltype"]].to_csv(WAVEFORM_LABELS_PATH, index=False)
    print(f"Wrote labels: {WAVEFORM_LABELS_PATH}  ({len(df)} units)")

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    pd.DataFrame([info]).to_csv(STATS_PATH, index=False)
    print(f"Wrote stats: {STATS_PATH}")


if __name__ == "__main__":
    main()
