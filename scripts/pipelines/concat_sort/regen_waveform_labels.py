"""Regenerate waveform cell-type labels from mean spike waveforms.

Reads the RawWaveforms .npy files produced by prep_concat_waveforms.py,
computes trough-to-peak (T2P) duration on the peak channel,
classifies Narrow (FSI) vs Broad (MSN/Proj) via 2-component GMM,
and writes a CSV with columns: session_date, cluster_id, celltype.

The cluster_id column uses the concat-sort global UID convention
(the filenames are Unit{global_uid}_RawSpikes.npy).

Output: AI_exploration/figures/waveform_celltype_labels.csv  (overwrites)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[3]

WF_ROOT = ROOT / "data" / "unit_match_concat_sort" / "input" / "BG_046"
OUTPUT_CSV = ROOT / "AI_exploration" / "figures" / "waveform_celltype_labels.csv"
SR = 30_000  # sample rate (Hz)


def compute_t2p(waveform_1d):
    """Trough-to-peak duration (ms) for a single 1-D waveform."""
    w = waveform_1d / (np.abs(waveform_1d).max() + 1e-12)
    trough_idx = int(np.argmin(w))
    after = w[trough_idx:]
    if len(after) < 2:
        return np.nan
    peak_idx = trough_idx + int(np.argmax(after))
    return (peak_idx - trough_idx) / SR * 1000


def extract_session_waveform_features(session_dir: Path):
    """Load all Unit*_RawSpikes.npy in a session dir and return T2P per cluster."""
    rw_dir = session_dir / "RawWaveforms"
    if not rw_dir.is_dir():
        return []

    rows = []
    for fpath in sorted(rw_dir.glob("Unit*_RawSpikes.npy")):
        uid_str = fpath.stem.split("_")[0].replace("Unit", "")
        try:
            global_uid = int(uid_str)
        except ValueError:
            continue

        raw = np.load(fpath)  # (n_timepoints, n_channels, 2)
        if raw.size == 0:
            continue

        # Average the two CV halves
        mean_wf = raw.mean(axis=2)  # (n_timepoints, n_channels)
        ptp_per_ch = np.ptp(mean_wf, axis=0)
        best_ch = int(np.argmax(ptp_per_ch))
        w = mean_wf[:, best_ch].astype(float)

        if np.ptp(w) < 1e-12:
            continue

        t2p = compute_t2p(w)
        if np.isnan(t2p):
            continue

        rows.append({"global_uid": global_uid, "t2p_ms": t2p})

    return rows


def main():
    # Discover sessions across all shanks
    all_rows = []
    shank_dirs = sorted(WF_ROOT.glob("shank_*"))
    if not shank_dirs:
        print(f"No shank directories found in {WF_ROOT}")
        return

    for shank_dir in shank_dirs:
        shank_id = shank_dir.name
        session_dirs = sorted(d for d in shank_dir.iterdir() if d.is_dir())
        print(f"{shank_id}: {len(session_dirs)} sessions")

        for sess_dir in session_dirs:
            sess_name = sess_dir.name  # e.g. BG_046_01072025
            date_str = sess_name.split("_", 2)[-1]
            session_date = int(date_str)

            rows = extract_session_waveform_features(sess_dir)
            for r in rows:
                r["session_date"] = session_date
            all_rows.extend(rows)

        print(f"  Running total: {len(all_rows)} waveforms")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal waveforms loaded: {len(df)}")

    # De-duplicate: same global_uid can appear in multiple shanks' dirs
    # (shouldn't happen, but just in case)
    df = df.drop_duplicates(subset=["session_date", "global_uid"])

    # Filter valid T2P range
    valid = df[(df["t2p_ms"] > 0.02) & (df["t2p_ms"] < 1.5)].copy()
    print(f"Valid T2P (0.02-1.5 ms): {len(valid)}")

    # GMM classification (2 components)
    X = valid["t2p_ms"].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
    labels = gmm.predict(X)
    means = gmm.means_.flatten()

    narrow_label = int(np.argmin(means))
    threshold = np.mean(means)

    valid["celltype"] = np.where(
        labels == narrow_label, "Narrow (FSI)", "Broad (MSN/Proj)"
    )
    print(f"GMM means: {means[narrow_label]:.3f} ms (narrow), "
          f"{means[1 - narrow_label]:.3f} ms (broad)")
    print(f"Threshold: {threshold:.3f} ms")
    print(f"  Narrow (FSI):     {(valid['celltype'] == 'Narrow (FSI)').sum()}")
    print(f"  Broad (MSN/Proj): {(valid['celltype'] == 'Broad (MSN/Proj)').sum()}")

    # Use global_uid as the cluster_id column (matches concat-sort pkls)
    out = valid[["session_date", "global_uid", "celltype"]].copy()
    out = out.rename(columns={"global_uid": "cluster_id"})
    out = out.sort_values(["session_date", "cluster_id"]).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
