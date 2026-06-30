"""WS2 builder: per-unit ISI/spike-train SHAPE features for cell-type label validity.

The waveform width split (FSI/SPN) risks sorting partly on firing RATE (width<->rate
correlated). To de-confound we need a SECOND axis that is rate-INDEPENDENT. This builder
computes, per good/stable unit (reusing spike_times), shape features:
  - cv2        : local ISI irregularity (rate-independent; FSI regular -> low, SPN bursty -> high)
  - burst_frac : fraction of ISIs < 10 ms (bursting)
  - isi_mode_s : modal ISI from the canonical log-ISI histogram (reused isi_log_histogram)
  - median_isi_s, rate_hz : reported for context (rate is the CONFOUND, shown not used as the axis)

Session pkl loading is sequential (I/O-bound); per-unit ISI compute is O(n_spikes) and
light, so it runs inline (no pool needed — pools are reserved for the heavy resample loops).

Output: data/cache/talk_substrate/bg046_isi_features.csv
Usage: py scripts/talk_substrate/ws2_isi_features.py [--limit N]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gc
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

from visdetect.suite.loader import load_session                    # noqa: E402
from visdetect.analysis.utils import get_good_cluster_ids          # noqa: E402
from visdetect.analysis.tracking_qc import isi_log_histogram       # noqa: E402

OUT = C.isi_features_path(C.SUBJECT)   # BG_046 -> legacy bg046_isi_features.csv; others isi_features_<S>.csv


def isi_features(st: np.ndarray) -> dict:
    st = np.sort(np.asarray(st, dtype=float))
    out = dict(n_spikes=int(st.size), rate_hz=np.nan, cv2=np.nan,
               burst_frac=np.nan, median_isi_s=np.nan, isi_mode_s=np.nan)
    if st.size < 20:
        return out
    dur = st[-1] - st[0]
    out["rate_hz"] = st.size / dur if dur > 0 else np.nan
    isi = np.diff(st)
    isi = isi[isi > 0]
    if isi.size < 10:
        return out
    out["median_isi_s"] = float(np.median(isi))
    out["burst_frac"] = float(np.mean(isi < 0.010))
    # CV2: rate-independent local irregularity
    a, b = isi[:-1], isi[1:]
    out["cv2"] = float(np.mean(2.0 * np.abs(b - a) / (a + b)))
    h, centers = isi_log_histogram(st)
    if np.isfinite(h).any():
        out["isi_mode_s"] = float(centers[np.nanargmax(h)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ct_lookup, sessions_8 = C.celltype_and_sessions(C.SUBJECT)
    if args.limit:
        sessions_8 = sessions_8[: args.limit]
    rows = []
    for si, s8 in enumerate(sessions_8, 1):
        try:
            sess = load_session(s8)
        except Exception as e:  # noqa: BLE001
            print(f"  [{si}/{len(sessions_8)}] {s8}: load failed ({e}); skip"); continue
        cids = get_good_cluster_ids(sess)
        cmap = {c.cluster_id: c for c in sess.clusters}
        for c in cids:
            cl = cmap.get(c)
            st = getattr(cl, "spike_times", None) if cl is not None else None
            feats = isi_features(st if st is not None else np.array([]))
            feats.update(session_8=s8, cluster_id=int(c),
                         celltype=ct_lookup.get((s8, int(c)), C.UNKNOWN))
            rows.append(feats)
        print(f"  [{si}/{len(sessions_8)}] {s8}: {len(cids)} units")
        del sess; gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"[WS2] wrote {OUT}  ({len(df)} units)")
    # quick sanity: CV2 by cell type
    for ct in (C.NARROW, C.BROAD):
        s = df.loc[df.celltype == ct, "cv2"].dropna()
        print(f"  {ct}: cv2 median={s.median():.3f} (n={len(s)})")


if __name__ == "__main__":
    main()
