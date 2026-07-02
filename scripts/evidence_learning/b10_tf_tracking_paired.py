"""B10 tracking — paired within-session significance tests.

Settles the two SUGGESTIVE contrasts from the tracking figure with a paired test
(more powerful than comparing overlapping bootstrap bands):
  - engagement:  StimSens - Disengaged   (per session)
  - kernel class: sustained - transient   (per session)
For each session we take the mean tracking r in a fixed reference-lag WINDOW
(0.3-0.6 s, around the pooled peak) for each condition — a FIXED window avoids the
per-session max-over-lags selection bias. Then Wilcoxon signed-rank + a sign test
across sessions (session = replication unit), per region and pooled.

Run: py scripts/evidence_learning/b10_tf_tracking_paired.py
Out: data/cache/evidence_learning/b10_tf_tracking_paired.csv
"""
import os
import sys
import importlib.util as _u

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from visdetect.analysis import psychophysical_kernel as pk
from visdetect.analysis.evidence_learning_io import (
    CACHE_DIR, subject_sessions, tf_responsive_units, load_state_labels_by_key)

# reuse the tracking module's session_tracking + class/region config (by path)
_TRK = os.path.join(_ROOT, "scripts", "evidence_learning", "b10_tf_tracking.py")
_spec = _u.spec_from_file_location("b10trk", _TRK)
trk = _u.module_from_spec(_spec)
_spec.loader.exec_module(trk)

WIN = (0.30, 0.60)      # reference lag window (s) around the peak
CONF = 0.8


def _win_mean(curve, lags):
    mask = (lags >= WIN[0]) & (lags <= WIN[1])
    return float(np.nanmean(np.asarray(curve, float)[mask]))


def main():
    dt = pk.DT
    max_lag = int(round(trk.MAX_LAG_S / dt))
    lags = np.arange(max_lag + 1) * dt
    rows = []
    for region, subs in trk.REGION_POOLS.items():
        for subject in subs:
            resp = tf_responsive_units(subject, responsive=True)
            cls = trk.tf_responsive_classes(subject)
            rng = np.random.default_rng(pk.BOOT_SEED)
            for skey, sname, stage, sess in subject_sessions(subject):
                rsigns = resp.get(skey, {})
                if not rsigns:
                    continue
                _, _, per_trial = trk.session_tracking(sess, rsigns, rng)
                rec = {"region": region, "subject": subject, "skey": str(skey),
                       "stimsens": np.nan, "diseng": np.nan,
                       "transient": np.nan, "sustained": np.nan}
                # engagement: session-mean curve per state, then window-mean r
                labels = load_state_labels_by_key(subject, skey)
                if labels is not None:
                    b = {"StimSens": [], "Disengaged": []}
                    for idx, rc in per_trial.items():
                        if idx in labels.index:
                            row = labels.loc[idx]
                            if (float(row["state_confidence"]) >= CONF
                                    and row["state_label"] in b):
                                b[row["state_label"]].append(rc)
                    if b["StimSens"] and b["Disengaged"]:
                        rec["stimsens"] = _win_mean(np.nanmean(b["StimSens"], 0), lags)
                        rec["diseng"] = _win_mean(np.nanmean(b["Disengaged"], 0), lags)
                # kernel class: separate signed-sum for transient / sustained cells
                cmap = cls.get(skey, {})
                for name, want in (("transient", "transient"), ("sustained", "sustained")):
                    sig = {c: s for c, s in rsigns.items() if cmap.get(c) == want}
                    if sig:
                        cr, _, _ = trk.session_tracking(sess, sig, rng)
                        if cr is not None:
                            rec[name] = _win_mean(cr, lags)
                rows.append(rec)
    df = pd.DataFrame(rows)
    # SAVE FIRST (before any printing) so the expensive per-session computation is
    # never lost to a downstream error.
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(os.path.join(CACHE_DIR, "b10_tf_tracking_paired.csv"), index=False)
    print(f"saved {len(df)} session rows -> b10_tf_tracking_paired.csv")

    summ = []

    def paired(sub, region, a, b, label):
        d = sub[[a, b]].dropna()
        if len(d) < 5:
            print(f"  {label:34s}: n={len(d)} (too few)")
            return
        diff = (d[a] - d[b]).values
        npos = int((diff > 0).sum())
        try:
            _, p = wilcoxon(d[a].values, d[b].values)
        except ValueError:
            p = np.nan
        print(f"  {label:34s}: n={len(d):2d}  median_diff={np.median(diff):+.4f}  "
              f"{npos}/{len(d)} positive  Wilcoxon p={p:.4g}")
        summ.append({"region": region, "contrast": label, "n": len(d),
                     "median_diff": float(np.median(diff)), "n_positive": npos,
                     "wilcoxon_p": float(p)})

    for region in list(trk.REGION_POOLS) + ["ALL"]:
        sub = df if region == "ALL" else df[df.region == region]
        print(f"\n=== {region} (paired, window {WIN[0]}-{WIN[1]}s) ===")
        paired(sub, region, "stimsens", "diseng", "engagement (StimSens - Diseng)")
        paired(sub, region, "sustained", "transient", "kernel (sustained - transient)")
    pd.DataFrame(summ).to_csv(
        os.path.join(CACHE_DIR, "b10_tf_tracking_paired_summary.csv"), index=False)


if __name__ == "__main__":
    main()
