"""Apply the BG_046-fitted session-group rule to other subjects.

Fits the depth-3 tree on the BG_046 manual labels (the only manually labelled subject),
then predicts groups for every tagged session of the target subjects and reports whether
the rule TRANSFERS sensibly.

⚠ This is APPLICATION, not validation. Without manual labels for BG_039/BG_031 there is no
transfer kappa — only plausibility diagnostics (distribution, ordering along training time,
and whether the known VMS impulsive non-learner BG_031 looks different). To actually validate,
label those subjects manually and compare (the stronger design).

Run: py scripts/session_sorting/apply_session_group_rule.py
Out: data/cache/session_sorting/predicted_session_groups.csv
"""
import os
import sys
import argparse
import importlib.util

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier

from visdetect.analysis.config import canonical_session_id, session_date_key

# reuse the fitter's feature code (do not duplicate)
_spec = importlib.util.spec_from_file_location(
    "fitmod", os.path.join(os.path.dirname(__file__), "fit_session_group_rule.py"))
fitmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fitmod)

TAG_DIR = fitmod.TAG_DIR
OUT_DIR = fitmod.OUT_DIR
REF = "BG_046"


def tagged_sessions(subject):
    tdir = os.path.join(TAG_DIR, subject)
    out = []
    for f in sorted(os.listdir(tdir)):
        if not f.endswith(".csv") or f.startswith("_"):
            continue
        if fitmod.REQUIRED_TAG_COLS <= set(pd.read_csv(os.path.join(tdir, f), nrows=0).columns):
            out.append(f[:-4])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="BG_039,BG_031")
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--rank-normalize", action="store_true",
                    help="convert each feature to a WITHIN-SUBJECT percentile rank before "
                         "fitting/predicting, so thresholds mean 'high for this mouse' rather "
                         "than an absolute value. WARNING: this forces every mouse onto the same "
                         "feature distribution, so it erases genuine between-subject differences "
                         "by construction (see the printed group distributions).")
    args = ap.parse_args()

    # ── fit the reference rule on BG_046 manual labels ───────────────
    lab = pd.read_csv(fitmod.LABELS_PATH, dtype={"session_name": str})
    lab = lab[lab["subject"] == REF]
    first = (lab.sort_values("presentation_idx")
                .drop_duplicates("session_name", keep="first"))
    first = first[first["group"] != "Unsure"]
    ref = fitmod.build_features(REF, first["session_name"].tolist()).merge(
        first[["session_name", "group"]], on="session_name")
    keep = ref["group"].value_counts().loc[lambda s: s >= args.min_per_class].index
    ref = ref[ref["group"].isin(keep)]
    fcols = [c for c in ref.columns if c not in ("subject", "session_name", "group")]

    def prep(df):
        """Feature matrix; optionally within-subject percentile-ranked.

        NOTE a decision tree is invariant to MONOTONIC transforms of individual features, so
        rank-normalising the reference subject cannot change the fitted tree or its LOSO kappa —
        it only changes what the thresholds MEAN when applied to a different mouse.
        """
        X = df[fcols].astype(float)
        if args.rank_normalize:
            X = X.rank(pct=True)
        return X.fillna(0.0).values

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, class_weight="balanced",
                                  random_state=42).fit(prep(ref),
                                                       ref["group"].astype(str).values)
    print(f"Reference rule fitted on {REF}: n={len(ref)}, classes={sorted(set(ref['group']))}\n")

    rows = []
    for subj in [s.strip() for s in args.targets.split(",") if s.strip()]:
        feats = fitmod.build_features(subj, tagged_sessions(subj))
        if feats.empty:
            print(f"{subj}: no usable tag files"); continue
        feats = feats.reindex(columns=list(feats.columns) + [c for c in fcols
                                                             if c not in feats.columns])
        pred = tree.predict(prep(feats))          # ranks computed WITHIN this subject
        feats["predicted_group"] = pred

        man_path = os.path.join(_ROOT, "data", f"{subj}_staging_manifest.csv")
        man = pd.read_csv(man_path, dtype={"session_name": str})
        # Join on session_date_key, NOT canonical_session_id: BG_031 mixes 18 six-digit with
        # 24 eight-digit tokens (BG_039 2 + 30), and canonical_session_id maps '050325'->
        # '00050325' but '05032025'->'05032025', so the same date can MISJOIN across widths.
        man["k"] = man["session_name"].map(session_date_key)
        feats["k"] = feats["session_name"].map(session_date_key)
        m = feats.merge(man[["k", "stage", "d_prime"]], on="k", how="left")
        m["chron"] = m["k"].map(session_date_key)
        m = m.sort_values("chron").reset_index(drop=True)
        m["session_idx"] = range(len(m))
        m["subject"] = subj
        rows.append(m)

        print(f"=== {subj} ({len(m)} sessions) ===")
        print(m["predicted_group"].value_counts().to_string())
        summ = (m.groupby("predicted_group")
                  .agg(n=("predicted_group", "size"),
                       median_session_idx=("session_idx", "median"),
                       median_dprime=("d_prime", "median")).round(2))
        print(summ.to_string())
        if m["predicted_group"].nunique() > 1:
            rho, p = spearmanr(m["session_idx"], (m["predicted_group"] == "Balanced").astype(int))
            print(f"  P(Balanced) vs training time: rho={rho:+.2f}, p={p:.3f}")
            rho2, p2 = spearmanr(m["session_idx"],
                                 (m["predicted_group"] == "Impulsive-dominated").astype(int))
            print(f"  P(Impulsive)  vs training time: rho={rho2:+.2f}, p={p2:.3f}")
        print(pd.crosstab(m["predicted_group"], m["stage"]).to_string())
        print()

    if rows:
        out = pd.concat(rows, ignore_index=True)
        p = os.path.join(OUT_DIR, "predicted_session_groups.csv")
        out.to_csv(p, index=False)
        print(f"Saved: {p}")
        print("\nNOTE: APPLICATION, not validation - no transfer kappa without manual "
              "labels for these subjects.")


if __name__ == "__main__":
    main()
