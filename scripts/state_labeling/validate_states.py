"""CLI: validate the state rule vs the experimenter's labels (kappa, confusion) and
produce a re-shade figure per labeled session."""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.suite.loader import load_session
from visdetect.analysis.state_labeling import (
    load_episodes, build_outcome_raster, episodes_to_trial_labels, render_raster,
)
from visdetect.analysis.state_calibration import (
    CalibrationResult, extract_state_features, tag_features,
)


def main():
    ap = argparse.ArgumentParser(description="Validate state rule vs labels; re-shade figures.")
    ap.add_argument("--model", default="data/state_labels/state_rule.pkl")
    ap.add_argument("--labels", default="data/state_labels/state_episodes.csv")
    ap.add_argument("--fig-dir", default="figures/state_labeler")
    args = ap.parse_args()

    result = CalibrationResult.load(args.model)
    episodes = load_episodes(args.labels)
    os.makedirs(args.fig_dir, exist_ok=True)

    y_true, y_pred = [], []
    for sn in sorted({e.session_name for e in episodes}):
        sess = load_session(sn)
        raster = build_outcome_raster(sess)
        feats = extract_state_features(raster, result.window)
        tagged = tag_features(result.tree, feats, confidence_threshold=0.0)  # no gating for agreement
        lab = episodes_to_trial_labels(episodes, sn, len(raster))
        for i in range(len(raster)):
            if lab[i] is not None:
                y_true.append(lab[i])
                y_pred.append(tagged.loc[i, "state_label"])

        fig, ax = plt.subplots(figsize=(12, 2))
        render_raster(ax, raster, episodes=[e for e in episodes if e.session_name == sn])
        ax.set_title(f"{sn} — tagger vs your labels")
        fig.savefig(os.path.join(args.fig_dir, f"reshade_{sn}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        del sess
        gc.collect()

    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    if y_true:
        k = cohen_kappa_score(y_true, y_pred)
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"Cohen's kappa vs labels: {k:.3f}")
        print("Confusion (rows=true, cols=pred):", labels)
        print(pd.DataFrame(cm, index=labels, columns=labels))
    else:
        print("No labeled trials to validate.")


if __name__ == "__main__":
    main()
