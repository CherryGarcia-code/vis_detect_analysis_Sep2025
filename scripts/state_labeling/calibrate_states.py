"""CLI: fit the behavioral-state rule from labeled episodes; save model + rules.md."""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from visdetect.suite.loader import load_session
from visdetect.analysis.state_labeling import load_episodes, build_outcome_raster
from visdetect.analysis.state_calibration import calibrate_states


def main():
    ap = argparse.ArgumentParser(description="Calibrate behavioral-state rule from labeled episodes.")
    ap.add_argument("--labels", default="data/state_labels/state_episodes.csv")
    ap.add_argument("--out-model", default="data/state_labels/state_rule.pkl")
    ap.add_argument("--out-rules", default="data/state_labels/rules.md")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    episodes = load_episodes(args.labels)
    if not episodes:
        raise SystemExit(f"No episodes found in {args.labels}")
    label_sessions = sorted({e.session_name for e in episodes})

    rasters = {}
    for sn in label_sessions:
        sess = load_session(sn)
        rasters[sn] = build_outcome_raster(sess)
        del sess
        gc.collect()

    result = calibrate_states(rasters, episodes, seed=args.seed)
    result.save(args.out_model)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_rules)), exist_ok=True)
    with open(args.out_rules, "w", encoding="utf-8") as f:
        f.write(f"# Behavioral-state rule\n\nwindow W = {result.window}\n")
        f.write(f"LOSO Cohen's kappa = {result.loso_kappa:.3f}\n")
        f.write(f"states = {result.state_labels}\n\n```\n{result.rules_text}\n```\n")
    print(f"Saved model -> {args.out_model}  (W={result.window}, kappa={result.loso_kappa:.3f})")
    print(f"Saved rules -> {args.out_rules}")


if __name__ == "__main__":
    main()
