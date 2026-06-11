"""CLI: tag all manifest sessions with behavioral states -> per-session CSV cache."""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from visdetect.analysis.config import load_staging_manifest
from visdetect.suite.loader import load_session
from visdetect.analysis.state_calibration import CalibrationResult, decode_session_states


def main():
    ap = argparse.ArgumentParser(description="Tag sessions with behavioral states.")
    ap.add_argument("--model", default="data/state_labels/state_rule.pkl")
    ap.add_argument("--out-dir", default="data/cache/state_tags")
    ap.add_argument("--confidence", type=float, default=0.8)
    args = ap.parse_args()

    result = CalibrationResult.load(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = load_staging_manifest(qc_only=True)
    for _, row in manifest.iterrows():
        sn = str(row["session_name"])
        sess = load_session(sn)
        tagged = decode_session_states(result, sess, confidence_threshold=args.confidence)
        tagged.to_csv(os.path.join(args.out_dir, f"{sn}.csv"), index=False)
        print(f"tagged {sn}: {len(tagged)} trials")
        del sess, tagged
        gc.collect()


if __name__ == "__main__":
    main()
