"""Regenerate HMM state labels from saved model PKLs.

Re-runs auto_label_states() on every saved model_K{K}.pkl and overwrites:
  - data/hmm/{subject}/state_labels_K{K}.json
  - data/hmm/{subject}/state_assignments_K{K}.csv  (hmm_state_label column)

Run this after updating auto_label_states() or after refitting with a new
prior/config so that stale JSON label files are corrected without a full refit.

Usage
-----
    py scripts/utils/relabel_hmm_states.py
    py scripts/utils/relabel_hmm_states.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from visdetect.analysis.config import HMM_DIR
from visdetect.analysis.hmm import GLMHMM, auto_label_states


def relabel(hmm_dir: Path, dry_run: bool = False) -> None:
    sel_path = hmm_dir / "model_selection.csv"
    if not sel_path.exists():
        print(f"ERROR: {sel_path} not found. Run fitting first.")
        sys.exit(1)

    sel_df = pd.read_csv(sel_path)
    best_K = int(sel_df.loc[sel_df["bic"].idxmin(), "K"])

    for _, row in sel_df.iterrows():
        K = int(row["K"])
        model_path = hmm_dir / f"model_K{K}.pkl"
        if not model_path.exists():
            print(f"  SKIP K={K}: {model_path} not found")
            continue

        model = GLMHMM.load(model_path)
        old_json = hmm_dir / f"state_labels_K{K}.json"
        old_labels = json.load(open(old_json))["labels"] if old_json.exists() else []
        new_labels = auto_label_states(model)

        tag = " ← BEST" if K == best_K else ""
        print(f"  K={K}{tag}:  {old_labels}  →  {new_labels}")

        if not dry_run:
            with open(old_json, "w") as f:
                json.dump({"K": K, "labels": new_labels}, f, indent=2)

            assign_path = hmm_dir / f"state_assignments_K{K}.csv"
            if assign_path.exists():
                df = pd.read_csv(assign_path, dtype={"session_name": str})
                if "hmm_state" in df.columns:
                    label_map = {i: lbl for i, lbl in enumerate(new_labels)}
                    df["hmm_state_label"] = df["hmm_state"].map(label_map)
                    df.to_csv(assign_path, index=False)
                    print(f"           → updated {assign_path.name}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        print("\nDone. Run expert_anchor_diagnostic.py to regenerate the figure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing files.")
    args = parser.parse_args()
    relabel(Path(HMM_DIR), dry_run=args.dry_run)
