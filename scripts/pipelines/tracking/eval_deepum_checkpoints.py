#!/usr/bin/env python3
"""Run the DeepUM tracking pipeline for each fine-tuned export checkpoint and
tabulate the span distribution vs the UM and stock-DeepUM baselines.

For each export_epoch_*.pt (optionally strided), invokes run_deepunitmatch_all.py
--ckpt and reads its run_summary.json, then writes finetune_comparison.csv.
"""
from __future__ import annotations
import argparse, os, sys, json, glob, subprocess
from pathlib import Path
import pandas as pd

# Hard reference numbers (from memory/neuron_tracking_may2026.md side-by-side).
BASELINE_ROWS = [
    {"label": "UM 3.2.9", "ge_2_pct": 19.8, "ge_5_pct": 4.9, "ge_10_pct": 1.6,
     "ge_15_pct": 0.9, "ge_20_pct": 0.5, "max_span": 28},
    {"label": "DeepUM stock", "ge_2_pct": 6.3, "ge_5_pct": 0.4, "ge_10_pct": 0.03,
     "ge_15_pct": 0.0, "ge_20_pct": 0.0, "max_span": 14},
]
THRESHOLDS = [2, 5, 10, 15, 20]


def summarize_run(label, summary):
    n = summary.get("n_tracked_ids", 0) or 0
    row = {"label": label, "n_tracked_ids": n, "max_span": summary.get("max_span", 0)}
    for t in THRESHOLDS:
        c = summary.get(f"ge_{t}", 0)
        row[f"ge_{t}_pct"] = round(100 * c / n, 3) if n else 0.0
    return row


def select_checkpoints(ckpt_dir, stride):
    exports = sorted(glob.glob(os.path.join(ckpt_dir, "export_epoch_*.pt")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not exports:
        return []
    epochs = [int(p.split("_")[-1].split(".")[0]) for p in exports]
    chosen = [p for p, e in zip(exports, epochs) if e % stride == 0]
    if exports[-1] not in chosen:      # always include the last epoch
        chosen.append(exports[-1])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="dir with export_epoch_*.pt")
    ap.add_argument("--input", required=True, help="UM input dir (BG_046 sessions)")
    ap.add_argument("--out-root", required=True, help="where per-ckpt outputs + CSV go")
    ap.add_argument("--runner", default=str(Path(__file__).with_name("run_deepunitmatch_all.py")))
    ap.add_argument("--stride", type=int, default=10, help="eval every Nth epoch + last")
    ap.add_argument("--label-prefix", default="ft")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    rows = list(BASELINE_ROWS)
    for ckpt in select_checkpoints(args.ckpt_dir, args.stride):
        epoch = int(ckpt.split("_")[-1].split(".")[0])
        label = f"{args.label_prefix}_ep{epoch}"
        out_dir = os.path.join(args.out_root, label)
        subprocess.run([sys.executable, args.runner, "--input", args.input,
                        "--out-dir", out_dir, "--ckpt", ckpt], check=True)
        with open(os.path.join(out_dir, "run_summary.json")) as f:
            summary = json.load(f)
        rows.append(summarize_run(label, summary))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_root, "finetune_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
