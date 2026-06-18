"""Attach the per-frame stimulus log (phase/TF/vbl) to EXISTING pkls from raw
trials.json — without re-running Kilosort. Additive: only sets the new Trial
fields; spike data and all existing fields are untouched.

Usage (single):
    py scripts/conversion/backfill_stim_phase.py \
        --pkl   data/pkls/BG_046/BG_046_01072025.pkl \
        --raw   "X:/public/.../BG_046/Raw data/BG_046_01072025" \
        --out   data/pkls_stim_staging/BG_046/BG_046_01072025.pkl
Usage (batch over a subject):
    py scripts/conversion/backfill_stim_phase.py \
        --pkl-dir data/pkls/BG_046 \
        --raw-root "X:/public/.../BG_046/Raw data" \
        --out-dir  data/pkls_stim_staging/BG_046
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

from visdetect.core.session import load_session, save_session
from visdetect.core.ingest import extract_stim_timeseries


def _load_raw_trials(raw_session_dir: str) -> list:
    sess = Path(raw_session_dir) / "Session"
    raw = []
    for tf in sorted(sess.glob("*trials.json")):
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw.extend(data if isinstance(data, list) else [data])
    return raw


def backfill_session(pkl_path: str, raw_session_dir: str, out_path: str) -> dict:
    s = load_session(pkl_path)
    raw = _load_raw_trials(raw_session_dir)
    matched = len(raw) == len(s.trials)
    n_with = 0
    if matched:
        for trial, r in zip(s.trials, raw):
            stim = extract_stim_timeseries(r)
            trial.stim_phase = stim["stim_phase"]
            trial.stim_tf_disp = stim["stim_tf_disp"]
            trial.stim_vbl = stim["stim_vbl"]
            if stim["stim_phase"] is not None:
                n_with += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_session(s, out_path)
    return {"n_trials": len(s.trials), "n_with_phase": n_with, "matched": matched}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pkl"); p.add_argument("--raw"); p.add_argument("--out")
    p.add_argument("--pkl-dir"); p.add_argument("--raw-root"); p.add_argument("--out-dir")
    a = p.parse_args(argv)
    if a.pkl:
        info = backfill_session(a.pkl, a.raw, a.out)
        print(Path(a.pkl).name, info)
        return 0 if info["matched"] else 1
    pkls = sorted(glob.glob(str(Path(a.pkl_dir) / "*.pkl")))
    bad = []
    for pkl in pkls:
        sname = Path(pkl).stem
        raw = Path(a.raw_root) / sname
        out = Path(a.out_dir) / Path(pkl).name
        if not (raw / "Session").exists():
            print("NO RAW:", sname); bad.append(sname); continue
        info = backfill_session(pkl, str(raw), str(out))
        print(sname, info)
        if not info["matched"] or info["n_with_phase"] == 0:
            bad.append(sname)
    print(f"\nDONE: {len(pkls)} pkls, {len(bad)} need attention: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
