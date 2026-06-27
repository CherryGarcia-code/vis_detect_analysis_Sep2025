#!/usr/bin/env python3
"""Fair test: DANT curation WITH the functional corroborator ON vs biophysical-only.

The shipped DANT curation ran biophysical-only (empty states -> corroborator
abstains). The UM curation we compare against ran with the in-zone functional
corroborator ACTIVE, which both (a) restricts PSTHs to in-zone/StimSens-state
trials and (b) requires those PSTHs to correlate across sessions to stay trusted.
So UM-trusted was selected for state-matched PSTH agreement; DANT-trusted was not.

This script re-curates the SAME DANT registry with the corroborator ON (real state
tags), into an isolated dir/cache, and reports tier counts + held-out ISI AUC so we
can compare DANT-with-corroborator against the biophysical DANT run and the UM
yardstick on equal footing.

Run from the worktree root with the analysis interpreter.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import curate_dant as cd            # noqa: E402
import inclusive_trusted as it      # noqa: E402

WT = cd.WORKTREE_ROOT
PRIMARY = cd.PRIMARY_DEFAULT
REG = WT / "data" / "cache" / "dant" / "BG_046" / "dant_registry_curation.csv"
STATES = PRIMARY / "data" / "cache" / "states" / "BG_046"      # real state tags -> corroborator ON
OUT = WT / "FIGURES" / "tracking_dant" / "BG_046" / "curation_corrob"
CACHE = WT / "data" / "cache" / "dant" / "BG_046" / "curation_features_dant_corrob.pkl"
CURATE = WT / "scripts" / "pipelines" / "tracking" / "curate_tracks.py"
RAWWF = PRIMARY / "data" / "unit_match" / "input" / "BG_046"
PKL = PRIMARY / "data" / "pkls" / "BG_046"

# Biophysical-only reference (shipped DANT run) + UM yardstick, for the table.
BIOPHYS = {"trusted": (155, 0.9401491442023553), "review": (759, 0.7029232411620964),
           "suspect": (108, float("nan"))}
UM_YARD = {"trusted": (22, 0.96), "review": (567, float("nan")), "suspect": (160, float("nan"))}


def main() -> int:
    if not STATES.is_dir() or not any(STATES.iterdir()):
        print(f"ERROR: state tags missing at {STATES} -> corroborator cannot run", flush=True)
        return 1

    cmd = [sys.executable, str(CURATE), "--subject", "BG_046",
           "--registry", str(REG), "--liberal-col", "dant_uid",
           "--raw-wf-root", str(RAWWF), "--pkl-dir", str(PKL),
           "--states-dir", str(STATES),
           "--out-dir", str(OUT), "--cache-path", str(CACHE),
           "--drift-source", "none", "--min-span", "2", "--rebuild-cache"]
    print("[corrob] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Held-out ISI AUC on the corroborator-on tracks (normalized join).
    sjp, tc, _ = cd._import_pipeline("BG_046")
    tracks = pd.read_csv(OUT / "curated_tracks.csv")
    reg = pd.read_csv(REG, dtype={"session": str})
    kept_pairs = it.kept_pairs_from(tracks, reg, sjp.session_date_key)
    print(f"[corrob] kept_pairs={len(kept_pairs)}", flush=True)
    holdout = cd.collect_holdout_isi(kept_pairs, "BG_046", PKL)
    auc = tc.held_out_isi_auc_by_tier(tracks, holdout)

    counts = tracks["confidence_tier"].value_counts().to_dict()
    rows = []
    for tier in ["trusted", "review", "suspect"]:
        rows.append({
            "tier": tier,
            "corrob_n": int(counts.get(tier, 0)),
            "corrob_isi_auc": auc.get(tier, {}).get("auc", float("nan")),
            "biophys_n": BIOPHYS[tier][0], "biophys_isi_auc": BIOPHYS[tier][1],
            "um_n": UM_YARD[tier][0], "um_isi_auc": UM_YARD[tier][1],
        })
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "corroborator_compare.csv", index=False)
    with open(OUT / "corroborator_compare.json", "w") as f:
        json.dump({"counts": counts, "isi_auc": auc}, f, indent=2)
    print("[corrob] tiers:", counts, flush=True)
    print(table.to_string(index=False), flush=True)
    print(f"wrote {OUT / 'corroborator_compare.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
