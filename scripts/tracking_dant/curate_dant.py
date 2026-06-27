#!/usr/bin/env python3
"""Curate + QC-render DANT's BG_046 cross-session tracks (spec 2026-06-25).

Thin orchestration runner. Writes a curation-ready registry (dant_uid>0), then
drives the EXISTING registry-agnostic curation pipeline (curate_tracks.py /
render_curation_sheets.py) via subprocess with --liberal-col dant_uid, biophysical
-only (empty states dir -> corroborator abstains), into a DANT-specific out-dir so
the UnitMatch curation outputs are never touched. Held-out ISI AUC is computed
IN-PROCESS (validate_curation.py hardcodes the UM dir and would clobber it).

Run from the worktree root with the analysis interpreter:
    <PRIMARY>/.venv/Scripts/python.exe scripts/tracking_dant/curate_dant.py \
        [--steps registry,curate,validate,render,summary]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_DEFAULT = Path("E:/python_analysis/git_repos/vis_detect_analysis_Sep2025")

# UnitMatch curation yardstick (project records, memory neuron_tracking_may2026);
# referenced for the summary, NOT re-run here.
UM_YARDSTICK: Dict[str, dict] = {
    "trusted": {"n": 22, "auc": 0.96},
    "review": {"n": 567},
    "suspect": {"n": 160},
}


def write_curation_registry(in_csv, out_csv) -> Tuple[int, int]:
    """Keep only tracked rows (dant_uid > 0); write session, ks_unit_id, dant_uid.

    Drops the untracked (dant_uid <= 0) rows so they cannot collapse into one bogus
    mega-track (the pipeline filters only on --min-span, not on uid value).
    Returns (n_rows_kept, n_distinct_uids).
    """
    df = pd.read_csv(in_csv, dtype={"session": str})
    kept = df[df["dant_uid"].astype(int) > 0][["session", "ks_unit_id", "dant_uid"]].copy()
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_csv, index=False)
    return len(kept), int(kept["dant_uid"].nunique())


@dataclass(frozen=True)
class DantCurationPaths:
    """All paths the runner needs. Worktree-local outputs; PRIMARY data inputs."""
    worktree_root: Path
    primary_root: Path
    registry_in: Path          # data/cache/dant/BG_046/dant_registry.csv
    registry_curation: Path    # data/cache/dant/BG_046/dant_registry_curation.csv
    raw_wf_root: Path          # <PRIMARY>/data/unit_match/input/BG_046
    pkl_dir: Path              # <PRIMARY>/data/pkls/BG_046
    states_empty: Path         # empty -> corroborator abstains
    out_dir: Path              # FIGURES/tracking_dant/BG_046/curation
    cache_path: Path           # curation_features_dant.pkl
    sheets_dir: Path           # out_dir/sheets
    curate_script: Path        # scripts/pipelines/tracking/curate_tracks.py
    render_script: Path        # scripts/pipelines/tracking/render_curation_sheets.py

    @classmethod
    def default(cls, worktree_root, primary_root) -> "DantCurationPaths":
        wt = Path(worktree_root)
        pr = Path(primary_root)
        cache = wt / "data" / "cache" / "dant" / "BG_046"
        out = wt / "FIGURES" / "tracking_dant" / "BG_046" / "curation"
        tracking = wt / "scripts" / "pipelines" / "tracking"
        return cls(
            worktree_root=wt,
            primary_root=pr,
            registry_in=cache / "dant_registry.csv",
            registry_curation=cache / "dant_registry_curation.csv",
            raw_wf_root=pr / "data" / "unit_match" / "input" / "BG_046",
            pkl_dir=pr / "data" / "pkls" / "BG_046",
            states_empty=cache / "states_empty",
            out_dir=out,
            cache_path=cache / "curation_features_dant.pkl",
            sheets_dir=out / "sheets",
            curate_script=tracking / "curate_tracks.py",
            render_script=tracking / "render_curation_sheets.py",
        )


def build_curate_cmd(python_exe, paths: DantCurationPaths,
                     rebuild_cache: bool = True) -> List[str]:
    """argv for curate_tracks.py: biophysical-only, DANT out-dir, dant_uid column."""
    cmd = [
        str(python_exe), str(paths.curate_script),
        "--subject", "BG_046",
        "--registry", str(paths.registry_curation),
        "--liberal-col", "dant_uid",
        "--raw-wf-root", str(paths.raw_wf_root),
        "--pkl-dir", str(paths.pkl_dir),
        "--states-dir", str(paths.states_empty),
        "--out-dir", str(paths.out_dir),
        "--cache-path", str(paths.cache_path),
        "--drift-source", "none",
        "--min-span", "2",
    ]
    if rebuild_cache:
        cmd.append("--rebuild-cache")
    return cmd


def build_render_cmd(python_exe, paths: DantCurationPaths, tier: str,
                     max_uids: Optional[int] = None,
                     uids: Optional[List[int]] = None) -> List[str]:
    """argv for render_curation_sheets.py: one tier, DANT sheets dir, no pair scores."""
    cmd = [
        str(python_exe), str(paths.render_script),
        "--subject", "BG_046",
        "--tracks", str(paths.out_dir / "curated_tracks.csv"),
        "--registry", str(paths.registry_curation),
        "--liberal-col", "dant_uid",
        "--raw-wf-root", str(paths.raw_wf_root),
        "--pkl-dir", str(paths.pkl_dir),
        "--out-dir", str(paths.sheets_dir),
        "--tier", tier,
        "--no-pair-scores",
    ]
    if max_uids is not None:
        cmd += ["--max-uids", str(max_uids)]
    if uids:
        cmd += ["--uids", *[str(u) for u in uids]]
    return cmd


def write_validation_json(result: dict, out_dir) -> Path:
    """Write the per-tier AUC result to the GIVEN out_dir (never the UM dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "curation_validation.json"
    with open(p, "w") as f:
        json.dump(result, f, indent=2)
    return p


def _import_pipeline(subj: str):
    """Lazy-import the worktree pipeline modules (visdetect + _subject_paths).

    VISDETECT_SUBJECT must be set before _subject_paths is imported. We prepend the
    worktree src + tracking dir so we get THIS worktree's code, not the editable
    install pinned to PRIMARY (memory worktree_editable_install_pythonpath).
    """
    os.environ["VISDETECT_SUBJECT"] = subj
    sys.path.insert(0, str(WORKTREE_ROOT / "src"))
    sys.path.insert(0, str(WORKTREE_ROOT / "scripts" / "pipelines" / "tracking"))
    import _subject_paths as sjp
    from visdetect.analysis import track_curation as tc
    from visdetect.core.session import load_session
    return sjp, tc, load_session


def collect_holdout_isi(kept_pairs: Dict[Tuple[int, str], int], subj: str,
                        pkl_dir) -> Dict[Tuple[int, str], "object"]:
    """Holdout (odd-partition) log-ISI hist per kept (uid, session). Loads each
    session pkl once. Faithful to validate_curation.py lines 67-80."""
    import numpy as np
    sjp, tc, load_session = _import_pipeline(subj)
    holdout: Dict[Tuple[int, str], object] = {}
    for sess in sorted({s for (_, s) in kept_pairs}):
        pkl = sjp.session_pkl(subj, sess, pkl_dir)
        if pkl is None:
            print(f"  [validate] skip {sess}: no pkl", flush=True)
            continue
        S = load_session(str(pkl))
        cmap = {c.cluster_id: c for c in S.clusters}
        for (uid, s), kid in kept_pairs.items():
            if s != sess or kid not in cmap:
                continue
            _, hold = tc.partitioned_isi_hists(np.asarray(cmap[kid].spike_times))
            holdout[(uid, s)] = hold
        del S; gc.collect()
    return holdout


def step_validate(paths: DantCurationPaths, subj: str = "BG_046") -> dict:
    """Held-out ISI AUC by tier, written IN-PROCESS to the DANT out-dir."""
    sjp, tc, load_session = _import_pipeline(subj)
    tracks = pd.read_csv(paths.out_dir / "curated_tracks.csv")
    reg = pd.read_csv(paths.registry_curation, dtype={"session": str})
    reg["uid"] = reg["dant_uid"].astype(int)
    # (uid, session) -> ks_unit_id, restricted to each track's kept sessions.
    # NORMALIZE the join on session_date_key: curate_tracks.py reads the registry
    # WITHOUT dtype=str, so pandas strips leading zeros and writes kept_sessions
    # 7-digit ("8092025") for single-digit-day sessions, while here the registry is
    # read as str (padded "08092025"). Raw string equality would silently drop the
    # 14 single-digit-day sessions (~31% of pairs). See memory session-zfill issue.
    lut = {(int(u), sjp.session_date_key(k)): int(ks)
           for u, k, ks in zip(reg["uid"], reg["session"], reg["ks_unit_id"])}
    kept_pairs: Dict[Tuple[int, str], int] = {}
    for _, row in tracks.iterrows():
        uid = int(row["curated_uid"])
        for s in [s for s in str(row["kept_sessions"]).split(";") if s]:
            ks = lut.get((uid, sjp.session_date_key(s)))
            if ks is not None:
                kept_pairs[(uid, s)] = ks
    holdout = collect_holdout_isi(kept_pairs, subj, paths.pkl_dir)
    result = tc.held_out_isi_auc_by_tier(tracks, holdout)
    write_validation_json(result, paths.out_dir)
    print(f"[validate] kept_pairs={len(kept_pairs)}; held-out ISI AUC by tier: {result}",
          flush=True)
    return result


def build_summary_table(tier_counts: Dict[str, int], auc_by_tier: Dict[str, dict],
                        yardstick: Dict[str, dict] = UM_YARDSTICK) -> pd.DataFrame:
    """One row per tier: DANT track count + held-out ISI AUC, with the UM yardstick."""
    rows = []
    for tier in ["trusted", "review", "suspect"]:
        a = auc_by_tier.get(tier, {})
        y = yardstick.get(tier, {})
        rows.append({
            "tier": tier,
            "dant_n_tracks": int(tier_counts.get(tier, 0)),
            "dant_auc": a.get("auc", float("nan")),
            "dant_n_matched": int(a.get("n_matched", 0)),
            "dant_n_nonmatched": int(a.get("n_nonmatched", 0)),
            "um_n_tracks": y.get("n", float("nan")),
            "um_auc": y.get("auc", float("nan")),
        })
    return pd.DataFrame(rows)


def plot_summary(table: pd.DataFrame, out_png) -> None:
    """2-panel summary: tier counts (DANT vs UM) + held-out ISI AUC vs UM yardstick."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tiers = list(table["tier"])
    x = np.arange(len(tiers))
    fig, (axc, axa) = plt.subplots(1, 2, figsize=(11, 4.2))

    axc.bar(x - 0.2, table["dant_n_tracks"], width=0.4, label="DANT", color="#3474ae")
    axc.bar(x + 0.2, table["um_n_tracks"], width=0.4, label="UnitMatch", color="#9e9e9e")
    axc.set_xticks(x); axc.set_xticklabels(tiers)
    axc.set_ylabel("tracks (span>=2)"); axc.set_title("Tier counts")
    axc.legend(frameon=False)

    axa.bar(x, table["dant_auc"], width=0.5, color="#6baed6", label="DANT")
    for xi, v in zip(x, table["um_auc"]):
        if np.isfinite(v):
            axa.hlines(v, xi - 0.25, xi + 0.25, color="#ef6548", lw=2,
                       label="UM yardstick" if xi == 0 else None)
    axa.axhline(0.5, color="k", lw=0.8, ls=":", label="chance")
    axa.set_xticks(x); axa.set_xticklabels(tiers)
    axa.set_ylim(0.4, 1.0); axa.set_ylabel("held-out ISI AUC")
    axa.set_title("Held-out ISI AUC (quasi-independent; ISI-gated, no ablation)")
    axa.legend(frameon=False, fontsize=8)

    fig.suptitle("DANT BG_046 track curation vs UnitMatch yardstick")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def step_summary(paths: DantCurationPaths) -> pd.DataFrame:
    tracks = pd.read_csv(paths.out_dir / "curated_tracks.csv")
    tier_counts = tracks["confidence_tier"].value_counts().to_dict()
    val_path = paths.out_dir / "curation_validation.json"
    auc_by_tier = json.loads(val_path.read_text()) if val_path.exists() else {}
    table = build_summary_table(tier_counts, auc_by_tier)
    table.to_csv(paths.out_dir / "dant_curation_summary.csv", index=False)
    plot_summary(table, paths.out_dir / "dant_curation_summary.png")
    print(f"[summary] tiers={tier_counts}", flush=True)
    print(table.to_string(index=False), flush=True)
    return table


STEPS = ["registry", "curate", "validate", "render", "summary"]


def parse_steps(s: str) -> List[str]:
    """Comma list -> validated steps in canonical order."""
    want = {tok.strip() for tok in s.split(",") if tok.strip()}
    bad = want - set(STEPS)
    if bad:
        raise ValueError(f"unknown step(s): {sorted(bad)}; valid: {STEPS}")
    return [s for s in STEPS if s in want]


def step_registry(paths: DantCurationPaths) -> Tuple[int, int]:
    paths.states_empty.mkdir(parents=True, exist_ok=True)
    n_rows, n_uids = write_curation_registry(paths.registry_in, paths.registry_curation)
    print(f"[registry] kept {n_rows} rows / {n_uids} dant_uids (dant_uid>0) "
          f"-> {paths.registry_curation}", flush=True)
    return n_rows, n_uids


def step_curate(paths: DantCurationPaths, rebuild_cache: bool = True) -> None:
    paths.states_empty.mkdir(parents=True, exist_ok=True)
    cmd = build_curate_cmd(sys.executable, paths, rebuild_cache)
    print("[curate]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def step_render(paths: DantCurationPaths, tier: str,
                max_uids: Optional[int] = None,
                uids: Optional[List[int]] = None) -> None:
    cmd = build_render_cmd(sys.executable, paths, tier, max_uids=max_uids, uids=uids)
    print(f"[render:{tier}]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default=",".join(STEPS),
                    help="comma list of steps to run; default all")
    ap.add_argument("--primary", type=Path, default=PRIMARY_DEFAULT,
                    help="PRIMARY repo root (raw waveforms + pkls live there)")
    ap.add_argument("--review-max-uids", type=int, default=25,
                    help="cap on review-tier sheets (spot-check sample)")
    ap.add_argument("--trusted-max-uids", type=int, default=None,
                    help="cap on trusted-tier sheets (None = render all)")
    ap.add_argument("--no-rebuild-cache", action="store_true",
                    help="reuse an existing feature cache instead of rebuilding")
    args = ap.parse_args(argv)
    steps = parse_steps(args.steps)
    paths = DantCurationPaths.default(WORKTREE_ROOT, args.primary)
    print(f"DANT curation runner — steps={steps}\n  out_dir={paths.out_dir}", flush=True)

    if "registry" in steps:
        step_registry(paths)
    if "curate" in steps:
        step_curate(paths, rebuild_cache=not args.no_rebuild_cache)
    if "validate" in steps:
        step_validate(paths)
    if "render" in steps:
        step_render(paths, "trusted", max_uids=args.trusted_max_uids)
        step_render(paths, "review", max_uids=args.review_max_uids)
    if "summary" in steps:
        step_summary(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
