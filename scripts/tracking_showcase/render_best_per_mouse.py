#!/usr/bin/env python3
"""Representative figure: the best *tracked AND functional* neuron per mouse.

One column per mouse (BG_046 DMS, BG_031 VMS, BG_039 DMS), three rows:
  row 1  waveform overlaid across all tracked sessions (light->dark by order)
         -> the "same neuron" tracking evidence.
  row 2  Change_ON large-hit PSTH, per-session overlay (light->dark) + bold
         across-session mean with 95% CI -> the functional response, shown to
         persist/evolve across the tracked span.
  row 3  each neuron's functional signature:
         046 -> change-evoked Hz, Learning vs Expert (it GROWS with learning)
         031 -> TF-kernel c1_r per session (purple = GLM TF-encoding) + AUROC
         039 -> change-evoked Hz per session (strong, maintained over 16 sess)

Candidates were chosen from cached tracking + functional evaluations
(data/cache/tracking_consensus, FIGURES/tracking_dant/.../behavior_figs,
data/cache/tf_responsive). All PKLs / raw waveforms are local (no X: compute).

Usage:  py scripts/tracking_showcase/render_best_per_mouse.py
Output: FIGURES/tracking_showcase/best_per_mouse.png (+ .pdf)
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/pipelines/tracking"))

from visdetect.analysis.config import session_date_key      # noqa: E402
from visdetect.core.session import load_session             # noqa: E402
from visdetect.analysis.tracking_qc import (                # noqa: E402
    extract_unit_psths, load_raw_mean_waveform, extract_peak_channel)
from visdetect.suite.config import STAGE_COLORS             # noqa: E402
from qc_sheet_figures import _shade_ramp                    # noqa: E402  (light->dark ramp)

STAGE_COLORS_LOCAL = {**STAGE_COLORS, "Unknown": "#9e9e9e"}
CHANGE_KEY = "change_on_big_hit"
PRE = (-0.5, 0.0)
POST = (0.0, 0.5)

OUT_DIR = REPO_ROOT / "FIGURES" / "tracking_showcase"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── candidate definitions ──────────────────────────────────────────────────
CANDIDATES = [
    dict(subj="BG_046", region="DMS · medial striatum", neuron="UM#942 ∩ DANT#631",
         tracker="UM ∩ DANT consensus", uid_kind="dant", uid=631,
         isi="held-out ISI r 0.97", tier="DANT trusted",
         headline="large-change response grows Learning→Expert", sig="stage"),
    dict(subj="BG_031", region="VMS · ventral striatum", neuron="DANT#756",
         tracker="DANT", uid_kind="dant", uid=756,
         isi="held-out-ISI validated", tier="DANT trusted",
         headline="TF-encoding 4/5 sessions · choice AUROC 0.72", sig="tf"),
    dict(subj="BG_039", region="DMS · medial striatum", neuron="UM#217",
         tracker="UnitMatch", uid_kind="um", uid=217,
         isi="longest track in cohort", tier="UM trusted",
         headline="strong change response held across 16 sessions", sig="session"),
]


# ── path helpers (per-subject) ──────────────────────────────────────────────
def _subject_paths(subj: str):
    os.environ["VISDETECT_SUBJECT"] = subj
    import importlib
    import _subject_paths as sjp
    importlib.reload(sjp)
    return sjp


def _stage_map(subj: str) -> Dict[tuple, str]:
    """session_date_key -> stage from the SUBJECT-SPECIFIC staging manifest
    (the default loader only covers BG_046). Naive folded to Learning;
    Excluded/absent -> Unknown."""
    sjp = _subject_paths(subj)
    path = REPO_ROOT / f"data/{subj}_staging_manifest.csv"
    out: Dict[tuple, str] = {}
    if not path.exists():
        return out
    man = pd.read_csv(path)
    for _, r in man.iterrows():
        st = str(r["stage"])
        st = "Learning" if st == "Naive" else ("Unknown" if st == "Excluded" else st)
        out[sjp.session_date_key(r["session_name"])] = st
    return out


def _resolve_track(cand: dict) -> Tuple[List[str], Dict[str, int]]:
    """Return (chronological kept sessions, {session -> ks_unit_id}) for a candidate."""
    subj = cand["subj"]
    sjp = _subject_paths(subj)
    if cand["uid_kind"] == "dant":
        reg = pd.read_csv(REPO_ROOT / f"data/cache/dant/{subj}/dant_registry.csv",
                          dtype={"session": str})
        reg = reg[reg.dant_uid.astype(int) == cand["uid"]]
        ksmap = {str(r.session): int(r.ks_unit_id) for _, r in reg.iterrows()}
        if subj == "BG_046":                       # consensus: restrict to agreed sessions
            coh = pd.read_csv(REPO_ROOT / "data/cache/tracking_consensus/BG_046/consensus_cohort.csv")
            row = coh[coh.dant_uid.astype(int) == cand["uid"]].iloc[0]
            kept = [s for s in str(row["agreed_sessions"]).split(";") if s]
        else:                                      # DANT curation kept set
            ct = pd.read_csv(REPO_ROOT / f"FIGURES/tracking_dant/{subj}/curation/curated_tracks.csv")
            row = ct[ct.curated_uid.astype(int) == cand["uid"]].iloc[0]
            kept = [s for s in str(row["kept_sessions"]).split(";") if s]
    else:                                          # UnitMatch curated track
        ct = pd.read_csv(sjp.curation_out_dir(subj) / "curated_tracks.csv")
        row = ct[ct.curated_uid.astype(int) == cand["uid"]].iloc[0]
        kept = [s for s in str(row["kept_sessions"]).split(";") if s]
        reg = pd.read_csv(sjp.um_registry(subj), dtype={"session": str})
        reg = reg[reg.global_uid.astype(int) == int(row["liberal_uid"])]
        ksmap = {str(r.session): int(r.ks_unit_id) for _, r in reg.iterrows()}
    kept = sorted([s for s in kept if s in ksmap], key=sjp.session_date_key)
    return kept, ksmap


def _win(centers: np.ndarray, w) -> np.ndarray:
    c = np.asarray(centers)
    return (c >= w[0]) & (c < w[1])


def _evoked(psth: np.ndarray, centers: np.ndarray) -> float:
    pre, post = _win(centers, PRE), _win(centers, POST)
    if not pre.any() or not post.any():
        return float("nan")
    return float(psth[post].mean() - psth[pre].mean())


# ── data collection (one session load per session) ─────────────────────────
def collect(cand: dict) -> dict:
    subj = cand["subj"]
    sjp = _subject_paths(subj)
    stage_of = _stage_map(subj)
    kept, ksmap = _resolve_track(cand)
    raw_root = sjp.raw_wf_root(subj)
    pkl_dir = sjp.pkl_dir(subj)

    waves, psths, sems, centers = [], [], [], None
    stages, evoked, sess_out = [], [], []
    for s in kept:
        ks = ksmap[s]
        # waveform (peak channel)
        mean_wf = load_raw_mean_waveform(raw_root, s, ks)
        peak_wave = None
        if mean_wf is not None:
            pc = extract_peak_channel(mean_wf)
            peak_wave = np.asarray(mean_wf[:, pc], dtype=float)
        # PSTH
        pkl = sjp.session_pkl(subj, s, pkl_dir)
        if pkl is None:
            continue
        S = load_session(str(pkl))
        d = extract_unit_psths(S, ks, with_sem=True)
        p, c, n, sem = d.get(CHANGE_KEY, (None, None, 0, None))
        del S
        gc.collect()
        if p is None:
            continue
        centers = np.asarray(c, dtype=float)
        waves.append(peak_wave)
        psths.append(np.asarray(p, dtype=float))
        sems.append(np.asarray(sem, dtype=float) if sem is not None else None)
        stages.append(stage_of.get(sjp.session_date_key(s), "Unknown"))
        evoked.append(_evoked(np.asarray(p, dtype=float), centers))
        sess_out.append(s)

    # TF c1_r per session (031 signature)
    tf_c1r, tf_resp = [], []
    if cand["sig"] == "tf":
        tf = pd.read_csv(REPO_ROOT / f"data/cache/tf_responsive/{subj.lower().replace('_','')}_tf_responsive.csv")
        tf_key = {(sjp.session_date_key(r["session"]), int(r["unit"])):
                  (float(r["c1_r_log2"]), bool(r["resp_log2"])) for _, r in tf.iterrows()}
        for s in sess_out:
            v = tf_key.get((sjp.session_date_key(s), ksmap[s]))
            tf_c1r.append(v[0] if v else np.nan)
            tf_resp.append(v[1] if v else False)

    return dict(cand=cand, sessions=sess_out, stages=stages, waves=waves,
                psths=psths, sems=sems, centers=centers, evoked=evoked,
                tf_c1r=tf_c1r, tf_resp=tf_resp)


# ── plotting ────────────────────────────────────────────────────────────────
def _grade_colors(n: int, stages: List[str]) -> List[tuple]:
    return [_shade_ramp(STAGE_COLORS_LOCAL.get(stages[i], "#888888"),
                        i / (n - 1) if n > 1 else 1.0) for i in range(n)]


def draw_column(fig, gs_col, data: dict) -> None:
    cand = data["cand"]
    n = len(data["psths"])
    cols = _grade_colors(n, data["stages"])
    centers = data["centers"]

    # ── row 1: waveform overlay ─────────────────────────────────────────────
    ax = fig.add_subplot(gs_col[0])
    have_wave = any(w is not None for w in data["waves"])
    if have_wave:
        for i, w in enumerate(data["waves"]):
            if w is None:
                continue
            t = np.arange(w.size) / 30.0  # ms @ 30 kHz
            ax.plot(t, w, color=cols[i], linewidth=1.0, alpha=0.85)
        ax.set_xlabel("time (ms)", fontsize=10)
        ax.set_ylabel("µV", fontsize=10)
    else:
        ax.text(0.5, 0.5, "waveform n/a", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set_axis_off()
    ax.set_title("waveform · every tracked session", fontsize=11)
    ax.tick_params(labelsize=9)

    # ── row 2: Change_ON PSTH per-session overlay + bold mean + 95% CI ───────
    ax = fig.add_subplot(gs_col[1])
    base = _win(centers, PRE)
    mat = np.vstack([p for p in data["psths"]])
    # baseline-subtract each session so traces are comparable
    matb = mat - mat[:, base].mean(axis=1, keepdims=True)
    for i in range(n):
        ax.plot(centers, matb[i], color=cols[i], linewidth=1.0, alpha=0.85, zorder=2)
    gmean = matb.mean(axis=0)
    if n >= 2:
        sem = matb.std(axis=0, ddof=1) / np.sqrt(n)
        ax.fill_between(centers, gmean - 1.96 * sem, gmean + 1.96 * sem,
                        color="0.25", alpha=0.18, linewidth=0, zorder=3)
    ax.plot(centers, gmean, color="white", linewidth=3.4, zorder=4)
    ax.plot(centers, gmean, color="0.12", linewidth=2.2, zorder=5, label="mean ± 95% CI")
    ax.axvline(0, color="0.5", linewidth=0.7)
    ax.axhline(0, color="0.7", linewidth=0.5, zorder=0)
    ax.set_title("Change onset · large hit  (per session)", fontsize=11)
    ax.set_xlabel("time from change (s)", fontsize=10)
    ax.set_ylabel("Hz (rel. baseline)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.text(0.02, 0.02, f"{n} sessions · light→dark", transform=ax.transAxes,
            fontsize=8, color="0.35", ha="left", va="bottom")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    # ── row 3: functional signature (per neuron) ─────────────────────────────
    ax = fig.add_subplot(gs_col[2])
    sig = cand["sig"]
    if sig == "stage":                       # 046: Learning vs Expert evoked Hz
        order = ["Learning", "Expert"]
        vals, errs, labs, bcols = [], [], [], []
        for st in order:
            v = [e for e, s in zip(data["evoked"], data["stages"]) if s == st]
            if not v:
                continue
            vals.append(np.mean(v)); errs.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            labs.append(f"{st}\n(n={len(v)})"); bcols.append(STAGE_COLORS_LOCAL[st])
        x = np.arange(len(vals))
        ax.bar(x, vals, yerr=errs, color=bcols, capsize=4, width=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
        ax.set_ylabel("change-evoked Hz", fontsize=10)
        ax.set_title("response GROWS with learning", fontsize=11)
    elif sig == "tf":                        # 031: TF c1_r per session
        x = np.arange(n)
        bcols = ["#6a51a3" if r else "#bdbdbd" for r in data["tf_resp"]]
        ax.bar(x, data["tf_c1r"], color=bcols, width=0.7)
        ax.axhline(0.2, color="#c0392b", linestyle="--", linewidth=1.2, label="GLM TF thresh")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(f"{cand['subj']}_", "") for s in data["sessions"]],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("TF-kernel c1_r", fontsize=10)
        n_tf = int(np.sum(data["tf_resp"]))
        ax.set_title(f"TF-encoding {n_tf}/{n} sessions (purple)", fontsize=11)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    else:                                    # 039: change-evoked Hz per session
        x = np.arange(n)
        bcols = [STAGE_COLORS_LOCAL.get(s, "#888888") for s in data["stages"]]
        ax.bar(x, data["evoked"], color=bcols, width=0.8)
        ax.set_xticks(x[::2])
        ax.set_xticklabels([data["sessions"][i].replace(f"{cand['subj']}_", "")
                            for i in x[::2]], rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("change-evoked Hz", fontsize=10)
        ax.set_title("response maintained across sessions", fontsize=11)
    ax.tick_params(labelsize=9)


def main() -> int:
    datas = []
    for cand in CANDIDATES:
        print(f"collecting {cand['subj']} {cand['neuron']} ...", flush=True)
        d = collect(cand)
        print(f"  {len(d['psths'])} sessions with PSTH; stages={sorted(set(d['stages']))}", flush=True)
        datas.append(d)

    fig = plt.figure(figsize=(15.5, 12.5))
    outer = gridspec.GridSpec(1, 3, wspace=0.28, left=0.06, right=0.985,
                              top=0.82, bottom=0.105, figure=fig)
    left, span = 0.06, 0.925
    for j, d in enumerate(datas):
        cand = d["cand"]
        col = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[j], hspace=0.55,
            height_ratios=[1.0, 1.25, 1.0])
        draw_column(fig, col, d)
        # per-column header block (sits in the gap above top=0.82)
        xc = left + span * (j + 0.5) / 3
        fig.text(xc, 0.965, f"{cand['subj']}  ·  {cand['region']}",
                 ha="center", va="center", fontsize=14, fontweight="bold")
        fig.text(xc, 0.945, f"{cand['neuron']}   —   {cand['tracker']}",
                 ha="center", va="center", fontsize=11.5, color="0.15")
        fig.text(xc, 0.927, f"{len(d['psths'])} sessions · {cand['isi']} · {cand['tier']}",
                 ha="center", va="center", fontsize=10, color="0.35")
        fig.text(xc, 0.908, cand["headline"], ha="center", va="center",
                 fontsize=10.5, style="italic", color="#1a7a3a")

    fig.suptitle("One tracked & functional neuron per mouse — followed across learning",
                 fontsize=17, fontweight="bold", y=0.995)
    fig.text(0.5, 0.028,
             "shade = recording order (light → dark = earlier → later session)      ·      "
             "green = Learning / Expert stage,   grey = excluded / unstaged",
             ha="center", va="bottom", fontsize=10, color="0.3")
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"best_per_mouse.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
