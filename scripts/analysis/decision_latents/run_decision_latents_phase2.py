"""B8 Phase 2 — generative decision-latents ORCHESTRATION (Engine A, real data).

Plain English: Phase 1 *measured* three behavioural dials (sharpness / itchiness-
caution / timing) per cell. Phase 2 *fits a generative model* of them — a closed-
form cloglog hazard-accumulator — anchored at the expert end and seeded backward
across learning. This script wires the whole Phase-2 pipeline on BG_046 and
produces the two headline science answers:

  * LEARNING ladder  -> which dial learning turns (across anchors);
  * STATE   ladder    -> which dial the mood states load on (within an anchor);

plus the per-trial GENERATIVE latent table appended to the Phase-1 deliverable
(never overwriting the 25 Phase-1 columns), each dial tagged with its recovery
trust verdict.

Pipeline (reuses the library — does NOT reimplement):
  inventory -> select_expert_anchors (the GATE)
    if mode == "fallback": append Phase-1 proxies as 'descriptive' latents, STOP.
    else:
      mu_by_session   = change_time_anchor per session (reached trials)
      regime_by_session = expert vs naive (post-comprehension high-d' rule, below)
      rectification   = select_rectification on the most-expert anchor
      build_anchor_designs -> backward_sweep -> learning_ladder + per-anchor
        state_ladder  (AIC-only fast path; CV too slow on ~30 anchors)
      recovery verdict: INGEST data/cache/decision_latents/recovery_results.json
        (the CLUSTER harness output) if present, else a clearly-marked PENDING
        placeholder (all dials 'descriptive', recovery_pending=True) so the table
        stays honest until the cluster result lands.
      append_generative_latents -> the deliverable.

HARD RULE (this script): a `--quick` smoke runs 2-3 anchors + AIC-only ladders and
writes to `_smoke`-suffixed paths; it NEVER overwrites the real deliverable. The
full ~30-anchor run is launched by the controller as a background job, NOT here.

regime rule (documented): a session is 'expert' iff it is post-comprehension
(``assign_comprehension_flags`` rule='dprime', threshold 0.5 — the low knows-the-
rule bar) AND its d' > 0.7 (the same expert-anchor sensitivity bar used by the
Task-0.8 inventory / Task-0.9 gate). Everything else is 'naive'. This matches the
two recovery regimes the cluster harness validates ('expert' / 'naive'), so the
per-dial trust row is selected by the regime the dial was actually recovered at.

Worktree run recipe (PYTHONPATH MUST point at the worktree src or you silently
test main's code; memory/worktree_editable_install_pythonpath):

  WT="$(pwd)"   # .../.claude/worktrees/B8-phase2-generative
  # quick smoke (2-3 anchors, AIC-only, ~5-15 min; writes *_smoke paths):
  PYTHONPATH="$WT/src" py scripts/analysis/decision_latents/run_decision_latents_phase2.py --quick
  # FULL run (controller launches this in the background; ~30 anchors):
  PYTHONPATH="$WT/src" py scripts/analysis/decision_latents/run_decision_latents_phase2.py

Outputs:
  data/cache/decision_latents/decision_latents_by_state.csv        (FULL: appended)
  data/cache/decision_latents/decision_latents_by_state_smoke.csv  (--quick)
  data/cache/decision_latents/decision_latents_phase2_results.json  (+_smoke)
  FIGURES/decision_latents/BG_046/decision_latents_phase2_stats.csv (+_smoke)
"""
from __future__ import annotations

# ── BLAS single-thread BEFORE any numpy-importing module (process-parallel) ─────
# We process-parallelise the ladders, so each worker must keep BLAS to one thread
# or the threads oversubscribe the 20 cores and the ProcessPool stops scaling.
# This MUST run before importing numpy / visdetect / config / dlg (any of which
# pulls in numpy). Mirrors the pattern at the top of cluster_recovery_harness.py.
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import gc
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# cp1252-safe console (the user runs on a Windows cp1252 terminal locally).
try:  # pragma: no cover - console only
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style          # styling only
from visdetect.analysis.config import ROOT, SUBJECT, canonicalize_session_column
from visdetect.analysis import decision_latents as dl
from visdetect.analysis import decision_latents_generative as dlg

setup_style()

# ── paths (repo-structure convention: scripts/, FIGURES/, data/cache/) ──────────
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

INVENTORY_CSV = os.path.join(CACHE_DIR, "b8p2_expert_anchor_inventory.csv")
DELIVERABLE_CSV = os.path.join(CACHE_DIR, "decision_latents_by_state.csv")  # Phase-1 deliverable (read-only here)
RECOVERY_JSON = os.path.join(CACHE_DIR, "recovery_results.json")            # cluster harness output (may be absent)

# regime / expert thresholds (documented in the module docstring) ──────────────
EXPERT_DPRIME = 0.7        # same sensitivity bar as the Task-0.8 inventory
COMPREHENSION_THRESHOLD = 0.5  # low "knows-the-rule" bar (spec §7), rule='dprime'
SIGMA = dlg.ParamSpec().urgency_sigma  # FIXED urgency-bump width (a ParamSpec field)

# ── --quick smoke knobs (machinery proof, NOT the science) ──────────────────────
# The real expert anchors carry hundreds of trials each; the M_full ladder rung
# (a 12-param combined fit, 4 restarts) over the pooled trials dominates runtime.
# For the smoke we (a) keep only the 3 most-expert anchors, (b) SUBSAMPLE each
# anchor's Design to QUICK_N_TRIALS trials (Design.subset), and (c) use fewer
# restarts. This proves the end-to-end path in a few minutes without touching the
# FULL run (which uses every trial + every anchor).
QUICK_N_TRIALS = 200
QUICK_N_RESTARTS = 2
QUICK_N_ANCHORS = 3


# ── state-ladder process-pool worker (MODULE-LEVEL so it is picklable) ──────────
# Each anchor's state_ladder is fully INDEPENDENT (one Design, no cross-anchor
# seeding), so it is the biggest parallel win. The in-memory ``Design`` +
# ``param_spec`` are picklable (audit-verified) and sent directly — NO session
# reloading in the worker (which would hammer the X: gateway). Determinism is
# preserved because state_ladder derives all its seeds from the fixed ``seed`` arg;
# results are collected BY KEY (sname), never by arrival order.
def _state_ladder_worker(args):
    """ONE anchor's state ladder. Returns ``(sname, result_dict)``."""
    sname, design, param_spec, n_restarts, compute_cvll, seed = args
    sl = dlg.state_ladder(design, param_spec, n_restarts=n_restarts,
                          compute_cvll=compute_cvll, seed=seed)
    return sname, sl


def _quick_subsample(anchor_designs, n_trials=QUICK_N_TRIALS, seed=0):
    """Cap each anchor Design to ``n_trials`` random trials (smoke only). Keeps the
    ragged structure intact via :meth:`Design.subset`."""
    out = {}
    rng = np.random.default_rng(seed)
    for sname, d in anchor_designs.items():
        if len(d) > n_trials:
            idx = rng.choice(len(d), size=n_trials, replace=False)
            out[sname] = d.subset(np.sort(idx))
        else:
            out[sname] = d
    return out


def save_fig(fig, name):
    """Write a presentation-ready PNG to top-level FIGURES/ (not analysis_suite/)."""
    p = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# ════════════════════════════════════════════════════════════════════════════
# Task 4.4 — figures F6 / F7 / F8 (built from CACHED outputs; no recompute of the
# science). The ONLY compute these touch is the Engine-C spot-check for F8, which
# is run once and cached to ENGINEC_CSV. Mood colours come from STATE_LABEL_COLORS
# (the labeler palette — NOT the HMM palette); every figure carries a short
# plain-English title/caption a non-expert can read.
# ════════════════════════════════════════════════════════════════════════════
from matplotlib import gridspec  # noqa: E402  (figure-only import; keep local-ish)
from visdetect.analysis.config import STATE_LABEL_COLORS  # mood palette

RESULTS_JSON = os.path.join(CACHE_DIR, "decision_latents_phase2_results.json")
STATS_CSV = os.path.join(FIG_DIR, "decision_latents_phase2_stats.csv")
ENGINEC_CSV = os.path.join(CACHE_DIR, "enginec_spotcheck.csv")  # F8 Engine-C cache

# fitted moods that carry a generative dial (Disengaged is NOT fit -> no dials).
_FIG_MOODS = ("Impulsive", "StimSens")
# the top-3 expert anchors by d' (== the brief's 03092025/01092025/26082025);
# int-form session_name as stored in the deliverable / stats CSVs.
_ENGINEC_ANCHORS = ("3092025", "1092025", "26082025")
# moderate, SEEDED differential-evolution config: more thorough than the unit-test
# fast config but bounded so the spot-check completes in a sane wall-time.
_ENGINEC_FITPARAMS = {"seed": 0, "maxiter": 12, "popsize": 6, "polish": True}


def _zfill8(s):
    """int-form / float-form / str session id -> canonical zfill8 string."""
    return dlg.canonical_session_id(s)


# ── F6: can we trust the dials? (recovery at the real long-baseline regime) ─────
def make_f6_recovery(recovery_json=RECOVERY_JSON):
    """F6 — recovery at the real long-baseline regime, from the CORRECTED per-dial
    verdict in ``recovery_results.json`` (caution + timing = generative, sharpness =
    descriptive). Reads ``point`` (r/CCC/coverage per dial x regime), ``confusion``
    (3x3 matrices) and ``gate.<regime>.per_dial_trust`` — i.e. the SAME corrected
    blob the orchestration ingests; NOT the scalar-shrunk *.cluster_raw.json."""
    with open(recovery_json, "r", encoding="utf-8") as fh:
        R = json.load(fh)

    regimes = ["expert", "naive"]
    dial_pub = ["sharpness", "itchiness", "timing"]   # JSON keys
    dial_show = ["Sharpness\n(drift v)", "Itchiness/caution\n(start z)",
                 "Timing\n(urgency u)"]
    C = {"expert": "#1b7837", "naive": "#d6604d"}     # engaged-green vs hair-trigger-red

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.50,
                           wspace=0.34, left=0.07, right=0.97, top=0.88,
                           bottom=0.11)
    x = np.arange(3)
    w = 0.38

    def _quality_panel(ax, key, thresh, ylabel, title, thr_label):
        for k, reg in enumerate(regimes):
            vals = [R["point"][reg][d].get(key, np.nan) for d in dial_pub]
            bars = ax.bar(x + (k - 0.5) * w, vals, w, color=C[reg],
                          label=("Expert (engaged)" if reg == "expert"
                                 else "Naive (hair-trigger)"))
            for rect, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02,
                            f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.axhline(thresh, ls="--", lw=1.4, color="#333333")
        ax.text(2.46, thresh + 0.01, thr_label, ha="right", fontsize=8,
                color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(dial_show, fontsize=8.3)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold")

    axA = fig.add_subplot(gs[0, 0])
    _quality_panel(axA, "r", 0.80, "recovered-vs-true  r",
                   "A. Do we recover the dial?", "r >= 0.80")
    axA.legend(frameon=False, fontsize=7.8, loc="lower left")

    axB = fig.add_subplot(gs[0, 1])
    _quality_panel(axB, "ccc", 0.70, "Lin's concordance (CCC)",
                   "B. Is it concordant (bias/scale)?", "CCC >= 0.70")

    axC = fig.add_subplot(gs[0, 2])
    _quality_panel(axC, "ci_coverage", 0.90, "bootstrap CI coverage",
                   "C. Is the error bar honest?", "coverage >= 0.90")

    # ── D/E: confusion heatmaps (3x3) per regime ──
    conf_labels = ["sharp.", "caution", "timing"]
    for k, reg in enumerate(regimes):
        ax = fig.add_subplot(gs[1, k])
        M = np.asarray(R["confusion"][reg]["matrix"], float)
        im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="equal")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if M[i, j] > 0.5 else "#222222")
        ax.set_xticks(range(3)); ax.set_xticklabels(conf_labels, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(conf_labels, fontsize=8)
        ax.set_xlabel("model picked as", fontsize=8.5)
        ax.set_ylabel("dial we actually turned", fontsize=8.5)
        ax.set_title(f"D{k+1}. Mix-up matrix — {reg}", fontsize=10.5,
                     fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── F: per-(dial x regime) trust verdict table (CORRECTED) ──
    axT = fig.add_subplot(gs[1, 2])
    axT.axis("off")
    axT.set_title("F. Trust verdict (corrected, per dial x regime)",
                  fontsize=10.5, fontweight="bold")
    rows, cell_colors = [], []
    name_show = {"sharpness": "sharpness", "caution": "caution", "timing": "timing"}
    for reg in regimes:
        verdict = R["gate"][reg]["per_dial_trust"]
        for gd in ("sharpness", "caution", "timing"):
            v = verdict.get(gd, "?")
            rows.append([reg, name_show[gd], v])
            col = "#a1d99b" if v == "generative" else "#fdbe85"
            cell_colors.append(["#f0f0f0", "#f0f0f0", col])
    tbl = axT.table(cellText=rows, colLabels=["regime", "dial", "trust"],
                    cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    tbl.scale(1.0, 1.45)

    fig.suptitle("B8 F6 -- Can we trust each behavioural dial as a real mechanism?  "
                 "(recovery at the real long-baseline regime)",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.012,
             "A dial is 'trustworthy / generative' only if it RECOVERS (r>=0.80), is "
             "CONCORDANT (CCC>=0.70), has an HONEST error bar (coverage>=0.90) and is "
             "NOT confused with another dial. Caution & timing pass everywhere -> "
             "real mechanisms; sharpness fails (it trades off against the others) -> "
             "fall back to the descriptive Phase-1 proxy.",
             ha="center", fontsize=8.4, color="#555555")
    return fig


# ── F7: latent distributions, TIMING-LED ────────────────────────────────────────
def _mood_color(mood):
    return STATE_LABEL_COLORS.get(mood, "#888888")


def make_f7_latents(stats_csv=STATS_CSV, deliverable_csv=DELIVERABLE_CSV):
    """F7 — the three generative dials by mood + across learning anchors, LED by the
    labeler-INDEPENDENT readouts (timing-urgency + RT variability), with FA-rate /
    criterion x mood shown as CONFIRMATORY (state-label circularity). Sharpness is
    marked 'descriptive-trust' (hatched) so trust is legible. Mood colours from
    STATE_LABEL_COLORS."""
    stats = pd.read_csv(stats_csv)
    stats = stats[stats["mood"].isin(_FIG_MOODS)].copy()
    stats["sid"] = stats["session"].map(_zfill8)

    df = pd.read_csv(deliverable_csv)
    gen = df[~df["generative_omitted"].astype(bool)].copy()
    gen["sid"] = gen["session_name"].map(_zfill8)
    gen = gen[gen["state_label"].isin(_FIG_MOODS)]

    # per-anchor x mood confirmatory aggregates (labeler-dependent: FA / criterion)
    agg = gen.groupby(["sid", "state_label"]).agg(
        rtcv=("rt_cv_by_cs", "median"),
        lme=("lick_minus_expected", "median"),
        fa=("fa_rate_cell", "median"),
        crit=("criterion_c", "median"),
        sidx=("session_idx", "first"),
        dpr=("session_dprime", "first")).reset_index()
    M = stats.merge(agg, left_on=["sid", "mood"], right_on=["sid", "state_label"],
                    how="inner")
    M["sdate"] = M["sid"].map(lambda s: int(s))           # chrono-ish ordering key
    M = M.sort_values("dpr")

    moods = [m for m in _FIG_MOODS if m in set(M["mood"])]

    fig = plt.figure(figsize=(15.5, 9.0))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.42,
                           wspace=0.30, left=0.06, right=0.985, top=0.88,
                           bottom=0.12)

    def _violin_by_mood(ax, col, title, ylabel, descriptive=False):
        data, labels, colors = [], [], []
        for m in moods:
            v = pd.to_numeric(M.loc[M["mood"] == m, col], errors="coerce").dropna()
            if len(v):
                data.append(v.values); labels.append(m); colors.append(_mood_color(m))
        if not data:
            ax.text(0.5, 0.5, "no data", ha="center"); return
        parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.8)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor("#333333")
            if descriptive:
                pc.set_hatch("////")
        parts["cmeans"].set_color("#222222")
        for i, (d, c) in enumerate(zip(data, colors)):
            jx = np.random.default_rng(0).normal(i + 1, 0.04, size=len(d))
            ax.scatter(jx, d, s=10, color=c, edgecolor="white", linewidth=0.3,
                       zorder=3, alpha=0.85)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, fontsize=9)
        ttl = title + ("  [descriptive proxy]" if descriptive else "")
        ax.set_title(ttl, fontsize=10.5, fontweight="bold")
        if descriptive:
            ax.text(0.98, 0.97, "rough estimate\n(not recovered)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.6, color="#7a1f12",
                    bbox=dict(boxstyle="round", fc="#fff0e8", ec="#d6604d", lw=0.8))

    # ── TOP ROW: LEAD with the labeler-INDEPENDENT readouts ──
    # A: timing urgency u by mood (generative, trusted)
    axA = fig.add_subplot(gs[0, 0])
    _violin_by_mood(axA, "timing_u",
                    "A. TIMING urgency by mood  (generative)",
                    "fitted urgency dial  u")
    # B: RT variability (RT CV) by mood — labeler-independent timing readout
    axB = fig.add_subplot(gs[0, 1])
    _violin_by_mood(axB, "rtcv",
                    "B. RT variability by mood  (labeler-independent)",
                    "RT coefficient of variation")
    # C: timing urgency u ACROSS learning (vs d'), per mood
    axC = fig.add_subplot(gs[0, 2])
    for m in moods:
        sub = M[M["mood"] == m].sort_values("dpr")
        y = pd.to_numeric(sub["timing_u"], errors="coerce")
        axC.scatter(sub["dpr"], y, s=26, color=_mood_color(m), label=m,
                    edgecolor="white", linewidth=0.4, alpha=0.9)
    axC.set_xlabel("session sensitivity d'  (learning axis ->)", fontsize=9)
    axC.set_ylabel("fitted urgency dial  u", fontsize=9)
    axC.set_title("C. TIMING urgency across learning", fontsize=10.5,
                  fontweight="bold")
    axC.legend(frameon=False, fontsize=8, title="mood")

    # ── BOTTOM ROW: confirmatory (sharpness descriptive + caution circular) ──
    # D: sharpness drift v by mood — MARKED descriptive
    axD = fig.add_subplot(gs[1, 0])
    _violin_by_mood(axD, "sharpness_v",
                    "D. SHARPNESS drift by mood", "fitted drift dial  v",
                    descriptive=True)
    # E: itchiness/caution z by mood — CONFIRMATORY (circular w.r.t. early licks)
    axE = fig.add_subplot(gs[1, 1])
    _violin_by_mood(axE, "itchiness_z",
                    "E. ITCHINESS/caution by mood  (confirmatory)",
                    "fitted start-point dial  z")
    axE.text(0.98, 0.04, "caution x mood is partly\nDEFINITIONAL (circularity)",
             transform=axE.transAxes, ha="right", va="bottom", fontsize=7.6,
             color="#7a1f12",
             bbox=dict(boxstyle="round", fc="#fff0e8", ec="#d6604d", lw=0.8))
    # F: FA rate by mood (confirmatory readout that caution loads on mood)
    axF = fig.add_subplot(gs[1, 2])
    _violin_by_mood(axF, "fa",
                    "F. False-alarm rate by mood  (confirmatory)",
                    "FA rate (catch-trial licks)")
    axF.text(0.98, 0.04, "partly definitional\n(labels use early licks)",
             transform=axF.transAxes, ha="right", va="bottom", fontsize=7.6,
             color="#7a1f12",
             bbox=dict(boxstyle="round", fc="#fff0e8", ec="#d6604d", lw=0.8))

    fig.suptitle("B8 F7 -- The three behavioural dials by mood and across learning  "
                 "(timing-led)", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.018,
             "Leading with TIMING (labeler-independent) and RT variability. "
             "FA-rate / criterion x mood (E,F) are CONFIRMATORY -- the mood labels are "
             "defined partly from early-lick features (state-label circularity), so "
             "caution x mood is partly definitional. Sharpness (D) is a descriptive "
             "proxy (did not pass recovery) -- shown hatched. Mood colours: "
             "Impulsive (red), StimSens (light blue).",
             ha="center", fontsize=8.3, color="#555555")
    return fig


# ── F8: construct validity (GLM dials vs descriptive scores + Engine-C DDM) ──────
def run_enginec_spotcheck(force=False):
    """Run (or load) the Engine-C pyddm spot-check on the top-3 expert anchors and
    cache to ENGINEC_CSV. This is the ONLY compute Task 4.4 performs. Cached on
    disk so re-rendering F8 never re-fits pyddm."""
    if os.path.exists(ENGINEC_CSV) and not force:
        print(f"[enginec] loading cached spot-check -> {ENGINEC_CSV}")
        return pd.read_csv(ENGINEC_CSV)
    from visdetect.analysis.decision_latents_enginec import engine_c_spotcheck
    rows = []
    for a in _ENGINEC_ANCHORS:
        sess = load_session(a)
        try:
            d = engine_c_spotcheck([sess], dt=0.02, fitparams=_ENGINEC_FITPARAMS)
            d["anchor"] = a
            rows.append(d)
            print(f"[enginec] {a}: " + d.iloc[0][["v", "u", "a", "z", "ll",
                                                  "failed", "n_trials"]].to_dict().__str__())
        finally:
            del sess
            gc.collect()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(ENGINEC_CSV, index=False)
    print(f"[enginec] wrote {ENGINEC_CSV}")
    return out


def make_f8_construct_validity(stats_csv=STATS_CSV, deliverable_csv=DELIVERABLE_CSV,
                               enginec_df=None):
    """F8 — construct validity. (a) the three generative dials vs the Phase-1
    DESCRIPTIVE scores on the same anchors x moods (sharpness<->lapse-aware
    psychometric slope; caution<->FA-rate; timing<->lick-minus-expected); (b) the
    Engine-C panel: GLM dials (sharpness=v, timing=u, caution=z) vs the full pyddm
    DDM params (v, u; a/z noted) on 2-3 expert anchors."""
    from scipy.stats import spearmanr

    stats = pd.read_csv(stats_csv)
    stats = stats[stats["mood"].isin(_FIG_MOODS)].copy()
    stats["sid"] = stats["session"].map(_zfill8)

    df = pd.read_csv(deliverable_csv)
    gen = df[~df["generative_omitted"].astype(bool)].copy()
    gen["sid"] = gen["session_name"].map(_zfill8)
    gen = gen[gen["state_label"].isin(_FIG_MOODS)]
    agg = gen.groupby(["sid", "state_label"]).agg(
        psy=("sharpness_psy_slope", "median"),
        fa=("fa_rate_cell", "median"),
        crit=("criterion_c", "median"),
        lme=("lick_minus_expected", "median")).reset_index()
    M = stats.merge(agg, left_on=["sid", "mood"], right_on=["sid", "state_label"],
                    how="inner")

    if enginec_df is None:
        enginec_df = run_enginec_spotcheck()

    fig = plt.figure(figsize=(15.5, 9.0))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.40,
                           wspace=0.32, left=0.06, right=0.985, top=0.88,
                           bottom=0.13)

    def _scatter_corr(ax, xcol, ycol, xlabel, ylabel, title, descriptive=False,
                      circular=False):
        xx = pd.to_numeric(M[xcol], errors="coerce")
        yy = pd.to_numeric(M[ycol], errors="coerce")
        ok = xx.notna() & yy.notna()
        for m in _FIG_MOODS:
            sel = ok & (M["mood"] == m)
            ax.scatter(xx[sel], yy[sel], s=34, color=_mood_color(m), label=m,
                       edgecolor="white", linewidth=0.4, alpha=0.9)
        if ok.sum() >= 3:
            r, p = spearmanr(xx[ok], yy[ok])
            # OLS guide line
            b, a0 = np.polyfit(xx[ok], yy[ok], 1)
            xs = np.linspace(xx[ok].min(), xx[ok].max(), 50)
            ax.plot(xs, b * xs + a0, color="#444444", lw=1.3, ls="--")
            ax.text(0.04, 0.95, f"Spearman r = {r:+.2f}\np = {p:.1e}  (n={ok.sum()})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.6,
                    fontweight="bold",
                    bbox=dict(boxstyle="round", fc="white", ec="#999999", lw=0.7))
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ttl = title + ("  [descriptive]" if descriptive else "")
        ax.set_title(ttl, fontsize=10.3, fontweight="bold")
        if circular:
            ax.text(0.96, 0.06, "partly definitional\n(circularity)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4,
                    color="#7a1f12",
                    bbox=dict(boxstyle="round", fc="#fff0e8", ec="#d6604d", lw=0.8))

    # ── ROW 1: generative dial vs Phase-1 descriptive score ──
    axA = fig.add_subplot(gs[0, 0])
    _scatter_corr(axA, "timing_u", "lme",
                  "generative TIMING dial  u", "lick - expected change time (s)",
                  "A. Timing dial vs licking-early")
    axA.legend(frameon=False, fontsize=8, title="mood", loc="lower left")
    axB = fig.add_subplot(gs[0, 1])
    _scatter_corr(axB, "itchiness_z", "fa",
                  "generative CAUTION dial  z", "false-alarm rate",
                  "B. Caution dial vs FA rate", circular=True)
    axC = fig.add_subplot(gs[0, 2])
    _scatter_corr(axC, "sharpness_v", "psy",
                  "generative SHARPNESS dial  v", "lapse-aware psychometric slope",
                  "C. Sharpness dial vs psychometric", descriptive=True)

    # ── ROW 2: Engine-C — GLM dials vs full pyddm DDM params ──
    ec = enginec_df.copy()
    ec["sid"] = ec["anchor"].map(_zfill8) if "anchor" in ec.columns \
        else ec["session"].map(_zfill8)
    # per-anchor GLM dials = mean across the fitted moods (Impulsive/StimSens)
    glm = stats.groupby("sid").agg(glm_v=("sharpness_v", "mean"),
                                   glm_u=("timing_u", "mean"),
                                   glm_z=("itchiness_z", "mean")).reset_index()
    E = ec.merge(glm, on="sid", how="left")
    ok_fit = ~E["failed"].astype(bool) if "failed" in E.columns else np.ones(len(E), bool)
    E = E[ok_fit].copy()

    def _ec_scatter(ax, glm_col, ddm_col, xlabel, ylabel, title, note=""):
        if len(E) == 0 or E[glm_col].isna().all() or E[ddm_col].isna().all():
            ax.text(0.5, 0.5, "no successful DDM fit", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=10.3, fontweight="bold"); return
        ax.scatter(E[glm_col], E[ddm_col], s=70, color="#54278f",
                   edgecolor="white", linewidth=0.5, zorder=3)
        for _, r in E.iterrows():
            ax.annotate(str(int(r["sid"])), (r[glm_col], r[ddm_col]),
                        fontsize=7.2, xytext=(4, 3), textcoords="offset points")
        if E[[glm_col, ddm_col]].dropna().shape[0] >= 3:
            rr, pp = spearmanr(E[glm_col], E[ddm_col])
            ax.text(0.04, 0.95, f"Spearman r = {rr:+.2f}  (n={len(E)})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.4,
                    fontweight="bold",
                    bbox=dict(boxstyle="round", fc="white", ec="#999999", lw=0.7))
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10.3, fontweight="bold")
        if note:
            ax.text(0.5, -0.26, note, transform=ax.transAxes, ha="center",
                    fontsize=7.6, color="#555555")

    axD = fig.add_subplot(gs[1, 0])
    _ec_scatter(axD, "glm_v", "v", "GLM sharpness dial  v (mean)",
                "full-DDM drift  v", "D. Engine-C: sharpness vs DDM drift",
                note="sharpness is descriptive-trust")
    axE = fig.add_subplot(gs[1, 1])
    _ec_scatter(axE, "glm_u", "u", "GLM timing dial  u (mean)",
                "full-DDM urgency  u", "E. Engine-C: timing vs DDM urgency")
    axF = fig.add_subplot(gs[1, 2])
    _ec_scatter(axF, "glm_z", "z", "GLM caution dial  z (mean)",
                "full-DDM start-point  z", "F. Engine-C: caution vs DDM start-point",
                note="caution x mood is partly definitional (circularity)")

    fig.suptitle("B8 F8 -- Construct validity: do the generative dials track "
                 "independent measures of the same thing?", fontsize=14,
                 fontweight="bold")
    fig.text(0.5, 0.022,
             "Top row: each generative dial vs its Phase-1 DESCRIPTIVE counterpart "
             "(timing<->licking early, caution<->FA rate, sharpness<->psychometric "
             "slope). Bottom row: Engine-C cross-check -- the GLM dials vs a full "
             "drift-diffusion (pyddm) fit on the top-3 expert sessions (n=3 -> a "
             "QUALITATIVE spot-check, not a statistical test). The caution<->FA panel "
             "is partly DEFINITIONAL (state-label circularity); sharpness is "
             "descriptive-trust (failed recovery).",
             ha="center", fontsize=8.3, color="#555555")
    return fig


def make_task44_figures(force_enginec=False):
    """Render F6 / F7 / F8 from cached outputs (+ the Engine-C spot-check for F8)."""
    print("[F6] recovery (corrected per-dial verdict) ...", flush=True)
    save_fig(make_f6_recovery(), "fig_b8_F6_recovery")
    print("[F7] latent distributions (timing-led) ...", flush=True)
    save_fig(make_f7_latents(), "fig_b8_F7_latents")
    print("[F8] construct validity (+ Engine-C spot-check) ...", flush=True)
    ec = run_enginec_spotcheck(force=force_enginec)
    save_fig(make_f8_construct_validity(enginec_df=ec), "fig_b8_F8_construct_validity")
    print("[done] F6/F7/F8 written to", FIG_DIR, flush=True)
    return ec


def _csv_key(sname) -> str:
    """Canonical zfill8 session-id key (project DDMMYYYY convention).

    A session id like ``01072025`` (1 Jul 2025) is stored int64 in the deliverable,
    which drops the leading-zero DAY -> ``1072025`` (there is no ``1072025``
    session; it is just the int form of ``01072025``). This keys every per-session
    dict by the canonical zfill8 form. `append_generative_latents` canonicalizes
    BOTH its CSV ``session_name`` column and the dicts the same way, so the keys
    match regardless of representation (and sort chronologically). Delegates to the
    single source of truth in :func:`decision_latents_generative.canonical_session_id`.
    """
    return dlg.canonical_session_id(sname)


# ════════════════════════════════════════════════════════════════════════════
# Recovery verdict ingest (cluster harness output) or a clearly-marked PENDING
# ════════════════════════════════════════════════════════════════════════════
_GATE_DIALS = ("sharpness", "caution", "timing")


def _pending_recovery(regimes):
    """A clearly-marked 'pending' recovery verdict: every dial 'descriptive', with
    ``recovery_pending=True`` so the appended table is HONEST until the cluster
    ``recovery_results.json`` lands (we do NOT run full recovery here)."""
    out = {}
    for reg in regimes:
        out[reg] = {
            "per_dial_trust": {d: "descriptive" for d in _GATE_DIALS},
            "regime": reg,
            "recovery_pending": True,
        }
    return out


def load_recovery_by_regime(regimes, path=RECOVERY_JSON):
    """Ingest the per-(dial x regime) gate verdict from the cluster harness, or a
    pending placeholder if absent.

    Returns ``(recovery_by_regime, source)`` where ``source`` is 'cluster' or
    'pending'. The cluster JSON's ``gate.<regime>`` dict is exactly the
    ``recovery_gate`` output (``per_dial_trust`` keyed sharpness/caution/timing),
    which ``append_generative_latents`` consumes directly.
    """
    if not os.path.exists(path):
        print(f"[recovery] {path} ABSENT -> per-dial trust = PENDING "
              f"(all 'descriptive'; honest until the cluster result lands).")
        return _pending_recovery(regimes), "pending"

    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    gate = blob.get("gate", {}) or {}
    rec = {}
    for reg in regimes:
        g = gate.get(reg)
        if g and isinstance(g.get("per_dial_trust"), dict):
            rec[reg] = g
        else:
            # cluster file present but missing this regime -> pending for it only
            print(f"[recovery] WARNING: recovery_results.json has no gate for "
                  f"regime '{reg}' -> PENDING (descriptive) for it.")
            rec[reg] = _pending_recovery([reg])[reg]
    print(f"[recovery] ingested {path}")
    for reg in regimes:
        pdt = rec[reg].get("per_dial_trust", {})
        line = "  ".join(f"{d}={pdt.get(d, '?')}" for d in _GATE_DIALS)
        print(f"           {reg:7s}: {line}")
    return rec, "cluster"


# ════════════════════════════════════════════════════════════════════════════
# Per-session geometry: mu (change-time anchor) + regime (expert/naive)
# ════════════════════════════════════════════════════════════════════════════
def compute_session_geometry(anchors_chrono):
    """For every anchor session compute its mu (change_time_anchor on reached
    trials) and its regime (expert/naive). Returns dicts keyed by BOTH the
    canonical 8-digit form (for the library) and the CSV int form (for the
    appender) — they coincide except for the leading zero.

    Loads each session ONCE, builds the Phase-1 trial table (for mu + d'), then
    ``del sess; gc.collect()``.
    """
    mu_by_session = {}          # CSV-key -> mu
    dprime_by_session = {}      # canonical-key -> d'  (for comprehension flags)
    evidence_by_session = {}    # CSV-key -> build_trial_evidence_corrected DataFrame
    for sname in anchors_chrono:
        sess = load_session(sname)
        try:
            labels = dl.load_state_labels(sname)
            trial_table = dl.build_trial_table(sess, labels, sname)
            mu = dl.change_time_anchor(trial_table)
            dprime_by_session[sname] = dl.session_dprime(sess)
            ev_df = dl.build_trial_evidence_corrected(sess, dt=0.05)
            mu_by_session[_csv_key(sname)] = mu
            evidence_by_session[_csv_key(sname)] = ev_df
        finally:
            del sess
            gc.collect()

    # comprehension flags (rule='dprime', low knows-the-rule bar): chronological,
    # latch-on. Then regime = post-comprehension AND d' > EXPERT_DPRIME.
    comp_flags = dl.assign_comprehension_flags(
        dprime_by_session, threshold=COMPREHENSION_THRESHOLD, rule="dprime")
    regime_by_session = {}
    for sname in anchors_chrono:
        post = comp_flags.get(sname) == "post"
        expert = post and (dprime_by_session.get(sname, float("nan")) > EXPERT_DPRIME)
        regime_by_session[_csv_key(sname)] = "expert" if expert else "naive"
    return mu_by_session, dprime_by_session, regime_by_session, evidence_by_session


# ════════════════════════════════════════════════════════════════════════════
# Fallback: append the Phase-1 proxies as 'descriptive' latents (no generative fit)
# ════════════════════════════════════════════════════════════════════════════
def write_fallback_table(out_csv):
    """Contingency-gate FALLBACK (mode=='fallback'): ship the Phase-1 proxies as the
    latent table with every dial ``latent_trust='descriptive'`` and
    ``generative_omitted=True`` (no generative fit was run). The 25 Phase-1 columns
    are preserved verbatim; only honest provenance columns are appended."""
    df = pd.read_csv(DELIVERABLE_CSV)
    for col in ("sharpness_drift", "itchiness_caution", "timing_urgency_at_decision",
                "evidence_integral_at_decision", "expected_change_time",
                "lick_minus_expected"):
        df[col] = np.nan
    df["anchor_id"] = None
    df["rectification_kind"] = None
    df["leak_tau"] = np.nan
    df["recovery_regime"] = None
    df["trust_sharpness"] = "descriptive"
    df["trust_caution"] = "descriptive"
    df["trust_timing"] = "descriptive"
    df["generative_omitted"] = True
    df["latent_trust"] = "descriptive"
    df = canonicalize_session_column(df)
    df.to_csv(out_csv, index=False)
    print(f"[fallback] wrote Phase-1 proxies (descriptive) -> {out_csv}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# Stats CSV + results JSON (the ladder winners are the headline)
# ════════════════════════════════════════════════════════════════════════════
def build_stats_rows(anchor_fits, learn, state_ladders, regime_by_session,
                     recovery_by_regime, rectification):
    """Per-anchor stats rows: dial values per mood, regime, state-ladder winner."""
    rows = []
    for sname, fit in anchor_fits.items():
        regime = regime_by_session.get(_csv_key(sname), "naive")
        sl = state_ladders.get(sname, {})
        pdt = recovery_by_regime.get(regime, {}).get("per_dial_trust", {})
        for mood, dials in (fit.dials or {}).items():
            rows.append({
                "session": sname,
                "regime": regime,
                "mood": mood,
                "sharpness_v": dials.get("sharpness"),
                "itchiness_z": dials.get("itchiness"),
                "timing_u": dials.get("timing"),
                "state_ladder_winner": sl.get("winner"),
                "rectification": rectification,
                "trust_sharpness": pdt.get("sharpness", "descriptive"),
                "trust_caution": pdt.get("caution", "descriptive"),
                "trust_timing": pdt.get("timing", "descriptive"),
            })
    return pd.DataFrame(rows)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="B8 Phase-2 generative decision-latents orchestration.")
    p.add_argument("--quick", action="store_true",
                   help="SMOKE: 2-3 anchors, AIC-only ladders, tiny; writes *_smoke "
                        "paths and NEVER overwrites the real deliverable.")
    p.add_argument("--force", action="store_true",
                   help="recompute even if a cached results JSON exists.")
    p.add_argument("--figures", action="store_true",
                   help="FIGURE-ONLY mode (Task 4.4): render F6/F7/F8 from the cached "
                        "outputs (recovery_results.json + the deliverable + stats CSVs) "
                        "and the Engine-C spot-check. Runs NO recovery / orchestration "
                        "science; the only compute is the cached Engine-C pyddm fit.")
    p.add_argument("--force-enginec", action="store_true",
                   help="with --figures: re-run the Engine-C pyddm spot-check even if "
                        "enginec_spotcheck.csv is cached.")
    p.add_argument("--l2", type=float, default=1.0,
                   help="ridge strength toward the more-expert neighbour in the "
                        "backward sweep (default 1.0).")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="process-parallel workers for the (CPU-bound) ladders: the "
                        "per-anchor STATE ladders and the LEARNING ladder's rung x "
                        "restart fits (default cpu_count-2). Session LOADING and the "
                        "backward sweep stay SEQUENTIAL (gateway + true dependency). "
                        "Results are byte-identical regardless of --workers (same seeds).")
    p.add_argument("--with-cvll", action="store_true",
                   help="ALSO compute the k-fold cross-validated LL on BOTH ladders "
                        "(SLOW: k refits per rung, single-threaded over ~30 anchors "
                        "-> hours). The ladder winner is argmin AIC and does NOT "
                        "depend on CV-LL, so by DEFAULT we skip it (AIC + BIC only, "
                        "~minutes). Use this only for the CV robustness supplement "
                        "(better run parallelized / on the cluster).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ── Task 4.4 FIGURE-ONLY mode: render F6/F7/F8 from cached outputs. NO recovery
    # / orchestration science is run here (the only compute is the cached Engine-C
    # pyddm spot-check). Decoupled so figures can be regenerated without the ~5h run.
    if args.figures:
        if not os.path.exists(RECOVERY_JSON):
            raise SystemExit(f"FATAL: {RECOVERY_JSON} absent — F6 needs the corrected "
                             "recovery verdict. (Run the cluster harness first.)")
        if not os.path.exists(DELIVERABLE_CSV) or not os.path.exists(STATS_CSV):
            raise SystemExit("FATAL: deliverable / stats CSV absent — F7/F8 need the "
                             "FULL orchestration outputs.")
        make_task44_figures(force_enginec=args.force_enginec)
        return 0

    suffix = "_smoke" if args.quick else ""
    out_csv = os.path.join(CACHE_DIR, f"decision_latents_by_state{suffix}.csv")
    results_json = os.path.join(CACHE_DIR, f"decision_latents_phase2_results{suffix}.json")
    stats_csv = os.path.join(FIG_DIR, f"decision_latents_phase2_stats{suffix}.csv")

    print("=" * 72)
    print(f"B8 Phase 2 orchestration  ({'QUICK SMOKE' if args.quick else 'FULL'})")
    print(f"  visdetect: {dlg.__file__}")
    print(f"  l2={args.l2}  sigma={SIGMA}  out_csv={out_csv}")
    print("=" * 72, flush=True)

    # ── cache skip (honour --force): a prior FULL run is reused unless --force.
    # Re-run with --force after the cluster recovery_results.json lands to flip the
    # per-dial trust columns. --quick always recomputes (writes throwaway _smoke).
    if not args.quick and not args.force \
            and os.path.exists(results_json) and os.path.exists(out_csv):
        print(f"[cache] results already exist:\n    {results_json}\n    {out_csv}\n"
              "  Use --force to recompute (e.g. after the cluster JSON lands). Skipping.")
        return 0

    # ── 1. inventory -> the contingency GATE ─────────────────────────────────
    if not os.path.exists(INVENTORY_CSV):
        raise SystemExit(f"FATAL: inventory not found: {INVENTORY_CSV} "
                         "(run _expert_anchor_inventory.py first).")
    inv = pd.read_csv(INVENTORY_CSV)
    sel = dlg.select_expert_anchors(inv)
    # canonical zfill8 form at the SOURCE so every downstream key (anchors_chrono,
    # anchor_designs, anchor_fits, mu_by_session_canon) is the leading-zero form,
    # never the int-form '1072025' (which is just int('01072025')).
    anchors = [dlg.canonical_session_id(a) for a in sel["anchors"]]
    mode = sel["mode"]
    print(f"[gate] mode={mode!r}  n_anchors={len(anchors)}")

    # chronological order (oldest -> newest); the sweep walks this in reverse.
    from visdetect.analysis.config import parse_session_date
    anchors_chrono = sorted(anchors, key=parse_session_date)

    # ── fallback branch: ship Phase-1 proxies, STOP (no generative fit) ──────
    if mode == "fallback":
        print("[gate] FALLBACK: <3 adequate expert anchors even after pooling -> "
              "shipping Phase-1 descriptive proxies; NO generative fit.")
        write_fallback_table(out_csv)
        with open(results_json, "w", encoding="utf-8") as fh:
            json.dump({"mode": "fallback", "anchors": anchors,
                       "note": "Phase-1 proxies shipped; latent_trust=descriptive"},
                      fh, indent=2)
        print(f"[done] fallback results -> {results_json}")
        return 0

    # ── quick smoke: keep only the most-expert anchors (newest), AIC-only ────
    if args.quick:
        anchors_chrono = anchors_chrono[-QUICK_N_ANCHORS:]
        print(f"[quick] reduced to {len(anchors_chrono)} most-expert anchors: "
              f"{anchors_chrono}")
    # Default AIC-only on BOTH ladders (winner = argmin AIC, contract-locked; AIC +
    # BIC still computed). CV-LL is slow + single-threaded over ~30 anchors -> opt-in
    # via --with-cvll (and better run parallelized / on the cluster).
    compute_cvll = bool(args.with_cvll)
    n_restarts = QUICK_N_RESTARTS if args.quick else 4

    # ── 2. per-session geometry: mu + regime + evidence ─────────────────────
    print("[geometry] computing mu (change-time anchor) + regime per anchor ...",
          flush=True)
    (mu_by_session, dprime_by_session, regime_by_session,
     evidence_by_session) = compute_session_geometry(anchors_chrono)
    regimes_present = sorted(set(regime_by_session.values()))
    print(f"[geometry] regimes present: {regimes_present}")
    for sname in anchors_chrono:
        ck = _csv_key(sname)
        print(f"           {sname}: mu={mu_by_session[ck]:.3f}s  "
              f"d'={dprime_by_session.get(sname, float('nan')):.2f}  "
              f"regime={regime_by_session[ck]}")

    # ── 3. rectification: select on the MOST-EXPERT anchor (last chrono) ─────
    expert_anchor = anchors_chrono[-1]
    print(f"[rectification] selecting on most-expert anchor {expert_anchor} ...",
          flush=True)
    param_spec_default = dlg.ParamSpec()
    expert_ev = evidence_by_session[_csv_key(expert_anchor)]
    expert_labels = dl.load_state_labels(expert_anchor)
    expert_labels = expert_labels[expert_labels["state_label"].isin(dl.MAIN_MOODS)]
    rect_k = 3 if args.quick else 5
    rect_res = dlg.select_rectification(
        dlg.build_design, expert_ev, expert_labels,
        mu_by_session[_csv_key(expert_anchor)], SIGMA, k=rect_k)
    rectification = rect_res["winner"]
    print(f"[rectification] winner={rectification!r}  scores="
          f"{ {k: round(v, 1) for k, v in rect_res['scores'].items()} }")
    param_spec = dlg.ParamSpec(rectification=rectification)

    # mu_by_session keyed by canonical form for build_anchor_designs (it looks up
    # mu_by_session[sname] with the canonical session name it iterates).
    mu_by_session_canon = {sname: mu_by_session[_csv_key(sname)]
                           for sname in anchors_chrono}

    # ── 4. build anchor Designs (loads sessions; QC-gates to usable cells) ───
    print(f"[designs] building anchor Designs for {len(anchors_chrono)} anchors "
          f"(rectification={rectification}) ...", flush=True)
    anchor_designs = dlg.build_anchor_designs(
        anchors_chrono, param_spec, mu_by_session_canon, SIGMA,
        rectification=rectification)
    fitted_keys = list(anchor_designs.keys())
    print(f"[designs] {len(fitted_keys)} sessions produced a usable Design: "
          f"{fitted_keys}  (trials/anchor: "
          f"{ {k: len(d) for k, d in anchor_designs.items()} })")
    if len(anchor_designs) < 2:
        raise SystemExit("FATAL: <2 usable anchor Designs -> cannot run the "
                         "learning ladder. (Try more anchors / check QC.)")

    # smoke: subsample each Design so the ladders fit in a few minutes (the FULL
    # run uses every trial). This is a tractability lever ONLY — not the science.
    if args.quick:
        anchor_designs = _quick_subsample(anchor_designs, QUICK_N_TRIALS)
        print(f"[quick] subsampled Designs to <= {QUICK_N_TRIALS} trials/anchor: "
              f"{ {k: len(d) for k, d in anchor_designs.items()} }")

    # ── 5. backward sweep + the two ladders (THE SCIENCE) ────────────────────
    print(f"[sweep] backward sweep (expert-first, l2={args.l2}) ...", flush=True)
    anchor_fits = dlg.backward_sweep(
        anchor_designs, anchors_chrono, param_spec, l2=args.l2)
    print(f"[sweep] fit {len(anchor_fits)} anchors.")

    n_workers = max(1, int(args.workers))
    print(f"[ladder] learning ladder (which dial moves with learning; "
          f"compute_cvll={compute_cvll}; n_workers={n_workers}) ...", flush=True)
    learn = dlg.learning_ladder(anchor_designs, param_spec,
                                compute_cvll=compute_cvll, n_restarts=n_restarts,
                                n_workers=n_workers)
    print(f"[ladder] LEARNING winner = {learn['winner']}")
    print("         AIC: " + "  ".join(f"{k}={v:.1f}" for k, v in learn["aic"].items()))

    # ── state ladders: one INDEPENDENT job per anchor (the biggest parallel win) ──
    # Collect BY KEY (sname), never by arrival order, so the dict is deterministic.
    print(f"[ladder] state ladders over {len(anchor_designs)} anchors "
          f"(n_workers={n_workers}) ...", flush=True)
    state_ladders = {}
    sl_tasks = [(sname, design, param_spec, n_restarts, compute_cvll, 0)
                for sname, design in anchor_designs.items()]
    if n_workers <= 1 or len(sl_tasks) <= 1:
        for t in sl_tasks:
            sname, sl = _state_ladder_worker(t)
            state_ladders[sname] = sl
    else:
        ctx = multiprocessing.get_context("spawn")  # Windows-safe
        with ProcessPoolExecutor(max_workers=min(n_workers, len(sl_tasks)),
                                 mp_context=ctx) as ex:
            for sname, sl in ex.map(_state_ladder_worker, sl_tasks):
                state_ladders[sname] = sl
    # report the per-anchor state-ladder winners + the modal winner
    sl_winners = [sl["winner"] for sl in state_ladders.values()]
    from collections import Counter
    modal = Counter(sl_winners).most_common(1)[0] if sl_winners else (None, 0)
    print(f"[ladder] STATE winners per anchor: "
          f"{dict(Counter(sl_winners))}  (modal={modal[0]}, n={modal[1]})")

    # ── 6/7. recovery verdict (ingest cluster JSON or PENDING) -> trust ─────
    recovery_by_regime, rec_source = load_recovery_by_regime(regimes_present)

    # ── 8. append generative latents to the deliverable ─────────────────────
    print(f"[append] appending generative latents -> {out_csv} "
          f"({'SMOKE — not the real deliverable' if args.quick else 'REAL deliverable'})",
          flush=True)
    # anchor_fits is keyed by the canonical session form (anchors_chrono / zfill8);
    # the appender canonicalizes BOTH its CSV session_name column (int64-stored) and
    # the dicts to zfill8, so the keys match regardless of representation.
    appended = dlg.append_generative_latents(
        DELIVERABLE_CSV, anchor_fits, recovery_by_regime, param_spec,
        mu_by_session, evidence_by_session, regime_by_session, SIGMA,
        rectification=rectification)
    # regression guard (would have caught the leading-zero-day key bug): every fitted
    # anchor that actually appears in the deliverable MUST have non-omitted rows.
    # Both sides canonicalized so int64 / int-form / zfill8 all compare equal. Fails
    # loudly rather than silently shipping a corrupt latent table.
    csv_sess = appended["session_name"].map(dlg.canonical_session_id)
    fitted_canon = {dlg.canonical_session_id(k) for k in anchor_fits}
    omitted_fits = sorted(
        s for s in (fitted_canon & set(csv_sess))
        if bool(appended.loc[csv_sess == s, "generative_omitted"].all()))
    if omitted_fits:
        raise SystemExit(
            "FATAL: fitted anchors written as generative_omitted (session-key "
            f"mismatch between anchor_fits and the deliverable): {omitted_fits}")
    appended = canonicalize_session_column(appended)
    appended.to_csv(out_csv, index=False)
    n_gen = int((~appended["generative_omitted"]).sum())
    print(f"[append] wrote {len(appended)} rows ({n_gen} with a fitted anchor); "
          f"{len(appended.columns)} cols.")

    # ── 9. stats CSV + results JSON + summary ───────────────────────────────
    stats = build_stats_rows(anchor_fits, learn, state_ladders,
                             regime_by_session, recovery_by_regime, rectification)
    stats.to_csv(stats_csv, index=False)
    print(f"[stats] wrote {stats_csv}")

    results = {
        "mode": mode,
        "quick": args.quick,
        "anchors_chrono": anchors_chrono,
        "fitted_anchors": fitted_keys,
        "rectification": rectification,
        "rectification_scores": rect_res["scores"],
        "l2": args.l2,
        "sigma": SIGMA,
        "regime_by_session": regime_by_session,
        "mu_by_session": mu_by_session,
        "learning_ladder": {"winner": learn["winner"], "aic": learn["aic"],
                            "bic": learn["bic"], "cvll": learn["cvll"]},
        "state_ladder_winners": {s: sl["winner"] for s, sl in state_ladders.items()},
        "state_ladder_modal_winner": modal[0],
        "recovery_source": rec_source,
        "recovery_by_regime": recovery_by_regime,
    }
    with open(results_json, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2)
    print(f"[json] wrote {results_json}")

    # ── headline summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("HEADLINE — the ladder winners")
    print("=" * 72)
    print(f"  LEARNING (which dial learning turns): {learn['winner']}")
    print(f"  STATE    (which dial mood loads on) : modal {modal[0]} "
          f"({modal[1]}/{len(state_ladders)} anchors)")
    print(f"  rectification: {rectification}   regimes: {regimes_present}")
    print(f"  recovery: {rec_source}"
          + ("  (PENDING — all dials descriptive until the cluster lands)"
             if rec_source == "pending" else ""))
    if args.quick:
        print("  *** QUICK SMOKE — proves the pipeline on 2-3 anchors; "
              "NOT the published science. ***")
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
