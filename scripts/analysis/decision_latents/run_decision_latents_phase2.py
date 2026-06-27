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
from visdetect.analysis.config import ROOT, SUBJECT
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
