"""Fig43: Optotagging — antidromic identification of D1 / D2 SPNs.

Protocol per session (post-task):
  Block 1 → 501 laser pulses to GPe  → D2-SPN tagging (indirect pathway)
  Block 2 → 501 laser pulses to SNr  → D1-SPN tagging (direct pathway)

Method: two-tier antidromic classification (see optotagging.py). Per (unit, fiber)
metrics — baseline-corrected excess reliability/jitter, canonical SALT, Poisson
excess test, and an offline collision test — feed `fiber_tier` (none / candidate /
high_confidence); `classify_unit` then combines GPe+SNr into a D1/D2 `UnitTag` via
bridging-collateral logic. The candidate tier is sensitive; the high-confidence tier
is collision-confirmed antidromic. (The legacy latency<8 / jitter<3.5 / reliability>=0.1
thresholds survive only to reconstruct the "old pipeline" comparison bar in fig43c.)

Produces:
  fig43a_optotagging_distributions.png
    Histograms of peak latency and excess jitter for candidate/high-confidence
    tagged units, split by fiber (GPe = D2, SNr = D1).

  fig43b_yield_by_stage_tier.png
    Grouped bar chart of D1 / D2 candidate vs high-confidence unit counts
    per learning stage (Learning / Expert).

  fig43c_old_vs_new_and_sweep.png
    Left: old-pipeline vs new-candidate vs new-high-confidence yield bars.
    Right: yield as a function of strict jitter-cap threshold sweep.

  Caches:
    cache/optotagging_results.csv   (one row per unit × fiber)
    cache/optotagging_unit_tags.csv (one row per unique unit: pathway + tier)
"""

import argparse
import os
import gc


import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis.config import STAGE_ORDER, canonicalize_session_column
from visdetect.suite.loader import load_staging_manifest, load_session, load_waveform_labels
from visdetect.analysis.utils import get_good_cluster_ids
from visdetect.suite.plotting import setup_style

# Transported out of analysis_suite (2026-07-01): outputs now live in the topic dirs.
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
CACHE_DIR = str(_ROOT / "data" / "cache" / "optotagging")
_FIG_DIR = _ROOT / "FIGURES" / "optotagging" / "BG_046"


def save_figure(fig, name, module_name=None, formats=("png",)):
    """Write to FIGURES/optotagging/BG_046 (replaces the analysis_suite figures path)."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(_FIG_DIR / f"{name}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)

from visdetect.analysis.optotagging import (
    OptoTagger, OptoMetrics,
    SALT_ALPHA, MIN_RELIABILITY, MAX_LATENCY_MS, MAX_JITTER_MS,
    STRICT_SALT_ALPHA, STRICT_MAX_JITTER_MS, CANDIDATE_MIN_EXCESS_REL,
    fiber_tier, classify_unit, is_spn_plausible_waveform,
)

setup_style()

MODULE_NAME = "09_optotagging"

# ── Fiber colors ──────────────────────────────────────────────────────
FIBER_COLORS = {
    "GPe": "#9b59b6",   # purple — D2 pathway
    "SNr": "#e67e22",   # orange — D1 pathway
}


# ── Helper: build per-session results ─────────────────────────────────
def _run_session(sname, stage, salt_n_jitter):
    """Load session, run OptoTagger, return list of result dicts."""
    try:
        sess = load_session(sname)
    except (FileNotFoundError, Exception) as exc:
        return sname, stage, [], f"SKIP: {exc}"

    ni = getattr(sess, "ni_events", {}) or {}
    laser_keys = [k for k in ni if "laser" in k.lower() or "opto" in k.lower()]
    if not laser_keys:
        del sess
        gc.collect()
        return sname, stage, [], "no laser data"

    try:
        tagger = OptoTagger(sess, salt_n_jitter=salt_n_jitter)
    except ValueError as exc:
        del sess
        gc.collect()
        return sname, stage, [], str(exc)

    good_ids = get_good_cluster_ids(sess)
    results = tagger.analyze_all(cluster_ids=good_ids)

    rows = []
    for m in results:
        rows.append({
            "session_name": int(sname), "stage": stage,
            "cluster_id": m.cluster_id, "fiber": m.fiber, "n_pulses": m.n_pulses,
            # legacy raw
            "latency_ms": m.latency_ms, "jitter_ms": m.jitter_ms,
            "reliability": m.reliability, "salt_p": m.salt_p,
            # enriched
            "baseline_rate_hz": m.baseline_rate_hz,
            "win_lo": m.response_window_ms[0], "win_hi": m.response_window_ms[1],
            "peak_latency_ms": m.peak_latency_ms,
            "excess_reliability": m.excess_reliability,
            "excess_jitter_ms": m.excess_jitter_ms,
            "poisson_p": m.poisson_p,
            "collision_status": m.collision_status,
            "collision_suppression_index": m.collision_suppression_index,
            "n_collision_free": m.n_collision_free,
            "n_collision_expected": m.n_collision_expected,
        })

    n_candidate = sum(1 for r in rows if r["salt_p"] < 0.05)
    del sess
    gc.collect()
    return sname, stage, rows, f"{len(rows)} units, {n_candidate} candidate"


def _process_session_worker(args):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    sname, stage, salt_n_jitter = args
    return _run_session(sname, stage, salt_n_jitter)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Optotagging analysis")
    parser.add_argument("--n-jitter", type=int, default=500,
                        help="Number of SALT jitter iterations (default 500)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel worker processes (default: 1 = sequential). "
                             "Each worker loads and processes one session independently.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cache exists")
    args = parser.parse_args()

    print("=" * 70)
    print("[09a] Optotagging — Antidromic D1/D2 identification (SALT test)")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  QC-passed sessions: {len(manifest)}")

    cache_path = os.path.join(CACHE_DIR, "optotagging_results.csv")

    # ── Run or load cache ─────────────────────────────────────────────
    if os.path.exists(cache_path) and not args.force:
        print(f"  Loading cached results from {cache_path}")
        df_all = pd.read_csv(cache_path)
        _required = {"win_lo", "win_hi", "collision_status", "excess_jitter_ms",
                     "peak_latency_ms", "excess_reliability", "poisson_p"}
        missing = _required - set(df_all.columns)
        if missing:
            raise SystemExit(
                f"  Cached {cache_path} is from an old schema (missing {sorted(missing)}). "
                "Re-run with --force to rebuild.")
    else:
        print(f"  Running SALT test (n_jitter={args.n_jitter}) ...")
        tasks = [
            (int(row["session_name"]), row["stage"], args.n_jitter)
            for _, row in manifest.iterrows()
        ]

        all_rows = []
        if args.n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            print(f"  Using {args.n_workers} parallel workers")
            with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
                for sname, stage, rows, msg in ex.map(_process_session_worker, tasks):
                    print(f"    {sname} ({stage}): {msg}")
                    all_rows.extend(rows)
        else:
            for idx, (sname, stage, salt_n) in enumerate(tasks):
                print(f"  [{idx+1}/{len(tasks)}] {sname} ({stage})")
                _, _, rows, msg = _run_session(sname, stage, salt_n)
                print(f"    {msg}")
                all_rows.extend(rows)

        if not all_rows:
            print("  No optotagging data found in any session!")
            return

        df_all = pd.DataFrame(all_rows)
        df_all = canonicalize_session_column(df_all)
        df_all.to_csv(cache_path, index=False)
        print(f"  Saved {len(df_all)} rows to {cache_path}")

    # ── waveform labels (optional) ─────────────────────────────────────
    try:
        wf = load_waveform_labels()
        wf_map = {(int(r.session_name), int(r.cluster_id)): r.cell_type
                  for r in wf.itertuples()}
    except FileNotFoundError:
        print("  Waveform labels not found - skipping FSI cross-check (annotation only).")
        wf_map = {}

    def _metrics_from_row(r):
        return OptoMetrics(
            cluster_id=int(r.cluster_id), fiber=r.fiber, is_responsive=False,
            latency_ms=r.latency_ms, jitter_ms=r.jitter_ms, reliability=r.reliability,
            salt_p=r.salt_p, n_pulses=int(r.n_pulses),
            baseline_rate_hz=r.baseline_rate_hz, response_window_ms=(r.win_lo, r.win_hi),
            peak_latency_ms=r.peak_latency_ms, excess_reliability=r.excess_reliability,
            excess_jitter_ms=r.excess_jitter_ms, poisson_p=r.poisson_p,
            collision_status=r.collision_status,
            collision_suppression_index=r.collision_suppression_index,
            n_collision_free=int(r.n_collision_free),
            n_collision_expected=int(r.n_collision_expected))

    # per-fiber tier (waveform applied per unit below)
    df_all["fiber_tier"] = [fiber_tier(_metrics_from_row(r)) for r in df_all.itertuples()]

    # ── unit-level classification (bridging logic + waveform gate) ─────
    unit_rows = []
    for (sn, cid), grp in df_all.groupby(["session_name", "cluster_id"]):
        g = grp[grp.fiber == "GPe"]
        s = grp[grp.fiber == "SNr"]
        gm = _metrics_from_row(next(g.itertuples())) if len(g) else None
        sm = _metrics_from_row(next(s.itertuples())) if len(s) else None
        cell_type = wf_map.get((int(sn), int(cid)))
        wf_ok = is_spn_plausible_waveform(cell_type)
        tag = classify_unit(gm, sm, waveform_ok=wf_ok)
        unit_rows.append({
            "session_name": int(sn), "cluster_id": int(cid),
            "stage": grp.iloc[0]["stage"], "pathway": tag.pathway, "tier": tag.tier,
            "gpe_tier": tag.gpe_tier, "snr_tier": tag.snr_tier,
            "contributing_fiber": tag.contributing_fiber,
            "cell_type": cell_type, "waveform_ok": wf_ok,
        })
    units = pd.DataFrame(unit_rows)
    units_path = os.path.join(CACHE_DIR, "optotagging_unit_tags.csv")
    units = canonicalize_session_column(units)
    units.to_csv(units_path, index=False)
    print(f"  Saved unit tags to {units_path}")

    # ── two-tier yield summary ─────────────────────────────────────────
    print("\n  === Yield by tier x pathway (unique units) ===")
    for pathway in ["D1", "D2"]:
        sub = units[units.pathway == pathway]
        n_cand = (sub.tier.isin(["candidate", "high_confidence"])).sum()
        n_hc = (sub.tier == "high_confidence").sum()
        print(f"    {pathway}: candidate={n_cand}  high_confidence={n_hc}")
    n_untestable = (df_all.collision_status == "untestable").mean()
    print(f"    Collision-untestable fraction (all unit x fiber): {n_untestable:.2f}")

    print("\n  Generating figures ...")

    # Panel set 1: latency + excess-jitter distributions for tagged units
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    tagged_fibers = df_all[df_all.fiber_tier.isin(["candidate", "high_confidence"])]
    for fiber, color in FIBER_COLORS.items():
        sub = tagged_fibers[tagged_fibers.fiber == fiber]
        axes[0].hist(sub.peak_latency_ms.dropna(), bins=np.linspace(0, 10, 41),
                     alpha=0.6, color=color, label=f"{fiber} (n={len(sub)})")
        axes[1].hist(sub.excess_jitter_ms.dropna(), bins=np.linspace(0, 3, 31),
                     alpha=0.6, color=color, label=fiber)
    axes[0].set(xlabel="Peak latency (ms)", ylabel="Count", title="Tagged-unit latency")
    axes[1].axvline(STRICT_MAX_JITTER_MS, color="k", ls="--", lw=1, label="strict cap")
    axes[1].set(xlabel="Excess jitter (ms)", ylabel="Count", title="Tagged-unit jitter")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    save_figure(fig, "fig43a_optotagging_distributions", MODULE_NAME)

    # Panel set 2: yield by stage x tier x pathway
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    stages = [s for s in STAGE_ORDER if s in units.stage.values]
    x = np.arange(len(stages)); bw = 0.35
    for ax, pathway in zip(axes2, ["D1", "D2"]):
        for k, (tier, alpha) in enumerate([("candidate", 0.5), ("high_confidence", 1.0)]):
            if tier == "high_confidence":
                counts = [((units.stage == st) & (units.pathway == pathway)
                           & (units.tier == "high_confidence")).sum() for st in stages]
            else:
                counts = [((units.stage == st) & (units.pathway == pathway)
                           & (units.tier.isin(["candidate", "high_confidence"]))).sum()
                          for st in stages]
            ax.bar(x + (k - 0.5) * bw, counts, bw, alpha=alpha,
                   color=FIBER_COLORS["SNr" if pathway == "D1" else "GPe"], label=tier)
        ax.set(xticks=x, title=f"{pathway} yield by stage", xlabel="Stage")
        ax.set_xticklabels(stages); ax.legend(fontsize=8)
    axes2[0].set_ylabel("Tagged units")
    save_figure(fig2, "fig43b_yield_by_stage_tier", MODULE_NAME)

    # Panel set 3: old-vs-new comparison + jitter-threshold sweep
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
    old = {"D2": int(((df_all.fiber == "GPe") & (df_all.salt_p < SALT_ALPHA)
                      & (df_all.latency_ms < MAX_LATENCY_MS)
                      & (df_all.jitter_ms < MAX_JITTER_MS)
                      & (df_all.reliability >= MIN_RELIABILITY)).sum()),
           "D1": int(((df_all.fiber == "SNr") & (df_all.salt_p < SALT_ALPHA)
                      & (df_all.latency_ms < MAX_LATENCY_MS)
                      & (df_all.jitter_ms < MAX_JITTER_MS)
                      & (df_all.reliability >= MIN_RELIABILITY)).sum())}
    new_cand = {p: int((units.pathway == p).sum()) for p in ["D1", "D2"]}
    new_hc = {p: int(((units.pathway == p) & (units.tier == "high_confidence")).sum())
              for p in ["D1", "D2"]}
    xp = np.arange(2)
    axes3[0].bar(xp - 0.25, [old["D1"], old["D2"]], 0.25, label="old pipeline", color="#888")
    axes3[0].bar(xp, [new_cand["D1"], new_cand["D2"]], 0.25, label="new candidate", color="#5dade2")
    axes3[0].bar(xp + 0.25, [new_hc["D1"], new_hc["D2"]], 0.25, label="new high-conf", color="#1f618d")
    axes3[0].set(xticks=xp, title="Old vs new yield", ylabel="Units")
    axes3[0].set_xticklabels(["D1", "D2"]); axes3[0].legend(fontsize=8)

    jit_grid = np.linspace(0.25, 3.0, 12)
    for pathway, fiber in [("D1", "SNr"), ("D2", "GPe")]:
        fib = df_all[df_all.fiber == fiber]
        ys = [int(((fib.salt_p < STRICT_SALT_ALPHA) & (fib.collision_status == "pass")
                   & (fib.excess_jitter_ms < j) & (fib.excess_reliability > CANDIDATE_MIN_EXCESS_REL)).sum())
              for j in jit_grid]
        axes3[1].plot(jit_grid, ys, "-o", color=FIBER_COLORS[fiber], label=pathway)
    axes3[1].axvline(STRICT_MAX_JITTER_MS, color="k", ls="--", lw=1)
    axes3[1].set(xlabel="Strict jitter cap (ms)", ylabel="High-conf units",
                 title="Yield vs jitter threshold"); axes3[1].legend(fontsize=8)
    save_figure(fig3, "fig43c_old_vs_new_and_sweep", MODULE_NAME)

    print("\n[09a] Done.")


if __name__ == "__main__":
    main()
