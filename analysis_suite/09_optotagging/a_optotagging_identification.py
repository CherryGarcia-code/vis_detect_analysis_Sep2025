"""Fig43: Optotagging — antidromic identification of D1 / D2 SPNs.

Protocol per session (post-task):
  Block 1 → 501 laser pulses to GPe  → D2-SPN tagging (indirect pathway)
  Block 2 → 501 laser pulses to SNr  → D1-SPN tagging (direct pathway)

Statistical method: SALT test (Stimulus-Associated spike Latency Test,
Kvitsiani et al. 2013) with Jensen–Shannon divergence.  Additional
criteria: latency < 8 ms, jitter < 3.5 ms, reliability >= 0.1.

Produces:
  fig43a_optotagging_overview.png
    Panel A: Example raster + PSTH for a tagged D1 and D2 unit
    Panel B: First-spike latency distributions (GPe vs SNr responsive)
    Panel C: Summary counts per session (stacked bar: tagged / total)
    Panel D: Tagged fractions by learning stage

  Caches: cache/optotagging_results.csv  (one row per unit × fiber)
"""

import argparse
import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import STAGE_ORDER, STAGE_COLORS, CACHE_DIR
from loader import load_staging_manifest, load_session, load_waveform_labels
from visdetect.analysis.utils import get_good_cluster_ids
from plotting import setup_style, save_figure

from visdetect.analysis.optotagging import (
    OptoTagger, OptoMetrics,
    SALT_ALPHA, RESPONSE_WINDOW_MS,
    MAX_LATENCY_MS, MAX_JITTER_MS, MIN_RELIABILITY,
    _first_spike_latencies,
)
from visdetect.analysis.align import align_spikes_to_events

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
            "session_name": int(sname),
            "stage": stage,
            "cluster_id": m.cluster_id,
            "fiber": m.fiber,
            "is_responsive": m.is_responsive,
            "latency_ms": m.latency_ms,
            "jitter_ms": m.jitter_ms,
            "reliability": m.reliability,
            "salt_p": m.salt_p,
            "n_pulses": m.n_pulses,
        })

    n_tagged = sum(1 for r in rows if r["is_responsive"])
    del sess
    gc.collect()
    return sname, stage, rows, f"{len(rows)} units, {n_tagged} tagged"


def _process_session_worker(args):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    sname, stage, salt_n_jitter = args
    return _run_session(sname, stage, salt_n_jitter)


# ── Plotting helpers ──────────────────────────────────────────────────
def _plot_raster_psth(ax_raster, ax_psth, spike_times, pulse_times, fiber,
                      window_s=(-0.01, 0.02), bin_ms=0.25):
    """Draw raster + PSTH for one unit vs one fiber's pulses."""
    win = window_s
    bin_s = bin_ms / 1000.0

    # Raster
    for i, p in enumerate(pulse_times):
        rel = spike_times - p
        in_win = rel[(rel >= win[0]) & (rel <= win[1])]
        ax_raster.vlines(in_win * 1000, i, i + 1, linewidth=0.3, color="k")
    ax_raster.set_ylabel("Pulse #")
    ax_raster.set_xlim(win[0] * 1000, win[1] * 1000)
    ax_raster.set_ylim(0, len(pulse_times))
    ax_raster.axvline(0, color="deepskyblue", linewidth=1, alpha=0.7, label="Laser")
    ax_raster.set_title(f"{fiber} stimulation", fontsize=10)

    # PSTH
    mat, bc = align_spikes_to_events(spike_times, list(pulse_times),
                                     window=win, bin_size=bin_s)
    mean_fr = mat.mean(axis=0) if mat.shape[0] > 0 else np.zeros_like(bc)
    ax_psth.bar(bc * 1000, mean_fr, width=bin_ms, color=FIBER_COLORS[fiber],
                alpha=0.8, edgecolor="none")
    ax_psth.axvline(0, color="deepskyblue", linewidth=1, alpha=0.7)
    ax_psth.set_xlabel("Time from pulse (ms)")
    ax_psth.set_ylabel("FR (Hz)")
    ax_psth.set_xlim(win[0] * 1000, win[1] * 1000)


def _plot_latency_distributions(ax, df_resp):
    """Histogram of first-spike latencies for GPe vs SNr responsive units."""
    bins = np.linspace(0, 10, 41)
    for fiber, color in FIBER_COLORS.items():
        sub = df_resp[df_resp["fiber"] == fiber]
        if sub.empty:
            continue
        ax.hist(sub["latency_ms"].dropna(), bins=bins, alpha=0.6,
                color=color, label=f"{fiber} (n={len(sub)})", edgecolor="white")
    ax.set_xlabel("First-spike latency (ms)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.set_title("Latency of tagged units")


def _plot_session_counts(ax, df_all, manifest):
    """Stacked bar: tagged vs untagged units per session."""
    sessions = manifest["session_name"].astype(int).values
    has_laser = df_all["session_name"].unique()

    x_labels, n_tagged_gpe, n_tagged_snr, n_total = [], [], [], []
    for i, sn in enumerate(sessions):
        sub = df_all[df_all["session_name"] == sn]
        if sub.empty:
            continue
        n_units = sub[sub["fiber"] == "GPe"]["cluster_id"].nunique()
        if n_units == 0:
            n_units = sub[sub["fiber"] == "SNr"]["cluster_id"].nunique()
        n_gpe = sub[(sub["fiber"] == "GPe") & sub["is_responsive"]].shape[0]
        n_snr = sub[(sub["fiber"] == "SNr") & sub["is_responsive"]].shape[0]
        x_labels.append(str(sn))
        n_tagged_gpe.append(n_gpe)
        n_tagged_snr.append(n_snr)
        n_total.append(n_units)

    x = np.arange(len(x_labels))
    n_untagged = [t - g - s for t, g, s in zip(n_total, n_tagged_gpe, n_tagged_snr)]
    ax.bar(x, n_tagged_gpe, color=FIBER_COLORS["GPe"], label="D2-tagged (GPe)")
    ax.bar(x, n_tagged_snr, bottom=n_tagged_gpe, color=FIBER_COLORS["SNr"],
           label="D1-tagged (SNr)")
    ax.bar(x, n_untagged,
           bottom=[g + s for g, s in zip(n_tagged_gpe, n_tagged_snr)],
           color="#cccccc", label="Untagged")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
    ax.set_ylabel("Units")
    ax.set_title("Tagged units per session")
    ax.legend(fontsize=7, loc="upper right")


def _plot_stage_fractions(ax, df_all):
    """Fraction of tagged units by stage for each fiber."""
    bar_width = 0.35
    stages_present = [s for s in STAGE_ORDER if s in df_all["stage"].values]
    x = np.arange(len(stages_present))

    for fi, (fiber, color) in enumerate(FIBER_COLORS.items()):
        fracs, ns = [], []
        for stage in stages_present:
            sub = df_all[(df_all["stage"] == stage) & (df_all["fiber"] == fiber)]
            n = len(sub)
            n_resp = sub["is_responsive"].sum()
            fracs.append(n_resp / n if n > 0 else 0)
            ns.append(n)
        offset = (fi - 0.5) * bar_width
        bars = ax.bar(x + offset, fracs, bar_width, color=color, alpha=0.8,
                      label=fiber)
        for xi, (f, n) in enumerate(zip(fracs, ns)):
            ax.text(xi + offset, f + 0.01, f"{int(f*100)}%\n(n={n})",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(stages_present)
    ax.set_ylabel("Fraction tagged")
    ax.set_title("Optotagging yield by stage")
    ax.legend(fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)


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
        df_all.to_csv(cache_path, index=False)
        print(f"  Saved {len(df_all)} rows to {cache_path}")

    # ── Re-derive is_responsive from current thresholds ────────────────
    df_all["is_responsive"] = (
        (df_all["salt_p"] < SALT_ALPHA)
        & (df_all["latency_ms"] < MAX_LATENCY_MS)
        & (df_all["jitter_ms"] < MAX_JITTER_MS)
        & (df_all["reliability"] >= MIN_RELIABILITY)
    )
    print(f"  Applied thresholds: SALT<{SALT_ALPHA}, lat<{MAX_LATENCY_MS}ms, "
          f"jitter<{MAX_JITTER_MS}ms, rel>={MIN_RELIABILITY}")

    # ── Flag units tagged by both fibers ───────────────────────────────
    tagged_keys = df_all.loc[df_all["is_responsive"], ["session_name", "cluster_id"]]
    dual_counts = tagged_keys.groupby(["session_name", "cluster_id"]).size()
    dual_ids = set(dual_counts[dual_counts > 1].index)
    df_all["dual_fiber"] = df_all.apply(
        lambda r: (r["session_name"], r["cluster_id"]) in dual_ids, axis=1
    )
    n_dual = len(dual_ids)
    if n_dual > 0:
        print(f"  WARNING: {n_dual} units tagged by BOTH GPe and SNr (flagged in dual_fiber column)")

    # ── Summary stats ─────────────────────────────────────────────────
    n_units = df_all.groupby("fiber")["cluster_id"].nunique()
    n_resp = df_all[df_all["is_responsive"]].groupby("fiber")["cluster_id"].nunique()
    print("\n  === Optotagging Summary ===")
    for fiber in ["GPe", "SNr"]:
        total = n_units.get(fiber, 0)
        tagged = n_resp.get(fiber, 0)
        pct = 100 * tagged / total if total > 0 else 0
        pathway = "D2" if fiber == "GPe" else "D1"
        print(f"    {fiber} ({pathway}): {tagged}/{total} tagged ({pct:.1f}%)")

    n_sessions_with_data = df_all["session_name"].nunique()
    print(f"    Sessions with laser data: {n_sessions_with_data}")

    df_resp = df_all[df_all["is_responsive"]]

    # ── Figure ────────────────────────────────────────────────────────
    print("\n  Generating figure ...")

    fig = plt.figure(figsize=(16, 12))
    gs_top = gridspec.GridSpec(2, 2, figure=fig, top=0.95, bottom=0.55,
                               hspace=0.5, wspace=0.3)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig, top=0.45, bottom=0.05,
                               wspace=0.35)

    # Panel A: example raster + PSTH (pick best tagged unit per fiber)
    # Find example units with lowest SALT p-value for each fiber
    for fi, fiber in enumerate(["GPe", "SNr"]):
        sub = df_resp[df_resp["fiber"] == fiber].copy()
        if sub.empty:
            ax_r = fig.add_subplot(gs_top[0, fi])
            ax_p = fig.add_subplot(gs_top[1, fi])
            ax_r.text(0.5, 0.5, f"No {fiber}-tagged units", transform=ax_r.transAxes,
                      ha="center", va="center", fontsize=12, color="gray")
            ax_p.set_visible(False)
            continue

        best = sub.sort_values("salt_p").iloc[0]
        sname = int(best["session_name"])
        cid = int(best["cluster_id"])

        # Reload session to get spike times and pulse times for plotting
        try:
            sess = load_session(sname)
            tagger = OptoTagger(sess)
            pulses = tagger.gpe_pulses if fiber == "GPe" else tagger.snr_pulses
            cluster = next((c for c in sess.clusters if c.cluster_id == cid), None)

            if cluster is not None and pulses is not None:
                ax_r = fig.add_subplot(gs_top[0, fi])
                ax_p = fig.add_subplot(gs_top[1, fi])
                _plot_raster_psth(ax_r, ax_p, cluster.spike_times, pulses, fiber)
                pathway = "D2" if fiber == "GPe" else "D1"
                ax_r.set_title(
                    f"{fiber} → {pathway}  |  unit {cid}  |  "
                    f"p={best['salt_p']:.4f}  lat={best['latency_ms']:.1f}ms",
                    fontsize=9,
                )
            del sess
            gc.collect()
        except Exception as exc:
            print(f"    Warning: could not plot example for {fiber}: {exc}")

    # Panel B: latency distributions
    ax_lat = fig.add_subplot(gs_bot[0, 0])
    _plot_latency_distributions(ax_lat, df_resp)

    # Panel C: stage fractions
    ax_stage = fig.add_subplot(gs_bot[0, 1])
    _plot_stage_fractions(ax_stage, df_all)

    fig.suptitle("Optotagging: Antidromic D1/D2 SPN Identification (SALT test)",
                 fontsize=14, fontweight="bold", y=0.99)

    save_figure(fig, "fig43a_optotagging_overview", MODULE_NAME)
    print(f"  Saved fig33a_optotagging_overview.png")

    # ── Second figure: per-session counts ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    _plot_session_counts(ax2, df_all, manifest)
    save_figure(fig2, "fig43b_optotagging_session_counts", MODULE_NAME)
    print(f"  Saved fig33b_optotagging_session_counts.png")

    # ── Save stats summary ────────────────────────────────────────────
    stats_rows = []
    for fiber in ["GPe", "SNr"]:
        for stage in STAGE_ORDER:
            sub = df_all[(df_all["fiber"] == fiber) & (df_all["stage"] == stage)]
            n = len(sub)
            n_resp_s = int(sub["is_responsive"].sum())
            stats_rows.append({
                "fiber": fiber,
                "stage": stage,
                "n_units": n,
                "n_tagged": n_resp_s,
                "fraction": n_resp_s / n if n > 0 else np.nan,
                "mean_latency_ms": sub.loc[sub["is_responsive"], "latency_ms"].mean(),
                "mean_jitter_ms": sub.loc[sub["is_responsive"], "jitter_ms"].mean(),
                "mean_reliability": sub.loc[sub["is_responsive"], "reliability"].mean(),
            })

    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(CACHE_DIR, "optotagging_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved stats to {stats_path}")
    print(stats_df.to_string(index=False))

    print("\n[09a] Done.")


if __name__ == "__main__":
    main()
