"""Fig43d: Optotagging candidate gallery — per-unit raster + PSTH with collision split.

For every TAGGED unit (pathway != None in optotagging_unit_tags.csv) this draws, for the
unit's contributing fiber (SNr for D1, GPe for D2), a raster of spikes around the laser pulse
with each pulse colored by whether a spontaneous spike preceded it within the collision window
(collision-EXPECTED, red) vs not (collision-FREE, grey), plus a PSTH overlaying the free and
expected pulse subsets. For a true antidromic unit the red (expected) trace is suppressed at the
response latency while the grey (free) trace shows the locked response — so the collision logic
is visible per unit. Each panel is annotated with tier / pathway / latency / excess-reliability /
SALT p / collision status and the free-vs-expected response rates.

High-confidence units are shown first (★, red border). Output is paginated:
  figures/09_optotagging/fig43d_candidate_gallery_p{NN}.png

Reads the caches produced by a_optotagging_identification.py:
  cache/optotagging_results.csv (per unit x fiber), cache/optotagging_unit_tags.csv (per unit).
Reloads each session pkl once to get spike times + laser pulse times.

Usage:
  cd analysis_suite && py 09_optotagging/b_candidate_gallery.py [--per-page 12]
"""
import argparse
import os
import sys
import gc

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import CACHE_DIR
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.optotagging import OptoTagger, COLLISION_REFRACTORY_MS

setup_style()

MODULE_NAME = "09_optotagging"
FIBER_COLORS = {"GPe": "#9b59b6", "SNr": "#e67e22"}
RASTER_WIN_S = (-0.005, 0.015)   # window shown around each pulse (s)
PSTH_BIN_MS = 0.25
EXP_COLOR = "#d62728"            # collision-expected (red)
FREE_COLOR = "0.45"             # collision-free (grey)
TIER_RANK = {"high_confidence": 0, "candidate": 1}
PATHWAY_RANK = {"D1": 0, "D2": 1}


# ── Per-unit data extraction (session must be loaded) ──────────────────
def _extract_unit(spikes, pulses, peak_latency_ms, win_lo_ms, win_hi_ms):
    """Return raster + collision-split render data for one unit x fiber."""
    spikes = np.asarray(spikes, float).ravel()
    pulses = np.asarray(pulses, float).ravel()
    cw = (peak_latency_ms + COLLISION_REFRACTORY_MS) / 1000.0
    a, b = RASTER_WIN_S
    rwa, rwb = win_lo_ms / 1000.0, win_hi_ms / 1000.0
    n = len(pulses)
    rel_times, rel_idx = [], []
    has_pre = np.zeros(n, bool)
    has_resp = np.zeros(n, bool)
    for i, p in enumerate(pulses):
        i0 = np.searchsorted(spikes, p + a)
        i1 = np.searchsorted(spikes, p + b)
        if i1 > i0:
            rel_times.append(spikes[i0:i1] - p)
            rel_idx.append(np.full(i1 - i0, i))
        has_pre[i] = np.searchsorted(spikes, p) - np.searchsorted(spikes, p - cw) > 0
        has_resp[i] = np.searchsorted(spikes, p + rwb) - np.searchsorted(spikes, p + rwa) > 0
    rel_times = np.concatenate(rel_times) if rel_times else np.array([])
    rel_idx = np.concatenate(rel_idx) if rel_idx else np.array([], int)
    n_free = int((~has_pre).sum())
    n_exp = int(has_pre.sum())
    return {
        "rel_times": rel_times, "rel_idx": rel_idx, "pulse_expected": has_pre,
        "n_pulses": n, "n_free": n_free, "n_exp": n_exp,
        "p_free": float(has_resp[~has_pre].mean()) if n_free else float("nan"),
        "p_exp": float(has_resp[has_pre].mean()) if n_exp else float("nan"),
    }


# ── Per-unit plotting (raster + PSTH into a nested gridspec cell) ───────
def _plot_unit(ax_r, ax_p, d, meta):
    exp_of_spike = d["pulse_expected"][d["rel_idx"]] if len(d["rel_idx"]) else np.array([], bool)
    t_ms = d["rel_times"] * 1000.0
    # raster: grey = collision-free, red = collision-expected
    ax_r.scatter(t_ms[~exp_of_spike], d["rel_idx"][~exp_of_spike], s=1.2,
                 c=FREE_COLOR, marker="|", linewidths=0.4)
    ax_r.scatter(t_ms[exp_of_spike], d["rel_idx"][exp_of_spike], s=1.2,
                 c=EXP_COLOR, marker="|", linewidths=0.4)
    ax_r.axvline(0, color="deepskyblue", lw=0.8)
    ax_r.axvline(meta["peak_latency_ms"], color="green", ls="--", lw=0.6)
    ax_r.axvspan(meta["win_lo"], meta["win_hi"], color="green", alpha=0.10, lw=0)
    ax_r.set_xlim(RASTER_WIN_S[0] * 1000, RASTER_WIN_S[1] * 1000)
    ax_r.set_ylim(0, max(d["n_pulses"], 1))
    ax_r.set_xticklabels([])
    ax_r.tick_params(labelsize=5, length=2)
    ax_r.set_ylabel("pulse", fontsize=5)

    star = "★ " if meta["tier"] == "high_confidence" else ""
    exp_str = f"{d['p_exp'] * 100:.0f}%" if d["n_exp"] else "n=0"
    ax_r.set_title(
        f"{star}S{meta['session']} c{meta['cluster']} {meta['pathway']}/{meta['tier']}\n"
        f"lat {meta['peak_latency_ms']:.1f}ms  rel {meta['excess_reliability']:.2f}  "
        f"salt {meta['salt_p']:.2g}  coll:{meta['collision_status']}\n"
        f"resp free {d['p_free'] * 100:.0f}% vs exp {exp_str}",
        fontsize=5, linespacing=0.95)
    if meta["tier"] == "high_confidence":
        for s in ax_r.spines.values():
            s.set_color(EXP_COLOR)
            s.set_linewidth(1.3)

    # PSTH: free vs expected (per-pulse normalized) — collision suppression made visible
    bins = np.arange(RASTER_WIN_S[0], RASTER_WIN_S[1] + 1e-9, PSTH_BIN_MS / 1000.0)
    ctr = (bins[:-1] + bins[1:]) / 2.0 * 1000.0
    if d["n_free"]:
        h, _ = np.histogram(d["rel_times"][~exp_of_spike], bins=bins)
        ax_p.plot(ctr, h / d["n_free"], color=FREE_COLOR, lw=0.8)
    if d["n_exp"]:
        h, _ = np.histogram(d["rel_times"][exp_of_spike], bins=bins)
        ax_p.plot(ctr, h / d["n_exp"], color=EXP_COLOR, lw=0.8)
    ax_p.axvline(0, color="deepskyblue", lw=0.8)
    ax_p.set_xlim(RASTER_WIN_S[0] * 1000, RASTER_WIN_S[1] * 1000)
    ax_p.tick_params(labelsize=5, length=2)
    ax_p.set_xlabel("ms from pulse", fontsize=5)
    ax_p.set_ylabel("spk/pulse", fontsize=5)


def main():
    parser = argparse.ArgumentParser(description="Optotagging per-candidate gallery")
    parser.add_argument("--per-page", type=int, default=12, help="units per page (default 12)")
    args = parser.parse_args()
    ncol = 3
    nrow = int(np.ceil(args.per_page / ncol))

    results = pd.read_csv(os.path.join(CACHE_DIR, "optotagging_results.csv"))
    units = pd.read_csv(os.path.join(CACHE_DIR, "optotagging_unit_tags.csv"))
    tagged = units[units.pathway.notna()].copy()
    if tagged.empty:
        print("  No tagged units (pathway != None) — nothing to plot.")
        return

    # metrics of the CONTRIBUTING fiber for each tagged unit
    rkey = results.set_index(["session_name", "cluster_id", "fiber"])
    print(f"  Tagged units to render: {len(tagged)} "
          f"(high_confidence={int((tagged.tier=='high_confidence').sum())}, "
          f"candidate={int((tagged.tier=='candidate').sum())})")

    # ── Pass 1: load each session once, extract render data ────────────
    render = []
    for sname, grp in tagged.groupby("session_name"):
        try:
            sess = load_session(int(sname))
            tagger = OptoTagger(sess)
        except Exception as exc:
            print(f"    SKIP session {sname}: {exc}")
            continue
        pulses_by_fiber = {"GPe": tagger.gpe_pulses, "SNr": tagger.snr_pulses}
        clusters = {c.cluster_id: c for c in sess.clusters}
        for _, u in grp.iterrows():
            fiber = u["contributing_fiber"]
            pulses = pulses_by_fiber.get(fiber)
            cl = clusters.get(int(u["cluster_id"]))
            if pulses is None or cl is None:
                continue
            try:
                m = rkey.loc[(int(sname), int(u["cluster_id"]), fiber)]
            except KeyError:
                continue
            d = _extract_unit(cl.spike_times, pulses,
                              float(m["peak_latency_ms"]), float(m["win_lo"]), float(m["win_hi"]))
            render.append({
                "d": d,
                "session": int(sname), "cluster": int(u["cluster_id"]),
                "pathway": u["pathway"], "tier": u["tier"], "fiber": fiber,
                "peak_latency_ms": float(m["peak_latency_ms"]),
                "win_lo": float(m["win_lo"]), "win_hi": float(m["win_hi"]),
                "excess_reliability": float(m["excess_reliability"]),
                "salt_p": float(m["salt_p"]), "collision_status": m["collision_status"],
            })
        del sess
        gc.collect()
        print(f"    {sname}: rendered {sum(r['session']==int(sname) for r in render)} units so far")

    if not render:
        print("  No renderable units found.")
        return

    # ── Sort: high-confidence first, then D1 before D2, then by SALT p ─
    render.sort(key=lambda r: (TIER_RANK.get(r["tier"], 9),
                               PATHWAY_RANK.get(r["pathway"], 9), r["salt_p"]))

    # ── Pass 2: paginate + plot ────────────────────────────────────────
    per_page = ncol * nrow
    n_pages = int(np.ceil(len(render) / per_page))
    print(f"  Plotting {len(render)} units across {n_pages} page(s) "
          f"({ncol}x{nrow}/page) ...")
    for pg in range(n_pages):
        chunk = render[pg * per_page:(pg + 1) * per_page]
        fig = plt.figure(figsize=(ncol * 3.2, nrow * 2.4))
        outer = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.75, wspace=0.3)
        for k, meta in enumerate(chunk):
            r, c = divmod(k, ncol)
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[r, c], height_ratios=[3, 1], hspace=0.08)
            ax_r = fig.add_subplot(inner[0])
            ax_p = fig.add_subplot(inner[1])
            _plot_unit(ax_r, ax_p, meta["d"], meta)
        fig.suptitle(
            f"Optotagging candidate gallery (page {pg+1}/{n_pages}) — "
            f"grey = collision-free, red = collision-expected; ★ = high-confidence",
            fontsize=8, y=0.995)
        save_figure(fig, f"fig43d_candidate_gallery_p{pg+1:02d}", MODULE_NAME)
        print(f"    saved page {pg+1}/{n_pages} ({len(chunk)} units)")

    print(f"\n[09b] Done — {len(render)} units, {n_pages} page(s).")


if __name__ == "__main__":
    main()
