"""Render talk-ready per-unit optotagging figures for the best BG_046 candidates.

Builds one clean 4-panel PNG per unit for:
  * the 3 collision-confirmed antidromic D1 units (tier==high_confidence), plus
  * the top short-latency CANDIDATE-tier units (2 D1 on SNr + 3 D2 on GPe),
    ranked by ``excess_reliability`` within pathway.

For each unit the panels are:
  1. Raster around the laser pulse (ms; pulse line at 0; response window shaded;
     peak-latency dashed). Spikes on collision-FREE pulses are grey, spikes on
     collision-EXPECTED pulses (a spontaneous spike preceded the pulse inside the
     collision window) are red.
  2. Collision PSTH: response rate on collision-free vs collision-expected pulses,
     overlaid. For a true antidromic unit the red (expected) trace is suppressed at
     the response latency while the grey (free) trace shows the locked response.
  3. Raw mean-waveform panel: peak-channel trace + a small footprint heatmap
     (or an "unavailable" placeholder if the raw waveform file is missing).
  4. A stats text box with the EXACT cached values for that (session, cluster,
     contributing_fiber).

Plus one summary figure: a scatter of peak_latency_ms vs excess_reliability for
ALL candidate/high-confidence units, with the rendered units labelled, and a
companion README.md.

Inputs (cached — the optotagging pipeline is NOT re-run):
  data/cache/optotagging/optotagging_results.csv   (per unit x fiber)
  data/cache/optotagging/optotagging_unit_tags.csv (per unit)

Reuses ``_extract_unit`` from scripts/optotagging/b_candidate_gallery.py
for the collision-split raster/PSTH extraction (no re-implementation of alignment).

Usage:
  py scripts/optotagging_figs/render_opto_candidates.py
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Repo-relative paths ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
OPTO_DIR = Path(__file__).resolve().parent          # b_candidate_gallery is a sibling now
TRACKING = REPO_ROOT / "scripts" / "pipelines" / "tracking"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "optotagging"
OUT_DIR = REPO_ROOT / "FIGURES" / "optotagging" / "BG_046"
CAND_DIR = OUT_DIR / "candidates"

for p in (str(SRC), str(OPTO_DIR), str(TRACKING)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Library imports ────────────────────────────────────────────────────
from visdetect.analysis.config import canonical_session_id
from visdetect.core.session import load_session
from visdetect.analysis.optotagging import OptoTagger
from visdetect.analysis.tracking_qc import (
    load_raw_mean_waveform, extract_peak_channel, extract_footprint,
)
import _subject_paths as sjp
from b_candidate_gallery import _extract_unit  # collision-split raster/PSTH extractor

SUBJECT = "BG_046"
FREE_COLOR = "0.45"          # grey — collision-free pulses
EXP_COLOR = "#d62728"        # red — collision-expected pulses
HC_COLOR = "#d62728"         # border/star for high-confidence units
PSTH_BIN_MS = 0.25
RASTER_WIN_S = (-0.005, 0.015)   # matches _extract_unit's internal window

# Fields shown verbatim in the per-unit stats box + README (from the results CSV row).
STAT_FIELDS = [
    "pathway", "tier", "stage", "fiber",
    "peak_latency_ms", "excess_reliability", "excess_jitter_ms",
    "salt_p", "poisson_p", "collision_status", "collision_suppression_index",
    "n_collision_free", "n_collision_expected", "baseline_rate_hz",
]


# ── Unit selection ─────────────────────────────────────────────────────
def select_units(results: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Return the ordered list of units to render as a DataFrame with all cached
    metrics of the CONTRIBUTING fiber joined in.

    3 high-confidence D1 (all) + top-2 D1 candidates (SNr) + top-3 D2 candidates
    (GPe), ranked by excess_reliability within pathway.
    """
    rkey = results.set_index(["session_name", "cluster_id", "fiber"])

    def _join(u):
        fiber = u["contributing_fiber"]
        try:
            m = rkey.loc[(int(u["session_name"]), int(u["cluster_id"]), fiber)]
        except KeyError:
            return None
        # rkey may have duplicate index rows in principle; take the first.
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        row = {
            "session_name": int(u["session_name"]),
            "cluster_id": int(u["cluster_id"]),
            "pathway": u["pathway"],
            "tier": u["tier"],
            "stage": u["stage"],
            "fiber": fiber,
        }
        for c in ("peak_latency_ms", "excess_reliability", "excess_jitter_ms",
                  "salt_p", "poisson_p", "collision_status",
                  "collision_suppression_index", "n_collision_free",
                  "n_collision_expected", "baseline_rate_hz", "win_lo", "win_hi",
                  "n_pulses"):
            row[c] = m[c]
        return row

    # 3 confirmed high-confidence D1
    hc = tags[tags.tier == "high_confidence"].copy()
    hc_rows = [r for r in (_join(u) for _, u in hc.iterrows()) if r is not None]

    # candidate tier -> join -> rank within pathway by excess_reliability
    cand = tags[tags.tier == "candidate"].copy()
    cand_rows = [r for r in (_join(u) for _, u in cand.iterrows()) if r is not None]
    cand_df = pd.DataFrame(cand_rows)

    d1 = (cand_df[(cand_df.pathway == "D1") & (cand_df.fiber == "SNr")]
          .sort_values("excess_reliability", ascending=False).head(2))
    d2 = (cand_df[(cand_df.pathway == "D2") & (cand_df.fiber == "GPe")]
          .sort_values("excess_reliability", ascending=False).head(3))

    sel = pd.concat([pd.DataFrame(hc_rows), d1, d2], ignore_index=True)
    # render order: high-confidence first, then D1 candidates, then D2 candidates
    tier_rank = {"high_confidence": 0, "candidate": 1}
    path_rank = {"D1": 0, "D2": 1}
    sel["_tr"] = sel.tier.map(tier_rank).fillna(9)
    sel["_pr"] = sel.pathway.map(path_rank).fillna(9)
    sel = sel.sort_values(
        ["_tr", "_pr", "excess_reliability"],
        ascending=[True, True, False]).reset_index(drop=True)
    sel["session8"] = sel.session_name.map(canonical_session_id)
    return sel.drop(columns=["_tr", "_pr"])


# ── Panel 3: raw waveform ──────────────────────────────────────────────
def _plot_waveform(ax_trace, ax_foot, wf):
    """Peak-channel trace (ax_trace) + footprint heatmap (ax_foot). If wf is None,
    both axes show an 'unavailable' placeholder."""
    if wf is None:
        for ax in (ax_trace, ax_foot):
            ax.text(0.5, 0.5, "raw waveform\nunavailable", ha="center", va="center",
                    fontsize=8, color="0.4", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        return
    peak = extract_peak_channel(wf)
    ax_trace.plot(wf[:, peak], color="#333333", lw=1.2)
    ax_trace.set_title(f"peak-channel waveform (ch {peak})", fontsize=8)
    ax_trace.set_xlabel("sample", fontsize=7)
    ax_trace.set_ylabel("a.u.", fontsize=7)
    ax_trace.tick_params(labelsize=6)

    snippet, chans = extract_footprint(wf, peak)
    # snippet: (n_samples, n_channels_kept) -> heatmap with time on x, channel on y
    im = ax_foot.imshow(snippet.T, aspect="auto", cmap="RdBu_r", origin="lower",
                        extent=[0, snippet.shape[0], chans[0], chans[-1] + 1])
    ax_foot.set_title("footprint (channels x time)", fontsize=8)
    ax_foot.set_xlabel("sample", fontsize=7)
    ax_foot.set_ylabel("channel", fontsize=7)
    ax_foot.tick_params(labelsize=6)
    cb = plt.colorbar(im, ax=ax_foot, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)


# ── Panel 4: stats text box ────────────────────────────────────────────
def _fmt(v):
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(v)


def _plot_stats_box(ax, row, d):
    ax.axis("off")
    lines = []
    for f in STAT_FIELDS:
        lines.append(f"{f:<28s}: {_fmt(row[f])}")
    # measured response rates from the collision split
    exp_str = f"{d['p_exp'] * 100:.0f}%" if d["n_exp"] else "n=0"
    free_str = f"{d['p_free'] * 100:.0f}%" if d["n_free"] else "n=0"
    lines.append("")
    lines.append(f"{'resp rate (collision-free)':<28s}: {free_str}")
    lines.append(f"{'resp rate (collision-expected)':<28s}: {exp_str}")
    lines.append(f"{'n pulses (this fiber)':<28s}: {int(d['n_pulses'])}")
    ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=8, transform=ax.transAxes)


# ── Per-unit figure ────────────────────────────────────────────────────
def render_unit(row, d, wf, out_path):
    is_hc = row["tier"] == "high_confidence"
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    # Panel 1: raster ---------------------------------------------------
    ax_r = fig.add_subplot(gs[0, 0])
    exp_of_spike = (d["pulse_expected"][d["rel_idx"]]
                    if len(d["rel_idx"]) else np.array([], bool))
    t_ms = d["rel_times"] * 1000.0
    ax_r.scatter(t_ms[~exp_of_spike], d["rel_idx"][~exp_of_spike], s=4,
                 c=FREE_COLOR, marker="|", linewidths=0.6,
                 label=f"collision-free (n={d['n_free']})")
    ax_r.scatter(t_ms[exp_of_spike], d["rel_idx"][exp_of_spike], s=6,
                 c=EXP_COLOR, marker="|", linewidths=0.8,
                 label=f"collision-expected (n={d['n_exp']})")
    ax_r.axvline(0, color="deepskyblue", lw=1.2)
    ax_r.axvline(row["peak_latency_ms"], color="green", ls="--", lw=1.0,
                 label=f"peak latency {row['peak_latency_ms']:.2f} ms")
    ax_r.axvspan(row["win_lo"], row["win_hi"], color="green", alpha=0.10, lw=0)
    ax_r.set_xlim(RASTER_WIN_S[0] * 1000, RASTER_WIN_S[1] * 1000)
    ax_r.set_ylim(0, max(int(d["n_pulses"]), 1))
    ax_r.set_xlabel("time from laser pulse (ms)", fontsize=9)
    ax_r.set_ylabel("laser pulse #", fontsize=9)
    ax_r.set_title("Spike raster around the laser pulse", fontsize=10)
    ax_r.legend(fontsize=6, loc="upper right", framealpha=0.9)

    # Panel 2: collision PSTH ------------------------------------------
    ax_p = fig.add_subplot(gs[0, 1])
    bins = np.arange(RASTER_WIN_S[0], RASTER_WIN_S[1] + 1e-9, PSTH_BIN_MS / 1000.0)
    ctr = (bins[:-1] + bins[1:]) / 2.0 * 1000.0
    if d["n_free"]:
        h, _ = np.histogram(d["rel_times"][~exp_of_spike], bins=bins)
        ax_p.plot(ctr, h / d["n_free"], color=FREE_COLOR, lw=1.6,
                  label="collision-free")
    if d["n_exp"]:
        h, _ = np.histogram(d["rel_times"][exp_of_spike], bins=bins)
        ax_p.plot(ctr, h / d["n_exp"], color=EXP_COLOR, lw=1.6,
                  label="collision-expected")
    ax_p.axvline(0, color="deepskyblue", lw=1.2)
    ax_p.axvline(row["peak_latency_ms"], color="green", ls="--", lw=1.0)
    ax_p.axvspan(row["win_lo"], row["win_hi"], color="green", alpha=0.10, lw=0)
    ax_p.set_xlim(RASTER_WIN_S[0] * 1000, RASTER_WIN_S[1] * 1000)
    ax_p.set_xlabel("time from laser pulse (ms)", fontsize=9)
    ax_p.set_ylabel("spikes per pulse", fontsize=9)
    ax_p.set_title("Collision PSTH: free vs expected pulses\n"
                   "(antidromic units suppress on expected)", fontsize=10)
    ax_p.legend(fontsize=7, loc="upper right", framealpha=0.9)

    # Panel 3: waveform -------------------------------------------------
    inner_wf = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, 0], wspace=0.45, width_ratios=[1, 1.3])
    ax_trace = fig.add_subplot(inner_wf[0])
    ax_foot = fig.add_subplot(inner_wf[1])
    _plot_waveform(ax_trace, ax_foot, wf)

    # Panel 4: stats box ------------------------------------------------
    ax_s = fig.add_subplot(gs[1, 1])
    _plot_stats_box(ax_s, row, d)

    # Header ------------------------------------------------------------
    star = "★ " if is_hc else ""
    conf = "collision-confirmed antidromic" if is_hc else "short-latency putative"
    fig.suptitle(
        f"{star}{SUBJECT}   session {row['session8']}   unit c{int(row['cluster_id'])}   "
        f"—   {row['pathway']} {row['tier']}\n"
        f"contributing fiber: {row['fiber']}   ({conf})",
        fontsize=13, y=0.985)

    if is_hc:
        # red border around the whole figure
        fig.patch.set_edgecolor(HC_COLOR)
        fig.patch.set_linewidth(6)

    fig.savefig(out_path, dpi=150, facecolor="white",
                edgecolor=(HC_COLOR if is_hc else "white"))
    plt.close(fig)


# ── Summary figure ─────────────────────────────────────────────────────
def render_summary(all_tagged, sel, out_path):
    """Scatter peak_latency_ms vs excess_reliability for ALL candidate/HC units,
    coloured by pathway/tier, with the rendered units labelled."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # background: all candidate + high-confidence units (contributing fiber metrics)
    bg = all_tagged.copy()
    styles = {
        ("D1", "candidate"): dict(c="#1f77b4", marker="o", alpha=0.35, s=30),
        ("D2", "candidate"): dict(c="#9467bd", marker="o", alpha=0.35, s=30),
    }
    for (pw, tr), g in bg.groupby(["pathway", "tier"]):
        if tr == "high_confidence":
            continue
        st = styles.get((pw, tr), dict(c="0.6", marker="o", alpha=0.35, s=30))
        ax.scatter(g.peak_latency_ms, g.excess_reliability,
                   label=f"{pw} {tr} (n={len(g)})", **st)

    # rendered units: filled, larger, labelled
    for _, r in sel.iterrows():
        is_hc = r["tier"] == "high_confidence"
        col = {"D1": "#1f77b4", "D2": "#9467bd"}[r["pathway"]]
        ax.scatter(r.peak_latency_ms, r.excess_reliability,
                   s=180 if is_hc else 110,
                   c=(HC_COLOR if is_hc else col),
                   edgecolors="black", linewidths=(2.0 if is_hc else 1.0),
                   marker=("*" if is_hc else "D"), zorder=5)
        lab = f"{r['session8'][:4]} c{int(r['cluster_id'])}"
        ax.annotate(lab, (r.peak_latency_ms, r.excess_reliability),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel("peak response latency (ms)", fontsize=11)
    ax.set_ylabel("excess reliability (vs jittered null)", fontsize=11)
    ax.set_title(
        f"{SUBJECT} optotagging candidates — latency vs reliability\n"
        "★ red = collision-confirmed antidromic D1 (n=3); "
        "◆ = rendered top candidates", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


# ── README ─────────────────────────────────────────────────────────────
def write_readme(sel, tags, path):
    tier_counts = tags.tier.value_counts()
    n_cand = int(tier_counts.get("candidate", 0))
    n_hc = int(tier_counts.get("high_confidence", 0))
    putative = tags[tags.tier.isin(["candidate", "high_confidence"])]
    n_put = len(putative)
    n_put_d1 = int((putative.pathway == "D1").sum())
    n_put_d2 = int((putative.pathway == "D2").sum())
    cand_only = tags[tags.tier == "candidate"]
    n_cand_d1 = int((cand_only.pathway == "D1").sum())
    n_cand_d2 = int((cand_only.pathway == "D2").sum())

    def _md(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "nan"
            if abs(v) < 1e-3 and v != 0:
                return f"{v:.2e}"
            return f"{v:.4g}"
        return str(v)

    lines = []
    lines.append("# BG_046 optotagging candidates — what the laser tagging shows")
    lines.append("")
    lines.append("Plain-language companion to the per-unit figures in "
                 "`candidates/`. These identify striatal units that are driven "
                 "antidromically (i.e. their axon terminals in GPe or SNr were "
                 "directly activated) by brief laser pulses — the gold-standard "
                 "for cell-type (pathway) identity in this preparation.")
    lines.append("")
    lines.append("## Yield summary")
    lines.append("")
    lines.append(f"- **22 laser sessions** in mouse {SUBJECT}.")
    lines.append(f"- **Candidate tier: {n_put} putative** short-latency responsive "
                 f"units ({n_put_d1} D1 on SNr + {n_put_d2} D2 on GPe). "
                 f"(Of these, {n_cand} are tier `candidate` — {n_cand_d1} D1 + "
                 f"{n_cand_d2} D2 — and {n_hc} are the confirmed high-confidence set.)")
    lines.append(f"- **High-confidence, collision-confirmed = {n_hc} (all D1, 0 D2).** "
                 "These are the only units that pass the offline collision test.")
    lines.append("- **~90% of short-latency SALT/Poisson-significant responses FAIL "
                 "the offline collision test** — they are *synaptic* "
                 "(trans-synaptic / network) responses, NOT antidromic. Terminal "
                 "stimulation in GPe/SNr drives mostly *network* responses in medial "
                 "striatum, not direct antidromic spikes.")
    lines.append("")
    lines.append("## What the collision test is")
    lines.append("")
    lines.append("A laser-evoked spike is **truly antidromic only if it disappears "
                 "when a spontaneous spike has just travelled down the same axon** "
                 "(the two collide and cancel) — so on pulses preceded by a "
                 "spontaneous spike within the collision window ('collision-expected') "
                 "the evoked response should be *suppressed* relative to pulses with a "
                 "clear axon ('collision-free'). **Passing this test = true antidromic "
                 "identity** (a real terminal-activated projection neuron); a synaptic "
                 "response does not collide and so fails.")
    lines.append("")
    lines.append("## Rendered units")
    lines.append("")
    lines.append("Each row is one figure in `candidates/`. Values are pulled directly "
                 "from `optotagging_results.csv` for the unit's contributing fiber. "
                 "The 3 high-confidence units are collision-confirmed D1; the rest are "
                 "the strongest short-latency *candidates* (by excess reliability), "
                 "shown for completeness — they are putative, not confirmed.")
    lines.append("")
    header = ("| session | cluster | pathway | tier | fiber | peak_latency_ms | "
              "excess_reliability | excess_jitter_ms | salt_p | poisson_p | "
              "collision_status | suppression_index | n_free | n_expected |")
    sep = "|" + "|".join(["---"] * 14) + "|"
    lines.append(header)
    lines.append(sep)
    for _, r in sel.iterrows():
        lines.append(
            f"| {r['session8']} | {int(r['cluster_id'])} | {r['pathway']} | "
            f"{r['tier']} | {r['fiber']} | {_md(r['peak_latency_ms'])} | "
            f"{_md(r['excess_reliability'])} | {_md(r['excess_jitter_ms'])} | "
            f"{_md(r['salt_p'])} | {_md(r['poisson_p'])} | {r['collision_status']} | "
            f"{_md(r['collision_suppression_index'])} | "
            f"{int(r['n_collision_free'])} | {int(r['n_collision_expected'])} |")
    lines.append("")
    lines.append("## Honest interpretation")
    lines.append("")
    lines.append(f"**Optotagging alone cannot anchor cell-type-resolved population "
                 f"analyses in {SUBJECT}: only {n_hc} units are confirmed antidromic** "
                 "(all D1, no D2). Treat the ~162 candidate-tier units as "
                 "*short-latency responsive / putative*, **not** *confirmed "
                 "antidromic*. The low antidromic yield may be intrinsic to this "
                 "preparation (terminal stimulation in GPe/SNr recruits striatal "
                 "networks far more than it back-propagates to soma). Any pathway "
                 "(D1/D2) labelling that leans on these candidates should be flagged "
                 "as provisional.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- **The excess-jitter gate is effectively vacuous.** Jitter is "
                 "measured inside the ±0.75 ms response window, so it is "
                 "mechanically ≤ 0.47 ms for every unit — it does no work. "
                 "**Collision is the real binding gate.**")
    lines.append("- **Session `05092025` is skipped.** Its pkl is "
                 "`BG_046_05092025_b.pkl`; the loader's date→filename match "
                 "misses the `_b` suffix, so no units from that session appear here.")
    lines.append("- **`salt_p` floors at 0.000816** (= 1 / 1226 permutations); a "
                 "value of 0.000816 means 'at the permutation floor', not literally "
                 "zero.")
    lines.append("- **Waveform `cell_type` labels were absent at build time** "
                 "(the `cell_type` column is all NaN in the tags cache), so the "
                 "waveform panels show the raw mean waveform only — no narrow/"
                 "broad classification. One rendered unit "
                 "(`10092025` c225) has no raw-waveform file and shows an "
                 "'unavailable' placeholder.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Verification ───────────────────────────────────────────────────────
def verify(sel, results):
    """Re-read each rendered unit's row from the results CSV and assert the shown
    collision_status / peak_latency_ms / excess_reliability match exactly. Assert
    the 3 high-confidence D1 are all present with collision_status==pass. Assert
    each PNG exists and is > 10 KB."""
    rkey = results.set_index(["session_name", "cluster_id", "fiber"])
    problems = []
    for _, r in sel.iterrows():
        m = rkey.loc[(int(r["session_name"]), int(r["cluster_id"]), r["fiber"])]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        for c in ("peak_latency_ms", "excess_reliability"):
            a, b = float(r[c]), float(m[c])
            if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > 1e-9:
                problems.append(f"{r['session8']} c{int(r['cluster_id'])}: {c} "
                                f"rendered {a} != csv {b}")
        if str(r["collision_status"]) != str(m["collision_status"]):
            problems.append(f"{r['session8']} c{int(r['cluster_id'])}: "
                            f"collision_status rendered {r['collision_status']} != "
                            f"csv {m['collision_status']}")

    hc = sel[sel.tier == "high_confidence"]
    if len(hc) != 3:
        problems.append(f"expected 3 high_confidence, got {len(hc)}")
    for _, r in hc.iterrows():
        if r["pathway"] != "D1":
            problems.append(f"HC {r['session8']} c{int(r['cluster_id'])} not D1")
        if str(r["collision_status"]) != "pass":
            problems.append(f"HC {r['session8']} c{int(r['cluster_id'])} "
                            f"collision_status={r['collision_status']} != pass")

    for _, r in sel.iterrows():
        p = CAND_DIR / r["png_name"]
        if not p.exists():
            problems.append(f"missing PNG: {p}")
        elif p.stat().st_size < 10 * 1024:
            problems.append(f"PNG too small (<10KB): {p} = {p.stat().st_size} B")
    return problems


# ── Main ───────────────────────────────────────────────────────────────
def main():
    CAND_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(CACHE_DIR / "optotagging_results.csv")
    tags = pd.read_csv(CACHE_DIR / "optotagging_unit_tags.csv")

    sel = select_units(results, tags)
    print(f"Selected {len(sel)} units to render "
          f"(high_confidence={int((sel.tier=='high_confidence').sum())}, "
          f"candidate={int((sel.tier=='candidate').sum())}):")
    for _, r in sel.iterrows():
        print(f"  {r['session8']} c{int(r['cluster_id'])}  {r['pathway']}/{r['tier']} "
              f"fiber={r['fiber']}  rel={r['excess_reliability']:.4f} "
              f"lat={r['peak_latency_ms']:.2f}ms coll={r['collision_status']}")

    # png filename per unit
    sel["png_name"] = [
        f"{r['tier']}_{r['pathway']}_{r['session8']}_c{int(r['cluster_id'])}.png"
        for _, r in sel.iterrows()
    ]

    pkldir = sjp.pkl_dir(SUBJECT)
    wfroot = sjp.raw_wf_root(SUBJECT)

    # Group by session so each pkl loads once.
    for sname, grp in sel.groupby("session_name"):
        pkl = sjp.session_pkl(SUBJECT, int(sname), pkldir)
        if pkl is None:
            print(f"  SKIP session {sname}: pkl not found")
            continue
        print(f"  Loading {pkl.name} ...")
        S = load_session(str(pkl))
        tagger = OptoTagger(S)
        pulses_by_fiber = {"GPe": tagger.gpe_pulses, "SNr": tagger.snr_pulses}
        clusters = {int(c.cluster_id): c for c in S.clusters}

        for _, r in grp.iterrows():
            fiber = r["fiber"]
            pulses = pulses_by_fiber.get(fiber)
            cl = clusters.get(int(r["cluster_id"]))
            if pulses is None or cl is None:
                print(f"    SKIP c{int(r['cluster_id'])}: "
                      f"pulses/cluster missing (fiber={fiber})")
                continue
            spikes = np.asarray(cl.spike_times, float)
            d = _extract_unit(spikes, pulses, float(r["peak_latency_ms"]),
                              float(r["win_lo"]), float(r["win_hi"]))
            wf = load_raw_mean_waveform(wfroot, r["session8"], int(r["cluster_id"]))
            out_path = CAND_DIR / r["png_name"]
            render_unit(r, d, wf, out_path)
            print(f"    rendered {out_path.name} "
                  f"({out_path.stat().st_size // 1024} KB, wf="
                  f"{'ok' if wf is not None else 'MISSING'})")

        del S
        gc.collect()

    # Summary figure: all candidate/HC units with contributing-fiber metrics
    rkey = results.set_index(["session_name", "cluster_id", "fiber"])
    put = tags[tags.tier.isin(["candidate", "high_confidence"])].copy()
    bg_rows = []
    for _, u in put.iterrows():
        fiber = u["contributing_fiber"]
        try:
            m = rkey.loc[(int(u["session_name"]), int(u["cluster_id"]), fiber)]
        except KeyError:
            continue
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        bg_rows.append(dict(pathway=u["pathway"], tier=u["tier"],
                            peak_latency_ms=float(m["peak_latency_ms"]),
                            excess_reliability=float(m["excess_reliability"])))
    all_tagged = pd.DataFrame(bg_rows)
    summ_path = OUT_DIR / "opto_candidates_summary.png"
    render_summary(all_tagged, sel, summ_path)
    print(f"  summary -> {summ_path} ({summ_path.stat().st_size // 1024} KB)")

    # README
    readme_path = OUT_DIR / "README.md"
    write_readme(sel, tags, readme_path)
    print(f"  README  -> {readme_path}")

    # Self-verification
    print("\n=== SELF-VERIFICATION ===")
    problems = verify(sel, results)
    if problems:
        print("FAILED:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("All assertions PASSED:")
    print(f"  - {len(sel)} rendered units match optotagging_results.csv exactly "
          "(peak_latency_ms, excess_reliability, collision_status)")
    print("  - 3 high_confidence units all present, all D1, all collision_status==pass")
    print("  - all PNGs exist and are > 10 KB")

    # emit the rendered table for the caller report
    cols = ["session8", "cluster_id"] + STAT_FIELDS + [
        "collision_suppression_index"]
    print("\n=== RENDERED UNITS TABLE ===")
    disp = sel[["session8", "cluster_id"] + STAT_FIELDS].copy()
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
