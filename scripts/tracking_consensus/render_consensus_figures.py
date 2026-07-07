"""Clean 1-page 'representative neuron' figure per top UM x DANT consensus track.

Each figure is built to convince a skeptic that ONE physical neuron was followed
across many sessions of learning, using FOUR independent lines of evidence plus
TWO independent trackers:
  * waveform shape stability across sessions (peak-channel overlay)
  * multi-channel footprint stability (first vs last agreed session)
  * probe-depth stability
  * held-out log-ISI fingerprint vs the simultaneously-recorded population (the
    independent validation axis; neither tracker uses spike timing to match)
  * a two-tracker agreement strip (UnitMatch AND DANT independently include it)

Inputs (all LOCAL; no pkl loads here -- waveforms/positions are .npy, ISI is cached):
  data/cache/tracking_consensus/BG_046/consensus_cohort.csv   (augmented)
  data/cache/tracking_consensus/BG_046/consensus_members.csv
  data/cache/tracking_consensus/BG_046/isi_holdout.pkl
  data/cache/tracking_consensus/BG_046/nonmatched_corrs.npy
  data/cache/tracking_consensus/BG_046/isi_validation.json
  data/cache/um_ref/unit_index.csv            (UM full track spans)
  data/cache/dant/BG_046/dant_registry.csv    (DANT full track spans)
  data/unit_match/input/BG_046/<sess>/RawWaveforms + channel_positions.npy
  data/BG_046_staging_manifest.csv

Output: FIGURES/tracking_consensus/BG_046/candidates/consensus_<um>_<dant>.png
        FIGURES/tracking_consensus/BG_046/cohort_summary.png

Usage: py scripts/tracking_consensus/render_consensus_figures.py [--uids 942 776 ...]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
plt.rcParams.update({"xtick.labelsize": 10, "ytick.labelsize": 10, "axes.labelsize": 11.5})

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from visdetect.analysis.config import canonical_session_id, session_date_key  # noqa: E402
from visdetect.analysis.tracking_qc import (  # noqa: E402
    load_raw_mean_waveform, load_channel_positions,
    extract_peak_channel, extract_footprint, isi_log_histogram,
)

SUBJECT = "BG_046"
CACHE = ROOT / "data/cache/tracking_consensus/BG_046"
RAW_WF_ROOT = ROOT / "data/unit_match/input/BG_046"
STAGING = ROOT / f"data/{SUBJECT}_staging_manifest.csv"
UM_REG = ROOT / "data/cache/um_ref/unit_index.csv"
DANT_REG = ROOT / "data/cache/dant/BG_046/dant_registry.csv"
OUT_DIR = ROOT / "FIGURES/tracking_consensus/BG_046"
CAND_DIR = OUT_DIR / "candidates"

STAGE_COLORS = {"Naive": "#c7e9c0", "Learning": "#74c476", "Expert": "#238b45",
                "Excluded": "#d9d9d9", "Unknown": "#f0f0f0"}
_, ISI_CENTERS = isi_log_histogram(np.array([]))  # bin centres (s)

# default render set: 5 cleanest longitudinal + 1 strict Naive->Expert exemplar
DEFAULT_UIDS = [942, 776, 827, 1132, 995, 349]


# ---------------------------------------------------------------- helpers
def _stage_map():
    st = pd.read_csv(STAGING, dtype={"session_name": str})
    return {canonical_session_id(s): stg for s, stg in zip(st["session_name"], st["stage"])}


def _reg_sessions(path, uid_col):
    """uid -> set of canonical session tokens where that tracker includes the unit."""
    df = pd.read_csv(path, dtype=str)
    df["sk"] = df["session"].map(canonical_session_id)
    df[uid_col] = df[uid_col].astype(int)
    if uid_col == "dant_uid":
        df = df[df[uid_col] >= 0]
    return {u: set(g["sk"]) for u, g in df.groupby(uid_col)}


def _chron(sessions):
    return sorted(sessions, key=session_date_key)


def _pairwise_corr(traces):
    vals = []
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            a, b = traces[i], traces[j]
            if a is None or b is None or a.std() == 0 or b.std() == 0:
                continue
            n = min(len(a), len(b))
            vals.append(np.corrcoef(a[:n], b[:n])[0, 1])
    return float(np.mean(vals)) if vals else np.nan


def _color_for_order(n):
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------- per-candidate
def render_candidate(um_uid, cohort, members, holdout, nonmatched, stage_of,
                     um_sessions, dant_sessions, cohort_auc):
    row = cohort[cohort["um_uid"] == um_uid]
    if row.empty:
        print(f"  uid {um_uid}: not in cohort -> skip")
        return None
    row = row.iloc[0]
    dant_uid = int(row["dant_uid"])
    mem = members[members["um_uid"] == um_uid].copy()
    mem["sk"] = mem["session_key"].map(canonical_session_id)
    mem = mem.sort_values("sk", key=lambda s: s.map(session_date_key))
    sess_list = list(mem["sk"])
    ks_of = dict(zip(mem["sk"], mem["ks_unit_id"].astype(int)))

    # ---- load per-session waveform / footprint / depth (npy, fast) ----
    traces, depths, valid_sess, footprints = [], [], [], []
    for sk in sess_list:
        wf = load_raw_mean_waveform(RAW_WF_ROOT, sk, ks_of[sk])
        pos = load_channel_positions(RAW_WF_ROOT, sk)
        if wf is None:
            continue
        pc = extract_peak_channel(wf)
        traces.append(wf[:, pc].astype(float))
        depths.append(float(pos[pc, 1]) if pos is not None and pc < len(pos) else np.nan)
        snip, _ = extract_footprint(wf, pc)
        footprints.append(snip)
        valid_sess.append(sk)
    if len(valid_sess) < 2:
        print(f"  uid {um_uid}: <2 usable waveforms -> skip")
        return None
    wave_r = _pairwise_corr(traces)
    colors = _color_for_order(len(valid_sess))

    # ---- figure ----
    fig = plt.figure(figsize=(15, 9.4))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.0, 1.0, 0.65],
                           hspace=0.46, wspace=0.30,
                           left=0.06, right=0.975, top=0.83, bottom=0.07)

    # header
    stages = [stage_of.get(s, "Unknown") for s in sess_list]
    srange = f"{stages[0]}→{stages[-1]}"
    n2e = "Naive→Expert" if bool(row["naive_to_expert"]) else (
        "Learning→Expert" if bool(row["learning_to_expert"]) else "single-stage")
    depth_range = np.nanmax(depths) - np.nanmin(depths)
    fig.suptitle(
        f"Consensus neuron  —  UnitMatch #{um_uid}  ∩  DANT #{dant_uid}"
        f"   ({int(row['n_agree'])} sessions both trackers agree, {srange})",
        fontsize=15.5, fontweight="bold", y=0.980)
    badge1 = (
        f"agreement: Jaccard {row['jaccard']:.2f}  (UM purity {row['purity_um']:.2f} / "
        f"DANT purity {row['purity_dant']:.2f})        "
        f"waveform shape r = {wave_r:.2f}        "
        f"probe-depth range {depth_range:.0f} µm")
    badge2 = (
        f"held-out ISI r = {row['matched_isi_corr']:.2f}  "
        f"(> {row['matched_isi_pctile']*100:.0f}% of unrelated pairs; cohort AUC {cohort_auc:.2f})"
        f"        span: {n2e}")
    tiers = (f"curation tiers — UnitMatch: {row['um_tier']}   DANT per-link: {row['dant_tier']}"
             f"   DANT whole-track biophysical: {row['dant_composite']}")
    fig.text(0.06, 0.947, badge1, fontsize=10.5, color="#1a1a1a")
    fig.text(0.06, 0.925, badge2, fontsize=10.5, color="#1a1a1a")
    fig.text(0.06, 0.903, tiers, fontsize=9.5, color="#666666", style="italic")

    # (0,0) waveform overlay
    ax_wave = fig.add_subplot(gs[0, 0])
    for tr, c in zip(traces, colors):
        ax_wave.plot(tr, color=c, lw=1.2, alpha=0.85)
    ax_wave.set_title(f"Spike waveform, every session\n(peak channel; shape r = {wave_r:.2f})",
                      fontsize=12.5)
    ax_wave.set_xlabel("sample"); ax_wave.set_ylabel("µV")
    ax_wave.spines[["top", "right"]].set_visible(False)
    # session-order colorbar (early→late) in the empty top-right header gap
    cax = fig.add_axes([0.80, 0.905, 0.145, 0.013])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["early", "late"])
    cb.ax.tick_params(labelsize=7.5); cb.set_label("session order (waveform / ISI colours)", fontsize=7.5)

    # (0,1) held-out ISI overlay
    ax = fig.add_subplot(gs[0, 1])
    n_isi = 0
    for sk, c in zip(sess_list, [colors[valid_sess.index(s)] if s in valid_sess
                                 else (0.6, 0.6, 0.6, 1) for s in sess_list]):
        h = holdout.get((sk, ks_of[sk]))
        if h is None or not np.all(np.isfinite(h)):
            continue
        ax.plot(ISI_CENTERS, h, color=c, lw=1.1, alpha=0.8); n_isi += 1
    ax.set_xscale("log")
    ax.set_title(f"Inter-spike-interval fingerprint\n(held-out spikes, {n_isi} sessions)",
                 fontsize=12.5)
    ax.set_xlabel("ISI (s)"); ax.set_ylabel("prob."); ax.spines[["top", "right"]].set_visible(False)

    # (0,2) held-out ISI vs population
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(nonmatched, bins=40, color="#cccccc", density=True,
            label="unrelated pairs (null)")
    mval = float(row["matched_isi_corr"])
    ax.axvline(mval, color="#d7301f", lw=2.5,
               label=f"this neuron (r={mval:.2f})")
    ax.set_title(f"ISI match vs population\n(cohort AUC {cohort_auc:.2f}; "
                 f"beats {row['matched_isi_pctile']*100:.0f}% of null)", fontsize=12.5)
    ax.set_xlabel("cross-session ISI correlation"); ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)

    # (1,0)/(1,1) footprint first vs last
    for col, idx, lab in [(0, 0, "first"), (1, len(valid_sess) - 1, "last")]:
        ax = fig.add_subplot(gs[1, col])
        snip = footprints[idx]
        ax.imshow(snip.T, aspect="auto", cmap="RdBu_r", origin="lower",
                  vmin=-np.abs(snip).max(), vmax=np.abs(snip).max())
        ax.set_title(f"Footprint — {lab} session\n{valid_sess[idx]} "
                     f"({stage_of.get(valid_sess[idx],'?')})", fontsize=12.5)
        ax.set_xlabel("sample"); ax.set_ylabel("channel (near peak)")

    # (1,2) depth trajectory
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(range(len(valid_sess)), depths, "-o", color="#238b45", lw=1.5, ms=5)
    ax.set_title(f"Probe depth stability\n(range {np.nanmax(depths)-np.nanmin(depths):.0f} µm)",
                 fontsize=12.5)
    ax.set_xlabel("session (early → late)"); ax.set_ylabel("depth on probe (µm)")
    ax.spines[["top", "right"]].set_visible(False)

    # (2,:) two-tracker agreement strip
    ax = fig.add_subplot(gs[2, :])
    um_s = um_sessions.get(um_uid, set())
    dant_s = dant_sessions.get(dant_uid, set())
    agreed = set(sess_list)
    union = _chron(um_s | dant_s)
    x = np.arange(len(union))
    for xi, sk in zip(x, union):
        ax.add_patch(plt.Rectangle((xi - 0.45, 1.55), 0.9, 0.4,
                     color=STAGE_COLORS.get(stage_of.get(sk, "Unknown"), "#eeeeee")))
        if sk in um_s:
            ax.plot(xi, 1.0, "s", ms=13, color=("#d7301f" if sk in agreed else "#fdae6b"))
        if sk in dant_s:
            ax.plot(xi, 0.4, "s", ms=13, color=("#d7301f" if sk in agreed else "#9ecae1"))
    ax.set_xlim(-0.7, len(union) - 0.3); ax.set_ylim(0.0, 2.1)
    ax.set_yticks([0.4, 1.0, 1.75]); ax.set_yticklabels(["DANT", "UnitMatch", "stage"])
    ax.set_xticks(x); ax.set_xticklabels(union, rotation=90, fontsize=6.5)
    ax.set_title(f"Two independent trackers across all sessions "
                 f"(red = both agree, {len(agreed)} sessions; "
                 f"UM spans {len(um_s)}, DANT spans {len(dant_s)})", fontsize=12.5)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)

    CAND_DIR.mkdir(parents=True, exist_ok=True)
    out = CAND_DIR / f"consensus_um{um_uid}_dant{dant_uid}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out.name}  (waveforms {len(valid_sess)}, ISI {n_isi}, wave_r {wave_r:.2f})")
    return {"um_uid": um_uid, "dant_uid": dant_uid, "n_agree": int(row["n_agree"]),
            "wave_r": round(wave_r, 3), "matched_isi_corr": float(row["matched_isi_corr"]),
            "out": out.name}


# ---------------------------------------------------------------- cohort summary
def render_summary(cohort, val):
    fig = plt.figure(figsize=(14, 4.2))
    gs = gridspec.GridSpec(1, 3, wspace=0.32, left=0.06, right=0.97, top=0.82, bottom=0.16)
    fig.suptitle("UM ∩ DANT consensus cohort — BG_046 medial striatum "
                 f"({len(cohort)} neurons tracked by BOTH trackers across ≥2 sessions)",
                 fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(cohort["n_agree"], bins=range(2, int(cohort["n_agree"].max()) + 2),
            color="#238b45", edgecolor="white")
    ax.set_title("agreed span (sessions)", fontsize=12.5)
    ax.set_xlabel("# sessions both trackers agree"); ax.set_ylabel("# neurons")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    m = cohort["matched_isi_corr"].dropna()
    ax.hist(m, bins=30, color="#74c476", edgecolor="white")
    ax.axvline(val["nonmatched_corr_mean"], color="#999999", ls="--",
               label=f"null mean {val['nonmatched_corr_mean']:.2f}")
    ax.set_title(f"held-out ISI match  (cohort AUC {val['auc_matched_vs_nonmatched']:.2f})",
                 fontsize=12.5)
    ax.set_xlabel("cross-session ISI correlation"); ax.set_ylabel("# neurons")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    counts = [(cohort["n_agree"] >= k).sum() for k in [2, 3, 5, 7, 10]]
    ax.bar([str(k) for k in [2, 3, 5, 7, 10]], counts, color="#238b45")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
    n2e = int(cohort["naive_to_expert"].sum()); l2e = int(cohort["learning_to_expert"].sum())
    ax.set_title(f"tracked ≥ k sessions\n(Learning→Expert: {l2e}; strict Naive→Expert: {n2e})",
                 fontsize=12.5)
    ax.set_xlabel("k (min agreed sessions)"); ax.set_ylabel("# neurons")
    ax.spines[["top", "right"]].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "cohort_summary.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cohort = pd.read_csv(CACHE / "consensus_cohort.csv")
    members = pd.read_csv(CACHE / "consensus_members.csv", dtype={"session_key": str})
    with open(CACHE / "isi_holdout.pkl", "rb") as f:
        holdout = pickle.load(f)
    nonmatched = np.load(CACHE / "nonmatched_corrs.npy")
    val = json.load(open(CACHE / "isi_validation.json"))
    stage_of = _stage_map()
    um_sessions = _reg_sessions(UM_REG, "global_uid")
    dant_sessions = _reg_sessions(DANT_REG, "dant_uid")
    auc = val["auc_matched_vs_nonmatched"]

    uids = args.uids if args.uids else DEFAULT_UIDS
    print(f"rendering {len(uids)} consensus candidates: {uids}")
    done = []
    for u in uids:
        r = render_candidate(u, cohort, members, holdout, nonmatched, stage_of,
                             um_sessions, dant_sessions, auc)
        if r:
            done.append(r)
    if done:
        pd.DataFrame(done).to_csv(CAND_DIR / "rendered_stats.csv", index=False)
    render_summary(cohort, val)
    print(f"\nrendered {len(done)} candidate figures + cohort summary")


if __name__ == "__main__":
    main()
