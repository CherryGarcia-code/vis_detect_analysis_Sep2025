"""Illustrate 'gain': Impulsive vs StimSens change response is the same time-course,
scaled up — not shifted earlier. BG_046 Expert, pooled across sessions.

Three panels (pedagogical):
  A. Grand population PSTH (z), Impulsive vs StimSens overlaid (mean +/- SEM across
     sessions). Same onset/peak timing, taller for Impulsive = gain.
  B. Gain demo: multiply the StimSens curve by a single best-fit factor g; it lands
     on the Impulsive curve. One curve is a scaled copy of the other => pure gain.
  C. The Impulsive-minus-StimSens gap in the early vs late window (per session):
     the gap doesn't grow over time => uniform scaling, not a timing difference
     (the flat state x window interaction, mixedLM p=0.84).

Pools the same 12 qualifying Expert sessions / Change-responsive units / go-Hit
trials as the cross-session + mixed-effects analyses. Caches per-session traces.
Output: figures/state_labeler/BG_046/gain_illustration.png
"""

import os
import gc

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import STATE_LABEL_COLORS
from visdetect.suite.loader import load_session, load_staging_manifest
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.utils import build_population_tensor, smooth_psth
from visdetect.analysis.constants import DEFAULT_SIGMA_MS

setup_style()

SUBJECT = "BG_046"
STATES = ["Impulsive", "StimSens"]
BIN = 0.01
SIGMA_MS = DEFAULT_SIGMA_MS
WINDOW = (-0.5, 1.0)
BASELINE_WIN = (-0.4, -0.05)
EARLY_WIN = (0.0, 0.25)
LATE_WIN = (0.25, 0.6)
FIT_WIN = (0.0, 0.6)          # window over which to fit the gain factor
GO_SET = {1.25, 1.35, 1.5, 2.0, 4.0}
MIN_UNITS = 8
MIN_TRIALS_STATE = 8

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG_DIR = os.path.join(_REPO, "data", "cache", "state_tags", SUBJECT)
RESP_CACHE = os.path.join(_REPO, "data", "cache", "state_labeling", "responsiveness_all_sessions.csv")
OUT_DIR = os.path.join(_REPO, "FIGURES", "state_labeler", SUBJECT)
TRACE_CACHE = os.path.join(_REPO, "data", "cache", "state_labeling", "state_gain_traces.npz")


def session_traces(sname, resp_all):
    """Per-session population-mean z trace for each state (go-Hit). Returns (bc, {state: trace}) or None."""
    sid8 = str(sname).zfill(8)
    resp = resp_all[(resp_all["session_name"].astype(str).str.zfill(8) == sid8)
                    & (resp_all["is_responsive"])]
    uids = [int(c) for c in resp["cluster_id"].tolist()]
    if len(uids) < MIN_UNITS:
        return None
    tag_csv = os.path.join(TAG_DIR, f"{sid8}.csv")
    if not os.path.exists(tag_csv):
        return None
    try:
        sess = load_session(sid8)
    except FileNotFoundError:
        return None
    present = {c.cluster_id for c in sess.clusters}
    uids = [u for u in uids if u in present]
    if len(uids) < MIN_UNITS:
        del sess; gc.collect(); return None

    tags = pd.read_csv(tag_csv)
    state_of = dict(zip(tags["trial_idx"].astype(int), tags["state_label"]))
    csize = {i: float(getattr(t, "change_size", np.nan)) for i, t in enumerate(sess.trials)}
    oc = {i: (getattr(t, "trialoutcome", "") or "").lower() for i, t in enumerate(sess.trials)}

    tensor, bc, vt = build_population_tensor(
        sess, uids, event_name="Change_ON", window=WINDOW, bin_size=BIN,
        outcome_filter={"Hit", "Miss"})
    del sess; gc.collect()
    vt = np.array([int(t) for t in vt])
    st = np.array([state_of.get(t) for t in vt])
    sz = np.array([csize.get(t, np.nan) for t in vt])
    ocl = np.array([oc.get(t) for t in vt])
    go = np.array([s in GO_SET for s in sz])
    hit_go = go & (ocl == "hit")
    base_bins = (bc >= BASELINE_WIN[0]) & (bc < BASELINE_WIN[1])
    nU = len(uids)
    if any(int((hit_go & (st == s)).sum()) < MIN_TRIALS_STATE for s in STATES):
        return None

    bm = np.array([tensor[go][:, base_bins, j].ravel().mean() for j in range(nU)])
    bs = np.array([max(tensor[go][:, base_bins, j].ravel().std(), 1e-6) for j in range(nU)])
    out = {}
    for s in STATES:
        m = hit_go & (st == s)
        mt = tensor[m].mean(axis=0)
        z = (mt - bm[None, :]) / bs[None, :]
        out[s] = z.mean(axis=1)            # population mean z trace
    del tensor; gc.collect()
    return bc, out


def main(force=False):
    if os.path.exists(TRACE_CACHE) and not force:
        d = np.load(TRACE_CACHE, allow_pickle=True)
        bc = d["bc"]
        traces = {s: d[s] for s in STATES}   # (n_sessions, n_bins)
        print(f"[gain] loaded cache: {traces[STATES[0]].shape[0]} sessions")
    else:
        resp_all = pd.read_csv(RESP_CACHE)
        man = load_staging_manifest(qc_only=False)
        sess_list = [str(s) for s in man.loc[man["stage"] == "Expert", "session_name"]]
        acc = {s: [] for s in STATES}
        bc = None
        for sname in sess_list:
            r = session_traces(sname, resp_all)
            if r is None:
                continue
            bc, out = r
            for s in STATES:
                acc[s].append(out[s])
            print(f"  {str(sname).zfill(8)}: ok")
        traces = {s: np.vstack(acc[s]) for s in STATES}
        os.makedirs(os.path.dirname(TRACE_CACHE), exist_ok=True)
        np.savez(TRACE_CACHE, bc=bc, **traces)
        print(f"[gain] computed + cached: {traces[STATES[0]].shape[0]} sessions")

    nS = traces[STATES[0]].shape[0]
    grand = {s: traces[s].mean(axis=0) for s in STATES}
    sem = {s: traces[s].std(axis=0) / np.sqrt(nS) for s in STATES}
    grand_s = {s: smooth_psth(grand[s], BIN, SIGMA_MS) for s in STATES}
    sem_s = {s: smooth_psth(sem[s], BIN, SIGMA_MS) for s in STATES}

    # gain factor g: g*StimSens ~= Impulsive over FIT_WIN (least squares through 0)
    fit = (bc >= FIT_WIN[0]) & (bc < FIT_WIN[1])
    imp, ss = grand_s["Impulsive"][fit], grand_s["StimSens"][fit]
    g = float(np.dot(imp, ss) / np.dot(ss, ss))

    # early/late gap per session
    early = (bc >= EARLY_WIN[0]) & (bc < EARLY_WIN[1])
    late = (bc >= LATE_WIN[0]) & (bc < LATE_WIN[1])
    gap_early = (traces["Impulsive"][:, early] - traces["StimSens"][:, early]).mean(axis=1)
    gap_late = (traces["Impulsive"][:, late] - traces["StimSens"][:, late]).mean(axis=1)

    # ----------------------------- figure -----------------------------
    fig = plt.figure(figsize=(15, 4.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.30, left=0.06, right=0.98, top=0.84, bottom=0.16)

    # A: overlay
    axA = fig.add_subplot(gs[0, 0])
    for s in STATES:
        axA.plot(bc, grand_s[s], color=STATE_LABEL_COLORS[s], lw=2.2, label=s, zorder=3)
        axA.fill_between(bc, grand_s[s] - sem_s[s], grand_s[s] + sem_s[s],
                         color=STATE_LABEL_COLORS[s], alpha=0.18, lw=0, zorder=2)
    axA.axvspan(*EARLY_WIN, color="0.85", alpha=0.5, zorder=0)
    axA.axvspan(*LATE_WIN, color="0.92", alpha=0.6, zorder=0)
    axA.axvline(0, color="k", ls="--", lw=0.9, alpha=0.6)
    axA.axhline(0, color="k", lw=0.5, alpha=0.3)
    axA.set_xlim(-0.4, 1.0)
    axA.set_xlabel("Time from change onset (s)"); axA.set_ylabel("Population z-score")
    axA.legend(frameon=False, fontsize=9, loc="upper left")
    axA.set_title("A. The response 'bump' is taller in Impulsive\n(same timing, more height)",
                  fontsize=10, fontweight="bold")
    axA.text(0.97, 0.05, f"{nS} sessions", transform=axA.transAxes, fontsize=8,
             color="gray", ha="right")

    # B: gain demo
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(bc, grand_s["Impulsive"], color=STATE_LABEL_COLORS["Impulsive"], lw=2.2,
             label="Impulsive", zorder=3)
    axB.plot(bc, grand_s["StimSens"], color=STATE_LABEL_COLORS["StimSens"], lw=1.4,
             alpha=0.6, label="StimSens (raw)", zorder=2)
    axB.plot(bc, g * grand_s["StimSens"], color="#222", lw=1.8, ls="--",
             label=f"StimSens × {g:.2f}", zorder=4)
    axB.axvline(0, color="k", ls="--", lw=0.9, alpha=0.6)
    axB.axhline(0, color="k", lw=0.5, alpha=0.3)
    axB.set_xlim(-0.4, 1.0)
    axB.set_xlabel("Time from change onset (s)"); axB.set_ylabel("Population z-score")
    axB.legend(frameon=False, fontsize=8.5, loc="upper left")
    axB.set_title(f"B. Scale StimSens by one number ({g:.2f})\n→ it lands on Impulsive = gain",
                  fontsize=10, fontweight="bold")

    # C: gap early vs late
    axC = fig.add_subplot(gs[0, 2])
    for i in range(nS):
        axC.plot([0, 1], [gap_early[i], gap_late[i]], color="0.75", lw=0.8, alpha=0.7, zorder=1)
    for k, (vals, lab) in enumerate([(gap_early, "early\n0–250 ms"), (gap_late, "late\n250–600 ms")]):
        axC.scatter(np.full(nS, k) + np.random.default_rng(1).normal(0, 0.04, nS), vals,
                    s=22, color="#7a4fb5", alpha=0.7, edgecolors="white", linewidths=0.3, zorder=3)
        axC.scatter([k], [np.mean(vals)], s=130, color="#4b2e83", edgecolors="k",
                    linewidths=1.2, marker="D", zorder=4)
    axC.axhline(0, color="k", lw=0.5, alpha=0.3)
    axC.set_xticks([0, 1]); axC.set_xticklabels(["early\n0–250 ms", "late\n250–600 ms"])
    axC.set_xlim(-0.5, 1.5)
    axC.set_ylabel("Impulsive − StimSens gap (z)")
    axC.set_title("C. The gap is the same early & late\n(uniform scaling, not earlier onset)",
                  fontsize=10, fontweight="bold")
    axC.text(0.5, 0.04, "state×window interaction p=0.84", transform=axC.transAxes,
             ha="center", fontsize=8, bbox=dict(boxstyle="round", fc="0.95", ec="none", alpha=0.85))

    fig.suptitle(
        f"What 'gain' means here — {SUBJECT}, Expert ({nS} sessions, Change-responsive units, go-Hit)",
        fontsize=12.5, fontweight="bold", y=0.99)
    save_figure(fig, "gain_illustration", f"state_labeler/{SUBJECT}")
    plt.close(fig)
    print("[gain] done.")


if __name__ == "__main__":
    main()
