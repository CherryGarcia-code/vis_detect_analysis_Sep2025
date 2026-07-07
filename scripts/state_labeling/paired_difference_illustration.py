"""Why overlapping PSTH error bands can still be a significant difference: it's a
PAIRED (within-unit) effect. BG_046 Expert, 12 sessions, Change-responsive units.

Panel A: per-unit scatter, StimSens (x) vs Impulsive (y) early-window evoked z.
         The two marginal spreads overlap hugely, but most dots sit ABOVE the
         unity line -> within-unit, Impulsive > StimSens.
Panel B: histogram of the per-unit paired difference (Impulsive - StimSens),
         shifted positive of zero. % of units > 0 + mean annotated.

The authoritative test remains the mixed-effects model (unit nested in session,
p=1.4e-9); this figure just makes the paired logic visible.
Output: figures/state_labeler/BG_046/paired_difference_illustration.png
"""

import os
import gc

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import STATE_LABEL_COLORS
from visdetect.suite.loader import load_session, load_staging_manifest
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.utils import build_population_tensor

setup_style()

SUBJECT = "BG_046"
STATES = ["Impulsive", "StimSens"]
BIN = 0.01
WINDOW = (-0.5, 1.0)
BASELINE_WIN = (-0.4, -0.05)
EARLY_WIN = (0.0, 0.25)
GO_SET = {1.25, 1.35, 1.5, 2.0, 4.0}
MIN_UNITS = 8
MIN_TRIALS_STATE = 8

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG_DIR = os.path.join(_REPO, "data", "cache", "state_tags", SUBJECT)
RESP_CACHE = os.path.join(_REPO, "data", "cache", "state_labeling", "responsiveness_all_sessions.csv")
OUT_DIR = os.path.join(_REPO, "FIGURES", "state_labeler", SUBJECT)


def session_unit_evoked(sname, resp_all):
    """Per-unit early-window evoked z for each state (go-Hit). Returns dict{state:array} or None."""
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
    early_bins = (bc >= EARLY_WIN[0]) & (bc < EARLY_WIN[1])
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
        out[s] = z[early_bins, :].mean(axis=0)     # per-unit evoked
    del tensor; gc.collect()
    return out


def main():
    resp_all = pd.read_csv(RESP_CACHE)
    man = load_staging_manifest(qc_only=False)
    sess_list = [str(s) for s in man.loc[man["stage"] == "Expert", "session_name"]]
    imp, ss = [], []
    for sname in sess_list:
        r = session_unit_evoked(sname, resp_all)
        if r is None:
            continue
        imp.append(r["Impulsive"]); ss.append(r["StimSens"])
        print(f"  {str(sname).zfill(8)}: ok")
    imp = np.concatenate(imp); ss = np.concatenate(ss)
    diff = imp - ss
    nU = len(diff)
    frac_pos = float(np.mean(diff > 0))
    W, p = wilcoxon(imp, ss)
    print(f"\n[paired] {nU} units; {100*frac_pos:.0f}% Impulsive>StimSens; "
          f"mean diff={diff.mean():.3f} z; Wilcoxon p={p:.2e}")

    fig = plt.figure(figsize=(11, 4.6))
    gs = gridspec.GridSpec(1, 2, wspace=0.28, left=0.08, right=0.97, top=0.84, bottom=0.16)

    # A: scatter vs unity line
    axA = fig.add_subplot(gs[0, 0])
    lim = [min(ss.min(), imp.min()) - 0.02, max(ss.max(), imp.max()) + 0.02]
    axA.plot(lim, lim, color="k", lw=1.0, ls="--", alpha=0.6, zorder=1)
    above = imp > ss
    axA.scatter(ss[above], imp[above], s=22, color=STATE_LABEL_COLORS["Impulsive"],
                alpha=0.6, edgecolors="none", zorder=3, label=f"Impulsive > StimSens ({100*frac_pos:.0f}%)")
    axA.scatter(ss[~above], imp[~above], s=22, color=STATE_LABEL_COLORS["StimSens"],
                alpha=0.6, edgecolors="none", zorder=3, label=f"StimSens ≥ Impulsive ({100*(1-frac_pos):.0f}%)")
    axA.set_xlim(lim); axA.set_ylim(lim); axA.set_aspect("equal")
    axA.set_xlabel("StimSens evoked z (per unit)")
    axA.set_ylabel("Impulsive evoked z (per unit)")
    axA.legend(frameon=False, fontsize=8.5, loc="upper left")
    axA.set_title(f"A. Each dot = one unit ({nU} units)\nmost sit above the line = Impulsive bigger",
                  fontsize=10, fontweight="bold")

    # B: histogram of paired difference
    axB = fig.add_subplot(gs[0, 1])
    axB.hist(diff, bins=40, color="#7a4fb5", alpha=0.8, edgecolor="white", linewidth=0.3)
    axB.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    axB.axvline(diff.mean(), color="#c0392b", lw=2.0,
                label=f"mean = {diff.mean():+.3f} z")
    axB.set_xlabel("Per-unit difference:  Impulsive − StimSens (z)")
    axB.set_ylabel("Number of units")
    axB.legend(frameon=False, fontsize=9, loc="upper right")
    axB.set_title(f"B. The paired difference is shifted positive\n"
                  f"{100*frac_pos:.0f}% of units > 0  ·  mixedLM p=1.4e-9",
                  fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Overlapping bands, but a real PAIRED effect — {SUBJECT}, Expert "
        f"({nU} units, early window)\nmarginal spreads overlap; the within-unit "
        "difference is consistent",
        fontsize=12, fontweight="bold", y=0.99)
    save_figure(fig, "paired_difference_illustration", f"state_labeler/{SUBJECT}")
    plt.close(fig)
    print("[paired] done.")


if __name__ == "__main__":
    main()
