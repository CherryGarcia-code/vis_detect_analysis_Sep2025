"""Cross-session replication: state-conditioned sensory response across Expert sessions.

Replicates the single-session result (engaged 'StimSens' state shows the smallest
& slowest-onset change-evoked striatal response; Impulsive/Abort larger & faster)
with **session as the replication unit**.

Per session (Change_ON-responsive units, go-Hit trials, per-unit shared-baseline z):
  * evoked z in early window (0-250 ms), per state  -> session mean across units
  * onset latency to 50% of each state's OWN peak    (amplitude-normalized; fixes the
        fixed-threshold confound)
  * absolute pre-change baseline FR (Hz), per state   (tests 'already sensitive')
  * median Hit RT, per state                          (tests 'trigger-happy')
  * Hit vs Miss evoked z (go trials)

Across sessions (paired, n = sessions):
  * Friedman over the 3 states (evoked z) + Wilcoxon StimSens-vs-others (Holm)
  * Wilcoxon StimSens-vs-Impulsive onset latency
  * Friedman baseline FR (expected n.s.)
  * Wilcoxon Hit-vs-Miss contrast vs 0
Effect sizes: Kendall's W, matched-pairs r. Separate questions not cross-corrected.

Usage:  py population_state_sensory_xsession.py [--sessions 17092025,16092025]
Outputs:
  figures/state_labeler/BG_046/xsession_state_sensory.png
  figures/state_labeler/BG_046/xsession_state_sensory_session_metrics.csv
  figures/state_labeler/BG_046/xsession_state_sensory_stats.csv
"""

import os
import gc
import sys
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import STATE_LABEL_COLORS
from visdetect.suite.loader import load_session, load_staging_manifest
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.utils import build_population_tensor, smooth_psth
from visdetect.analysis.align import get_event_times_by_trial
from visdetect.analysis.constants import DEFAULT_SIGMA_MS

setup_style()

SUBJECT = "BG_046"
STATES = ["Impulsive", "StimSens", "Abort"]
BIN = 0.01
SIGMA_MS = DEFAULT_SIGMA_MS
WINDOW = (-0.5, 1.0)
BASELINE_WIN = (-0.4, -0.05)
SENSORY_WIN = (0.0, 0.25)
GO_SET = {1.25, 1.35, 1.5, 2.0, 4.0}
MIN_UNITS = 8
MIN_TRIALS_STATE = 8

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG_DIR = os.path.join(_REPO, "data", "cache", "state_tags", SUBJECT)
RESP_CACHE = os.path.join(_REPO, "analysis_suite", "cache", "responsiveness_all_sessions.csv")
OUT_DIR = os.path.join(_REPO, "analysis_suite", "figures", "state_labeler", SUBJECT)


def wilcoxon_paired(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = x - y
    n = int(np.sum(d != 0))
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n
    res = wilcoxon(x, y, method="approx", zero_method="wilcox")
    z = float(res.zstatistic)
    return float(res.statistic), float(res.pvalue), z, z / np.sqrt(n), n


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx]); adj[idx] = min(run, 1.0)
    return adj


def omnibus_states(arrs, nS, k):
    """State omnibus: Friedman for k>=3, else the single paired Wilcoxon.
    Returns (stat_name, stat_value, p, effect_name, effect_value)."""
    if k >= 3:
        chi2, p = friedmanchisquare(*arrs)
        return "chi2(Friedman)", round(chi2, 3), p, "Kendall_W", round(chi2 / (nS * (k - 1)), 3)
    Wst, p, z, r, n = wilcoxon_paired(arrs[0], arrs[1])
    return "W(Wilcoxon)", round(Wst, 1), p, "r_matched", round(r, 3)


def latency_to_half_peak(ztrace, bc):
    """Latency (s) to 50% of the post-change peak; NaN if no positive peak."""
    post = bc >= 0
    zp = ztrace[post]; tp = bc[post]
    pk = np.argmax(zp)
    if zp[pk] <= 0:
        return np.nan
    half = 0.5 * zp[pk]
    cross = np.where(zp[:pk + 1] >= half)[0]
    return float(tp[cross[0]]) if len(cross) else np.nan


def process_session(sname, resp_all):
    # Normalize to canonical 8-digit DDMMYYYY id. The staging manifest drops the
    # leading zero for Sept 1-9 (e.g. '1092025'), but state-tag files and pkls are
    # zero-padded ('01092025.csv'); the responsiveness cache may use either form.
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
    hit_t = np.array(get_event_times_by_trial(sess, "Hit"), float)
    chg_t = np.array(get_event_times_by_trial(sess, "Change_ON"), float)

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
    sens_bins = (bc >= SENSORY_WIN[0]) & (bc < SENSORY_WIN[1])
    nU = len(uids)

    # per-unit shared baseline (all go trials)
    bm = np.array([tensor[go][:, base_bins, j].ravel().mean() for j in range(nU)])
    bs = np.array([max(tensor[go][:, base_bins, j].ravel().std(), 1e-6) for j in range(nU)])

    out = {"session": sid8, "n_units": nU}
    # require each state present in hit-go
    for s in STATES:
        m = hit_go & (st == s)
        out[f"n_{s}"] = int(m.sum())
    if any(out[f"n_{s}"] < MIN_TRIALS_STATE for s in STATES):
        return None  # insufficient state coverage

    for s in STATES:
        m = hit_go & (st == s)
        mean_tr = tensor[m].mean(axis=0)                       # (bins, units)
        ztr = (mean_tr - bm[None, :]) / bs[None, :]            # (bins, units)
        evoked = ztr[sens_bins, :].mean(axis=0)               # per unit
        out[f"evoked_{s}"] = float(np.mean(evoked))
        pop_z = ztr.mean(axis=1)                               # population mean trace
        out[f"onset_{s}"] = latency_to_half_peak(pop_z, bc)
        out[f"baseFR_{s}"] = float(np.mean([tensor[m][:, base_bins, j].mean()
                                            for j in range(nU)]))
        # RT
        rts = [hit_t[t] - chg_t[t] for t in vt[m]
               if np.isfinite(hit_t[t]) and np.isfinite(chg_t[t])]
        out[f"RT_{s}"] = float(np.median(rts)) if rts else np.nan

    # Hit vs Miss (go), per-unit evoked z
    def evoked_for(mask):
        if mask.sum() < MIN_TRIALS_STATE:
            return np.nan
        mean_tr = tensor[mask].mean(axis=0)
        ztr = (mean_tr - bm[None, :]) / bs[None, :]
        return float(np.mean(ztr[sens_bins, :].mean(axis=0)))
    out["evoked_Hit"] = evoked_for(go & (ocl == "hit"))
    out["evoked_Miss"] = evoked_for(go & (ocl == "miss"))
    del tensor; gc.collect()
    return out


def main():
    global STATES
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=None, help="comma list to override (smoke test)")
    ap.add_argument("--states", default=",".join(STATES),
                    help="comma list of states to require/compare (e.g. Impulsive,StimSens)")
    args = ap.parse_args()
    STATES = [s.strip() for s in args.states.split(",")]
    suffix = "" if len(STATES) >= 3 else "_" + "_".join(STATES).lower()
    print(f"[xsession] states: {STATES}")

    resp_all = pd.read_csv(RESP_CACHE)
    if args.sessions:
        sess_list = args.sessions.split(",")
    else:
        man = load_staging_manifest(qc_only=False)
        sess_list = [str(s) for s in man.loc[man["stage"] == "Expert", "session_name"]]
    print(f"[xsession] candidate Expert sessions: {len(sess_list)}")

    recs = []
    for sname in sess_list:
        r = process_session(sname, resp_all)
        if r is None:
            print(f"  {sname}: skipped (coverage)")
            continue
        print(f"  {sname}: {r['n_units']} units  evoked "
              + " ".join(f"{s}={r[f'evoked_{s}']:+.2f}" for s in STATES))
        recs.append(r)

    df = pd.DataFrame(recs)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, f"xsession_state_sensory{suffix}_session_metrics.csv"), index=False)
    nS = len(df)
    print(f"\n[xsession] usable sessions: {nS}")
    if nS < 4:
        print("  too few sessions for paired stats; metrics CSV written.")
        return

    # -------- across-session stats (session = unit) --------
    rows = []
    def add(test, sname, sval, p, esname, esval, n, interp, notes=""):
        rows.append(dict(test=test, statistic_name=sname, statistic_value=sval,
                         p_value=p, effect_size_name=esname, effect_size_value=esval,
                         n_sessions=n, interpretation=interp, notes=notes))

    ev = {s: df[f"evoked_{s}"].values for s in STATES}
    o_name, o_val, p, o_es_name, W = omnibus_states([ev[s] for s in STATES], nS, len(STATES))
    add("evoked_state_effect", o_name, o_val, p, o_es_name, W,
        nS, f"{'sig' if p < 0.05 else 'n.s.'} state effect on evoked z (across sessions)",
        "means " + " ".join(f"{s}={np.mean(ev[s]):.3f}" for s in STATES))
    pvals, recs2 = [], []
    for a, b in combinations(STATES, 2):
        Wst, pp, z, r, n = wilcoxon_paired(ev[a], ev[b]); pvals.append(pp); recs2.append((a, b, Wst, pp, r, n))
    adj = holm(pvals)
    for (a, b, Wst, pp, r, n), pa in zip(recs2, adj):
        med = float(np.median(ev[a] - ev[b]))
        add(f"evoked_{a}_vs_{b}", "W(Wilcoxon)", round(Wst, 1), pp, "r_matched", round(r, 3), n,
            f"{a} {'>' if med > 0 else '<'} {b} ({'sig' if pa < 0.05 else 'n.s.'} Holm)",
            f"p_holm={pa:.4f}; median_diff={med:.3f}")

    # onset latency StimSens vs Impulsive
    Wo, po, zo, ro, no = wilcoxon_paired(df["onset_StimSens"].values, df["onset_Impulsive"].values)
    medo = float(np.nanmedian(df["onset_StimSens"].values - df["onset_Impulsive"].values))
    add("onset_StimSens_vs_Impulsive", "W(Wilcoxon)", round(Wo, 1), po, "r_matched", round(ro, 3), no,
        f"StimSens onset {'later' if medo > 0 else 'earlier'} than Impulsive "
        f"({'sig' if (po == po and po < 0.05) else 'n.s.'})",
        f"median_diff={medo:.3f}s")

    # baseline FR state effect (expect n.s.)
    b_name, b_val, bp, b_es_name, b_es = omnibus_states(
        [df[f"baseFR_{s}"].values for s in STATES], nS, len(STATES))
    add("baseline_FR_state_effect", b_name, b_val, bp, b_es_name, b_es, nS,
        f"{'sig' if bp < 0.05 else 'n.s.'} baseline-FR state effect",
        "means " + " ".join(f"{s}={np.mean(df[f'baseFR_{s}']):.2f}Hz" for s in STATES))

    # RT StimSens vs Impulsive
    Wr, pr, zr, rr, nr = wilcoxon_paired(df["RT_StimSens"].values, df["RT_Impulsive"].values)
    medr = float(np.nanmedian(df["RT_StimSens"].values - df["RT_Impulsive"].values))
    add("RT_StimSens_vs_Impulsive", "W(Wilcoxon)", round(Wr, 1), pr, "r_matched", round(rr, 3), nr,
        f"StimSens RT {'slower' if medr > 0 else 'faster'} than Impulsive "
        f"({'sig' if (pr == pr and pr < 0.05) else 'n.s.'})", f"median_diff={medr:.3f}s")

    # Hit vs Miss
    Wh, ph, zh, rh, nh = wilcoxon_paired(df["evoked_Hit"].values, df["evoked_Miss"].values)
    medh = float(np.nanmedian(df["evoked_Hit"].values - df["evoked_Miss"].values))
    add("hit_vs_miss", "W(Wilcoxon)", round(Wh, 1), ph, "r_matched", round(rh, 3), nh,
        f"Hit {'>' if medh > 0 else '<'} Miss ({'sig' if (ph == ph and ph < 0.05) else 'n.s.'})",
        f"median_diff={medh:.3f}")

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(OUT_DIR, f"xsession_state_sensory{suffix}_stats.csv"), index=False)
    print("\n=== ACROSS-SESSION STATS ===")
    print(stats_df[["test", "statistic_value", "p_value", "effect_size_value",
                    "interpretation"]].to_string(index=False))

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.28,
                           left=0.09, right=0.97, top=0.89, bottom=0.09)
    xs = np.arange(len(STATES))

    def spaghetti(ax, prefix, ylabel, title, ylog=False):
        for _, r in df.iterrows():
            ax.plot(xs, [r[f"{prefix}_{s}"] for s in STATES], color="0.75", lw=0.8,
                    alpha=0.6, zorder=1)
        means = [np.nanmean(df[f"{prefix}_{s}"]) for s in STATES]
        for i, s in enumerate(STATES):
            ax.scatter([xs[i]], [means[i]], s=90, color=STATE_LABEL_COLORS[s],
                       edgecolors="k", linewidths=1.0, zorder=4)
        ax.plot(xs, means, color="0.2", lw=1.8, zorder=3)
        ax.set_xticks(xs); ax.set_xticklabels(STATES)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.97, 0.04, f"{nS} sessions", transform=ax.transAxes,
                fontsize=8, color="gray", ha="right")

    axA = fig.add_subplot(gs[0, 0])
    spaghetti(axA, "evoked", "Evoked z (0–250 ms)", "A.  Evoked response by state")
    axA.text(0.02, 0.98, f"{o_name.split('(')[0]} p={p:.1e}, eff={W:.2f}", transform=axA.transAxes,
             fontsize=8, va="top", bbox=dict(boxstyle="round", fc="0.95", ec="none", alpha=0.85))

    axB = fig.add_subplot(gs[0, 1])
    spaghetti(axB, "onset", "Onset latency to 50% peak (s)", "B.  Response onset by state")
    axB.text(0.02, 0.98, f"StimSens vs Impulsive\np={po:.1e}, r={abs(ro):.2f}",
             transform=axB.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", fc="0.95", ec="none", alpha=0.85))

    axC = fig.add_subplot(gs[1, 0])
    spaghetti(axC, "baseFR", "Baseline FR (Hz)", "C.  Pre-change baseline by state")
    axC.text(0.02, 0.98, f"{b_name.split('(')[0]} p={bp:.2f} ({'n.s.' if bp>=0.05 else 'sig'})",
             transform=axC.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", fc="0.95", ec="none", alpha=0.85))

    axD = fig.add_subplot(gs[1, 1])
    diff = df["evoked_Hit"].values - df["evoked_Miss"].values
    axD.axhline(0, color="k", lw=0.6, alpha=0.4)
    jx = np.random.default_rng(42).normal(0, 0.04, size=nS)
    axD.scatter(jx, diff, s=40, color="#4CAF50", edgecolors="white", linewidths=0.4, zorder=3)
    axD.scatter([0], [np.nanmean(diff)], s=140, color="#1b5e20", edgecolors="k",
                linewidths=1.2, marker="D", zorder=4)
    axD.set_xlim(-0.5, 0.5); axD.set_xticks([0]); axD.set_xticklabels(["Hit − Miss"])
    axD.set_ylabel("Δ evoked z (Hit − Miss)")
    axD.set_title("D.  Hit vs Miss (per session)", fontsize=11, fontweight="bold")
    axD.text(0.5, 0.04, f"Wilcoxon p={ph:.1e}, r={abs(rh):.2f}", transform=axD.transAxes,
             ha="center", fontsize=8, bbox=dict(boxstyle="round", fc="0.95", ec="none", alpha=0.85))

    fig.suptitle(
        f"Cross-session replication — state-conditioned sensory response ({SUBJECT}, Expert)\n"
        f"{nS} sessions · states: {', '.join(STATES)} · session as replication unit · go-Hit, early window",
        fontsize=12.5, fontweight="bold", y=0.985)
    save_figure(fig, f"xsession_state_sensory{suffix}", f"state_labeler/{SUBJECT}")
    plt.close(fig)
    print("[xsession] done.")


if __name__ == "__main__":
    main()
