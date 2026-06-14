"""Mixed-effects / unit-pooled re-test of the state-conditioned sensory response.

The session-level (n=12) test is conservative but underpowered for the dynamics,
RT, and Hit-vs-Miss contrasts. Here we pool every Change-responsive unit across the
qualifying Expert sessions with **unit nested in session** as random effects
(statsmodels MixedLM, REML) — power without pseudoreplication.

Per-unit metric: evoked firing as a shared-baseline z-score (one baseline per unit
from all go trials' pre-change window), in an early (0-250 ms) and a late
(250-600 ms) window. RT is trial-level (change->lick, go-Hit).

Models (state reference = StimSens; window reference = early; outcome reference = Miss):
  1. evoked_z ~ C(state)            | RE: session + unit(session)   -> state effect
  2. evoked_z ~ C(state)*C(window)  | RE: session + unit(session)   -> dynamics (interaction)
  3. evoked_z ~ C(outcome)          | RE: session + unit(session)   -> Hit vs Miss
  4. rt       ~ C(state)            | RE: session                   -> RT (trial-level)

Outputs:
  figures/state_labeler/BG_046/mixedeffects_state_sensory.png
  figures/state_labeler/BG_046/mixedeffects_state_sensory_stats.csv
"""

import os
import gc
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import STATE_LABEL_COLORS
from visdetect.suite.loader import load_session, load_staging_manifest
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.utils import build_population_tensor
from visdetect.analysis.align import get_event_times_by_trial

setup_style()
warnings.filterwarnings("ignore")

SUBJECT = "BG_046"
STATES = ["Impulsive", "StimSens"]
BIN = 0.01
WINDOW = (-0.5, 1.0)
BASELINE_WIN = (-0.4, -0.05)
EARLY_WIN = (0.0, 0.25)
LATE_WIN = (0.25, 0.6)
GO_SET = {1.25, 1.35, 1.5, 2.0, 4.0}
MIN_UNITS = 8
MIN_TRIALS_STATE = 8

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG_DIR = os.path.join(_REPO, "data", "cache", "state_tags", SUBJECT)
RESP_CACHE = os.path.join(_REPO, "analysis_suite", "cache", "responsiveness_all_sessions.csv")
OUT_DIR = os.path.join(_REPO, "analysis_suite", "figures", "state_labeler", SUBJECT)


def collect_session(sname, resp_all):
    """Return (unit_rows, hm_rows, rt_rows) for one session, or None if it fails coverage."""
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
    early_bins = (bc >= EARLY_WIN[0]) & (bc < EARLY_WIN[1])
    late_bins = (bc >= LATE_WIN[0]) & (bc < LATE_WIN[1])
    nU = len(uids)

    # coverage: each state needs enough go-Hit trials
    nstate = {s: int((hit_go & (st == s)).sum()) for s in STATES}
    if any(nstate[s] < MIN_TRIALS_STATE for s in STATES):
        return None

    bm = np.array([tensor[go][:, base_bins, j].ravel().mean() for j in range(nU)])
    bs = np.array([max(tensor[go][:, base_bins, j].ravel().std(), 1e-6) for j in range(nU)])

    def evoked(mask, bins):
        mt = tensor[mask].mean(axis=0)                       # (bins, units)
        z = (mt - bm[None, :]) / bs[None, :]
        return z[bins, :].mean(axis=0)                       # per unit

    unit_rows, hm_rows = [], []
    for j, cid in enumerate(uids):
        for s in STATES:
            m = hit_go & (st == s)
            ze = evoked(m, early_bins)[j]
            zl = evoked(m, late_bins)[j]
            unit_rows.append(dict(session=sid8, unit=f"{sid8}_{cid}", state=s,
                                  window="early", evoked_z=ze))
            unit_rows.append(dict(session=sid8, unit=f"{sid8}_{cid}", state=s,
                                  window="late", evoked_z=zl))
    # Hit vs Miss (go), early window
    mh, mm = go & (ocl == "hit"), go & (ocl == "miss")
    if mm.sum() >= MIN_TRIALS_STATE:
        zh, zmi = evoked(mh, early_bins), evoked(mm, early_bins)
        for j, cid in enumerate(uids):
            hm_rows.append(dict(session=sid8, unit=f"{sid8}_{cid}", outcome="Hit", evoked_z=zh[j]))
            hm_rows.append(dict(session=sid8, unit=f"{sid8}_{cid}", outcome="Miss", evoked_z=zmi[j]))

    rt_rows = []
    for ti in vt[hit_go]:
        s = state_of.get(int(ti))
        if s in STATES and np.isfinite(hit_t[ti]) and np.isfinite(chg_t[ti]):
            rt_rows.append(dict(session=sid8, state=s, rt=hit_t[ti] - chg_t[ti]))

    del tensor; gc.collect()
    return unit_rows, hm_rows, rt_rows


def fit_and_extract(model, terms):
    """Fit MixedLM and pull (coef, se, z, p, ci_lo, ci_hi) for named fixed-effect terms."""
    res = model.fit(reml=True, method="lbfgs")
    out = {}
    ci = res.conf_int()
    for label, term in terms.items():
        if term in res.params.index:
            out[label] = dict(coef=res.params[term], se=res.bse[term],
                              z=res.tvalues[term], p=res.pvalues[term],
                              lo=ci.loc[term, 0], hi=ci.loc[term, 1])
    return out, res


def main():
    resp_all = pd.read_csv(RESP_CACHE)
    man = load_staging_manifest(qc_only=False)
    sess_list = [str(s) for s in man.loc[man["stage"] == "Expert", "session_name"]]

    U, HM, RT = [], [], []
    used = []
    for sname in sess_list:
        r = collect_session(sname, resp_all)
        if r is None:
            continue
        u, hm, rt = r
        U += u; HM += hm; RT += rt
        used.append(str(sname).zfill(8))
        print(f"  {str(sname).zfill(8)}: +{len(set(d['unit'] for d in u))} units")
    udf = pd.DataFrame(U); hmdf = pd.DataFrame(HM); rtdf = pd.DataFrame(RT)
    n_units = udf["unit"].nunique(); n_sess = len(used)
    print(f"\n[mixed] {n_sess} sessions, {n_units} units, {len(rtdf)} go-Hit trials")

    rows = []
    def add(model_name, term, d, interp, notes=""):
        rows.append(dict(model=model_name, term=term, coef=round(d["coef"], 4),
                         se=round(d["se"], 4), z=round(d["z"], 3), p_value=d["p"],
                         ci_lo=round(d["lo"], 4), ci_hi=round(d["hi"], 4),
                         interpretation=interp, notes=notes))

    VC = {"unit": "0 + C(unit)"}
    impulse_term = "C(state, Treatment('StimSens'))[T.Impulsive]"

    # Model 1: state effect (early window)
    d1 = udf[udf["window"] == "early"]
    m1, r1 = fit_and_extract(
        smf.mixedlm("evoked_z ~ C(state, Treatment('StimSens'))", d1,
                    groups=d1["session"], vc_formula=VC, re_formula="1"),
        {"Impulsive_vs_StimSens": impulse_term})
    add("M1_state_early", "Impulsive-StimSens", m1["Impulsive_vs_StimSens"],
        f"Impulsive {'>' if m1['Impulsive_vs_StimSens']['coef'] > 0 else '<'} StimSens "
        f"({'sig' if m1['Impulsive_vs_StimSens']['p'] < 0.05 else 'n.s.'})")

    # Model 2: state x window interaction (dynamics)
    inter = "C(state, Treatment('StimSens'))[T.Impulsive]:C(window, Treatment('early'))[T.late]"
    m2, r2 = fit_and_extract(
        smf.mixedlm("evoked_z ~ C(state, Treatment('StimSens'))*C(window, Treatment('early'))",
                    udf, groups=udf["session"], vc_formula=VC, re_formula="1"),
        {"state_x_window": inter, "Impulsive_early": impulse_term})
    add("M2_state_x_window", "Impulsive:late", m2["state_x_window"],
        f"state effect {'grows' if m2['state_x_window']['coef'] > 0 else 'shrinks'} late "
        f"({'sig' if m2['state_x_window']['p'] < 0.05 else 'n.s.'} interaction)",
        "early Impulsive-StimSens coef={:.3f}".format(m2["Impulsive_early"]["coef"]))

    # Model 3: Hit vs Miss (early)
    hit_term = "C(outcome, Treatment('Miss'))[T.Hit]"
    m3, r3 = fit_and_extract(
        smf.mixedlm("evoked_z ~ C(outcome, Treatment('Miss'))", hmdf,
                    groups=hmdf["session"], vc_formula=VC, re_formula="1"),
        {"Hit_vs_Miss": hit_term})
    add("M3_hit_vs_miss_early", "Hit-Miss", m3["Hit_vs_Miss"],
        f"Hit {'>' if m3['Hit_vs_Miss']['coef'] > 0 else '<'} Miss "
        f"({'sig' if m3['Hit_vs_Miss']['p'] < 0.05 else 'n.s.'})")

    # Model 4: RT ~ state (trial-level)
    m4, r4 = fit_and_extract(
        smf.mixedlm("rt ~ C(state, Treatment('StimSens'))", rtdf, groups=rtdf["session"]),
        {"Impulsive_vs_StimSens": impulse_term})
    add("M4_RT_state", "Impulsive-StimSens (s)", m4["Impulsive_vs_StimSens"],
        f"Impulsive RT {'faster' if m4['Impulsive_vs_StimSens']['coef'] < 0 else 'slower'} "
        f"({'sig' if m4['Impulsive_vs_StimSens']['p'] < 0.05 else 'n.s.'})")

    stats_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    stats_df.to_csv(os.path.join(OUT_DIR, "mixedeffects_state_sensory_stats.csv"), index=False)
    print("\n=== MIXED-EFFECTS FIXED EFFECTS ===")
    print(stats_df[["model", "term", "coef", "se", "p_value", "interpretation"]].to_string(index=False))

    # ----------------------------- figure -----------------------------
    fig = plt.figure(figsize=(14, 4.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.32, left=0.07, right=0.98, top=0.84, bottom=0.16)
    rng = np.random.default_rng(42)

    # A: evoked-z by state (early), unit-level
    axA = fig.add_subplot(gs[0, 0])
    for k, s in enumerate(STATES):
        vals = d1[d1["state"] == s]["evoked_z"].values
        axA.scatter(rng.normal(k, 0.06, len(vals)), vals, s=10, alpha=0.4,
                    color=STATE_LABEL_COLORS[s], edgecolors="none")
        axA.scatter([k], [np.mean(vals)], s=130, color=STATE_LABEL_COLORS[s],
                    edgecolors="k", linewidths=1.2, marker="D", zorder=4)
    axA.axhline(0, color="k", lw=0.5, alpha=0.3)
    axA.set_xticks(range(len(STATES))); axA.set_xticklabels(STATES)
    axA.set_ylabel("Evoked z (early, 0–250 ms)")
    p1 = m1["Impulsive_vs_StimSens"]["p"]
    axA.set_title(f"A. State effect (n={n_units} units)\nMixedLM p={p1:.1e}",
                  fontsize=10, fontweight="bold")

    # B: Hit vs Miss (early)
    axB = fig.add_subplot(gs[0, 1])
    cols = {"Hit": "#4CAF50", "Miss": "#F44336"}
    for k, oc in enumerate(["Hit", "Miss"]):
        vals = hmdf[hmdf["outcome"] == oc]["evoked_z"].values
        axB.scatter(rng.normal(k, 0.06, len(vals)), vals, s=10, alpha=0.4,
                    color=cols[oc], edgecolors="none")
        axB.scatter([k], [np.mean(vals)], s=130, color=cols[oc], edgecolors="k",
                    linewidths=1.2, marker="D", zorder=4)
    axB.axhline(0, color="k", lw=0.5, alpha=0.3)
    axB.set_xticks([0, 1]); axB.set_xticklabels(["Hit", "Miss"])
    axB.set_ylabel("Evoked z (early)")
    p3 = m3["Hit_vs_Miss"]["p"]
    axB.set_title(f"B. Hit vs Miss (unit-pooled)\nMixedLM p={p3:.1e}",
                  fontsize=10, fontweight="bold")

    # C: RT by state (trial-level)
    axC = fig.add_subplot(gs[0, 2])
    for k, s in enumerate(STATES):
        vals = rtdf[rtdf["state"] == s]["rt"].values
        parts = axC.violinplot([vals], positions=[k], widths=0.7, showmeans=False,
                               showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(STATE_LABEL_COLORS[s]); pc.set_alpha(0.45)
        axC.scatter([k], [np.median(vals)], s=90, color=STATE_LABEL_COLORS[s],
                    edgecolors="k", zorder=4)
    axC.set_xticks(range(len(STATES))); axC.set_xticklabels(STATES)
    axC.set_ylabel("Hit RT (s)"); axC.set_ylim(0, min(2.5, np.percentile(rtdf["rt"], 98)))
    p4 = m4["Impulsive_vs_StimSens"]["p"]
    axC.set_title(f"C. Reaction time ({len(rtdf)} trials)\nMixedLM p={p4:.1e}",
                  fontsize=10, fontweight="bold")

    p2 = m2["state_x_window"]["p"]
    fig.suptitle(
        f"Mixed-effects re-test (unit nested in session) — {SUBJECT}, Expert · "
        f"{n_sess} sessions, {n_units} units\n"
        f"state×window (dynamics) interaction p={p2:.2f}",
        fontsize=12, fontweight="bold", y=0.99)
    save_figure(fig, "mixedeffects_state_sensory", f"state_labeler/{SUBJECT}")
    plt.close(fig)
    print("[mixed] done.")


if __name__ == "__main__":
    main()
