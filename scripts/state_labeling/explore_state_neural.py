"""Four exploratory state x neural analyses (BG_046 Expert), off the cached
extraction (scripts/state_labeling/explore_extract.py) + state_gain_traces.npz.

  #1 Sensory vs motor specificity  -> figures/.../explore1_sensory_vs_motor.png
  #2 Pre-change anticipatory ramp  -> figures/.../explore2_prechange_ramp.png
  #3 State-conditioned decoding    -> figures/.../explore3_decoding.png
  #4 Single-trial response -> RT   -> figures/.../explore4_response_rt.png
Combined stats -> figures/.../explore_state_neural_stats.csv

States compared: Impulsive vs StimSens (reference). All per-unit z use shared
baselines; tests pair within unit/session. Honest about power where thin.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import STATE_LABEL_COLORS
from visdetect.suite.plotting import setup_style, save_figure

setup_style()
warnings.filterwarnings("ignore")

SUBJECT = "BG_046"
STATES = ["Impulsive", "StimSens"]
BIN = 0.01
PRECHANGE_WIN = (-0.5, 0.0)
MIN_TRIALS = 8
MIN_DEC = 8                      # per class per state for decoding

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(_REPO, "analysis_suite", "cache", "state_neural_explore")
GAIN_TRACES = os.path.join(_REPO, "analysis_suite", "cache", "state_gain_traces.npz")
OUT = f"state_labeler/{SUBJECT}"
STATS_CSV = os.path.join(_REPO, "analysis_suite", "figures", "state_labeler", SUBJECT,
                         "explore_state_neural_stats.csv")
VC = {"unit": "0 + C(unit)"}
IMP_TERM = "C(state, Treatment('StimSens'))[T.Impulsive]"
ROWS = []


def add(model, term, coef, se, p, interp, notes=""):
    ROWS.append(dict(analysis=model, term=term, coef=coef, se=se, p_value=p,
                     interpretation=interp, notes=notes))


def load_cache():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def mixedlm_imp_vs_ss(df):
    """df cols: session, unit, state, evoked. Returns (coef, se, p, ci)."""
    m = smf.mixedlm("evoked ~ C(state, Treatment('StimSens'))", df,
                    groups=df["session"], vc_formula=VC, re_formula="1").fit(reml=True, method="lbfgs")
    ci = m.conf_int()
    return (m.params[IMP_TERM], m.bse[IMP_TERM], m.pvalues[IMP_TERM],
            ci.loc[IMP_TERM, 0], ci.loc[IMP_TERM, 1])


# ----------------------------------------------------------------- #1 ----
def analysis1_sensory_vs_motor(cache):
    aligns = [("change", "Sensory\n(change)"), ("fa", "Motor\n(FA lick)"), ("hl", "Motor\n(Hit lick)")]
    results = {}
    for key, _ in aligns:
        rows = []
        for d in cache:
            ni = {"change": (MIN_TRIALS, MIN_TRIALS),
                  "fa": (int(d["n_fa_Impulsive"]), int(d["n_fa_StimSens"])),
                  "hl": (int(d["n_hl_Impulsive"]), int(d["n_hl_StimSens"]))}[key]
            if min(ni) < MIN_TRIALS:
                continue
            imp = d[f"{key}_evoked_Impulsive"]; ss = d[f"{key}_evoked_StimSens"]
            sid = str(d["sid8"])
            for j in range(len(imp)):
                if np.isfinite(imp[j]) and np.isfinite(ss[j]):
                    rows.append((sid, f"{sid}_{j}", "Impulsive", float(imp[j])))
                    rows.append((sid, f"{sid}_{j}", "StimSens", float(ss[j])))
        df = pd.DataFrame(rows, columns=["session", "unit", "state", "evoked"])
        nU = df["unit"].nunique(); nS = df["session"].nunique()
        if nU < 10:
            results[key] = dict(coef=np.nan, se=np.nan, p=np.nan, lo=np.nan, hi=np.nan,
                                nU=nU, nS=nS, frac=np.nan)
            continue
        coef, se, p, lo, hi = mixedlm_imp_vs_ss(df)
        diff = (df[df.state == "Impulsive"].set_index("unit")["evoked"]
                - df[df.state == "StimSens"].set_index("unit")["evoked"])
        frac = float((diff > 0).mean())
        results[key] = dict(coef=coef, se=se, p=p, lo=lo, hi=hi, nU=nU, nS=nS, frac=frac)
        add(f"#1_{key}", "Impulsive-StimSens", round(coef, 4), round(se, 4), p,
            f"{key}: Impulsive {'>' if coef > 0 else '<'} StimSens "
            f"({'sig' if p < 0.05 else 'n.s.'})", f"{nU} units/{nS} sess; {100*frac:.0f}% units +")

    fig = plt.figure(figsize=(11, 4.4))
    gs = gridspec.GridSpec(1, 2, wspace=0.32, left=0.10, right=0.97, top=0.85, bottom=0.18)
    axA = fig.add_subplot(gs[0, 0]); ys = np.arange(len(aligns))[::-1]
    for y, (key, lab) in zip(ys, aligns):
        r = results[key]
        c = STATE_LABEL_COLORS["Impulsive"] if (np.isfinite(r["p"]) and r["p"] < 0.05) else "0.6"
        axA.errorbar([r["coef"]], [y], xerr=[[r["coef"] - r["lo"]], [r["hi"] - r["coef"]]],
                     fmt="o", color=c, capsize=4, ms=9)
        axA.text(0.98, y + 0.18, f"p={r['p']:.1e}" if np.isfinite(r['p']) else "n/a",
                 transform=axA.get_yaxis_transform(), ha="right", fontsize=8, color="0.3")
    axA.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    axA.set_yticks(ys); axA.set_yticklabels([l for _, l in aligns])
    axA.set_xlabel("Impulsive − StimSens evoked z  (mixedLM ± 95% CI)")
    axA.set_title("A. Is the gain effect sensory-specific?", fontsize=10, fontweight="bold")
    axB = fig.add_subplot(gs[0, 1])
    fr = [results[k]["frac"] * 100 for k, _ in aligns]
    axB.bar(range(len(aligns)), fr, color=["#3474ae", "#9aa0a6", "#9aa0a6"])
    axB.axhline(50, color="k", ls="--", lw=0.8, alpha=0.5)
    axB.set_xticks(range(len(aligns))); axB.set_xticklabels([l for _, l in aligns])
    axB.set_ylabel("% units Impulsive > StimSens"); axB.set_ylim(0, 100)
    axB.set_title("B. Fraction leaning Impulsive-bigger", fontsize=10, fontweight="bold")
    fig.suptitle(f"#1 Sensory vs motor specificity of the state gain effect — {SUBJECT} Expert",
                 fontsize=12, fontweight="bold", y=0.98)
    save_figure(fig, "explore1_sensory_vs_motor", OUT); plt.close(fig)


# ----------------------------------------------------------------- #2 ----
def analysis2_prechange_ramp(cache):
    d = np.load(GAIN_TRACES, allow_pickle=True)
    bc = d["bc"]; pre = (bc >= PRECHANGE_WIN[0]) & (bc < PRECHANGE_WIN[1])
    t = bc[pre]
    slopes = {s: [] for s in STATES}
    for s in STATES:
        arr = d[s]                       # (n_sessions, n_bins)
        for i in range(arr.shape[0]):
            y = arr[i, pre]
            slopes[s].append(np.polyfit(t, y, 1)[0])   # z per second
    slopes = {s: np.array(v) for s, v in slopes.items()}
    nS = len(slopes["Impulsive"])
    W, p = wilcoxon(slopes["Impulsive"], slopes["StimSens"])
    med = float(np.median(slopes["Impulsive"] - slopes["StimSens"]))
    add("#2_prechange_ramp", "Impulsive-StimSens slope", round(med, 4), np.nan, p,
        f"Impulsive pre-change ramp {'steeper' if med > 0 else 'shallower'} "
        f"({'sig' if p < 0.05 else 'n.s.'})", f"{nS} sessions; median Δslope={med:.3f} z/s")

    fig = plt.figure(figsize=(11, 4.4))
    gs = gridspec.GridSpec(1, 2, wspace=0.3, left=0.09, right=0.97, top=0.85, bottom=0.16)
    axA = fig.add_subplot(gs[0, 0])
    for s in STATES:
        gm = d[s].mean(axis=0)
        axA.plot(bc, gm, color=STATE_LABEL_COLORS[s], lw=2, label=s)
    axA.axvspan(*PRECHANGE_WIN, color="0.85", alpha=0.5)
    axA.axvline(0, color="k", ls="--", lw=0.9, alpha=0.6); axA.axhline(0, color="k", lw=0.5, alpha=0.3)
    axA.set_xlim(-0.5, 0.4); axA.set_xlabel("Time from change onset (s)")
    axA.set_ylabel("Population z"); axA.legend(frameon=False, fontsize=9)
    axA.set_title("A. Pre-change window (shaded)", fontsize=10, fontweight="bold")
    axB = fig.add_subplot(gs[0, 1])
    for i in range(nS):
        axB.plot([0, 1], [slopes["Impulsive"][i], slopes["StimSens"][i]],
                 color="0.75", lw=0.8, alpha=0.7)
    for k, s in enumerate(STATES):
        axB.scatter(np.full(nS, k), slopes[s], s=25, color=STATE_LABEL_COLORS[s], alpha=0.7, zorder=3)
        axB.scatter([k], [np.mean(slopes[s])], s=130, color=STATE_LABEL_COLORS[s],
                    edgecolors="k", linewidths=1.2, marker="D", zorder=4)
    axB.axhline(0, color="k", lw=0.5, alpha=0.3)
    axB.set_xticks([0, 1]); axB.set_xticklabels(STATES); axB.set_ylabel("Pre-change ramp slope (z/s)")
    axB.set_title(f"B. Ramp slope by state\nWilcoxon p={p:.2f}", fontsize=10, fontweight="bold")
    fig.suptitle(f"#2 Pre-change anticipatory ramp — {SUBJECT} Expert ({nS} sessions)",
                 fontsize=12, fontweight="bold", y=0.98)
    save_figure(fig, "explore2_prechange_ramp", OUT); plt.close(fig)


# ----------------------------------------------------------------- #3 ----
def _auc(X, y):
    if len(np.unique(y)) < 2:
        return np.nan
    n_min = min(np.bincount(y))
    k = int(min(5, n_min))
    if k < 2:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    return float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")))


def analysis3_decoding(cache):
    per = {s: [] for s in STATES}
    for d in cache:
        z = d["trial_z"]; st = d["trial_state"].astype(str); go = d["trial_is_go"]
        oc = d["trial_outcome"].astype(str)
        for s in STATES:
            sel = st == s
            # class 1 = real change (go & hit); class 0 = catch (not go)
            is_change = sel & go & (oc == "hit")
            is_catch = sel & (~go)
            y = np.r_[np.ones(is_change.sum()), np.zeros(is_catch.sum())].astype(int)
            if (is_change.sum() < MIN_DEC) or (is_catch.sum() < MIN_DEC):
                per[s].append(np.nan); continue
            X = np.vstack([z[is_change], z[is_catch]])
            per[s].append(_auc(X, y))
    per = {s: np.array(v) for s, v in per.items()}
    paired = np.isfinite(per["Impulsive"]) & np.isfinite(per["StimSens"])
    nS = int(paired.sum())
    if nS >= 3:
        W, p = wilcoxon(per["Impulsive"][paired], per["StimSens"][paired])
        med = float(np.median(per["Impulsive"][paired] - per["StimSens"][paired]))
    else:
        W, p, med = np.nan, np.nan, np.nan
    for s in STATES:
        v = per[s][np.isfinite(per[s])]
        add(f"#3_decode_{s}", "CV AUC change-vs-catch", round(float(np.mean(v)), 3) if len(v) else np.nan,
            np.nan, np.nan, f"{s}: mean AUC={np.mean(v):.3f} ({len(v)} sess)" if len(v) else "no data")
    add("#3_decode_Imp_vs_SS", "AUC diff", round(med, 3) if np.isfinite(med) else np.nan, np.nan, p,
        f"decoding {'better' if (np.isfinite(med) and med>0) else 'worse'} in Impulsive "
        f"({'sig' if (np.isfinite(p) and p<0.05) else 'n.s.'})", f"{nS} paired sessions")

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    fig.subplots_adjust(left=0.16, right=0.95, top=0.85, bottom=0.14)
    for k, s in enumerate(STATES):
        v = per[s][np.isfinite(per[s])]
        ax.scatter(np.full(len(v), k) + np.random.default_rng(1).normal(0, 0.04, len(v)),
                   v, s=30, color=STATE_LABEL_COLORS[s], alpha=0.7, zorder=3)
        if len(v):
            ax.scatter([k], [np.mean(v)], s=130, color=STATE_LABEL_COLORS[s],
                       edgecolors="k", linewidths=1.2, marker="D", zorder=4)
    if nS >= 3:
        idxI, idxS = np.where(paired)[0], np.where(paired)[0]
        for i in idxI:
            ax.plot([0, 1], [per["Impulsive"][i], per["StimSens"][i]], color="0.8", lw=0.7, alpha=0.7)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5, label="chance")
    ax.set_xticks([0, 1]); ax.set_xticklabels(STATES); ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("CV AUC (change vs catch)")
    ax.set_title(f"#3 Population decoding by state\n{nS} paired sessions · Wilcoxon p="
                 + (f"{p:.2f}" if np.isfinite(p) else "n/a"), fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "explore3_decoding", OUT); plt.close(fig)


# ----------------------------------------------------------------- #4 ----
def analysis4_response_rt(cache):
    rho = {s: [] for s in STATES}
    pooled = {s: ([], []) for s in STATES}     # (response, rt)
    for d in cache:
        z = d["trial_z"]; st = d["trial_state"].astype(str); go = d["trial_is_go"]
        oc = d["trial_outcome"].astype(str); rt = d["trial_rt"]
        pop = np.nanmean(z, axis=1)            # per-trial population response
        for s in STATES:
            sel = (st == s) & go & (oc == "hit") & np.isfinite(rt)
            if sel.sum() < MIN_TRIALS:
                rho[s].append(np.nan); continue
            r, _ = spearmanr(pop[sel], rt[sel])
            rho[s].append(r)
            pooled[s][0].extend(pop[sel]); pooled[s][1].extend(rt[sel])
    rho = {s: np.array(v) for s, v in rho.items()}
    for s in STATES:
        v = rho[s][np.isfinite(rho[s])]
        W, p = wilcoxon(v) if len(v) >= 3 else (np.nan, np.nan)
        add(f"#4_resp_rt_{s}", "Spearman rho (resp vs RT)", round(float(np.mean(v)), 3) if len(v) else np.nan,
            np.nan, p, f"{s}: mean rho={np.mean(v):.3f} ({'sig' if (np.isfinite(p) and p<0.05) else 'n.s.'}, "
            f"{len(v)} sess)" if len(v) else "no data",
            "negative rho = bigger response -> faster RT")

    fig = plt.figure(figsize=(11, 4.4))
    gs = gridspec.GridSpec(1, 2, wspace=0.3, left=0.09, right=0.97, top=0.85, bottom=0.16)
    axA = fig.add_subplot(gs[0, 0])
    for s in STATES:
        x, y = np.array(pooled[s][0]), np.array(pooled[s][1])
        axA.scatter(x, y, s=8, color=STATE_LABEL_COLORS[s], alpha=0.25, edgecolors="none")
        if len(x) > 10:
            b = np.polyfit(x, y, 1); xs = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 50)
            axA.plot(xs, np.polyval(b, xs), color=STATE_LABEL_COLORS[s], lw=2, label=s)
    axA.set_xlabel("Per-trial population response (z, early)"); axA.set_ylabel("Hit RT (s)")
    axA.set_ylim(0, np.percentile(np.concatenate([pooled[s][1] for s in STATES]), 97))
    axA.legend(frameon=False, fontsize=9)
    axA.set_title("A. Bigger response → faster RT? (pooled)", fontsize=10, fontweight="bold")
    axB = fig.add_subplot(gs[0, 1])
    for k, s in enumerate(STATES):
        v = rho[s][np.isfinite(rho[s])]
        axB.scatter(np.full(len(v), k) + np.random.default_rng(2).normal(0, 0.04, len(v)),
                    v, s=30, color=STATE_LABEL_COLORS[s], alpha=0.7, zorder=3)
        if len(v):
            axB.scatter([k], [np.mean(v)], s=130, color=STATE_LABEL_COLORS[s],
                        edgecolors="k", linewidths=1.2, marker="D", zorder=4)
    axB.axhline(0, color="k", lw=0.6, alpha=0.4)
    axB.set_xticks([0, 1]); axB.set_xticklabels(STATES)
    axB.set_ylabel("Spearman ρ (response vs RT) per session")
    axB.set_title("B. Brain→behavior coupling by state", fontsize=10, fontweight="bold")
    fig.suptitle(f"#4 Single-trial response → RT — {SUBJECT} Expert", fontsize=12, fontweight="bold", y=0.98)
    save_figure(fig, "explore4_response_rt", OUT); plt.close(fig)


def main():
    cache = load_cache()
    print(f"[explore] loaded {len(cache)} cached sessions")
    analysis1_sensory_vs_motor(cache)
    analysis2_prechange_ramp(cache)
    analysis3_decoding(cache)
    analysis4_response_rt(cache)
    df = pd.DataFrame(ROWS)
    df.to_csv(STATS_CSV, index=False)
    print("\n=== EXPLORE STATS ===")
    print(df[["analysis", "coef", "p_value", "interpretation"]].to_string(index=False))
    print("[explore] done.")


if __name__ == "__main__":
    main()
