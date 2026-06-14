"""#4 (controlled): does single-trial response predict RT BEYOND change size?

The raw response->RT correlation is confounded: big changes evoke big responses
AND fast RTs, so pooling across sizes manufactures a negative correlation. Here we
control for change size two ways, off the cached extraction:

  (A) Per-session, per-state PARTIAL Spearman rho(response, RT | change_size);
      Wilcoxon across sessions vs 0. Non-parametric, project-convention, paired.
  (B) Mixed model  RT ~ response + C(change_size), groups=session, per state;
      the 'response' coefficient is the partial effect (powered, pools trials).

Go-Hit trials only (they have an RT and a real change size). Output:
  figures/state_labeler/BG_046/explore4_partial_rt.png
  figures/state_labeler/BG_046/explore4_partial_rt_stats.csv
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, wilcoxon

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
GO_SET = {1.25, 1.35, 1.5, 2.0, 4.0}
MIN_TRIALS = 15

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(_REPO, "analysis_suite", "cache", "state_neural_explore")
OUT = f"state_labeler/{SUBJECT}"
STATS_CSV = os.path.join(_REPO, "analysis_suite", "figures", "state_labeler", SUBJECT,
                         "explore4_partial_rt_stats.csv")


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z."""
    rxy = spearmanr(x, y).correlation
    rxz = spearmanr(x, z).correlation
    ryz = spearmanr(y, z).correlation
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if not np.isfinite(denom) or denom < 1e-8:
        return np.nan
    return (rxy - rxz * ryz) / denom


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    raw = {s: [] for s in STATES}
    part = {s: [] for s in STATES}
    long = {s: [] for s in STATES}      # for mixed model
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        sid = str(d["sid8"])
        z = d["trial_z"]; st = d["trial_state"].astype(str); go = d["trial_is_go"]
        oc = d["trial_outcome"].astype(str); rt = d["trial_rt"]; cs = d["trial_csize"]
        pop = np.nanmean(z, axis=1)
        for s in STATES:
            sel = (st == s) & go & (oc == "hit") & np.isfinite(rt)
            if sel.sum() < MIN_TRIALS or len(np.unique(cs[sel])) < 2:
                raw[s].append(np.nan); part[s].append(np.nan); continue
            r_pop, r_rt, r_cs = pop[sel], rt[sel], cs[sel]
            raw[s].append(spearmanr(r_pop, r_rt).correlation)
            part[s].append(partial_spearman(r_pop, r_rt, r_cs))
            for i in np.where(sel)[0]:
                long[s].append((sid, pop[i], rt[i], float(cs[i])))

    rows = []
    mixed_coef = {}
    for s in STATES:
        rv = np.array(raw[s]); pv = np.array(part[s])
        rvf, pvf = rv[np.isfinite(rv)], pv[np.isfinite(pv)]
        Wr, pr = wilcoxon(rvf) if len(rvf) >= 3 else (np.nan, np.nan)
        Wp, pp = wilcoxon(pvf) if len(pvf) >= 3 else (np.nan, np.nan)
        rows.append(dict(test=f"raw_rho_{s}", value=round(float(np.mean(rvf)), 3),
                         p_value=pr, n=len(rvf),
                         interpretation=f"{s} raw rho={np.mean(rvf):.3f} ({'sig' if pr<0.05 else 'n.s.'})"))
        rows.append(dict(test=f"partial_rho_{s}", value=round(float(np.mean(pvf)), 3),
                         p_value=pp, n=len(pvf),
                         interpretation=f"{s} partial rho (|change size)={np.mean(pvf):.3f} "
                                        f"({'sig' if (np.isfinite(pp) and pp<0.05) else 'n.s.'})"))
        # mixed model RT ~ response + C(change size)
        df = pd.DataFrame(long[s], columns=["session", "response", "rt", "csize"])
        m = smf.mixedlm("rt ~ response + C(csize)", df, groups=df["session"]).fit(method="lbfgs")
        coef, se, pv2 = m.params["response"], m.bse["response"], m.pvalues["response"]
        ci = m.conf_int().loc["response"]
        mixed_coef[s] = (coef, ci[0], ci[1], pv2)
        rows.append(dict(test=f"mixed_response_coef_{s}", value=round(coef, 4), p_value=pv2,
                         n=len(df), interpretation=f"{s}: RT changes {coef:+.3f}s per +1z response, "
                         f"controlling change size ({'sig' if pv2<0.05 else 'n.s.'})"))

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(STATS_CSV, index=False)
    print("=== #4 controlled stats ===")
    print(stats_df.to_string(index=False))

    # figure
    fig = plt.figure(figsize=(11, 4.4))
    gs = gridspec.GridSpec(1, 2, wspace=0.32, left=0.09, right=0.97, top=0.85, bottom=0.16)

    axA = fig.add_subplot(gs[0, 0])
    xpos = {"Impulsive": 0, "StimSens": 1}
    for s in STATES:
        rv = np.array(raw[s]); pv = np.array(part[s])
        ok = np.isfinite(rv) & np.isfinite(pv)
        x0 = xpos[s] - 0.18; x1 = xpos[s] + 0.18
        for i in np.where(ok)[0]:
            axA.plot([x0, x1], [rv[i], pv[i]], color="0.8", lw=0.7, alpha=0.7, zorder=1)
        axA.scatter(np.full(ok.sum(), x0), rv[ok], s=18, color="0.5", alpha=0.6, zorder=2)
        axA.scatter(np.full(ok.sum(), x1), pv[ok], s=18, color=STATE_LABEL_COLORS[s], alpha=0.7, zorder=2)
        axA.scatter([x0], [np.nanmean(rv)], s=120, color="0.4", edgecolors="k", marker="D", zorder=4)
        axA.scatter([x1], [np.nanmean(pv)], s=120, color=STATE_LABEL_COLORS[s], edgecolors="k", marker="D", zorder=4)
    axA.axhline(0, color="k", lw=0.6, alpha=0.4)
    axA.set_xticks([0, 1]); axA.set_xticklabels(STATES)
    axA.set_ylabel("Spearman ρ (response vs RT)")
    axA.set_title("A. Raw (grey) → partial | change size (color)\nper session", fontsize=10, fontweight="bold")

    axB = fig.add_subplot(gs[0, 1])
    for k, s in enumerate(STATES):
        coef, lo, hi, p = mixed_coef[s]
        axB.errorbar([k], [coef], yerr=[[coef - lo], [hi - coef]], fmt="o",
                     color=STATE_LABEL_COLORS[s], capsize=4, ms=10)
        axB.text(k, hi + 0.005, f"p={p:.1e}", ha="center", fontsize=8, color="0.3")
    axB.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    axB.set_xticks([0, 1]); axB.set_xticklabels(STATES); axB.set_xlim(-0.5, 1.5)
    axB.set_ylabel("RT change per +1z response (s)\n(mixedLM, controls change size)")
    axB.set_title("B. Partial effect of response on RT", fontsize=10, fontweight="bold")

    fig.suptitle(f"#4 controlled: response→RT beyond change size — {SUBJECT} Expert",
                 fontsize=12, fontweight="bold", y=0.98)
    save_figure(fig, "explore4_partial_rt", OUT); plt.close(fig)
    print("[#4-controlled] done.")


if __name__ == "__main__":
    main()
