"""Visualize BG_046 TF-pulse responses: do GLM-identified TF-responsive units
actually LOOK TF-responsive?

Reads the run_tf_glm_bg046.py outputs (per-session CSV + per-unit fast/slow pulse
PETHs) and produces:
  fig1_exemplar_responsive.png  grid of the most TF-responsive units: actual
        fast (red) vs slow (blue) pulse-triggered firing (+ GLM prediction,
        dashed). A real TF cell should separate fast from slow around the pulse.
  fig2_responsive_vs_not.png    population mean fast-minus-slow pulse PETH for
        responsive vs non-responsive units (+ a few non-responsive exemplars for
        contrast), and the C1 distribution with the 0.2 threshold.

Run AFTER run_tf_glm_bg046.py (reads whatever sessions are done).
"""
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_bg046/src")
try:
    from visdetect.viz.plotting import set_style, despine
    set_style("talk")
except Exception:
    def despine(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

CACHE = Path("E:/python_analysis/git_repos/vd_tf_bg046/data/cache/tf_glm_bg046")
FIG = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/BG_046")
FIG.mkdir(parents=True, exist_ok=True)


def load():
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(str(CACHE / "bg046_*.csv")))
           if "_peth" not in f]
    if not dfs:
        raise SystemExit(f"no bg046_*.csv in {CACHE} yet")
    m = pd.concat(dfs, ignore_index=True)
    m["is_responsive"] = m["is_responsive"].astype(str).str.lower().isin(["true", "1", "1.0"])
    peth, t_axis = {}, None
    for f in sorted(glob.glob(str(CACHE / "bg046_*_peth.npz"))):
        sess = Path(f).stem.replace("bg046_", "").replace("_peth", "")
        z = np.load(f, allow_pickle=True)
        t_axis = z["t_axis"]
        for u in z["units"]:
            key = f"u{u}"
            if key in z:
                peth[(sess, int(u))] = z[key]  # 4 x nlags: a_fast,a_slow,p_fast,p_slow (Hz)
    return m, peth, t_axis


def _panel(ax, t, arr, title):
    af, as_, pf, ps = arr  # actual fast/slow, predicted fast/slow (Hz)
    # baseline-subtract each curve by its pre-pulse (t<0) mean for readability
    def bs(v):
        pre = v[t < 0]
        return v - (np.nanmean(pre) if pre.size else 0.0)
    ax.axvline(0, color="0.7", lw=0.8, zorder=0)
    ax.plot(t, bs(af), color="#d6322a", lw=2, label="fast (actual)")
    ax.plot(t, bs(as_), color="#2b6fb3", lw=2, label="slow (actual)")
    ax.plot(t, bs(pf), color="#d6322a", lw=1, ls="--", alpha=0.8, label="fast (GLM)")
    ax.plot(t, bs(ps), color="#2b6fb3", lw=1, ls="--", alpha=0.8, label="slow (GLM)")
    ax.set_title(title, fontsize=9)
    despine(ax)


def fig_exemplars(m, peth, t, n=12):
    resp = m[m.is_responsive].sort_values("c1_r", ascending=False)
    keys = [(r.session, int(r.unit)) for _, r in resp.iterrows()
            if (r.session, int(r.unit)) in peth][:n]
    if not keys:
        print("no responsive units with PETHs yet")
        return
    nc = 4; nr = int(np.ceil(len(keys) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 2.6 * nr), squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // nc][i % nc]
        r = resp[(resp.session == k[0]) & (resp.unit == k[1])].iloc[0]
        _panel(ax, t, peth[k], f"{k[0][-8:]} u{k[1]}\nC1={r.c1_r:.2f} p={r.c2_p:.1e}")
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="best")
        if i % nc == 0:
            ax.set_ylabel("Δ firing (Hz)")
        if i // nc == nr - 1:
            ax.set_xlabel("time from TF pulse (s)")
    for j in range(len(keys), nr * nc):
        axes[j // nc][j % nc].axis("off")
    fig.suptitle(f"BG_046 most TF-responsive units (n={len(keys)} of "
                 f"{int(m.is_responsive.sum())} responsive / {len(m)} units): "
                 f"fast vs slow TF-pulse-triggered firing", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = FIG / "fig1_exemplar_responsive.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig_population(m, peth, t):
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16, 4.6))
    # mean fast-minus-slow PETH for responsive vs non-responsive
    for grp, col, lab in [(True, "#d6322a", "responsive"), (False, "0.5", "non-responsive")]:
        diffs = []
        for _, r in m[m.is_responsive == grp].iterrows():
            k = (r.session, int(r.unit))
            if k in peth:
                af, as_, _, _ = peth[k]
                d = (af - as_)
                d = d - (np.nanmean(d[t < 0]) if (t < 0).any() else 0)
                diffs.append(d)
        if diffs:
            D = np.vstack(diffs); mu = np.nanmean(D, 0); se = np.nanstd(D, 0) / np.sqrt(len(D))
            axL.plot(t, mu, color=col, lw=2, label=f"{lab} (n={len(D)})")
            axL.fill_between(t, mu - se, mu + se, color=col, alpha=0.2)
    axL.axvline(0, color="0.7", lw=0.8); axL.axhline(0, color="0.7", lw=0.8)
    axL.set_xlabel("time from TF pulse (s)"); axL.set_ylabel("Δ firing, fast−slow (Hz)")
    axL.set_title("Population fast−slow pulse response"); axL.legend(frameon=False); despine(axL)
    # C1 distribution
    axM.hist(m["c1_r"].dropna(), bins=30, color="#6baed6", edgecolor="w")
    axM.axvline(0.2, color="#d6322a", ls="--", lw=1.5, label="C1=0.2 threshold")
    axM.set_xlabel("C1 (full-model fast−slow pulse-PETH corr)"); axM.set_ylabel("units")
    axM.set_title(f"C1 distribution\n{100*np.mean(m.c1_r>0.2):.0f}% > 0.2, "
                  f"{100*m.is_responsive.mean():.0f}% responsive (C1+C2)"); despine(axM)
    # responsive fraction per session
    fr = m.groupby("session")["is_responsive"].agg(["mean", "size"])
    axR.bar(range(len(fr)), 100 * fr["mean"], color="#5aa469")
    axR.set_xticks(range(len(fr)))
    axR.set_xticklabels([s[-8:] for s in fr.index], rotation=45, ha="right", fontsize=8)
    axR.set_ylabel("% TF-responsive"); axR.set_title("Responsive fraction per session"); despine(axR)
    fig.tight_layout()
    out = FIG / "fig2_population.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    m, peth, t = load()
    print(f"loaded {len(m)} units across {m.session.nunique()} sessions; "
          f"{int(m.is_responsive.sum())} TF-responsive; {len(peth)} PETHs")
    fig_exemplars(m, peth, t)
    fig_population(m, peth, t)
