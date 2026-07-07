"""LEARNING axis for the transient/sustained TF-cell dissociation.

⚠️ CONFOUND: units are NOT cross-session tracked, and there is documented chronic-
probe DRIFT (BG_046 broad/SPN% 89->15% across stages), so cross-stage comparisons
of the recorded population mix learning with drift. Therefore:
  (A) DRIFT-ROBUST — within-stage robustness: does sustained>transient coupling hold
      WITHIN each stage separately (Learning, Expert)? A within-population contrast at
      each stage is drift-robust (drift changes WHICH cells you record, not the within-
      sample class contrast). This is the defensible result.
  (B) CAVEATED — does the sustained-transient GAP strengthen Learning->Expert?
      (bootstrap CI on Expert_gap - Learning_gap). Cross-stage → drift-confounded; a
      positive here is suggestive, NOT a clean learning claim.
  (C) CAVEATED — per-session coupling gap vs behavioural d' (Spearman), with a
      session-order (drift proxy) partial correlation to probe learning-vs-drift.

Stages from data/<SUBJ>_staging_manifest.csv (Naive merged into Learning per the
project SESSION_FILTER). d' = manifest d_prime. Reads load_cells() (cache) — fast.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import mannwhitneyu, spearmanr, rankdata

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO, _pdate                              # noqa: E402
from transient_vs_sustained import load_cells, TCOL, SCOL                  # noqa: E402

OUTCOMES = [("change_on", "Change_ON response"), ("hit_ramp", "Hit motor ramp"),
            ("fa_ramp", "FA motor ramp")]
STAGES = ["Learning", "Expert"]
STAGE_C = {"Learning": "#fdae6b", "Expert": "#31a354"}
OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/learning_transient_sustained")
MIN_T, MIN_S = 3, 3


def attach_stage(cells):
    frames = []
    for subj in cells.subject.unique():
        man = pd.read_csv(f"{REPO}/data/{subj}_staging_manifest.csv", dtype={"session_name": str})
        man = man[~man.qc_fail.astype(bool)]
        man["stage2"] = man["stage"].replace({"Naive": "Learning"})
        man["order"] = man["date"].map(_pdate)
        man = man.sort_values("order").reset_index(drop=True)
        man["chrono"] = np.arange(len(man))
        smap = man.set_index("session_name")[["stage2", "d_prime", "chrono"]]
        c = cells[cells.subject == subj].copy()
        c["date_str"] = [str(s).split(f"{subj}_", 1)[-1] for s in c.session]
        c = c.merge(smap, left_on="date_str", right_index=True, how="left")
        frames.append(c)
    return pd.concat(frames, ignore_index=True)


def _mwu(a, b):
    a = pd.Series(a).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(b).replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < MIN_T or len(b) < MIN_S:
        return np.nan, np.nan, len(a), len(b), np.nan
    return float(a.median()), float(b.median()), len(a), len(b), float(mannwhitneyu(a, b).pvalue)


def gap_diff_boot(df, col, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    sub = {stg: (df[(df.stage2 == stg) & (df.cls == "transient")][col].replace([np.inf, -np.inf], np.nan).dropna().values,
                 df[(df.stage2 == stg) & (df.cls == "sustained")][col].replace([np.inf, -np.inf], np.nan).dropna().values)
           for stg in STAGES}
    for _ in range(n):
        g = {}
        ok = True
        for stg in STAGES:
            t, s = sub[stg]
            if len(t) < MIN_T or len(s) < MIN_S:
                ok = False; break
            g[stg] = np.median(rng.choice(s, len(s))) - np.median(rng.choice(t, len(t)))
        if ok:
            diffs.append(g["Expert"] - g["Learning"])
    if len(diffs) < 100:
        return np.nan, np.nan, np.nan, np.nan
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs < 0).mean(), (diffs > 0).mean())
    return float(np.median(diffs)), float(lo), float(hi), float(max(p, 1 / n))


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]

    def resid(a):
        c, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ c
    ex, ey = resid(rx), resid(ry)
    return float(np.corrcoef(ex, ey)[0, 1])


def session_gaps(df, col):
    """per-session sustained-transient gap + d' + chrono (sessions with enough of both)."""
    rows = []
    for sess, g in df.groupby("session"):
        t = g[g.cls == "transient"][col].replace([np.inf, -np.inf], np.nan).dropna()
        s = g[g.cls == "sustained"][col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(t) >= MIN_T and len(s) >= MIN_S:
            rows.append(dict(session=sess, gap=s.median() - t.median(),
                             d_prime=g.d_prime.iloc[0], chrono=g.chrono.iloc[0],
                             region=g.region.iloc[0]))
    return pd.DataFrame(rows)


def main():
    cells = attach_stage(load_cells().rename(columns={"class": "cls"}))
    cells = cells[cells.cls.isin(["transient", "sustained"])]
    n_join = cells.stage2.notna().sum()
    cells = cells.dropna(subset=["stage2"])
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})
    lines = [f"cells with stage joined: {n_join}/{len(cells)}",
             "stage counts: " + str(cells.groupby(['stage2', 'cls']).size().to_dict())]

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)

    # Row 1: within-stage transient vs sustained + gap-strengthening test
    for j, (col, lab) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[0, j])
        xpos = {("Learning", "transient"): 0, ("Learning", "sustained"): 0.8,
                ("Expert", "transient"): 2.0, ("Expert", "sustained"): 2.8}
        for (stg, cl), x in xpos.items():
            v = cells[(cells.stage2 == stg) & (cells.cls == cl)][col].replace([np.inf, -np.inf], np.nan).dropna()
            jit = (np.random.default_rng(int(x * 10)).random(len(v)) - 0.5) * 0.5
            ax.scatter(np.full(len(v), x) + jit, v, s=8, alpha=0.35,
                       color=(TCOL if cl == "transient" else SCOL), edgecolors="none")
            ax.hlines(v.median(), x - 0.32, x + 0.32, color="k", lw=2.2, zorder=5)
        # within-stage MWU
        for stg, xc in (("Learning", 0.4), ("Expert", 2.4)):
            mt, ms, nt, ns, p = _mwu(cells[(cells.stage2 == stg) & (cells.cls == "transient")][col],
                                     cells[(cells.stage2 == stg) & (cells.cls == "sustained")][col])
            ax.text(xc, ax.get_ylim()[1] * 0.95, f"{stg}\np={p:.1e}\n(t{nt}/s{ns})", ha="center", va="top", fontsize=8)
            lines.append(f"[{col}] WITHIN {stg}: transient={mt:.2f} sustained={ms:.2f} (n {nt}/{ns}) MWU p={p:.2e}")
        gd, glo, ghi, gp = gap_diff_boot(cells, col)
        ax.set_title(f"{lab}\nExpert−Learning gap Δ={gd:+.2f} [{glo:+.2f},{ghi:+.2f}] p={gp:.2f}", fontsize=10)
        ax.set_xticks([0.4, 2.4]); ax.set_xticklabels(["Learning", "Expert"], fontsize=10)
        ax.axhline(0, color="0.7", lw=0.7, ls=":")
        if j == 0:
            ax.set_ylabel("Δ firing (Hz)", fontsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lines.append(f"[{col}] GAP-STRENGTHEN Expert-Learning Δ={gd:+.3f} CI[{glo:+.3f},{ghi:+.3f}] p={gp:.3f}")

    # Row 2: per-session gap vs d' (with session-order partial)
    for j, (col, lab) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[1, j])
        sg = session_gaps(cells, col)
        if len(sg) >= 6:
            for reg, cc in (("DMS", "#3474ae"), ("VMS", "#ef6548")):
                m = sg.region == reg
                ax.scatter(sg.d_prime[m], sg.gap[m], s=40, color=cc, alpha=0.7, label=reg, edgecolors="none")
            rho, p = spearmanr(sg.d_prime, sg.gap)
            pr = partial_spearman(sg.gap.values, sg.d_prime.values, sg.chrono.values)
            rho_ord, _ = spearmanr(sg.chrono, sg.gap)
            b1, b0 = np.polyfit(sg.d_prime, sg.gap, 1)
            xs = np.linspace(sg.d_prime.min(), sg.d_prime.max(), 20)
            ax.plot(xs, b0 + b1 * xs, color="k", lw=1.5)
            ax.set_title(f"{lab}\nρ(gap,d')={rho:+.2f} p={p:.2f} · partial|order={pr:+.2f} · ρ(gap,order)={rho_ord:+.2f}",
                         fontsize=9)
            lines.append(f"[{col}] per-session gap~d' rho={rho:+.3f} p={p:.2e} (n={len(sg)}); "
                         f"partial|chrono={pr:+.3f}; gap~chrono rho={rho_ord:+.3f}")
            if j == 0:
                ax.legend(frameon=False, fontsize=9)
        ax.axhline(0, color="0.7", lw=0.7, ls=":")
        ax.set_xlabel("session d′", fontsize=12)
        if j == 0:
            ax.set_ylabel("per-session gap\n(sustained − transient, Hz)", fontsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Transient/sustained coupling across LEARNING (Naive+Learning vs Expert)\n"
                 "TOP: dissociation holds WITHIN each stage (drift-robust) + does the gap strengthen? · "
                 "BOTTOM: gap vs d' (caution: cross-session = drift-confounded; partial | session-order shown)",
                 fontsize=12.5, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"learning_transient_sustained.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (OUT / "learning_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/learning_transient_sustained.png")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
