# scripts/tf_responsiveness/state_conditioned/width_vs_waveform.py
"""Part 2: does the transient/sustained (temporal-width) axis map onto narrow/broad
(spike-waveform FSI/SPN)? Overlap crosstab (centerpiece) + continuous 2D joint
distribution + four-quadrant coupling + an independence test (does width predict
coupling controlling for t2p?). Striatum only; carries the yield-bias caveat.
Reads caches only.

t2p cache filename asymmetry: BG_031/039 = waveform_t2p_BG_{id}.csv, BG_046 =
bg046_waveform_t2p.csv. Resolve per subject (do NOT glob waveform_t2p_BG_*).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import spearmanr, chi2_contingency

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO                                  # noqa: E402
from visdetect.analysis.config import canonical_session_id            # noqa: E402

CACHE = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/width_vs_waveform"
REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}
T2P = {"BG_046": "data/cache/talk_substrate/bg046_waveform_t2p.csv",
       "BG_039": "data/cache/talk_substrate/waveform_t2p_BG_039.csv",
       "BG_031": "data/cache/talk_substrate/waveform_t2p_BG_031.csv"}
WIDTH = "interp_fwhm"
OUTCOMES = ["change_on", "hit_ramp", "fa_ramp"]


def _load_t2p(subj):
    df = pd.read_csv(Path(REPO) / T2P[subj])
    df["skey"] = df["session_8"].map(canonical_session_id)
    df["unit"] = df["cluster_id"].astype(int)
    return df[["skey", "unit", "t2p_ms"]].drop_duplicates(["skey", "unit"])


def _load_label(subj):
    f = Path(REPO) / f"data/{subj}/waveform_celltype_labels.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df["skey"] = df["session_date"].map(canonical_session_id)
    df["unit"] = df["cluster_id"].astype(int)
    return df[["skey", "unit", "celltype"]].drop_duplicates(["skey", "unit"])


def attach(d):
    d = d.copy()
    d["skey"] = [canonical_session_id(str(s).split(f"{sub}_", 1)[-1])
                 for s, sub in zip(d.session, d.subject)]
    out = []
    for subj in d.subject.unique():
        sub = d[d.subject == subj].merge(_load_t2p(subj), on=["skey", "unit"], how="left")
        lab = _load_label(subj)
        sub = sub.merge(lab, on=["skey", "unit"], how="left") if lab is not None else sub.assign(celltype=np.nan)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def main():
    import statsmodels.formula.api as smf
    d = attach(pd.read_csv(CACHE))
    d["region"] = d.subject.map(REGION)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    lines = []
    lab = d.dropna(subset=["celltype"]); lab = lab[lab.celltype.isin(["FSI", "SPN"])]
    lines.append(f"cells={len(d)}; with t2p={d.t2p_ms.notna().sum()}; with FSI/SPN label={len(lab)}")

    # ── overlap crosstab (centerpiece): class x celltype ──
    med = d[WIDTH].median()
    d["wclass"] = np.where(d[WIDTH] <= med, "transient", "sustained")
    ct = pd.crosstab(lab.assign(wclass=np.where(lab[WIDTH] <= med, "transient", "sustained")).wclass,
                     lab.celltype)
    chi2, pchi, *_ = chi2_contingency(ct)
    lines.append(f"OVERLAP crosstab (median-split width x waveform):\n{ct.to_string()}")
    lines.append(f"  chi2={chi2:.2f} p={pchi:.2e}")

    # ── continuous 2D: t2p vs width ──
    dd = d.dropna(subset=["t2p_ms", WIDTH])
    rho_all, p_all = spearmanr(dd.t2p_ms, dd[WIDTH])
    lines.append(f"CONTINUOUS t2p vs width: rho={rho_all:+.3f} p={p_all:.2e} (n={len(dd)})")
    for rg in ("DMS", "VMS"):
        sub = dd[dd.region == rg]
        if len(sub) > 10:
            r, p = spearmanr(sub.t2p_ms, sub[WIDTH])
            lines.append(f"    {rg}: rho={r:+.3f} p={p:.2e} (n={len(sub)})")

    # ── four-quadrant coupling ──
    tmed = lab.t2p_ms.median()
    lines.append(f"FOUR-QUADRANT (t2p median={tmed:.3f} ms, width median={med:.3f} s) — median Δ firing (Hz):")
    for wc in ("transient", "sustained"):
        for narrow in (True, False):
            q = lab[(np.where(lab[WIDTH] <= med, "transient", "sustained") == wc) &
                    ((lab.t2p_ms <= tmed) == narrow)]
            wf = "narrow/FSI" if narrow else "broad/SPN"
            meds = {c: round(float(q[c].median()), 2) for c in OUTCOMES if q[c].notna().any()}
            lines.append(f"    {wc:9s} x {wf:11s} n={len(q):3d}  {meds}")

    # ── independence: does width predict coupling controlling for t2p? ──
    # Two models per outcome: mixedlm (session RE) AND a cluster-robust OLS
    # (cluster on session) as a convergence-robust cross-check. The mixedlm session
    # RE variance is ~0 here (it warns / may not converge), so the cluster-robust OLS
    # is the model to prefer when a convergence flag fires — write BOTH so the
    # deliverable is self-documenting. Coupling metrics are raw-Hz Δfiring; the
    # width→coupling MAGNITUDE is established elsewhere — this test isolates the
    # width-vs-t2p INDEPENDENCE (t2p ns while width wins ⇒ FR-confounding is not
    # driving it).
    lines.append("INDEPENDENCE (outcome ~ width + t2p, standardized): width beta | t2p beta")
    lines.append("  metrics are raw-Hz Δfiring; MAGNITUDE established elsewhere — this isolates width-vs-t2p independence")
    lines.append("  mixedlm=session random-intercept; OLS=cluster-robust (cluster on session, prefer if mixedlm non-converged)")
    for col in OUTCOMES:
        m = d.dropna(subset=[col, WIDTH, "t2p_ms"]).copy()
        m["w"] = (m[WIDTH] - m[WIDTH].mean()) / m[WIDTH].std()
        m["t"] = (m.t2p_ms - m.t2p_ms.mean()) / m.t2p_ms.std()
        try:
            fit = smf.mixedlm(f"{col} ~ w + t", m, groups=m["session"]).fit(reml=False)
            conv = getattr(fit, "converged", True)
            flag = "" if conv else "  [!] mixedlm did NOT converge (RE var~0) → prefer OLS below"
            lines.append(f"  [{col}] mixedlm width b={fit.params['w']:+.3f} p={fit.pvalues['w']:.2e} | "
                         f"t2p b={fit.params['t']:+.3f} p={fit.pvalues['t']:.2e}{flag}")
        except Exception as e:
            lines.append(f"  [{col}] mixedlm failed: {e}")
        try:
            ols = smf.ols(f"{col} ~ w + t", m).fit(cov_type="cluster",
                                                   cov_kwds={"groups": m["session"]})
            lines.append(f"  [{col}] OLS(cl) width b={ols.params['w']:+.3f} p={ols.pvalues['w']:.2e} | "
                         f"t2p b={ols.params['t']:+.3f} p={ols.pvalues['t']:.2e}  n={len(m)}")
        except Exception as e:
            lines.append(f"  [{col}] cluster-robust OLS failed: {e}")

    # ── caveats (make the deliverable self-documenting) ──
    n_fsi = int((lab.celltype == "FSI").sum()); n_spn = int((lab.celltype == "SPN").sum())
    lines.append(f"YIELD-BIAS CAVEAT: FSI:SPN = {n_fsi}:{n_spn} in the labeled sample — narrow cells are")
    lines.append("  OVER-SAMPLED; do NOT read population fractions as biology. The within-sample t2p↔width")
    lines.append("  relationship + the independence test (which don't depend on the FSI/SPN marginal) are what matter.")

    # ── figure ──
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.32)
    axA = fig.add_subplot(gs[0, 0])
    for rg, c in (("DMS", "#3474ae"), ("VMS", "#ef6548")):
        sub = dd[dd.region == rg]
        axA.scatter(sub.t2p_ms, sub[WIDTH], s=12, alpha=0.4, color=c, edgecolors="none", label=rg)
    axA.axvline(tmed, color="0.6", ls=":"); axA.axhline(med, color="0.6", ls=":")
    axA.set_xlabel("trough-to-peak t2p (ms)  [narrow←|→broad]"); axA.set_ylabel(f"kernel width {WIDTH} (s)")
    axA.set_title(f"2D joint: t2p vs width  rho={rho_all:+.2f}", fontsize=10.5); axA.legend(frameon=False)

    axB = fig.add_subplot(gs[0, 1])
    frac = ct.div(ct.sum(1), axis=0)
    bottom = np.zeros(len(frac))
    for cc, col in (("FSI", "#d94801"), ("SPN", "#08519c")):
        if cc in frac:
            axB.bar(frac.index, frac[cc], bottom=bottom, color=col, label=cc); bottom += frac[cc].values
    axB.set_ylabel("fraction"); axB.set_title(f"overlap: width-class × waveform\nχ²={chi2:.1f} p={pchi:.1e}", fontsize=10.5)
    axB.legend(frameon=False)

    axT = fig.add_subplot(gs[:, 2]); axT.axis("off")
    axT.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=6.6, family="monospace",
             transform=axT.transAxes)
    fig.suptitle("Part 2 — Does the transient/sustained axis map onto narrow/broad (FSI/SPN)?",
                 fontsize=13, y=1.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"width_vs_waveform.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (OUT / "width_vs_waveform_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/width_vs_waveform.png")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
