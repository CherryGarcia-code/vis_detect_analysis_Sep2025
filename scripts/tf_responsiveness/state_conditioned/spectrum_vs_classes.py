# scripts/tf_responsiveness/state_conditioned/spectrum_vs_classes.py
"""Part 1: is the transient/sustained (temporal) identity a spectrum or two classes?

Continuous width (Component A) -> modality battery (GMM ΔBIC primary; Silverman;
Sarle BC; optional Hartigan dip) pooled + per region, latency⊥width check, and a
graded-vs-stepped (segmented-vs-linear BIC) test of outcome coupling on width.
Reads the cache only; no session reloads.
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
from scipy.stats import spearmanr

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO                                  # noqa: E402
from visdetect.analysis.spectrum_stats import (                       # noqa: E402
    gmm_delta_bic, bimodality_coefficient, silverman_bootstrap, dip_test,
    segmented_vs_linear,
)

CACHE = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"
PULSE_ALL = Path(REPO) / "data/cache/tf_glm_bg046/pulse_fwhm_allpulses.csv"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/spectrum_vs_classes"
REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"}
# hit_ramp is NOT an independent outcome: the hit lick follows the change by ~0.64 s, so
# its window is largely the change-evoked response (rho~+0.58 with change_on). It is a
# consistency check, not a third test; fa_ramp (early lick, NO change stimulus) is the
# independent motor probe (width->fa_ramp survives controlling change_on, partial +0.28).
OUTCOMES = [("change_on", "Change_ON (sensory)"),
            ("hit_ramp", "Hit pre-lick (≈change resp)"),
            ("fa_ramp", "FA motor ramp (independent)")]
WIDTH = "interp_fwhm"  # primary continuous width
# Model-free width measure: the GUARDED all-pulse fast-minus-slow contrast, NOT the stale
# 600-cap `pulse_fwhm` column (which was noise, rho=+0.045 to interp_fwhm).
PULSE_MEASURE = "pulse_fwhm_all"


def main():
    d = pd.read_csv(CACHE)
    if PULSE_ALL.exists():
        pa = pd.read_csv(PULSE_ALL, dtype={"session": str})[
            ["subject", "session", "unit", "pulse_fwhm_all"]]
        d = d.merge(pa, on=["subject", "session", "unit"], how="left")
    d["region"] = d.subject.map(REGION)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass

    stat_rows, lines = [], []
    # ── modality battery on each width measure, pooled + per region ──
    for measure in [WIDTH, "temporal_spread", PULSE_MEASURE]:
        for scope, sub in [("pooled", d)] + [(rg, d[d.region == rg]) for rg in ("DMS", "VMS")]:
            x = sub[measure].replace([np.inf, -np.inf], np.nan).dropna().values
            gm = gmm_delta_bic(x); si = silverman_bootstrap(x, n_boot=300)
            dp = dip_test(x); bc = bimodality_coefficient(x)
            stat_rows.append(dict(measure=measure, scope=scope, n=len(x),
                                  gmm_delta_bic=gm["delta_bic"], gmm_means=gm["means"],
                                  gmm_weights=gm["weights"], silverman_p_unimodal=si["p_unimodal"],
                                  dip=dp["dip"], dip_p=dp["p"], bimodality_coef=bc))
    lines.append("MODALITY (positive ΔBIC & low silverman-p & BC>0.555 => classes; else spectrum):")
    for r in stat_rows:
        lines.append(f"  [{r['measure']}/{r['scope']}] n={r['n']} ΔBIC={r['gmm_delta_bic']:+.1f} "
                     f"silverman_p={r['silverman_p_unimodal']:.3f} dip_p={r['dip_p']} BC={r['bimodality_coef']:.3f}")

    # ── latency ⊥ width ──
    rho_lw, p_lw = spearmanr(d.kernel_peak_t_registry, d[WIDTH], nan_policy="omit")
    lines.append(f"latency(peak_t) vs width({WIDTH}): rho={rho_lw:+.3f} p={p_lw:.2e}")

    # ── graded vs stepped: outcome ~ width ──
    seg_rows = []
    for col, lab in OUTCOMES:
        sub = d[[WIDTH, col]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p = spearmanr(sub[WIDTH], sub[col])
        seg = segmented_vs_linear(sub[WIDTH].values, sub[col].values)
        seg_rows.append(dict(outcome=col, spearman_rho=rho, spearman_p=p, **seg))
        lines.append(f"  [{col}] Spearman rho={rho:+.3f} p={p:.2e} | segmented ΔBIC(seg-vs-lin)="
                     f"{seg['delta_bic']:+.1f} breakpoint={seg['breakpoint']:.3f} "
                     f"(ΔBIC<=6 => graded continuum; >10 => threshold)")

    # ── figure: A width hist+GMM, B latency-vs-width, C-E outcome-vs-width curves ──
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)
    axA = fig.add_subplot(gs[0, 0])
    for rg, c in (("DMS", "#3474ae"), ("VMS", "#ef6548")):
        axA.hist(d.loc[d.region == rg, WIDTH].dropna(), bins=np.linspace(0, 0.8, 33),
                 histtype="step", lw=2, density=True, color=c, label=rg)
    gm_pool = gmm_delta_bic(d[WIDTH].dropna().values)
    axA.set_title(f"continuous kernel width — GMM ΔBIC={gm_pool['delta_bic']:+.1f}\n"
                  f"(means {['%.2f'%m for m in gm_pool['means']]})", fontsize=10.5)
    axA.set_xlabel(f"{WIDTH} (s)"); axA.set_ylabel("density"); axA.legend(frameon=False)

    axB = fig.add_subplot(gs[0, 1])
    axB.scatter(d.kernel_peak_t_registry, d[WIDTH], s=10, alpha=0.4, color="0.4", edgecolors="none")
    axB.set_xlabel("kernel peak latency (s)"); axB.set_ylabel(f"{WIDTH} (s)")
    axB.set_title(f"latency ⊥ width  rho={rho_lw:+.2f}", fontsize=10.5)

    for i, (col, lab) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[1, i])
        sub = d[[WIDTH, col]].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(sub[WIDTH], sub[col], s=8, alpha=0.25, color="0.5", edgecolors="none")
        q = pd.qcut(sub[WIDTH], 8, duplicates="drop")
        binned = sub.groupby(q, observed=True).agg(x=(WIDTH, "median"), y=(col, "median"))
        ax.plot(binned.x, binned.y, "o-", color="#238b45", lw=2, label="binned median")
        sr = next(r for r in seg_rows if r["outcome"] == col)
        ax.set_title(f"{lab}: rho={sr['spearman_rho']:+.2f}, segΔBIC={sr['delta_bic']:+.1f}", fontsize=10)
        ax.set_xlabel(f"{WIDTH} (s)"); ax.set_ylabel("Δ firing (Hz)"); ax.legend(frameon=False, fontsize=8)

    axT = fig.add_subplot(gs[0, 2]); axT.axis("off")
    axT.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=7.2, family="monospace",
             transform=axT.transAxes)
    fig.suptitle("Part 1 — Is transient/sustained a spectrum or two classes? (continuous kernel width)",
                 fontsize=13, y=1.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"spectrum_vs_classes.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stat_rows).to_csv(OUT / "spectrum_vs_classes_modality.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(OUT / "spectrum_vs_classes_segmented.csv", index=False)
    (OUT / "spectrum_vs_classes_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/spectrum_vs_classes.png")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
