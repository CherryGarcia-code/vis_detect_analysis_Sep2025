"""Log-scale view of the transient->sustained kernel-width distribution.

The kernel-width axis interp_fwhm (transient=narrow -> sustained=broad) is strongly
right-skewed and heavy-tailed (see core_metrics_continuum, width-distribution panel).
That is the shape Buzsaki & Mizuseki 2014 ("The log-dynamic brain") describe for many
neural variables: they are ~LOGNORMAL, and LOG is their natural axis. This figure
displays the width distribution on a log axis and formally asks whether it is
lognormal or merely gamma/otherwise skewed.

ADDED ALONGSIDE the existing (linear-axis) width displays -- nothing is overwritten.

Honest scope: this is a DISPLAY/shape figure, not a change to any conclusion. The
width->coupling results use Spearman (rank-based) and equal-count bins, BOTH invariant
to a monotone log transform -- so the correlations and the "graded not stepped" reading
are unchanged. What the log axis buys is (a) a natural scale for a heavy-tailed positive
quantity and (b) the direct lognormal test: if width is lognormal, log(width) is a
symmetric bell.

Panels:
  A  linear-axis width histogram + lognormal & gamma MLE fits (right-skewed)
  B  log10(width) histogram + normal fit -> the "symmetrised bell" (lognormal signature)
  C  Q-Q vs the fitted lognormal AND gamma -> which family tracks the tail
  D/E per-region (DMS, VMS) log10(width) + normal fit + AIC-best family
  F  fit-comparison stats (AIC/BIC/KS per family, pooled + per region) + notes

fit_compare_distributions (visdetect.analysis.spectrum_stats) does the MLE + AIC/KS.
Cache-only (kernel_width_continuous.csv via load_width_metrics).

Usage:  py scripts/tf_responsiveness/state_conditioned/width_logscale_distribution.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import load_width_metrics, WIDTH, REPO  # noqa: E402
from visdetect.analysis.spectrum_stats import fit_compare_distributions  # noqa: E402

OUT = Path(REPO) / "FIGURES/tf_glm_bg046/width_logscale_distribution"
LN_C = "#2171b5"   # lognormal
GA_C = "#d95f0e"   # gamma
NO_C = "#238b45"   # normal (of log-width)
WIDTH_TICKS = [0.03, 0.05, 0.1, 0.2, 0.4, 0.7]   # actual-width ticks for the log axes


def _frozen(fam, params):
    return getattr(stats, fam)(*params)


def _pos(w):
    w = np.asarray(w, float)
    return w[np.isfinite(w) & (w > 0)]


def _linear_panel(ax, w, fits):
    w = _pos(w)
    bins = np.linspace(w.min(), np.nanpercentile(w, 99.5), 28)
    ax.hist(w, bins=bins, density=True, color="0.82", edgecolor="white", lw=0.3)
    grid = np.linspace(w.min(), w.max(), 500)
    for fam, c, lab in [("lognorm", LN_C, "lognormal"), ("gamma", GA_C, "gamma")]:
        ax.plot(grid, _frozen(fam, fits["families"][fam]["params"]).pdf(grid),
                color=c, lw=2.2, label=lab)
    ax.set_xlabel("kernel width interp_fwhm (s)")
    ax.set_ylabel("density")
    ax.set_title(f"linear axis — right-skewed (skew={fits['skew_linear']:+.2f})", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _log_panel(ax, w, title, fits=None):
    """log10(width) histogram + fitted normal-of-log (the lognormal 'symmetric bell')."""
    w = _pos(w)
    lg = np.log10(w)
    mu, sd = float(np.mean(lg)), float(np.std(lg, ddof=1))
    bins = np.linspace(lg.min(), lg.max(), 26)
    ax.hist(lg, bins=bins, density=True, color="0.82", edgecolor="white", lw=0.3)
    grid = np.linspace(lg.min(), lg.max(), 400)
    ax.plot(grid, stats.norm.pdf(grid, mu, sd), color=NO_C, lw=2.2,
            label=f"normal of log (skew={stats.skew(lg):+.2f})")
    ax.axvline(mu, color="k", ls="--", lw=1.2)
    ax.set_xticks(np.log10(WIDTH_TICKS))
    ax.set_xticklabels([f"{t:g}" for t in WIDTH_TICKS])
    ax.set_xlabel("kernel width interp_fwhm (s, log scale)")
    ax.set_ylabel("density")
    if fits is not None:
        best = fits["best_aic"]
        d_ai = fits["families"]["gamma"]["aic"] - fits["families"]["lognorm"]["aic"]
        ax.set_title(f"{title}\nAIC-best = {best.upper()}  "
                     f"(lognorm−gamma AIC Δ={-d_ai:+.1f})", fontsize=10)
    else:
        ax.set_title(title, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _qq_panel(ax, w, fits):
    w = np.sort(_pos(w))
    n = w.size
    p = (np.arange(1, n + 1) - 0.5) / n
    for fam, c, lab in [("lognorm", LN_C, "lognormal"), ("gamma", GA_C, "gamma")]:
        theo = _frozen(fam, fits["families"][fam]["params"]).ppf(p)
        ax.scatter(theo, w, s=8, color=c, alpha=0.5, edgecolors="none",
                   label=f"{lab} (KS p={fits['families'][fam]['ks_p']:.2g})")
    lo, hi = float(w.min()), float(np.nanpercentile(w, 99.8))
    ax.plot([lo, hi], [lo, hi], color="k", lw=1.0, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("fitted-distribution quantile (s)")
    ax.set_ylabel("observed width quantile (s)")
    ax.set_title("Q-Q: which family tracks the tail\n(points on the dashed line = good fit)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _fmt_fit(tag, f):
    fam = f["families"]
    best = f["best_aic"]
    out = [f"{tag}: n={f['n']}  skew(lin)={f['skew_linear']:+.2f}  skew(log)={f['skew_log']:+.2f}"
           f"  -> AIC-best = {best.upper()}"]
    for name in ("lognorm", "gamma", "norm"):
        fm = fam[name]
        out.append(f"    {name:8s} AIC={fm['aic']:+9.1f}  BIC={fm['bic']:+9.1f}  "
                   f"KS={fm['ks_stat']:.3f} (p={fm['ks_p']:.2g})")
    return out


def main():
    d = load_width_metrics()
    w = d[WIDTH].to_numpy(float)
    fits = fit_compare_distributions(w)
    reg_fits = {reg: fit_compare_distributions(d.loc[d.region == reg, WIDTH].to_numpy(float))
                for reg in ("DMS", "VMS")}

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)

    _linear_panel(fig.add_subplot(gs[0, 0]), w, fits)
    _log_panel(fig.add_subplot(gs[0, 1]), w, "log10(width) — the symmetrised bell (all cells)", fits)
    _qq_panel(fig.add_subplot(gs[0, 2]), w, fits)
    _log_panel(fig.add_subplot(gs[1, 0]), d.loc[d.region == "DMS", WIDTH].to_numpy(float),
               "DMS (BG_046 + BG_039)", reg_fits["DMS"])
    _log_panel(fig.add_subplot(gs[1, 1]), d.loc[d.region == "VMS", WIDTH].to_numpy(float),
               "VMS (BG_031)", reg_fits["VMS"])

    axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
    lines = ["FIT COMPARISON  (MLE; lognorm/gamma floc=0; AIC-best wins)", ""]
    lines += _fmt_fit("ALL", fits) + [""]
    lines += _fmt_fit("DMS", reg_fits["DMS"]) + [""]
    lines += _fmt_fit("VMS", reg_fits["VMS"]) + [""]
    lines += [
        "Buzsaki & Mizuseki 2014 (log-dynamic brain): many neural",
        "variables are lognormal; LOG is their natural axis.",
        "",
        "DISPLAY ONLY: width->coupling uses Spearman (rank-based) +",
        "equal-count bins, both invariant to a log transform -> the",
        "correlations and 'graded not stepped' reading are UNCHANGED.",
        "Even where gamma wins, the log axis remains the right display",
        "for a heavy-tailed positive quantity.",
    ]
    axt.text(0.0, 1.0, "\n".join(lines), transform=axt.transAxes, va="top", ha="left",
             fontsize=8.2, family="monospace")

    fig.suptitle("Transient→sustained kernel width on a LOGARITHMIC scale — is it lognormal? "
                 "(Buzsáki log-dynamic view; added alongside the linear displays)\n"
                 "log10(width) collapses the right skew toward a symmetric bell — the lognormal "
                 "signature; Q-Q + AIC say which family (lognormal vs gamma) tracks the tail",
                 fontsize=12.5, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"width_logscale_distribution.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)

    stats_txt = ["Transient->sustained kernel-width distribution — log-scale / lognormal test", ""]
    stats_txt += _fmt_fit("ALL", fits) + [""] + _fmt_fit("DMS", reg_fits["DMS"]) + [""] \
        + _fmt_fit("VMS", reg_fits["VMS"])
    (OUT / "width_logscale_distribution_stats.txt").write_text("\n".join(stats_txt), encoding="utf-8")
    print(f"wrote {OUT}/width_logscale_distribution.png (+.pdf, +_stats.txt)")
    for s in stats_txt:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
