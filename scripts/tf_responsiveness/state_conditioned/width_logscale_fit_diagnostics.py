"""Supplementary fit diagnostics for the log-scale kernel-width distribution.

Companion to `width_logscale_distribution.py` (which is the HEADLINE, parsimonious
2-parameter lognormal view). This figure interrogates the residual misfit the plain
lognormal leaves behind, WITHOUT abandoning the lognormal story:

  * a KDE of the data on each panel, so the fit is judged against a smooth empirical
    density rather than noisy histogram bars;
  * a dashed 3-PARAMETER SHIFTED lognormal (location left free) next to the 2-parameter
    (floc=0) lognormal -- the residual right-skew collapses to a tiny location offset
    and the Kolmogorov-Smirnov test then stops rejecting, i.e. the deviation is a small
    offset, not a failure of lognormality;
  * a PER-MOUSE row -- the single-mouse VMS (BG_031) is a near-exact lognormal
    (log-skew ~0), BG_046 keeps a small genuine right-skew, and pooling mice with
    different median widths (a mixture) is part of why the pooled panel looks least snug.

Honest framing (kept from the headline figure): this is DISPLAY/goodness-of-fit only.
The width->coupling results use Spearman (rank-based) + equal-count bins, both invariant
to a monotone transform, so no conclusion changes.

Self-contained: fits are plain scipy MLE (lognorm floc=0 vs loc-free), no dependency on
the spectrum_stats helper. Cache-only (kernel_width_continuous.csv via load_width_metrics).

Usage:  py scripts/tf_responsiveness/state_conditioned/width_logscale_fit_diagnostics.py
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

OUT = Path(REPO) / "FIGURES/tf_glm_bg046/width_logscale_fit_diagnostics"
KDE_C = "0.45"          # empirical density
LN2_C = "#2171b5"       # 2-parameter lognormal (floc=0)
LN3_C = "#cb181d"       # 3-parameter shifted lognormal (loc free)
WIDTH_TICKS = [0.03, 0.05, 0.1, 0.2, 0.4, 0.7]


def _pos(w):
    w = np.asarray(w, float)
    return w[np.isfinite(w) & (w > 0)]


def _fits(w):
    """2-param (floc=0) and 3-param (loc-free) lognormal MLE + KS + log-skew."""
    w = _pos(w)
    lg = np.log10(w)
    p2 = stats.lognorm.fit(w, floc=0)
    p3 = stats.lognorm.fit(w)
    return {
        "w": w, "lg": lg, "n": int(w.size),
        "p2": p2, "ks2": float(stats.kstest(w, "lognorm", args=p2).pvalue),
        "p3": p3, "ks3": float(stats.kstest(w, "lognorm", args=p3).pvalue),
        "skew_log": float(stats.skew(lg)),
        "loc3": float(p3[1]),
    }


def _logdens(params, u):
    """A linear-x lognormal pdf expressed as a density in log10(x)=u units
    (Jacobian dx/du = x*ln10), so it can overlay a histogram of log10(width)."""
    x = 10.0 ** u
    return stats.lognorm.pdf(x, *params) * x * np.log(10.0)


def _kde(data, grid, factor=1.4):
    """A gently OVER-smoothed KDE (1.4x Scott bandwidth): the display job here is a
    clean empirical density, and the default bandwidth injects spurious little modes
    (finite-sample noise) that would misread as bimodality in a unimodal continuum."""
    k = stats.gaussian_kde(data)
    k.set_bandwidth(k.factor * factor)
    return k(grid)


def _log_panel(ax, f, title, legend=False):
    lg = f["lg"]
    bins = np.linspace(lg.min(), lg.max(), 24)
    ax.hist(lg, bins=bins, density=True, color="0.85", edgecolor="white", lw=0.3)
    u = np.linspace(lg.min(), lg.max(), 400)
    ax.plot(u, _kde(lg, u), color=KDE_C, lw=1.6, label="KDE (data)")
    ax.plot(u, _logdens(f["p2"], u), color=LN2_C, lw=2.2, label="lognormal (2-param)")
    ax.plot(u, _logdens(f["p3"], u), color=LN3_C, lw=2.0, ls="--",
            label="shifted lognormal (3-param)")
    ax.set_xticks(np.log10(WIDTH_TICKS))
    ax.set_xticklabels([f"{t:g}" for t in WIDTH_TICKS])
    ax.set_xlabel("kernel width interp_fwhm (s, log scale)")
    ax.set_ylabel("density")
    ax.set_title(f"{title}\nn={f['n']}  log-skew={f['skew_log']:+.2f}  "
                 f"KS: 2p={f['ks2']:.3f} → 3p={f['ks3']:.3f}", fontsize=9.5)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _linear_panel(ax, f):
    w = f["w"]
    bins = np.linspace(w.min(), np.nanpercentile(w, 99.5), 26)
    ax.hist(w, bins=bins, density=True, color="0.85", edgecolor="white", lw=0.3)
    xg = np.linspace(w.min(), np.nanpercentile(w, 99.9), 500)
    ax.plot(xg, _kde(w, xg), color=KDE_C, lw=1.6, label="KDE (data)")
    ax.plot(xg, stats.lognorm.pdf(xg, *f["p2"]), color=LN2_C, lw=2.2, label="lognormal (2-param)")
    ax.plot(xg, stats.lognorm.pdf(xg, *f["p3"]), color=LN3_C, lw=2.0, ls="--",
            label="shifted lognormal (3-param)")
    ax.set_xlabel("kernel width interp_fwhm (s)")
    ax.set_ylabel("density")
    ax.set_title("pooled — linear axis (tail compressed → misfit hidden)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _qq_panel(ax, f):
    w = np.sort(f["w"]); n = w.size
    p = (np.arange(1, n + 1) - 0.5) / n
    ax.scatter(stats.lognorm.ppf(p, *f["p2"]), w, s=8, color=LN2_C, alpha=0.5,
               edgecolors="none", label=f"2-param (KS p={f['ks2']:.3f})")
    ax.scatter(stats.lognorm.ppf(p, *f["p3"]), w, s=8, color=LN3_C, alpha=0.5,
               edgecolors="none", label=f"3-param (KS p={f['ks3']:.3f})")
    lo, hi = float(w.min()), float(np.nanpercentile(w, 99.8))
    ax.plot([lo, hi], [lo, hi], color="k", lw=1.0, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("fitted quantile (s)")
    ax.set_ylabel("observed width quantile (s)")
    ax.set_title("pooled Q-Q: 2-param vs 3-param lognormal\n(points on the line = good fit)",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    d = load_width_metrics()
    pooled = _fits(d[WIDTH].to_numpy(float))
    mice = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
    mouse_fits = {m: _fits(d.loc[d.subject == m, WIDTH].to_numpy(float)) for m, _ in mice}

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.46, wspace=0.30)

    _linear_panel(fig.add_subplot(gs[0, 0]), pooled)
    _log_panel(fig.add_subplot(gs[0, 1]), pooled, "pooled — log axis (KDE + 2p vs 3p)", legend=True)
    _qq_panel(fig.add_subplot(gs[0, 2]), pooled)
    for j, (m, reg) in enumerate(mice):
        tag = "near-exact lognormal" if m == "BG_031" else "small residual skew"
        _log_panel(fig.add_subplot(gs[1, j]), mouse_fits[m], f"{m} ({reg}) — {tag}",
                   legend=(j == 0))

    fig.suptitle("Kernel-width lognormal fit — DIAGNOSTICS (supplementary to the parsimonious "
                 "2-param headline figure)\n"
                 "KDE vs 2-param vs dashed 3-param shifted lognormal; the residual right-skew "
                 "collapses to a tiny location offset (KS stops rejecting), and single-mouse VMS "
                 "(BG_031) is near-exact — pooling different-median mice is part of the misfit",
                 fontsize=12, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"width_logscale_fit_diagnostics.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)

    lines = ["Kernel-width lognormal fit diagnostics (2-param floc=0 vs 3-param loc-free)",
             "KS p: larger = better; 3-param frees the location (a small shift).", ""]
    for name, f in [("ALL", pooled)] + [(m, mouse_fits[m]) for m, _ in mice]:
        lines.append(f"{name:8s} n={f['n']:3d}  log-skew={f['skew_log']:+.2f}  "
                     f"KS 2p={f['ks2']:.3f} -> 3p={f['ks3']:.3f}  (3p loc shift={f['loc3']:+.4f} s)")
    lines += ["",
              "Headline stays the parsimonious 2-param lognormal (beats gamma+normal, Buzsaki).",
              "3-param shifted lognormal shows the residual is a small OFFSET, not non-lognormality.",
              "Single-mouse BG_031 (VMS) is near-exact; pooling different-median mice adds mixture blur.",
              "DISPLAY ONLY: width->coupling Spearman + equal-count bins are log-invariant -> no",
              "conclusion changes."]
    (OUT / "width_logscale_fit_diagnostics_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/width_logscale_fit_diagnostics.png (+.pdf, +_stats.txt)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
