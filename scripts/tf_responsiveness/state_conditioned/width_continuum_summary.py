"""Clean summary of the transient->sustained TF-kernel width axis — a denoised alternative
to the (inherently noisy) per-cell raw-pulse-PETH heatmap.

Why not the heatmap: the per-cell raw fast-pulse PETH is ~20x below spiking noise, so a
520-row heatmap of it is speckled even though each cell's kernel WIDTH (the sort key) is
exact. The message lives in the DENOISED GLM kernel, so we (A) average kernels within width
bins — noise cancels across cells — and (B) show every cell as one point on the continuous
width axis and its coupling.

Panel A  width-binned MEAN GLM kernel, stacked narrow(top)->broad(bottom). Averaged at the
         BIN level (not per cell), then each bin-mean scaled to unit peak so bins are shape-
         comparable; +/-1 SEM shaded. A half-max bar marks each bin's width. The peak
         visibly broadens down the stack.
Panel B  width (x) vs FA-motor coupling (y), each cell a point coloured by width, with a
         decile mean+-bootstrap-CI trend and a top marginal width histogram. Shows the axis
         is a CONTINUOUS spectrum (one mode, no gap) AND that broad cells couple more.

Cache-only (kernel_width_continuous.csv + kernel_vectors_*.npz). No session reloads.
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
from continuum_common import load_width_metrics, binned_trend, _cmap, REPO, WIDTH  # noqa: E402
from latency_outcome_coupling import load_unresponsive_reference                   # noqa: E402
from visdetect.analysis.kernel_width import interpolated_fwhm                      # noqa: E402

OUT = Path(REPO) / "FIGURES/tf_glm_bg046/width_continuum_summary"
CACHE = Path(REPO) / "data/cache/tf_glm_bg046"
N_BINS = 6
COUPLE = ("fa_ramp", "FA-lick motor coupling (Δ Hz)")


def _load_kernels():
    """(subject, session, unit) -> sign-aligned kernel; plus the lag axis."""
    kmap, lags = {}, None
    for subj in ("BG_031", "BG_039", "BG_046"):
        f = CACHE / f"kernel_vectors_{subj}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        lags = np.asarray(z["lags"], float)
        for k in z.files:
            if k in ("lags", "units"):
                continue
            sess, uid = k.rsplit("_u", 1)
            K = np.asarray(z[k], float)
            s = np.sign(K[np.argmax(np.abs(K))]) or 1.0
            kmap[(subj, sess, int(uid))] = K * s        # dominant deflection made positive
    return kmap, lags


def main():
    d = load_width_metrics()
    kmap, lags = _load_kernels()
    d = d[np.isfinite(d[WIDTH])].copy()
    # attach each cell's sign-aligned kernel
    K = np.full((len(d), lags.size), np.nan)
    for i, r in enumerate(d.itertuples()):
        k = kmap.get((str(r.subject), str(r.session), int(r.unit)))
        if k is not None:
            K[i] = k
    ok = np.all(np.isfinite(K), axis=1)
    d, K = d[ok].reset_index(drop=True), K[ok]
    width = d[WIDTH].values

    # equal-count width bins
    edges = np.quantile(width, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    binid = np.clip(np.searchsorted(edges, width, side="right") - 1, 0, N_BINS - 1)
    cmap = _cmap()

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass

    fig = plt.figure(figsize=(15, 7.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.15], wspace=0.24)

    # ── Panel A: width-binned mean-kernel ridgeline ─────────────────────────────
    # PEAK-ALIGNED average: each cell's |kernel| peaks at a different lag, so averaging the
    # raw (unaligned) kernels smears the mean and destroys the width it is meant to show.
    # We interpolate each kernel onto a PEAK-RELATIVE lag grid (t=0 at that cell's peak),
    # then average within the bin. This is an illustrative "average response shape" — the
    # width of each bin's mean is dominated by the cells' own (independently measured) FWHM,
    # which is what defines the bins, so it is illustrative-by-construction, not a test.
    axA = fig.add_subplot(gs[0, 0])
    rel = np.arange(-0.30, 1.00 + 1e-9, float(np.median(np.diff(lags))))   # peak-relative lags
    Krel = np.full((len(K), rel.size), np.nan)
    for i in range(len(K)):
        pk_lag = lags[int(np.argmax(np.abs(K[i])))]
        Krel[i] = np.interp(pk_lag + rel, lags, K[i], left=0.0, right=0.0)
    off = 1.25
    lines = ["WIDTH-BINNED MEAN KERNEL (ridgeline) — PEAK-ALIGNED bin average, unit-peak scaled",
             "(illustrative average response shape; bins defined by each cell's own interp_fwhm)",
             "-" * 62]
    for b in range(N_BINS):
        sel = binid == b
        km = np.nanmean(Krel[sel], axis=0)
        sem = np.nanstd(Krel[sel], axis=0) / np.sqrt(max(sel.sum(), 1))
        pk = np.max(np.abs(km)) + 1e-12
        km, sem = km / pk, sem / pk                     # unit-peak (bin mean, not per cell)
        y0 = (N_BINS - 1 - b) * off
        c = cmap(b / (N_BINS - 1))
        axA.fill_between(rel, y0 + km - sem, y0 + km + sem, color=c, alpha=0.30, lw=0)
        axA.fill_between(rel, y0, y0 + km, color=c, alpha=0.75, lw=0)
        axA.plot(rel, y0 + km, color="k", lw=0.9)
        # half-max width bar of the bin-mean kernel (peak-relative)
        fw = interpolated_fwhm(km, rel)
        ipk = int(np.argmax(np.abs(km)))
        axA.plot([rel[ipk] - fw / 2, rel[ipk] + fw / 2], [y0 + 0.5, y0 + 0.5],
                 color="k", lw=2.0, solid_capstyle="butt")
        axA.text(rel[-1], y0 + 0.18,
                 f"width {edges[b]:.02f}–{edges[b+1]:.02f}s  (n={sel.sum()})",
                 fontsize=8.2, va="bottom", ha="right")
        lines.append(f"  bin{b} width {edges[b]:.3f}-{edges[b+1]:.3f}s n={sel.sum()}: "
                     f"peak-aligned bin-mean fwhm={fw:.3f}s")
    axA.axvline(0, color="0.5", lw=0.9, ls="--")
    axA.set_yticks([])
    axA.set_xlabel("lag relative to each cell's kernel peak (s)")
    axA.set_title("A  peak-aligned mean GLM kernel by width bin\nnarrow (top) → broad "
                  "(bottom): the response duration grows", fontsize=11)
    axA.set_xlim(rel[0], rel[-1])

    # ── Panel B: width vs coupling scatter + width marginal ─────────────────────
    gsB = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1],
                                           height_ratios=[1, 4], hspace=0.06)
    axM = fig.add_subplot(gsB[0]); axS = fig.add_subplot(gsB[1], sharex=None)
    col, clab = COUPLE
    sub = d[[WIDTH, col]].replace([np.inf, -np.inf], np.nan).dropna()
    # top marginal: width distribution (unimodal spectrum, no gap)
    axM.hist(width, bins=np.linspace(width.min(), np.percentile(width, 99.5), 40),
             color="0.6", edgecolor="none")
    axM.axvline(np.median(width), color="k", ls="--", lw=1)
    axM.set_yticks([]); axM.tick_params(labelbottom=False)
    for sp in ("top", "right", "left"):
        axM.spines[sp].set_visible(False)
    axM.set_title("B  every cell = one point on the continuous width axis (spectrum, no gap);\n"
                  "broad-kernel cells couple more to the impulsive lick", fontsize=11)
    # ── TF-UNRESPONSIVE reference band ─────────────────────────────────────────
    # The 520 analysed cells are only 4.5% of the 11,598 recorded. This band is where the
    # other ~11k (TF-UNRESPONSIVE) cells sit on the SAME coupling metric, measured with the
    # identical windows/clamp — i.e. what "no TF response" looks like. It is a horizontal
    # band, not a scatter: an unresponsive cell's kernel width is noise, so it has no
    # position on the x-axis.
    ref = load_unresponsive_reference()
    r = ref.get(col)
    if r:
        axS.axhspan(r["q25"], r["q75"], color="0.55", alpha=0.18, zorder=0,
                    label=f"TF-unresponsive IQR (n={r['n']:,})")
        axS.axhline(r["median"], color="0.35", lw=1.5, ls="--", zorder=2,
                    label=f"TF-unresponsive median ({r['median']:+.2f} Hz)")

    # main scatter coloured by width + decile trend
    axS.scatter(sub[WIDTH], sub[col], c=sub[WIDTH], cmap="viridis", s=16, alpha=0.7,
                edgecolors="none", zorder=1)
    binned_trend(axS, sub[WIDTH].values, sub[col].values, n_bins=10, color="#d7301f",
                 scatter=False, label="decile mean ± 95% CI")
    axS.axhline(0, color="0.6", lw=0.8, ls=":")
    axS.set_xlabel("kernel FWHM (s)  — transient ↔ sustained")
    axS.set_ylabel(clab)
    # Y-CLIP (disclosed): a handful of extreme cells stretch the axis to ~[-20,+30] and squash
    # the 0-8 Hz band where the trend and the unresponsive baseline actually live. Clip to the
    # 1-99th percentile of the responsive cells and SAY how many points fall outside — never
    # silently. The statistic (Spearman, rank-based) uses every point regardless.
    ylo, yhi = np.percentile(sub[col], [1, 99])
    pad = 0.12 * (yhi - ylo)
    n_out = int(np.sum((sub[col] < ylo - pad) | (sub[col] > yhi + pad)))
    axS.set_ylim(ylo - pad, yhi + pad)
    if n_out:
        axS.text(0.985, 0.015, f"{n_out}/{len(sub)} cells outside axis range\n"
                               f"(stats use all points)",
                 transform=axS.transAxes, ha="right", va="bottom", fontsize=7, color="0.35")
    axS.legend(frameon=False, fontsize=7.6, loc="upper right")
    # align marginal x to scatter x
    axM.set_xlim(axS.get_xlim())

    rho, p = spearmanr(sub[WIDTH], sub[col])
    lines += ["", f"width vs {col}: Spearman rho={rho:+.3f} p={p:.2e} n={len(sub)}",
              f"width distribution: n={len(width)} median={np.median(width):.3f}s "
              f"range {width.min():.3f}-{width.max():.3f}s"]

    # ── is the NARROW end already at the TF-unresponsive baseline? ──────────────
    if r:
        from scipy.stats import mannwhitneyu
        unresp = pd.read_csv(Path(REPO) / "FIGURES/tf_glm_bg046/latency_outcome_coupling/"
                             "latency_outcome_metrics_unresponsive.csv")
        uv = unresp[col].replace([np.inf, -np.inf], np.nan).dropna().values
        lines += ["", "TF-UNRESPONSIVE REFERENCE (same windows/clamp; ~4.5% of cells are responsive)",
                  f"  unresponsive {col}: median={r['median']:+.3f} IQR[{r['q25']:+.3f},{r['q75']:+.3f}] n={r['n']}"]
        # narrowest vs broadest responsive bin, each vs the unresponsive baseline
        for lab, sel in (("narrowest bin", binid == 0), ("broadest bin", binid == N_BINS - 1)):
            v = d.loc[sel, col].replace([np.inf, -np.inf], np.nan).dropna().values
            if v.size:
                u, pu = mannwhitneyu(v, uv)
                # verdict must respect significance, not just which median is larger
                if pu >= 0.05:
                    verdict = "-> INDISTINGUISHABLE from the unresponsive baseline"
                else:
                    verdict = ("-> significantly ABOVE baseline" if np.median(v) > r["median"]
                               else "-> significantly BELOW baseline")
                lines.append(f"  {lab:14s} median={np.median(v):+.3f} (n={v.size}) "
                             f"vs unresponsive: MWU p={pu:.2e}  {verdict}")

    fig.suptitle("The transient→sustained TF-kernel width axis (denoised summary; replaces the "
                 "noisy per-cell PETH heatmap)", fontsize=13, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"width_continuum_summary.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (OUT / "width_continuum_summary_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    for ln in lines:
        print(ln.encode("ascii", "replace").decode())
    print(f"\nwrote {OUT}/width_continuum_summary.png")


if __name__ == "__main__":
    main()
