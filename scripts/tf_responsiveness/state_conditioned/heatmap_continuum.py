"""Continuum re-render of the §3 transient/sustained heatmap figure.

The class figure (`heatmap_transient_sustained.py`) ordered the population heatmap
rows by a transient/sustained CLASS block and summarised each alignment with two
class-mean PSTHs. The spectrum result (docs 2026-07 transient-sustained-spectrum)
showed kernel width is a GRADED axis, not two classes, so here:

  * every row is ordered by the CONTINUOUS kernel width (`interp_fwhm`) ascending —
    one global order shared across all three alignments (narrow at TOP, broad at
    BOTTOM), so one heatmap row is one cell everywhere (no transient/sustained blocks);
  * a continuous-width colour strip (viridis) runs down the left of each heatmap,
    oriented to MATCH the rows (narrow top -> broad bottom) and the numeric width
    colorbar;
  * above each heatmap, a PSTH FAMILY replaces the two class-mean traces: cells are
    split into 5 equal-COUNT width bins (`width_bin_assign`; equal n per bin, so the
    top/broad bin spans a wider width range because the width distribution is
    right-skewed) and the mean z-trace per bin is drawn along the viridis gradient,
    with 95% CI (+/-1.96 SEM) shading.

FAST TF PULSE — SIGN-ALIGNED BY THE GLM KERNEL (not by the PETH itself).
A sizeable minority of TF-responsive cells are SUPPRESSION-type (fire less to a fast
pulse; GLM responsiveness is sign-agnostic), so a signed pop-mean cancels excitation
against suppression and we must sign-align before averaging.

⚠️ HOW **NOT** TO DO IT (a bug this figure used to have). The first version took each
cell's sign from its OWN post-pulse (0, 0.4) s window and then averaged that same
trace. That is circular (double-dipping), and it produced two artifacts:
  * a SPURIOUS PRE-PULSE RISE — the traces appeared to climb ~150 ms BEFORE t=0. The
    stimulus cannot cause this: the baseline TF is white noise (autocorrelation
    r ~ 0.000 at 50-200 ms) and the pulse-triggered average of the TF itself is a
    clean delta at t=0. The cause was that a smoothed PETH's pre- and post-pulse bins
    are correlated across cells (r ~ +0.20), so choosing the sign on the post window
    dragged the pre window positive too.
  * an INFLATED RESPONSE — flipping each cell's own noise to be positive lifted the
    population post-pulse mean ~7x (+0.0144 vs +0.0019 with an honest sign).
FIX: take the sign from the GLM TF KERNEL (mean over the same 0-0.4 s lag window).
The kernel is an independent estimator of the same quantity — fit on the full
continuous TF regression with lick/movement/reward nuisances regressed out — so it is
not derived from the trace being averaged. The suppression fraction it reports
(~30-40%, window-dependent: 25-51% across defensible kernel lag windows, so quote a
RANGE not a point) is the honest one; the old PETH-derived "~49%" was a coin-flip on
noise (its post-window response sits below its own noise floor).

Change / FA are predominantly excitatory and are shown raw (no sign-alignment).

Per-panel colour scaling: change / FA on an asymmetric `TwoSlopeNorm(-1.5, 0, 3)`
baseline-z (preserves magnitude); the sign-aligned pulse on a robust percentile-
clipped baseline-z (cleaner than the old peak-norm, which amplified weak-cell noise).
Cache-only: reads the rebuilt peth_traces_all.npz (all 520 responsive cells, incl.
the ~106 intermediates the class figure dropped) + kernel_width_continuous.csv.

Usage:  py scripts/tf_responsiveness/state_conditioned/heatmap_continuum.py
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
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter1d

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import (  # noqa: E402
    load_width_metrics, width_bin_assign, WIDTH, REPO, MICE, _cmap,
)

NPZ = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_all.npz"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/heatmap_continuum"

KEYS = ["pulse", "change", "fa"]
TITLES = {"pulse": "fast TF pulse (sign-aligned)", "change": "Change_ON (hit trials)",
          "fa": "FA (early lick)"}
XLAB = {"pulse": "t from fast TF pulse (s)", "change": "t from Change_ON (s)",
        "fa": "t from FA lick (s)"}
N_WIDTH_BINS = 5
PULSE_SIGN_WIN = (0.0, 0.4)   # post-pulse window used to define each cell's response sign


def _join_width(D):
    """One continuous width (interp_fwhm) per npz cell, keyed (subject, session, unit)."""
    d = load_width_metrics()
    wmap = {(str(r.subject), str(r.session), int(r.unit)): float(getattr(r, WIDTH))
            for r in d.itertuples()}
    w = np.array([wmap.get((str(D["meta_subject"][i]), str(D["meta_session"][i]),
                            int(D["meta_unit"][i])), np.nan)
                  for i in range(len(D["meta_unit"]))])
    return w


def _kernel_sign(D, win=PULSE_SIGN_WIN):
    """Per-cell excitation(+1) / suppression(-1) sign taken from the GLM TF KERNEL.

    NOT from the pulse PETH being averaged — deriving the sign from the same trace you
    then average is circular and manufactures both a spurious pre-pulse rise and a ~7x
    inflated response (see the module docstring). The kernel is an independent estimator
    of the same sign, so this average is unbiased. Uses the kernel's mean over the SAME
    (0, 0.4) s lag window the PETH is summarised on, so the two are comparable.
    """
    kmap = {}
    for subj, _ in MICE:
        f = Path(REPO) / f"data/cache/tf_glm_bg046/kernel_vectors_{subj}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        lags = np.asarray(z["lags"], float)
        m = (lags >= win[0]) & (lags <= win[1])
        for k in z.files:
            if k in ("lags", "units"):
                continue
            sess, uid = k.rsplit("_u", 1)
            kmap[(sess, int(uid))] = float(np.sign(np.asarray(z[k], float)[m].mean()) or 1.0)
    s = np.array([kmap.get((str(D["meta_session"][i]), int(D["meta_unit"][i])), np.nan)
                  for i in range(len(D["meta_unit"]))])
    n_miss = int(np.sum(~np.isfinite(s)))
    assert n_miss == 0, f"_kernel_sign: {n_miss} cells have no cached GLM kernel — key mismatch?"
    s[s == 0] = 1.0
    return s


def _unresponsive_trace(key):
    """(t, mean, ci95, n) for the TF-UNRESPONSIVE reference on the `change`/`fa` panels, or
    None if unavailable / not applicable.

    Returns None for `pulse` BY DESIGN — that panel is sign-aligned by each cell's GLM kernel
    and unresponsive cells have no kernel; signing them by a noise-derived sign would fabricate
    a response (the circularity this figure was fixed to remove). Build the cache with
    `py rebuild_peth_traces_all.py --unresponsive`.
    """
    if key not in ("change", "fa"):
        return None
    f = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_unresponsive.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    M = np.asarray(z[f"mat_{key}"], float)
    t = np.asarray(z[f"t_{key}"], float)
    if M.size == 0 or t.size == 0:
        return None
    mean, ci = _mean_ci(M)
    return t, mean, ci, int(np.sum(np.any(np.isfinite(M), axis=1)))


def _mean_ci(rows):
    """Per-timepoint mean and 95% CI half-width (1.96 * SEM) across cells in a bin."""
    mean = np.nanmean(rows, axis=0)
    n = np.sum(np.isfinite(rows), axis=0).clip(1)
    sem = np.nanstd(rows, axis=0, ddof=1) / np.sqrt(n)
    return mean, 1.96 * sem


def main():
    D = {k: v for k, v in np.load(NPZ, allow_pickle=True).items()}
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    n = len(D["meta_unit"])
    w = _join_width(D)
    n_fin = int(np.isfinite(w).sum())
    if n_fin < n:
        print(f"WARNING: {n - n_fin} cells missing interp_fwhm — dropped from ordering")

    # one global row order = ascending continuous width (narrow first -> plotted at TOP)
    order = np.argsort(w, kind="stable")
    order = order[np.isfinite(w[order])]
    w_sorted = w[order]
    wmin, wmax = float(np.nanmin(w)), float(np.nanmax(w))

    # 5 equal-count width bins (idx on ORIGINAL cell order; equal n, unequal width span)
    bin_idx, edges = width_bin_assign(w, n=N_WIDTH_BINS)
    cmap = _cmap()
    bin_counts = [int(np.sum(bin_idx == b)) for b in range(N_WIDTH_BINS)]

    # ONE shared width->colour mapping — a QUANTILE (equal-count) scale — used by the
    # strip, the colorbar AND the PSTH-family line colours. Why not linear, and why not
    # even log:
    #  * width is LOGNORMAL, so a LINEAR colour scale crushes ~85% of cells (0.026-0.19 s)
    #    into one dark shade and the strip carries no legible gradient;
    #  * a LOG scale fixes the strip, but the 5 EQUAL-COUNT bin medians then bunch in the
    #    middle of the colormap (~0.26-0.68), so the PSTH families never reach yellow and
    #    adjacent families are hard to tell apart;
    #  * RANK is a strictly monotone function of width, so ordering is preserved exactly,
    #    while the colours spread over the FULL colormap: the 5 bins land at 0.1/0.3/0.5/
    #    0.7/0.9 (purple -> yellow). The colorbar ticks below still carry the TRUE width
    #    values (unevenly spaced — so the skew stays visible).
    # Because one mapping drives all three, a bin's line colour IS the strip colour of
    # its cells (previously the lines were evenly spaced in cmap-space while the strip
    # used true width, so a bin-4 cell was green in the legend but dark blue in the strip).
    qrank = np.full(len(w), np.nan)
    _fin = np.isfinite(w)
    qrank[_fin] = rankdata(w[_fin], method="average") / int(_fin.sum())   # (0, 1]
    q_sorted = qrank[order]
    bin_colors = [cmap(float(np.nanmean(qrank[bin_idx == b]))) for b in range(N_WIDTH_BINS)]

    # sign-align the fast-pulse traces (flip suppression cells); track % flipped / bin.
    # Sign comes from the GLM KERNEL, NOT from this PETH (which would be circular).
    psign = _kernel_sign(D)
    mat_signed = {"pulse": D["mat_pulse"] * psign[:, None],
                  "change": D["mat_change"], "fa": D["mat_fa"]}
    pct_flip_bin = [100.0 * np.mean(psign[bin_idx == b] < 0) if np.any(bin_idx == b) else 0.0
                    for b in range(N_WIDTH_BINS)]
    pct_flip_all = 100.0 * np.mean(psign < 0)
    # robust colour range for the (sign-aligned) pulse heatmap
    pv = np.abs(mat_signed["pulse"])
    pmax = float(np.nanpercentile(pv, 97)) or 1.0

    fig = plt.figure(figsize=(17, 11.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3.0], hspace=0.24, wspace=0.24)

    ims, hax = {}, {}
    for j, k in enumerate(KEYS):
        t = D[f"t_{k}"]
        Mfull = mat_signed[k]
        M = Mfull[order]
        if k == "pulse":
            # per-cell pulse response is near the baseline z-noise floor; light
            # display smoothing reveals the (sign-aligned) response band.
            M = gaussian_filter1d(M, 1.3, axis=1)

        # ── top panel: width-binned PSTH family (mean + 95% CI shading) ─────────
        axp = fig.add_subplot(gs[0, j])
        # TF-UNRESPONSIVE reference trace (Change_ON / FA only). These panels are shown RAW
        # (no sign-alignment), so the reference drops in cleanly. It is deliberately absent
        # from the PULSE panel: that panel is sign-aligned by each cell's GLM kernel, and
        # signing a non-responding cell by its own noise kernel would fabricate a bump — the
        # exact circularity we removed. (It is also tautological there: they are DEFINED as
        # having no TF response.) See _unresponsive_trace().
        uref = _unresponsive_trace(k)
        if uref is not None:
            ut, umean, uci, un = uref
            axp.fill_between(ut, umean - uci, umean + uci, color="0.45", alpha=0.22, lw=0,
                             zorder=1)
            axp.plot(ut, umean, color="0.25", lw=2.0, ls="--", zorder=2,
                     label=f"TF-UNRESPONSIVE (n={un:,})")
        for b in range(N_WIDTH_BINS):
            rows = Mfull[bin_idx == b]
            if rows.size == 0:
                continue
            mean, ci = _mean_ci(rows)
            axp.fill_between(t, mean - ci, mean + ci, color=bin_colors[b], alpha=0.20, lw=0)
            extra = f", {pct_flip_bin[b]:.0f}% supp" if k == "pulse" else ""
            lab = f"{edges[b]:.3f}–{edges[b+1]:.3f}s (n={bin_counts[b]}{extra})"
            axp.plot(t, mean, color=bin_colors[b], lw=2.0, label=lab)
        axp.axvline(0, color="0.6", lw=0.8)
        axp.axhline(0, color="0.85", lw=0.8)
        axp.set_xlim(t[0], t[-1])
        yl = "sign-aligned z (pop mean)" if k == "pulse" else "z-score (pop mean)"
        axp.set_ylabel(yl, fontsize=12)
        axp.set_title(f"{TITLES[k]}  — width families (±95% CI)", fontsize=12.5, fontweight="bold")
        axp.tick_params(labelsize=10)
        for sp in ("top", "right"):
            axp.spines[sp].set_visible(False)
        # each panel carries its own legend so n / width-ranges are self-explanatory
        ttl = "width bin (interp_fwhm, s) — n=104 each" if k != "pulse" else \
              "width bin — n=104 each (% suppression flipped)"
        axp.legend(frameon=False, fontsize=7.6, title=ttl, title_fontsize=8.2,
                   loc="upper left", handlelength=1.2)

        # ── heatmap: rows ordered by continuous width (narrow top -> broad bottom) ─
        axh = fig.add_subplot(gs[1, j])
        hax[k] = axh
        if k == "pulse":
            imkw = dict(norm=TwoSlopeNorm(vmin=-pmax, vcenter=0.0, vmax=pmax))
        else:
            imkw = dict(norm=TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=3.0))
        ims[k] = axh.imshow(M, aspect="auto", cmap="RdBu_r",
                            extent=[t[0], t[-1], len(M), 0],
                            interpolation="nearest", **imkw)
        axh.axvline(0, color="k", lw=1.0, ls="--")
        axh.set_xlabel(XLAB[k], fontsize=13)
        axh.tick_params(labelsize=10)
        axh.set_yticks([])

        # continuous-width colour strip (viridis), narrow(top) -> broad(bottom),
        # matching the row order; end-labelled to remove any orientation ambiguity.
        strip = axh.inset_axes([-0.060, 0.0, 0.030, 1.0])
        strip.imshow(q_sorted[:, None], aspect="auto", origin="upper",   # shared QUANTILE scale
                     cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        strip.set_xticks([]); strip.set_yticks([])
        if j == 0:
            strip.set_title("narrow", fontsize=9, pad=3)
            strip.set_xlabel("broad", fontsize=9, labelpad=3)
            strip.text(-1.7, 0.5, "cells ordered by width", rotation=90, va="center",
                       ha="center", transform=strip.transAxes, fontsize=10)

    # ── colorbars ───────────────────────────────────────────────────────────
    cbz = fig.colorbar(ims["fa"], ax=[hax["change"], hax["fa"]], fraction=0.02, pad=0.015)
    cbz.set_label("change / FA:  z-score (per-unit, baseline)", fontsize=12)
    cbz.ax.tick_params(labelsize=10)
    fig.canvas.draw()
    pp = hax["pulse"].get_position()
    cax = fig.add_axes([pp.x0, pp.y0 - 0.075, pp.width, 0.014])
    cbp = fig.colorbar(ims["pulse"], cax=cax, orientation="horizontal")
    cbp.set_label("pulse: sign-aligned z (suppression flipped +)", fontsize=10)
    cbp.ax.tick_params(labelsize=9)
    # numeric continuous-width colorbar (far left) — INVERTED so narrow is at TOP,
    # matching the left strips and the heatmap row order.
    # SAME quantile scale as the strips and the PSTH line colours; ticks carry the TRUE
    # width at each quantile (unevenly spaced — the lognormal skew stays visible).
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    cbw = fig.colorbar(sm, ax=[hax["pulse"], hax["change"], hax["fa"]],
                       location="left", fraction=0.015, pad=0.07, aspect=40)
    cbw.set_label("kernel width  interp_fwhm (s)  —  equal-count (quantile) colour scale  "
                  "[narrow top]", fontsize=10.5)
    qticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbw.set_ticks(qticks)
    cbw.set_ticklabels([f"{float(np.nanquantile(w, q)):.3f}" for q in qticks])
    cbw.ax.invert_yaxis()
    cbw.ax.tick_params(labelsize=9)

    n_interm = int(np.sum(D["meta_cls"].astype(str) == "intermediate"))
    fig.suptitle(
        f"TF-responsive cells ordered by CONTINUOUS kernel width (interp_fwhm, narrow top -> broad bottom) "
        f"— n={len(order)} cells, no transient/sustained blocks\n"
        f"all responsive cells incl. the {n_interm} intermediate-width cells; PSTH families = "
        f"mean +/-95% CI per equal-count width bin; pulse sign-aligned BY THE GLM KERNEL "
        f"(independent of this PETH — {pct_flip_all:.0f}% suppression-type at this window; "
        f"window-dependent ~30-40%)",
        fontsize=12.5, y=0.998,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"heatmap_continuum.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ── stats ───────────────────────────────────────────────────────────────
    lines = [
        "Width-ordered population heatmap + width-binned PSTH families (mean +/-95% CI)",
        "(continuum re-render of the transient/sustained class heatmap)",
        "",
        f"n cells plotted = {len(order)} (finite interp_fwhm; total npz cells = {n})",
        "row ordering = CONTINUOUS kernel width interp_fwhm, ASCENDING; NARROW at TOP, BROAD at BOTTOM",
        "  (one global order shared across pulse / change / FA; NO transient/sustained blocks)",
        f"width range = {wmin:.4f} - {wmax:.4f} s   median = {np.nanmedian(w):.4f} s",
        "",
        f"{N_WIDTH_BINS} EQUAL-COUNT width bins (equal n, UNEQUAL width span — the width",
        "distribution is right-skewed, so the broad bin spans a wider range):",
        "  edges (s) = [" + ", ".join(f"{e:.4f}" for e in edges) + "]",
    ]
    for b in range(N_WIDTH_BINS):
        lines.append(f"    bin {b}: [{edges[b]:.4f}, {edges[b+1]:.4f}) s   n = {bin_counts[b]} cells"
                     f"   suppression-flipped (pulse) = {pct_flip_bin[b]:.0f}%")
    lines += [
        "",
        f"FAST PULSE SIGN-ALIGNMENT: sign taken from the GLM TF KERNEL (mean over "
        f"{PULSE_SIGN_WIN} s lags), NOT from this PETH — signing on the PETH's own "
        f"post-window is circular and produced a spurious pre-pulse rise + a ~7x "
        f"inflated response. {pct_flip_all:.1f}% of cells are suppression-type at the "
        f"{PULSE_SIGN_WIN} s kernel window (window-dependent: ~30-40%, 25-51% across "
        f"defensible windows — quote a range; the old PETH-derived figure said ~49%, "
        f"which was a coin-flip on noise).",
        "ALL fast pulses are now used (~41k/session; the old 600-pulse cap discarded "
        "~98.5% of them and left the raw PETH noise-dominated).",
        "  (a signed pop-mean cancels exc vs supp -> the raw family looks flat; sign-",
        "   alignment makes the population pulse response coherent.)",
        "",
        "per-panel colour scaling:",
        f"  pulse  : SIGN-ALIGNED baseline-z, RdBu_r, TwoSlopeNorm(+/-{pmax:.2f}) "
        "(robust 97th-pct clip; replaces the noisy peak-norm)",
        "  change : baseline-z, RdBu_r, TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=3)",
        "  fa     : baseline-z, RdBu_r, TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=3)",
        "left strip + numeric width colorbar both run NARROW(top) -> BROAD(bottom).",
    ]
    (OUT / "heatmap_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT}/heatmap_continuum.png (+.pdf, +_stats.txt)  [n={len(order)} cells, "
          f"{pct_flip_all:.0f}% pulse-suppression flipped]")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
