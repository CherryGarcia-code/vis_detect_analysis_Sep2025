"""Fig5h: preparatory-activity onset vs TF-response width.

Within-striatum port of Khilkevich & Lohse (Nature 2024) Fig 5h. Their per-region
scatter (onset vs median TF-pulse half-peak width, r=-0.55) becomes a within-region
per-width-decile scatter here.

  * PRIMARY (faithful): per width-decile dot — x = decile population activation
    onset (bootstrap-over-neurons, 100ms/80ms/mean>0.1/lo>0 rule), y = decile
    median interp_fwhm (log axis). Pearson + Spearman + bootstrap CI (10,000x
    resample over the underlying cells).
  * SUPPLEMENT (higher-n, extends the paper): per-cell scatter — x = each cell's own
    pre-lick onset, y = interp_fwhm, coloured by class; decile-of-onset trend
    overlaid; non-TF cells drawn as an x-axis onset rug (no width).

PER-REGION ALWAYS: pooled + DMS + VMS. Cache-only (prep_<lick>.npz). No reload.

The width axis (interp_fwhm) is estimated from the TF-pulse GLM, independent of the
lick alignment, so onset-vs-width is NOT circular. A flat/positive result is
reported as-is (striatum need not mirror the brain-wide sign).

Usage:  py fig5h_onset_vs_width.py [--lick hit|fa]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import pearsonr, spearmanr, linregress

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import prep_common as C  # noqa: E402
from visdetect.analysis.preparatory import (  # noqa: E402
    active_mask, bootstrap_fraction_ci, population_onset, cell_onset, width_deciles)

FIGROOT = C.REPO / "FIGURES/preparatory_fig5"
REGIONS = [("pooled", None), ("DMS", "DMS"), ("VMS", "VMS")]
N_DEC = 10
N_BOOT_ONSET = 5000     # per-decile point-onset bootstrap (matches panel f)
N_BOOT_CI = 10000       # resample-over-cells CI on the correlation/slope
CLS_COLORS = {"transient": "#3182bd", "intermediate": "#756bb1", "sustained": "#e6550d"}
NONTF_COLOR = "#969696"


# ── vectorized "first sustained crossing" (matches preparatory._first_sustained) ──
def _first_sustained_vec(cond, win=4, need=3):
    cond = np.asarray(cond, bool)
    n = len(cond)
    cs = np.concatenate(([0], np.cumsum(cond.astype(int))))
    idx = np.arange(n)
    end = np.minimum(idx + win, n)
    ok = cond & ((cs[end] - cs[idx]) >= need)
    w = np.where(ok)[0]
    return int(w[0]) if w.size else -1


def _analytic_decile_onsets(A_resp, w_resp, t, base_mask):
    """Fast per-decile (onset, median_width) using a Wald analytic lower-CI onset,
    for the 10,000x cell-resample. Returns (onsets, median_widths) over all deciles
    (NaN onset where the ramp never sustains)."""
    idx, _ = width_deciles(w_resp, n=N_DEC)
    onsets = np.full(N_DEC, np.nan)
    medws = np.full(N_DEC, np.nan)
    for d in range(N_DEC):
        sel = idx == d
        n = int(sel.sum())
        if n == 0:
            continue
        p = A_resp[sel].mean(0)                       # fraction per bin in [0,1]
        base = float(np.nanmean(p[base_mask]))
        frac = p - base
        se = np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / max(n, 1))
        lo = (p - 1.96 * se) - base
        cond = (lo > 0) & (frac > 0.1)
        i = _first_sustained_vec(cond)
        if i >= 0:
            onsets[d] = float(t[i])
        medws[d] = float(np.median(w_resp[sel]))
    return onsets, medws


def _boot_decile_ci(A_resp, w_resp, t, base_mask, n=N_BOOT_CI, seed=42):
    """Bootstrap over cells -> percentile CI of Pearson r and OLS slope (y=width on x=onset)."""
    rng = np.random.default_rng(seed)
    nC = A_resp.shape[0]
    rs, sl = [], []
    for _ in range(n):
        bi = rng.integers(0, nC, nC)
        on, mw = _analytic_decile_onsets(A_resp[bi], w_resp[bi], t, base_mask)
        m = np.isfinite(on) & np.isfinite(mw)
        if m.sum() >= 3:
            rs.append(pearsonr(on[m], mw[m])[0])
            sl.append(np.polyfit(on[m], mw[m], 1)[0])
    rs = np.asarray(rs, float)
    sl = np.asarray(sl, float)
    return rs, sl


def _boot_cell_slope_ci(x, y, n=N_BOOT_CI, seed=42):
    """Bootstrap over cells -> percentile CI of OLS slope (y=width on x=cell onset)."""
    rng = np.random.default_rng(seed)
    nC = len(x)
    sl = []
    for _ in range(n):
        bi = rng.integers(0, nC, nC)
        if np.ptp(x[bi]) > 0:
            sl.append(np.polyfit(x[bi], y[bi], 1)[0])
    return np.asarray(sl, float)


def _corr_row(region, unit_of_obs, x, y, ci_lo, ci_hi, slope):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 3:
        pr, pp = pearsonr(x[m], y[m])
        sr, sp = spearmanr(x[m], y[m])
    else:
        pr = pp = sr = sp = np.nan
    return {"region": region, "unit_of_obs": unit_of_obs,
            "pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp,
            "n": int(m.sum()), "slope": slope, "ci_lo": ci_lo, "ci_hi": ci_hi}


def main(lick: str = "hit") -> None:
    path = C.REPO / f"data/cache/preparatory_fig5/prep_{lick}.npz"
    if not path.exists():
        raise SystemExit(f"cache missing: {path} — run build_prep_cache.py --lick {lick}")

    D = np.load(path, allow_pickle=True)
    t = np.asarray(D["t"], float)
    z = np.asarray(D["z"], float)
    resp = np.asarray(D["resp"], bool)
    region = D["region"].astype(str)
    interp = np.asarray(D["interp_fwhm"], float)
    cls = D["cls"].astype(str)

    A = active_mask(z)
    base_mask = (t >= C.BASE_FRAC_WIN[0]) & (t <= C.BASE_FRAC_WIN[1])
    lick_lbl = lick.upper()

    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass

    for rname, rval in REGIONS:
        rmask = np.ones(len(resp), bool) if rval is None else (region == rval)

        # ── PRIMARY: per-width-decile ────────────────────────────────────────
        f_sel = rmask & resp & np.isfinite(interp)
        A_resp = A[f_sel]
        w_resp = interp[f_sel]
        cls_resp = cls[f_sel]
        # point onsets: full bootstrap_fraction_ci (matches panel f)
        idx, _ = width_deciles(w_resp, n=N_DEC)
        dec_onset = np.full(N_DEC, np.nan)
        dec_medw = np.full(N_DEC, np.nan)
        for d in range(N_DEC):
            sel = idx == d
            if sel.sum() == 0:
                continue
            mean, lo, _hi = bootstrap_fraction_ci(A_resp[sel], baseline_bins=base_mask, n=N_BOOT_ONSET)
            dec_onset[d] = population_onset(t, mean, lo)
            dec_medw[d] = float(np.median(w_resp[sel]))
        md = np.isfinite(dec_onset) & np.isfinite(dec_medw)
        if md.sum() >= 3:
            dslope, dintercept = linregress(dec_onset[md], dec_medw[md])[:2]
        else:
            dslope, dintercept = np.nan, np.nan
        # CI over cells (analytic onset for tractable 10,000x)
        boot_r, boot_sl = _boot_decile_ci(A_resp, w_resp, t, base_mask)
        d_ci = (float(np.percentile(boot_sl, 2.5)), float(np.percentile(boot_sl, 97.5))) \
            if boot_sl.size else (np.nan, np.nan)
        d_r_ci = (float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))) \
            if boot_r.size else (np.nan, np.nan)

        # ── SUPPLEMENT: per-cell ─────────────────────────────────────────────
        cell_on = np.array([cell_onset(t, z[f_sel][i]) for i in range(A_resp.shape[0])])
        cm = np.isfinite(cell_on) & np.isfinite(w_resp)
        if cm.sum() >= 3:
            cslope = linregress(cell_on[cm], w_resp[cm])[0]
        else:
            cslope = np.nan
        boot_csl = _boot_cell_slope_ci(cell_on[cm], w_resp[cm]) if cm.sum() >= 3 else np.array([])
        c_ci = (float(np.percentile(boot_csl, 2.5)), float(np.percentile(boot_csl, 97.5))) \
            if boot_csl.size else (np.nan, np.nan)
        # non-TF onset rug
        g_sel = rmask & (~resp)
        nontf_on = np.array([cell_onset(t, z[g_sel][i]) for i in range(int(g_sel.sum()))])

        # y-limits from the (always-positive, non-empty) responsive widths so the
        # log axis stays valid even when a region has no finite onsets to plot.
        wfin = w_resp[np.isfinite(w_resp)]
        wlo = float(np.nanmin(wfin)) * 0.9 if wfin.size else 0.02
        whi = float(np.nanmax(wfin)) * 1.1 if wfin.size else 0.7

        # ── figure ───────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(15.0, 6.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.15], wspace=0.24)

        axP = fig.add_subplot(gs[0, 0])
        axP.scatter(dec_onset[md], dec_medw[md], s=90, c=dec_medw[md], cmap=C.WIDTH_CMAP,
                    edgecolors="k", linewidths=0.8, zorder=3)
        if np.isfinite(dslope):
            xs = np.linspace(np.nanmin(dec_onset[md]), np.nanmax(dec_onset[md]), 50)
            axP.plot(xs, dintercept + dslope * xs, "k--", lw=1.5, zorder=2)
        pr = pearsonr(dec_onset[md], dec_medw[md]) if md.sum() >= 3 else (np.nan, np.nan)
        srho = spearmanr(dec_onset[md], dec_medw[md]) if md.sum() >= 3 else (np.nan, np.nan)
        axP.set_yscale("log")
        axP.set_ylim(wlo, whi)
        axP.set_xlabel(f"decile activation onset (s from {lick_lbl} lick)")
        axP.set_ylabel("decile median width  interp_fwhm (s)")
        axP.set_title("h  primary — per-width-decile (faithful)", fontsize=13, loc="left")
        axP.text(0.03, 0.05,
                 f"Pearson r={pr[0]:+.2f} (p={pr[1]:.2g})\n"
                 f"Spearman rho={srho[0]:+.2f} (p={srho[1]:.2g})\n"
                 f"slope={dslope:+.3g} [{d_ci[0]:+.3g}, {d_ci[1]:+.3g}]\n"
                 f"n={int(md.sum())} deciles",
                 transform=axP.transAxes, va="bottom", ha="left", fontsize=10)
        for sp in ("top", "right"):
            axP.spines[sp].set_visible(False)

        axS = fig.add_subplot(gs[0, 1])
        for cl in ("transient", "intermediate", "sustained"):
            s = cls_resp == cl
            if s.any():
                axS.scatter(cell_on[s], w_resp[s], s=18, alpha=0.5,
                            color=CLS_COLORS[cl], edgecolors="none", label=cl, zorder=2)
        # decile-of-onset trend (median width per onset decile)
        if cm.sum() >= 10:
            xo, yo = cell_on[cm], w_resp[cm]
            oedges = np.quantile(xo, np.linspace(0, 1, 11))
            oedges[-1] += 1e-9
            oc, ow = [], []
            for b in range(10):
                bsel = (xo >= oedges[b]) & (xo < oedges[b + 1])
                if bsel.any():
                    oc.append(float(np.median(xo[bsel])))
                    ow.append(float(np.median(yo[bsel])))
            axS.plot(oc, ow, "-o", color="k", lw=1.8, ms=5, zorder=4, label="onset-decile trend")
        # non-TF onset rug along the x-axis
        rug = nontf_on[np.isfinite(nontf_on)]
        if rug.size:
            axS.plot(rug, np.full(rug.size, wlo), "|", color=NONTF_COLOR, ms=8,
                     alpha=0.35, zorder=1, label=f"non-TF onset rug (n={rug.size})")
        cpr = pearsonr(cell_on[cm], w_resp[cm]) if cm.sum() >= 3 else (np.nan, np.nan)
        csr = spearmanr(cell_on[cm], w_resp[cm]) if cm.sum() >= 3 else (np.nan, np.nan)
        axS.set_yscale("log")
        axS.set_ylim(wlo, whi)
        axS.set_xlabel(f"per-cell onset (s from {lick_lbl} lick)")
        axS.set_ylabel("width  interp_fwhm (s)")
        axS.set_title("h  supplement — per-cell (extends paper)", fontsize=13, loc="left")
        axS.text(0.03, 0.05,
                 f"Pearson r={cpr[0]:+.2f} (p={cpr[1]:.2g})\n"
                 f"Spearman rho={csr[0]:+.2f} (p={csr[1]:.2g})\n"
                 f"slope={cslope:+.3g} [{c_ci[0]:+.3g}, {c_ci[1]:+.3g}]\n"
                 f"n={int(cm.sum())} cells",
                 transform=axS.transAxes, va="bottom", ha="left", fontsize=10)
        axS.legend(frameon=False, fontsize=9, loc="upper right")
        for sp in ("top", "right"):
            axS.spines[sp].set_visible(False)

        fig.suptitle(
            f"Fig5h  preparatory onset vs TF-response width — {rname} ({lick_lbl}); "
            f"decile r_CI=[{d_r_ci[0]:+.2f}, {d_r_ci[1]:+.2f}]",
            fontsize=13, y=1.01)

        outdir = FIGROOT / rname
        outdir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"fig5h_{lick}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

        # ── stats ────────────────────────────────────────────────────────────
        rows = [
            _corr_row(rname, "decile", dec_onset, dec_medw, d_ci[0], d_ci[1], dslope),
            _corr_row(rname, "cell", cell_on, w_resp, c_ci[0], c_ci[1], cslope),
        ]
        pd.DataFrame(rows).to_csv(outdir / f"fig5h_{lick}_stats.csv", index=False)
        print(f"[{rname}] decile: n={int(md.sum())} r={pr[0]:+.2f} slope={dslope:+.3g} "
              f"slopeCI=[{d_ci[0]:+.3g},{d_ci[1]:+.3g}] rCI=[{d_r_ci[0]:+.2f},{d_r_ci[1]:+.2f}] | "
              f"cell: n={int(cm.sum())} r={cpr[0]:+.2f} slope={cslope:+.3g}", flush=True)

    print(f"wrote {FIGROOT}/{{pooled,DMS,VMS}}/fig5h_{lick}.{{png,pdf}} (+_stats.csv)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lick", choices=["hit", "fa"], default="hit")
    a = ap.parse_args()
    main(lick=a.lick)
