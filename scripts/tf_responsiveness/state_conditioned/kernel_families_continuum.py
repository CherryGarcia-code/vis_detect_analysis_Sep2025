"""What the transient -> sustained continuum IS: the deconvolved GLM TF kernel per
width bin.

The raw fast-TF-pulse PSTH does NOT separate by kernel width, because the grating's
TF fluctuates every ~50 ms around baseline so a fast pulse is never isolated: the
pulse-triggered average mixes the cell's response with the correlated neighbouring
fluctuations (stimulus autocorrelation), which smears the temporal width -- and
because `interp_fwhm` is a GLM-DECONVOLVED quantity the raw average can't recover. This figure plots the thing the width axis is actually defined on:
the per-cell GLM TF FIR kernel (the deconvolved impulse response), averaged per width
bin. It shows the narrow -> broad (transient -> sustained) progression directly, and
that the progression is SMOOTH (a continuum), not two discrete shapes.

CAVEAT (honest): the width bins are DEFINED on each kernel's FWHM, so the fact that
broad-bin kernels are broader is illustrative-by-construction, not an independent
result. What is informative is (a) what "narrow" vs "broad" LOOKS like, and (b) that
the family morphs smoothly across bins (supporting the spectrum reading).

Kernels are sign-aligned (dominant deflection made positive; ~half of TF-responsive
cells are suppression-type and the FWHM is sign-agnostic) and peak-normalised, so the
families show SHAPE/DURATION independent of amplitude.

Reads the cached GLM kernel vectors (data/cache/tf_glm_bg046/kernel_vectors_{subj}.npz,
520 cells) + kernel_width_continuous.csv. Cache-only.

Usage:  py scripts/tf_responsiveness/state_conditioned/kernel_families_continuum.py
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

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import load_width_metrics, width_bin_assign, WIDTH, REPO, MICE, _cmap  # noqa: E402

OUT = Path(REPO) / "FIGURES/tf_glm_bg046/kernel_families_continuum"
N_WIDTH_BINS = 5


def _load_kernels():
    """Per-cell GLM TF kernel keyed (session, unit); returns dict + the lag grid."""
    kmap, lags = {}, None
    for subj, _ in MICE:
        f = Path(REPO) / f"data/cache/tf_glm_bg046/kernel_vectors_{subj}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        lags = z["lags"] if lags is None else lags
        for key in z.files:
            if key in ("lags", "units"):
                continue
            sess, uid = key.rsplit("_u", 1)      # "BG_046_01072025_u278" -> sess, 278
            kmap[(sess, int(uid))] = np.asarray(z[key], float)
    return kmap, np.asarray(lags, float)


HALF = 14   # +/- lags around the peak for the latency-aligned shape (0.05 s grid -> +/-0.7 s)


def _prep(K, half=HALF):
    """Sign-align to the dominant deflection, LATENCY-ALIGN to the peak, peak-normalise.

    interp_fwhm is a latency-invariant WIDTH, so to show narrow-vs-broad SHAPE we
    centre each kernel on its own peak (else averaging kernels whose peaks sit at
    different lags smears the sharpness). Output is length 2*half+1, index `half` =
    peak; out-of-range lags are NaN."""
    ip = int(np.argmax(np.abs(K)))
    s = np.sign(K[ip]) or 1.0
    Kn = K * s
    pk = np.max(np.abs(Kn))
    Kn = Kn / pk if pk > 1e-9 else Kn
    out = np.full(2 * half + 1, np.nan)
    lo, hi = max(0, ip - half), min(len(Kn), ip + half + 1)
    out[(lo - ip) + half:(hi - ip) + half] = Kn[lo:hi]
    return out


def _mean_ci(rows):
    mean = np.nanmean(rows, axis=0)
    nrow = np.sum(np.isfinite(rows), axis=0).clip(1)
    sem = np.nanstd(rows, axis=0, ddof=1) / np.sqrt(nrow)
    return mean, 1.96 * sem


def main():
    d = load_width_metrics()
    kmap, lags = _load_kernels()
    d["kkey"] = list(zip(d.session.astype(str), d.unit.astype(int)))
    d["hasK"] = d.kkey.map(lambda k: k in kmap)
    d = d[d.hasK & np.isfinite(d[WIDTH])].reset_index(drop=True)
    assert len(d) > 0, (
        "kernel_families: 0 responsive cells have a cached GLM kernel — "
        "kernel_vectors_{subj}.npz missing/empty or a (session, unit) key mismatch.")
    K = np.vstack([_prep(kmap[k]) for k in d.kkey])          # latency-aligned, peak-norm, sign-aligned
    w = d[WIDTH].to_numpy(float)
    bin_s = float(lags[1] - lags[0])
    rel_lags = np.arange(-HALF, HALF + 1) * bin_s            # lag relative to each kernel's peak

    order = np.argsort(w, kind="stable")
    w_sorted = w[order]
    wmin, wmax = float(w.min()), float(w.max())
    bin_idx, edges = width_bin_assign(w, n=N_WIDTH_BINS)
    cmap = _cmap()
    bin_colors = cmap(np.linspace(0.08, 0.95, N_WIDTH_BINS))

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    fig = plt.figure(figsize=(14, 6.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.26)

    # ── Panel A: mean GLM kernel per width bin (families + 95% CI) ─────────────
    axA = fig.add_subplot(gs[0, 0])
    lines = ["GLM TF kernel families by continuous width bin (sign-aligned, peak-normalised)",
             f"n = {len(d)} responsive cells with a cached kernel; {N_WIDTH_BINS} equal-count width bins", ""]
    for b in range(N_WIDTH_BINS):
        rows = K[bin_idx == b]
        if rows.size == 0:
            continue
        mean, ci = _mean_ci(rows)
        med_w = float(np.median(w[bin_idx == b]))
        axA.fill_between(rel_lags, mean - ci, mean + ci, color=bin_colors[b], alpha=0.18, lw=0)
        axA.plot(rel_lags, mean, color=bin_colors[b], lw=2.4,
                 label=f"{edges[b]:.3f}–{edges[b+1]:.3f}s  (n={int(np.sum(bin_idx==b))}, med {med_w:.3f})")
        lines.append(f"  bin {b} width [{edges[b]:.3f},{edges[b+1]:.3f}) s (median {med_w:.3f}): "
                     f"kernel peak-normalised mean plotted")
    axA.axhline(0, color="0.8", lw=0.8)
    axA.axvline(0, color="0.6", lw=0.8, ls=":")
    axA.set_xlabel("lag relative to kernel peak (s)", fontsize=12)
    axA.set_ylabel("GLM TF kernel (sign-aligned, peak-norm)", fontsize=12)
    axA.set_title("Deconvolved TF kernel broadens narrow→broad\n(latency-aligned; what the transient→sustained axis IS)",
                  fontsize=12.5, fontweight="bold")
    axA.legend(frameon=False, fontsize=8.2, title="kernel width bin (interp_fwhm)", title_fontsize=8.6,
               loc="upper right")
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # ── Panel B: per-cell kernel heatmap, ordered by continuous width ──────────
    axB = fig.add_subplot(gs[0, 1])
    im = axB.imshow(K[order], aspect="auto", cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=1.0),
                    extent=[rel_lags[0], rel_lags[-1], len(K), 0], interpolation="nearest")
    axB.axvline(0, color="k", lw=0.8, ls=":")
    axB.set_xlabel("lag relative to peak (s)", fontsize=12)
    axB.set_yticks([])
    axB.set_title("per-cell kernels, narrow(top)→broad(bottom)", fontsize=12.5, fontweight="bold")
    strip = axB.inset_axes([-0.05, 0.0, 0.028, 1.0])
    strip.imshow(w_sorted[:, None], aspect="auto", origin="upper", cmap=cmap,
                 vmin=wmin, vmax=wmax, interpolation="nearest")
    strip.set_xticks([]); strip.set_yticks([])
    strip.set_title("narrow", fontsize=9, pad=3)
    strip.set_xlabel("broad", fontsize=9, labelpad=3)
    cb = fig.colorbar(im, ax=axB, fraction=0.03, pad=0.02)
    cb.set_label("kernel (sign-aligned, peak-norm)", fontsize=10)
    smw = ScalarMappable(norm=Normalize(vmin=wmin, vmax=wmax), cmap=cmap)
    cbw = fig.colorbar(smw, ax=axB, location="left", fraction=0.03, pad=0.10, aspect=32)
    cbw.set_label("kernel width interp_fwhm (s) [narrow top]", fontsize=10)
    cbw.ax.invert_yaxis()

    fig.suptitle("Why a transient→sustained continuum: the GLM TF kernel (deconvolved impulse response) "
                 "broadens SMOOTHLY with width\n(illustrative-by-construction — bins are defined on this "
                 "kernel's FWHM — but shows what 'narrow/broad' means and that the morph is graded, not two shapes)",
                 fontsize=11.5, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"kernel_families_continuum.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)

    lines += ["", "CAVEAT: width bins are defined on interp_fwhm (a FWHM of these kernels), so the",
              "broadening is illustrative-by-construction; the informative content is the SHAPE",
              "of narrow vs broad and that the family morphs SMOOTHLY (a continuum, not 2 classes).",
              "", "Raw fast-pulse PSTHs do NOT show this separation (dense 50ms-pulse convolution +",
              "deconvolution) — see heatmap_continuum.py; the width difference lives in this kernel",
              "and in the change/lick COUPLING."]
    (OUT / "kernel_families_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/kernel_families_continuum.png (+.pdf, +_stats.txt)  [n={len(d)} cells]")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
