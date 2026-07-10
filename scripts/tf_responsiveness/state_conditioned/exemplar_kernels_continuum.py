"""Single-neuron exemplars of the transient <-> sustained extremes.

Shows the actual GLM TF kernel (each neuron's deconvolved response to a unit change
in the continuous temporal-frequency signal -- the FIR impulse response, NOT a
discrete fast/slow pulse) for a few clean cells at each end of the width axis:
SUSTAINED cells whose response stays elevated for hundreds of ms, and TRANSIENT cells
whose response is a brief blip. These are real individual neurons -- the continuum is
not just a population-average effect.

Why the GLM kernel and not the raw pulse PETH: the grating's TF fluctuates every
~50 ms around baseline (log2 ~ N(0, 0.25 octave)), so a fast pulse is never isolated
-- a raw fast-pulse-triggered average mixes the response to that pulse with the
correlated neighbouring fluctuations (stimulus autocorrelation), which smears it, and
is weak per cell (~0.05 z). The GLM regresses the spike train against the WHOLE
continuous TF timeseries and deconvolves, so the single-cell response duration is
visible in the kernel. For contrast, each panel overlays the cell's raw fast-pulse
PETH (thin grey, sign-aligned) so you can see it is muddier than the kernel.

Each kernel carries a 95% CI band (shaded) from a per-cell TRIAL BOOTSTRAP of the
GLM refit (resample the cell's trials, refit the ridge-Poisson at the point-estimate
lambda, read off the TF FIR coefficients; 200 resamples). A band that stays clear of
zero at the peak is the direct evidence that the single-cell kernel is a reliable
estimate, not the ridge prior. The band is precomputed by
`compute_exemplar_ci.py` into exemplar_kernel_ci.npz; if that cache is absent the
figure still renders (falls back to a shaded FWHM span, no CI).

Exemplars are picked programmatically (reproducible): among "clean" cells (kernel
peak, TF-selectivity, and spike count all above median) the 3 broadest and 3
narrowest kernel FWHM. Reads the cached GLM kernels + peth_traces_all.npz. Cache-only.

Usage:  py scripts/tf_responsiveness/state_conditioned/exemplar_kernels_continuum.py
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
from scipy.ndimage import gaussian_filter1d

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import load_width_metrics, WIDTH, REPO, MICE  # noqa: E402

OUT = Path(REPO) / "FIGURES/tf_glm_bg046/exemplar_kernels_continuum"
NPZ = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_all.npz"
CI_NPZ = Path(REPO) / "data/cache/tf_glm_bg046/exemplar_kernel_ci.npz"
SUS_C = "#d94801"   # sustained
TRA_C = "#3182bd"   # transient
DISPLAY_SMOOTH = 0.7   # gaussian sigma (bins) for display; the CI band uses the SAME


def _load_kernels():
    kmap, lags = {}, None
    for subj, _ in MICE:
        f = Path(REPO) / f"data/cache/tf_glm_bg046/kernel_vectors_{subj}.npz"
        z = np.load(f, allow_pickle=True)
        lags = z["lags"] if lags is None else lags
        for k in z.files:
            if k in ("lags", "units"):
                continue
            sess, uid = k.rsplit("_u", 1)
            kmap[(sess, int(uid))] = np.asarray(z[k], float)
    return kmap, np.asarray(lags, float)


def _signed(K):
    ip = int(np.argmax(np.abs(K)))
    return K * (np.sign(K[ip]) or 1.0)


def select_exemplars(d, kmap):
    """The 6 exemplar cells (3 broadest + 3 narrowest kernel FWHM among clean cells).

    Factored out so compute_exemplar_ci.py bootstraps EXACTLY the cells this figure
    plots (the selection is deterministic given the cached width CSV + kernels).
    Returns (sus, tra) DataFrames carrying kkey + kpeak columns."""
    d = d.copy()
    d["kkey"] = list(zip(d.session.astype(str), d.unit.astype(int)))
    d = d[d.kkey.map(lambda k: k in kmap)].copy()
    d["kpeak"] = d.kkey.map(lambda k: float(np.max(np.abs(kmap[k]))))
    clean = d[(d.kpeak > d.kpeak.median()) & (d.c1_r_log2 > d.c1_r_log2.median()) &
              (d.n_spikes > 2000) & (d.kernel_peak_t_registry.between(0.0, 0.70))]
    sus = clean[clean[WIDTH] > 0.35].nlargest(3, "kpeak")   # broad + strongest kernels
    tra = clean[clean[WIDTH] < 0.09].nlargest(3, "kpeak")   # narrow + strongest kernels
    return sus, tra


def _load_ci():
    """Precomputed 95% CI bands keyed 'SUBJ_DATE_uUID_{lo,hi}'; None if absent."""
    if not CI_NPZ.exists():
        return None
    return np.load(CI_NPZ, allow_pickle=True)


def main():
    d = load_width_metrics()
    kmap, lags = _load_kernels()
    sus, tra = select_exemplars(d, kmap)
    cib = _load_ci()

    # raw pulse PETHs for the same cells (sign-aligned) for the grey overlay
    Z = {k: v for k, v in np.load(NPZ, allow_pickle=True).items()}
    prow = {(str(Z["meta_subject"][i]), str(Z["meta_session"][i]), int(Z["meta_unit"][i])): i
            for i in range(len(Z["meta_unit"]))}
    tp = Z["t_pulse"]

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    fig = plt.figure(figsize=(15, 7.2))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.26)
    lines = ["Single-neuron exemplars: transient vs sustained GLM TF kernels", ""]
    any_band = False

    for ri, (grp, col, tag) in enumerate([(sus, SUS_C, "SUSTAINED"), (tra, TRA_C, "TRANSIENT")]):
        for ci, (_, r) in enumerate(grp.iterrows()):
            ax = fig.add_subplot(gs[ri, ci])
            K = gaussian_filter1d(_signed(kmap[r.kkey]), DISPLAY_SMOOTH)   # display smoothing
            pk = K.max()
            half = pk / 2.0
            ip = int(np.argmax(K)); pk_lag = float(lags[ip])
            w_fw = float(r[WIDTH])
            x0, x1 = pk_lag - w_fw / 2, pk_lag + w_fw / 2

            # 95% CI band (trial bootstrap) if precomputed; else fall back to an FWHM span
            bkey = f"{r.session}_u{int(r.unit)}"
            have_band = cib is not None and f"{bkey}_lo" in cib.files
            if have_band:
                lo, hi = np.asarray(cib[f"{bkey}_lo"], float), np.asarray(cib[f"{bkey}_hi"], float)
                ax.fill_between(lags, lo, hi, color=col, alpha=0.22, lw=0, zorder=2,
                                label="95% CI (trial bootstrap)")
                any_band = True
            else:
                ax.axvspan(x0, x1, color=col, alpha=0.18, zorder=0)

            ax.plot(lags, K, color=col, lw=2.4, zorder=3)
            # half-max FWHM bracket + label (the width the continuum is defined on)
            ax.annotate("", xy=(x1, half), xytext=(x0, half),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1.6))
            ax.text(pk_lag, half * 1.2, f"fwhm={w_fw:.3f}s", color=col, fontsize=8.5,
                    ha="center", va="bottom", fontweight="bold")
            ax.axhline(half, color="0.7", lw=0.6, ls=":")
            ax.axhline(0, color="0.85", lw=0.8)
            # raw pulse PETH overlay (grey, sign-aligned, smoothed, scaled to the kernel peak)
            key3 = (str(r.subject), str(r.session), int(r.unit))
            if key3 in prow:
                raw = Z["mat_pulse"][prow[key3]]
                post = (tp >= 0) & (tp <= 0.4)
                raw = raw * (np.sign(np.nanmean(raw[post])) or 1.0)
                raw = gaussian_filter1d(raw, 1.3)
                rp = np.nanmax(np.abs(raw)) or 1.0
                ax.plot(tp, raw / rp * pk, color="0.55", lw=1.0, alpha=0.8, zorder=1,
                        label="raw fast-pulse PETH (scaled)")
            ax.set_xlim(0, 1.45)
            ax.set_title(f"{tag}  fwhm={r[WIDTH]:.3f}s\n{r.subject} {r.session.split('_',2)[-1]} u{int(r.unit)}",
                         fontsize=10, color=col, fontweight="bold")
            if ci == 0:
                ax.set_ylabel("GLM TF kernel\n(sign-aligned)", fontsize=11)
            if ri == 1:
                ax.set_xlabel("lag from TF change (s)", fontsize=11)
            if ri == 0 and ci == 2:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            band_txt = ""
            if have_band:
                band_txt = (f" peak95CI=[{lo[ip]:+.4f},{hi[ip]:+.4f}] "
                            f"excl0={not (lo[ip] <= 0 <= hi[ip])}")
            lines.append(f"  {tag:9s} {r.subject} {r.session} u{int(r.unit)}: "
                         f"fwhm={r[WIDTH]:.3f}s peak_t={r.kernel_peak_t_registry:.2f}s "
                         f"c1_r={r.c1_r_log2:.2f} kpeak={r.kpeak:.4f}{band_txt}")

    band_note = "Shaded band = 95% CI (trial bootstrap, 200×). " if any_band else ""
    fig.suptitle("Real single neurons at the transient↔sustained extremes — GLM TF kernel "
                 "(each cell's deconvolved response to a unit change in temporal frequency)\n"
                 "TOP: SUSTAINED (response stays elevated ~0.4–0.7 s)   BOTTOM: TRANSIENT (a brief blip). "
                 f"{band_note}Grey = the same cell's raw fast-pulse PETH (muddier per cell).",
                 fontsize=12, y=1.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"exemplar_kernels_continuum.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)
    lines += ["", "GLM kernel = deconvolved response to a unit change in the continuous TF signal;",
              "grey raw fast-pulse PETH is weak/smeared per cell (stimulus autocorrelation + weak signal)",
              "— the reason we read duration off the kernel. Exemplars = 3 broadest + 3 narrowest fwhm",
              "among clean cells (kpeak/c1/spikes > median).",
              ("Shaded band = 95% CI from a per-cell trial bootstrap (compute_exemplar_ci.py)."
               if any_band else "No CI cache found — run compute_exemplar_ci.py for the 95% bands.")]
    (OUT / "exemplar_kernels_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/exemplar_kernels_continuum.png (+.pdf, +_stats.txt)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
