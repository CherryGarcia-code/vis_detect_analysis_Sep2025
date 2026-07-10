"""Single-neuron exemplars of the transient <-> sustained extremes.

Shows the actual GLM TF kernel (each neuron's deconvolved response to an isolated
TF pulse) for a few clean cells at each end of the width axis: SUSTAINED cells whose
response stays elevated for hundreds of ms, and TRANSIENT cells whose response is a
brief blip. These are real individual neurons -- the continuum is not just a
population-average effect.

Why the GLM kernel and not the raw pulse PETH: the fast pulses come every ~50 ms, so
the raw pulse-triggered average is the impulse response convolved with the dense
pulse train (smeared) and is weak per cell (~0.05 z). The GLM deconvolves that, so
the single-cell response duration is visible in the kernel. For contrast, each panel
also overlays the cell's raw fast-pulse PETH (thin grey, sign-aligned) so you can see
it is muddier than the kernel.

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
SUS_C = "#d94801"   # sustained
TRA_C = "#3182bd"   # transient


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


def main():
    d = load_width_metrics()
    kmap, lags = _load_kernels()
    d["kkey"] = list(zip(d.session.astype(str), d.unit.astype(int)))
    d = d[d.kkey.map(lambda k: k in kmap)].copy()
    d["kpeak"] = d.kkey.map(lambda k: float(np.max(np.abs(kmap[k]))))
    clean = d[(d.kpeak > d.kpeak.median()) & (d.c1_r_log2 > d.c1_r_log2.median()) &
              (d.n_spikes > 2000) & (d.kernel_peak_t_registry.between(0.0, 0.70))]
    sus = clean.nlargest(3, WIDTH)
    tra = clean.nsmallest(3, WIDTH)

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

    for ri, (grp, col, tag) in enumerate([(sus, SUS_C, "SUSTAINED"), (tra, TRA_C, "TRANSIENT")]):
        for ci, (_, r) in enumerate(grp.iterrows()):
            ax = fig.add_subplot(gs[ri, ci])
            K = gaussian_filter1d(_signed(kmap[r.kkey]), 0.7)   # light display smoothing
            pk = K.max()
            ax.plot(lags, K, color=col, lw=2.4, zorder=3)
            # half-max marker + CONTIGUOUS FWHM shading (walk out from the peak, not
            # first-to-last crossing — else a noisy far-off wiggle shades the whole axis)
            half = pk / 2.0
            ax.axhline(half, color="0.6", lw=0.8, ls=":")
            ip = int(np.argmax(K)); lo, hi = ip, ip
            while lo > 0 and K[lo - 1] >= half:
                lo -= 1
            while hi < len(K) - 1 and K[hi + 1] >= half:
                hi += 1
            ax.axvspan(lags[lo], lags[hi], color=col, alpha=0.14, zorder=0)
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
                        label="raw pulse PETH (scaled)")
            ax.set_xlim(0, 1.45)
            ax.set_title(f"{tag}  fwhm={r[WIDTH]:.3f}s\n{r.subject} {r.session.split('_',2)[-1]} u{int(r.unit)}",
                         fontsize=10, color=col, fontweight="bold")
            if ci == 0:
                ax.set_ylabel("GLM TF kernel\n(sign-aligned)", fontsize=11)
            if ri == 1:
                ax.set_xlabel("lag from TF pulse (s)", fontsize=11)
            if ri == 0 and ci == 2:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            lines.append(f"  {tag:9s} {r.subject} {r.session} u{int(r.unit)}: "
                         f"fwhm={r[WIDTH]:.3f}s peak_t={r.kernel_peak_t_registry:.2f}s "
                         f"c1_r={r.c1_r_log2:.2f} kpeak={r.kpeak:.4f}")

    fig.suptitle("Real single neurons at the transient↔sustained extremes — GLM TF kernel "
                 "(each cell's deconvolved response to a TF pulse)\n"
                 "TOP: SUSTAINED (response stays elevated ~0.4–0.7 s)   BOTTOM: TRANSIENT (a brief blip). "
                 "Grey = the same cell's raw fast-pulse PETH (muddier per cell).",
                 fontsize=12, y=1.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"exemplar_kernels_continuum.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)
    lines += ["", "GLM kernel = deconvolved impulse response (the clean single-cell TF-pulse response);",
              "grey raw pulse PETH is weak/smeared per cell (dense 50ms pulses) — the reason we use the kernel.",
              "Exemplars = 3 broadest + 3 narrowest fwhm among clean cells (kpeak/c1/spikes > median)."]
    (OUT / "exemplar_kernels_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/exemplar_kernels_continuum.png (+.pdf, +_stats.txt)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
