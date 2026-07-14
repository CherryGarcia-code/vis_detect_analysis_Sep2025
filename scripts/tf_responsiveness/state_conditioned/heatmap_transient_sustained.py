"""Presentation figure: population heatmaps + grand-average PSTHs of TF-responsive
cells split by kernel-width class (transient vs sustained), across three
alignments — fast TF pulse | Hit @ Change_ON | FA @ early-lick.

Per-unit z-score to a local pre-event baseline (normalize-then-average for the
PSTHs). Cells ordered identically across all three heatmaps (class block, then
fast-pulse peak latency), so one row = one cell everywhere. Builds a trace cache
(one session load each) so re-plotting is instant.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import gc
import argparse
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
from representative_cells import REPO, _spikes, load_session, get_event_times_by_trial  # noqa: E402
from transient_vs_sustained import load_cells, TCOL, SCOL, NARROW, BROAD      # noqa: E402
from heatmap_continuum import _kernel_sign, _join_width                        # noqa: E402
from matplotlib.colors import TwoSlopeNorm                                    # noqa: E402
from visdetect.analysis.align import align_spikes_to_events                   # noqa: E402
from visdetect.analysis.tf_glm import TFGLMConfig, assemble_design, pulse_times_from_tf  # noqa: E402
from visdetect.analysis.tf_glm_data import session_trial_regressors           # noqa: E402
from visdetect.analysis.constants import DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS    # noqa: E402

BIN = DEFAULT_BIN_SIZE
SIG = DEFAULT_SIGMA_MS / 1000.0 / BIN
ALIGN = {  # name: (display window, baseline window)
    "pulse": ((-0.4, 0.8), (-0.4, -0.05)),
    "change": ((-0.5, 1.0), (-0.5, -0.05)),
    "fa": ((-0.8, 0.8), (-0.8, -0.40)),
}
TITLES = {"pulse": "fast TF pulse", "change": "Change_ON (hit trials)", "fa": "FA (early lick)"}
XLAB = {"pulse": "t from fast TF pulse (s)", "change": "t from Change_ON (s)",
        "fa": "t from FA lick (s)"}
MIN_EV = 5
# Use ALL fast pulses (None = no cap). This was 600, which threw away ~98.5% of the
# ~41k fast pulses in a session on the assumption that 600 was "plenty for a mean
# PETH". It is not: the per-pulse response sits ~20x BELOW the spiking noise, so the
# subsample left the raw pulse PETH noise-dominated (its post-window sign agreed with
# the GLM kernel's only ~55% of the time = near chance). All pulses costs ~9 s/cell
# and buys sqrt(41309/600) ~ 8x SNR.
PULSE_CAP = None
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/heatmap_transient_sustained"   # was a stale
# `vd_tf_bg046` path from the old repo — re-running wrote nothing where anyone looked.
CACHE = OUT / "peth_traces.npz"          # legacy per-figure cache (only build() writes it)
# The shared cache already holds ALL 520 responsive cells over the SAME windows/BIN/SIG,
# rebuilt with ALL ~41k fast pulses. Read it instead of rebuilding (~1 h) here.
SHARED_CACHE = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_all.npz"
_RNG = np.random.default_rng(42)


def _cfg():
    return TFGLMConfig(include_movement=False, include_phase=False, include_tiled_baseline=True,
                       standardize_design=True, fast_fit=True, tf_encoding="log2", min_pulses_per_label=20)


def _ztrace(spk, times, win, base):
    if len(times) < MIN_EV:
        return None, None
    binned, t = align_spikes_to_events(spk, list(times), window=win, bin_size=BIN)
    binned = np.asarray(binned, float)
    bmask = (t >= base[0]) & (t < base[1])
    base_vals = binned[:, bmask].ravel()
    mu, sd = base_vals.mean(), base_vals.std()
    if not np.isfinite(sd) or sd < 1e-6:
        sd = max(base_vals.mean(), 1.0)
    z = gaussian_filter1d(binned.mean(0), SIG) if SIG > 0 else binned.mean(0)
    return (z - mu) / sd, t


def _outcome_times(session, event, outcome):
    et = np.asarray(get_event_times_by_trial(session, event), float)
    return [et[i] for i, tr in enumerate(session.trials)
            if str(getattr(tr, "trialoutcome", "") or "").lower() == outcome
            and i < et.size and np.isfinite(et[i])]


def build(force=False):
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    cells = load_cells()
    cells = cells[cells["class"].isin(["transient", "sustained"])].reset_index(drop=True)
    tax = {k: None for k in ALIGN}
    rows = []
    mats = {k: [] for k in ALIGN}
    for (subj, sess), g in cells.groupby(["subject", "session"]):
        pkl = Path(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
        if not pkl.exists():
            continue
        s = load_session(str(pkl))
        cfg = _cfg()
        trials, _ = session_trial_regressors(s, cfg)
        d = assemble_design(trials, cfg)
        fast, _slow = pulse_times_from_tf(d, cfg)
        fast = np.asarray(fast, float)
        if PULSE_CAP is not None and fast.size > PULSE_CAP:
            fast = np.sort(_RNG.choice(fast, PULSE_CAP, replace=False))
        ev = {"pulse": fast,
              "change": _outcome_times(s, "Change_ON", "hit"),
              "fa": _outcome_times(s, "FA", "fa")}
        for _, r in g.iterrows():
            spk = np.sort(_spikes(s, int(r["unit"])))
            trace = {}
            for k, (win, base) in ALIGN.items():
                z, t = _ztrace(spk, ev[k], win, base)
                trace[k] = z
                if t is not None and tax[k] is None:
                    tax[k] = t
            rows.append(dict(subject=subj, session=sess, unit=int(r["unit"]), cls=r["class"]))
            for k in ALIGN:
                mats[k].append(trace[k])
        del s
        gc.collect()
        print(f"  {subj}/{sess}: {len(g)} cells", flush=True)
    meta = pd.DataFrame(rows)
    # pad missing traces to NaN rows of the right length
    out = {"meta_subject": meta.subject.values, "meta_session": meta.session.values,
           "meta_unit": meta.unit.values, "meta_cls": meta.cls.values}
    for k in ALIGN:
        L = len(tax[k])
        M = np.full((len(rows), L), np.nan)
        for i, tr in enumerate(mats[k]):
            if tr is not None and len(tr) == L:
                M[i] = tr
        out[f"mat_{k}"] = M
        out[f"t_{k}"] = tax[k]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if SHARED_CACHE.exists() and not a.force:
        D = {k: v for k, v in np.load(SHARED_CACHE, allow_pickle=True).items()}
    else:
        D = build(force=a.force)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    # the shared cache carries all 520 cells; this figure is the 2-CLASS view
    cls_all = D["meta_cls"].astype(str)
    keep = np.isin(cls_all, ("transient", "sustained"))
    D = {k: (v[keep] if (k.startswith("mat_") or k.startswith("meta_")) else v)
         for k, v in D.items()}
    cls = D["meta_cls"].astype(str)

    # SIGN-ALIGN the pulse traces by the GLM KERNEL (independent of this PETH). Signing
    # on the trace's own post-pulse window is circular: it flips each cell's own noise
    # positive, fabricating a pre-pulse rise and an identical bump on every cell.
    psign = _kernel_sign(D)
    D["mat_pulse"] = D["mat_pulse"] * psign[:, None]

    # ORDER: transient block, then sustained block; WITHIN a block by kernel WIDTH.
    # (It used to sort by each cell's OWN fast-pulse peak latency — circular: sorting a
    # noise-dominated trace by its own peak ALWAYS manufactures a diagonal, which is what
    # the old "every cell is TF-locked / latency tiling" impression actually was.)
    width = _join_width(D)
    order = []
    for c in ("transient", "sustained"):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(np.nan_to_num(width[idx], nan=1e9))]
        order.append(idx)
    n_tr = len(order[0])
    order = np.concatenate(order)

    # kernel_fwhm per cell (registry) for the width-distribution top panel — the
    # actual quantity that DEFINES the classes (sign-agnostic; unlike the raw
    # pulse-PETH pop-mean, which is ~flat only because cells tile response latencies).
    cells_reg = load_cells()
    fwhm_map = {(r.subject, r.session, int(r.unit)): r.kernel_fwhm for r in cells_reg.itertuples()}
    fwhm = np.array([fwhm_map.get((str(D["meta_subject"][i]), str(D["meta_session"][i]),
                                   int(D["meta_unit"][i])), np.nan) for i in range(len(cls))])

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3.2], hspace=0.16, wspace=0.22)
    keys = ["pulse", "change", "fa"]
    # The TF-pulse response (~1 Hz) is small vs ongoing-firing SD, so a baseline-z
    # scale washes out its SHAPE. Peak-normalise the pulse column per unit to show
    # the narrow(transient) vs broad(sustained) WIDTH; keep change/FA on baseline-z
    # to preserve their MAGNITUDE difference.
    ims = {}
    for j, k in enumerate(keys):
        t = D[f"t_{k}"]
        M = D[f"mat_{k}"][order].copy()
        peaknorm = (k == "pulse")
        # ⚠️ NO PEAK-NORMALISATION any more. Dividing each cell by its OWN peak is
        # circular: it rescales that cell's largest noise excursion up to 1.0, so every
        # cell — signal or not — is drawn with a full-amplitude "response". The pulse is
        # already sign-aligned by the GLM kernel and stays on the baseline-z scale, with a
        # robust percentile colour range (same treatment as heatmap_continuum).
        # ── top panel ────────────────────────────────────────────────
        axp = fig.add_subplot(gs[0, j])
        if peaknorm:
            # kernel-WIDTH distribution by class (defines the classes). The raw
            # pulse-PETH pop-mean is uninformative: cells tile latencies AND ~50%
            # are suppression-type, so their signed mean cancels to ~flat.
            bins = np.arange(0, 0.66, 0.05)
            for c, col in (("transient", TCOL), ("sustained", SCOL)):
                axp.hist(fwhm[cls == c], bins=bins, color=col, alpha=0.6, label=c,
                         density=True, edgecolor="white", linewidth=0.4)
            axp.axvline(NARROW, color=TCOL, ls="--", lw=1.3)
            axp.axvline(BROAD, color=SCOL, ls="--", lw=1.3)
            axp.set_title("kernel WIDTH defines the classes", fontsize=14, fontweight="bold")
            axp.set_xlabel("GLM TF-kernel FWHM (s)", fontsize=12)
            axp.set_ylabel("density", fontsize=13)
            axp.legend(frameon=False, fontsize=11)
        else:
            for c, col in (("transient", TCOL), ("sustained", SCOL)):
                sub = M[cls[order] == c]
                if not len(sub):
                    continue
                mean = np.nanmean(sub, 0)
                sem = np.nanstd(sub, 0) / np.sqrt(np.sum(np.isfinite(sub), 0).clip(1))
                axp.plot(t, mean, color=col, lw=2.2, label=c)
                axp.fill_between(t, mean - sem, mean + sem, color=col, alpha=0.2)
            axp.axvline(0, color="0.6", lw=0.8); axp.axhline(0, color="0.85", lw=0.8)
            axp.set_title(TITLES[k] + "  — magnitude", fontsize=14, fontweight="bold")
            axp.set_xlim(t[0], t[-1])
            axp.set_ylabel("z-score (pop mean)", fontsize=13) if j == 1 else None
        axp.tick_params(labelsize=11)
        for sp in ("top", "right"):
            axp.spines[sp].set_visible(False)
        # ── heatmap ──────────────────────────────────────────────────
        axh = fig.add_subplot(gs[1, j])
        if peaknorm:
            pmax = float(np.nanpercentile(np.abs(M), 97)) or 1.0
            imkw = dict(norm=TwoSlopeNorm(vmin=-pmax, vcenter=0.0, vmax=pmax))
        else:
            # asymmetric TwoSlopeNorm: responses are mostly EXCITATORY (0..3),
            # only mild suppression (down to ~-1.5), so don't waste the deep-blue.
            norm = TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=3.0)
            imkw = dict(norm=norm)
        ims[k] = axh.imshow(M, aspect="auto", cmap="RdBu_r",
                            extent=[t[0], t[-1], len(M), 0], interpolation="nearest", **imkw)
        axh.axhline(n_tr, color="k", lw=1.5)
        axh.axvline(0, color="k", lw=1.0, ls="--")
        axh.set_xlabel(XLAB[k], fontsize=13)
        axh.tick_params(labelsize=11)
        if j == 0:
            # the two colored labels ARE the y-axis descriptor (block above/below
            # the divider); no separate ylabel → no duplication. tick numbers = cell index.
            axh.text(-0.145, n_tr / 2, "transient", rotation=90, va="center", ha="center",
                     transform=axh.get_yaxis_transform(), color=TCOL, fontsize=13, fontweight="bold")
            axh.text(-0.145, n_tr + (len(M) - n_tr) / 2, "sustained", rotation=90, va="center",
                     ha="center", transform=axh.get_yaxis_transform(), color=SCOL,
                     fontsize=13, fontweight="bold")
        else:
            axh.set_yticks([])
    # colorbars: pulse (peak-norm shape) separate from change/FA (z magnitude);
    # change/FA on symmetric ±3 (was ±4, washed out) so 0 stays white and the bulk 0–2 shows colour
    cb2 = fig.colorbar(ims["fa"], ax=[fig.axes[3], fig.axes[5]], fraction=0.02, pad=0.015)
    cb2.set_label("change / FA:  z-score (per-unit, baseline)", fontsize=12)
    cb2.ax.tick_params(labelsize=10)
    cb1 = fig.colorbar(ims["pulse"], ax=fig.axes[1], fraction=0.05, pad=0.03)
    cb1.set_label("pulse: sign-aligned z (suppression flipped +)", fontsize=10)
    cb1.ax.tick_params(labelsize=9)
    n_su = len(order) - n_tr
    fig.suptitle(f"TF-responsive cells by kernel width — transient (n={n_tr}) vs sustained (n={n_su})\n"
                 "sustained cells carry the change- and lick-related signals; transient cells are near-pure fast sensory\n"
                 "pulse: ALL ~41k pulses/session, sign-aligned BY THE GLM KERNEL, rows sorted by kernel WIDTH "
                 "(no peak-norm, no peak-latency sort — both were circular)",
                 fontsize=12, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"heatmap_transient_sustained.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/heatmap_transient_sustained.png (+.pdf)  [transient={n_tr}, sustained={n_su}]")


if __name__ == "__main__":
    main()
