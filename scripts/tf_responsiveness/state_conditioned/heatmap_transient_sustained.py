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
from transient_vs_sustained import load_cells, TCOL, SCOL                     # noqa: E402
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
TITLES = {"pulse": "fast TF pulse", "change": "Hit @ Change_ON", "fa": "FA @ early-lick"}
MIN_EV = 5
PULSE_CAP = 600          # subsample fast pulses (thousands/session) — plenty for a mean PETH
OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/heatmap_transient_sustained")
CACHE = OUT / "peth_traces.npz"
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
        if fast.size > PULSE_CAP:
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
    D = build(force=a.force)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    cls = D["meta_cls"].astype(str)
    # order: transient block then sustained block; within block by fast-pulse peak time
    tp = D["t_pulse"]
    peak_t = np.full(len(cls), np.nan)
    for i, row in enumerate(D["mat_pulse"]):
        if np.isfinite(row).any():
            peak_t[i] = tp[np.nanargmax(row)]
    order = []
    for c in ("transient", "sustained"):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(np.nan_to_num(peak_t[idx], nan=1e9))]
        order.append(idx)
    n_tr = len(order[0])
    order = np.concatenate(order)

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
        if peaknorm:
            pk = np.nanmax(np.abs(M), axis=1, keepdims=True)
            pk[~np.isfinite(pk) | (pk < 1e-9)] = 1.0
            M = M / pk
        vlim = 1.0 if peaknorm else 4.0
        ylab = "peak-norm (pop mean)" if peaknorm else "z (pop mean)"
        # PSTH (normalize-then-average): mean across each class
        axp = fig.add_subplot(gs[0, j])
        for c, col in (("transient", TCOL), ("sustained", SCOL)):
            m = cls[order] == c
            sub = M[m]
            if not len(sub):
                continue
            mean = np.nanmean(sub, 0)
            sem = np.nanstd(sub, 0) / np.sqrt(np.sum(np.isfinite(sub), 0).clip(1))
            axp.plot(t, mean, color=col, lw=2, label=c)
            axp.fill_between(t, mean - sem, mean + sem, color=col, alpha=0.2)
        axp.axvline(0, color="0.6", lw=0.8); axp.axhline(0, color="0.85", lw=0.8)
        axp.set_title(TITLES[k] + ("  (shape)" if peaknorm else "  (magnitude)"),
                      fontsize=12, fontweight="bold")
        axp.set_xlim(t[0], t[-1]); axp.set_ylabel(ylab if j == 0 else "")
        if j == 0:
            axp.legend(frameon=False, fontsize=9)
        for sp in ("top", "right"):
            axp.spines[sp].set_visible(False)
        # heatmap
        axh = fig.add_subplot(gs[1, j])
        ims[k] = axh.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                            extent=[t[0], t[-1], len(M), 0], interpolation="nearest")
        axh.axhline(n_tr, color="k", lw=1.5)
        axh.axvline(0, color="k", lw=0.8, ls="--")
        axh.set_xlabel(f"t from {TITLES[k].split('@')[0].strip()} (s)")
        if j == 0:
            axh.set_ylabel("cell  (transient ↑ | sustained ↓)")
            axh.text(-0.14, n_tr / 2, "transient", rotation=90, va="center", ha="center",
                     transform=axh.get_yaxis_transform(), color=TCOL, fontsize=11, fontweight="bold")
            axh.text(-0.14, n_tr + (len(M) - n_tr) / 2, "sustained", rotation=90, va="center", ha="center",
                     transform=axh.get_yaxis_transform(), color=SCOL, fontsize=11, fontweight="bold")
        else:
            axh.set_yticks([])
    cb1 = fig.colorbar(ims["pulse"], ax=fig.axes[1], fraction=0.05, pad=0.02)
    cb1.set_label("pulse: peak-norm (shape)", fontsize=8)
    cb2 = fig.colorbar(ims["fa"], ax=[fig.axes[3], fig.axes[5]], fraction=0.015, pad=0.01)
    cb2.set_label("change / FA: z-score (per-unit, baseline)")
    n_su = len(order) - n_tr
    fig.suptitle(f"TF-responsive cells by kernel width — transient (n={n_tr}) vs sustained (n={n_su})\n"
                 "sustained cells carry the change- and lick-related signals; transient cells are near-pure fast sensory",
                 fontsize=13.5, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"heatmap_transient_sustained.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/heatmap_transient_sustained.png (+.pdf)  [transient={n_tr}, sustained={n_su}]")


if __name__ == "__main__":
    main()
