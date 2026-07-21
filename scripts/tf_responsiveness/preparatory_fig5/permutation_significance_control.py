"""Robustness: PERMUTATION single-unit significance (CLAUDE.md-preferred) instead of
the paper's parametric |z|>2.576. Writes a NEW figure + CSV; does NOT overwrite the
parametric panels.

Per unit, "active at bin t" is defined NON-parametrically: the observed lick-aligned
mean firing-rate deviation from the pre-change baseline exceeds the 99th percentile of
a RANDOM-TIME shuffle null (S draws of n_lick random times across the task span,
deviations pooled per unit). No z-scoring -> sidesteps the baseline-sigma choice (the
shuffle supplies the noise scale). Everything downstream (fraction-active, bootstrap-
over-neurons CI, population onset, per-cell onset~width) is IDENTICAL to the parametric
pipeline, so any change in conclusion is attributable to the significance test.

Output (NEW): FIGURES/preparatory_fig5/permutation/fig_permutation_vs_parametric_hit.{png,pdf}
overlays PERMUTATION (solid) vs PARAMETRIC |z|>2.576 (dashed, SAME cells) fraction-active
per class x region; + permutation_significance.csv. Responsive cells (all 520) + non-TF
subsample (default 2000, seeded). LOCAL ProcessPool.

Usage:  py permutation_significance_control.py [--shuffles 200] [--nontf 2000] [--workers 10]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import gc
import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_common as C
from visdetect.core.session import load_session
from visdetect.analysis.align import get_event_times_by_trial, align_spikes_to_events
from visdetect.analysis.preparatory import (
    active_mask, baseline_mean_sd, fraction_active, bootstrap_fraction_ci, population_onset,
    first_sustained, onset_win_need)

_WIN, _NEED = onset_win_need(0.1, 0.08, C.BIN)
PCTL = 99.0  # two-sided P<0.01 analog: |obs| beyond 99th pct of |null|
GROUPS = ["transient", "sustained", "non-TF"]


def _perm_active(spk, lick_t, change_t, rand_sets):
    """Per-unit permutation active mask (0/1 per bin) + time axis."""
    b_binned, _ = align_spikes_to_events(spk, list(change_t), window=C.BASE_WIN, bin_size=C.BIN)
    mu, _sd = baseline_mean_sd(b_binned)
    l_binned, t = align_spikes_to_events(spk, list(lick_t), window=C.LICK_WIN, bin_size=C.BIN)
    obs = gaussian_filter1d(np.nanmean(np.asarray(l_binned, float), 0), C.SIG_BINS) - mu
    null = np.empty((len(rand_sets), len(t)))
    for s, rt in enumerate(rand_sets):
        rb, _ = align_spikes_to_events(spk, list(rt), window=C.LICK_WIN, bin_size=C.BIN)
        null[s] = gaussian_filter1d(np.nanmean(np.asarray(rb, float), 0), C.SIG_BINS) - mu
    thr = float(np.percentile(np.abs(null), PCTL))
    return (np.abs(obs) > thr).astype(float), np.asarray(t, float)


def _process(task):
    subj, sess, recs, S = task
    pkl = C.REPO / "data/pkls" / subj / f"{sess}.pkl"
    if not pkl.exists():
        return {"rows": [], "err": f"MISSING {pkl}"}
    try:
        s = load_session(str(pkl))
        change = np.asarray(get_event_times_by_trial(s, "Change_ON"), float)
        licks = np.asarray(get_event_times_by_trial(s, "Hit"), float)
        rt = licks - change
        licks = np.where(np.isfinite(rt) & (rt >= C.MIN_RT), licks, np.nan)
        change_t = change[np.isfinite(change)]
        lick_t = licks[np.isfinite(licks)]
        if len(lick_t) < C.MIN_LICKS or len(change_t) < 1:
            del s
            return {"rows": [], "err": None}
        rng = np.random.default_rng(zlib.crc32(str(sess).encode()))
        lo, hi = float(np.min(change_t)) - 2.0, float(np.max(change_t)) + 2.0
        rand_sets = [np.sort(rng.uniform(lo, hi, len(lick_t))) for _ in range(S)]
        rows = []
        for r in recs:
            uid = int(r["unit"])
            spk = C.spikes_for(s, uid)
            if spk.size == 0:
                continue
            active, _t = _perm_active(spk, lick_t, change_t, rand_sets)
            rows.append({"subject": subj, "session": str(sess), "unit": uid,
                         "region": C.REGION[subj], "resp": bool(r["resp"]),
                         "cls": (C.class_from_fwhm(float(r["kernel_fwhm"])) if r["resp"] else "non-TF"),
                         "interp_fwhm": float(r["interp_fwhm"]), "active": active})
        del s
        gc.collect()
        return {"rows": rows, "err": None}
    except Exception as e:
        import traceback
        return {"rows": [], "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def _group_mask(cls, resp, group):
    return (~resp) if group == "non-TF" else (cls == group)


def main(shuffles=200, n_nontf=2000, n_workers=10):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    width = C.load_width()
    wmap = {(str(r.subject), str(r.session), int(r.unit)): float(r.interp_fwhm)
            for r in width.itertuples()}
    resp_rows, nontf_rows = [], []
    for subj, _ in C.MICE:
        reg = C.load_registry(subj)
        reg = reg[reg.session_date.isin(C.good_dates(subj))]
        for _, rr in reg.iterrows():
            rec = {"subject": subj, "session": str(rr["session"]), "unit": int(rr["unit"]),
                   "resp": bool(rr["resp"]), "kernel_fwhm": float(rr["kernel_fwhm"]),
                   "interp_fwhm": wmap.get((subj, str(rr["session"]), int(rr["unit"])), np.nan)}
            (resp_rows if rr["resp"] else nontf_rows).append(rec)
    rng = np.random.default_rng(42)
    if len(nontf_rows) > n_nontf:
        nontf_rows = [nontf_rows[i] for i in rng.choice(len(nontf_rows), n_nontf, replace=False)]
    bysess = {}
    for rec in resp_rows + nontf_rows:
        bysess.setdefault((rec["subject"], rec["session"]), []).append(rec)
    tasks = [(subj, sess, recs, shuffles) for (subj, sess), recs in bysess.items()]
    print(f"START permutation-sig | {len(tasks)} sessions | {len(resp_rows)} resp + "
          f"{len(nontf_rows)} non-TF | S={shuffles} | {n_workers} workers", flush=True)
    rows, errs = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_process, t) for t in tasks])):
            res = fut.result()
            rows += res["rows"]
            if res["err"]:
                errs.append(res["err"])
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(tasks)} sessions", flush=True)

    A = np.array([r["active"] for r in rows])                       # permutation active mask
    cls = np.array([r["cls"] for r in rows])
    reg = np.array([r["region"] for r in rows])
    resp = np.array([r["resp"] for r in rows])
    w = np.array([r["interp_fwhm"] for r in rows])
    t = np.arange(A.shape[1]) * C.BIN + (C.LICK_WIN[0] + C.BIN / 2)
    base_mask = (t >= C.BASE_FRAC_WIN[0]) & (t <= C.BASE_FRAC_WIN[1])

    # PARAMETRIC active for the SAME cells (match prep_hit.npz by subject/session/unit)
    D = np.load(C.REPO / "data/cache/preparatory_fig5/prep_hit.npz", allow_pickle=True)
    Apar_all = active_mask(np.asarray(D["z"], float))
    ckey = {(str(D["meta_subject"][i]), str(D["meta_session"][i]), int(D["meta_unit"][i])): i
            for i in range(len(D["meta_unit"]))}
    midx = np.array([ckey.get((r["subject"], r["session"], r["unit"]), -1) for r in rows])
    Apar = np.full_like(A, np.nan)
    ok = midx >= 0
    Apar[ok] = Apar_all[midx[ok]]

    print(f"\nPERMUTATION significance (S={shuffles}, {PCTL:.0f}th-pct null) | {len(rows)} cells "
          f"| matched parametric {int(ok.sum())} | {len(errs)} errors", flush=True)
    print("Parametric ref (|z|>2.576, full non-TF): sustained -0.738 / transient -0.613 / non-TF -0.338\n", flush=True)

    REGS = [("pooled", np.ones(len(rows), bool)), ("DMS", reg == "DMS"), ("VMS", reg == "VMS")]
    traces, out = {}, []
    for rname, rmask in REGS:
        print(f"[{rname}]", flush=True)
        for grp in GROUPS:
            m = rmask & _group_mask(cls, resp, grp)
            if m.sum() < 3:
                continue
            pm, plo, phi = bootstrap_fraction_ci(A[m], baseline_bins=base_mask, n=2000)
            qm = fraction_active(Apar[m], baseline_bins=base_mask)   # parametric, same cells
            on_perm = population_onset(t, pm, plo)
            traces[(rname, grp)] = (pm, plo, phi, qm)
            print(f"    {grp:10s} n={int(m.sum()):5d}  PERM onset={on_perm if np.isnan(on_perm) else round(on_perm,3)} "
                  f"peak={np.nanmax(pm):.3f}", flush=True)
            out.append({"region": rname, "group": grp, "n": int(m.sum()), "perm_onset_s": on_perm,
                        "perm_peak": float(np.nanmax(pm))})
        rm = rmask & resp & np.isfinite(w)
        idxs = np.where(rm)[0]
        onc = np.array([(t[first_sustained(A[i], _WIN, _NEED)] if first_sustained(A[i], _WIN, _NEED) >= 0
                         else np.nan) for i in idxs])
        wm = w[rm]
        fin = np.isfinite(onc) & np.isfinite(wm)
        if fin.sum() >= 3:
            rho, p = spearmanr(onc[fin], wm[fin])
            print(f"    per-cell onset~width: PERM Spearman rho={rho:+.3f} (p={p:.2g}, n={int(fin.sum())}) "
                  f"[parametric: pooled -0.18 / DMS -0.41 / VMS ~0]", flush=True)
            out.append({"region": rname, "group": "onset~width_spearman", "n": int(fin.sum()),
                        "perm_onset_s": float(rho), "perm_peak": np.nan})

    # ── figure: permutation (solid) vs parametric-same-cells (dashed), 1x3 regions ──
    OUT = C.REPO / "FIGURES/preparatory_fig5/permutation"
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    fig = plt.figure(figsize=(16.5, 5.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.18)
    for ci, (rname, _rm) in enumerate(REGS):
        ax = fig.add_subplot(gs[0, ci])
        ax.axvspan(C.BASE_FRAC_WIN[0], C.BASE_FRAC_WIN[1], color="0.9", zorder=0)
        for grp in GROUPS:
            if (rname, grp) not in traces:
                continue
            pm, plo, phi, qm = traces[(rname, grp)]
            col = C.CLASS_COLORS[grp]
            ax.fill_between(t, plo, phi, color=col, alpha=0.15, lw=0)
            ax.plot(t, pm, color=col, lw=2.4, ls="-")            # permutation
            ax.plot(t, qm, color=col, lw=1.8, ls="--")           # parametric, same cells
        ax.axvline(0, color="k", lw=0.9, ls=":")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(-0.05, 1.0)
        ax.set_title(rname, fontsize=15, fontweight="bold")
        ax.set_xlabel("time from HIT lick (s)")
        if ci == 0:
            ax.set_ylabel("fraction active above baseline")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], color=C.CLASS_COLORS[g], lw=2.4, label=g) for g in GROUPS]
    handles += [Line2D([0], [0], color="0.3", lw=2.4, ls="-", label="permutation (99th-pct null)"),
                Line2D([0], [0], color="0.3", lw=1.8, ls="--", label="parametric |z|>2.576 (same cells)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Single-unit significance: PERMUTATION vs PARAMETRIC — recruitment order is preserved",
                 fontsize=14, y=1.10)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_permutation_vs_parametric_hit.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(out).to_csv(OUT / "permutation_significance.csv", index=False)
    print(f"\nwrote {OUT}/fig_permutation_vs_parametric_hit.png (+pdf, +permutation_significance.csv)", flush=True)
    print("END OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--nontf", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    main(shuffles=a.shuffles, n_nontf=a.nontf, n_workers=a.workers)
