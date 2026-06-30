"""Fig B (talk substrate): Task-event-aligned population PSTHs, split by cell type.

Plain-English: "What does striatal activity look like around the key task events?"
Each panel is the population z-scored firing rate (per-unit z to a shared baseline,
then averaged across units; mean +/- SEM) aligned to one event:

  - Baseline onset (trial start)          : all outcomes
  - Change onset (the visual change)       : go trials only (hit + miss)
  - Response lick (hit)                    : motor-aligned
  - Early lick / false alarm (fa)          : motor-aligned

Traces are split by putative cell type (narrow/FSI vs broad/MSN-Proj). BG_046 is
all dorsal striatum (CP), so this is the striatum panel; cortex pending a second
animal. NOTE: per Fig A the spike-width cell-type split is unreliable here (84%
narrow), so read the cell-type *difference* cautiously; the per-event response
*shape* is the robust message.

Alignment honours EVENT_VALID_OUTCOMES (e.g. Change_ON excludes fa/abort, where
the change never occurred). Normalisation = per-unit z to a shared baseline,
normalise-then-average, div-by-zero guarded (CLAUDE.md golden rule).

Usage:
    py scripts/talk_substrate/fig_b_event_psths.py [--force]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gc
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

from visdetect.analysis import config as cfg                       # noqa: E402
from visdetect.suite.loader import load_session                    # noqa: E402
from visdetect.suite.plotting import setup_style                   # noqa: E402
from visdetect.analysis.constants import DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS  # noqa: E402
from visdetect.analysis.utils import (                             # noqa: E402
    get_good_cluster_ids, build_population_tensor,
    compute_zscore_normalized, smooth_psth,
)

setup_style()

BIN = DEFAULT_BIN_SIZE
CACHE = C.CACHE_DIR / "fig_b_event_psth.npz"

# Per-event plot window + z-score baseline window (baseline must lie inside window).
EVENTS = {
    "Baseline_ON": dict(window=(-1.0, 2.0), baseline=(-1.0, -0.3),
                        title="Baseline onset (trial start)\nall outcomes",
                        xlabel="time from baseline onset (s)"),
    "Change_ON":   dict(window=(-1.0, 1.5), baseline=(-0.4, -0.05),
                        title="Change onset (visual change)\ngo trials: hit + miss",
                        xlabel="time from change onset (s)"),
    "Hit":         dict(window=(-2.0, 0.75), baseline=(-1.75, -1.25),
                        title="Response lick (hit)\nmotor-aligned",
                        xlabel="time from response lick (s)"),
    "FA":          dict(window=(-2.0, 0.75), baseline=(-1.75, -1.25),
                        title="Early lick / false alarm\nmotor-aligned",
                        xlabel="time from early lick (s)"),
}


def load_celltype_lookup() -> dict:
    lab = pd.read_csv(cfg.WAVEFORM_LABELS_PATH)
    lab["session_8"] = lab["session_date"].map(C.canon)
    lab["cluster_id"] = lab["cluster_id"].astype(int)
    lab["ct"] = lab["celltype"].map(C.normalize_celltype)
    return {(r.session_8, r.cluster_id): r.ct for r in lab.itertuples()}


def build_cache(limit=None):
    ct_lookup = load_celltype_lookup()
    sessions_8 = sorted({s for (s, _c) in ct_lookup.keys()})
    if limit:
        sessions_8 = sessions_8[:limit]
    print(f"[B] {len(sessions_8)} sessions with cell-type labels"
          + (f" (LIMIT {limit})" if limit else ""))

    # accumulators: per event -> per celltype -> list of per-unit mean z-traces
    acc = {ev: {C.NARROW: [], C.BROAD: []} for ev in EVENTS}
    bc_by_event = {}
    n_trials_by_event = {ev: {C.NARROW: 0, C.BROAD: 0} for ev in EVENTS}

    for si, s8 in enumerate(sessions_8, 1):
        try:
            sess = load_session(s8)
        except Exception as e:  # noqa: BLE001
            print(f"  [{si}/{len(sessions_8)}] {s8}: load failed ({e}); skip")
            continue
        cids = get_good_cluster_ids(sess)
        ct = np.array([ct_lookup.get((s8, int(c)), C.UNKNOWN) for c in cids])
        n_lab = int(np.isin(ct, [C.NARROW, C.BROAD]).sum())
        msg = [f"  [{si}/{len(sessions_8)}] {s8}: {len(cids)} good units, {n_lab} labelled"]
        for ev, spec in EVENTS.items():
            try:
                tensor, bc, valid = build_population_tensor(
                    sess, list(cids), event_name=ev,
                    window=spec["window"], bin_size=BIN)
            except ValueError:
                msg.append(f"{ev}:0tr")
                continue
            if tensor.shape[0] == 0:
                msg.append(f"{ev}:0tr")
                continue
            z = compute_zscore_normalized(tensor, bc, spec["baseline"])  # (tr,bins,units)
            unit_mean = np.nanmean(z, axis=0)  # (bins, units)
            bc_by_event[ev] = bc
            for ci in range(unit_mean.shape[1]):
                lab = ct[ci]
                if lab not in (C.NARROW, C.BROAD):
                    continue
                tr = unit_mean[:, ci]
                if not np.all(np.isfinite(tr)):
                    continue
                acc[ev][lab].append(smooth_psth(tr, BIN, sigma_ms=DEFAULT_SIGMA_MS))
            n_trials_by_event[ev][C.NARROW] += tensor.shape[0]
            msg.append(f"{ev}:{tensor.shape[0]}tr")
        print(" | ".join(msg))
        del sess
        gc.collect()

    # collapse to mean/sem and save
    out = {}
    for ev in EVENTS:
        out[f"{ev}__bc"] = bc_by_event.get(ev, np.array([]))
        for lab in (C.NARROW, C.BROAD):
            mat = np.vstack(acc[ev][lab]) if acc[ev][lab] else np.zeros((0, 0))
            out[f"{ev}__{lab}__mat_n"] = np.array([mat.shape[0]])
            if mat.shape[0] > 0:
                out[f"{ev}__{lab}__mean"] = np.nanmean(mat, axis=0)
                out[f"{ev}__{lab}__sem"] = (np.nanstd(mat, axis=0) /
                                            np.sqrt(mat.shape[0]))
            else:
                out[f"{ev}__{lab}__mean"] = np.array([])
                out[f"{ev}__{lab}__sem"] = np.array([])
    np.savez(CACHE, **out)
    print(f"[B] wrote cache {CACHE}")
    return out


def load_cache():
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only first N sessions (smoke test)")
    args = ap.parse_args()

    if args.limit:
        build_cache(limit=args.limit)
        return
    data = build_cache() if (args.force or not CACHE.exists()) else load_cache()

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.22)
    stat_rows = []
    for idx, (ev, spec) in enumerate(EVENTS.items()):
        ax = fig.add_subplot(gs[idx])
        bc = data[f"{ev}__bc"]
        if bc.size == 0:
            ax.set_title(spec["title"] + "  (no data)")
            continue
        # shade baseline window
        ax.axvspan(spec["baseline"][0], spec["baseline"][1], color="0.85",
                   alpha=0.5, zorder=0, label="_baseline")
        ax.axvline(0, color="k", lw=1.0, ls="-", zorder=1)
        ax.axhline(0, color="0.6", lw=0.7, ls=":", zorder=1)
        for lab in (C.NARROW, C.BROAD):
            mean = data[f"{ev}__{lab}__mean"]
            sem = data[f"{ev}__{lab}__sem"]
            n = int(data[f"{ev}__{lab}__mat_n"][0])
            if mean.size == 0:
                continue
            color = C.celltype_color(lab)
            ax.plot(bc, mean, color=color, lw=1.8, label=f"{lab} (n={n})", zorder=3)
            ax.fill_between(bc, mean - sem, mean + sem, color=color, alpha=0.2, zorder=2)
            # stats: peak |z| in post-event window (0, end)
            post = bc >= 0
            if post.any():
                pk_i = int(np.nanargmax(np.abs(mean[post])))
                pk_t = bc[post][pk_i]
                pk_z = mean[post][pk_i]
                stat_rows.append({"event": ev, "celltype": lab, "n_units": n,
                                  "peak_abs_z": round(float(pk_z), 3),
                                  "peak_time_s": round(float(pk_t), 3)})
        ax.set_title(spec["title"], fontsize=10)
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel("z-score (shared baseline)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(f"{C.SUBJECT} striatum (CP): task-event-aligned population activity by cell type",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.04,
             "Per-unit z-score to a shared pre-event baseline (grey band), then averaged across "
             "units (mean +/- SEM). Change-onset uses go trials only (hit+miss); fa/abort excluded "
             "because the change never occurred. Cell-type split is unreliable here (see Fig A) — "
             "read the response SHAPE, not the FSI-vs-SPN difference.",
             ha="center", fontsize=8, color="#555555", wrap=True)

    out = C.save_talk_figure(fig, "fig_b_event_psths")
    print(f"[fig] wrote {out}")
    sdf = pd.DataFrame(stat_rows)
    spath = C.stats_csv_path("fig_b_event_psths")
    sdf.to_csv(spath, index=False)
    print(f"[fig] wrote {spath}")
    print("\n=== Fig B peak responses ===")
    if not sdf.empty:
        print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
