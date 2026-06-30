"""Fig A (talk substrate): Cell-type inventory for BG_046 striatum.

Plain-English: "How many neurons did we record, and what kinds?" Putative cell
types are split by spike width (trough-to-peak, t2p): narrow ~ fast-spiking
interneuron (FSI), broad ~ projection neuron (SPN). The split is purely
waveform-shape (a 2-component GMM, threshold = mean of component means); in
striatum narrow≈FSI / broad≈SPN. No D1/D2 claim (not waveform-separable).

Two panels:
  (1) t2p distribution, coloured by on-disk cell-type label, with the GMM threshold.
  (2) per-session unit yield (good/stable units), stacked by cell type, stage-shaded.

t2p is recomputed from RawWaveforms using the SAME functions that produced the
on-disk labels, so colours separate cleanly at the threshold.

Usage:
    py scripts/talk_substrate/fig_a_celltype_inventory.py [--force] [--n_workers N]
"""
from __future__ import annotations

import os
# Pin BLAS/threads per worker BEFORE numpy import (project convention).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

from visdetect.analysis import config as cfg                       # noqa: E402
from visdetect.suite.loader import load_staging_manifest           # noqa: E402
from visdetect.suite.plotting import setup_style                   # noqa: E402
from visdetect.analysis.config import STAGE_COLORS                 # noqa: E402
from visdetect.analysis.tracking_qc import (                       # noqa: E402
    load_raw_mean_waveform, extract_peak_channel,
)
from visdetect.analysis.waveform_celltype import compute_waveform_features  # noqa: E402

setup_style()

RAW_WF_DIR = cfg.RAW_WF_DIR
T2P_CACHE = C.CACHE_DIR / "bg046_waveform_t2p.csv"


def _date_key(session_8: str):
    """(yyyy, mm, dd) sort key from a canonical DDMMYYYY string."""
    try:
        return (int(session_8[4:8]), int(session_8[2:4]), int(session_8[0:2]))
    except (ValueError, IndexError):
        return (9999, 99, 99)


def _t2p_worker(session_8: str):
    """Compute t2p for every RawWaveforms unit in one session. Top-level for ProcessPool."""
    rw_dir = os.path.join(RAW_WF_DIR, session_8, "RawWaveforms")
    rows = []
    if not os.path.isdir(rw_dir):
        return rows
    for fn in os.listdir(rw_dir):
        if not (fn.startswith("Unit") and fn.endswith("_RawSpikes.npy")):
            continue
        try:
            kid = int(fn[len("Unit"):-len("_RawSpikes.npy")])
        except ValueError:
            continue
        mean_wf = load_raw_mean_waveform(RAW_WF_DIR, session_8, kid)
        if mean_wf is None or mean_wf.ndim != 2:
            continue
        pc = extract_peak_channel(mean_wf)
        feats = compute_waveform_features(mean_wf[:, pc])
        rows.append({"session_8": session_8, "cluster_id": int(kid),
                     "t2p_ms": feats["t2p_ms"], "half_width_ms": feats["half_width_ms"]})
    return rows


def compute_or_load_t2p(sessions_8, force=False, n_workers=8) -> pd.DataFrame:
    if T2P_CACHE.exists() and not force:
        df = pd.read_csv(T2P_CACHE, dtype={"session_8": str})
        have = set(df["session_8"].unique())
        if set(sessions_8).issubset(have):
            print(f"[t2p] loaded cache {T2P_CACHE} ({len(df)} units, {len(have)} sessions)")
            return df
        print("[t2p] cache missing some sessions -> recompute")
    print(f"[t2p] computing t2p from RawWaveforms for {len(sessions_8)} sessions "
          f"(n_workers={n_workers}) ...")
    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for rows in ex.map(_t2p_worker, sessions_8):
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(T2P_CACHE, index=False)
    print(f"[t2p] wrote {T2P_CACHE} ({len(df)} units)")
    return df


def load_celltype_labels() -> pd.DataFrame:
    lab = pd.read_csv(cfg.WAVEFORM_LABELS_PATH)
    lab["session_8"] = lab["session_date"].map(C.canon)
    lab["cluster_id"] = lab["cluster_id"].astype(int)
    lab["celltype_display"] = lab["celltype"].map(C.normalize_celltype)
    return lab[["session_8", "cluster_id", "celltype_display"]]


def load_glt_units() -> pd.DataFrame:
    glt = pd.read_csv(cfg.GLT_PATH, usecols=["Session_Date", "Cluster_ID"])
    glt["session_8"] = glt["Session_Date"].map(C.canon)
    glt["cluster_id"] = glt["Cluster_ID"].astype(int)
    return glt[["session_8", "cluster_id"]]


def stage_map() -> dict:
    try:
        man = load_staging_manifest(qc_only=False)
        man["session_8"] = man["session_name"].map(C.canon)
        return dict(zip(man["session_8"], man["stage"]))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] staging manifest unavailable: {e}")
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute t2p cache")
    ap.add_argument("--n_workers", type=int, default=min(8, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    labels = load_celltype_labels()
    glt = load_glt_units()
    stages = stage_map()

    labeled_sessions = sorted(labels["session_8"].unique(), key=_date_key)
    t2p = compute_or_load_t2p(labeled_sessions, force=args.force, n_workers=args.n_workers)

    # ── Histogram source: units with both t2p and a cell-type label ──────────
    hist_df = t2p.merge(labels, on=["session_8", "cluster_id"], how="inner")
    hist_df = hist_df[np.isfinite(hist_df["t2p_ms"])]
    narrow_t2p = hist_df.loc[hist_df["celltype_display"] == C.NARROW, "t2p_ms"].values
    broad_t2p = hist_df.loc[hist_df["celltype_display"] == C.BROAD, "t2p_ms"].values

    # GMM threshold line
    thr = np.nan
    if C.WAVEFORM_STATS_PATH.exists():
        thr = float(pd.read_csv(C.WAVEFORM_STATS_PATH)["threshold_ms"].iloc[0])

    # ── Per-session counts: all good/stable units (GLT) labelled where possible ─
    units = glt.merge(labels, on=["session_8", "cluster_id"], how="left")
    units["celltype_display"] = units["celltype_display"].fillna(C.UNKNOWN)
    per_sess = (units.groupby(["session_8", "celltype_display"]).size()
                .unstack(fill_value=0))
    for col in [C.NARROW, C.BROAD, C.UNKNOWN]:
        if col not in per_sess.columns:
            per_sess[col] = 0
    per_sess["total"] = per_sess[[C.NARROW, C.BROAD, C.UNKNOWN]].sum(axis=1)
    per_sess["stage"] = [stages.get(s, "excluded") for s in per_sess.index]
    per_sess = per_sess.reindex(sorted(per_sess.index, key=_date_key))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.5], wspace=0.22)

    # Panel 1: t2p distribution
    ax1 = fig.add_subplot(gs[0])
    bins = np.linspace(0, 1.0, 41)
    ax1.hist([broad_t2p, narrow_t2p], bins=bins, stacked=True,
             color=[C.celltype_color(C.BROAD), C.celltype_color(C.NARROW)],
             label=[f"{C.BROAD}  (n={len(broad_t2p)})",
                    f"{C.NARROW}  (n={len(narrow_t2p)})"],
             edgecolor="white", linewidth=0.3)
    if np.isfinite(thr):
        ax1.axvline(thr, ls="--", color="k", lw=1.3)
        ax1.text(thr + 0.01, ax1.get_ylim()[1] * 0.92, f"GMM split\n{thr:.2f} ms",
                 fontsize=8, va="top")
    n_lab = len(narrow_t2p) + len(broad_t2p)
    pct_narrow = 100 * len(narrow_t2p) / max(n_lab, 1)
    ax1.set_xlabel("trough-to-peak width (ms)")
    ax1.set_ylabel("# units")
    ax1.set_title(f"Putative cell types by spike width\n"
                  f"{n_lab} labelled units · {pct_narrow:.0f}% narrow", fontsize=11)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel 2: per-session yield + composition
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(per_sess))
    b = np.zeros(len(per_sess))
    for ct in [C.BROAD, C.NARROW, C.UNKNOWN]:
        vals = per_sess[ct].values
        color = C.celltype_color(ct) if ct != C.UNKNOWN else "#d9d9d9"
        ax2.bar(x, vals, bottom=b, color=color, width=0.85,
                label=ct if ct != C.UNKNOWN else "Unlabelled", edgecolor="none")
        b = b + vals
    ax2.set_ylim(0, per_sess["total"].max() * 1.12)
    # stage shading (contiguous runs) — colour the background only, no inline text
    stage_seq = per_sess["stage"].values
    seen_stages = []
    start = 0
    for i in range(1, len(stage_seq) + 1):
        if i == len(stage_seq) or stage_seq[i] != stage_seq[start]:
            st = stage_seq[start]
            col = STAGE_COLORS.get(st, "#cccccc")
            ax2.axvspan(x[start] - 0.5, x[i - 1] + 0.5, color=col, alpha=0.12, zorder=0)
            if st not in seen_stages:
                seen_stages.append(st)
            start = i
    ax2.set_xticks(x)
    ax2.set_xticklabels([s for s in per_sess.index], rotation=90, fontsize=6)
    ax2.set_xlabel("session (chronological, DDMMYYYY)")
    ax2.set_ylabel("# good/stable units")
    n_lab_sess = int((per_sess[[C.NARROW, C.BROAD]].sum(axis=1) > 0).sum())
    ax2.set_title(f"Per-session unit yield & cell-type composition\n"
                  f"{int(per_sess['total'].sum())} units · {len(per_sess)} sessions · "
                  f"cell-type labels for {n_lab_sess}/{len(per_sess)} sessions",
                  fontsize=11)
    handles = [Patch(facecolor=C.celltype_color(C.BROAD), label=C.BROAD),
               Patch(facecolor=C.celltype_color(C.NARROW), label=C.NARROW),
               Patch(facecolor="#d9d9d9", label="Unlabelled")]
    for st in seen_stages:
        handles.append(Patch(facecolor=STAGE_COLORS.get(st, "#cccccc"), alpha=0.45,
                             label=f"stage: {st}"))
    ax2.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", ncol=2)

    fig.suptitle(f"{C.SUBJECT} striatum (CP): cell-type substrate & recording yield",
                 fontsize=13, y=1.02)
    fig.text(0.5, -0.04,
             "Putative cell types split by spike width (trough-to-peak); a 2-component GMM sets "
             "the dashed split. CAVEAT: 84% of units fall in the narrow/FSI mode — biologically "
             "implausible for striatum, where projection neurons should dominate — so the "
             "spike-width cell-type split is treated as unreliable (likely chronic-probe SPN "
             "under-yield), not a real ~5:1 FSI:SPN ratio.",
             ha="center", fontsize=8, color="#555555", wrap=True)

    out = C.save_talk_figure(fig, "fig_a_celltype_inventory")
    print(f"[fig] wrote {out}")

    # ── Stats CSV ───────────────────────────────────────────────────────────
    table = per_sess.reset_index().rename(columns={
        "index": "session", C.NARROW: "n_narrow", C.BROAD: "n_broad",
        C.UNKNOWN: "n_unlabelled", "total": "n_good_stable"})
    table = table[["session_8", "stage", "n_good_stable", "n_narrow",
                   "n_broad", "n_unlabelled"]].rename(columns={"session_8": "session"})
    stats_path = C.stats_csv_path("fig_a_celltype_inventory")
    table.to_csv(stats_path, index=False)
    print(f"[fig] wrote {stats_path}")

    # ── Report to stdout (yield bounds everything) ───────────────────────────
    print("\n=== BG_046 cell-type inventory ===")
    print(f"good/stable units (GLT)   : {int(per_sess['total'].sum())} "
          f"across {len(per_sess)} sessions")
    print(f"cell-type labelled        : {n_lab} units across {n_lab_sess} sessions")
    print(f"  narrow (FSI)            : {len(narrow_t2p)} ({pct_narrow:.1f}%)")
    print(f"  broad  (MSN/Proj)       : {len(broad_t2p)} ({100 - pct_narrow:.1f}%)")
    print(f"GMM split threshold       : {thr:.3f} ms")
    print(f"per-session units: min={int(per_sess['total'].min())}, "
          f"median={int(per_sess['total'].median())}, max={int(per_sess['total'].max())}")


if __name__ == "__main__":
    main()
