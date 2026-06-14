#!/usr/bin/env python3
"""Validate UnitMatch tracks with ISI-fingerprint stability (paper Fig 4).

A real same-neuron's inter-spike-interval distribution is highly stable across
days; two different neurons within one session have different ISI distributions.
The discriminability of (matched cross-day ISI corr) vs (non-matched within-day
ISI corr) measures whether tracking is real or chance.

UnitMatch paper (van Beest et al. 2024 Nat Methods): AUC ~0.95 within-day,
~0.82 at 183 days for the example mouse. Tracks that fall well below the
matched distribution are likely false-positive merges.

Outputs (FIGURES/tracking_qc/):
    track_validation.png        - 4-panel: ISI examples, distributions, ROC,
                                  per-track corr matrices for top long tracks
    track_validation_stats.csv  - per-track mean ISI corr, rank, span
    track_validation_summary.json - overall AUC + medians

Usage:
    py scripts/pipelines/tracking/validate_long_tracks.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from visdetect.core.session import load_session  # noqa: E402
from visdetect.analysis.config import TRACKING_QC_DIR  # noqa: E402

DEFAULT_REGISTRY = ("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
                    "BG_046/unit_match/output/all42/unit_index.csv")
DEFAULT_PKL_DIR = REPO_ROOT / "data" / "pkls" / "BG_046"
OUT_DIR = Path(TRACKING_QC_DIR)

# ISI histogram: 50 log-spaced bins from 1 ms to 10 s (paper-style)
ISI_BINS = np.logspace(-3, 1, 51)
ISI_CENTERS = 0.5 * (ISI_BINS[:-1] + ISI_BINS[1:])
RNG = np.random.default_rng(42)
N_NONMATCHED_PER_SESSION = 500     # control pairs
LONG_SPAN_MIN = 10                 # "long track" threshold for highlighting


def isi_hist(spike_times: np.ndarray) -> np.ndarray:
    """Normalised log-ISI histogram. NaN if too few spikes."""
    if len(spike_times) < 20:
        return np.full(len(ISI_BINS) - 1, np.nan)
    isis = np.diff(np.sort(spike_times))
    isis = isis[(isis > 0) & (isis < 10)]
    if len(isis) < 10:
        return np.full(len(ISI_BINS) - 1, np.nan)
    h, _ = np.histogram(isis, bins=ISI_BINS)
    return h / h.sum() if h.sum() > 0 else np.full_like(h, np.nan, dtype=float)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def session_pkl(pkl_dir: Path, session_name: str) -> Path | None:
    """DDMMYYYY may or may not have a leading zero on the day."""
    for s in (session_name, "0" + session_name if len(session_name) == 7 else session_name):
        p = pkl_dir / f"BG_046_{s}.pkl"
        if p.exists():
            return p
    return None


def build_isi_cache(unit_index: pd.DataFrame, pkl_dir: Path) -> dict:
    """{(session, ks_unit_id) -> isi histogram (50,)}. Loads each pkl once."""
    cache = {}
    needed_sess = sorted(unit_index["session"].unique())
    print(f"Building ISI cache from {len(needed_sess)} pkls ...", flush=True)
    for sess in needed_sess:
        p = session_pkl(pkl_dir, sess)
        if p is None:
            print(f"  skip {sess}: no pkl"); continue
        S = load_session(str(p))
        ks_ids_needed = set(unit_index.loc[unit_index.session == sess, "ks_unit_id"].astype(int))
        n_hit = 0
        for c in S.clusters:
            cid = int(c.cluster_id)
            if cid in ks_ids_needed:
                cache[(sess, cid)] = isi_hist(np.asarray(c.spike_times))
                n_hit += 1
        del S
        print(f"  {sess}: {n_hit}/{len(ks_ids_needed)} units cached", flush=True)
    return cache


def matched_pairs(unit_index: pd.DataFrame, cache: dict):
    """ISI corr for pairs of (session, ks_id) sharing a uid -- CROSS-SESSION ONLY.
    Within-session pairs (same uid in same session = within-day oversplit merge)
    are NOT cross-day evidence and would inflate the matched distribution."""
    rows = []
    for uid, grp in unit_index.groupby("global_uid"):
        if grp["session"].nunique() < 2:
            continue
        entries = grp[["session", "ks_unit_id"]].values
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                s_i, k_i = entries[i][0], int(entries[i][1])
                s_j, k_j = entries[j][0], int(entries[j][1])
                if s_i == s_j:                       # skip within-session pairs
                    continue
                c = corr(cache.get((s_i, k_i)), cache.get((s_j, k_j)))
                if not np.isnan(c):
                    rows.append((uid, s_i, k_i, s_j, k_j, c))
    return pd.DataFrame(rows, columns=["uid", "s1", "k1", "s2", "k2", "isi_corr"])


def nonmatched_pairs(unit_index: pd.DataFrame, cache: dict, n_per_sess: int):
    """Within-session, different-uid ISI correlations (control)."""
    rows = []
    for sess, grp in unit_index.groupby("session"):
        rows_grp = grp[["ks_unit_id", "global_uid"]].values
        n = len(rows_grp)
        if n < 4:
            continue
        idx = RNG.integers(0, n, size=(min(n_per_sess, n * (n - 1) // 2), 2))
        for a, b in idx:
            if a == b or rows_grp[a, 1] == rows_grp[b, 1]:
                continue
            c = corr(cache.get((sess, int(rows_grp[a, 0]))),
                     cache.get((sess, int(rows_grp[b, 0]))))
            if not np.isnan(c):
                rows.append((sess, c))
    return pd.DataFrame(rows, columns=["session", "isi_corr"])


def auc_score(matched: np.ndarray, nonmatched: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Manual ROC AUC + curves (avoids sklearn dep)."""
    scores = np.concatenate([matched, nonmatched])
    labels = np.concatenate([np.ones_like(matched), np.zeros_like(nonmatched)])
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = tp / max(1, labels.sum())
    fpr = fp / max(1, (1 - labels).sum())
    auc = float(np.trapz(tpr, fpr))
    return auc, fpr, tpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR)
    args = ap.parse_args()

    unit_index = pd.read_csv(args.registry, dtype={"session": str})
    print(f"loaded registry: {len(unit_index)} unit-session entries, "
          f"{unit_index['global_uid'].nunique()} global UIDs", flush=True)

    cache = build_isi_cache(unit_index, args.pkl_dir)
    print(f"\nISI cache: {len(cache)} (session, unit) histograms")

    matched = matched_pairs(unit_index, cache)
    nonmatched = nonmatched_pairs(unit_index, cache, N_NONMATCHED_PER_SESSION)
    print(f"matched pairs: {len(matched)}, non-matched pairs: {len(nonmatched)}")

    m_vals = matched["isi_corr"].values
    n_vals = nonmatched["isi_corr"].values
    auc, fpr, tpr = auc_score(m_vals, n_vals)

    # per-track stats -- span = # distinct sessions the uid appears in
    span = unit_index.groupby("global_uid")["session"].nunique().rename("span")
    per_track = (matched.groupby("uid")["isi_corr"]
                 .agg(["mean", "median", "min", "count"]).reset_index()
                 .rename(columns={"uid": "global_uid"}))
    per_track = per_track.merge(span.reset_index(), on="global_uid")
    per_track["nonmatched_rank_pct"] = per_track["median"].apply(
        lambda v: 100 * (n_vals < v).mean())     # higher = more above nonmatched
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_track.sort_values("span", ascending=False).to_csv(
        OUT_DIR / "track_validation_stats.csv", index=False)

    long = per_track[per_track["span"] >= LONG_SPAN_MIN].sort_values("span", ascending=False)
    n_long_high = int((long["median"] > np.median(n_vals)).sum())

    print("\n" + "=" * 64)
    print("TRACK VALIDATION (ISI fingerprint, UnitMatch-paper method)")
    print("=" * 64)
    print(f"  matched   ISI corr: median {np.median(m_vals):.3f}, "
          f"25-75% {np.quantile(m_vals, 0.25):.3f}-{np.quantile(m_vals, 0.75):.3f}")
    print(f"  nonmatched ISI corr: median {np.median(n_vals):.3f}, "
          f"25-75% {np.quantile(n_vals, 0.25):.3f}-{np.quantile(n_vals, 0.75):.3f}")
    print(f"  AUC = {auc:.3f}   (paper benchmark: 0.95 within-day, 0.82 at 183d)")
    print(f"  long tracks (>= {LONG_SPAN_MIN} sessions): {len(long)}; "
          f"with median corr > nonmatched median: {n_long_high}/{len(long)}")

    sum_ = dict(auc=auc, matched_median=float(np.median(m_vals)),
                nonmatched_median=float(np.median(n_vals)),
                n_matched_pairs=len(matched), n_nonmatched_pairs=len(nonmatched),
                n_long_tracks=len(long), n_long_good=n_long_high)
    with open(OUT_DIR / "track_validation_summary.json", "w") as f:
        json.dump(sum_, f, indent=2)

    _figure(matched, nonmatched, per_track, long, cache, unit_index, auc, fpr, tpr)
    print(f"\nfigure + CSV -> {OUT_DIR}")


def _figure(matched, nonmatched, per_track, long, cache, unit_index, auc, fpr, tpr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: example long-track ISI histograms across sessions
    ax = fig.add_subplot(gs[0, 0])
    if len(long) > 0:
        ex_uid = long.iloc[0]["global_uid"]
        entries = unit_index[unit_index.global_uid == ex_uid].sort_values("session")
        cmap = plt.cm.viridis(np.linspace(0, 1, len(entries)))
        for (_, row), c in zip(entries.iterrows(), cmap):
            h = cache.get((row["session"], int(row["ks_unit_id"])))
            if h is not None and not np.isnan(h).any():
                ax.plot(ISI_CENTERS * 1000, h, color=c, alpha=0.7, lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("ISI (ms)"); ax.set_ylabel("density")
        ax.set_title(f"uid {ex_uid}: ISI histograms across {len(entries)} sessions")

    # Panel B: matched vs nonmatched correlation distributions
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-0.5, 1, 60)
    ax.hist(nonmatched["isi_corr"], bins=bins, alpha=0.5, color="steelblue",
            label=f"non-matched (within day)  n={len(nonmatched)}", density=True)
    ax.hist(matched["isi_corr"], bins=bins, alpha=0.5, color="crimson",
            label=f"matched (across days)  n={len(matched)}", density=True)
    ax.set_xlabel("ISI histogram correlation"); ax.set_ylabel("density")
    ax.set_title("Matched cross-day vs non-matched within-day")
    ax.legend(fontsize=8)

    # Panel C: ROC
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(fpr, tpr, color="darkred", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC matched vs non-matched   AUC = {auc:.3f}")

    # Panel D: per-track median ISI corr vs span
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(per_track["span"], per_track["median"], s=10, alpha=0.5, color="gray")
    ax.scatter(long["span"], long["median"], s=30, color="crimson",
               label=f"long tracks (>= {LONG_SPAN_MIN} sessions)")
    ax.axhline(np.median(nonmatched["isi_corr"]), color="steelblue", ls="--",
               label="nonmatched median")
    ax.set_xlabel("track span (sessions)"); ax.set_ylabel("median cross-day ISI corr")
    ax.set_title("Per-track stability vs span")
    ax.legend(fontsize=8)

    # Panel E,F: per-track corr matrices for top long tracks
    if len(long) >= 2:
        for plot_i, (_, trow) in enumerate(long.head(2).iterrows()):
            ax = fig.add_subplot(gs[1, 1 + plot_i])
            uid = trow["global_uid"]
            entries = unit_index[unit_index.global_uid == uid].sort_values("session").reset_index(drop=True)
            n = len(entries)
            C = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(n):
                    h1 = cache.get((entries.iloc[i]["session"], int(entries.iloc[i]["ks_unit_id"])))
                    h2 = cache.get((entries.iloc[j]["session"], int(entries.iloc[j]["ks_unit_id"])))
                    C[i, j] = corr(h1, h2)
            im = ax.imshow(C, vmin=-0.2, vmax=1, cmap="RdBu_r")
            ax.set_title(f"uid {uid}: {n}-session ISI corr matrix")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels([s[-4:] for s in entries["session"]], rotation=90, fontsize=5)
            ax.set_yticklabels([s[-4:] for s in entries["session"]], fontsize=5)
            plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"UnitMatch track validation -- ISI fingerprint stability  "
                 f"(BG_046, {unit_index['global_uid'].nunique()} tracked IDs)",
                 fontsize=13)
    fig.savefig(OUT_DIR / "track_validation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
