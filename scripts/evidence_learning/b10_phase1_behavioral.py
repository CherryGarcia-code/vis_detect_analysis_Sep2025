"""B10 Phase 1 (behavioral) — the impulsivity kernel across learning, 3 mice.

Fig B10.1: (A) pooled kernel + CI; (B) Naive vs Expert per subject; (C) kernel
half-width vs stage. The kernel = FA-triggered log2-TF minus time-in-trial-
matched withhold. Aligned to the RECORDED lick (no hardware-delay constant).

Run: py scripts/evidence_learning/b10_phase1_behavioral.py
Out: FIGURES/evidence_learning/pooled/b10_behavioral_kernel.png,
     data/cache/evidence_learning/b10_behavioral_kernel_stats.csv
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis import psychophysical_kernel as pk
from visdetect.suite.plotting import setup_style
from visdetect.analysis.evidence_learning_io import (
    SUBJECTS, CACHE_DIR, FIG_DIR, subject_sessions)

setup_style()
HEADLINE = ("Naive", "Expert")


def session_kernel(session, rng):
    """(paired-mean kernel, n_pairs) for one session; empty kernel if no pairs."""
    eps = pk.fa_kernel_epochs(session)
    wh = pk.withhold_epochs(session, eps, rng=rng)
    pairs = [(e["window"], w) for e, w in zip(eps, wh) if w is not None]
    if not pairs:
        return np.zeros(0), 0
    fa_w, wh_w = zip(*pairs)
    return pk.reverse_correlation_kernel(list(fa_w), list(wh_w)), len(pairs)


def collect_windows(subject, stages):
    """{stage: (fa_windows, withhold_windows)} pooling all sessions of a subject."""
    rng = np.random.default_rng(pk.BOOT_SEED)
    acc = {s: ([], []) for s in stages}
    for skey, sname, stage, sess in subject_sessions(subject, stages):
        if stage not in acc:
            continue
        eps = pk.fa_kernel_epochs(sess)
        wh = pk.withhold_epochs(sess, eps, rng=rng)
        for e, w in zip(eps, wh):
            if w is not None:
                acc[stage][0].append(e["window"])
                acc[stage][1].append(w)
    return acc


def stage_kernel(fa_w, wh_w, n_match, seed=pk.BOOT_SEED):
    """n-matched kernel + CI (subsample to n_match pairs)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fa_w), size=min(n_match, len(fa_w)), replace=False)
    return pk.bootstrap_kernel_ci([fa_w[i] for i in idx], [wh_w[i] for i in idx])


def main():
    lags = pk.kernel_lags()
    stats = []
    pooled = {s: ([], []) for s in HEADLINE}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for subject in SUBJECTS:
        acc = collect_windows(subject, HEADLINE)
        nmin = min((len(acc[s][0]) for s in HEADLINE if acc[s][0]), default=0)
        for s in HEADLINE:
            fa_w, wh_w = acc[s]
            pooled[s][0].extend(fa_w)
            pooled[s][1].extend(wh_w)
            if len(fa_w) >= nmin > 0:
                k, lo, hi = stage_kernel(fa_w, wh_w, nmin)
                stats.append({"subject": subject, "stage": s, "n_pairs": len(fa_w),
                              "n_match": nmin, **pk.kernel_shape_metrics(k)})
                axes[1].plot(lags, k, label=f"{subject} {s}")
    # Panel A: pooled kernel + CI
    nmin = min((len(pooled[s][0]) for s in HEADLINE if pooled[s][0]), default=0)
    for s in HEADLINE:
        if len(pooled[s][0]) >= nmin > 0:
            k, lo, hi = stage_kernel(pooled[s][0], pooled[s][1], nmin)
            axes[0].plot(lags, k, label=f"pooled {s}")
            axes[0].fill_between(lags, lo, hi, alpha=0.2)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Impulsivity kernel (pooled, 3 mice)")
    axes[0].set_xlabel("time before recorded lick (s)")
    axes[0].set_ylabel("log2-TF (FA - withhold)")
    axes[0].legend()
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("Per subject: Naive vs Expert")
    axes[1].set_xlabel("time before recorded lick (s)")
    axes[1].legend(fontsize=7)
    # Panel C: half-width vs stage
    sdf = pd.DataFrame(stats)
    if not sdf.empty:
        for subject in SUBJECTS:
            d = sdf[sdf.subject == subject]
            if not d.empty:
                axes[2].plot(d["stage"], d["half_width_s"], "o-", label=subject)
        axes[2].set_title("Kernel half-width vs stage")
        axes[2].set_ylabel("half-width (s)")
        axes[2].legend()
    fig.text(0.5, -0.02, "Limitation: no video -> 'stimulus history preceding "
             "impulsive licks', not pure sensory evidence. n=3 mice.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    outdir = os.path.join(FIG_DIR, "pooled")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "b10_behavioral_kernel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    os.makedirs(CACHE_DIR, exist_ok=True)
    sdf.to_csv(os.path.join(CACHE_DIR, "b10_behavioral_kernel_stats.csv"), index=False)
    print(sdf)


if __name__ == "__main__":
    main()
