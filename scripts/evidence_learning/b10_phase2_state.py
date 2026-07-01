"""B10 Phase 2 — impulsivity kernel split by behavioral state (StimSens vs
Impulsive).

NON-CIRCULAR: state labels come from lick RATES / outcomes; the kernel (what
stimulus pattern precedes the lick) is an independent measurement the labeler
never sees. Hypothesis: StimSens FAs = genuine stimulus-driven false alarms
(SHARP kernel); Impulsive FAs = internal itch, stimulus-decoupled (FLAT kernel).

Run: py scripts/evidence_learning/b10_phase2_state.py
Out: FIGURES/evidence_learning/state/b10_state_kernel.png,
     data/cache/evidence_learning/b10_state_kernel_stats.csv
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
    SUBJECTS, CACHE_DIR, FIG_DIR, subject_sessions, load_state_labels_by_key)

setup_style()
STATES = ("StimSens", "Impulsive")
CONF = 0.8


def fa_epochs_by_state(session, subject, skey=None, states=STATES):
    """{state: [fa_epoch dicts]} — FA epochs whose trial's confident state label
    is in `states`."""
    eps = pk.fa_kernel_epochs(session)
    labels = load_state_labels_by_key(subject, skey)
    if labels is None:
        return {s: [] for s in states}
    by = {s: [] for s in states}
    for e in eps:
        idx = e["trial_idx"]
        if idx not in labels.index:
            continue
        row = labels.loc[idx]
        if float(row["state_confidence"]) >= CONF and row["state_label"] in by:
            by[row["state_label"]].append(e)
    return by


def main():
    lags = pk.kernel_lags()
    stats = []
    acc = {s: ([], []) for s in STATES}
    for subject in SUBJECTS:
        for skey, sname, stage, sess in subject_sessions(subject, ("Naive", "Expert")):
            by = fa_epochs_by_state(sess, subject, skey=skey)
            rng = np.random.default_rng(pk.BOOT_SEED)
            for state, eps in by.items():
                wh = pk.withhold_epochs(sess, eps, rng=rng)
                for e, w in zip(eps, wh):
                    if w is not None:
                        acc[state][0].append(e["window"])
                        acc[state][1].append(w)
    nmin = min((len(acc[s][0]) for s in STATES if acc[s][0]), default=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s in STATES:
        fa_w, wh_w = acc[s]
        if len(fa_w) >= nmin > 0:
            rng = np.random.default_rng(pk.BOOT_SEED)
            idx = rng.choice(len(fa_w), nmin, replace=False)
            k, lo, hi = pk.bootstrap_kernel_ci([fa_w[i] for i in idx],
                                               [wh_w[i] for i in idx])
            axes[0].plot(lags, k, label=s)
            axes[0].fill_between(lags, lo, hi, alpha=0.2)
            stats.append({"state": s, "n_pairs": len(fa_w), "n_match": nmin,
                          **pk.kernel_shape_metrics(k)})
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Behavioral kernel by state (pooled)")
    axes[0].set_xlabel("time before recorded lick (s)")
    axes[0].set_ylabel("log2-TF (FA - withhold)")
    axes[0].legend()
    axes[1].axis("off")
    axes[1].text(0.02, 0.5,
                 "Non-circular: state labels use lick rates/outcomes;\n"
                 "the kernel shape is an independent measurement.\n"
                 "Naive-StimSens is the thinnest cell (wide CI).",
                 va="center")
    fig.tight_layout()
    outdir = os.path.join(FIG_DIR, "state")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "b10_state_kernel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.DataFrame(stats).to_csv(
        os.path.join(CACHE_DIR, "b10_state_kernel_stats.csv"), index=False)
    print(pd.DataFrame(stats))


if __name__ == "__main__":
    main()
