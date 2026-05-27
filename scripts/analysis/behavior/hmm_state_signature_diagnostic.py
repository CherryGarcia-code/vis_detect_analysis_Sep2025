"""Plot fitted GLM-HMM states in the (P(lick|catch), P(lick|large-go)) plane.

This is the F4 diagnostic figure: it confirms that K=3 fitted states fall
into the three predicted corners corresponding to the a priori state structure
(spec §1.1):
    - Impulsive:        high p(catch) AND high p(go)
    - Stim-sensitive:   low p(catch)  AND high p(go)
    - Disengaged:       low p(catch)  AND low p(go)

Usage
-----
    py scripts/analysis/behavior/hmm_state_signature_diagnostic.py \
        --model data/hmm/BG_046/best_model.pkl \
        --out FIGURES/behavior/BG_046/hmm/state_signature.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from visdetect.analysis.hmm import GLMHMM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path,
                    help="Path to fitted GLMHMM pickle")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output PNG path")
    ap.add_argument("--stim-high", type=float, default=2.0,
                    help="log2 of largest change_size used as 'high stim' (default 2.0 = 4x)")
    args = ap.parse_args()

    model = GLMHMM.load(args.model)

    if model.weights is None:
        raise ValueError(
            f"Model loaded from {args.model} has not been fitted "
            "(weights are None). Fit the model before generating the diagnostic."
        )

    K, D = model.n_states, model.n_features

    x_catch = np.zeros(D); x_catch[0] = 1.0
    x_high  = np.zeros(D); x_high[0]  = 1.0; x_high[1] = args.stim_high

    p_catch = np.array([float(expit(model.weights[k] @ x_catch)) for k in range(K)])
    p_high  = np.array([float(expit(model.weights[k] @ x_high))  for k in range(K)])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Reference regions (a priori state predictions).
    # axhspan x-bounds are AXES fractions; with xlim=0..1, fraction == data coord.
    ax.axhspan(0.5, 1.0, xmin=0.5, xmax=1.0, alpha=0.08, color="#d95f02",
               label="Impulsive region")
    ax.axhspan(0.5, 1.0, xmin=0.0, xmax=0.2, alpha=0.08, color="#1b9e77",
               label="Stim-sensitive region")
    ax.axhspan(0.0, 0.2, xmin=0.0, xmax=0.2, alpha=0.08, color="#7570b3",
               label="Disengaged region")

    palette = plt.cm.tab10(np.arange(K))
    for k in range(K):
        ax.scatter(p_catch[k], p_high[k], s=120, color=palette[k],
                   edgecolor="k", zorder=10)
        ax.annotate(f"State {k}", (p_catch[k], p_high[k]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("P(lick | catch, baseline history)")
    ax.set_ylabel(f"P(lick | large go [log2={args.stim_high}])")
    ax.set_title("Fitted state signature in P(lick) plane (F4 diagnostic)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
