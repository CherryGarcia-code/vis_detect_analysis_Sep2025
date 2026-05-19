"""Online (causal) single-trial state prediction from a fitted GLM-HMM.

Uses the forward algorithm only (no backward pass) to produce
P(z_t | y_{1:t}) — the probability of each state given *only past and
present* observations.  This is the appropriate inference for:
  - Real-time closed-loop experiments
  - Held-out prediction without future information
  - Simulating what an experimenter could know trial-by-trial

Outputs:
  - online_predictions.csv            — per-trial predictions
  - online_prediction_accuracy.png    — rolling accuracy plot
  - online_state_posteriors.png       — state posterior timecourse

Usage
-----
    python scripts/analysis/behavior/hmm_predict.py \
        --data-dir  data/hmm/BG_046 \
        --pkl-dir   data/pkls/BG_046 \
        --out       FIGURES/behavior/BG_046/hmm/prediction \
        --data-out  data/hmm/BG_046

    # Single session:
    python scripts/analysis/behavior/hmm_predict.py \
        --data-dir  data/hmm/BG_046 \
        --pkl-dir   data/pkls/BG_046 \
        --session   01092025 \
        --out       FIGURES/behavior/BG_046/hmm/prediction
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from visdetect.analysis.config import load_staging_manifest
from visdetect.core.session import load_session
from visdetect.analysis.hmm import prepare_session_data
from visdetect.analysis.hmm_downstream import (
    forward_only_state_posteriors,
    load_hmm_results,
    predict_trial_by_trial,
)
from visdetect.viz.plotting import set_style, despine


def _state_palette(K):
    base = ["#7570b3", "#1b9e77", "#d95f02", "#e7298a", "#66a61e", "#e6ab02"]
    return base[:K]


def plot_rolling_accuracy(
    pred_df: pd.DataFrame,
    session_name: str,
    out_dir: Path,
    window: int = 30,
):
    """Plot rolling prediction accuracy (causal vs full-posterior)."""
    fig, ax = plt.subplots(figsize=(12, 4))

    correct = (pred_df["pred_choice"] == pred_df["y_true"]).astype(float)
    rolling = correct.rolling(window, min_periods=1).mean()

    ax.plot(pred_df["trial_idx"], rolling, color="tab:blue", linewidth=1.5,
            label=f"Causal (rolling {window})")
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", label="Chance")

    overall = correct.mean()
    ax.axhline(overall, color="tab:blue", linewidth=1, linestyle=":",
               alpha=0.5, label=f"Overall={overall:.3f}")

    ax.set_xlabel("Trial")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f"Online Prediction Accuracy — {session_name}")
    ax.legend(fontsize=8)
    despine(ax)
    plt.tight_layout()

    sess_dir = out_dir / session_name
    sess_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(sess_dir / "online_prediction_accuracy.png", dpi=150)
    plt.close(fig)


def plot_online_posteriors(
    pred_df: pd.DataFrame,
    state_labels: list,
    n_states: int,
    session_name: str,
    out_dir: Path,
):
    """Plot causal state posteriors as stacked area."""
    palette = _state_palette(n_states)
    T = len(pred_df)
    trials = pred_df["trial_idx"].values

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )

    # Top: outcome raster
    y = pred_df["y_true"].values
    for t in range(T):
        color = "green" if y[t] == 1 else "gray"
        ax_top.axvline(trials[t], color=color, linewidth=0.5, alpha=0.6)
    ax_top.set_ylabel("Lick")
    ax_top.set_yticks([])
    despine(ax_top)

    # Bottom: stacked posteriors
    bottom = np.zeros(T)
    for k in range(n_states):
        vals = pred_df[f"p_state_{k}"].values
        ax_bot.fill_between(trials, bottom, bottom + vals,
                            color=palette[k], alpha=0.7, label=state_labels[k])
        bottom += vals
    ax_bot.set_ylim(0, 1)
    ax_bot.set_xlabel("Trial")
    ax_bot.set_ylabel("P(state | past)")
    ax_bot.legend(loc="upper right", fontsize=8)
    despine(ax_bot)

    fig.suptitle(f"Online (Causal) State Posteriors — {session_name}", fontsize=12)
    plt.tight_layout()

    sess_dir = out_dir / session_name
    sess_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(sess_dir / "online_state_posteriors.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Online single-trial prediction from fitted GLM-HMM."
    )
    parser.add_argument("--data-dir", required=True,
                        help="HMM results directory.")
    parser.add_argument("--pkl-dir", required=True,
                        help="Directory with session pkl files.")
    parser.add_argument("--session", default=None,
                        help="Single session to predict (optional).")
    parser.add_argument("--manifest", default=None,
                        help="Manifest CSV for batch mode.")
    parser.add_argument("--out", default="FIGURES/behavior/hmm/prediction",
                        help="Plot output directory.")
    parser.add_argument("--data-out", default=None,
                        help="CSV output directory (defaults to --data-dir).")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of states to load (e.g. 3). "
                             "Default: highest-K model found on disk.")
    parser.add_argument("--exclude-qc-fail", action="store_true",
                        help="DEPRECATED: SESSION_FILTER handles QC.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Bypass SESSION_FILTER.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_out = Path(args.data_out) if args.data_out else data_dir
    data_out.mkdir(parents=True, exist_ok=True)
    set_style(context="talk")

    # Load model
    model, assignments_df, state_labels = load_hmm_results(data_dir, K=args.K)
    K = model.n_states
    print(f"Loaded K={K} model, labels={state_labels}")

    pkl_dir = Path(args.pkl_dir)

    # Select sessions
    if args.session:
        session_names = [args.session]
    elif args.manifest or not args.session:
        manifest = load_staging_manifest(
            manifest_path=args.manifest,
            apply_filter=not getattr(args, 'no_filter', False),
        )
        session_names = manifest["session_name"].tolist()
    else:
        session_names = assignments_df["session_name"].unique().tolist()

    all_predictions = []

    for sname in session_names:
        candidates = list(pkl_dir.glob(f"*{sname}*.pkl"))
        if not candidates:
            print(f"  SKIP {sname}: pkl not found")
            continue

        try:
            session = load_session(str(candidates[0]))
            sd = prepare_session_data(session)
            sd["session_name"] = sname
        except Exception as exc:
            print(f"  SKIP {sname}: {exc}")
            continue

        # Causal prediction
        pred_df = predict_trial_by_trial(model, sd, causal=True)
        pred_df.insert(0, "session_name", sname)
        all_predictions.append(pred_df)

        acc = (pred_df["pred_choice"] == pred_df["y_true"]).mean()
        print(f"  {sname}: {len(pred_df)} trials, "
              f"accuracy={acc:.3f}")

        # Plots
        plot_rolling_accuracy(pred_df, sname, out_dir)
        plot_online_posteriors(pred_df, state_labels, K, sname, out_dir)

    # Save all predictions
    if all_predictions:
        full_df = pd.concat(all_predictions, ignore_index=True)
        csv_path = data_out / "online_predictions.csv"
        full_df.to_csv(csv_path, index=False)
        print(f"\nAll predictions saved: {csv_path}")

        # Summary
        summary = (
            full_df.groupby("session_name")
            .apply(lambda g: pd.Series({
                "n_trials": len(g),
                "accuracy": (g["pred_choice"] == g["y_true"]).mean(),
                "mean_p_lick": g["p_lick"].mean(),
            }))
            .reset_index()
        )
        print("\n" + "=" * 50)
        print("Online Prediction Summary")
        print("=" * 50)
        print(summary.to_string(index=False))
        print(f"\nOverall accuracy: "
              f"{(full_df['pred_choice'] == full_df['y_true']).mean():.3f}")

    print(f"\nPlots saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
