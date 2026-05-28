"""F22: External behavioral validation per HMM state for BG_046.

Produces a four-panel figure analogous to Ashwood Fig 6:
  Panel A  Lick latency distributions per state (boxplot or violin).
  Panel B  Response-time quantile bars (analog of Q-Q tail).
  Panel C  Per-state psychometric curves (P(lick) vs log2 change_size).
  Panel D  TF-pulse responsiveness per state.

The figure is the "are the states real?" evidence for manuscript purposes.

Usage
-----
    py scripts/analysis/behavior/hmm_external_validation.py \\
        --model data/hmm/BG_046/best_model.pkl \\
        --manifest data/BG_046_staging_manifest_v2.csv \\
        --pkl-dir data/pkls/BG_046 \\
        --out FIGURES/behavior/BG_046/hmm/external_validation.png \\
        --data-out data/hmm/BG_046/external_validation \\
        --confidence-threshold 0.8
"""

import argparse
import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Ensure worktree's src/ shadows the editable install when running directly.
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visdetect.analysis.config import load_staging_manifest
from visdetect.analysis.hmm import GLMHMM, decode_session, auto_label_states_explicit
from visdetect.analysis.hmm_validation import (
    per_state_lick_latency,
    per_state_response_time_quantiles,
    per_state_psychometric_slope,
    per_state_tf_pulse_lick_rate,
)
from visdetect.core.session import load_session
from visdetect.viz.plotting import set_style, despine


STATE_COLORS = {
    "Impulsive": "#d95f02",
    "Stimulus_sensitive": "#1b9e77",
    "Disengaged": "#7570b3",
}


def _color_for_label(label: str) -> str:
    if label.startswith("Stimulus"):
        return STATE_COLORS["Stimulus_sensitive"]
    return STATE_COLORS.get(label, "#666666")


def _resolve_pkl_path(manifest_row: pd.Series, pkl_dir: Path):
    """Find a session .pkl file. Tries an explicit 'pkl_path' column first,
    then falls back to globbing pkl_dir by session_name (the idiom in
    scripts/analysis/behavior/hmm_cross_validation.py)."""
    if "pkl_path" in manifest_row and pd.notna(manifest_row["pkl_path"]):
        return Path(manifest_row["pkl_path"])
    sname = str(manifest_row.get("session_name", ""))
    if not sname:
        return None
    candidates = list(pkl_dir.glob(f"*{sname}*.pkl"))
    return candidates[0] if candidates else None


def gather_assignments(
    model: GLMHMM,
    manifest: pd.DataFrame,
    pkl_dir: Path,
    *,
    confidence_threshold: float = 0.8,
):
    """Decode every session and concatenate into a single trial DataFrame.

    Returns (assignments_df, state_labels, sessions_by_name) where the third
    item is a dict {session_name -> Session} for downstream per-session
    access (needed for TF-pulse responsiveness which uses trial-level traces).
    """
    state_labels = auto_label_states_explicit(model)
    rows = []
    sessions_by_name: dict = {}
    for _, mrow in manifest.iterrows():
        pkl = _resolve_pkl_path(mrow, pkl_dir)
        if pkl is None or not pkl.exists():
            print(f"  Skip {mrow.get('session_name')}: pkl not found")
            continue
        try:
            sess = load_session(str(pkl))
        except Exception as exc:
            print(f"  Skip {pkl}: {exc}")
            continue
        df = decode_session(model, sess, state_labels=state_labels,
                            confidence_threshold=confidence_threshold)
        sname = sess.session_name or str(mrow.get("session_name", ""))
        df["session_name"] = sname
        rows.append(df)
        sessions_by_name[sname] = sess
    if not rows:
        raise RuntimeError("No sessions decoded.")
    return pd.concat(rows, ignore_index=True), state_labels, sessions_by_name


def plot_validation(
    latency_df: pd.DataFrame,
    rt_q_df: pd.DataFrame,
    psy_df: pd.DataFrame,
    tf_df: pd.DataFrame,
    state_labels,
    out_path: Path,
):
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Panel A — lick latency
    ax = axes[0]
    K = len(state_labels)
    for k in range(K):
        row = latency_df[latency_df["state"] == k].iloc[0]
        if not np.isnan(row["median_latency_s"]):
            ax.bar(k, row["median_latency_s"],
                   color=_color_for_label(state_labels[k]), edgecolor="k")
            ax.errorbar(k, row["median_latency_s"], yerr=row["iqr_s"] / 2,
                        fmt="none", color="k", capsize=3)
    ax.set_xticks(range(K))
    ax.set_xticklabels(state_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Lick latency (s, median ± IQR/2)")
    ax.set_title("A. Lick latency per state")
    despine(ax)

    # Panel B — RT quantiles
    ax = axes[1]
    for k in range(K):
        row = rt_q_df[rt_q_df["state"] == k].iloc[0]
        ax.plot([0.25, 0.5, 0.75, 0.9],
                [row["q25"], row["q50"], row["q75"], row["q90"]],
                "-o", color=_color_for_label(state_labels[k]),
                label=state_labels[k])
    ax.set_xlabel("Quantile")
    ax.set_ylabel("RT (s)")
    ax.set_title("B. RT distribution shape")
    ax.legend(fontsize=8)
    despine(ax)

    # Panel C — psychometric slope
    ax = axes[2]
    for k in range(K):
        row = psy_df[psy_df["state"] == k].iloc[0]
        if np.isnan(row["slope"]):
            continue
        xx = np.linspace(0, 2.2, 60)
        yy = 1.0 / (1.0 + np.exp(-(row["intercept"] + row["slope"] * xx)))
        ax.plot(xx, yy, color=_color_for_label(state_labels[k]),
                label=f"{state_labels[k]} (slope={row['slope']:.2f})")
    ax.set_xlabel("log2(change_size)")
    ax.set_ylabel("P(lick)")
    ax.set_ylim(0, 1)
    ax.set_title("C. Per-state psychometric")
    ax.legend(fontsize=7)
    despine(ax)

    # Panel D — TF-pulse responsiveness
    ax = axes[3]
    for k in range(K):
        row = tf_df[tf_df["state"] == k].iloc[0]
        if not np.isnan(row["p_lick_pulse_locked"]):
            ax.bar(k, row["p_lick_pulse_locked"],
                   color=_color_for_label(state_labels[k]), edgecolor="k")
            ax.text(k, row["p_lick_pulse_locked"] + 0.005,
                    f"n={row['n_pulses']}", ha="center", fontsize=7)
    ax.set_xticks(range(K))
    ax.set_xticklabels(state_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("P(lick within 400 ms of sub-threshold TF pulse)")
    ax.set_title("D. TF-pulse responsiveness  (key discriminator)")
    despine(ax)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    # NOTE: manifest comes from load_staging_manifest() (uses the project's
    # configured staging-manifest path). No --manifest argument is exposed.
    ap.add_argument("--pkl-dir", required=True, type=Path,
                    help="Directory containing per-session .pkl files (e.g. data/pkls/BG_046)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-out", required=True, type=Path)
    ap.add_argument("--confidence-threshold", type=float, default=0.8)
    args = ap.parse_args()

    model = GLMHMM.load(args.model)
    K = model.n_states
    manifest = load_staging_manifest(qc_only=True)

    assignments_df, state_labels, sessions_by_name = gather_assignments(
        model, manifest, args.pkl_dir,
        confidence_threshold=args.confidence_threshold,
    )

    state_col = "hmm_state"

    args.data_out.mkdir(parents=True, exist_ok=True)

    latency_df = per_state_lick_latency(assignments_df, n_states=K, state_col=state_col)
    latency_df.to_csv(args.data_out / "lick_latency_per_state.csv", index=False)

    rt_q_df = per_state_response_time_quantiles(assignments_df, n_states=K, state_col=state_col)
    rt_q_df.to_csv(args.data_out / "rt_quantiles_per_state.csv", index=False)

    psy_df = per_state_psychometric_slope(assignments_df, n_states=K, state_col=state_col)
    psy_df.to_csv(args.data_out / "psychometric_slope_per_state.csv", index=False)

    tf_rows = []
    for sess_name, sub in assignments_df.groupby("session_name"):
        sess = sessions_by_name.get(sess_name)
        if sess is None:
            continue
        # Critical: reset the index so trial_idx in per_state_tf_pulse_lick_rate
        # corresponds to the session's local trials list, not the pooled
        # DataFrame's row position.
        sub_local = sub.reset_index(drop=True)
        tf_local = per_state_tf_pulse_lick_rate(sess, sub_local, n_states=K, state_col=state_col)
        tf_local["session_name"] = sess_name
        tf_rows.append(tf_local)
    sessions_by_name.clear()
    gc.collect()
    if tf_rows:
        tf_concat = pd.concat(tf_rows, ignore_index=True)
        tf_concat.to_csv(args.data_out / "tf_pulse_per_state_per_session.csv", index=False)
        tf_pooled = (
            tf_concat.groupby("state")
                     .agg(n_pulses=("n_pulses", "sum"),
                          n_pulse_locked_licks=("n_pulse_locked_licks", "sum"))
                     .reset_index()
        )
        tf_pooled["p_lick_pulse_locked"] = np.where(
            tf_pooled["n_pulses"] > 0,
            tf_pooled["n_pulse_locked_licks"] / tf_pooled["n_pulses"],
            np.nan,
        )
        tf_pooled.to_csv(args.data_out / "tf_pulse_per_state_pooled.csv", index=False)
    else:
        tf_pooled = pd.DataFrame({
            "state": list(range(K)),
            "n_pulses": [0] * K,
            "n_pulse_locked_licks": [0] * K,
            "p_lick_pulse_locked": [np.nan] * K,
        })

    plot_validation(latency_df, rt_q_df, psy_df, tf_pooled, state_labels, args.out)


if __name__ == "__main__":
    main()
