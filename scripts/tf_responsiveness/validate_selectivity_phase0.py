"""Phase-B early-validation gate for the fast-minus-slow selectivity core.

Runs the selectivity detector on a real session's good-and-stable units and
answers the gate question: does a *sparse* population of units have a
fast-minus-slow selectivity peak that exits the label-shuffle null at short
latency (~0.12-0.17 s)? Re-picks clean exemplars for the eventual HITL tagger.

Usage:
    cd /e/python_analysis/git_repos/vd_tf_phase0
    PYTHONPATH=src py scripts/tf_responsiveness/validate_selectivity_phase0.py \
        --session BG_046_16092025

Outputs (under the worktree root):
    data/cache/tf_selectivity/<session>_features.csv
    figures/tf_responsiveness/<session>_selectivity_gate.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Self-insert this worktree's src so we never run the primary repo's editable
# install by accident (the editable visdetect is pinned to the primary src).
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis.tf_pulse import _collect_pulses
from visdetect.analysis.tf_selectivity import (
    TFSelectivityConfig,
    compute_session_selectivity,
    unit_features,
)
from visdetect.analysis.tf_pulse import TFRespPulseConfig
from visdetect.analysis.constants import LOHSE_SENSORY_CD_WINDOW, TF_PULSE_TRACE_PRE

_ROOT = Path(__file__).resolve().parents[2]
_CACHE = _ROOT / "data" / "cache" / "tf_selectivity"
_FIGS = _ROOT / "figures" / "tf_responsiveness"


def build_feature_table(session, cluster_ids, cfg=None) -> pd.DataFrame:
    """Pure seam: corrected pulses -> per-unit selectivity -> feature table."""
    if cfg is None:
        cfg = TFSelectivityConfig()
    fast_times, slow_times = _collect_pulses(session, cfg.pulse)
    sels = compute_session_selectivity(session, cluster_ids, fast_times, slow_times, cfg)
    rows = [unit_features(s) for s in sels]
    df = pd.DataFrame(rows)
    df.attrs["selectivities"] = sels
    df.attrs["n_fast_total"] = int(np.asarray(fast_times).size)
    df.attrs["n_slow_total"] = int(np.asarray(slow_times).size)
    return df


def _render_gate_figure(df, cfg, out_png, session_name):
    sels = df.attrs.get("selectivities", [])
    sig = df[(df["shuffle_p"] < 0.05) & (df["sufficient"])]
    sig_ids = set(sig["cluster_id"].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: significant units' selectivity traces + mean
    ax = axes[0]
    traces = []
    for s in sels:
        if int(s.cluster_id) in sig_ids and np.all(np.isfinite(s.selectivity)):
            ax.plot(s.t_vec, s.selectivity, color="0.6", lw=0.6, alpha=0.7)
            traces.append(s.selectivity)
    if traces:
        m = np.nanmean(np.vstack(traces), axis=0)
        ax.plot(sels[0].t_vec, m, color="k", lw=2.0, label=f"mean (n={len(traces)})")
        ax.legend(fontsize=8)
    ax.axvline(0, color="r", ls="--", lw=0.8)
    ax.axvspan(*LOHSE_SENSORY_CD_WINDOW, color="orange", alpha=0.15)
    ax.set_xlabel("time from pulse (s)")
    ax.set_ylabel("selectivity (fast-slow)/sigma_b")
    ax.set_title("Significant-unit selectivity")

    # Panel 2: peak-latency histogram of significant units
    ax = axes[1]
    if len(sig):
        ax.hist(sig["sel_peak_latency"], bins=20, range=(0, 0.5), color="steelblue")
    ax.axvspan(*LOHSE_SENSORY_CD_WINDOW, color="orange", alpha=0.25,
               label="Lohse 0.122-0.167 s")
    ax.set_xlabel("peak latency (s)")
    ax.set_ylabel("# significant units")
    ax.set_title("Peak latency")
    ax.legend(fontsize=8)

    # Panel 3: |sel_peak| vs sel_z_vs_null, coloured by significance
    ax = axes[2]
    finite = df[np.isfinite(df["sel_z_vs_null"])]
    is_sig = (finite["shuffle_p"] < 0.05) & (finite["sufficient"])
    ax.scatter(finite.loc[~is_sig, "sel_peak"].abs(),
               finite.loc[~is_sig, "sel_z_vs_null"], s=10, color="0.7", label="ns")
    ax.scatter(finite.loc[is_sig, "sel_peak"].abs(),
               finite.loc[is_sig, "sel_z_vs_null"], s=14, color="crimson", label="p<0.05")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("|sel peak|")
    ax.set_ylabel("selectivity z vs null")
    ax.set_title("Detection scatter")
    ax.legend(fontsize=8)

    fig.suptitle(f"TF selectivity gate — {session_name} "
                 f"(fast={df.attrs.get('n_fast_total')}, slow={df.attrs.get('n_slow_total')})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="e.g. BG_046_16092025 or 16092025")
    ap.add_argument("--n-shuffles", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.001,
                    help="trace bin (s); 0.004 is ~4x faster and fine for the gate")
    ap.add_argument("--top", type=int, default=15, help="# exemplar candidates to print")
    args = ap.parse_args()

    from visdetect.suite.loader import load_session
    from visdetect.analysis.utils import get_good_cluster_ids

    sess = load_session(args.session)
    cluster_ids = get_good_cluster_ids(sess)
    print(f"[gate] {args.session}: {len(cluster_ids)} good-and-stable units")

    cfg = TFSelectivityConfig(
        pulse=TFRespPulseConfig(trace_pre=TF_PULSE_TRACE_PRE, dt=args.dt),
        n_shuffles=args.n_shuffles)
    df = build_feature_table(sess, cluster_ids, cfg)

    sname = str(getattr(sess, "session_name", args.session))
    _CACHE.mkdir(parents=True, exist_ok=True)
    csv_path = _CACHE / f"{sname}_features.csv"
    df.drop(columns=[]).to_csv(csv_path, index=False)
    print(f"[gate] wrote {csv_path}")

    _render_gate_figure(df, cfg, _FIGS / f"{sname}_selectivity_gate.png", sname)
    print(f"[gate] wrote {_FIGS / f'{sname}_selectivity_gate.png'}")

    sig = df[(df["shuffle_p"] < 0.05) & (df["sufficient"])].copy()
    n_total = int((df["sufficient"]).sum())
    frac = (len(sig) / n_total) if n_total else float("nan")
    print(f"[gate] significant responders: {len(sig)} / {n_total} sufficient "
          f"units ({100*frac:.1f}%)")
    in_win = sig[(sig["sel_peak_latency"] >= LOHSE_SENSORY_CD_WINDOW[0]) &
                 (sig["sel_peak_latency"] <= LOHSE_SENSORY_CD_WINDOW[1])]
    print(f"[gate] of those, {len(in_win)} peak in Lohse window "
          f"{LOHSE_SENSORY_CD_WINDOW}")

    print(f"[gate] top {args.top} exemplar candidates (by sel_z_vs_null):")
    cols = ["cluster_id", "sel_peak", "sel_peak_latency", "sel_z_vs_null",
            "shuffle_p", "split_half_r", "n_fast", "n_slow"]
    top = sig.sort_values("sel_z_vs_null", ascending=False).head(args.top)
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
