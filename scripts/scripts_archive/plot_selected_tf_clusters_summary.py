"""Plot a summarized TF-pulse figure for selected clusters per session.

Features:
- Sessions as rows (earliest top), clusters as columns.
- Omit cluster titles; place session label once at left of each row.
- Add a rightmost column showing the session-mean of selected clusters;
  for the mean we flip negative-going slow responses so the summary always
  shows positive-going responsiveness.
- Larger axis text for readability.

Usage:
    python scripts/analysis/plot_selected_tf_clusters_summary.py --out FIGURES
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import csv

# repo root
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from visdetect.core.session import load_session
from visdetect.analysis.tf_pulse import TFRespPulseConfig, collect_tf_pulse_traces
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MAP = {
    "BG_046_02072025": [151, 43, 51, 366, 135, 526],
    "BG_046_03072025": [522, 313, 575, 201, 92, 199, 312, 574],
    "BG_046_04072025": [130, 322, 392, 382, 537],
    "BG_046_15092025": [528, 189, 643, 570, 519, 514],
    "BG_046_16092025": [579, 145, 422, 602, 379, 134, 30],
    "BG_046_17092025": [540, 492, 440, 410, 630, 245],
}


def find_session_pickle(session_name: str) -> Path | None:
    p1 = REPO / "pkls" / f"{session_name}.pkl"
    if p1.exists():
        return p1
    p2 = REPO / "data" / f"{session_name}.pkl"
    if p2.exists():
        return p2
    return None


def normalized_trace_for_unit(e, t_vec):
    """Return a sign-normalized trace (responsiveness positive).

    If the fast-peak magnitude is larger than the slow negative magnitude,
    return fast_z; otherwise return -slow_z so the direction is flipped.
    """
    fast_peak = e.z_max_fast if np.isfinite(e.z_max_fast) else -np.inf
    slow_mag = -e.z_min_slow if np.isfinite(e.z_min_slow) else -np.inf
    # prefer fastest response if both present
    if fast_peak >= slow_mag:
        return e.fast_z
    else:
        # flip slow so negative deflections become positive
        return -e.slow_z


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot selected TF clusters summary")
    ap.add_argument("--out", default="FIGURES")
    ap.add_argument("--which", choices=["fast", "slow", "both"], default="both")
    ap.add_argument("--no-cache", action="store_true", help="Ignore existing caches and recompute")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    mapping = DEFAULT_MAP
    sessions = list(mapping.keys())
    cfg = TFRespPulseConfig()

    t_vec = None
    per_session_entries = {}
    max_cols = max(len(v) for v in mapping.values())

    # collect traces
    for sess_name in sessions:
        print(f"Collecting traces for {sess_name}")
        pkl = find_session_pickle(sess_name)
        if pkl is None:
            print(f"  WARNING: pickle not found for {sess_name}; skipping")
            per_session_entries[sess_name] = {}
            continue
        session = load_session(str(pkl))
        ident = f"{getattr(session,'subject','unknown')}_{getattr(session,'session_name','unknown')}"
        cache_p = Path("table_output") / "tf_pulse" / ident / "tf_pulse_traces.npz"
        cache_path = None if args.no_cache else (str(cache_p) if cache_p.exists() else None)
        try:
            t_vec_local, entries = collect_tf_pulse_traces(session, cfg=cfg, cache_path=cache_path, show_progress=False)
        except Exception as e:
            print(f"  ERROR collecting traces for {sess_name}: {e}")
            per_session_entries[sess_name] = {}
            continue
        if t_vec is None:
            t_vec = t_vec_local
        per_session_entries[sess_name] = {e.cluster_id: e for e in entries}

    if t_vec is None:
        print("No traces available; aborting")
        return

    # layout includes an extra column for the mean
    n_rows = len(sessions)
    n_cols = max_cols + 1
    fig_w = 2.2 * n_cols
    fig_h = 1.6 * n_rows + 1.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Styling: larger fonts
    label_font = 12
    tick_font = 10
    for ax in np.ravel(axes):
        ax.tick_params(labelsize=tick_font)

    for r, sess_name in enumerate(sessions):
        lookup = per_session_entries.get(sess_name, {})
        clu_list = mapping[sess_name]
        norm_stack = []
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(clu_list):
                clu = clu_list[c]
                e = lookup.get(clu)
                if e is None:
                    ax.axis('off')
                    continue
                # show both traces in cluster panels
                if args.which in ("slow", "both"):
                    ax.plot(t_vec, e.slow_z, color="#d62728", lw=1)
                    ax.fill_between(t_vec, e.slow_z - e.slow_z_sem, e.slow_z + e.slow_z_sem, color="#d62728", alpha=0.15)
                if args.which in ("fast", "both"):
                    ax.plot(t_vec, e.fast_z, color="#1f77b4", lw=1)
                    ax.fill_between(t_vec, e.fast_z - e.fast_z_sem, e.fast_z + e.fast_z_sem, color="#1f77b4", alpha=0.15)
                ax.axvline(0, color='k', linestyle='--', lw=0.6)
                # no cluster title
                if c == 0:
                    # session label once on the left, rotated vertical and positioned beside y-axis
                    ax.set_ylabel(sess_name, fontsize=label_font, rotation=90, labelpad=10)
                if r == n_rows - 1:
                    ax.set_xlabel('time (s)', fontsize=label_font)
                # collect normalized trace for mean
                norm = normalized_trace_for_unit(e, t_vec)
                if norm is not None:
                    norm_stack.append(norm)
            elif c == n_cols - 1:
                # mean column for this session
                if norm_stack:
                    stack = np.stack(norm_stack)
                    mean_trace = np.nanmean(stack, axis=0)
                    if stack.shape[0] > 1:
                        sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
                    else:
                        sem = np.zeros_like(mean_trace)
                    ax.plot(t_vec, mean_trace, color="#4e79a7", lw=2)
                    ax.fill_between(t_vec, mean_trace - sem, mean_trace + sem, color="#4e79a7", alpha=0.2)
                    ax.axvline(0, color='k', linestyle='--', lw=0.8)
                else:
                    ax.axis('off')
                if r == n_rows - 1:
                    ax.set_xlabel('time (s)', fontsize=label_font)
                # put a small label above mean column only for top row
                if r == 0:
                    ax.set_title('mean (responsiveness)', fontsize=label_font)
            else:
                ax.axis('off')

    # Final formatting
    # Increase overall font sizes for axis labels
    for ax in axes[:, 0]:
        # place the vertical session label to the left of the axis
        ax.yaxis.set_label_coords(-0.14, 0.5)
    fig.suptitle('Selected TF clusters — summary (rows=sessions, cols=clusters + mean)', fontsize=14)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    out_png = out_root / 'selected_tf_clusters_summary.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save per-session mean traces CSV
    out_csv = out_root / 'selected_tf_clusters_summary_means.csv'
    header = ['session'] + [f't_{i:.3f}' for i in t_vec]
    with out_csv.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for sess_name in sessions:
            lookup = per_session_entries.get(sess_name, {})
            clu_list = mapping[sess_name]
            norm_stack = []
            for clu in clu_list:
                e = lookup.get(clu)
                if e is None:
                    continue
                norm = normalized_trace_for_unit(e, t_vec)
                if norm is not None:
                    norm_stack.append(norm)
            if norm_stack:
                mean_trace = np.nanmean(np.stack(norm_stack), axis=0)
                w.writerow([sess_name] + mean_trace.tolist())
            else:
                w.writerow([sess_name] + [''] * len(t_vec))

    print(f'Wrote figure: {out_png}')
    print(f'Wrote means CSV: {out_csv}')


if __name__ == '__main__':
    main()
