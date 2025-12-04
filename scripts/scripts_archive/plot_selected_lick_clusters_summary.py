"""Plot a summarized lick-response figure for selected clusters per session.

This uses the MATLAB-faithful lick routines in
`visdetect.utils.matlab_ports.lick` to collect FA-lick aligned z-traces per unit.

Outputs:
- FIGURES/selected_lick_clusters_summary.png
- FIGURES/selected_lick_clusters_summary_means.csv
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import csv

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from visdetect.analysis import lick as lick_mod
from visdetect.core.session import load_session
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


def sign_normalize_trace(z_trace: np.ndarray, time_axis: np.ndarray, cfg: lick_mod.MatlabLickConfig) -> np.ndarray:
    """Flip the trace sign if necessary so the post-window responsiveness is positive.

    We compare the maximum positive deflection to the absolute minimum (negative deflection)
    within the configured post_window; if the negative is larger we flip sign.
    """
    post_mask = (time_axis >= cfg.post_window[0]) & (time_axis < cfg.post_window[1])
    if not np.any(post_mask):
        return z_trace
    post_vals = z_trace[post_mask]
    max_pos = np.nanmax(post_vals) if post_vals.size else -np.inf
    max_neg = -np.nanmin(post_vals) if post_vals.size else -np.inf
    if max_neg > max_pos:
        return -z_trace
    return z_trace


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot selected FA-lick clusters summary")
    ap.add_argument("--out", default="FIGURES")
    ap.add_argument("--no-cache", action="store_true", help="Ignore caches (not applicable for lick collector)")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    mapping = DEFAULT_MAP
    sessions = list(mapping.keys())
    cfg = lick_mod.MatlabLickConfig()

    t_vec = None
    per_session_lookup = {}
    max_cols = max(len(v) for v in mapping.values())

    for sess_name in sessions:
        print(f"Collecting lick traces for {sess_name}")
        pkl = find_session_pickle(sess_name)
        if pkl is None:
            print(f"  WARNING: pickle not found for {sess_name}; skipping")
            per_session_lookup[sess_name] = {}
            continue
        session = load_session(str(pkl))
        t_vec_local, entries = lick_mod.collect_fa_lick_traces(session, cfg=cfg, good_ids=None, show_progress=False)
        if t_vec is None:
            t_vec = t_vec_local
        lookup = {e.cluster_id: e for e in entries}
        per_session_lookup[sess_name] = lookup

    if t_vec is None:
        print("No lick traces found; aborting")
        return

    # layout: extra column for mean
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

    label_font = 12
    tick_font = 10
    for ax in np.ravel(axes):
        ax.tick_params(labelsize=tick_font)

    for r, sess_name in enumerate(sessions):
        lookup = per_session_lookup.get(sess_name, {})
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
                # plot unit z-trace
                ax.plot(t_vec, e.z_trace, color="#2ca02c", lw=1)
                ax.fill_between(t_vec, e.z_trace - e.sem_trace, e.z_trace + e.sem_trace, color="#2ca02c", alpha=0.15)
                ax.axvline(0, color='k', linestyle='--', lw=0.6)
                if c == 0:
                    ax.set_ylabel(sess_name, fontsize=label_font, rotation=90, labelpad=10)
                if r == n_rows - 1:
                    ax.set_xlabel('time (s)', fontsize=label_font)
                # normalized trace
                norm = sign_normalize_trace(e.z_trace, t_vec, cfg)
                norm_stack.append(norm)
            elif c == n_cols - 1:
                # mean column
                if norm_stack:
                    stack = np.stack(norm_stack)
                    mean_trace = np.nanmean(stack, axis=0)
                    if stack.shape[0] > 1:
                        sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
                    else:
                        sem = np.zeros_like(mean_trace)
                    ax.plot(t_vec, mean_trace, color="#9467bd", lw=2)
                    ax.fill_between(t_vec, mean_trace - sem, mean_trace + sem, color="#9467bd", alpha=0.2)
                    ax.axvline(0, color='k', linestyle='--', lw=0.8)
                else:
                    ax.axis('off')
                if r == n_rows - 1:
                    ax.set_xlabel('time (s)', fontsize=label_font)
                if r == 0:
                    ax.set_title('mean (responsiveness)', fontsize=label_font)
            else:
                ax.axis('off')

    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.14, 0.5)
    fig.suptitle('Selected FA-lick clusters — summary', fontsize=14)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    out_png = out_root / 'selected_lick_clusters_summary.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # write means CSV
    out_csv = out_root / 'selected_lick_clusters_summary_means.csv'
    header = ['session'] + [f't_{i:.3f}' for i in t_vec]
    with out_csv.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for sess_name in sessions:
            lookup = per_session_lookup.get(sess_name, {})
            clu_list = mapping[sess_name]
            norm_stack = []
            for clu in clu_list:
                e = lookup.get(clu)
                if e is None:
                    continue
                norm = sign_normalize_trace(e.z_trace, t_vec, cfg)
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
