"""Plot selected TF-pulse clusters across sessions into a single summary PNG.

Usage (example):
    python scripts/analysis/plot_selected_tf_clusters.py --out FIGURES

The script looks for session pickles in `pkls/{SESSION}.pkl` (falling back to
`data/{SESSION}.pkl`) and attempts to reuse `table_output/tf_pulse/{IDENT}/tf_pulse_traces.npz`
if present to avoid recomputation.
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import csv

# repo root on path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.visdetect.core.session import load_session
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
    # prefer pkls/, then data/
    p1 = REPO / "pkls" / f"{session_name}.pkl"
    if p1.exists():
        return p1
    p2 = REPO / "data" / f"{session_name}.pkl"
    if p2.exists():
        return p2
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot selected TF clusters across sessions")
    ap.add_argument("--out", default="FIGURES", help="Output folder for summary PNG and CSV")
    ap.add_argument("--which", choices=["fast", "slow", "both"], default="both")
    ap.add_argument("--map-file", default=None, help="Optional CSV file with session,cluster_id rows to override embedded map")
    ap.add_argument("--no-cache", action="store_true", help="Ignore existing tf_pulse_traces.npz caches and force recompute")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load mapping from file if provided
    mapping = DEFAULT_MAP.copy()
    if args.map_file:
        mf = Path(args.map_file)
        if mf.exists():
            mapping = {}
            with mf.open("r", newline="") as fh:
                rdr = csv.reader(fh)
                for row in rdr:
                    if not row:
                        continue
                    sess = row[0].strip()
                    ids = [int(x) for x in row[1:] if x.strip()]
                    mapping[sess] = ids

    sessions = list(mapping.keys())
    # ensure earliest first as requested (DEFAULT_MAP already ordered that way)

    cfg = TFRespPulseConfig()

    # collect traces per session and store lookups
    t_vec = None
    per_session_entries = {}
    summary_rows = []
    max_cols = max(len(v) for v in mapping.values())

    for sess_name in sessions:
        print(f"Processing session: {sess_name}")
        pkl = find_session_pickle(sess_name)
        if pkl is None:
            print(f"  WARNING: session pickle not found for {sess_name}; skipping")
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

        lookup = {e.cluster_id: e for e in entries}
        per_session_entries[sess_name] = lookup

        # build summary rows for CSV
        for clu in mapping[sess_name]:
            e = lookup.get(clu)
            if e is None:
                summary_rows.append({"session": sess_name, "cluster_id": int(clu), "found": False})
            else:
                summary_rows.append({
                    "session": sess_name,
                    "cluster_id": int(clu),
                    "found": True,
                    "z_max_fast": float(e.z_max_fast) if np.isfinite(e.z_max_fast) else "",
                    "z_min_fast": float(e.z_min_fast) if np.isfinite(e.z_min_fast) else "",
                    "z_max_slow": float(e.z_max_slow) if np.isfinite(e.z_max_slow) else "",
                    "z_min_slow": float(e.z_min_slow) if np.isfinite(e.z_min_slow) else "",
                })

    # Plot layout: rows = sessions, cols = max_cols
    n_rows = len(sessions)
    n_cols = max_cols
    fig_h = 1.8 * n_rows + 1.2
    fig_w = 2.4 * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for r, sess_name in enumerate(sessions):
        lookup = per_session_entries.get(sess_name, {})
        clu_list = mapping[sess_name]
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(clu_list):
                ax.axis("off")
                continue
            clu = clu_list[c]
            e = lookup.get(clu)
            if e is None or t_vec is None:
                ax.text(0.5, 0.5, f"{sess_name}\nclu {clu}\n(not found)", ha="center", va="center")
                ax.set_axis_off()
                continue
            if args.which in ("slow", "both"):
                ax.plot(t_vec, e.slow_z, color="#d62728")
                ax.fill_between(t_vec, e.slow_z - e.slow_z_sem, e.slow_z + e.slow_z_sem, color="#d62728", alpha=0.25)
            if args.which in ("fast", "both"):
                ax.plot(t_vec, e.fast_z, color="#1f77b4")
                ax.fill_between(t_vec, e.fast_z - e.fast_z_sem, e.fast_z + e.fast_z_sem, color="#1f77b4", alpha=0.25)
            ax.axvline(0, color="k", linestyle="--", lw=0.8)
            ax.set_title(f"{sess_name}\nclu {clu}", fontsize=8)
            if c == 0:
                ax.set_ylabel("z-score")
            if r == n_rows - 1:
                ax.set_xlabel("time (s)")

    fig.suptitle("Selected TF clusters (rows=sessions, cols=clusters)", fontsize=12)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    out_png = out_root / "selected_tf_clusters.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # write CSV summary
    out_csv = out_root / "selected_tf_clusters_summary.csv"
    keys = ["session", "cluster_id", "found", "z_max_fast", "z_min_fast", "z_max_slow", "z_min_slow"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in summary_rows:
            # coerce missing fields to empty strings
            nr = {k: row.get(k, "") for k in keys}
            w.writerow(nr)

    print(f"Wrote summary PNG: {out_png}")
    print(f"Wrote summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
