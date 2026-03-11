"""Plot a grid of TF pulse per-unit z-scored mean traces for visual inspection.

Usage:
    python scripts/analysis/plot_tf_pulse_grid.py --file data/BG_046_17092025.pkl --out png_output/tf_pulse_grids --which both --cols 5
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Ensure repo root/src on path
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / 'src'))

from visdetect.core.session import load_session
from visdetect.analysis.tf_pulse import TFRespPulseConfig, plot_tf_pulse_grid
from visdetect.core.qc import compute_unit_selection_table, apply_unit_filters, load_qc_profile
import pandas as pd
import visdetect.analysis.tf_pulse as tf_pulse_mod


def main(argv=None):
    ap = argparse.ArgumentParser(description="TF pulse z-trace grid per session")
    ap.add_argument("--file", required=True, help="Path to session pickle file")
    ap.add_argument("--out", default="png_output/tf_pulse_grids", help="Output root for grid image")
    ap.add_argument("--which", choices=["fast", "slow", "both"], default="slow", help="Which traces to plot")
    ap.add_argument("--cols", type=int, default=10, help="Number of columns in the grid")
    ap.add_argument("--kept-only", action="store_true", help="Plot only kept/good clusters (legacy flag)")
    ap.add_argument("--profile", default="striatal_strict", help="QC profile name (e.g. striatal_strict)")
    ap.add_argument("--sort", action="store_true", default=True, help="Sort by responsiveness strength")
    ap.add_argument("--no-sort", action="store_false", dest="sort", help="Do not sort")
    ap.add_argument("--parallel", action="store_true", default=True, help="Compute traces in parallel")
    ap.add_argument("--workers", type=int, default=None, help="Number of worker processes for parallel compute")
    ap.add_argument("--cluster-ids-file", default=None, help="Optional file with cluster IDs to plot (one per line)")
    args = ap.parse_args(argv)

    sess = load_session(args.file)
    # ident = f"{getattr(sess,'subject','unknown')}_{getattr(sess,'session_name','unknown')}"
    
    # Unit selection logic: prefer good_and_stable_ids, then good_cluster_ids, else all clusters
    if getattr(sess, "good_and_stable_ids", None):
        cluster_id_list = sess.good_and_stable_ids
    elif getattr(sess, "good_cluster_ids", None):
        cluster_id_list = sess.good_cluster_ids
    else:
        cluster_id_list = [c.cluster_id for c in sess.clusters]

    candidate_ids = cluster_id_list  # <-- define candidate_ids for filtering
    # If a cluster-ids-file is provided, restrict to those IDs
    if args.cluster_ids_file:
        with open(args.cluster_ids_file, 'r') as f:
            file_ids = set(int(line.strip()) for line in f if line.strip())
        candidate_ids = [cid for cid in candidate_ids if int(cid) in file_ids]
    # 2. Apply QC profile if requested
    final_ids = list(candidate_ids)
    if args.profile:
        profile = load_qc_profile(args.profile)
        if profile:
            print(f"Applying QC profile: {args.profile}")
            # Compute metrics for ALL clusters
            qc_df = compute_unit_selection_table(sess)
            
            # Apply filters
            # We disable require_good_cluster in apply_unit_filters because we handle the base set manually
            filt_df = apply_unit_filters(
                qc_df,
                require_good_cluster=False,
                min_total_spikes=profile.get("min_total_spikes", 0),
                min_mean_rate_hz=profile.get("min_mean_rate_hz", 0.0),
                max_isi_viol_frac=profile.get("max_isi_viol_frac", 1.0),
                min_median_spikes_per_trial=profile.get("min_median_spikes_per_trial", 0.0),
                max_median_spikes_per_trial=profile.get("max_median_spikes_per_trial", None),
            )
            
            kept_qc_ids = set(filt_df.loc[filt_df["keep"], "cluster_id"].astype(int))
            
            # Intersect with candidate_set (preserving order if possible, though sort will override)
            final_ids = [cid for cid in candidate_ids if cid in kept_qc_ids]
            print(f"Units kept after QC: {len(final_ids)} / {len(candidate_ids)}")
        else:
            print(f"[WARN] Profile {args.profile} not found. Using raw candidate list.")

    cfg = TFRespPulseConfig(kept_only=False) # We handle filtering manually via filter_ids
    out_png = Path(args.out) / f"tf_pulse_grid_{args.which}.png"
    out_csv = out_png.with_suffix(".csv")
    # Ensure output directory exists (avoid matplotlib OSError when saving)
    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort: ignore errors and let downstream code attempt to create
        pass
    print(f"Collecting traces for {len(final_ids)} units...")
    # Try to reuse precomputed tf pulse times and traces if available to avoid recomputing
    times_csv = Path("table_output") / "tf_pulse"  / "tf_pulse_times.csv"
    cache_npz = Path("table_output") / "tf_pulse" / "tf_pulse_traces.npz"
    try:
        # Prefer using explicit times CSV if present to keep provenance explicit
        if times_csv.exists():
            df_times = pd.read_csv(times_csv)
            fast = df_times.get("fast_times")
            slow = df_times.get("slow_times")
            if fast is not None and slow is not None:
                fast_times = fast.dropna().astype(float).to_numpy()
                slow_times = slow.dropna().astype(float).to_numpy()
                # Compute/load traces with cache support (use parallel if requested)
                t_vec, entries = tf_pulse_mod.collect_tf_pulse_traces(
                    sess,
                    cfg=cfg,
                    selection_csv=None,
                    fast_times=fast_times,
                    slow_times=slow_times,
                    cache_path=str(cache_npz),
                    show_progress=True,
                    parallel=args.parallel,
                    n_workers=args.workers,
                )
                # Monkeypatch the module collector so plot_tf_pulse_grid will reuse these entries
                tf_pulse_mod.collect_tf_pulse_traces = lambda *a, **k: (t_vec, entries)
                p = plot_tf_pulse_grid(
                    sess,
                    str(out_png),
                    cfg=cfg,
                    selection_csv=None,
                    n_cols=args.cols,
                    which=args.which,
                    filter_ids=final_ids,
                    sort_by_strength=args.sort,
                    show_progress=True,
                    save_csv_path=str(out_csv),
                )
            else:
                # Fall back to using any existing cache
                if cache_npz.exists():
                    t_vec, entries = tf_pulse_mod.collect_tf_pulse_traces(
                        sess, cfg=cfg, selection_csv=None, cache_path=str(cache_npz), show_progress=True, parallel=args.parallel, n_workers=args.workers
                    )
                    tf_pulse_mod.collect_tf_pulse_traces = lambda *a, **k: (t_vec, entries)
                p = plot_tf_pulse_grid(
                    sess,
                    str(out_png),
                    cfg=cfg,
                    selection_csv=None,
                    n_cols=args.cols,
                    which=args.which,
                    filter_ids=final_ids,
                    sort_by_strength=args.sort,
                    show_progress=True,
                    save_csv_path=str(out_csv),
                )
        else:
            # No explicit times CSV; try to use an existing cache, otherwise compute and cache
            if cache_npz.exists():
                t_vec, entries = tf_pulse_mod.collect_tf_pulse_traces(
                    sess, cfg=cfg, selection_csv=None, cache_path=str(cache_npz), show_progress=True, parallel=args.parallel, n_workers=args.workers
                )
                tf_pulse_mod.collect_tf_pulse_traces = lambda *a, **k: (t_vec, entries)
            p = plot_tf_pulse_grid(
                sess,
                str(out_png),
                cfg=cfg,
                selection_csv=None,
                n_cols=args.cols,
                which=args.which,
                filter_ids=final_ids,
                sort_by_strength=args.sort,
                show_progress=True,
                save_csv_path=str(out_csv),
            )
    except Exception as e:
        print(f"[WARN] failed to reuse precomputed tf_pulse_times ({e}), falling back to compute", file=sys.stderr)
        p = plot_tf_pulse_grid(
            sess,
            str(out_png),
            cfg=cfg,
            selection_csv=None,
            n_cols=args.cols,
            which=args.which,
            filter_ids=final_ids,
            sort_by_strength=args.sort,
            show_progress=True,
            save_csv_path=str(out_csv),
        )
    print(f"Wrote grid: {p}")
    print(f"Wrote data: {out_csv}")


if __name__ == "__main__":
    raise SystemExit(main())
