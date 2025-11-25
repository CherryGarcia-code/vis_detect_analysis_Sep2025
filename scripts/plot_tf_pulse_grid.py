"""Plot a grid of TF pulse per-unit z-scored mean traces for visual inspection.

Usage:
  python scripts/plot_tf_pulse_grid.py --file data/BG_031_260325.pkl --out png_output/tf_pulse_grids --which slow --cols 10
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visdetect.core.legacy_io import load_session
from visdetect.analysis.tf_pulse import TFRespPulseConfig, plot_tf_pulse_grid
from visdetect.core.qc import compute_unit_selection_table, apply_unit_filters, load_qc_profile


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
    args = ap.parse_args(argv)

    sess = load_session(args.file)
    ident = f"{getattr(sess,'subject','unknown')}_{getattr(sess,'session_name','unknown')}"
    
    # Unit selection logic
    # 1. Start with good_and_stable_ids if available, else good_cluster_ids
    candidate_ids = sess.good_and_stable_ids
    if candidate_ids is None:
        print(f"[WARN] good_and_stable_ids not found in {args.file}. Falling back to good_cluster_ids.")
        candidate_ids = sess.good_cluster_ids
    
    if candidate_ids is None:
        # Fallback to all clusters if no good list
        candidate_ids = [c.cluster_id for c in sess.clusters]
        
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
    out_png = Path(args.out) / ident / f"tf_pulse_grid_{args.which}.png"
    out_csv = out_png.with_suffix(".csv")
    
    print(f"Collecting traces for {len(final_ids)} units...")
    
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
        save_csv_path=str(out_csv)
    )
    print(f"Wrote grid: {p}")
    print(f"Wrote data: {out_csv}")


if __name__ == "__main__":
    raise SystemExit(main())
