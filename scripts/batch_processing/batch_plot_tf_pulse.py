"""Batch generate TF pulse grids for all sessions.

Usage:
    python scripts/batch_plot_tf_pulse.py --data-dir data --out-dir png_output/tf_pulse_grids --profile striatal_strict
"""
import argparse
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add repo root/src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / 'src'))

from visdetect.core.session import load_session
from visdetect.analysis.tf_pulse import TFRespPulseConfig, plot_tf_pulse_grid
from visdetect.core.qc import compute_unit_selection_table, apply_unit_filters, load_qc_profile

def process_session(pkl_path, out_dir, profile_name, which="both", sort=True):
    try:
        sess = load_session(str(pkl_path))
        ident = f"{getattr(sess,'subject','unknown')}_{getattr(sess,'session_name','unknown')}"
        
        # Unit selection logic: prefer good_and_stable_ids, then good_cluster_ids, else all clusters
        if getattr(sess, "good_and_stable_ids", None):
            cluster_id_list = sess.good_and_stable_ids
        elif getattr(sess, "good_cluster_ids", None):
            cluster_id_list = sess.good_cluster_ids
        else:
            cluster_id_list = [c.cluster_id for c in sess.clusters]
            
        # Apply QC profile
        final_ids = list(candidate_ids)
        if profile_name:
            profile = load_qc_profile(profile_name)
            if profile:
                qc_df = compute_unit_selection_table(sess)
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
                final_ids = [cid for cid in candidate_ids if cid in kept_qc_ids]

        cfg = TFRespPulseConfig(kept_only=False)
        out_png = out_dir / ident / f"tf_pulse_grid_{which}.png"
        out_csv = out_png.with_suffix(".csv")
        
        plot_tf_pulse_grid(
            sess, 
            str(out_png), 
            cfg=cfg, 
            selection_csv=None, 
            n_cols=10, 
            which=which,
            filter_ids=final_ids,
            sort_by_strength=sort,
            show_progress=False, # Disable inner progress bar
            save_csv_path=str(out_csv)
        )
        return True
    except Exception as e:
        logging.error(f"Failed to process {pkl_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch TF pulse grid generator")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing .pkl files")
    parser.add_argument("--out-dir", type=Path, default=Path("png_output/tf_pulse_grids"), help="Output directory")
    parser.add_argument("--profile", default="striatal_strict", help="QC profile name")
    parser.add_argument("--which", default="both", choices=["fast", "slow", "both"])
    parser.add_argument("--no-sort", action="store_false", dest="sort", help="Disable sorting")
    parser.add_argument("--pattern", default="*.pkl", help="Glob pattern for filtering files (default: *.pkl)")
    parser.set_defaults(sort=True)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    pkl_files = sorted(args.data_dir.glob(args.pattern))
    if not pkl_files:
        logging.error(f"No files found matching '{args.pattern}' in {args.data_dir}")
        return
        
    logging.info(f"Found {len(pkl_files)} sessions matching '{args.pattern}'. Output: {args.out_dir}")
    
    success = 0
    for pkl in tqdm(pkl_files, desc="Generating grids"):
        if process_session(pkl, args.out_dir, args.profile, args.which, args.sort):
            success += 1
            
    logging.info(f"Completed {success}/{len(pkl_files)} sessions.")

if __name__ == "__main__":
    main()
