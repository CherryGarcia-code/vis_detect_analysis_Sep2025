"""Batch convert all BG_046 .mat files to .pkl format.

This script processes all BG_046 session .mat files in the data/ directory,
converts them to normalized Session pickles, and logs progress/errors.

Usage:
    python scripts/batch_convert_bg046.py [--data-dir data/] [--force]
"""

import argparse
import logging
from pathlib import Path
import sys
from tqdm import tqdm

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / 'src') not in sys.path:
    sys.path.insert(0, str(repo_root / 'src'))

from visdetect.core.io import load_mat_file_to_session
from visdetect.core.session import save_session


def setup_logging(verbose: bool = False):
    """Configure logging with timestamps and appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def convert_single_session(mat_path: Path, pkl_path: Path, force: bool = False) -> bool:
    """Convert a single .mat file to .pkl.
    
    Args:
        mat_path: Path to input .mat file
        pkl_path: Path to output .pkl file
        force: If True, overwrite existing .pkl files
        
    Returns:
        True if conversion successful, False otherwise
    """
    if pkl_path.exists() and not force:
        logging.info(f"SKIP (exists): {pkl_path.name}")
        return True
    
    try:
        logging.info(f"Converting: {mat_path.name} -> {pkl_path.name}")
        session = load_mat_file_to_session(str(mat_path))
        
        # Log summary
        n_trials = len(session.trials) if session.trials else 0
        n_clusters = len(session.clusters) if session.clusters else 0
        # Prefer good_and_stable_ids, then good_cluster_ids, else all clusters
        if getattr(session, "good_and_stable_ids", None):
            cluster_id_list = session.good_and_stable_ids
        elif getattr(session, "good_cluster_ids", None):
            cluster_id_list = session.good_cluster_ids
        else:
            cluster_id_list = [c.cluster_id for c in session.clusters]
        n_good = len(cluster_id_list)
        
        logging.info(f"  Loaded: {session.subject}/{session.session_name}")
        logging.info(f"  Trials: {n_trials}, Clusters: {n_clusters}, Good: {n_good}")
        
        # Save pickle
        save_session(session, str(pkl_path))
        logging.info(f"  SUCCESS: Saved {pkl_path.name}")
        return True
        
    except Exception as e:
        logging.error(f"  FAILED: {mat_path.name}")
        logging.error(f"  Error: {str(e)}", exc_info=True)
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch convert BG_046 .mat files to .pkl format"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=repo_root / 'data',
        help='Directory containing .mat files (default: data/)'
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=repo_root / 'data',
        help='Directory to save .pkl files (default: data/)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing .pkl files'
    )
    parser.add_argument(
        '--subject',
        type=str,
        default='BG_046',
        help='Subject ID to filter files (default: BG_046)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files that would be converted without converting'
    )
    
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    
    # Find all matching .mat files
    pattern = f"{args.subject}_*.mat"
    mat_files = sorted(args.data_dir.glob(pattern))
    
    if not mat_files:
        logging.error(f"No .mat files found matching pattern: {pattern}")
        logging.error(f"Searched in: {args.data_dir.resolve()}")
        return 1
    
    logging.info(f"Found {len(mat_files)} .mat files for {args.subject}")
    logging.info(f"Data directory: {args.data_dir.resolve()}")
    
    if args.dry_run:
        logging.info("\n=== DRY RUN MODE ===")
        for mat_file in mat_files:
            pkl_file = mat_file.with_suffix('.pkl')
            status = "EXISTS" if pkl_file.exists() else "NEEDS CONVERSION"
            logging.info(f"  {status}: {mat_file.name}")
        logging.info(f"\nTotal files: {len(mat_files)}")
        return 0
    
    # Process each file
    logging.info("\n=== STARTING CONVERSION ===\n")
    success_count = 0
    failed_files = []
    
    for mat_file in tqdm(mat_files, desc="Converting sessions", unit="file"):
        pkl_file = mat_file.with_suffix('.pkl')
        # logging.info(f"Processing {mat_file.name}") # Reduced verbosity for tqdm
        
        if convert_single_session(mat_file, pkl_file, args.force):
            success_count += 1
        else:
            failed_files.append(mat_file.name)
        
        # logging.info("")  # Blank line between files
    
    # Summary
    logging.info("=== CONVERSION SUMMARY ===")
    logging.info(f"Total files: {len(mat_files)}")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {len(failed_files)}")
    
    if failed_files:
        logging.error("\nFailed files:")
        for fname in failed_files:
            logging.error(f"  - {fname}")
        return 1
    
    logging.info("\n✓ All conversions completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
