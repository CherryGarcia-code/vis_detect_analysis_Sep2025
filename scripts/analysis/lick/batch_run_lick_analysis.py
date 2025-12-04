"""
Batch runner for lick analysis pipeline.

Example usage (for BG_031 sessions in the data directory):
    python scripts/analysis/lick/batch_run_lick_analysis.py --pkl-dir pkls/BG_031 --out FIGURES/lick/BG_031 --workers 4

This will process all .pkl files in pkls/BG_031 and output figures to FIGURES/lick/BG_031.
"""
import subprocess
from pathlib import Path
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

def run_session_analysis(args):
    """
    Worker function to run analysis for a single session.
    args: (session_name, pkl_path, out_dir_base, repo_root)
    """
    session_name, pkl_path, out_dir_base, repo_root = args
    
    # Create specific output directory for this session
    out_dir = out_dir_base / session_name
    
    script_path = repo_root / "scripts" / "analysis" / "lick" / "run_lick_analysis_pipeline.py"
    
    cmd = f"python {script_path} --session {session_name} --pkl {pkl_path} --out {out_dir}"
    
    # Capture output to avoid interleaved printing in parallel mode
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return session_name, False, result.stderr
        else:
            return session_name, True, result.stdout
    except Exception as e:
        return session_name, False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Batch runner for lick analysis pipeline.")
    parser.add_argument('--pkl-dir', required=True, help='Directory containing session pkls')
    parser.add_argument('--out', required=True, help='Output directory for figures')
    parser.add_argument('--manifest', help='Optional: Path to manifest CSV to filter sessions (e.g. _clean.csv)')
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    # Define paths
    repo_root = Path(__file__).resolve().parents[3]
    pkl_dir = Path(args.pkl_dir)
    out_dir_base = Path(args.out)
    
    # Filter by manifest if provided
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Error: Manifest not found: {manifest_path}")
            sys.exit(1)
            
        print(f"Loading sessions from manifest: {manifest_path.name}")
        df = pd.read_csv(manifest_path, dtype={'session_name': str})
        
        tasks = []
        for _, row in df.iterrows():
            session_name = str(row['session_name'])
            if session_name.isdigit() and len(session_name) == 7:
                session_name = session_name.zfill(8)
            
            # Try to find the pkl file
            # 1. Use pkl_path from manifest if it exists
            if 'pkl_path' in row and pd.notna(row['pkl_path']):
                pkl_path = Path(row['pkl_path'])
                if not pkl_path.exists():
                    # Try relative to pkl_dir if absolute path fails (e.g. different machine)
                    pkl_path = pkl_dir / pkl_path.name
            else:
                # 2. Construct from session_name
                # Try exact match or with subject prefix
                candidates = list(pkl_dir.glob(f"*{session_name}*.pkl"))
                if candidates:
                    pkl_path = candidates[0] # Take first match
                else:
                    pkl_path = None
            
            if pkl_path and pkl_path.exists():
                # Use the stem as session name for output consistency, or the manifest name?
                # Let's use the manifest name to match the clean list
                tasks.append((session_name, pkl_path, out_dir_base, repo_root))
            else:
                print(f"Warning: Could not find pkl file for session {session_name}")

        print(f"  - Manifest entries: {len(df)}")
        print(f"  - Found pkl files: {len(tasks)}")
        
        if not tasks:
            print("No valid sessions found from manifest!")
            sys.exit(1)

    else:
        # Legacy mode: Scan directory
        if not pkl_dir.exists():
            print(f"Error: Pickle directory not found: {pkl_dir}")
            sys.exit(1)

        # Find all pkl files
        pkl_files = sorted(list(pkl_dir.glob("*.pkl")))
        
        if not pkl_files:
            print(f"No pkl files found in {pkl_dir}")
            sys.exit(1)
            
        print(f"Found {len(pkl_files)} sessions to process in {pkl_dir}.")
        
        tasks = []
        for pkl_path in pkl_files:
            # session_name usually matches filename stem
            session_name = pkl_path.stem.replace('.new', '')
            tasks.append((session_name, pkl_path, out_dir_base, repo_root))

    print(f"Outputting to {out_dir_base}")
    print(f"Running with {args.workers} workers.")
    
    failed_sessions = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_session = {executor.submit(run_session_analysis, task): task[0] for task in tasks}
        
        for future in as_completed(future_to_session):
            session_name = future_to_session[future]
            try:
                name, success, output = future.result()
                if not success:
                    failed_sessions.append(name)
                    print(f"[-] Session {name} FAILED.")
                    # print(output) # Uncomment to see error details
                else:
                    print(f"[+] Session {name} COMPLETED.")
            except Exception as exc:
                print(f"[-] Session {session_name} generated an exception: {exc}")
                failed_sessions.append(session_name)
            
    print("\nBatch processing complete.")
    if failed_sessions:
        print(f"The following sessions failed: {failed_sessions}")
    else:
        print("All sessions processed successfully.")

if __name__ == "__main__":
    main()
