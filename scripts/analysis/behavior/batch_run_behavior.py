"""
Batch runner for behavior analysis pipeline.

Usage:
    python scripts/analysis/behavior/batch_run_behavior.py --pkl-dir pkls/BG_046 --out FIGURES/behavior/BG_046 --workers 4
"""
import argparse
import subprocess
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

def run_session_analysis(args):
    """
    Worker function to run analysis for a single session.
    args: (session_name, pkl_path, out_dir_base, repo_root)
    """
    session_name, pkl_path, out_dir_base, repo_root = args
    
    out_dir = out_dir_base / session_name
    script_path = repo_root / "scripts" / "analysis" / "behavior" / "run_behavior_pipeline.py"
    
    cmd = f"python {script_path} --session {session_name} --pkl {pkl_path} --out {out_dir}"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return session_name, False, result.stderr
        else:
            return session_name, True, result.stdout
    except Exception as e:
        return session_name, False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Batch runner for behavior analysis.")
    parser.add_argument('--pkl-dir', required=True, help='Directory containing session pkls')
    parser.add_argument('--out', required=True, help='Output directory for figures')
    parser.add_argument('--manifest', help='Optional: Path to manifest CSV to filter sessions (e.g. _clean.csv)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    pkl_dir = Path(args.pkl_dir)
    out_dir_base = Path(args.out)
    
    tasks = []

    # Filter by manifest if provided
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Error: Manifest not found: {manifest_path}")
            sys.exit(1)
            
        print(f"Loading sessions from manifest: {manifest_path.name}")
        df = pd.read_csv(manifest_path, dtype={'session_name': str})
        
        for _, row in df.iterrows():
            session_name = str(row['session_name'])
            
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
                tasks.append((session_name, pkl_path, out_dir_base, repo_root))
            else:
                print(f"Warning: Could not find pkl file for session {session_name}")

        print(f"  - Manifest entries: {len(df)}")
        print(f"  - Found pkl files: {len(tasks)}")
        
        if not tasks:
            print("No valid sessions found from manifest!")
            sys.exit(1)

    else:
        pkl_files = sorted(list(pkl_dir.glob("*.pkl")))
        
        if not pkl_files:
            print(f"No pkl files found in {pkl_dir}")
            return

        print(f"Found {len(pkl_files)} sessions. Starting batch processing...")
        
        for pkl in pkl_files:
            # Extract session name (e.g. BG_046_17092025)
            # If filename is BG_046_17092025.pkl -> BG_046_17092025
            session_name = pkl.stem.replace('.new', '') 
            tasks.append((session_name, pkl, out_dir_base, repo_root))
        
    failed_sessions = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_session_analysis, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Sessions"):
            session_name = futures[future]
            try:
                name, success, output = future.result()
                if not success:
                    failed_sessions.append(name)
                    # print(f"\n[-] {name} Failed: {output[:200]}...") # Optional verbose error
            except Exception as e:
                failed_sessions.append(session_name)
                print(f"\n[-] Exception for {session_name}: {e}")

    print("\nBatch processing complete.")
    if failed_sessions:
        print(f"Failed sessions ({len(failed_sessions)}): {failed_sessions}")
    else:
        print("All sessions processed successfully.")

    # Run cross-session analysis if manifest is present
    if args.manifest:
        print("\nRunning cross-session analysis...")
        cross_session_out = out_dir_base / "cross_session_summary"
        cross_session_script = repo_root / "scripts" / "analysis" / "behavior" / "plot_cross_session_behavior.py"
        
        cmd = f"python {cross_session_script} --manifest {args.manifest} --out {cross_session_out}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Cross-session analysis complete. Output: {cross_session_out}")
        except subprocess.CalledProcessError as e:
            print(f"Error running cross-session analysis: {e}")
    else:
        print("\nSkipping cross-session analysis (requires --manifest).")

if __name__ == "__main__":
    main()
