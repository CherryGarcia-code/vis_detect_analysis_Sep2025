"""
Batch runner for behavior analysis pipeline.

Usage:
    python scripts/analysis/behavior/batch_run_behavior.py --pkl-dir pkls/BG_046 --workers 4
"""
import argparse
import subprocess
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def run_session_analysis(args):
    """
    Worker function to run analysis for a single session.
    args: (session_name, pkl_path, repo_root)
    """
    session_name, pkl_path, repo_root = args
    
    out_dir = repo_root / "FIGURES" / "behavior" / session_name
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
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    pkl_dir = Path(args.pkl_dir)
    
    pkl_files = sorted(list(pkl_dir.glob("*.pkl")))
    # Filter out .new.pkl if they exist alongside originals to avoid duplicates, 
    # or just take all if migration is done.
    # Let's assume standard naming.
    
    if not pkl_files:
        print(f"No pkl files found in {pkl_dir}")
        return

    print(f"Found {len(pkl_files)} sessions. Starting batch processing...")
    
    tasks = []
    for pkl in pkl_files:
        # Extract session name (e.g. BG_046_17092025)
        # If filename is BG_046_17092025.pkl -> BG_046_17092025
        session_name = pkl.stem.replace('.new', '') 
        tasks.append((session_name, pkl, repo_root))
        
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

if __name__ == "__main__":
    main()
