"""
Run behavior analysis pipeline for a single session.

Usage:
    python scripts/analysis/behavior/run_behavior_pipeline.py --session BG_046_17092025 --pkl pkls/BG_046/BG_046_17092025.pkl --out FIGURES/behavior/BG_046_17092025
"""
import argparse
import subprocess
from pathlib import Path
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run behavior analysis pipeline.")
    parser.add_argument('--session', required=True, help='Session name')
    parser.add_argument('--pkl', required=True, help='Path to session pkl')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    # Define scripts to run
    # Currently just one main plotting script, but structure allows adding more
    scripts = [
        ("plot_session_behavior.py", f"--session {args.session} --pkl {args.pkl} --out {args.out}")
    ]
    
    repo_root = Path(__file__).resolve().parents[3]
    
    for script_name, script_args in scripts:
        script_path = repo_root / "scripts" / "analysis" / "behavior" / script_name
        cmd = f"python {script_path} {script_args}"
        run_command(cmd)

if __name__ == "__main__":
    main()
