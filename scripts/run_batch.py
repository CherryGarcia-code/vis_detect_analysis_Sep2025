"""Simple batch runner to execute analyses across multiple session PKLs.

This script will locate PKL files under `data/` and run a subset of analyses
(such as the demo runner, optotagging detection, responsivity summary) and
write per-session outputs under `notebooks/batch_outputs/`.

It's intentionally small - extend as needed.
"""
from pathlib import Path
import logging
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / 'data'
OUT_DIR = REPO / 'notebooks' / 'batch_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def find_session_pkls(data_dir: Path):
    return sorted(data_dir.glob('*.pkl'))


def run_demo_for_pkl(pkl: Path, out_dir: Path, use_good_only: bool = True, overwrite: bool = False):
    # Use the `visdetect` conda environment (created earlier)
    cmd = [
        'C:/Users/Ben/anaconda3/Scripts/conda.exe', 'run', '-n', 'visdetect', '--no-capture-output',
        sys.executable if False else 'python', str(REPO / 'scripts' / 'run_demo_pipeline.py'), '--session-pkl', str(pkl), '--out-dir', str(out_dir), '--mode', 'quick'
    ]
    if use_good_only:
        cmd.append('--use-good-only')
    else:
        cmd.append('--no-use-good-only')
    if overwrite:
        cmd.append('--overwrite')
    logging.info('Running demo for %s (use_good_only=%s, overwrite=%s)', pkl, use_good_only, overwrite)
    subprocess.run(cmd, check=False)


if __name__ == '__main__':
    pkls = find_session_pkls(DATA_DIR)
    # Default behavior: prefer good clusters and do not overwrite unless requested
    use_good_only = True
    overwrite = True
    for p in pkls:
        out = OUT_DIR / p.stem
        out.mkdir(exist_ok=True)
        run_demo_for_pkl(p, out, use_good_only=use_good_only, overwrite=overwrite)
    logging.info('Batch run complete')
