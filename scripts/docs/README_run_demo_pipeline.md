# run_demo_pipeline.py

Small helper and README for the demo single-session pipeline runner.

Usage examples (from repository root):

# Run quickly (fewer permutations)
python scripts/run_demo_pipeline.py --mode quick --overwrite

# Run full (more permutations)
python scripts/run_demo_pipeline.py --mode full --n-permutations 500

# Specify a different session pickle and output directory
python scripts/run_demo_pipeline.py --session-pkl data/BG_031_260325.pkl --out-dir notebooks/my_outputs --overwrite

# Verbose logging
python scripts/run_demo_pipeline.py --verbose

# If you prefer to run inside the conda environment without activating it manually (Windows example):
C:/Users/Ben/anaconda3/Scripts/conda.exe run -p "C:/Users/Ben/anaconda3" --no-capture-output python "e:/python_analysis/git_repos/vis_detect_analysis_Sep2025/scripts/run_demo_pipeline.py" --mode quick --overwrite

Notes:
- The script uses matplotlib Agg backend and writes PNG/CSV outputs to the `--out-dir` (default: `notebooks/`).
- Use `--overwrite` to replace existing outputs.
- The `quick` mode reduces the number of permutations for a faster run suitable for demos.
