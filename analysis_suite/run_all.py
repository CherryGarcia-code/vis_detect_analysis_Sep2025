#!/usr/bin/env python3
"""Run all 29 analysis scripts in sequence, with timing and error tracking.

Usage:
  python run_all.py                  # sequential (default)
  python run_all.py --n_workers 4   # pass parallelism flag to heavy scripts

The --n_workers flag is forwarded only to the scripts that support it:
  03_population/a_coding_direction.py
  04_decoding/a_hit_miss_decoding.py
  07_advanced/a_glm_encoding.py
All other scripts are unaffected and run sequentially as normal.
"""
import subprocess, sys, os, time, argparse

SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SUITE_DIR)
PYTHON = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")

SCRIPTS = [
    ("01_behavior/a_learning_curve.py",            "Fig01 Learning Curve"),
    ("01_behavior/b_hmm_state_dynamics.py",        "Fig02 HMM State Dynamics"),
    ("01_behavior/c_reaction_time_analysis.py",    "Fig03 Reaction Times"),
    ("02_single_unit/a_responsiveness_screen.py",  "Fig04 Responsiveness Screen"),
    ("02_single_unit/b_outcome_selectivity.py",    "Fig05 Outcome Selectivity"),
    ("02_single_unit/c_change_size_tuning.py",     "Fig06 Change-Size Tuning"),
    ("02_single_unit/d_state_modulation.py",       "Fig07 State Modulation"),
    ("02_single_unit/e_cell_type_comparison.py",   "Fig08 Cell-Type Comparison"),
    ("03_population/a_coding_direction.py",        "Fig09 Coding Direction"),
    ("03_population/b_population_psth_heatmap.py", "Fig10 Population Heatmap"),
    ("03_population/c_dimensionality_reduction.py","Fig11 PCA Dimensionality"),
    ("04_decoding/a_hit_miss_decoding.py",         "Fig12 Hit/Miss Decoding"),
    ("04_decoding/b_change_size_decoding.py",      "Fig13 Change-Size Decoding"),
    ("04_decoding/c_state_decoding.py",            "Fig14 State Decoding"),
    ("05_longitudinal/a_neural_learning_curves.py","Fig15 Neural Learning"),
    ("05_longitudinal/b_celltype_learning.py",     "Fig16 Cell-Type Learning"),
    ("05_longitudinal/c_population_geometry_shift.py","Fig17 Geometry Shift"),
    ("06_lick_motor/a_fa_neural_signatures.py",    "Fig18 FA Neural Signatures"),
    ("06_lick_motor/b_pre_lick_ramping.py",        "Fig19 Pre-Lick Ramping"),
    ("06_lick_motor/c_motor_vs_sensory.py",        "Fig20 Motor vs Sensory"),
    ("07_advanced/a_glm_encoding.py",              "Fig21 GLM Encoding"),
    ("07_advanced/b_dpca.py",                      "Fig22 dPCA"),
    ("07_advanced/c_noise_correlations.py",        "Fig23 Noise Correlations"),
    ("08_tf_pulse/a_tf_responsiveness.py",         "Fig24 TF Responsiveness"),
    ("08_tf_pulse/b_tf_response_properties.py",    "Fig25 TF Response Properties"),
    ("08_tf_pulse/c_tf_pulse_integration.py",      "Fig26 Two-Pulse Integration"),
    ("08_tf_pulse/d_tf_learning_emergence.py",     "Fig27 TF Learning Emergence"),
    ("08_tf_pulse/e_tf_state_modulation.py",       "Fig28 TF × HMM State"),
    ("08_tf_pulse/f_tf_sensory_motor.py",          "Fig29 TF Sensory-Motor"),
]

# Scripts that accept --n_workers (session-level parallelism)
PARALLEL_SCRIPTS = {
    "03_population/a_coding_direction.py",
    "04_decoding/a_hit_miss_decoding.py",
    "07_advanced/a_glm_encoding.py",
}


def main():
    parser = argparse.ArgumentParser(description="Run all 29 analysis scripts.")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Worker processes forwarded to heavy scripts "
                             "(03a coding direction, 04a hit/miss decoding, "
                             "07a GLM encoding). Default: 1 (sequential).")
    args = parser.parse_args()

    results = []
    t_total = time.time()
    log_path = os.path.join(SUITE_DIR, "run_all_log.txt")
    log = open(log_path, "w", buffering=1)

    def out(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    if args.n_workers > 1:
        out(f"n_workers={args.n_workers} (forwarded to parallelism-aware scripts)")

    for i, (script, label) in enumerate(SCRIPTS, 1):
        script_path = os.path.join(SUITE_DIR, script)
        out(f"\n{'='*70}")
        out(f"[{i:2d}/{len(SCRIPTS)}] {label}  ({script})")
        out(f"{'='*70}")

        # Build command — forward --n_workers only to scripts that support it
        cmd = [PYTHON, "-u", script_path]
        if args.n_workers > 1 and script in PARALLEL_SCRIPTS:
            cmd += ["--n_workers", str(args.n_workers)]

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=SUITE_DIR,
                capture_output=True,
                timeout=1800,
            )
            elapsed = time.time() - t0
            status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
            # Print last few lines of output
            stdout_tail = result.stdout.decode("utf-8", errors="replace").strip().split("\n")[-5:]
            for line in stdout_tail:
                out(f"  {line}")
            if result.returncode != 0:
                stderr_tail = result.stderr.decode("utf-8", errors="replace").strip().split("\n")[-10:]
                for line in stderr_tail:
                    out(f"  ERR: {line}")
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            status = "TIMEOUT"
        except Exception as e:
            elapsed = time.time() - t0
            status = f"ERROR: {e}"

        results.append((label, status, elapsed))
        out(f"  >> {status} ({elapsed:.1f}s)")

    total_time = time.time() - t_total
    out(f"\n\n{'='*70}")
    out(f"SUMMARY  (total: {total_time/60:.1f} min)")
    out(f"{'='*70}")
    for label, status, elapsed in results:
        mark = "OK" if status == "OK" else "XX"
        out(f"  [{mark}] {label:35s}  {elapsed:6.1f}s  {status}")

    n_ok = sum(1 for _, s, _ in results if s == "OK")
    out(f"\n  {n_ok}/{len(SCRIPTS)} completed successfully.")
    log.close()

if __name__ == "__main__":
    main()