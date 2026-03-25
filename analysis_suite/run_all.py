#!/usr/bin/env python3
"""Run all analysis scripts in sequence, with timing and error tracking.

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
    # ── 01_behavior (Figs 01-07) ──────────────────────────────────────
    ("01_behavior/a_learning_curve.py",            "Fig01 Learning Curve"),
    ("01_behavior/b_hmm_state_dynamics.py",        "Fig02 HMM State Dynamics"),
    ("01_behavior/c_reaction_time_analysis.py",    "Fig03 Reaction Times"),
    ("01_behavior/d_post_error_psychometric.py",   "Fig04 Post-Error Psychometric"),
    ("01_behavior/e_post_error_dynamics.py",       "Fig05 Post-Error Dynamics"),
    ("01_behavior/f_post_error_controls.py",       "Fig06 Post-Error Controls"),
    ("01_behavior/g_post_error_streak_controls.py","Fig07 Post-Error Streak Controls"),
    # ── 02_single_unit (Figs 08-12) ───────────────────────────────────
    ("02_single_unit/a_responsiveness_screen.py",  "Fig08 Responsiveness Screen"),
    ("02_single_unit/b_outcome_selectivity.py",    "Fig09 Outcome Selectivity"),
    ("02_single_unit/c_change_size_tuning.py",     "Fig10 Change-Size Tuning"),
    ("02_single_unit/d_state_modulation.py",       "Fig11 State Modulation"),
    ("02_single_unit/e_cell_type_comparison.py",   "Fig12 Cell-Type Comparison"),
    # ── 03_population (Figs 13-17) ────────────────────────────────────
    ("03_population/a_coding_direction.py",        "Fig13 Coding Direction"),
    ("03_population/b_population_psth_heatmap.py", "Fig14 Population Heatmap"),
    ("03_population/c_dimensionality_reduction.py","Fig15 PCA Dimensionality"),
    ("03_population/d_state_matched_cd.py",        "Fig16 State-Matched CD"),
    ("03_population/e_sensory_dose_response.py",   "Fig17 Sensory Dose-Response"),
    # ── 04_decoding (Figs 18-20) ──────────────────────────────────────
    ("04_decoding/a_hit_miss_decoding.py",         "Fig18 Hit/Miss Decoding"),
    ("04_decoding/b_change_size_decoding.py",      "Fig19 Change-Size Decoding"),
    ("04_decoding/c_state_decoding.py",            "Fig20 State Decoding"),
    # ── 05_longitudinal (Figs 21-23) ──────────────────────────────────
    ("05_longitudinal/a_neural_learning_curves.py","Fig21 Neural Learning"),
    ("05_longitudinal/b_celltype_learning.py",     "Fig22 Cell-Type Learning"),
    ("05_longitudinal/c_population_geometry_shift.py","Fig23 Geometry Shift"),
    # ── 06_lick_motor (Figs 24-26) ────────────────────────────────────
    ("06_lick_motor/a_fa_neural_signatures.py",    "Fig24 FA Neural Signatures"),
    ("06_lick_motor/b_pre_lick_ramping.py",        "Fig25 Pre-Lick Ramping"),
    ("06_lick_motor/c_motor_vs_sensory.py",        "Fig26 Motor vs Sensory"),
    # ── 07_advanced (Figs 27-34) ──────────────────────────────────────
    ("07_advanced/a_glm_encoding.py",              "Fig27 GLM Encoding"),
    ("07_advanced/b_dpca.py",                      "Fig28 dPCA"),
    ("07_advanced/c_noise_correlations.py",        "Fig29 Noise Correlations"),
    ("07_advanced/d_impulsivity_regression.py",    "Fig30 Impulsivity Regression"),
    ("07_advanced/e_trial_outcome_prediction.py",  "Fig31 Trial Outcome Prediction"),
    ("07_advanced/f_fa_subtype_lick_triggered_tf.py","Fig32 FA Subtype Lick-Triggered TF"),
    ("07_advanced/g_fa_subtype_prediction.py",     "Fig33 FA Subtype Prediction"),
    ("07_advanced/h_second_pulse_analysis.py",     "Fig34 Second Pulse Analysis"),
    ("07_advanced/i_fa_circular_shuffle_classification.py", "Fig32i FA Circular Shuffle"),
    ("07_advanced/j_fa_matched_null_classification.py",     "Fig32j FA Matched Null"),
    # ── 08_tf_pulse (Figs 35-42) ──────────────────────────────────────
    ("08_tf_pulse/a_tf_responsiveness.py",         "Fig35 TF Responsiveness"),
    ("08_tf_pulse/b_tf_response_properties.py",    "Fig36 TF Response Properties"),
    ("08_tf_pulse/c_tf_pulse_integration.py",      "Fig37 Two-Pulse Integration"),
    ("08_tf_pulse/d_tf_learning_emergence.py",     "Fig38 TF Learning Emergence"),
    ("08_tf_pulse/e_tf_state_modulation.py",       "Fig39 TF × HMM State"),
    ("08_tf_pulse/f_tf_sensory_motor.py",          "Fig40 TF Sensory-Motor"),
    ("08_tf_pulse/g_tf_cell_classifier.py",        "Fig41 TF Cell Classifier"),
    ("08_tf_pulse/g2_tf_tier_gallery.py",          "Fig41g TF Tier Gallery"),
    ("08_tf_pulse/h_tf_post_error_modulation.py",  "Fig42 TF Post-Error Modulation"),
    # ── 09_optotagging (Fig 43) ───────────────────────────────────────
    ("09_optotagging/a_optotagging_identification.py","Fig43 Optotagging Identification"),
]

# Scripts that accept --n_workers (session-level parallelism)
PARALLEL_SCRIPTS = {
    "03_population/a_coding_direction.py",
    "04_decoding/a_hit_miss_decoding.py",
    "07_advanced/a_glm_encoding.py",
}


def main():
    parser = argparse.ArgumentParser(description="Run all analysis scripts.")
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