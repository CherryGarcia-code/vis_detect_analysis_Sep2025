#!/usr/bin/env python3
"""Run remaining analysis scripts (11-23) with timing and error tracking."""
import subprocess, sys, os, time

SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SUITE_DIR)
PYTHON = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")

SCRIPTS = [
    ("03_population/c_dimensionality_reduction.py", "Fig11 PCA Dimensionality"),
    ("04_decoding/a_hit_miss_decoding.py",          "Fig12 Hit/Miss Decoding"),
    ("04_decoding/b_change_size_decoding.py",       "Fig13 Change-Size Decoding"),
    ("04_decoding/c_state_decoding.py",             "Fig14 State Decoding"),
    ("05_longitudinal/a_neural_learning_curves.py", "Fig15 Neural Learning"),
    ("05_longitudinal/b_celltype_learning.py",      "Fig16 Cell-Type Learning"),
    ("05_longitudinal/c_population_geometry_shift.py", "Fig17 Geometry Shift"),
    ("06_lick_motor/a_fa_neural_signatures.py",     "Fig18 FA Neural Signatures"),
    ("06_lick_motor/b_pre_lick_ramping.py",         "Fig19 Pre-Lick Ramping"),
    ("06_lick_motor/c_motor_vs_sensory.py",         "Fig20 Motor vs Sensory"),
    ("07_advanced/a_glm_encoding.py",               "Fig21 GLM Encoding"),
    ("07_advanced/b_dpca.py",                       "Fig22 dPCA"),
    ("07_advanced/c_noise_correlations.py",         "Fig23 Noise Correlations"),
]

def main():
    results = []
    t_total = time.time()
    log_path = os.path.join(SUITE_DIR, "run_remaining_log.txt")
    log = open(log_path, "w", buffering=1)

    def out(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    for i, (script, label) in enumerate(SCRIPTS, 11):
        script_path = os.path.join(SUITE_DIR, script)
        out(f"\n{'='*70}")
        out(f"[{i:2d}/23] {label}  ({script})")
        out(f"{'='*70}")

        t0 = time.time()
        try:
            result = subprocess.run(
                [PYTHON, "-u", script_path],
                cwd=SUITE_DIR,
                capture_output=True,
                timeout=3600,  # 60 min timeout per script
            )
            elapsed = time.time() - t0
            status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
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
    out(f"\n  {n_ok}/13 completed successfully.")
    log.close()

if __name__ == "__main__":
    main()
