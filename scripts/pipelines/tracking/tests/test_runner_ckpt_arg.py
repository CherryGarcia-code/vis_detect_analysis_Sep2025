import os, sys, subprocess
TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = subprocess.run(["python", os.path.join(TRACK, "run_deepunitmatch_all.py"), "--help"],
                     capture_output=True, text=True)
assert "--ckpt" in out.stdout, "runner is missing --ckpt"
print("test_runner_ckpt_arg PASS")
