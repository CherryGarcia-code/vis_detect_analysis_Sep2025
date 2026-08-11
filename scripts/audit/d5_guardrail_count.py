# scripts/audit/d5_guardrail_count.py
"""D5: true guardrail violation count after the approved SKIP_DIRS fix."""
import re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

out = subprocess.run(
    [sys.executable, str(REPO / "scripts/qc/check_refactor_guardrails.py")],
    capture_output=True, text=True)
hard = re.search(r"HARD violations \((\d+)\)", out.stdout)
n = int(hard.group(1)) if hard else -1
(REPO / "data/cache/audit/guardrails_after.txt").write_text(out.stdout, encoding="utf-8")
record("d5.guardrail.after", "D5", "real HARD violations after .claude excluded", n,
       "count", "py scripts/audit/d5_guardrail_count.py", "d5_guardrail_count.py",
       "scripts/qc/check_refactor_guardrails.py", "recon predicted ~218")
print("HARD after fix:", n)
