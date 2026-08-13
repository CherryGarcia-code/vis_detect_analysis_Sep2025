# scripts/audit/d6_retraction_survey.py
"""D6 supplement (Task 11 step 2): retraction-marker survey of docs/superpowers/.

Greps every file under docs/superpowers/ for the terminal-status tokens
RETRACTED|REFUTED|SUPERSEDED (case-sensitive, per the audit brief) and records
how many files carry at least one token vs the corpus total. Marker presence
is an UPPER bound on terminal-status labelling: inspection shows most hits are
cautionary mentions of OTHER work's retractions inside living design docs, not
a status header declaring the file's own work retracted (details in
docs/audit/06-ai-layer.md).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d6_retraction_survey.py"
S = "d6_retraction_survey.py"

pat = re.compile(r"RETRACTED|REFUTED|SUPERSEDED")
files = sorted(p for p in (REPO / "docs/superpowers").rglob("*") if p.is_file())
hits = [str(p.relative_to(REPO)) for p in files
        if pat.search(p.read_text(encoding="utf-8", errors="replace"))]
record("d6.superpowers.retraction_markers", "D6",
       "docs/superpowers files containing RETRACTED|REFUTED|SUPERSEDED",
       len(hits), "files", CMD, S, ";".join(hits),
       notes=f"corpus total {len(files)} files (spec recon: 76); marker presence is "
             "an UPPER bound on terminal-status labelling - hits are mostly mentions "
             "of other work's retractions, and the corpus has no status-header "
             "convention; the retraction record lives in the memory layer instead")
print(f"{len(hits)}/{len(files)} docs/superpowers files carry a marker")
