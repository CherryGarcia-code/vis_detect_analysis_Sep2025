# scripts/audit/d4_traceability_census.py
"""D4c: stratified sample ~5 figures per FIGURES/<topic>; for each, try to
identify the producing script by (a) FIGURE_DIR usage + topic match,
(b) verbatim stem in source, (c) sidecar. No figure is regenerated."""
import csv, random, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d4_traceability_census.py"; S = "d4_traceability_census.py"
random.seed(20260809)   # fixed: sample is reproducible

# PRE-FLIGHT FIX (A4): git-TRACKED figures are deliverables and must ALL be in
# the census, in addition to the stratified random sample — otherwise acceptance
# criterion A4 ("covers every tracked figure") is not dischargeable.
tracked = set(subprocess.run(["git", "ls-files", "FIGURES"], capture_output=True,
                             text=True, cwd=REPO).stdout.split("\n"))
tracked = {t for t in tracked if t.lower().endswith((".png", ".pdf", ".svg"))}

rows = []
for topic in sorted(p for p in (REPO / "FIGURES").iterdir() if p.is_dir()):
    figs = [f for f in topic.rglob("*") if f.suffix.lower() in (".png", ".pdf", ".svg")]
    sample = set(random.sample(figs, min(5, len(figs))))
    sample |= {topic.parent.parent / t for t in tracked
               if t.startswith(f"FIGURES/{topic.name}/")}
    for fig in sorted(sample):
        stem = fig.stem
        method, producer = "untraceable", ""
        hits = subprocess.run(["git", "grep", "-lF", stem, "--", "scripts/", "src/"],
                              capture_output=True, text=True, cwd=REPO).stdout.splitlines()
        if hits:
            method, producer = "stem-in-source", hits[0]
        else:
            sidecars = list(fig.parent.glob(fig.stem + "*.json")) + \
                       list(fig.parent.glob(fig.stem + "*_notes.md"))
            if sidecars:
                method, producer = "sidecar", str(sidecars[0].relative_to(REPO))
            else:
                thits = subprocess.run(
                    ["git", "grep", "-lF", f"FIGURES/{topic.name}", "--", "scripts/"],
                    capture_output=True, text=True, cwd=REPO).stdout.splitlines()
                if len(thits) == 1:
                    method, producer = "unique-topic-writer", thits[0]
        rows.append([str(fig.relative_to(REPO)), topic.name, method, producer])

with (REPO / "data/cache/audit/traceability_sample.csv").open("w", newline="",
                                                              encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["figure", "topic", "method", "producer"]); w.writerows(rows)
n = len(rows)
untraceable = sum(1 for r in rows if r[2] == "untraceable")
record("d4.trace.sample", "D4", "figures in census (stratified sample + ALL git-tracked)", n,
       "figures", CMD, S, "data/cache/audit/traceability_sample.csv")
record("d4.trace.tracked_covered", "D4", "git-tracked deliverable figures included (A4)",
       len(tracked), "figures", CMD, S,
       notes="A4 requires every TRACKED figure covered; the stratified sample "
             "extends coverage to the untracked bulk")
record("d4.trace.untraceable_frac", "D4", "fraction of censused figures with NO identifiable producer",
       round(untraceable / n, 2), "fraction", CMD, S,
       notes="evidence for the S5 provenance layer, not a target (spec D4)")
print(f"sample={n} tracked={len(tracked)} untraceable={untraceable}")
