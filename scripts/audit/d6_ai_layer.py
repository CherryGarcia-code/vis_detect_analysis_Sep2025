# scripts/audit/d6_ai_layer.py
"""D6: (1) every `SYMBOL` = value claim in CLAUDE.md/docs/skills resolved
against code (2) dead path references (3) model ids in .claude prose."""
import ast, csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d6_ai_layer.py"; S = "d6_ai_layer.py"

# ground truth values
def consts_of(p):
    out = {}
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for n in tree.body:
        tgts = n.targets if isinstance(n, ast.Assign) else \
               ([n.target] if isinstance(n, ast.AnnAssign) else [])
        for t in tgts:
            if isinstance(t, ast.Name) and t.id.isupper() and getattr(n, "value", None):
                out[t.id] = ast.unparse(n.value)
    return out
truth = {}
for f in ["src/visdetect/analysis/constants.py", "src/visdetect/analysis/config.py"]:
    truth.update(consts_of(REPO / f))

DOCS = [REPO / "CLAUDE.md"] + list((REPO / "docs").rglob("*.md")) + \
       list((REPO / ".claude/skills").rglob("SKILL.md"))
DOCS = [d for d in DOCS if "audit" not in d.parts and "superpowers" not in d.parts]

claim_pat = re.compile(r"`([A-Z][A-Z0-9_]{2,})`\s*(?:\||=|:)\s*`?([^`|\n]{1,40})")
lit_rows, dead_rows = [], []
path_pat = re.compile(r"`((?:scripts|src|docs|config|analysis_suite|data)/[\w./\-]+)`")
for d in DOCS:
    for i, line in enumerate(d.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for sym, claimed in claim_pat.findall(line):
            if sym in truth:
                code_v = truth[sym]
                same = claimed.strip().rstrip("|").strip() in code_v or \
                       code_v in claimed
                lit_rows.append([str(d.relative_to(REPO)), i, sym,
                                 claimed.strip()[:40], code_v[:40],
                                 "match" if same else "MISMATCH"])
            elif re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", sym):
                lit_rows.append([str(d.relative_to(REPO)), i, sym,
                                 claimed.strip()[:40], "", "symbol-not-found"])
        for pth in path_pat.findall(line):
            if not (REPO / pth).exists():
                dead_rows.append([str(d.relative_to(REPO)), i, pth])

for name, rows, hdr in [("doc_literals.csv", lit_rows,
                         ["doc", "line", "symbol", "doc_value", "code_value", "verdict"]),
                        ("dead_paths.csv", dead_rows, ["doc", "line", "path"])]:
    with (REPO / "data/cache/audit" / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(hdr); w.writerows(rows)

record("d6.literals.checked", "D6", "SYMBOL=value claims in docs resolved against code",
       len(lit_rows), "claims", CMD, S, "data/cache/audit/doc_literals.csv")
record("d6.literals.mismatch", "D6", "claims that MISMATCH the code",
       sum(1 for r in lit_rows if r[5] == "MISMATCH"), "claims", CMD, S)
record("d6.literals.symbol_missing", "D6", "claims naming symbols that do not exist",
       sum(1 for r in lit_rows if r[5] == "symbol-not-found"), "claims", CMD, S)
record("d6.deadpaths", "D6", "doc references to non-existent paths",
       len(dead_rows), "refs", CMD, S, "data/cache/audit/dead_paths.csv",
       notes="recon: 181 analysis_suite refs across 42 files")

models = []
for f in (REPO / ".claude").rglob("*.md"):
    if "worktrees" in f.parts:   # PRE-FLIGHT FIX: 1,150 of 1,159 md files under
        continue                 # .claude are duplicate worktree checkouts
    for m in re.findall(r"claude-[a-z0-9\-\.\[\]]+|[Oo]pus[- ][\d.]+", f.read_text(encoding="utf-8", errors="replace")):
        models.append(f"{f.relative_to(REPO)}:{m}")
record("d6.modelids", "D6", "model ids hardcoded in primary .claude prose (worktrees excluded)",
       len(models), "mentions", CMD, S, notes=";".join(sorted(set(models))[:10]))

# canonical-authority claimants (spec: count files claiming to be THE instruction file)
auth = []
for f in [REPO / "CLAUDE.md"] + list((REPO / "docs/AI_interaction").glob("*.md")):
    if f.exists() and re.search(r"canonical|authoritative|single source of truth",
                                f.read_text(encoding="utf-8", errors="replace"), re.I):
        auth.append(str(f.relative_to(REPO)))
record("d6.authority.claimants", "D6", "instruction files claiming canonical authority",
       len(auth), "files", CMD, S, ";".join(auth),
       notes="only CLAUDE.md is actually loaded by the harness")
record("d6.dup_pair_agreement", "D6", "line-level agreement diff of duplicated doc pairs",
       "not-measured", "n/a", CMD, S,
       notes="recon measured overlap %s and the one known divergence (GOTCHAS session-id "
             "row); full pairwise diff deferred - the new repo deletes the copies (ADR-005)")
print("done")
