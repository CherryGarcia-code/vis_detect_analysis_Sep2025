# scripts/audit/d1_constants_census.py
"""D1: for each of the 82 canonical constants - re-export, importers, retypes,
agreement. Then classify the 130 divergent multi-file names into buckets
(a) scientific parameter (b) path alias (c) genuinely local.
AST-based; no imports of scanned files."""
import ast, csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

SCOPE = ["src", "scripts", "tests"]
SKIP = {".venv", "archive", "__pycache__", ".claude", "_DeepUnitMatch_repo",
        "refactor_baseline", "_preserved_from_worktrees_20260628"}


def py_files():
    for top in SCOPE:
        for p in (REPO / top).rglob("*.py"):
            if not any(s in p.parts for s in SKIP):
                yield p


def module_constants(path):
    """Module-level UPPERCASE assignments -> {name: value-source-string}."""
    out = {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return out
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            val = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, val = [node.target], node.value
        else:
            continue
        for t in targets:
            if t.id.isupper() and val is not None:
                out[t.id] = ast.unparse(val)
    return out


canon = module_constants(REPO / "src/visdetect/analysis/constants.py")
cfg = module_constants(REPO / "src/visdetect/analysis/config.py")
cfg_src = (REPO / "src/visdetect/analysis/config.py").read_text(encoding="utf-8")

# PRE-FLIGHT FIX (blocker): importer detection MUST be AST-based. The line-regex
# `import...NAME` misses multi-line parenthesized imports (13 in-scope files,
# incl. config.py's own 40-name re-export block and video_sync.py:69) and was
# measured to misclassify 48 of 82 constants as zero-importer.
all_defs = {}          # name -> list of (path, lineno, value_src)
importers = {}         # canonical name -> set of files importing it
for f in py_files():
    src = f.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in tree.body:
        targets, val = [], None
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            val = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, val = [node.target], node.value
        for t in targets:
            if t.id.isupper() and val is not None:
                all_defs.setdefault(t.id, []).append((f, node.lineno, ast.unparse(val)))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and \
           node.module.rsplit(".", 1)[-1] in ("constants", "config"):
            for alias in node.names:
                if alias.name in canon:
                    importers.setdefault(alias.name, set()).add(f)
    for name in canon:   # attribute-style usage: constants.NAME / config.NAME
        if re.search(rf"\b(constants|config)\.{name}\b", src):
            importers.setdefault(name, set()).add(f)

rows = []
for name, val in sorted(canon.items()):
    defs = all_defs.get(name, [])
    shadow = [(p, ln, v) for p, ln, v in defs
              if "src\\visdetect\\analysis\\constants.py" not in str(p)]
    agree = all(v == val for *_x, v in shadow) if shadow else True
    rows.append({
        "name": name, "defined_in": "constants.py",
        "reexported_by_config": name in cfg or f" {name}" in cfg_src,
        "n_importers": len(importers.get(name, set())),
        "n_retype_sites": len(shadow), "retypes_agree": agree,
        "bucket": "canonical"})

# multi-file names (not in canon): three buckets per spec — (a) divergent
# parameter, (b) path alias, (c) genuinely local (multi-file but values agree)
for name, defs in sorted(all_defs.items()):
    if name in canon or len({str(p) for p, _l, _v in defs}) < 2:
        continue
    sites = ";".join(sorted({f"{p.relative_to(REPO)}:{ln}" for p, ln, _v in defs})[:6])
    values = {v for *_x, v in defs}
    is_path = any(k in name for k in ("DIR", "PATH", "ROOT", "FILE", "OUT"))
    bucket = ("path-alias" if is_path
              else "divergent-parameter" if len(values) > 1
              else "genuinely-local")
    rows.append({"name": name, "defined_in": sites,
                 "reexported_by_config": False, "n_importers": 0,
                 "n_retype_sites": len(defs), "retypes_agree": len(values) == 1,
                 "bucket": bucket})

# D1 TF-period consumer census (spec: every TF_SAMPLE_PERIOD consumer AND every
# bare dt=0.05 / DT_GEN / DT=0.02 site; lowercase sites are invisible to the
# uppercase census above)
tf_rows = []
for f in py_files():
    for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if re.search(r"\bTF_SAMPLE_PERIOD\b", line):
            tf_rows.append([str(f.relative_to(REPO)), i, "TF_SAMPLE_PERIOD", line.strip()[:80]])
        if re.search(r"\bdt\s*[=:]\s*0\.05\b|\bDT_GEN\s*=\s*0\.05\b|\bDT\s*=\s*0\.02\b", line):
            tf_rows.append([str(f.relative_to(REPO)), i, "bare-dt", line.strip()[:80]])
with (REPO / "data/cache/audit/tf_dt_sites.csv").open("w", newline="", encoding="utf-8") as fh:
    w2 = csv.writer(fh, lineterminator="\n")
    w2.writerow(["file", "line", "kind", "source"]); w2.writerows(tf_rows)

out = REPO / "data/cache/audit/constants_census.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader(); w.writerows(rows)

dead = sum(1 for r in rows if r["bucket"] == "canonical"
           and r["n_importers"] == 0 and r["n_retype_sites"] == 0)
not_reexported = sum(1 for r in rows if r["bucket"] == "canonical"
                     and not r["reexported_by_config"])
disagreeing = sum(1 for r in rows if r["bucket"] == "canonical"
                  and not r["retypes_agree"])
div_param = sum(1 for r in rows if r["bucket"] == "divergent-parameter")

cmd = "py scripts/audit/d1_constants_census.py"
record("d1.constants.total", "D1", "canonical constants in constants.py",
       len(canon), "count", cmd, "d1_constants_census.py",
       "src/visdetect/analysis/constants.py")
record("d1.constants.dead", "D1", "canonical constants with zero importers and zero retypes",
       dead, "count", cmd, "d1_constants_census.py", str(out.relative_to(REPO)))
record("d1.constants.not_reexported", "D1", "canonical constants config.py fails to re-export",
       not_reexported, "count", cmd, "d1_constants_census.py")
record("d1.constants.shadow_disagree", "D1", "canonical constants with DISAGREEING retyped copies",
       disagreeing, "count", cmd, "d1_constants_census.py")
record("d1.constants.divergent_params", "D1", "non-canonical divergent parameter names (bucket a)",
       div_param, "count", cmd, "d1_constants_census.py")
record("d1.tfperiod.consumer_sites", "D1",
       "TF_SAMPLE_PERIOD consumers + bare dt=0.05/DT_GEN/DT=0.02 sites",
       len(tf_rows), "sites", cmd, "d1_constants_census.py",
       "data/cache/audit/tf_dt_sites.csv")
record("d1.tfperiod.figure_attribution", "D1",
       "which published figures/caches were produced under each dt", "not-measured",
       "n/a", cmd, "d1_constants_census.py",
       notes="requires per-figure provenance that does not exist (the D4 census "
             "measures exactly that gap); direction carried in the register")
print(f"canon={len(canon)} dead={dead} not_reexported={not_reexported} "
      f"disagree={disagreeing} divergent_params={div_param}")
