# scripts/audit/d5_test_inventory_ast.py
"""AST ground-truth supplement to the byte-faithful D5 regex census.

The shipped ``d5_test_inventory.py`` (byte-faithful to the Task-10 brief) maps
test-file -> visdetect-module coverage with a regex that cannot match
parenthesised multi-line imports (``from X import (`` has no ``[\\w,\\s]+``
after ``import ``), so its ``d5.tests.untested_modules`` count is an overcount.
This script recomputes the untested-module count from ``ast.parse`` ground
truth over the same test set and the same exact-or-descendant crediting rule,
and records it as ``d5.tests.untested_modules_ast`` — the number the audit
instructs everyone to cite. It supplements, not replaces, the shipped census:
both measurement rows stand (dual-recording ruling, Task-10 review).
"""
import ast
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d5_test_inventory_ast.py"
S = "d5_test_inventory_ast.py"

# Same test set as the shipped census: tests/**/test_*.py, audit-owned excluded.
tests = [p for p in (REPO / "tests").rglob("test_*.py") if "audit" not in p.parts]

ast_cover = {}
for t in tests:
    mods = set()
    tree = ast.parse(t.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "visdetect" or a.name.startswith("visdetect."):
                    mods.add(a.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            m = node.module or ""
            if m == "visdetect" or m.startswith("visdetect."):
                mods.add(m)
                for a in node.names:
                    if a.name != "*":
                        mods.add(f"{m}.{a.name}")
    ast_cover[str(t.relative_to(REPO))] = mods

# Regex-credited coverage from the committed census CSV, for the miss count.
with (REPO / "data/cache/audit/test_partition.csv").open(
        newline="", encoding="utf-8") as fh:
    regex_cover = {
        r["test_file"]: (set(r["covers_modules"].split(";"))
                         if r["covers_modules"] else set())
        for r in csv.DictReader(fh)
    }
under_credited = sum(1 for tf, mods in ast_cover.items()
                     if mods - regex_cover.get(tf, set()))

lib_modules = {"visdetect." + str(p.relative_to(REPO / "src/visdetect"))[:-3].replace("\\", ".")
               for p in (REPO / "src/visdetect").rglob("*.py") if p.stem != "__init__"}


def untested(covered):
    # Exact-or-descendant crediting, identical to the shipped census rule.
    return sorted(m for m in lib_modules
                  if not any(c == m or c.startswith(m + ".") for c in covered))


u_ast = untested(set().union(*ast_cover.values()))
u_regex = untested(set().union(*regex_cover.values()))
falsely = sorted(set(u_regex) - set(u_ast))


def short(m):
    m = m.removeprefix("visdetect.")
    return m.removeprefix("analysis.") if m.startswith("analysis.") else m


notes = (
    "the shipped regex cannot match parenthesised multi-line imports "
    "('from X import (' has no [\\w,\\s]+ after 'import '), under-crediting "
    f"{under_credited}/{len(ast_cover)} test files; {len(falsely)} of the "
    f"shipped {len(u_regex)} 'untested' modules DO have AST-visible test "
    "imports (incl. kernel_width, spectrum_stats, state_calibration, "
    "core.qc); truly untested = " + ";".join(short(m) for m in u_ast)
)
record("d5.tests.untested_modules_ast", "D5",
       "library modules with zero test coverage (AST-corrected)",
       len(u_ast), "modules", CMD, S,
       "data/cache/audit/test_partition.csv", notes)
print(f"untested_ast={len(u_ast)} (regex={len(u_regex)}, "
      f"falsely_untested={len(falsely)}, "
      f"under_credited_files={under_credited}/{len(ast_cover)})")
