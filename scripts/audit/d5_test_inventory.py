# scripts/audit/d5_test_inventory.py
import csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d5_test_inventory.py"; S = "d5_test_inventory.py"
DATA_PAT = re.compile(r"load_session|\.pkl|staging_manifest|PKL_DIR|data/cache")
rows, covered = [], set()
tests = [p for p in (REPO / "tests").rglob("test_*.py") if "audit" not in p.parts]
for t in tests:
    src = t.read_text(encoding="utf-8", errors="replace")
    # ROUND-2 FIX: `from PKG import name` must credit PKG.name, not just PKG —
    # the dominant test style is `from visdetect.analysis import decision_latents`
    # (21 occurrences), which the old regex reduced to the bare package.
    mods = set()
    for pkg, names, plain in re.findall(
            r"from (visdetect(?:\.\w+)*) import ([\w,\s]+)|import (visdetect(?:\.\w+)*)", src):
        if plain:
            mods.add(plain)
        if pkg:
            mods.add(pkg)
            for n in names.split(","):
                n = n.strip().split(" as ")[0].strip()
                if n and n.isidentifier():
                    mods.add(f"{pkg}.{n}")
    covered |= mods
    rows.append([str(t.relative_to(REPO)), bool(DATA_PAT.search(src)),
                 ";".join(sorted(mods))])
with (REPO / "data/cache/audit/test_partition.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["test_file", "needs_real_data", "covers_modules"]); w.writerows(rows)

lib_modules = {"visdetect." + str(p.relative_to(REPO / "src/visdetect"))[:-3].replace("\\", ".")
               for p in (REPO / "src/visdetect").rglob("*.py") if p.stem != "__init__"}
# ROUND-2 FIX: exact-or-descendant ONLY. The old `m.startswith(c)` direction let a
# single `from visdetect.analysis import X` line blanket all ~40 analysis modules.
untested = sorted(m for m in lib_modules
                  if not any(c == m or c.startswith(m + ".") for c in covered))
record("d5.tests.total", "D5", "test files", len(rows), "files", CMD, S)
record("d5.tests.need_real_data", "D5", "test files requiring real sessions/caches",
       sum(1 for r in rows if r[1]), "files", CMD, S, "data/cache/audit/test_partition.csv")
record("d5.tests.untested_modules", "D5", "library modules with ZERO test coverage",
       len(untested), "modules", CMD, S, notes=";".join(untested[:15]))
print(f"tests={len(rows)} real-data={sum(1 for r in rows if r[1])} untested={len(untested)}")
