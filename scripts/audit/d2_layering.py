# scripts/audit/d2_layering.py
"""D2: (1) src.visdetect vs visdetect importers (2) sys.path.insert
classification (3) module-level upward layer edges via AST
(4) import wall-times. Wheel/clean-venv check is Step 3 (shell)."""
import ast, csv, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d2_layering.py"; S = "d2_layering.py"
SKIP = {".venv", "archive", "__pycache__", ".claude", "_DeepUnitMatch_repo",
        "refactor_baseline", "_preserved_from_worktrees_20260628"}
files = [p for top in ("src", "scripts", "tests")
         for p in (REPO / top).rglob("*.py") if not any(x in p.parts for x in SKIP)]

# (1) dual import roots
dual, srcroot = set(), set()
for f in files:
    src = f.read_text(encoding="utf-8", errors="replace")
    has_src = re.search(r"^\s*(from|import)\s+src\.visdetect", src, re.M)
    has_plain = re.search(r"^\s*(from|import)\s+visdetect", src, re.M)
    if has_src: srcroot.add(f)
    if has_src and has_plain: dual.add(f)
record("d2.dualroot.src_importers", "D2", "files importing src.visdetect.*",
       len(srcroot), "files", CMD, S)
record("d2.dualroot.mixed", "D2", "files mixing BOTH import roots (distinct class objects)",
       len(dual), "files", CMD, S,
       ";".join(sorted(str(p.relative_to(REPO)) for p in dual)[:8]))

# (2) sys.path.insert classification
rows = []
for f in files:
    for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        m = re.search(r"sys\.path\.(insert|append)\(.*?['\"](.+?)['\"]", line)
        if not m:
            if "sys.path" in line and ("insert" in line or "append" in line):
                rows.append((f, i, "<computed>", "computed"))
            continue
        target = m.group(2)
        if "vd_tf" in target or (target.startswith(("E:", "e:")) and "vis_detect" not in target):
            cat = "foreign-absolute"
        elif target.endswith("src") or target.endswith("src/"):
            cat = "repo-src"
        else:
            cat = "other-literal"
        exists = Path(target).exists() if ":" in target else None
        rows.append((f, i, target, cat + ("" if exists in (None, True) else "-MISSING")))
with (REPO / "data/cache/audit/syspath_sites.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["file", "line", "target", "category"])
    for f, i, t, c in rows:
        w.writerow([str(f.relative_to(REPO)), i, t, c])
record("d2.syspath.total", "D2", "sys.path mutation sites (maintained tree)",
       len(rows), "sites", CMD, S, "data/cache/audit/syspath_sites.csv")
record("d2.syspath.foreign_missing", "D2", "sites pointing at NON-EXISTENT foreign src trees",
       sum(1 for *_x, c in rows if c == "foreign-absolute-MISSING"), "sites", CMD, S,
       notes="silent fall-through to ambient visdetect - provenance unverifiable")

# (3) module-level upward edges
LAYER = {"core": 0, "anatomy": 1, "analysis": 2, "suite": 3}
edges = []
for f in (REPO / "src/visdetect").rglob("*.py"):
    parts = f.relative_to(REPO / "src/visdetect").parts
    src_layer = LAYER.get(parts[0], None)
    if src_layer is None:
        continue
    try:
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        continue
    for node in tree.body:  # module level only - lazy imports excluded by design
        mods = []
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("visdetect."):
            mods = [node.module]
        elif isinstance(node, ast.Import):
            mods = [a.name for a in node.names if a.name.startswith("visdetect.")]
        for m in mods:
            tgt = m.split(".")[1] if len(m.split(".")) > 1 else ""
            if tgt in LAYER and LAYER[tgt] > src_layer:
                edges.append(f"{f.relative_to(REPO)} -> {m}")
record("d2.layers.upward_module_level", "D2",
       "module-level upward import edges (core/anatomy importing analysis/suite)",
       len(edges), "edges", CMD, S, ";".join(edges[:6]))

# (4) import wall-times (fresh interpreter each)
for mod in ["visdetect", "visdetect.core", "visdetect.analysis.constants",
            "visdetect.suite.loader"]:
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c",
                        f"import {mod}, sys; print(len(sys.modules))"],
                       capture_output=True, text=True, cwd=REPO)
    dt = time.perf_counter() - t0
    nmod = r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "FAIL"
    record(f"d2.importtime.{mod}", "D2", f"cold import wall-time of {mod}",
           round(dt, 2), "s", CMD, S, notes=f"sys.modules={nmod}")
print("done")

# --- Task-5 addition (brief Step 3): parents[N] census over src/ ---
# Path(...).parents[N] is the idiom class that produced the live parents[1]
# qc-profile bug (core/qc.py resolving config/ under src/visdetect instead of
# the repo root). Census every src/ site so Task 15 can triage fragile indices.
prows = []
for f in sorted((REPO / "src").rglob("*.py")):
    if any(x in f.parts for x in SKIP):
        continue
    for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for m in re.finditer(r"parents\[\d\]", line):
            prows.append((str(f.relative_to(REPO)), i, m.group(0), line.strip()))
with (REPO / "data/cache/audit/parents_sites.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["file", "line", "match", "source_line"])
    w.writerows(prows)
record("d2.parents.sites", "D2", "Path(...).parents[N] sites in src/ (fragile-root idiom)",
       len(prows), "sites", CMD, S, "data/cache/audit/parents_sites.csv",
       notes="idiom class that produced the live parents[1] qc-profile bug (core/qc.py:218)")
print("parents census done")
