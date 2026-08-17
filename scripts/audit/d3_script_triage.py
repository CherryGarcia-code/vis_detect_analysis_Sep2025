# scripts/audit/d3_script_triage.py
"""Task 15 wave 4 (finding C1): discharge the promised script-classification triage.

`d3.scripts.no_output`'s note and 03-scripts.md:122-125 both promised that Task 15
would triage `data/cache/audit/script_classification.csv` into shared-module vs
job-body vs dead for the drop-list. This script does it.

FULL triage — the 46 orphan-nonentry scripts (`in_degree_0` AND no `__main__` AND
no argparse; `d3.scripts.orphan_nonentry`), on three evidence axes:
  1. AST body shape: does the file perform top-level WORK when executed (bare
     calls, loops, non-import try blocks, non-scaffold call-bearing assigns), or
     is it defs/constants-only? A guard-less script with top-level work is a
     runnable job body; a defs-only file is only ever useful if something imports
     or names it.
  2. Name references: is the script's stem referenced by name anywhere in the
     maintained tracked tree (subprocess launches, shell/Slurm wrappers, docs)?
     The import DAG cannot see these (03-scripts.md import-DAG note), so this is
     the rescue axis. `scripts/audit/`, `docs/audit/`, `.superpowers/` and
     `data/` are EXCLUDED as referencers — the audit's own documents name many
     of these scripts precisely because they are suspect, and a mention in an
     audit deliverable must not count as life.
  3. Git recency: last-commit date of the file (`untracked` when never
     committed) — evidence, not a verdict axis.

Verdict rule (deliberately conservative toward NOT-dead; false name-reference
matches on generic stems can only rescue, never condemn):
  __init__.py                     -> package-marker (never dead-by-orphanhood;
                                     scripts/__init__.py is register A12's
                                     wheel-build trigger. Dunder-stem name
                                     matches are noise and are not used)
  test_*.py / *_test.py           -> job-body (pytest module - collected by
                                     path, invisible to both DAG and grep)
  referenced-by-name + defs-only  -> shared-module
  referenced-by-name + work       -> job-body
  unreferenced + top-level work   -> job-body (runnable via `py <path>`;
                                     invisible to both the DAG and the grep)
  unreferenced + defs-only        -> dead (in-degree 0, nothing names it, and
                                     executing it does nothing)

COARSE triage — the 108 no-output scripts (`d3.scripts.no_output`), from the
census columns plus axis 1 only. The three buckets partition the 108 exactly
(anything in-degree-0 without an entry point is by definition in the 46):
  shared-module  : in_degree_0 = False (imported by at least one sibling)
  entry-point    : has __main__ or argparse (CLI/job body writing no detected
                   artefact — may plot interactively, print, or mutate state)
  orphan-nonentry: member of the 46 -> the FULL verdict above applies

NOT triaged (stated so the drop-list can say exactly what was and wasn't done):
runtime string dispatch (the DAG blind spot stands); references from untracked
files; per-file execution. No script was run, imported, edited or deleted.

Output: data/cache/audit/script_triage.csv (gitignored; commit with git add -f).
Measurements: d8.scripts.orphan_triage, d8.scripts.nooutput_triage; re-records
d3.scripts.no_output verbatim with its promise note marked discharged.
"""
import ast
import csv
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d3_script_triage.py"
S = "d3_script_triage.py"
SRC_CSV = REPO / "data/cache/audit/script_classification.csv"
OUT_CSV = REPO / "data/cache/audit/script_triage.csv"

# Call tails that are scaffolding, not work, when they appear on an assign RHS.
_SCAFFOLD_TAILS = {
    "Path", "resolve", "absolute", "parent", "parents", "joinpath", "insert",
    "append", "dirname", "abspath", "realpath", "join", "getcwd", "getenv",
    "get", "getLogger", "basicConfig", "compile", "use", "namedtuple",
    "TypeVar", "field", "dataclass", "deepcopy", "OrderedDict", "defaultdict",
    "ArgumentParser", "add_argument", "set_defaults", "FullLoader",
}


def _is_main_guard(node):
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
            and t.left.id == "__name__")


def _call_tail(call):
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


def _assign_has_work_call(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and _call_tail(sub) not in _SCAFFOLD_TAILS:
            return True
    return False


def toplevel_work(path):
    """'work' | 'defs-only' | 'unparseable' for the module's top level."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return "unparseable"
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring / bare literal
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if _assign_has_work_call(node):
                return "work"
            continue
        if isinstance(node, ast.If) and _is_main_guard(node):
            continue  # censused separately as has_main
        if isinstance(node, ast.Try) and all(
                isinstance(n, (ast.Import, ast.ImportFrom, ast.Pass))
                for n in node.body):
            continue  # import-guard try
        return "work"  # bare call, loop, with, non-guard if, real try, etc.
    return "defs-only"


def _git(args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True).stdout


def last_commit(relpath):
    out = _git(["log", "-1", "--format=%cs", "--", relpath]).strip()
    return out if out else "untracked"


# ---------------------------------------------------------------- census input
rows = list(csv.DictReader(SRC_CSV.open(encoding="utf-8")))
for r in rows:
    r["file"] = r["file"].replace("\\", "/")

orphans = [r for r in rows if r["in_degree_0"] == "True"
           and r["has_main"] == "False" and r["has_argparse"] == "False"]
no_output = [r for r in rows if r["writes_figure"] == "False"
             and r["writes_data"] == "False"]
assert len(orphans) == 46, f"orphan set moved: {len(orphans)} != 46"
assert len(no_output) == 108, f"no-output set moved: {len(no_output)} != 108"
orphan_files = {r["file"] for r in orphans}

# ------------------------------------------------- referencer corpus (tracked)
_TEXT_EXT = {".py", ".sh", ".bash", ".sbatch", ".md", ".yml", ".yaml", ".toml",
             ".cfg", ".ini", ".ps1", ".bat", ".txt", ".json"}
_EXCLUDE_PREFIX = ("scripts/audit/", "docs/audit/", ".superpowers/", "data/",
                   "tests/audit/")
corpus = {}
for rel in _git(["ls-files"]).splitlines():
    rel = rel.strip()
    if not rel or rel.startswith(_EXCLUDE_PREFIX):
        continue
    if Path(rel).suffix.lower() not in _TEXT_EXT:
        continue
    p = REPO / rel
    try:
        if p.stat().st_size > 2_000_000:
            continue
        corpus[rel] = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue


def name_refs(relfile):
    stem = Path(relfile).stem
    pat = re.compile(r"\b" + re.escape(stem) + r"\b")
    hits = [ref for ref, text in corpus.items()
            if ref != relfile and pat.search(text)]
    return hits


# ------------------------------------------------------------------ FULL (46)
out_rows = []
full_verdicts = {"dead": [], "shared-module": [], "job-body": [],
                 "package-marker": []}
for r in sorted(orphans, key=lambda r: r["file"]):
    f = r["file"]
    base = Path(f).name
    body = toplevel_work(REPO / f)
    refs = name_refs(f)
    when = last_commit(f)
    if base == "__init__.py":
        verdict = "package-marker"
        why = ("package marker; not dead-by-orphanhood (scripts/__init__.py "
               "is register A12's wheel-build trigger); dunder-stem name "
               "matches are noise")
    elif base.startswith("test_") or base.endswith("_test.py"):
        verdict = "job-body"
        why = ("pytest test module - collected by path (py -m pytest <path>), "
               "invisible to both the import DAG and the name grep")
    elif refs:
        verdict = "job-body" if body == "work" else "shared-module"
        why = f"referenced by name in {len(refs)} tracked file(s), e.g. {refs[0]}"
    elif body in ("work", "unparseable"):
        verdict = "job-body"
        why = ("top-level work runs on `py <path>` despite no __main__ guard"
               if body == "work" else "unparseable - cannot rule out a job body")
    else:
        verdict = "dead"
        why = "in-degree 0, no entry point, defs-only, named nowhere tracked"
    full_verdicts[verdict].append(f)
    out_rows.append({
        "file": f, "set": "orphan-nonentry", "triage_depth": "full",
        "verdict": verdict, "toplevel_body": body,
        "name_ref_count": len(refs),
        "name_ref_example": refs[0] if refs else "",
        "last_commit": when,
        "writes_figure": r["writes_figure"], "writes_data": r["writes_data"],
        "why": why,
    })

# --------------------------------------------------------------- COARSE (108)
coarse_counts = {"shared-module": 0, "entry-point": 0, "orphan-nonentry": 0}
for r in sorted(no_output, key=lambda r: r["file"]):
    f = r["file"]
    if r["in_degree_0"] == "False":
        bucket = "shared-module"
    elif r["has_main"] == "True" or r["has_argparse"] == "True":
        bucket = "entry-point"
    else:
        bucket = "orphan-nonentry"
        assert f in orphan_files, f"partition broke on {f}"
    coarse_counts[bucket] += 1
    if bucket != "orphan-nonentry":  # orphans already have a full row above
        out_rows.append({
            "file": f, "set": "no-output", "triage_depth": "coarse",
            "verdict": bucket, "toplevel_body": toplevel_work(REPO / f),
            "name_ref_count": "", "name_ref_example": "",
            "last_commit": "", "writes_figure": "False", "writes_data": "False",
            "why": "coarse: census columns only (in_degree/has_main/has_argparse)",
        })

overlap = sum(1 for r in orphans if r["writes_figure"] == "False"
              and r["writes_data"] == "False")
assert coarse_counts["orphan-nonentry"] == overlap

# ---------------------------------------------------------------------- write
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()),
                       lineterminator="\n")
    w.writeheader()
    w.writerows(out_rows)

n_dead = len(full_verdicts["dead"])
n_shared = len(full_verdicts["shared-module"])
n_job = len(full_verdicts["job-body"])
n_pkg = len(full_verdicts["package-marker"])
record(
    "d8.scripts.orphan_triage", "D8",
    "FULL triage of the 46 orphan-nonentry scripts (dead / shared-module / job-body / package-marker)",
    f"dead:{n_dead} | shared-module:{n_shared} | job-body:{n_job} | "
    f"package-marker:{n_pkg}",
    "files", CMD, S, "data/cache/audit/script_triage.csv",
    notes=("Discharges the promise in d3.scripts.no_output's note and "
           "03-scripts.md:122-125. Axes: AST top-level-work, tracked "
           "name-reference grep (audit's own files excluded as referencers), "
           "git last-commit recency (evidence column). Rule is conservative "
           "toward NOT-dead: a name match anywhere tracked rescues; only "
           "in-degree-0 + no entry point + defs-only + named-nowhere = dead. "
           "HEADLINE: the 46 'strongest dead-code candidates' are mostly NOT "
           "dead - guard-less runnable job bodies dominate. NOT triaged: "
           "runtime string dispatch, references from untracked files, "
           "per-file execution. Dead files listed in the CSV and in "
           "drop-list.md section 2.8."))
record(
    "d8.scripts.nooutput_triage", "D8",
    "COARSE triage of the 108 no-output scripts",
    f"shared-module:{coarse_counts['shared-module']} | "
    f"entry-point:{coarse_counts['entry-point']} | "
    f"orphan-nonentry:{coarse_counts['orphan-nonentry']} (full triage above)",
    "files", CMD, S, "data/cache/audit/script_triage.csv",
    notes=("Coarse = census columns only; the three buckets partition the 108 "
           "exactly (in-degree-0 without an entry point is by definition in "
           "the 46, which got the full three-axis triage; the overlap is the "
           f"{overlap} orphan-nonentry rows). shared-module and entry-point "
           "rows are NOT drop candidates on this evidence alone."))
# Re-record the promise row verbatim (same measurement, same provenance),
# with the note marked discharged - record() is the only writer of the CSV.
record(
    "d3.scripts.no_output", "D3",
    "scripts writing neither figure nor data artefact (shared-module / job-body / dead)",
    108, "files", "py scripts/audit/d3_scripts_census.py", "d3_scripts_census.py",
    "data/cache/audit/script_classification.csv",
    notes=("Task 15 triages this CSV into shared-module vs dead for the "
           "drop-list. [DISCHARGED 2026-08-17, Task 15 wave 4: "
           "d8.scripts.orphan_triage + d8.scripts.nooutput_triage, "
           "data/cache/audit/script_triage.csv, drop-list.md section 2.8]"))

print(f"full 46 -> dead:{n_dead} shared:{n_shared} job:{n_job}")
print(f"coarse 108 -> {coarse_counts}")
print("dead files:")
for f in full_verdicts["dead"]:
    print("  ", f)
