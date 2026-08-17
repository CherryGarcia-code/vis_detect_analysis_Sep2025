# scripts/audit/d3_parser_recount.py
"""Task 15 wave 4 (finding I2): recount the local date-parser sites.

The register's hand-count "23" (the census CSV's 19 strptime rows + 4
out-of-regex sites enumerated in entry 3) is itself an undercount, because the
shipped census regex (`strptime\\([^)]*%d%m%Y` on a single line, scripts/ only)
misses (a) `pd.to_datetime(..., format='%d%m%Y')` sites, (b) the 6-digit
`'%d%m%y'` variant, (c) multi-line calls, and (d) all of `src/`. Wave-4
verified four further sites first-hand (hmm_behavioral_states.py:45,
build_concat_windows.py:56, build_qc_sheets.py:116, run_unitmatch_all.py:67)
plus the hand-rolled repair `strptime("0"+s, "%d%m%Y")` at
run_unitmatch_all.py:78 (already in the CSV, but proof the class breeds).

This census walks the AST of every .py under scripts/ + src/ (excluding
scripts/audit/ and __pycache__) and counts every `strptime` / `to_datetime`
CALL whose literal format argument contains %d%m%Y or %d%m%y — single- or
multi-line, positional or keyword. Known residual blind spot, stated: a format
passed through a variable (e.g. `strptime(s, DATE_FMT)`) is not counted, so
the computed number is still a LOWER bound on the true population.

Output: data/cache/audit/date_parser_recount.csv (gitignored; commit with
git add -f). Measurement: d8.dateparser.recount.
"""
import ast
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d3_parser_recount.py"
S = "d3_parser_recount.py"
OUT_CSV = REPO / "data/cache/audit/date_parser_recount.csv"


def _call_name(call):
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


def sites_in(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in ("strptime", "to_datetime"):
            continue
        fmts = [a.value for a in
                list(node.args) + [k.value for k in node.keywords]
                if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        hit = [f for f in fmts if "%d%m%Y" in f or "%d%m%y" in f]
        if hit:
            variant = "%d%m%Y" if "%d%m%Y" in hit[0] else "%d%m%y"
            yield node.lineno, name, variant


rows = []
for root in ("scripts", "src"):
    for p in sorted((REPO / root).rglob("*.py")):
        parts = p.relative_to(REPO).parts
        if "__pycache__" in parts:
            continue
        if parts[0] == "scripts" and len(parts) > 1 and parts[1] == "audit":
            continue
        rel = str(p.relative_to(REPO)).replace("\\", "/")
        for lineno, func, variant in sites_in(p) or ():
            rows.append({"file": rel, "line": lineno, "func": func,
                         "variant": variant, "tree": parts[0]})

rows.sort(key=lambda r: (r["file"], r["line"]))
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "line", "func", "variant",
                                       "tree"], lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

n = len(rows)
n_scripts = sum(1 for r in rows if r["tree"] == "scripts")
n_src = n - n_scripts
n_todt = sum(1 for r in rows if r["func"] == "to_datetime")
n_ddmmyy = sum(1 for r in rows if r["variant"] == "%d%m%y")
record(
    "d8.dateparser.recount", "D8",
    "strptime/to_datetime sites with a literal %d%m%Y or %d%m%y format, scripts/ + src/ (AST census)",
    n, "sites", CMD, S, "data/cache/audit/date_parser_recount.csv",
    notes=(f"Supersedes the hand-count 23 in register entry 3 / drop-list s5 "
           f"(itself a correction of d3.dateparser.sites=19). Breakdown: "
           f"scripts/ {n_scripts}, src/ {n_src}; to_datetime {n_todt}; "
           f"6-digit %d%m%y variant {n_ddmmyy}. AST call census: catches "
           f"multi-line and keyword-arg calls the line regex missed; still a "
           f"LOWER bound (formats passed via variables are invisible). "
           f"scripts/audit/ excluded. [Task 15 wave 4]"))
print(f"total {n} (scripts {n_scripts}, src {n_src}; "
      f"to_datetime {n_todt}, %d%m%y {n_ddmmyy})")
for r in rows:
    print(f"  {r['file']}:{r['line']} {r['func']} {r['variant']}")
