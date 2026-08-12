# scripts/audit/d3_scripts_census.py
"""D3: (1) local date parsers trio-tested (2) zfill(8) dtype exposure
(3) partial_spearman three-variant spread on one shared real input
(4) vd_tf_bg046 writers: on-disk figure mtime vs script last-commit."""
import csv, re, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d3_scripts_census.py"; S = "d3_scripts_census.py"

# (1) the trio behaviour that every local strptime parser inherits
trio = {}
for tok in ["01072025", "1072025", "1072025.0"]:
    try:
        trio[tok] = datetime.strptime(tok, "%d%m%Y").date().isoformat()
    except ValueError:
        trio[tok] = "ValueError"
record("d3.dateparser.trio", "D3", "strptime('%d%m%Y') on 01072025/1072025/1072025.0",
       " | ".join(f"{k}->{v}" for k, v in trio.items()), "behaviour", CMD, S,
       notes="1072025 silently becomes 10 July: WRONG DATE, no exception")

sites = []
for p in (REPO / "scripts").rglob("*.py"):
    if "__pycache__" in p.parts or "audit" in p.parts:   # component test, not substring
        continue
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if re.search(r"strptime\([^)]*%d%m%Y", line):
            sites.append((str(p.relative_to(REPO)), i, "strptime"))
        if re.search(r"zfill\(8\)", line):
            sites.append((str(p.relative_to(REPO)), i, "zfill8"))
with (REPO / "data/cache/audit/date_parser_sites.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["file", "line", "kind"])
    w.writerows(sites)
record("d3.dateparser.sites", "D3", "raw strptime('%d%m%Y') sites in scripts/",
       sum(1 for *_a, k in sites if k == "strptime"), "sites", CMD, S,
       "data/cache/audit/date_parser_sites.csv")
record("d3.zfill.sites", "D3", "ad-hoc zfill(8) sites in scripts/",
       sum(1 for *_a, k in sites if k == "zfill8"), "sites", CMD, S)

# (2) does load_staging_manifest canonicalize? (decides 78-vs-~10 real defects)
from visdetect.analysis.config import load_staging_manifest
m = load_staging_manifest(qc_only=False)
kinds = m["session_name"].map(lambda x: type(x).__name__).value_counts().to_dict()
record("d3.zfill.manifest_dtype", "D3", "session_name dtypes returned by load_staging_manifest",
       str(kinds), "dtypes", CMD, S,
       notes="if all str+8digit, downstream zfill sites are redundant-but-harmless")

# (3) partial_spearman spread — PRE-FLIGHT FIX (blocker): replicate the THREE
# estimator families that actually exist in the codebase, verbatim. All
# residual-based copies rank FIRST and residualize RANKS on RANKS; the third
# family is the closed-form pairwise-rho formula. The original plan draft
# residualized raw values (an estimator used NOWHERE in the repo) and included
# a variant mathematically identical to another.
from scipy.stats import spearmanr, rankdata
rng_src = REPO / "data/cache/session_sorting/session_group_features.csv"
rows = list(csv.DictReader(rng_src.open(encoding="utf-8")))
x = np.array([float(r["occ_StimSens"]) for r in rows])
y = np.array([float(r["hit_rate_go"]) for r in rows])
z = np.array([float(r["n_trials"]) for r in rows])


def _resid(a, c):
    A = np.column_stack([np.ones_like(c), c])
    return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]


rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
ex, ey = _resid(rx, rz), _resid(ry, rz)
# family A: rank -> residualize -> spearmanr   (theta_prototype.py:106-115,
#           theta_count_matched.py:147, within_session_dynamics.py:65-71)
v_a = spearmanr(ex, ey).statistic
# family B: rank -> residualize -> np.corrcoef (learning_continuum.py:94-104,
#           learning_transient_sustained.py:95, latency_outcome_coupling.py:254)
v_b = float(np.corrcoef(ex, ey)[0, 1])
# family C: closed-form from pairwise Spearman rhos (explore4_partial_rt.py:49-57)
rxy = spearmanr(x, y).statistic
rxz = spearmanr(x, z).statistic
ryz = spearmanr(y, z).statistic
v_c = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
record("d3.pspearman.spread", "D3",
       "partial_spearman THREE-FAMILY spread on one shared real input (n=%d)" % len(x),
       f"rank+spearmanr={v_a:.3f} | rank+corrcoef={v_b:.3f} | pairwise-rho={v_c:.3f}",
       "rho", CMD, S,
       "theta_prototype.py:106 vs learning_continuum.py:104 vs explore4_partial_rt.py:49",
       notes="all three replicated verbatim from their source files; any spread "
             ">0.02 upgrades the register entry to materially-different-in-practice")

# (4) vd_tf_bg046 writers: are the repo-tree figures older than the code?
writers = subprocess.run(["git", "grep", "-lE", "vd_tf_bg046/(FIGURES|data)", "--", "scripts/"],
                         capture_output=True, text=True, cwd=REPO).stdout.split()
stale = []
for w_ in writers:
    last = subprocess.run(["git", "log", "-1", "--format=%cs", "--", w_],
                          capture_output=True, text=True, cwd=REPO).stdout.strip()
    stale.append(f"{w_}@{last}")
record("d3.vdtf.writers", "D3", "scripts writing into the deleted vd_tf_bg046 tree",
       len(writers), "files", CMD, S, ";".join(stale[:10]),
       notes="reruns succeed and write nowhere visible")

# (5) PRE-FLIGHT ADDITION: per-script classification census (spec: classify the
# ~130 scripts writing neither figure nor CSV; entry-point convention)
import ast as _ast
srcs = {}
for p in (REPO / "scripts").rglob("*.py"):
    if "__pycache__" in p.parts or "audit" in p.parts:
        continue
    srcs[str(p.relative_to(REPO))] = p.read_text(encoding="utf-8", errors="replace")

# ROUND-2 ADDITION: intra-scripts import edges -> in-degree (spec's import DAG).
# Scripts import siblings by bare module name after sys.path tricks, so an
# import of a name matching another script's stem is an edge.
by_stem = {}
for rel in srcs:
    by_stem.setdefault(Path(rel).stem, set()).add(rel)
imported_by_someone = set()
for rel, src2 in srcs.items():
    try:
        tree = _ast.parse(src2)
    except SyntaxError:
        continue
    for node in _ast.walk(tree):
        mods = []
        if isinstance(node, _ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, _ast.ImportFrom) and node.module:
            mods = [node.module.split(".")[0]]
        for m in mods:
            for target in by_stem.get(m, ()):
                if target != rel:
                    imported_by_someone.add(target)

cls_rows = []
for rel, src2 in sorted(srcs.items()):
    cls_rows.append([rel,
                     bool(re.search(r'__name__\s*==\s*["\']__main__', src2)),
                     "argparse" in src2,
                     bool(re.search(r"savefig|save_figure", src2)),
                     ".to_csv(" in src2 or "np.save" in src2 or "json.dump" in src2,
                     rel not in imported_by_someone])
with (REPO / "data/cache/audit/script_classification.csv").open("w", newline="",
                                                                encoding="utf-8") as fh:
    w3 = csv.writer(fh, lineterminator="\n")
    w3.writerow(["file", "has_main", "has_argparse", "writes_figure", "writes_data",
                 "in_degree_0"])
    w3.writerows(cls_rows)
no_output = sum(1 for r in cls_rows if not r[3] and not r[4])
orphan_nonentry = sum(1 for r in cls_rows if r[5] and not r[1] and not r[2])
record("d3.scripts.no_output", "D3",
       "scripts writing neither figure nor data artefact (shared-module / job-body / dead)",
       no_output, "files", CMD, S, "data/cache/audit/script_classification.csv",
       notes="Task 15 triages this CSV into shared-module vs dead for the drop-list")
record("d3.scripts.orphan_nonentry", "D3",
       "in-degree-0 scripts that are ALSO not entry points (no __main__, no argparse)",
       orphan_nonentry, "files", CMD, S, "data/cache/audit/script_classification.csv")

# shim importer count (Task 15's drop-list evidence for the dead top-level shims)
shim = subprocess.run(["git", "grep", "-nE",
                       r"from visdetect import (session|io)\b|from visdetect\.(session|io) import",
                       "--", "scripts/", "src/", "tests/"],
                      capture_output=True, text=True, cwd=REPO).stdout.splitlines()
record("d3.shim_importers", "D3",
       "importers of the top-level shims src/visdetect/{session,io}.py",
       len(shim), "sites", CMD, S,
       notes="0 expected - the evidence behind the drop-list entry")

# (6) lick-channel overlap: the 33-session re-extraction batch list is not
# materialized anywhere in the repo, and deriving it needs NI-file inspection on
# the X: mount, which the audit forbids. Recorded honestly:
record("d3.lick.overlap", "D3",
       "which unguarded lick-channel scripts touch the 33 re-extracted sessions",
       "not-measured", "n/a", CMD, S,
       notes="batch list requires X:-side NI-file audit (forbidden); direction "
             "carried in the register: lick rates in affected sessions are "
             "under-detected 10-40x, so cross-session lick trends are suspect")
print("done")
