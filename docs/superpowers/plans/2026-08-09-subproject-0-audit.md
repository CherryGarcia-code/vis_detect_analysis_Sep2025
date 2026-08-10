# Sub-project 0 — Deep Empirical Audit: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the audit findings corpus (`docs/audit/`) specified by
`docs/superpowers/specs/2026-08-06-subproject-0-audit-spec.md`, from executed measurements — the
evidence base from which specs 1–6 and the known-defect register are written.

**Architecture:** A small library of one-shot measurement scripts under `scripts/audit/`, all
appending typed rows to one `docs/audit/measurements.csv` through a shared recorder, plus one
markdown report per domain assembled from those rows. Nothing in the existing repo is modified
except the single approved `SKIP_DIRS` exception. Heavy real-data measurements load at most a
handful of sessions, locally.

**Tech Stack:** Python 3.10 (the repo's existing `.venv`, invoked as `py`), stdlib `ast`/`csv`/
`hashlib`, pandas, pytest for the harness tests. D9 only: a *separate* scratch venv with
`pynwb`/`neuroconv` so the repo venv is never mutated.

## Global Constraints

- **Working copy:** primary checkout, branch `design/new-repo-foundation`. No worktree (junction
  hazard; audit writes are additive-only).
- **Read-only rule:** the audit modifies NOTHING outside `scripts/audit/`, `docs/audit/`,
  `data/cache/audit/` — with exactly one approved exception: the `SKIP_DIRS` line in
  `scripts/qc/check_refactor_guardrails.py` (Task 2).
- **Never run compute over the `X:` Samba mount.** All pkls are local under `data/pkls/`.
- `py` not `python` (Windows + Git Bash). The `.venv` has an editable install; `import visdetect`
  must work without `sys.path` hacks (verified in Task 1).
- **Measured, not read:** every finding row carries the command run and its output; every report
  claim carries `file:line` + blast radius. A claim that can be settled by execution is settled by
  execution.
- **Session ids:** any session id used as a key goes through the Task-1 classifier/canonicaliser;
  never `int()`, never ad-hoc `zfill`.
- **Memory:** any script loading sessions does `del sess; gc.collect()` per session.
- **Budget (ADR-020):** ~5 working days for Tasks 1–13 + 15–16; D9 (Task 14) is hard time-boxed to
  1 day. If the budget is blown, remaining measurements are recorded as `not-measured` rows rather
  than silently skipped.
- **Subagents:** every dispatched subagent runs on the newest Opus (`model: 'opus'` explicit).
- All file writes UTF-8, LF; `measurements.csv` written via the recorder only.

**measurements.csv schema (fixed):**
`measurement_id,domain,metric,value,unit,command,script,evidence,notes` — `measurement_id` unique
(re-recording overwrites), `command` is the exact reproduction command, `evidence` is a repo path
or `file:line`.

---

### Task 1: Audit scaffolding — recorder + session-token classifier

**Files:**
- Create: `scripts/audit/_audit_lib.py`
- Create: `scripts/audit/__init__.py` (empty)
- Create: `tests/audit/test_audit_lib.py`
- Create: `docs/audit/.gitkeep`, `data/cache/audit/.gitkeep`

**Interfaces:**
- Produces: `record(measurement_id: str, domain: str, metric: str, value, unit: str, command: str, script: str, evidence: str = "", notes: str = "") -> None` (appends/overwrites a row in `docs/audit/measurements.csv`); `REPO` (pathlib.Path repo root); `classify_token(s: str) -> str` returning one of `{"8digit","6digit","7digit-stripped","00-padded","float-string","suffixed","subject-prefixed","other"}`; `canonical(s: str) -> str` (thin wrapper over `visdetect.analysis.config.canonical_session_id`).
- Consumes: nothing.

- [ ] **Step 1: Verify the editable install works (no sys.path hacks needed)**

Run: `py -c "import visdetect, sys; print(visdetect.__file__); print(sys.version)"`
Expected: a path under `...\src\visdetect\__init__.py`, Python 3.10.x. If this fails, STOP — fix
the venv before anything else (`py -m pip install -e .` from repo root).

- [ ] **Step 2: Write the failing tests**

```python
# tests/audit/test_audit_lib.py
import csv, importlib, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit._audit_lib import classify_token, record, MEASUREMENTS_CSV


def test_classify_token_trio_and_edges():
    assert classify_token("01072025") == "8digit"
    assert classify_token("1072025") == "7digit-stripped"
    assert classify_token("1072025.0") == "float-string"
    assert classify_token("050325") == "6digit"
    assert classify_token("00050325") == "00-padded"
    assert classify_token("23042025_v2") == "suffixed"
    assert classify_token("BG_046_01072025") == "subject-prefixed"
    assert classify_token("garbage") == "other"


def test_record_appends_and_overwrites(tmp_path, monkeypatch):
    target = tmp_path / "m.csv"
    monkeypatch.setattr("audit._audit_lib.MEASUREMENTS_CSV", target)
    record("t.one", "D1", "demo", 42, "count", "py x.py", "x.py")
    record("t.one", "D1", "demo", 43, "count", "py x.py", "x.py")  # overwrite same id
    rows = list(csv.DictReader(target.open()))
    assert len(rows) == 1 and rows[0]["value"] == "43"
    assert rows[0]["measurement_id"] == "t.one"
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `py -m pytest tests/audit/test_audit_lib.py -v`
Expected: FAIL (ModuleNotFoundError: audit).

- [ ] **Step 4: Implement `_audit_lib.py`**

```python
# scripts/audit/_audit_lib.py
"""Shared audit harness: measurement recorder + session-token classifier.

Audit rule: every measurement lands in docs/audit/measurements.csv through
record(), never by hand, so the executive summary can cite ids (spec A6).
"""
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEASUREMENTS_CSV = REPO / "docs" / "audit" / "measurements.csv"
_FIELDS = ["measurement_id", "domain", "metric", "value", "unit",
           "command", "script", "evidence", "notes"]


def record(measurement_id, domain, metric, value, unit, command, script,
           evidence="", notes=""):
    MEASUREMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if MEASUREMENTS_CSV.exists():
        with MEASUREMENTS_CSV.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if r["measurement_id"] != measurement_id]
    rows.append({"measurement_id": measurement_id, "domain": domain,
                 "metric": metric, "value": str(value), "unit": unit,
                 "command": command, "script": script,
                 "evidence": evidence, "notes": notes})
    rows.sort(key=lambda r: (r["domain"], r["measurement_id"]))
    with MEASUREMENTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def classify_token(s):
    s = str(s)
    if re.fullmatch(r"BG_\d{3}_\d{6,8}(_[A-Za-z0-9]+)*", s):
        return "subject-prefixed"
    if re.fullmatch(r"\d{6,8}_[A-Za-z0-9_]+", s):
        return "suffixed"
    if re.fullmatch(r"\d+\.0", s):
        return "float-string"
    if re.fullmatch(r"00\d{6}", s):
        return "00-padded"
    if re.fullmatch(r"\d{8}", s):
        return "8digit"
    if re.fullmatch(r"\d{7}", s):
        return "7digit-stripped"
    if re.fullmatch(r"\d{6}", s):
        return "6digit"
    return "other"


def canonical(s):
    from visdetect.analysis.config import canonical_session_id
    return canonical_session_id(s)
```

Note the ordering trap encoded in the tests: `00-padded` must be checked before `8digit`
(`00050325` matches both), and `float-string` before the digit classes.

- [ ] **Step 5: Run tests, verify pass**

Run: `py -m pytest tests/audit/test_audit_lib.py -v` → PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/ tests/audit/ docs/audit/.gitkeep data/cache/audit/.gitkeep
git commit -m "feat(audit): measurement recorder + session-token classifier (Task 1)"
```

---

### Task 2: The approved SKIP_DIRS fix + true guardrail count

**Files:**
- Modify: `scripts/qc/check_refactor_guardrails.py` (the `SKIP_DIRS` set, ~line 35–41)
- Create: `scripts/audit/d5_guardrail_count.py`

**Interfaces:**
- Consumes: `record` from Task 1.
- Produces: measurement ids `d5.guardrail.before`, `d5.guardrail.after`, `d5.syspath.real`.

- [ ] **Step 1: Record the BEFORE count**

Run: `py scripts/qc/check_refactor_guardrails.py > data/cache/audit/guardrails_before.txt; echo "exit=$?"`
Expected: exit=1, ~1,375 HARD violations (recon baseline; 84 % phantom from `.claude/worktrees`).

- [ ] **Step 2: Apply the one-line fix (the audit's sole modification)**

In `check_refactor_guardrails.py`, add to `SKIP_DIRS`:
```python
    ".claude", "_preserved_from_worktrees_20260628",
```

- [ ] **Step 3: Record the AFTER count and write the measurement script**

```python
# scripts/audit/d5_guardrail_count.py
"""D5: true guardrail violation count after the approved SKIP_DIRS fix."""
import re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

out = subprocess.run(
    [sys.executable, str(REPO / "scripts/qc/check_refactor_guardrails.py")],
    capture_output=True, text=True)
hard = re.search(r"HARD violations \((\d+)\)", out.stdout)
n = int(hard.group(1)) if hard else -1
(REPO / "data/cache/audit/guardrails_after.txt").write_text(out.stdout, encoding="utf-8")
record("d5.guardrail.after", "D5", "real HARD violations after .claude excluded", n,
       "count", "py scripts/audit/d5_guardrail_count.py", "d5_guardrail_count.py",
       "scripts/qc/check_refactor_guardrails.py", "recon predicted ~218")
print("HARD after fix:", n)
```

Run: `py scripts/audit/d5_guardrail_count.py`
Expected: a number near 218 (recon prediction). Manually record the before count too:
`py -c "import sys; sys.path.insert(0,'scripts'); from audit._audit_lib import record; record('d5.guardrail.before','D5','HARD violations before fix',1375,'count','see data/cache/audit/guardrails_before.txt','manual','scripts/qc/check_refactor_guardrails.py:35')"`

- [ ] **Step 4: Commit (fix and measurement in separate commits)**

```bash
git add scripts/qc/check_refactor_guardrails.py
git commit -m "fix(qc): exclude .claude and preserved-worktree trees from guardrail scan

The audit's sole approved modification (spec section 6): 84% of reported
violations were phantom copies from .claude/worktrees, which is why the gate
was never wired up."
git add scripts/audit/d5_guardrail_count.py docs/audit/measurements.csv data/cache/audit/
git commit -m "feat(audit): D5 guardrail before/after counts (Task 2)"
```

---

### Task 3: D1 — constants census (static)

**Files:**
- Create: `scripts/audit/d1_constants_census.py`
- Create: `docs/audit/01-constants.md` (started here, finished Task 4)

**Interfaces:**
- Consumes: `record`, `REPO`.
- Produces: `data/cache/audit/constants_census.csv` with columns
  `name,defined_in,reexported_by_config,n_importers,n_retype_sites,retypes_agree,bucket` —
  consumed by Task 15's register. Measurement ids `d1.constants.*`.

- [ ] **Step 1: Write the census script**

```python
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

all_defs = {}          # name -> list of (path, value_src)
importers = {}         # canonical name -> set of files importing it
for f in py_files():
    consts = module_constants(f)
    for name, v in consts.items():
        all_defs.setdefault(name, []).append((f, v))
    src = f.read_text(encoding="utf-8", errors="replace")
    for name in canon:
        if re.search(rf"\bimport\b[^\n]*\b{name}\b", src) or \
           re.search(rf"\b(constants|config)\.{name}\b", src):
            importers.setdefault(name, set()).add(f)

rows = []
for name, val in sorted(canon.items()):
    defs = all_defs.get(name, [])
    shadow = [(p, v) for p, v in defs
              if "src\\visdetect\\analysis\\constants.py" not in str(p)]
    agree = all(v == val for _, v in shadow) if shadow else True
    rows.append({
        "name": name, "defined_in": "constants.py",
        "reexported_by_config": name in cfg or f" {name}" in cfg_src,
        "n_importers": len(importers.get(name, set())),
        "n_retype_sites": len(shadow), "retypes_agree": agree,
        "bucket": "canonical"})

# divergent multi-file names (not in canon)
for name, defs in sorted(all_defs.items()):
    if name in canon or len({str(p) for p, _ in defs}) < 2:
        continue
    values = {v for _, v in defs}
    if len(values) < 2:
        continue
    is_path = any(k in name for k in ("DIR", "PATH", "ROOT", "FILE", "OUT"))
    rows.append({"name": name, "defined_in": ";".join(sorted({str(p.relative_to(REPO)) for p, _ in defs})[:6]),
                 "reexported_by_config": False, "n_importers": 0,
                 "n_retype_sites": len(defs), "retypes_agree": False,
                 "bucket": "path-alias" if is_path else "divergent-parameter"})

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
print(f"canon={len(canon)} dead={dead} not_reexported={not_reexported} "
      f"disagree={disagreeing} divergent_params={div_param}")
```

- [ ] **Step 2: Run and sanity-check against recon baselines**

Run: `py scripts/audit/d1_constants_census.py`
Expected: canon ≈ 82, dead ≈ 22, not_reexported ≈ 42 (recon numbers; deviations are fine but must
be explained in the report). Spot-check one row by hand: `CHANGE_SIZES` must show as *not* in
constants.py-canon (it lives in config.py) — if the census says otherwise, the script is wrong.

- [ ] **Step 3: Start `docs/audit/01-constants.md`** — header + the census table summary +
`file:line` for the worst 10 disagreeing names (pull from the CSV; each row cites its
`defined_in`). Leave a `## Executed measurements` section for Task 4.

- [ ] **Step 4: Commit**

```bash
git add scripts/audit/d1_constants_census.py docs/audit/01-constants.md docs/audit/measurements.csv data/cache/audit/constants_census.csv
git commit -m "feat(audit): D1 constants census - dead/not-reexported/disagreeing counts (Task 3)"
```

---

### Task 4: D1 — executed measurements on real data (QC profiles, FR floors, ref trials, TF period)

**Files:**
- Create: `scripts/audit/d1_executed_checks.py`
- Modify: `docs/audit/01-constants.md` (finish)

**Interfaces:**
- Consumes: `record`; `visdetect.suite.loader.load_session`; `visdetect.core.qc`
  (`load_qc_profile`, `apply_unit_filters`, `find_good_stable_units`);
  `visdetect.analysis.utils.get_good_cluster_ids`.
- Produces: ids `d1.qcprofile.*`, `d1.frfloor.*`, `d1.ref.*`, `d1.tfperiod.*` — these decide two
  quarantine entries (Task 15).

- [ ] **Step 1: Write the script**

```python
# scripts/audit/d1_executed_checks.py
"""D1 executed checks on ONE real session (BG_046 01072025 - also exercises the
day-1-9 id path) + ref-trial check over 5 sessions.

(1) load_qc_profile for all 4 named profiles -> recorded dicts
(2) unit count per selection path on the same session
(3) ref trials: was the change presented? (RT relative to change_time)
(4) TF sample period measured from the stimulus log
"""
import gc, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

from visdetect.core.qc import load_qc_profile, apply_unit_filters, find_good_stable_units
from visdetect.suite.loader import load_session
from visdetect.analysis.utils import get_good_cluster_ids

CMD = "py scripts/audit/d1_executed_checks.py"
S = "d1_executed_checks.py"

# --- (1) profiles ---
for name in ["default", "qc_only", "striatal_strict", "striatal_lenient"]:
    prof = load_qc_profile(name)
    record(f"d1.qcprofile.{name}", "D1", f"load_qc_profile('{name}') returned dict",
           repr(prof), "dict", CMD, S, "src/visdetect/core/qc.py:218",
           "empty dict == silent no-op defect confirmed")

# --- (2) unit counts per path on one session ---
sess = load_session("01072025")
n_good_and_stable = len(getattr(sess, "good_and_stable_ids", []) or [])
n_getgood_1hz = len(get_good_cluster_ids(sess, min_rate_hz=1.0))
n_getgood_01hz = len(get_good_cluster_ids(sess, min_rate_hz=0.1))
record("d1.frfloor.good_and_stable", "D1", "units via good_and_stable_ids (0.5Hz ingest gate)",
       n_good_and_stable, "units", CMD, S, "core/qc.py:269 find_good_stable_units")
record("d1.frfloor.getgood_1hz", "D1", "units via get_good_cluster_ids(min 1.0Hz)",
       n_getgood_1hz, "units", CMD, S, "analysis/utils.py:216")
record("d1.frfloor.getgood_01hz", "D1", "units via get_good_cluster_ids(min 0.1Hz, yml value)",
       n_getgood_01hz, "units", CMD, S, "config/qc_profiles.yml:8")
record("d1.frfloor.spread", "D1", "unit-count spread across live selection paths (session 01072025)",
       f"{n_good_and_stable}/{n_getgood_1hz}/{n_getgood_01hz}", "units", CMD, S,
       notes="one session, three floors, three different populations")

# --- (3) ref trials across 5 sessions ---
REF_SESSIONS = ["01072025", "23062025", "08072025", "15072025", "30062025"]
del sess; gc.collect()
tot_ref, ref_with_change, rts = 0, 0, []
for sname in REF_SESSIONS:
    try:
        s = load_session(sname)
    except Exception as e:
        print(f"skip {sname}: {e}"); continue
    for t in s.trials:
        if str(getattr(t, "outcome", getattr(t, "trialoutcome", ""))).lower() == "ref":
            tot_ref += 1
            ct = getattr(t, "change_time", None)
            rt = getattr(t, "RT", getattr(t, "rt", None))
            if ct is not None and not (isinstance(ct, float) and np.isnan(ct)):
                ref_with_change += 1
                if rt is not None and not (isinstance(rt, float) and np.isnan(rt)):
                    rts.append(float(rt))
    del s; gc.collect()
record("d1.ref.total", "D1", "ref trials across 5 sessions", tot_ref, "trials", CMD, S)
record("d1.ref.with_change_time", "D1", "ref trials with a valid change_time",
       ref_with_change, "trials", CMD, S,
       notes="if ~=total, the change WAS presented on ref trials -> "
             "CHANGE_PRESENTED_OUTCOMES incl. Ref is factually right and "
             "EVENT_VALID_OUTCOMES excluding ref is a scientific choice, not a fact")
if rts:
    record("d1.ref.rt_median_ms", "D1", "median RT on ref trials (from change)",
           round(1000 * float(np.median(rts))), "ms", CMD, S,
           notes="small positive RT = lick AFTER change onset = reflex")

# --- (4) TF sample period from the stimulus log ---
s = load_session("01072025")
period = None
for t in s.trials[:80]:
    vbl = getattr(t, "stim_vbl", None)
    tfd = getattr(t, "stim_tf_disp", None)
    if vbl is None or tfd is None:
        continue
    vbl, tfd = np.asarray(vbl, float).ravel(), np.asarray(tfd, float).ravel()
    if len(vbl) < 20 or len(tfd) < 20:
        continue
    changes = np.where(np.diff(tfd) != 0)[0]
    if len(changes) > 5:
        period = float(np.median(np.diff(vbl[changes])))
        break
if period is not None:
    record("d1.tfperiod.measured_s", "D1", "measured TF update period from stim log",
           round(period, 4), "s", CMD, S, "constants.py:113 TF_SAMPLE_PERIOD=0.25",
           notes="expected ~0.05; 0.25 is the known-wrong canonical value")
else:
    record("d1.tfperiod.measured_s", "D1", "measured TF update period", "not-measured",
           "s", CMD, S, notes="stim logs None on this pkl (legacy, pre-backfill); "
           "fall back to psychophysical_kernel.py:18 documentary evidence")
del s; gc.collect()
print("done")
```

- [ ] **Step 2: Run**

Run: `py scripts/audit/d1_executed_checks.py`
Expected: all four profile rows show `{}` (confirming the defect); three distinct unit counts;
`ref_with_change ≈ tot_ref` with small positive median RT (settling the quarantine); TF period
≈ 0.05 s or an honest `not-measured` with the fallback noted. If `load_session("01072025")` fails
on the id, that is itself a finding — record it and use the zero-padded literal path.

- [ ] **Step 3: Finish `01-constants.md`** — executed-measurements section citing the ids;
verdict paragraphs for: qc-profile no-op (blast radius = every `--profile` invocation), FR-floor
spread, the ref-trial resolution, TF period. Palette census gets one summary line (the 717-hex /
194-distinct recon numbers) with `d1.palette.*` rows recorded via a 10-line grep script inline:
`git grep -oh "#[0-9a-fA-F]\{6\}" -- scripts/ | sort | uniq -c | sort -rn > data/cache/audit/hexes.txt`
and record top-line counts manually with `record(...)` one-liners.

- [ ] **Step 4: Commit**

```bash
git add scripts/audit/d1_executed_checks.py docs/audit/01-constants.md docs/audit/measurements.csv data/cache/audit/
git commit -m "feat(audit): D1 executed checks - qc-profile no-op, FR floors, ref trials, TF period (Task 4)"
```

---

### Task 5: D2 — layering, imports, packaging

**Files:**
- Create: `scripts/audit/d2_layering.py`
- Create: `docs/audit/02-layering.md`

**Interfaces:**
- Consumes: `record`, `REPO`.
- Produces: ids `d2.*`; `data/cache/audit/syspath_sites.csv` (`file,line,target,category`)
  consumed by Task 15.

- [ ] **Step 1: Write the script** — four independent measurements in one file:

```python
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
```

- [ ] **Step 2: Run** — `py scripts/audit/d2_layering.py`. Baselines: 9 src-importers / 7 mixed;
~228 sites with 17 foreign-missing; 3 upward module-level edges (`video_sync.py:69,105`,
`peak_channel.py:10`); `constants` import ≥ 2 s.

- [ ] **Step 3: Wheel + clean-venv check (shell, recorded manually)**

```bash
cd /e/python_analysis/git_repos/vis_detect_analysis_Sep2025
py -m pip wheel . --no-deps -w data/cache/audit/wheel 2>&1 | tail -2
py -m venv data/cache/audit/cleanvenv
data/cache/audit/cleanvenv/Scripts/python -m pip install data/cache/audit/wheel/*.whl -q
data/cache/audit/cleanvenv/Scripts/python -c "import visdetect.viz" ; echo "viz-exit=$?"
```
Expected: `viz-exit=1` (ModuleNotFoundError — `viz`/`integrations` lack `__init__.py`, dropped by
`find_packages`). Record via one-liner: id `d2.packaging.viz_missing`, value = the exit code, note
"50 importers break on any non-editable install".

- [ ] **Step 4: Write `02-layering.md`** citing the ids; then commit:

```bash
git add scripts/audit/d2_layering.py docs/audit/02-layering.md docs/audit/measurements.csv data/cache/audit/syspath_sites.csv
git commit -m "feat(audit): D2 layering - dual roots, syspath classes, upward edges, import cost, packaging break (Task 5)"
```

---

### Task 6: D3 — scripts tree: date parsers, zfill dtypes, partial_spearman spread, dead writers

**Files:**
- Create: `scripts/audit/d3_scripts_census.py`
- Create: `docs/audit/03-scripts.md`

**Interfaces:**
- Consumes: `record`, `classify_token`.
- Produces: ids `d3.*`; `data/cache/audit/date_parser_sites.csv` for Task 15.

- [ ] **Step 1: Write the script.** Four measurements:

```python
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
    if "__pycache__" in p.parts or str(p).find("audit") >= 0:
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

# (3) partial_spearman spread: three variants, one real input
from scipy.stats import spearmanr
rng_src = REPO / "data/cache/session_sorting/session_group_features.csv"
rows = list(csv.DictReader(rng_src.open(encoding="utf-8")))
x = np.array([float(r["occ_StimSens"]) for r in rows])
y = np.array([float(r["hit_rate_go"]) for r in rows])
z = np.array([float(r["n_trials"]) for r in rows])


def _resid(a, c):
    A = np.column_stack([np.ones_like(c), c])
    return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]


ex, ey = _resid(x, z), _resid(y, z)
v_spear = spearmanr(ex, ey).statistic                       # variant A (2 copies)
v_spear_rank = spearmanr(np.argsort(np.argsort(ex)),
                         np.argsort(np.argsort(ey))).statistic  # rank-then-corr
v_pearson = float(np.corrcoef(ex, ey)[0, 1])                # variant C (learning_continuum)
record("d3.pspearman.spread", "D3",
       "partial_spearman three-variant spread on one shared real input (n=%d)" % len(x),
       f"spearmanr={v_spear:.3f} | rank-corr={v_spear_rank:.3f} | corrcoef={v_pearson:.3f}",
       "rho", CMD, S, "learning_continuum.py:104 vs theta_prototype.py:106",
       notes="corrcoef-on-residuals is a DIFFERENT estimator; spread quantified")

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
print("done")
```

- [ ] **Step 2: Run** — `py scripts/audit/d3_scripts_census.py`. Baselines: 23 strptime + 78
zfill sites; 10 vd_tf writers; the three-variant spread is NEW data — if `corrcoef` differs from
`spearmanr` by > 0.02 on real input, the register entry upgrades from "different estimator in
principle" to "materially different results in practice".

- [ ] **Step 3: Write `03-scripts.md`** (include the import-DAG note: in-degree-0 classification
is deferred to the drop-list task with the census CSVs as input). Commit:

```bash
git add scripts/audit/d3_scripts_census.py docs/audit/03-scripts.md docs/audit/measurements.csv data/cache/audit/date_parser_sites.csv
git commit -m "feat(audit): D3 scripts census - date trio, zfill dtypes, partial_spearman spread, dead writers (Task 6)"
```

---

### Task 7: D4a — session-id integrity extended + join-loss quantification

**Files:**
- Create: `scripts/audit/d4_session_id_integrity.py`
- Create: `docs/audit/04-artefacts.md` (started)

**Interfaces:**
- Consumes: `record`, `classify_token`, `canonical`.
- Produces: ids `d4.ids.*`; `data/cache/audit/csv_key_domains.csv`
  (`file,n_rows,domains,joinable_to_manifest,rows_lost_on_join`).

- [ ] **Step 1: Write the script**

```python
# scripts/audit/d4_session_id_integrity.py
"""D4a: extend the session-id integrity check to FIGURES/ and table_output/,
classify the key domain of every CSV carrying a session-id column, and
QUANTIFY rows lost when joined to the staging manifest."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record, classify_token, canonical

CMD = "py scripts/audit/d4_session_id_integrity.py"; S = "d4_session_id_integrity.py"
ID_COLS = {"session_name", "session", "session_id", "sess", "sid", "session_key", "k"}
ROOTS = ["data", "FIGURES", "table_output"]
SKIP = {"audit", "__pycache__"}

import pandas as pd
manifest = pd.read_csv(REPO / "data/BG_046_staging_manifest.csv", dtype=str)
manifest_keys = {canonical(x) for x in manifest["session_name"]}

out_rows, bad_files, total_bad_rows = [], 0, 0
for root in ROOTS:
    for f in (REPO / root).rglob("*.csv"):
        if any(x in f.parts for x in SKIP) or f.stat().st_size > 200_000_000:
            continue
        try:
            head = pd.read_csv(f, nrows=5, dtype=str)
        except Exception:
            continue
        col = next((c for c in head.columns if c.lower() in ID_COLS), None)
        if col is None:
            continue
        df = pd.read_csv(f, usecols=[col], dtype=str)
        toks = df[col].dropna()
        domains = toks.map(classify_token).value_counts().to_dict()
        n_bad = sum(v for k, v in domains.items()
                    if k in ("7digit-stripped", "float-string", "00-padded"))
        # join-loss vs manifest (BG_046-keyed files only; others report n/a)
        canon_toks = toks.map(lambda x: canonical(x))
        lost = int((~canon_toks.isin(manifest_keys)).sum()) if "BG_046" in str(f) or root == "data" else -1
        out_rows.append([str(f.relative_to(REPO)), len(toks), str(domains),
                         lost >= 0, max(lost, 0)])
        if n_bad:
            bad_files += 1; total_bad_rows += n_bad

with (REPO / "data/cache/audit/csv_key_domains.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["file", "n_rows", "domains", "joinable_to_manifest", "rows_lost_on_join"])
    w.writerows(out_rows)

record("d4.ids.files_scanned", "D4", "CSV files with a session-id column (data+FIGURES+table_output)",
       len(out_rows), "files", CMD, S, "data/cache/audit/csv_key_domains.csv")
record("d4.ids.files_corrupt", "D4", "files containing stripped/float/00-padded tokens",
       bad_files, "files", CMD, S)
record("d4.ids.rows_corrupt", "D4", "total corrupted-token rows",
       total_bad_rows, "rows", CMD, S,
       notes="recon baseline 15,802 across 6 caches; this extends scope to FIGURES+table_output")
print(f"files={len(out_rows)} corrupt_files={bad_files} corrupt_rows={total_bad_rows}")
```

- [ ] **Step 2: Run** — expect ≥ 15,802 corrupted rows (scope grew), and the three git-tracked
`FIGURES/popgeom_theta/*.csv` deliverables to appear among corrupt files (recon found `00050325`
keys there). Also run the existing red test for the record:
`py -m pytest tests/test_session_id_csv_integrity.py -q > data/cache/audit/integrity_test.txt; echo $?`
and record its failure as `d4.ids.integrity_test_red`.

- [ ] **Step 3: Start `04-artefacts.md`**, commit:

```bash
git add scripts/audit/d4_session_id_integrity.py docs/audit/04-artefacts.md docs/audit/measurements.csv data/cache/audit/csv_key_domains.csv
git commit -m "feat(audit): D4a session-id integrity extended to FIGURES/table_output + join loss (Task 7)"
```

---

### Task 8: D4b — cache staleness ranking, SESSION_FILTER divergence, twin collisions

**Files:**
- Create: `scripts/audit/d4_staleness.py`
- Modify: `docs/audit/04-artefacts.md`

**Interfaces:**
- Consumes: `record`; `visdetect.analysis.config.load_staging_manifest`;
  `visdetect.analysis.evidence_learning_io` (`_pkl_index` collision check by re-implementation).
- Produces: ids `d4.stale.*`, `d4.filter.*`, `d4.twins.*`;
  `data/cache/audit/stale_caches.csv` (ranked).

- [ ] **Step 1: Write the script** — three parts:

```python
# scripts/audit/d4_staleness.py
"""D4b: (1) rank caches by mtime-vs-producing-code (2) SESSION_FILTER
divergence for direct-manifest readers (3) twin date-key collisions
against the real pkl tree."""
import csv, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d4_staleness.py"; S = "d4_staleness.py"

# (1) staleness: cache mtime vs last commit touching the writer that names it
writers = {}   # topic -> newest commit date of any script mentioning data/cache/<topic>
for topic_dir in sorted((REPO / "data/cache").iterdir()):
    if not topic_dir.is_dir() or topic_dir.name == "audit":
        continue
    hits = subprocess.run(["git", "grep", "-l", f"data/cache/{topic_dir.name}",
                           "--", "scripts/", "src/"],
                          capture_output=True, text=True, cwd=REPO).stdout.split()
    newest = ""
    for h in hits:
        d = subprocess.run(["git", "log", "-1", "--format=%cs", "--", h],
                           capture_output=True, text=True, cwd=REPO).stdout.strip()
        newest = max(newest, d)
    files = [p for p in topic_dir.rglob("*") if p.is_file()]
    if not files:
        continue
    newest_artefact = max(datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                          for p in files).date().isoformat()
    writers[topic_dir.name] = (newest, newest_artefact, len(files),
                               newest_artefact < newest)  # artefact older than code
rows = sorted(((t, *v) for t, v in writers.items()), key=lambda r: r[4], reverse=True)
with (REPO / "data/cache/audit/stale_caches.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["topic", "newest_writer_commit", "newest_artefact_mtime", "n_files", "stale"])
    w.writerows(rows)
record("d4.stale.topics", "D4", "cache topics whose newest artefact predates its writer's last commit",
       sum(1 for r in rows if r[4]), "topics", CMD, S, "data/cache/audit/stale_caches.csv",
       notes="mtime heuristic; known-stale tf_responsive must appear here or the metric is wrong")

# (2) SESSION_FILTER divergence
from visdetect.analysis.config import load_staging_manifest
import pandas as pd
filt = set(load_staging_manifest(qc_only=True)["session_name"].astype(str))
raw = set(pd.read_csv(REPO / "data/BG_046_staging_manifest.csv", dtype=str)["session_name"])
record("d4.filter.divergence", "D4",
       "sessions a direct-manifest reader sees that load_staging_manifest(qc_only=True) filters out",
       len(raw - filt), "sessions", CMD, S,
       notes="28 scripts read the CSV directly (recon); each sees this many extra sessions")

# (3) twin collisions against the real pkl tree
from visdetect.analysis.config import parse_session_date  # only to DEMONSTRATE, never to key
coll = {}
for subj_dir in sorted((REPO / "data/pkls").iterdir()):
    if not subj_dir.is_dir():
        continue
    seen = {}
    for p in subj_dir.glob("*.pkl"):
        stem = p.stem.split("_")[0] if p.stem[0].isdigit() else p.stem
        key = stem[-8:] if len(stem) >= 8 else stem
        seen.setdefault(key[-8:], []).append(p.name)
    coll[subj_dir.name] = {k: v for k, v in seen.items() if len(v) > 1}
total = sum(len(v) for v in coll.values())
record("d4.twins.colliding_date_keys", "D4", "date keys with >1 pkl (twins) across subjects",
       total, "keys", CMD, S,
       notes="; ".join(f"{s}:{len(v)}" for s, v in coll.items() if v))
print("done")
```

- [ ] **Step 2: Run and validate** — `tf_responsive` MUST rank as stale (its README says so); if
it does not, the heuristic is broken — fix before committing. BG_012 must show ~9 colliding keys.

- [ ] **Step 3: Extend `04-artefacts.md`**, commit:

```bash
git add scripts/audit/d4_staleness.py docs/audit/04-artefacts.md docs/audit/measurements.csv data/cache/audit/stale_caches.csv
git commit -m "feat(audit): D4b staleness ranking, filter divergence, twin collisions (Task 8)"
```

---

### Task 9: D4c — stratified traceability census (~100 figures, no re-running)

**Files:**
- Create: `scripts/audit/d4_traceability_census.py`
- Modify: `docs/audit/04-artefacts.md` (finish)

**Interfaces:**
- Consumes: `record`.
- Produces: ids `d4.trace.*`; `data/cache/audit/traceability_sample.csv`
  (`figure,topic,method,producer`) — evidence for S5's provenance layer.

- [ ] **Step 1: Write the script**

```python
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

rows = []
for topic in sorted(p for p in (REPO / "FIGURES").iterdir() if p.is_dir()):
    figs = [f for f in topic.rglob("*") if f.suffix.lower() in (".png", ".pdf", ".svg")]
    for fig in random.sample(figs, min(5, len(figs))):
        stem = fig.stem
        method, producer = "untraceable", ""
        hits = subprocess.run(["git", "grep", "-l", stem, "--", "scripts/", "src/"],
                              capture_output=True, text=True, cwd=REPO).stdout.split()
        if hits:
            method, producer = "stem-in-source", hits[0]
        else:
            sidecars = list(fig.parent.glob(fig.stem + "*.json")) + \
                       list(fig.parent.glob(fig.stem + "*_notes.md"))
            if sidecars:
                method, producer = "sidecar", str(sidecars[0].relative_to(REPO))
            else:
                thits = subprocess.run(
                    ["git", "grep", "-l", f"FIGURES/{topic.name}", "--", "scripts/"],
                    capture_output=True, text=True, cwd=REPO).stdout.split()
                if len(thits) == 1:
                    method, producer = "unique-topic-writer", thits[0]
        rows.append([str(fig.relative_to(REPO)), topic.name, method, producer])

with (REPO / "data/cache/audit/traceability_sample.csv").open("w", newline="",
                                                              encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["figure", "topic", "method", "producer"]); w.writerows(rows)
n = len(rows)
untraceable = sum(1 for r in rows if r[2] == "untraceable")
record("d4.trace.sample", "D4", "figures in stratified traceability sample", n,
       "figures", CMD, S, "data/cache/audit/traceability_sample.csv")
record("d4.trace.untraceable_frac", "D4", "fraction of sampled figures with NO identifiable producer",
       round(untraceable / n, 2), "fraction", CMD, S,
       notes="evidence for the S5 provenance layer, not a target (spec D4)")
print(f"sample={n} untraceable={untraceable}")
```

- [ ] **Step 2: Run; finish `04-artefacts.md`** (integrity + staleness + census + the
`tf_responsive` flip-count note: the actual flip count requires re-running the GLM post-lick-fix,
which is compute the audit does not do — record as `d4.tfresp.flips = not-measured`, with the
register entry carrying direction only). Commit:

```bash
git add scripts/audit/d4_traceability_census.py docs/audit/04-artefacts.md docs/audit/measurements.csv data/cache/audit/traceability_sample.csv
git commit -m "feat(audit): D4c stratified traceability census (Task 9)"
```

---

### Task 10: D5 — tests and tooling

**Files:**
- Create: `scripts/audit/d5_test_inventory.py`
- Create: `docs/audit/05-tests-tooling.md`

**Interfaces:**
- Consumes: `record`.
- Produces: ids `d5.tests.*`; `data/cache/audit/test_partition.csv`
  (`test_file,needs_real_data,covers_modules`).

- [ ] **Step 1: Write the script** — partition tests by real-data dependency (grep for
`load_session|\.pkl|staging_manifest|PKL_DIR|data/cache`), map covered `visdetect` modules by
import statements, and name every library module with zero tests:

```python
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
    mods = set(re.findall(r"from (visdetect(?:\.\w+)*) import|import (visdetect(?:\.\w+)*)", src))
    mods = {m for pair in mods for m in pair if m}
    covered |= mods
    rows.append([str(t.relative_to(REPO)), bool(DATA_PAT.search(src)),
                 ";".join(sorted(mods))])
with (REPO / "data/cache/audit/test_partition.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["test_file", "needs_real_data", "covers_modules"]); w.writerows(rows)

lib_modules = {"visdetect." + str(p.relative_to(REPO / "src/visdetect"))[:-3].replace("\\", ".")
               for p in (REPO / "src/visdetect").rglob("*.py") if p.stem != "__init__"}
covered_stems = {m for c in covered for m in [c] } 
untested = sorted(m for m in lib_modules
                  if not any(c == m or c.startswith(m) or m.startswith(c) for c in covered))
record("d5.tests.total", "D5", "test files", len(rows), "files", CMD, S)
record("d5.tests.need_real_data", "D5", "test files requiring real sessions/caches",
       sum(1 for r in rows if r[1]), "files", CMD, S, "data/cache/audit/test_partition.csv")
record("d5.tests.untested_modules", "D5", "library modules with ZERO test coverage",
       len(untested), "modules", CMD, S, notes=";".join(untested[:15]))
print(f"tests={len(rows)} real-data={sum(1 for r in rows if r[1])} untested={len(untested)}")
```

- [ ] **Step 2: Run offline partition once for runtime** —
`py -m pytest tests -q --ignore=tests/audit -m "not slow" --co -q | tail -3` (collection only —
full runs of the real-data tier are out of audit scope; record collection count). Write
`05-tests-tooling.md`: partition table, untested-module list, the Task-2 guardrail numbers, and
the de-facto-gate statement (recon: zero CI/linters/pre-commit; sole hook = delete guard).

- [ ] **Step 3: Commit**

```bash
git add scripts/audit/d5_test_inventory.py docs/audit/05-tests-tooling.md docs/audit/measurements.csv data/cache/audit/test_partition.csv
git commit -m "feat(audit): D5 test inventory - real-data partition + untested modules (Task 10)"
```

---

### Task 11: D6 — AI layer and docs: literal resolver, duplication agreement, dead paths

**Files:**
- Create: `scripts/audit/d6_ai_layer.py`
- Create: `docs/audit/06-ai-layer.md`

**Interfaces:**
- Consumes: `record`.
- Produces: ids `d6.*`; `data/cache/audit/doc_literals.csv`
  (`doc,line,symbol,doc_value,code_value,verdict`), `data/cache/audit/dead_paths.csv`.

- [ ] **Step 1: Write the script** — three checks:

```python
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
    for m in re.findall(r"claude-[a-z0-9\-\.\[\]]+|[Oo]pus[- ][\d.]+", f.read_text(encoding="utf-8", errors="replace")):
        models.append(f"{f.relative_to(REPO)}:{m}")
record("d6.modelids", "D6", "model ids hardcoded in .claude prose",
       len(models), "mentions", CMD, S, notes=";".join(sorted(set(models))[:10]))
print("done")
```

- [ ] **Step 2: Run; write `06-ai-layer.md`** — literal-mismatch table (the CLAUDE.md
constants-table check, now mechanical), dead-path counts, duplication-agreement summary (recon
percentages restated, with the *divergent* GOTCHAS row called out), retraction-marker survey of
`docs/superpowers/` (grep for `RETRACTED|REFUTED|SUPERSEDED` per file → count of terminal-status
files vs total; one `record` line), skill-overlap table (from the seven skill descriptions —
judgment, cite recon). Commit:

```bash
git add scripts/audit/d6_ai_layer.py docs/audit/06-ai-layer.md docs/audit/measurements.csv data/cache/audit/doc_literals.csv data/cache/audit/dead_paths.csv
git commit -m "feat(audit): D6 doc-vs-code literal resolver, dead paths, model ids (Task 11)"
```

---

### Task 12: D7 — work-at-risk inventory + branch disposition draft

**Files:**
- Create: `scripts/audit/d7_work_at_risk.py`
- Create: `docs/audit/07-work-at-risk.md`
- Create: `docs/audit/branch-disposition.md` (DRAFT — owner decides at review)

**Interfaces:**
- Consumes: `record`.
- Produces: ids `d7.*`; the disposition table (deliverable 7).

- [ ] **Step 1: Write the script** — per-worktree gitignored inventory + stash-tag diffs:

```python
# scripts/audit/d7_work_at_risk.py
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d7_work_at_risk.py"; S = "d7_work_at_risk.py"

def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout

# gitignored artefact volume per worktree + primary
wt_root = REPO / ".claude" / "worktrees"
lines = []
for wt in ([REPO] + (sorted(wt_root.iterdir()) if wt_root.exists() else [])):
    for sub in ("data", "FIGURES"):
        p = wt / sub
        if not p.exists() or p.is_symlink():
            continue
        files = [x for x in p.rglob("*") if x.is_file()]
        size = sum(x.stat().st_size for x in files)
        lines.append(f"{wt.name}/{sub}: {len(files)} files, {size/1e9:.1f} GB")
record("d7.gitignored.volume", "D7", "gitignored data/FIGURES volume per tree",
       " | ".join(lines), "inventory", CMD, S,
       notes="no branch migration carries any of this")

# unmerged branches with unique commits
branches = sh("git", "for-each-ref", "--format=%(refname:short)", "refs/heads").split()
uniq = {}
for b in branches:
    if b == "main":
        continue
    n = sh("git", "rev-list", "--count", f"main..{b}").strip()
    cherry = sh("git", "cherry", "main", b)
    real = sum(1 for l in cherry.splitlines() if l.startswith("+"))
    uniq[b] = (n, real)
record("d7.branches.unmerged", "D7", "branches ahead of main (raw/cherry-verified unique)",
       "; ".join(f"{b}:{n}/{r}" for b, (n, r) in uniq.items()), "commits", CMD, S,
       notes="cherry-verified 0 == already applied under a rewritten sha (safe to drop)")

# stash-tags content
for tag in ["pre-tidy-20260628/stash-0", "pre-tidy-20260628/stash-1"]:
    stat = sh("git", "show", "--stat", "--format=%cs %s", tag).strip().splitlines()
    record(f"d7.stash.{tag[-1]}", "D7", f"stash-tag {tag} content",
           " | ".join(stat[:6]), "difftext", CMD, S)
print("done")
```

- [ ] **Step 2: Write `branch-disposition.md`** as a table with the evidence-backed RECOMMENDED
column filled and the DECISION column empty (owner fills at review). Rows: every branch from the
script output plus every untracked file in `git status --short`. Recommended values follow the
migration brief: `feature/camera-tagger-2b` → port-on-first-use (whole subsystem);
`hardening/fa-psth-and-manifest-sort` → carry fix into new-repo foundation (its sort fix is
register entry #6); `feature/fig5eh-preparatory-cellclass` → drop after verifying still-ancestor;
`feature/tf-transient-sustained-spectrum` → drop (cherry-verified applied);
`feature/population-field-plan2` → merge docs to main before freeze; QC1 untracked scripts →
owner decision (live work). Also write `07-work-at-risk.md` around the measurements.

- [ ] **Step 3: Commit**

```bash
git add scripts/audit/d7_work_at_risk.py docs/audit/07-work-at-risk.md docs/audit/branch-disposition.md docs/audit/measurements.csv
git commit -m "feat(audit): D7 work-at-risk inventory + branch disposition draft (Task 12)"
```

---

### Task 13: D6/D7 sibling check + docs-science agreement (small residuals)

**Files:**
- Modify: `docs/audit/06-ai-layer.md`, `docs/audit/07-work-at-risk.md`

**Interfaces:** consumes `record` only.

- [ ] **Step 1: Sibling-repo duplication check** (read-only, shell):

```bash
grep -rlE "canonical_session_id|zfill\(8\)|CHANGE_SIZES|staging_manifest" \
  /e/python_analysis/git_repos/vis_detect_analysis_Apr2023 --include=*.py | head -20
```
Record count as `d7.sibling.duplication` with the file list in notes — external sources of truth
the new repo's design must account for (ADR-011/ADR-015 boundary).

- [ ] **Step 2: docs/science vs memory agreement** — for each of the 12 `docs/science/*.md`
results docs, check whether a superseding/retraction note exists in memory but not in the doc
(the two known cases: transient/sustained state retraction; StimSens-expert refutation). Record
`d6.science.stale_docs` = count of results docs carrying claims later walked back without an
in-doc marker, listing them in notes. This is judgment + grep, not a script — cite both sides
(`docs/science/<file>:line` and the memory note name).

- [ ] **Step 3: Commit**

```bash
git add docs/audit/06-ai-layer.md docs/audit/07-work-at-risk.md docs/audit/measurements.csv
git commit -m "feat(audit): sibling-repo duplication + stale results-doc survey (Task 13)"
```

---

### Task 14: D9 — storage-format spike (HARD time-box: 1 day)

**Files:**
- Create: `scripts/audit/d9_nwb_spike.py`
- Create: `docs/audit/09-storage-spike.md`

**Interfaces:**
- Consumes: `record`; a *scratch* venv (`data/cache/audit/nwbvenv`) — the repo `.venv` is NOT
  modified.
- Produces: ids `d9.*` — the numbers ADR-015 cites.

- [ ] **Step 1: Scratch venv**

```bash
py -m venv data/cache/audit/nwbvenv
data/cache/audit/nwbvenv/Scripts/python -m pip install -q pynwb neuroconv "numpy<3" pandas
data/cache/audit/nwbvenv/Scripts/python -c "import pynwb; print(pynwb.__version__)"
```

- [ ] **Step 2: Write the conversion script.** Minimal direct pynwb (not full NeuroConv
interfaces — the spike measures feasibility and numbers, not production conversion):

```python
# scripts/audit/d9_nwb_spike.py
"""D9: convert 3 sessions (small BG_049, large BG_046, BG_012 twin) to NWB.
Measure size, three read patterns, and round-trip equality.
RUN WITH THE SCRATCH VENV's python, from repo root, PYTHONPATH=src."""
import gc, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from audit._audit_lib import record
from visdetect.core.session import load_session as load_pkl

from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject

TARGETS = [  # (subject, pkl path picked at run time by glob)
    ("BG_049", sorted((REPO / "data/pkls/BG_049").glob("*.pkl"))[0]),
    ("BG_046", REPO / "data/pkls/BG_046/01092025.pkl"),
    ("BG_012", next((REPO / "data/pkls/BG_012").glob("*_b.pkl"),
                    sorted((REPO / "data/pkls/BG_012").glob("*.pkl"))[0])),
]
CMD = "nwbvenv python scripts/audit/d9_nwb_spike.py"; S = "d9_nwb_spike.py"

for subj, pkl in TARGETS:
    t0 = time.perf_counter()
    sess = load_pkl(str(pkl))
    load_pkl_s = time.perf_counter() - t0

    nwb = NWBFile(session_description=f"visdetect {subj}", identifier=str(uuid.uuid4()),
                  session_start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
                  session_id=pkl.stem, subject=Subject(subject_id=subj, species="Mus musculus"))
    nwb.add_trial_column("change_size", "TF change ratio")
    nwb.add_trial_column("outcome", "behavioural label")
    for i, t in enumerate(sess.trials):
        ct = getattr(t, "change_time", None) or 0.0
        nwb.add_trial(start_time=float(i), stop_time=float(i) + 1.0,
                      change_size=float(getattr(t, "change_size", 0) or 0),
                      outcome=str(getattr(t, "outcome", getattr(t, "trialoutcome", ""))))
    for c in sess.clusters:
        nwb.add_unit(spike_times=np.asarray(c.spike_times, float))
    out = REPO / "data/cache/audit" / f"{subj}_{pkl.stem}.nwb"
    with NWBHDF5IO(str(out), "w") as io:
        io.write(nwb)
    write_s = time.perf_counter() - t0 - load_pkl_s

    # read patterns
    t0 = time.perf_counter()
    with NWBHDF5IO(str(out), "r") as io:
        f = io.read()
        _trials = f.trials.to_dataframe()          # pattern: trials table alone
    read_trials_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    with NWBHDF5IO(str(out), "r") as io:
        f = io.read()
        _one_unit = f.units["spike_times"][0]      # pattern: one unit's spikes
    read_unit_s = time.perf_counter() - t0

    # round-trip equality on spike times of unit 0
    orig = np.asarray(sess.clusters[0].spike_times, float)
    ok = bool(np.array_equal(orig, np.asarray(_one_unit, float)))

    ratio = out.stat().st_size / pkl.stat().st_size
    record(f"d9.size_ratio.{subj}", "D9", f"NWB/pkl size ratio ({pkl.stem})",
           round(ratio, 2), "ratio", CMD, S, str(out.relative_to(REPO)))
    record(f"d9.readtimes.{subj}", "D9",
           "load_pkl / write_nwb / read_trials / read_one_unit",
           f"{load_pkl_s:.1f}/{write_s:.1f}/{read_trials_s:.2f}/{read_unit_s:.2f}",
           "s", CMD, S,
           notes="pkl load is ALL-OR-NOTHING; NWB unit read is lazy - compare col 1 vs col 4")
    record(f"d9.roundtrip.{subj}", "D9", "spike-time round-trip exact equality",
           ok, "bool", CMD, S)
    del sess; gc.collect()
print("done")
```

- [ ] **Step 3: Run with the scratch venv**

Run: `data/cache/audit/nwbvenv/Scripts/python scripts/audit/d9_nwb_spike.py`
Expected: three size ratios (< 1.0 anticipated — pkls are uncompressed), lazy unit-read ≪ pkl
full-load, `roundtrip=True` for all three. Any failure is a *finding*, recorded, not fixed.
Also record `d9.keep_all_good` (`grep -n "keep_all_good" src/visdetect/core/ingest.py` + whether
Kilosort outputs for one session contain spike times for non-good clusters — check
`spike_clusters.npy` unique count vs pkl cluster count; one `record` line). **Stop at the
time-box** regardless of completeness; write `09-storage-spike.md` with the numbers and the
frame-log open question, and delete the `.nwb` scratch outputs after recording sizes.

- [ ] **Step 4: Commit**

```bash
git add scripts/audit/d9_nwb_spike.py docs/audit/09-storage-spike.md docs/audit/measurements.csv
git commit -m "feat(audit): D9 NWB storage spike - size/read/round-trip measured (Task 14)"
```

---

### Task 15: D8 — known-defect register, module classification, quarantine, drop/cold lists

**Files:**
- Create: `scripts/audit/d8_module_classifier.py`
- Create: `docs/audit/known-defect-register.md`
- Create: `docs/audit/quarantine.md`
- Create: `docs/audit/drop-list.md`
- Create: `docs/audit/cold-list.md`

**Interfaces:**
- Consumes: everything above — census CSVs + measurement ids.
- Produces: the deliverable that gates sub-project 3 (spec A2/A3).

- [ ] **Step 1: Module classifier script** — for each of the ~40 `src/visdetect/analysis` modules
plus `core/*`, grep which defect-implicated symbols it touches:

```python
# scripts/audit/d8_module_classifier.py
"""D8: classify every library module against the defect register.
A module is register-affected if it imports/uses any implicated symbol."""
import csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

DEFECT_SYMBOLS = {
    "qc-profile-noop": r"load_qc_profile|qc_profiles",
    "tf-period-5x": r"TF_SAMPLE_PERIOD",
    "session-order": r"parse_session_date",
    "id-corruption": r"session_name|canonical_session_id|zfill",
    "lick-channel": r"lick_times|Piezo|lick_channel",
    "stale-tf-registries": r"tf_responsive",
    "alignment-QC1": r"trial_event_index|Change_ON|align_spikes",
    "change-sizes-membership": r"CHANGE_SIZES",
    "ref-ambiguity": r"EVENT_VALID_OUTCOMES|CHANGE_PRESENTED",
    "state-tags": r"state_tags|state_label",
}
rows = []
for p in sorted((REPO / "src/visdetect").rglob("*.py")):
    if p.stem == "__init__":
        continue
    src = p.read_text(encoding="utf-8", errors="replace")
    hits = sorted(k for k, pat in DEFECT_SYMBOLS.items() if re.search(pat, src))
    rows.append([str(p.relative_to(REPO / "src")), ";".join(hits) or "clean"])
with (REPO / "data/cache/audit/module_register_map.csv").open("w", newline="",
                                                              encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["module", "register_entries"]); w.writerows(rows)
record("d8.modules.classified", "D8", "library modules classified against the register",
       len(rows), "modules", "py scripts/audit/d8_module_classifier.py",
       "d8_module_classifier.py", "data/cache/audit/module_register_map.csv")
record("d8.modules.clean", "D8", "modules touching NO register entry",
       sum(1 for r in rows if r[1] == "clean"), "modules",
       "py scripts/audit/d8_module_classifier.py", "d8_module_classifier.py")
print("classified", len(rows))
```

- [ ] **Step 2: Write `known-defect-register.md`** — the 12 master-design seeds + 5 ephys entries
(ADR-018), each row: defect | direction of effect | affected modules (from the classifier CSV) |
affected artefacts | evidence (measurement ids + file:line) | status. **Resolve the two
quarantines with Task-4 data**: if `d1.ref.with_change_time ≈ d1.ref.total`, the ref entry
becomes "change WAS presented; exclusion from Change_ON PETHs is a scientific choice — new repo
must state it as one" (direction: trial counts on hard/fast conditions rise if included).
`CHANGE_SIZES` membership: direction = "any consumer of tf_glm's tuple mixed catch into go-trial
loops; per-consumer check listed". Anything still undetermined stays in `quarantine.md` with the
specific check that settles it.

- [ ] **Step 3: Write `drop-list.md` and `cold-list.md`** — drop-list: cherry-verified-applied
branches, the 7 orphaned `tf_response` leaf scripts, `AI_exploration` references, dead top-level
shims (`src/visdetect/session.py`, `io.py`, 0 importers), `docs/BrainBulb` zips — each with its
evidence. Cold-list seed: every `src/visdetect/analysis` module NOT touched by the currently
live analysis lines (early-lick/QC1, camera, population-field, state labeling, tf_glm) — port on
first use per ADR-020.

- [ ] **Step 4: Commit**

```bash
git add scripts/audit/d8_module_classifier.py docs/audit/known-defect-register.md docs/audit/quarantine.md docs/audit/drop-list.md docs/audit/cold-list.md docs/audit/measurements.csv data/cache/audit/module_register_map.csv
git commit -m "feat(audit): D8 known-defect register + module map + quarantine/drop/cold lists (Task 15)"
```

---

### Task 16: Executive summary + acceptance self-check

**Files:**
- Create: `docs/audit/00-executive-summary.md`

- [ ] **Step 1: Write the summary** — ranked cross-domain findings (every number cited by
`measurement_id`), split into: *must fix before building* vs *made impossible by the new design*
(map each to its ADR). ≤ 300 lines. Include the traceability fraction, the corrupted-row total,
the qc-profile confirmation, the ref-trial resolution, the D9 numbers, and the guardrail
before/after.

- [ ] **Step 2: Acceptance self-check (spec §5)** — verify in writing, in the summary's final
section: A1 (every finding has evidence+command+blast radius — spot-check 5 random measurement
rows), A2 (module map covers all library modules), A3 (every must-differ entry has a direction),
A4 (census fraction reported with sample size), A5 (corpus is the reports, not the raw CSVs —
raw stays in `data/cache/audit/`), A6 (grep the summary for numbers; each traces to an id).

- [ ] **Step 3: Final commit + push**

```bash
git add docs/audit/00-executive-summary.md
git commit -m "docs(audit): executive summary + acceptance self-check (Task 16) - sub-project 0 complete"
git push origin design/new-repo-foundation
```

---

## Self-review notes (performed at plan-writing time)

- **Spec coverage:** D1→Tasks 3–4; D2→5; D3→6; D4→7–9; D5→2+10; D6→11+13; D7→12–13; D8→15;
  D9→14; deliverables 1–8 → Tasks 16/3–14/15/15/15/15/12/15 respectively. The
  `d4.tfresp.flips` item is recorded as `not-measured` by design (needs GLM recompute — out of
  audit scope, direction captured in the register).
- **Placeholders:** none; every script is complete; shell steps carry exact commands.
- **Type consistency:** `record()` signature identical across all consumers; census CSV column
  names checked against their consumers in Tasks 15.
- **Known risk:** Task 4's attribute names (`t.outcome` vs `t.trialoutcome`, `t.RT` vs `t.rt`)
  are handled with `getattr` fallbacks; if a pkl schema surprises us, the script records
  `not-measured` rather than crashing the task.
