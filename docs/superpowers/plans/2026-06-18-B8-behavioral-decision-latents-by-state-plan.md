# B8 — Behavioral Decision-Latents by State: Implementation Plan (Phase 0 + Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the *descriptive* layer of B8 — per-(session×mood) decision-dial scores (sharpness/itchiness/timing) over the learning axis, a clean censored survival-hazard, and a cached per-trial descriptive latent table + presentation-ready figures — so the science question ("which dial does learning turn / which dial do states load on") is answered on robust ground before any generative model.

**Architecture:** A new behavior-only library module `visdetect.analysis.decision_latents` holds all reusable logic (state-label accessor, two-tier session selection, per-trial table builder, censored hazard, the three dial-score functions, table assembler). A thin script under `scripts/analysis/decision_latents/` orchestrates it into figures (top-level `FIGURES/`) + a cached table (`data/cache/`). Everything reuses `visdetect.analysis.behavior` (SDT/psychometrics) and `visdetect.analysis.ddm.build_trial_evidence` (per-trial evidence trace); nothing is duplicated. TDD throughout with synthetic fixtures.

**Tech Stack:** Python 3.10 (`.venv`, invoke via `py`), numpy, pandas, scipy, matplotlib (Agg), pytest. No new dependencies (pyddm already present from B0 but **not used in Phase 1**).

**Spec:** `docs/superpowers/specs/2026-06-18-B8-behavioral-decision-latents-by-state-design.md`

**Scope note — why Phase 1 only:** The spec is two-step and *descriptive-first*: Step 2's regression-accumulator is explicitly **seeded/constrained by Step 1's results and is recovery-gated** (spec §4, §6, §9 — it may even fall back to Step 1). Planning Step 2 before Step 1's outputs exist would be guesswork, so **Phase 2 (the generative accumulator + per-trial generative latents + parameter recovery) gets its own plan written after Phase 1 lands.** Phase 1 is independently shippable and is the spec's must-have deliverable (§9).

## Global Constraints

_Every task implicitly includes these (copied from the spec/CLAUDE.md):_

- **Invoke Python as `py`** (Windows + Git Bash), never `python`.
- **Worktree execution:** this work lives in the git worktree `…/.claude/worktrees/B8-decision-latents` on branch `feature/B8-decision-latents`. The editable `visdetect` install points at the **primary** repo's `src`, so **set `PYTHONPATH=<worktree>/src`** for every `py`/`pytest` invocation or you silently test main's code (`memory/worktree_editable_install_pythonpath`). Gitignored data inputs are **not** in the worktree checkout — junction/copy them (Task 0.0), and **never `rm -rf` without deleting junctions first** (`memory/worktree_realdata_inputs_junctions`).
- **Behavior-only.** No spike data is loaded anywhere in Phase 1. Load behavior/trials, not clusters.
- **Constants from the canonical source** `visdetect.analysis.constants` (e.g. `CHANGE_SIZES`, `FA_RT_SPLIT`, `TF_FAST_THRESH_LOG2`, `TF_SLOW_THRESH_LOG2`). Never hardcode a value that lives there.
- **Integration / hazard grid `dt = 0.05 s`** (the verified 50 ms TF update period — Task 0.1 confirms before any TF-derived quantity).
- **State source = the new labeler tags** at `data/cache/state_tags/BG_046/{session}.csv`, behind one accessor. **Main fits: Impulsive vs StimSens. Disengaged: reported separately. Abort: excluded** (labeler-state `Abort` ≠ trial-outcome `abort`; both dropped).
- **Sessions: two-tier filter, no d′ gate.** Tier 1 = data-integrity floor (valid task recording + min total trials + tag file exists). Tier 2 = d′ as a continuous covariate + `comprehension_flag` as a label. Keep the `load_staging_manifest(qc_only=True)` subset only for a robustness comparison.
- **Trial typing by `change_size`** (go > 1.0; catch ≈ 1.0). `fa` label = anticipatory lick ≠ SDT-FA (catch-trial `hit`).
- **Repo structure (new convention, `memory/feedback_repo_structure_scripts_figures`):** scripts live in **`scripts/analysis/decision_latents/`** (NOT `analysis_suite/`); figures in top-level **`FIGURES/decision_latents/BG_046/`**; caches in **`data/cache/decision_latents/`**. Library code stays in `src/visdetect/`. **Do not use `suite.plotting.save_figure`** (it writes to `analysis_suite/figures`); reuse `suite.plotting.setup_style` for *styling only* and save with the local `save_fig()` helper defined in Task 1.9.
- **Every analysis step saves a presentation-ready figure** to `FIGURES/decision_latents/BG_046/` after `setup_style()`, with a **plain-language title + caption** (`memory/feedback_plain_language_and_save_figures`). Glossary used in captions: *sharpness = how clearly the mouse tells the change happened; itchiness = how trigger-happy it is before evidence; timing = how strongly it expects the change now.*
- **Memory hygiene:** `del sess; gc.collect()` after each session in loops.
- **TDD + frequent commits.** Each task ends green and committed. Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## File Structure

- **Create** `src/visdetect/analysis/decision_latents.py` — all Phase-1 library logic. One responsibility: turn raw behavior + state tags into per-(session×mood) dial scores and a per-trial descriptive table. Functions (signatures locked in Task interfaces): `load_state_labels`, `MAIN_MOODS`/`SEPARATE_MOODS`/`EXCLUDED_MOODS`, `enumerate_valid_sessions`, `session_dprime`, `assign_comprehension_flags`, `build_trial_table`, `censored_hazard`, `sharpness_scores`, `itchiness_scores`, `timing_scores`, `descriptive_cell_table`, `descriptive_latent_table`.
- **Create** `tests/analysis/test_decision_latents.py` — unit tests (synthetic fixtures; no real data).
- **Create** `tests/analysis/conftest.py` *(if absent)* — shared synthetic fixtures (a tagged synthetic session).
- **Create** `scripts/analysis/decision_latents/run_decision_latents_by_state.py` — orchestration: build the all-sessions table (cached), emit figures F1–F5 + F-summary, write the descriptive latent table + stats CSV.
- **Create** `scripts/analysis/decision_latents/_tf_sampling_check.py`, `scripts/analysis/decision_latents/_label_reliability.py` — Phase-0 diagnostics.
- **Modify** `docs/science/QUESTION_INDEX.md` — bump B8 status `spec-draft` → `plan-draft` and add the plan link (final task).
- **Outputs (gitignored):** `data/cache/decision_latents/decision_latents_by_state.csv` (per-trial table), `data/cache/decision_latents/decision_latents_cell_scores.csv`; figures `FIGURES/decision_latents/BG_046/fig_b8_*.png`; stats `FIGURES/decision_latents/BG_046/decision_latents_stats.csv`.

---

## Phase 0 — Environment, data wiring, and prerequisites

### Task 0.0: Make the worktree runnable (PYTHONPATH + data junctions)

**Files:** none (environment only).

**Interfaces:**
- Produces: a shell in which `py`/`pytest` import `visdetect` from the **worktree** `src` and can load BG_046 sessions + state tags.

- [ ] **Step 1: Confirm the import path resolves to the worktree**

Run (from the worktree root):
```bash
WT="$(pwd)"
PYTHONPATH="$WT/src" py -c "import visdetect, os; print(visdetect.__file__)"
```
Expected: a path under `…/.claude/worktrees/B8-decision-latents/src/visdetect/__init__.py`. If it points at the primary repo, `PYTHONPATH` was not honored — fix before proceeding.

- [ ] **Step 2: Junction the big gitignored inputs (pkls), copy the small ones (tags, manifest)**

Run (Git Bash; junctions avoid duplicating ~GBs of pkls):
```bash
WT="$(pwd)"; PRIMARY="E:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
mkdir -p "$WT/data/pkls"
cmd //c mklink /J "$(cygpath -w "$WT/data/pkls/BG_046")" "$(cygpath -w "$PRIMARY/data/pkls/BG_046")"
# state tags: COPY the FULL set (the user tags all sessions externally in primary, see below)
mkdir -p "$WT/data/cache/state_tags"
cp -r "$PRIMARY/data/cache/state_tags/BG_046" "$WT/data/cache/state_tags/BG_046"
cp "$PRIMARY/data/BG_046_staging_manifest.csv" "$WT/data/" 2>/dev/null || true
```

- [ ] **Step 3: Verify a session + its tags load through the library**

Run:
```bash
PYTHONPATH="$WT/src" py -c "
from visdetect.suite.loader import load_session
s = load_session('01072025'); print('trials:', len(s.trials))
import pandas as pd, glob
print('tag files:', len(glob.glob('data/cache/state_tags/BG_046/*.csv')))
"
```
Expected: prints a positive trial count and a tag-file count (~27). If `load_session` fails on paths, the pkl junction is wrong — fix it.

- [ ] **Step 4: Record the environment recipe**

Append a short `## Worktree run recipe` block (the two commands above) to the top of `scripts/analysis/decision_latents/run_decision_latents_by_state.py`'s module docstring later (Task 1.9). No commit yet (no tracked change).

---

### Task 0.1: Verify the 50 ms TF update period (resolve the `BASELINE_STRIDE` doubt)

**Files:**
- Create: `scripts/analysis/decision_latents/_tf_sampling_check.py` (diagnostic; saves a figure).

**Interfaces:**
- Produces: an empirical answer "what is the real per-sample period of `trial.baseline_values`?" → confirms `dt = 0.05` and whether striding is needed (it must **not** be, per `memory/tf_fluctuation_50ms_vs_constant`).

- [ ] **Step 1: Write the diagnostic**

```python
"""B8 prereq: verify the TF baseline-vector sample period (should be ~50 ms).

Plain English: each trial stores a vector of temporal-frequency values shown
during the baseline. We need to know how many milliseconds each value covers,
so our time grid (dt) matches reality and we don't (as an old script did)
silently sub-sample every 3rd value.
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from visdetect.suite.loader import load_staging_manifest, load_session
from visdetect.suite.plotting import setup_style          # styling only
from visdetect.analysis.config import ROOT, SUBJECT
setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
os.makedirs(FIG_DIR, exist_ok=True)

rows = []
man = load_staging_manifest(qc_only=True)
for sname in man["session_name"].astype(str).head(8):
    s = load_session(sname)
    for t in s.trials:
        bv = getattr(t, "baseline_values", None)
        ct = getattr(t, "change_time", None)
        nseen = getattr(t, "n_seen", None)
        if bv is None or ct is None or not nseen:
            continue
        # period implied if n_seen samples fill the pre-change window [0, change_time]
        rows.append(ct / nseen)
    del s
periods = np.asarray(rows, float)
periods = periods[np.isfinite(periods) & (periods > 0)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(periods * 1000, bins=60)
ax.axvline(50, color="r", ls="--", label="50 ms (expected)")
ax.set_xlabel("implied TF sample period (ms)"); ax.set_ylabel("trials")
ax.set_title("B8 prereq — TF baseline sample period\n(should peak at 50 ms)")
ax.legend(frameon=False)
fig.savefig(os.path.join(FIG_DIR, "fig_b8_prereq_tf_sample_period.png"), dpi=300, bbox_inches="tight")
print(f"median implied period: {np.median(periods)*1000:.1f} ms  (n={periods.size})")
```

- [ ] **Step 2: Run it and read the result**

Run: `PYTHONPATH="$(pwd)/src" py scripts/analysis/decision_latents/_tf_sampling_check.py`
Expected: median implied period ≈ **50 ms**; figure peak at 50 ms. **Decision rule:** if ≈50 ms → use `dt=0.05` with **no striding** (one `baseline_values` sample per 50 ms bin). If materially different, STOP and reconcile with `n_seen` semantics before continuing (record the finding in the plan).

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/decision_latents/_tf_sampling_check.py
git commit -m "feat(b8): TF baseline sample-period diagnostic (confirm 50 ms grid)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 0.2: Label-reliability check (tagging done externally — see Prerequisite)

**Prerequisite (done by the user in the *primary* repo, NOT a step in this plan):** all BG_046 sessions are state-tagged via `tag_sessions.py` with the full session list passed explicitly to bypass the QC-manifest default:
```bash
py scripts/state_labeling/tag_sessions.py \
  --sessions $(py -c "from visdetect.suite.loader import list_pkl_sessions; print(' '.join(list_pkl_sessions()))") \
  --figures --fig-dir FIGURES/state_labeler/BG_046
```
This task only **verifies** coverage and sanity-checks the out-of-distribution naive-session labels (spec §7).

**Files:**
- Create: `scripts/analysis/decision_latents/_label_reliability.py` (saves a reliability figure + coverage CSV).

**Interfaces:**
- Consumes: the state-tag CSVs in `data/cache/state_tags/BG_046/`; `visdetect.analysis.behavior.compute_session_performance`.
- Produces: a reliability figure (`FIGURES/decision_latents/BG_046/`) + `data/cache/decision_latents/b8_label_coverage.csv`.

- [ ] **Step 1: Write the reliability check (mood proportions + confidence per session)**

```python
"""B8 prereq: sanity-check state labels, esp. on newly-labeled naive sessions.

Plain English: the mood labeler was trained on good-behavior sessions. The
early/naive sessions are 'out of distribution', so before we trust their moods
we look at: how much of each session is each mood, and how confident the
labeler is. Low confidence on the new sessions = treat their moods as shaky.
"""
import os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from visdetect.suite.plotting import setup_style          # styling only
from visdetect.analysis.behavior import compute_session_performance
from visdetect.suite.loader import load_session
from visdetect.analysis.config import ROOT, SUBJECT
setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT); os.makedirs(FIG_DIR, exist_ok=True)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents"); os.makedirs(CACHE_DIR, exist_ok=True)

rows = []
for f in sorted(glob.glob("data/cache/state_tags/BG_046/*.csv")):
    sname = os.path.splitext(os.path.basename(f))[0]
    if not sname.isdigit():
        continue
    df = pd.read_csv(f)
    perf = compute_session_performance(load_session(sname))
    props = df["state_label"].value_counts(normalize=True)
    rows.append({"session": sname, "dprime": perf.get("dprime", np.nan),
                 "mean_conf": df["state_confidence"].mean(),
                 **{m: props.get(m, 0.0) for m in
                    ["Impulsive", "StimSens", "Disengaged", "Abort"]}})
tab = pd.DataFrame(rows).sort_values("session")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(tab["dprime"], tab["mean_conf"])
axes[0].set_xlabel("session d′"); axes[0].set_ylabel("mean label confidence")
axes[0].set_title("Label confidence vs performance\n(low-d′ naive sessions = watch here)")
bottom = np.zeros(len(tab))
for m, c in [("Impulsive", "#d62728"), ("StimSens", "#1f77b4"),
             ("Disengaged", "#9488bf"), ("Abort", "#7f7f7f")]:
    axes[1].bar(range(len(tab)), tab[m], bottom=bottom, label=m, color=c)
    bottom += tab[m].values
axes[1].set_xlabel("session (chronological)"); axes[1].set_ylabel("mood fraction")
axes[1].set_title("Mood composition per session"); axes[1].legend(frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG_DIR, "fig_b8_prereq_label_reliability.png"), dpi=300, bbox_inches="tight")
tab.to_csv(os.path.join(CACHE_DIR, "b8_label_coverage.csv"), index=False)
print(tab.to_string(index=False))
```

- [ ] **Step 2: Run it; eyeball the reliability**

Run: `PYTHONPATH="$(pwd)/src" py scripts/analysis/decision_latents/_label_reliability.py`
Expected: coverage table for ~45 sessions + figure. **Gate:** if newly-labeled naive sessions show pathological label confidence (e.g. mean_conf collapses, or 100% one mood with low confidence), note it — Phase 1 will then treat those sessions at coarse (no-mood) level (spec §7). This is a judgment checkpoint, not an automated pass/fail.

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/decision_latents/_label_reliability.py
git commit -m "feat(b8): label-reliability check on (out-of-distribution) naive sessions

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 1 — Step 1: the descriptive dial decomposition

### Task 1.1: State-label accessor + module skeleton

**Files:**
- Create: `src/visdetect/analysis/decision_latents.py`
- Create: `tests/analysis/conftest.py` (if absent)
- Create: `tests/analysis/test_decision_latents.py`

**Interfaces:**
- Produces:
  - `MAIN_MOODS = ("Impulsive", "StimSens")`, `SEPARATE_MOODS = ("Disengaged",)`, `EXCLUDED_MOODS = ("Abort",)`
  - `load_state_labels(session_name: str, subject: str = "BG_046", tag_dir: str | None = None) -> pd.DataFrame` — index `trial_idx` (int), columns `["state_label", "state_confidence"]`. Resolves the session id by trying both the raw string and `str(int(session_name)).zfill(8)` filename forms (the leading-zero gotcha, `memory/state_labeler_neural_validation_jun2026`). Raises `FileNotFoundError` if no tag file.

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_decision_latents.py
import pandas as pd, pytest
from visdetect.analysis import decision_latents as dl

def test_load_state_labels_reads_trial_indexed_moods(tmp_path):
    d = tmp_path / "BG_046"; d.mkdir()
    pd.DataFrame({"trial_idx": [0, 1, 2],
                  "state_label": ["Impulsive", "StimSens", "Disengaged"],
                  "state_confidence": [0.9, 0.8, 0.95]}).to_csv(d / "01072025.csv", index=False)
    out = dl.load_state_labels("01072025", tag_dir=str(tmp_path))
    assert list(out.index) == [0, 1, 2]
    assert out.loc[1, "state_label"] == "StimSens"
    assert dl.MAIN_MOODS == ("Impulsive", "StimSens")

def test_load_state_labels_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dl.load_state_labels("99999999", tag_dir=str(tmp_path))
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -q`
Expected: FAIL (`ModuleNotFoundError: decision_latents` or `AttributeError`).

- [ ] **Step 3: Write the minimal module + accessor**

```python
# src/visdetect/analysis/decision_latents.py
"""B8 — behavioral decision-latents decomposed by state (Phase 1: descriptive).

Plain English: for every trial we want three numbers — 'sharpness' (how clearly
the mouse can tell the grating changed), 'itchiness' (how trigger-happy it is
before any real change), and 'timing' (how strongly it expects the change now).
Phase 1 measures these directly from behaviour, split by the mouse's mood
(Impulsive vs StimSens), across learning. No model fitting here.
"""
from __future__ import annotations
import os
import pandas as pd

MAIN_MOODS = ("Impulsive", "StimSens")
SEPARATE_MOODS = ("Disengaged",)
EXCLUDED_MOODS = ("Abort",)
_DEFAULT_TAG_DIR = os.path.join("data", "cache", "state_tags")

def load_state_labels(session_name, subject="BG_046", tag_dir=None):
    base = os.path.join(tag_dir or _DEFAULT_TAG_DIR, subject)
    candidates = [str(session_name)]
    try:
        candidates.append(str(int(session_name)).zfill(8))  # leading-zero form
    except (TypeError, ValueError):
        pass
    for cand in candidates:
        path = os.path.join(base, f"{cand}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[df["trial_idx"].notna()].copy()
            df["trial_idx"] = df["trial_idx"].astype(int)
            return df.set_index("trial_idx")[["state_label", "state_confidence"]]
    raise FileNotFoundError(f"No state-tag file for {session_name} under {base}")
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/decision_latents.py tests/analysis/test_decision_latents.py
git commit -m "feat(b8): decision_latents module + state-label accessor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.2: Two-tier session selection, per-session d′, comprehension flag

**Files:**
- Modify: `src/visdetect/analysis/decision_latents.py`
- Modify: `tests/analysis/test_decision_latents.py`

**Interfaces:**
- Consumes: `load_state_labels`; `visdetect.analysis.behavior.compute_session_performance`.
- Produces:
  - `enumerate_valid_sessions(subject="BG_046", tag_dir=None, min_total_trials=50) -> list[str]` — every session with a tag file (Tier-1 integrity floor; sorted chronologically by date).
  - `session_dprime(session) -> float` — thin wrapper over `compute_session_performance(session)["dprime"]`.
  - `assign_comprehension_flags(dprime_by_session: dict[str, float], threshold: float = 0.5) -> dict[str, str]` — first chronological session whose d′ ≥ `threshold` (and stays ≥ for the rest) marks the pre→post boundary; returns `{session: "pre"|"post"}`. (`threshold=0.5` is the low "knows-the-rule" bar, distinct from the QC `0.8`; spec §7.)

- [ ] **Step 1: Write the failing tests**

```python
from visdetect.analysis import decision_latents as dl

def test_enumerate_valid_sessions_sorted_and_filtered(tmp_path):
    d = tmp_path / "BG_046"; d.mkdir()
    for s in ["30062025", "01072025"]:
        (d / f"{s}.csv").write_text("trial_idx,state_label,state_confidence\n0,Impulsive,0.9\n")
    out = dl.enumerate_valid_sessions(tag_dir=str(tmp_path), min_total_trials=0)
    assert out == ["30062025", "01072025"]  # chronological (30 Jun before 01 Jul)

def test_assign_comprehension_flags_marks_boundary():
    dprime = {"30062025": 0.1, "01072025": 0.2, "02072025": 0.7, "03072025": 0.9}
    flags = dl.assign_comprehension_flags(dprime, threshold=0.5)
    assert flags["30062025"] == "pre" and flags["01072025"] == "pre"
    assert flags["02072025"] == "post" and flags["03072025"] == "post"
```

- [ ] **Step 2: Run to confirm failure**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -q`
Expected: FAIL (`AttributeError: enumerate_valid_sessions`).

- [ ] **Step 3: Implement (reuse `parse_session_date` for chronological order)**

```python
import glob
from visdetect.analysis.behavior import compute_session_performance
from visdetect.analysis.config import parse_session_date  # DDMMYYYY parser

def enumerate_valid_sessions(subject="BG_046", tag_dir=None, min_total_trials=50):
    base = os.path.join(tag_dir or _DEFAULT_TAG_DIR, subject)
    sessions = []
    for path in glob.glob(os.path.join(base, "*.csv")):
        sname = os.path.splitext(os.path.basename(path))[0]
        if not sname.isdigit():               # skip _tag_summary.csv etc.
            continue
        n = sum(1 for _ in open(path)) - 1     # rows minus header (Tier-1 floor)
        if n >= min_total_trials:
            sessions.append(sname)
    return sorted(sessions, key=parse_session_date)

def session_dprime(session):
    # NOTE: the key is "d_prime" (NOT "dprime") — confirmed in behavior.py; wrong key = silent NaN
    return float(compute_session_performance(session).get("d_prime", float("nan")))

def assign_comprehension_flags(dprime_by_session, threshold=0.5):
    ordered = sorted(dprime_by_session, key=parse_session_date)
    flags, comprehended = {}, False
    for s in ordered:
        if (dprime_by_session[s] or 0) >= threshold:
            comprehended = True
        flags[s] = "post" if comprehended else "pre"
    return flags
```

> If `parse_session_date` lives elsewhere, find it with `grep -rn "def parse_session_date" src/` and import accordingly (`visdetect.analysis.config` per CLAUDE.md).

- [ ] **Step 4: Run to confirm pass**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/decision_latents.py tests/analysis/test_decision_latents.py
git commit -m "feat(b8): two-tier session selection + comprehension flag

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.3: Per-trial table builder (behavior + mood + evidence)

**Files:**
- Modify: `src/visdetect/analysis/decision_latents.py`
- Create: `tests/analysis/conftest.py`
- Modify: `tests/analysis/test_decision_latents.py`

**Interfaces:**
- Consumes: `visdetect.utils.synthetic.make_synthetic_session`; `visdetect.analysis.ddm.build_trial_evidence`; `load_state_labels`.
- Produces: `build_trial_table(session, state_labels: pd.DataFrame, session_name: str, dt: float = 0.05) -> pd.DataFrame` — one row per usable trial (drops outcome `abort`/`ref` and mood `Abort`), columns: `session_name, trial_idx, outcome, change_size, change_time_planned, change_reached(bool), decision_time, lick(int), censored(bool), state_label, state_confidence, trial_in_session(int), n_bins(int)`. (`evidence` arrays stay out of the flat table; recomputed in Phase 2.)

- [ ] **Step 1: Shared fixture**

```python
# tests/analysis/conftest.py
import pytest, pandas as pd
from visdetect.utils.synthetic import make_synthetic_session

@pytest.fixture
def synth_session():
    return make_synthetic_session(n_trials=40, n_clusters=2, seed=0)

@pytest.fixture
def synth_state_labels():
    # alternate Impulsive/StimSens, with a couple Disengaged/Abort to exercise filtering
    labels = []
    for i in range(40):
        m = ["Impulsive", "StimSens"][i % 2]
        if i in (5, 6): m = "Disengaged"
        if i in (7,):   m = "Abort"
        labels.append({"trial_idx": i, "state_label": m, "state_confidence": 0.9})
    return pd.DataFrame(labels).set_index("trial_idx")[["state_label", "state_confidence"]]
```

- [ ] **Step 2: Write the failing test**

```python
def test_build_trial_table_filters_and_columns(synth_session, synth_state_labels):
    from visdetect.analysis import decision_latents as dl
    tab = dl.build_trial_table(synth_session, synth_state_labels, "07072025", dt=0.05)
    assert "Abort" not in tab["state_label"].values          # mood Abort dropped
    assert {"sharpness", "itchiness"}.isdisjoint(tab.columns) # not here yet
    for col in ["session_name", "trial_idx", "outcome", "change_size",
                "change_time_planned", "change_reached", "decision_time",
                "lick", "censored", "state_label", "trial_in_session"]:
        assert col in tab.columns
    # change_reached True only for hit/miss
    assert (tab.loc[tab["change_reached"], "outcome"].isin(["hit", "miss"])).all()
    assert tab["trial_in_session"].is_monotonic_increasing
```

- [ ] **Step 3: Run to confirm failure**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -q`
Expected: FAIL (`AttributeError: build_trial_table`).

- [ ] **Step 4: Implement (reuse `ddm.build_trial_evidence` for geometry, then join mood)**

```python
import numpy as np
from visdetect.analysis import ddm

def build_trial_table(session, state_labels, session_name, dt=0.05):
    ev = ddm.build_trial_evidence(session, dt=dt)   # trial_uid, outcome, change_size,
                                                    # change_time, decision_time, lick, censored, evidence
    rows = []
    for _, r in ev.iterrows():
        uid = int(r["trial_uid"])
        mood = state_labels["state_label"].get(uid)
        conf = state_labels["state_confidence"].get(uid)
        if mood in EXCLUDED_MOODS:                  # drop labeler 'Abort'
            continue
        outcome = str(r["outcome"]).lower()
        rows.append({
            "session_name": session_name, "trial_idx": uid, "outcome": outcome,
            "change_size": float(r["change_size"]),
            "change_time_planned": float(r["change_time"]),
            "change_reached": outcome in ("hit", "miss"),
            "decision_time": float(r["decision_time"]),
            "lick": int(r["lick"]), "censored": bool(r["censored"]),
            "state_label": mood, "state_confidence": conf,
            "n_bins": int(len(r["evidence"])),
        })
    tab = pd.DataFrame(rows).sort_values("trial_idx").reset_index(drop=True)
    tab["trial_in_session"] = np.arange(len(tab))   # within-session position (satiety covariate)
    return tab
```

- [ ] **Step 5: Run to confirm pass**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/visdetect/analysis/decision_latents.py tests/analysis/test_decision_latents.py tests/analysis/conftest.py
git commit -m "feat(b8): per-trial table builder (behavior + mood + geometry)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.4: Censored discrete-time survival hazard (clean reimplementation)

**Files:**
- Modify: `src/visdetect/analysis/decision_latents.py`
- Modify: `tests/analysis/test_decision_latents.py`

**Interfaces:**
- Produces: `censored_hazard(durations, events, dt=0.05, t_max=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(bin_centers, hazard, survival)`. At bin *k*: `hazard[k] = (#events in bin k) / (#still at risk at start of bin k)`; `survival = cumprod(1 - hazard)`. A trial with `events[i] = False` is **right-censored** at `durations[i]` (contributes to the risk set up to its bin, never an event) — this is how a planned-but-unreached change (the 15 s-planned, 3 s-FA case, spec §4 / user 2026-06-18) is handled.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

def test_censored_hazard_counts_events_and_censoring():
    from visdetect.analysis import decision_latents as dl
    # 3 trials: event at 0.10s, event at 0.10s, CENSORED at 0.05s (no event)
    dur = np.array([0.10, 0.10, 0.05]); ev = np.array([True, True, False])
    centers, hz, surv = dl.censored_hazard(dur, ev, dt=0.05, t_max=0.15)
    # bin0 [0,0.05): risk=3, events=0 -> hz=0 ; the censored trial leaves after bin0
    assert hz[0] == 0.0
    # bin1 [0.05,0.10): risk=2 (censored gone), events=2 -> hz=1.0
    assert np.isclose(hz[1], 1.0)
    assert np.all(surv <= 1.0) and np.all(np.diff(surv) <= 1e-9)

def test_censored_hazard_survival_is_one_minus_prod():
    from visdetect.analysis import decision_latents as dl
    dur = np.array([0.10, 0.15]); ev = np.array([True, True])
    _, hz, surv = dl.censored_hazard(dur, ev, dt=0.05, t_max=0.20)
    assert np.isclose(surv[-1], np.prod(1 - hz))
```

- [ ] **Step 2: Run to confirm failure**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k hazard -q`
Expected: FAIL (`AttributeError: censored_hazard`).

- [ ] **Step 3: Implement**

```python
def censored_hazard(durations, events, dt=0.05, t_max=None):
    durations = np.asarray(durations, float); events = np.asarray(events, bool)
    if t_max is None:
        t_max = float(np.nanmax(durations)) + dt
    edges = np.arange(0.0, t_max + dt, dt)
    centers = 0.5 * (edges[:-1] + edges[1:])
    event_bin = np.floor(durations / dt).astype(int)   # bin in which the trial ends
    hazard = np.zeros(len(centers))
    for k in range(len(centers)):
        at_risk = np.sum(durations >= edges[k] - 1e-12)        # still running at bin start
        n_event = np.sum(events & (event_bin == k))
        hazard[k] = (n_event / at_risk) if at_risk > 0 else 0.0
    survival = np.cumprod(1.0 - hazard)
    return centers, hazard, survival
```

- [ ] **Step 4: Run to confirm pass**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k hazard -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/decision_latents.py tests/analysis/test_decision_latents.py
git commit -m "feat(b8): censored discrete-time survival hazard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.5: Sharpness scores (psychometric + d′ + RT mean/variability)

**Files:**
- Modify: `src/visdetect/analysis/decision_latents.py`
- Modify: `tests/analysis/test_decision_latents.py`

**Interfaces:**
- Consumes: `visdetect.analysis.behavior.calculate_dprime`; `visdetect.analysis.constants.CHANGE_SIZES`.
- Produces: `sharpness_scores(trial_df) -> dict` with keys `psy_slope` (logistic slope of P(lick on go) vs log2 change_size), `dprime` (go hit-rate vs catch FA-rate via `calculate_dprime`), and per-change-size `rt_mean_cs{c}` / `rt_cv_cs{c}` (Hit RT = `decision_time − change_time_planned`). Operates on one (session×mood) cell's rows.

- [ ] **Step 1: Write the failing test**

```python
def test_sharpness_scores_keys_and_dprime_direction():
    import numpy as np, pandas as pd
    from visdetect.analysis import decision_latents as dl
    rng = np.random.default_rng(0)
    rows = []
    for cs, p in [(1.0, 0.1), (1.25, 0.4), (2.0, 0.8), (4.0, 0.95)]:
        for _ in range(50):
            lick = rng.random() < p
            outcome = "hit" if (cs > 1.0 and lick) else ("fa" if (cs == 1.0 and lick) else "miss")
            ct = 5.0
            rows.append({"change_size": cs, "lick": int(lick), "outcome": outcome,
                         "change_time_planned": ct,
                         "decision_time": ct + rng.uniform(0.2, 0.6) if outcome == "hit" else ct + 2.0})
    sc = dl.sharpness_scores(pd.DataFrame(rows))
    assert "psy_slope" in sc and "dprime" in sc
    assert any(k.startswith("rt_cv_cs") for k in sc)
    assert sc["dprime"] > 0          # more hits on big changes than FAs on catch
```

- [ ] **Step 2: Run to confirm failure**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k sharpness -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
from scipy.optimize import curve_fit
from visdetect.analysis.behavior import calculate_dprime
from visdetect.analysis.constants import CHANGE_SIZES

def _logistic(x, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * x)))

def sharpness_scores(trial_df):
    go = trial_df[trial_df["change_size"] > 1.0]
    catch = trial_df[np.isclose(trial_df["change_size"], 1.0)]
    out = {}
    # psychometric slope: P(lick) vs log2(change_size) on go trials
    if len(go) >= 8 and go["change_size"].nunique() >= 2:
        x = np.log2(go["change_size"].values); y = go["lick"].values.astype(float)
        try:
            (a, b), _ = curve_fit(_logistic, x, y, p0=[0.0, 1.0], maxfev=5000)
            out["psy_slope"] = float(b)
        except Exception:
            out["psy_slope"] = float("nan")
    else:
        out["psy_slope"] = float("nan")
    hit_rate = float(go["lick"].mean()) if len(go) else float("nan")
    fa_rate = float(catch["lick"].mean()) if len(catch) else float("nan")
    out["dprime"] = float(calculate_dprime(hit_rate, fa_rate))
    # per-change-size Hit RT mean + CV
    hits = go[go["outcome"] == "hit"].copy()
    hits["rt"] = hits["decision_time"] - hits["change_time_planned"]
    for cs in CHANGE_SIZES:
        rt = hits.loc[np.isclose(hits["change_size"], cs), "rt"].values
        out[f"rt_mean_cs{cs}"] = float(np.mean(rt)) if rt.size >= 3 else float("nan")
        out[f"rt_cv_cs{cs}"] = float(np.std(rt) / np.mean(rt)) if rt.size >= 3 and np.mean(rt) > 0 else float("nan")
    return out
```

- [ ] **Step 4: Run to confirm pass**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k sharpness -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/decision_latents.py tests/analysis/test_decision_latents.py
git commit -m "feat(b8): sharpness scores (psychometric, d', RT mean/CV)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.6: Itchiness scores (criterion c, FA-rate, baseline hazard)

**Files:**
- Modify: `src/visdetect/analysis/decision_latents.py`
- Modify: `tests/analysis/test_decision_latents.py`

**Interfaces:**
- Consumes: `censored_hazard`; `scipy.stats.norm`.
- Produces: `itchiness_scores(trial_df, dt=0.05) -> dict` with `criterion_c` (SDT criterion `-(z(H)+z(FA))/2`, log-linear corrected), `fa_rate` (fraction of trials with `outcome=="fa"`), `baseline_hazard` (mean lick hazard over the pre-change window across trials, from `censored_hazard` on FA-latency events). Operates on one cell.

- [ ] **Step 1: Write the failing test**

```python
def test_itchiness_scores_more_fa_higher_criterion_shift():
    import numpy as np, pandas as pd
    from visdetect.analysis import decision_latents as dl
    def make(fa_frac):
        rows = []
        for _ in range(200):
            if np.random.default_rng().random() < fa_frac:
                rows.append({"change_size": 2.0, "lick": 1, "outcome": "fa",
                             "decision_time": 1.0, "change_time_planned": 5.0, "censored": False})
            else:
                rows.append({"change_size": 2.0, "lick": 1, "outcome": "hit",
                             "decision_time": 5.3, "change_time_planned": 5.0, "censored": False})
        return pd.DataFrame(rows)
    hi = dl.itchiness_scores(make(0.6)); lo = dl.itchiness_scores(make(0.1))
    assert hi["fa_rate"] > lo["fa_rate"]
    assert "criterion_c" in hi and "baseline_hazard" in hi
```

- [ ] **Step 2: Run to confirm failure**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k itchiness -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
from scipy.stats import norm

def _loglinear(rate, n):
    return (rate * n + 0.5) / (n + 1.0)          # clip to (0,1), avoids inf z

def itchiness_scores(trial_df, dt=0.05):
    go = trial_df[trial_df["change_size"] > 1.0]
    catch = trial_df[np.isclose(trial_df["change_size"], 1.0)]
    H = _loglinear(go["lick"].mean() if len(go) else 0.0, max(len(go), 1))
    FA = _loglinear(catch["lick"].mean() if len(catch) else 0.0, max(len(catch), 1))
    crit = -(norm.ppf(H) + norm.ppf(FA)) / 2.0
    fa_rate = float((trial_df["outcome"] == "fa").mean())
    # baseline lick hazard: FA-latency events vs everything else censored at decision_time
    is_fa = (trial_df["outcome"] == "fa").values
    dur = trial_df["decision_time"].values.copy()
    _, hz, _ = censored_hazard(dur, is_fa, dt=dt)
    return {"criterion_c": float(crit), "fa_rate": fa_rate,
            "baseline_hazard": float(np.nanmean(hz)) if hz.size else float("nan")}
```

- [ ] **Step 4: Run to confirm pass**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k itchiness -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/decision_latents.py tests/analysis/test_decision_latents.py
git commit -m "feat(b8): itchiness scores (criterion c, FA-rate, baseline hazard)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.7: Timing scores (change-onset vs lick hazard, peak/spread)

**Files:**
- Modify: `src/visdetect/analysis/decision_latents.py`
- Modify: `tests/analysis/test_decision_latents.py`

**Interfaces:**
- Consumes: `censored_hazard`.
- Produces:
  - `change_onset_hazard(trial_df, dt=0.05) -> tuple[centers, hazard, survival]` — event = change occurring (`change_reached==True`) at `change_time_planned`; trials without a reached change are censored at `decision_time` (the planned-but-unreached case).
  - `lick_hazard(trial_df, dt=0.05) -> tuple[centers, hazard, survival]` — event = first lick (`lick==1`) at `decision_time`; non-lick trials censored at `decision_time`.
  - `timing_scores(trial_df, dt=0.05) -> dict` with `lick_hazard_peak_time`, `lick_hazard_spread` (std of the hazard-weighted time distribution), `change_hazard_peak_time`, `peak_offset = lick_peak − change_peak` (how far the mouse's licking sits from the true change timing).

- [ ] **Step 1: Write the failing test**

```python
def test_timing_scores_peak_and_offset():
    import numpy as np, pandas as pd
    from visdetect.analysis import decision_latents as dl
    rows = []
    for _ in range(300):
        # changes cluster near 5s; licks (hits) shortly after
        ct = 5.0 + np.random.default_rng().normal(0, 0.2)
        rows.append({"change_reached": True, "change_time_planned": ct,
                     "lick": 1, "outcome": "hit", "decision_time": ct + 0.3})
    sc = dl.timing_scores(pd.DataFrame(rows), dt=0.05)
    assert 4.0 < sc["change_hazard_peak_time"] < 6.0
    assert sc["lick_hazard_peak_time"] >= sc["change_hazard_peak_time"]  # licks after change
    assert "peak_offset" in sc and "lick_hazard_spread" in sc
```

- [ ] **Step 2: Run to confirm failure**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k timing -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def change_onset_hazard(trial_df, dt=0.05):
    reached = trial_df["change_reached"].values.astype(bool)
    dur = np.where(reached, trial_df["change_time_planned"].values,
                   trial_df["decision_time"].values).astype(float)
    return censored_hazard(dur, reached, dt=dt)

def lick_hazard(trial_df, dt=0.05):
    ev = (trial_df["lick"].values.astype(int) == 1)
    return censored_hazard(trial_df["decision_time"].values.astype(float), ev, dt=dt)

def _peak_and_spread(centers, hazard):
    w = np.clip(hazard, 0, None)
    if w.sum() <= 0:
        return float("nan"), float("nan")
    peak = centers[int(np.argmax(hazard))]
    mean = np.average(centers, weights=w)
    spread = float(np.sqrt(np.average((centers - mean) ** 2, weights=w)))
    return float(peak), spread

def timing_scores(trial_df, dt=0.05):
    cc, ch, _ = change_onset_hazard(trial_df, dt=dt)
    lc, lh, _ = lick_hazard(trial_df, dt=dt)
    ch_peak, _ = _peak_and_spread(cc, ch)
    l_peak, l_spread = _peak_and_spread(lc, lh)
    return {"change_hazard_peak_time": ch_peak, "lick_hazard_peak_time": l_peak,
            "lick_hazard_spread": l_spread,
            "peak_offset": (l_peak - ch_peak) if np.isfinite(l_peak) and np.isfinite(ch_peak) else float("nan")}
```

- [ ] **Step 4: Run to confirm pass**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k timing -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/decision_latents.py tests/analysis/test_decision_latents.py
git commit -m "feat(b8): timing scores (change-onset vs lick hazard, peak/spread)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.8: Cell-score table + per-trial descriptive latent table

**Files:**
- Modify: `src/visdetect/analysis/decision_latents.py`
- Modify: `tests/analysis/test_decision_latents.py`

**Interfaces:**
- Consumes: all score functions above.
- Produces:
  - `descriptive_cell_table(all_trials_df) -> pd.DataFrame` — one row per `(session_name, state_label)` cell for moods in `MAIN_MOODS` **and** `SEPARATE_MOODS` (Disengaged reported, flagged via a `reported_separately` bool), with all sharpness/itchiness/timing scores + `n_trials`, `session_dprime`, `comprehension_flag`. Cells with `n_trials < min_cell_trials` (default 20) get `NaN` scores but are kept (flagged `underpowered=True`).
  - `descriptive_latent_table(all_trials_df, cell_table) -> pd.DataFrame` — the per-trial deliverable (spec §5, Step-1 columns): each trial row joined to its cell's `sharpness_psy_slope, rt_cv_by_cs, criterion_c, fa_rate_cell, hazard_peak_cell` + identifiers (`session_dprime, comprehension_flag, state_confidence, trial_in_session, change_time_planned, change_reached, decision_time, lick, censored`).

- [ ] **Step 1: Write the failing test**

```python
def test_cell_and_latent_tables(synth_session, synth_state_labels):
    from visdetect.analysis import decision_latents as dl
    tab = dl.build_trial_table(synth_session, synth_state_labels, "07072025")
    tab["session_dprime"] = 0.9; tab["comprehension_flag"] = "post"
    cells = dl.descriptive_cell_table(tab, min_cell_trials=1)
    assert set(cells["state_label"]).issubset(set(dl.MAIN_MOODS + dl.SEPARATE_MOODS))
    assert {"criterion_c", "psy_slope", "lick_hazard_peak_time", "n_trials"}.issubset(cells.columns)
    lat = dl.descriptive_latent_table(tab, cells)
    assert len(lat) == len(tab)
    assert {"criterion_c", "sharpness_psy_slope", "trial_in_session"}.issubset(lat.columns)
```

- [ ] **Step 2: Run to confirm failure**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k "cell_and_latent" -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def descriptive_cell_table(all_trials_df, min_cell_trials=20, dt=0.05):
    keep = list(MAIN_MOODS) + list(SEPARATE_MOODS)
    rows = []
    for (sname, mood), cell in all_trials_df.groupby(["session_name", "state_label"]):
        if mood not in keep:
            continue
        n = len(cell)
        rec = {"session_name": sname, "state_label": mood, "n_trials": n,
               "reported_separately": mood in SEPARATE_MOODS,
               "underpowered": n < min_cell_trials,
               "session_dprime": cell["session_dprime"].iloc[0],
               "comprehension_flag": cell["comprehension_flag"].iloc[0]}
        if n >= min_cell_trials:
            rec.update(sharpness_scores(cell))
            rec.update(itchiness_scores(cell, dt=dt))
            rec.update(timing_scores(cell, dt=dt))
        rows.append(rec)
    return pd.DataFrame(rows)

def descriptive_latent_table(all_trials_df, cell_table):
    key = ["session_name", "state_label"]
    cols = ["psy_slope", "criterion_c", "fa_rate", "lick_hazard_peak_time"]
    avail = [c for c in cols if c in cell_table.columns]
    joined = all_trials_df.merge(cell_table[key + avail], on=key, how="left")
    return joined.rename(columns={"psy_slope": "sharpness_psy_slope",
                                  "fa_rate": "fa_rate_cell",
                                  "lick_hazard_peak_time": "hazard_peak_cell"})
```

- [ ] **Step 4: Run to confirm pass**

Run: `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect/analysis/decision_latents.py tests/analysis/test_decision_latents.py
git commit -m "feat(b8): cell-score table + per-trial descriptive latent table

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.9: Orchestration script — figures F1–F5 + summary, cached table, stats

**Files:**
- Create: `scripts/analysis/decision_latents/run_decision_latents_by_state.py`

**Interfaces:**
- Consumes: everything in `decision_latents`; `visdetect.suite.loader.load_session`; `visdetect.suite.plotting.setup_style` (styling only); `visdetect.analysis.config.{ROOT,SUBJECT}`.
- Produces: `data/cache/decision_latents/decision_latents_by_state.csv`, `data/cache/decision_latents/decision_latents_cell_scores.csv`, `FIGURES/decision_latents/BG_046/fig_b8_*.png`, `FIGURES/decision_latents/BG_046/decision_latents_stats.csv`.

- [ ] **Step 1: Write the builder (cached) and run it on tagged sessions**

```python
"""B8 Fig: decision-latents by state (Step 1, descriptive).

Plain English: measures three behavioural 'dials' — sharpness (can it tell the
change happened), itchiness (is it trigger-happy), timing (does it expect the
change now) — split by mood (Impulsive vs StimSens), across learning, and
saves them as figures + a per-trial table.

Worktree run recipe:
  WT=$(pwd); PYTHONPATH="$WT/src" py scripts/analysis/decision_latents/run_decision_latents_by_state.py
"""
import os, sys, gc, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style          # styling only
from visdetect.analysis.config import ROOT, SUBJECT, STATE_LABEL_COLORS  # canonical new-labeler mood palette
from visdetect.analysis import decision_latents as dl
setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents")
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)
CACHE = os.path.join(CACHE_DIR, "decision_latents_by_state.csv")        # deliverable: per-trial LATENT table
TRIAL_CACHE = os.path.join(CACHE_DIR, "decision_latents_trialtable.csv")  # raw per-trial table (build() cache)

def save_fig(fig, name):                       # writes to top-level FIGURES/, not analysis_suite/
    p = os.path.join(FIG_DIR, f"{name}.png"); fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig); return p

def build(force=False):
    if os.path.exists(TRIAL_CACHE) and not force:
        return pd.read_csv(TRIAL_CACHE)
    sessions = dl.enumerate_valid_sessions()
    dprime = {}
    parts = []
    for sname in sessions:
        sess = load_session(sname)
        dprime[sname] = dl.session_dprime(sess)
        labels = dl.load_state_labels(sname)
        parts.append((sname, dl.build_trial_table(sess, labels, sname)))
        del sess; gc.collect()
    flags = dl.assign_comprehension_flags(dprime)
    frames = []
    for sname, tab in parts:
        tab["session_dprime"] = dprime[sname]; tab["comprehension_flag"] = flags[sname]
        frames.append(tab)
    all_trials = pd.concat(frames, ignore_index=True)
    all_trials.to_csv(TRIAL_CACHE, index=False)
    return all_trials
```

- [ ] **Step 2: Run the build; sanity-check shape**

Run: `PYTHONPATH="$(pwd)/src" py -c "import sys; sys.argv=['x']; import importlib.util as u; m=u.spec_from_file_location('b8','scripts/analysis/decision_latents/run_decision_latents_by_state.py'); mod=u.module_from_spec(m); m.loader.exec_module(mod); df=mod.build(force=True); print(df.shape, df['state_label'].value_counts().to_dict())"`
Expected: a few-thousand-row table; `state_label` counts show Impulsive/StimSens/Disengaged, **no Abort**.

- [ ] **Step 3: Add the figure functions (F1–F5 + summary) and `__main__`**

Append plain-language-captioned panels. Each uses the local `save_fig(fig, name)` helper (writes to `FIGURES/decision_latents/BG_046/`):
```python
def fig_sharpness(cells):                 # F1
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for mood, c in [(m, STATE_LABEL_COLORS[m]) for m in ("Impulsive", "StimSens")]:
        sub = cells[cells["state_label"] == mood].sort_values("session_dprime")
        ax[0].plot(sub["session_dprime"], sub["psy_slope"], "o-", color=c, label=mood)
    ax[0].set_xlabel("session d′ (learning →)"); ax[0].set_ylabel("psychometric slope")
    ax[0].set_title("F1  Sharpness rises with learning\n(steeper = tells the change apart better)")
    ax[0].legend(frameon=False)
    rt_cols = [k for k in cells.columns if k.startswith("rt_cv_cs")]
    if rt_cols:
        ax[1].plot(cells.sort_values("session_dprime")["session_dprime"],
                   cells.sort_values("session_dprime")[rt_cols].mean(axis=1), "o-")
        ax[1].set_title("F2  RT variability shrinks with learning")
        ax[1].set_xlabel("session d′"); ax[1].set_ylabel("mean RT CV (across change sizes)")
    return save_fig(fig, "fig_b8_F1_F2_sharpness")

def fig_itchiness(cells):                 # F3
    fig, ax = plt.subplots(figsize=(7, 4))
    for mood, c in [(m, STATE_LABEL_COLORS[m]) for m in ("Impulsive", "StimSens")]:
        sub = cells[cells["state_label"] == mood]
        ax.scatter(sub["criterion_c"], sub["fa_rate"], color=c, label=mood)
    ax.set_xlabel("criterion c  (low = trigger-happy)"); ax.set_ylabel("FA rate")
    ax.set_title("F3  Itchiness separates the moods\n(Impulsive = liberal criterion, more early licks)")
    ax.legend(frameon=False)
    return save_fig(fig, "fig_b8_F3_itchiness")

def fig_timing(all_trials):               # F4
    fig, ax = plt.subplots(figsize=(8, 4))
    cc, ch, _ = dl.change_onset_hazard(all_trials)
    lc, lh, _ = dl.lick_hazard(all_trials)
    ax.plot(cc, ch / max(ch.max(), 1e-9), label="change-onset hazard (when the change comes)")
    ax.plot(lc, lh / max(lh.max(), 1e-9), label="lick hazard (when it licks)")
    ax.set_xlim(0, 12); ax.set_xlabel("time from baseline on (s)"); ax.set_ylabel("hazard (norm.)")
    ax.set_title("F4  Temporal expectation\n(does licking line up with when the change actually comes?)")
    ax.legend(frameon=False)
    return save_fig(fig, "fig_b8_F4_timing")

def fig_bias_not_gain(cells):             # F5
    fig, ax = plt.subplots(figsize=(7, 4))
    for mood, c in [(m, STATE_LABEL_COLORS[m]) for m in ("Impulsive", "StimSens")]:
        sub = cells[cells["state_label"] == mood]
        ax.scatter(sub["psy_slope"], sub["dprime"], color=c, label=mood)
    ax.set_xlabel("psychometric slope"); ax.set_ylabel("d′ (true sensitivity)")
    ax.set_title("F5  Bias-not-gain test\n(Impulsive looks eager but d′ should NOT be higher)")
    ax.legend(frameon=False)
    return save_fig(fig, "fig_b8_F5_bias_not_gain")

if __name__ == "__main__":
    force = "--force" in sys.argv
    all_trials = build(force=force)
    cells = dl.descriptive_cell_table(all_trials)
    lat = dl.descriptive_latent_table(all_trials, cells)
    cells.to_csv(os.path.join(CACHE_DIR, "decision_latents_cell_scores.csv"), index=False)
    lat.to_csv(CACHE, index=False)
    fig_sharpness(cells); fig_itchiness(cells); fig_timing(all_trials); fig_bias_not_gain(cells)
    # F-summary: which dial moves with learning / which separates moods
    summ = cells.groupby("state_label")[["psy_slope", "dprime", "criterion_c",
                                          "fa_rate", "lick_hazard_peak_time"]].mean()
    summ.to_csv(os.path.join(FIG_DIR, "decision_latents_stats.csv"))
    print(summ)
```

- [ ] **Step 4: Run end-to-end; eyeball the figures**

Run: `PYTHONPATH="$(pwd)/src" py scripts/analysis/decision_latents/run_decision_latents_by_state.py --force`
Expected: writes `fig_b8_F1_F2_sharpness.png`, `fig_b8_F3_itchiness.png`, `fig_b8_F4_timing.png`, `fig_b8_F5_bias_not_gain.png` + `decision_latents_stats.csv` to `FIGURES/decision_latents/BG_046/`, plus the two cache CSVs in `data/cache/decision_latents/`; prints the summary table. **Eyeball:** F5 — Impulsive should not show higher d′ than StimSens (bias-not-gain). Record observations (these are presentation figures).

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/decision_latents/run_decision_latents_by_state.py
git commit -m "feat(b8): Step-1 orchestration script (figures F1-F5 + table + stats)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.10: Update the question index

**Files:**
- Modify: `docs/science/QUESTION_INDEX.md`

- [ ] **Step 1: Bump B8 status and add the plan link**

In the B8 row, change status `spec-draft` → `plan-draft` and replace the Plan cell `—` with `[plan](../superpowers/plans/2026-06-18-B8-behavioral-decision-latents-by-state-plan.md)`. Update the `_Last updated_` line to `2026-06-18`.

- [ ] **Step 2: Commit**

```bash
git add docs/science/QUESTION_INDEX.md
git commit -m "docs(b8): index status spec-draft -> plan-draft + plan link

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 2 (separate plan, after Phase 1 results)

To be written as `docs/superpowers/plans/2026-06-…-B8-phase2-generative-latents-plan.md` once Phase 1 figures + cell scores exist. It will cover (spec §4 Step 2, §6): the temporal-expectation-shaped urgency, the minimal regression-accumulator with `v/z/urgency` as state functions, the **expert-anchored backward-seeded** fit, **parameter recovery at the real long-baseline regime** (the gate), the generative per-trial latents appended to the table, and figures F6–F8. Phase 1's cell scores **seed and sanity-check** it; if recovery fails, Phase 1's descriptive latents are the shipped deliverable (spec §9).

**Carried-forward refinements (from Phase-1 review + user figure review):**
- **Evidence reconstruction** must index `baseline_values` at the 60 Hz frame rate / collapse runs of 3 (n_seen is None; `ddm.build_trial_evidence`'s `ct/len(bv)` + n_seen truncation are wrong for this data — needs a corrected evidence builder).
- **F1 sharpness-metric rethink (bigger):** the single logistic `psy_slope` is shape-blind (the 1.25→1.5 segment ≠ 1.5→4 segment) and fragile (capped ±20). Move to a proper psychometric model (threshold + slope + lapse) and/or per-change-size d′; Phase 1 adds threshold + curves as an interim, Phase 2 does the full model.
- **State-definition circularity:** the bias-not-gain claim needs a labeler-independent sensitivity measure and/or neural confirmation (itchiness/criterion is partly definitional — labeler uses `f_inapplick`, `f_hit_hard`, `f_miss_easy`); see spec §7.
- **`baseline_hazard` window:** restrict to a pre-change window (currently a whole-decision-timeline mean, not comparable across cells with different max decision_time).
- **change-time anchor:** report the empirical change-time mode/median directly (not the hazard peak, which is biased late by at-risk depletion).

---

## Self-Review

**Spec coverage (Phase-1 scope):** §0 working-style → Global Constraints + figure-per-step (Tasks 0.1–1.9). §3 two-tier sessions + labeling prerequisite → Tasks 0.0, 0.2, 1.2. §3 state accessor (Abort/Disengaged, zfill) → Tasks 1.1, 1.3. §4 Step 1 three dials → Tasks 1.5–1.7. §4 two hazards (censored) → Task 1.4, 1.7. §5 latent-table schema (Step-1 columns + comprehension_flag + trial_in_session) → Tasks 1.3, 1.8. §7 confounds (within-session position, label reliability, comprehension split) → Tasks 0.2, 1.2, 1.3. §8 hazard reimplementation (not the old script) → Task 1.4. §10 deliverables (module/tests/script/figures/index) → all tasks. **Step 2 (§4 Step-2, §6 anchoring, §9 recovery) is intentionally deferred to the Phase-2 plan** — flagged above. TF-sampling resolve-at-planning (§11) → Task 0.1.

**Placeholder scan:** No "TBD/handle edge cases/similar to Task N". Each code step shows real code; each run step shows the command + expected output. The external tagging prerequisite (Task 0.2) is given as a verbatim `tag_sessions.py` command, not a hand-wave.

**Structure check (post-revision):** no task writes to `analysis_suite/` — scripts under `scripts/analysis/decision_latents/`, figures under `FIGURES/decision_latents/BG_046/`, caches under `data/cache/decision_latents/`; `save_figure` replaced by the local `save_fig` helper; `setup_style` reused for styling only (`memory/feedback_repo_structure_scripts_figures`).

**Type consistency:** `load_state_labels` → DataFrame indexed by `trial_idx` with `state_label`/`state_confidence` (used identically in 1.3). `build_trial_table` columns consumed unchanged by 1.5–1.8 (`change_size`, `lick`, `outcome`, `decision_time`, `change_time_planned`, `change_reached`, `state_label`). `censored_hazard` 3-tuple signature reused by 1.6/1.7. `descriptive_cell_table` keys (`session_name`,`state_label`) match the merge in `descriptive_latent_table`. Score-dict keys (`psy_slope`,`dprime`,`criterion_c`,`fa_rate`,`lick_hazard_peak_time`) are produced in 1.5–1.7 and consumed in 1.8/1.9.
