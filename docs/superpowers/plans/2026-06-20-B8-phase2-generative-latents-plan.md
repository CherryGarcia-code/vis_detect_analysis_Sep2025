# B8 Phase 2 — Generative Decision-Latents: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build **Engine A** — a closed-form discrete-time survival hazard-accumulator that emits three per-trial decision latents (sharpness / itchiness-caution / timing), fit expert-first then backward-seeded, **gated on parameter recovery at the real long-baseline regime** — and append the recovery-validated latents to the per-trial deliverable, plus an Engine-C pyddm spot-check and figures F6–F8.

**Architecture:** A new behaviour-only library module `visdetect.analysis.decision_latents_generative` holds the Engine-A model (corrected evidence → leaky accumulator → cloglog hazard likelihood → penalized MLE → anchored backward sweep → two model-comparison ladders → the recovery/confusion gate → the latent appender). It reuses Phase-1 `decision_latents.py` (which gets the five carried-forward fixes) and imports `ddm.py` **only** for the Engine-C pyddm spot-check (never for evidence; **`ddm.py` is reference-only and is NOT mutated**). A thin orchestration script wires it into the deliverable CSV + figures. TDD throughout with synthetic fixtures of **known ground truth**; **parameter recovery + the which-dial-varies confusion matrix are the primary tests**.

**Tech Stack:** Python 3.10 (`.venv`, invoke via `py`), numpy, scipy (`optimize`, `stats`), pandas, matplotlib (Agg), pytest. pyddm 0.9.0 (Engine-C only). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-20-B8-phase2-generative-latents-design.md` (extends `2026-06-18-B8-behavioral-decision-latents-by-state-design.md` §4 Step 2, §6, §9).

## Global Constraints

_Every task implicitly includes these (verbatim from the spec / CLAUDE.md):_

- **Invoke Python as `py`** (Windows + Git Bash), never `python`.
- **Worktree execution.** This work lives in the git worktree `…/.claude/worktrees/B8-phase2-generative` on branch `feature/B8-phase2-generative-latents`. The editable `visdetect` install points at the **primary** repo's `src`, so **set `PYTHONPATH="<worktree>/src"`** for every `py`/`pytest` invocation or you silently test main's code (`memory/worktree_editable_install_pythonpath`). Gitignored data inputs are **not** in the worktree checkout — junction/copy them (Task 0.0), and **never `rm -rf` without deleting junctions first** (`memory/worktree_realdata_inputs_junctions`).
- **Behaviour-only.** No spike data is loaded anywhere. Load behaviour/trials, not clusters.
- **Constants from the canonical source** `visdetect.analysis.constants` (e.g. `CHANGE_SIZES`, `FA_RT_SPLIT`). Never hardcode a value that lives there.
- **Grid `dt = 0.05 s`** everywhere in Engine A (verified 50 ms TF update). **Leak `τ = 0.27 s` fixed** (Khilkevich & Lohse 2024), with a sensitivity sweep `(0.15, 0.27, 0.40)`. Engine A evidence comes **only** from `build_trial_evidence_corrected` (Task 0.1) — **never** `ddm.build_trial_evidence`.
- **Do not mutate `ddm.py`.** Import its helpers (`rectify`, `build_model`, `fit_model`) read-only for the Engine-C spot-check only.
- **State source = the new-labeler tags** behind the Phase-1 `load_state_labels` accessor. **Main fits: Impulsive vs StimSens. Disengaged: reported separately. Abort: excluded** (labeler-state `Abort` ≠ trial-outcome `abort`; both dropped).
- **Repo structure** (`memory/feedback_repo_structure_scripts_figures`): scripts in `scripts/analysis/decision_latents/`; figures in top-level `FIGURES/decision_latents/BG_046/`; caches in `data/cache/decision_latents/`; library in `src/visdetect/`. **Not** `analysis_suite/`. Use `suite.plotting.setup_style` for **styling only** + the local `save_fig()` helper (Task 4.2). Mood colours from `config.STATE_LABEL_COLORS` = `{Impulsive:#ef6548, StimSens:#6baed6, Disengaged:#3474ae, Abort:#bdbdbd}`.
- **Every analysis step saves a presentation-ready PNG** with a plain-language title + caption (`memory/feedback_plain_language_and_save_figures`). Glossary for captions: *sharpness = how clearly the mouse tells the change happened; itchiness/caution = how trigger-happy it is before evidence; timing = how strongly it expects the change now.*
- **Lead figures with labeler-INDEPENDENT dials** (timing, RT variability); explicitly label FA-rate/criterion×mood **"confirmatory"** (state-label circularity, `memory/state_labeler_circularity_caveat`).
- **Memory hygiene:** `del sess; gc.collect()` after each session in loops.
- **TDD + frequent commits.** Each task ends green and committed. Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## §A. Locked Types, Math & Test Contract (BINDING — overrides any divergent inline code below)

This section is the **single source of truth** for the Engine-A core types, the likelihood math, and the recovery-test rigor. Every task that touches these uses **exactly** these definitions. Where a task's prose and this section disagree, **this section wins**. All of the following lives in `src/visdetect/analysis/decision_latents_generative.py` unless noted.

### A.1 The cloglog hazard link (Task 1.2)

The per-bin lick probability (hazard) is the **inverse complementary-log-log** of a linear predictor `lp`. Naming avoids the `cloglog(h)` vs `inv_cloglog(x)` confusion: one function maps **linear-predictor → hazard**, the other its inverse.

```python
import numpy as np

def hazard_from_lp(lp):
    """Inverse cloglog link: h = 1 - exp(-exp(lp)), numerically stable, h in (0,1)."""
    exp_lp = np.exp(np.clip(lp, -30.0, 30.0))
    return -np.expm1(-exp_lp)                      # 1 - exp(-exp(lp))

def lp_from_hazard(h):
    """Forward cloglog link: lp = log(-log(1-h))."""
    h = np.clip(np.asarray(h, float), 1e-12, 1 - 1e-12)
    return np.log(-np.log1p(-h))
```
Round-trip invariant (a test): `hazard_from_lp(lp_from_hazard(h)) ≈ h` for `h ∈ (0,1)`.

### A.2 Corrected evidence builder (Task 0.1) — the 60 Hz / collapse-runs-of-3 algorithm

`baseline_values` (`bv`) is stored at **60 Hz**; each TF value is **held 3 frames (50 ms)**; `n_seen` is **always `None`**. The `dt=0.05 s` grid is exactly one TF update, so **bin `k` reads frame `3k`** (NOT `ct/len(bv)`). The change is a TF *increase* by `change_size` on go trials.

```python
def build_trial_evidence_corrected(session, dt=0.05, tf_base=None):
    """Per-trial log2-TF evidence on the dt grid, truncated to [0, decision_time].
    Returns DataFrame: trial_idx, outcome, change_size, change_time, decision_time,
    lick, censored, evidence(np.ndarray, len==n_bins), n_bins.
    Evidence bin k reads baseline frame index 3*k (60 Hz storage, 50 ms holds)."""
    MONITOR_HZ = 60.0
    frames_per_bin = int(round(dt * MONITOR_HZ))   # == 3 for dt=0.05
    trials = getattr(session, "trials", []) or []
    rows = []
    for uid, t in enumerate(trials):
        oc = (getattr(t, "trialoutcome", "") or "").lower()
        if oc in ("abort", "ref"):
            continue
        bv = getattr(t, "baseline_values", None)
        if bv is None:
            continue
        bv = np.asarray(bv, float).ravel()
        if bv.size == 0:
            continue
        cs = float(getattr(t, "change_size", np.nan) or np.nan)
        ct = float(getattr(t, "change_time", np.nan) or np.nan)
        base = float(tf_base) if tf_base is not None else float(np.nanmedian(bv)) or 1.0
        dec_t, lick, censored = _decision_time_dl(t)        # Phase-1 helper (Task 0.1 Step 1)
        if not np.isfinite(dec_t) or dec_t <= 0:
            continue
        n_bins = int(round(dec_t / dt))
        if n_bins < 1:
            continue
        ev = np.empty(n_bins, float)
        for k in range(n_bins):
            j = min(bv.size - 1, k * frames_per_bin)        # 60 Hz frame for this 50 ms bin
            tau = k * dt
            tf = bv[j] * cs if (np.isfinite(ct) and tau >= ct and cs > 1.0) else bv[j]
            ev[k] = np.log2(tf / base) if tf > 0 else 0.0
        ev = np.nan_to_num(ev, nan=0.0)
        rows.append({"trial_idx": uid, "outcome": oc, "change_size": cs,
                     "change_time": ct, "decision_time": dec_t, "lick": int(lick),
                     "censored": bool(censored), "evidence": ev, "n_bins": n_bins})
    import pandas as pd
    return pd.DataFrame(rows)
```
Validation (Task 0.1): on a synthetic session with a known `bv` of repeated runs-of-3, `evidence[k]` must equal `log2(bv[3k]/base)` pre-change. Spot-check against `scripts/analysis/decision_latents/_tf_sampling_check.py`.

### A.3 Leaky accumulator (Task 1.1) + the urgency bump (Task 1.2)

Leak uses the **exponential** decay `exp(-dt/τ)` (always in `(0,1)`; no `τ→0` instability). `R` is the selected rectification (`signed`/`halfwave`/`asym`).

```python
def leaky_accumulate(evidence, dt=0.05, leak_tau=0.27, rectification="signed", g_up=1.0, g_down=1.0):
    """A[k] = decay*A[k-1] + R(e[k])*dt, decay = exp(-dt/leak_tau)."""
    from visdetect.analysis.ddm import rectify
    kind = {"signed": "symmetric"}.get(rectification, rectification)   # ddm uses 'symmetric'
    r = rectify(np.asarray(evidence, float), kind, g_up=g_up, g_down=g_down)
    decay = np.exp(-dt / float(leak_tau))
    A = np.empty(len(r), float)
    acc = 0.0
    for k in range(len(r)):
        acc = decay * acc + r[k] * dt
        A[k] = acc
    return A

def expectation_bump(t_grid, mu, sigma):
    """Gaussian temporal-expectation profile, peak 1.0 at mu (sigma FIXED, not fitted)."""
    t_grid = np.asarray(t_grid, float)
    return np.exp(-0.5 * ((t_grid - mu) / float(sigma)) ** 2)
```
**Locked decision (resolves the φ circularity):** `mu` is the **per-session empirical change-time anchor** (Task 0.4) and `sigma` is **FIXED** (a `ParamSpec` field, default 0.8 s; not a free parameter). Only the urgency **amplitude** is fit. So `phi` is fully determined in `build_design` and never depends on the fitted parameters.

### A.4 `ParamSpec` — declarative parameter layout (Task 1.4)

Owns the `theta` ↔ dial/mood mapping so no task hardcodes indices. Three dials `v` (sharpness), `z` (itchiness/caution), `u` (timing-amplitude); each is per-mood if in `state_terms`, else shared.

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ParamSpec:
    moods: tuple = ("Impulsive", "StimSens")
    dials: tuple = ("v", "z", "u")
    state_terms: tuple = ("v", "z", "u")     # which dials carry a per-mood term
    rectification: str = "signed"
    leak_tau: float = 0.27
    urgency_sigma: float = 0.8               # FIXED seconds

    def _len(self, dial):
        return len(self.moods) if dial in self.state_terms else 1

    def n_params(self):
        return sum(self._len(d) for d in self.dials)

    def _offset(self, dial):
        off = 0
        for d in self.dials:
            if d == dial:
                return off
            off += self._len(d)
        raise KeyError(dial)

    def value(self, theta, dial, mood):
        off = self._offset(dial)
        return theta[off + self.moods.index(mood)] if dial in self.state_terms else theta[off]

    def per_trial(self, theta, mood_code):
        """mood_code: int array indexing self.moods. Returns (v, z, u) per-trial arrays."""
        out = {}
        for dial in ("v", "z", "u"):
            off = self._offset(dial)
            if dial in self.state_terms:
                vals = np.asarray([theta[off + m] for m in mood_code])
            else:
                vals = np.full(len(mood_code), theta[off])
            out[dial] = vals
        return out["v"], out["z"], out["u"]
```
**Layout-invariance test (Task 1.4):** building `theta` for a reordered `state_terms` and reading back via `value(...)` yields identical per-mood values — indices never hardcoded.

### A.5 `Design` — ragged per-trial precompute (Task 1.3)

Ragged (list-of-arrays), **not** NaN-padded 2-D. Built once per anchor; sliceable for CV/resampling.

```python
@dataclass
class Design:
    A: list            # list[np.ndarray]  leaky-accumulated evidence per trial (len == n_bins_i)
    phi: list          # list[np.ndarray]  urgency bump per trial (same lengths)
    event_bin: np.ndarray   # int   index of the decision bin per trial (== n_bins_i - 1)
    lick: np.ndarray        # int 0/1
    censored: np.ndarray    # bool
    mood_code: np.ndarray   # int   index into ParamSpec.moods
    trial_idx: np.ndarray   # int
    dt: float

    def __len__(self):
        return len(self.A)

    def subset(self, idx):
        idx = np.asarray(idx, int)
        return Design(A=[self.A[i] for i in idx], phi=[self.phi[i] for i in idx],
                      event_bin=self.event_bin[idx], lick=self.lick[idx],
                      censored=self.censored[idx], mood_code=self.mood_code[idx],
                      trial_idx=self.trial_idx[idx], dt=self.dt)

def build_design(trial_evidence_df, state_labels, mu, sigma, dt=0.05,
                 leak_tau=0.27, rectification="signed"):
    """Precompute A (leaky_accumulate) and phi (expectation_bump on the trial's t-grid)
    per trial; map mood to code (drop EXCLUDED_MOODS, keep MAIN_MOODS for fitting).
    event_bin = n_bins-1 (decision in the last bin). Returns a Design."""
    ...
```
Each trial's `phi` is `expectation_bump(np.arange(n_bins)*dt, mu, sigma)`. **Only `MAIN_MOODS` (Impulsive, StimSens) enter the fit Design**; Disengaged handled in reporting, Abort dropped (Phase-1 rule). `A[i]` and `phi[i]` have length `n_bins_i`; `event_bin[i] = n_bins_i - 1`.

### A.6 `hazard_nll` — closed-form censored negative log-likelihood (Task 1.4)

Lick at bin `K`: `h_K · Π_{k<K}(1−h_k)`. Censored (no-lick / Miss, right-censored at `K`): `Π_{k≤K}(1−h_k)`.

```python
def hazard_nll(theta, design, param_spec, l2=0.0, seed_theta=None):
    v, z, u = param_spec.per_trial(theta, design.mood_code)
    nll = 0.0
    for i in range(len(design)):
        A, phi = design.A[i], design.phi[i]
        K = int(design.event_bin[i])                     # == len(A) - 1
        lp = z[i] + v[i] * A + u[i] * phi                # linear predictor, len == K+1
        h = np.clip(hazard_from_lp(lp), 1e-12, 1 - 1e-12)
        log_surv = np.sum(np.log1p(-h[:K]))              # log Π_{k<K}(1-h_k)
        if design.lick[i] == 1 and not design.censored[i]:
            nll -= log_surv + np.log(h[K])               # event in bin K
        else:
            nll -= log_surv + np.log1p(-h[K])            # survived through bin K (censored)
    if l2 > 0 and seed_theta is not None:
        nll += float(l2) * np.sum((np.asarray(theta) - np.asarray(seed_theta)) ** 2)
    return float(nll)
```
**Ragged-safety test (Task 1.4):** a Design mixing a 3-bin trial and a 200-bin trial gives a finite NLL equal to the sum of the two trials computed independently.

### A.7 `FitResult` + `fit_anchor` (Task 1.5)

```python
@dataclass
class FitResult:
    theta: np.ndarray
    dials: dict          # {mood: {"sharpness": v, "itchiness": z, "timing": u}}
    ll: float
    n_params: int
    cov: np.ndarray      # inverse Hessian (may be None if singular)
    hessian: np.ndarray  # finite-difference Hessian at the optimum
    hessian_cond: float  # condition number (np.inf if singular)
```
`fit_anchor(design, param_spec, seed_theta=None, l2=0.0, n_restarts=4, seed=0) -> FitResult` minimizes `hazard_nll` (scipy `minimize`, L-BFGS-B, `n_restarts` random inits + the `seed_theta` init), computes the finite-difference Hessian (`scipy.optimize.approx_fprime` of the gradient, or `numdifftools` if available) and `hessian_cond = np.linalg.cond(hessian)`, and fills `dials` via `param_spec.value(theta, dial, mood)` for each mood. **`dials` structure is locked** here and consumed unchanged by Tasks 2.x/3.x/4.1.

### A.8 `simulate_licks` (Task 3.1) — draw through the per-bin hazard

```python
def simulate_licks(design, true_theta, param_spec, seed=0):
    """Generate (event_bin, lick, censored) by walking each trial's per-bin hazard.
    Uses the SAME A/phi/mood as `design` (so a refit Design reuses them)."""
    assert len(true_theta) == param_spec.n_params()
    rng = np.random.default_rng(seed)
    v, z, u = param_spec.per_trial(true_theta, design.mood_code)
    n = len(design)
    event_bin = np.empty(n, int); lick = np.zeros(n, int); censored = np.zeros(n, bool)
    for i in range(n):
        A, phi = design.A[i], design.phi[i]
        h = np.clip(hazard_from_lp(z[i] + v[i] * A + u[i] * phi), 1e-12, 1 - 1e-12)
        fired = -1
        draws = rng.random(len(h))
        for k in range(len(h)):
            if draws[k] < h[k]:
                fired = k; break
        if fired >= 0:
            event_bin[i] = fired; lick[i] = 1
        else:
            event_bin[i] = len(h) - 1; censored[i] = True
    return event_bin, lick, censored

def design_with_outcomes(design, event_bin, lick, censored):
    """Return a copy of `design` with simulated outcomes (A/phi/mood unchanged), so it can
    be refit. NOTE: trials are truncated at event_bin for the likelihood via event_bin only;
    A/phi keep full length (hazard_nll reads only [:K+1])."""
    import copy
    d = copy.copy(design)
    d.event_bin = np.asarray(event_bin, int); d.lick = np.asarray(lick, int)
    d.censored = np.asarray(censored, bool)
    return d
```

### A.9 Recovery-test rigor (Tasks 3.0–3.5) — NON-NEGOTIABLE assertions

Weak/tautological tests are forbidden. All recovery tests use **shared ground-truth fixtures** (Task 3.0): synthetic `Design`s for an **expert-like** and a **naïve-like** regime built on **real per-trial evidence** at `dt=0.05`, plus a known `true_theta` per regime.

- **`recover_point` (3.2)** — simulate from `true_theta`, refit, repeat `N≥100`. Assert **per-dial `|mean(recovered) − true| ≤ 0.1·SD(true_across_reps)`** (bias) AND **Pearson `r ≥ 0.8`** between recovered and true across a grid of true values; AND bootstrap **CI coverage ∈ [0.90, 0.97]**. A test that only checks `ll > random_ll` is INVALID.
- **`recover_confusion` (3.3, decisive)** — three scenarios where **only one** dial truly varies across two anchors (only-`v`, only-`z`, only-`u`). For each, run `learning_ladder`; the winner must equal the true dial. Build the **3×3 confusion matrix** over `n_rep` repeats. Assert **diagonal ≥ 0.8** AND **every off-diagonal ≤ 0.2**. Checking only "diagonal > 0.55" is INVALID.
- **`recover_true_difference` (3.4)** — with a genuinely different true dial across stages (e.g. `v_naïve=1.0`, `v_expert=2.5`), the **L2-seeded backward fit must recover the difference**: assert `recovered_delta` has the right sign and `|recovered − true| ≤ 0.3·|true|`, and `shrunk == False` (the prior informed but did not erase the difference).
- **`recovery_gate` (3.5)** — provisional thresholds `r_min=0.8`, `bias_max_frac=0.1`, `confusion_min_diag=0.8`, `confusion_max_offdiag=0.2`, CI coverage band `[0.90,0.97]`; naïve-regime relaxation is a **single declared constant** (`NAIVE_RELAX=0.1` off `r_min`/`diag`), flagged **statistician-confirmable** (spec §11). Returns `{per_dial_trust: {dial: "generative"|"descriptive"}, regime, passed}`. **Per-dial**, not binary — a dial that fails goes `descriptive`.

### A.10 Resolved decisions (apply throughout)

1. **σ (urgency width) FIXED**, μ per-session seeded → `phi` is data-determined, not fitted (A.3).
2. **Leak fixed at 0.27 s + sweep (0.15/0.27/0.40)**; "is τ learned" → B1.
3. **Anchor-design dict is built in an explicit Task 1.7** (`build_anchor_designs`), not hidden inside `backward_sweep`. `backward_sweep(anchor_designs: dict[str, Design], …) -> dict[str, FitResult]`. `learning_ladder(anchor_designs: dict[str, Design], …)` (re-fits per ladder rung; GLM dof, **not** the pyddm formula).
4. **Engine-C is REQUIRED** (computed on ≥1 expert session); F8 includes it; a genuine pyddm failure is logged + the panel notes "unavailable," never silently skipped.
5. **Expert-anchor contingency is a GATE, not a warning** (Task 0.9 + 4.2): `<3` adequate expert anchors → branch to **pool late post-comprehension sessions** as the anchor; if still `<3` → **ship Phase-1 proxies** (`latent_trust="descriptive"` for all) and stop the generative fit. Tested.
6. **F7/F8 captions explicitly** state the circularity caveat and lead with timing/RT (Task 4.4).

---

## Phase 0 — Prerequisites (BLOCKING: each must pass before any generative fit)

### Task 0.0: Make the worktree runnable (PYTHONPATH + data junctions)

**Files:** none (environment only; no commit).

**Interfaces:** Produces a shell where `py`/`pytest` import `visdetect` from the **worktree** `src` and can load BG_046 sessions + state tags + the Phase-1 caches.

- [ ] **Step 1: Confirm imports resolve to the worktree.** Run from the worktree root:
  ```bash
  WT="$(pwd)"; PYTHONPATH="$WT/src" py -c "import visdetect, os; print(visdetect.__file__)"
  ```
  Expected: a path under `…/.claude/worktrees/B8-phase2-generative/src/visdetect/__init__.py`. If it points at the primary repo, `PYTHONPATH` was not honored — fix before continuing.
- [ ] **Step 2: Junction big gitignored inputs (pkls), copy small ones (tags, manifest, Phase-1 caches).**
  ```bash
  WT="$(pwd)"; PRIMARY="E:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
  mkdir -p "$WT/data/pkls" "$WT/data/cache"
  cmd //c mklink /J "$(cygpath -w "$WT/data/pkls/BG_046")" "$(cygpath -w "$PRIMARY/data/pkls/BG_046")"
  cp -r "$PRIMARY/data/cache/state_tags" "$WT/data/cache/state_tags"
  cp -r "$PRIMARY/data/cache/decision_latents" "$WT/data/cache/decision_latents"
  ```
- [ ] **Step 3: Verify a session + its tags + the Phase-1 cache load.**
  ```bash
  PYTHONPATH="$WT/src" py -c "
  from visdetect.suite.loader import load_session
  from visdetect.analysis import decision_latents as dl
  import pandas as pd
  s = load_session('01072025'); print('trials:', len(s.trials))
  print('tagged sessions:', len(dl.enumerate_valid_sessions()))
  print('deliverable rows:', len(pd.read_csv('data/cache/decision_latents/decision_latents_by_state.csv')))
  "
  ```
  Expected: positive trial count, ~28 tagged sessions, 16,692 deliverable rows. **No commit** (no tracked change).

---

### Task 0.1: Corrected evidence builder (fix a) — 60 Hz / collapse-runs-of-3

**Files:**
- Modify: `src/visdetect/analysis/decision_latents.py` (add `build_trial_evidence_corrected`; reuse/extract a `_decision_time_dl(trial)` helper mirroring `ddm._decision_time` — **do not import from ddm to avoid the bug surface**).
- Modify: `tests/analysis/test_decision_latents.py`.
- Create: `scripts/analysis/decision_latents/_evidence_builder_check.py` (validation figure).

**Interfaces:**
- Consumes: `visdetect.suite.loader.load_session`; `RESPONSE_WINDOW` (define a module constant `RESPONSE_WINDOW_S = 2.155`, mirroring ddm; confirm against task params at run).
- Produces: `build_trial_evidence_corrected(session, dt=0.05, tf_base=None) -> DataFrame [trial_idx, outcome, change_size, change_time, decision_time, lick, censored, evidence(np.ndarray), n_bins]` — **the canonical code is §A.2** (use it verbatim).

- [ ] **Step 1: Write the failing test** (synthetic session with a known runs-of-3 `bv`).
  ```python
  def test_build_trial_evidence_corrected_reads_every_third_frame():
      import numpy as np
      from types import SimpleNamespace
      from visdetect.analysis import decision_latents as dl
      # bv: 60 Hz, each TF held 3 frames. Pre-change TF doubles every 3 frames: 1,1,1,2,2,2,4,4,4,...
      bv = np.repeat([1.0, 2.0, 4.0, 8.0], 3)            # 12 frames = 0.2 s at 60 Hz
      t = SimpleNamespace(trialoutcome="hit", change_size=4.0, change_time=10.0,
                          reactiontimes={"RT": 0.10}, baseline_values=bv, n_seen=None)
      sess = SimpleNamespace(trials=[t])
      df = dl.build_trial_evidence_corrected(sess, dt=0.05, tf_base=1.0)
      ev = df.iloc[0]["evidence"]
      # bin k reads frame 3k: bv[0]=1->log2(1)=0, bv[3]=2->1, bv[6]=4->2 (pre-change, ct=10 s)
      assert ev[0] == 0.0 and abs(ev[1] - 1.0) < 1e-9 and abs(ev[2] - 2.0) < 1e-9
      assert df.iloc[0]["n_bins"] == int(round((10.0 + 0.10) / 0.05)) or df.iloc[0]["n_bins"] == 202
  ```
- [ ] **Step 2: Run → FAIL** (`AttributeError: build_trial_evidence_corrected`). `PYTHONPATH="$(pwd)/src" py -m pytest tests/analysis/test_decision_latents.py -k evidence_corrected -q`
- [ ] **Step 3: Implement** `_decision_time_dl` + `build_trial_evidence_corrected` exactly as §A.2.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Validation figure** — `_evidence_builder_check.py`: load 3 real sessions, overlay reconstructed `evidence` vs raw `bv` (downsampled) for a few trials; save `FIGURES/decision_latents/BG_046/fig_b8_P2_evidence_builder_check.png` with a plain-language caption ("evidence reconstructed at the true 50 ms TF cadence"). Confirm peaks align with `_tf_sampling_check.py` cadence.
- [ ] **Step 6: Commit** — `feat(b8p2): corrected 60Hz/collapse-3 evidence builder + validation`.

---

### Task 0.2: Lapse-aware psychometric (fix b) + re-run Phase-1 cell table

**Files:** Modify `src/visdetect/analysis/decision_latents.py` (`sharpness_scores`); Modify `tests/analysis/test_decision_latents.py`; re-run the Phase-1 cell/latent caches.

**Interfaces:** Produces `sharpness_scores(trial_df)` with **added** keys `psy_lapse`, `psy_threshold_lapse`. Model: `P(lick|cs) = lapse + (1-2*lapse)*logistic(a + b*log2(cs))`, `lapse ∈ [0, 0.3]`, fit by `scipy.optimize.curve_fit` with bounds; keep the existing `psy_slope`/`psy_threshold` for back-compat.

- [ ] **Step 1: Failing test** — synthetic go-trial data generated with a **known lapse** (e.g. `lapse=0.15`, steep slope); assert the 3-param fit recovers `psy_lapse` within `±0.07` and that on lapse-free data `psy_lapse ≈ 0`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the 3-param model alongside the 2-param (guard `b>0` for `psy_threshold_lapse = 2**(-a/b)`, clamp `[1.0, 8.0]`; on convergence failure return `nan`).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Re-run Phase-1 caches** so Phase-1 and Phase-2 measure the same psychometric (else F8 disagrees): `PYTHONPATH=... py scripts/analysis/decision_latents/run_decision_latents_by_state.py --force`; confirm `psy_lapse` column appears in `decision_latents_cell_scores.csv`.
- [ ] **Step 6: Commit** — `feat(b8p2): lapse-aware psychometric (threshold+slope+lapse) + cell-table re-run`.

---

### Task 0.3: Pre-change baseline-hazard window (fix c)

**Files:** Modify `decision_latents.py` (`itchiness_scores`); Modify the test file.

**Interfaces:** `itchiness_scores(trial_df, dt=0.05)` — `baseline_hazard` now computed over the **pre-change window only** (censor non-FA trials at `change_time_planned`, mirroring `fa_lick_hazard`), so it is comparable across cells with different max decision times.

- [ ] **Step 1: Failing test** — two cells identical except post-change lick density; the new `baseline_hazard` must be **insensitive** to post-change activity (differs from the old full-timeline value), and equal between the two cells.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (reuse `fa_lick_hazard` censoring at `change_time_planned`). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `fix(b8p2): restrict baseline_hazard to the pre-change window`.

---

### Task 0.4: Empirical change-time anchor (fix d)

**Files:** Modify `decision_latents.py` (add `change_time_anchor`; add `change_time_anchor_median` key to `timing_scores`); Modify the test file.

**Interfaces:** `change_time_anchor(trial_df) -> float = median(change_time_planned | change_reached)`. `timing_scores(...)` gains `change_time_anchor_median` (keep the existing `*_peak_time` keys for the late-bias comparison). This `μ` seeds `φ` (A.3) and the `expected_change_time` latent.

- [ ] **Step 1: Failing test** — data with changes drawn around 7 s but a hazard peak biased late (by at-risk depletion); assert `change_time_anchor ≈ 7 s` and that it is **earlier** than `change_hazard_peak_time`.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement.** [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): empirical change-time anchor (median, not hazard peak)`.

---

### Task 0.5: Generative-sufficiency QC flag (fix e, part 1)

**Files:** Modify `decision_latents.py` (`compute_cell_qc`); Modify the test file; Modify `scripts/analysis/decision_latents/behavioral_qc_profile.py` (add the new distributions).

**Interfaces:** `compute_cell_qc(trial_df)` gains counts `n_lick_events, n_censored, n_trials_spanning_anchor, n_evidence_excursions` and a boolean `usable_generative`. Thresholds are **distribution-justified** (declare `QC_GEN_MIN_LICK_EVENTS`, `QC_GEN_MIN_CENSORED`, `QC_GEN_MIN_SPAN`, `QC_GEN_MIN_EXCURSION` as module constants, **set from the profiler run in Step 5**, not guessed).

- [ ] **Step 1: Failing test** — a cell with plenty of licks but **zero censored trials** (can't identify a hazard slope) → `usable_generative == False`; a balanced cell → `True`.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (counts + flag; thresholds as named constants).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Profiler distributions** — extend `behavioral_qc_profile.py` to plot the four new per-cell quantities; **set the `QC_GEN_*` constants from where each distribution's mass sits** (presentation figure `fig_b8_P2_generative_qc_distributions.png`). Document the chosen values in the docstring.
- [ ] **Step 6: Commit** — `feat(b8p2): distribution-justified usable_generative QC gate`.

---

### Task 0.6: comprehension_flag operationalization (fix f) + sensitivity

**Files:** Modify `decision_latents.py` (`assign_comprehension_flags` gains `rule` + `hitrate_by_session`); Create `scripts/analysis/decision_latents/_comprehension_flag_explore.py`; Modify the test file.

**Interfaces:** `assign_comprehension_flags(dprime_by_session, threshold=0.5, rule="dprime", hitrate_by_session=None) -> {session: "pre"|"post"}`, `rule ∈ {"dprime", "easy_hitrate"}`.

- [ ] **Step 1: Failing test** — both rules mark the expected pre→post boundary on a synthetic chronology; switching `rule` changes the boundary as designed.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** both rules. [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Sensitivity script** — `_comprehension_flag_explore.py`: overlay the d′-rule and easy-hitrate-rule boundaries on the chronological d′/hit-rate trajectory; annotate which sessions flip; report the **±1-session sensitivity** of downstream counts. Save `fig_b8_P2_comprehension_boundary.png`. **Pick the rule by inspection** and record it in the script docstring + this plan's Task-4.2 call site.
- [ ] **Step 6: Commit** — `feat(b8p2): comprehension_flag rule options + sensitivity diagnostic`.

---

### Task 0.7: Naïve-session label-reliability protocol (fix g)

**Files:** Modify `scripts/analysis/decision_latents/_label_reliability.py`.

**Interfaces:** Consumes the state-tag CSVs + `compute_session_performance`. Produces a reliability figure + `data/cache/decision_latents/b8p2_label_reliability.csv` and a **confidence-gating rule** applied downstream.

- [ ] **Step 1: Confirm tag coverage.** Run `dl.enumerate_valid_sessions()`; if the earliest naïve sessions are **untagged**, run the labeler on them (external prerequisite, per the base plan Task 0.2: `py scripts/state_labeling/tag_sessions.py --sessions <list> --figures`). Record which sessions were newly tagged.
- [ ] **Step 2: Extend `_label_reliability.py`** — per session: mood proportions, mean `state_confidence`, and a `naive_reliable` boolean = **≥80 % of trials have `state_confidence > 0.7`**. Plot confidence-vs-d′ and mood composition (chronological); flag sessions failing the rule (those drop to **coarse, no-mood** treatment downstream).
- [ ] **Step 3: Run + eyeball.** Save `fig_b8_P2_label_reliability.png` + the CSV; print the flagged-session list. **Judgment checkpoint**, not an automated pass/fail.
- [ ] **Step 4: Commit** — `feat(b8p2): naive-session label-reliability protocol + confidence gate`.

---

### Task 0.8: Expert-anchor data inventory (fix h, part 1)

**Files:** Create `scripts/analysis/decision_latents/_expert_anchor_inventory.py`.

**Interfaces:** Produces `data/cache/decision_latents/b8p2_expert_anchor_inventory.csv` (per session: d′, per-mood `n_trials`, `usable_generative` cell count) + the **expert subset** = sessions with `d′>0.7` AND per-mood `n≥20`.

- [ ] **Step 1: Write the inventory script** — iterate `enumerate_valid_sessions`, compute `session_dprime` + per-(session×mood) counts via `compute_cell_qc`; mark the expert subset; print its size.
- [ ] **Step 2: Run it.** Save `fig_b8_P2_expert_anchor_inventory.png` (d′ × per-mood-n scatter, expert subset highlighted) + the CSV. **Record the expert-subset size** — it drives the Task 0.9 contingency.
- [ ] **Step 3: Commit** — `feat(b8p2): expert-anchor data inventory`.

---

### Task 0.9: Expert-anchor contingency GATE (fix h, part 2)

**Files:** Modify `src/visdetect/analysis/decision_latents_generative.py` (create the module here with the gate fn); Create `tests/analysis/test_decision_latents_generative.py`.

**Interfaces:** Produces `select_expert_anchors(inventory_df, min_d=0.7, min_mood_n=20, min_anchors=3) -> {"anchors": list, "mode": "expert"|"pooled"|"fallback"}`. **`expert`** if ≥3 qualify; else **`pooled`** = the latest post-comprehension sessions topped up to 3; if still impossible, **`fallback`** (ship Phase-1 proxies, no generative fit).

- [ ] **Step 1: Failing test** — 3 qualifying → `mode=="expert"`; 1 qualifying + post-comprehension sessions available → `mode=="pooled"` with 3 anchors; none → `mode=="fallback"`.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement.** [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): expert-anchor contingency gate (expert/pooled/fallback)`.

---

## Phase 1 — Engine A: model + likelihood

> All core code is in **§A**; tasks below add tests + wire it together. Module: `src/visdetect/analysis/decision_latents_generative.py` (created in Task 0.9). Test file: `tests/analysis/test_decision_latents_generative.py`.

### Task 1.1: Leaky accumulator + rectification

**Interfaces:** Produces `leaky_accumulate(evidence, dt=0.05, leak_tau=0.27, rectification="signed", g_up=1.0, g_down=1.0) -> np.ndarray` (§A.3).

- [ ] **Step 1: Failing tests.** (a) On constant positive evidence the accumulator approaches the steady state `R(e)·τ` (within 5 % after `5τ`). (b) `rectification="halfwave"` zeros a negative-evidence trace's accumulator; `"signed"` drives it negative. (c) `decay = exp(-dt/leak_tau)` is in `(0,1)` for `leak_tau∈{0.15,0.27,0.40}`.
  ```python
  def test_leaky_accumulate_steady_state_and_rectification():
      import numpy as np
      from visdetect.analysis import decision_latents_generative as g
      e = np.ones(int(5*0.27/0.05))                    # ~5 tau of constant evidence
      A = g.leaky_accumulate(e, dt=0.05, leak_tau=0.27, rectification="signed")
      assert abs(A[-1] - 1.0*0.27) < 0.05*0.27         # steady state R*tau
      neg = -np.ones(20)
      assert np.all(g.leaky_accumulate(neg, rectification="halfwave") == 0.0)
      assert g.leaky_accumulate(neg, rectification="signed")[-1] < 0
  ```
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement §A.3 `leaky_accumulate`.** [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): leaky accumulator + rectification`.

### Task 1.2: cloglog link + urgency bump

**Interfaces:** `hazard_from_lp(lp)`, `lp_from_hazard(h)` (§A.1); `expectation_bump(t_grid, mu, sigma)` (§A.3).

- [ ] **Step 1: Failing tests.** Round-trip `hazard_from_lp(lp_from_hazard(h)) ≈ h` for `h∈{0.01,…,0.99}`; `hazard_from_lp` stays in `(0,1)` for `lp∈[-50,50]` (no overflow); `expectation_bump` peaks at `mu` with value 1.0 and is symmetric.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement §A.1 + §A.3.** [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): cloglog hazard link + temporal-expectation bump`.

### Task 1.3: `Design` + `build_design`

**Interfaces:** `Design` dataclass + `build_design(trial_evidence_df, state_labels, mu, sigma, dt=0.05, leak_tau=0.27, rectification="signed") -> Design` (§A.5). Consumes `build_trial_evidence_corrected` (0.1), `load_state_labels` (Phase-1), `MAIN_MOODS`.

- [ ] **Step 1: Failing tests.** A 2-trial evidence frame (one 3-bin, one 200-bin) + state labels → `Design` with `len==2`, ragged `A`/`phi` of matching lengths, `event_bin == [2, 199]`, `mood_code` indexing `MAIN_MOODS`; **`design.subset([1])`** returns a 1-trial Design with the long trial only. Trials whose mood ∉ `MAIN_MOODS` are dropped.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** `Design` (incl. `subset`) + `build_design` per §A.5 (`phi = expectation_bump(arange(n_bins)*dt, mu, sigma)`; `A = leaky_accumulate(evidence,…)`). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): ragged Design + build_design (A, phi, mood, subset)`.

### Task 1.4: `ParamSpec` + `hazard_nll`

**Interfaces:** `ParamSpec` (§A.4) + `hazard_nll(theta, design, param_spec, l2=0.0, seed_theta=None) -> float` (§A.6).

- [ ] **Step 1: Failing tests.** (a) **Layout-invariance:** for `state_terms=("v","z","u")` and a reordered `("u","z","v")`, `param_spec.value(theta, "v", "StimSens")` reads the correct slot in both; `n_params()==6` for 2 moods × 3 state dials. (b) **Ragged-safety:** `hazard_nll` on the mixed 3-bin/200-bin Design equals the sum of the two trials' NLLs computed singly, and is finite. (c) Adding `l2>0` with `seed_theta=theta` adds 0; with `seed_theta≠theta` increases the NLL.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement §A.4 + §A.6.** [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): ParamSpec layout + closed-form censored hazard NLL`.

### Task 1.5: `fit_anchor` + `FitResult`

**Interfaces:** `FitResult` (§A.7) + `fit_anchor(design, param_spec, seed_theta=None, l2=0.0, n_restarts=4, seed=0) -> FitResult`.

- [ ] **Step 1: Failing test (ground-truth recovery, not a tautology).** Build a Design on synthetic evidence; pick a known `true_theta`; **simulate licks via `simulate_licks` (Task 3.1 — implement 3.1 first or import the §A.8 code)** with ~2000 trials; refit; assert **per-dial `|recovered − true| < 0.3`** for each mood, and `result.hessian_cond < 1e6`. (Sequencing: this test depends on §A.8 `simulate_licks`; do Task 3.1's `simulate_licks` implementation before 1.5's test, or inline the §A.8 function — they live in the same module.)
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** `fit_anchor` (scipy `minimize` L-BFGS-B, `n_restarts` inits + `seed_theta`; finite-difference Hessian; fill `dials` via `param_spec.value`). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): penalized-MLE fit_anchor + FitResult (dials, Hessian, cond)`.

### Task 1.6: Rectification selection by CV-LL

**Interfaces:** `select_rectification(design_builder, expert_evidence, state_labels, mu, sigma, candidates=("signed","halfwave","asym"), k=5, seed=0) -> {"scores": dict, "winner": str}`.

- [ ] **Step 1: Failing test (non-vacuous).** Simulate expert data from a **signed**-evidence ground truth; `select_rectification` must score `signed` strictly above `halfwave` (CV-LL gap > a small margin), i.e. the scores must **differ meaningfully** and `winner=="signed"`.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (k-fold CV-LL over candidates via `Design.subset`; freeze the winner for the sweep). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): rectification selection by cross-validated log-likelihood`.

### Task 1.7: `build_anchor_designs` (the anchor-design dict)

**Interfaces:** `build_anchor_designs(sessions, param_spec, mu_by_session, sigma, dt=0.05, leak_tau=0.27, rectification="signed") -> dict[str, Design]` — loads each session, builds corrected evidence + state labels, filters to **`usable_generative`** cells, builds a `Design`, keyed by session. Resolves the §A.10-3 interface gap (explicit, not hidden in `backward_sweep`).

- [ ] **Step 1: Failing test** — two synthetic tagged sessions → a dict with 2 keyed `Design`s; a session with all-unusable cells is **omitted** from the dict.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (`del sess; gc.collect()` per session). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): build_anchor_designs (per-session Design dict, QC-gated)`.

---

## Phase 2 — Anchoring + model-comparison ladders

### Task 2.1: Expert-first, backward-seeded sweep

**Interfaces:** `backward_sweep(anchor_designs: dict[str, Design], anchors_chrono: list[str], param_spec, l2=1.0, seed=0) -> dict[str, FitResult]`. Fits the **most-expert** anchor freely, then walks earlier anchors in reverse-chronological order, each `fit_anchor(..., seed_theta=<neighbour's theta>, l2=l2)`.

- [ ] **Step 1: Failing test** — three synthetic anchors with a true `v` ramp; `backward_sweep` returns a `FitResult` per anchor whose recovered `v` ramps in the right direction; the expert anchor (fit first, `l2=0` implicitly via `seed_theta=None`) is closest to its truth.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (expert first with `seed_theta=None`; then reverse-chrono seeded). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): expert-first backward-seeded anchored sweep`.

### Task 2.2: Learning ladder (which dial moves)

**Interfaces:** `learning_ladder(anchor_designs: dict[str, Design], param_spec, dt=0.05, k=5, seed=0) -> {"winner": str, "aic": dict, "bic": dict, "cvll": dict}` with rungs `M_shared / M_sharpness / M_caution / M_timing / M_full`. Each rung fits the dials it lets vary across anchors (others shared); score with **GLM** AIC/BIC (`k_params` = the fitted free-parameter count, **not** the pyddm formula) and held-out CV-LL via `Design.subset`.

- [ ] **Step 1: Failing test (unambiguous).** Two anchors where **only `v` truly varies** (z, u identical, evidence non-trivial); assert `winner == "M_sharpness"` as the **strict argmin AIC** (not "in top-2"), and `M_sharpness` beats `M_caution` and `M_timing` on CV-LL.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (per-rung `ParamSpec` with the rung's dials in `state_terms`/shared-across-anchors; GLM dof). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): learning ladder (GLM AIC/BIC/CV-LL, which-dial-moves)`.

### Task 2.3: State ladder (which dial loads on mood)

**Interfaces:** `state_ladder(anchor_design: Design, param_spec, k=5, seed=0) -> {"winner": str, "aic": dict, "cvll": dict}` — within one anchor, which dial must carry a mood term (`M_none / M_v / M_z / M_u / M_all`).

- [ ] **Step 1: Failing test** — a Design where **only `z` differs by mood** → `winner == "M_z"` (strict argmin), beating `M_v`/`M_u`.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement.** [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): state ladder (which dial loads on mood)`.

### Task 2.4: Backward-seeding guardrails (Hessian conditioning + L2 sensitivity)

**Interfaces:** `hessian_conditioning(fit: FitResult) -> {"cond_number": float, "rank": int, "deficient": bool}` (`deficient` if `cond_number > 1e8` or `rank < n_params`); `l2_weight_sensitivity(anchor_designs, anchors_chrono, param_spec, weights=(0,0.01,0.1,1,10), seed=0) -> DataFrame` (the learning-ladder winner + key dial deltas per weight).

- [ ] **Step 1: Failing tests.** (a) A well-conditioned synthetic fit → `deficient==False`; a rank-deficient one (duplicate column) → `True`. (b) `l2_weight_sensitivity` returns one row per weight and the ladder winner is **stable** across `weights≥0.01` on a clean only-`v`-varies dataset (guards against the regularization manufacturing the trajectory).
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement.** [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): Hessian-conditioning + L2-weight sensitivity guardrails`.

---

## Phase 3 — Recovery gate (PRIMARY validation; rigor per §A.9)

### Task 3.0: Shared recovery fixtures (expert-like & naïve-like ground truth)

**Files:** Modify `tests/analysis/conftest.py` (add fixtures); Create `scripts/analysis/decision_latents/_recovery_fixtures.py` (reusable builder).

**Interfaces:** `make_recovery_design(regime: str, n_trials=2000, seed=0) -> (Design, true_theta, ParamSpec)` — `regime ∈ {"expert","naive"}`. Expert-like = more change-driven licks (higher `v`, lower `z`); naïve-like = more flat-evidence hair-trigger licks (low `v`, high `z`). Built on **realistic per-trial evidence** at `dt=0.05` (change_time ≥ 6 s).

- [ ] **Step 1: Implement the builder** (synthesize evidence with a TF change at a sampled `change_time≥6 s`; set `true_theta` per regime). Add `@pytest.fixture`s `recovery_design_expert`, `recovery_design_naive`.
- [ ] **Step 2: Sanity test** — both regimes produce Designs with a realistic lick/censor mix (e.g. 30–70 % licks) so recovery is non-degenerate.
- [ ] **Step 3: Commit** — `test(b8p2): shared recovery fixtures (expert-like & naive-like)`.

### Task 3.1: `simulate_licks`

**Interfaces:** `simulate_licks(design, true_theta, param_spec, seed=0) -> (event_bin, lick, censored)` + `design_with_outcomes(...)` (§A.8).

- [ ] **Step 1: Failing test (survival-aware, not a tautology).** With a **single** trial repeated `N=5000×` (same A/phi/mood), the empirical first-lick-bin distribution must match the theoretical `h_k·Π_{j<k}(1−h_j)` (chi-square / KS over bins, p>0.05), and the empirical censor rate matches `Π_k(1−h_k)`.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement §A.8.** [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): simulate_licks (draw through per-bin hazard)`.

### Task 3.2: `recover_point`

**Interfaces:** `recover_point(design, true_theta, param_spec, n_rep=100, seed=0) -> {per_dial: {"r": float, "bias": float, "ci_coverage": float}}` (§A.9).

- [ ] **Step 1: Failing test (ground-truth tolerances).** On `recovery_design_expert`, over a small grid of true dial values, assert **`r ≥ 0.8`** and **`|bias| ≤ 0.1·SD(true)`** per dial, and **CI coverage ∈ [0.90,0.97]**. Explicitly assert `abs(recovered_mean_v - true_v) < 0.1*SD` — NOT `ll > random_ll`.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (`n_rep` simulate→refit; collect dial estimates; bootstrap CI coverage). [ ] **Step 4: Run → PASS (expert regime); record naïve-regime numbers.**
- [ ] **Step 5: Commit** — `feat(b8p2): recover_point (per-dial r/bias/CI coverage)`.

### Task 3.3: `recover_confusion` (decisive)

**Interfaces:** `recover_confusion(design_template, base_theta, param_spec, n_rep=50, seed=0) -> {"matrix": np.ndarray(3,3), "labels": ("sharpness","caution","timing")}` (§A.9).

- [ ] **Step 1: Failing test.** Three only-one-dial-varies scenarios over two anchors; run `learning_ladder`; build the 3×3. Assert **`diag ≥ 0.8`** AND **`every off-diagonal ≤ 0.2`** (proves the sharpness↔caution and urgency↔itchiness trade-offs don't fool the ladder). A "diag > 0.55" check is INVALID (§A.9).
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement.** [ ] **Step 4: Run → PASS (or, if off-diagonals are high, that is a real finding → the conflated dial is marked descriptive-only downstream).**
- [ ] **Step 5: Commit** — `feat(b8p2): recover_confusion (which-dial-varies 3x3 matrix)`.

### Task 3.4: `recover_true_difference`

**Interfaces:** `recover_true_difference(design_naive, design_expert, param_spec, true_delta: dict, l2=1.0, seed=0) -> {"recovered_delta": dict, "shrunk": bool}` (§A.9).

- [ ] **Step 1: Failing test** — true `v_naive=1.0`, `v_expert=2.5`; the L2-seeded backward fit must recover `recovered_delta["v"]` with the right sign and `|recovered − 1.5| ≤ 0.3·1.5`, and `shrunk == False`.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (`shrunk = |recovered_delta| < 0.5·|true_delta|`). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): recover_true_difference (seeding informs, not erases)`.

### Task 3.5: `recovery_gate` (per-dial decision)

**Interfaces:** `recovery_gate(point_res, confusion_res, truediff_res, cond_res, regime, r_min=0.8, bias_max_frac=0.1, confusion_min_diag=0.8, confusion_max_offdiag=0.2, naive_relax=0.1) -> {"per_dial_trust": {dial: "generative"|"descriptive"}, "regime": str, "passed": dict}` (§A.9).

- [ ] **Step 1: Failing test (edge cases).** Mock inputs where sharpness passes all, caution fails the confusion off-diagonal, timing fails point-recovery, and one anchor is Hessian-deficient → `per_dial_trust == {sharpness:"generative", caution:"descriptive", timing:"descriptive"}`; naïve-regime relaxation applies the single `naive_relax` constant.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement.** [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): recovery_gate (per-dial generative/descriptive trust)`.

---

## Phase 4 — Latents, orchestration, Engine C, figures

### Task 4.1: `append_generative_latents`

**Interfaces:** `append_generative_latents(per_trial_csv, anchor_fits: dict, recovery: dict, param_spec, mu_by_session, trial_evidence_by_session) -> DataFrame` — appends, per trial: `sharpness_drift, itchiness_caution, timing_urgency_at_decision, evidence_integral_at_decision, expected_change_time, lick_minus_expected, anchor_id, rectification_kind, leak_tau, recovery_regime, latent_trust`. **Never overwrites the 25 Phase-1 columns**; `latent_trust` per dial comes from `recovery["per_dial_trust"]`.

- [ ] **Step 1: Failing test** — a 2-session mock with fitted anchors → output has the 25 original + 11 new columns; `timing_urgency_at_decision == (u0+u_state)*phi[event_bin]` (a genuinely trial-specific realized value, not the coefficient); `evidence_integral_at_decision == A[event_bin]`; dials failing recovery get `latent_trust=="descriptive"`.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (compute realized quantities from the trial's `A`/`phi` at `event_bin`; `lick_minus_expected = decision_time − μ_session`). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): append generative latents + provenance to the deliverable`.

### Task 4.2: Orchestration script (with the contingency gate)

**Files:** Create `scripts/analysis/decision_latents/run_decision_latents_phase2.py` (module docstring carries the worktree run-recipe + a local `save_fig`).

- [ ] **Step 1: Write the pipeline** — load inventory → `select_expert_anchors` (Task 0.9). If `mode=="fallback"`, write the Phase-1 proxies as the latent table (`latent_trust="descriptive"`), emit a clear log, and **stop** (no generative fit). Else: `build_anchor_designs` → `select_rectification` on the expert anchor → `backward_sweep` → `learning_ladder` + `state_ladder` → recovery (`recover_point`/`recover_confusion`/`recover_true_difference` at both regimes) → `recovery_gate` → `append_generative_latents`. Cache results.
- [ ] **Step 2: Run end-to-end on real data** `PYTHONPATH="$(pwd)/src" py scripts/analysis/decision_latents/run_decision_latents_phase2.py`. Confirm the appended CSV, the ladder winners, and the per-dial trust print. **Record observations** (these drive F6–F8).
- [ ] **Step 3: Commit** — `feat(b8p2): Phase-2 orchestration (sweep+ladders+recovery+append, contingency-gated)`.

### Task 4.3: Engine-C pyddm spot-check (REQUIRED)

**Interfaces:** `engine_c_spotcheck(expert_sessions, dt=0.02) -> DataFrame` — fit B0's pyddm model (`ddm.build_model`/`ddm.fit_model`, **no mutation**) on ≥1 expert session; return `{session, v, u, a, z, ll}`. A genuine pyddm failure is caught + logged and the row marked `failed=True` (never silently skipped).

- [ ] **Step 1: Failing test** — on a tiny synthetic pyddm-friendly session the fn returns a row with finite `v`,`u`; an induced failure returns `failed=True` with a logged reason.
- [ ] **Step 2: Run → FAIL.** [ ] **Step 3: Implement** (use `ddm.fit_model` per its API contract; `dt=0.02` for pyddm). [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8p2): Engine-C pyddm spot-check (GLM-vs-DDM construct validity)`.

### Task 4.4: Figures F6 / F7 / F8 (timing-led; circularity caveat in captions)

**Files:** Add figure fns to `run_decision_latents_phase2.py`; save to `FIGURES/decision_latents/BG_046/` via the local `save_fig`.

- [ ] **Step 1: F6 (recovery)** — recovered-vs-true scatter per dial (both regimes) + the 3×3 confusion matrix + the Hessian-condition / L2-sensitivity summary. Title: *"Can we trust the dials? (recovery at the real long-baseline regime)."*
- [ ] **Step 2: F7 (latent distributions, timing-led)** — the three dials by mood and across learning anchors, **leading with timing + RT variability**; a caption/callout: *"Leading with timing (labeler-independent) and RT variability. FA-rate / criterion × mood are confirmatory — the mood labels are defined partly from early-lick features (state-label circularity; memory/state_labeler_circularity_caveat)."* Mood colours from `STATE_LABEL_COLORS`.
- [ ] **Step 3: F8 (construct validity)** — generative dials vs Phase-1 descriptive scores (sharpness↔lapse-aware psychometric, caution↔criterion/FA, timing↔change-time anchor) **+ the Engine-C panel** (GLM sharpness/urgency vs DDM `v`/`u`; note `a` vs `z` where afforded). Caption repeats the circularity caveat for the caution panel.
- [ ] **Step 4: Run; eyeball; confirm all three PNGs render.**
- [ ] **Step 5: Commit** — `feat(b8p2): figures F6 (recovery) / F7 (latents) / F8 (construct validity)`.

---

## Phase 5 — Index

### Task 5.1: Update the question index

**Files:** Modify `docs/science/QUESTION_INDEX.md`.

- [ ] **Step 1:** Add/borrow the B8 row: status → `phase2-plan`; add a Plan cell linking `../superpowers/plans/2026-06-20-B8-phase2-generative-latents-plan.md` and the Phase-2 spec; update `_Last updated_` to `2026-06-20`.
- [ ] **Step 2: Commit** — `docs(b8p2): index B8 Phase-2 plan + spec`.

---

## Self-Review

**Spec coverage (Phase-2 spec → task):**
- §2 model (cloglog hazard-accumulator, 3 dials, σ-fixed φ, declarative ParamSpec) → §A + Tasks 1.1–1.4.
- §3 knob decisions: rectification by CV-LL → 1.6; leak fixed 0.27 + sweep → §A.4 + 2.4 (L2/sweep) + Global Constraints; which-dial-moves tested not assumed → 2.2/2.3.
- §4 anchoring (expert-first backward-seeded; two ladders; distance-from-template) → 1.7 + 2.1/2.2/2.3; manufactured-trajectory guardrails (Hessian, L2 sensitivity, recover-a-true-difference) → 2.4 + 3.4.
- §5 recovery gate (both regimes; point + confusion + true-difference; per-dial latent_trust; Hessian pre-flight; provisional thresholds) → §A.9 + 3.0–3.5.
- §6 confounds: bound/start-point conflation validated by confusion + Engine-C → 3.3 + 4.3; urgency↔itchiness separability (σ-fixed + empirical anchor) → §A.3 + 0.4 + 3.3; **state-label circularity led-with-timing** → 4.4 captions; two-impulsivities/comprehension → 0.6; satiety covariate `trial_in_session` (carried in the Phase-1 table) → 4.1; naïve-label reliability → 0.7; expert-anchor sufficiency → 0.8/0.9; n=1 / F3 architecture (subject-parameterized) → preserved via `subject=` in Phase-1 accessors.
- §7 the five carried-forward fixes (a evidence builder, b lapse psychometric, c baseline-hazard window, d change-time anchor, e QC re-derivation + generative gate) → 0.1/0.2/0.3/0.4/0.5; comprehension (f) → 0.6; naïve labels (g) → 0.7; expert inventory + contingency (h) → 0.8/0.9.
- §8 Engine-C required → 4.3 + 4.4(F8). §9 latent columns + provenance + figures → 4.1/4.4. §10 fallbacks (per-dial descriptive + full Phase-1 fallback) → 3.5 + 4.2 contingency. §13 index → 5.1.

**Placeholder scan:** Core code is fully specified in §A and reused verbatim; per-task code is concrete (real tests with ground-truth assertions, exact commands). No "TBD / handle edge cases / similar to Task N." The only deferred items are the **provisional recovery thresholds** (explicitly flagged statistician-confirmable, spec §11) and the **`QC_GEN_*` / comprehension-rule / leak values**, which are *set from a profiler/inventory run inside their own task* (the data-quality-gate-first discipline), not left blank.

**Type consistency:** `Design`/`ParamSpec`/`FitResult` are defined once (§A.4–A.7) and consumed unchanged everywhere; the cross-task interface gaps the review flagged are closed — `build_anchor_designs` (1.7) builds the `dict[str, Design]` that `backward_sweep`/`learning_ladder` consume; `fit_anchor` returns the locked `FitResult` (incl. `hessian`/`hessian_cond`) that `hessian_conditioning` (2.4) reads; `dials = {mood:{sharpness,itchiness,timing}}` is the single structure `append_generative_latents` (4.1) reads; `simulate_licks` (§A.8/3.1) feeds every recovery test; `hazard_from_lp`/`lp_from_hazard` replace the ambiguous `cloglog`/`inv_cloglog`. `ParamSpec.per_trial` removes all hardcoded `theta` indexing.

**Sequencing note:** Task 1.5's recovery test imports `simulate_licks` (§A.8), implemented in Task 3.1 — both live in the same module, so implement §A.8 when first needed (the executor may pull Task 3.1's `simulate_licks` forward; it is pure and self-contained).

