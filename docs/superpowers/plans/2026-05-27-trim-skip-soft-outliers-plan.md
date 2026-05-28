# Skip-able Trimming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `find_stable_subset` keep non-contiguous good sessions across soft-outlier gaps when ISI fingerprint consistency confirms cell identity holds across the gap.

**Architecture:** Extend `session_outlier_flags` with soft/hard classification. Replace the contiguous-slice `longest_good_run` with a skip-able algorithm gated by set-wide `baseline_isi_hist_corr`. The old algorithm becomes a private fallback. `find_stable_subset` exposes `skipped_indices` alongside the redefined `kept_indices`/`dropped_indices`. The CLI driver adds one CSV column and unions skipped+dropped at the render call site (renderer code unchanged).

**Tech Stack:** Python 3.10, numpy, pytest. Project's `py` launcher (Windows).

**Spec:** `docs/superpowers/specs/2026-05-27-trim-skip-soft-outliers-design.md` (commit `5e6f54c` on main).

**Prerequisites:**

A worktree on `main` exists at `.claude/worktrees/trim-skip-review/`. Create the feature branch IN THAT WORKTREE so all implementation work stays isolated from the parallel chats sharing the main work-dir:

```bash
cd .claude/worktrees/trim-skip-review
git checkout -b feature/trim-skip-soft-outliers
git branch --show-current   # must print: feature/trim-skip-soft-outliers
```

All `Bash` commands in this plan assume `cwd` is the worktree path. **Before EVERY commit, run `git branch --show-current` and confirm it prints `feature/trim-skip-soft-outliers`.** The user runs parallel chats; the main work-dir's branch can shift unexpectedly.

---

## Task 1: Add `is_hard_outlier` and `is_soft_outlier` keys to `session_outlier_flags`

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py` (`session_outlier_flags` function, lines ~989-1089)
- Modify: `tests/analysis/test_tracking_qc.py` (append 1 test)

- [ ] **Step 1: Write the failing test**

Append to `tests/analysis/test_tracking_qc.py`:

```python
def test_session_outlier_flags_classifies_hard_vs_soft():
    """is_hard_outlier = wave OR depth (independent of composite is_outlier rule).
    is_soft_outlier = is_outlier AND NOT is_hard_outlier.

    Note: is_hard_outlier and is_outlier are independent signals — a session
    with ONLY a wave flag (strikes=1) is is_hard_outlier=True but NOT
    is_outlier (the existing composite rule requires isi_peak OR strikes>=2
    OR unknown_stage). The new algorithm uses both flags separately."""
    h_clean = np.zeros(50, dtype=np.float32); h_clean[15] = 1.0
    h_bimodal = np.zeros(50, dtype=np.float32); h_bimodal[35] = 1.0
    wave_clean = np.array([0.0, 1.0, 0.0, -1.0, 0.0] * 16 + [0.0, 1.0], dtype=np.float32)
    wave_flipped = -wave_clean
    def mk_rec(name, stage, peak_hist, fr, wave, depth):
        return qc.SessionRecord(
            session_name=name, ks_unit_id=0, stage=stage,
            peak_chan=0, peak_depth_um=float(depth), amplitude=1.0,
            baseline_fr_hz=float(fr), waveform_peak=wave,
            footprint=np.zeros((82, 17), dtype=np.float32),
            footprint_channels=np.arange(17),
            isi_hist=peak_hist, isi_centers=np.zeros(50, dtype=np.float32),
        )
    sessions = [
        mk_rec("s00", "Learning", h_clean,    5.0, wave_clean,   1000.0),  # clean
        mk_rec("s01", "Learning", h_clean,    5.0, wave_flipped, 1000.0),  # wave-only: HARD, NOT is_outlier
        mk_rec("s02", "Learning", h_bimodal,  5.0, wave_clean,   1000.0),  # isi_peak alone: SOFT, is_outlier
        mk_rec("s03", "Learning", h_clean,    5.0, wave_clean,   1000.0),  # clean
        mk_rec("s04", "Unknown",  h_clean,    5.0, wave_clean,   1000.0),  # unknown_stage: SOFT, is_outlier
    ]
    uid = qc.UIDIntermediate(
        global_uid=1, span=5, has_naive_to_expert=False,
        suspect_known=False, sessions=sessions,
    )
    f = qc.session_outlier_flags(uid)
    assert "is_hard_outlier" in f
    assert "is_soft_outlier" in f
    # s01: wave-only triggers is_hard_outlier but NOT is_outlier (strikes=1)
    # s02: isi_peak alone triggers is_outlier directly; soft (no wave/depth)
    # s04: unknown_stage triggers is_outlier directly; soft (no wave/depth)
    assert f["is_hard_outlier"] == [False, True,  False, False, False]
    assert f["is_soft_outlier"] == [False, False, True,  False, True]
    assert f["is_outlier"]      == [False, False, True,  False, True]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
py -m pytest tests/analysis/test_tracking_qc.py::test_session_outlier_flags_classifies_hard_vs_soft -v 2>&1 | tail -10
```

Expected: FAIL with `AssertionError` on `"is_hard_outlier" in f` (the key doesn't exist yet).

- [ ] **Step 3: Add the two new keys to `session_outlier_flags`**

In `src/visdetect/analysis/tracking_qc.py`, find the `out = {...}` initialization at the top of `session_outlier_flags` (around line 1002). Add two new keys alongside the existing ones:

```python
    out = {
        "isi_peak":         [False] * n,
        "fr":               [False] * n,
        "wave":             [False] * n,
        "depth":            [False] * n,
        "unknown_stage":    [False] * n,
        "is_hard_outlier":  [False] * n,   # NEW
        "is_soft_outlier":  [False] * n,   # NEW
        "is_outlier":       [False] * n,
    }
```

Then find the composite-outlier loop at the end of the function (around line 1078-1087). Replace it with:

```python
    # Composite outlier rule
    for i in range(n):
        strikes = sum([out["isi_peak"][i], out["fr"][i], out["wave"][i], out["depth"][i]])
        # ISI peak divergence alone is sufficient (strongest single signal);
        # otherwise need >=2 criteria; unknown-stage always forces outlier.
        out["is_outlier"][i] = (
            out["isi_peak"][i]
            or strikes >= 2
            or out["unknown_stage"][i]
        )
        # Hard vs soft classification (used by skip-able trimming).
        # Hard = wave or depth outlier — strongly suggests a different physical
        # unit at this probe position; never skip across these. Soft = any
        # other outlier type (unknown_stage, fr, isi_peak) — data-quality or
        # transient issues; cell identity may be intact, eligible for skip.
        out["is_hard_outlier"][i] = out["wave"][i] or out["depth"][i]
        out["is_soft_outlier"][i] = out["is_outlier"][i] and not out["is_hard_outlier"][i]

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -q 2>&1 | tail -3
```

Expected: all tests pass (was 68 before, now 69).

- [ ] **Step 5: Confirm branch + commit**

```bash
git branch --show-current   # must print: feature/trim-skip-soft-outliers
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py
git commit -m "Add is_hard_outlier and is_soft_outlier classification to session_outlier_flags"
```

---

## Task 2: Skip-able `longest_good_run` + thread through `find_stable_subset`

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py` (`longest_good_run` and `find_stable_subset`, lines ~1092-1131)
- Modify: `tests/analysis/test_tracking_qc.py` (rewire 3 existing tests, add 4 new tests)

The old `longest_good_run(is_outlier) -> Tuple[int, int]` is renamed to a private `_longest_good_run_contiguous` (used as fallback). The new public `longest_good_run` takes hard-outlier flags + ISI hists and returns a `Dict[str, List[int]]` with `kept_indices` and `skipped_indices`. `find_stable_subset` is updated to consume the new return shape and expose `skipped_indices`.

- [ ] **Step 1: Rewire the 3 existing `longest_good_run` tests to point at the renamed private helper**

In `tests/analysis/test_tracking_qc.py`, find the existing `test_longest_good_run_basic`, `test_longest_good_run_all_bad_returns_zero`, `test_longest_good_run_all_good_returns_full` (around lines 318-329). Rename their function targets:

```python
def test_longest_good_run_contiguous_basic():
    # 0..2 good, 3 bad, 4..7 good (length 4) → best (4,8)
    flags = [False, False, False, True, False, False, False, False]
    assert qc._longest_good_run_contiguous(flags) == (4, 8)


def test_longest_good_run_contiguous_all_bad_returns_zero():
    assert qc._longest_good_run_contiguous([True, True, True]) == (0, 0)


def test_longest_good_run_contiguous_all_good_returns_full():
    assert qc._longest_good_run_contiguous([False, False, False, False]) == (0, 4)
```

(Test names changed from `test_longest_good_run_*` to `test_longest_good_run_contiguous_*`. The bodies are unchanged except `qc.longest_good_run` → `qc._longest_good_run_contiguous`.)

- [ ] **Step 2: Add 4 new failing tests for the new `longest_good_run` and `find_stable_subset` skip behavior**

Append to `tests/analysis/test_tracking_qc.py`:

```python
# ─── Skip-able longest_good_run ───────────────────────────────────────

def _identical_isi_hists(n: int) -> list:
    """Helper: n copies of a fixed peak-15 log-ISI histogram. Guarantees
    set-wide isi_hist_corr == 1.0 (passes gate trivially)."""
    h = np.zeros(50, dtype=np.float32); h[15] = 0.5; h[14] = 0.25; h[16] = 0.25
    return [h.copy() for _ in range(n)]


def test_longest_good_run_skips_soft_with_high_consistency():
    """Sequence [G, G, S, G, G] with identical ISI hists → all 4 good kept,
    soft outlier at index 2 is skipped, NO sessions dropped (no hard
    outliers). Set-wide isi_hist_corr = 1.0 passes 0.85 gate trivially."""
    is_outlier      = [False, False, True,  False, False]
    is_hard_outlier = [False, False, False, False, False]
    hists = _identical_isi_hists(5)
    out = qc.longest_good_run(is_outlier, is_hard_outlier, hists)
    assert out["kept_indices"] == [0, 1, 3, 4]
    assert out["skipped_indices"] == [2]


def test_longest_good_run_falls_back_when_consistency_fails():
    """Sequence [G, G, S, G, G] where the two good halves have DIFFERENT
    ISI shapes → set-wide isi_hist_corr fails 0.85 gate → falls back to
    longest contiguous all-good run = [0,1] (length 2; ties broken by
    first-encountered)."""
    is_outlier      = [False, False, True,  False, False]
    is_hard_outlier = [False, False, False, False, False]
    h_a = np.zeros(50, dtype=np.float32); h_a[10] = 1.0
    h_b = np.zeros(50, dtype=np.float32); h_b[40] = 1.0
    # Soft outlier at index 2 (any shape — will be skipped); halves divergent
    hists = [h_a.copy(), h_a.copy(), h_a.copy(), h_b.copy(), h_b.copy()]
    out = qc.longest_good_run(is_outlier, is_hard_outlier, hists)
    # Set [0,1,3,4]: pairs (0,1)=1, (0,3)=-1, (0,4)=-1, (1,3)=-1, (1,4)=-1, (3,4)=1
    # median = -1 < 0.85 → fallback
    # Fallback picks first longest contiguous good run = [0,1]
    assert out["kept_indices"] == [0, 1]
    assert out["skipped_indices"] == []


def test_longest_good_run_never_skips_hard_outliers():
    """Sequence [G, G, H, G, G] where H is a HARD outlier → must NEVER appear
    in kept or skipped. Result is one of [0,1] or [3,4] (both length 2)."""
    is_outlier      = [False, False, True, False, False]
    is_hard_outlier = [False, False, True, False, False]
    hists = _identical_isi_hists(5)
    out = qc.longest_good_run(is_outlier, is_hard_outlier, hists)
    assert 2 not in out["kept_indices"]
    assert 2 not in out["skipped_indices"]
    # Tie-break: largest kept_set; ties → longest span (kept+skipped); ties
    # → earliest start. [0,1] starts earlier, so it wins on the last tie-break.
    assert out["kept_indices"] == [0, 1]
    assert out["skipped_indices"] == []


def test_find_stable_subset_returns_skipped_indices():
    """find_stable_subset exposes skipped_indices and redefines dropped_indices
    to exclude skipped. Set: [Learning, Learning, Unknown, Learning, Learning]
    with identical ISI → kept=[0,1,3,4], skipped=[2], dropped=[]."""
    h = np.zeros(50, dtype=np.float32); h[15] = 0.5; h[14] = 0.25; h[16] = 0.25
    wave = np.array([0.0, 1.0, 0.0, -1.0, 0.0] * 16 + [0.0, 1.0], dtype=np.float32)
    def mk_rec(name, stage):
        return qc.SessionRecord(
            session_name=name, ks_unit_id=0, stage=stage,
            peak_chan=0, peak_depth_um=1000.0, amplitude=1.0,
            baseline_fr_hz=5.0, waveform_peak=wave,
            footprint=np.zeros((82, 17), dtype=np.float32),
            footprint_channels=np.arange(17),
            isi_hist=h.copy(), isi_centers=np.zeros(50, dtype=np.float32),
        )
    sessions = [mk_rec(f"s{i:02d}", "Unknown" if i == 2 else "Learning")
                for i in range(5)]
    uid = qc.UIDIntermediate(
        global_uid=1, span=5, has_naive_to_expert=False,
        suspect_known=False, sessions=sessions,
    )
    out = qc.find_stable_subset(uid)
    assert "skipped_indices" in out
    assert out["kept_indices"]    == [0, 1, 3, 4]
    assert out["skipped_indices"] == [2]
    assert out["dropped_indices"] == []
    # Sanity: union covers all sessions and the three sets are disjoint
    union = set(out["kept_indices"]) | set(out["skipped_indices"]) | set(out["dropped_indices"])
    assert union == set(range(5))
    assert (set(out["kept_indices"]) & set(out["skipped_indices"])) == set()
    assert (set(out["kept_indices"]) & set(out["dropped_indices"])) == set()
    assert (set(out["skipped_indices"]) & set(out["dropped_indices"])) == set()
```

- [ ] **Step 3: Run tests to verify the 4 new tests fail and the 3 renamed tests fail**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -v -k "longest_good_run or find_stable_subset_returns_skipped" 2>&1 | tail -20
```

Expected:
- 3 renamed tests fail with `AttributeError: module 'visdetect.analysis.tracking_qc' has no attribute '_longest_good_run_contiguous'`.
- 4 new tests fail with `TypeError` (longest_good_run signature mismatch) or `AssertionError`.

- [ ] **Step 4: Rename old `longest_good_run` to `_longest_good_run_contiguous`**

In `src/visdetect/analysis/tracking_qc.py`, find the existing `longest_good_run` function (around lines 1092-1108). Rename the function:

```python
def _longest_good_run_contiguous(is_outlier: Sequence[bool]) -> Tuple[int, int]:
    """Return (start_idx, end_idx_exclusive) of the longest contiguous run of
    non-outlier sessions. (0, 0) if no good sessions.

    Internal helper: used as the fallback inside `longest_good_run` when the
    skip-able algorithm cannot find a span whose kept set passes the
    consistency gate."""
    best_start, best_end = 0, 0
    cur_start = None
    arr = list(is_outlier) + [True]  # sentinel
    for i, bad in enumerate(arr):
        if not bad:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                length = i - cur_start
                if length > (best_end - best_start):
                    best_start, best_end = cur_start, i
                cur_start = None
    return best_start, best_end
```

- [ ] **Step 5: Add the new `longest_good_run` with skip logic**

In `src/visdetect/analysis/tracking_qc.py`, IMMEDIATELY AFTER `_longest_good_run_contiguous`, add the new public `longest_good_run`:

```python
def longest_good_run(
    is_outlier: Sequence[bool],
    is_hard_outlier: Sequence[bool],
    isi_hists: Sequence[Optional[np.ndarray]],
    *,
    threshold: float = ISI_HIST_CORR_PASS,
) -> Dict[str, List[int]]:
    """Skip-able trim: largest set of non-outlier sessions inside any
    hard-outlier-free span whose set-wide ISI hist correlation passes
    `threshold`.

    Algorithm:
      1. Find all maximal contiguous spans containing NO hard outliers.
      2. For each span, candidate kept_set = sessions in the span that are
         NOT outliers of any kind (soft or hard).
      3. Compute set-wide baseline_isi_hist_corr on kept_set's hists.
      4. If correlation >= threshold (or fewer than 2 kept — gate
         trivially satisfied for size 1; size 0 disqualifies), the span
         qualifies. The skipped_set = soft outliers inside the span.
      5. Pick the span with the LARGEST kept_set (ties → longest span,
         then earliest start).
      6. If NO span qualifies, fall back to longest contiguous all-good
         run (no skipping).

    Returns
    -------
    Dict[str, List[int]] with keys 'kept_indices' (sorted) and
    'skipped_indices' (sorted). Indices outside the chosen span (or
    hard outliers anywhere) are NOT returned by this function — the
    caller computes 'dropped' as the complement.
    """
    n = len(is_outlier)
    if n == 0:
        return {"kept_indices": [], "skipped_indices": []}

    # Step 1: maximal hard-outlier-free spans
    spans: List[Tuple[int, int]] = []  # [(start, end_exclusive), ...]
    cur_start: Optional[int] = None
    for i in range(n):
        if is_hard_outlier[i]:
            if cur_start is not None:
                spans.append((cur_start, i))
                cur_start = None
        else:
            if cur_start is None:
                cur_start = i
    if cur_start is not None:
        spans.append((cur_start, n))

    # Step 2-4: evaluate each span
    best_kept: List[int] = []
    best_skipped: List[int] = []
    best_span_len = 0
    best_span_start = 10**9
    for (s, e) in spans:
        kept = [i for i in range(s, e) if not is_outlier[i]]
        skipped = [i for i in range(s, e) if is_outlier[i]]
        if not kept:
            continue
        if len(kept) >= 2:
            kept_hists = [isi_hists[i] for i in kept]
            corr = baseline_isi_hist_corr(kept_hists)
            if not (np.isfinite(corr) and corr >= threshold):
                continue
        # Step 5 tie-breaking: larger kept, then longer span, then earlier start
        span_len = e - s
        better = (
            len(kept) > len(best_kept)
            or (len(kept) == len(best_kept) and span_len > best_span_len)
            or (len(kept) == len(best_kept) and span_len == best_span_len and s < best_span_start)
        )
        if better:
            best_kept = kept
            best_skipped = skipped
            best_span_len = span_len
            best_span_start = s

    if best_kept:
        return {"kept_indices": best_kept, "skipped_indices": best_skipped}

    # Step 6: fallback to contiguous-all-good
    start, end = _longest_good_run_contiguous(is_outlier)
    return {"kept_indices": list(range(start, end)), "skipped_indices": []}
```

- [ ] **Step 6: Update `find_stable_subset` to consume the new shape and expose `skipped_indices`**

In `src/visdetect/analysis/tracking_qc.py`, find `find_stable_subset` (around lines 1111-1131). Replace its body:

```python
def find_stable_subset(uid: "UIDIntermediate") -> Dict[str, object]:
    """Identify a stable kept subset of sessions for this UID, allowing
    skip-over of soft outliers when cross-gap ISI fingerprint consistency
    holds.

    Returns
    -------
    dict with keys:
        outlier_flags    : Dict[str, List[bool]]  (from session_outlier_flags)
        kept_indices     : List[int]              (GOOD sessions in kept span)
        skipped_indices  : List[int]              (soft outliers inside span)
        dropped_indices  : List[int]              (outside span, or hard
                                                    outliers anywhere)
        trimmed_span     : int                    (len of kept_indices)

    Invariants:
        kept ∪ skipped ∪ dropped == range(len(uid.sessions))
        the three sets are pairwise disjoint
    """
    flags = session_outlier_flags(uid)
    isi_hists = [r.isi_hist for r in uid.sessions]
    run = longest_good_run(
        flags["is_outlier"], flags["is_hard_outlier"], isi_hists,
    )
    kept = run["kept_indices"]
    skipped = run["skipped_indices"]
    kept_set = set(kept)
    skipped_set = set(skipped)
    dropped = [i for i in range(len(uid.sessions))
               if i not in kept_set and i not in skipped_set]
    return {
        "outlier_flags": flags,
        "kept_indices": kept,
        "skipped_indices": skipped,
        "dropped_indices": dropped,
        "trimmed_span": len(kept),
    }
```

- [ ] **Step 7: Run all tracking_qc tests to verify**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -q 2>&1 | tail -3
```

Expected: all tests pass (was 69, now 73 — added 4 new tests; 3 existing `test_longest_good_run_*` renamed in place).

- [ ] **Step 8: Confirm branch + commit**

```bash
git branch --show-current   # must print: feature/trim-skip-soft-outliers
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py
git commit -m "Replace longest_good_run with skip-able algorithm; thread skipped_indices through find_stable_subset"
```

---

## Task 3: Wire `skipped_sessions` into CSV + visual union in `build_qc_sheets`

**Files:**
- Modify: `scripts/pipelines/tracking/build_qc_sheets.py` (`uid_trim_info` build loop, render loop, `trimmed_rows.append` blocks)

No new unit tests — integration verified by Task 4 smoke. There's a smoke `--help` check.

- [ ] **Step 1: Capture `skipped` in `uid_trim_info`**

In `scripts/pipelines/tracking/build_qc_sheets.py`, find the `uid_trim_info` build loop (around line 301). Find:

```python
    uid_trim_info: Dict[int, Dict[str, object]] = {}
    for uid in uids_to_render:
        iv = intermediates[uid]
        if not iv.sessions:
            continue
        stable = find_stable_subset(iv)
        kept = stable["kept_indices"]
        dropped = stable["dropped_indices"]
```

Add `skipped` extraction immediately after:

```python
    uid_trim_info: Dict[int, Dict[str, object]] = {}
    for uid in uids_to_render:
        iv = intermediates[uid]
        if not iv.sessions:
            continue
        stable = find_stable_subset(iv)
        kept = stable["kept_indices"]
        skipped = stable["skipped_indices"]
        dropped = stable["dropped_indices"]
```

Then find the `uid_trim_info[uid] = {...}` dict (around line 328) and add `skipped_indices`:

```python
        uid_trim_info[uid] = {
            "stable": stable,
            "kept_indices": kept,
            "skipped_indices": skipped,
            "dropped_indices": dropped,
            "trimmed_metrics": tm,
            "trimmed_verdict": tv,
        }
```

- [ ] **Step 2: Pass `dropped ∪ skipped` as the "visually dropped" set to `write_uid_pdf`**

Find the `write_uid_pdf` call in the main render loop (around line 348). Find:

```python
        verdict_pdf = write_uid_pdf(
            out_path, iv, pair_scores.get(uid),
            isi_score=isi,
            depth_std=metrics["depth_std_um"],
            wave_corr=metrics["wave_corr"],
            fr_cv_val=metrics["fr_cv"],
            dropped_indices=list(trim["dropped_indices"]),
            n_kept=len(trim["kept_indices"]),
            trimmed_verdict=str(trim["trimmed_verdict"]),
        )
```

Replace the `dropped_indices=` line with the union of dropped + skipped. Renderer still receives a single set; that set is now "dropped ∪ skipped" = "visually dim/hatch these":

```python
        # Skipped sessions render identically to dropped per spec §3.5;
        # union both into the single "visually dim" set the renderer expects.
        visually_dropped = sorted(set(trim["dropped_indices"]) | set(trim["skipped_indices"]))
        verdict_pdf = write_uid_pdf(
            out_path, iv, pair_scores.get(uid),
            isi_score=isi,
            depth_std=metrics["depth_std_um"],
            wave_corr=metrics["wave_corr"],
            fr_cv_val=metrics["fr_cv"],
            dropped_indices=visually_dropped,
            n_kept=len(trim["kept_indices"]),
            trimmed_verdict=str(trim["trimmed_verdict"]),
        )
```

- [ ] **Step 3: Add `skipped_sessions` column to `verdicts_trimmed.csv` (kept branch)**

Find the `trimmed_rows.append({...})` block in the kept-sessions branch (around line 429). Find:

```python
        trimmed_rows.append({
            "global_uid": uid,
            "original_span": iv.span,
            "trimmed_span": len(kept),
            "n_dropped": len(dropped_sessions),
            "dropped_sessions": ";".join(r.session_name for r in dropped_sessions),
            "kept_sessions": ";".join(r.session_name for r in kept_sessions),
            "trimmed_depth_std_um":            tm["depth_std_um"],
```

Modify it to capture `skipped` from `trim` and add the new column:

```python
        kept_sessions = [iv.sessions[i] for i in kept]
        skipped_sessions = [iv.sessions[i] for i in trim["skipped_indices"]]
        dropped_sessions = [iv.sessions[i] for i in dropped]
        # Look up the original CSV verdict for comparison
        original_verdict = next((r["verdict"] for r in rows if r["global_uid"] == uid), "")
        rescued = (original_verdict == "suspect" and tv in ("trusted", "review")
                   and len(kept) >= 5)
        trimmed_rows.append({
            "global_uid": uid,
            "original_span": iv.span,
            "trimmed_span": len(kept),
            "n_dropped": len(dropped_sessions),
            "dropped_sessions": ";".join(r.session_name for r in dropped_sessions),
            "skipped_sessions": ";".join(r.session_name for r in skipped_sessions),
            "kept_sessions": ";".join(r.session_name for r in kept_sessions),
            "trimmed_depth_std_um":            tm["depth_std_um"],
```

(Add the `skipped_sessions = ...` extraction near the existing `kept_sessions` and `dropped_sessions` extractions, and add the `"skipped_sessions"` dict key between `"dropped_sessions"` and `"kept_sessions"`.)

- [ ] **Step 4: Add `skipped_sessions` to the no-kept (early-exit) row**

Find the early-exit `trimmed_rows.append` block (around line 415) used when `not kept`. Find:

```python
        if not kept:
            trimmed_rows.append({
                "global_uid": uid, "original_span": iv.span,
                "trimmed_span": 0,
                "dropped_sessions": ";".join(r.session_name for r in iv.sessions),
                "kept_sessions": "", "trimmed_verdict": "suspect",
                "rescued": False,
            })
            continue
```

Add `"skipped_sessions": ""` to maintain column symmetry (pandas will fill missing as NaN otherwise, but explicit empty string is cleaner):

```python
        if not kept:
            trimmed_rows.append({
                "global_uid": uid, "original_span": iv.span,
                "trimmed_span": 0,
                "dropped_sessions": ";".join(r.session_name for r in iv.sessions),
                "skipped_sessions": "",
                "kept_sessions": "", "trimmed_verdict": "suspect",
                "rescued": False,
            })
            continue
```

- [ ] **Step 5: Smoke check that the module still imports + `--help` runs**

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --help 2>&1 | head -5
```

Expected: argparse help block, no exception.

- [ ] **Step 6: Confirm branch + commit**

```bash
git branch --show-current   # must print: feature/trim-skip-soft-outliers
git add scripts/pipelines/tracking/build_qc_sheets.py
git commit -m "Wire skipped_sessions into verdicts_trimmed.csv + union dropped+skipped for renderer"
```

---

## Task 4: End-to-end smoke + cohort shift report

**Files:** none modified — verification + final annotation commit.

- [ ] **Step 1: Rebuild cache (required — new algorithm changes per-UID trim outputs)**

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --rebuild-cache 2>&1 | tail -15
```

Expected: takes ~5-10 min. Output ends with `Wrote .../verdicts.csv` and `Wrote .../verdicts_trimmed.csv (N rescued)`.

- [ ] **Step 2: Verify `skipped_sessions` column is present in `verdicts_trimmed.csv` + verify cohort cross-stage coverage**

```bash
py -c "
import pandas as pd
import pickle
t = pd.read_csv('FIGURES/tracking_qc/verdicts_trimmed.csv')
print('Trimmed CSV columns:', list(t.columns))
print()
assert 'skipped_sessions' in t.columns, 'skipped_sessions column missing!'

print('Trimmed verdict distribution:')
print(t['trimmed_verdict'].value_counts().to_string())
print()
print(f\"Rescued count: {t['rescued'].sum()}\")
print()

# Cross-stage coverage of usable UIDs
with open('data/cache/tracking_qc_intermediates.pkl','rb') as f:
    inter = pickle.load(f)
rows = []
for _, r in t.iterrows():
    uid = int(r['global_uid'])
    if uid not in inter:
        continue
    iv = inter[uid]
    kept_set = set(str(r.get('kept_sessions','')).split(';')) if pd.notna(r.get('kept_sessions')) else set()
    kept_recs = [rec for rec in iv.sessions if rec.session_name in kept_set]
    n_learn  = sum(1 for rec in kept_recs if rec.stage == 'Learning')
    n_expert = sum(1 for rec in kept_recs if rec.stage == 'Expert')
    rows.append({'uid': uid, 'verdict': r['trimmed_verdict'], 'n_learn': n_learn, 'n_expert': n_expert})
df = pd.DataFrame(rows)
use = df[df['verdict'].isin(['trusted','review'])]
n_cross = ((use['n_learn'] >= 1) & (use['n_expert'] >= 1)).sum()
print(f'Usable UIDs with >=1 Learning AND >=1 Expert: {n_cross} / {len(use)} (was 1/20 pre-skip)')
n_cross_3 = ((use['n_learn'] >= 3) & (use['n_expert'] >= 3)).sum()
print(f'Usable UIDs with >=3 Learning AND >=3 Expert: {n_cross_3} / {len(use)} (was 0/20 pre-skip)')
"
```

Expected:
- `skipped_sessions` column present
- Verdict distribution: any direction shift OK, but `rescued` count should be > 4 (the pre-skip baseline). Suspect count may decrease.
- Cross-stage `>=1 L AND >=1 E` rises from 1 to (hopefully) several UIDs.

- [ ] **Step 3: Verify UID 942's rescue specifically**

```bash
py -c "
import pandas as pd
t = pd.read_csv('FIGURES/tracking_qc/verdicts_trimmed.csv')
row = t[t.global_uid == 942].iloc[0]
print(f'UID 942 trimmed_span: {row.trimmed_span} (was 4 pre-skip)')
print(f'UID 942 trimmed_verdict: {row.trimmed_verdict} (was trusted)')
print(f'UID 942 kept_sessions: {row.kept_sessions}')
print(f'UID 942 skipped_sessions: {row.skipped_sessions}')
print(f'UID 942 dropped_sessions: {row.dropped_sessions}')
"
```

Expected:
- `trimmed_span` rises from 4 to ~10-11 (most of the 14 sessions are kept now)
- `trimmed_verdict` stays trusted
- `skipped_sessions` lists the Unknown sessions (13082025, 18082025, 26082025)
- `dropped_sessions` is empty or contains only hard-outliers/trim-edges

- [ ] **Step 4: Spot-check UID 942 PDF — verify visual rendering still works**

Read `FIGURES/tracking_qc/per_uid_sheets/uid_0942.pdf` page 1 (use the Read tool with `pages: "1"`). Verify:
- Stage stripe now shows Learning + Expert cells filled (no hatch) and Unknown cells with hatch (matching the new visual scheme: skipped = same visual as dropped)
- Header trim annotation shows `kept N/14` with the new larger N
- ISI panel: grey traces for the 3 Unknown sessions; stage-colored traces for the kept Learning+Expert

- [ ] **Step 5: Run full pytest one more time to confirm no regressions**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -q 2>&1 | tail -3
```

Expected: all tests pass (73 total).

- [ ] **Step 6: Confirm branch + final annotation commit**

```bash
git branch --show-current   # must print: feature/trim-skip-soft-outliers
git commit --allow-empty -m "$(cat <<'EOF'
End-to-end smoke for skip-able trimming

Cohort size: 61  (unchanged)
verdicts_trimmed.csv:  <FILL FROM STEP 2 OUTPUT>
Rescued count: <FILL>  (pre-skip baseline was 4)
Cross-stage coverage (>=1 L AND >=1 E): <FILL>/<TOTAL>  (was 1/20 pre-skip)
Cross-stage coverage (>=3 L AND >=3 E): <FILL>/<TOTAL>  (was 0/20 pre-skip)
UID 942 trimmed_span: <FILL>  (was 4 pre-skip); verdict: trusted; skipped: <list>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Fill in the `<...>` placeholders from the Step 2 and Step 3 outputs.

---

## Self-review (post-write)

**Spec coverage:**

| Spec section | Plan task |
|---|---|
| §3.1 soft/hard classification in session_outlier_flags | Task 1 |
| §3.2 new longest_good_run algorithm | Task 2 (Steps 4-5) |
| §3.3 find_stable_subset return-shape changes | Task 2 (Step 6) |
| §3.4 CSV column changes | Task 3 (Steps 3-4) |
| §3.5 PDF rendering (no structural change; union at call site) | Task 3 (Step 2) |
| §6 testing (5 unit tests) | Tasks 1-2 (1 + 4 = 5 unit tests) |
| §6 end-to-end smoke | Task 4 |

All spec sections covered. No placeholders in any task. Names consistent across tasks: `is_hard_outlier`, `is_soft_outlier`, `longest_good_run` (public, new shape), `_longest_good_run_contiguous` (private fallback), `skipped_indices`, `skipped_sessions`. The four-task structure mirrors the spec's component layout.
