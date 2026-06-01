# Heatmap + PSTH-Summary Normalization and ISI Auto-Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-event baseline-subtracted diverging-cmap heatmaps + matching PSTH-summary y-axis, with a `--shared-baseline` mode toggle; add `isi_hist_corr` auto-pass (Option B) to the composite verdict.

**Architecture:** Three tightly coupled changes share the QC-sheet rendering surface area: (a) widen the PSTH extract windows so canonical baselines from `EVENT_RESPONSIVENESS_WINDOWS` fit (requires cache rebuild); (b) add baseline-scalar helpers and rewire `_draw_heatmap` + `_draw_psth_summary` to subtract the scalar (diverging cmap for heatmap, y=0 ref line for summary); (c) add a new verdict-level `apply_isi_autopass` function that promotes a UID to `trusted` when ISI shape correlation is ≥0.95 and no hard biophysical badge fails.

**Tech Stack:** Python 3.10, numpy, matplotlib (RdBu_r diverging cmap), pytest. Project's `py` launcher (Windows).

**Spec:** `docs/superpowers/specs/2026-06-01-heatmap-psth-norm-and-isi-autopass-design.md` (commit `872bc6d` on main).

**Prerequisites:**

A worktree on `main` already exists at `.claude/worktrees/trim-skip-review/` (with `data/` and `FIGURES/` junctions to the main repo). Create the feature branch IN THAT WORKTREE so all implementation work stays isolated from parallel chats:

```bash
cd .claude/worktrees/trim-skip-review
git checkout -b feature/heatmap-psth-norm-isi-autopass
git branch --show-current   # must print: feature/heatmap-psth-norm-isi-autopass
```

All `Bash` commands in this plan assume `cwd` is the worktree path. **Before EVERY commit, run `git branch --show-current` and confirm it prints `feature/heatmap-psth-norm-isi-autopass`.** User runs parallel chats; verify branch every time.

---

## Task 1: ISI auto-pass (Option B)

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py` (add constant + function)
- Modify: `tests/analysis/test_tracking_qc.py` (append 5 tests)
- Modify: `scripts/pipelines/tracking/build_qc_sheets.py` (wire into both verdict computations + 2 CSV columns)

### Step 1: Write 5 failing tests

Append to `tests/analysis/test_tracking_qc.py`:

```python
# ─── Option B: isi_hist_corr auto-pass ────────────────────────────────

def test_apply_isi_autopass_promotes_when_threshold_met():
    """ISI 0.97 + no wave/depth fail + suspect verdict → trusted."""
    assert qc.apply_isi_autopass("suspect", 0.97, "pass", "pass") == "trusted"
    assert qc.apply_isi_autopass("review",  0.97, "warn", "warn") == "trusted"
    assert qc.apply_isi_autopass("trusted", 0.97, "pass", "pass") == "trusted"


def test_apply_isi_autopass_blocks_on_wave_fail():
    """High ISI + wave FAIL → verdict unchanged (hard biophysical block)."""
    assert qc.apply_isi_autopass("suspect", 0.99, "fail", "pass") == "suspect"
    assert qc.apply_isi_autopass("review",  0.99, "fail", "warn") == "review"


def test_apply_isi_autopass_blocks_on_depth_fail():
    """High ISI + depth FAIL → verdict unchanged."""
    assert qc.apply_isi_autopass("suspect", 0.99, "pass", "fail") == "suspect"
    assert qc.apply_isi_autopass("review",  0.99, "warn", "fail") == "review"


def test_apply_isi_autopass_below_threshold_no_change():
    """ISI 0.94 (just below 0.95 threshold) → no promotion."""
    assert qc.apply_isi_autopass("suspect", 0.94, "pass", "pass") == "suspect"
    assert qc.apply_isi_autopass("review",  0.85, "pass", "pass") == "review"


def test_apply_isi_autopass_nan_no_change():
    """NaN ISI → no promotion (the threshold check fails)."""
    assert qc.apply_isi_autopass("suspect", float("nan"), "pass", "pass") == "suspect"
    assert qc.apply_isi_autopass("review",  float("nan"), "pass", "pass") == "review"
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -v -k "apply_isi_autopass" 2>&1 | tail -10
```

Expected: 5 tests fail with `AttributeError: module 'visdetect.analysis.tracking_qc' has no attribute 'apply_isi_autopass'`.

- [ ] **Step 3: Add constant + function to tracking_qc.py**

In `src/visdetect/analysis/tracking_qc.py`, near the other badge threshold constants (right after `ISI_HIST_CORR_WARN: float = 0.65`):

```python
# ISI hist-corr auto-pass: a UID whose set-wide ISI shape correlation is
# exceptionally consistent (>= 0.95, top ~25% of BG_046 cohort) is promoted
# to trusted regardless of marginal failures on other badges. Hard biophysical
# signals (wave or depth FAIL) still block — same philosophy as the
# skip-able-trim hard-outlier rule.
ISI_HIST_CORR_AUTOPASS: float = 0.95
```

Then add the function near the other verdict helpers (e.g., right after `composite_verdict`):

```python
def apply_isi_autopass(verdict: str,
                       isi_hist_corr: float,
                       wave_badge: str,
                       depth_badge: str,
                       threshold: float = ISI_HIST_CORR_AUTOPASS) -> str:
    """Promote verdict to 'trusted' when ISI shape correlation is exceptionally
    strong AND no hard biophysical badge fails.

    Hard biophysical signals (wave_badge or depth_badge == 'fail') block the
    promotion — they suggest a physically different unit at the recording
    position, which ISI alone cannot overrule.

    Parameters
    ----------
    verdict : str
        Current composite verdict ('trusted', 'review', 'suspect').
    isi_hist_corr : float
        Set-wide median pairwise Pearson r of per-session log-ISI hists.
        NaN values fail the threshold check (no promotion).
    wave_badge, depth_badge : str
        Individual badge levels ('pass', 'warn', 'fail').
    threshold : float
        Promotion threshold (default ISI_HIST_CORR_AUTOPASS).

    Returns
    -------
    str
        'trusted' if promotion conditions are met, else unchanged `verdict`.
    """
    if (np.isfinite(isi_hist_corr)
        and isi_hist_corr >= threshold
        and wave_badge != "fail"
        and depth_badge != "fail"):
        return "trusted"
    return verdict
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -q 2>&1 | tail -3
```

Expected: all tests pass (was 75 before, now 80).

- [ ] **Step 5: Add `apply_isi_autopass` to the build_qc_sheets import block**

In `scripts/pipelines/tracking/build_qc_sheets.py`, find the existing import block from `visdetect.analysis.tracking_qc`. Add `apply_isi_autopass` to the imported names. Final form:

```python
from visdetect.analysis.tracking_qc import (        # noqa: E402
    UIDIntermediate, SessionRecord,
    select_long_tracks, annotate_naive_to_expert,
    extract_session_records, load_channel_positions,
    load_isi_scores, load_um_pair_scores,
    depth_std_um, waveform_corr, fr_cv,
    isi_peak_agreement, baseline_psth_corr, baseline_isi_hist_corr,
    badge_isi, badge_depth, badge_waveform, badge_fr,
    badge_isi_peak, badge_func_resp, badge_isi_hist_corr, composite_verdict,
    apply_isi_autopass,
    estimate_session_drift, depth_std_um_corrected,
    save_cache, load_cache,
    find_stable_subset,
)
```

- [ ] **Step 6: Wire `apply_isi_autopass` into the main render-loop verdict**

In `scripts/pipelines/tracking/build_qc_sheets.py`, find the main render loop's `verdict_csv = composite_verdict([...])` line (around line 371). Add the autopass call AFTER it and capture whether it changed the verdict:

```python
        verdict_csv = composite_verdict([b_isi, b_depth, b_wave, b_fr, b_hist, b_func])
        verdict_csv_pre_autopass = verdict_csv
        verdict_csv = apply_isi_autopass(
            verdict_csv, metrics["isi_hist_corr"], b_wave, b_depth,
        )
        autopass_applied = (verdict_csv != verdict_csv_pre_autopass)
```

Then in the `rows.append({...})` dict that follows, add the new column right after `"badge_func_resp"`:

```python
        rows.append({
            # ...existing fields above...
            "badge_func_resp":  b_func,
            "autopass_applied": autopass_applied,
            # ...existing verdict/verdict_pdf/pdf_csv_disagree fields below...
        })
```

- [ ] **Step 7: Wire `apply_isi_autopass` into the trimmed-verdict precompute**

In the same file, find the trimmed-verdict block in the `uid_trim_info` precompute loop (around line 317). Find:

```python
            tv = composite_verdict([
                badge_isi(isi_scores[uid]),
                badge_depth(_depth_for_badge(tm)),
                badge_waveform(tm["wave_corr"]),
                badge_fr(tm["fr_cv"]),
                badge_isi_hist_corr(tm["isi_hist_corr"]),
                badge_func_resp(tm["func_resp_corr"]),
            ])
```

Replace with:

```python
            tv_pre_autopass = composite_verdict([
                badge_isi(isi_scores[uid]),
                badge_depth(_depth_for_badge(tm)),
                badge_waveform(tm["wave_corr"]),
                badge_fr(tm["fr_cv"]),
                badge_isi_hist_corr(tm["isi_hist_corr"]),
                badge_func_resp(tm["func_resp_corr"]),
            ])
            tv = apply_isi_autopass(
                tv_pre_autopass, tm["isi_hist_corr"],
                badge_waveform(tm["wave_corr"]),
                badge_depth(_depth_for_badge(tm)),
            )
            trimmed_autopass_applied = (tv != tv_pre_autopass)
```

Then store `trimmed_autopass_applied` in `uid_trim_info[uid]`:

```python
        uid_trim_info[uid] = {
            "stable": stable,
            "kept_indices": kept,
            "skipped_indices": skipped,
            "dropped_indices": dropped,
            "trimmed_metrics": tm,
            "trimmed_verdict": tv,
            "trimmed_autopass_applied": trimmed_autopass_applied,
        }
```

For the no-kept early-exit branch (where `tm` and `tv` are set to None/suspect), set `trimmed_autopass_applied = False`:

```python
        else:
            tm = None
            tv = "suspect"
            trimmed_autopass_applied = False
        uid_trim_info[uid] = {...}
```

- [ ] **Step 8: Add `trimmed_autopass_applied` to `verdicts_trimmed.csv` rows**

Find the `trimmed_rows.append({...})` block in the kept-sessions branch (around line 429). Add the new column at the end of the dict (before the closing `})`):

```python
        trimmed_rows.append({
            # ...existing fields above...
            "trimmed_verdict": tv,
            "trimmed_autopass_applied": trim["trimmed_autopass_applied"],
            "rescued": rescued,
        })
```

And in the no-kept (early-exit) row, add `"trimmed_autopass_applied": False`:

```python
        if not kept:
            trimmed_rows.append({
                "global_uid": uid, "original_span": iv.span,
                "trimmed_span": 0,
                "dropped_sessions": ";".join(r.session_name for r in iv.sessions),
                "skipped_sessions": "",
                "kept_sessions": "", "trimmed_verdict": "suspect",
                "trimmed_autopass_applied": False,
                "rescued": False,
            })
            continue
```

- [ ] **Step 9: Smoke check that the module still imports + `--help` runs**

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --help 2>&1 | head -8
```

Expected: argparse help block, no exception.

- [ ] **Step 10: Confirm branch + commit**

```bash
git branch --show-current   # must print: feature/heatmap-psth-norm-isi-autopass
git add src/visdetect/analysis/tracking_qc.py tests/analysis/test_tracking_qc.py scripts/pipelines/tracking/build_qc_sheets.py
git commit -m "Add ISI auto-pass (Option B): promote to trusted when isi_hist_corr >= 0.95 and no wave/depth fail"
```

---

## Task 2: Widen PSTH extract windows + cache rebuild

**Files:**
- Modify: `src/visdetect/analysis/tracking_qc.py` (`PSTH_SPECS` dict)

No new tests; cache rebuild is the verification.

- [ ] **Step 1: Update `PSTH_SPECS` window values**

In `src/visdetect/analysis/tracking_qc.py`, find the `PSTH_SPECS` dict (around line 682). Update two entries:

```python
PSTH_SPECS: Dict[str, Dict[str, Any]] = {
    "baseline_on":        {"event": "Baseline_ON", "outcomes": None,           "sizes": None,       "window": (-2.0, 1.5)},
    "change_on_big_hit":  {"event": "Change_ON",   "outcomes": {"hit"},        "sizes": BIG_POOL,   "window": (-0.5, 0.5)},
    "change_on_big_miss": {"event": "Change_ON",   "outcomes": {"miss"},       "sizes": BIG_POOL,   "window": (-0.5, 0.5)},
    "change_on_sm_hit":   {"event": "Change_ON",   "outcomes": {"hit"},        "sizes": SMALL_POOL, "window": (-0.5, 0.5)},
    "change_on_sm_miss":  {"event": "Change_ON",   "outcomes": {"miss"},       "sizes": SMALL_POOL, "window": (-0.5, 0.5)},
    "hit_lick":           {"event": "Hit",         "outcomes": {"hit"},        "sizes": None,       "window": (-2.0, 1.0)},
}
```

Changes: `baseline_on` window `(-0.5, 1.5)` → `(-2.0, 1.5)`; `hit_lick` window `(-1.0, 1.0)` → `(-2.0, 1.0)`. Other entries unchanged.

- [ ] **Step 2: Rebuild cache (required — PSTHs are baked in with new windows)**

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --rebuild-cache 2>&1 | tail -5
```

Expected: takes ~5-10 min. Output ends with `Wrote .../verdicts.csv` and `Wrote .../verdicts_trimmed.csv (N rescued)`.

- [ ] **Step 3: Verify the cache PSTHs now span the new windows**

```bash
py -c "
import pickle
import numpy as np
with open('data/cache/tracking_qc_intermediates.pkl','rb') as f:
    inter = pickle.load(f)
uid = inter[942]
rec = uid.sessions[0]
for key in ['baseline_on', 'hit_lick', 'change_on_big_hit']:
    psth, c, n = rec.psths.get(key, (None, None, 0))
    if c is None:
        print(f'{key}: no data')
        continue
    print(f'{key}: time range ({c[0]:.2f}, {c[-1]:.2f}), n_bins {len(c)}, n_trials {n}')
"
```

Expected:
- `baseline_on`: time range `(-2.00, 1.48)` (or similar — bin edges may differ by ~bin_size); `n_bins` ≈ 140 (was ~80 before widening)
- `hit_lick`: time range `(-2.00, 0.98)`; `n_bins` ≈ 120 (was ~80)
- `change_on_big_hit`: unchanged, time range `(-0.50, 0.48)`; `n_bins` ≈ 40

- [ ] **Step 4: Confirm branch + commit**

```bash
git branch --show-current   # must print: feature/heatmap-psth-norm-isi-autopass
git add src/visdetect/analysis/tracking_qc.py
git commit -m "Widen baseline_on and hit_lick PSTH extract windows to (-2.0, ...) so canonical baselines from EVENT_RESPONSIVENESS_WINDOWS fit"
```

---

## Task 3: Baseline-subtracted heatmaps + PSTH summaries + --shared-baseline CLI flag

**Files:**
- Modify: `scripts/pipelines/tracking/qc_sheet_figures.py` (helpers + `_draw_heatmap` + `_draw_psth_summary` + `render_page2` + `write_uid_pdf`)
- Modify: `scripts/pipelines/tracking/build_qc_sheets.py` (CLI flag + pass-through to `write_uid_pdf`)

No new unit tests — visual changes verified by Task 4 smoke.

### Step 1: Add `_PSTH_KEY_TO_EVENT` dict and import `EVENT_RESPONSIVENESS_WINDOWS`

In `scripts/pipelines/tracking/qc_sheet_figures.py`, find the existing imports near the top. Add the constants import (after the existing `from visdetect.suite.config import ...` line):

```python
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS  # noqa: E402
```

Then, near the other module-level constants (after `STAGE_COLORS_LOCAL`), add the key-to-event mapping:

```python
# Maps the PSTH series keys used in this module to the canonical event names
# in EVENT_RESPONSIVENESS_WINDOWS. Used by the baseline-scalar helpers below
# to look up the per-event baseline window.
_PSTH_KEY_TO_EVENT: Dict[str, str] = {
    "baseline_on":        "Baseline_ON",
    "change_on_big_hit":  "Change_ON",
    "change_on_big_miss": "Change_ON",
    "change_on_sm_hit":   "Change_ON",
    "change_on_sm_miss":  "Change_ON",
    "hit_lick":           "Hit",
    "fa_lick":            "FA",
}
```

### Step 2: Add `_per_event_baseline_scalar` and `_shared_baseline_scalar` helpers

In `scripts/pipelines/tracking/qc_sheet_figures.py`, right after `_psth_matrix` (around line 289), add the two helpers:

```python
def _per_event_baseline_scalar(mat: np.ndarray,
                                centers: np.ndarray,
                                psth_key: str) -> float:
    """Per-UID pooled baseline scalar for one heatmap.

    Looks up the canonical baseline window for this PSTH key from
    EVENT_RESPONSIVENESS_WINDOWS, selects the matching bins in `centers`,
    and returns the mean rate pooled across all sessions and those bins.

    Returns NaN if the baseline window does not intersect the matrix.
    Caller is responsible for falling back when this happens.
    """
    event = _PSTH_KEY_TO_EVENT.get(psth_key, "Baseline_ON")
    (lo, hi), _ = EVENT_RESPONSIVENESS_WINDOWS[event]
    mask = (centers >= lo) & (centers < hi)
    if not mask.any():
        return float("nan")
    return float(mat[:, mask].mean())


def _shared_baseline_scalar(uid: UIDIntermediate) -> float:
    """One-baseline-for-all-heatmaps scalar for this UID, computed from the
    Baseline_ON PSTHs using the canonical baseline window (-1.75, -1.25).

    Pools across all sessions of the UID. Returns NaN if no Baseline_ON
    PSTH data is available (caller should fall back to per-event baselines).
    """
    rows: List[np.ndarray] = []
    centers: Optional[np.ndarray] = None
    for r in uid.sessions:
        psth, c, _n = r.psths.get("baseline_on", (None, None, 0))
        if psth is not None:
            rows.append(psth)
            if centers is None:
                centers = c
    if not rows or centers is None:
        return float("nan")
    mat = np.stack(rows)
    (lo, hi), _ = EVENT_RESPONSIVENESS_WINDOWS["Baseline_ON"]
    mask = (centers >= lo) & (centers < hi)
    if not mask.any():
        return float("nan")
    return float(mat[:, mask].mean())
```

### Step 3: Modify `_draw_heatmap` to subtract baseline + diverging cmap

In `scripts/pipelines/tracking/qc_sheet_figures.py`, find `_draw_heatmap` (around line 292). Change its signature to accept a `baseline_scalar` kwarg, and replace the `imshow` call. Final form:

```python
def _draw_heatmap(ax, uid: UIDIntermediate, key: str, title: str,
                  *,
                  dropped_indices: Optional[List[int]] = None,
                  baseline_scalar: float = 0.0) -> None:
    """Render the chronological PSTH heatmap into `ax` with baseline-subtracted
    diverging-cmap rendering. `baseline_scalar` is subtracted from every value
    before rendering; 0 leaves the matrix unchanged.

    If dropped_indices is supplied, draw a thin red rectangle just to the
    LEFT of each dropped row to flag sessions excluded by find_stable_subset.
    """
    data = _psth_matrix(uid, key)
    if data is None:
        ax.text(0.5, 0.5, f"no trials for {key}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        return

    mat, centers, _stages, _ = data
    # Baseline-subtract: shifts the matrix so the chosen baseline window
    # corresponds to 0 (white on the diverging cmap below).
    mat_sub = mat - float(baseline_scalar)
    # Symmetric vmax from absolute-value 95th percentile, with a floor so
    # near-zero-modulation cells don't get a degenerate scale.
    vmax = max(float(np.percentile(np.abs(mat_sub), 95)), 0.5)
    ax.imshow(mat_sub, aspect="auto", origin="upper", cmap="RdBu_r",
              extent=[centers[0], centers[-1], mat_sub.shape[0], 0],
              vmin=-vmax, vmax=vmax)
    ax.axvline(0, color="0.3", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("session #")
    # Inline ±vmax annotation so reviewers can calibrate the color scale.
    ax.text(0.98, 0.02, f"±{vmax:.1f} Hz",
            transform=ax.transAxes, fontsize=7, color="0.3",
            ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1))

    if dropped_indices:
        # Build uid_idx -> heatmap_row_idx mapping using the same filter as
        # _psth_matrix (only rows where psths[key] is non-None).
        heatmap_row = 0
        uid_to_row = {}
        for uid_idx, rec in enumerate(uid.sessions):
            psth, _c, _n = rec.psths.get(key, (None, None, 0))
            if psth is None:
                continue
            uid_to_row[uid_idx] = heatmap_row
            heatmap_row += 1
        x0, x1 = centers[0], centers[-1]
        pad = 0.03 * (x1 - x0)
        drew_any_marker = False
        for uid_idx in dropped_indices:
            row = uid_to_row.get(uid_idx)
            if row is None:
                continue
            ax.add_patch(Rectangle((x0 - pad, row), pad, 1.0,
                                    facecolor="red", edgecolor="none",
                                    clip_on=False, zorder=5))
            drew_any_marker = True
        if drew_any_marker:
            # Extend the visible x-range slightly so the red stripe is not clipped
            ax.set_xlim(x0 - pad, x1)
            ax.text(0.02, 0.97, "red bar = dropped row",
                    transform=ax.transAxes, fontsize=7, color="black",
                    ha="left", va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))
```

Three small visual tweaks alongside the cmap change:
- `axvline` color changed from `white` (invisible on white) to `"0.3"` (medium grey, visible on RdBu_r)
- `red bar = dropped row` legend changed from white-on-dark to black-on-white-bbox (was white-on-magma; needs adjustment for RdBu_r)
- `±vmax Hz` annotation added in bottom-right of each heatmap

### Step 4: Modify `_draw_psth_summary` to subtract baseline + y=0 line

In `scripts/pipelines/tracking/qc_sheet_figures.py`, find `_draw_psth_summary` (around line 351). Change its signature and body:

```python
def _draw_psth_summary(ax, uid: UIDIntermediate, key: str,
                        miss_keys: Optional[List[str]] = None,
                        *, baseline_scalar: float = 0.0) -> None:
    """Render L vs E stage-mean PSTH traces into `ax` as a normal (white) plot,
    baseline-subtracted to match the heatmap above. `baseline_scalar` is
    subtracted from each stage-mean trace before plotting; 0 leaves traces
    unchanged.

    miss_keys (optional): list of keys whose stage-mean traces to overlay as
    dashed lines for hit/miss comparison (e.g. ["change_on_big_miss"]).
    """
    data = _psth_matrix(uid, key)
    if data is None:
        ax.text(0.5, 0.5, f"no trials for {key}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
        ax.set_axis_off()
        return

    mat, centers, stages, _ = data
    has_label = False
    for st in STAGE_ORDER:
        mask = np.array([s == st for s in stages])
        if mask.sum() == 0:
            continue
        label_solid = f"{st} hit" if miss_keys else st
        # Stage-mean trace, baseline-subtracted to match the heatmap above
        stage_mean = mat[mask].mean(axis=0) - float(baseline_scalar)
        ax.plot(centers, stage_mean, color=STAGE_COLORS_LOCAL[st],
                linewidth=1.2, label=label_solid)
        has_label = True

    if miss_keys:
        for mk in miss_keys:
            mdata = _psth_matrix(uid, mk)
            if mdata is None:
                continue
            mmat, mcenters, mstages, _ = mdata
            for st in STAGE_ORDER:
                mask = np.array([s == st for s in mstages])
                if mask.sum() == 0:
                    continue
                stage_mean = mmat[mask].mean(axis=0) - float(baseline_scalar)
                ax.plot(mcenters, stage_mean,
                        color=STAGE_COLORS_LOCAL[st], linewidth=1.0,
                        linestyle="--", alpha=0.7,
                        label=f"{st} miss")
                has_label = True

    ax.axvline(0, color="0.5", linewidth=0.7)
    # NEW: y=0 reference line (the baseline subtraction's zero point)
    ax.axhline(0, color="0.7", linewidth=0.5, alpha=0.8, zorder=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz (rel. baseline)")
    ax.tick_params(labelsize=8)
    if has_label:
        ax.legend(loc="upper right", fontsize=6 if miss_keys else 7, frameon=False)
```

### Step 5: Modify `render_page2` to compute baseline scalar per row + thread through

In `scripts/pipelines/tracking/qc_sheet_figures.py`, find `render_page2` (around line 400). Add a `shared_baseline` kwarg, compute the scalar per heatmap row, and pass it to both `_draw_heatmap` and `_draw_psth_summary`.

Add a helper computation function at the TOP of `render_page2`'s body (right after the docstring):

```python
def render_page2(uid: UIDIntermediate, isi_score: float, depth_std: float,
                 wave_corr: float, fr_cv_val: float,
                 *,
                 dropped_indices: Optional[List[int]] = None,
                 n_kept: Optional[int] = None,
                 trimmed_verdict: Optional[str] = None,
                 shared_baseline: bool = False) -> plt.Figure:
    """Render page 2 (physical) — returns the Figure.

    Layout (5 rows × 2 cols master): each row pairs a heatmap (left) with its
    L vs E PSTH-summary panel (right) at matched row heights.

    shared_baseline : if True, use ONE baseline scalar derived from Baseline_ON
        applied to all heatmaps + summaries. Default (False) uses per-event
        baseline windows from EVENT_RESPONSIVENESS_WINDOWS.
    """
    fig = plt.figure(figsize=(8.5, 11.0))
    gs = gridspec.GridSpec(
        nrows=5, ncols=2,
        height_ratios=[1.25, 1.75, 1.75, 1.75, 1.75],
        width_ratios=[1, 1],
        hspace=0.70, wspace=0.30,
        top=0.96, bottom=0.05, left=0.09, right=0.96,
        figure=fig,
    )

    # ── Row 0: Header (spans both columns)
    ax_hdr = fig.add_subplot(gs[0, :])
    draw_header(ax_hdr, uid, isi_score, depth_std, wave_corr, fr_cv_val,
                dropped_indices=dropped_indices,
                n_kept=n_kept, trimmed_verdict=trimmed_verdict)

    # Compute baseline scalar for each row. The same scalar is passed to BOTH
    # the heatmap and its companion PSTH-summary so they show the same zero.
    # Shared mode: one Baseline_ON-derived scalar applied to every row.
    # Per-event mode (default): each row gets its own canonical baseline.
    if shared_baseline:
        shared_scalar = _shared_baseline_scalar(uid)
        if not np.isfinite(shared_scalar):
            shared_scalar = None  # signal fall back to per-event

    def _scalar_for_key(key: str) -> float:
        """Resolve baseline scalar with fallback chain: shared (if requested
        and finite) -> per-event (computed from this heatmap's matrix) -> 0."""
        if shared_baseline and shared_scalar is not None:
            return shared_scalar
        data = _psth_matrix(uid, key)
        if data is None:
            return 0.0
        mat, centers, _stages, _ = data
        scalar = _per_event_baseline_scalar(mat, centers, key)
        return scalar if np.isfinite(scalar) else 0.0

    # ── Row 1: Baseline_ON
    bs_baseline = _scalar_for_key("baseline_on")
    _draw_heatmap(
        fig.add_subplot(gs[1, 0]), uid, "baseline_on",
        title="PSTH · Baseline_ON · all outcomes pooled [TODO: split by outcome in v2]",
        dropped_indices=dropped_indices,
        baseline_scalar=bs_baseline,
    )
    _draw_psth_summary(
        fig.add_subplot(gs[1, 1]), uid, "baseline_on",
        baseline_scalar=bs_baseline,
    )

    # ── Row 2: Change_ON Big-Hit (+ Big-Miss dashed overlay in summary)
    bh_baseline = _scalar_for_key("change_on_big_hit")
    _draw_heatmap(
        fig.add_subplot(gs[2, 0]), uid, "change_on_big_hit",
        title="Change_ON · Big-Hit (2.0× + 4.0×)",
        dropped_indices=dropped_indices,
        baseline_scalar=bh_baseline,
    )
    _draw_psth_summary(
        fig.add_subplot(gs[2, 1]), uid, "change_on_big_hit",
        miss_keys=["change_on_big_miss"],
        baseline_scalar=bh_baseline,
    )

    # ── Row 3: Change_ON Small-Hit (+ Small-Miss dashed overlay in summary)
    sh_baseline = _scalar_for_key("change_on_sm_hit")
    _draw_heatmap(
        fig.add_subplot(gs[3, 0]), uid, "change_on_sm_hit",
        title="Change_ON · Small-Hit (1.25× + 1.35×)",
        dropped_indices=dropped_indices,
        baseline_scalar=sh_baseline,
    )
    _draw_psth_summary(
        fig.add_subplot(gs[3, 1]), uid, "change_on_sm_hit",
        miss_keys=["change_on_sm_miss"],
        baseline_scalar=sh_baseline,
    )

    # ── Row 4: Hit-lick
    hl_baseline = _scalar_for_key("hit_lick")
    _draw_heatmap(
        fig.add_subplot(gs[4, 0]), uid, "hit_lick",
        title="PSTH · Hit lick",
        dropped_indices=dropped_indices,
        baseline_scalar=hl_baseline,
    )
    _draw_psth_summary(
        fig.add_subplot(gs[4, 1]), uid, "hit_lick",
        baseline_scalar=hl_baseline,
    )

    return fig
```

(`shared_scalar` is declared in the `if shared_baseline:` block; need to ensure it's defined in scope. Either set it to `None` outside the if too, or use a sentinel — Python's late binding inside the nested function will see whatever was set in the enclosing scope. Make sure the variable is always defined before `_scalar_for_key` is called.)

### Step 6: Modify `write_uid_pdf` to accept + thread `shared_baseline`

In `scripts/pipelines/tracking/qc_sheet_figures.py`, find `write_uid_pdf` (around line 481). Add the kwarg + pass it to `render_page2`:

```python
def write_uid_pdf(out_path: Path, uid: UIDIntermediate,
                  um_pair_scores: Optional[np.ndarray],
                  isi_score: float, depth_std: float,
                  wave_corr: float, fr_cv_val: float,
                  *,
                  dropped_indices: Optional[List[int]] = None,
                  n_kept: Optional[int] = None,
                  trimmed_verdict: Optional[str] = None,
                  shared_baseline: bool = False) -> str:
    """Write the 2-page PDF; return the composite verdict string."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        f1 = render_page1(uid, um_pair_scores, isi_score, depth_std, wave_corr, fr_cv_val,
                          dropped_indices=dropped_indices,
                          n_kept=n_kept, trimmed_verdict=trimmed_verdict)
        pdf.savefig(f1); plt.close(f1)
        f2 = render_page2(uid, isi_score, depth_std, wave_corr, fr_cv_val,
                          dropped_indices=dropped_indices,
                          n_kept=n_kept, trimmed_verdict=trimmed_verdict,
                          shared_baseline=shared_baseline)
        pdf.savefig(f2); plt.close(f2)
    # Re-run the composite using the same inputs (cheap; keeps the API tidy)
    from visdetect.analysis.tracking_qc import (
        badge_isi, badge_depth, badge_waveform, badge_fr, composite_verdict,
    )
    return composite_verdict([
        badge_isi(isi_score), badge_depth(depth_std),
        badge_waveform(wave_corr), badge_fr(fr_cv_val),
    ])
```

### Step 7: Add `--shared-baseline` CLI flag to build_qc_sheets.py

In `scripts/pipelines/tracking/build_qc_sheets.py`, find the existing `parser.add_argument` calls (around line 199). Add the new flag:

```python
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--uids", type=int, nargs="*", default=None,
                        help="render only these UIDs")
    parser.add_argument("--max-uids", type=int, default=None,
                        help="cap on number of UIDs rendered")
    parser.add_argument(
        "--shared-baseline", action="store_true",
        help="Use one Baseline_ON-derived baseline scalar for ALL heatmaps "
             "in each UID's page 2 (cross-event comparison mode). Default: "
             "per-event baseline from EVENT_RESPONSIVENESS_WINDOWS.",
    )
```

### Step 8: Pass `--shared-baseline` to `write_uid_pdf`

Find the `write_uid_pdf(...)` call in the main render loop (around line 348). Add the `shared_baseline` argument:

```python
        verdict_pdf = write_uid_pdf(
            out_path, iv, pair_scores.get(uid),
            isi_score=isi,
            depth_std=metrics["depth_std_um"],
            wave_corr=metrics["wave_corr"],
            fr_cv_val=metrics["fr_cv"],
            dropped_indices=visually_dropped,
            n_kept=len(trim["kept_indices"]),
            trimmed_verdict=str(trim["trimmed_verdict"]),
            shared_baseline=args.shared_baseline,
        )
```

### Step 9: Smoke check that the module still imports + `--help` runs

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --help 2>&1 | head -10
```

Expected: argparse help block, including the new `--shared-baseline` flag, no exception.

### Step 10: Smoke render UID 942 (default mode + --shared-baseline mode)

Cache from Task 2 is already populated. Render UID 942 twice — once default, once with `--shared-baseline`. Compare to verify:

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --uids 942 2>&1 | tail -3
```

Expected: `uid 942: csv=trusted pdf=trusted` (or similar), no exception.

Inspect `FIGURES/tracking_qc/per_uid_sheets/uid_0942.pdf` page 2 (use the Read tool with `pages: "2"`). Verify:
- Heatmaps now use RdBu_r diverging cmap with baseline near white
- Y-axis labels: "Hz (rel. baseline)" on summary panels
- Y=0 reference line visible on summary panels
- "±N.N Hz" annotation in bottom-right of each heatmap
- "red bar = dropped row" legend visible (on UIDs with dropped sessions); now black-on-white

Then re-render with `--shared-baseline`:

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --uids 942 --shared-baseline 2>&1 | tail -3
```

Re-inspect the same PDF. The Change_ON and Hit-lick heatmaps should look subtly different (baseline is now Baseline_ON-derived ITI rate, not pre-change). Baseline_ON heatmap should look ~identical between the two modes (both use Baseline_ON's baseline window).

### Step 11: Confirm branch + commit

```bash
git branch --show-current   # must print: feature/heatmap-psth-norm-isi-autopass
git add scripts/pipelines/tracking/qc_sheet_figures.py scripts/pipelines/tracking/build_qc_sheets.py
git commit -m "Heatmap + PSTH-summary baseline subtraction with diverging cmap; --shared-baseline CLI flag"
```

---

## Task 4: End-to-end smoke + Option B promotion verification

**Files:** none modified — verification + final annotation commit.

- [ ] **Step 1: Verify `autopass_applied` columns are present in both CSVs**

```bash
py -c "
import pandas as pd
d = pd.read_csv('FIGURES/tracking_qc/verdicts.csv')
t = pd.read_csv('FIGURES/tracking_qc/verdicts_trimmed.csv')
print('verdicts.csv has autopass_applied:', 'autopass_applied' in d.columns)
print('verdicts_trimmed.csv has trimmed_autopass_applied:', 'trimmed_autopass_applied' in t.columns)
print()
print('autopass_applied=True count (full):', d['autopass_applied'].sum())
print('trimmed_autopass_applied=True count:', t['trimmed_autopass_applied'].sum())
"
```

Expected:
- Both columns present
- At least 1 UID in either CSV with autopass=True (gold-standard UIDs have isi_hist_corr ≥ 0.95 but might already be trusted by composite, in which case autopass is False)

- [ ] **Step 2: Identify a real Option B promotion example**

Find a UID where Option B promoted the trimmed verdict from suspect/review to trusted:

```bash
py -c "
import pandas as pd
t = pd.read_csv('FIGURES/tracking_qc/verdicts_trimmed.csv')
promoted = t[t.trimmed_autopass_applied == True]
print(f'Total Option B promotions: {len(promoted)}')
print()
print('First few promoted UIDs:')
cols = ['global_uid','trimmed_isi_hist_corr','trimmed_verdict','rescued']
print(promoted[cols].head(10).to_string(index=False))
"
```

Expected: at least 1 promoted UID. All have `trimmed_isi_hist_corr ≥ 0.95` and `trimmed_verdict == 'trusted'`.

- [ ] **Step 3: Inspect a non-promotable case for contrast**

Find a UID with high ISI but a wave or depth fail that blocked Option B:

```bash
py -c "
import pandas as pd
t = pd.read_csv('FIGURES/tracking_qc/verdicts_trimmed.csv')
high_isi = t[t.trimmed_isi_hist_corr >= 0.95]
blocked = high_isi[high_isi.trimmed_autopass_applied == False]
blocked = blocked[blocked.trimmed_verdict != 'trusted']
print(f'High-ISI UIDs with autopass blocked + still not trusted: {len(blocked)}')
print()
cols = ['global_uid','trimmed_isi_hist_corr','trimmed_wave_corr','trimmed_verdict']
print(blocked[cols].head(5).to_string(index=False))
"
```

Expected: if any present, their `trimmed_wave_corr` should be low (a wave fail). Confirms the Option B block-on-fail logic.

- [ ] **Step 4: Render UID 942 + 779 + one Option B-promoted UID; visual check on page 2**

Pick the first Option B-promoted UID from Step 2 (call it `<UID_BPROMOTED>`):

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --uids 942 779 <UID_BPROMOTED> 2>&1 | tail -5
```

Inspect each PDF page 2. Verify:
- All three have RdBu_r cmap, "Hz (rel. baseline)" y-label, y=0 line, ±vmax annotation
- UID 942: gold standard, baseline near white throughout
- UID 779: heatmaps may look more diverse (matching failure)
- `<UID_BPROMOTED>`: now trusted; visual identity should match a typical trusted UID

- [ ] **Step 5: Verify cross-mode rendering (default vs --shared-baseline)**

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --uids 942 --shared-baseline 2>&1 | tail -3
```

Re-inspect `uid_0942.pdf` page 2. Confirm:
- The Change_ON heatmaps look subtly different from the default-mode version (baseline is now the cell's true ITI rate, not pre-change rate)
- The Baseline_ON heatmap is unchanged (both modes use Baseline_ON's baseline window)
- Re-render WITHOUT `--shared-baseline` to restore default state for any reviewer:

```bash
py scripts/pipelines/tracking/build_qc_sheets.py --uids 942 2>&1 | tail -3
```

- [ ] **Step 6: Re-run pytest to confirm no regressions**

```bash
py -m pytest tests/analysis/test_tracking_qc.py -q 2>&1 | tail -3
```

Expected: 80 passed (75 prior + 5 new from Task 1).

- [ ] **Step 7: Confirm branch + final annotation commit**

Fill the `<...>` placeholders below from the actual outputs of Steps 1, 2, 3:

```bash
git branch --show-current   # must print: feature/heatmap-psth-norm-isi-autopass
git commit --allow-empty -m "$(cat <<'EOF'
End-to-end smoke for heatmap+PSTH normalization and ISI auto-pass

Cohort size: 61 (unchanged)
verdicts.csv autopass=True count: <FILL>
verdicts_trimmed.csv trimmed_autopass=True count: <FILL>
Option B promotions: <FILL> UIDs (real example: UID <UID_BPROMOTED>
  trimmed_isi_hist_corr=<R> → trimmed_verdict=trusted)
Wave/depth-blocked high-ISI UIDs: <FILL>
PDF visual checks: RdBu_r cmap, baseline-relative y-axis, y=0 ref line,
  ±vmax Hz annotation all present on UID 942/779/<UID_BPROMOTED>
--shared-baseline mode renders cleanly; Baseline_ON heatmap identical to
  default mode (sanity check on shared-from-baseline-on logic)
All 80 unit tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review (post-write)

**Spec coverage:**

| Spec section | Plan task |
|---|---|
| §3.1 Widen PSTH extract windows | Task 2 |
| §3.2 _per_event_baseline_scalar helper | Task 3 (Step 2) |
| §3.3 _shared_baseline_scalar helper | Task 3 (Step 2) |
| §3.4 Render-time baseline mode toggle (CLI flag + orchestration) | Task 3 (Steps 5, 7, 8) |
| §3.5 Diverging cmap + symmetric vmax + ±vmax annotation | Task 3 (Step 3) |
| §3.6 PSTH-summary baseline subtraction + y=0 line | Task 3 (Step 4) |
| §3.7 ISI auto-pass + CSV columns | Task 1 |
| §6 Unit tests (5 for autopass) | Task 1 (Step 1) |
| §6 End-to-end smoke | Task 4 |

All spec sections covered. No placeholders. Names consistent across tasks: `ISI_HIST_CORR_AUTOPASS`, `apply_isi_autopass`, `autopass_applied`, `trimmed_autopass_applied`, `_PSTH_KEY_TO_EVENT`, `_per_event_baseline_scalar`, `_shared_baseline_scalar`, `baseline_scalar` (kwarg), `shared_baseline` (kwarg + CLI flag).
