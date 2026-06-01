# Heatmap + PSTH-Summary Normalization and ISI Auto-Pass

**Date:** 2026-06-01
**Owner:** Ben (UCL)
**Status:** Design

## 1. Purpose

Bundle three improvements that share the same surface area (`qc_sheet_figures.py` rendering + composite-verdict logic in `tracking_qc.py`/`build_qc_sheets.py`):

1. Replace the unnormalized heatmap rendering (single shared `vmax`, magma cmap, raw Hz) with per-event baseline-subtracted rendering plus a diverging cmap. Fixes the long-standing "saturated/dark heatmaps" usability complaint from the May 2026 review (Q3). Provides a togglable mode for using a single shared baseline across all heatmaps in a UID.
2. Apply the same baseline subtraction to the PSTH-summary line plots so each row's heatmap and summary panel share an identical quantitative interpretation.
3. Add an `isi_hist_corr` auto-pass rule to the composite verdict (Option B from the prior brainstorm): if a UID's set-wide ISI shape correlation is exceptionally strong (≥0.95) AND no hard biophysical signal fails (wave/depth), promote the verdict to `trusted` regardless of other warnings.

## 2. Background

### 2.1 — Heatmap usability

`_draw_heatmap` in `scripts/pipelines/tracking/qc_sheet_figures.py` currently calls `imshow(mat, vmin=0, vmax=max(vmax, 1e-6), cmap="magma")` where `vmax` is the per-matrix maximum. Two practical consequences:

- High-FR sessions set the cap, washing out lower-FR rows.
- Low-FR baselines saturate dark on the magma scale, making subtle modulation invisible.

The visual signal needed for QC review is *modulation amplitude* (does the cell respond to the event?) and *cross-session offset* (is the baseline rate consistent?). Current rendering makes both hard to read.

### 2.2 — PSTH-summary scale mismatch

The right-column line plots (`_draw_psth_summary`) plot stage-mean traces in raw Hz. High-FR cells make their per-stage traces look offset rather than co-modulated. For the QC use case — "do Learning and Expert show similar modulation around the event?" — the absolute offset is a distractor.

### 2.3 — Canonical event baseline windows vs current PSTH extract windows

`EVENT_RESPONSIVENESS_WINDOWS` in `constants.py` defines the *canonical* baseline window for each event:

| Event | Canonical baseline (s) |
|---|---|
| `Baseline_ON` | (−1.75, −1.25) |
| `Change_ON`   | (−0.4, −0.05)   |
| `Hit`         | (−1.75, −1.25)  |
| `FA`          | (−1.75, −1.25)  |

The lick-aligned events deliberately use an early-trial window well before any motor ramp — critical because pre-lick activity is already ramping in many cells.

However, the current PSTH extract windows in `tracking_qc.py` PSTH_SPECS are narrower:

| PSTH key | Current extract window | Canonical baseline | Fits in matrix? |
|---|---|---|---|
| `baseline_on` | (−0.5, 1.5) | (−1.75, −1.25) | ❌ |
| `change_on_*` | (−0.5, 0.5) | (−0.4, −0.05) | ✓ |
| `hit_lick` | (−1.0, 1.0) | (−1.75, −1.25) | ❌ |

To use canonical baselines for `baseline_on` and `hit_lick`, we must widen the extract windows and rebuild the cache.

### 2.4 — Per-event vs shared-baseline interpretation

For a UID's page 2, each heatmap is a separate panel — Change_ON heatmap is not compared on the same axis to the Baseline_ON heatmap. So CLAUDE.md's "shared baseline across all conditions being compared" rule applies *within each heatmap* (all sessions of a UID for that event), not necessarily *across heatmaps* of the UID.

Per-event baseline (the default): each heatmap uses its own canonical baseline window. Interpretation: "how does this cell modulate from its pre-event resting state?" Different heatmaps have different reference baselines (Change_ON vs ITI).

Shared baseline (optional toggle): one scalar derived from Baseline_ON's pre-event window applied to all heatmaps. Interpretation: "how does each event's activity compare to the cell's true resting ITI rate?" More directly cross-condition-comparable; equivalent for cells with similar baseline-TF and ITI rates.

Both modes are valid for different purposes. Per-event is the natural default; shared is a flag for QC reviewers wanting cross-event comparison.

### 2.5 — Option B context

The post-merge cohort has 13 trusted + 7 review + 41 suspect UIDs (after skip-able trimming). Several borderline UIDs have very high `isi_hist_corr` (≥0.95) but are marked review/suspect due to a single non-fatal failure on another badge. ISI fingerprint is the strongest biophysical identity signal we measure; when it's exceptionally consistent across sessions, marginal failures on other badges should not override that signal.

Cohort percentiles for `isi_hist_corr` (5/25/50/75/95): 0.51 / 0.79 / 0.86 / 0.94 / 0.98. Gold-standard UIDs (942, 1207, 1712): 0.97–0.99. The 0.95 threshold isolates the top ~25% of the cohort.

## 3. Components

### 3.1 — Widen PSTH extract windows

**Location:** `src/visdetect/analysis/tracking_qc.py`, `PSTH_SPECS` dict (around line 682).

Update two extract windows so the canonical baselines from `EVENT_RESPONSIVENESS_WINDOWS` fit inside the matrix:

```python
PSTH_SPECS: Dict[str, Dict[str, Any]] = {
    "baseline_on":        {... "window": (-2.0, 1.5)},   # was (-0.5, 1.5)
    "change_on_big_hit":  {... "window": (-0.5, 0.5)},   # unchanged (already fits)
    "change_on_big_miss": {... "window": (-0.5, 0.5)},   # unchanged
    "change_on_sm_hit":   {... "window": (-0.5, 0.5)},   # unchanged
    "change_on_sm_miss":  {... "window": (-0.5, 0.5)},   # unchanged
    "hit_lick":           {... "window": (-2.0, 1.0)},   # was (-1.0, 1.0)
}
```

`baseline_on` becomes (-2.0, 1.5) and `hit_lick` becomes (-2.0, 1.0) so each can use its canonical (-1.75, -1.25) baseline. Other keys unchanged.

**Consequence:** the cache (`data/cache/tracking_qc_intermediates.pkl`) must be rebuilt because PSTHs are stored per-session with these windows baked in. Cache rebuild takes ~5–10 min and is included in the implementation plan's smoke task.

### 3.2 — Per-event baseline scalar helper

**Location:** `scripts/pipelines/tracking/qc_sheet_figures.py` (new module-level helpers).

Add two helpers:

```python
_PSTH_KEY_TO_EVENT: Dict[str, str] = {
    "baseline_on":        "Baseline_ON",
    "change_on_big_hit":  "Change_ON",
    "change_on_big_miss": "Change_ON",
    "change_on_sm_hit":   "Change_ON",
    "change_on_sm_miss":  "Change_ON",
    "hit_lick":           "Hit",
    "fa_lick":            "FA",
}

def _per_event_baseline_scalar(mat: np.ndarray,
                                centers: np.ndarray,
                                psth_key: str) -> float:
    """Per-UID pooled baseline scalar for one heatmap.

    Looks up the canonical baseline window for this PSTH key from
    EVENT_RESPONSIVENESS_WINDOWS, selects the matching bins in `centers`,
    and returns the mean rate pooled across all sessions and those bins.

    Returns NaN if the baseline window does not intersect the matrix (safe
    fallback — caller should not subtract).
    """
    event = _PSTH_KEY_TO_EVENT.get(psth_key, "Baseline_ON")
    (lo, hi), _ = EVENT_RESPONSIVENESS_WINDOWS[event]
    mask = (centers >= lo) & (centers < hi)
    if not mask.any():
        return float("nan")
    return float(mat[:, mask].mean())
```

### 3.3 — Shared baseline scalar helper

**Location:** `scripts/pipelines/tracking/qc_sheet_figures.py` (new module-level helper).

```python
def _shared_baseline_scalar(uid: UIDIntermediate) -> float:
    """One-baseline-for-all-heatmaps scalar for this UID, computed from the
    Baseline_ON PSTHs using the canonical baseline window (-1.75, -1.25).

    Pools across all sessions of the UID. Returns NaN if no Baseline_ON
    PSTH data is available (caller should fall back to per-event baselines).
    """
    psths_centers_n = [r.psths.get("baseline_on", (None, None, 0))
                        for r in uid.sessions]
    rows, centers = [], None
    for (psth, c, _n) in psths_centers_n:
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

### 3.4 — Render-time baseline mode toggle

**Location:** `scripts/pipelines/tracking/build_qc_sheets.py` (new CLI flag); `qc_sheet_figures.py` `_draw_heatmap` and `_draw_psth_summary` signature change.

**CLI flag:**

```python
parser.add_argument(
    "--shared-baseline", action="store_true",
    help="Use one Baseline_ON-derived baseline scalar for ALL heatmaps in "
         "each UID's page 2 (cross-event comparison mode). Default: per-event "
         "baseline from EVENT_RESPONSIVENESS_WINDOWS.",
)
```

`build_qc_sheets.py` passes the flag value through to `write_uid_pdf`, which passes it to `_draw_heatmap` and `_draw_psth_summary`:

```python
def _draw_heatmap(ax, uid, key, *, shared_baseline: bool = False, ...):
    ...
    if shared_baseline:
        baseline_scalar = _shared_baseline_scalar(uid)
        if not np.isfinite(baseline_scalar):
            baseline_scalar = _per_event_baseline_scalar(mat, centers, key)
    else:
        baseline_scalar = _per_event_baseline_scalar(mat, centers, key)
    if not np.isfinite(baseline_scalar):
        baseline_scalar = 0.0  # no subtraction; render raw values
    ...
```

The fallback chain: shared (if requested) → per-event → no subtraction. The "no subtraction" fallback should never trigger for canonical PSTH keys (after Component 3.1's window widening) but guards against future event types or empty matrices.

### 3.5 — Diverging colormap + symmetric vmax

**Location:** `scripts/pipelines/tracking/qc_sheet_figures.py`, `_draw_heatmap`.

Replace the existing `imshow` call:

```python
    mat_sub = mat - baseline_scalar
    vmax = max(float(np.percentile(np.abs(mat_sub), 95)), 0.5)
    ax.imshow(
        mat_sub, aspect="auto", origin="upper", cmap="RdBu_r",
        extent=[centers[0], centers[-1], mat_sub.shape[0], 0],
        vmin=-vmax, vmax=vmax,
    )
```

Add a small inline colorbar (or a text annotation showing the ±vmax in Hz) so reviewers can calibrate the color scale. Concretely: `ax.text(0.98, 0.02, f"±{vmax:.1f} Hz", transform=ax.transAxes, fontsize=7, color="0.4", ha="right", va="bottom")`.

The `vmax >= 0.5` floor prevents degenerate near-zero scales for cells with almost no modulation.

All other `_draw_heatmap` behavior unchanged: origin='upper' for row 0 at top (from prior work), red trim-marker overlay for dropped rows, "red bar = dropped row" legend.

### 3.6 — PSTH-summary baseline subtraction

**Location:** `scripts/pipelines/tracking/qc_sheet_figures.py`, `_draw_psth_summary`.

Use the SAME baseline_scalar that was computed for this row's heatmap (passed in as parameter to keep them in lockstep). Subtract from each stage-mean trace before plotting. Add reference line:

```python
def _draw_psth_summary(ax, uid, key, miss_keys=None, *, baseline_scalar: float = 0.0):
    ...
    # subtract baseline_scalar from each stage-mean trace before plotting
    ax.axhline(0, color="0.7", linewidth=0.5, alpha=0.8, zorder=0)
    ax.set_ylabel("Hz (rel. baseline)")
    ...
```

Sharing the scalar via parameter ensures that the heatmap's "zero" (white, on RdBu_r) corresponds to the line plot's y=0 reference — the two panels display the same baseline.

Stage colors, hit/miss line styles, and the existing legend changes from prior work all stay unchanged.

### 3.7 — ISI auto-pass (Option B)

**Location:** `src/visdetect/analysis/tracking_qc.py` (new constant + new function); `scripts/pipelines/tracking/build_qc_sheets.py` (call-site wiring).

**New constant:**

```python
# ISI hist-corr auto-pass: a UID whose set-wide ISI shape correlation is
# exceptionally consistent (>= 0.95, top ~25% of BG_046 cohort) is promoted
# to trusted regardless of marginal failures on other badges. Hard biophysical
# signals (wave or depth FAIL) still block — same philosophy as the
# skip-able-trim hard-outlier rule.
ISI_HIST_CORR_AUTOPASS: float = 0.95
```

**New function:**

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

**Call-site integration in `build_qc_sheets.py`:**

After both `composite_verdict(...)` invocations (main render loop AND trimmed-verdict precompute), apply autopass:

```python
verdict_csv = composite_verdict([...])
verdict_csv_pre_autopass = verdict_csv
verdict_csv = apply_isi_autopass(verdict_csv, metrics["isi_hist_corr"], b_wave, b_depth)
autopass_applied = (verdict_csv != verdict_csv_pre_autopass)
```

Similarly in the trimmed block, using `tm["isi_hist_corr"]` and the trimmed-metric badges.

**CSV columns:**

- `autopass_applied` in `verdicts.csv` (bool): True iff Option B changed the verdict.
- `trimmed_autopass_applied` in `verdicts_trimmed.csv` (bool): same for the trimmed verdict.

## 4. Data flow

```
EVENT_RESPONSIVENESS_WINDOWS (constants.py)
       │
       ▼
PSTH_SPECS (tracking_qc.py)  ── widened to fit canonical baselines
       │
       ▼  (cache rebuild)
data/cache/tracking_qc_intermediates.pkl
       │
       ▼
qc_sheet_figures.py:
   ├── _per_event_baseline_scalar(mat, centers, key)
   ├── _shared_baseline_scalar(uid)
   └── _draw_heatmap / _draw_psth_summary:
        ├── pick scalar based on --shared-baseline flag (fallback chain)
        ├── subtract from mat / traces
        └── render with RdBu_r + symmetric vmax (heatmap) or y=0 ref line (summary)


composite_verdict([b_isi, b_depth, b_wave, b_fr, b_hist, b_func])
       │
       ▼
apply_isi_autopass(verdict, isi_hist_corr, b_wave, b_depth)   ── NEW
       │
       ▼
verdicts.csv / verdicts_trimmed.csv (+ autopass_applied columns)
```

## 5. File-level changes

| File | Change |
|---|---|
| `src/visdetect/analysis/tracking_qc.py` | (a) Widen `PSTH_SPECS["baseline_on"]["window"]` to (−2.0, 1.5) and `PSTH_SPECS["hit_lick"]["window"]` to (−2.0, 1.0). (b) Add `ISI_HIST_CORR_AUTOPASS = 0.95` constant. (c) Add `apply_isi_autopass(verdict, isi_hist_corr, wave_badge, depth_badge, threshold=...)` function. |
| `tests/analysis/test_tracking_qc.py` | Append 5 unit tests for `apply_isi_autopass` covering: threshold met, wave fail block, depth fail block, below threshold, NaN. |
| `scripts/pipelines/tracking/build_qc_sheets.py` | (a) Import `apply_isi_autopass`. (b) Add `--shared-baseline` CLI flag. (c) Pass flag through to `write_uid_pdf`. (d) After both `composite_verdict(...)` calls, apply autopass and capture `autopass_applied`. (e) Add `autopass_applied` and `trimmed_autopass_applied` columns to the two CSV row dicts. |
| `scripts/pipelines/tracking/qc_sheet_figures.py` | (a) Import `EVENT_RESPONSIVENESS_WINDOWS`. (b) Add `_PSTH_KEY_TO_EVENT` dict, `_per_event_baseline_scalar`, `_shared_baseline_scalar` helpers. (c) `_draw_heatmap`: accept `shared_baseline` kwarg, compute baseline via fallback chain, subtract from `mat`, switch cmap to `RdBu_r`, use symmetric `vmin=-vmax, vmax=max(95p(|·|), 0.5)`, add inline ±vmax Hz annotation. (d) `_draw_psth_summary`: accept `baseline_scalar` kwarg, subtract from each stage trace, add y=0 reference line, update y-axis label. (e) `write_uid_pdf` orchestrates the per-row baseline computation and passes the same scalar to both `_draw_heatmap` and `_draw_psth_summary`. |
| `docs/superpowers/specs/2026-06-01-heatmap-psth-norm-and-isi-autopass-design.md` | This spec, committed. |

## 6. Testing

**Unit tests** (5 new in `tests/analysis/test_tracking_qc.py`):

- `test_apply_isi_autopass_promotes_when_threshold_met`: ISI 0.97 + no fails → trusted
- `test_apply_isi_autopass_blocks_on_wave_fail`: ISI 0.99 + wave fail → unchanged
- `test_apply_isi_autopass_blocks_on_depth_fail`: ISI 0.99 + depth fail → unchanged
- `test_apply_isi_autopass_below_threshold_no_change`: ISI 0.94 + no fails → unchanged
- `test_apply_isi_autopass_nan_no_change`: ISI NaN → unchanged

**No new unit tests for the normalization changes** — visual; verified by smoke render.

**End-to-end smoke** (cache rebuild required for the widened PSTH windows):

1. Rebuild cache: `py scripts/pipelines/tracking/build_qc_sheets.py --rebuild-cache`. Expect ~5–10 min, 61 UIDs unchanged.
2. Confirm `verdicts.csv` and `verdicts_trimmed.csv` have new `autopass_applied` / `trimmed_autopass_applied` columns.
3. Render UID 942 (gold, isi_hist_corr ≈ 0.99) and UID 779 (matching failure, isi_hist_corr ≈ 0.58):
   - UID 942: heatmaps now in RdBu_r with baseline near white; PSTH summaries show modulation from y=0; verdict still trusted; `autopass_applied` is False (the flag is True only when autopass CHANGED the verdict from a lower level — UID 942 was already trusted).
   - UID 779: heatmaps render with diverging cmap; `autopass_applied` is False; verdict stays suspect.
4. Identify a real Option B promotion example via:
   ```bash
   py -c "
   import pandas as pd
   t = pd.read_csv('FIGURES/tracking_qc/verdicts_trimmed.csv')
   cand = t[(t.trimmed_isi_hist_corr >= 0.95) & (t.trimmed_verdict != 'trusted')]
   print(cand[['global_uid','trimmed_isi_hist_corr','trimmed_verdict']].to_string())
   "
   ```
   Pick one UID; re-render; verify `trimmed_autopass_applied=True` and `trimmed_verdict=trusted` in the output CSV.
5. Render UID 942 with `--shared-baseline` flag; visually compare heatmap appearance to the default per-event mode (Change_ON should look subtly different because the baseline is now ITI-derived instead of pre-change-derived). Confirm both modes work.
6. Re-run `py -m pytest tests/analysis/test_tracking_qc.py` and confirm 80/80 passing (was 75 + 5 new tests).

## 7. Non-goals

The following are explicitly out of scope:

- **Per-row baseline subtraction** (rejected during brainstorming — hides cross-session FR-drift signal).
- **Stage-shared baseline** (rejected — hides cross-stage identity changes).
- **Cohort-shared baseline** (rejected — distorts most heatmaps).
- **Tightening depth thresholds** — depends on drift correction (separate large project).
- **Adjusting `ISI_HIST_CORR_AUTOPASS` threshold beyond 0.95** — empirical tuning after seeing post-implementation cohort distribution. Revisit in a future spec if needed.
- **Adjusting the consistency gate threshold (0.85) in the skip-able trim** — independent setting.
- **Persisting `--shared-baseline` per-UID** (e.g., as a CSV column) — for now a one-off CLI flag for visual comparison; revisit if reviewers want it baked into the production output.

## 8. Open questions

None at design time. All clarifications resolved in brainstorming.
