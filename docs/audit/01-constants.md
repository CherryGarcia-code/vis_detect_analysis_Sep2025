# D1 — Constants audit

Static census of canonical constants and their shadows across `src/`, `scripts/`, `tests/`
(excluding `.venv`, `archive`, `__pycache__`, `.claude`, `_DeepUnitMatch_repo`,
`refactor_baseline`, `_preserved_from_worktrees_20260628`).

- Script: `scripts/audit/d1_constants_census.py` (AST-based; scanned files are never imported)
- Census CSV: `data/cache/audit/constants_census.csv` (gitignored; committed with `git add -f`)
- TF-period site census: `data/cache/audit/tf_dt_sites.csv`
- Measurement ids: `d1.constants.*`, `d1.tfperiod.*` in `docs/audit/measurements.csv`

## Census summary (Task 3, static)

| Measurement | Value | Recon baseline | Verdict |
|---|---|---|---|
| `d1.constants.total` — canonical constants in `constants.py` | 82 | ≈ 82 | exact match |
| `d1.constants.dead` — zero importers AND zero retypes | 16 | ≈ 22 | below baseline; explained below |
| `d1.constants.not_reexported` — config.py fails to re-export | 42 | ≈ 42 | exact match |
| `d1.constants.shadow_disagree` — canonical with DISAGREEING retyped copies | 0 | — | no canonical constant is retyped anywhere in scope (0 retype sites) |
| `d1.constants.divergent_params` — non-canonical divergent parameter names (bucket a) | 98 | — | see multi-file note below |
| `d1.tfperiod.consumer_sites` — TF_SAMPLE_PERIOD + bare dt=0.05/DT_GEN/DT=0.02 sites | 83 | — | full listing in `tf_dt_sites.csv` |
| `d1.tfperiod.figure_attribution` | not-measured | — | requires per-figure provenance that does not exist (D4 measures that gap) |

**Bucket breakdown** (298 census rows): canonical 82, divergent-parameter 98,
path-alias 36, genuinely-local 82.

### Deviation: dead = 16 vs recon ≈ 22

The delta is methodological, not repo drift. Recon's ≈ 22 predates the pre-flight fix that
replaced line-regex importer detection with AST `ImportFrom` parsing; the regex missed
multi-line parenthesized imports (13 in-scope files, including config.py's 40-name re-export
block) and was measured to misclassify 48 of 82 constants as zero-importer, so recon's
dead count was inflated. Cross-check: re-running the census with the post-recon untracked
files and the audit's own `scripts/audit/` files excluded from importer detection still
yields dead = 16 — the six rescued constants are found by better detection, not by new code.
The 16 dead names: LOHSE_EVIDENCE_THRESH, LOHSE_PULSE_SIGMA_MS, LOHSE_TRIAL_NORM_BASELINE,
LOHSE_TRIAL_SIGMA_MS, MOTION_ENERGY_DOWNSAMPLE, MOTION_ENERGY_MOUTH_ROI,
PUPIL_BLINK_MAX_GAP_MS, PUPIL_BLUR_KERNEL, PUPIL_EYE_ROI, PUPIL_MAX_AREA, PUPIL_MIN_AREA,
PUPIL_MIN_ROUNDNESS, PUPIL_THRESH_BLOCK_SIZE, STATE_LABEL_W_DEFAULT, TF_DETREND_BASELINE,
TF_DETREND_POST_WINDOW.

### Multi-file disagreeing names vs recon's "130 divergent"

127 census rows have `retypes_agree = False` (98 divergent-parameter + 29 path-alias with
differing values; all 127 are non-canonical), against recon's ≈ 130. The small shortfall is
bucketing granularity (path-alias is keyed on name keywords DIR/PATH/ROOT/FILE/OUT) plus
normal repo evolution; several multi-file names (e.g. `CACHE`, `OUT_DIR`, `SUBJECT`) count
definition sites inside the audit-era untracked scripts under `scripts/` — such deltas are
the audit's own footprint, not drift.

**CAVEAT — known bucket mislabels (Task 15 MUST read this).** The `"OUT" in name`
substring rule mislabels scientifically loaded names as `path-alias`, and Task 15 must
treat the following as **candidate divergent scientific parameters** when building the
register, even though the CSV buckets them `path-alias`:

- `OUTCOMES` — 8 disagreeing sites (`retypes_agree = False`); these are lists of which
  trial outcomes an analysis includes, not paths. `defined_in`:
  `scripts\tf_responsiveness\state_conditioned\continuum_common.py:21`;
  `scripts\tf_responsiveness\state_conditioned\diseng_sensitivity.py:60`;
  `scripts\tf_responsiveness\state_conditioned\learning_transient_sustained.py:38`;
  `scripts\tf_responsiveness\state_conditioned\null_controls_continuum.py:35`;
  `scripts\tf_responsiveness\state_conditioned\robustness_width_coupling.py:28`;
  `scripts\tf_responsiveness\state_conditioned\spectrum_vs_classes.py:38`.
- `OUTCOME_COLORS` — 3 disagreeing sites (`retypes_agree = False`); the config palette vs
  local overrides. `defined_in`: `scripts\state_labeling\population_state_sensory.py:64`;
  `scripts\video\sync_validation_figure.py:60`; `src\visdetect\analysis\config.py:220`.
- `N_TRIALS_PER_OUTCOME` — 2 sites, values currently agree (`retypes_agree = True`), but
  it is an analysis parameter, not a path; it belongs with the scientific names, not the
  path scaffolds. `defined_in`: `scripts\video\batch_sync_sessions.py:54`;
  `scripts\video\sync_validation_figure.py:43`.

Conversely, the `divergent-parameter` bucket (98, `d1.constants.divergent_params`) is
inflated by pure path/scaffold aliases that carry no keyword the heuristic recognizes —
e.g. `_HERE` (41 sites), `CACHE` (26), `_REPO` (18) are script-directory/cache-path
scaffolds sitting in bucket (a). So `d1.constants.divergent_params = 98` both omits real
scientific parameters and includes scaffolds; Task 15 should re-triage bucket membership
name-by-name from the CSV rather than consuming the bucket labels as ground truth.

### Mandatory spot-check: `CHANGE_SIZES`

PASS — `CHANGE_SIZES` is **not** in constants.py-canon. The census classifies it
`divergent-parameter` with 3 definition sites:
`src\visdetect\analysis\config.py:264`, `src\visdetect\analysis\tf_glm.py:210`,
`scripts\analysis\decision_latents\run_decision_latents_by_state.py:64` — and their values
disagree (`retypes_agree = False`). Note this contradicts CLAUDE.md, which presents
`CHANGE_SIZES` as living in `constants.py`. Only the derived sets
(`SMALL_CHANGE_SIZES`, `BIG_CHANGE_SIZES`, `ALL_GO_CHANGE_SIZES`) are canonical.

## Worst 10 disagreeing names

Non-canonical multi-file names with `retypes_agree = False`, ranked by number of retype
sites (`n_retype_sites`). Sites are quoted from the census CSV's `defined_in` column
(capped at 6 sites per name by the script).

| # | Name | Bucket | Sites | `defined_in` (file:line) |
|---|---|---|---|---|
| 1 | `OUT` | path-alias | 48 | `scripts\analysis\decision_latents\_concept_slide.py:29`; `scripts\analysis\decision_latents\_recovery_summary_fig.py:49`; `scripts\popgeom_theta\theta_count_matched.py:24`; `scripts\popgeom_theta\theta_prototype.py:27`; `scripts\popgeom_theta\theta_support_matched.py:23`; `scripts\state_dynamics\within_session_dynamics.py:25` |
| 2 | `_HERE` | divergent-parameter | 41 | `scripts\tf_responsiveness\cluster\tf_glm_cluster_task.py:43`; `scripts\tf_responsiveness\cluster_bg\tf_glm_bg_task.py:42`; `scripts\tf_responsiveness\preparatory_fig5\fig5e_fraction_active.py:27`; `scripts\tf_responsiveness\preparatory_fig5\fig5fg_onset_heatmaps.py:33`; `scripts\tf_responsiveness\preparatory_fig5\fig5h_onset_vs_width.py:36`; `scripts\tf_responsiveness\preparatory_fig5\fig_hit_vs_fa.py:26` |
| 3 | `REPO_ROOT` | path-alias | 30 | `scripts\analysis\build_waveform_celltype_labels.py:13`; `scripts\analysis\run_deep_unitmatch.py:58`; `scripts\batch_processing\aggregate_modulation_across_sessions.py:20`; `scripts\optotagging\render_opto_candidates.py:51`; `scripts\pipelines\concat_sort\build_concat_pkls.py:38`; `scripts\pipelines\concat_sort\build_concat_windows.py:29` |
| 4 | `CACHE` | divergent-parameter | 26 | `scripts\analysis\decision_latents\run_decision_latents_by_state.py:21`; `scripts\chronic_feasibility\chronic_feasibility_figure.py:77`; `scripts\optotagging\render_opto_exemplar_figure.py:57`; `scripts\state_tf_learning\b9_summary_figure.py:31`; `scripts\talk_substrate\_events_plot.py:29`; `scripts\talk_substrate\build_event_cache.py:59` |
| 5 | `SUBJECT` | divergent-parameter | 25 | `scripts\chronic_feasibility\chronic_feasibility_figure.py:70`; `scripts\data_management\organize_bg012_behavior.py:39`; `scripts\optotagging\render_opto_candidates.py:73`; `scripts\optotagging\render_opto_exemplar_figure.py:54`; `scripts\pipelines\concat_sort\build_concat_pkls.py:59`; `scripts\pipelines\concat_sort\compare_old_vs_concat.py:31` |
| 6 | `OUT_DIR` | path-alias | 24 | `scripts\QC_technical\audit_trial_baselineon_alignment.py:51`; `scripts\QC_technical\characterize_unsolvable_alignment.py:110`; `scripts\QC_technical\repair_trial_event_alignment.py:29`; `scripts\batch_processing\aggregate_batch_results.py:14`; `scripts\optotagging\render_opto_candidates.py:56`; `scripts\pipelines\concat_sort\qc_ks4_runs.py:35` |
| 7 | `_ROOT` | path-alias | 22 | `scripts\QC_technical\audit_trial_baselineon_alignment.py:36`; `scripts\QC_technical\characterize_unsolvable_alignment.py:83`; `scripts\QC_technical\repair_trial_event_alignment.py:19`; `scripts\analysis\behavior\early_lick_learning_trajectory.py:33`; `scripts\analysis\behavior\early_lick_replication.py:26`; `scripts\analysis\behavior\fa_lick_hazard_learning.py:36` |
| 8 | `STATES` | divergent-parameter | 21 | `scripts\evidence_learning\b10_phase2_state.py:32`; `scripts\session_sorting\fit_session_group_rule.py:53`; `scripts\state_dynamics\within_session_dynamics.py:29`; `scripts\state_labeling\exemplar_sensory_decomposition.py:43`; `scripts\state_labeling\exemplar_state_conditioned_units.py:56`; `scripts\state_labeling\explore4_partial_rt.py:38` |
| 9 | `CACHE_DIR` | path-alias | 20 | `scripts\analysis\behavior\early_lick_learning_trajectory.py:63`; `scripts\analysis\behavior\fa_lick_hazard_learning.py:55`; `scripts\analysis\decision_latents\_comprehension_flag_explore.py:57`; `scripts\analysis\decision_latents\_diag_f1b_vs_f1d.py:20`; `scripts\analysis\decision_latents\_expert_anchor_inventory.py:48`; `scripts\analysis\decision_latents\_label_reliability.py:53` |
| 10 | `_REPO` | divergent-parameter | 18 | `scripts\state_labeling\exemplar_sensory_decomposition.py:56`; `scripts\state_labeling\exemplar_state_conditioned_units.py:62`; `scripts\state_labeling\explore4_partial_rt.py:42`; `scripts\state_labeling\explore_extract.py:46`; `scripts\state_labeling\explore_state_neural.py:45`; `scripts\state_labeling\gain_illustration.py:50` |

The top of this ranking is dominated by per-script path/scaffold aliases (`OUT`, `_HERE`,
`REPO_ROOT`, `CACHE`, `OUT_DIR`, `_ROOT`, `CACHE_DIR`, `_REPO`) — many definition sites,
low scientific risk. The scientifically loaded disagreements (e.g. `CHANGE_SIZES`,
`STATES`, `SUBJECT`, and the mislabeled path-alias rows `OUTCOMES` / `OUTCOME_COLORS` /
`N_TRIALS_PER_OUTCOME` named in the caveat above) carry fewer sites but feed Task 15's
defect register directly.

## Executed measurements

Runtime checks on real BG_046 pkls (Task 4). Script: `scripts/audit/d1_executed_checks.py`
(`py scripts/audit/d1_executed_checks.py`, exit 0). Primary session `01072025` (deliberately a
day-1-9 leading-zero id — `load_session("01072025")` resolved it correctly, so no id-handling
finding); ref-trial check over 5 sessions. Measurement ids: `d1.qcprofile.*`, `d1.frfloor.*`,
`d1.ref.*`, `d1.tfperiod.measured_s`, `d1.palette.*` in `docs/audit/measurements.csv`.

| Measurement | Value | Expectation | Verdict |
|---|---|---|---|
| `d1.qcprofile.{default,qc_only,striatal_strict,striatal_lenient}` | `{}` × 4 | `{}` × 4 | no-op defect CONFIRMED by execution |
| `d1.qcprofile.diff.*` (YAML-intended thresholds, session 01072025) | 108 units under all 4 profiles | counts differ across profiles | deviation — see verdict below |
| `d1.frfloor.spread` (good_and_stable / getgood 1.0 Hz / getgood 0.1 Hz) | 108/92/108 | three distinct counts | deviation — two distinct populations, see verdict |
| `d1.ref.total` / `d1.ref.with_change_time` | 18 / 18 | `ref_with_change ≈ tot_ref` | change WAS presented on every ref trial |
| `d1.ref.rt_median_ms` | +83 ms | small positive median RT | reflex interpretation confirmed |
| `d1.ref.rt_dict_keys` | `FA;Miss;RT;Ref;abort;gray` | — | `Ref` key present as pre-flight predicted |
| `d1.tfperiod.measured_s` | not-measured | ≈ 0.05 s or honest fallback | stim logs `None` on this pkl; documentary fallback |
| `d1.palette.hex_total` / `hex_distinct` | 692 / 174 | recon ≈ 717 / 194 | census confirmed at slightly smaller tracked-only scope |

### Verdict: qc-profile no-op (`d1.qcprofile.*`)

CONFIRMED at runtime. All four named profiles return `{}` from `load_qc_profile()`. Mechanism:
`src/visdetect/core/qc.py:215-221` resolves the YAML as
`Path(__file__).resolve().parents[1] / "config" / "qc_profiles.yml"` =
`src/visdetect/config/qc_profiles.yml`, which does not exist — the real file lives at repo root
`config/qc_profiles.yml` — and the function silently returns `{}` on the missing path. **Blast
radius: every `--profile` invocation that does not pass an explicit `profiles_path`.** Live call
sites: `scripts/batch_processing/batch_plot_tf_pulse.py:35`,
`scripts/analysis/tf_response/plot_tf_pulse_grid.py:56`, and
`src/visdetect/analysis/unit_selection.py:249` (which then does `used_params.update(prof)` — an
empty dict updates nothing, so every profile silently collapses to the function-default
parameters; `scripts/batch_processing/batch_plot_tf_grids.py` forwards `--profile` into
`plot_tf_pulse_grid.py` and inherits the defect). A user requesting `striatal_strict`
(1200 spikes, 3% ISI) actually ran the defaults (500 spikes, 20% ISI) with no warning.

The intended-vs-actual comparison (`d1.qcprofile.diff.*`) shows a second, structural fact: on
session 01072025 all four YAML-intended profiles pass the same 108 units, because the pkl stores
spikes only for the 108 `good_and_stable` clusters (of 260 KS-good), and that ingest gate already
dominates every YAML floor (measured on-pkl minima: 5223 spikes ≥ any `min_total_spikes`; max
ISI-violation fraction 0.014 ≤ even strict's 0.03). So on today's pre-filtered pkls the four
profiles are indistinguishable anyway — the no-op's practical blast radius is historical runs on
fuller populations (raw KS outputs, pre-filter pkls) and, more importantly, the *silent* failure
mode itself: the mechanism will misconfigure any future profile whose thresholds do bind.
(Caveat: the intended-threshold check applies the three numeric floors only, not
`require_good_cluster`/`min_median_spikes_per_trial`, per the audit script's explicit metric
construction.)

### Verdict: FR-floor spread (`d1.frfloor.*`)

Deviation from the plan's prediction of three distinct counts: measured 108/92/108 — two
distinct populations. Explanation from source: `get_good_cluster_ids` (`analysis/utils.py:239`)
starts from `good_and_stable_ids` (108 units, ingest-gated at ≥ 0.5 Hz by
`find_good_stable_units`, `core/qc.py:269`) and then applies its own rate floor. The yml's
0.1 Hz floor (`config/qc_profiles.yml:8`) can therefore **never bind** downstream of the 0.5 Hz
ingest gate — it is structurally vacuous on these pkls, which is why paths 1 and 3 coincide at
108. The only live, binding floor is `get_good_cluster_ids`' hardcoded 1.0 Hz default, which
drops 16/108 units (14.8%) on this session. The three *nominal* FR floors (0.1 yml / 0.5 ingest
/ 1.0 code default) thus produce two *actual* populations, and which one an analysis used
depends on whether it called `get_good_cluster_ids` or read `good_and_stable_ids` directly —
a real, silent 15% population difference between scripts that believe they use "the" QC'd units.

### Verdict: ref trials (`d1.ref.*`) — settles the Task 15 quarantine entry

Across 5 sessions (01072025, 23062025, 08072025, 15072025, 30062025): 18 ref trials, and **all
18 have a valid `change_time`** (`d1.ref.with_change_time` = 18 = total). The change stimulus
WAS presented on ref trials. Median RT from change onset is **+83 ms** (`d1.ref.rt_median_ms`,
keyed `Ref` in `Trial.reactiontimes`; observed keys `FA;Miss;RT;Ref;abort;gray`) — the lick
lands after change onset but far below any plausible detection latency, i.e. a reflex/chance
lick, exactly the behavioral-software definition. Resolution: `CHANGE_PRESENTED_OUTCOMES`
including `Ref` is **factually right** (the stimulus event exists and could be aligned to), and
`EVENT_VALID_OUTCOMES` excluding `ref` from `Change_ON` is a **scientific choice** (excluding
trials whose lick is uninterpretable as detection), not a data fact. Task 15 should quarantine
it as a documented convention, not a defect. Ref trials are rare (18 over 5 sessions, 0 in two
of them), so no analysis is materially biased either way.

### Verdict: TF sample period (`d1.tfperiod.measured_s`)

`not-measured` — the honest fallback the plan anticipated. Session 01072025's trials carry the
`stim_vbl` / `stim_tf_disp` fields but all are `None`: these are legacy-pkl placeholders "until
backfilled from raw trials.json" (`core/session.py:32-35`), so no flip-timestamped TF trace
exists in the pkl to measure a period from. The dt = 0.05 s value therefore rests on documentary
evidence: `src/visdetect/analysis/psychophysical_kernel.py:18` ("Everything is dt = 0.05 s (the
50 ms TF update). Never 0.25.") and the memory note `tf_fluctuation_50ms_vs_constant`. The
canonical `TF_SAMPLE_PERIOD = 0.25` (`constants.py:113`) remains the known-wrong value, 5× the
documented true period; its 83 consumer/`dt` sites are already censused in
`d1.tfperiod.consumer_sites` (`data/cache/audit/tf_dt_sites.csv`). An executed measurement
requires either a backfilled pkl or the raw `trials.json` — flagged for the register rather than
guessed here.

### Palette census (`d1.palette.*`)

One-line census per the plan: `git grep -oh "#[0-9a-fA-F]\{6\}" -- scripts/ | sort | uniq -c |
sort -rn > data/cache/audit/hexes.txt`. Measured: **692 hex-literal occurrences, 174 distinct
colors** in tracked `scripts/` files (recon context: ≈ 717 / 194; the delta is scope — `git
grep` sees tracked files only, excluding audit-era untracked scripts — plus normal drift).
Top repeats: `#3474ae` ×38 and `#ef6548` ×28 — which are exactly the canonical
`STATE_LABEL_COLORS` values (Disengaged, Impulsive) re-hardcoded per script instead of imported
from config — then `#333333` ×27. Full ranking in `data/cache/audit/hexes.txt`.
