# Presentation-Prep Technical Roadmap — Design

**Date:** 2026-06-03
**Author / owner:** Ben (b.gonzales@ucl.ac.uk)
**Internal deadline:** ~Tue 2026-06-17 (self-imposed; real talk is later → buffer exists)
**Branch:** `docs/presentation-prep-roadmap`

---

## 1. Purpose

Ben is giving a research-update talk. The talk should show **new biological findings**, but
those depend on first finishing a set of **technical / label-producing tasks**. This document is
the *master roadmap* that sequences those tasks over a ~2-week internal sprint, defines how they
fit together, and sets the realistic "what's presentable" cut.

This is a **program plan**, not a single feature spec. Several workstreams are large enough to need
their own brainstorm → spec → plan → implementation cycle; this roadmap defines the interfaces
between them and the order, and flags which ones spin off their own design session.

The seven workstreams (six from Ben's list + video sync, added during brainstorm):

1. Long-term neuronal tracking — settle method, get results
2. Optotagging — reliable classification
3. GLM-HMM — settle method, get state classifications
4. TF-responsive neurons — reliable classification
5. Adding subjects — spike sorting + behavior
6. FSI / SPN separation by waveform
7. **Video sync** — per-trial start + motion-energy onset timing

---

## 2. Decisions taken during brainstorm

| # | Decision | Rationale |
|---|---|---|
| D1 | This chat builds the **full master roadmap** for all seven and will drive execution (directly or via sequential worktree-isolated chats). | Ben wants one coordinated plan; the three live chats become input producers. |
| D2 | Finding priority (naive, since findings are unknown until labels exist): **Learning > stimulus/evidence > cell-type > HMM**. | Catch-22: can't know the finding until the labels are trustworthy, so prioritise label quality. |
| D3 | **Depth on the critical path**: go deep on the top levers to real figures; the rest reach "method settled / in-progress". | 2 weeks cannot bring all seven to finding-ready results. |
| D4 | Roadmap structure = **Approach A: foundation-first, label-table-centric.** | Only structure where the three live chats are *inputs* (a column each), not merge conflicts. |
| D5 | The **two deep levers = Learning (mechanical) + TF-responsive (the stretch)**. | Learning is near-guaranteed; TF directly targets the stim/evidence headline despite research risk. |
| D6 | **First move = P0 spine audit.** | The shared table gates both headlines and no live chat is covering it. |

---

## 3. Architecture — the spine

**The integrating artifact is one per-unit label table.** Each workstream owns exactly one column
plus a documented *trust rule*. Findings never re-derive labels — they `groupby` the frozen table.

A `build_unit_table()` already exists at `src/visdetect/suite/loader.py:316` and already merges
`Global_UID`, `celltype`, `is_tf_responsive`, `is_lick_responsive`. **But it is old code built on
shaky inputs** (it reads a separately-produced `GLT_PATH` CSV; `load_waveform_labels()` reads a
stale `AI_exploration/analysis_3_waveform_celltype.py` output). Therefore the spine is treated as
**audit / test / maybe-rebuild**, never trusted as-is. See P0.

### Column contract (the interface every workstream writes to)

| Column | Producer workstream | Current state | What "settled" adds | Bucket |
|---|---|---|---|---|
| `Global_UID` + **`track_verdict`** | Tracking | UID exists; verdict missing | Lock UM 3.2.9; add trusted/review/suspect from QC-sheets | 🔧 mechanical |
| `celltype` (FSI/SPN) | Waveform | labels stale (preTprime / AI_exploration path) | Regenerate on **current** KS4 output + cutoff justification | 🔧 mechanical |
| **`opto_tag`** (D1/D2/none) | Optotagging | not in table | SALT methodology audit; add column even if sparse | 🧠 brainstorm |
| `is_tf_responsive` → **`tf_class`** | TF-responsive | z-threshold only; Ben dissatisfied | Build classifier matched to Ben's by-eye labels | 🧠 brainstorm |
| `hmm_state` (trial-level) | GLM-HMM | separate; trial- not unit-level | Lock K + labels (hmm-track-a) + fix switching latency | 🧠 brainstorm |
| motor-time per trial | Video sync | clock-sync in progress | Per-trial start + motion-energy onset; manual-tag fallback | 🧠 brainstorm |
| (rows for subject 2+) | Adding subjects | only BG_046 | Same columns for additional subjects + photometry arm | 🔧 mechanical |

Each finished workstream commits its column + trust rule on its own branch; the table is then
**rebuilt and frozen** before figures.

> **Frozen as of P0 (2026-06-03):** the canonical column set, dtypes, label defaults,
> and allowed values now live in `src/visdetect/suite/unit_table_schema.py`
> (`CONTRACT_COLUMNS`, `LABEL_DEFAULTS`, `ALLOWED_VALUES`, `validate_unit_table`).
> `build_unit_table(validate=True)` enforces it. Workstreams overwrite their one
> column; they must not rename or drop contract columns.
>
> **P0 audit finding:** on a fresh checkout the GLT (`table_output/Grand_Longitudinal_Table.csv`)
> and the configured waveform-label CSV are both ABSENT, so the table cannot be built until
> regenerated. The GLT is regenerated by `scripts/analysis/build_longitudinal_table.py` and
> inherits `Global_UID` from the UnitMatch registry — i.e. the spine is downstream of the
> tracking workstream (M1). Regenerate the GLT once M1 locks the UM 3.2.9 registry, then the
> skip-if test in `tests/suite/test_unit_table_build.py` must pass.

---

## 4. The decomposition — two buckets

### 🔧 Mechanical (decide + execute; this chat / sequential chats can just drive)

- **P0 — Spine audit/rebuild** *(prerequisite, gates everything)*: write tests for
  `build_unit_table()` + the GLT producer; verify joins, stage assignment, dtype of session keys;
  decide trust-or-rebuild. Output: a tested table builder and a frozen column schema.
- **Tracking**: lock **UM 3.2.9** (data already favours it: 19.8% ≥2-session vs 6.3% stock DeepUM);
  run `scripts/pipelines/tracking/validate_long_tracks.py` + `build_qc_sheets.py` (the
  `2026-05-22-tracking-qc-sheets-plan.md` plan) → `verdicts.csv` → `track_verdict`.
  **`feature/deepum-finetune` is upside only — never a blocker.**
- **FSI/SPN waveform**: a clean producer on the **current per-session KS4 output** (NOT
  `scripts/pipelines/concat_sort/` — that approach is retired; `regen_waveform_labels.py` is
  suspect). Trough-to-peak width (+ half-width / repolarisation), bimodality check for the cutoff,
  emit `celltype`.
- **Adding subjects**: `scripts/conversion/raw_to_pkl.py` + `validate_pkl.py` for **as many
  subjects as pipeline throughput allows** (not just 1–2). Long-running KS4 sorting runs in the
  **background from Day 1**. Behavior curves per subject while sorting runs. **Photometry arm**:
  the sibling repo `E:\python_analysis\git_repos\vis_detect_analysis_Apr2023` (same task) can pool
  population-level signal; if D1/D2-Cre–specific, it is a **cell-type-resolved learning signal that
  does not depend on optotagging yield**.

### 🧠 Needs its own brainstorm (open methodology; each spawns spec→plan)

- **TF-responsive** *(2nd deep lever; on the stim/evidence critical path)*: Ben is dissatisfied with
  current results but confident by eye. **Ben's by-eye labels are ground truth.** Task = build a
  classifier / curation that matches his eye and reconcile with the manual GUI labels
  (`src/visdetect/analysis/tf_labeling.py`) and `analysis_suite/08_tf_pulse/g_tf_cell_classifier.py`.
  Not "tune a z-threshold."
- **Video sync** *(promoted to high)*: today all lick times come from the piezo lick-spout contact
  (sometimes estimated as *contact − X ms*, which is sub-optimal). Goal = (a) reliable **per-trial
  start from video** (auto + **manual-tag fallback** — Ben is willing to tag), (b) **motion-energy
  onset** to replace the piezo-minus-Xms lick timing. Coordinate with the live
  `feature/video-sync-anchor-barcode` chat (it does clock-sync; this is a new capability on top).
- **HMM switching latency** *(medium)*: `worktree-feature+hmm-track-a` (plan
  `2026-05-27-hmm-track-a.md`) fixes *labeling defensibility* (F4/F25 explicit labels, F14
  confidence gating) but **not** Ben's two observed problems: switches happen **too late**, and
  activity that looks "in-the-zone / stimulus-sensitive" is **mislabelled Impulsive**. Add a
  switching-dynamics investigation: sticky/self-transition prior, posterior smoothing, emission
  definition. Hand off from hmm-track-a once it finishes.
- **Optotagging methodology** *(medium)*: prior yield was very low (one example animal). Audit the
  method (`src/visdetect/analysis/optotagging.py`): SALT window/params, waveform-collision test,
  alternative tagging metrics — can we do better? Emit `opto_tag` even if sparse; report yield
  honestly.

---

## 5. Schedule (internal target ~2026-06-17; real talk later = buffer)

Units run in **worktree-isolated chats** (matches Ben's parallel-chat workflow). Each off-branch
chat sets `PYTHONPATH=<worktree>/src` so it tests its own code, not main's (known gotcha). The three
live chats keep running as input producers.

```
WEEK 1 ── foundations + commit to TF
 Day 1   P0  Spine audit: test build_unit_table + GLT producer → tested
             builder + frozen column contract. (THIS chat first.)
 Day 1   BG  Kick off added-subject KS4 sorting in BACKGROUND (long-running).
             Pick subjects; behavior curves run while it sorts.
 Day 1   D1  Open TF-responsive brainstorm chat (Ben's eye = ground truth).
 D1–3    M1  Tracking: lock UM 3.2.9; validate_long_tracks + build_qc_sheets
             → verdicts.csv → track_verdict.
 D1–3    M2  FSI/SPN waveform producer on current KS4 → celltype + cutoff.
             [M1 ∥ M2 ∥ D1, separate chats]
 D2–5    D1  TF brainstorm → spec → start build (classifier matched to labels).
 Day 5   ★   F-Learning first pass: Naive→Expert on trusted tracked units × celltype.

WEEK 2 ── stretch finding + settle tier + assembly
 D6–8    D1  Finish TF build → tf_class → ★ F-StimEvidence figure.
 D6–8    S1  Video-sync brainstorm+build (motion-init + manual-tag)   ⏱ time-box
 D6–8    S2  HMM switching-latency (handoff from hmm-track-a)          ⏱ time-box
 D6–8    S3  Optotagging methodology audit → opto_tag                  ⏱ time-box
 D7–8    BG  Subject sorting completes → behavior + pooled example.
 D9–10   ▣   FREEZE unit-label table; finalize figures; assemble tiers.
```

**Critical path:** `P0 → (M1 ∥ M2) → F-Learning` and `P0 → D1 brainstorm → D1 build → F-StimEvidence`.
Everything in the ⏱ time-boxes can slip without touching the two headline figures.

---

## 6. Risks and the presentable cut (three slide tiers)

| Tier | Slides show | Workstreams | Main risk → fallback |
|---|---|---|---|
| **1 — Results** | Learning figure; TF/evidence figure | Tracking, Waveform, TF | **TF doesn't converge** → by-eye curated subset for the figure. **Too few trusted tracks** → Learning from **per-session unit populations** across stages (tracking-independent; tracked is nicer but not required). |
| **2 — Method settled** | Tracking decision + N trusted long-tracks + QC sheet; FSI/SPN counts; new-subject behavior curves | Tracking, Waveform, Subjects | Subject sorting stalls → behavior-only for new subjects |
| **3 — In progress** | HMM (current + diagnosed late-switch + fix plan); optotagging yield + method options; video-sync status + motion-init plan | HMM, Optotagging, Video | These are *expected* in-progress; honest framing is the deliverable |

### Cross-cutting risks being designed around
- **Don't collide with live chats** — consume their *committed outputs* (branch/CSV/column); never
  edit their files. Live chats: `deepum-finetune` (tracking upside), `hmm-track-a` (HMM labeling),
  `video-sync-anchor-barcode` (clock-sync).
- **The spine is the single point of failure** — if `build_unit_table` is wrong, both headlines are
  wrong. P0 (audit + tests) is Day 1 and gates everything.
- **TF is genuine research risk** — the by-eye-curated fallback must exist before relying on it.
- **Learning is the safe headline** — mostly mechanical and has a tracking-independent fallback.

---

## 7. Workstream → existing assets (so chats don't re-derive)

| Workstream | Library | Scripts | Notes |
|---|---|---|---|
| Spine | `suite/loader.py` (`build_unit_table`, `load_glt`, `load_waveform_labels`) | — | Audit + tests first |
| Tracking | `analysis/tracking_qc.py` (+ tests) | `pipelines/tracking/{run_unitmatch_all, run_deepunitmatch_all, validate_long_tracks, validate_waveforms, diagnose_intersession_drift, build_qc_sheets, qc_sheet_figures}.py` | QC-sheets plan `2026-05-22-…` |
| Waveform | (new clean producer) | retire `concat_sort/regen_waveform_labels.py`; current KS4 output | Bimodality cutoff |
| Optotagging | `analysis/optotagging.py` | — | SALT audit |
| TF-responsive | `analysis/tf_labeling.py`, `analysis/tf_pulse.py` | `tf_labeling/`, `08_tf_pulse/{g_tf_cell_classifier, _eval_cutoffs}.py`; caches `tf_cell_classification*.csv` | Own brainstorm |
| GLM-HMM | `analysis/hmm.py`, `hmm_downstream.py` | hmm-track-a plan/spec `2026-05-27-…` | Switching add-on |
| Video sync | `core/video_sync.py` | `video/`; plans `2026-05-27-video-sync-anchor-barcode-*` | Own brainstorm |
| Subjects | `core/ingest.py` | `conversion/{raw_to_pkl, validate_pkl}.py`, `data_management/organize_subject_data.py` | + photometry repo `…Apr2023` |

---

## 8. Next action

Begin **P0 — spine audit** via the writing-plans skill: produce an implementation plan to test
(and, if needed, rebuild) `build_unit_table()` and the GLT producer, and freeze the column schema
that every other workstream writes to. Then spin up M1 / M2 / D1 in parallel worktree chats.
