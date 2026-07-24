# Preparatory activity vs behavioural regulation — an *across-regulation* reframe of the Fig 5 e–h cell-class extension

**Date:** 2026-07-21
**Status:** DESIGN (spec) — awaiting user review before writing the implementation plan
**Parent work:** `docs/science/2026-07-20-preparatory-activity-transient-sustained.md`
(within-striatum port of Khilkevich & Lohse Nature 2024 Fig 5 e–h), memory
`prep_activity_transient_sustained_jul2026`.
**Sibling line this shares an axis with:** `FIGURES/popgeom_fa_cutoff/`
(behavioural state-space / regulation axis; `improvement_vectors.csv`).

---

## 1. One-line honest summary

The parent result's **robust core** is "TF-responsive (esp. sustained) striatal cells
carry the earliest/largest pre-lick preparatory ramp; non-TF cells last." This spec
extends that across *learning* — but reframed from arbitrary discrete Learning/Expert
**stages** to a **continuous, session-level behavioural-regulation axis** (impulsivity =
anticipatory-FA rate ↓, sensitivity = d′ ↑). The primary question: **does the sustained-
vs-non-TF preparatory lead scale with behavioural regulation, within and across animals?**
The design is **cache-only** (reuses `prep_{hit,fa}.npz`; no ~30-min rebuild), **drift-immune**
(the neural quantity is a *within-session* contrast, so chronic drift never enters a
cross-stage neuron comparison), and **may legitimately return a clean null.**

---

## 2. Why this reframe (verified feasibility facts, 2026-07-21)

All numbers below were measured from the live caches/manifests this session, not from notes.

**2.1 The discrete-stage design is weak and the transient rung is empty.**
TF-responsive cell-sessions per mouse × stage × class (from `prep_hit.npz` joined to the
staging manifests, Naive→Learning merged):

| Mouse | Region | Stage | transient | sustained | intermediate | non-TF |
|---|---|---|---|---|---|---|
| BG_046 | DMS | Learning | **1** | 13 | 49 | 2492 |
| BG_046 | DMS | Expert | **3** | 32 | 64 | 2786 |
| BG_031 | VMS | Learning | **4** | 48 | 86 | 2293 |
| BG_031 | VMS | Expert | **6** | 62 | 113 | 2644 |
| BG_039 | DMS | Learning | 0 | 0 | 2 | 78 |
| BG_039 | DMS | Expert | 1 | 12 | 24 | 785 |

- **Transient cannot support a per-stage population onset** (1–6 cells) → the parent's
  3-way ordering-across-learning is untestable. Dropped from the primary.
- **BG_039 is Expert-only** (1 Learning QC session, 0 sustained) → contributes only at
  the high-d′/low-FA end of the axis.
- Only **BG_046 and BG_031** have both stages.

**2.2 Tracking cannot anchor this (decided + empirically empty).** Of the 46 BG_046
UM∩DANT consensus neurons spanning Learning→Expert, **4** are ever TF-responsive and
**0** are TF-responsive in *both* stages. Within-neuron cross-stage trajectories for the
TF classes are therefore impossible with the current tracked set. Tracking is **out of
scope** here (it needs curation + possibly a better tracker — future work).

**2.3 The stage table understates what the animals actually sample.** From
`improvement_vectors.csv` (animal-mean early→late vectors): only **BG_046** has a real
trajectory (ΔFA −0.09, Δd′ +0.50); **BG_039** barely moves (already good); **BG_031**'s
net vector ≈ 0 (ΔFA +0.01, Δd′ −0.03). Yet BG_031's *individual sessions* span the whole
regulation range — measured here from its 31 QC sessions, SDT fa_rate 0.07–0.61 and d′
0.83–1.64 (and the sibling's anticipatory-FA range is wider still). A continuous per-session
axis turns BG_031 from an "unusable non-learner" into a within-animal supplier of both ends
of the axis, and the overlapping clouds across animals (§2.4) enable a **behaviour-matched**
cross-animal test.

**2.4 The behavioural coordinates exist and span/overlap** (all QC-pass sessions, valid):

| Mouse | n QC sess | d′ range | SDT fa_rate range |
|---|---|---|---|
| BG_046 | 34 | 0.86–2.45 | 0.04–0.47 |
| BG_039 | 22 | 0.94–2.20 | 0.00–0.45 |
| BG_031 | 31 | 0.83–1.64 | 0.07–0.61 |

⚠️ The manifest's `fa_rate` is the **SDT** false-alarm rate. The sibling axis
(`improvement_vectors.csv`) uses an **anticipatory-FA / early-lick** impulsivity rate
(BG_046 ≈ 0.4–0.5). The plan MUST reconcile to a single shared definition (§6, checkpoint C1).

---

## 3. Scientific questions

- **Q1 (primary).** Does the **sustained-vs-non-TF** preparatory *lead* scale with
  session-level behavioural regulation (lower anticipatory-FA rate and/or higher d′),
  pooled across animals?
- **Q1b (robustness).** Same for **TF-responsive-pooled (resp=True) vs non-TF** (max power).
- **Q2 (drift-immune stat).** Does the *within-session* sustained-vs-non-TF lead depend on
  that session's regulation coordinate after accounting for animal (mixed model)?
- **Q3 (cross-animal, region-confound-resistant).** At **matched** behavioural coordinates,
  is the within-session lead the same regardless of animal/region — i.e., is the neural
  geometry set by behaviour rather than animal?
- **Q4 (states — overlay & robustness).** Does behavioural-state occupancy (StimSens /
  Impulsive / Disengaged) vary along the regulation axis as expected, and does the Q1/Q2
  regulation effect **survive controlling for state occupancy** (i.e., is it more than a
  state-mixture proxy)? Trial-level within-session state contrasts are secondary and
  circularity-caveated. 
- **Decomposition (mandatory).** For every "lead grows" result, report an **amplitude-
  normalized (timing)** version alongside the **absolute-threshold (magnitude)** version,
  because the parent effect is a magnitude/reliability ordering, not a proven latency one.

**Acceptance = a clean, hardened answer in either direction.** A null slope that survives
the shuffle control is a legitimate deliverable. No slope will be manufactured by tuning
bins/metrics.

---

## 4. Data & definitions

**4.1 Cohort / sessions.** Reuse the parent's session gate exactly: `prep_common.good_dates()`
(QC-pass AND <50% Disengaged). 3 mice: BG_046 (DMS), BG_039 (DMS), BG_031 (VMS).

**4.2 Neural source = the existing cache (no rebuild).** `data/cache/preparatory_fig5/
prep_{hit,fa}.npz` — per-cell lick-aligned z-traces `z [11598×140]`, `t`, `resp`,
`interp_fwhm`, `region`, `meta_{subject,session,unit}`. All estimators below are computed
from these arrays. Both **hit** (decision) and **fa** (impulsive) licks are analysed.

**4.3 Cell groups.**
- **sustained** = `resp=True & interp_fwhm ≥ 0.15` (project BROAD cut).
- **non-TF** = `resp=False`.
- **TF-responsive-pooled** = `resp=True` (robustness arm, Q1b).
- transient / intermediate: **excluded from the primary** (documented, not silently dropped).

**4.4 Behavioural coordinate (per session).** A 2-D point `(impulsivity, sensitivity)`:
- **sensitivity** = d′ (SDT, log-linear corrected) via `behavior.compute_session_performance`
  / manifest `d_prime`.
- **impulsivity** = the **anticipatory-FA / early-lick rate** as defined by the sibling
  `popgeom_fa_cutoff` line (NOT the SDT `fa_rate`). Plan checkpoint C1 locates and reuses
  that exact function; if unavailable, define impulsivity = fraction of trials with a
  `trialoutcome=='fa'` anticipatory lick, computed once and cached.
- **Circularity guard:** the axis uses **raw** behavioural rates only. Behavioural *state
  labels* are computed from lick/outcome features (`f_inapplick`, `f_hit_hard`, …) — hence
  ~relabels of FA rate, mechanically coupled to the very lick our preparatory measure aligns
  to — and are NEVER the axis; they enter only as overlay/robustness (§5.4).
  [[state_labeler_circularity_caveat]]

**4.5 Per-cell preparatory scalars** (from each cell's z-trace, reusing parent windows /
`preparatory.active_mask`):
- **magnitude scalar** = mean z in a pre-lick window (default `[-0.3, 0] s`).
- **timing scalar** = the cell's pre-lick onset (`preparatory.cell_onset`) OR pre-lick
  center-of-mass of active(t); noisy per cell, so it enters only through the mixed model /
  bin medians, never a single-cell claim.
Both are per-cell, so the primary stat (E2) aggregates them with a cell-level mixed model
rather than a per-session median — necessary because per-session **sustained** N is only
~1–4 cells (§9), too few for a stable per-session sustained median.

**4.6 Behavioural state tags (role-a overlay/robustness).** Per-**trial** state labels exist
for every session (`data/cache/state_tags/<subj>/<date>.csv`; `trial_idx`, `state_label` ∈
StimSens / Impulsive / Disengaged / Abort; BG_046 46 / BG_039 30 / BG_031 42 sessions).
Two derived quantities are used: (i) per-session **state occupancy** = fraction of trials in
each state (cache-free); (ii) per-lick **trial state**, joinable to licks by `trial_idx`, for
the optional trial-level contrast (§5.4-S3). ⚠️ `state_label` is derived from lick/outcome
features, so it is mechanically coupled to the lick alignment — states serve overlay/
robustness, never the primary axis.

---

## 5. Estimators (the design)

**E1 — Binned "see-it" curve (population, faithful machinery).**
Pool ALL cell-sessions; bin by the session behavioural coordinate (primary: anticipatory-FA
**tertiles**; also d′ tertiles), pooled across animals so each bin has enough neurons. Per
bin, run the parent's faithful population-onset + active-fraction-CI machinery
(`bootstrap_fraction_ci`, `population_onset`) for sustained and non-TF (and TF-resp).
Output: onset and peak-active-fraction, and the **sustained−nonTF lead**, as a function of
regulation bin. This is the visual "does the lead grow toward the well-regulated corner."
Bootstrap CIs over neurons within bin. (Tertiles chosen so the sparse sustained group has
workable N/bin; bin count is a documented, not tuned, choice.)

**E2 — Cell-level mixed model (drift-immune, the primary stat).**
One row per cell-session (sustained + non-TF), columns = per-cell scalar (§4.5), `group`
∈ {sustained, nonTF}, the session `impulsivity`/`d′`, `session`, `subject`. Fit
`scalar ~ group * impulsivity + group * d' + (1 | subject) + (1 | session)`
(statsmodels mixedlm; session RE nested in subject). **The `group:impulsivity` (and
`group:d'`) interaction IS the effect of interest**: "does the sustained-vs-non-TF lead
scale with regulation." The session random intercept absorbs per-session baseline/drift, and
because both groups share each session's coordinate, the contrast is within-session and
drift-immune — this is what replaces tracking. Report interaction estimate, CI, p, plus
per-animal fits. A **per-session-median lead** version (`lead ~ impulsivity + d′ +
(1|subject)`) is a secondary robustness run **only on the well-powered TF-resp-pooled group**
(≥ N_min=5 per session), never on the sparse sustained group.

**E3 — Behaviour-matched cross-animal test.**
Partition standardized (impulsivity, d′) space into coordinate cells; keep cells containing
sessions from ≥2 animals. Within each matched cell, compute the **pooled** sustained−nonTF
lead (E1-style, pooling that cell's cell-sessions across animals — NOT a per-session
sustained median) and test whether it differs by animal/region:
`scalar ~ group * animal` within matched cells, and a matched-cell-level
`lead ~ (1 | matched_cell) + animal`. If behaviour sets geometry, the animal/region term is
~0. Drift- and region-confound-resistant, doable now with 3 animals.

### 5.4 Behavioural states — overlay & robustness (role a)

- **S1 (overlay, cache-free).** Plot per-session state occupancy (%StimSens / %Impulsive /
  %Disengaged) against the regulation coordinate, and colour the E1/E2 session points by
  dominant state. Makes the behaviour↔state tie explicit (well-regulated ⇒ more StimSens).
  Use `config.STATE_LABEL_COLORS`. [[reference_state_label_colors]]
- **S2 (robustness, cache-free — the key state use).** Re-fit the E2 model adding session
  state-occupancy fractions as covariates; the `group:regulation` interaction (Q1/Q2) must
  **survive**. If it vanishes, the regulation effect was just a state-mixture proxy.
- **S3 (optional, heavier — bridge to future).** Trial-level state-conditioned preparatory
  recompute: rebuild per-cell z-traces from spikes using only **StimSens** licks (and
  separately Impulsive), then compare the sustained-vs-non-TF lead across states. Requires a
  state-split recompute (local ProcessPool, like `build_prep_cache.py`) and further thins
  per-cell trial counts; Disengaged excluded (near-zero hits). Flagged, not required for the
  first pass. ⚠️ Only the *within-session, cross-state* contrast on **FA-aligned** activity
  carries a mechanical-coupling caveat (Impulsive trials ≡ FA trials) — there, use hit-aligned
  or lick-independent signatures. The *across-session, state-fixed* trajectory (which is NOT
  circular) is §5.5.

**Sign convention:** lead > 0 ⇒ sustained leads/exceeds non-TF (magnitude: larger ramp;
timing: earlier onset ⇒ define as nonTF_onset − sustained_onset).

### 5.5 Phase 2 (extension) — across-session, state-conditioned signature trajectories

**Goal.** Fix behavioural **state** (segment each session into single-state blocks), then ask how a
neural **signature** evolves across the recording timeline / regulation axis — pooling state-blocks
across sessions into learning-ordered bins for power (per-session where dense enough; session
identity retained).

**Why it is NOT circular (scope correction).** Conditioning on a state and asking whether neural
activity changes *across sessions within that fixed state* is not circular: the state is a constant
segmentation criterion, never inferred from the neural data and never the outcome. The one case with
genuine mechanical coupling is a *within-session, cross-state* contrast on a **lick-aligned** measure
(e.g. FA-aligned preparatory activity Impulsive-vs-StimSens — Impulsive trials *are* the FA trials).
Fixing the state across the comparison, or using a lick-independent signature, removes it. **Impulsive-
state trajectories are in scope.**

**Signatures (not just the preparatory lick-lead).** The evolving quantity may be:
- **Sensory TF-pulse responses** — fast and/or slow pulse-evoked responses (reuse `tf_pulse` / GLM-TF);
  lick-independent, cleanest against state.
- **Coding directions** — sensory / choice-outcome / motor CDs (or new ones): track strength & geometry
  within a fixed state across sessions.
- The **sustained-vs-non-TF preparatory lead** (parent signature), state-conditioned.

**Drift discipline (unchanged, load-bearing).** State-conditioning does **not** remove composition
drift — the probe still samples different cells across weeks (BG_046 89%→15% broad). So the tracked
quantity must be drift-robust: a **within-session relative/between-group contrast** (e.g. sustained−nonTF)
or a **normalized population-geometry** measure (angles, not raw magnitudes), never an absolute
per-session level. Per-animal breakdown + regulation-shuffle null required.

**State × lick pairing.** For across-session trajectories (state fixed) both StimSens+hit and Impulsive+FA
are usable; the mechanical-coupling caveat applies only to within-session cross-state lick-aligned
contrasts. Disengaged stays too sparse in hits.

**Cost / status.** Needs the §5.4-S3 state-split recompute from spikes; cross-session binning supplies
power. Not cache-only. **Second pass** — the §5 first pass ships independently.

**Related in-house work (accurate characterisation).** A separate, **in-development** method registers
**probe physical locations / channels** — binning activity along the probe rather than tracking
individual cells — to compare population activity across sessions without cell tracking. It is **not
finished** and **not purpose-built** for state-conditioned trajectories; a state-conditioned application
could eventually host this, but verify its status before relying on it — do not assume it fits.

---

## 6. Hardening battery (mandatory — project rule)

- **H1 Regulation-shuffle null.** Permute the session→coordinate mapping (≥500 shuffles,
  seed 42); the E1 slope / E2 fixed-effect must collapse toward flat. [[feedback_circular_analysis_null_controls]]
- **H2 Per-animal.** Report E1/E2 slopes per animal (BG_046 real trajectory; BG_031 spans
  axis = key within-animal test; BG_039 = high-regulation anchor). No pooled claim without
  the per-animal breakdown.
- **H3 Magnitude vs timing.** Every headline reported for BOTH the absolute-threshold
  (magnitude) and amplitude-normalized (timing) lead. Divergence is a result, not a failure.
- **H4 Hit vs FA lick.** Run on both alignments; a regulation effect specific to the
  decision (hit) vs impulsive (fa) lick is interpretively important.
- **H5 Pseudoreplication.** subject random effect in E2/E3; per-session sign test as a
  non-parametric backstop; disclose within-subject repeated-neuron inflation (untracked).
- **Inherited caveats (carry verbatim):** magnitude≈timing confound; FSI-vs-mixed cell-type
  on the non-TF rung; VMS = 1 mouse; "preparatory" peaks peri-lick (broad-kernel sensory
  responses bleed pre-lick via RT). The decisive movement-regressed re-derivation remains
  open (future; the regulation regression could later be run on movement-regressed leads).
- **Checkpoint C1 (behavioural axis).** Before any neural stat: reconcile the impulsivity
  definition with `popgeom_fa_cutoff` so both lines share ONE axis; verify per-session
  coordinates reproduce `improvement_vectors.csv` at the animal-mean level.

---

## 7. Deliverables

- **Code:** `scripts/tf_responsiveness/preparatory_fig5/regulation_axis.py` (E1–E3 + H1–H5),
  reusing `visdetect.analysis.preparatory` primitives + `prep_common`. New pure helpers, if
  any (e.g. behaviour-matching), go in `preparatory.py` with unit tests (TDD, project rule).
- **Behavioural-coordinate cache:** `data/cache/preparatory_fig5/session_regulation.csv`
  (subject, session, impulsivity, d′, stage) — small, `git add -f`'d.
- **Figures** (`FIGURES/preparatory_fig5/regulation/{pooled,DMS,VMS}/`): E1 lead-vs-regulation
  curve; E2 within-session-lead scatter + fit (stage as colour overlay only); E3
  behaviour-matched cross-animal panel; per-animal slope panel; magnitude-vs-timing pair.
- **State overlay/robustness (§5.4):** `.../regulation/states/` — occupancy-vs-regulation
  figure (S1) + state-covariate robustness table (S2); S3 artifacts only if run.
- **Stats CSVs** next to each figure; **null** results under `.../regulation/hardening/`.
- **Write-up:** `docs/science/2026-07-21-preparatory-vs-regulation.md` (honest, caveated,
  null-friendly). Update memory `prep_activity_transient_sustained_jul2026` with the outcome.

---

## 8. Out of scope (explicit)

- Tracked-unit within-neuron learning trajectories (needs curation / better tracker).
- Transient-class per-bin split (N too small).
- Movement-regressed / change-aligned re-derivation (parent's open control; future).
- **Single-state-conditioned within-neuron *tracking* across learning (future).** Restricting
  to one state before comparing the **same tracked neurons** across learning would remove the
  state-mixture confound and make tracking apples-to-apples — but needs the tracking-curation
  work first. The *population-level*, tracking-free version of state-conditioned across-session
  comparison is now scoped as **Phase 2 (§5.5)** — second-pass, not first-pass.
- Full state×learning factorial designs (state-null already; circularity; sparse Disengaged
  hits) — parked.
- Any compute over the X: (Samba) drive; any non-Opus/Fable subagent (project rules).

---

## 9. Feasibility / power summary

- **Well-powered:** non-TF (thousands/mouse); TF-resp-pooled (BG_046 63L/99E, BG_031
  138L/181E); the E1 tertile bins pooled across animals (~55 sustained/bin).
- **Sparse per session → drives the E2 design choice:** per-session **sustained** N ≈ 1–4
  (BG_046 45 sustained / 32 sessions ≈ 1.4; BG_031 110 / 29 ≈ 3.8). This is why E2 is a
  **cell-level** mixed model (all sustained cells contribute; session RE handles the within-
  session contrast) rather than a per-session sustained median, and why per-session-median
  runs are restricted to TF-resp-pooled.
- **Workable but thin:** sustained in the low-regulation (Learning-like) region for BG_046
  (n=13 total). Mitigated by pooling across animals in E1 and the cell-level model in E2.
- **Not powered → excluded:** transient per bin; any BG_039 low-d′ bin.
- **Realistic outcome distribution:** a modest positive regulation slope carried mainly by
  BG_046/BG_031, OR a clean null. Both are acceptable, hardened deliverables.
