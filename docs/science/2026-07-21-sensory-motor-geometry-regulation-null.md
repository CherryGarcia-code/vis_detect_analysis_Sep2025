# Sensory–motor population geometry vs behavioural regulation: a controlled near-null (with a VMS hint to revisit)

**One-line honest summary.** The tempting result — that the angle θ between a striatal
**sensory axis** (β_TF, per-unit signed GLM temporal-frequency kernel) and an **impulsive-motor
axis** (β_FA, false-alarm pre-lick ramp) *orthogonalises as behavioural regulation improves* — is a
**two-confound artifact**. After a physical-matching confound battery, **no significant geometry
effect survives**: DMS is null, and only a **weak, non-significant single-animal VMS hint**
(BG_031, ρ ≈ −0.25, ns) remains — a hypothesis to carry to the growing VMS cohort, not a result.

**Status.** Prototype campaign on the CHEAP axes (cached GLM kernels + cached FA ramps — **no
full-population GLM refit**), 2026-07-21. Branch `worktree-theta-prototype`
(`0fbe308`). This is a deliberately-cheap first pass whose job was to decide whether the
expensive dense-β_TF refit is worth doing; its verdict is **not yet**.

---

## 1. Question & axes

Behavioural spine: mice learn to *suppress impulsivity while boosting sensitivity*. Geometric
restatement: do the population's **sensory** and **impulsive-lick** directions **align** when the
animal is impulsive and **orthogonalise** when it is well-regulated? Organising variable is
**behavioural regulation** (per-session false-alarm rate; the behavioural `fa` = early anticipatory
lick, NOT the SDT false alarm), not training stage (§ chosen because the 3 mice do not share a
monotonic learning trajectory; see `subject_usability` memory).

- **β_TF (sensory)** — per-unit **signed** peak of the cached GLM TF-kernel
  (`kernel_vectors_*.npz`), responsive cells only, non-responsive = 0. Sparse: **~5 (BG_046) to
  ~11 (BG_031) non-zero cells/session**. *(This sparsity is the source of confound #2.)*
- **β_FA (motor)** — FA pre-lick ramp, z-scored Δrate `(−0.3,−0.15) − (−1.75,−1.25) s` re. the FA
  lick (canonical `EVENT_RESPONSIVENESS_WINDOWS['FA']`), complete-case FAs (latency ≥ 1.75 s).
- **θ** = `arccos |β̂_TF · β̂_FA|` per session; tested vs FA rate, Spearman, session-level, **per
  animal** (no cross-animal pooling).

Mice: **BG_046 (DMS), BG_039 (DMS), BG_031 (VMS)**; `good_dates` sessions (qc_fail = False AND
< 50 % Disengaged); 73 sessions with usable spikes.

---

## 2. Result — the confound battery

### Raw prototype (lumped β_FA from `prep_fa.npz`, sparse β_TF) — **looks like the hypothesis**
θ decreases with impulsivity in all three mice: ρ = **−0.30** (BG_046), **−0.66** (BG_039, n = 6),
**−0.54** (BG_031, p < 0.01, n = 28). Consistent sign, 3/3.

### Confound #1 — β_FA reliability (FA count). **Handled by physical count-matching.**
FA rate is ~collinear with FA-lick count (ρ = **+0.89 / +0.66 / +0.79**), and θ tracks FA count
directly (ρ = **−0.44 / −1.00 / −0.43**) — the attenuation signature (more FAs → cleaner β_FA →
smaller θ). Recomputing β_FA from spikes (complete-case ≥ 1.75 s, z-normalised) and **subsampling
every session to a fixed K FA-trials** (K-sweep 40/60/91, robust):
- **BG_046 (DMS): ρ ≈ 0 (null).** The prototype's −0.30 was **early-FA contamination** in the
  lumped `prep_fa` (which did not enforce complete-case), *not* geometry.
- **BG_031 (VMS): ρ ≈ −0.48 to −0.51 (p = 0.01)** — survives reliability-matching.
- BG_039: untestable (< 6 sessions clear K).

*Methods note:* the partial-correlation control (θ ~ FA-rate | FA-count) **over-controlled** —
FA-count and FA-rate are collinear (ρ ≈ 0.8), so partialling the proxy removes most of the signal.
The **physical count-match is the trustworthy control**, and it disagreed.

### Confound #2 — β_TF support (responsive-cell count). **The killer.**
Responsive-cell count covaries with impulsivity (BG_031: support ~ FA-rate ρ = **+0.62**) and θ
tracks support (ρ = **−0.49**). This is an **artifact of the sparse responsive-only β_TF** (a
variable-membership axis). Matching **both** confounds — fixed K = 60 FA-trials **and** fixed
S responsive cells (S-sweep 3/4/5):
- **BG_046 (DMS): ρ = −0.07 to −0.18 (ns).**
- **BG_031 (VMS): ρ ≈ −0.28 (ns, p ≈ 0.16), consistent across S.**
- This **converges with the full-power partial** (θ ~ FA | support = −0.24), so the residual is
  **not a power artifact** of the thin support-match — it is a genuinely weak, non-significant
  effect.

| stage | BG_046 (DMS) | BG_031 (VMS) |
|---|---|---|
| raw prototype | −0.30 | −0.54 |
| + complete-case β_FA, count-matched | **~0 (null)** | −0.48 (p 0.01) |
| + β_TF support matched (both) | ~0 | **−0.28 (ns)** |

---

## 3. Verdict & interpretation

**No significant sensory–motor geometry effect survives the confound battery.** The naive effect
was ~half β_FA-reliability confound and ~half β_TF-support confound, both of which covary with
impulsivity. What remains is a **weak (ρ ≈ −0.25), non-significant, single-animal VMS trend**
(BG_031) in the predicted direction — notable only because BG_031 is both the one VMS probe and the
impulsive non-learner with the most impulsivity variance to detect an effect, but **n = 1 VMS
affords no inference**.

**Why the dense refit is deprioritised.** The support confound is specifically an artifact of the
*sparse responsive-only* β_TF; a full-population **dense** β_TF (the refit) would make support
≈ constant and is the only clean, full-power fix. But its expected value is **modest**: DMS is
null even in the sparse test, and the confound-free VMS residual is already ns. The VMS hint is a
**single-animal** phenomenon — best adjudicated by **more VMS animals** (histology-pending
BG_012/040/041, once localised) than by a refit on n = 1.

---

## 4. Caveats (mandatory framing)
- **Cheap axes.** β_TF is the **responsive-only** signed kernel from an **8-day-stale cache**
  (`kernel_vectors_*.npz`, Jul-08), not a fresh full-population fit; β_FA is a lumped-over-all-FAs
  ramp (not split at the per-subject impulsive/anticipatory τ). Both limit power, not just the
  confound reading.
- **Single VMS animal.** The only surviving signal is in BG_031; region (VMS) and animal are
  confounded at n = 1.
- **Drift / untracked units.** Cross-session comparison mixes learning-state with chronic-probe
  drift; θ is between per-session axes of changing membership (part of confound #2).
- **State-label independence.** FA rate (the axis) is behavioural; θ is neural, so the
  behaviour↔neural test is not circular — but the FA windows use the raw event time (no −200 ms
  hardware correction, per the decision to await video-derived lick onset).
- What the data **can** say: the cheap-axis geometry effect is confound-dominated. What it
  **cannot** say: whether a clean, dense, well-powered β_TF would reveal a real VMS effect.

---

## 5. Deliverables & reproduction
- `scripts/popgeom_theta/theta_prototype.py` — prototype + partial-correlation control.
- `scripts/popgeom_theta/theta_count_matched.py` — β_FA count-match + K-sweep + support-confound check.
- `scripts/popgeom_theta/theta_support_matched.py` — both-confounds-matched + S-sweep.
- Figures + per-session CSVs under `FIGURES/popgeom_theta/`.
- Inputs (read-only from the primary tree): `prep_fa.npz`, `kernel_vectors_*.npz`, staging manifests.

**Related:** audit + FA-cutoff + subject-usability in memory `popgeom_sensory_motor_audit_jul2026`;
the sensory-vs-task-state geometry precedent (`state_tf_encoding_population_geometry_jul2026`);
mandatory null-control rule (`feedback_circular_analysis_null_controls`).
