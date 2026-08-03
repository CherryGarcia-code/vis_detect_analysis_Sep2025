# S1 — Session grouping and a non-circular learning axis (design)

**Date:** 2026-07-31
**Status:** Design (brainstormed + approved; Phase 1 tooling BUILT and smoke-tested)
**ID:** S1 (new — methodological infrastructure serving the spine, not a spine question itself)

---

## 1. The problem

Every stage label in this project currently comes from d′: sessions are **gated** at `d′ ≥ 0.8`,
stage boundaries are d′ thresholds (Naive→Learning at >1.0, →Expert at >1.5, one-way, 3-of-4
window), and d′ is *also* the performance readout. So **the learning axis is built out of the
outcome**. Two consequences, both observed:

1. **"Naive" is not naive.** BG_046's three Naive sessions have d′ 0.98–1.45. The genuinely naive
   early sessions were QC-excluded (12 `Excluded` sessions for BG_046, 10 BG_039, 11 BG_031).
2. **Any behavioural DV correlated with d′ is partly definitional**, so a clean progression cannot
   be drawn — the axis is contaminated by exactly the sessions it is built from.

Underneath this is the **competence vs performance** distinction: what the mouse *knows*
(monotone, only increases) vs what it *did that day* (fluctuates with engagement/satiety/health).
Session-level gating conflates them, and it is the wrong granularity anyway — engagement drifts
*within* a session, so no session is wholly "good" or "bad".

---

## 2. What was tested and REFUTED (do not retry)

An earlier design proposed anchoring the axis on **"expert sessions with long sustained StimSens
runs (plus enough Impulsive to contrast)"** and growing outward. A feasibility pass over all
tagged sessions killed it. Recorded here so it is not re-proposed:

| Check | Result |
|---|---|
| Is sustained StimSens an *expert* signature? | **No.** BG_046 max StimSens run, median by stage: Naive **35** [8–43], Learning 32 [12–73], Expert **40** [12–83]. 2/3 Naive sessions already pass "run ≥ 30". BG_039's Naive `02042025` has an **81-trial run at 62% occupancy**. |
| Does bout length add over occupancy? | **No.** ρ(max_run, occupancy)=+0.815; partial(max_run, session_idx \| occ) = −0.158 (p=0.30, n.s.) while partial(occ, session_idx \| max_run) = **+0.514** (p=0.0003). The axis collapses to occupancy — reproducing the prior finding that occupancy is the sufficient regulation statistic. |
| Does the endpoint rule isolate Expert? | **No.** BG_046: 11 Expert **+ 7 Learning + 2 Naive** pass. BG_031: dominated by Learning (6) over Expert (4). |
| Does it replicate across mice? | **No.** BG_046 occupancy ρ=+0.624; BG_039 flat (ρ=−0.07 n.s.); BG_031 **decreasing** (max_run ρ=−0.465, p=0.002). |

**Scientific content of the refutation (worth keeping):** the mouse **already has the good state in
Naive** and can sustain it; what learning changes is **how much of the session it spends there**
(StimSens occupancy 0.14 → 0.40). Learning shifts state *mixing*, not the state repertoire —
consistent with the GLM-HMM learning literature. Hence there is no qualitatively new capability to
anchor an endpoint on. Also: median StimSens run is only ~3–4.5 trials in *every* stage, so
"sustained bouts" are rare tail events, and **max-run is an extremum statistic that scales with
`n_trials`** — normalise it if ever used.

**And the sting:** occupancy — the one statistic that does track learning — is derived from
`f_hit_hard`/`f_miss_easy`/`f_inapplick`, so a StimSens-occupancy axis is near-circular with the
d′ staging it was meant to replace. It hides the circularity one layer down rather than removing it.

---

## 3. Revised design

Three separated roles, so no quantity does double duty:

| Role | Quantity | Why |
|---|---|---|
| **Learning axis** | **Exogenous**: training time (session index / cumulative trials / days). Optionally a latent monotone competence curve (state-space or isotonic) fitted *to* it. | Independent of every behavioural DV **by construction** — the only non-circular option available. |
| **Eligibility filter** | The user's rule: enough StimSens **and** enough Impulsive, in runs long enough to analyse. Applied **uniformly at every stage**, not as a stage definition. | A *coverage/contrast* criterion, not a performance one. Selects on the **structure** of behaviour, not the mean of the DV. |
| **Contamination covariate** | State **occupancy** (the sufficient statistic). | Enters the model as a covariate, never as the axis: `DV ~ training_time + occupancy + (1|session)`. |

This is the competence/performance decomposition the exercise wanted, **without a selection step**:
training time carries competence, occupancy carries the day's state, and the eligibility rule only
decides which sessions can support a within-session state contrast at all.

**The refutation is what makes the eligibility rule attractive:** because it passes 20/45 BG_046
sessions spanning Naive, Learning *and* Expert, it yields comparable sessions **at every point on
the learning axis** — better for state-conditioned analysis than an expert-only endpoint would be.

### Session inclusion (explicit)
- **Drop the `d′ ≥ 0.8` gate** for anything learning-axis related; recover the `Excluded` sessions.
- Keep DV-independent gates only: `n_trials ≥ N`, data-integrity QC (sync, clock drift).
- **Never** gate on `n_fa` when the DV is FA-based — that selects on the DV's own numerator.

---

## 4. Sub-project: manual session sorter (Phase 1, BUILT)

Learn the grouping rule from the expert eye rather than inventing one, mirroring the proven
trial-level pipeline (hand-labelled episodes → shallow tree → LOSO κ=0.731 → tag everything).

**Taxonomy (agreed):** `Balanced`, `Impulsive-dominated`, `Disengaged-dominated`, `Deteriorating`,
`Low-yield`, plus `Unsure`. `Deteriorating` is deliberate: it is a **dynamics** category that
rate-based features cannot express, so if the labels reliably separate it, the manual pass added
information rather than recapitulating d′.

**Two design requirements, both non-negotiable:**

1. **Blinding.** Sessions are presented in fixed random order with stage, date, session id and d′
   **hidden**. Otherwise labels inherit the staging being replaced and "groups track learning"
   becomes a tautology. Blinded, it is a real testable finding.
2. **Test–retest.** ~15% of sessions are silently shown twice, giving the labeller's own
   self-consistency κ. This is the **ceiling**: no fitted rule can be more reliable than its labels.
   If it is low, fix the taxonomy before fitting anything.

### ⚠ Circularity scope (must travel with any result)
Groups are learned **from behaviour**, therefore:
- **Clean** for NEURAL DVs → "do striatal dynamics differ between Impulsive and Balanced sessions?" is legitimate.
- **Circular** for BEHAVIOURAL DVs → "Impulsive-dominated sessions have more early licks" is definitional, not a finding.

---

## 5. Deliverables

| Artefact | Path |
|---|---|
| Sorter GUI | `scripts/session_sorting/run_session_sorter.py` ✅ built + smoke-tested |
| Rule fitter | `scripts/session_sorting/fit_session_group_rule.py` ✅ built + smoke-tested |
| Manual labels | `data/cache/session_sorting/manual_session_labels.csv` |
| Presentation queue | `data/cache/session_sorting/<SUBJ>_presentation_queue.csv` (45 sessions + 7 repeats = 52) |
| Learned rule + κ | `data/cache/session_sorting/session_group_rule.txt`, `FIGURES/session_sorting/<SUBJ>/session_group_rule.png` |

---

## 6. Open questions

1. **Latent competence model** — state-space (Smith & Brown binomial random-walk) vs simple
   isotonic envelope? Decide after seeing whether the manual groups already separate bad days.
2. **Per-mouse vs global anchors** — inclination is **per-mouse rules with a shared rule form**;
   BG_031 (VMS impulsive non-learner) legitimately has a different endpoint, and that is the
   negative control, not a nuisance to normalise away.
3. **Rule-based labels vs `hmm_state`** — all feasibility numbers used the rule-based
   `state_label` (a tree on composition features, circular by construction). A GLM-HMM fit might
   behave differently; the one check that could change §2's conclusion.
4. **Truly independent engagement** — no behaviour-only feature set is independent of both DVs
   simultaneously. Video/pupil/movement from the camera pipeline is the clean answer; this is a
   concrete scientific argument for finishing sub-project A.

---

## 7. Related

- Feature-definition audit: `state_calibration.py:30-55`, `state_labeling.py:21-40`
  (`f_miss_easy` conditions on `is_go` **and** `change_size ≥ 2.0` → it is an error rate;
  `f_nolick` pools go-miss with catch-CR → correctness-agnostic).
- Memory: `state_labeler_circularity_caveat`, `within_session_state_dynamics_jul2026`,
  `early_lick_learning_bg046_jul2026`, `feedback_circular_analysis_null_controls`.
