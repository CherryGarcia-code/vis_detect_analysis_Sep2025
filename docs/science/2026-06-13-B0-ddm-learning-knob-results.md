# B0 — Which DDM knob does learning turn? (first real-data result)

**Date:** 2026-06-13 · **Subject:** BG_046 · **Status:** in-progress (identifiability-aware)
**Spec:** `docs/superpowers/specs/2026-06-10-B0-ddm-learning-knob-design.md`
**Code:** `src/visdetect/analysis/ddm.py` · `analysis_suite/01_behavior/h_ddm_learning_knob.py`
**Figure/stats:** `analysis_suite/figures/01_behavior/fig0N_ddm_learning_knob.png`,
`…/ddm_learning_stats.csv`, `analysis_suite/cache/ddm_per_stage_fits.csv`

## Headline

Across the Naive→Expert trajectory (merged **Learning** vs **Expert**), the parameter
that moves most is the **drift gain `v` (sensitivity)**: it roughly doubles,
**Δv = +0.65 (Learning 0.56 → Expert 1.21)** — consistent with **H1** (Marica:
striatal/sensory responsiveness rises with learning). The bound `a` also rises
(2.61 → 3.32, more caution) and the starting-point ratio `z` falls slightly
(0.40 → 0.33, marginally less anticipatory bias).

Nested model comparison (AIC, lower = better, 2 stages):

| model | free across stages | AIC |
|-------|--------------------|-----|
| M_shared | none | 3406.3 |
| **M_v** | v | **3372.8** |
| M_a | a | 3401.4 |
| M_zu | z,u | 3393.0 |
| M_full | v,a,z,u | 3370.4 |

`M_full` is the nominal winner, but it beats `M_v` by only **2.4 AIC** while costing 3
extra parameters; letting **`v` alone** vary recovers essentially all of the
improvement over `M_shared` (Δ33 AIC). **Read-out: learning primarily turns the drift/
sensitivity knob, with a secondary caution (bound) increase.** The impulsivity/urgency
route `u` is negligible (≈0 in both stages).

## Secondary results

- **Route attribution (Step 0b):** the two-route model does **not** beat a TF-only
  model (CV-LL two-route −315.8 vs TF-only −304.8; `two_route_wins = False`). Once a
  starting-point `z` is in the model, a separate time/urgency route adds nothing here —
  contrasting the earlier κ≈0.02 "impulsivity-dominated FA" prior (held loosely per
  spec §4). Early licks are captured by drift + start-point, not a dedicated urgency drive.
- **State-resolved route mixture (spec §5 secondary):** TF-route share of early licks was
  **engaged = 0.00, impulsive = 0.82** — i.e. **reversed** from the predicted
  *engaged > impulsive*. On these GLM-HMM labels, impulsive-state early licks track the TF
  stream more than engaged-state ones. Treat as exploratory: the GLM-HMM state↔route
  mapping is evidently not the simple engaged=evidence-driven story (caveat below).

## Why "in-progress", not "done" (identifiability caveats — spec §6/§8)

The change-detection task has **long, variable baselines** (decision_time median ~7 s,
p99 ~13.6 s). The Baseline_ON-aligned accumulator therefore runs over a long horizon
(`T_dur = 11 s`, `dt = 0.05 s`), a weakly-identified regime:

1. The bound `a` sits high (2.6–3.3) and needed a **widened Fittable range (→8.0)** to
   avoid being pinned; `u` collapses to ≈0. The `v`/`a`/`z` trade-off is only weakly
   constrained without error-RTs (detection task, spec §6).
2. **Structure selection (Step 0) was skipped** for this run (`RUN_STRUCTURE = False`);
   the lit-predicted `R = halfwave`, `urgency = rising` were used. Re-run with
   `RUN_STRUCTURE = True` to confirm the rectification form.
3. **Parameter recovery was validated only on the short-baseline synthetic regime**
   (panel E, clean), **not** on the real long-baseline configuration. A recovery check at
   `T_dur = 11`, `dt = 0.05` with the real per-trial TF streams is the key follow-up.
4. No **bootstrap CIs** on the stage parameters yet (panel C shows point estimates), and
   the **state-composition control of the knob comparison** (spec §6 — refit the stage
   comparison conditioned on / matched for HMM state) is **not** done; only the secondary
   route-mixture used state.
5. Fits use a **bounded+seeded DE** and **trial-count caps** (≤250/stage) for tractability.

## Next steps to close out

- Long-baseline parameter recovery (validate `v`/`a`/`z` are recoverable at `T_dur=11`).
- Bootstrap CIs on `v,a,z,u` and on Δv, Δa, Δz (session-level resampling).
- State-controlled knob comparison (spec §6); reconcile the reversed state route-mixture.
- Turn on `RUN_STRUCTURE` to lock the rectification form; cross-check against the
  lick-hazard GLM (spec §7).
