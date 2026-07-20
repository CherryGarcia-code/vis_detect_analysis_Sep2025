# Fig-5 e-h preparatory-by-cell-class — 6-lens adversarial refutation (Opus 4.8)

Six independent skeptics re-derived from `prep_hit.npz` (each spot-checked the
cache, several rebuilt PETHs from pkls) and tried to REFUTE the two claims.

- CLAIM 1 = per-class ordering: sustained recruited earliest < transient < non-TF.
- CLAIM 2 = onset scales with kernel width (wider = earlier).

| Lens | Claim 1 | Claim 2 | Strongest point |
|---|---|---|---|
| firing-rate / yield / unequal-n | SURVIVES | SURVIVES | base_hz→onset ρ=−0.05 ns; rate-matching non-TF closes only ~0.05 s of the 0.35 s gap; subsample non-TF to n=99 → −0.363 (never earlier than sustained). **Caveat:** BG_046 responsive cells are 127/133 FSI → non-TF rung is partly FSI-vs-mixed. |
| movement / lick / RT leakage | SURVIVES | **PARTIAL** | Ordering established at every bin from −0.838 s (0.33/0.13/0.05 at −0.5 s), n_licks matched. **But** per-cell width→onset collapses when restricted to strictly pre-lick (DMS onset<−0.2 ρ=−0.16 p=0.09 ns); broad-kernel cells' long sensory responses bleed into the pre-lick window via RT → partly guaranteed by construction. |
| baseline-σ / circularity | SURVIVES | SURVIVES | Rebuilt PETHs match cache exactly; ordering survives SEM/fixed-1Hz/Poisson σ; divisor is *largest* for sustained (works against it). width→onset mediated by response gain (peak_Hz ρ=0.34), not algebraically circular. **Caveat:** onset conflates magnitude with latency — amplitude-normalized timing does NOT preserve the ordering. |
| onset-metric / smoothing | **PARTIAL** | SURVIVES | Ordering survives all threshold/smoothing knobs. **But** absolute mean_frac>0.1 threshold means a bigger ramp crosses earlier for amplitude reasons; under peak-relative normalization VMS sustained≈transient (the robust core is TF-classes<non-TF everywhere + full 3-way in DMS/pooled). |
| pseudoreplication / single-mouse | SURVIVES | **PARTIAL** | Ordering replicates independently in all 3 mice. **But** leave-one-mouse-out: dropping BG_046 makes pooled per-cell width→onset n.s. (ρ=−0.067, p=0.22); VMS is a lone mouse; untracked cross-session units inflate per-cell N. |
| statistics / multiple-comparisons | SURVIVES | **PARTIAL** | Label-shuffle nulls survive all 3 regions under Bonferroni. **But** of 8 region×lick×metric combos only HIT-DMS per-cell (ρ=−0.41, p=4e-9) survives FDR; the n=10 decile r=−0.66 FAILS FDR (p_adj=0.18), sits at the shuffle noise floor (95th |r|=0.63 vs 0.64), and is apples-to-oranges vs the paper's across-area −0.55. Absent on FA, null in VMS. |

## Bottom line
- **CLAIM 1 (ordering): robust but reframed** — it is an ordering of preparatory-
  response MAGNITUDE/reliability (sustained carry ~23 vs ~6 Hz pre-lick ramps),
  not a proven latency ordering; robust to σ-convention, rate-matching, nulls, and
  replicates across mice; carries an FSI-vs-mixed confound on the non-TF rung.
- **CLAIM 2 (width→onset): weak** — the ONLY defensible statistic is the per-cell
  DMS Spearman ρ=−0.41 (weaker than the paper's −0.55, not stronger); carried by a
  single mouse, collapses strictly pre-lick, absent on FA, null in VMS. Do NOT
  headline the decile r=−0.66 or claim it reproduces/exceeds the paper.
- **Decisive control still open:** change-aligned (not lick-aligned) or
  video-movement-regressed re-derivation, to separate sensory-response persistence
  from decision preparation. The lick-time-shuffle (ramp is lick-locked) is done.
