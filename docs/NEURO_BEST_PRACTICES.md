# Neuroscience Best Practices

These standards apply to all analyses and reflect current best practices in systems neuroscience and decision-making research.

---

## Spike Data Analysis

- **Never average raw firing rates across units without normalization.** Use z-scoring (baseline-subtracted, divided by baseline SD) or baseline-subtracted rates to compare across units with different firing rates.
- **PSTH smoothing**: Gaussian kernel with σ=25 ms is standard for striatal neurons. Do not over-smooth (σ>50 ms obscures temporal dynamics).
- **Permutation tests** (500–1000 shuffles) for single-unit significance. Do not rely solely on parametric tests for neural data.
- **FDR correction** (Benjamini-Hochberg, α=0.05) for mass screening across units. Do NOT apply FDR across separate scientific questions/figures.
- **Report effect sizes** alongside every p-value. Neural data with large n can yield tiny p-values for biologically meaningless effects.

---

## Population Analysis

- **Trial-match conditions** when comparing population responses (e.g., Hit vs Miss). Unequal trial counts bias variance estimates. Subsample the larger group or report both matched and unmatched results.
- **Cross-validation** for all decoders. Never test on training data. Use stratified k-fold (k=5) to maintain class balance.
- **Null distributions** for decoding: label-shuffle permutation (≥200 shuffles), report chance as mean ± 2 SD of null.
- **Coding direction (CD) vectors**: Compute on training folds, project on held-out data to avoid circularity.

---

## Signal Detection Theory

- d′ = z(hit_rate) − z(fa_rate), with log-linear correction clipping rates to [0.01, 0.99]
- Hit rate computed on **go trials only**. FA rate on **catch trials only**.
- Always report criterion c alongside d′ when assessing response bias.

---

## Statistical Standards

- **Non-parametric by default** for neural data (Kruskal-Wallis, Mann-Whitney U, Spearman ρ). Neural distributions are almost never Gaussian.
- **Two-sided tests** unless the biological hypothesis is strongly directional.
- **Bootstrap CI** (1000 resamples, seed=42, percentile method) for key estimates.
- See the **Research Statistician** skill for the full decision framework and effect size formulas.

---

## Normalization

→ Full guide (decision tree, methods, pitfalls, code examples, quick reference): `docs/NORMALIZATION.md`

**Golden rule**: Normalize each unit separately using a **shared baseline** across all conditions being compared. Normalize-then-average, never average-then-normalize.
