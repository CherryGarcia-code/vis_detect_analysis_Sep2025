# Analysis Documentation: Sequential Dynamics & Distributed TF Encoding

**Date**: April 5, 2026
**Analyses**: Fig14b (Sequence Significance) + Fig35b (TF Encoding Distribution)

---

## 1. TF Encoding Distribution Analysis (Fig35b)

### Scientific Question
Is TF encoding strength across the striatal population distributed continuously (log-normal/Gamma) or bimodally (responsive vs non-responsive two-class model)?

### Motivation
- Prior classification used |z_abs_max| >= 3.0 threshold to label neurons as "TF-responsive" (~8.5% of population)
- Manual review during GUI labeling suggested many sub-threshold neurons show weak but above-chance encoding
- Buzsaki & Mizuseki (2014, Nat Neurosci) showed neural response strengths follow log-normal distributions, predicting distributed rather than concentrated coding

### Methods

#### Data Source
- Pre-computed TF screening cache: `analysis_suite/cache/tf_responsiveness.csv` (4725 units, 25 sessions)
- Per-session NPZ trace files: `data/cache/tf_traces/BG_046/` (fast/slow z-scored PSTHs per unit)

#### Detrending
- Linear baseline detrending applied to each unit's pulse-triggered trace
- Fits a line to baseline period (-400 to -10 ms pre-pulse), subtracts extrapolated trend from full trace
- Rationale: slow baseline drift from task engagement or drifting grating fluctuations can mask or inflate pulse responses
- Analysis run with BOTH detrended (primary) and standard (secondary) z-scores

#### Distribution Fitting
Five candidate models compared by AIC/BIC (maximum likelihood):

| Model | Parameters | Rationale |
|-------|-----------|-----------|
| Half-normal | sigma | Proper null for |z| values (z ~ N(0,1) => |z| is half-normal if no encoding) |
| Log-normal | mu, sigma | Buzsaki hypothesis: neural response strengths are log-normally distributed |
| Gamma | alpha, beta | Flexible positive distribution, common for neural data |
| Exponential | lambda | Simplest heavy-tailed model |
| 2-component Gaussian mixture | pi, mu1, sigma1, mu2, sigma2 | Two-class model (responsive vs non-responsive) |

Model comparison criterion: ΔAIC/ΔBIC > 10 = strong evidence.

#### Cumulative Information Curve
- All 4725 units ranked by |z_abs_max| (strongest first)
- Cumulative sum of |z_abs_max| plotted vs fraction of neurons
- Plot on both linear and log(N) x-axes (Buzsaki prediction: linear on log scale)
- Break-even point: where bottom (N-k) neurons collectively match top 10%

#### Per-Session Discriminability
- For each of 25 sessions: cumulative discriminability (|mean_fast_z - mean_slow_z| in response window) as function of neurons included
- Averaged across sessions per learning stage with SEM

### Normalization Decisions
1. **z-scores computed per-unit relative to pre-pulse baseline (standard)**: Pre-pulse window (-0.4, 0) s, each unit independently normalized. This is the established pipeline standard.
2. **Detrended z-scores (primary)**: Additional linear detrending of baseline to remove slow drift. This inflates z-scores (91.4% vs 42.9% responsive) but may better reflect true encoding.
3. **No across-session normalization needed**: z-scores are already per-unit, per-session scale-free measures.

### Results
- **Best-fit model**: Gamma (AIC best), Log-normal close (ΔAIC=18.8)
- **Two-class model strongly rejected**: ΔAIC=113.5 vs Gamma
- **Noise-only null overwhelmingly rejected**: ΔAIC=5752
- **Break-even**: Bottom 24% of neurons collectively match top 10%
- **Buzsaki log(N) linearity**: r=0.91 (strong support for distributed coding)
- **No stage effect**: Distribution shape identical across Learning vs Expert (KW: small effect)
- **Detrended vs standard correlation**: rho=0.047 (weak), indicating detrending reveals different information than raw z

### Statistical Considerations
- AIC/BIC comparison avoids multiple testing issues (model selection, not null hypothesis testing)
- Anderson-Darling used as secondary goodness-of-fit test (more tail-sensitive than KS)
- Per-session CDF overlay confirms distribution shape consistency across recording sessions
- Q-Q plot (Panel H) shows slight heavy-tailed deviation from log-normal in upper quantiles (consistent with Gamma being preferred)

---

## 2. Sequence Significance Analysis (Fig14b)

### Scientific Question
Does the sequential activation pattern in the population heatmap (Fig14A) represent genuine temporal tiling, or is it driven by (a) the argmax sorting artifact, (b) variable reaction-time jitter?

### Motivation
- Fig14 shows ~600 neurons with sequential peaks tiling 0-250ms post-change on Hit trials
- The same sort order applied to Miss trials does NOT show this diagonal (good sign)
- However, sorting by argmax ALWAYS produces a diagonal — this is a known artifact
- Furthermore, variable Hit RTs (200-500ms) could create an apparent sequence if neurons are actually lick-locked

### Methods

#### Test 1: Split-Half Peak-Order Stability
- Split Hit trials randomly into two halves
- Compute argmax (peak time in response window) for each unit on each half independently
- Measure Spearman ρ between the two peak orders
- Null model: Circular-shift each unit's PSTH independently (500 permutations) — preserves temporal autocorrelation but destroys coordinated ordering
- Significance: permutation p-value = (n_null >= observed + 1) / (N_perm + 1)

#### Test 2: Cross-Validated Time Decoding
- At each time bin in the response window, predict elapsed time from the population vector
- Ridge regression with 5-fold cross-validation at the TRIAL level
- Critical: CV must split trials, not time bins, to avoid within-trial autocorrelation leakage
- Metric: R² on held-out trials (variance of elapsed time explained)
- Null: **Circular time-shift permutation** (200 permutations) — for each trial independently, circularly shift the time axis by a random amount. Preserves per-neuron autocorrelation and firing rate but destroys coordinated population timing.
- Control: Same analysis on Miss trials (should show lower R² if sequence is outcome-specific)

**Bug fix (April 6, 2026)**: Original implementation used trial-identity shuffle, which is a no-op for time decoding (every trial gets identically tiled time labels, so permuting trial order doesn't break the time-to-activity mapping). Corrected to circular time-shift null.

#### Test 3: RT-Controlled Time Decoding
- Bin Hit trials by reaction time: 150-250ms, 250-350ms, 350-500ms, 500-800ms
- Within each RT bin, compute time-decoding R²
- If the diagonal is RT-driven, within-bin R² should collapse ~0
- If genuine sequence, R² persists within RT bins

#### Test 4: Lick-Aligned Comparison
- Re-align all activity to lick time (Change_ON + RT) instead of Change_ON
- Compute time-decoding R² on lick-aligned data
- If R²_lick >> R²_change: sequence is motor/lick-locked
- If R²_change >> R²_lick: sequence is stimulus-locked

### Normalization Decisions
1. **Z-score to shared baseline (-0.5, -0.05 s)**: Equalizes units with different baseline firing rates. Critical for time decoding — otherwise high-FR units dominate the Ridge regression coefficients.
2. **StandardScaler within CV folds**: Additional within-fold standardization for the Ridge regression (fit on training, transform test).
3. **25ms bins**: Standard bin size for sequence analysis. Finer binning (10ms) gives more time points but increases noise per bin.

### Key Statistical Decisions
1. **Trial-level CV, not time-bin-level**: Avoids inflated R² from within-trial temporal autocorrelation.
2. **Ridge (not Lasso/OLS)**: Regularization handles collinearity between nearby time bins. RidgeCV selects optimal alpha from [0.1, 1, 10, 100].
3. **Permutation null (not parametric)**: Circular time-shift null gives conservative, assumption-free null distribution that preserves per-neuron autocorrelation.
4. **Circular-shift null for both split-half and time decoding**: Preserves per-unit temporal autocorrelation (critical — a simple shuffle would destroy autocorrelation and make ANY stable signal look significant).

### Multiple Comparisons
- Tests 1-4 address different aspects of ONE hypothesis (is the sequence real?). No FDR correction across tests — they form a logical chain.
- RT-bin results reported descriptively (question: does the sequence persist in ALL bins, not which bin is significant).
- Per-session p-values from permutation tests NOT corrected across sessions — each session is an independent replicate.

---

## 3. Key Insights for Next Steps

### TF Distribution Implications
1. The z=3.0 threshold is arbitrary — the encoding continuum extends well below it
2. CD_sensory in the 2D decomposition should use full population, not just "responsive" neurons
3. The encoding substrate (distribution shape) is stable across learning — what changes is readout, not representation

### Sequence Analysis Results (Initial Run — Broken Null)

The first run (April 5, 2026) used a trial-permutation null for time decoding that was effectively a no-op. Key observations from other metrics:

- **Split-half peak stability**: Highly significant (W=0.0, p<10⁻⁷, r=0.87). Expert ρ=0.51, Learning ρ=0.26.
- **Hit vs Miss R²**: Massive asymmetry (W=1.0, p<10⁻⁷, median diff=0.32). Whatever signal exists is outcome-specific.
- **RT-bin R²**: Persists within narrow RT bins, even increases for 350-500ms. Argues against pure RT-jitter artifact.
- **Lick vs stimulus alignment**: Lick-aligned R² >> change-aligned R² (median diff=0.35, p<10⁻⁷). Population is more lick-locked than stimulus-locked.
- **Stage effect**: Expert R² >> Learning R² (U=23, p=0.003, r=0.70).

### Sequence Analysis Results (Corrected Null — Pending)

Re-run with circular time-shift null in progress (April 6, 2026). Expected to show significant R² if the coordinated sequential structure is genuine.

### Sequence Analysis Implications
- The **lick-locked dominance** changes the interpretation: rather than a "temporal basis set" for sensory processing, this is more consistent with **sequential motor preparation**.
- Neurons have stable preferred timing (high split-half ρ), but this timing is relative to the impending lick, not stimulus onset.
- The learning effect (Expert > Learning) could reflect motor sequences becoming more stereotyped and reliable.
- The sequential vs sustained subpopulation split (top ~600 vs bottom ~2100 in Fig14) should be characterized against TF responsiveness, lick responsiveness, and cell type.

### Connection to Lohse AND-Gate Framework
- The distributed encoding finding suggests the "sensory dimension" in the AND-gate is not carried by specialist neurons but by the collective activity of the entire population
- This may explain why the AND-gate geometry is present from early learning (Panel G: identical distributions) — the encoding substrate already exists, but the cortical readout (task-state dimension) learns to gate it
- The lick-locked sequential dynamics may represent the **motor output dimension** — orthogonal to both CD_task and CD_sensory
