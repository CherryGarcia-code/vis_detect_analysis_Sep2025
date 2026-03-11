# Figure Explanations — Visual Change-Detection Analysis Suite

**Subject:** BG_046 (mouse), medial striatum Neuropixels 2.0 chronic recordings  
**Task:** Visual change-detection (orientation grating changes at varied contrast)  
**Sessions:** 38 QC-passed sessions spanning Naive → Learning → Expert stages  
**Cell types:** Narrow-spiking (putative FSI) vs Broad-spiking (putative MSN/Proj)  
**HMM states:** Disengaged, Engaged, Impulsive (K=3, pre-computed per trial)

---

## Module 01 — Behavior

### Fig 01: Learning Curve (`01_behavior/a_learning_curve.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** d′ trajectory across sessions with stage-colored background bands. **B:** Hit rate and FA rate across sessions (dual y-axis). **C:** Psychometric curves (hit rate vs change size) per learning stage. **D:** Reaction time distributions by trial outcome. |
| **Data source** | Trial-level behavioral outcomes from all 38 sessions. |
| **d′ computation** | Hit rate = fraction correct on go trials (change_size > 1.0 Hz); FA rate = fraction responding on catch trials (change_size ≤ 1.01 Hz). Rates clipped to [0.01, 0.99] before z-transform: d′ = z(HR) − z(FAR). |
| **Staging** | Naive (sessions 1–6), Learning (7–25), Expert (26–38). From `staging_manifest.csv`. |
| **Statistics** | Spearman ρ for d′ vs session index; Kruskal-Wallis for d′ across stages; Spearman for FA rate vs session; Kruskal-Wallis for RT across stages. |
| **Output** | `figures/01_behavior/fig01_learning_curve.png`, `learning_curve_stats.csv` |

---

### Fig 02: HMM State Dynamics (`01_behavior/b_hmm_state_dynamics.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Stacked area plot of HMM state fractions (Disengaged / Engaged / Impulsive) across sessions. **B:** Per-state d′ trajectories across sessions. **C:** State transition matrices for each learning stage. **D:** Per-state psychometric curves (Expert sessions only). |
| **Data source** | Pre-computed HMM state assignments (`hmm_k3_assignments.csv`) mapped onto trials. |
| **HMM states** | Disengaged (low performance), Engaged (good performance), Impulsive (high FA rate). Renamed from original Engaged_1 / Engaged_2 / Biased. |
| **Statistics** | Spearman ρ for each state fraction vs session; χ² contingency test (state × stage); Kruskal-Wallis for d′ across stages within each state. |
| **Output** | `figures/01_behavior/fig02_hmm_state_dynamics.png`, `hmm_state_stats.csv` |

---

### Fig 03: Reaction Times (`01_behavior/c_reaction_time_analysis.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Hit RT distributions by learning stage (violin plots). **B:** Hit RT vs change size (speed-accuracy tradeoff, per stage). **C:** FA and abort RT distributions by learning stage. **D:** Hit RT by HMM state (Expert sessions). **E:** Median Hit RT trajectory across sessions. **F:** FA RT (from baseline) trajectory across sessions. |
| **RT definitions** | **Hit:** `reactiontimes["RT"]` — time from Change_ON to first lick. **FA:** `reactiontimes["FA"]` — time from Baseline_ON to first lick. **abort:** `reactiontimes["abort"]` — time from Baseline_ON to first lick (very early). **Miss:** `reactiontimes["Miss"]` = 2.155 s constant (response window limit, NOT a real RT — excluded from analysis). |
| **Key parameters** | `FA_RT_SPLIT` from config separates FA from abort RTs. |
| **Statistics** | Kruskal-Wallis for RT across stages; Spearman ρ for median RT vs session; Mann-Whitney U for Hit RT vs FA RT in Expert sessions. |
| **Output** | `figures/01_behavior/fig03_reaction_times.png`, `reaction_time_stats.csv` |

---

## Module 02 — Single Unit

### Fig 04: Responsiveness Screen (`02_single_unit/a_responsiveness_screen.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Volcano plot (response d′ vs −log₁₀ p-value) colored by responsiveness. **B:** Fraction responsive units by learning stage (bar chart). **C:** Population PSTH heatmap sorted by peak latency (Expert sessions). **D:** Distribution of response magnitudes (delta firing rate) by cell type. |
| **Method** | Permutation test (500 shuffles) comparing mean FR in response window (0–250 ms post Change_ON) vs baseline (−400 to −50 ms). A unit is "responsive" if p < 0.05. Tests pooled Hit + Miss trials. |
| **Key parameters** | Event: Change_ON. Baseline: (−0.4, −0.05 s). Response: (0.0, 0.25 s). Bin: 10 ms. Min trials: 5. n_perm: 500. Min firing rate: 1.0 Hz (good cluster filter). |
| **Statistics** | Kruskal-Wallis for fraction responsive across stages; Spearman ρ for fraction responsive vs session index. |
| **Caching** | Saves/loads `cache/responsiveness_all_sessions.csv` to avoid re-computation. |
| **Output** | `figures/02_single_unit/fig04_responsiveness_screen.png`, `responsiveness_stats.csv` |

---

### Fig 05: Outcome Selectivity (`02_single_unit/b_outcome_selectivity.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** auROC distribution (histogram) with significance threshold. **B:** Fraction of selective units per learning stage. **C:** Selectivity heatmap (Expert sessions, units sorted by auROC). **D:** Mean PSTH for top Hit-preferring vs top Miss-preferring units. |
| **Method** | Area under ROC curve (auROC) comparing single-trial FR in response window for Hit vs Miss trials. auROC > 0.5 = Hit-preferring. Significance assessed by permutation test (500 shuffles, FDR-corrected). |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 25 ms. Response: (0.0, 0.25 s). Baseline: (−0.4, −0.05 s). Min trials per class: 5. n_perm: 500. |
| **Statistics** | Wilcoxon signed-rank vs 0.5 (population-level selectivity); Kruskal-Wallis for auROC across stages; Spearman ρ for mean auROC and fraction selective vs session. |
| **Caching** | Saves/loads `cache/selectivity_all_sessions.csv`. |
| **Output** | `figures/02_single_unit/fig05_outcome_selectivity.png`, `outcome_selectivity_stats.csv` |

---

### Fig 06: Change-Size Tuning (`02_single_unit/c_change_size_tuning.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Population-average tuning curve (FR vs change size) by learning stage. **B:** Distribution of tuning strength (Spearman ρ of FR vs change magnitude). **C:** Fraction significantly tuned per stage. **D:** Mean tuning slope (ρ) vs session index. |
| **Method** | Per-unit Spearman correlation of mean response FR across change sizes. Significance: Kruskal-Wallis across change sizes, FDR-corrected. |
| **Key parameters** | Window: (−0.5, 0.5 s). Bin: 25 ms. Response: (0.0, 0.25 s). Baseline: (−0.4, −0.05 s). Min trials per size: 3. |
| **Statistics** | Wilcoxon signed-rank for ρ vs 0 (population-level tuning); Spearman ρ for mean ρ vs session index. |
| **Caching** | Saves/loads `cache/tuning_all_sessions.csv`. |
| **Output** | `figures/02_single_unit/fig06_change_size_tuning.png`, `change_size_tuning_stats.csv` |

---

### Fig 07: State Modulation (`02_single_unit/d_state_modulation.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Population PSTH by HMM state (Expert sessions, Change_ON aligned). **B:** Per-unit modulation index distribution (Engaged vs Disengaged delta FR). **C:** Modulation index by cell type (FSI vs MSN). **D:** State modulation across learning stages. |
| **Method** | Modulation index = (FR_Engaged − FR_Disengaged) / (\|FR_Engaged\| + \|FR_Disengaged\| + ε). Computed from mean FR in response window. |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 25 ms. Baseline: (−0.4, −0.05 s). Response: (0.0, 0.25 s). Min trials per state: 5. |
| **Statistics** | Wilcoxon signed-rank for MI vs 0; Mann-Whitney U for MI FSI vs MSN; Kruskal-Wallis for MI across stages. |
| **Caching** | Saves/loads `cache/state_modulation.csv`. |
| **Output** | `figures/02_single_unit/fig07_state_modulation.png`, `state_modulation_stats.csv` |

---

### Fig 08: Cell-Type Comparison (`02_single_unit/e_cell_type_comparison.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Mean z-scored PSTH by cell type (Expert sessions, Change_ON aligned). **B:** Overall firing rate distributions (FSI vs MSN). **C:** Response magnitude (delta FR) by cell type × learning stage. **D:** Outcome selectivity (auROC) by cell type. |
| **Method** | Merges waveform cell-type labels (`waveform_celltypes.csv`) with neural responses. Peak-trough time < threshold → Narrow (FSI); ≥ threshold → Broad (MSN/Proj). |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 25 ms. Baseline: (−0.4, −0.05 s). Response: (0.0, 0.25 s). |
| **Statistics** | Mann-Whitney U for FSI vs MSN firing rate, delta FR, and selectivity; Mann-Whitney for Expert vs Naive within each cell type. |
| **Output** | `figures/02_single_unit/fig08_celltype_comparison.png`, `celltype_comparison_stats.csv` |

---

## Module 03 — Population

### Fig 09: Coding Direction (`03_population/a_coding_direction.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** CD projection for a single Expert session (Hit vs Miss mean time courses). **B:** Grand-average CD across Expert sessions. **C:** CD effect size vs session index (emergence across learning). **D:** CD projection in Engaged vs Impulsive HMM states (Expert sessions). |
| **Method** | Coding direction (CD) = normalized difference of population-mean Hit vs Miss activity. Population activity projected onto CD axis at each time bin. Cross-validated: condition means from one split, tested on held-out split (5-fold). |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 25 ms. Min units: 10. Min trials per class: 8. N_splits: 5. N_perm: 200 (significance). |
| **Statistics** | Spearman ρ for CD effect vs session; Kruskal-Wallis for CD by stage; Mann-Whitney for Expert vs Naive. |
| **Caching** | Saves per-session results to `cache/cd_results/`. |
| **Output** | `figures/03_population/fig09_coding_direction.png`, `coding_direction_stats.csv` |

---

### Fig 10: Population PSTH Heatmap (`03_population/b_population_psth_heatmap.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Z-scored population heatmap aligned to Change_ON (Hit trials, Expert sessions), sorted by peak latency. **B:** Same unit ordering, Miss trials. **C:** Hit − Miss difference heatmap. **D:** Population average PSTH ± SEM for Hit vs Miss. |
| **Method** | Concatenates all Expert-session units. Z-scores each unit against its baseline. Sorts by peak latency (argmax in 0–1 s). Fine binning (10 ms) for smooth heatmaps. |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 10 ms. Baseline: (−0.5, −0.05 s). Expert sessions only. Min units: 5. |
| **Statistics** | Descriptive: peak latency distribution, Hit-preferring fraction, peak population activity difference. |
| **Output** | `figures/03_population/fig10_population_heatmap.png`, `population_heatmap_stats.csv` |

---

### Fig 11: PCA Dimensionality (`03_population/c_dimensionality_reduction.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** PC1 vs PC2 state-space trajectories for Hit vs Miss (one Expert session). **B:** Scree plot (variance explained by top PCs) averaged per learning stage. **C:** Effective dimensionality across sessions. **D:** PC1 temporal profile by outcome (Expert grand-average). |
| **Method** | PCA on concatenated condition-mean PSTHs (Hit and Miss). Effective dimensionality = participation ratio: (Σλ)² / Σλ². Computed on full trial-by-trial z-scored population activity. |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 25 ms. Baseline: (−0.5, −0.05 s). Min units: 10. Min trials per class: 5. n_components: min(10, n_units). |
| **Statistics** | Spearman ρ for effective dimensionality vs session index. |
| **Output** | `figures/03_population/fig11_dimensionality.png`, `dimensionality_stats.csv` |

---

## Module 04 — Decoding

### Fig 12: Hit/Miss Decoding (`04_decoding/a_hit_miss_decoding.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Time-resolved decoding accuracy for one Expert session (with permutation chance band). **B:** Grand-average decoding accuracy across Expert sessions. **C:** Decoding onset latency vs session index. **D:** Peak decoding accuracy by learning stage (boxplot). |
| **Method** | Logistic Regression (L2, C=1.0) with stratified 5-fold CV at each time bin. Features = population firing rates (z-scored). Chance estimated from 20 label-shuffled permutations. Onset = first time bin significantly above chance. |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 50 ms. Min units: 10. Min trials per class: 8. N_folds: 5. N_perm: 20. |
| **Statistics** | Spearman ρ for peak accuracy vs session; Kruskal-Wallis for peak accuracy across stages. |
| **Output** | `figures/04_decoding/fig12_hit_miss_decoding.png`, `hit_miss_decoding_stats.csv` |

---

### Fig 13: Change-Size Decoding (`04_decoding/b_change_size_decoding.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Time-resolved decoding accuracy (Big vs Small change, Expert grand-average). **B:** Peak decoding accuracy across sessions. **C:** Decoding accuracy by learning stage (boxplot). **D:** Confusion matrix (Expert sessions, pooled). |
| **Method** | Binary classification: Big ({2.0, 4.0 Hz}) vs Small ({1.25, 1.35, 1.5 Hz}). Logistic Regression with stratified 5-fold CV per time bin. |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 50 ms. Min units: 5. Min trials per class: 8. N_folds: 5. |
| **Statistics** | Spearman ρ for peak accuracy vs session; confusion matrix analysis. |
| **Output** | `figures/04_decoding/fig13_change_size_decoding.png`, `change_size_decoding_stats.csv` |

---

### Fig 14: State Decoding (`04_decoding/c_state_decoding.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Decoding accuracy across sessions (scatter + stage background). **B:** Accuracy by learning stage (boxplot). **C:** Confusion matrix (Expert sessions, pooled across folds). **D:** Feature importance (mean \|LR coefficients\| histogram). |
| **Method** | Multinomial Logistic Regression decoding HMM state (3 classes) from **pre-trial** mean firing rates in (−1.5, −0.5 s) before Change_ON. Stratified 5-fold CV. |
| **Interpretation** | Tests whether pre-trial population activity predicts upcoming behavioral state — i.e., whether internal state is represented in striatal neural activity before the trial begins. |
| **Key parameters** | Tensor window: (−1.5, 0.0 s). Feature extraction: mean FR in (−1.5, −0.5 s). Min units: 5. Min trials per state: 5. Chance: 1/3. |
| **Statistics** | Wilcoxon signed-rank vs chance (1/3); Spearman ρ for accuracy vs session; Kruskal-Wallis across stages. |
| **Output** | `figures/04_decoding/fig14_state_decoding.png`, `state_decoding_stats.csv` |

---

## Module 05 — Longitudinal

### Fig 15: Neural Learning Curves (`05_longitudinal/a_neural_learning_curves.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Per-session mean FR change (response − baseline) vs session index with learning stage bands. **B:** Fraction responsive vs d′ (neural-behavioral correlation). **C:** Population response magnitude by stage (boxplot). **D:** Neural metric summary table across stages. |
| **Method** | Tracks population-averaged neural response magnitude (z-scored delta FR) across sessions. Correlates with behavioral d′ to test neural-behavioral coupling. |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 25 ms. Baseline: (−0.4, −0.05 s). Response: (0.0, 0.25 s). |
| **Statistics** | Spearman ρ for neural metric vs session and vs d′; Kruskal-Wallis across stages. |
| **Output** | `figures/05_longitudinal/fig15_neural_learning.png`, `neural_learning_stats.csv` |

---

### Fig 16: Cell-Type Learning (`05_longitudinal/b_celltype_learning.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Mean response magnitude (delta FR) across sessions, separate lines for FSI and MSN. **B:** Fraction responsive by stage and cell type. **C:** Response latency by cell type across stages. **D:** Cell-type ratio (FSI/MSN) of responsive neurons across sessions. |
| **Method** | Tracks cell-type-specific neural metrics across learning. Tests whether FSIs and MSNs show different learning trajectories. |
| **Key parameters** | Window: (−0.5, 0.5 s). Bin: 25 ms. Baseline: (−0.5, −0.05 s). Response: (0.0, 0.25 s). Z-score responsiveness threshold: 2.0. |
| **Statistics** | Spearman ρ for cell-type metrics vs session; Mann-Whitney U for FSI vs MSN. |
| **Output** | `figures/05_longitudinal/fig16_celltype_learning.png`, `celltype_learning_stats.csv` |

---

### Fig 17: Population Geometry Shift (`05_longitudinal/c_population_geometry_shift.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** CD magnitude (Hit vs Miss separation) across sessions. **B:** CD angle stability (cosine similarity between consecutive sessions). **C:** CD angle relative to first Expert session. **D:** Variance along vs orthogonal to CD across stages. |
| **Method** | Tracks the coding direction (CD) vector across sessions. Cosine similarity quantifies stability of the population discrimination axis. Variance decomposition along CD vs orthogonal quantifies signal vs noise power. |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 25 ms. Baseline: (−0.5, −0.05 s). Response: (0.0, 0.25 s). Min units: 10. Min trials per class: 5. |
| **Statistics** | Spearman ρ for CD magnitude, angle stability, and variance ratio vs session. |
| **Output** | `figures/05_longitudinal/fig17_geometry_shift.png`, `geometry_shift_stats.csv` |

---

## Module 06 — Lick/Motor

### Fig 18: FA Neural Signatures (`06_lick_motor/a_fa_neural_signatures.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Population mean PSTH for FA trials (aligned to FA lick) vs Miss trials (aligned to Change_ON), Expert sessions. **B:** Per-unit pre-event activity difference heatmap (FA − Miss). **C:** Pre-FA ramp magnitude across sessions. **D:** Pre-FA ramp split by HMM behavioral state. |
| **Method** | FA trials aligned to the FA lick time; Miss trials aligned to Change_ON. Both z-scored against distant baseline (−2.0, −1.5 s). Compares pre-event (−0.5, 0.0 s) activity between FA and Miss — tests whether neural "ramp-up" precedes FA licks. |
| **Key parameters** | FA/Miss window: (−2.0, 0.5 s). Bin: 25 ms. Baseline: (−2.0, −1.5 s). Early: (−1.5, −1.0 s). Late: (−0.5, 0.0 s). Min trials: 5. Min units: 3. |
| **Statistics** | Wilcoxon signed-rank for pre-FA vs pre-Miss activity; Spearman ρ for ramp magnitude vs session; Kruskal-Wallis for ramp by HMM state. |
| **Output** | `figures/06_lick_motor/fig18_fa_neural_signatures.png`, `fa_neural_signatures_stats.csv` |

---

### Fig 19: Pre-Lick Ramping (`06_lick_motor/b_pre_lick_ramping.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Example ramping unit PSTHs (top 3 units by ramp ρ, FA-aligned). **B:** Distribution of ramp ρ values across all units. **C:** Fraction of ramping units by cell type (FSI vs MSN). **D:** Fraction of ramping units by learning stage. |
| **Method** | For each unit, computes Spearman ρ between FA-aligned mean PSTH and time in the ramp window (−1.0, 0.0 s). A unit is classified as "ramping" if ρ > 0 and p < 0.05. |
| **Key parameters** | FA window: (−2.0, 0.5 s). Bin: 25 ms. Ramp window: (−1.0, 0.0 s). Min FA trials: 5. Ramp p threshold: 0.05. |
| **Statistics** | Spearman ρ (ramp detection per unit); χ² contingency for ramp fraction FSI vs MSN; Kruskal-Wallis for ramp fraction across stages. |
| **Output** | `figures/06_lick_motor/fig19_pre_lick_ramping.png`, `pre_lick_ramping_stats.csv` |

---

### Fig 20: Motor vs Sensory Dissociation (`06_lick_motor/c_motor_vs_sensory.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Population PSTH for Hit (Change_ON aligned) vs FA (FA-aligned), Expert sessions. **B:** Scatter of Hit response vs FA response per unit. **C:** Sensory Index distribution colored by cell type. **D:** Sensory Index by learning stage (boxplot). |
| **Method** | Sensory Index (SI) = (Hit_resp − FA_resp) / (\|Hit_resp\| + \|FA_resp\| + ε). SI > 0 → sensory-driven (responds to change but not FA lick). SI < 0 → motor/lick-driven (responds to both equally or prefers FA). |
| **Key parameters** | Windows: (−0.5, 1.0 s). Bin: 25 ms. Baseline: (−0.4, −0.05 s). Response: (0.0, 0.25 s). Min trials: 5. ε = 1e-6. |
| **Statistics** | Wilcoxon signed-rank for SI vs 0; Mann-Whitney U for SI FSI vs MSN; Kruskal-Wallis for SI across stages. |
| **Output** | `figures/06_lick_motor/fig20_motor_vs_sensory.png`, `motor_vs_sensory_stats.csv` |

---

## Module 07 — Advanced

### Fig 21: GLM Encoding (`07_advanced/a_glm_encoding.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Distribution of total deviance explained (DE) across units. **B:** Partial deviance explained (PDE) by predictor (stacked bars). **C:** PDE by predictor across learning stages. **D:** PDE by predictor for FSI vs MSN. |
| **Method** | Poisson GLM per unit with predictors: stimulus (change_size), choice (hit/miss indicator), HMM behavioral state, and pre-event baseline FR. Response variable: spike count in (0, 0.25 s). Partial deviance explained (PDE) = reduction in deviance when a predictor is added to the null model. |
| **Key parameters** | Response: (0.0, 0.25 s). Baseline: (−0.4, −0.05 s). Min trials: 20. Optimizer: L-BFGS-B. Max iterations: 200. |
| **Statistics** | Spearman ρ for PDE vs session; Mann-Whitney U for PDE FSI vs MSN; deviance comparison (full vs null). |
| **Output** | `figures/07_advanced/fig21_glm_encoding.png`, `glm_encoding_stats.csv` |

---

### Fig 22: Demixed PCA (`07_advanced/b_dpca.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** Variance explained by each marginalization (time, decision, stimulus, interaction). **B:** Top dPC for stimulus marginalization (time course). **C:** Top dPC for decision marginalization (time course). **D:** Variance partition pie charts by learning stage. |
| **Method** | Demixed PCA (Kobak et al., 2016 eLife) decomposes population activity into components driven by different task variables. Conditions: 2 outcomes (Hit/Miss) × 2 stimulus groups (Big/Small). Marginalizations: time, decision (hit/miss), stimulus (big/small), and interaction (residual). |
| **Key parameters** | Window: (−0.5, 1.0 s). Bin: 25 ms. Baseline: (−0.5, −0.05 s). Min units: 10. Min trials per condition: 5. Big: {2.0, 4.0 Hz}. Small: {1.25, 1.35, 1.5 Hz}. |
| **Statistics** | Spearman ρ for variance fractions vs session. |
| **Output** | `figures/07_advanced/fig22_dpca.png`, `dpca_stats.csv` |

---

### Fig 23: Noise Correlations (`07_advanced/c_noise_correlations.py`)

| Item | Detail |
|------|--------|
| **Panels** | **A:** r_noise distribution by learning stage. **B:** Mean r_noise across sessions (learning trajectory). **C:** r_noise by HMM state (Expert sessions). **D:** r_noise by cell-type pair (FSI-FSI, FSI-MSN, MSN-MSN). |
| **Method** | Noise correlation = Pearson r of trial-by-trial residuals between simultaneously recorded neuron pairs. Residuals computed by subtracting condition means (outcome-specific) from single-trial responses. Computed in post-change window (0, 0.5 s). |
| **Interpretation** | r_noise reflects shared trial-to-trial variability. Changes across learning may indicate shifts in network coupling or shared modulatory inputs. |
| **Key parameters** | Window: (0.0, 0.5 s). Bin: 25 ms. Min units: 5. Min trials: 10. Max pairs per session: 500 (random subsampling for efficiency). |
| **Statistics** | Spearman ρ for mean r_noise vs session; Mann-Whitney U for r_noise by cell-type pair and HMM state. |
| **Output** | `figures/07_advanced/fig23_noise_correlations.png`, `noise_correlation_stats.csv` |

---

---

# Module 08: TF Pulse Analysis

Inspired by Khilkevich & Lohse (2024, Nature) — brain-wide dynamics linking
sensation to action.  TF pulses during the baseline period provide
motor-confound-free sensory responses.  These analyses examine how medial
striatal neurons encode TF pulses and how this encoding interacts with
learning, behavioral state, and motor planning.

---

## Fig 24 – TF Pulse Responsiveness Screening

| | |
|------------|------|
| **Script** | `08_tf_pulse/a_tf_responsiveness.py` |
| **Question** | How prevalent are TF-responsive neurons in medial striatum? |
| **Panels** | **A:** Example TF-responsive unit PSTHs (fast and slow pulse-aligned z-scored traces with SEM). **B:** Heatmap of all TF-responsive units' fast pulse responses sorted by peak time. **C:** Z-score distribution across all units with responsive fraction pie chart. **D:** Responsiveness broken down by learning stage and cell type (FSI vs MSN). |
| **Method** | Align spikes to fast/slow TF pulses during baseline, compute Gaussian-smoothed (σ=17ms) firing rates, z-score relative to pre-pulse baseline (−0.4 to 0 s). A unit is "TF-responsive" if any post-pulse |z| ≥ 3.0. |
| **Key parameters** | Fast threshold: log₂(TF) ≥ 0.25. Slow threshold: log₂(TF) ≤ −0.25. Z-threshold: 3.0. Pre-window: (−0.4, 0) s. Post-window: (0, 0.5) s. Constraints: min 1s after baseline start, min 1s before change, min 2s before FA/abort. |
| **Output** | `figures/08_tf_pulse/fig24_tf_responsiveness.png`, `tf_responsiveness.csv` |

---

## Fig 25 – TF Response Properties

| | |
|------------|------|
| **Script** | `08_tf_pulse/b_tf_response_properties.py` |
| **Question** | What are the temporal properties of TF pulse responses in striatum — fast/transient (relay) or slow/sustained (integration)? |
| **Panels** | **A:** Population mean fast TF pulse response ± SEM across all TF-responsive units, with peak annotated. **B:** Peak latency distribution with cell-type comparison (FSI vs MSN, Mann-Whitney U). **C:** Response half-width (duration) distribution with cell-type comparison. **D:** Latency vs half-width scatter — positive correlation suggests integrative responses. |
| **Method** | For each TF-responsive unit: peak latency = time of max |z| in post-pulse window; half-width = duration above 50% of peak |z|. Pearson correlation between latency and half-width tests integration hypothesis. |
| **Key parameters** | Same as fig24 (shared TFRespPulseConfig). |
| **Statistics** | Mann-Whitney U for cell-type comparisons, Pearson r for latency-width correlation. |
| **Output** | `figures/08_tf_pulse/fig25_tf_response_properties.png`, `tf_response_properties.csv` |

---

## Fig 26 – Two-Pulse TF Integration

| | |
|------------|------|
| **Script** | `08_tf_pulse/c_tf_pulse_integration.py` |
| **Question** | Do striatal neurons integrate sequential TF pulses? (Khilkevich & Lohse 2024 Fig 2e-f: ~250 ms integration timescale in forebrain.) |
| **Panels** | **A:** Facilitation index at each inter-pulse interval (IPI). **B:** Facilitation curve with exponential decay fit → integration timescale τ. **C:** Facilitation by learning stage (box plots). **D:** Facilitation by cell type (FSI vs MSN). |
| **Method** | Classify fast pulses as isolated (>500ms gap before) or paired. For each IPI bin (50-500 ms), align to the second pulse and measure peak |z|. Facilitation index = (peak_double − peak_single) / peak_single. Fit exponential decay a·exp(−IPI/τ) + c to estimate integration timescale. |
| **Key parameters** | IPI bins: 50, 100, 150, 200, 300, 400, 500 ms (±15 ms tolerance). Min isolated pulses: 5. Min paired pulses per bin: 3. |
| **Statistics** | Exponential curve fit (scipy.optimize.curve_fit), Mann-Whitney U for cell-type comparison. |
| **Output** | `figures/08_tf_pulse/fig26_tf_two_pulse_integration.png`, `tf_two_pulse_integration.csv` |

---

## Fig 27 – TF Encoding Emergence Across Learning

| | |
|------------|------|
| **Script** | `08_tf_pulse/d_tf_learning_emergence.py` |
| **Question** | Does TF pulse encoding in striatum emerge with learning? (Paper: trained mice develop widespread sensory encoding absent in untrained.) |
| **Panels** | **A:** Fraction TF-responsive by learning stage (bar chart, χ² test). **B:** Response amplitude (peak |z|) by stage (box plots, Kruskal-Wallis). **C:** Longitudinal session-by-session trajectory of % TF-responsive with linear trend. **D:** Response latency and half-width evolution across stages (dual y-axis). |
| **Method** | Pool per-unit TF responsiveness across all 38 sessions. Group by Naive/Learning/Expert. Track fraction responsive and amplitude per session. |
| **Statistics** | χ² test for fraction differences, Kruskal-Wallis for amplitude, linear regression for trajectory. |
| **Output** | `figures/08_tf_pulse/fig27_tf_learning_emergence.png`, `tf_learning_emergence.csv` |

---

## Fig 28 – TF × HMM Behavioral State Modulation

| | |
|------------|------|
| **Script** | `08_tf_pulse/e_tf_state_modulation.py` |
| **Question** | Does the animal's behavioral state (Engaged/Disengaged/Impulsive) gate sensory TF encoding? |
| **Panels** | **A:** Population TF pulse response split by HMM state (mean ± SEM traces). **B:** Fraction TF-responsive by HMM state (bar chart). **C:** Response amplitude in state × learning stage interaction (grouped bars). **D:** Modulation index (Engaged − Disengaged)/(Engaged + Disengaged) distribution with Wilcoxon test. |
| **Method** | For each session, split fast TF pulses by the HMM state of the parent trial. Compute state-specific z-scored responses per unit. Compare Engaged vs Disengaged TF encoding strength. |
| **Key parameters** | HMM states from K=3 model. Pulse classification same as fig24. Per-trial state assignment from hmm_k3_assignments.csv. |
| **Statistics** | Wilcoxon signed-rank for modulation index ≠ 0. |
| **Output** | `figures/08_tf_pulse/fig28_tf_state_modulation.png`, `tf_state_modulation.csv` |

---

## Fig 29 – TF Sensory-Motor Separation

| | |
|------------|------|
| **Script** | `08_tf_pulse/f_tf_sensory_motor.py` |
| **Question** | Do TF-responsive (sensory) neurons behave differently from non-responsive neurons around visual change and pre-lick periods? |
| **Panels** | **A:** Change-aligned PSTH for TF-responsive vs non-responsive units. **B:** Pre-lick PSTH for TF-responsive vs non-responsive units. **C:** Overlap bar chart: TF-responsive only / lick-responsive only / both / neither. **D:** Change-size selectivity curves for TF-responsive vs non-responsive. |
| **Method** | Classify units as TF-responsive (z ≥ 3.0 on post-pulse response) or not. Align to Change_ON and first Hit lick. Compare population PSTHs between groups. Overlap with lick responsiveness from pre-computed lick analysis. Change-size tuning computed from post-change (0–0.5s) firing rates at each TF multiplier. |
| **Key parameters** | Change window: (−0.5, 1.5) s. Lick window: (−1.5, 0.5) s. Gaussian σ = 20 ms. Change sizes: [1.25, 1.35, 1.5, 2.0, 4.0]. |
| **Output** | `figures/08_tf_pulse/fig29_tf_sensory_motor.png` |

---

## Summary of Shared Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Good cluster filter | min_rate_hz = 1.0 | Exclude units with overall FR < 1 Hz |
| Alignment event | Change_ON | Visual change onset (default) |
| Standard window | (−0.5, 1.0 s) | Peri-event window |
| Baseline window | (−0.4, −0.05 s) | Pre-stimulus baseline |
| Response window | (0.0, 0.25 s) | Post-stimulus response |
| Standard bin size | 25 ms | Temporal resolution |
| z-score normalization | Per-unit | (FR − baseline_mean) / baseline_std |
| Learning stages | Naive, Learning, Expert | From staging_manifest.csv |
| HMM states | Disengaged, Engaged, Impulsive | K=3 from hmm_k3_assignments.csv |
| Cell types | Narrow (FSI), Broad (MSN/Proj) | From waveform_celltypes.csv |

---

## File Organization

```
analysis_suite/
├── config.py                    # Central configuration
├── loader.py                    # Unified data loading
├── utils.py                     # Shared utilities (tensor building, z-scoring)
├── plotting.py                  # Common plotting functions
├── run_all.py                   # Sequential runner for all scripts
├── FIGURE_EXPLANATIONS.md       # This file
├── 01_behavior/                 # Scripts a–c → Figs 01–03
├── 02_single_unit/              # Scripts a–e → Figs 04–08
├── 03_population/               # Scripts a–c → Figs 09–11
├── 04_decoding/                 # Scripts a–c → Figs 12–14
├── 05_longitudinal/             # Scripts a–c → Figs 15–17
├── 06_lick_motor/               # Scripts a–c → Figs 18–20
├── 07_advanced/                 # Scripts a–c → Figs 21–23
├── 08_tf_pulse/                 # Scripts a–f → Figs 24–29  (TF pulse analysis)
├── figures/                     # Output PNGs and stats CSVs
│   ├── 01_behavior/
│   ├── 02_single_unit/
│   ├── 03_population/
│   ├── 04_decoding/
│   ├── 05_longitudinal/
│   ├── 06_lick_motor/
│   ├── 07_advanced/
│   └── 08_tf_pulse/
└── cache/                       # Pre-computed intermediate results
```
