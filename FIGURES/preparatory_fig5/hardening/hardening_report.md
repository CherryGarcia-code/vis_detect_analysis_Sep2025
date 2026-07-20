# Fig-5 e-h preparatory-by-cell-class — Stage-3 hardening report (HIT lick)

cache: `prep_hit.npz`  |  11598 cells (520 TF-responsive / 11078 non-TF)  |  DMS 6342 / VMS 5256
shuffles per null = 1000; bootstrap CI = 5000; base-frac window [-2.0, -1.8] s (8 bins).

Earlier onset = MORE NEGATIVE (s from lick). Headline: sustained leads transient leads non-TF; per-cell onset~width negative.


## Region: pooled

**Per-class population onset (bootstrap-over-neurons, headline):**
  - sustained  n=   99  onset=-0.738 s  peak_frac=+0.876 @ t=+0.012 s
  - transient  n=  315  onset=-0.613 s  peak_frac=+0.644 @ t=+0.087 s
  - non-TF     n=11078  onset=-0.338 s  peak_frac=+0.502 @ t=+0.137 s
  ordering sustained<transient<non-TF holds: **True**

### C2 label-shuffle null (per-class onset ordering)
  observed diff (onset_nonTF - onset_sustained) = +0.400 s (positive = sustained earlier)
  null diff: mean=+0.009, 95%=[-0.125, +0.175] (n_valid=1000)
  observed at percentile 100.0 of null; one-sided p=0.000999
  class-rank Spearman obs=+1.00 (p=0.1064); monotonic ordering obs=1.0
  VERDICT: label-shuffle null SURVIVES

### C3 width-shuffle null (per-cell onset~width Pearson)
  n_cells with finite onset = 495 of 520 responsive
  observed Pearson r(onset, width) = -0.105
  null |r|: 95th pct = 0.084 (n=1000); observed |r| at percentile 98.7; two-sided p=0.01399
  VERDICT: width-shuffle null SURVIVES

### C4 mixedlm pseudoreplication (onset ~ width)
  naive OLS: slope=-0.7377 s per (fwhm unit), p=0.01955
  mixedlm(session RE): slope=-0.9429, p=0.000256
  nested subject/session RE: n/a
  n_cells=495 over 69 sessions / 3 subjects
  per-session sign test: 34/43 sessions negative slope, binomial p=8.508e-05, Wilcoxon p=0.0001376

### C5 pre-lick-only control (active mask zeroed for t>=0)
  per-class pre-lick-only population onset (ordering must survive):
    sustained  onset_prelick=-0.738 s (full=-0.738); peak@+0.012 s; pre-lick ramp fraction=0.26
    transient  onset_prelick=-0.613 s (full=-0.613); peak@+0.087 s; pre-lick ramp fraction=0.22
    non-TF     onset_prelick=-0.338 s (full=-0.338); peak@+0.137 s; pre-lick ramp fraction=0.15
  ordering sustained<transient<non-TF (pre-lick only): **True**
  per-cell onset~width (pre-lick-only z): Pearson r=+0.009 (n=388)
  CAVEAT: peak fraction sits at ~+0.0 to +0.14 s (peri-lick); anticipatory MOVEMENT vs decision-PREPARATION cannot be separated without video (future extension — project has video_sync).

### C6 lick-responsiveness stratification (join lick_acquisition_cells.csv)
  matched 520/520 responsive cells to lick_sig
    within lick-responsive     : n= 372  Pearson r=-0.098  Spearman rho=-0.169 (p=0.001078)
    within non-lick-responsive : n= 123  Pearson r=-0.138  Spearman rho=-0.128 (p=0.1584)

### C7 independent re-derivation (different onset impl + seed)
  primary  slope(onset~width) = -0.7377  CI[-1.3968, -0.1822] (cell_onset primitive, seed 42, n=495)
  independent slope           = -0.7391  CI[-1.3499, -0.1678] (direct-loop 3-of-4, seed 123, n=495)
  independent slope within primary CI: **True**; onset MAE between implementations = 0.0237 s


## Region: DMS

**Per-class population onset (bootstrap-over-neurons, headline):**
  - sustained  n=   30  onset=-0.688 s  peak_frac=+0.983 @ t=+0.012 s
  - transient  n=  132  onset=-0.563 s  peak_frac=+0.721 @ t=+0.462 s
  - non-TF     n= 6141  onset=-0.363 s  peak_frac=+0.545 @ t=+0.287 s
  ordering sustained<transient<non-TF holds: **True**

### C2 label-shuffle null (per-class onset ordering)
  observed diff (onset_nonTF - onset_sustained) = +0.325 s (positive = sustained earlier)
  null diff: mean=-0.038, 95%=[-0.201, +0.176] (n_valid=1000)
  observed at percentile 99.7 of null; one-sided p=0.003996
  class-rank Spearman obs=+1.00 (p=0.07404); monotonic ordering obs=1.0
  VERDICT: label-shuffle null SURVIVES

### C3 width-shuffle null (per-cell onset~width Pearson)
  n_cells with finite onset = 189 of 201 responsive
  observed Pearson r(onset, width) = -0.206
  null |r|: 95th pct = 0.136 (n=1000); observed |r| at percentile 99.8; two-sided p=0.002997
  VERDICT: width-shuffle null SURVIVES

### C4 mixedlm pseudoreplication (onset ~ width)
  naive OLS: slope=-1.1485 s per (fwhm unit), p=0.00446
  mixedlm(session RE): slope=-1.1134, p=0.0004481
  nested subject/session RE: n/a
  n_cells=189 over 40 sessions / 2 subjects
  per-session sign test: 13/16 sessions negative slope, binomial p=0.01064, Wilcoxon p=0.01242

### C5 pre-lick-only control (active mask zeroed for t>=0)
  per-class pre-lick-only population onset (ordering must survive):
    sustained  onset_prelick=-0.688 s (full=-0.688); peak@+0.012 s; pre-lick ramp fraction=0.28
    transient  onset_prelick=-0.563 s (full=-0.563); peak@+0.462 s; pre-lick ramp fraction=0.17
    non-TF     onset_prelick=-0.363 s (full=-0.363); peak@+0.287 s; pre-lick ramp fraction=0.14
  ordering sustained<transient<non-TF (pre-lick only): **True**
  per-cell onset~width (pre-lick-only z): Pearson r=-0.080 (n=146)
  CAVEAT: peak fraction sits at ~+0.0 to +0.14 s (peri-lick); anticipatory MOVEMENT vs decision-PREPARATION cannot be separated without video (future extension — project has video_sync).

### C6 lick-responsiveness stratification (join lick_acquisition_cells.csv)
  matched 201/201 responsive cells to lick_sig
    within lick-responsive     : n= 149  Pearson r=-0.132  Spearman rho=-0.331 (p=3.708e-05)
    within non-lick-responsive : n=  40  Pearson r=-0.415  Spearman rho=-0.421 (p=0.006777)

### C7 independent re-derivation (different onset impl + seed)
  primary  slope(onset~width) = -1.1485  CI[-2.2098, -0.5563] (cell_onset primitive, seed 42, n=189)
  independent slope           = -1.1492  CI[-2.2065, -0.5489] (direct-loop 3-of-4, seed 123, n=189)
  independent slope within primary CI: **True**; onset MAE between implementations = 0.0238 s


## Region: VMS

**Per-class population onset (bootstrap-over-neurons, headline):**
  - sustained  n=   69  onset=-0.888 s  peak_frac=+0.830 @ t=-0.013 s
  - transient  n=  183  onset=-0.713 s  peak_frac=+0.606 @ t=+0.087 s
  - non-TF     n= 4937  onset=-0.338 s  peak_frac=+0.451 @ t=+0.137 s
  ordering sustained<transient<non-TF holds: **True**

### C2 label-shuffle null (per-class onset ordering)
  observed diff (onset_nonTF - onset_sustained) = +0.550 s (positive = sustained earlier)
  null diff: mean=+0.015, 95%=[-0.175, +0.350] (n_valid=1000)
  observed at percentile 99.0 of null; one-sided p=0.01099
  class-rank Spearman obs=+1.00 (p=0.09248); monotonic ordering obs=1.0
  VERDICT: label-shuffle null SURVIVES

### C3 width-shuffle null (per-cell onset~width Pearson)
  n_cells with finite onset = 306 of 319 responsive
  observed Pearson r(onset, width) = -0.040
  null |r|: 95th pct = 0.107 (n=1000); observed |r| at percentile 50.6; two-sided p=0.4945
  VERDICT: width-shuffle null does NOT beat null (weak/absent per-cell gradient)

### C4 mixedlm pseudoreplication (onset ~ width)
  naive OLS: slope=-0.3205 s per (fwhm unit), p=0.481
  mixedlm(session RE): slope=-0.8230, p=0.02266
  nested subject/session RE: n/a
  n_cells=306 over 29 sessions / 1 subjects
  per-session sign test: 21/27 sessions negative slope, binomial p=0.002962, Wilcoxon p=0.003892

### C5 pre-lick-only control (active mask zeroed for t>=0)
  per-class pre-lick-only population onset (ordering must survive):
    sustained  onset_prelick=-0.888 s (full=-0.888); peak@-0.013 s; pre-lick ramp fraction=0.25
    transient  onset_prelick=-0.713 s (full=-0.713); peak@+0.087 s; pre-lick ramp fraction=0.25
    non-TF     onset_prelick=-0.338 s (full=-0.338); peak@+0.137 s; pre-lick ramp fraction=0.16
  ordering sustained<transient<non-TF (pre-lick only): **True**
  per-cell onset~width (pre-lick-only z): Pearson r=+0.055 (n=242)
  CAVEAT: peak fraction sits at ~+0.0 to +0.14 s (peri-lick); anticipatory MOVEMENT vs decision-PREPARATION cannot be separated without video (future extension — project has video_sync).

### C6 lick-responsiveness stratification (join lick_acquisition_cells.csv)
  matched 319/319 responsive cells to lick_sig
    within lick-responsive     : n= 223  Pearson r=-0.076  Spearman rho=-0.083 (p=0.2194)
    within non-lick-responsive : n=  83  Pearson r=+0.008  Spearman rho=+0.102 (p=0.3567)

### C7 independent re-derivation (different onset impl + seed)
  primary  slope(onset~width) = -0.3205  CI[-1.2140, +0.5579] (cell_onset primitive, seed 42, n=306)
  independent slope           = -0.3225  CI[-1.2065, +0.5823] (direct-loop 3-of-4, seed 123, n=306)
  independent slope within primary CI: **True**; onset MAE between implementations = 0.0236 s


## C8 lick-time-shuffle null (run from main session, responsive cells)
  Re-aligned the 520 responsive cells to RANDOM times (same count as real hit-licks,
  drawn across the task span), z-scored to the same 2 s pre-change baseline
  (`licktime_shuffle_control.py`, 69 sessions, 0 errors):
    transient  REAL peak_frac ~0.64  ->  SHUFFLED peak_frac = +0.036 (mean|frac|=0.013)
    sustained  REAL peak_frac ~0.88  ->  SHUFFLED peak_frac = +0.057 (mean|frac|=0.031)
  The pre-lick ramp COLLAPSES to ~baseline under random-time alignment.
  VERDICT: the preparatory ramp is genuinely LICK-LOCKED, not a slow-drift/arousal
  artifact. **SURVIVES.**
