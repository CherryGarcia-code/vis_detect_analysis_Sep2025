# TF-responsive cell registry

> ## ⚠️ STALE — these registries predate the lick-channel fix (2026-07-31)
>
> These CSVs were built **before** `visdetect.analysis.lick_channels` corrected
> how the GLM's lick nuisance regressor is built, so **they are no longer
> reproducible from current code** and must be regenerated before being trusted.
>
> **What changed.** `session_trial_regressors` previously POOLED every present NI
> lick channel. It now resolves exactly ONE. This alters the lick regressor on
> essentially every session:
> * **BG_046** — was `Piezo_1 ∪ Piezo_2` (33 sessions) or `Lick_L ∪ Lick_R` (13);
>   measured lick-count inflation **1.12×–4.99×** (median 1.40×). The 4.99×
>   outlier is `30062025`, where the old pool masked a **truncated `Lick_L`**
>   (see below).
> * **BG_031** — **7/43** sessions had a **contaminated `Lick_L`** (>10 Hz; worst
>   is `170325` at 751 793 events ≈ 63 Hz) that is now rejected in favour of
>   `Lick_R`. On the other 28 lick-convention sessions `Lick_L` was already clean
>   and is merely de-pooled from `Lick_R`; 8 are `Piezo_1`. So **all 43 sessions
>   change**, but only 7 were driven by corrupted input. *(An earlier draft of
>   this banner said "35/43 contaminated" — that was wrong: 35 is the number of
>   sessions in the `Lick_*` convention, not the number contaminated.)*
> * **BG_039** — was `Lick_L ∪ Lick_R`, now `Lick_L` only (32/32).
> * **Truncated recordings** — on `BG_046_30062025` and `BG_041_30062025` the
>   `Lick_L` line stops early (39% / 88% trial coverage) and the resolver now
>   falls back to `Lick_R`. Two sessions remain below 90% coverage with **no
>   alternative channel** (`BG_046_15072025` 66%, `BG_031_050325` 82%, both
>   `Piezo_1`-only) — check `LickChannelResult.trial_coverage` before trusting
>   lick-derived quantities there.
>
> **Why it matters.** The lick regressor is the point of the "lick-controlled
> GLM": changing it changes fitted coefficients, the TF-ablated residual (C2) and
> C1, so borderline `resp_log2` calls will flip. The headline comparison
> **VMS (BG_031) 5.3 % > DMS (BG_046 2.8 % / BG_039 3.1 %)** must be re-derived
> before being repeated — every session of all three subjects had its lick
> regressor changed, and BG_031 additionally had 7 sessions fit on a corrupted
> (~63 Hz) lick line.
>
> **Cheapest way to size the impact** (do this before a full rebuild): a PAIRED
> within-unit re-fit on a stratified subsample. `trial_index` is unchanged and
> `make_trial_folds` is seed-fixed, so both regressor variants fit on identical
> CV folds — far more sensitive than comparing responsive fractions. ~150–500
> near-threshold units ≈ 1–4 wall-hours locally; no cluster needed.
>
> ⚠️ **Do not resume over old outputs.** `run_tf_glm_bg046.py` skips sessions
> whose output files exist and the cluster task resumes per-unit, so a re-run
> would silently interleave old pooled rows with new single-channel rows. Clear
> `data/cache/tf_glm_*` and the cluster `results_bg_*` first.

Per-unit TF-responsiveness calls from the Khilkevich & Lohse (2024) replication:
a per-neuron ridge-Poisson GLM (50 ms bins, log2-TF, 10-fold CV). A unit is
**TF-responsive** (`resp_log2 = True`) iff BOTH:
- **C1** `c1_r_log2 > 0.2` — corr between the actual and full-model-predicted
  fast-minus-slow TF-pulse response (denoised pulse-PETH), and
- **C2** `c2_p_log2 < 0.01` — the TF-ablated residual is still significantly
  predicted across the 10 CV folds (one-sided t-test).

`*_lin` columns are the linear-Hz TF-encoding control (should be ≤ log2).

## Files
| File | Subject | Region | Sessions | Units | TF-resp |
|------|---------|--------|----------|-------|---------|
| `bg046_tf_responsive.csv` | BG_046 | DMS | 46 | 7047 | 195 (2.8%) |
| `bg039_tf_responsive.csv` | BG_039 | DMS | 32 | 2442 | 75 (3.1%) |
| `bg031_tf_responsive.csv` | BG_031 | VMS | 42 | 7537 | 399 (5.3%) |

All three complete. Pattern: **VMS (BG_031, 5.3%) > DMS (BG_046 2.8% / BG_039
3.1%)**. Do NOT pool across regions (DMS = {BG_046, BG_039}; VMS = {BG_031}
separate), and confirm bank positions before pooling cells *within* a region.
BG_031's session `20052025` is excluded (0 ingested trials).

## Columns
- `subject`, `session` (full pkl stem, e.g. `BG_039_02062025`), `session_date`
  (subject-stripped, e.g. `02062025`), `unit` (cluster_id) — **join keys**.
- `resp_log2` — the TF-responsive call (the thing you want).
- `c1_r_log2`, `c2_p_log2` — strength + significance behind the call.
- `kernel_peak_t`, `kernel_fwhm` — the unit's TF-kernel timing/width (s).
- `n_spikes` — total spikes used (all units here cleared the ≥500 gate).
- `resp_lin`, `c1_r_lin` — linear-encoding control.
- `region`, `region_bank_confirmed` — see caveats.

## How to join (in another analysis)
```python
import pandas as pd
from visdetect.analysis import config
reg = pd.read_csv("data/cache/tf_responsive/bg039_tf_responsive.csv")
# canonicalize BOTH sides of the join (leading-zero-day / 6-digit footgun)
reg["sess_key"] = reg["session_date"].map(config.canonical_session_id)
my_df["sess_key"] = my_df["session"].map(config.canonical_session_id)
out = my_df.merge(reg[["subject","sess_key","unit","resp_log2","c1_r_log2"]],
                  on=["subject","sess_key","unit"], how="left")
```

## Caveats (read before pooling)
1. **No-movement version.** BG mice lack processed movement/phase regressors, so
   this is the lick/wheel-controlled GLM *without* motion-energy/pupil/phase.
   TF-responsiveness here is **not movement-controlled** (first-pass).
2. **Regional pooling.** `region` is provisional. **DMS = {BG_046, BG_039}**;
   **BG_031 = VMS (keep separate)**. `region_bank_confirmed = False` because the
   per-session recorded bank/depth has **not** yet been audited — chronic probes
   drift, so do **not** pool cells across sessions/subjects until co-location is
   confirmed. Per-session/per-unit use is fine.
3. **Off-format sessions.** A few session ids carry suffixes / non-8-digit dates
   (e.g. `01042025_v2`); always key through `config.canonical_session_id`.
4. **Regenerable** from `…/tf_glm_cluster/bg_mice/results_bg_039/` via
   `scripts/tf_responsiveness/cluster_bg/aggregate_bg.py`.

Provenance: cluster job 3220157 (SUBJ=039), branch `feature/tf-glm-bg046`.
