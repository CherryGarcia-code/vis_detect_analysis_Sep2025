# Best tracked & functional neuron per mouse

`render_best_per_mouse.py` → `FIGURES/tracking_showcase/best_per_mouse.{png,pdf}`

One representative figure: the single best neuron per mouse that is **both
well-tracked and functionally responsive**, followed across learning. One
column per mouse, three rows (waveform stability · change-onset response ·
functional signature).

## Candidates

| Mouse | Region | Neuron | Tracker | Span | Tracking credential | Functional signature |
|-------|--------|--------|---------|------|---------------------|----------------------|
| BG_046 | DMS (medial striatum) | UM#942 ∩ DANT#631 | UM ∩ DANT **consensus** | 13 sessions (Learning→Expert) | held-out ISI r 0.97; Jaccard 0.81; DANT biophysically trusted | large-change response **grows** Learning (5.8 Hz) → Expert (11.0 Hz) |
| BG_031 | VMS (ventral striatum) | DANT#756 | DANT | 5 sessions (all Learning) | DANT trusted; held-out-ISI validated | **TF-encoding in 4/5 sessions** (Khilkevich–Lohse GLM); **choice AUROC 0.72** |
| BG_039 | DMS (medial striatum) | UM#217 | UnitMatch | 16 sessions (Learning→Expert; **longest track in the cohort**) | UM trusted | strong change response (median ~14z) held across all 16 sessions |

## How the candidates were chosen ("tracked AND functional")

Per the user's criterion, each mouse's pick had to satisfy **both** axes; the
best that did so was selected from cached tracking + functional evaluations
(no new scoring compute for 046/031):

- **BG_046** — shortlist = the 9 UM∩DANT consensus neurons
  (`data/cache/tracking_consensus/BG_046/consensus_cohort.csv` +
  `.../candidates/rendered_stats.csv`). #631 wins: best-tracked span that runs
  Learning→Expert (13 agreed sessions, ISI r 0.97) **and** the clearest,
  learning-modulated task response (verified in
  `FIGURES/tracking_consensus/BG_046/behavior/behavior_um942_dant631.png`).
  Note it is **not** TF-encoding (0/13 sessions) — "functional" here means a
  robust task-evoked (change / lick) response, which it has strongly.
- **BG_031** — shortlist = DANT tracks scored for TF-encoding
  (`FIGURES/tracking_dant/BG_031/curation/behavior_figs/behavior_stats.csv`).
  #756 wins the "AND": decent tracking (trusted, span 5) with the strongest
  function in the mouse — TF-encoding 4/5 sessions + choice AUROC 0.72. The
  longer tracks (#588/#790, span 12) are functionally flat (TF 1/12), so they
  lose on the functional axis.
- **BG_039** — flagship long track #217 (span 16, UM trusted). Functional signal
  confirmed by a direct per-session probe (change-onset response median ~14z,
  up to 53z; sustained across all 16 sessions).

## Methods

- **Waveform (row 1)** — raw mean waveform on the peak channel
  (`load_raw_mean_waveform` + `extract_peak_channel`), one trace per tracked
  session, overlaid. Tight overlap = the same physical unit across sessions.
  Absolute µV amplitude varies session-to-session (drift / re-referencing);
  the shape is the stable signature.
- **Change-onset PSTH (row 2)** — `extract_unit_psths(..., with_sem=True)`,
  condition `change_on_big_hit` (2.0× + 4.0× go-trial hits), 25 ms bins,
  σ = 25 ms Gaussian smoothing (`DEFAULT_BIN_SIZE` / `DEFAULT_SIGMA_MS`). Each
  session baseline-subtracted over (−0.5, 0) s. Bold line = across-session mean;
  band = **95% CI** (mean ± 1.96·SEM across sessions).
- **Functional signature (row 3)**
  - 046: change-evoked Hz = mean(0–0.5 s) − mean(−0.5–0 s), averaged within stage.
  - 031: per-session GLM TF-kernel `c1_r_log2` from
    `data/cache/tf_responsive/bg031_tf_responsive.csv`; purple = `resp_log2`
    True (`c1_r > 0.2` AND `c2_p < 0.01`); red line = the 0.2 threshold.
  - 039: per-session change-evoked Hz.
- **Shade / colour** — light→dark within a column = earliest→latest session
  (recording order). Green hue = Learning/Expert stage from the
  **subject-specific** staging manifest (`data/{subj}_staging_manifest.csv`);
  grey = Excluded/unstaged sessions. Naive folded to Learning.
- **Session ordering / joins** — all via `session_date_key` (subject-aware;
  handles BG_031/039 6-digit `DDMMYY` + prefix). PKLs and raw waveforms are
  **local** (`data/pkls`, `data/unit_match/input`) — no compute over X:.

## Caveats

- **Mixed trackers** — 046/031 = DANT (031 also consensus for 046); 039 =
  UnitMatch. Each column labels its tracker; credentials are not identical
  across trackers.
- **BG_031 is single-stage** — all 5 tracked sessions fall in Learning, so its
  "across learning" story is within-Learning stability, not Learning→Expert.
- **Consensus ≠ TF** — the BG_046 pick is task-functional but not TF-encoding;
  none of the 9 BG_046 consensus neurons are (base rate 2.8%). TF-encoding is
  the BG_031 pick's signature.
- **Change-onset for 031** is modest/idiosyncratic (a pre-rise dip); 756's
  strength is TF + choice (row 3 + header), not change-onset magnitude. Row 2
  is shown for uniformity across columns.

## Regenerate

```
py scripts/tracking_showcase/render_best_per_mouse.py
```
(force UTF-8 stdout on Windows consoles: `PYTHONIOENCODING=utf-8`.)
