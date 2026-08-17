# Raw NI-DAQ and task-event specification

**Status**: derived from a full re-extraction of one session (BG_046, 17 Sep 2025) directly from the
raw SpikeGLX `nidq.bin`, 13–14 Aug 2026, then **adversarially audited by six independent reviewers**
and corrected. Claims that survived audit are stated plainly; claims that failed are marked and the
corrected version given. Where a quantity is not identified, it is bracketed rather than quoted.

**Purpose**: a new extraction pipeline should be buildable from this document without repeating the
investigation, and without inheriting either the defects in the existing MATLAB extractions or the
mistakes made in this one.

**Scope caveat**: ONE session, ONE subject. Nothing here has been replicated on a second recording.
The channel map is reported to be byte-identical across all 50 BG_046 raw sessions, but every numeric
constant should be re-derived per session — the procedures below are per-session by construction.

---

## 0. Read the settings files FIRST — the biggest lesson

The rig logs its own parameters. In this investigation two constants were laboriously *fitted* that
were sitting in `Session/*_session_settings.json` all along, because the file was searched with
keyword regexes (`wheel|encoder|radius|circumference|cpr`) that matched none of the actual names.

**Dump the settings files in full and read them before deriving anything.**

| key | value | what it fixes |
|---|---|---|
| `spdrnghigh` / `spdrnglow` | 5 / −5 (cm/s) | abort threshold — with `spdavgbin` gives 2.5 mm / 50 ms |
| `spdavgbin` | 0.05 (s) | the averaging bin for the abort rule |
| `cRunning` | 1 | running-abort enabled |
| `refLine` | 0.005 | **the "5 ms offset"** — a flip-lead subtracted at scheduling |
| `Tstimdelaymin` / `Tstimdelaymeanadd` | 3 / 0.5 | trial start needs ≥3 s stationary + Exp(0.5 s) |
| `itirunning` | `'auto'` | ITI extends until the animal settles |
| `Torientationdelaymin` / `…meanadd` | 6 / 2 | change time = 6 s + Exp(2 s) — explains 6–11 s latencies |
| `Trewdavailable` | 2 | 2 s reward window — explains the ~2000 ms second pulses on Miss trials |
| `rewd1` / `rewd2` | 0.2 | reward valve open time (s) |
| `punishT` | 0.2 | ⚠ also 0.2 — see §6, this makes the "rewd matches" argument non-unique |
| `punishearly` | `'End trial on Stim1 lick'` | early lick ends the trial (the `fa` outcome) |
| `lickthreshold` | 0.25 | the online lick threshold (units unverified — see §7) |
| `temporalfreq` / `TFsd1` | 1 / 0.25 | base TF 1 Hz, 0.25 octave SD |
| `changelist1` / `changelist2` | [4, 2] / [1.5, 1.35, 1.25] | the change sizes |
| `monitors.left.photodiode_location` | [0, 930, 150, 1080] | sync square = bottom 150 rows of 1080 |

`computer_settings.json` additionally holds monitor geometry (`width` 49 cm, `distance` 20 cm) and
the gamma table. `trials.json` carries per-frame `vbl` flip timestamps and a per-frame content `tag`
— an independent display-side clock that this analysis under-used.

---

## 1. Why not trust the existing extractions

Two MATLAB extractions exist and **neither is usable alone**:

| | 2025 original | 2026 re-extract (6 Mar) |
|---|---|---|
| licks | `Lick_L` 10,093 / `Lick_R` 13,358 crossings | `Piezo_1` 797 / `Piezo_2` 349 |
| laser | **absent** | 1,003 pulses |

They read the *same physical lines* at different thresholds. This is now **measured, not inferred**:
at a 0.150 V threshold, ch4 reproduces all 10,093 `Lick_L` times at **offset zero, 100.00%** (best
alternative channel 26.4%), and ch5 reproduces `Lick_R` likewise; the swap is incompatible. At 1.000 V
the same lines give exactly 797 and 349 — the 2026 counts.

- `Lick_L` ≡ `Piezo_1` ≡ analog **ch4**; `Lick_R` ≡ `Piezo_2` ≡ **ch5**; `Valve_L` ≡ **ch6** (251/251).
- ❌ *Earlier claim*: "the 2025 run's `Valve_R` was empty because the laser's 0.38 V never crossed
  threshold." **False** — at 0.150 V the laser line has 1,007 crossings. The real reason is
  structural: `Valve_R` is a per-trial field and **no laser pulse falls inside any trial**.

**Never treat either extraction as ground truth.** Both are undocumented threshold choices.

---

## 2. File layout

```
{root}/{subject}/Raw data/{subject}_{DDMMYYYY}/
    EphysNidaq/..._g0_t0.nidq.bin / .nidq.meta
    EphysNidaq/..._g0_imec0/..._t0.imec0.ap.bin / .ap.meta
    Session/{subject}_{YYYYMMDD}_{HHMMSS}__trials.json + *_settings.json
    Cameras/                                    # may be EMPTY
{root}/{subject}/Processed data/{subject}_{DDMMYYYY}/
    Kilosort&Phy/..._imec0/                     # KS4 output + CatGT tcat
    Nidaq/..._NIdaq_events.mat                  # MATLAB output — see §1
    Nidaq/NI_Sync.txt                           # CatGT sync edges (TPrime input)
```

`nidq.bin` is **interleaved int16**, `nSavedChans` per sample. Sample count =
`filesize / (nSavedChans × 2)` and must divide exactly — a free integrity check (here
2,019,796,434 / 18 = **112,210,913**). Scale with `niAiRangeMax / niMaxInt` (5/32768); XA gain is 1.

---

## 3. Channel map

`acqMnMaXaDw = 0,0,8,1` → 8 analog (XA) then 1 digital word (DW). Parse names from `~snsChanMap`;
never hard-code order.

| idx | name | what it is | measured swing |
|---|---|---|---|
| 0 | `Photodiode` | corner sync square, **not** the grating | analog |
| 1 | `Baseline_ON` | trial/stimulus onset | 3.45 V |
| 2 | `Change_ON` | change-stimulus marker | 3.36 V |
| 3 | `Airpuff` | ⚠ **UNCONNECTED on this session** | **0.028 V** |
| 4 | `Piezo_1` | **lick sensor** (analog) | rails to 5 V |
| 5 | `Piezo_2` | second lick line, weaker | rails to 5 V |
| 6 | `Valve_1` | **reward solenoid** | 3.28 V |
| 7 | `Laser` | optotagging trigger | **0.263 V** |

⚠ **Guard unconnected lines.** `Airpuff` swings 0.028 V; a naive level estimator put a threshold
inside its own noise and emitted **17.4 million spurious edges**. Require a minimum low→high swing
(0.10 V here) before treating a line as TTL — that keeps `Laser` (0.263 V) and rejects `Airpuff`.

⚠ **The Laser line peaks at 0.383 V.** A conventional 2.5 V TTL threshold finds **zero** pulses. It is
also not a standard logic HIGH, so "TTL" is loose — the data cannot distinguish an attenuated digital
trigger from an analog command held constant. Either way it says nothing about light at the fibre tip.

⚠ **`Baseline_ON` contains one excursion to the negative rail** while resting near 0 V, so a min/max
midpoint threshold sits *below* the resting level and the line reads permanently high (0 of 739 pulses
recovered). Use robust levels: `low = median(x)`, `high = median(x[x > low + 0.5·(max−low)])`.

### Digital word (ch8; the meta does not say which bit is which)

| bit | n rises | identity |
|---|---|---|
| 0 | 10,593 | **Sync** (1 Hz) |
| 1, 2 | 1 each | glitch — ⚠ **1 rise, 0 falls**: these lines go high and never return |
| 3, 7 | 0 | unused |
| 4 | 761 | carries the **trial marker** (see caveat) |
| 5 / 6 | 244,319 / 244,447 | rotary encoder A / B |

⚠ **`dig4` is not a "duplicate" of the analog `Baseline_ON`.** Only **663/761** rise indices are
sample-identical, and the 22 slivers on each line occur at *completely different times* (0 of 22
coincide). Matching counts is not matching signals. Its real value is being **threshold-free**, which
is what adjudicates the edge rule in §5.

⚠ **Quadrature-decode the encoder; do not count edges.** Reconstruct A/B states and walk the Gray
sequence, summing signed counts. But note two things the first pass got wrong:
- "100% valid transitions" is a **tautology** — a threshold extractor emits strictly alternating
  rise/fall per line, so every consecutive pair flips one bit, and any one-bit change of a 2-bit Gray
  state is ±1. It returns 100% for pure noise. The informative test is whether consecutive transitions
  **alternate between lines**: here 5.3% do not (real bounce), so honest validity is **94.7%**.
- The "6× gross/net inflation" is a divide-by-small-denominator artifact of quiet windows;
  session-wide the ratio is **1.25×**. Decoding is still correct — that number just isn't the reason.

---

## 4. Time base — **use `niSampRate` = 10593.2 Hz**

| source | value |
|---|---|
| `niSampRate` (meta) | 10593.2 |
| `niClockSource` line | 10593.220339 |
| fitted from the 1 Hz sync | 10593.289960 (max resid 0.183 samples) |

⚠ **The fitted value is not "the true rate in Hz".** `syncSourceIdx=3` means the 1 Hz pulse is
generated by the **imec basestation**, so the fit gives *NI samples per basestation-second*. The
probe's own nominal 30 kHz is itself ~+9.9 ppm off the same generator. **Nothing in this dataset is
traceable to absolute time.**

❌ *Earlier evidence, retracted*: "reproduces `NI_Sync.txt` and the `.mat` to 0.000 ms". That is a
**mathematical identity** — `NI_Sync.txt` is CatGT's edge extraction converted with the same meta
rate, so testing `index/10593.2` against it carries zero information. The two `.mat` files are also
**byte-identical** to each other on `Synch`/`Baseline_ON`, so three "independent confirmations" were
one measurement.

✅ **The real evidence**: reconstruct TPrime's piecewise-linear map from the two sync-edge files and
compare to `spike_times_sec_adj`.

| target frame | median residual | drift |
|---|---|---|
| **meta 10593.2** | **13.9 µs** (p99 71 µs) | **flat (+0.00006 ppm)** |
| fitted 10593.28996 | **46.1 ms** | −8.49 ppm ramp |

Adopting the fitted rate would misalign events against spikes by up to **90 ms**, growing across the
session, so no constant offset can absorb it.

⚠ **Bug found and fixed in this codebase**: `extract_nidq.py` originally wrote the stored `*_t`
arrays using the *fitted* rate while everything downstream used the meta rate — a latent 75–89 ms
trap inside the deliverable `.npz`. Write times with the adopted rate, or store only `*_idx` and force
consumers to divide.

**Spikes**: prefer `spike_times_sec_adj.npy` (TPrime-mapped into the NI frame; median shift −9.0 ms,
range −14.9 → +0.03 ms — not negligible). Two traps: it is **not sorted** (117 backsteps up to
19.7 µs, min value −0.0002 s), and `spike_times_sec.npy` has shape **(N,1)** while `_adj` is **(N,)**.

---

## 5. Event extraction — merge split edges, then first pulse per trial

❌ *Earlier rule, retracted*: discard 4.9–5.0 ms "slivers" on a ≥15 ms width cut.

**That rule was wrong on 20 of 739 trials.** Twenty of the 22 slivers are separated from the following
pulse by **exactly one sample** — they are the *leading edge* of a single trial marker split by a
momentary dip. The threshold-free `dig4` copy shows one continuous pulse whose rise coincides with the
**sliver** on 20/20 (within 1 sample) and never with the second edge. The width cut therefore placed
20 trial onsets **5.02 ms late**. The trial *count* cannot detect this — both rules return 739.

✅ **Correct rule**: merge pulses separated by **≤2 samples**, then take the **first pulse per trial**.
No width cut, no fallback.

| channel | n | exact vs MATLAB | max diff |
|---|---|---|---|
| `Baseline_ON` | 739 | **739** | **0.0000 ms** |
| `Change_ON` | 323 | **323** | **0.0000 ms** |
| `Valve_1` | 251 | **251** | **0.0000 ms** |

(Under the old rule these were 719/739 and 315/323, differing by ~5 ms. The doc previously claimed
"0.000 ms" for them, which was false.)

Scored against the programmed `stimT`, merging tightens `Change_ON` from an **8.98 ms** spread with 8
outliers to **0.227 ms** with none — a 40× improvement. MATLAB was right on all 8.

### `Valve_1` → reward deliveries

Widths are bimodal: **242 at ~200 ms** (the deliveries, one per go-trial Hit) and **9 at ~0.09 ms**
(one sample). The blips occur on catch-trial Hits — the line is briefly commanded but reward is
correctly withheld. Keep pulses ≥100 ms. ⚠ Only 9 of the **14** catch-trial Hits have a blip.
⚠ The 200 ms group is itself two-valued (158 at 200.034, 84 at 199.940 ms — one sample apart); do not
quote its median to finer than the 0.0944 ms sample period.

### `Laser` → optotagging pulses

Require width ≥2 ms. Real pulses are 10.01 ms. There are **1004 raw rises**: 1002 real plus **two**
0.094 ms artifacts (2717.7 s and 5223.2 s — ⚠ **both** are mid-behaviour; behaviour ends at 8857 s).
MATLAB's list contains one of them. ❌ The earlier "Laser 1003/1003" agreement row was wrong; no rule
yields 1003 from a re-extraction.

### ⚠ Edge-pairing guards

Downstream code pairs `rise[i]` with `fall[i]`. That requires the line to start LOW and every rise to
have a later fall. **Both conditions are unguarded in most code and one is live here**: `dig1`/`dig2`
have 1 rise and 0 falls. A line HIGH at sample 0 would pair every event with the wrong partner and
produce negative widths **with no error raised**. Assert both.

---

## 6. Verifying channel identity — the method

Meta labels are unreliable (§1). Verify against something non-circular. Worked example, ch6 = valve:

| check | result |
|---|---|
| only channel with ~200 ms pulses | 0 such pulses on all six other TTL lines |
| count == go-trial Hits | 242 == 242, one each |
| never on FA / abort / Miss / catch-Hit | 0 |
| follows the **measured** lick (not software RT) | **4.72 ms** median |
| never during the optotagging epoch | 0 |

⚠ **The `rewd1 = 0.2` match is suggestive, not decisive.** Four settings parameters equal 0.2 —
`pprobe0`, `rewd1`, `rewd2` and **`punishT`** — so a 200 ms pulse on a punishment line would match
equally well. And `rewd` takes only two values (0.2 on all 658 go trials, 0.0 on 81 catch), so there
is no variance to test against. ⚠ `rewd` is **not** a delivery record: it is 0.2 on every go trial
regardless of outcome, including FA and abort.

---

## 7. Lick detection — what it is, honestly

The piezo is analog; MATLAB thresholded it as TTL, which is what produced the under-detection.

**Derived detector**: threshold **8σ = 0.214 V** on `Piezo_1` (σ = 1.4826 × MAD of the trace itself),
**30 ms** refractory → **3,580 contacts**.

| detector | contacts | rewards preceded | lead time | ILI 100–200 ms |
|---|---|---|---|---|
| derived, 8σ | 3,580 | 100% | 4.63 ms | 46.3% |
| `Lick_L` (2025) debounced 30 ms | 4,651 | 100% | **5.00 ms** | 45.9% |
| `Piezo_1` (2026) | 494 | **14.9%** | (see below) | 9.7% |

### ⚠ What the audit established about this derivation

- ❌ **"Independent of any previous extraction" is false.** The valve opens ~5 ms after the *online
  task computer* registered a lick — itself a threshold on the **same `Piezo_1` line**. So "highest
  threshold explaining 100% of rewards" recovers **the online detector's threshold**. It is independent
  of the two MATLAB *re-extractions*, which is a weaker and different claim. (`lickthreshold: 0.25` is
  logged, but its units are unverified and it does not reproduce the observed coverage in NI volts —
  a Teensy handles the I/O and may scale differently.)
- ❌ **"Simultaneously maximises rhythm and surge" is false.** Rhythm peaks at 7.75σ and is a
  **plateau** (5.75–8.5σ within 1 pp); the post-change surge peaks at 12.5σ; the lead-time criterion
  points to **5.5σ** (5.003 ms vs the valve's 4.973 ms — 11× closer than 8σ). The criteria disagree.
- ❌ The selection rule *guarantees* 100% coverage by construction. It is a tautology, not evidence.
- ❌ **The 30 ms refractory is asserted, not derived.** Reward coverage is 100% at every refractory
  from 0–100 ms, and ~10% of intervals pile against the 30 ms floor — a sign it truncates real
  structure. Contact count moves ±8% between 20 and 50 ms.
- ❌ "Every train is dominated by sub-10 ms intervals" is wrong: only `Lick_R` exceeds 50%, and the
  2026 trains have median ILIs inside the lick band.
- ❌ The "35,923 ms lead time" for the 2026 train measures **train sparsity**, not misidentification:
  on the 36 rewards it does cover, its lead is **2.08 ms**. Its defect is sensitivity, not wrongness.
- ⚠ `Lick_L` de-bounced at 30 ms **matches or beats** the derived detector on every stated criterion.
  The honest framing is "we independently arrived at essentially the 2025 train", not "we did better".
- ⚠ 186 contacts (5.2%) fall in the optotagging epoch where no reward-seeking lick is possible, and
  they are **not rhythmic** (21% in band vs 46%) — a noise floor of order 30% of the behavioural train.

### ✅ What does survive

**Post-reward consumption bouts**, never used in fitting: **4.98 Hz** in the 1 s after reward vs
1.00 Hz before, median post-reward ILI 155 ms (6.4 Hz), **72.1%** of post-reward ILIs in 100–200 ms,
**98.8%** of rewards followed by ≥3 contacts. This establishes the contacts **are** licks. It does not
validate 8σ specifically.

⚠ The `r = 0.9994` agreement with the software RT is **not** independent validation — both descend
from the online threshold on the same line, and sub-millisecond agreement between two "independent"
transducers is itself proof of shared provenance. Also `r` is uninformative here: a detector with
50 ms RMS error still scores 0.9935.

---

## 8. Task timing

**A fixed 5.13 ms software→hardware offset — NOT a "tick".** ❌ The earlier "~5 ms task tick" claim is
refuted. It is `refLine: 0.005`, a flip-lead subtracted at scheduling: `stimT = (integer frames)/60 −
0.005` holds for **732/732 trials to 2×10⁻¹³ s**. The task's real quantum is the **16.67 ms video
frame**. The published quantisation test was degenerate — a pure constant scores R = 0.95–1.00 at
*every* candidate tick. A proper test (reward minus the continuous piezo-derived contact) gives a
p5–p95 spread of **0.94 ms**: fixed latency, no tick.

**Display latency ~67 ms — an upper bound, and not "4 frames".** The photodiode watches a sync square
in the **bottom 150 rows** of a 1080-line display (logged in `computer_settings.json`), painted last in
the raster. Measured TTL→photodiode is **+67.3 ms** (IQR 55–79), robust to detector settings
(3σ→65.5, 24σ→72.0). But:
- ❌ "~4 frames" is refuted — the latency is **not frame-quantised** and spans 2.8 frames p5–p95.
- The **TTL is not frame-locked** (sd 14.6 ms against the stimulus PC's own `vbl` log) while the
  photodiode **is** (corr −0.988; re-referencing to the frame removes 97.6% of the variance). So
  67.3 ms = (TTL→frame, unknown mean, sd 14.6 ms) + (frame→photons, sd 2.29 ms), and the data cannot
  split it. Do not attribute all of it to the display.
- The detector is unvalidated beyond onset: it returns **0/323** detections at `Change_ON`, firing
  only on the gray→grating contrast step.

> Still actionable: aligning visual responses to the raw `Baseline_ON` TTL places "stimulus onset"
> tens of ms early, comparable to striatal visual latency. Use `vbl` and/or the photodiode.

**Trial start requires stationarity, not lick suppression.** `Tstimdelaymin = 3` +
`Tstimdelaymeanadd = 0.5` → ≥3 s stationary plus Exp(0.5 s), with `itirunning: 'auto'`. Measured ITI
median 5.755 s, min 3.660 s, max 40.8 s.

> ❌ An earlier draft claimed the 3.09 s minimum last-lick-to-onset gap revealed an enforced
> *lick-free* rule. **Wrong** — it follows from the ITI. Licking is concentrated in the **ITI**
> (0.663 Hz) rather than the trial (0.064 Hz): roughly one response lick ends the trial and the
> consumption bout falls in the ITI.

**Movement aborts trials** — this is the rig's *configuration* (`cRunning: 1`, `spdrnghigh/low = ±5`),
not a discovery. Only the effect size is informative: measuring the last 1 s **inside** each trial,
abort median **47.0** counts/s vs Hit/Miss **1.0** (p = 3.7×10⁻⁶⁸), and 0/153 aborts show zero net
motion versus 26.8% of completed trials.

> ⚠ An earlier version anchored aborts to `onset + 2 s` with a 1 s window. 81/153 abort trials are
> shorter than 3 s, so ~43% of that window fell **after** the trial ended and measured post-abort ITI
> running — inflating the abort mean to 247 counts/s. Anchor to each trial's own end.

### Wheel calibration — bracketed, not point-identified

The rule is the rig's own: `spdrnghigh = 5` cm/s × `spdavgbin = 0.05` s = **2.5 mm per 50 ms**. Finding
the count threshold that predicts abort times gives a **bracket, not a value**:

- **T ≤ 15** counts/50 ms → 100% abort coverage (aborts are movement-only, so this is required)
- **T ≥ 18** → zero pre-trial stationarity violations

No single T satisfies both, which is expected: the rig thresholds a **signed velocity** averaged over
`spdavgbin`, while this measures **|net displacement|** in a sliding window. Adopt the midpoint:

**T = 16 → 0.156 mm/count, bracket [0.139, 0.167] (±9%).**

⚠ Do not quote 4 significant figures. The earlier "15 counts = 0.1667 mm/count" moved 33% when the
selection criterion changed, and its stated corroborations were circular — the −29 ms "detect-and-
terminate latency" is a monotone function of T (T=20 makes it ~0 ms), and the 733/739 stationarity
figure *falsifies* rather than supports.

---

## 9. Behavioural JSON

- Multiple runs per session are normal: here an aborted 11:00:30 run (**7 trials**) and the main
  11:01:46 run (**732**). Both started after NI recording began, so both appear in the NI stream;
  concatenating in timestamp order gives 739 = the `Baseline_ON` count.
- ⚠ **De-duplicate by hash.** This session has a byte-identical `__trials (2).json`; globbing
  `*trials.json` and concatenating gives **1,471** trials. (The `(2)` settings files are duplicates too.)
- Useful fields: `trialoutcome`, `Stim2TF` (change_size), `stimT`, `reactiontimes` (`rt_RT`, `rt_FA`),
  `rewd` (valve open time — see §6), `St1TrialVector`, per-frame `TF`, `vbl`, `tag`.
- Invariant that should always hold: `count(Hit) + count(Miss) + count(Ref) == n Change_ON`
  (here 256 + 58 + 9 = **323**, and the trial *sets* are identical).

### No change stimulus on `fa` / `abort` — the valid argument

❌ *Earlier argument, retracted*: "all 43 surplus `Change_ON` pulses fall on Hit/Miss/Ref and none on
fa/abort, independently confirming `EVENT_VALID_OUTCOMES`." This is **logically entailed** by the set
equality above, so carries no extra information — and 35 of the 43 start *after* their assigned trial
ended (median 0.495 s into the ITI); attributing those forwards instead, equally arbitrary, puts 17 on
FA/abort. The conclusion was an artifact of `searchsorted` attributing ITI events backwards.

✅ **The valid argument**: the trial's own `Baseline_ON` pulse ends **before the change was even
scheduled** on **100% of FA trials** (n=263, median margin 3.163 s) and **100% of aborts** (n=153,
4.749 s). No change could have been presented.

---

## 10. Optotagging

Two blocks of 501 pulses (10.01 ms) after behaviour: block 0 = **SNr** (direct pathway, putative D1),
block 1 = **GPe** (indirect, putative D2) — antidromic stimulation of striatal terminals.

⚠ The block→target mapping is **not in any settings file**; it came from the experimenter. Capture it
at acquisition. The attribution of the response asymmetry to **fibre placement** is likewise
experimenter-supplied and untestable from these files.

**Per-block screen** (exact Poisson on the 1–10 ms window, BH-FDR q<0.01, ≥10 evoked spikes):

| class | n |
|---|---|
| block 0 only (SNr) | **17** |
| both | **3** |
| block 1 only (GPe) | **0** |
| tested and negative | 437 |
| **untestable** (too few evoked spikes) | 199 |
| not screened (<50 spikes in epoch) | 14 |

⚠ Report untestable separately — an earlier version collapsed these into "638 neither", overstating
specificity. ⚠ A **baseline**-count guard is the wrong fix for the `inf`-z problem: requiring ≥20
baseline spikes made any unit below 0.887 Hz untestable and discarded a genuine responder (cluster 209,
q≈5×10⁻¹⁷). Use an exact Poisson test with an **evoked**-count floor — a low expectation makes the test
conservative, not invalid.

**The asymmetry is not a detection-power artifact** — attacked directly and survived: responders are
not higher-firing (p=0.705), and the detectability floor *rises* with baseline rate. An empirical null
over 8,397 sham tests gave **zero** false positives, so the screen is conservative.

⚠ "Never pool the blocks" was overstated: pooled ranking does *not* bury the weaker block
(Spearman(pooled, best-per-block) = 0.986). The defensible principle is narrower — **a pooled
statistic cannot assign pathway identity**.

### Expression strategy — read this before interpreting anything

ChR2 reaches these cells by **anterograde transsynaptic Cre injected in MOs**, which jumps to
downstream striatal neurons. Three consequences:

- **ChR2 is in striatal SOMATA and their axons**, not in the terminals of some other structure.
- **The ~3% response yield is expected**, not suspicious: anterograde transsynaptic labelling is
  inefficient and MOs-recipient cells are a sparse subset.
- **Responders are spatially intermixed with non-responders**, which is also expected — MOs-recipient
  cells are scattered. Measured: responder depths 1590–2145 µm, indistinguishable from
  non-responders (MWU p=0.10, KS p=0.13), no contiguous band, no meaningful depth gradient
  (Spearman ρ = −0.15) and no latency-vs-depth relationship (p=0.19).

A **direct somatic illumination** account was considered and **rejected on fibre position**: fibres
are confirmed at SNr and GPe, and SNr is 2–3 mm from striatum, so no light reaches striatal somata.
With opsin in striatal somata and axons, the only route by which SNr/GPe light can drive a striatal
soma is **antidromic invasion from illuminated terminals**.

### Antidromic status: **plausible but NOT established**

The responses are unambiguously **optogenetically driven** — laser-locked, absent in the other block
for the same units, no onset artifact (no excess at 0–1 ms), no spike-identity artifact
(evoked/spontaneous amplitude ratio 1.016). Whether they are **antidromic** is not resolved, and the
reason is the stimulation protocol, not the analysis.

**Measured properties** (block 0, strongest responders): earliest reliable spikes at **1.7–2.0 ms**
(p5–p10; sub-1 ms events occur at exactly the chance rate for these baselines and are spontaneous),
modal first spike 2.7–6.2 ms, first-band width ~2 ms, repetitive firing at each cell's own preferred
rate through the pulse, and continued firing for tens of ms after light offset.

**Why these do not refute antidromic**, contrary to two earlier drafts of this document:

- **Jitter.** 2–3 ms was called disqualifying. That standard comes from *electrical* antidromic
  activation. **Optogenetic** antidromic activation is inherently jittier because spike initiation
  waits on opsin current accumulating at the terminal; published optotagging routinely accepts
  latencies to ~10 ms with jitter of a few ms.
- **Repetitive firing and the post-pulse tail.** A **10 ms sustained** pulse holds the terminal
  depolarised, so repeated antidromic invasion is expected; ChR2 closes with a ~10 ms time constant,
  so firing outlasting the light is expected too. Neither is evidence against antidromic *under this
  protocol* — they are consequences of it.
- **Collision.** Negative and numerically well powered (short/long-gap ratio **1.197, 95% CI
  [1.048, 1.367]**), but **the test is invalid with a sustained pulse**: even if the first antidromic
  spike is annihilated, continued illumination regenerates a spike inside the scoring window. This is
  the single decisive test and the protocol prevents it from being run.
- **Latency FWHM.** An intermediate draft cited 0.4–1.0 ms as supportive. That measure has **no
  discriminative power**: a sham control (same estimator, no laser) returns a median FWHM of 0.40 ms,
  and the estimate tracks bin width (0.18→1.50 ms for 0.05→0.5 ms bins). Do not cite it either way.

**Remaining tension**: 1.7–2 ms is fast even for antidromic conduction over the ~4 mm striatum→SNr
path (it implies >2 m/s in thin, poorly myelinated axons), and far too fast for any polysynaptic
route (≥3–5 ms minimum). No account is entirely comfortable with it.

**Bottom line: these are ChR2⁺ MOs-recipient striatal neurons driven by SNr/GPe illumination, most
plausibly antidromically, but pathway assignment (SNr→D1, GPe→D2) is NOT established for any
individual unit.** Treat the screen as a candidate list. Repo precedent: 162 candidates,
3 collision-confirmed.

The 17-vs-3 block asymmetry is attributed to fibre placement (experimenter-supplied). Note it is also
what you would see if the MOs-recipient population sampled here is biased toward one pathway; the data
cannot separate those.

**These are candidates, not identified cell types.** Repo precedent: 162 candidates, 3 collision-confirmed.
To settle it, acquire **short (1–2 ms) pulses** for collision testing and compare spontaneous vs evoked
waveforms from the raw AP band.

### The intra-pulse rhythm is real — and it argues the same way

Strong responders fire **several time-locked bands inside the 10 ms pulse** (cluster 317: ~2.5, 5 and
8 ms, i.e. a ~360 Hz train) plus further bands *after* the light ends. Periodic banding is a classic
artifact signature, so it was tested four ways; all four exonerate it:

| test | result |
|---|---|
| laser command modulated? | single-pulse / averaged plateau noise ratio **22.10×** vs √N = 22.4 — pure noise, no ripple |
| one shared cause across units? | first-peak latency spans **1.70–5.10 ms** (sd 0.92 ms), periods 0.5–4.6 ms — independent per unit |
| spike-sorting double-counting? | in-pulse ISIs <1 ms **2.1%** vs **1.6%** spontaneously (cluster 317) — no excess |
| physically possible for the cell? | in-pulse ISI **2.75 ms** vs that cell's own 1st-percentile spontaneous ISI **0.867 ms** — it already fires faster unaided (median ratio across units **1.58×**) |

⚠ **Do not explain this by cell type.** 19/20 responders have narrow waveforms, but **85.4% of all 670
clusters do** — Fisher OR 3.33, **p = 0.34**, and trough-to-peak is identical between responders and
non-responders (0.200 vs 0.200 ms, MWU p = 0.68). That 85% is the known BG_046 yield bias
(see `qc_celltype_yield_jun2026`), so waveform cell-typing is unreliable in this session. The valid
argument is the **unit-specific** one in the last row, which assumes no cell type at all.

The caveat that cannot be closed from these files: the laser line is a **command**, not a photodiode
on the fibre, so a driver holding a steady command while delivering modulated light is not excluded.

**Interpretation**: the banding is genuine driven firing. It does *not* support antidromic tagging —
a rhythmic burst outlasting the light is what synaptic/network drive looks like. An exemplar raster
showing this is `figures/20_optotag_exemplar.png`; the diagnostics are `21_intrapulse_rhythm.png`,
`extracted/intrapulse_rhythm.csv` and `extracted/responder_waveforms.csv`.

### No non-bursting responder exists — a further argument against antidromic

Searching all 20 responders for one that fires a **single** time-locked spike per pulse: there is no
such unit that is also reliable. Spikes-per-response tracks response probability almost perfectly,
**Spearman ρ = +0.895**:

| cluster | responds on | spikes / response | PSTH peaks |
|---|---|---|---|
| 317 | 100% | 3.73 | 4 |
| 204 | 95% | 2.34 | 6 |
| 15 | 77% | 1.61 | 5 |
| 16 (GPe) | 68% | 1.11 | 4 |
| 51 (GPe) | 14% | 1.00 | 1 |
| 44 | 8.6% | 1.00 | 1 |

The only single-spike units are the ones that almost never respond — they fire one spike because they
are barely driven, not because they belong to a distinct high-fidelity class. A genuine antidromic
population would look the opposite: **highly reliable AND single-spike with sub-millisecond jitter**.
No unit here has that combination (the most reliable single-spike unit, cluster 16, still has 4 latency
modes and 2.77 ms jitter). See `figures/23_burstiness_continuum.png` and
`extracted/responder_burstiness.csv`.

---

### Recommended optotagging protocol for future sessions

Almost every ambiguity above traces to **one design choice: 10 ms sustained pulses**. They invalidate
collision testing, manufacture repetitive firing, and produce a post-offset tail through opsin
kinetics — so the three observations that looked like evidence against antidromic are all
consequences of the protocol rather than of the biology. Changing this one parameter would have made
the session interpretable.

> References below are standard methods citations from general knowledge, not verified against a PDF
> in this repo. Check them before citing in a manuscript; several are likely already in
> `literature/synthesis-phase3-celltypes.md`.

**1. Pulse duration: 1–2 ms, not 10 ms.** ⭐ The single highest-value change. Short pulses evoke at
most one spike per pulse, make latency and jitter interpretable, and are a precondition for collision
testing. (PINP: Lima, Hromádka, Zador, *PLoS ONE* 2009; Cohen et al., *Nature* 2012.)

**2. Titrate power, and run a power series.** Use the lowest power giving reliable spikes, then
sample 3–4 levels. This is diagnostic in itself: direct/antidromic activation shows latency
shortening smoothly with power, whereas synaptic recruitment is more threshold-like. Excess power
recruits non-specifically and worsens photoelectric artifact.

**3. Add an explicit collision test.** The decisive test for antidromic invasion. Either closed-loop
(trigger a pulse at a controlled delay after a spontaneous spike) or open-loop with enough pulses to
sample the delays post hoc. Score the spike **at the antidromic latency only**, and require the
effect to be confined to gaps shorter than the collision interval (~2× conduction time + refractory)
and to recover beyond it. This is worthless with sustained pulses — hence item 1.

**4. Add a frequency-following test.** Trains at 50 / 100 / 200 Hz. Antidromic invasion follows high
rates with near-constant latency; polysynaptic drive fails and depresses. Cheap and highly diagnostic.
(Used for antidromic photo-tagging in e.g. Economo et al., *Nature* 2018.)

**5. Interleave the fibre conditions.** Randomly interleave SNr and GPe pulses rather than running
them as two long blocks. Blocked designs confound target with time, drift and opsin desensitisation —
here the two 501-pulse blocks were 763 s apart, so any slow change is perfectly confounded with target.

**6. Controls.** (a) A light-only control in an opsin-negative animal, or at a wavelength the opsin
does not absorb, to quantify photoelectric artifact. (b) Verify no light reaches the recording site —
especially relevant when a fibre sits in a structure bordering the recorded one, as GPe does with
striatum.

**7. Statistics: SALT, not a rate z-score.** The stimulus-associated spike latency test
(Kvitsiani et al., *Nature* 2013) tests whether the *latency distribution* changes, which is the right
question; it is implemented in `visdetect.analysis.optotagging`. Pair it with a **waveform criterion**
— evoked and spontaneous spike waveforms should correlate >0.9 — and report units failing either.

**8. Record the metadata.** The block→target mapping existed in nobody's file for this session and had
to be supplied from memory; the same is true of fibre coordinates and delivered power. Log all three
at acquisition. The Laser analog line is a **trigger command**, not a power monitor — if delivered
power matters, put a photodiode on the fibre and record it.

**9. Keep 200–500 pulses per condition at ≥1 s ISI.** The current session's 501 pulses at ~1.53 s is
well judged and needs no change.

---

## 11. Checklist for a new pipeline

1. **Dump and read the settings JSONs in full** before deriving any constant (§0).
2. Verify `filesize % (nSavedChans × 2) == 0`.
3. Parse the channel map from `~snsChanMap`; never hard-code order.
4. Derive thresholds from robust levels, and **reject lines whose swing is below ~0.1 V** as unconnected.
5. Assert edge pairing: line starts LOW, every rise has a fall.
6. Use `t = sample / niSampRate` (the **meta** rate). Validate by reconstructing the TPrime map, not
   by comparing to `NI_Sync.txt` (that is an identity).
7. **Merge pulses split by ≤2 samples**, then take the first pulse per trial. Confirm exact agreement
   with MATLAB on `Baseline_ON`, `Change_ON` and `Valve`.
8. De-duplicate behavioural JSON **by hash**; concatenate runs in timestamp order; confirm
   `Baseline_ON` count == trial count and `Hit+Miss+Ref == n Change_ON`.
9. Derive the lick threshold per session against reward deliveries, **and state plainly that this
   recovers the online detector's threshold on the same line** (§7). Report the refractory's effect.
10. Confirm the valve by pulse-width bimodality and one-to-one mapping onto go-trial Hits.
11. Prefer `spike_times_sec_adj.npy`; **do not assume it is sorted**; check its shape.
12. Keep the analog traces if disk allows (~1.7 GB/session) so thresholds can be re-explored.
13. Never let a guard silently convert "untestable" into "negative" — count and report both.

---

## 12. Where the working code lives

`tmpclaude-BG_046_17092025/` in this repo (git-ignored, ~6.3 GB): per-step scripts, an exploration
notebook, and figures. `README.md` there gives the run order; `make_all_figures.py` regenerates every
figure and verifies the result. Superseded approaches are under `superseded/` with reasons.
The six adversarial audits that produced these corrections left their scratch work in `_refute1/`…`_refute6/`.
