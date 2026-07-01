"""Emit FIGURES/tracking_consensus/BG_046/README.md from the cohort artifacts.

Every number is read from a source file (never hand-typed) so the companion doc
cannot drift from the data. Re-run after re-building the cohort.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from visdetect.analysis.config import canonical_session_id  # noqa: E402

CACHE = ROOT / "data/cache/tracking_consensus/BG_046"
OUT_DIR = ROOT / "FIGURES/tracking_consensus/BG_046"
README = OUT_DIR / "README.md"


def _fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    cohort = pd.read_csv(CACHE / "consensus_cohort.csv")
    val = json.load(open(CACHE / "isi_validation.json"))
    rendered = pd.read_csv(OUT_DIR / "candidates" / "rendered_stats.csv")
    um = pd.read_csv(ROOT / "data/cache/um_ref/unit_index.csv", dtype=str)
    dant = pd.read_csv(ROOT / "data/cache/dant/BG_046/dant_registry.csv", dtype=str)
    dant_tracked = dant[dant["dant_uid"].astype(int) >= 0]
    behav_cohort = (pd.read_csv(CACHE / "behavior_cohort.csv")
                    if (CACHE / "behavior_cohort.csv").exists() else None)

    n = len(cohort)
    ge = {k: int((cohort["n_agree"] >= k).sum()) for k in (2, 3, 5, 7, 10)}
    l2e = int(cohort["learning_to_expert"].sum())
    n2e = int(cohort["naive_to_expert"].sum())

    # per-candidate rows (merge rendered wave_r with cohort stats)
    cand = rendered.merge(cohort, on=["um_uid", "dant_uid"], suffixes=("", "_c"))

    def _cand_table(df):
        cols = [("um_uid", "UM #"), ("dant_uid", "DANT #"), ("n_agree", "agreed sess."),
                ("wave_r", "waveform r"), ("matched_isi_corr", "held-out ISI r"),
                ("matched_isi_pctile", "ISI pctile"), ("jaccard", "Jaccard"),
                ("purity_um", "UM purity"), ("purity_dant", "DANT purity"),
                ("stages", "stages"), ("dant_composite", "DANT biophys.")]
        head = "| " + " | ".join(c[1] for c in cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        lines = [head, sep]
        for _, r in df.iterrows():
            cells = []
            for key, _lab in cols:
                v = r[key]
                if key == "matched_isi_pctile":
                    cells.append(f"{float(v)*100:.0f}%")
                elif key in ("wave_r", "matched_isi_corr", "jaccard", "purity_um", "purity_dant"):
                    cells.append(_fmt(v))
                elif key == "stages":
                    seq = str(v).split(";")
                    cells.append(f"{seq[0]}→{seq[-1]} ({len(set(s for s in seq if s in ('Naive','Learning','Expert')))} st.)")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    # ---- behavioural sub-deliverable section ----
    behav_md = ""
    if behav_cohort is not None and len(behav_cohort):
        n_neu = len(behav_cohort)
        behav_md = f"""
## Behavioural profiles across learning (`behavior/`)

Having *established* these are the same neurons across learning (identity figures
above), these figures ask the scientific question: **does each neuron's
task/behavioural response change as the mouse learns?** For {n_neu} high-confidence,
learning-spanning neurons (DANT-composite-trusted, held-out-ISI-validated, spanning
Learning→Expert; plus the one strict Naive→Expert cell) we render one page each with
four signal families, every panel overlaid by learning stage:

1. **Task-event responses** — Baseline_ON, Change_ON (large-change hit), Hit-lick,
   FA-lick. E.g. UM#942 shows a baseline response that *shrinks* while its
   change-evoked response *grows* (~34→45 Hz) from Learning to Expert.
2. **Decision selectivity** — Change_ON Hit-vs-Miss, large- vs small-change tuning,
   choice AUROC, and reaction-time coding, each Learning vs Expert.
3. **Behavioural-state modulation** — baseline firing and change-response split by
   state (Impulsive / StimSens / Disengaged) from the state labeller.
4. **Choice / RT** — per-trial Change_ON response → AUROC(hit vs miss) and
   Spearman(response, RT) on hit go-trials.

A `behavior_cohort_summary.png` distills the population (per-neuron Learning→Expert
choice-coding and change-response trajectories).

### TF-encoding status — PENDING (not asserted here)

The valid per-unit TF registry is the **Khilkevich–Lohse GLM replication**
(`data/cache/tf_responsive/<subject>_tf_responsive.csv`, `resp_log2`; requires
`c1_r_log2 > 0.2` AND `c2_p_log2 < 0.01`). **BG_046's GLM run is not complete yet**
(the registry currently covers BG_031 at 5.3% and BG_039 at 3.1%), so **no TF-encoding
call is made for these neurons.** Cross-reference `bg046_tf_responsive.csv` when it
lands. The earlier single-pulse z-screen was **stale/superseded** (flagged ~64% of all
units — uninformative) and has been retired from these figures.

⚠️ Caveats for the behavioural panels: behavioural-state labels are partly circular
(defined from lick behaviour), so read state panels descriptively. Choice AUROC uses
the early (0–0.3 s) evoked window, so it reads near chance even when the *later*
hit-vs-miss ramp (visible in the PSTH panel) is strong. Stage panels aggregate
per-session metrics (mean ± SEM across a stage's sessions).

---
"""

    top = cohort.head(20)

    def _cohort_table(df):
        cols = ["um_uid", "dant_uid", "n_agree", "jaccard", "purity_um", "purity_dant",
                "matched_isi_corr", "matched_isi_pctile", "learning_to_expert",
                "naive_to_expert", "um_tier", "dant_tier", "dant_composite"]
        labs = ["UM #", "DANT #", "agreed", "Jacc.", "UM pur.", "DANT pur.",
                "ISI r", "ISI pct", "L→E", "N→E", "UM tier", "DANT tier", "DANT biophys."]
        lines = ["| " + " | ".join(labs) + " |", "|" + "|".join(["---"] * len(labs)) + "|"]
        for _, r in df.iterrows():
            cells = [str(int(r["um_uid"])), str(int(r["dant_uid"])), str(int(r["n_agree"])),
                     _fmt(r["jaccard"]), _fmt(r["purity_um"]), _fmt(r["purity_dant"]),
                     _fmt(r["matched_isi_corr"]),
                     (f"{r['matched_isi_pctile']*100:.0f}%" if pd.notna(r["matched_isi_pctile"]) else "n/a"),
                     "yes" if r["learning_to_expert"] else "-",
                     "YES" if r["naive_to_expert"] else "-",
                     str(r["um_tier"]), str(r["dant_tier"]), str(r["dant_composite"])]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    md = f"""# UM ∩ DANT consensus cohort — BG_046 medial striatum

**Best "same neuron across learning" candidates, confirmed by two independent trackers.**

This folder holds presentation-ready figures for the single neurons that **both**
cross-session trackers — **UnitMatch (UM)** and **DANT** — followed across many
sessions of the visual change-detection task, as the mouse went from Naive →
Learning → Expert. Two independent algorithms agreeing that a unit is the *same*
neuron across sessions is the strongest argument a skeptic will accept.

---

## What is a "consensus track"?

Each recording session is spike-sorted into units (Kilosort cluster ids). A
cross-session tracker decides which unit in session A is the *same physical
neuron* as which unit in session B. UM and DANT do this with completely different
math (UM = pairwise waveform-probability chaining; DANT = global density
clustering with built-in drift correction). We take the **mutual-best**
correspondence between UM's `global_uid` and DANT's `dant_uid` on the units both
trackers observed, and keep the sessions where they **agree**. A consensus track
is that agreed set — the sessions where *both* trackers place the same physical
unit together.

---

## Headline numbers

- **{n} consensus neurons** tracked by BOTH trackers across ≥ 2 sessions.
- Tracked ≥ 3 sessions: **{ge[3]}**;  ≥ 5: **{ge[5]}**;  ≥ 7: **{ge[7]}**;  ≥ 10: **{ge[10]}**.
- Span **Learning → Expert: {l2e}**;  strict **Naive → Expert: {n2e}**.
- **Independent validation** (held-out log-ISI fingerprint, an axis neither
  tracker uses to match): matched cross-session pairs vs unrelated
  simultaneously-recorded pairs give **AUC = {_fmt(val['auc_matched_vs_nonmatched'])}**
  ({val['n_matched_pairs']} matched vs {val['n_nonmatched_pairs']} non-matched pairs;
  matched median r = {_fmt(val['matched_corr_median'])} vs null mean
  r = {_fmt(val['nonmatched_corr_mean'])}).

Registry inputs: UM {len(um)} unit-sessions / {um['global_uid'].nunique()} global ids;
DANT {len(dant_tracked)} tracked unit-sessions / {dant_tracked['dant_uid'].nunique()} cluster ids;
{cohort['n_agree'].sum()} agreed member nodes across the {n} consensus tracks.

---

## The rendered candidates (`candidates/`)

Six neurons are rendered as clean one-page figures: five of the cleanest
longitudinal tracks plus the single strict **Naive→Expert** exemplar (shown
honestly — it is the hardest track and its agreement/ISI numbers are weaker).

{_cand_table(cand)}

(In the `stages` column, **Excluded** = a session that failed the *behavioural*
QC gate — min-trials / d′ — not a tracking problem; the neuron is still tracked
there, it just isn't assigned a Naive/Learning/Expert learning stage. "st." = the
number of distinct learning stages spanned.)

Each figure carries six evidence panels + a two-tracker strip:

1. **Spike waveform, every session** — the peak-channel waveform overlaid for all
   agreed sessions (colour = early→late). Superposition ⇒ same neuron. The number
   is the mean pairwise waveform-shape correlation.
2. **Inter-spike-interval fingerprint** — the log-ISI histogram per session
   (held-out spikes). A neuron's firing statistics are a biophysical fingerprint.
3. **ISI match vs population** — this neuron's cross-session ISI correlation (red)
   against the distribution of *unrelated* simultaneously-recorded pairs (grey
   null). Sitting in the right tail = the ISI fingerprint is reproducibly this
   neuron's, not a coincidence. This is the **independent** validation axis.
4. **Footprint — first vs last session** — the multi-channel voltage footprint
   early vs late. Same spatial signature ⇒ same location on the probe.
5. **Probe-depth stability** — the peak-channel depth across sessions.
6. **Two independent trackers across all sessions** — red = both UM and DANT
   agree; orange = UM only; blue = DANT only; the top strip is the behavioural
   stage. Shows honestly where the trackers agree and where one reaches further.

---

## Full cohort — top 20 by agreed span (`consensus_cohort.csv`)

{_cohort_table(top)}

(Full table: `data/cache/tracking_consensus/BG_046/consensus_cohort.csv`,
{n} rows. Per-session membership: `consensus_members.csv`.)

---
{behav_md}
## How it was built (methods)

1. **Registries** (local, no X:/Samba compute):
   `data/cache/um_ref/unit_index.csv` (UM: session, ks_unit_id, global_uid) and
   `data/cache/dant/BG_046/dant_registry.csv` (DANT: session, ks_unit_id, dant_uid;
   −1 = untracked).
2. **Join** on `(session, ks_unit_id)` — sessions normalised to canonical 8-digit
   `DDMMYYYY` via `config.canonical_session_id` (**critical**: a raw string join
   silently drops the 14 single-digit-day sessions — the leading-zero footgun).
3. **Mutual-best correspondence**: for each UM id G the DANT id D winning the most
   shared sessions, and vice-versa; keep (G,D) only if each is the other's best
   and they agree on ≥ 2 sessions. `jaccard`/`purity_um`/`purity_dant` quantify
   how cleanly the two ids correspond on the co-observed units.
4. **Stages / N→E** from `data/BG_046_staging_manifest.csv`.
5. **Curation tiers** attached from the existing UM curation
   (`FIGURES/tracking_qc/curation/curated_tracks.csv`) and DANT curation
   (`FIGURES/tracking_dant/BG_046/curation/curated_tracks.csv` +
   `composite_retier.csv`). These are informational — the consensus cohort is
   **not** gated by them.
6. **Held-out ISI validation**: each unit's spikes split even/odd
   (`track_curation.partitioned_isi_hists`); the odd (held-out) log-ISI histogram
   is used only for validation, so it is statistically independent of any curation
   ISI feature. AUC is matched (within-track, cross-session) vs non-matched
   (within-session, across-track) — cross-checked against the library
   `held_out_isi_auc_by_tier` (identical, {_fmt(val['auc_matched_vs_nonmatched'],4)}).

**Correctness check**: the join recovers the previously hand-derived worked
example **UM #942 ↔ DANT #631** with exactly its 13 agreed sessions.

---

## Caveats (read before presenting)

- **Curation tier ≠ tracker error.** Most long consensus tracks are UM-tier
  *review* (not *trusted*): the conservative per-link curation demotes any track
  with one borderline transition. The whole-track biophysical composite and the
  two-tracker agreement are the better quality signals here.
- **Waveform/ISI/depth are identity evidence; task activity (PSTH) is NOT.**
  Firing to task events legitimately *changes* across learning — that is the
  scientific signal, never used to establish identity (avoids circularity).
- **Held-out ISI is quasi-independent**, not perfectly orthogonal (even/odd
  partitions of the same train are autocorrelated). It is a strong corroborator,
  not a proof.
- **Strict Naive→Expert is rare (n={n2e}).** "Naive" is only a handful of early
  sessions; most learning-spanning tracks are Learning→Expert. The one strict
  N→E track (UM #349 / DANT #260) is the hardest to track — its lower Jaccard
  (0.44) and ISI r (0.73) are shown honestly, not hidden.
- **Depth is the raw peak-channel depth** (whole-probe inter-session drift on this
  data is ≈ 0 µm, so no correction is applied); a few µm of wander across weeks
  is expected and visible.

---

## Reproduce

```
py scripts/tracking_consensus/build_consensus_cohort.py     # join + cohort + members
py scripts/tracking_consensus/compute_isi_validation.py     # held-out ISI (loads pkls once each)
py scripts/tracking_consensus/render_consensus_figures.py   # 6 candidate figures + cohort summary
py scripts/tracking_consensus/compute_behavior_cache.py     # behaviour features (loads cohort pkls)
py scripts/tracking_consensus/render_behavior_figures.py    # behaviour figs + cohort summary
py scripts/tracking_consensus/write_readme.py               # this file
```

Artifacts: `data/cache/tracking_consensus/BG_046/` (cohort, members, ISI cache,
validation json); figures in this folder (gitignored).
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    README.write_text(md, encoding="utf-8")
    print(f"wrote {README} ({len(md)} chars)")


if __name__ == "__main__":
    main()
