# driftplots QC spike — instructions (run in a separate chat)

**Status:** ready to run. Self-contained. Hand this whole file to a fresh chat.

## Why we're doing this

The track-curation pipeline gates cross-session links partly on **probe depth**. We
established (via `scripts/pipelines/tracking/diagnose_intersession_drift.py`, the
amplitude-depth *fingerprint* method) that **whole-probe inter-session drift on BG_046
is ~0** (best rigid shift 0 µm for all 41 session pairs, fingerprint alignment corr 0.88),
so the curation gate now uses **raw depth** by default. See
`memory/neuron_tracking_may2026.md` ("Track-curation pipeline SHIPPED") and the figure
`FIGURES/tracking_qc/intersession_drift.png`.

That diagnostic uses **mean waveforms** (no time axis), so it can only see *between*-session
drift. This QC spike uses **[driftplots](https://github.com/neuroinformatics-unit/driftplots)**
(NeuroInformatics Unit, UCL) to add two things it can't:

1. **Within-session driftmaps** (per-spike depth vs **time**, amplitude-colored) — confirm
   there's no intra-session drift hiding behind the stable between-session numbers.
2. **A maintained cross-session visual comparison** (`MultiSessionDriftmapWidget`, linked
   zoom) — an independent eyeball check of the "~0 cross-session drift" claim, on the actual
   spikes rather than mean-waveform fingerprints.

**Scope:** this is QC/visualization only. driftplots does **not** output a drift offset
number; it will not change the curation code. Its job is to *confirm* (or challenge) the
raw-depth decision and surface any per-session drift surprises.

## What driftplots is (quick API)

```python
from driftplots import DriftPlotter, MultiSessionDriftmapWidget, get_amplitudes
```

- `DriftPlotter(kilosort_output_dir)` — also accepts a SpikeInterface `SortingAnalyzer`.
  Reads spike depths (KS4: `spike_positions.npy`) and amplitudes.
- `.drift_map_plot_matplotlib(add_histogram_plot=True, weight_histogram_by_amplitude=True)`
  → static driftmap (depth vs time, amplitude color) + an amplitude-weighted depth
  histogram on the side (this histogram is the per-spike analogue of our fingerprint).
- `.drift_map_plot_interactive(good_units_only=True, title=...)` → interactive; click a spike
  to see its template. Docs: *"useful for checking the alignment of two sorted sessions."*
- `MultiSessionDriftmapWidget([plot1, plot2, ...]).plot()` → multiple sessions side by side,
  linked zoom.
- `get_amplitudes([dirs...], good_units_only=True, concatenate=True)` → pool amplitudes
  across sessions so you can set a *consistent* color/filter scale on every panel.

## Data location (BG_046 Kilosort4 outputs)

Each session is a standard Phy/KS4 output directory on the X: drive:

```
X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Processed data/BG_046_<DATE>/Kilosort&Phy/BG_046_<DATE>_g0_imec0/
```

`<DATE>` is `DDMMYYYY` (e.g. `01072025`). That dir contains `params.py`, `spike_times.npy`,
`spike_clusters.npy`, `spike_positions.npy`, `templates.npy`, `amplitudes.npy`,
`channel_positions.npy`, `cluster_group.tsv` — everything `DriftPlotter` needs.

Pick ~4 sessions spanning the training arc (Naive → Expert). Suggested first pass:

| Role | Session (`<DATE>`) |
|------|--------------------|
| Early (Naive/Learning) | `23062025` or `30062025` |
| Mid | `28072025` or `04082025` |
| Late (Expert) | `16092025` or `17092025` |

For exact stage labels use the staging manifest (`from visdetect.suite.loader import
load_staging_manifest`) — but any early/mid/late triplet is fine for a first look.

## Environment

driftplots is a separate pip package; install it into a throwaway/QC venv (don't pollute the
`unitmatch` conda env). From the repo root:

```bash
# Option A: reuse the project venv
.venv\Scripts\python.exe -m pip install driftplots

# Option B: a clean QC venv
py -m venv .venv_driftplots
.venv_driftplots\Scripts\python.exe -m pip install driftplots matplotlib numpy
```

Interactive mode needs a GUI backend. On Windows, run the interactive widgets from a
**Jupyter notebook** (`%matplotlib widget` or `qt`) or a Python REPL with a Qt/Tk backend.
The **static** `drift_map_plot_matplotlib` works headless (`matplotlib.use("Agg")`) and is the
safest for a first scripted pass.

## Step 1 — static driftmaps per session (headless, scripted)

Save as `scripts/pipelines/tracking/driftplots_qc.py` (or a notebook). This renders one static
driftmap + amplitude-weighted depth histogram per session, with a **shared amplitude scale**
so panels are comparable.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from driftplots import DriftPlotter, get_amplitudes

BASE = Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Processed data")
OUT  = Path("FIGURES/tracking_qc/driftplots"); OUT.mkdir(parents=True, exist_ok=True)

def ks_dir(date):
    return BASE / f"BG_046_{date}" / "Kilosort&Phy" / f"BG_046_{date}_g0_imec0"

dates = ["23062025", "28072025", "16092025"]      # early / mid / late — edit as desired
dirs  = [str(ks_dir(d)) for d in dates]

# Shared amplitude scale across all sessions (so color/contrast is comparable)
amps = get_amplitudes(dirs, good_units_only=True, concatenate=True)
lo, hi = np.percentile(amps, (0, 95))

for date, d in zip(dates, dirs):
    plotter = DriftPlotter(d)
    fig = plotter.drift_map_plot_matplotlib(
        add_histogram_plot=True,
        weight_histogram_by_amplitude=True,
        filter_amplitude_mode="absolute",
        filter_amplitude_values=(lo, hi),
        amplitude_cmap_scaling=(lo, hi),
        n_color_bins=25,
    )
    fig.suptitle(f"BG_046 {date}")
    fig.savefig(OUT / f"driftmap_{date}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"driftmap_{date}.png")
```

> Note: argument names above (`add_histogram_plot`, `weight_histogram_by_amplitude`,
> `filter_amplitude_mode`, `filter_amplitude_values`, `amplitude_cmap_scaling`, `n_color_bins`)
> come from the driftplots docs. If the installed version differs, run
> `help(DriftPlotter.drift_map_plot_matplotlib)` and adjust — the kwargs may have been renamed
> between beta releases.

**What to look for (within-session):** in each driftmap the spike cloud should be roughly
**horizontal over time** — no slow vertical ramp or sudden jumps. KS4's own drift correction
should already have flattened it; this confirms it. A visible ramp/step in any session is a
flag worth noting (it would mean that session's depths are time-varying and the single
per-session depth we feed the gate is a smear).

## Step 2 — cross-session alignment (interactive, in a notebook)

```python
from driftplots import DriftPlotter, MultiSessionDriftmapWidget

dates = ["23062025", "28072025", "16092025"]
panels = []
for date in dates:
    d = f"X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Processed data/BG_046_{date}/Kilosort&Phy/BG_046_{date}_g0_imec0"
    panels.append(DriftPlotter(d).drift_map_plot_interactive(title=date, good_units_only=True))

MultiSessionDriftmapWidget(panels).plot()
```

**What to look for (cross-session):** the dense depth **bands** (rows where many big units sit)
should land at the **same depths** across the three panels. If they do, that's an independent,
spike-level confirmation of the fingerprint's "~0 drift" result and the raw-depth gate is well
justified. If a band is clearly shifted in one session, note the session + approximate µm shift
and report back — it would mean per-session depth needs correction after all (and we'd revisit
the gate's `--drift-source`).

## Step 3 (optional) — collate all sessions into a QC PDF

The docs mention collating driftmaps into a PDF across many sessions. Loop Step 1 over **all**
manifest sessions and write each figure into a single multi-page PDF
(`matplotlib.backends.backend_pdf.PdfPages`) for a one-glance QC sweep of the whole dataset.

## What to report back

1. Any session whose **within-session** driftmap shows a ramp/step (vs. flat). List session + rough µm.
2. Whether the **cross-session** depth bands align across early/mid/late (yes/no; if no, which session + µm).
3. Overall verdict: does the spike-level view **confirm** the ~0-drift / raw-depth decision, or challenge it?

If everything looks flat and aligned (expected), no code change is needed — this just hardens
the raw-depth gate decision. If something is off, ping back and we'll reconsider `--drift-source`
(the curation runner already supports `none` / `fingerprint` / `match`).
