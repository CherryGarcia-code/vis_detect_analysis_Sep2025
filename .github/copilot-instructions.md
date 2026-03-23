# Project Context for AI Assistants

## What This Project Is
Single-subject (BG_046) electrophysiology analysis of medial striatal neurons during a visual detection task with temporal frequency (TF) drifting grating stimuli. The mouse progresses through Naive → Learning → Expert stages across ~42 recording sessions.

## Project Structure (ignore `archive/`)
- **`src/visdetect/`** — Core Python package: `core/session.py` (Session/Trial/Cluster dataclasses), `analysis/` (behavior, align, tf_pulse, constants, config), `core/qc.py` (unit selection)
- **`analysis_suite/`** — 8 analysis modules (`01_behavior/` through `08_tf_pulse/`), each producing numbered figures. Shared infra: `config.py`, `loader.py`, `utils.py`, `plotting.py`. Outputs → `figures/`, `cache/`
- **`AI_exploration/`** — Standalone analysis scripts (`analysis_1` through `analysis_7`) with `shared_config.py`
- **`scripts/analysis/tf_response/`** — Per-session TF pulse grid plotting and splitter analysis pipeline
- **`data/`** — Session pickles (`pkls/BG_046/`), staging manifest, HMM state assignments, cached TF traces (`cache/tf_traces/BG_046/*.npz`)
- **`table_output/preTPrime/`** — Grand Longitudinal Table (GLT), TF pulse CSVs
- **`config/`** — QC profiles (YAML), UnitMatch session config

## Key Data Files
| File | Purpose |
|---|---|
| `data/BG_046_staging_manifest.csv` | Session QC + stage labels (Excluded/Naive/Learning/Expert) |
| `data/pkls/BG_046/BG_046_DDMMYYYY.pkl` | Per-session neural + behavioral data |
| `data/cache/tf_traces/BG_046/*.npz` | Pre-computed TF pulse z-scored traces (42 sessions) |
| `table_output/preTPrime/Grand_Longitudinal_Table.csv` | Per-unit longitudinal metrics (TF z-scores, QC) |
| `AI_exploration/figures/waveform_celltype_labels.csv` | Cell-type labels (Narrow FSI / Broad MSN) |
| `data/hmm/BG_046/state_assignments_K3.csv` | HMM behavioral state labels per trial |

## Coding Patterns
- **analysis_suite scripts**: `from config import ...`, `from loader import ...`, `from utils import ...`, `from plotting import setup_style, save_figure`
- **sys.path**: `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
- **Session loop**: `manifest = load_staging_manifest(qc_only=True)` → iterate rows → `load_session(sname)` → process → `del sess; gc.collect()`
- **Figure saving**: `save_figure(fig, "fig_name", "subfolder")` → `analysis_suite/figures/subfolder/fig_name.png`
- **TF traces from cache**: `load_tf_traces_npz(sname)` returns dict with `t_vec`, `cluster_ids`, `fast_z`, `slow_z`, `z_max_fast`, `z_min_fast`, `z_max_slow`, `z_min_slow`

## Environments
- Windows `.venv`: `.venv\Scripts\python.exe` (PowerShell)
- Conda `copilot_ephys`: activate in bash terminal

## Key Constants
| Constant | Value | Location |
|---|---|---|
| `CHANGE_SIZES` | [1.25, 1.35, 1.5, 2.0, 4.0] | `visdetect.analysis.config` |
| `DEFAULT_Z_THRESH_TF` | 3.0 | `visdetect.analysis.constants` |
| `TF_PULSE_PRE_WINDOW` | (-0.4, 0.0) s | `visdetect.analysis.constants` |
| `TF_PULSE_POST_WINDOW` | (0.0, 0.5) s | `visdetect.analysis.constants` |
| `TF_FAST_THRESH_LOG2` | 0.25 | `visdetect.analysis.constants` |
| `TF_SLOW_THRESH_LOG2` | -0.25 | `visdetect.analysis.constants` |
| `STAGE_ORDER` | ["Learning", "Expert"] | `visdetect.analysis.config` |

## Scientific Context
- **Task**: Mouse detects changes in temporal frequency (TF) of a drifting grating. Go trials have change_size > 1.0; catch trials ≈ 1.0.
- **Baseline TF pulses**: During baseline period, TF fluctuates stochastically (~50 ms updates). Periods where log₂(TF) crosses ±0.25 are "fast" or "slow" pulse events. These are motor-confound-free sensory probes.
- **TF-responsive**: A unit is TF-responsive if its post-pulse z-score (relative to pre-pulse baseline) has |z| ≥ 3.0 for either fast or slow pulses.
- **Cell types**: Narrow-spiking = putative FSI; Broad-spiking = putative MSN/Projection neuron. Classified by trough-to-peak waveform timing.
- **Framework reference**: Khilkevich & Lohse, Nature 2024 (brain-wide dynamics, ~250 ms integration timescale)
