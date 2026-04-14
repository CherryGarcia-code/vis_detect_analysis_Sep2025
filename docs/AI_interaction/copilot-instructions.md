# # AI Assistant Instructions for Visual Detection Analysis

**Purpose**: Comprehensive instructions for AI assistants working on this neuroscience analysis repository.
**Priority**: This is the **canonical** instruction file for the project.

---

## 🎯 **Project Context**

### **What This Project Is**
Single-subject (BG_046) electrophysiology analysis of medial striatal neurons during a visual detection task with temporal frequency (TF) drifting grating stimuli. The mouse progresses through Naive → Learning → Expert stages across ~42 recording sessions.

### **Scientific Framework**
- **Reference**: Khilkevich & Lohse, Nature 2024 (brain-wide dynamics, ~250 ms integration timescale)
- **Task**: Mouse detects changes in temporal frequency (TF) of a drifting grating
- **Trial Types**: Go trials (change_size > 1.0), Catch trials (≈ 1.0)
- **Key Innovation**: TF pulses during baseline provide motor-confound-free sensory probes

---

## 📁 **Project Structure**

### **Core Architecture**
- **`src/visdetect/`** — Core Python package
  - `core/session.py` — Session/Trial/Cluster dataclasses
  - `analysis/` — behavior, align, tf_pulse, constants, config
  - `core/qc.py` — unit selection and quality control
- **`analysis_suite/`** — Production figure generation (43 scripts → Figs 1-43)
  - Shared infrastructure: `config.py`, `loader.py`, `utils.py`, `plotting.py`
  - Outputs: `figures/`, `cache/`
- **`scripts/`** — Pipeline utilities, data conversion, QC tools
- **`docs/`** — Documentation, manuals, AI interaction plans

### **Key Data Files**
| File | Purpose |
|------|---------|
| `data/BG_046_staging_manifest.csv` | Session QC + stage labels |
| `data/pkls/BG_046/*.pkl` | Per-session neural + behavioral data |
| `data/cache/tf_traces/BG_046/*.npz` | Pre-computed TF pulse z-scored traces |
| `table_output/preTPrime/Grand_Longitudinal_Table.csv` | Per-unit longitudinal metrics |
| `data/hmm/BG_046/state_assignments_K3.csv` | HMM behavioral state labels |

---

## 🤖 **Agent Roles & Personas**

Use these specialized personas when helpful:

### **DataWrangler**
- **Purpose**: Parse and normalize session data
- **Tools**: pandas, numpy, data validation
- **Focus**: JSON → dataclasses, data quality, format conversion

### **NeuroAnalyst**
- **Purpose**: Statistical and neural analyses
- **Tools**: scipy, scikit-learn, statistical testing
- **Focus**: PETHs, population dynamics, significance testing

### **VizBot**
- **Purpose**: Publication-quality visualizations
- **Tools**: matplotlib, seaborn, figure formatting
- **Focus**: Multi-panel figures, color consistency, publication standards

---

## 💻 **Coding Conventions**

### **Modern Python Practices**
```python
# Use type hints, docstrings, and clear naming
def analyze_population_response(
    session: Session,
    event: str = "Change_ON"
) -> Tuple[np.ndarray, List[int]]:
    """Analyze population response to behavioral events."""
    pass
```

### **Analysis Suite Patterns**
```python
# Standard import pattern
from config import STAGE_COLORS, DEFAULT_BIN_SIZE
from loader import load_staging_manifest, load_session
from utils import get_good_cluster_ids, build_population_tensor
from plotting import setup_style, save_figure

# Standard session processing loop
manifest = load_staging_manifest(qc_only=True)
for _, row in manifest.iterrows():
    sess = load_session(row['session_name'])
    # ... process session ...
    del sess; gc.collect()  # Memory management
```

### **Environment Setup**
```bash
# Development setup
pip install -e .

# Multi-subject configuration
export VISDETECT_SUBJECT=BG_046  # Default

# Cross-platform Python
python -c "import visdetect; print('OK')"  # Test install
```

---

## 🧪 **Scientific Context & Constants**

### **Key Parameters**
| Constant | Value | Purpose |
|----------|-------|---------|
| `CHANGE_SIZES` | [1.25, 1.35, 1.5, 2.0, 4.0] | Go-trial TF ratios |
| `DEFAULT_Z_THRESH_TF` | 3.0 | TF responsiveness threshold |
| `TF_PULSE_PRE_WINDOW` | (-0.4, 0.0) s | Baseline for TF pulses |
| `TF_PULSE_POST_WINDOW` | (0.0, 0.5) s | Response window |
| `STAGE_ORDER` | ["Learning", "Expert"] | Analysis stages |

### **TF Pulse Analysis**
- **Fast pulses**: log₂(TF) > 0.25 (higher temporal frequency)
- **Slow pulses**: log₂(TF) < -0.25 (lower temporal frequency)
- **TF-responsive unit**: |z-score| ≥ 3.0 for fast OR slow pulses
- **Motor-confound-free**: Occurs during baseline, no behavioral requirement

### **Cell Type Classification**
- **Narrow-spiking**: Putative Fast-Spiking Interneurons (FSIs)
- **Broad-spiking**: Putative Medium Spiny Neurons (MSNs)
- **Classification**: Trough-to-peak waveform timing

---

## 📋 **Priority Rules**

When instructions conflict, follow this order:

1. **Safety & Privacy**: Never expose credentials or sensitive data
2. **Explicit user request** in the current conversation
3. **This file** (`docs/AI_interaction/copilot-instructions.md`)
4. **Project documentation** (`CLAUDE.md`, `README.md`, `RUNNING.md`)
5. **Analysis manual** (`docs/ANALYSIS_MANUAL.md`)

---

## 🔧 **Best Practices**

### **Code Quality**
- ✅ Use proper package imports: `from visdetect.analysis.utils import ...`
- ✅ Follow memory management: `del sess; gc.collect()`
- ✅ Use canonical constants from `visdetect.analysis.constants`
- ✅ Handle missing data gracefully
- ⚠️ Avoid hardcoded paths or subject names

### **Analysis Standards**
- ✅ Use shared baseline normalization for cross-condition comparisons
- ✅ Apply proper statistical corrections (FDR, permutation tests)
- ✅ Include effect sizes with p-values
- ✅ Use consistent color palettes (`STAGE_COLORS`, `OUTCOME_COLORS`)
- ⚠️ Never average raw firing rates across neurons without normalization

### **Documentation**
- ✅ Use clear docstrings with parameter descriptions
- ✅ Include figure captions that explain the analysis
- ✅ Link to relevant documentation sections
- ✅ Document any assumptions or limitations

---

## 📖 **Key Documentation**

- **Setup**: `README.md` → `RUNNING.md` → `analysis_suite/README.md`
- **Analysis Guide**: `CLAUDE.md` → `docs/ANALYSIS_MANUAL.md`
- **Pipeline Monitoring**: `docs/AI_interaction/concat-sort/MONITORING_GUIDE.md`
- **All Plans & Docs**: `docs/DOCUMENTATION_INDEX.md`

---

*This is the canonical AI assistant instruction file. Other instruction files should point here to avoid duplication.*
-----------------
- Use `data/RAW_SESSION_SCHEMA_BG_031_260325.json` for field names and expected structures.
- Consult `README.md` for experimental context and research questions.
- If a field or behavior is ambiguous, ask the user or suggest a reasonable default and document the assumption in your change.

Allowed and Disallowed Actions
------------------------------
- Allowed (with caution): read repository files, create/modify code and notebooks, run local lint/tests if requested, and suggest environment changes.
- Disallowed: making external network requests or disclosing secrets; do not publish private data to external services without explicit user consent.

Preferred Outputs
-----------------
- Small, self-contained changes that are easy to review.
- When modifying code, include or update a minimal test demonstrating the change where practical.
- For notebooks, provide a clean scaffold (imports, env checks, small example) and keep examples reproducible using local files.

Examples of Useful Tasks
------------------------
- Create Python dataclasses for session/trial formats derived from the JSON schema.
- Add a notebook scaffold that imports core libraries and loads a small sample of the JSON schema.
- Implement PETH computation and a raster+PSTH figure for a single session.

If You Need Clarification
-------------------------
- Ask the user for missing details (e.g., which files to touch, whether to overwrite a notebook). When in doubt, propose a short plan and request approval before large changes.

Maintenance
-----------
- Keep this file short and prioritized. If you update `PROMPT.md` or `AGENTS.md`, consider consolidating important changes here.

Contact / Notes
---------------
If you want this file renamed, or to change the priority order, tell me and I'll update it.
