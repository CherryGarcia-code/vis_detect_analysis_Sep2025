# Stage Filtering Examples

This guide shows how to use the new `load_filtered_manifest()` function for flexible learning stage comparisons.

## Quick Reference

The helper function is available in both locations:
- **analysis_suite**: `from loader import load_filtered_manifest`
- **AI_exploration**: `from shared_config import load_filtered_manifest`

## Common Use Cases

### 1. Learning vs Expert Only (Exclude Naive)

```python
# In analysis_suite scripts:
from loader import load_filtered_manifest

manifest = load_filtered_manifest(include_stages=['Learning', 'Expert'])
```

**Use case**: Compare stable performance periods, excluding early noisy sessions.

---

### 2. Extremes Comparison (Naive vs Expert)

```python
manifest = load_filtered_manifest(
    include_stages=['Naive', 'Expert'],
    min_trials=200,  # Only well-powered sessions
    stage_specific_dprime={'Naive': 0.5, 'Expert': 1.5}  # Quality gates
)
```

**Use case**: Maximum contrast between learning stages with high-quality sessions only.

---

### 3. Early vs Expert (Merge Naive + Learning)

```python
manifest = load_filtered_manifest(
    include_stages=['Naive', 'Learning', 'Expert'],
    merge_naive_learning=True
)

# Use manifest['stage_group'] for analysis:
# - 'Learning' (was Naive or Learning)
# - 'Expert'
```

**Use case**: Compare early learning trajectory as a whole against expertise.

---

### 4. All Stages with Quality Gate

```python
manifest = load_filtered_manifest(
    exclude_stages=['Excluded', 'Disengaged'],
    min_dprime=0.8,
    min_trials=150
)
```

**Use case**: Full learning trajectory with minimum quality standards.

---

### 5. Custom Quality Thresholds per Stage

```python
manifest = load_filtered_manifest(
    include_stages=['Naive', 'Learning', 'Expert'],
    stage_specific_dprime={
        'Naive': 0.3,      # Lower bar for naive sessions
        'Learning': 0.8,   # Moderate bar
        'Expert': 1.2      # High bar for expert sessions
    }
)
```

**Use case**: Stage-appropriate quality filters (accounts for expected performance levels).

---

## Examples in analysis_suite Scripts

### Example 1: Population PSTH Comparison (Learning vs Expert)

```python
# analysis_suite/03_population/some_analysis.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loader import load_filtered_manifest, load_session
import numpy as np

# Only compare Learning and Expert sessions
manifest = load_filtered_manifest(include_stages=['Learning', 'Expert'])

psths_by_stage = {}
for stage in ['Learning', 'Expert']:
    stage_sessions = manifest[manifest['stage'] == stage]['session_name'].values
    
    all_psths = []
    for sname in stage_sessions:
        session = load_session(sname)
        # ... compute PSTH ...
        all_psths.append(psth)
    
    psths_by_stage[stage] = np.array(all_psths)

# Compare Learning vs Expert
# ...
```

---

### Example 2: Single-Unit Analysis (Extremes Comparison)

```python
# analysis_suite/02_single_unit/compare_extremes.py

from loader import load_filtered_manifest, load_glt
from config import STAGE_COLORS

# High-quality Naive vs Expert sessions only
manifest = load_filtered_manifest(
    include_stages=['Naive', 'Expert'],
    min_trials=200,
    stage_specific_dprime={'Naive': 0.5, 'Expert': 1.5}
)

# Load GLT and filter to these sessions
glt = load_glt(qc_only=False)
glt = glt[glt['Session_Date'].isin(manifest['session_name'].values)]

# Group by stage and compare
for stage in ['Naive', 'Expert']:
    stage_glt = glt[glt['stage'] == stage]
    # ... analyze units ...
```

---

## Examples in AI_exploration Scripts

### Example 1: Population Dynamics (Early vs Expert)

```python
# AI_exploration/analysis_1_learning_population_dynamics.py

from shared_config import (
    load_filtered_manifest,
    STAGE_COLORS,
    FIG_DIR
)

# Merge Naive + Learning into "Early" group
manifest = load_filtered_manifest(
    include_stages=['Naive', 'Learning', 'Expert'],
    merge_naive_learning=True,
    min_trials=150
)

# Use stage_group for comparisons
early_sessions = manifest[manifest['stage_group'] == 'Learning']['session_name'].values
expert_sessions = manifest[manifest['stage_group'] == 'Expert']['session_name'].values

print(f"Early sessions (Naive+Learning): {len(early_sessions)}")
print(f"Expert sessions: {len(expert_sessions)}")

# ... rest of analysis ...
```

---

### Example 2: HMM State Dynamics (Learning vs Expert)

```python
# AI_exploration/analysis_2_hmm_state_dynamics.py

from shared_config import (
    load_filtered_manifest,
    load_hmm_k3,
    HMM_STATE_COLORS,
    FIG_DIR
)

# Compare HMM dynamics in Learning vs Expert only
manifest = load_filtered_manifest(
    include_stages=['Learning', 'Expert'],
    min_dprime=1.0  # Minimum performance threshold
)

# Load HMM data
state_assign, per_sess, traj = load_hmm_k3()

# Filter HMM data to selected sessions
valid_sessions = set(manifest['session_name'].values)
state_assign = state_assign[state_assign['session_name'].isin(valid_sessions)]

# Merge stage info
state_assign = state_assign.merge(
    manifest[['session_name', 'stage']],
    on='session_name',
    how='left'
)

# Compare HMM state fractions between Learning and Expert
# ...
```

---

### Example 3: Encoding Analysis (Custom Quality Gates)

```python
# AI_exploration/analysis_5_encoding_across_learning.py

from shared_config import load_filtered_manifest, load_glt

# Stage-specific quality thresholds
manifest = load_filtered_manifest(
    include_stages=['Naive', 'Learning', 'Expert'],
    min_trials=150,
    stage_specific_dprime={
        'Naive': 0.3,      # Allow lower performance in naive
        'Learning': 0.8,   # Moderate threshold
        'Expert': 1.2      # Higher bar for expert
    }
)

# Load GLT and filter
glt = load_glt(qc_only=False)
glt = glt[glt['Session_Date'].isin(manifest['session_name'].values)]

# Compare encoding strength across stages
# ...
```

---

## Parameter Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_stages` | `List[str]` | Stages to include (e.g., `['Learning', 'Expert']`) |
| `exclude_stages` | `List[str]` | Stages to exclude |
| `merge_naive_learning` | `bool` | If True, create `stage_group` column with Naive→Learning |
| `min_trials` | `int` | Minimum `n_go + n_catch` to include |
| `min_dprime` | `float` | Global d' threshold |
| `stage_specific_dprime` | `Dict[str, float]` | Per-stage d' thresholds (overrides `min_dprime`) |

---

## Tips

1. **Always check session counts** after filtering to ensure you have enough data:
   ```python
   print(f"Filtered to {len(manifest)} sessions")
   print(manifest.groupby('stage').size())
   ```

2. **Use `stage_group` when merging stages**:
   ```python
   manifest = load_filtered_manifest(merge_naive_learning=True, ...)
   # Use manifest['stage_group'] NOT manifest['stage']
   ```

3. **Document your filtering choices** in comments:
   ```python
   # Extremes comparison: Naive (d'>0.5) vs Expert (d'>1.5), min 200 trials
   manifest = load_filtered_manifest(...)
   ```

4. **Start with extremes** for exploratory analysis, then expand:
   ```python
   # Step 1: Find effects with clear contrast
   manifest = load_filtered_manifest(include_stages=['Naive', 'Expert'], ...)
   
   # Step 2: If effects are found, include Learning to see trajectory
   manifest = load_filtered_manifest(include_stages=['Naive', 'Learning', 'Expert'], ...)
   ```

---

## See Also

- `src/visdetect/analysis/behavior.py::filter_manifest_by_stage()` - Core implementation with detailed docstring
- `scripts/analysis/stage_sessions.py` - How stages are assigned
- `analysis_suite/loader.py` - Data loading utilities for analysis_suite
- `AI_exploration/shared_config.py` - Configuration for AI_exploration scripts
