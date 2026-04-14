# Codebase Reorganization Plan

**Created**: March 31, 2026
**Priority**: High - Address technical debt and improve maintainability

## 🎯 Goals

1. **Centralize utilities** - Move shared code to canonical locations
2. **Remove hard-coded dependencies** - Parameterize subject and paths
3. **Standardize imports** - Use proper package imports throughout
4. **Consolidate archives** - Single location for legacy code
5. **Improve documentation** - Clear setup and usage instructions

## 📋 Task Breakdown

### Phase 1: Quick Wins (1-2 days)
- [x] **Task 1**: Consolidate legacy code into `archive/`
- [x] **Task 2**: Add README documentation for key directories
- [ ] **Task 3**: Fix problematic filenames (spaces, special characters)

### Phase 2: Core Infrastructure (2-4 days)
- [ ] **Task 4**: Move shared utilities to `src/visdetect/analysis/utils.py`
- [ ] **Task 5**: Remove `sys.path` manipulation, use proper package imports
- [ ] **Task 6**: Replace hard-coded SUBJECT and drive paths with configuration

### Phase 3: Polish and Testing (1-2 days)
- [ ] **Task 7**: Add developer documentation and contribution guidelines
- [ ] **Task 8**: Run full test suite and validate all imports work
- [ ] **Task 9**: Update CI/automation to use new structure

## 🔧 Implementation Details

### Task 4: Centralize Shared Utilities

**Current Problem**:
- `analysis_suite/utils.py` contains utilities used across 30+ scripts
- Functions like `bootstrap_ci`, `compute_zscore_normalized`, `get_good_cluster_ids` should be in the library

**Solution**:
```bash
# Create new canonical utilities module
touch src/visdetect/analysis/utils.py

# Move functions from analysis_suite/utils.py:
# - bootstrap_ci, permutation_test, fdr_correct
# - compute_zscore_normalized, compute_baseline_subtracted
# - build_population_tensor, smooth_psth
# - get_good_cluster_ids, load_kept_ids

# Update imports in ~30 analysis_suite scripts:
# FROM: from utils import bootstrap_ci
# TO:   from visdetect.analysis.utils import bootstrap_ci
```

### Task 5: Fix Import System

**Current Problem**:
- Scripts use `sys.path.insert(0, '../src')` then `from config import ...`
- Fragile and prevents proper package installation

**Solution**:
```python
# Remove from analysis_suite/config.py:
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

# Replace with proper imports:
from visdetect.analysis.config import *
from visdetect.analysis.constants import *

# Update RUNNING.md to require: pip install -e .
```

### Task 6: Remove Hard-coded Dependencies

**Current Problem**:
```python
# In src/visdetect/analysis/config.py
SUBJECT: str = "BG_046"  # Hard-coded!

# In scripts/pipelines/concat_sort/build_concat_pkls.py
FINAL_OUTPUT = "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/..."  # Hard-coded!
```

**Solution**:
```python
# Make config parameterizable
SUBJECT: str = os.getenv("VISDETECT_SUBJECT", "BG_046")

# Add CLI arguments to pipeline scripts
parser.add_argument("--subject", default=os.getenv("VISDETECT_SUBJECT", "BG_046"))
parser.add_argument("--data-root", default=os.getenv("VISDETECT_DATA_ROOT", "X:/public/..."))
```

## 📁 File Movements Required

### Archive Consolidation
```bash
# Move legacy code to single archive
mv scripts/scripts_archive/* archive/scripts_archive/
mv src/visdetect/analysis/archive/* archive/src_analysis_archive/

# Remove duplicate helper versions
rm archive/scripts_archive/database/vis_detect_helpers_v[6-8].py
# Keep only v9 as reference
```

### Utility Centralization
```bash
# Move shared utilities
analysis_suite/utils.py → src/visdetect/analysis/utils.py (selected functions)

# Update imports in these files:
analysis_suite/01_behavior/*.py
analysis_suite/02_single_unit/*.py
analysis_suite/03_population/*.py
analysis_suite/04_decoding/*.py
# ... (all analysis_suite scripts)
```

## ⚠️ Risk Mitigation

### Before Starting
1. **Commit current state** - Clean working directory
2. **Run tests** - Establish baseline: `python -m pytest tests/`
3. **Document current imports** - Grep for import patterns

### During Migration
1. **Move incrementally** - One utility function at a time
2. **Test continuously** - Run affected scripts after each change
3. **Keep git history clean** - Separate commits for moves vs. logic changes

### Validation Steps
```bash
# After each phase, verify:
python -c "import visdetect; print('Package imports work')"
cd analysis_suite && python 01_behavior/a_learning_curve.py  # Test analysis script
python -m pytest tests/  # Run test suite
```

## 🎯 Success Metrics

**Phase 1 Complete When**:
- [x] All legacy code in single `archive/` directory
- [x] READMEs exist for main directories
- [ ] No filenames with spaces/special characters

**Phase 2 Complete When**:
- [ ] No `sys.path.insert()` calls in codebase
- [ ] All utilities in `src/visdetect/analysis/utils.py`
- [ ] No hard-coded SUBJECT or drive paths
- [ ] `pip install -e .` enables all analysis scripts

**Phase 3 Complete When**:
- [ ] Full `run_all.py` execution passes
- [ ] All tests pass with new import structure
- [ ] Documentation covers new developer setup

## 🚀 Quick Start Commands

```bash
# Phase 1: Quick cleanup
git add archive/README.md analysis_suite/README.md
git commit -m "docs: Add archive and analysis_suite documentation"

# Phase 2: Start utility migration
mkdir -p src/visdetect/analysis && touch src/visdetect/analysis/utils.py
# (Implement Task 4 step by step)

# Validation after each step
cd analysis_suite && python run_all.py --dry-run  # Check script loading
```

## 📞 Implementation Support

This reorganization plan addresses the technical debt identified in the March 2026 audit. Each task includes specific file paths and validation steps.

**Estimated Total Time**: 4-8 days (depending on testing thoroughness)
**Risk Level**: Medium (many files affected, but changes are mechanical)
**Benefit**: High (improved maintainability, easier onboarding, better testing)