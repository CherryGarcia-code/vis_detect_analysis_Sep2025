#!/usr/bin/env python3
"""Update analysis_suite imports to use centralized utilities.

This script updates all analysis_suite files to import from the new
visdetect.analysis.utils module instead of the local utils.py.
"""

import os
import re
from pathlib import Path

# Files that need import updates
FILES_TO_UPDATE = [
    "analysis_suite/03_population/f_2d_decomposition.py",
    "analysis_suite/04_decoding/c_state_decoding.py",
    "analysis_suite/04_decoding/b_change_size_decoding.py",
    "analysis_suite/04_decoding/a_hit_miss_decoding.py",
    "analysis_suite/03_population/e_sensory_dose_response.py",
    "analysis_suite/03_population/d_state_matched_cd.py",
    "analysis_suite/03_population/a_coding_direction.py",
    "analysis_suite/07_advanced/_fa_helpers.py",
    "analysis_suite/07_advanced/h_second_pulse_analysis.py",
    "analysis_suite/07_advanced/g_fa_subtype_prediction.py",
    "analysis_suite/07_advanced/f_fa_subtype_lick_triggered_tf.py",
    "analysis_suite/02_single_unit/a_responsiveness_screen.py",
    "analysis_suite/07_advanced/e_trial_outcome_prediction.py",
    "analysis_suite/02_single_unit/e_cell_type_comparison.py",
    "analysis_suite/02_single_unit/d_state_modulation.py",
    "analysis_suite/03_population/c_dimensionality_reduction.py",
    "analysis_suite/07_advanced/b_dpca.py",
    "analysis_suite/07_advanced/d_impulsivity_regression.py",
    "analysis_suite/07_advanced/c_noise_correlations.py",
    "analysis_suite/06_lick_motor/a_fa_neural_signatures.py",
    "analysis_suite/05_longitudinal/b_celltype_learning.py",
    "analysis_suite/06_lick_motor/b_pre_lick_ramping.py",
    "analysis_suite/05_longitudinal/c_population_geometry_shift.py",
    "analysis_suite/06_lick_motor/c_motor_vs_sensory.py",
    "analysis_suite/02_single_unit/b_outcome_selectivity.py",
    "analysis_suite/05_longitudinal/a_neural_learning_curves.py",
    "analysis_suite/09_optotagging/a_optotagging_identification.py",
    "analysis_suite/07_advanced/a_glm_encoding.py",
    "analysis_suite/03_population/b_population_psth_heatmap.py",
    "analysis_suite/02_single_unit/c_change_size_tuning.py",
]

# Functions that were moved to the new utils module
UTILITY_FUNCTIONS = [
    'build_population_tensor',
    'smooth_psth',
    'compute_zscore_normalized',
    'compute_baseline_subtracted',
    'get_good_cluster_ids',
    'bootstrap_ci',
    'permutation_test',
    'fdr_correct',
    'compute_auroc',
]

def update_imports_in_file(file_path: Path):
    """Update imports in a single file."""
    print(f"Updating imports in: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern 1: from utils import func1, func2, ...
    pattern1 = r'from utils import ([^\n]+)'
    matches = re.findall(pattern1, content)

    for match in matches:
        # Parse imported functions
        imported_funcs = [func.strip() for func in match.split(',')]

        # Check which functions are in our utility list
        util_funcs = [f for f in imported_funcs if f in UTILITY_FUNCTIONS]
        other_funcs = [f for f in imported_funcs if f not in UTILITY_FUNCTIONS]

        if util_funcs:
            # Create new import for utility functions
            new_util_import = f"from visdetect.analysis.utils import {', '.join(util_funcs)}"

            if other_funcs:
                # Keep import for remaining functions
                old_import = f"from utils import {match}"
                new_other_import = f"from utils import {', '.join(other_funcs)}"
                replacement = f"{new_util_import}\n{new_other_import}"
            else:
                # Replace entire import
                old_import = f"from utils import {match}"
                replacement = new_util_import

            content = content.replace(old_import, replacement)

    # Pattern 2: import utils (then usage like utils.function_name)
    if 'import utils' in content and 'from utils import' not in content:
        # This is trickier - need to add specific imports and update usage
        # For now, let's add the new import and keep the old one
        # (will need manual cleanup later)
        if 'from visdetect.analysis.utils import' not in content:
            # Add all utility imports
            new_import = f"from visdetect.analysis.utils import {', '.join(UTILITY_FUNCTIONS)}"
            # Insert after existing imports
            import_section_end = content.find('\n\n')
            if import_section_end != -1:
                content = content[:import_section_end] + f"\n{new_import}" + content[import_section_end:]

    # Only write if content changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Updated {file_path}")
        return True
    else:
        print(f"  - No changes needed in {file_path}")
        return False

def main():
    """Update imports in all analysis_suite files."""
    print("Updating analysis_suite imports to use visdetect.analysis.utils")
    print("=" * 60)

    updated_count = 0

    for file_path in FILES_TO_UPDATE:
        full_path = Path(file_path)
        if full_path.exists():
            if update_imports_in_file(full_path):
                updated_count += 1
        else:
            print(f"  Warning: File not found: {file_path}")

    print("=" * 60)
    print(f"Updated imports in {updated_count} files")
    print("\nNext steps:")
    print("1. Test that imports work: cd analysis_suite && python 01_behavior/a_learning_curve.py")
    print("2. Remove old analysis_suite/utils.py after testing")
    print("3. Update analysis_suite/config.py to remove sys.path manipulation")

if __name__ == "__main__":
    main()