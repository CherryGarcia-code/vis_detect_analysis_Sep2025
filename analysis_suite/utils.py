"""DEPRECATED: Shared computation utilities for the analysis suite.

⚠️  WARNING: This file is deprecated!
⚠️  All utilities have been moved to src/visdetect/analysis/utils.py

This file now simply re-exports functions from the centralized location
for backward compatibility. Please update your imports to:

    from visdetect.analysis.utils import function_name

instead of:

    from utils import function_name

This file will be removed in a future version.
"""

# Re-export all functions from the centralized location
from visdetect.analysis.utils import (
    # Population analysis
    build_population_tensor,
    smooth_psth,
    compute_zscore_normalized,
    compute_baseline_subtracted,

    # Unit selection
    get_good_cluster_ids,

    # Statistical utilities
    bootstrap_ci,
    permutation_test,
    fdr_correct,
    compute_auroc,

    # LDA coding direction
    compute_lda_cd,
)

# Show deprecation warning
import warnings
warnings.warn(
    "analysis_suite.utils is deprecated. Use 'from visdetect.analysis.utils import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)