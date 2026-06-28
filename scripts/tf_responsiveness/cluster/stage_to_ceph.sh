#!/usr/bin/env bash
#
# Stage the TF-GLM cluster code + targets to the writable ceph project
# (BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster). Re-runnable.
#
# IMPORTANT: empties the heavy `visdetect` package __init__ files IN THE STAGED
# COPY. The worker only needs `visdetect.analysis.tf_glm` + `.tf_glm_data`, and
# neither imports anything else from visdetect -- but the normal package
# __init__ chain eagerly imports core/qc (PyYAML), analysis/hmm, align (h5py),
# etc., which the minimal `tfglm` conda env (numpy/pandas/sklearn/scipy/pyarrow)
# does NOT have. Stubbing the two __init__ files makes `import
# visdetect.analysis.tf_glm` pull in ONLY numpy/sklearn -> runs on the lean env.
#
# Never writes under the read-only MoHa data tree; everything goes to $STAGE.
#
# Usage:
#   bash stage_to_ceph.sh [SRC] [STAGE] [TARGETS_CSV]
set -euo pipefail

SRC=${1:-/e/python_analysis/git_repos/vd_tf_phase0}
STAGE=${2:-/x/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster}
TARGETS=${3:-$SRC/data/cache/tf_glm/cluster/targets_decisive.csv}

mkdir -p "$STAGE/code/src" "$STAGE/code/scripts/tf_responsiveness/cluster" \
         "$STAGE/logs" "$STAGE/results"

# Fresh copy of the package (clean any prior staged tree).
rm -rf "$STAGE/code/src/visdetect"
cp -r "$SRC/src/visdetect" "$STAGE/code/src/"
find "$STAGE/code/src/visdetect" -name __pycache__ -type d -prune \
     -exec rm -rf {} + 2>/dev/null || true

# Minimal-import stubs (see header). The worker imports only leaf modules.
: > "$STAGE/code/src/visdetect/__init__.py"
: > "$STAGE/code/src/visdetect/analysis/__init__.py"

# Scripts + targets.
cp -r "$SRC/scripts/tf_responsiveness/cluster/." \
      "$STAGE/code/scripts/tf_responsiveness/cluster/"
cp "$TARGETS" "$STAGE/targets.csv"

echo "Staged to $STAGE"
echo "  visdetect/__init__.py        bytes: $(wc -c < "$STAGE/code/src/visdetect/__init__.py")"
echo "  visdetect/analysis/__init__  bytes: $(wc -c < "$STAGE/code/src/visdetect/analysis/__init__.py")"
echo "  tf_glm.py present:                  $([ -f "$STAGE/code/src/visdetect/analysis/tf_glm.py" ] && echo yes || echo NO)"
echo "  tf_glm_data.py present:             $([ -f "$STAGE/code/src/visdetect/analysis/tf_glm_data.py" ] && echo yes || echo NO)"
echo "  targets.csv lines:                  $(wc -l < "$STAGE/targets.csv")"
echo "  results/ task CSVs:                 $(ls "$STAGE"/results/task_*.csv 2>/dev/null | wc -l)"
