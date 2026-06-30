#!/usr/bin/env bash
#
# Stage the BG-mouse TF-GLM cluster CODE + targets to the writable ceph project
# (BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/bg_mice). Re-runnable.
#
# Differs from the MoHa stage_to_ceph.sh in ONE way: the BG worker imports
# `visdetect.core.session` (load_session for pkls), so we stub THREE package
# __init__ files, not two -- root, analysis, AND core. core/__init__ eagerly
# imports qc (PyYAML), kilosort, spikeglx etc. which the lean `tfglm` env lacks;
# session.py itself only needs pickle+numpy, so the empty stub makes
# `import visdetect.core.session` lean.
#
# The pkls are NOT staged here (they are ~25-40 GB). Stage them separately and
# ONCE with a resumable transfer (the gateway blocks cp-overwrite; robocopy /
# rsync --ignore-existing skip already-copied files):
#
#   # Windows (resumable, skips existing):
#   robocopy "E:\python_analysis\git_repos\vis_detect_analysis_Sep2025\data\pkls" \
#            "X:\public\projects\BeJG_20230130_VisDetect\wEPhys\tf_glm_cluster\bg_mice\bg_pkls" \
#            /E /XO /R:2 /W:5 /MT:8
#   # or git-bash rsync:
#   rsync -a --ignore-existing /e/.../data/pkls/ /x/.../bg_mice/bg_pkls/
#
# Usage:
#   bash stage_bg_to_ceph.sh [SRC] [STAGE] [TARGETS_CSV]
set -euo pipefail

SRC=${1:-/e/python_analysis/git_repos/vd_tf_bg046}
STAGE=${2:-/x/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster}
TARGETS=${3:-$SRC/data/cache/tf_glm_bg046/targets_bg.csv}
BG="$STAGE/bg_mice"

mkdir -p "$BG/code/src" "$BG/code/scripts/tf_responsiveness/cluster_bg" \
         "$BG/logs" "$BG/results_bg"

# Fresh copy of the package (delete-then-copy avoids the gateway's cp-overwrite
# "File exists" failure).
rm -rf "$BG/code/src/visdetect"
cp -r "$SRC/src/visdetect" "$BG/code/src/"
find "$BG/code/src/visdetect" -name __pycache__ -type d -prune \
     -exec rm -rf {} + 2>/dev/null || true

# Minimal-import stubs: the worker imports only leaf modules
# (core.session, analysis.tf_glm, analysis.tf_glm_data).
: > "$BG/code/src/visdetect/__init__.py"
: > "$BG/code/src/visdetect/analysis/__init__.py"
: > "$BG/code/src/visdetect/core/__init__.py"

# Scripts (delete-then-copy) + targets (cat> truncate-write, not cp).
rm -rf "$BG/code/scripts/tf_responsiveness/cluster_bg"
mkdir -p "$BG/code/scripts/tf_responsiveness/cluster_bg"
cp -r "$SRC/scripts/tf_responsiveness/cluster_bg/." \
      "$BG/code/scripts/tf_responsiveness/cluster_bg/"
if [ -f "$TARGETS" ]; then
    cat "$TARGETS" > "$BG/targets_bg.csv"
fi

echo "Staged BG code to $BG"
echo "  visdetect/__init__.py        bytes: $(wc -c < "$BG/code/src/visdetect/__init__.py")"
echo "  visdetect/analysis/__init__  bytes: $(wc -c < "$BG/code/src/visdetect/analysis/__init__.py")"
echo "  visdetect/core/__init__      bytes: $(wc -c < "$BG/code/src/visdetect/core/__init__.py")"
echo "  core/session.py present:            $([ -f "$BG/code/src/visdetect/core/session.py" ] && echo yes || echo NO)"
echo "  analysis/tf_glm.py present:         $([ -f "$BG/code/src/visdetect/analysis/tf_glm.py" ] && echo yes || echo NO)"
echo "  analysis/tf_glm_data.py present:    $([ -f "$BG/code/src/visdetect/analysis/tf_glm_data.py" ] && echo yes || echo NO)"
echo "  worker present:                     $([ -f "$BG/code/scripts/tf_responsiveness/cluster_bg/tf_glm_bg_task.py" ] && echo yes || echo NO)"
echo "  targets_bg.csv lines:               $([ -f "$BG/targets_bg.csv" ] && wc -l < "$BG/targets_bg.csv" || echo MISSING)"
echo "  bg_pkls present:                    $([ -d "$BG/bg_pkls" ] && echo "$(find "$BG/bg_pkls" -name '*.pkl' 2>/dev/null | wc -l) pkls" || echo "NOT staged -- see header for robocopy/rsync")"
echo "  results_bg task CSVs:               $(ls "$BG"/results_bg/task_*.csv 2>/dev/null | wc -l)"
