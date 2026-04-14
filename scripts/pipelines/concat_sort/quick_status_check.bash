#!/usr/bin/env bash
# Quick Concat-Sort Status Check
# Run this script for immediate pipeline status

cd "$(dirname "$0")"

echo "=========================================="
echo "CONCAT-SORT PIPELINE QUICK STATUS CHECK"
echo "=========================================="

echo "1. Checking completion status..."
python check_ks4_completion_status.py

echo -e "\n2. Analyzing any failures..."
python analyze_cluster_logs.py

echo -e "\n3. Summary and next steps..."
python summary_and_next_steps.py

echo -e "\n=========================================="
echo "QUICK REFERENCE:"
echo "- Retry failed jobs: sbatch retry_failed_ks4_jobs.bash"
echo "- Monitor queue: squeue -u \$USER"
echo "- Detailed guide: docs/AI_interaction/concat-sort/MONITORING_GUIDE.md"
echo "=========================================="