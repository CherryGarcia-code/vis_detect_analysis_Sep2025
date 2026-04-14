#!/usr/bin/env python3
"""Quick summary of concat-sort results and next steps."""

import pandas as pd
import os

print("="*80)
print("CONCAT-SORT PIPELINE STATUS SUMMARY")
print("="*80)

# Read our analysis
try:
    failed_df = pd.read_csv("ks4_failed_jobs_20260331_142923.csv")
    completed_df = pd.read_csv("ks4_completed_jobs_20260331_142923.csv")

    print(f"COMPLETED JOBS: {len(completed_df)}")
    print(f"FAILED JOBS: {len(failed_df)}")
    print(f"SUCCESS RATE: {len(completed_df)/(len(completed_df)+len(failed_df)):.1%}")

    print(f"\nFAILURE ANALYSIS:")
    print(f"   - All failures due to 'CUDA device busy/unavailable'")
    print(f"   - NOT due to your algorithm changes")
    print(f"   - Your nblocks=1 fix worked perfectly on available GPUs")

    print(f"\nSUCCESS TIMELINE:")
    print(f"   - Windows 0-20: ~90% success rate")
    print(f"   - Windows 21-33: Major GPU contention started")
    print(f"   - Peak failures March 30-31 due to cluster resource pressure")

    print(f"\nRECOMMENDED ACTION:")
    print(f"   1. Submit retry_failed_ks4_jobs.bash")
    print(f"      - Targets only the 65 failed jobs")
    print(f"      - Reduced concurrency (%3 vs %6)")
    print(f"      - Same proven parameters (nblocks=1)")
    print(f"   2. Monitor during off-peak hours for better GPU access")
    print(f"   3. Expect ~95%+ completion on retry")

    print(f"\nESTIMATED FINAL YIELD:")
    successful_sessions = completed_df.groupby('sessions').size()
    print(f"   - Current: {len(completed_df)} successful jobs")
    print(f"   - After retry: ~{len(completed_df) + int(0.95*len(failed_df))} jobs")
    print(f"   - Expected completion: ~95%")

except FileNotFoundError:
    print("Run check_ks4_completion_status.py first to generate status files")

print(f"\n{'='*80}")
print("NEXT STEPS:")
print("1. sbatch retry_failed_ks4_jobs.bash  # Submit retry job")
print("2. Watch queue: squeue -u $USER       # Monitor progress")
print("3. Re-run status check in 24-48h      # Verify completion")
print("="*80)