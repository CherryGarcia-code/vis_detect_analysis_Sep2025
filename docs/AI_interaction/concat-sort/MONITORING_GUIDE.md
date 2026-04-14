# Concat-Sort Pipeline Monitoring Guide

**Date Created**: March 31, 2026
**Purpose**: Standard operating procedures for monitoring and troubleshooting concat-sort pipeline runs

---

## Quick Status Check Workflow

### 1. Basic Completion Status
```bash
# Run from project root
cd scripts/pipelines/concat_sort
python check_ks4_completion_status.py --verbose

# Outputs:
# - Total jobs and completion percentage
# - List of failed jobs by window/shank
# - CSV files: ks4_completed_jobs_*.csv, ks4_failed_jobs_*.csv
```

### 2. Log Analysis (if failures detected)
```bash
# Analyze SLURM logs for error patterns
python analyze_cluster_logs.py --verbose

# Outputs:
# - Error type frequency (cuda_oom, timeout, file_not_found, etc.)
# - Failure patterns by window
# - Specific error excerpts for debugging
```

### 3. Summary and Next Steps
```bash
# Get actionable summary
python summary_and_next_steps.py

# Shows current status and recommended actions
```

---

## File Locations and Paths

### Key Directories
| Path | Purpose |
|------|---------|
| `X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/ks4_runs/` | Run directories |
| `Z:/Documents/ks4/logs_ks4_resort/` | SLURM logs (.out/.err files) |
| `scripts/pipelines/concat_sort/` | Monitoring scripts (this repo) |

### Important Files
- `ks4_run_manifest.json` - Job definitions (136 total jobs)
- `ks4_complete.txt` - Completion markers in each run directory
- `ks4_resort-*_N.out/.err` - SLURM logs for job N

---

## Monitoring Scripts Reference

### check_ks4_completion_status.py
**Purpose**: Check which jobs completed successfully vs failed
**How it works**: Reads manifest, checks for `ks4_complete.txt` markers
**Key outputs**:
- Completion percentage
- Failed jobs by window/shank
- Failure patterns (random vs systematic)

**Usage**:
```bash
# Basic check
python check_ks4_completion_status.py

# Verbose (show each job status)
python check_ks4_completion_status.py --verbose

# Custom manifest path
python check_ks4_completion_status.py --manifest /path/to/manifest.json
```

### analyze_cluster_logs.py
**Purpose**: Analyze SLURM logs to identify failure root causes
**How it works**: Reads .out/.err files, extracts error patterns
**Key outputs**:
- Error type classification
- Temporal failure patterns
- Specific error excerpts

**Common Error Types**:
- `cuda_oom` - GPU out of memory
- `timeout` - Job time limit exceeded
- `file_not_found` - Missing input files
- `kilosort_error` - General KS4 failures
- `node_failure` - Hardware issues

**Usage**:
```bash
# Analyze all failed jobs
python analyze_cluster_logs.py

# Show detailed error excerpts
python analyze_cluster_logs.py --verbose

# Custom log directory
python analyze_cluster_logs.py --log-dir /path/to/logs
```

---

## Common Failure Patterns and Solutions

### Pattern 1: CUDA Resource Contention
**Symptoms**:
- `cuda_oom` or "CUDA device busy/unavailable"
- Failures clustered in time (not random)
- Early windows succeed, later windows fail

**Diagnosis**: GPU queue saturation or node issues
**Solution**: Retry with reduced concurrency (`%3` instead of `%6`)

### Pattern 2: Systematic File Issues
**Symptoms**:
- `file_not_found` errors
- Failures clustered by session or window
- All shanks in a window fail

**Diagnosis**: Network storage or data integrity issues
**Solution**: Check file paths, verify binary files exist

### Pattern 3: Memory/Time Limits
**Symptoms**:
- `timeout` or memory-related errors
- Larger windows fail more often
- Node-specific failures

**Diagnosis**: Insufficient resources for job size
**Solution**: Reduce window size or increase time/memory limits

### Pattern 4: Random Scattered Failures
**Symptoms**:
- Mixed error types
- No temporal or spatial pattern
- Low failure rate (<10%)

**Diagnosis**: Normal cluster variability
**Solution**: Simple retry of failed jobs

---

## Retry Workflow

### 1. Generate Retry Script
For systematic failures (like GPU contention):
```bash
# Use the template: retry_failed_ks4_jobs.bash
# Edit the FAILED_JOBS array with actual failed job IDs
# Adjust concurrency (%N) based on failure pattern
```

For simple retries:
```bash
# Just resubmit the original array job
# SLURM will skip completed jobs automatically
sbatch run_ks4_resort_nblocks1.bash
```

### 2. Monitor Retry Progress
```bash
# Check queue status
squeue -u $USER

# Re-run completion check periodically
python check_ks4_completion_status.py

# Target: >95% completion rate
```

---

## Troubleshooting Checklist

### Before Investigating Failures:
- [ ] Are >90% of jobs successful? (If yes, simple retry may be sufficient)
- [ ] Are failures clustered in time? (GPU contention)
- [ ] Are failures clustered by window? (Data issues)
- [ ] Are failures clustered by node? (Hardware issues)

### For Systematic Investigation:
1. **Run completion status check**
   ```bash
   python check_ks4_completion_status.py --verbose > status_report.txt
   ```

2. **Analyze failure patterns**
   ```bash
   python analyze_cluster_logs.py --verbose > error_analysis.txt
   ```

3. **Check specific failed jobs manually**
   ```bash
   # Example: Check job 133 logs
   less "Z:/Documents/ks4/logs_ks4_resort/ks4_resort-*_133.out"
   less "Z:/Documents/ks4/logs_ks4_resort/ks4_resort-*_133.err"
   ```

4. **Verify run directories**
   ```bash
   # Check if completion marker exists but status shows failed
   ls "X:/public/projects/.../ks4_runs/window_XXX/shank_Y/ks4_complete.txt"
   ```

---

## Expected Performance Benchmarks

### Successful Job Characteristics:
- **Runtime**: 8-12 hours per job (96 channels × 5 sessions)
- **GPU memory**: 30-35 GB peak usage (A100-40GB)
- **Unit yield**: 100-200 total units, 30-60 good units
- **Files generated**: spike_times.npy, templates.npy, cluster_info.tsv, etc.

### Warning Signs:
- **Runtime** >15 hours (may indicate drift/clustering issues)
- **GPU memory** >38 GB (OOM risk)
- **Unit yield** <20 total units (poor data quality)
- **Missing outputs** (incomplete processing)

---

## Automation Suggestions

### For Regular Monitoring:
```bash
# Daily status check (add to cron)
cd /path/to/concat_sort && python check_ks4_completion_status.py | mail -s "KS4 Status" user@domain.com
```

### For Large Pipelines:
- Set up automated retry for failed jobs
- Monitor GPU queue status before submission
- Log completion status to shared dashboard

---

## Files Created (March 31, 2026)

| File | Purpose |
|------|---------|
| `check_ks4_completion_status.py` | Main status checker |
| `analyze_cluster_logs.py` | Log analysis and error pattern detection |
| `summary_and_next_steps.py` | Quick status summary and recommendations |
| `retry_failed_ks4_jobs.bash` | Template for retrying failed jobs |
| `check_completion_marker.py` | Debug specific completion marker issues |

---

## Contact and Maintenance

**Created by**: Claude Code assistant
**Last updated**: March 31, 2026
**Tested on**: BG_046 concat-sort pipeline (136 jobs)

**Future improvements**:
- [ ] Add automatic retry script generation
- [ ] Integrate with SLURM job dependency system
- [ ] Add email notifications for completion status
- [ ] Create dashboard for multi-subject monitoring