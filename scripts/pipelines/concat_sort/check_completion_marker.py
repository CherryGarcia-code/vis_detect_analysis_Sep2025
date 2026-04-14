#!/usr/bin/env python3
"""Quick check for completion marker discrepancy.

Cross-reference successful log completions with ks4_complete.txt markers.
"""

import os
import re
from pathlib import Path

# Jobs that completed successfully in logs but were marked as failed
suspected_completed = [131]  # Job 131 from our analysis

for job_id in suspected_completed:
    print(f"Checking job {job_id}...")

    # Read the manifest to get run directory
    manifest_path = "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/ks4_runs/ks4_run_manifest.json"

    with open(manifest_path, 'r') as f:
        import json
        manifest = json.load(f)

    if job_id <= len(manifest['windows']):
        window = manifest['windows'][job_id - 1]  # job_id is 1-based
        run_dir = Path(window['run_dir'])
        complete_marker = run_dir / "ks4_complete.txt"

        print(f"  Run dir: {run_dir}")
        print(f"  Complete marker exists: {complete_marker.exists()}")

        if complete_marker.exists():
            with open(complete_marker, 'r') as f:
                content = f.read()
                print(f"  Marker content preview: {content[:100]}...")
        else:
            print(f"  -> This explains why it was marked as failed!")

            # Check if KS4 outputs exist
            ks4_outputs = ['spike_times.npy', 'spike_clusters.npy', 'templates.npy']
            outputs_exist = [str((run_dir / f).exists()) for f in ks4_outputs]
            print(f"  KS4 outputs exist: {outputs_exist}")