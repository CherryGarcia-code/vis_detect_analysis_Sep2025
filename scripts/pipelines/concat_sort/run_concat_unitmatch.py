#!/usr/bin/env python3
"""Run UnitMatch (Bayesian) on concat-sort waveforms, per-shank.

Takes the prepared waveforms from prep_concat_waveforms.py and runs the
UnitMatchPy pipeline to assign cross-session unique IDs.

Runs independently per shank (4 shanks → 4 separate UnitMatch runs).
Uses sliding-window batches for memory management on large session counts.

Input:  data/unit_match_concat_sort/input/BG_046/shank_{N}/{session}/RawWaveforms/
Output: data/unit_match_concat_sort/output/BG_046/shank_{N}/{batch_label}/

Usage:
    python scripts/pipelines/concat_sort/run_concat_unitmatch.py --shank 0
    python scripts/pipelines/concat_sort/run_concat_unitmatch.py --shank all --batch_size 12
    python scripts/pipelines/concat_sort/run_concat_unitmatch.py --shank 0 --sessions BG_046_01072025 BG_046_02072025
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# UnitMatch imports
try:
    import UnitMatchPy.utils as util
    import UnitMatchPy.overlord as ov
    import UnitMatchPy.bayes_functions as bf
    import UnitMatchPy.assign_unique_id as aid
    import UnitMatchPy.save_utils as su
    import UnitMatchPy.default_params as default_params
except ImportError:
    print("Error: UnitMatchPy not found. Activate the unitmatch_env environment.")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = REPO_ROOT / "data" / "unit_match_concat_sort" / "input" / "BG_046"
OUTPUT_ROOT = REPO_ROOT / "data" / "unit_match_concat_sort" / "output" / "BG_046"


def parse_date(name):
    """Parse date from session name like BG_046_DDMMYYYY or DDMMYYYY."""
    if len(name) == 8 and name.isdigit():
        try:
            return datetime.strptime(name, "%d%m%Y")
        except ValueError:
            pass
    parts = name.split('_')
    if len(parts) >= 2:
        d_str = parts[-1]
        if len(d_str) == 8 and d_str.isdigit():
            try:
                return datetime.strptime(d_str, "%d%m%Y")
            except ValueError:
                pass
    return datetime.min


def sync_tsv_with_waveforms(ks_dir):
    """Ensure cluster_group.tsv only lists clusters with extracted waveforms."""
    ks_path = Path(ks_dir)
    wav_dir = ks_path / "RawWaveforms"
    tsv_path = ks_path / "cluster_group.tsv"

    if not tsv_path.exists() or not wav_dir.exists():
        return

    available_ids = set()
    for f in wav_dir.glob("Unit*_RawSpikes.npy"):
        try:
            uid_str = f.stem.split('_')[0].replace('Unit', '')
            available_ids.add(int(uid_str))
        except ValueError:
            pass

    if not available_ids:
        return

    try:
        df = pd.read_csv(tsv_path, sep='\t')
        df_filtered = df[df['cluster_id'].isin(available_ids)].copy()
        # Ensure all units marked 'good' for UnitMatch
        if 'KSLabel' in df_filtered.columns:
            df_filtered['KSLabel'] = 'good'
        elif 'group' in df_filtered.columns:
            df_filtered['group'] = 'good'
        df_filtered.to_csv(tsv_path, sep='\t', index=False)
        print(f"  Synced {ks_path.name}: {len(df)} → {len(df_filtered)} units in TSV.")
    except Exception as e:
        print(f"  Error syncing TSV in {ks_dir}: {e}")


def run_pipeline_batch(session_dirs, output_dir, label_suffix=""):
    """Run UnitMatchPy on a batch of session directories."""
    print(f"\n{'='*60}")
    print(f"Batch: {label_suffix}")
    print(f"Sessions: {[d.name for d in session_dirs]}")
    print(f"Output:  {output_dir}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sync TSVs
    for d in session_dirs:
        sync_tsv_with_waveforms(d)

    KS_dirs = [str(d) for d in session_dirs]

    # UnitMatch default parameters
    param = default_params.get_default_param()

    # Memory optimization for large batches
    n_sess = len(session_dirs)
    if n_sess >= 8:
        param['n_bins'] = 50
        print(f"  Memory optimization: n_bins reduced to 50 for {n_sess} sessions.")

    # Load data via UnitMatchPy
    try:
        wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(KS_dirs)
        waveform, session_id, session_switch, within_session, good_units, param = \
            util.load_good_waveforms(wave_paths, unit_label_paths, param)
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Loaded {len(np.concatenate(good_units))} total units across {n_sess} sessions.")

    clus_info = {
        'good_units': good_units,
        'session_switch': session_switch,
        'session_id': session_id,
        'original_ids': np.concatenate(good_units)
    }

    param = util.get_probe_geometry(channel_pos[0], param)

    # STEP 1: Extract waveform parameters
    print("Extracting waveform parameters...")
    extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)

    # STEP 2-4: Metric scores
    print("Extracting metric scores...")
    total_score, candidate_pairs, scores_to_include, predictors = ov.extract_metric_scores(
        extracted_wave_properties, session_switch, within_session, param, niter=2
    )

    # STEP 5: Bayesian probability
    print("Running probability analysis...")
    n_units = param['n_units']
    prior_match = 1 - (param['n_expected_matches'] / n_units**2)
    priors = np.array((prior_match, 1 - prior_match))
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)

    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one=1)

    try:
        probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)
    except MemoryError as e:
        print(f"MEMORY ERROR: {e}. Try reducing --batch_size.")
        return
    except Exception as e:
        if "Unable to allocate" in str(e):
            print(f"MEMORY ERROR (Allocation): {e}. Try reducing --batch_size.")
            return
        raise

    output_prob_matrix = probability[:, 1].reshape(n_units, n_units)

    match_threshold = 0.5
    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1

    # Assign Unique IDs
    print("Assigning Unique IDs...")
    matches = np.argwhere(output_threshold == 1)
    UID_collection = aid.assign_unique_id(output_prob_matrix, param, clus_info)

    if isinstance(UID_collection, (list, tuple)) and len(UID_collection) >= 2:
        UIDs_for_registry = UID_collection[1]  # Intermediate
        print("Using Intermediate UIDs (Index 1).")
    elif hasattr(UID_collection, '__len__') and len(UID_collection) == len(clus_info['original_ids']):
        UIDs_for_registry = UID_collection
        UID_collection = [UID_collection] * 4
    else:
        UIDs_for_registry = UID_collection[0]

    # Save UnitMatch native output
    save_path = str(output_dir)
    print(f"Saving results to {save_path}...")

    amplitude = extracted_wave_properties['amplitude']
    spatial_decay = extracted_wave_properties['spatial_decay']
    avg_centroid = extracted_wave_properties['avg_centroid']
    avg_waveform = extracted_wave_properties['avg_waveform']
    avg_waveform_per_tp = extracted_wave_properties['avg_waveform_per_tp']
    wave_idx = extracted_wave_properties['good_wave_idxs']
    max_site = extracted_wave_properties['max_site']

    su.save_to_output(
        save_path, scores_to_include, matches, output_prob_matrix,
        avg_centroid, avg_waveform, avg_waveform_per_tp, max_site,
        total_score, output_threshold, clus_info, param,
        UIDs=UID_collection, matches_curated=None, save_match_table=True
    )

    # --- Cell Registry ---
    print("Generating Cell Registry...")
    session_names = [Path(k).name for k in KS_dirs]
    with open(output_dir / "SessionList.txt", "w") as f:
        for s in session_names:
            f.write(s + "\n")

    sess_ids = clus_info['session_id']
    orig_ids = clus_info['original_ids']

    registry_variants = {}
    if isinstance(UID_collection, (list, tuple)) and len(UID_collection) >= 3:
        registry_variants['Liberal'] = UID_collection[0]
        registry_variants['Intermediate'] = UID_collection[1]
        registry_variants['Conservative'] = UID_collection[2]
    else:
        registry_variants['Default'] = UIDs_for_registry

    for var_name, var_uids in registry_variants.items():
        if len(var_uids) != len(sess_ids):
            print(f"Warning: {var_name} UID length mismatch. Skipping.")
            continue

        data_list = []
        for i, uid in enumerate(var_uids):
            s_idx = sess_ids[i]
            c_val = orig_ids[i]
            if hasattr(c_val, 'item'):
                c_val = c_val.item()
            if s_idx < len(session_names):
                data_list.append({
                    'UID': uid,
                    'Session': session_names[s_idx],
                    'ClusterID': c_val,
                    'GlobalIndex': i
                })

        df_long = pd.DataFrame(data_list)
        if var_name in ['Intermediate', 'Default']:
            df_long.to_csv(output_dir / "Unit_Long_Table.csv", index=False)

        # Pivot to wide format
        if df_long.duplicated(subset=['UID', 'Session']).any():
            print(f"Warning: {var_name} has duplicate UID-Session pairs. Aggregating.")

            def agg_clusters(x):
                return ";".join(str(v) for v in sorted(set(x)))

            df_registry = df_long.pivot_table(
                index='UID', columns='Session', values='ClusterID', aggfunc=agg_clusters
            )
        else:
            df_registry = df_long.pivot(index='UID', columns='Session', values='ClusterID')

        existing_cols = [s for s in session_names if s in df_registry.columns]
        df_registry = df_registry[existing_cols]

        df_registry.to_csv(output_dir / f"CellRegistry_{var_name}.csv")
        if var_name in ['Intermediate', 'Default']:
            df_registry.to_csv(output_dir / "CellRegistry.csv")
            print(f"Saved CellRegistry.csv ({var_name}): {len(df_registry)} unique units.")
        else:
            print(f"Saved CellRegistry_{var_name}.csv: {len(df_registry)} units.")

    print(f"Batch {label_suffix} completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Run UnitMatch on concat-sort waveforms (per-shank)")
    parser.add_argument('--shank', type=str, required=True,
                        help='Shank ID (0-3) or "all"')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Sessions per sliding-window batch (default: 12, safe for 64 GB RAM)')
    parser.add_argument('--sessions', nargs='+',
                        help='Specific session names to include')
    args = parser.parse_args()

    shanks = list(range(4)) if args.shank == 'all' else [int(args.shank)]

    for shank_id in shanks:
        data_root = INPUT_ROOT / f"shank_{shank_id}"
        out_root = OUTPUT_ROOT / f"shank_{shank_id}"
        out_root.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            print(f"No input data for shank {shank_id} at {data_root}")
            continue

        session_dirs = [d for d in data_root.iterdir()
                        if d.is_dir() and (d / "RawWaveforms").exists()]

        if args.sessions:
            session_dirs = [d for d in session_dirs
                           if any(t in d.name for t in args.sessions)]

        # Sort by date (most recent first — matches original pipeline convention)
        session_dirs.sort(key=lambda x: parse_date(x.name), reverse=True)

        print(f"\n{'#'*60}")
        print(f"# SHANK {shank_id}: {len(session_dirs)} sessions")
        print(f"# Input:  {data_root}")
        print(f"# Output: {out_root}")
        print(f"{'#'*60}")

        if not session_dirs:
            print("No prepared sessions. Run prep_concat_waveforms.py first.")
            continue

        bs = args.batch_size
        n_sessions = len(session_dirs)

        if bs >= n_sessions:
            run_pipeline_batch(session_dirs, out_root, label_suffix=f"shank{shank_id}_ALL")
        else:
            print(f"Running sliding-window batches of size {bs}...")
            for i in range(0, n_sessions - bs + 1):
                batch_dirs = session_dirs[i:i + bs]
                start_name = batch_dirs[0].name
                end_name = batch_dirs[-1].name
                label = f"{start_name}_to_{end_name}"
                sub_out = out_root / label
                run_pipeline_batch(batch_dirs, sub_out, label_suffix=label)


if __name__ == "__main__":
    main()
