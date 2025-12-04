#!/usr/bin/env python3
"""Run UnitMatch tracking across multiple sessions.

This script implements the full UnitMatch algorithm from the paper:
- Iteratively matches units across all session pairs
- Applies prob > 0.5 threshold with neighboring recordings check
- Allows units to disappear and reappear
- Outputs tracking table with match chains

Based on Windolf et al. 2024 Nature Methods:
"The default version of the algorithm iteratively inspects all pairs, and merges
a unit with a target group if its probability of matching with all of the units
in the target group that are within the recording and in neighboring recordings
is higher than 0.5."

Usage:
    python scripts/run_unitmatch_batch.py --config config/unitmatch_sessions.yml
    python scripts/run_unitmatch_batch.py --sessions BG_031_* --prob-threshold 0.6
"""

import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import UnitMatchPy.overlord as ov
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.default_params as dp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_waveforms_for_session(
    session_config: dict,
    waveform_source: str,
    waveform_dir: Path,
    use_iti: bool
) -> np.ndarray:
    """Load waveforms for a session."""
    session_name = session_config.get('name', 'unknown')
    
    if waveform_source == 'prepared':
        suffix = 'iti' if use_iti else 'full'
        # Try kilosort first, then bombcell
        wf_file = waveform_dir / f"{session_name}_waveforms_kilosort_{suffix}.npy"
        if not wf_file.exists():
            wf_file = waveform_dir / f"{session_name}_waveforms_bombcell_{suffix}.npy"
        
        if not wf_file.exists():
            raise FileNotFoundError(f"Prepared waveforms not found for {session_name}")
        
        return np.load(wf_file)
    
    else:
        raise NotImplementedError(f"Source '{waveform_source}' not yet implemented. Use 'prepared' waveforms.")


def compute_pairwise_probabilities(
    waveforms_list: list,
    channel_positions_list: list,
    spike_times_list: list,
    spike_clusters_list: list,
    param: dict
) -> np.ndarray:
    """Compute pairwise match probabilities between all units across all sessions.
    
    Returns:
        prob_matrix: (n_units_total, n_units_total) matrix of match probabilities
    """
    # Prepare data structures for UnitMatch
    n_sessions = len(waveforms_list)
    n_units_per_session = [w.shape[0] for w in waveforms_list]
    n_units_total = sum(n_units_per_session)
    
    # Concatenate waveforms
    waveform_all = np.concatenate(waveforms_list, axis=0)
    
    # Build session_switch and within_session arrays
    session_switch = []
    within_session = []
    for si, n_units in enumerate(n_units_per_session):
        session_switch.extend([si] * n_units)
        within_session.extend([si] * n_units)
    
    session_switch = np.array(session_switch)
    within_session = np.array(within_session)
    
    # Build clus_info
    original_ids_list = []
    session_id_list = []
    
    for si, spike_clusters in enumerate(spike_clusters_list):
        cluster_ids = np.unique(spike_clusters)
        original_ids_list.append(cluster_ids)
        session_id_list.append(np.full(len(cluster_ids), si))
    
    original_ids = np.concatenate(original_ids_list)
    session_id = np.concatenate(session_id_list)
    clus_info = {'original_ids': original_ids, 'session_id': session_id}
    
    # Update param
    param['n_units'] = n_units_total
    param['n_sessions'] = n_sessions
    param['n_channels'] = waveform_all.shape[2]
    param['spike_width'] = waveform_all.shape[1]
    
    # Set waveidx to central 50%
    spike_w = param['spike_width']
    start = spike_w // 4
    end = spike_w - start
    param['waveidx'] = np.arange(start, end)
    param['peak_loc'] = spike_w // 2
    
    # Align channel positions
    n_ch = param['n_channels']
    aligned_positions = []
    for cp in channel_positions_list:
        cp = np.asarray(cp)
        if cp.ndim == 2 and cp.shape[1] == 2:
            cp = np.concatenate([cp, np.zeros((cp.shape[0], 1))], axis=1)
        if cp.shape[0] >= n_ch:
            cp = cp[:n_ch, :]
        else:
            pad = np.zeros((n_ch - cp.shape[0], cp.shape[1]))
            cp = np.vstack([cp, pad])
        aligned_positions.append(cp)
    
    # Run UnitMatch computations
    logger.info("Extracting waveform parameters...")
    extracted = ov.extract_parameters(waveform_all, aligned_positions, clus_info, param)
    
    logger.info("Computing metric scores...")
    total_score, candidate_pairs, scores_to_include, predictors = ov.extract_metric_scores(
        extracted, session_switch, within_session, param
    )
    
    logger.info("Computing Bayesian probabilities...")
    labels = session_switch
    cond = np.unique(labels)
    
    try:
        res = bf.get_parameter_kernels(scores_to_include, param)
    except TypeError:
        res = bf.get_parameter_kernels(scores_to_include, labels, cond, param)
    
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        param_kernels = res[0]
        priors = res[1]
    else:
        param_kernels = res
        counts = np.bincount(labels.astype(int))
        priors = counts / counts.sum()
    
    output_prob = bf.apply_naive_bayes(param_kernels, priors, predictors, param, cond)
    
    # Convert to probability matrix
    if hasattr(output_prob, 'shape'):
        if tuple(output_prob.shape) == (n_units_total * n_units_total, 2):
            prob_matrix = output_prob.reshape((n_units_total, n_units_total, 2))[:, :, 1]
        elif tuple(output_prob.shape) == (n_units_total, n_units_total, 2):
            prob_matrix = output_prob[:, :, 1]
        elif tuple(output_prob.shape) == (n_units_total, n_units_total):
            prob_matrix = output_prob
        else:
            flat = np.ravel(output_prob)
            if flat.size == n_units_total * n_units_total:
                prob_matrix = flat.reshape((n_units_total, n_units_total))
            else:
                raise ValueError(f"Cannot interpret output_prob shape: {output_prob.shape}")
    else:
        raise ValueError(f"output_prob has no shape attribute: {type(output_prob)}")
    
    return prob_matrix, clus_info


def build_tracking_chains(
    prob_matrix: np.ndarray,
    clus_info: dict,
    prob_threshold: float = 0.5,
    check_neighboring: bool = True
) -> pd.DataFrame:
    """Build unit tracking chains across sessions using iterative matching.
    
    Implements the default UnitMatch algorithm from the paper.
    
    Args:
        prob_matrix: (n_units, n_units) match probability matrix
        clus_info: Dict with 'original_ids' and 'session_id' arrays
        prob_threshold: Probability threshold for matches (default 0.5)
        check_neighboring: Check neighboring recordings (default True)
    
    Returns:
        DataFrame with columns: track_id, session_id, unit_id, match_prob
    """
    n_units = prob_matrix.shape[0]
    original_ids = clus_info['original_ids']
    session_ids = clus_info['session_id']
    
    # Initialize tracking groups
    # Each group is a dict: {session_id: [(unit_idx, unit_id)]}
    tracking_groups = []
    unit_to_group = {}  # unit_idx -> group_index
    
    # Sort all potential matches by probability
    matches = []
    for i in range(n_units):
        for j in range(i + 1, n_units):
            if session_ids[i] != session_ids[j]:  # Only cross-session matches
                prob = prob_matrix[i, j]
                if prob >= prob_threshold:
                    matches.append((i, j, prob))
    
    matches.sort(key=lambda x: x[2], reverse=True)  # Sort by probability descending
    
    logger.info(f"Found {len(matches)} candidate matches above threshold {prob_threshold}")
    
    # Iteratively build tracking groups
    for unit_i, unit_j, prob in matches:
        sess_i = int(session_ids[unit_i])
        sess_j = int(session_ids[unit_j])
        id_i = int(original_ids[unit_i])
        id_j = int(original_ids[unit_j])
        
        # Check if either unit is already in a group
        group_i = unit_to_group.get(unit_i)
        group_j = unit_to_group.get(unit_j)
        
        if group_i is None and group_j is None:
            # Create new group
            new_group = {
                sess_i: [(unit_i, id_i)],
                sess_j: [(unit_j, id_j)]
            }
            group_idx = len(tracking_groups)
            tracking_groups.append(new_group)
            unit_to_group[unit_i] = group_idx
            unit_to_group[unit_j] = group_idx
        
        elif group_i is not None and group_j is None:
            # Add unit_j to group_i
            group = tracking_groups[group_i]
            
            # Check if this session already has a unit in the group
            if sess_j in group:
                # Session already has a unit - can't add another (1-to-1 matching per session)
                continue
            
            # Check neighboring recordings if enabled
            if check_neighboring:
                # Check match probability with units in neighboring sessions
                neighboring_sessions = [sess_j - 1, sess_j, sess_j + 1]
                passes_neighbor_check = True
                
                for neighbor_sess in neighboring_sessions:
                    if neighbor_sess in group:
                        for existing_unit_idx, _ in group[neighbor_sess]:
                            neighbor_prob = prob_matrix[unit_j, existing_unit_idx]
                            if neighbor_prob < prob_threshold:
                                passes_neighbor_check = False
                                break
                    if not passes_neighbor_check:
                        break
                
                if not passes_neighbor_check:
                    continue
            
            # Add to group
            group[sess_j] = [(unit_j, id_j)]
            unit_to_group[unit_j] = group_i
        
        elif group_i is None and group_j is not None:
            # Add unit_i to group_j (symmetric to above)
            group = tracking_groups[group_j]
            
            if sess_i in group:
                continue
            
            if check_neighboring:
                neighboring_sessions = [sess_i - 1, sess_i, sess_i + 1]
                passes_neighbor_check = True
                
                for neighbor_sess in neighboring_sessions:
                    if neighbor_sess in group:
                        for existing_unit_idx, _ in group[neighbor_sess]:
                            neighbor_prob = prob_matrix[unit_i, existing_unit_idx]
                            if neighbor_prob < prob_threshold:
                                passes_neighbor_check = False
                                break
                    if not passes_neighbor_check:
                        break
                
                if not passes_neighbor_check:
                    continue
            
            group[sess_i] = [(unit_i, id_i)]
            unit_to_group[unit_i] = group_j
        
        else:
            # Both in groups - try to merge if compatible
            if group_i == group_j:
                continue  # Already in same group
            
            group_a = tracking_groups[group_i]
            group_b = tracking_groups[group_j]
            
            # Check if sessions overlap
            sessions_a = set(group_a.keys())
            sessions_b = set(group_b.keys())
            
            if sessions_a & sessions_b:
                # Can't merge - sessions overlap
                continue
            
            # Merge groups (add group_b to group_a)
            for sess, units in group_b.items():
                group_a[sess] = units
                for unit_idx, _ in units:
                    unit_to_group[unit_idx] = group_i
            
            # Mark group_b as merged
            tracking_groups[group_j] = None
    
    # Remove None entries (merged groups)
    tracking_groups = [g for g in tracking_groups if g is not None]
    
    logger.info(f"Built {len(tracking_groups)} tracking chains")
    
    # Convert to DataFrame
    rows = []
    for track_id, group in enumerate(tracking_groups):
        for sess_id, units in group.items():
            for unit_idx, unit_id in units:
                # Get best match probability with any unit in the track
                best_prob = 0.0
                for other_sess, other_units in group.items():
                    if other_sess != sess_id:
                        for other_idx, _ in other_units:
                            prob = prob_matrix[unit_idx, other_idx]
                            best_prob = max(best_prob, prob)
                
                rows.append({
                    'track_id': track_id,
                    'session_id': sess_id,
                    'unit_id': unit_id,
                    'unit_idx': unit_idx,
                    'max_match_prob': best_prob
                })
    
    # Add unmatched units as single-session tracks
    for unit_idx in range(n_units):
        if unit_idx not in unit_to_group:
            rows.append({
                'track_id': len(tracking_groups) + unit_idx,
                'session_id': int(session_ids[unit_idx]),
                'unit_id': int(original_ids[unit_idx]),
                'unit_idx': unit_idx,
                'max_match_prob': 0.0
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['track_id', 'session_id'])
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Run batch UnitMatch tracking')
    parser.add_argument('--config', type=str, default='config/unitmatch_sessions.yml',
                        help='Path to UnitMatch config file')
    parser.add_argument('--sessions', type=str, nargs='+', default=None,
                        help='Process specific sessions (default: all in config)')
    parser.add_argument('--waveform-dir', type=str, default='png_output/unitmatch_waveforms',
                        help='Directory with prepared waveforms')
    parser.add_argument('--prob-threshold', type=float, default=None,
                        help='Probability threshold (overrides config, default 0.5)')
    parser.add_argument('--no-neighbor-check', action='store_true',
                        help='Disable neighboring recordings check')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: table_output/unitmatch/tracking_chains.csv)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters
    wf_config = config.get('waveform_config', {})
    match_config = config.get('matching_params', {})
    
    waveform_dir = Path(args.waveform_dir)
    prob_threshold = args.prob_threshold or match_config.get('prob_threshold', 0.5)
    check_neighboring = not args.no_neighbor_check and match_config.get('check_neighboring_recordings', True)
    use_iti = wf_config.get('use_iti', False)
    
    output_file = args.output
    if output_file is None:
        output_dir = Path(config.get('report_dir', 'table_output/unitmatch'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'tracking_chains.csv'
    output_file = Path(output_file)
    
    logger.info(f"Configuration:")
    logger.info(f"  Waveform dir: {waveform_dir}")
    logger.info(f"  Probability threshold: {prob_threshold}")
    logger.info(f"  Check neighboring: {check_neighboring}")
    logger.info(f"  Use ITI: {use_iti}")
    logger.info(f"  Output: {output_file}")
    
    # Get sessions
    sessions_list = config.get('sessions', [])
    if args.sessions:
        sessions_list = [s for s in sessions_list if s.get('name') in args.sessions]
    
    if len(sessions_list) < 2:
        logger.error("Need at least 2 sessions for tracking")
        return 1
    
    logger.info(f"Processing {len(sessions_list)} sessions")
    
    # Load waveforms and metadata for all sessions
    waveforms_list = []
    channel_positions_list = []
    spike_times_list = []
    spike_clusters_list = []
    session_names = []
    
    for sess_config in sessions_list:
        session_name = sess_config.get('name', 'unknown')
        ks_path = Path(sess_config['path'])
        
        logger.info(f"Loading session: {session_name}")
        
        # Load waveforms
        try:
            waveforms = load_waveforms_for_session(sess_config, 'prepared', waveform_dir, use_iti)
            waveforms_list.append(waveforms)
        except Exception as e:
            logger.error(f"Failed to load waveforms: {e}")
            return 1
        
        # Load Kilosort metadata
        ch_pos = np.load(ks_path / "channel_positions.npy")
        if ch_pos.ndim == 2 and ch_pos.shape[1] == 2:
            ch_pos = np.concatenate([ch_pos, np.zeros((ch_pos.shape[0], 1))], axis=1)
        channel_positions_list.append(ch_pos)
        
        spike_times = np.load(ks_path / "spike_times.npy")
        spike_clusters = np.load(ks_path / "spike_clusters.npy").flatten()
        spike_times_list.append(spike_times)
        spike_clusters_list.append(spike_clusters)
        session_names.append(session_name)
    
    # Get default UnitMatch parameters
    param = dp.get_default_param()
    
    # Apply config overrides
    param['no_shanks'] = config.get('no_shanks', 4)
    param['shank_dist'] = config.get('shank_dist', 250)
    
    spatial_config = config.get('spatial_constraints', {})
    if 'max_cross_shank_distance_um' in spatial_config:
        param['max_cross_shank_distance_um'] = spatial_config['max_cross_shank_distance_um']
    
    # Compute pairwise probabilities
    logger.info("Computing pairwise match probabilities...")
    prob_matrix, clus_info = compute_pairwise_probabilities(
        waveforms_list,
        channel_positions_list,
        spike_times_list,
        spike_clusters_list,
        param
    )
    
    logger.info(f"Probability matrix shape: {prob_matrix.shape}")
    logger.info(f"Probability range: [{np.min(prob_matrix):.3f}, {np.max(prob_matrix):.3f}]")
    
    # Build tracking chains
    logger.info("Building tracking chains...")
    tracking_df = build_tracking_chains(
        prob_matrix,
        clus_info,
        prob_threshold=prob_threshold,
        check_neighboring=check_neighboring
    )
    
    # Add session names
    session_id_to_name = {i: name for i, name in enumerate(session_names)}
    tracking_df['session_name'] = tracking_df['session_id'].map(session_id_to_name)
    
    # Save results
    tracking_df.to_csv(output_file, index=False)
    logger.info(f"Saved tracking chains to: {output_file}")
    
    # Print summary statistics
    n_tracks = tracking_df['track_id'].nunique()
    n_multi_session = tracking_df.groupby('track_id')['session_id'].nunique()
    n_multi_session = (n_multi_session > 1).sum()
    
    logger.info(f"\nTracking summary:")
    logger.info(f"  Total tracks: {n_tracks}")
    logger.info(f"  Multi-session tracks: {n_multi_session}")
    logger.info(f"  Single-session tracks: {n_tracks - n_multi_session}")
    
    # Show longest tracks
    track_lengths = tracking_df.groupby('track_id').size().sort_values(ascending=False)
    logger.info(f"\nLongest tracks:")
    for track_id, length in track_lengths.head(10).items():
        logger.info(f"  Track {track_id}: {length} sessions")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
