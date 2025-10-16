"""Run UnitMatch on sessions listed in config/unitmatch_sessions.yml (pairwise)

This script is a thin adapter that uses UnitMatchPy internals to run a
pairwise UnitMatch on the KS folders listed in the config. It produces a
match table saved to `table_output/unitmatch/`.

This is intended for quick testing with two sessions. For larger batches
or GPU/parallel setups, prefer the UnitMatch project's recommended runner.
"""
import os
from pathlib import Path
import yaml
import UnitMatchPy.extract_raw_data as er
import UnitMatchPy.overlord as ov
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.save_utils as su
import inspect
import importlib


def derive_shanks_from_channel_positions(ch_pos, expected_shanks=None, gap_factor=3.0):
    """
    Heuristic to derive shank assignments from channel_positions (no external deps).

    - ch_pos: (n_ch, 2 or 3) array of (x,y[,z]) coordinates
    - expected_shanks: if provided, try to use this as number of shanks
    - gap_factor: multiplier on median adjacent-x-gap to define large gaps

    Returns: (no_shanks, shank_dist, channel_shanks)
    - channel_shanks: array of length n_ch with shank index (0..no_shanks-1)
    - shank_dist: median center-to-center distance between shanks (or fallback)

    This is intentionally conservative: if we cannot find clear gaps we
    fall back to using `expected_shanks` (if given) or return None to let the
    caller keep defaults.
    """
    import numpy as _np

    cp = _np.asarray(ch_pos)
    if cp.ndim != 2 or cp.shape[1] < 1:
        return None, None, None

    x = cp[:, 0].astype(float)
    # Work with unique sorted x positions (round to reduce floating noise)
    xs_unique = _np.unique(_np.round(x, 6))
    if xs_unique.size <= 1:
        return None, None, None

    diffs = _np.diff(_np.sort(xs_unique))
    if diffs.size == 0:
        return None, None, None

    med = float(_np.median(diffs))
    # Define a threshold for large gaps between channel columns
    thresh = max(med * gap_factor, _np.percentile(diffs, 90))
    # Find indices in the unique-sorted x where a large gap occurs
    breaks = _np.where(diffs > thresh)[0]
    # Number of shanks is segments between breaks
    no_shanks = int(breaks.size + 1)

    # If we found only one shank but expected_shanks is provided, attempt
    # to split into that many shanks by quantiles
    if no_shanks == 1 and expected_shanks is not None and expected_shanks > 1:
        no_shanks = int(expected_shanks)

    # Build centroids for each segment
    segments = []
    start = 0
    for b in list(breaks) + [len(xs_unique) - 1]:
        end = b + 1
        seg_xs = xs_unique[start:end]
        if seg_xs.size > 0:
            segments.append(float(_np.mean(seg_xs)))
        start = end

    # If segments detection failed or inconsistent, fall back to quantile-based split
    if len(segments) < 1 or len(segments) != no_shanks:
        # fallback: split unique x into `no_shanks` quantiles
        if no_shanks <= 1:
            return None, None, None
        qs = [_np.percentile(xs_unique, 100.0 * i / no_shanks) for i in range(no_shanks + 1)]
        centers = []
        for i in range(no_shanks):
            seg = xs_unique[(xs_unique >= qs[i]) & (xs_unique <= qs[i + 1])]
            centers.append(float(_np.mean(seg)) if seg.size > 0 else float((qs[i] + qs[i + 1]) / 2.0))
        segments = centers

    centers = _np.array(segments)
    # Assign each channel to nearest center
    channel_shanks = _np.argmin(_np.abs(x[:, None] - centers[None, :]), axis=1)

    # Compute median spacing between adjacent shank centers as shank_dist
    if centers.size > 1:
        shank_dist = float(_np.median(_np.abs(_np.diff(_np.sort(centers)))))
    else:
        shank_dist = None

    return no_shanks, shank_dist, channel_shanks


def _save_shank_assignment_plot(channel_positions, channel_shanks, out_path):
    """Save a quick diagnostic plot of channel x positions colored by derived shank."""
    try:
        import matplotlib.pyplot as plt
        import numpy as _np
        cp = _np.asarray(channel_positions)
        x = cp[:, 0].astype(float)
        y = cp[:, 1].astype(float) if cp.shape[1] > 1 else _np.zeros_like(x)
        sh = _np.asarray(channel_shanks)
        plt.figure(figsize=(6, 4))
        sc = plt.scatter(x, y, c=sh, cmap='tab10', s=40)
        plt.colorbar(sc, label='derived shank')
        for i, (xx, yy) in enumerate(zip(x, y)):
            plt.text(xx, yy, str(i), fontsize=6, alpha=0.6)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        plt.title('Derived shank assignment (channel positions)')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print('Failed to save shank assignment plot:', e)


def main():
    cfg = yaml.safe_load(Path('config/unitmatch_sessions.yml').read_text())
    sessions = [s['path'] for s in cfg.get('sessions', [])]
    if len(sessions) < 2:
        raise SystemExit('Need at least two sessions to run UnitMatch')
    # For test we take the first two sessions
    ks_dirs = sessions[:2]
    print('Running UnitMatch on:', ks_dirs)


    # Build inputs for UnitMatch using Bombcell's average waveforms (already extracted)
    import numpy as np
    avg_waveforms_list = []
    channel_positions_list = []
    spike_times_list = []
    spike_ids_list = []
    clus_info_list = []
    session_switch = []
    within_session = []
    n_units_total = 0

    for si, ks in enumerate(ks_dirs):
        ks_path = Path(ks)
        # Prefer explicit mapping in the config
        bc_dir = None
        try:
            cfg = yaml.safe_load(Path('config/unitmatch_sessions.yml').read_text())
            bc_dir = cfg.get('sessions', [])[si].get('bombcell_dir')
            if bc_dir is not None:
                bc_dir = Path(bc_dir)
        except Exception:
            bc_dir = None

        if bc_dir is None or not bc_dir.exists():
            bc_dir = Path('notebooks') / ks_path.name.replace('_g0_imec0', '') / 'bombcell'
            # fallback: try to find a bombcell folder under notebooks that contains the session name
            if not bc_dir.exists():
                candidates = list(Path('notebooks').glob('*'))
                found = None
                for c in candidates:
                    if (c / 'bombcell').exists():
                        found = c / 'bombcell'
                        break
                if found is not None:
                    bc_dir = found

        templates_file = bc_dir / 'templates._bc_rawWaveforms.npy'
        if not templates_file.exists():
            templates_file = bc_dir / '_bc_rawWaveforms_kilosort_format.npy'
        if not templates_file.exists():
            raise SystemExit(f'No Bombcell waveform file found for session {ks}')

        w = np.load(templates_file, allow_pickle=True)
        # If 0-d array wrapping None, try alternate file
        if getattr(w, 'ndim', None) == 0 and w.item() is None:
            raise SystemExit(f'Bombcell waveform file for {ks} is empty: {templates_file}')

        # Ensure waveform shape is (n_units, spike_w, n_ch) or (n_units, spike_w, n_ch, 2)
        if w.ndim == 3:
            w = w[..., np.newaxis]
            w = np.repeat(w, 2, axis=-1)
        elif w.ndim == 4:
            pass
        else:
            raise SystemExit(f'Unrecognized waveform array shape for {ks}: {w.shape}')

        avg_waveforms_list.append(w)
        n_units = w.shape[0]
        n_units_total += n_units
        session_switch.extend([si] * n_units)
        within_session.extend([si] * n_units)

        # load channel positions and spike files from KS folder
        ch_pos = np.load(Path(ks) / 'channel_positions.npy')
        # UnitMatch expects channel positions with 3 columns (x,y,z). If only x,y
        # are present, pad a zero z-coordinate so shapes match.
        if ch_pos.ndim == 2 and ch_pos.shape[1] == 2:
            ch_pos = np.concatenate([ch_pos, np.zeros((ch_pos.shape[0], 1), dtype=ch_pos.dtype)], axis=1)
        spike_times = np.load(Path(ks) / 'spike_times.npy')
        spike_clusters = np.load(Path(ks) / 'spike_clusters.npy')
        channel_positions_list.append(ch_pos)
        spike_times_list.append(spike_times)
        spike_ids_list.append(spike_clusters)

        # Build clus_info for this session
        # original_ids: unique cluster ids in this session
        orig_ids = np.unique(spike_clusters)
        sess_id = np.ones_like(orig_ids) * si
        clus_info_list.append({'original_ids': orig_ids.reshape(-1,1), 'session_id': sess_id.reshape(-1,1)})

    # Concatenate waveforms
    waveform_all = np.concatenate(avg_waveforms_list, axis=0)

    # Concatenate clus_info
    original_ids = np.concatenate([ci['original_ids'].squeeze() for ci in clus_info_list])
    session_id = np.concatenate([ci['session_id'].squeeze() for ci in clus_info_list])
    clus_info = {'original_ids': original_ids, 'session_id': session_id}

    # Load default params from UnitMatchPy.default_params
    import UnitMatchPy.default_params as dp
    param = dp.get_default_param()
    # Diagnostic: print UnitMatchPy location and key function signatures
    try:
        um_mod = importlib.import_module('UnitMatchPy')
        print('UnitMatchPy module file:', getattr(um_mod, '__file__', 'builtin'))
    except Exception as e:
        print('Could not import UnitMatchPy module location:', e)
    try:
        print('ov.extract_metric_scores signature:', inspect.signature(ov.extract_metric_scores))
    except Exception:
        print('Could not inspect ov.extract_metric_scores')
    try:
        print('bf.get_parameter_kernels signature:', inspect.signature(bf.get_parameter_kernels))
    except Exception:
        print('Could not inspect bf.get_parameter_kernels')
    try:
        print('bf.apply_naive_bayes signature:', inspect.signature(bf.apply_naive_bayes))
    except Exception:
        print('Could not inspect bf.apply_naive_bayes')
    param['n_units'] = n_units_total
    param['n_sessions'] = len(ks_dirs)
    # Use Neuropixels 2.0 4-shank geometry by default for these probes
    # (user provided: 4 shanks, center-to-center spacing ~250 um)
    # Allow opt-in auto shank derivation via config: use_auto_shanks: true
    cfg_all = yaml.safe_load(Path('config/unitmatch_sessions.yml').read_text())
    use_auto = bool(cfg_all.get('use_auto_shanks', False))

    # Default fallbacks
    param['no_shanks'] = 4
    param['shank_dist'] = param.get('shank_dist', 250)

    if use_auto:
        # Try to derive shanks from the first session's channel_positions (assumes same probe)
        try:
            derived = derive_shanks_from_channel_positions(channel_positions_list[0], expected_shanks=4)
            derived_no_shanks, derived_shank_dist, channel_shanks = derived
            if derived_no_shanks is not None:
                print(f'Auto-derived {derived_no_shanks} shanks; shank_dist={derived_shank_dist}')
                param['no_shanks'] = derived_no_shanks
                if derived_shank_dist is not None:
                    param['shank_dist'] = derived_shank_dist
                # attach channel_shanks for UnitMatch usage if it's expected downstream
                # pad/truncate to n_ch
                try:
                    cs = channel_shanks.astype(int)
                    if cs.shape[0] >= n_ch:
                        cs = cs[:n_ch]
                    else:
                        pad_rows = np.zeros((n_ch - cs.shape[0],), dtype=int)
                        cs = np.concatenate([cs, pad_rows])
                    param['channel_shanks'] = cs
                    # Save a diagnostic plot showing derived shanks
                    try:
                        out_dir = Path(cfg_all.get('report_dir', 'table_output/unitmatch'))
                        out_dir.mkdir(parents=True, exist_ok=True)
                        png_path = out_dir / 'shank_assignment.png'
                        _save_shank_assignment_plot(channel_positions_list[0], cs, str(png_path))
                        print('Saved shank assignment plot to', png_path)
                    except Exception as _e:
                        print('Could not save shank assignment plot:', _e)
                except Exception:
                    # best-effort: skip attaching channel_shanks
                    pass
            else:
                print('Auto-shank derivation returned no result; keeping defaults')
        except Exception as e:
            print('Auto-shank derivation failed:', e)
            print('Keeping default shank settings')

    # Run UnitMatch computations
    print('Extracting waveform parameters...')
    channel_pos = channel_positions_list
    # number of channels inferred from waveform arrays
    n_ch = int(waveform_all.shape[2])
    param['n_channels'] = n_ch

    # Ensure spike width and waveidx match the Bombcell templates
    spike_w = int(waveform_all.shape[1])
    param['spike_width'] = spike_w
    # choose a central window for waveidx (middle 50% of samples)
    start = spike_w // 4
    end = spike_w - start
    param['waveidx'] = np.arange(start, end)
    param['peak_loc'] = spike_w // 2

    # Ensure each channel_positions array has shape (n_ch, 3) by truncating or padding
    aligned_channel_positions = []
    for cp in channel_positions_list:
        cp = np.asarray(cp)
        # pad a 3rd column if missing
        if cp.ndim == 2 and cp.shape[1] == 2:
            cp = np.concatenate([cp, np.zeros((cp.shape[0], 1), dtype=cp.dtype)], axis=1)
        # truncate or pad rows to n_ch
        if cp.shape[0] >= n_ch:
            cp = cp[:n_ch, :]
        else:
            # pad rows with zeros
            pad_rows = np.zeros((n_ch - cp.shape[0], cp.shape[1]), dtype=cp.dtype)
            cp = np.vstack([cp, pad_rows])
        aligned_channel_positions.append(cp)
    channel_positions_list = aligned_channel_positions
    # use the aligned channel positions for extraction
    channel_pos = channel_positions_list
    extracted = ov.extract_parameters(waveform_all, channel_pos, clus_info, param)
    print('Computing metric scores...')
    # extract_metric_scores returns (total_score, candidate_pairs, scores_to_include, predictors)
    total_score, candidate_pairs, scores_to_include, predictors = ov.extract_metric_scores(
        extracted, np.array(session_switch), np.array(within_session), param
    )

    print('Computing parameter kernels and priors for Bayesian combination...')
    # get_parameter_kernels will compute kernels and priors needed by apply_naive_bayes
    # Build labels and cond for versions that require them
    labels = np.array(session_switch)
    cond = np.unique(labels)
    # Call get_parameter_kernels with fallbacks and accept variable-length returns
    try:
        res = bf.get_parameter_kernels(scores_to_include, param)
    except TypeError:
        res = bf.get_parameter_kernels(scores_to_include, labels, cond, param)

    # unpack kernel/prior from returned object (support different return types)
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        param_kernels = res[0]
        priors = res[1]
    elif hasattr(res, 'ndim'):
        # Some versions return only the parameter_kernels ndarray; compute simple priors from labels
        param_kernels = res
        counts = np.bincount(labels.astype(int))
        # normalize to get priors per cond (align to unique cond ordering)
        # cond is np.unique(labels)
        priors = counts / counts.sum()
    else:
        raise RuntimeError('Unexpected return from get_parameter_kernels: %r' % (res,))

    print('Applying Bayes...')
    # predictors is expected from extract_metric_scores; cond may be None or returned elsewhere
    # Call apply_naive_bayes with explicit cond (UnitMatchPy versions expect this)
    output_prob = bf.apply_naive_bayes(param_kernels, priors, predictors, param, cond)

    print('Making match table (custom writer)...')
    import json
    out_dir = Path(cfg.get('report_dir', 'table_output/unitmatch'))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostic dump to help debugging shapes
    diag = {
        'waveform_all_shape': getattr(waveform_all, 'shape', None),
        'total_score_shape': getattr(total_score, 'shape', None),
        'candidate_pairs_shape': getattr(candidate_pairs, 'shape', None) if 'candidate_pairs' in locals() else None,
        'scores_to_include_shape': getattr(scores_to_include, 'shape', None) if 'scores_to_include' in locals() else None,
        'predictors_shape': getattr(predictors, 'shape', None) if 'predictors' in locals() else None,
        'output_prob_shape': getattr(output_prob, 'shape', None),
        'param_kernels_shape': getattr(param_kernels, 'shape', None) if 'param_kernels' in locals() else None,
        'priors_shape': getattr(priors, 'shape', None) if 'priors' in locals() else None,
        'n_units': param.get('n_units')
    }
    # Additional debug info placeholders
    diag.update({
        'candidate_mask_true_count': None,
        'sess_ids_unique': None,
        'prob_matrix_max': None,
        'prob_matrix_min': None,
    })
    (out_dir / 'unitmatch_pair_diagnostic.json').write_text(json.dumps(diag, indent=2))

    import csv
    # Helper arrays
    orig_ids = clus_info['original_ids']
    sess_ids = clus_info['session_id']
    session_names = [Path(p).name for p in ks_dirs]

    # Interpret candidate_pairs and output_prob shapes produced by UnitMatchPy
    n = param['n_units']
    pairs = None
    probs = None

    # Simplified and deterministic extraction: reshape output_prob into (n,n,2) when possible
    prob_matrix = None
    if hasattr(output_prob, 'shape'):
        if tuple(output_prob.shape) == (n * n, 2):
            out_reshaped = np.array(output_prob).reshape((n, n, 2))
            prob_matrix = out_reshaped[:, :, 1]
        elif tuple(output_prob.shape) == (n, n, 2):
            prob_matrix = np.array(output_prob)[:, :, 1]
        elif tuple(output_prob.shape) == (n, n):
            prob_matrix = np.array(output_prob)

    if prob_matrix is None:
        # try ravel fallback
        flat = np.ravel(output_prob)
        if flat.size == n * n:
            prob_matrix = flat.reshape((n, n))

    if prob_matrix is None:
        raise RuntimeError('Unable to interpret output_prob into an (n,n) match-probability matrix. See diagnostic JSON.')

    # record prob stats
    try:
        diag['prob_matrix_max'] = float(np.nanmax(prob_matrix))
        diag['prob_matrix_min'] = float(np.nanmin(prob_matrix))
    except Exception:
        pass

    # Build list of cross-session pairs (i<j) and probabilities, then sort by prob desc
    pair_list = []
    prob_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if sess_ids[i] != sess_ids[j]:
                # ensure ordering session_i < session_j for consistency
                if sess_ids[i] < sess_ids[j]:
                    pair_list.append((i, j))
                    prob_list.append(float(prob_matrix[i, j]))
                else:
                    pair_list.append((j, i))
                    prob_list.append(float(prob_matrix[j, i]))

    pairs = np.array(pair_list)
    probs = np.array(prob_list)
    # sort by probability descending
    order = np.argsort(-probs)
    pairs = pairs[order]
    probs = probs[order]

    if pairs is None or probs is None:
        raise RuntimeError('Unable to interpret candidate_pairs/output_prob shapes for CSV output. See diagnostic JSON.')

    # finalize diag fields about session ids
    try:
        diag['sess_ids_unique'] = np.unique(sess_ids).tolist()
    except Exception:
        diag['sess_ids_unique'] = None
    (out_dir / 'unitmatch_pair_diagnostic.json').write_text(json.dumps(diag, indent=2))

    # Debug: print number of pairs and first few samples
    try:
        num_pairs = int(pairs.shape[0])
    except Exception:
        num_pairs = 0
    print(f'Preparing to write {num_pairs} pairs to CSV')
    sample_n = min(10, num_pairs)
    if num_pairs > 0:
        print('Sample pairs (first', sample_n, '):')
        for (a, b), p in list(zip(pairs.tolist(), probs.tolist()))[:sample_n]:
            print('  ', int(a), int(b), float(p))

    # Build CSV rows
    csv_path = out_dir / 'unitmatch_pair_matches.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['unit_idx_a', 'unit_idx_b', 'unit_id_a', 'unit_id_b', 'session_a', 'session_b', 'prob'])
        for (a, b), p in zip(pairs.tolist(), probs.tolist()):
            a = int(a); b = int(b)
            writer.writerow([a, b, int(orig_ids[a]), int(orig_ids[b]), session_names[int(sess_ids[a])], session_names[int(sess_ids[b])], float(p)])

    print('Saved custom match CSV to', csv_path)


if __name__ == '__main__':
    main()
