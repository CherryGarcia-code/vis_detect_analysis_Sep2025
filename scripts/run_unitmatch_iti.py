"""Run UnitMatch on sessions listed in config/unitmatch_sessions.yml (pairwise)

This script is a thin adapter that uses UnitMatchPy internals to run a
pairwise UnitMatch on the KS folders listed in the config. It produces a
match table saved to `table_output/unitmatch/`.

This is intended for quick testing with two sessions. For larger batches
or GPU/parallel setups, prefer the UnitMatch project's recommended runner.

Updated to support flexible waveform loading from:
- Bombcell pre-computed waveforms
- Kilosort templates (direct extraction)
- Pre-prepared waveforms from prepare_waveforms_for_unitmatch.py
"""

from pathlib import Path
import argparse
import inspect
import importlib
import sys
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import yaml
import UnitMatchPy.overlord as ov
import UnitMatchPy.bayes_functions as bf


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from visdetect.core.legacy_io import load_session, save_session
from src.visdetect.io import load_mat_file_to_session
from src.unit_tracking import extract_iti_waveforms_from_raw


def _stable_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable dict with sorted keys for caching tokens."""
    serializable = {}
    for key in sorted(meta.keys()):
        val = meta[key]
        if isinstance(val, Path):
            serializable[key] = str(val)
        else:
            serializable[key] = val
    return serializable


def _cache_token(meta: Dict[str, Any]) -> str:
    payload = json.dumps(_stable_meta(meta), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _cache_paths(cache_dir: Path, token: str) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{token}.npz", cache_dir / f"{token}.json"


def _load_cached_waveforms(cache_dir: Path, meta: Dict[str, Any]):
    token = _cache_token(meta)
    npz_path, meta_path = _cache_paths(cache_dir, token)
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        stored_meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    if _stable_meta(meta) != stored_meta.get("meta"):
        return None
    data = np.load(npz_path, allow_pickle=False)
    wave = data["waveforms"]
    clusters = data["cluster_ids"].astype(int).tolist()
    diag = stored_meta.get("diagnostics", {})
    return wave, clusters, diag, token


def _save_cached_waveforms(cache_dir: Path, meta: Dict[str, Any], wave, clusters, diagnostics):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = _cache_token(meta)
    npz_path, meta_path = _cache_paths(cache_dir, token)
    np.savez_compressed(npz_path, waveforms=wave, cluster_ids=np.asarray(clusters, dtype=int))
    payload = {
        "meta": _stable_meta(meta),
        "diagnostics": diagnostics,
        "cached_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    return token


def _infer_session_label(session_config: dict) -> Optional[str]:
    if session_config.get("name"):
        return session_config["name"]
    path = session_config.get("path")
    if not path:
        return None
    ks_path = Path(path)
    name_parts = ks_path.name.split("_g0")[0]
    return name_parts


def resolve_session_pickle(session_config: dict) -> Optional[Path]:
    explicit = session_config.get("session_pkl")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    label = _infer_session_label(session_config)
    if label:
        candidate = REPO / "data" / f"{label}.pkl"
        if candidate.exists():
            return candidate
        mat_candidate = candidate.with_suffix('.mat')
        if mat_candidate.exists():
            print(f"Session pickle missing for {label}, converting from MAT: {mat_candidate.name}")
            session = load_mat_file_to_session(str(mat_candidate))
            save_session(session, str(candidate))
            return candidate
    return None


def resolve_raw_ap_path(session_config: dict) -> Optional[Path]:
    explicit = session_config.get("raw_ap")
    if explicit:
        return Path(explicit)
    path_value = session_config.get("path")
    if not path_value:
        return None
    ks_path = Path(path_value)
    folder = ks_path.name
    base = folder
    if folder.endswith("_imec0"):
        base = folder[: -len("_imec0")]
    raw_name = f"{base}_tcat.imec0.ap.bin"
    candidate = ks_path / raw_name
    if candidate.exists():
        return candidate
    # fallback: search for *.ap.bin in folder
    matches = list(ks_path.glob("*.ap.bin"))
    return matches[0] if matches else None


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
        qs = [
            _np.percentile(xs_unique, 100.0 * i / no_shanks)
            for i in range(no_shanks + 1)
        ]
        centers = []
        for i in range(no_shanks):
            seg = xs_unique[(xs_unique >= qs[i]) & (xs_unique <= qs[i + 1])]
            centers.append(
                float(_np.mean(seg))
                if seg.size > 0
                else float((qs[i] + qs[i + 1]) / 2.0)
            )
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
        sc = plt.scatter(x, y, c=sh, cmap="tab10", s=40)
        plt.colorbar(sc, label="derived shank")
        for i, (xx, yy) in enumerate(zip(x, y)):
            plt.text(xx, yy, str(i), fontsize=6, alpha=0.6)
        plt.xlabel("x (um)")
        plt.ylabel("y (um)")
        plt.title("Derived shank assignment (channel positions)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print("Failed to save shank assignment plot:", e)


def load_waveforms_from_source(
    session_config: dict,
    session_idx: int,
    waveform_source: str,
    waveform_dir: Path = None,
    use_iti: bool = False
):
    """Load waveforms from specified source.
    
    Args:
        session_config: Session configuration dict from config
        session_idx: Index of this session
        waveform_source: 'bombcell', 'kilosort', or 'prepared'
        waveform_dir: Directory containing prepared waveforms (if source='prepared')
        use_iti: Whether to use ITI-filtered waveforms
    
    Returns:
        waveforms: (n_units, spike_w, n_ch, 2) array
    """
    session_name = session_config.get('name', f'session_{session_idx}')
    ks_path = Path(session_config['path'])
    
    if waveform_source == 'prepared':
        # Load pre-prepared waveforms
        if waveform_dir is None:
            waveform_dir = Path('png_output/unitmatch_waveforms')
        
        suffix = 'iti' if use_iti else 'full'
        # Try to find matching prepared waveform file
        wf_file = waveform_dir / f"{session_name}_waveforms_kilosort_{suffix}.npy"
        if not wf_file.exists():
            wf_file = waveform_dir / f"{session_name}_waveforms_bombcell_{suffix}.npy"
        
        if not wf_file.exists():
            raise FileNotFoundError(f"Prepared waveforms not found: {wf_file}")
        
        print(f"Loading prepared waveforms from: {wf_file}")
        w = np.load(wf_file)
        return w
    
    elif waveform_source == 'kilosort':
        # Load directly from Kilosort templates
        templates_file = ks_path / "templates.npy"
        if not templates_file.exists():
            raise FileNotFoundError(f"Kilosort templates not found: {templates_file}")
        
        print(f"Loading Kilosort templates from: {templates_file}")
        templates = np.load(templates_file)  # (n_templates, n_samples, n_channels)
        spike_templates = np.load(ks_path / "spike_templates.npy").flatten()
        spike_clusters = np.load(ks_path / "spike_clusters.npy").flatten()
        
        # Get unique clusters
        cluster_ids = np.unique(spike_clusters)
        n_units = len(cluster_ids)
        n_samples = templates.shape[1]
        n_channels = templates.shape[2]
        
        # Compute mean waveform for each cluster with cross-validation
        waveforms = np.zeros((n_units, n_samples, n_channels, 2), dtype=np.float32)
        
        for i, cluster_id in enumerate(cluster_ids):
            spike_idx = np.where(spike_clusters == cluster_id)[0]
            if len(spike_idx) < 10:
                continue
            
            cluster_template_ids = spike_templates[spike_idx]
            
            # Split for cross-validation
            n_half = len(spike_idx) // 2
            first_half = cluster_template_ids[:n_half]
            second_half = cluster_template_ids[n_half:2*n_half]
            
            waveforms[i, :, :, 0] = templates[first_half].mean(axis=0)
            waveforms[i, :, :, 1] = templates[second_half].mean(axis=0)
        
        return waveforms
    
    elif waveform_source == 'bombcell':
        # Load from Bombcell
        bc_dir = session_config.get('bombcell_dir')
        if bc_dir is None or not Path(bc_dir).exists():
            # Try to find bombcell dir
            bc_dir = Path('notebooks') / ks_path.name.replace('_g0_imec0', '') / 'bombcell'
            if not bc_dir.exists():
                raise FileNotFoundError(f"Bombcell directory not found for session {session_name}")
        
        bc_dir = Path(bc_dir)
        templates_file = bc_dir / "templates._bc_rawWaveforms.npy"
        if not templates_file.exists():
            templates_file = bc_dir / "_bc_rawWaveforms_kilosort_format.npy"
        if not templates_file.exists():
            raise FileNotFoundError(f"Bombcell waveforms not found: {templates_file}")
        
        print(f"Loading Bombcell waveforms from: {templates_file}")
        w = np.load(templates_file, allow_pickle=True)
        
        if getattr(w, "ndim", None) == 0 and w.item() is None:
            raise ValueError(f"Bombcell waveform file is empty: {templates_file}")
        
        # Ensure shape is (n_units, spike_w, n_ch, 2)
        if w.ndim == 3:
            w = w[..., np.newaxis]
            w = np.repeat(w, 2, axis=-1)
        elif w.ndim == 4:
            pass
        else:
            raise ValueError(f"Unrecognized waveform shape: {w.shape}")
        
        return w
    
    else:
        raise ValueError(f"Unknown waveform source: {waveform_source}")


def main():
    parser = argparse.ArgumentParser(description='Run UnitMatch on a pair of sessions')
    parser.add_argument('--config', type=str, default='config/unitmatch_sessions.yml',
                        help='Path to UnitMatch config file')
    parser.add_argument('--waveform-source', type=str, default=None,
                        choices=['bombcell', 'kilosort', 'prepared'],
                        help='Source for waveforms (overrides config)')
    parser.add_argument('--waveform-dir', type=str, default=None,
                        help='Directory with prepared waveforms (for source=prepared)')
    parser.add_argument('--use-iti', action='store_true',
                        help='Use ITI-filtered waveforms')
    parser.add_argument('--iti-max-spikes', type=int, default=None,
                        help='Max spikes per unit when building ITI waveforms (default 500)')
    parser.add_argument('--iti-min-spikes', type=int, default=None,
                        help='Minimum spikes per unit required for ITI waveform (default 80)')
    parser.add_argument('--iti-min-half', type=int, default=None,
                        help='Minimum spikes per half-split (default 20)')
    parser.add_argument('--iti-window-mode', type=str, default=None,
                        choices=['all', 'uniform'],
                        help="How to sample ITI windows when building waveforms ('all' or 'uniform')")
    parser.add_argument('--iti-max-windows', type=int, default=None,
                        help='If window-mode=uniform, limit to this many ITI windows per session (default from config)')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable tqdm progress bars during waveform extraction')
    parser.add_argument('--iti-cache-dir', type=str, default=None,
                        help='Directory to cache ITI waveforms for reuse across runs')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable ITI waveform caching')
    
    args = parser.parse_args()
    
    cfg = yaml.safe_load(Path(args.config).read_text())
    
    # Get waveform source from args or config
    waveform_source = args.waveform_source
    if waveform_source is None:
        wf_config = cfg.get('waveform_config', {})
        waveform_source = wf_config.get('source', 'kilosort')
        # If source is 'both', prefer 'kilosort' for this script
        if waveform_source == 'both':
            waveform_source = 'kilosort'
    
    wf_cfg = cfg.get('waveform_config', {})
    use_iti = args.use_iti or wf_cfg.get('use_iti', False)
    waveform_dir = Path(args.waveform_dir) if args.waveform_dir else None
    max_spikes_per_unit = args.iti_max_spikes or wf_cfg.get('max_spikes_per_unit', 500)
    min_spikes_per_unit = args.iti_min_spikes or wf_cfg.get('min_spikes_per_unit', 80)
    min_spikes_per_half = args.iti_min_half or wf_cfg.get('min_spikes_per_half', 20)
    window_sampling = args.iti_window_mode or wf_cfg.get('iti_window_mode', 'all')
    max_iti_windows = args.iti_max_windows or wf_cfg.get('max_iti_windows')
    show_progress = False if args.no_progress else wf_cfg.get('show_progress', True)
    cache_waveforms = wf_cfg.get('cache_waveforms', True) and not args.no_cache
    cache_dir = args.iti_cache_dir or wf_cfg.get('iti_cache_dir')
    if cache_dir is None:
        cache_dir = Path(cfg.get('report_dir', 'table_output/unitmatch')) / 'iti_cache'
    else:
        cache_dir = Path(cache_dir)
    if cache_waveforms:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Waveform source: {waveform_source}")
    print(f"Use ITI: {use_iti}")
    if use_iti:
        print(f"ITI window sampling: {window_sampling}")
        if window_sampling == 'uniform' and max_iti_windows:
            print(f" - limiting to {max_iti_windows} windows per session")
        if cache_waveforms:
            print(f"ITI waveform cache: {cache_dir}")
        else:
            print("ITI waveform cache disabled")
    
    sessions = cfg.get("sessions", [])
    if len(sessions) < 2:
        raise SystemExit("Need at least two sessions to run UnitMatch")
    # For test we take the first two sessions
    ks_dirs = [s["path"] for s in sessions[:2]]
    print("Running UnitMatch on:", ks_dirs)

    # Build inputs for UnitMatch
    avg_waveforms_list = []
    channel_positions_list = []
    clus_info_list = []
    session_switch = []
    within_session = []
    n_units_total = 0
    run_session_logs: list[dict[str, Any]] = []

    for si in range(2):
        session_config = sessions[si]
        ks_path = Path(session_config["path"])
        session_label = _infer_session_label(session_config) or f"session_{si}"

        spike_clusters = np.load(ks_path / "spike_clusters.npy").astype(int)

        if use_iti:
            session_pkl = resolve_session_pickle(session_config)
            if session_pkl is None or not Path(session_pkl).exists():
                raise SystemExit(f"No session pickle found for {session_label}. Provide session_pkl in config.")
            session = load_session(str(session_pkl))
            raw_ap = resolve_raw_ap_path(session_config)
            if raw_ap is None or not raw_ap.exists():
                raise SystemExit(f"Raw AP file not found for {session_label}. Set raw_ap in config.")
            good_ids = session.good_cluster_ids if session.good_cluster_ids else None
            print(f"[{si + 1}/2] Extracting ITI waveforms for {session_label} (good clusters only) ...")
            wave, kept_clusters, diag = extract_iti_waveforms_from_raw(
                session,
                ks_path,
                raw_ap,
                cluster_ids=good_ids,
                max_spikes_per_unit=max_spikes_per_unit,
                min_spikes_per_unit=min_spikes_per_unit,
                min_spikes_per_half=min_spikes_per_half,
                rng_seed=si,
                session_label=session_label,
                window_sampling=window_sampling,
                max_windows=max_iti_windows,
                show_progress=show_progress,
            )
            if len(kept_clusters) == 0:
                raise SystemExit(f"No ITI-qualified units found for {session_label}")
            print(f"Finished {session_label}: {len(kept_clusters)} clusters with ITI waveforms")
            print(
                f"Session {session_label}: built ITI waveforms for {len(kept_clusters)} units using {raw_ap}"
            )
            w = wave
            cluster_ids_session = np.array(kept_clusters, dtype=int)
        else:
            try:
                w = load_waveforms_from_source(
                    session_config,
                    si,
                    waveform_source,
                    waveform_dir,
                    use_iti,
                )
            except Exception as e:
                print(f"Failed to load waveforms for session {si}: {e}")
                raise
            cluster_ids_session = np.unique(spike_clusters).astype(int)

        avg_waveforms_list.append(w)
        n_units = w.shape[0]
        n_units_total += n_units
        session_switch.extend([si] * n_units)
        within_session.extend([si] * n_units)

        # load channel positions from KS folder
        ch_pos = np.load(ks_path / "channel_positions.npy")
        # UnitMatch expects channel positions with 3 columns (x,y,z). If only x,y
        # are present, pad a zero z-coordinate so shapes match.
        if ch_pos.ndim == 2 and ch_pos.shape[1] == 2:
            ch_pos = np.concatenate(
                [ch_pos, np.zeros((ch_pos.shape[0], 1), dtype=ch_pos.dtype)], axis=1
            )
        channel_positions_list.append(ch_pos)

        # Build clus_info for this session
        orig_ids = cluster_ids_session
        sess_id = np.ones_like(orig_ids) * si
        clus_info_list.append(
            {
                "original_ids": orig_ids.reshape(-1, 1),
                "session_id": sess_id.reshape(-1, 1),
            }
        )

    # Concatenate waveforms
    waveform_all = np.concatenate(avg_waveforms_list, axis=0)

    # Concatenate clus_info
    original_ids = np.concatenate(
        [ci["original_ids"].squeeze() for ci in clus_info_list]
    )
    session_id = np.concatenate([ci["session_id"].squeeze() for ci in clus_info_list])
    clus_info = {"original_ids": original_ids, "session_id": session_id}

    # Load default params from UnitMatchPy.default_params
    import UnitMatchPy.default_params as dp

    param = dp.get_default_param()
    # Diagnostic: print UnitMatchPy location and key function signatures
    try:
        um_mod = importlib.import_module("UnitMatchPy")
        print("UnitMatchPy module file:", getattr(um_mod, "__file__", "builtin"))
    except Exception as e:
        print("Could not import UnitMatchPy module location:", e)
    try:
        print(
            "ov.extract_metric_scores signature:",
            inspect.signature(ov.extract_metric_scores),
        )
    except Exception:
        print("Could not inspect ov.extract_metric_scores")
    try:
        print(
            "bf.get_parameter_kernels signature:",
            inspect.signature(bf.get_parameter_kernels),
        )
    except Exception:
        print("Could not inspect bf.get_parameter_kernels")
    try:
        print(
            "bf.apply_naive_bayes signature:", inspect.signature(bf.apply_naive_bayes)
        )
    except Exception:
        print("Could not inspect bf.apply_naive_bayes")
    param["n_units"] = n_units_total
    param["n_sessions"] = len(ks_dirs)
    # Use Neuropixels 2.0 4-shank geometry by default for these probes
    # (user provided: 4 shanks, center-to-center spacing ~250 um)
    # Allow opt-in auto shank derivation via config: use_auto_shanks: true
    cfg_all = yaml.safe_load(Path("config/unitmatch_sessions.yml").read_text())
    use_auto = bool(cfg_all.get("use_auto_shanks", False))

    # Default fallbacks
    param["no_shanks"] = 4
    param["shank_dist"] = param.get("shank_dist", 250)

    if use_auto:
        # Try to derive shanks from the first session's channel_positions (assumes same probe)
        try:
            derived = derive_shanks_from_channel_positions(
                channel_positions_list[0], expected_shanks=4
            )
            derived_no_shanks, derived_shank_dist, channel_shanks = derived
            if derived_no_shanks is not None:
                print(
                    f"Auto-derived {derived_no_shanks} shanks; shank_dist={derived_shank_dist}"
                )
                param["no_shanks"] = derived_no_shanks
                if derived_shank_dist is not None:
                    param["shank_dist"] = derived_shank_dist
                # attach channel_shanks for UnitMatch usage if it's expected downstream
                # pad/truncate to n_ch
                try:
                    cs = channel_shanks.astype(int)
                    if cs.shape[0] >= n_ch:
                        cs = cs[:n_ch]
                    else:
                        pad_rows = np.zeros((n_ch - cs.shape[0],), dtype=int)
                        cs = np.concatenate([cs, pad_rows])
                    param["channel_shanks"] = cs
                    # Save a diagnostic plot showing derived shanks
                    try:
                        out_dir = Path(
                            cfg_all.get("report_dir", "table_output/unitmatch")
                        )
                        out_dir.mkdir(parents=True, exist_ok=True)
                        png_path = out_dir / "shank_assignment.png"
                        _save_shank_assignment_plot(
                            channel_positions_list[0], cs, str(png_path)
                        )
                        print("Saved shank assignment plot to", png_path)
                    except Exception as _e:
                        print("Could not save shank assignment plot:", _e)
                except Exception:
                    # best-effort: skip attaching channel_shanks
                    pass
            else:
                print("Auto-shank derivation returned no result; keeping defaults")
        except Exception as e:
            print("Auto-shank derivation failed:", e)
            print("Keeping default shank settings")

    # Run UnitMatch computations
    print("Extracting waveform parameters...")
    channel_pos = channel_positions_list
    # number of channels inferred from waveform arrays
    n_ch = int(waveform_all.shape[2])
    param["n_channels"] = n_ch

    # Ensure spike width and waveidx match the Bombcell templates
    spike_w = int(waveform_all.shape[1])
    param["spike_width"] = spike_w
    # choose a central window for waveidx (middle 50% of samples)
    start = spike_w // 4
    end = spike_w - start
    param["waveidx"] = np.arange(start, end)
    param["peak_loc"] = spike_w // 2

    # Ensure each channel_positions array has shape (n_ch, 3) by truncating or padding
    aligned_channel_positions = []
    for cp in channel_positions_list:
        cp = np.asarray(cp)
        # pad a 3rd column if missing
        if cp.ndim == 2 and cp.shape[1] == 2:
            cp = np.concatenate(
                [cp, np.zeros((cp.shape[0], 1), dtype=cp.dtype)], axis=1
            )
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
    print("Computing metric scores...")
    # extract_metric_scores returns (total_score, candidate_pairs, scores_to_include, predictors)
    total_score, candidate_pairs, scores_to_include, predictors = (
        ov.extract_metric_scores(
            extracted, np.array(session_switch), np.array(within_session), param
        )
    )

    print("Computing parameter kernels and priors for Bayesian combination...")
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
    elif hasattr(res, "ndim"):
        # Some versions return only the parameter_kernels ndarray; compute simple priors from labels
        param_kernels = res
        counts = np.bincount(labels.astype(int))
        # normalize to get priors per cond (align to unique cond ordering)
        # cond is np.unique(labels)
        priors = counts / counts.sum()
    else:
        raise RuntimeError("Unexpected return from get_parameter_kernels: %r" % (res,))

    print("Applying Bayes...")
    # predictors is expected from extract_metric_scores; cond may be None or returned elsewhere
    # Call apply_naive_bayes with explicit cond (UnitMatchPy versions expect this)
    output_prob = bf.apply_naive_bayes(param_kernels, priors, predictors, param, cond)

    print("Making match table (custom writer)...")
    import json

    out_dir = Path(cfg.get("report_dir", "table_output/unitmatch"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostic dump to help debugging shapes
    diag = {
        "waveform_all_shape": getattr(waveform_all, "shape", None),
        "total_score_shape": getattr(total_score, "shape", None),
        "candidate_pairs_shape": getattr(candidate_pairs, "shape", None)
        if "candidate_pairs" in locals()
        else None,
        "scores_to_include_shape": getattr(scores_to_include, "shape", None)
        if "scores_to_include" in locals()
        else None,
        "predictors_shape": getattr(predictors, "shape", None)
        if "predictors" in locals()
        else None,
        "output_prob_shape": getattr(output_prob, "shape", None),
        "param_kernels_shape": getattr(param_kernels, "shape", None)
        if "param_kernels" in locals()
        else None,
        "priors_shape": getattr(priors, "shape", None)
        if "priors" in locals()
        else None,
        "n_units": param.get("n_units"),
    }
    # Additional debug info placeholders
    diag.update(
        {
            "candidate_mask_true_count": None,
            "sess_ids_unique": None,
            "prob_matrix_max": None,
            "prob_matrix_min": None,
        }
    )
    (out_dir / "unitmatch_pair_diagnostic.json").write_text(json.dumps(diag, indent=2))

    import csv

    # Helper arrays
    orig_ids = clus_info["original_ids"]
    sess_ids = clus_info["session_id"]
    session_names = [Path(p).name for p in ks_dirs]

    # Interpret candidate_pairs and output_prob shapes produced by UnitMatchPy
    n = param["n_units"]
    pairs = None
    probs = None

    # Simplified and deterministic extraction: reshape output_prob into (n,n,2) when possible
    prob_matrix = None
    if hasattr(output_prob, "shape"):
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
        raise RuntimeError(
            "Unable to interpret output_prob into an (n,n) match-probability matrix. See diagnostic JSON."
        )

    # record prob stats
    try:
        diag["prob_matrix_max"] = float(np.nanmax(prob_matrix))
        diag["prob_matrix_min"] = float(np.nanmin(prob_matrix))
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
        raise RuntimeError(
            "Unable to interpret candidate_pairs/output_prob shapes for CSV output. See diagnostic JSON."
        )

    # finalize diag fields about session ids
    try:
        diag["sess_ids_unique"] = np.unique(sess_ids).tolist()
    except Exception:
        diag["sess_ids_unique"] = None
    (out_dir / "unitmatch_pair_diagnostic.json").write_text(json.dumps(diag, indent=2))

    # Debug: print number of pairs and first few samples
    try:
        num_pairs = int(pairs.shape[0])
    except Exception:
        num_pairs = 0
    print(f"Preparing to write {num_pairs} pairs to CSV")
    sample_n = min(10, num_pairs)
    if num_pairs > 0:
        print("Sample pairs (first", sample_n, "):")
        for (a, b), p in list(zip(pairs.tolist(), probs.tolist()))[:sample_n]:
            print("  ", int(a), int(b), float(p))

    # Build CSV rows
    csv_path = out_dir / "unitmatch_pair_matches.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "unit_idx_a",
                "unit_idx_b",
                "unit_id_a",
                "unit_id_b",
                "session_a",
                "session_b",
                "prob",
            ]
        )
        for (a, b), p in zip(pairs.tolist(), probs.tolist()):
            a = int(a)
            b = int(b)
            writer.writerow(
                [
                    a,
                    b,
                    int(orig_ids[a]),
                    int(orig_ids[b]),
                    session_names[int(sess_ids[a])],
                    session_names[int(sess_ids[b])],
                    float(p),
                ]
            )

    print("Saved custom match CSV to", csv_path)

    run_timestamp = datetime.utcnow()
    run_logs_dir = out_dir / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": run_timestamp.isoformat() + "Z",
        "config": str(Path(args.config).resolve()),
        "command": " ".join(sys.argv),
        "waveform_source": waveform_source,
        "use_iti": use_iti,
        "waveform_params": {
            "window_sampling": window_sampling,
            "max_iti_windows": max_iti_windows,
            "max_spikes_per_unit": max_spikes_per_unit,
            "min_spikes_per_unit": min_spikes_per_unit,
            "min_spikes_per_half": min_spikes_per_half,
        },
        "cache": {
            "enabled": cache_waveforms,
            "cache_dir": str(cache_dir) if cache_waveforms else None,
        },
        "sessions": run_session_logs,
        "output_csv": str(csv_path),
        "report_dir": str(out_dir),
    }
    log_path = run_logs_dir / f"unitmatch_iti_{run_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(summary, indent=2))
    print("Saved run summary to", log_path)


if __name__ == "__main__":
    main()
