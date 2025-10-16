"""Wrapper to integrate Bombcell (pyBombCell) QC into this project.

This module provides a thin adapter that:
- optionally attaches Kilosort waveforms to a Session using `src.kilosort_adapter`;
- prepares a minimal input for Bombcell if it is installed;
- returns a tidy DataFrame of QC metrics (or a placeholder DataFrame if Bombcell
  is not installed).

Bombcell uses GPL-3.0 licensing; this wrapper will not ship or copy Bombcell
code — it only calls it if the package is installed in the environment.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import logging
import pandas as pd

from src.kilosort_adapter import attach_kilosort_waveforms


def detect_with_bombcell(session: Any, kilosort_dir: Optional[str] = None, out_dir: Optional[str] = None, save_figures: bool = True, raw_data_file: Optional[str] = None, meta_file: Optional[str] = None, reextract_raw: bool = False, n_raw_spikes: Optional[int] = None) -> pd.DataFrame:
    """Run Bombcell QC on a session (if Bombcell is installed).

    Args:
        session: Session object from `src.session_io.load_session`.
        kilosort_dir: optional path to a Kilosort folder to attach waveforms.
        out_dir: optional directory to save Bombcell outputs (figures/CSV).
        save_figures: whether to save Bombcell-generated figures (if any).

    Returns:
        DataFrame indexed by cluster_id with QC metrics. If Bombcell is not
        available, returns a DataFrame produced from the kilosort_adapter
        summary (if available) or an empty DataFrame.
    """
    outp = Path(out_dir) if out_dir is not None else None
    if outp is not None:
        outp.mkdir(parents=True, exist_ok=True)

    # If kilosort_dir provided, attach mean_waveform to clusters and get a summary
    ks_df = None
    if kilosort_dir is not None:
        try:
            ks_df = attach_kilosort_waveforms(session, kilosort_dir)
            logging.info('Attached kilosort waveforms; summary rows=%d', len(ks_df))
            if outp is not None:
                ks_df.to_csv(str(outp / 'kilosort_summary.csv'))
        except Exception:
            logging.exception('Failed to attach kilosort waveforms from %s', kilosort_dir)

    # Try to import bombcell (pyBombCell). If not installed, return ks_df or empty DataFrame
    try:
        import bombcell as bc
        bc_present = True
    except Exception:
        try:
            import pyBombCell as bc
            bc_present = True
        except Exception:
            bc_present = False
            bc = None

    if not bc_present:
        logging.warning('Bombcell package not available in this environment. Returning kilosort summary (if any).')
        if ks_df is not None:
            return ks_df
        return pd.DataFrame()

    # At this point, bombcell is importable. The Python API surface may vary
    # between versions; we make a conservative, documented attempt to call a
    # likely entry point. If unavailable, we fall back to returning ks_df.
    try:
        # Common pattern: a high-level function that runs QC and returns metrics.
        if hasattr(bc, 'run_bombcell'):
            # pyBombCell provides run_bombcell(ks_dir, save_path, param, ...)
            # We'll prefer calling it with the Kilosort directory (if provided)
            # and use outp as the save_path. If kilosort_dir is not supplied,
            # attempt to discover a ks_dir from the session (best-effort) and
            # fall back to returning ks_df if not available.
            ks_dir_to_use = None
            if kilosort_dir is not None:
                ks_dir_to_use = kilosort_dir
            else:
                # best-effort: check session for an attribute pointing to ks dir
                ks_dir_to_use = getattr(session, 'kilosort_dir', None)

            if ks_dir_to_use is None:
                logging.warning('run_bombcell requires a Kilosort directory; none provided. Skipping Bombcell run.')
                metrics = None
            else:
                # Prefer to obtain a canonical param dict from bombcell itself
                try:
                    if hasattr(bc, 'default_parameters') and hasattr(bc.default_parameters, 'get_default_parameters'):
                        # get_default_parameters requires the kilosort path
                        param = bc.default_parameters.get_default_parameters(ks_dir_to_use)
                    else:
                        param = {}
                except Exception:
                    param = {}

                # If user supplied a raw_data_file, set it so Bombcell will extract raw snippets
                if raw_data_file is not None:
                    param['raw_data_file'] = raw_data_file
                    # enable extraction of raw snippets
                    param['extractRaw'] = True
                    param['reextractRaw'] = param.get('reextractRaw', False)
                if meta_file is not None:
                    # allow Bombcell to find and parse the meta explicitly
                    param['ephys_meta_file'] = meta_file
                else:
                    # Ensure keys exist and apply overrides: disable raw waveform extraction
                    param['raw_data_file'] = param.get('raw_data_file', None)
                    param['extractRaw'] = False
                    param['reextractRaw'] = False
                param['removeDuplicateSpikes'] = param.get('removeDuplicateSpikes', False)
                param['savePlots'] = bool(save_figures) or param.get('savePlots', False)
                if save_figures and param.get('plotsSaveDir') is None and outp is not None:
                    param['plotsSaveDir'] = str(Path(outp) / 'bombcell_plots')

                # if user requested a re-extraction, remove previous intermediates
                if reextract_raw and outp is not None:
                    try:
                        for fn in ['templates._bc_rawWaveforms.npy', '_bc_rawWaveforms_kilosort_format.npy']:
                            pth = outp / fn
                            if pth.exists():
                                pth.unlink()
                    except Exception:
                        logging.exception('Failed to remove previous raw-waveform intermediates')

                # call run_bombcell with the correct signature. Bombcell may save
                # object-dtype npy files and later load them without specifying
                # allow_pickle=True; numpy now blocks pickles by default. To
                # be robust we temporarily monkeypatch np.load to allow pickle
                # while Bombcell runs, then restore it.
                out_save = str(outp) if outp is not None else None
                np_load_orig = np.load
                def _np_load_allow_pickle(*args, **kwargs):
                    if 'allow_pickle' not in kwargs:
                        kwargs['allow_pickle'] = True
                    return np_load_orig(*args, **kwargs)
                np.load = _np_load_allow_pickle
                # Apply a safe in-memory monkeypatch to bombcell.extract_raw_waveforms
                # to fix a known indexing bug where code compares unique_clusters == i
                # (loop index) instead of unique_clusters == cid (cluster id). This
                # causes empty selections and zero-size arrays when cluster ids are
                # non-consecutive or when a masked subset of clusters is passed.
                try:
                    import importlib, inspect
                    er_mod = importlib.import_module('bombcell.extract_raw_waveforms')
                    src = inspect.getsource(er_mod)
                    if 'mask = unique_clusters == i' in src:
                        fixed_src = src.replace('mask = unique_clusters == i', 'mask = unique_clusters == cid')
                        # compile in a temporary namespace and replace function
                        ns = {}
                        exec(fixed_src, ns)
                        if 'extract_raw_waveforms' in ns:
                            patched_fn = ns['extract_raw_waveforms']
                            setattr(er_mod, 'extract_raw_waveforms', patched_fn)
                            # Also replace any reference in helper modules that may have imported
                            # the function earlier (run_bombcell often calls via helper_functions)
                            try:
                                hf = importlib.import_module('bombcell.helper_functions')
                                if hasattr(hf, 'extract_raw_waveforms'):
                                    setattr(hf, 'extract_raw_waveforms', patched_fn)
                            except Exception:
                                # not fatal; helper module may not exist or have been loaded
                                pass
                            logging.info('Applied runtime monkeypatch to bombcell.extract_raw_waveforms to avoid empty-index bug')
                except Exception:
                    logging.exception('Failed to apply runtime monkeypatch to bombcell.extract_raw_waveforms; continuing without patch')
                try:
                    try:
                        # if user requested a custom number of raw spikes, set it
                        if n_raw_spikes is not None:
                            param['nRawSpikesToExtract'] = int(n_raw_spikes)
                        # run_bombcell returns (quality_metrics, param, unit_type, unit_type_string, figures_optional)
                        metrics = bc.run_bombcell(ks_dir_to_use, out_save, param, save_figures=save_figures, return_figures=False)
                    except TypeError:
                        # fallback to simpler call signature
                        if n_raw_spikes is not None:
                            param['nRawSpikesToExtract'] = int(n_raw_spikes)
                        metrics = bc.run_bombcell(ks_dir_to_use, out_save, param)
                finally:
                    # restore numpy.load
                    np.load = np_load_orig
        elif hasattr(bc, 'run'):
            metrics = bc.run(session)
        elif hasattr(bc, 'compute_metrics'):
            metrics = bc.compute_metrics(session)
        elif hasattr(bc, 'analyze_session'):
            metrics = bc.analyze_session(session)
        else:
            # try lower-level unit-match runner if available
            if hasattr(bc, 'run_bombcell_unit_match'):
                try:
                    metrics = bc.run_bombcell_unit_match(session, outdir=str(outp) if outp is not None else None)
                except TypeError:
                    metrics = bc.run_bombcell_unit_match(session)
            else:
                logging.warning('Bombcell installed but no recognized API entrypoint found. Returning kilosort summary.')
                return ks_df if ks_df is not None else pd.DataFrame()

        # Bombcell may return a tuple: (quality_metrics_dict, param, unit_type, unit_type_string[, figures])
        if isinstance(metrics, tuple) or isinstance(metrics, list):
            if len(metrics) >= 1 and isinstance(metrics[0], dict):
                qmetrics = metrics[0]
                df = pd.DataFrame.from_dict(qmetrics, orient='index')
            else:
                # fallback: try to create DataFrame from the first element
                df = pd.DataFrame(metrics[0]) if len(metrics) > 0 else pd.DataFrame()
        else:
            # If the Bombcell function returned a dict or DataFrame-like, normalize
            if isinstance(metrics, pd.DataFrame):
                df = metrics
            elif isinstance(metrics, dict):
                df = pd.DataFrame.from_dict(metrics, orient='index')
            else:
                # try to convert
                df = pd.DataFrame(metrics)

        # Optionally save
        if outp is not None:
            try:
                df.to_csv(str(outp / 'bombcell_metrics.csv'))
            except Exception:
                logging.exception('Failed to save bombcell_metrics.csv')

        return df
    except Exception:
        logging.exception('Error while running Bombcell API; returning kilosort summary (if any)')
        return ks_df if ks_df is not None else pd.DataFrame()


__all__ = ['detect_with_bombcell']
