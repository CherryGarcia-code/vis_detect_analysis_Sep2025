"""
Convert Bombcell exports (from notebooks/*/bombcell/) into UnitMatch-style
RawWaveforms: one file per unit named Unit{cluster_id}_RawSpikes.npy saved into
the target Kilosort folder's `RawWaveforms/` directory.

Usage:
  python scripts/unitmatch_convert_bombcell_to_rawwaveforms.py \
      --bombcell-dir notebooks/260325/bombcell \
      --kilosort-dir "X:/.../BG_031_250325_g0_imec0" \
      [--good-only]

This script is intentionally defensive: it tries a few common Bombcell export
file names and CSV/parquet metadata to map unit indices to cluster IDs and
optionally filter GOOD units.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import logging


def find_bombcell_waveforms(bc_dir: Path):
    # Try known filenames
    # Prefer the templates._bc_rawWaveforms.npy file (commonly contains the
    # full waveform array). Some Bombcell runs write a kilosort-format stub
    # that can be a 0-d None; prefer the templates file first.
    candidates = [
        'templates._bc_rawWaveforms.npy',
        '_bc_rawWaveforms_kilosort_format.npy',
        '_bc_rawWaveforms.npy',
        'templates._bc_rawWaveforms_kilosort_format.npy'
    ]
    for c in candidates:
        p = bc_dir / c
        if p.exists():
            logging.info('Found bombcell waveform file: %s', p)
            return p
    return None


def load_unit_ids(bc_dir: Path):
    # Attempt to read kilosort_summary.csv or templates qMetrics to map indices to unit ids
    csv1 = bc_dir / 'kilosort_summary.csv'
    if csv1.exists():
        try:
            df = pd.read_csv(csv1)
            # try common columns
            for col in ['cluster_id', 'unit_id', 'id', 'unit']:
                if col in df.columns:
                    return df[col].astype(int).values
            # fallback to first column
            return df.iloc[:, 0].astype(int).values
        except Exception:
            logging.exception('Failed to parse kilosort_summary.csv')
    # Try qMetrics/parquet if present
    pq = bc_dir / 'templates._bc_qMetrics.parquet'
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
            for col in ['cluster_id', 'unit_id', 'id', 'unit']:
                if col in df.columns:
                    return df[col].astype(int).values
            return df.index.astype(int).values
        except Exception:
            logging.exception('Failed to parse parquet qMetrics')
    # If nothing, return None
    return None


def load_good_mask(bc_dir: Path):
    # Return boolean mask (same length as units) marking GOOD units if possible
    # Look for cluster_bc_unitType.tsv or qMetrics files with labels
    tsv = bc_dir / 'cluster_bc_unitType.tsv'
    if tsv.exists():
        try:
            df = pd.read_csv(tsv, sep='\t', header=None)
            # Common format: [unit_id, label]
            if df.shape[1] >= 2:
                labels = df.iloc[:, 1].astype(str).str.upper()
                unit_ids = df.iloc[:, 0].astype(int).values
                # GOOD or NON-SOMA GOOD
                good_ids = unit_ids[labels.isin(['GOOD', 'NON-SOMA GOOD', 'GOOD_UNIT', 'GOODUNIT'])]
                return set(good_ids)
        except Exception:
            logging.exception('Failed to read cluster_bc_unitType.tsv')
    # Try qMetrics csv with 'unit_type' or similar
    qcsv = bc_dir / 'templates._bc_qMetrics.csv'
    if qcsv.exists():
        try:
            df = pd.read_csv(qcsv)
            for col in ['unit_type', 'unitLabel', 'unit_label', 'type']:
                if col in df.columns:
                    good = df[df[col].astype(str).str.contains('GOOD', case=False, na=False)]
                    if 'cluster_id' in df.columns:
                        return set(good['cluster_id'].astype(int).values)
                    else:
                        return set(good.index.astype(int).values)
        except Exception:
            logging.exception('Failed to parse templates._bc_qMetrics.csv')
    # No good list found
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bombcell-dir', type=Path, required=True, help='Path to Bombcell export folder')
    p.add_argument('--kilosort-dir', type=Path, required=True, help='Target Kilosort folder to write RawWaveforms into')
    p.add_argument('--good-only', action='store_true', help='Write only GOOD units if Bombcell provides labels')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    bc_dir = args.bombcell_dir
    ks_dir = args.kilosort_dir
    if not bc_dir.exists():
        logging.error('Bombcell folder does not exist: %s', bc_dir)
        return
    if not ks_dir.exists():
        logging.error('Kilosort folder does not exist: %s', ks_dir)
        return

    wf_file = find_bombcell_waveforms(bc_dir)
    if wf_file is None:
        logging.error('No recognized Bombcell waveform file found in %s', bc_dir)
        return

    logging.info('Loading waveform array from %s', wf_file)
    # The array produced by Bombcell may be object-dtype and require pickle
    # loading. These files were produced locally by the Bombcell run in this
    # workspace, so allow_pickle=True is acceptable here.
    wave = np.load(wf_file, allow_pickle=True)
    logging.info('Waveform array shape: %s', wave.shape)

    # Determine expected shape and units axis
    # Preferred shape: (n_units, spike_width, n_channels, 2)
    if wave.ndim == 4 and wave.shape[0] > 1:
        n_units = wave.shape[0]
        # proceed as-is
    elif wave.ndim == 3:
        # Several shapes are possible. Common ones:
        #  - (spike_width, n_units, n_channels)
        #  - (n_units, spike_width, n_channels)
        # Try to detect which axis corresponds to units using available unit_ids
        # if possible; otherwise use heuristics.
        # First attempt: if unit_ids is known and matches one axis length, trust that.
        if 'unit_ids' in locals() and unit_ids is not None:
            maybe_n = len(unit_ids)
            if wave.shape[0] == maybe_n:
                # already (n_units, spike_w, n_ch)
                n_units = wave.shape[0]
                wave = wave[..., np.newaxis]
                # duplicate last axis to make 2 CVs if missing
                wave = np.repeat(wave, 2, axis=-1)
            elif wave.shape[1] == maybe_n:
                # shape is (spike_w, n_units, n_ch)
                spike_w, n_units, n_ch = wave.shape
                wave = wave.transpose(1, 0, 2)[..., np.newaxis]
                wave = np.repeat(wave, 2, axis=-1)
            elif wave.shape[2] == maybe_n:
                # unlikely, but handle for completeness
                n_units = wave.shape[2]
                wave = wave.transpose(2, 0, 1)[..., np.newaxis]
                wave = np.repeat(wave, 2, axis=-1)
            else:
                # fallback to heuristics below
                pass

        if wave.ndim == 3:
            # fallback heuristics when unit_ids not available or didn't match
            # prefer to interpret shape as (spike_width, n_units, n_channels)
            if wave.shape[1] > wave.shape[0]:
                spike_w, n_units, n_ch = wave.shape
                wave = wave.transpose(1, 0, 2)[..., np.newaxis]
                wave = np.repeat(wave, 2, axis=-1)
            elif wave.shape[0] > 1 and wave.shape[0] > wave.shape[1]:
                # interpret as (n_units, spike_w, n_ch)
                n_units = wave.shape[0]
                wave = wave[..., np.newaxis]
                wave = np.repeat(wave, 2, axis=-1)
            else:
                logging.error('Unrecognized waveform array shape: %s', wave.shape)
                return
    else:
        logging.error('Unrecognized waveform array shape: %s', wave.shape)
        return

    # Load mapping of unit indices to cluster IDs if available
    unit_ids = load_unit_ids(bc_dir)
    if unit_ids is None:
        logging.warning('Could not find unit id mapping; will use indices 0..n-1 as IDs')
        unit_ids = np.arange(n_units)
    else:
        if len(unit_ids) != n_units:
            logging.warning('Unit id mapping length (%d) does not match waveform n_units (%d). Using indices.', len(unit_ids), n_units)
            unit_ids = np.arange(n_units)

    good_set = None
    if args.good_only:
        good_set = load_good_mask(bc_dir)
        if good_set is None:
            logging.warning('Requested GOOD-only but no GOOD list found in bombcell outputs; writing all units instead')
            good_set = None

    out_dir = ks_dir / 'RawWaveforms'
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx in range(n_units):
        cid = int(unit_ids[idx])
        if good_set is not None and cid not in good_set:
            continue
        unit_wave = wave[idx]  # shape (spike_width, n_channels, 2)
        # Ensure float32
        unit_wave = unit_wave.astype(np.float32)
        out_name = out_dir / f'Unit{cid}_RawSpikes.npy'
        np.save(out_name, unit_wave)
        written += 1

    logging.info('Wrote %d unit waveform files to %s', written, out_dir)


if __name__ == '__main__':
    main()
