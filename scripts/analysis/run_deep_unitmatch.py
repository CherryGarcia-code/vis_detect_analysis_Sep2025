"""
DeepUnitMatch Pipeline for BG_046
==================================
Runs the DeepUnitMatch pre-trained model on existing raw waveforms to produce
a Global CellRegistry compatible with the Grand Longitudinal Table pipeline.

Architecture
------------
1. PREPROCESS  – convert raw (82, 383, 2) waveforms → (60, 30, 2) snippets
2. ENCODE      – embed each CV-half through SpatioTemporalCNN_V2 → 256-d vectors
3. MATCH       – CLIP cosine similarity + KDE threshold + spatial + conflict filter
4. REGISTRY    – assemble a Global CellRegistry CSV

Usage
-----
    conda activate unitmatch_env
    python scripts/analysis/run_deep_unitmatch.py

    # Use a different distance threshold (default 150 µm)
    python scripts/analysis/run_deep_unitmatch.py --dist-thresh 100

    # Use MAP-based threshold
    python scripts/analysis/run_deep_unitmatch.py --map-threshold

Output
------
    data/deep_unit_match/output/BG_046/
        DeepUM_CellRegistry.csv          – drop-in replacement for Global_CellRegistry_Stitched.csv
        similarity_matrix.npy            – full N×N cosine similarity matrix
        embeddings.npz                   – per-session embeddings for later analysis
        unit_index.csv                   – mapping from matrix row → (session, ks_id)
        DeepUM_MatchTable.csv            – all pairwise matches with probabilities

Requires
--------
    PyTorch (CPU), h5py, scikit-learn, numpy, pandas, scipy, tqdm
    DeepUnitMatch repo cloned at: _DeepUnitMatch_repo/
"""

from __future__ import annotations
import os

# Fix OpenMP conflict between numpy/torch on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.signal import detrend
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DUM_CODE  = REPO_ROOT / "_DeepUnitMatch_repo" / "UnitMatchPy" / "DeepUnitMatch"
sys.path.insert(0, str(DUM_CODE))

from utils.mymodel import SpatioTemporalCNN_V2
from utils.losses import clip_sim, clip_prob

INPUT_ROOT  = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
OUTPUT_ROOT = REPO_ROOT / "data" / "deep_unit_match" / "output" / "BG_046"
MODEL_PATH  = DUM_CODE / "utils" / "model"


def _parse_date(s) -> pd.Timestamp:
    """Parse a DDMMYYYY directory name, handling missing leading zero on day."""
    s = str(s)
    if len(s) == 7:
        s = '0' + s
    return pd.to_datetime(s, format='%d%m%Y')

# ---------------------------------------------------------------------------
# 1. PREPROCESSING  –  (82, N_ch, 2) → (60, 30, 2) snippets
# ---------------------------------------------------------------------------
# Adapted from DeepUnitMatch's param_fun.py with a fix for the odd-channel
# interleaving bug in sort_good_channels.

DEFAULT_PARAM = {
    'nTime': 82,
    'RnChannels': 30,
    'RnTime': 60,
    'ChannelRadius': 110,  # µm — radius around max-site for channel selection
}


def _extract_snippet(waveform: np.ndarray,
                     channel_pos: np.ndarray,
                     param: dict = DEFAULT_PARAM) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Convert a single unit's raw waveform to a (60, 30, 2) snippet.

    Parameters
    ----------
    waveform : (nTime, nChannels, 2)  – raw mean waveform per cross-validation half
    channel_pos : (nChannels, 2)      – (x, y) for each channel
    param : dict                      – preprocessing parameters

    Returns
    -------
    (snippet, max_site_pos) or None if the waveform is malformed.
    """
    nTime = param['nTime']
    RnTime = param['RnTime']
    RnCh = param['RnChannels']
    radius = param['ChannelRadius']

    # ---- temporal crop to middle 60 points ----
    start = (nTime - RnTime) // 2
    end = (nTime + RnTime) // 2
    wf = waveform[start:end, :, :]                       # (60, nCh, 2)

    # ---- linear detrend per channel ----
    wf = detrend(wf, axis=0)

    # ---- find max site (average over CV halves) ----
    mean_cv = np.mean(wf, axis=2)                        # (60, nCh)
    spatial_fp = np.max(np.abs(mean_cv), axis=0)          # (nCh,)
    max_site = int(np.argmax(spatial_fp))
    max_pos = channel_pos[max_site].copy()                # (2,)

    # ---- select channels within radius ----
    dists = np.linalg.norm(channel_pos - channel_pos[max_site], axis=1)
    good_mask = dists < radius
    good_indices = np.where(good_mask)[0]
    n_good = len(good_indices)

    if n_good == 0:
        return None

    # ---- sort good channels spatially (interleave two columns) ----
    good_pos = channel_pos[good_indices]
    unique_x = np.unique(good_pos[:, 0])

    if len(unique_x) == 2:
        # NP2.0 two-column layout — interleave columns sorted by depth
        col0 = good_indices[good_pos[:, 0] == unique_x[0]]
        col1 = good_indices[good_pos[:, 0] == unique_x[1]]
        col0 = col0[np.argsort(channel_pos[col0, 1])]
        col1 = col1[np.argsort(channel_pos[col1, 1])]

        # Interleave (handle unequal column lengths)
        sorted_ch = []
        for a, b in zip(col0, col1):
            sorted_ch.extend([a, b])
        # Append remaining from the longer column
        longer = col0 if len(col0) > len(col1) else col1
        for idx in range(min(len(col0), len(col1)), len(longer)):
            sorted_ch.append(longer[idx])
        sorted_ch = np.array(sorted_ch, dtype=np.int32)
    else:
        # Single column or unusual layout — just sort by depth
        sorted_ch = good_indices[np.argsort(channel_pos[good_indices, 1])]

    # ---- select and pad to RnChannels ----
    Rwf = wf[:, sorted_ch, :]                            # (60, n_good, 2)
    global_mean = np.mean(Rwf)
    Rwf = Rwf - global_mean                               # zero-mean
    pad_val = np.mean(Rwf)

    if n_good < RnCh:
        # Decide padding direction based on max site relative to good channels
        good_y = channel_pos[sorted_ch, 1]
        mid_y = np.mean(good_y)
        needed = RnCh - n_good
        if max_pos[1] < mid_y:
            pad_before, pad_after = needed, 0
        else:
            pad_before, pad_after = 0, needed
        Rwf = np.pad(Rwf, ((0, 0), (pad_before, pad_after), (0, 0)),
                      mode='constant', constant_values=pad_val)
    elif n_good > RnCh:
        # Take the RnCh channels closest to max site
        ch_dists = np.linalg.norm(channel_pos[sorted_ch] - channel_pos[max_site], axis=1)
        keep = np.argsort(ch_dists)[:RnCh]
        keep.sort()  # preserve spatial order
        Rwf = Rwf[:, keep, :]

    assert Rwf.shape == (RnTime, RnCh, 2), f"Unexpected shape {Rwf.shape}"
    return Rwf, max_pos


# ---------------------------------------------------------------------------
# 2. SESSION LOADING  –  discover sessions, preprocess all units
# ---------------------------------------------------------------------------

def load_session_waveforms(session_dir: Path, param: dict = DEFAULT_PARAM):
    """
    Load and preprocess all units from a session directory.

    Returns
    -------
    snippets   : list of (60, 30, 2) arrays
    positions  : list of (2,) arrays  (max-site x, y)
    ks_ids     : list of int  (original Kilosort cluster IDs)
    """
    cp = np.load(session_dir / "channel_positions.npy")
    wf_dir = session_dir / "RawWaveforms"
    if not wf_dir.exists():
        return [], [], []

    files = sorted(f for f in os.listdir(wf_dir) if f.endswith('_RawSpikes.npy'))
    snippets, positions, ks_ids = [], [], []

    for fname in files:
        ks_id = int(fname.replace('Unit', '').replace('_RawSpikes.npy', ''))
        wf = np.load(wf_dir / fname)

        # Validate shape
        if wf.ndim != 3 or wf.shape[0] != param['nTime'] or wf.shape[2] != 2:
            continue

        result = _extract_snippet(wf, cp, param)
        if result is None:
            continue

        snippet, max_pos = result
        snippets.append(snippet)
        positions.append(max_pos)
        ks_ids.append(ks_id)

    return snippets, positions, ks_ids


def discover_sessions(input_root: Path):
    """Return chronologically sorted list of (session_date_str, session_path) tuples."""
    sessions = []
    for name in os.listdir(input_root):
        d = input_root / name
        if d.is_dir() and (d / "RawWaveforms").is_dir():
            sessions.append((name, d))

    # Sort chronologically by parsing DDMMYYYY → datetime
    sessions.sort(key=lambda x: _parse_date(x[0]))
    return sessions


# ---------------------------------------------------------------------------
# 3. MODEL  –  load checkpoint, encode batches
# ---------------------------------------------------------------------------

def load_model(device: str = 'cpu') -> SpatioTemporalCNN_V2:
    model = SpatioTemporalCNN_V2(n_channel=30, n_time=60, n_output=256).to(device)
    model = model.double()
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def encode_session(model, snippets: list[np.ndarray], cv_half: int,
                   device: str = 'cpu', batch_size: int = 128) -> np.ndarray:
    """
    Encode one CV half of a session's snippets.

    Parameters
    ----------
    cv_half : 0 = first half, 1 = second half

    Returns
    -------
    embeddings : (N, 256) float64
    """
    if len(snippets) == 0:
        return np.empty((0, 256), dtype=np.float64)

    wfs = np.array([s[:, :, cv_half] for s in snippets])  # (N, 60, 30)

    # Min-max normalize per unit (matching NeuropixelsDataset._normalize_waveform)
    mn = wfs.min(axis=(1, 2), keepdims=True)
    mx = wfs.max(axis=(1, 2), keepdims=True)
    wfs = (wfs - mn) / (mx - mn + 1e-8)

    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(wfs), batch_size):
            batch = torch.tensor(wfs[i:i+batch_size], dtype=torch.float64, device=device)
            emb = model(batch)
            embeddings_list.append(emb.cpu().numpy())

    return np.vstack(embeddings_list)


def load_temp_tau(device: str = 'cpu') -> float:
    """Load the learned temperature parameter from the CLIP loss in the checkpoint."""
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    tau = checkpoint['clip_loss']['temp_tau'].item()
    return tau


# ---------------------------------------------------------------------------
# 4. MATCHING  –  per-pair softmax probability + rank-based matching
# ---------------------------------------------------------------------------
# Key insight: raw CLIP cosine similarities are saturated near 1.0 for all
# pairs (~0.995), making global thresholding useless. Instead, we use
# temperature-scaled softmax (clip_prob) on CONSECUTIVE session pairs.
# With temp_tau ≈ 0.026, even tiny cosine differences become decisive.
# ---------------------------------------------------------------------------

def match_consecutive_pair(
    emb_fh_a: np.ndarray, emb_sh_a: np.ndarray,
    emb_fh_b: np.ndarray, emb_sh_b: np.ndarray,
    pos_a: np.ndarray, pos_b: np.ndarray,
    ks_ids_a: np.ndarray, ks_ids_b: np.ndarray,
    temp_tau: float,
    dist_thresh: float = 150.0,
    min_prob: float = 0.01,
) -> pd.DataFrame:
    """
    Match units between two consecutive sessions using rank-based matching
    with softmax probability scoring.

    Steps
    -----
    1. Compute per-pair softmax probability matrix (temperature-scaled)
    2. Rank-based best-match assignment (argmax per row and per column)
    3. Bidirectional filter (keep only mutual best matches)
    4. Spatial filter with drift correction
    5. Probability floor filter

    Parameters
    ----------
    emb_fh_a, emb_sh_a : (nA, 256) first/second half embeddings for session A
    emb_fh_b, emb_sh_b : (nB, 256) first/second half embeddings for session B
    pos_a, pos_b        : (nA, 2), (nB, 2) — (x, y) positions per unit
    ks_ids_a, ks_ids_b  : (nA,), (nB,) — Kilosort cluster IDs
    temp_tau            : learned temperature from CLIP loss
    dist_thresh         : max drift-corrected spatial distance (µm)
    min_prob            : minimum probability to accept a match

    Returns
    -------
    DataFrame with columns [ID1, ID2, Prob, prob_AB, prob_BA, dist]
    """
    nA, nB = len(emb_fh_a), len(emb_fh_b)
    if nA == 0 or nB == 0:
        return pd.DataFrame()

    # ---- Softmax probability matrices ----
    # A→B: P(B_j | A_i) — how likely is B_j to be the match for A_i?
    # B→A: P(A_i | B_j) — how likely is A_i to be the match for B_j?
    with torch.no_grad():
        t_a_fh = torch.tensor(emb_fh_a, dtype=torch.float64)
        t_b_sh = torch.tensor(emb_sh_b, dtype=torch.float64)
        prob_AB = clip_prob(t_a_fh, t_b_sh, temp_tau=temp_tau).numpy()  # (nA, nB)

        t_b_fh = torch.tensor(emb_fh_b, dtype=torch.float64)
        t_a_sh = torch.tensor(emb_sh_a, dtype=torch.float64)
        prob_BA = clip_prob(t_b_fh, t_a_sh, temp_tau=temp_tau).numpy()  # (nB, nA)

    # ---- Rank-based best matches ----
    best_B_for_A = np.argmax(prob_AB, axis=1)               # (nA,)
    prob_A_to_B = prob_AB[np.arange(nA), best_B_for_A]      # (nA,)

    best_A_for_B = np.argmax(prob_BA, axis=1)               # (nB,)
    prob_B_to_A = prob_BA[np.arange(nB), best_A_for_B]      # (nB,)

    # ---- Bidirectional filter: keep only mutual best matches ----
    matches = []
    for i_a in range(nA):
        i_b = best_B_for_A[i_a]
        if best_A_for_B[i_b] == i_a:
            # Mutual best match — geometric mean of probabilities
            p = np.sqrt(float(prob_A_to_B[i_a]) * float(prob_B_to_A[i_b]))
            matches.append({
                'idx_a': i_a, 'idx_b': i_b,
                'ID1': int(ks_ids_a[i_a]), 'ID2': int(ks_ids_b[i_b]),
                'Prob': p,
                'prob_AB': float(prob_A_to_B[i_a]),
                'prob_BA': float(prob_B_to_A[i_b]),
                'x1': pos_a[i_a, 0], 'y1': pos_a[i_a, 1],
                'x2': pos_b[i_b, 0], 'y2': pos_b[i_b, 1],
            })

    if len(matches) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(matches)
    n_bidirectional = len(df)

    # ---- Adaptive probability floor ----
    # Fixed min_prob scales poorly: softmax probs ~ 1/N, so larger sessions
    # get penalized.  Use an adaptive floor: match must be at least half as
    # likely as a uniform guess (0.5/max(nA,nB)).  This keeps the same
    # selectivity regardless of session size.
    adaptive_floor = 0.5 / max(nA, nB)
    effective_min = max(min_prob, adaptive_floor) if min_prob > 0 else adaptive_floor
    df = df[df['Prob'] >= effective_min].copy()

    if len(df) == 0:
        return df

    # ---- Spatial filter with per-shank drift correction ----
    df['shank1'] = df['x1'].round(-2).astype(int)
    df['shank2'] = df['x2'].round(-2).astype(int)

    same_shank = df[df['shank1'] == df['shank2']].copy()
    if len(same_shank) > 0:
        same_shank['ydiff'] = same_shank['y2'] - same_shank['y1']
        corrections = (same_shank
                       .groupby('shank1')['ydiff']
                       .median()
                       .reset_index()
                       .rename(columns={'shank1': 'shank'}))
        df = df.merge(corrections[['shank', 'ydiff']], how='left',
                      left_on='shank1', right_on='shank')
        df['y2_corr'] = np.where(
            df['shank1'] != df['shank2'],
            1000,  # penalize cross-shank
            df['y2'] - df['ydiff'].fillna(0)
        )
        df.drop(columns=['shank'], inplace=True, errors='ignore')
    else:
        df['y2_corr'] = df['y2']
        df['ydiff'] = 0.0

    df['dist'] = np.sqrt((df['x1'] - df['x2'])**2 +
                          (df['y1'] - df['y2_corr'])**2)

    df = df[df['dist'] < dist_thresh].copy()

    return df


# ---------------------------------------------------------------------------
# 5. REGISTRY  –  per-pair CellRegistry + stitching via unit_tracking.py
# ---------------------------------------------------------------------------

def build_pair_registry(
    matches: pd.DataFrame,
    ks_ids_a: np.ndarray, ks_ids_b: np.ndarray,
    date_a: str, date_b: str,
) -> pd.DataFrame:
    """
    Build a 2-column CellRegistry CSV for one session pair.

    Format: index=UID, columns=[date_a, date_b].
    Matched units share a row; unmatched units get NaN in the other column.
    """
    matched_a = set()
    matched_b = set()
    rows = []
    uid = 0

    if len(matches) > 0:
        for _, m in matches.iterrows():
            rows.append({'UID': uid, date_a: int(m['ID1']), date_b: int(m['ID2'])})
            matched_a.add(int(m['ID1']))
            matched_b.add(int(m['ID2']))
            uid += 1

    # Unmatched units from A
    for ks_id in ks_ids_a:
        if int(ks_id) not in matched_a:
            rows.append({'UID': uid, date_a: int(ks_id), date_b: np.nan})
            uid += 1

    # Unmatched units from B
    for ks_id in ks_ids_b:
        if int(ks_id) not in matched_b:
            rows.append({'UID': uid, date_a: np.nan, date_b: int(ks_id)})
            uid += 1

    df = pd.DataFrame(rows).set_index('UID')
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepUnitMatch pipeline for BG_046")
    parser.add_argument("--dist-thresh", type=float, default=150.0,
                        help="Max spatial distance (µm) for matches (default: 150)")
    parser.add_argument("--min-prob", type=float, default=0.0,
                        help="Additional probability floor (default: 0.0; adaptive floor always applied)")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for model inference")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved embeddings.npz and unit_index.csv")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- Load temperature parameter from checkpoint ----
    temp_tau = load_temp_tau(args.device)
    print(f"Learned temperature (temp_tau): {temp_tau:.6f}")

    # ---- Per-session data stores ----
    session_emb_fh = {}     # date → (n, 256)
    session_emb_sh = {}     # date → (n, 256)
    session_positions = {}  # date → (n, 2)
    session_ks_ids = {}     # date → (n,)
    session_dates = []      # chronologically ordered

    # ---- Check for resume mode ----
    if args.resume:
        emb_path = OUTPUT_ROOT / "embeddings.npz"
        idx_path = OUTPUT_ROOT / "unit_index.csv"
        if emb_path.exists() and idx_path.exists():
            print("Resuming from saved embeddings...")
            data = np.load(emb_path)
            emb_fh_all = data['emb_fh']
            emb_sh_all = data['emb_sh']
            sid_all = data['session_ids']
            ks_all = data['ks_ids']

            unit_index = pd.read_csv(idx_path)
            session_dates_unsorted = list(dict.fromkeys(unit_index['session']))

            # Sort chronologically (directory names are DDMMYYYY)
            session_dates = sorted(session_dates_unsorted,
                                   key=lambda s: _parse_date(s))

            # Reconstruct per-session stores
            for sd in session_dates:
                mask = unit_index['session'].values == sd
                session_emb_fh[sd] = emb_fh_all[mask]
                session_emb_sh[sd] = emb_sh_all[mask]
                sub = unit_index[mask]
                session_positions[sd] = sub[['pos_x', 'pos_y']].values
                session_ks_ids[sd] = sub['ks_id'].values

            N = len(unit_index)
            print(f"  Loaded {N} units from {len(session_dates)} sessions")
        else:
            print("WARNING: --resume specified but saved data not found. Running from scratch.")
            args.resume = False

    if not args.resume:
        # ---- Discover sessions ----
        sessions = discover_sessions(INPUT_ROOT)
        print(f"Found {len(sessions)} sessions under {INPUT_ROOT}")

        # ---- Load model ----
        print("Loading pre-trained DeepUnitMatch model...")
        model = load_model(args.device)
        print("  Model loaded successfully.")

        # ---- Preprocess & encode all sessions ----
        all_session_ids = []
        all_ks_ids = []
        all_positions = []

        for sess_date, sess_path in tqdm(sessions, desc="Processing sessions"):
            snippets, positions, ks_ids = load_session_waveforms(sess_path)
            if len(snippets) == 0:
                print(f"  WARNING: No valid units in {sess_date}, skipping.")
                continue

            session_dates.append(sess_date)

            emb_fh = encode_session(model, snippets, cv_half=0,
                                    device=args.device, batch_size=args.batch_size)
            emb_sh = encode_session(model, snippets, cv_half=1,
                                    device=args.device, batch_size=args.batch_size)

            pos_arr = np.array(positions)
            ks_arr = np.array(ks_ids)

            session_emb_fh[sess_date] = emb_fh
            session_emb_sh[sess_date] = emb_sh
            session_positions[sess_date] = pos_arr
            session_ks_ids[sess_date] = ks_arr

            for i in range(len(snippets)):
                all_session_ids.append(sess_date)
                all_ks_ids.append(ks_ids[i])
                all_positions.append(positions[i])

        N = len(all_ks_ids)
        print(f"\nTotal units across all sessions: {N}")

        # ---- Save unit index ----
        unit_index = pd.DataFrame({
            'session': all_session_ids,
            'ks_id': all_ks_ids,
            'pos_x': [p[0] for p in all_positions],
            'pos_y': [p[1] for p in all_positions],
        })
        unit_index.to_csv(OUTPUT_ROOT / "unit_index.csv", index=False)

        # ---- Save embeddings ----
        emb_fh_all = np.vstack([session_emb_fh[sd] for sd in session_dates])
        emb_sh_all = np.vstack([session_emb_sh[sd] for sd in session_dates])
        np.savez(OUTPUT_ROOT / "embeddings.npz",
                 emb_fh=emb_fh_all, emb_sh=emb_sh_all,
                 session_ids=np.array(all_session_ids),
                 ks_ids=np.array(all_ks_ids))
        print(f"Saved embeddings and unit index.")

    # ---- Match consecutive session pairs ----
    print(f"\n{'='*60}")
    print(f"Per-pair softmax matching ({len(session_dates) - 1} consecutive pairs)")
    print(f"  temp_tau={temp_tau:.6f}, dist_thresh={args.dist_thresh}, min_prob={args.min_prob}")
    print(f"{'='*60}")

    pair_output_dir = OUTPUT_ROOT / "pair_registries"
    pair_output_dir.mkdir(parents=True, exist_ok=True)

    pair_registry_paths = []
    total_matches = 0
    all_match_rows = []

    for idx in range(len(session_dates) - 1):
        date_a = session_dates[idx]
        date_b = session_dates[idx + 1]

        matches = match_consecutive_pair(
            session_emb_fh[date_a], session_emb_sh[date_a],
            session_emb_fh[date_b], session_emb_sh[date_b],
            session_positions[date_a], session_positions[date_b],
            session_ks_ids[date_a], session_ks_ids[date_b],
            temp_tau=temp_tau,
            dist_thresh=args.dist_thresh,
            min_prob=args.min_prob,
        )

        nA = len(session_ks_ids[date_a])
        nB = len(session_ks_ids[date_b])
        n_matches = len(matches)
        total_matches += n_matches

        # Diagnostic: probability stats for this pair
        if n_matches > 0:
            p_mean = matches['Prob'].mean()
            p_min = matches['Prob'].min()
            d_mean = matches['dist'].mean() if 'dist' in matches.columns else 0
            print(f"  {date_a} -> {date_b}: {n_matches:3d} matches "
                  f"(of {nA}+{nB} units) | prob {p_min:.3f}-{p_mean:.3f} | "
                  f"dist {d_mean:.0f} um")

            # Save match details
            match_save = matches[['ID1', 'ID2', 'Prob', 'prob_AB', 'prob_BA', 'dist']].copy()
            match_save.insert(0, 'RecSes1', date_a)
            match_save.insert(2, 'RecSes2', date_b)
            all_match_rows.append(match_save)
        else:
            print(f"  {date_a} -> {date_b}:   0 matches (of {nA}+{nB} units)")

        # Build and save per-pair registry
        registry = build_pair_registry(
            matches, session_ks_ids[date_a], session_ks_ids[date_b],
            date_a, date_b
        )
        pair_path = pair_output_dir / f"CellRegistry_{date_a}_{date_b}.csv"
        registry.to_csv(pair_path)
        pair_registry_paths.append(pair_path)

    # ---- Save all matches ----
    if all_match_rows:
        all_matches_df = pd.concat(all_match_rows, ignore_index=True)
        all_matches_df.to_csv(OUTPUT_ROOT / "DeepUM_MatchTable.csv", index=False)
        print(f"\nSaved {len(all_matches_df)} total pairwise matches to DeepUM_MatchTable.csv")

    # ---- Build Global CellRegistry from match table via Union-Find ----
    print("\nBuilding Global CellRegistry via Union-Find...")

    # Union-Find data structure
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Initialize: each (session, ks_id) is its own component
    for sd in session_dates:
        for ks_id in session_ks_ids[sd]:
            key = (sd, int(ks_id))
            parent[key] = key

    # Union matched pairs from all consecutive pair matches
    if all_match_rows:
        for df_m in all_match_rows:
            for _, row in df_m.iterrows():
                a = (row['RecSes1'], int(row['ID1']))
                b = (row['RecSes2'], int(row['ID2']))
                if a in parent and b in parent:
                    union(a, b)

    # Group by component
    from collections import defaultdict
    groups = defaultdict(list)
    for unit_key in parent:
        groups[find(unit_key)].append(unit_key)

    # Build registry DataFrame
    registry_rows = []
    for uid, members in enumerate(sorted(groups.values(),
                                          key=lambda g: min(_parse_date(s)
                                                            for s, _ in g))):
        row_dict = {'UID': uid}
        for sess, ks_id in members:
            row_dict[sess] = ks_id
        registry_rows.append(row_dict)

    global_registry = pd.DataFrame(registry_rows).set_index('UID')

    # Ensure all session columns exist
    for sd in session_dates:
        if sd not in global_registry.columns:
            global_registry[sd] = np.nan

    # Reorder columns chronologically
    global_registry = global_registry[sorted(
        global_registry.columns,
        key=lambda c: _parse_date(c)
    )]

    registry_path = OUTPUT_ROOT / "DeepUM_CellRegistry.csv"
    global_registry.to_csv(registry_path)
    print(f"Saved Global CellRegistry to {registry_path}")

    # ---- Summary ----
    n_sessions_per_uid = global_registry.notna().sum(axis=1)
    multi_session = (n_sessions_per_uid > 1).sum()
    print(f"\n{'='*60}")
    print(f"DeepUnitMatch Pipeline Summary")
    print(f"{'='*60}")
    print(f"  Sessions processed:       {len(session_dates)}")
    print(f"  Consecutive pairs:        {len(session_dates) - 1}")
    print(f"  Total pairwise matches:   {total_matches}")
    print(f"  Tracked neurons (UIDs):   {len(global_registry)}")
    print(f"  Multi-session UIDs:       {multi_session}")
    print(f"  Max sessions tracked:     {n_sessions_per_uid.max()}")
    print(f"  Output directory:         {OUTPUT_ROOT}")
    print(f"{'='*60}")
    print(f"\nTo use with build_longitudinal_table.py:")
    print(f"  python scripts/analysis/build_longitudinal_table.py \\")
    print(f"    --registry {registry_path}")


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
