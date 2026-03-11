#!/usr/bin/env python3
"""Build .pkl session files from concat-sort KS4 output with TPrime correction.

For each session:
  1. Load existing .pkl for behavioral data (trials, NI events)
  2. Load concat-sort spike data from all 4 shanks
  3. Apply TPrime clock-drift correction (interpolated from original KS output)
  4. Merge shanks into a single flat cluster list with unique IDs
  5. Save to data/pkls/BG_046_concat_sort/

Cluster ID convention:
  new_id = original_cluster_id + shank_id * 100_000
  e.g., cluster 42 on shank 2 → ID 200042

Quality filtering:
  good_cluster_ids = KS4 "good" labeled clusters (all shanks).
  good_and_stable_ids = good clusters passing the stability filter
    (Python port of find_good_stable_units_PaperVersion.m:
     rate >= 0.5 Hz, stable across 5/10/20-min windows, clean ISI).
  The analysis pipeline further filters by firing rate >= 1 Hz at load time.

Usage:
    python scripts/pipelines/concat_sort/build_concat_pkls.py
    python scripts/pipelines/concat_sort/build_concat_pkls.py --sessions BG_046_01072025
    python scripts/pipelines/concat_sort/build_concat_pkls.py --dry-run
"""

import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import session module directly to avoid heavy transitive deps
# (visdetect/__init__ → core/__init__ → qc → align → h5py).
import types
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "visdetect.core.session",
    REPO_ROOT / "src" / "visdetect" / "core" / "session.py",
)
_session_mod = importlib.util.module_from_spec(_spec)
# Register in sys.modules so pickle can resolve the class paths
sys.modules.setdefault("visdetect", types.ModuleType("visdetect"))
sys.modules.setdefault("visdetect.core", types.ModuleType("visdetect.core"))
sys.modules["visdetect.core.session"] = _session_mod
_spec.loader.exec_module(_session_mod)
Session = _session_mod.Session
Cluster = _session_mod.Cluster
load_session = _session_mod.load_session
save_session = _session_mod.save_session

# ── Constants ─────────────────────────────────────────────────────────
SUBJECT = "BG_046"
FINAL_OUTPUT = Path(
    r"X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046"
    r"/concat_sort/final_output"
)
PROCESSED_BASE = Path(
    r"X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046"
    r"/Processed data"
)
PKL_INPUT_DIR = REPO_ROOT / "data" / "pkls" / SUBJECT
PKL_OUTPUT_DIR = REPO_ROOT / "data" / "pkls" / f"{SUBJECT}_concat_sort"
MANIFEST_PATH = REPO_ROOT / "data" / f"{SUBJECT}_staging_manifest.csv"

SAMPLE_RATE = 30_000.0
SHANK_ID_OFFSET = 100_000
N_SHANKS = 4


# ── TPrime correction ────────────────────────────────────────────────

def find_original_ks_path(session_name):
    """Locate the original per-session KS4 folder on X: drive."""
    pattern = str(PROCESSED_BASE / f"*{session_name}*" / "Kilosort&Phy" / "*imec0")
    found = glob.glob(pattern)
    return Path(found[0]) if found else None


def build_tprime_correction(ks_path):
    """Build lightweight TPrime interpolation anchors from original output.

    Returns (sec_anchors, adj_anchors) for np.interp, or None if data
    unavailable.  We subsample to ~2 000 evenly-spaced points, which is
    accurate to <1 µs across the full recording.
    """
    sec_path = ks_path / "spike_times_sec.npy"
    adj_path = ks_path / "spike_times_sec_adj.npy"
    if not sec_path.exists() or not adj_path.exists():
        return None

    sec = np.load(sec_path).flatten()
    adj = np.load(adj_path).flatten()
    if len(sec) != len(adj) or len(sec) < 2:
        return None

    # Subsample for efficiency — ~2 000 points captures drift perfectly
    n = len(sec)
    step = max(1, n // 2000)
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)

    return sec[idx].copy(), adj[idx].copy()


def apply_tprime(spike_times_samples, correction):
    """Convert spike times from samples → TPrime-corrected seconds."""
    t_sec = spike_times_samples.astype(np.float64) / SAMPLE_RATE
    if correction is None:
        return t_sec
    sec_anchors, adj_anchors = correction
    return np.interp(t_sec, sec_anchors, adj_anchors)


# ── Per-shank registry (global_uid -> original_cluster_id mapping) ────

def load_shank_registries():
    """Load per-shank registry CSVs produced by stitch_across_windows.

    Returns dict {shank_id: DataFrame} with columns including
    global_uid, session, original_cluster_id.
    """
    regs = {}
    for shank_id in range(N_SHANKS):
        path = FINAL_OUTPUT / f"_registry_shank_{shank_id}.csv"
        if path.exists():
            regs[shank_id] = pd.read_csv(path)
        else:
            print(f"  WARNING: registry not found: {path}")
    return regs


# ── Concat-sort data loading ─────────────────────────────────────────

def load_shank_spikes(session_name, shank_id, registry_df=None):
    """Load spike data and KS4 labels from one shank folder.

    The final_output spike_clusters.npy uses global UIDs (from stitching),
    but cluster_KSLabel.tsv still uses original (pre-remap) cluster IDs.
    We use the per-shank registry to map global_uid -> original_cluster_id
    and then look up the KS4 label for the original ID.

    Returns dict with spike_clusters, spike_times (samples), ks_good_set,
    or None if data missing.
    """
    shank_dir = FINAL_OUTPUT / session_name / f"shank_{shank_id}"
    sc_path = shank_dir / "spike_clusters.npy"
    st_path = shank_dir / "spike_times.npy"
    if not sc_path.exists() or not st_path.exists():
        return None

    spike_clusters = np.load(sc_path).flatten()
    spike_times = np.load(st_path).flatten()

    # Read KS4 quality labels (keyed by ORIGINAL cluster IDs)
    original_good_ids = set()
    for tsv_name in ("cluster_KSLabel.tsv", "cluster_group.tsv"):
        tsv_path = shank_dir / tsv_name
        if not tsv_path.exists():
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        label_col = "KSLabel" if "KSLabel" in df.columns else "group"
        original_good_ids = set(
            df.loc[df[label_col].str.strip() == "good", "cluster_id"]
            .astype(int)
        )
        break

    # Map global UIDs (in spike_clusters) -> original cluster IDs via registry
    ks_good = set()
    if registry_df is not None:
        sub = registry_df[registry_df["session"] == session_name]
        uid_to_orig = dict(zip(sub["global_uid"].astype(int),
                               sub["original_cluster_id"].astype(int)))
        for uid in np.unique(spike_clusters):
            orig = uid_to_orig.get(int(uid))
            if orig is not None and orig in original_good_ids:
                ks_good.add(int(uid))
    else:
        # Fallback: no registry available, use TSV IDs directly (lossy)
        ks_good = original_good_ids & set(np.unique(spike_clusters).tolist())

    return {
        "spike_clusters": spike_clusters,
        "spike_times": spike_times,
        "ks_good": ks_good,
    }


# ── Stability filter (port of find_good_stable_units_PaperVersion.m) ──

def find_good_stable_units(clusters, good_cluster_ids):
    """Keep only stable units from KS4 good units.

    Python port of find_good_stable_units_PaperVersion.m
    (Khilkevich & Lohse 2024 criteria):
      - Average firing rate >= 0.5 Hz
      - Rate in 20 / 10 / 5-min sliding window never drops below
        30% / 20% / 10% of the session average
      - ISI distribution peak in first 5 ms is at >= 2 ms
      - ISI distribution is smooth (tallest bin < 4x second-tallest)
    """
    BIN_SEC = 0.01  # 10 ms bins, matching MATLAB
    # Window sizes in bins: 20 min, 10 min, 5 min
    WIN_20 = int(20 * 60 / BIN_SEC)  # 120 000
    WIN_10 = int(10 * 60 / BIN_SEC)  #  60 000
    WIN_5  = int( 5 * 60 / BIN_SEC)  #  30 000

    # Recording duration = latest spike across ALL clusters
    max_t = max(
        (c.spike_times[-1] if len(c.spike_times) > 0 else 0)
        for c in clusters
    )
    rec_dur = max_t if max_t > 0 else 1.0

    good_set = set(good_cluster_ids)
    stable_ids = []

    for c in clusters:
        if c.cluster_id not in good_set:
            continue
        sp = c.spike_times
        if len(sp) < 2:
            continue

        avg_fr = len(sp) / rec_dur
        if avg_fr < 0.5:
            continue

        # ── Firing-rate stability ────────────────────────────
        n_bins = int(np.ceil(rec_dur / BIN_SEC))
        counts, _ = np.histogram(sp, bins=n_bins, range=(0, rec_dur))
        fr = counts.astype(np.float64) / BIN_SEC

        # uniform_filter1d with mode='nearest' ≈ MATLAB movmean
        if n_bins >= WIN_5:  # need at least a 5-min recording
            fr_20 = uniform_filter1d(fr, size=min(WIN_20, n_bins), mode="nearest")
            fr_10 = uniform_filter1d(fr, size=min(WIN_10, n_bins), mode="nearest")
            fr_5  = uniform_filter1d(fr, size=min(WIN_5,  n_bins), mode="nearest")

            if fr_20.min() < 0.3 * avg_fr:
                continue
            if fr_10.min() < 0.2 * avg_fr:
                continue
            if fr_5.min() < 0.1 * avg_fr:
                continue

        # ── ISI quality ──────────────────────────────────────
        isi = np.diff(sp)
        # 50 bins from 0-50 ms (edges: 0, 0.001, ..., 0.050)
        isi_counts, _ = np.histogram(isi, bins=np.arange(0, 0.051, 0.001))

        # Peak among first 5 bins must be at >= 2 ms
        # (MATLAB 1-indexed > 2 ↔ Python 0-indexed >= 2)
        first5 = isi_counts[:5]
        if first5.max() == 0:
            continue
        peak_positions = np.where(first5 == first5.max())[0]
        if peak_positions.min() < 2:
            continue

        # Smoothness: tallest bin < 4x second-tallest (all 50 bins)
        top2 = np.sort(isi_counts)[-2:]  # [second, first]
        if top2[0] == 0:  # only one non-zero bin -> not smooth
            continue
        if top2[1] >= 4 * top2[0]:
            continue

        stable_ids.append(c.cluster_id)

    return sorted(stable_ids)


# ── Session builder ──────────────────────────────────────────────────

def process_session(session_name, shank_registries=None):
    """Build a new Session from concat-sort spikes + existing behavioral data."""

    # 1. Load existing pkl for trials / NI events
    existing_pkl = PKL_INPUT_DIR / f"{session_name}.pkl"
    if existing_pkl.exists():
        old = load_session(str(existing_pkl))
        trials = old.trials
        ni_events = old.ni_events
        subject = old.subject
        sess_date = old.session_name
    else:
        print(f"  Warning: no existing pkl — behavioral data will be empty")
        trials, ni_events = [], {}
        subject = SUBJECT
        parts = session_name.split("_")
        sess_date = parts[-1] if len(parts) >= 3 else session_name

    # 2. TPrime correction from original KS output
    orig_ks = find_original_ks_path(session_name)
    correction = None
    if orig_ks:
        correction = build_tprime_correction(orig_ks)
    if correction is not None:
        print(f"  TPrime: {len(correction[0])} anchor pts")
    else:
        print(f"  TPrime: unavailable — using uncorrected times (samples/30 kHz)")

    # 3. Load all shanks, build Cluster list
    all_clusters = []
    good_ids = []

    for shank_id in range(N_SHANKS):
        reg_df = shank_registries.get(shank_id) if shank_registries else None
        shank = load_shank_spikes(session_name, shank_id, registry_df=reg_df)
        if shank is None:
            continue

        sc = shank["spike_clusters"]
        st_corrected = apply_tprime(shank["spike_times"], correction)
        ks_good = shank["ks_good"]

        unique_ids = np.unique(sc)
        # Pre-sort for fast per-cluster indexing
        order = np.argsort(sc)
        sorted_sc = sc[order]
        left = np.searchsorted(sorted_sc, unique_ids, side="left")
        right = np.searchsorted(sorted_sc, unique_ids, side="right")

        n_good = 0
        for i, cid in enumerate(unique_ids):
            cid_int = int(cid)
            new_id = cid_int + shank_id * SHANK_ID_OFFSET

            idx = order[left[i] : right[i]]
            times = np.sort(st_corrected[idx])

            quality = "good" if cid_int in ks_good else "mua"
            all_clusters.append(
                Cluster(cluster_id=new_id, spike_times=times, quality=quality)
            )
            if quality == "good":
                good_ids.append(new_id)
                n_good += 1

        print(
            f"  shank {shank_id}: {len(unique_ids)} clusters "
            f"({n_good} good, {len(sc):,} spikes)"
        )

    good_sorted = sorted(good_ids)

    # 4. Apply stability filter (port of find_good_stable_units_PaperVersion)
    stable_ids = find_good_stable_units(all_clusters, good_sorted)
    print(f"  Stability filter: {len(stable_ids)} / {len(good_sorted)} good clusters are stable")

    return Session(
        trials=trials,
        clusters=all_clusters,
        subject=subject,
        session_name=sess_date,
        good_cluster_ids=good_sorted,
        good_and_stable_ids=stable_ids,
        ni_events=ni_events,
    )


# ── Parallel worker ──────────────────────────────────────────────────

def _build_one_session(sess_name, shank_registries, output_dir, force):
    """Build a single session pkl. Used by both serial and parallel modes.

    Returns a summary string for logging.
    """
    out_path = output_dir / f"{sess_name}.pkl"
    if out_path.exists() and not force:
        return f"[{sess_name}] exists -- skip"

    new_session = process_session(sess_name, shank_registries)
    save_session(new_session, str(out_path))

    n_cl = len(new_session.clusters)
    n_gd = len(new_session.good_and_stable_ids or [])
    n_tr = len(new_session.trials)
    return (f"[{sess_name}] {n_cl} clusters ({n_gd} stable), "
            f"{n_tr} trials -> {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build .pkl sessions from concat-sort KS4 + TPrime"
    )
    parser.add_argument(
        "--sessions", nargs="+",
        help="Specific sessions (e.g. BG_046_01072025)"
    )
    parser.add_argument(
        "--output", type=Path, default=PKL_OUTPUT_DIR,
        help=f"Output directory (default: {PKL_OUTPUT_DIR})"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing pkls")
    parser.add_argument("--dry-run", action="store_true", help="Show plan, don't write")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (default: 1 = serial).  4-6 recommended.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Discover concat-sort sessions
    sessions = sorted(
        d.name
        for d in FINAL_OUTPUT.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "global"))
    )
    if args.sessions:
        targets = args.sessions
        sessions = [s for s in sessions if any(t in s for t in targets)]

    print(f"Concat-sort sessions: {len(sessions)}")
    print(f"Existing pkls dir:    {PKL_INPUT_DIR}")
    print(f"Output dir:           {args.output}")

    if args.dry_run:
        for s in sessions:
            pkl_ok = "Y" if (PKL_INPUT_DIR / f"{s}.pkl").exists() else "N"
            ks = find_original_ks_path(s)
            tp_ok = "Y" if ks and (ks / "spike_times_sec_adj.npy").exists() else "N"
            print(f"  {s}  pkl:{pkl_ok}  tprime:{tp_ok}")
        return

    # Load per-shank registries for correct global_uid -> original_id mapping
    shank_registries = load_shank_registries()

    n_workers = max(1, args.workers)

    if n_workers == 1:
        # Serial mode (original behavior, with per-session logging)
        for sess_name in tqdm(sessions, desc="Building pkls"):
            out_path = args.output / f"{sess_name}.pkl"
            if out_path.exists() and not args.force:
                print(f"\n[{sess_name}] exists -- skip (use --force to overwrite)")
                continue

            print(f"\n[{sess_name}]")
            new_session = process_session(sess_name, shank_registries)
            save_session(new_session, str(out_path))

            n_cl = len(new_session.clusters)
            n_gd = len(new_session.good_and_stable_ids or [])
            n_tr = len(new_session.trials)
            print(f"  Saved: {n_cl} clusters ({n_gd} stable), "
                  f"{n_tr} trials -> {out_path.name}")
    else:
        # Parallel mode
        print(f"Using {n_workers} parallel workers")
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for sess_name in sessions:
                fut = pool.submit(
                    _build_one_session,
                    sess_name, shank_registries, args.output, args.force,
                )
                futures[fut] = sess_name

            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Building pkls"
            ):
                sess_name = futures[fut]
                try:
                    msg = fut.result()
                    tqdm.write(msg)
                except Exception as exc:
                    tqdm.write(f"[{sess_name}] FAILED: {exc}")

    print(f"\nDone. Pkls in {args.output}")
    print(
        f"\nTo use in analysis_suite, update PKL_DIR in "
        f"src/visdetect/analysis/config.py:\n"
        f'  PKL_DIR = os.path.join(ROOT, "data", "pkls", "{SUBJECT}_concat_sort")'
    )


if __name__ == "__main__":
    main()
