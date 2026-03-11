"""Stitch unit identities across overlapping concatenation windows.

For each shank, uses overlapping sessions (sessions that appear in
consecutive windows) to match cluster IDs by spike-time agreement.
Produces a global unit registry and canonical per-session KS folders
with globally consistent cluster IDs.

The algorithm:
  1. For each pair of consecutive windows sharing overlapping sessions,
     load the split KS output from both.
  2. For each overlapping session, find cluster pairs whose spike times
     agree (>90% of spikes in common within a tolerance of +/-1 sample).
  3. Build a union-find structure across all windows to assign global UIDs.
  4. For each session, choose the "best" window (most central) and remap
     cluster IDs to global UIDs.

Usage:
    python scripts/pipelines/concat_sort/stitch_across_windows.py \
        --split-manifest <path_to_split_manifest.json> \
        --ks4-run-manifest <path_to_ks4_run_manifest.json> \
        [--overlap-threshold 0.9] \
        [--output-dir ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Union-Find ────────────────────────────────────────────────────────

class UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self):
        self.parent: Dict = {}
        self.rank: Dict = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> Dict:
        """Return {root: [members]}."""
        g = defaultdict(list)
        for x in self.parent:
            g[self.find(x)].append(x)
        return dict(g)


# ── Spike-time matching ───────────────────────────────────────────────

def match_clusters_by_spikes(
    spike_times_a: np.ndarray,
    spike_clusters_a: np.ndarray,
    spike_times_b: np.ndarray,
    spike_clusters_b: np.ndarray,
    threshold: float = 0.9,
    sample_tol: int = 1,
) -> List[Tuple[int, int, float]]:
    """Find matching clusters between two sort results for the SAME session.

    Uses a fully vectorized global nearest-neighbor approach: for every
    spike in A, find its nearest spike in B via searchsorted, then
    aggregate match counts per (cluster_a, cluster_b) pair.

    Returns list of (cluster_a, cluster_b, agreement_fraction).
    """
    if spike_times_a.size == 0 or spike_times_b.size == 0:
        return []

    # Ensure B is sorted by time for searchsorted
    order_b = np.argsort(spike_times_b, kind="mergesort")
    st_b = spike_times_b[order_b].astype(np.int64)
    sc_b = spike_clusters_b[order_b]

    st_a = spike_times_a.astype(np.int64)
    sc_a = spike_clusters_a

    # For each spike in A, find nearest spike in B
    idx = np.searchsorted(st_b, st_a, side="left")
    idx_right = np.clip(idx, 0, st_b.size - 1)
    idx_left = np.clip(idx - 1, 0, st_b.size - 1)

    diff_right = np.abs(st_a - st_b[idx_right])
    diff_left = np.abs(st_a - st_b[idx_left])

    use_left = diff_left < diff_right
    nearest_diff = np.where(use_left, diff_left, diff_right)
    nearest_idx = np.where(use_left, idx_left, idx_right)

    matched = nearest_diff <= sample_tol
    if not matched.any():
        return []

    ca_matched = sc_a[matched]
    cb_matched = sc_b[nearest_idx[matched]]

    # Spike counts per cluster
    clusters_a_u, counts_a = np.unique(sc_a, return_counts=True)
    clusters_b_u, counts_b = np.unique(spike_clusters_b, return_counts=True)
    count_a = dict(zip(clusters_a_u.tolist(), counts_a.tolist()))
    count_b = dict(zip(clusters_b_u.tolist(), counts_b.tolist()))

    # Count matches per (ca, cb) pair
    pairs = np.stack([ca_matched, cb_matched], axis=1)
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)

    # Find best match per cluster_a
    best_per_ca: Dict[int, Tuple[int, float]] = {}
    for i in range(len(unique_pairs)):
        ca, cb = int(unique_pairs[i, 0]), int(unique_pairs[i, 1])
        cnt = int(pair_counts[i])
        smaller = min(count_a.get(ca, 0), count_b.get(cb, 0))
        if smaller == 0:
            continue
        frac = cnt / smaller
        if frac >= threshold and (ca not in best_per_ca or frac > best_per_ca[ca][1]):
            best_per_ca[ca] = (cb, frac)

    return [(ca, cb, round(frac, 4)) for ca, (cb, frac) in best_per_ca.items()]


# ── Window centrality ────────────────────────────────────────────────

def session_centrality(session_name: str, window_sessions: List[str]) -> float:
    """Return a centrality score (0-1) for a session within a window.

    Sessions in the middle of the window get higher scores (better sort
    quality expected).
    """
    n = len(window_sessions)
    if n <= 1:
        return 1.0
    try:
        idx = window_sessions.index(session_name)
    except ValueError:
        return 0.0
    # Distance from centre, normalised
    centre = (n - 1) / 2.0
    return 1.0 - abs(idx - centre) / centre


# ── Main pipeline ────────────────────────────────────────────────────

def stitch_shank(
    shank_id: str,
    split_manifest: dict,
    run_manifest: dict,
    output_dir: Path,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Stitch unit identities for one shank across all windows.

    Returns a DataFrame with columns:
        global_uid, session, window_idx, original_cluster_id, n_spikes
    """
    # Collect all windows for this shank, sorted by window_idx
    shank_windows = [
        w for w in run_manifest["windows"]
        if str(w["shank_id"]) == str(shank_id)
    ]
    shank_windows.sort(key=lambda w: w["window_idx"])

    if not shank_windows:
        print(f"  No windows for shank {shank_id}")
        return pd.DataFrame()

    # Build union-find over (window_idx, cluster_id) keys
    uf = UnionFind()

    # Helper for parallel overlap-session matching
    def _match_overlap_session(session_name, w_a, w_b):
        """Load data and match clusters for one overlap session."""
        path_a = path_b = None
        for entry in split_manifest.get(session_name, []):
            if entry["window_idx"] == w_a["window_idx"] and str(entry["shank_id"]) == str(shank_id):
                path_a = Path(entry["path"])
            elif entry["window_idx"] == w_b["window_idx"] and str(entry["shank_id"]) == str(shank_id):
                path_b = Path(entry["path"])

        if path_a is None or path_b is None:
            return session_name, []
        if not (path_a / "spike_times.npy").exists() or not (path_b / "spike_times.npy").exists():
            return session_name, []

        st_a = np.load(str(path_a / "spike_times.npy"), mmap_mode="r").flatten()
        sc_a = np.load(str(path_a / "spike_clusters.npy"), mmap_mode="r").flatten()
        st_b = np.load(str(path_b / "spike_times.npy"), mmap_mode="r").flatten()
        sc_b = np.load(str(path_b / "spike_clusters.npy"), mmap_mode="r").flatten()

        return session_name, match_clusters_by_spikes(st_a, sc_a, st_b, sc_b, threshold)

    # For each pair of consecutive windows, find matches in overlapping sessions
    for i in range(len(shank_windows) - 1):
        w_a = shank_windows[i]
        w_b = shank_windows[i + 1]
        overlap = set(w_a["sessions"]) & set(w_b["sessions"])
        if not overlap:
            print(f"  WARNING: No overlap between window {w_a['window_idx']} and {w_b['window_idx']}")
            continue

        print(f"  Window pair {w_a['window_idx']}↔{w_b['window_idx']} "
              f"({len(overlap)} overlap sessions)...")

        # Match overlap sessions in parallel (I/O + numpy releases GIL)
        with ThreadPoolExecutor(max_workers=min(len(overlap), 4)) as pool:
            futures = [pool.submit(_match_overlap_session, s, w_a, w_b)
                       for s in overlap]
            for fut in as_completed(futures):
                session_name, matches = fut.result()
                for ca, cb, frac in matches:
                    uf.union((w_a["window_idx"], int(ca)),
                             (w_b["window_idx"], int(cb)))
                print(f"    Overlap {session_name}: {len(matches)} cluster matches "
                      f"(win {w_a['window_idx']} ↔ win {w_b['window_idx']})")

    # Assign global UIDs from union-find groups
    groups = uf.groups()
    global_uid_map = {}  # (window_idx, cluster_id) -> global_uid
    for uid, (root, members) in enumerate(groups.items()):
        for key in members:
            global_uid_map[key] = uid

    # Also assign UIDs to any clusters not involved in any match
    next_uid = len(groups)

    # Build registry rows
    rows = []
    for w in shank_windows:
        widx = w["window_idx"]
        for session_name in w["sessions"]:
            entries = [
                e for e in split_manifest.get(session_name, [])
                if e["window_idx"] == widx and str(e["shank_id"]) == str(shank_id)
            ]
            if not entries:
                continue
            entry = entries[0]
            ks_path = Path(entry["path"])
            if not (ks_path / "spike_clusters.npy").exists():
                continue

            sc = np.load(str(ks_path / "spike_clusters.npy"), mmap_mode="r").flatten()

            unique_clusters, cluster_counts = np.unique(sc, return_counts=True)
            centrality = round(session_centrality(session_name, w["sessions"]), 3)
            for cid, n_spk in zip(unique_clusters, cluster_counts):
                key = (widx, int(cid))
                if key not in global_uid_map:
                    global_uid_map[key] = next_uid
                    next_uid += 1

                rows.append({
                    "global_uid": global_uid_map[key],
                    "session": session_name,
                    "shank_id": shank_id,
                    "window_idx": widx,
                    "original_cluster_id": int(cid),
                    "n_spikes": int(n_spk),
                    "centrality": centrality,
                })

    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry

    # ── Choose canonical window per (global_uid, session) ──
    # Prefer the window where the session is most central
    registry = registry.sort_values(
        ["global_uid", "session", "centrality"], ascending=[True, True, False]
    )
    canonical = registry.groupby(["global_uid", "session"]).first().reset_index()

    # ── Write canonical per-session KS folders ──
    for session_name in canonical["session"].unique():
        sess_rows = canonical[canonical["session"] == session_name]
        sess_out = output_dir / session_name / f"shank_{shank_id}"

        # Skip if already written (resume support)
        if (sess_out / "spike_clusters.npy").exists():
            continue

        sess_out.mkdir(parents=True, exist_ok=True)

        # Pick the canonical window for this session (most common among its units)
        best_window = int(sess_rows["window_idx"].mode().iloc[0])
        source_entries = [
            e for e in split_manifest.get(session_name, [])
            if e["window_idx"] == best_window and str(e["shank_id"]) == str(shank_id)
        ]
        if not source_entries:
            continue
        source_path = Path(source_entries[0]["path"])

        if not (source_path / "spike_times.npy").exists():
            continue

        # Load source spikes
        st = np.load(str(source_path / "spike_times.npy"), mmap_mode="r").flatten()
        sc = np.load(str(source_path / "spike_clusters.npy"), mmap_mode="r").flatten()

        # Build remap: original_cluster_id -> global_uid
        remap = {}
        for _, row in sess_rows[sess_rows["window_idx"] == best_window].iterrows():
            remap[int(row["original_cluster_id"])] = int(row["global_uid"])

        # Vectorised remap of cluster IDs
        unique_orig = np.array(list(remap.keys()), dtype=np.int64)
        unique_glob = np.array(list(remap.values()), dtype=np.int64)
        max_cid = max(int(sc.max()), int(unique_orig.max())) + 1
        lut = np.arange(max_cid, dtype=np.int64)  # identity by default
        lut[unique_orig] = unique_glob
        sc_global = lut[sc.astype(np.int64)]

        np.save(str(sess_out / "spike_times.npy"), st)
        np.save(str(sess_out / "spike_clusters.npy"), sc_global)

        # Copy remaining files from source
        for fname in [
            "spike_templates.npy", "spike_detection_templates.npy",
            "amplitudes.npy", "spike_positions.npy",
            "templates.npy", "whitening_mat.npy", "whitening_mat_dat.npy",
            "channel_map.npy", "channel_positions.npy", "channel_shanks.npy",
            "chanMap.mat", "similar_templates.npy", "templates_ind.npy",
            "pc_features.npy", "pc_feature_ind.npy", "ops.npy",
            "cluster_group.tsv", "cluster_KSLabel.tsv",
            "cluster_Amplitude.tsv", "cluster_ContamPct.tsv",
            "params.py",
        ]:
            src = source_path / fname
            if src.exists():
                shutil.copy2(str(src), str(sess_out / fname))

        print(f"  {session_name}/shank_{shank_id}: "
              f"{len(np.unique(sc_global))} global units (from window {best_window})")

    return registry


def main(argv=None):
    p = argparse.ArgumentParser(description="Stitch unit identities across overlapping windows")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--ks4-run-manifest", type=Path, required=True)
    p.add_argument("--overlap-threshold", type=float, default=0.9,
                   help="Min fraction of spike agreement to consider a match")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: sibling of split_output)")
    args = p.parse_args(argv)

    with open(args.split_manifest) as f:
        split_manifest = json.load(f)
    with open(args.ks4_run_manifest) as f:
        run_manifest = json.load(f)

    # Default output: alongside the split_output directory
    if args.output_dir is None:
        args.output_dir = args.split_manifest.resolve().parent.parent / "final_output"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Split manifest:  {args.split_manifest}")
    print(f"KS4 manifest:    {args.ks4_run_manifest}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Threshold:       {args.overlap_threshold}")
    print()

    shank_ids = run_manifest.get("shank_ids", ["0", "1", "2", "3"])
    t0 = time.time()

    # Process shanks with per-shank checkpoint CSVs for resume support.
    # Run up to 4 in parallel; completed shanks are skipped on re-run.
    all_registries = []
    shanks_to_run = []
    for sid in shank_ids:
        ckpt = args.output_dir / f"_registry_shank_{sid}.csv"
        if ckpt.exists():
            print(f"  Shank {sid}: loading checkpoint {ckpt.name}")
            all_registries.append((sid, pd.read_csv(ckpt)))
        else:
            shanks_to_run.append(sid)

    if shanks_to_run:
        print(f"Processing {len(shanks_to_run)} shanks in parallel: {shanks_to_run}")

        def _run_shank(sid):
            print(f"\n══ Shank {sid} ══")
            t_shank = time.time()
            registry = stitch_shank(
                sid, split_manifest, run_manifest,
                args.output_dir, args.overlap_threshold,
            )
            elapsed = time.time() - t_shank
            if not registry.empty:
                # Save per-shank checkpoint
                ckpt = args.output_dir / f"_registry_shank_{sid}.csv"
                registry.to_csv(ckpt, index=False)
                print(f"══ Shank {sid} DONE ══ {registry['global_uid'].nunique()} UIDs, "
                      f"{len(registry)} rows ({elapsed:.1f}s)")
            return sid, registry

        with ThreadPoolExecutor(max_workers=len(shanks_to_run)) as pool:
            futures = [pool.submit(_run_shank, sid) for sid in shanks_to_run]
            for fut in as_completed(futures):
                sid, registry = fut.result()
                if not registry.empty:
                    all_registries.append((sid, registry))
    else:
        print("All shanks already checkpointed — skipping stitch.")

    # Sort by shank_id for deterministic UID assignment
    all_registries.sort(key=lambda x: str(x[0]))
    all_registries = [reg for _, reg in all_registries]

    # Combine registries across shanks (UIDs are shank-local; make them global)
    if all_registries:
        offset = 0
        for reg in all_registries:
            reg["global_uid"] += offset
            offset = reg["global_uid"].max() + 1

        combined = pd.concat(all_registries, ignore_index=True)
        registry_path = args.output_dir / "global_registry.csv"
        combined.to_csv(registry_path, index=False)
        print(f"\nGlobal registry → {registry_path}")
        print(f"  {combined['global_uid'].nunique()} unique global units")
        print(f"  {combined['session'].nunique()} sessions")
        print(f"  {len(combined)} total (unit × session × window) entries")

        # Also save canonical-only view
        canonical = combined.sort_values(
            ["global_uid", "session", "centrality"], ascending=[True, True, False]
        ).groupby(["global_uid", "session"]).first().reset_index()
        canonical_path = args.output_dir / "global_registry_canonical.csv"
        canonical.to_csv(canonical_path, index=False)
        print(f"  Canonical registry → {canonical_path}")
        print(f"  Canonical: {canonical['global_uid'].nunique()} units "
              f"across {canonical['session'].nunique()} sessions")
    else:
        print("\nNo registries produced.")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
