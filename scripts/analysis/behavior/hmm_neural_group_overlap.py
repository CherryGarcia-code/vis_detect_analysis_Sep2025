"""Overlap visualization between TF-responsive and lick-responsive neuron groups.

For each session, loads:
  - TF classification (from ``tf_pulse_grid_both.csv``)
  - Lick classification (from ``lick_responsiveness.csv``)

Produces:
  - **Per-session** 2-set Venn diagrams (TF-resp vs Lick-resp) and
    a detailed sub-group overlap matrix heatmap.
  - **Pooled** summary across all sessions.

Usage
-----
    python scripts/analysis/behavior/hmm_neural_group_overlap.py \\
        --tf-dir   FIGURES/tf \\
        --lick-dir FIGURES/lick/BG_046 \\
        --pkl-dir  data/pkls/BG_046 \\
        --manifest data/BG_046_staging_manifest_v2.csv \\
        --out      FIGURES/behavior/BG_046/group_overlap

    # Single session:
    python scripts/analysis/behavior/hmm_neural_group_overlap.py \\
        ... --session 03072025
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from visdetect.analysis.config import load_staging_manifest
from visdetect.core.session import load_session
from visdetect.analysis.constants import TF_FAST_THRESH_LOG2, TF_SLOW_THRESH_LOG2


# =====================================================================
# Classification loaders
# =====================================================================

def _quality_fr_ids(session, min_fr: float = 1.0) -> set:
    """Return cluster IDs passing quality + min-FR gates."""
    quality_ids = set(
        session.good_and_stable_ids
        or session.good_cluster_ids
        or [c.cluster_id for c in session.clusters]
    )
    fr_ok: set = set()
    for c in session.clusters:
        cid = c.cluster_id
        if cid not in quality_ids:
            continue
        st = c.spike_times
        if st is None or len(st) == 0:
            continue
        dur = float(st[-1] - st[0])
        if dur < 1e-6:
            continue
        if len(st) / dur >= min_fr:
            fr_ok.add(cid)
    return fr_ok


def _load_tf_groups(tf_csv: str, fr_ok: set,
                    z_thresh: float = 3.0) -> Dict[str, set]:
    """Classify units by TF responsiveness from CSV.

    Returns dict with keys: TF-excited, TF-suppressed, Non-TF.
    """
    groups: Dict[str, set] = {
        "TF-excited": set(), "TF-suppressed": set(), "Non-TF": set(),
    }
    if not Path(tf_csv).exists():
        groups["Non-TF"] = set(fr_ok)
        return groups

    df = pd.read_csv(tf_csv)
    classified = set()
    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        if cid not in fr_ok:
            continue
        z_max_f = abs(row.get("z_max_fast", 0.0))
        z_min_f = abs(row.get("z_min_fast", 0.0))
        z_max_s = abs(row.get("z_max_slow", 0.0))
        z_min_s = abs(row.get("z_min_slow", 0.0))

        fast_resp = z_max_f >= z_thresh or z_min_f >= z_thresh
        slow_resp = z_max_s >= z_thresh or z_min_s >= z_thresh

        if fast_resp or slow_resp:
            peak = max(z_max_f, z_max_s)
            trough = max(z_min_f, z_min_s)
            if peak >= trough:
                groups["TF-excited"].add(cid)
            else:
                groups["TF-suppressed"].add(cid)
        else:
            groups["Non-TF"].add(cid)
        classified.add(cid)

    groups["Non-TF"] |= (fr_ok - classified)
    return groups


def _load_lick_groups(lick_csv: str, fr_ok: set) -> Dict[str, set]:
    """Classify units by lick responsiveness from CSV.

    Returns dict with keys: Lick-excited, Lick-inhibited, Non-lick.
    """
    groups: Dict[str, set] = {
        "Lick-excited": set(), "Lick-inhibited": set(), "Non-lick": set(),
    }
    if not Path(lick_csv).exists():
        groups["Non-lick"] = set(fr_ok)
        return groups

    df = pd.read_csv(lick_csv)
    classified = set()
    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        if cid not in fr_ok:
            continue
        sig = bool(row.get("is_significant", False))
        delta = float(row.get("delta_mean", 0.0))
        if sig and delta > 0:
            groups["Lick-excited"].add(cid)
        elif sig and delta < 0:
            groups["Lick-inhibited"].add(cid)
        else:
            groups["Non-lick"].add(cid)
        classified.add(cid)

    groups["Non-lick"] |= (fr_ok - classified)
    return groups


# =====================================================================
# Venn diagram (2-set: TF-responsive vs Lick-responsive)
# =====================================================================

def _draw_venn2(
    tf_resp: set, lick_resp: set, all_units: set,
    ax: plt.Axes, title: str = "",
) -> None:
    """Draw a 2-set proportional-ish Venn with matplotlib patches."""
    only_tf = tf_resp - lick_resp
    only_lick = lick_resp - tf_resp
    both = tf_resp & lick_resp
    neither = all_units - tf_resp - lick_resp

    # Circle positions and size
    r = 0.35
    cx_tf, cy = -0.18, 0.0
    cx_lk = 0.18

    c1 = mpatches.Circle((cx_tf, cy), r, alpha=0.35, color="#d62728",
                          label="TF-responsive")
    c2 = mpatches.Circle((cx_lk, cy), r, alpha=0.35, color="#ff7f0e",
                          label="Lick-responsive")
    ax.add_patch(c1)
    ax.add_patch(c2)

    # Labels inside circles
    fs = 12
    ax.text(cx_tf - 0.12, cy, str(len(only_tf)), ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#d62728")
    ax.text(cx_lk + 0.12, cy, str(len(only_lick)), ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#ff7f0e")
    ax.text((cx_tf + cx_lk) / 2, cy, str(len(both)), ha="center",
            va="center", fontsize=fs, fontweight="bold", color="#333333")
    ax.text(0.0, -0.50, f"Neither: {len(neither)}", ha="center",
            va="center", fontsize=10, color="0.4")

    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.65, 0.55)
    ax.set_aspect("equal")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.7, ncol=2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


# =====================================================================
# Overlap matrix heatmap
# =====================================================================

def _draw_overlap_matrix(
    tf_groups: Dict[str, set],
    lick_groups: Dict[str, set],
    ax: plt.Axes,
    title: str = "",
) -> None:
    """Draw a heatmap showing pairwise overlap counts."""
    tf_labels = [k for k in tf_groups if tf_groups[k]]
    lick_labels = [k for k in lick_groups if lick_groups[k]]

    if not tf_labels or not lick_labels:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="0.5")
        ax.set_title(title, fontsize=11)
        return

    matrix = np.zeros((len(tf_labels), len(lick_labels)), dtype=int)
    for ri, tk in enumerate(tf_labels):
        for ci, lk in enumerate(lick_labels):
            matrix[ri, ci] = len(tf_groups[tk] & lick_groups[lk])

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=max(1, matrix.max()))
    ax.set_xticks(range(len(lick_labels)))
    ax.set_xticklabels(lick_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(tf_labels)))
    ax.set_yticklabels(tf_labels, fontsize=9)
    ax.set_xlabel("Lick groups", fontsize=10)
    ax.set_ylabel("TF groups", fontsize=10)

    # Annotate cells
    for ri in range(len(tf_labels)):
        for ci in range(len(lick_labels)):
            val = matrix[ri, ci]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(ci, ri, str(val), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_title(title, fontsize=11, fontweight="bold")


# =====================================================================
# Per-session figure
# =====================================================================

def _make_session_figure(
    tf_groups: Dict[str, set],
    lick_groups: Dict[str, set],
    all_units: set,
    session_name: str,
    out_dir: Path,
) -> str:
    """Produce a 2-panel figure: Venn + overlap matrix for one session."""
    fig, (ax_venn, ax_mat) = plt.subplots(1, 2, figsize=(11, 5))

    tf_resp = tf_groups.get("TF-excited", set()) | tf_groups.get(
        "TF-suppressed", set()
    )
    lick_resp = lick_groups.get("Lick-excited", set()) | lick_groups.get(
        "Lick-inhibited", set()
    )

    _draw_venn2(tf_resp, lick_resp, all_units, ax_venn,
                title=f"TF vs Lick responsive — {session_name}")
    _draw_overlap_matrix(tf_groups, lick_groups, ax_mat,
                         title="Sub-group overlap")

    fig.tight_layout()
    out_path = out_dir / session_name
    out_path.mkdir(parents=True, exist_ok=True)
    fp = out_path / "group_overlap.png"
    fig.savefig(str(fp), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(fp)


# =====================================================================
# Pooled summary figure
# =====================================================================

def _make_pooled_figure(
    pooled_tf: Dict[str, set],
    pooled_lick: Dict[str, set],
    pooled_all: set,
    out_dir: Path,
    n_sessions: int,
) -> str:
    """Pooled overlap figure across all sessions."""
    fig, (ax_venn, ax_mat) = plt.subplots(1, 2, figsize=(12, 5.5))

    tf_resp = pooled_tf.get("TF-excited", set()) | pooled_tf.get(
        "TF-suppressed", set()
    )
    lick_resp = pooled_lick.get("Lick-excited", set()) | pooled_lick.get(
        "Lick-inhibited", set()
    )

    _draw_venn2(tf_resp, lick_resp, pooled_all, ax_venn,
                title=f"TF vs Lick responsive — pooled ({n_sessions} sessions)")
    _draw_overlap_matrix(pooled_tf, pooled_lick, ax_mat,
                         title="Sub-group overlap (pooled)")

    fig.tight_layout()
    fp = out_dir / "group_overlap_pooled.png"
    fig.savefig(str(fp), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(fp)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize overlap between TF-responsive and "
                    "lick-responsive neuron groups.",
    )
    parser.add_argument("--tf-dir", required=True,
                        help="Root TF screening directory (e.g. FIGURES/tf).")
    parser.add_argument("--lick-dir", required=True,
                        help="Root lick analysis directory "
                             "(e.g. FIGURES/lick/BG_046).")
    parser.add_argument("--pkl-dir", required=True,
                        help="Directory with session pkl files "
                             "(needed for quality/FR gating).")
    parser.add_argument("--manifest", default=None,
                        help="Staging manifest CSV (session list).")
    parser.add_argument("--session", default=None,
                        help="Single session name (optional).")
    parser.add_argument("--out", default="FIGURES/behavior/group_overlap",
                        help="Output directory.")
    parser.add_argument("--exclude-qc-fail", action="store_true",
                        help="DEPRECATED: SESSION_FILTER handles QC.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Bypass SESSION_FILTER.")
    parser.add_argument("--z-thresh-tf", type=float, default=3.0,
                        help="Z-score threshold for TF responsiveness.")
    parser.add_argument("--min-fr", type=float, default=1.0,
                        help="Minimum firing rate (Hz).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style(context="talk")

    pkl_dir = Path(args.pkl_dir)
    tf_dir = Path(args.tf_dir)
    lick_dir = Path(args.lick_dir)
    subject = "BG_046"

    # Determine sessions
    if args.session:
        session_names = [args.session]
    elif args.manifest or True:  # Always load manifest for session filtering
        manifest = load_staging_manifest(
            manifest_path=args.manifest,
            apply_filter=not getattr(args, 'no_filter', False),
        )
        session_names = manifest["session_name"].tolist()
    else:
        # Fall back to sessions present in both TF and lick dirs
        tf_sessions = {p.name.replace(f"{subject}_", "")
                       for p in tf_dir.iterdir() if p.is_dir()}
        lick_sessions = {p.name for p in lick_dir.iterdir() if p.is_dir()}
        session_names = sorted(tf_sessions | lick_sessions)

    # Pooled accumulators (use (session_name, cluster_id) as unique key
    # to avoid collisions across sessions)
    pooled_tf: Dict[str, set] = defaultdict(set)
    pooled_lick: Dict[str, set] = defaultdict(set)
    pooled_all: set = set()

    fig_paths: List[str] = []
    n_ok = 0

    for sname in session_names:
        # Find pkl
        candidates = list(pkl_dir.glob(f"*{sname}*.pkl"))
        if not candidates:
            print(f"  SKIP {sname}: pkl not found")
            continue

        # Find TF CSV
        tf_csv: Optional[str] = None
        for d in [tf_dir / f"{subject}_{sname}", tf_dir / sname]:
            p = d / "tf_pulse_grid_both.csv"
            if p.exists():
                tf_csv = str(p)
                break

        # Find Lick CSV
        lick_csv: Optional[str] = None
        for d in [lick_dir / f"{subject}_{sname}", lick_dir / sname]:
            p = d / "lick_responsiveness.csv"
            if p.exists():
                lick_csv = str(p)
                break

        if tf_csv is None and lick_csv is None:
            print(f"  SKIP {sname}: no TF or lick CSV found")
            continue

        # Load session for quality gating
        try:
            session = load_session(str(candidates[0]))
        except Exception as exc:
            print(f"  ERR {sname}: {exc}")
            continue

        fr_ok = _quality_fr_ids(session, min_fr=args.min_fr)
        if not fr_ok:
            print(f"  SKIP {sname}: no quality units")
            continue

        # Classify
        tf_groups = _load_tf_groups(
            tf_csv or "", fr_ok, z_thresh=args.z_thresh_tf,
        )
        lick_groups = _load_lick_groups(lick_csv or "", fr_ok)

        # Accumulate pooled (prefix with session to keep unique)
        for gk, ids in tf_groups.items():
            pooled_tf[gk] |= {(sname, cid) for cid in ids}
        for gk, ids in lick_groups.items():
            pooled_lick[gk] |= {(sname, cid) for cid in ids}
        pooled_all |= {(sname, cid) for cid in fr_ok}

        # Per-session figure
        fp = _make_session_figure(
            tf_groups, lick_groups, fr_ok, sname, out_dir,
        )
        if fp:
            fig_paths.append(fp)
            n_ok += 1
            n_tf = sum(len(v) for k, v in tf_groups.items()
                       if k != "Non-TF")
            n_lk = sum(len(v) for k, v in lick_groups.items()
                       if k != "Non-lick")
            print(f"  [OK] {sname}: {len(fr_ok)} units, "
                  f"{n_tf} TF-resp, {n_lk} Lick-resp")

    # Pooled figure
    if n_ok > 1:
        fp = _make_pooled_figure(
            pooled_tf, pooled_lick, pooled_all, out_dir, n_ok,
        )
        if fp:
            fig_paths.append(fp)

    print(f"\n{'=' * 60}")
    print(f"DONE: {n_ok} session figures + "
          f"{'1 pooled' if n_ok > 1 else 'no pooled'}")
    print(f"Output: {out_dir}")

    # Summary CSV
    summary = []
    for sname in session_names:
        # Quick count from pooled
        t_ids = {cid for s, cid in pooled_all if s == sname}
        tf_r = sum(1 for s, cid in
                   (pooled_tf.get("TF-excited", set()) |
                    pooled_tf.get("TF-suppressed", set()))
                   if s == sname)
        lk_r = sum(1 for s, cid in
                   (pooled_lick.get("Lick-excited", set()) |
                    pooled_lick.get("Lick-inhibited", set()))
                   if s == sname)
        summary.append({
            "session_name": sname,
            "n_units": len(t_ids),
            "n_tf_responsive": tf_r,
            "n_lick_responsive": lk_r,
        })
    pd.DataFrame(summary).to_csv(
        out_dir / "overlap_summary.csv", index=False,
    )


if __name__ == "__main__":
    main()
