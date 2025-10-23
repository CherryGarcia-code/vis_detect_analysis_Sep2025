"""
Aggregate per-unit modulation indices across sessions.

- Scans png_output/**/ for 'modulation_index_by_cluster.csv'
- For each session, optionally joins with 'qc_filtered_typical_good.csv' in the same folder to flag good units
- Adds subject + session_name inferred from folder name (expects pattern 'demo_single_unit_<SESSION>' or similar)
- Writes:
  * table_output/modulation_across_sessions.csv (long format: one row per cluster per session)
  * table_output/modulation_population_stats_by_session.csv (summary stats per session, overall and for good units only)
- Optionally saves a simple per-day summary plot under png_output/learning_analysis
"""
from __future__ import annotations
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
PNG_ROOT = REPO_ROOT / "png_output"
OUT_TABLE_DIR = REPO_ROOT / "table_output"
OUT_PNG_DIR = PNG_ROOT / "learning_analysis"

OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)


def infer_subject_session(folder: Path) -> tuple[str|None, str|None, str]:
    """Infer subject and session strings from a folder name.
    Returns (subject, session_date, raw_label).
    """
    name = folder.name
    # common pattern: demo_single_unit_BG_046_15082025
    m = re.search(r"([A-Za-z]+_\d+)_(\d{6,8})", name)
    if m:
        subject = m.group(1)
        sess = m.group(2)
        return subject, sess, name
    # fallback: try to extract trailing date-like digits
    m2 = re.search(r"(\d{6,8})", name)
    return None, (m2.group(1) if m2 else None), name


def find_session_dirs(root: Path) -> list[Path]:
    session_dirs = []
    if not root.exists():
        return session_dirs
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("demo_single_unit_"):
            session_dirs.append(child)
    return session_dirs


def load_modulation_csvs(session_dir: Path) -> pd.DataFrame | None:
    mi_path = session_dir / "modulation_index_by_cluster.csv"
    if not mi_path.exists():
        return None
    df = pd.read_csv(mi_path)
    # normalize columns
    needed = {"cluster_id", "hit_mean_fr", "miss_mean_fr", "mi", "n_hit", "n_miss"}
    missing = needed - set(df.columns)
    if missing:
        # best effort: return None if not compatible
        return None
    return df


def load_good_units(session_dir: Path) -> set[int]:
    good_csv = session_dir / "qc_filtered_typical_good.csv"
    if not good_csv.exists():
        return set()
    g = pd.read_csv(good_csv)
    if "cluster_id" not in g.columns:
        return set()
    return set(int(x) for x in g["cluster_id"].tolist())


def aggregate():
    rows = []
    session_stats = []
    for sess_dir in find_session_dirs(PNG_ROOT):
        subject, sess_date, label = infer_subject_session(sess_dir)
        mi_df = load_modulation_csvs(sess_dir)
        if mi_df is None or mi_df.empty:
            continue
        good_set = load_good_units(sess_dir)
        mi_df["is_good"] = mi_df["cluster_id"].astype(int).isin(good_set).astype(int)
        mi_df["subject"] = subject
        mi_df["session_date"] = sess_date
        mi_df["session_label"] = label
        rows.append(mi_df)
        # per-session stats
        def summarize(sub: pd.DataFrame, tag: str):
            if sub.empty:
                return
            session_stats.append({
                "session_label": label,
                "subject": subject,
                "session_date": sess_date,
                "scope": tag,
                "n_units": int(sub.shape[0]),
                "mi_mean": float(np.nanmean(sub["mi"])) if sub.shape[0] else np.nan,
                "mi_median": float(np.nanmedian(sub["mi"])) if sub.shape[0] else np.nan,
                "mi_std": float(np.nanstd(sub["mi"])) if sub.shape[0] else np.nan,
                "frac_mi_gt0": float(np.mean(sub["mi"] > 0.0)) if sub.shape[0] else np.nan,
                "frac_mi_gt0_2sd": float(np.mean(sub["mi"] > (sub["mi"].mean() + 2*sub["mi"].std()))) if sub.shape[0] else np.nan,
            })
        summarize(mi_df, "all")
        summarize(mi_df.loc[mi_df["is_good"] == 1], "good")

    if not rows:
        print("No modulation_index_by_cluster.csv files found under", PNG_ROOT)
        return None

    all_df = pd.concat(rows, ignore_index=True)
    out_long = OUT_TABLE_DIR / "modulation_across_sessions.csv"
    all_df.to_csv(out_long, index=False)

    stats_df = pd.DataFrame(session_stats)
    out_stats = OUT_TABLE_DIR / "modulation_population_stats_by_session.csv"
    stats_df.to_csv(out_stats, index=False)

    # quick plot: mi_mean over sessions (sorted by date where available)
    try:
        # create a sortable key
        def sort_key(row):
            d = row.get("session_date")
            try:
                return int(d)
            except Exception:
                return 0
        stats_sorted = stats_df.sort_values(by=["subject", "session_date", "scope"], key=None)
        # Separate scopes
        for scope in ("all", "good"):
            sub = stats_sorted.loc[stats_sorted["scope"] == scope]
            if sub.empty:
                continue
            # x labels
            labels = [f"{r['subject'] or ''}_{r['session_date'] or r['session_label']}" for _, r in sub.iterrows()]
            plt.figure(figsize=(max(6, len(labels)*0.6), 3))
            plt.plot(range(len(sub)), sub["mi_mean"].values, marker='o', label='mi_mean')
            plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.ylabel('Mean modulation index (Hit vs Miss)')
            plt.title(f'Modulation across sessions ({scope})')
            plt.tight_layout()
            plt.savefig(OUT_PNG_DIR / f"modulation_across_sessions_{scope}.png", dpi=130)
            plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    return {
        "long_csv": str(out_long),
        "stats_csv": str(out_stats),
        "plots": [str(OUT_PNG_DIR / f"modulation_across_sessions_{s}.png") for s in ("all","good")]
    }


if __name__ == "__main__":
    res = aggregate()
    if res:
        print("Wrote:", res)
    else:
        print("Aggregation produced no outputs.")
