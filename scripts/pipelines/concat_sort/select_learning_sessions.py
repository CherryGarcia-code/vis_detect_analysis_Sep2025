"""Select sessions for concatenated spike sorting with sliding windows.

Strategy: include ALL viable sessions in chronological order so that the
sliding-window approach has no artificial temporal gaps.  Only sessions that
are QC-failed or show near-zero engagement are excluded.

Usage:
    python scripts/pipelines/concat_sort/select_learning_sessions.py \
        --manifest data/BG_046_staging_manifest_v2.csv \
        --subject BG_046 \
        --out data/concat_sort/learning_session_selection.json

    # Optionally restrict to a maximum number (picks contiguous block with
    # the steepest learning):
    python scripts/pipelines/concat_sort/select_learning_sessions.py \
        --manifest data/BG_046_staging_manifest_v2.csv \
        --n-sessions 25 \
        --out data/concat_sort/learning_session_selection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── helpers ────────────────────────────────────────────────────────────
def _parse_date(date_str: str) -> datetime:
    """Parse DDMMYYYY date string to datetime.

    The manifest stores dates like 01072025 (= 1 July 2025).  When pandas
    reads the column as integer the leading zero is dropped (→ 1072025, 7
    chars).  We zero-pad to 8 digits before parsing to handle both cases.
    """
    s = str(date_str).strip().zfill(8)   # '1072025' → '01072025'
    return datetime.strptime(s, "%d%m%Y")


def _day_gaps(dates: pd.Series) -> pd.Series:
    """Return inter-session gap in days (NaN for the first session)."""
    return dates.diff().dt.days


# ── main ───────────────────────────────────────────────────────────────
def select_sessions(
    manifest_path: str | Path,
    subject: str = "BG_046",
    n_sessions: int | None = None,
    min_hit_rate: float = 0.05,
    min_dprime: float = 0.1,
) -> dict:
    """Return a dict with the selected sessions and metadata.

    Selection strategy (contiguity-first):
      1. Remove QC-fail / missing-d' sessions.
      2. Remove truly disengaged sessions (hit_rate < min_hit_rate AND
         d' < min_dprime).
      3. Keep everything else in chronological order — Naive, Learning,
         and Expert — so the sliding window always has temporally close
         neighbours.
      4. If --n-sessions is given and fewer sessions remain, keep all.
         If more remain, pick the contiguous block of that length whose
         mean d' delta (learning slope) is steepest.
    """
    df = pd.read_csv(manifest_path, dtype={"session_name": str, "date": str})

    # 1. Filter QC-fail and missing d'
    df = df[df["qc_fail"] != True].copy()  # noqa: E712
    df = df[df["d_prime"].notna()].copy()

    # 2. Parse dates & sort chronologically
    df["date_dt"] = df["date"].apply(_parse_date)
    df = df.sort_values("date_dt").reset_index(drop=True)

    # 3. Remove truly disengaged sessions
    disengaged = (df["hit_rate"] < min_hit_rate) & (df["d_prime"] < min_dprime)
    n_dropped = int(disengaged.sum())
    dropped_names = df.loc[disengaged, "session_name"].tolist()
    df = df[~disengaged].reset_index(drop=True)

    # 4. Optionally trim to n_sessions via best contiguous block
    if n_sessions is not None and len(df) > n_sessions:
        # Find the contiguous block with the steepest learning (max Δd')
        dp = df["d_prime"].values.astype(float)
        best_start, best_slope = 0, -np.inf
        for i in range(len(dp) - n_sessions + 1):
            block = dp[i : i + n_sessions]
            # slope = mean first-difference (captures net improvement)
            slope = np.mean(np.diff(block))
            if slope > best_slope:
                best_slope = slope
                best_start = i
        df = df.iloc[best_start : best_start + n_sessions].reset_index(drop=True)

    # Compute day gaps for diagnostics
    day_gaps = _day_gaps(df["date_dt"])

    # Build output
    sessions_out = []
    for idx, row in df.iterrows():
        gap = int(day_gaps.iloc[idx]) if pd.notna(day_gaps.iloc[idx]) else None
        sessions_out.append(
            {
                "session_name": str(row["session_name"]),
                "date": row["date_dt"].strftime("%d%m%Y"),
                "date_iso": row["date_dt"].strftime("%Y-%m-%d"),
                "d_prime": round(float(row["d_prime"]), 4),
                "hit_rate": round(float(row["hit_rate"]), 4),
                "fa_rate": round(float(row["fa_rate"]), 4),
                "stage": str(row["stage"]),
                "days_since_prev": gap,
            }
        )

    result = {
        "subject": subject,
        "n_sessions_selected": len(sessions_out),
        "selection_criteria": {
            "strategy": "contiguity-first: all viable sessions in chronological order",
            "excluded_qc_fail": True,
            "excluded_disengaged": {
                "rule": f"hit_rate < {min_hit_rate} AND d_prime < {min_dprime}",
                "n_dropped": n_dropped,
                "dropped_sessions": dropped_names,
            },
            "n_sessions_cap": n_sessions,
        },
        "date_range": {
            "first": sessions_out[0]["date_iso"] if sessions_out else None,
            "last": sessions_out[-1]["date_iso"] if sessions_out else None,
        },
        "gap_stats": {
            "max_gap_days": int(day_gaps.max()) if len(day_gaps.dropna()) > 0 else None,
            "median_gap_days": round(float(day_gaps.median()), 1) if len(day_gaps.dropna()) > 0 else None,
        },
        "sessions": sessions_out,
    }
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Select sessions for concatenated sorting (contiguity-first)")
    p.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "BG_046_staging_manifest_v2.csv")
    p.add_argument("--subject", type=str, default="BG_046")
    p.add_argument("--n-sessions", type=int, default=None,
                   help="Max sessions to include (picks best contiguous block). Omit to include all viable.")
    p.add_argument("--min-hit-rate", type=float, default=0.05,
                   help="Sessions with hit_rate AND d' below thresholds are dropped.")
    p.add_argument("--min-dprime", type=float, default=0.1,
                   help="Sessions with hit_rate AND d' below thresholds are dropped.")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "concat_sort" / "learning_session_selection.json")
    args = p.parse_args(argv)

    result = select_sessions(args.manifest, args.subject, args.n_sessions,
                             args.min_hit_rate, args.min_dprime)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Selected {result['n_sessions_selected']} sessions → {args.out}")
    print(f"Date range: {result['date_range']['first']} → {result['date_range']['last']}")
    print(f"Max gap: {result['gap_stats']['max_gap_days']}d, median gap: {result['gap_stats']['median_gap_days']}d")
    if result["selection_criteria"]["excluded_disengaged"]["n_dropped"]:
        print(f"Dropped {result['selection_criteria']['excluded_disengaged']['n_dropped']} "
              f"disengaged session(s): {result['selection_criteria']['excluded_disengaged']['dropped_sessions']}")
    print()
    for s in result["sessions"]:
        gap_str = f"(+{s['days_since_prev']}d)" if s["days_since_prev"] is not None else "(start)"
        print(f"  {s['date_iso']} {gap_str:>7s}  d'={s['d_prime']:+.3f}  HR={s['hit_rate']:.2f}  FA={s['fa_rate']:.2f}  {s['stage']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
