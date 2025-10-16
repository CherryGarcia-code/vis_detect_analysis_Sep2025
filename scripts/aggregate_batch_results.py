from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

REPO = Path(__file__).resolve().parents[1]
BATCH_DIR = REPO / "notebooks" / "batch_outputs"
OUT_DIR = REPO / "notebooks" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_sessions(batch_dir: Path):
    # Expect structure: batch_outputs/<pkl_stem>/<session_name>/
    for p in sorted(batch_dir.iterdir()):
        if p.is_dir():
            for child in p.iterdir():
                if child.is_dir():
                    yield p.stem, child.name, child


def load_stats(session_dir: Path):
    change_csv = session_dir / "demo_pipeline_changeON_group_stats.csv"
    baseline_csv = session_dir / "demo_pipeline_baseline_group_stats.csv"
    change_trials_csv = session_dir / "demo_pipeline_changeON_scalar_pre0_by_trial.csv"
    baseline_trials_csv = (
        session_dir / "demo_pipeline_baseline_scalar_pre0_by_trial.csv"
    )
    if not change_csv.exists() or not baseline_csv.exists():
        return None
    try:
        ch = pd.read_csv(change_csv).set_index("group")
    except Exception:
        ch = None
    try:
        ba = pd.read_csv(baseline_csv).set_index("group")
    except Exception:
        ba = None
    # Also read per-trial scalars if present
    try:
        ct = pd.read_csv(change_trials_csv) if change_trials_csv.exists() else None
    except Exception:
        ct = None
    try:
        bt = pd.read_csv(baseline_trials_csv) if baseline_trials_csv.exists() else None
    except Exception:
        bt = None
    return ch, ba, ct, bt


def aggregate():
    rows = []
    import re

    for pkl_stem, session_name, session_dir in find_sessions(BATCH_DIR):
        stats = load_stats(session_dir)
        if stats is None:
            continue
        ch, ba, ct, bt = stats

        # Parse session date from session_name (formats: ddmmyy or ddmmyyyy)
        def parse_session_date(name: str):
            m = re.search(r"(\d{8}|\d{6})", name)
            if not m:
                logging.warning('Could not find date token in "%s"', name)
                return pd.NaT
            token = m.group(0)
            try:
                if len(token) == 8:
                    dt = datetime.datetime.strptime(token, "%d%m%Y").date()
                    return dt
                else:  # len == 6, ddmmyy -> convert yy to 2000+ by default
                    day = int(token[0:2])
                    month = int(token[2:4])
                    yy = int(token[4:6])
                    year = 2000 + yy if yy < 100 else yy
                    # Basic validation
                    return datetime.date(year, month, day)
            except Exception:
                logging.exception(
                    'Failed to parse date token "%s" from "%s"', token, name
                )
                return pd.NaT

        session_date = parse_session_date(session_name)

        # Harvest metrics with safe lookups
        def safe_get(df, key, col="mean"):
            try:
                return float(df.loc[key, col])
            except Exception:
                return float("nan")

        # Try to infer counts from group stats
        n_hit = (
            int(ch.loc["Hit", "n"])
            if (ch is not None and "Hit" in ch.index and "n" in ch.columns)
            else 0
        )
        n_miss = (
            int(ch.loc["Miss", "n"])
            if (ch is not None and "Miss" in ch.index and "n" in ch.columns)
            else 0
        )

        # Prefer per-trial pre-0 scalar CSVs when available for Change_ON
        if ct is not None:
            try:
                mean_change_hit = float(
                    ct.loc[ct["outcome"] == "Hit", "scalar_pre0"].mean()
                )
            except Exception:
                mean_change_hit = safe_get(ch, "Hit")
            try:
                mean_change_miss = float(
                    ct.loc[ct["outcome"] == "Miss", "scalar_pre0"].mean()
                )
            except Exception:
                mean_change_miss = safe_get(ch, "Miss")
        else:
            mean_change_hit = safe_get(ch, "Hit")
            mean_change_miss = safe_get(ch, "Miss")

        # For baseline, prefer per-trial baseline scalar for Hit; FA>3s typically not encoded in trial CSVs, so fall back to group stat
        if bt is not None:
            try:
                mean_baseline_hit = float(
                    bt.loc[bt["outcome"] == "Hit", "scalar_pre0"].mean()
                )
            except Exception:
                mean_baseline_hit = safe_get(ba, "Hit")
        else:
            mean_baseline_hit = safe_get(ba, "Hit")

        mean_baseline_FAgt3 = safe_get(ba, "FA>3s")

        rows.append(
            {
                "pkl_stem": pkl_stem,
                "session_name": session_name,
                "session_date": session_date,
                "n_hit": n_hit,
                "n_miss": n_miss,
                "mean_change_hit": mean_change_hit,
                "mean_change_miss": mean_change_miss,
                "mean_baseline_hit": mean_baseline_hit,
                "mean_baseline_FA_gt3": mean_baseline_FAgt3,
            }
        )

    df = pd.DataFrame(rows)

    # Ensure session_date is datetime-like and sort by it (NaT will be last)
    if "session_date" in df.columns:
        df["session_date"] = pd.to_datetime(df["session_date"])
        df = df.sort_values(["session_date", "session_name"])

    df_path = OUT_DIR / "aggregated_session_summary.csv"
    df.to_csv(df_path, index=False)

    # Plot 1: mean baseline Hit by session sorted by session_date
    df_plot = df.copy()
    # Use string labels but position ticks explicitly to avoid warnings
    labels = df_plot["session_name"].astype(str).tolist()
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, df_plot["mean_baseline_hit"], marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="small")
    ax.set_ylabel("Mean baseline CD (Hit)")
    ax.set_title("Session-sorted mean baseline CD (Hit)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mean_baseline_hit_by_session.png", dpi=150)
    plt.close(fig)

    # Plot 2: Change_ON scalar difference (Hit - Miss), sorted by date
    df_plot["change_diff"] = df_plot["mean_change_hit"] - df_plot["mean_change_miss"]
    labels2 = df_plot["session_name"].astype(str).tolist()
    x2 = list(range(len(labels2)))
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(x2, df_plot["change_diff"])
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, rotation=45, ha="right", fontsize="small")
    ax2.set_ylabel("Change_ON mean (Hit - Miss)")
    ax2.set_title("Session-sorted Change_ON scalar difference")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "change_on_diff_by_session.png", dpi=150)
    plt.close(fig2)

    print("Wrote aggregated CSV and plots to", OUT_DIR)


if __name__ == "__main__":
    aggregate()
