"""QC visualizations for the stitch-across-windows output.

Produces a multi-panel figure assessing:
  1. Unit yield per session × shank heatmap
  2. Session-span distribution (how many sessions each unit appears in)
  3. Spike count distribution per unit (log scale)
  4. Match quality: cluster-match counts per window pair
  5. Shank composition pie chart
  6. Longitudinal tracking: units tracked across ≥N sessions

Usage:
    python scripts/pipelines/concat_sort/qc_stitch_output.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

FINAL_OUTPUT = Path(
    "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046"
    "/concat_sort/final_output"
)
FIG_DIR = Path(__file__).resolve().parents[3] / "FIGURES" / "stitch_qc"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    reg = pd.read_csv(FINAL_OUTPUT / "global_registry.csv")
    can = pd.read_csv(FINAL_OUTPUT / "global_registry_canonical.csv")
    print(f"Full registry: {reg.shape[0]:,} rows, {reg.global_uid.nunique():,} UIDs")
    print(f"Canonical:     {can.shape[0]:,} rows, {can.global_uid.nunique():,} UIDs")
    print(f"Sessions: {can.session.nunique()}, Shanks: {sorted(can.shank_id.unique())}")
    return reg, can


def sort_sessions_chronologically(sessions):
    """Sort session names like BG_046_DDMMYYYY by date."""
    from datetime import datetime
    def _parse(s):
        parts = s.split("_")
        return datetime.strptime(parts[-1], "%d%m%Y")
    return sorted(sessions, key=_parse)


def plot_unit_yield_heatmap(can, ax):
    """Panel 1: units per session × shank."""
    sessions = sort_sessions_chronologically(can.session.unique())
    shanks = sorted(can.shank_id.unique())
    pivot = can.groupby(["session", "shank_id"]).global_uid.nunique().unstack(fill_value=0)
    pivot = pivot.reindex(index=sessions, columns=shanks, fill_value=0)

    im = ax.imshow(pivot.values.T, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(shanks)))
    ax.set_yticklabels([f"Shank {s}" for s in shanks])
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([s.split("_")[-1][:4] for s in sessions],
                       rotation=90, fontsize=6)
    ax.set_xlabel("Session")
    ax.set_title("Unit yield per session × shank")
    plt.colorbar(im, ax=ax, label="# units", shrink=0.8)

    # Print totals
    totals = pivot.sum(axis=1)
    print(f"\nUnit yield per session (total across shanks):")
    for s, t in zip(sessions, totals):
        print(f"  {s}: {int(t)}")
    print(f"  Mean: {totals.mean():.0f}, Min: {totals.min()}, Max: {totals.max()}")


def plot_session_span(can, ax):
    """Panel 2: how many sessions does each unit span?"""
    spans = can.groupby("global_uid").session.nunique()
    bins = np.arange(0.5, spans.max() + 1.5, 1)
    ax.hist(spans.values, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("# sessions unit appears in")
    ax.set_ylabel("# units")
    ax.set_title("Unit session-span distribution")
    ax.axvline(spans.median(), color="red", linestyle="--", label=f"median={spans.median():.0f}")
    ax.legend(fontsize=8)

    print(f"\nSession span stats:")
    print(f"  Mean: {spans.mean():.2f}")
    print(f"  Median: {spans.median():.0f}")
    print(f"  1 session only: {(spans==1).sum()} ({(spans==1).mean()*100:.1f}%)")
    print(f"  >=5 sessions: {(spans>=5).sum()} ({(spans>=5).mean()*100:.1f}%)")
    print(f"  >=10 sessions: {(spans>=10).sum()} ({(spans>=10).mean()*100:.1f}%)")
    print(f"  >=20 sessions: {(spans>=20).sum()} ({(spans>=20).mean()*100:.1f}%)")
    print(f"  All 38: {(spans==38).sum()}")


def plot_spike_count_dist(can, ax):
    """Panel 3: spike count distribution."""
    spk = can["n_spikes"].values
    spk = spk[spk > 0]
    ax.hist(np.log10(spk), bins=80, color="darkorange", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("log₁₀(spike count)")
    ax.set_ylabel("# unit×session entries")
    ax.set_title("Spike count distribution (canonical)")
    ax.axvline(np.log10(100), color="red", linestyle="--", alpha=0.7, label="100 spikes")
    ax.axvline(np.log10(1000), color="darkred", linestyle="--", alpha=0.7, label="1000 spikes")
    ax.legend(fontsize=7)

    print(f"\nSpike count stats (canonical entries):")
    print(f"  Total entries: {len(spk):,}")
    print(f"  Median spikes: {np.median(spk):.0f}")
    print(f"  <100 spikes: {(spk<100).sum()} ({(spk<100).mean()*100:.1f}%)")
    print(f"  >=1000 spikes: {(spk>=1000).sum()} ({(spk>=1000).mean()*100:.1f}%)")


def plot_match_counts(reg, ax):
    """Panel 4: cluster match counts per window pair (from full registry)."""
    # Count units that appear in multiple windows for the same session
    # This indicates stitch connectivity
    multi = reg.groupby(["global_uid", "session"]).window_idx.nunique()
    multi_dist = multi.value_counts().sort_index()
    ax.bar(multi_dist.index, multi_dist.values, color="seagreen", edgecolor="white")
    ax.set_xlabel("# windows contributing to same (unit, session)")
    ax.set_ylabel("Count")
    ax.set_title("Window multiplicity per (unit, session)")

    print(f"\nWindow multiplicity:")
    for k, v in multi_dist.items():
        print(f"  {k} windows: {v:,}")


def plot_shank_composition(can, ax):
    """Panel 5: units per shank."""
    per_shank = can.groupby("shank_id").global_uid.nunique()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    wedges, texts, autotexts = ax.pie(
        per_shank.values, labels=[f"Shank {s}" for s in per_shank.index],
        autopct="%1.0f%%", colors=colors[:len(per_shank)],
        textprops={"fontsize": 9}
    )
    ax.set_title("Units per shank")

    print(f"\nUnits per shank:")
    for s, n in per_shank.items():
        print(f"  Shank {s}: {n}")


def plot_longitudinal_tracking(can, ax):
    """Panel 6: cumulative units tracked across ≥N sessions."""
    spans = can.groupby("global_uid").session.nunique()
    thresholds = np.arange(1, spans.max() + 1)
    counts = [(spans >= t).sum() for t in thresholds]
    ax.plot(thresholds, counts, color="purple", linewidth=2)
    ax.fill_between(thresholds, counts, alpha=0.15, color="purple")
    ax.set_xlabel("Minimum # sessions")
    ax.set_ylabel("# units tracked across ≥N sessions")
    ax.set_title("Longitudinal tracking depth")
    ax.set_xlim(1, 38)

    # Mark key thresholds
    for t in [5, 10, 20, 38]:
        c = (spans >= t).sum()
        ax.annotate(f"{c}", xy=(t, c), fontsize=7,
                    ha="center", va="bottom", color="purple")


def main():
    reg, can = load_data()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Stitch QC — BG_046 concat sort (4-shank NP2.0)", fontsize=14, y=0.98)

    plot_unit_yield_heatmap(can, axes[0, 0])
    plot_session_span(can, axes[0, 1])
    plot_spike_count_dist(can, axes[0, 2])
    plot_match_counts(reg, axes[1, 0])
    plot_shank_composition(can, axes[1, 1])
    plot_longitudinal_tracking(can, axes[1, 2])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "stitch_qc_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}", flush=True)
    plt.close(fig)

    # === Additional detailed figure: per-shank session span ===
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
    fig2.suptitle("Session-span distribution by shank", fontsize=13)
    for i, shank in enumerate(sorted(can.shank_id.unique())):
        sub = can[can.shank_id == shank]
        spans = sub.groupby("global_uid").session.nunique()
        bins = np.arange(0.5, 39.5, 1)
        axes2[i].hist(spans.values, bins=bins, color=["#2196F3","#4CAF50","#FF9800","#F44336"][i],
                      edgecolor="white", linewidth=0.3)
        axes2[i].set_title(f"Shank {shank} ({spans.shape[0]} units)")
        axes2[i].set_xlabel("# sessions")
        med = spans.median()
        axes2[i].axvline(med, color="black", linestyle="--", alpha=0.5)
        axes2[i].text(med + 0.5, axes2[i].get_ylim()[1] * 0.9,
                      f"med={med:.0f}", fontsize=8)
    axes2[0].set_ylabel("# units")
    fig2.tight_layout()
    out2 = FIG_DIR / "stitch_qc_per_shank_spans.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out2}", flush=True)
    plt.close(fig2)

    # === Centrality distribution ===
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(can.centrality.values, bins=50, color="teal", edgecolor="white")
    ax3.set_xlabel("Centrality score")
    ax3.set_ylabel("# canonical entries")
    ax3.set_title("Centrality of chosen canonical windows")
    out3 = FIG_DIR / "stitch_qc_centrality.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out3}", flush=True)
    plt.close(fig3)


if __name__ == "__main__":
    main()
