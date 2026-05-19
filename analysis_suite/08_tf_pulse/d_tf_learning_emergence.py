"""Fig38: TF responsiveness emergence — tracking across learning stages.

Loads pre-computed NPZ caches (no session pickles needed).
Tracks how TF responsiveness fraction, amplitude, and latency evolve
across learning stages (Naive → Learning → Expert).

Can also consume the cached tf_responsiveness.csv from script (a) when
available.

Produces fig38_tf_learning_emergence.png:
  - Panel A: TF-responsive fraction per session across learning
  - Panel B: Mean z-score amplitude vs. session index
  - Panel C: Fraction responsive by stage (bar chart with stats)
  - Panel D: Amplitude trajectory by cell type across learning
"""

import argparse
import os
import sys


import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS, CACHE_DIR,
    DEFAULT_Z_THRESH_TF,
)
from visdetect.suite.loader import load_staging_manifest, load_waveform_labels, load_tf_traces_npz
from visdetect.suite.plotting import setup_style, save_figure

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

setup_style()

Z_THRESH = DEFAULT_Z_THRESH_TF


def main():
    parser = argparse.ArgumentParser(description="TF learning emergence")
    parser.add_argument("--n-workers", type=int, default=1, help="(unused)")
    args = parser.parse_args()

    print("=" * 70)
    print("[08d] TF Responsiveness Emergence Across Learning  [from NPZ cache]")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  Sessions: {len(manifest)}")

    ct_lookup = {}
    try:
        wf = load_waveform_labels()
        for _, r in wf.iterrows():
            ct_lookup[(int(r["session_name"]), int(r["cluster_id"]))] = r["cell_type"]
    except Exception:
        pass

    # ── Try loading cached CSV from script (a) ──────────────────────
    csv_path = os.path.join(CACHE_DIR, "tf_responsiveness.csv")
    if os.path.isfile(csv_path):
        print(f"  Loading cached tf_responsiveness.csv")
        df = pd.read_csv(csv_path)
        df["session_name"] = df["session_name"].astype(int)
    else:
        # Build from NPZ
        print("  Building unit table from NPZ cache…")
        records = []
        for _, row in manifest.iterrows():
            sname = int(row["session_name"])
            npz = load_tf_traces_npz(sname)
            if npz is None:
                continue
            for i, cid in enumerate(npz["cluster_ids"]):
                cid = int(cid)
                z_abs = max(
                    abs(npz["z_max_fast"][i]), abs(npz["z_min_fast"][i]),
                    abs(npz["z_max_slow"][i]), abs(npz["z_min_slow"][i]),
                )
                records.append({
                    "session_name": sname, "cluster_id": cid,
                    "stage": row["stage"], "session_idx": row["session_idx"],
                    "cell_type": ct_lookup.get((sname, cid), "Unknown"),
                    "z_abs_max": z_abs,
                    "is_tf_responsive": z_abs >= Z_THRESH,
                    "z_max_fast": float(npz["z_max_fast"][i]),
                    "z_min_fast": float(npz["z_min_fast"][i]),
                })
        df = pd.DataFrame(records)

    if len(df) == 0:
        print("  No data. Exiting.")
        return
    print(f"  Total units: {len(df)}")

    # Ensure session_idx and stage are available
    if "session_idx" not in df.columns or df["session_idx"].isna().all():
        man_map = {int(r["session_name"]): (r["stage"], r["session_idx"])
                   for _, r in manifest.iterrows()}
        df["stage"] = df["session_name"].map(lambda s: man_map.get(s, ("Unknown", -1))[0])
        df["session_idx"] = df["session_name"].map(lambda s: man_map.get(s, ("Unknown", -1))[1])

    # Compute amplitude if not present
    if "z_abs_max" not in df.columns:
        df["z_abs_max"] = df[["z_max_fast","z_min_fast","z_max_slow","z_min_slow"]].abs().max(axis=1)

    # ── Per-session summaries ─────────────────────────────────────
    sess_df = df.groupby("session_name").agg(
        n_units=("cluster_id", "count"),
        n_resp=("is_tf_responsive", "sum"),
        mean_z_abs=("z_abs_max", "mean"),
        stage=("stage", "first"),
        session_idx=("session_idx", "first"),
    ).reset_index()
    sess_df["frac_resp"] = sess_df["n_resp"] / sess_df["n_units"]
    sess_df = sess_df.sort_values("session_idx")

    # ── Create figure ─────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Fraction responsive per session ──────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for s in STAGE_ORDER:
        sub = sess_df[sess_df["stage"] == s]
        if len(sub):
            ax_a.scatter(sub["session_idx"], sub["frac_resp"] * 100,
                        color=STAGE_COLORS[s], s=60, edgecolors="black",
                        linewidth=0.5, label=s, zorder=5)
    # Trend line
    sidxs = sess_df["session_idx"].values.astype(float)
    fracs = sess_df["frac_resp"].values * 100
    if len(sidxs) >= 3:
        z = np.polyfit(sidxs, fracs, 2)
        xfit = np.linspace(sidxs.min(), sidxs.max(), 100)
        ax_a.plot(xfit, np.polyval(z, xfit), "k--", linewidth=1, alpha=0.5)
    ax_a.set_xlabel("Session index (chronological)")
    ax_a.set_ylabel("% TF-responsive")
    ax_a.set_title("A – TF responsiveness fraction across learning")
    ax_a.legend(fontsize=8)

    # ── Panel B: Mean z-score amplitude across sessions ───────────
    ax_b = fig.add_subplot(gs[0, 1])
    for s in STAGE_ORDER:
        sub = sess_df[sess_df["stage"] == s]
        if len(sub):
            ax_b.scatter(sub["session_idx"], sub["mean_z_abs"],
                        color=STAGE_COLORS[s], s=60, edgecolors="black",
                        linewidth=0.5, label=s, zorder=5)
    ax_b.axhline(Z_THRESH, color="red", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_b.set_xlabel("Session index (chronological)")
    ax_b.set_ylabel("Mean |z-score|")
    ax_b.set_title("B – Mean TF amplitude across learning")
    ax_b.legend(fontsize=8)

    # ── Panel C: Stage bar chart with chi-squared ─────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    stages = [s for s in STAGE_ORDER if s in df["stage"].values]
    x = np.arange(len(stages))
    fracs_stage = []
    ns = []
    for s in stages:
        sub = df[df["stage"] == s]
        frac = sub["is_tf_responsive"].mean() * 100 if len(sub) else 0
        fracs_stage.append(frac)
        ns.append(len(sub))
    bars = ax_c.bar(x, fracs_stage,
                    color=[STAGE_COLORS[s] for s in stages],
                    edgecolor="black", linewidth=0.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(stages, ns)])
    ax_c.set_ylabel("% TF-responsive")
    ax_c.set_title("C – TF responsiveness by learning stage")
    if len(stages) >= 2:
        cont = [[int(df[df["stage"]==s]["is_tf_responsive"].sum()),
                  len(df[df["stage"]==s]) - int(df[df["stage"]==s]["is_tf_responsive"].sum())]
                 for s in stages]
        if all(sum(r) > 0 for r in cont):
            try:
                _, p, _, _ = chi2_contingency(cont)
                ax_c.text(0.5, 0.95, f"χ² test: p={p:.2e}",
                         transform=ax_c.transAxes, fontsize=9, ha="center", va="top")
            except Exception:
                pass

    # ── Panel D: Amplitude by cell type across learning ───────────
    ax_d = fig.add_subplot(gs[1, 1])
    cell_types = sorted([c for c in df["cell_type"].unique() if c != "Unknown"])
    resp = df[df["is_tf_responsive"]]
    for ct in cell_types:
        ct_sess = resp[resp["cell_type"] == ct].groupby("session_idx")["z_abs_max"].mean()
        if len(ct_sess) >= 2:
            ax_d.plot(ct_sess.index, ct_sess.values,
                     "o-", color=CELLTYPE_COLORS.get(ct, "#999"),
                     markersize=5, linewidth=1.2, label=ct, alpha=0.8)
    ax_d.set_xlabel("Session index")
    ax_d.set_ylabel("Mean |z-score| (responsive units)")
    ax_d.set_title("D – TF amplitude trajectory by cell type")
    if cell_types:
        ax_d.legend(fontsize=8)

    fig.suptitle(
        "TF Pulse Responsiveness Across Learning Stages\n"
        "(Naive → Learning → Expert)",
        fontsize=13, fontweight="bold", y=0.98)
    save_figure(fig, "fig38_tf_learning_emergence", "08_tf_pulse")
    print("\n  Saved fig38_tf_learning_emergence.png")

    # Summary
    print("\n  Summary:")
    for s in stages:
        sub = df[df["stage"] == s]
        nr = int(sub["is_tf_responsive"].sum())
        print(f"    {s}: {nr}/{len(sub)} = {100*nr/len(sub):.1f}% responsive, "
              f"mean |z|={sub['z_abs_max'].mean():.2f}")


if __name__ == "__main__":
    main()
