"""08a – TF pulse responsiveness screening across medial striatum.

Loads pre-computed TF trace caches (NPZ files in data/cache/tf_traces/BG_046/).
No session pickles are loaded — all TF z-score data comes from the cache.

A unit is TF-responsive if its post-pulse z-score exceeds ±3.0 for either
fast or slow pulses.

Produces fig24_tf_responsiveness.png:
  - Panel A: Example TF-responsive unit PSTHs (fast + slow overlaid)
  - Panel B: Heatmap of fast TF pulse responses (all TF-responsive units)
  - Panel C: Distribution of z-scores; pie chart of responsive fraction
  - Panel D: TF responsiveness by cell type and learning stage
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS, CACHE_DIR,
    DEFAULT_Z_THRESH_TF,
)
from loader import load_staging_manifest, load_waveform_labels, load_tf_traces_npz
from plotting import setup_style, save_figure

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

setup_style()

Z_THRESH = DEFAULT_Z_THRESH_TF  # 3.0


def main():
    parser = argparse.ArgumentParser(description="TF pulse responsiveness screening")
    parser.add_argument("--n-workers", type=int, default=1, help="(unused — loads from cache)")
    args = parser.parse_args()

    print("=" * 70)
    print("[08a] TF Pulse Responsiveness Screening  [from NPZ cache]")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  Sessions: {len(manifest)}")

    # Load cell-type labels
    ct_lookup = {}
    try:
        wf = load_waveform_labels()
        for _, row in wf.iterrows():
            ct_lookup[(int(row["session_name"]), int(row["cluster_id"]))] = row["cell_type"]
    except (FileNotFoundError, KeyError):
        print("  Warning: cell-type labels not found")

    # Process sessions from NPZ cache
    all_units = []
    all_examples = []

    session_args = [
        (int(row["session_name"]), row["stage"], row["session_idx"])
        for _, row in manifest.iterrows()
    ]

    iterator = tqdm(session_args, desc="Sessions") if tqdm else session_args
    n_loaded = 0
    for sname, stage, sidx in iterator:
        npz = load_tf_traces_npz(sname)
        if npz is None:
            continue
        n_loaded += 1

        t_vec = npz["t_vec"]
        cluster_ids = npz["cluster_ids"]

        for i, cid in enumerate(cluster_ids):
            cid = int(cid)
            z_abs_max = max(
                abs(npz["z_max_fast"][i]), abs(npz["z_min_fast"][i]),
                abs(npz["z_max_slow"][i]), abs(npz["z_min_slow"][i]),
            )
            is_responsive = z_abs_max >= Z_THRESH
            ct = ct_lookup.get((sname, cid), "Unknown")

            all_units.append({
                "session_name": sname, "cluster_id": cid,
                "stage": stage, "session_idx": sidx, "cell_type": ct,
                "z_max_fast": float(npz["z_max_fast"][i]),
                "z_min_fast": float(npz["z_min_fast"][i]),
                "z_max_slow": float(npz["z_max_slow"][i]),
                "z_min_slow": float(npz["z_min_slow"][i]),
                "z_abs_max": z_abs_max,
                "is_tf_responsive": is_responsive,
            })

            all_examples.append((
                z_abs_max, npz["fast_z"][i].copy(), npz["slow_z"][i].copy(),
                t_vec.copy(), sname, cid, stage,
            ))

        if tqdm and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(
                units=len(all_units),
                resp=sum(1 for u in all_units if u["is_tf_responsive"]),
            )

    print(f"\n  NPZ files loaded: {n_loaded}/{len(session_args)}")

    df = pd.DataFrame(all_units)
    print(f"  Total units analyzed: {len(df)}")

    if len(df) == 0:
        print("  No data collected. Exiting.")
        return

    n_resp = int(df["is_tf_responsive"].sum())
    print(f"  TF-responsive: {n_resp}/{len(df)} ({100*n_resp/len(df):.1f}%)")

    # Cache results
    cache_path = os.path.join(CACHE_DIR, "tf_responsiveness.csv")
    df.to_csv(cache_path, index=False)
    print(f"  Cached: {cache_path}")

    # Sort examples for plotting
    all_examples.sort(key=lambda x: x[0], reverse=True)

    # ── Create figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Example TF-responsive unit PSTHs ─────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    n_ex = min(4, len(all_examples))
    c_fast = ["#E53935", "#C62828", "#B71C1C", "#880E4F"]
    c_slow = ["#1565C0", "#0D47A1", "#01579B", "#006064"]
    for i in range(n_ex):
        zv, fz, sz, tv, sn, ci, st = all_examples[i]
        ax_a.plot(tv*1000, fz, color=c_fast[i], linewidth=1.3,
                  label=f"Fast #{ci} (z={zv:.1f})" if i < 2 else None)
        ax_a.plot(tv*1000, sz, color=c_slow[i], linewidth=1.0,
                  linestyle="--", alpha=0.7,
                  label=f"Slow #{ci}" if i < 2 else None)
    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.axhline(Z_THRESH, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)
    ax_a.axhline(-Z_THRESH, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)
    ax_a.set_xlabel("Time from TF pulse (ms)")
    ax_a.set_ylabel("Z-score")
    ax_a.set_title("A – Example TF-responsive units")
    ax_a.legend(fontsize=7, loc="upper right")

    # ── Panel B: Heatmap ──────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    resp_ex = [ex for ex in all_examples if ex[0] >= Z_THRESH]
    if resp_ex:
        tref = resp_ex[0][3]
        fz_stack = np.array([ex[1] for ex in resp_ex])
        post_mask = tref >= 0
        if fz_stack.shape[0] > 0 and np.any(post_mask):
            pk_t = []
            for row in fz_stack:
                pv = row[post_mask]
                if np.any(np.isfinite(pv)):
                    pk_t.append(tref[post_mask][np.nanargmax(np.abs(pv))])
                else:
                    pk_t.append(0.5)
            fz_sorted = fz_stack[np.argsort(pk_t)]
            vmax = min(np.nanpercentile(np.abs(fz_sorted), 98), 15)
            im = ax_b.imshow(fz_sorted, aspect="auto",
                             extent=[tref[0]*1000, tref[-1]*1000, len(fz_sorted), 0],
                             cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                             interpolation="nearest")
            ax_b.axvline(0, color="k", linewidth=0.8, linestyle="--")
            plt.colorbar(im, ax=ax_b, label="Z-score", shrink=0.7)
    ax_b.set_xlabel("Time from fast TF pulse (ms)")
    ax_b.set_ylabel("Unit (sorted by peak time)")
    ax_b.set_title(f"B – Fast TF pulse responses (n={len(resp_ex)} responsive)")

    # ── Panel C: Z-score distribution ─────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    z_all = df["z_abs_max"].dropna().values
    ax_c.hist(z_all, bins=50, color="#78909C", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax_c.axvline(Z_THRESH, color="red", linewidth=1.5, linestyle="--",
                 label=f"Threshold={Z_THRESH}")
    ax_c.set_xlabel("Max |z-score| (fast or slow)")
    ax_c.set_ylabel("Number of units")
    ax_c.set_title("C – TF z-score distribution")
    ax_c.legend(fontsize=8)
    ax_pie = ax_c.inset_axes([0.65, 0.55, 0.3, 0.4])
    n_non = len(df) - n_resp
    ax_pie.pie([n_resp, n_non],
               labels=[f"Resp.\n({n_resp})", f"Non-resp.\n({n_non})"],
               colors=["#E53935", "#BDBDBD"], autopct="%1.0f%%",
               textprops={"fontsize": 7})

    # ── Panel D: By stage & cell type ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    stages = [s for s in STAGE_ORDER if s in df["stage"].values]
    x_pos = np.arange(len(stages))
    frac_by_stage, n_by_stage = [], []
    for s in stages:
        sub = df[df["stage"] == s]
        frac_by_stage.append(sub["is_tf_responsive"].mean() * 100 if len(sub) else 0)
        n_by_stage.append(len(sub))
    ax_d.bar(x_pos - 0.15, frac_by_stage, 0.3,
             color=[STAGE_COLORS[s] for s in stages], edgecolor="black", linewidth=0.5)
    cell_types = sorted([c for c in df["cell_type"].unique() if c != "Unknown"])
    if len(cell_types) >= 2:
        offsets = np.linspace(-0.15, 0.15, len(cell_types))
        w = 0.25 / len(cell_types)
        for ci, ct in enumerate(cell_types):
            fracs = [df[(df["stage"]==s)&(df["cell_type"]==ct)]["is_tf_responsive"].mean()*100
                     if len(df[(df["stage"]==s)&(df["cell_type"]==ct)]) else 0 for s in stages]
            ax_d.bar(x_pos + 0.2 + offsets[ci], fracs, w,
                     color=CELLTYPE_COLORS.get(ct,"#999"), edgecolor="black",
                     linewidth=0.3, alpha=0.7, label=ct)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([f"{s}\n(n={n})" for s,n in zip(stages, n_by_stage)])
    ax_d.set_ylabel("% TF-responsive")
    ax_d.set_title("D – TF responsiveness by stage & cell type")
    if len(cell_types) >= 2:
        ax_d.legend(fontsize=7, loc="upper left")
    if len(stages) >= 2:
        cont = [[int(df[df["stage"]==s]["is_tf_responsive"].sum()),
                  len(df[df["stage"]==s]) - int(df[df["stage"]==s]["is_tf_responsive"].sum())]
                 for s in stages]
        if all(sum(r) > 0 for r in cont):
            try:
                _, p, _, _ = chi2_contingency(cont)
                ax_d.text(0.5, 0.95, f"χ² test: p={p:.2e}",
                         transform=ax_d.transAxes, fontsize=8, ha="center", va="top")
            except Exception:
                pass

    fig.suptitle(
        "TF Pulse Responsiveness in Medial Striatum\n"
        "(Baseline TF fluctuations | Khilkevich & Lohse, Nature 2024 framework)",
        fontsize=13, fontweight="bold", y=0.98)
    save_figure(fig, "fig24_tf_responsiveness", "08_tf_pulse")
    print("\n  ✓ Saved fig24_tf_responsiveness.png")

    print("\n  Summary by stage:")
    for s in stages:
        sub = df[df["stage"] == s]
        nr = int(sub["is_tf_responsive"].sum())
        print(f"    {s}: {nr}/{len(sub)} = {100*nr/len(sub):.1f}% responsive")
    print("\n  Summary by cell type:")
    for ct in cell_types:
        sub = df[df["cell_type"] == ct]
        nr = int(sub["is_tf_responsive"].sum())
        print(f"    {ct}: {nr}/{len(sub)} = {100*nr/len(sub):.1f}% responsive")


if __name__ == "__main__":
    main()
