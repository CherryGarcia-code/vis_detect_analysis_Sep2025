# scripts/anatomy/plot_shank_anatomy.py
"""QC figure: region-by-depth per shank for a subject. See spec §9."""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COARSE_COLORS = {
    "CP": "#2c7fb8", "GPe": "#d95f0e", "CTX": "#7fbc41", "WM": "#999999",
    "VS": "#9e9ac8", "out": "#000000", "other": "#cccccc", "unknown": "#eeeeee",
}


def plot_subject_anatomy(subject, atlas_csv, out_png, unit_csv=None) -> str:
    atlas = pd.read_csv(atlas_csv)
    shanks = sorted(atlas["shank"].unique())
    fig, axes = plt.subplots(1, len(shanks), figsize=(2.2 * len(shanks), 7),
                             sharey=True, squeeze=False)
    for ax, sh in zip(axes[0], shanks):
        d = atlas[atlas["shank"] == sh].sort_values("y_um")
        for _, r in d.iterrows():
            ax.scatter(0, r["y_um"], s=18,
                       color=COARSE_COLORS.get(r["region_coarse"], "#cccccc"))
        # mark CTX->CP transition (most dorsal CP channel)
        cp = d[d["region_coarse"] == "CP"]
        if not cp.empty:
            yt = cp["y_um"].max()
            sig = float(d.loc[cp["y_um"].idxmax(), "sigma_um"])
            ax.axhspan(yt - sig, yt + sig, color="red", alpha=0.15)
            ax.axhline(yt, color="red", lw=0.8)
        ax.set_title(f"shank {sh}"); ax.set_xticks([])
    axes[0][0].set_ylabel("depth along shank (um)")
    fig.suptitle(f"{subject}: region by depth per shank")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=k)
               for k, c in COARSE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return str(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--anatomy-dir", default="data/anatomy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    atlas_csv = os.path.join(args.anatomy_dir, f"{args.subject}_channel_atlas.csv")
    out = args.out or os.path.join("figures", "anatomy", f"{args.subject}_shank_anatomy.png")
    plot_subject_anatomy(args.subject, atlas_csv, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
