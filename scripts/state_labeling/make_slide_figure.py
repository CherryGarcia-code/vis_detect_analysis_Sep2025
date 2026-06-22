"""Compose a presentation slide: representative tagged sessions across mice.

Reads each subject's `_slide_pick.json` (the most slide-worthy session, chosen by
the selection step) and renders one stacked **raster + tagger** panel per mouse,
with shared outcome/state legends. All panels share a common trial-width scale so
a "trial" is the same size across mice (shorter sessions end early, leaving white
space). Renders straight from the tagged CSVs — no session pkls needed.

    py scripts/state_labeling/make_slide_figure.py
    py scripts/state_labeling/make_slide_figure.py --subjects BG_046 BG_031 \
        --picks BG_046:17092025 BG_031:190325        # override auto-selection
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch

from visdetect.analysis.config import STATE_LABEL_COLORS
from visdetect.analysis.constants import STATE_LABELS
from visdetect.analysis.state_labeling import (
    render_raster, render_state_strip, lick_valence_legend_handles,
)

DEFAULT_ORDER = ["BG_046", "BG_031", "BG_038", "BG_039"]


def resolve_session(subject, tags_root, overrides):
    if subject in overrides:
        return overrides[subject]
    pick_path = os.path.join(tags_root, subject, "_slide_pick.json")
    with open(pick_path) as f:
        return str(json.load(f)["session"])


def main():
    ap = argparse.ArgumentParser(description="Cross-subject state-labeler slide figure.")
    ap.add_argument("--tags-root", default=os.path.join("data", "cache", "state_tags"))
    ap.add_argument("--subjects", nargs="*", default=DEFAULT_ORDER)
    ap.add_argument("--picks", nargs="*", default=[],
                    help="optional SUBJECT:SESSION overrides for the auto-selected sessions")
    ap.add_argument("--out", default=os.path.join("figures", "state_labeler",
                                                  "slide_representative_states.png"))
    args = ap.parse_args()

    overrides = dict(p.split(":", 1) for p in args.picks)

    panels = []
    for s in args.subjects:
        try:
            sess = resolve_session(s, args.tags_root, overrides)
        except FileNotFoundError:
            print(f"no _slide_pick.json for {s} — skipping (run selection or pass --picks)")
            continue
        df = pd.read_csv(os.path.join(args.tags_root, s, f"{sess}.csv"))
        panels.append((s, sess, df))
    if not panels:
        raise SystemExit("no panels to draw")

    common = max(len(df) for _, _, df in panels)   # shared trial-width scale
    n = len(panels)

    fig = plt.figure(figsize=(13, 1.9 + 2.1 * n))
    outer = gridspec.GridSpec(n, 1, hspace=0.5, top=0.85, bottom=0.15,
                              left=0.135, right=0.985)
    for i, (s, sess, df) in enumerate(panels):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i],
                                                 height_ratios=[1.9, 1.0], hspace=0.08)
        ax_r = fig.add_subplot(inner[0])
        ax_t = fig.add_subplot(inner[1])

        render_raster(ax_r, df, catch_lw=0.3)   # subtle catch outline keeps dense rasters clean
        ax_r.set_xlim(-0.5, common - 0.5)
        ax_r.set_xlabel(""); ax_r.tick_params(labelbottom=False)
        ax_r.set_ylabel(f"{s}\n{sess}", rotation=0, ha="right", va="center",
                        fontsize=10, fontweight="bold")
        ax_r.text(0.997, 1.07, f"n = {len(df)} trials", transform=ax_r.transAxes,
                  ha="right", va="bottom", fontsize=7.5, color="0.4")

        render_state_strip(ax_t, df["state_label"].tolist(),
                           gated=df["state_gated"].tolist(), ylabel="tagged\nstate")
        ax_t.set_xlim(-0.5, common - 0.5)
        ax_t.tick_params(labelbottom=(i == n - 1))   # trial-index ticks on the bottom panel only

    fig.suptitle("Behavioral-state labeler — representative sessions across mice",
                 fontsize=13, fontweight="bold", y=0.965)
    fig.legend(handles=lick_valence_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 0.925), ncol=6, frameon=False, fontsize=8,
               title="trial outcome  (upper strip)", title_fontsize=8)
    state_handles = [Patch(facecolor=STATE_LABEL_COLORS.get(x, "#999999"), label=x)
                     for x in STATE_LABELS]
    fig.legend(handles=state_handles, loc="lower center", bbox_to_anchor=(0.5, 0.05),
               ncol=len(state_handles), frameon=False, fontsize=9,
               title="tagged state  (lower strip)", title_fontsize=8)
    fig.text(0.5, 0.10, "trial index", ha="center", fontsize=10)
    fig.text(0.5, 0.012,
             "faded cells = low-confidence (gated)   ·   catch trials outlined in black",
             ha="center", fontsize=7, style="italic", color="0.45")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)
    for s, sess, df in panels:
        print(f"  {s}: {sess}  (n={len(df)})")


if __name__ == "__main__":
    main()
