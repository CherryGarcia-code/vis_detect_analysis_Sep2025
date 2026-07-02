"""Learning-arc slide: one subject across early/middle/late exemplar sessions.

Shows the behavioral-state labeler tracking a mouse's learning progression via
three representative sessions (e.g. disengaged -> engaged -> engaged), each a
stacked outcome-raster + tagged-state panel. Labels are free text (use '|' for a
line break) so panels can be annotated with phase + behavioral stage / d'.

    py scripts/state_labeling/make_learning_arc.py --subject BG_046 \
       --sessions 15072025 19082025 11092025 \
       --labels "Early|naive (QC-fail)" "Middle|Learning (d'=1.55)" "Late|Expert (d'=1.56)"
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser(description="Single-subject learning-arc slide.")
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--tags-root", default=os.path.join("data", "cache", "state_tags"))
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True,
                    help="one per session; '|' becomes a line break")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--equal-length", action="store_true",
                    help="truncate every session to the shortest so panels fill fully (no white tail)")
    ap.add_argument("--ethograms-only", action="store_true",
                    help="rasters only, no tagged-state strips (raw-behavior slide to show before the labels)")
    args = ap.parse_args()
    if len(args.labels) != len(args.sessions):
        raise SystemExit("--labels and --sessions must have equal length")

    raw = []
    for sess, lab in zip(args.sessions, args.labels):
        df = pd.read_csv(os.path.join(args.tags_root, args.subject, f"{sess}.csv"))
        raw.append((sess, lab.replace("|", "\n"), df))
    full_len = {sess: len(df) for sess, _, df in raw}
    if args.equal_length:
        common = min(full_len.values())
        panels = [(s, l, df.iloc[:common].reset_index(drop=True)) for s, l, df in raw]
    else:
        common = max(full_len.values())
        panels = raw
    n = len(panels)
    eo = args.ethograms_only

    fig = plt.figure(figsize=(13, (1.4 if eo else 1.9) + (1.6 if eo else 2.1) * n))
    outer = gridspec.GridSpec(n, 1, hspace=(0.45 if eo else 0.5),
                              top=(0.84 if eo else 0.85), bottom=(0.12 if eo else 0.15),
                              left=0.16, right=0.985)
    for i, (sess, lab, df) in enumerate(panels):
        if eo:
            ax_r = fig.add_subplot(outer[i])
        else:
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i],
                                                     height_ratios=[1.9, 1.0], hspace=0.08)
            ax_r = fig.add_subplot(inner[0])
            ax_t = fig.add_subplot(inner[1])

        render_raster(ax_r, df, catch_lw=0.3)
        ax_r.set_xlim(-0.5, common - 0.5)
        ax_r.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10,
                        fontweight="bold")
        full = full_len[sess]
        ntxt = (f"{sess}   ·   first {len(df)} of {full} trials" if len(df) < full
                else f"{sess}   ·   n = {full} trials")
        ax_r.text(0.997, 1.07, ntxt, transform=ax_r.transAxes, ha="right",
                  va="bottom", fontsize=7.5, color="0.4")
        ax_r.set_xlabel("")

        if eo:
            ax_r.tick_params(labelbottom=(i == n - 1))
        else:
            ax_r.tick_params(labelbottom=False)
            dom = df["state_label"].value_counts(normalize=True)
            dom_txt = "   ".join(f"{s} {dom.get(s, 0)*100:.0f}%"
                                 for s in STATE_LABELS if dom.get(s, 0) >= 0.10)
            render_state_strip(ax_t, df["state_label"].tolist(),
                               gated=df["state_gated"].tolist(), ylabel="tagged\nstate")
            ax_t.set_xlim(-0.5, common - 0.5)
            ax_t.text(0.997, -0.02, dom_txt, transform=ax_t.transAxes, ha="right",
                      va="top", fontsize=7.5, color="0.35")
            ax_t.tick_params(labelbottom=(i == n - 1))

    if args.title:
        title = args.title
    elif eo:
        title = f"{args.subject} — raw behavior across learning (early / middle / late)"
    else:
        title = f"{args.subject} — behavioral-state labeler tracks the learning arc"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.965)
    fig.legend(handles=lick_valence_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 0.9 if eo else 0.925), ncol=6, frameon=False,
               fontsize=8, title="trial outcome", title_fontsize=8)
    if eo:
        fig.text(0.5, 0.055, "trial index", ha="center", fontsize=10)
        fig.text(0.5, 0.015, "catch trials outlined in black", ha="center",
                 fontsize=7, style="italic", color="0.45")
    else:
        state_handles = [Patch(facecolor=STATE_LABEL_COLORS.get(x, "#999999"), label=x)
                         for x in STATE_LABELS]
        fig.legend(handles=state_handles, loc="lower center", bbox_to_anchor=(0.5, 0.05),
                   ncol=len(state_handles), frameon=False, fontsize=9,
                   title="tagged state  (lower strip)", title_fontsize=8)
        fig.text(0.5, 0.10, "trial index", ha="center", fontsize=10)
        fig.text(0.5, 0.012,
                 "faded cells = low-confidence (gated)   ·   catch trials outlined in black",
                 ha="center", fontsize=7, style="italic", color="0.45")

    default_name = f"slide_learning_arc_{args.subject}" + ("_ethograms" if eo else "") + ".png"
    out = args.out or os.path.join("figures", "state_labeler", default_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
