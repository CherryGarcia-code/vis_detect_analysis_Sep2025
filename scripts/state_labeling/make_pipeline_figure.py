"""Pipeline slide: how one session becomes a tagged session (worked example).

Four stages, drawn from real artifacts for a single session:
  (1) outcome raster  -> the session's lick-valence ethogram
  (2) local features  -> the windowed outcome-composition fractions the rule reads,
                         with the tree's cut thresholds drawn as labelled dashed lines
  (3) learned rule    -> a clean flowchart of the effective depth-3 tree (thresholds
                         read live from state_rule.pkl)
  (4) tagged session  -> the per-trial state strip

    py scripts/state_labeling/make_pipeline_figure.py --subject BG_046 --session 19082025
"""
import argparse
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch

from visdetect.analysis.config import STATE_LABEL_COLORS
from visdetect.analysis.constants import STATE_LABELS, STATE_FEATURE_COLS
from visdetect.analysis.state_labeling import render_raster, render_state_strip
from visdetect.analysis.state_calibration import CalibrationResult

# feature -> (line colour, friendly name, x-position of its threshold label as frac of n)
FEATURE_LINES = [
    ("f_inapplick", "#ef6548", "inappropriate lick", 0.60),
    ("f_applick",   "#3f93c9", "appropriate lick",   0.13),
    ("f_nolick",    "#3474ae", "no-lick",            0.44),
    ("f_abort",     "#7a7a7a", "abort",              0.28),
]
ARROW = dict(arrowstyle="-|>", lw=2.4, color="#333333")
DARK_TEXT = {"StimSens", "Abort"}


def primary_thresholds(tree):
    """{feature: shallowest threshold} via BFS from the root."""
    t = tree.tree_
    out, seen = {}, {}
    q = deque([(0, 0)])
    while q:
        node, depth = q.popleft()
        f = t.feature[node]
        if f >= 0:
            name = STATE_FEATURE_COLS[f]
            if name not in seen or depth < seen[name]:
                seen[name] = depth
                out[name] = float(t.threshold[node])
            q.append((t.children_left[node], depth + 1))
            q.append((t.children_right[node], depth + 1))
    return out


def node(ax, x, y, text, face, edge="#555555", tcolor="black", fs=10.5, bold=False):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tcolor,
            fontweight="bold" if bold else "normal", zorder=3,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=face, edgecolor=edge, linewidth=1.4))


def edge(ax, xy_from, xy_to, label=None):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#555555"))
    if label:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx, my, label, fontsize=9.5, color="0.25", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))


def draw_rule(ax, thr):
    """Clean flowchart of the effective (collapsed) depth-3 rule."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    nl = thr.get("f_nolick", 0.41); ab = thr.get("f_abort", 0.38)
    ia = thr.get("f_inapplick", 0.45); ap = thr.get("f_applick", 0.24)
    root = (0.5, 0.93); dL = (0.25, 0.66); dR = (0.78, 0.66)
    lAb = (0.09, 0.40); dI = (0.34, 0.40); lDi = (0.67, 0.40); lSt2 = (0.92, 0.40)
    lIm = (0.25, 0.12); lSt = (0.48, 0.12)
    C = STATE_LABEL_COLORS

    edge(ax, root, dL, "responding"); edge(ax, root, dR, "withdrawn")
    edge(ax, dL, lAb, "yes"); edge(ax, dL, dI, "no")
    edge(ax, dR, lDi, "yes"); edge(ax, dR, lSt2, "no")
    edge(ax, dI, lIm, "yes"); edge(ax, dI, lSt, "no")

    node(ax, *root, f"no-lick ≤ {nl:.2f}?", "white", fs=11.5, bold=True)
    node(ax, *dL, f"aborts > {ab:.2f}?", "white")
    node(ax, *dR, f"appropriate\nlicks ≤ {ap:.2f}?", "white")
    node(ax, *dI, f"inappropriate\nlicks > {ia:.2f}?", "white")
    for xy, s in [(lAb, "Abort"), (lDi, "Disengaged"), (lSt2, "StimSens"),
                  (lIm, "Impulsive"), (lSt, "StimSens")]:
        node(ax, *xy, s, C[s], edge="none",
             tcolor="black" if s in DARK_TEXT else "white", fs=10.5, bold=True)


def main():
    ap = argparse.ArgumentParser(description="State-tagger pipeline worked-example slide.")
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--session", default="19082025")
    ap.add_argument("--model", default="data/state_labels/state_rule.pkl")
    ap.add_argument("--tags-root", default=os.path.join("data", "cache", "state_tags"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.tags_root, args.subject, f"{args.session}.csv"))
    result = CalibrationResult.load(args.model)
    n = len(df)
    thr = primary_thresholds(result.tree)

    fig = plt.figure(figsize=(16, 8.6))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2.35, 1.0], height_ratios=[1.0, 2.3, 1.0],
                           left=0.08, right=0.985, top=0.9, bottom=0.085,
                           hspace=0.62, wspace=0.14)
    ax_r = fig.add_subplot(gs[0, 0]); ax_f = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[2, 0]); ax_tree = fig.add_subplot(gs[:, 1])

    def stage(ax, num, text):
        ax.set_title(f"{num}  {text}", fontsize=12.5, fontweight="bold", loc="left", pad=6)

    # (1) raster
    render_raster(ax_r, df, catch_lw=0.3)
    ax_r.set_xlim(-0.5, n - 0.5); ax_r.set_xlabel(""); ax_r.tick_params(labelbottom=False)
    stage(ax_r, "①", "Outcome raster  — each trial's lick outcome")
    ax_r.legend(handles=[Patch(facecolor="#6fb58f", label="hit"),
                         Patch(facecolor="#e3897c", label="FA / early"),
                         Patch(facecolor="#9488bf", label="no-lick"),
                         Patch(facecolor="#bdbdbd", label="abort")],
                loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=9, frameon=False,
                handlelength=1.1, labelspacing=0.3)

    # (2) features + labelled threshold lines
    for col, c, friendly, lx in FEATURE_LINES:
        ax_f.plot(df["trial_idx"], df[col], color=c, lw=1.4)
        if col in thr:
            tv = thr[col]
            ax_f.axhline(tv, color=c, ls="--", lw=1.1, alpha=0.55)
            ax_f.text(n * lx, tv + 0.015, f"{friendly}  cut = {tv:.2f}", color=c, fontsize=9,
                      ha="center", va="bottom", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
    ax_f.set_xlim(-0.5, n - 0.5); ax_f.set_ylim(0, 1.02)
    ax_f.set_ylabel("local fraction", fontsize=11); ax_f.set_xlabel("")
    ax_f.tick_params(labelbottom=False, labelsize=9.5)
    stage(ax_f, "②", f"Local outcome composition  — windowed fractions (W={result.window}).  "
                          "Dashed = the tree's cut thresholds.")

    # (3) learned rule flowchart
    draw_rule(ax_tree, thr)
    stage(ax_tree, "③", "Learned rule  (effective depth-3 tree)")
    ax_tree.text(0.5, 0.0, f"fit to sparse expert labels  ·  LOSO κ = {result.loso_kappa:.2f}",
                 transform=ax_tree.transAxes, ha="center", va="top", fontsize=10, color="0.35")

    # (4) tagged strip
    render_state_strip(ax_t, df["state_label"].tolist(), gated=df["state_gated"].tolist())
    ax_t.set_xlim(-0.5, n - 0.5); ax_t.set_xlabel("trial index", fontsize=11)
    ax_t.tick_params(labelsize=9.5)
    stage(ax_t, "④", "Tagged session  — per-trial state (faded = low-confidence)")
    ax_t.legend(handles=[Patch(facecolor=STATE_LABEL_COLORS[s], label=s) for s in STATE_LABELS],
                loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=9, frameon=False,
                handlelength=1.1, labelspacing=0.3)

    # flow arrows.  Both cross-axis arrows are drawn on ax_tree (added last -> on
    # top) with annotation_clip=False so nothing hides them.
    # (1) raster -> features (down)
    ax_f.annotate("", xy=(0.5, 1.14), xycoords=ax_f.transAxes,
                  xytext=(0.5, -0.12), textcoords=ax_r.transAxes, arrowprops=ARROW)
    # (2) features -> the tree's ROOT node ("run each trial through the rule")
    ax_tree.annotate("apply rule\nto each trial", xy=(0.34, 0.9), xycoords=ax_tree.transAxes,
                     xytext=(1.04, 0.78), textcoords=ax_f.transAxes, fontsize=11,
                     va="center", ha="left", color="#333333", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#333333",
                                     connectionstyle="arc3,rad=0.15"))
    # (3) tree -> tagged strip (the rule's output); on ax_tree so it draws above ax_t
    ax_tree.annotate("gives the\ntag", xy=(1.02, 0.5), xycoords=ax_t.transAxes,
                     xytext=(0.45, -0.03), textcoords=ax_tree.transAxes, fontsize=10.5,
                     va="top", ha="center", color="#333333", annotation_clip=False, zorder=30,
                     arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#333333",
                                     connectionstyle="arc3,rad=-0.25"))

    fig.suptitle(f"How a session becomes tagged — state-labeler pipeline "
                 f"(worked example: {args.subject} {args.session})",
                 fontsize=14.5, fontweight="bold", y=0.965)

    out = args.out or os.path.join("figures", "state_labeler",
                                   f"slide_pipeline_{args.subject}_{args.session}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    pdf = os.path.splitext(out)[0] + ".pdf"          # vector version for projection
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
