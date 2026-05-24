"""Figure-rendering helpers for the per-UID QC sheets.

Two pages per UID.  All gridspec ratios are picked per
docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md §6.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visdetect.suite.config import STAGE_COLORS, STAGE_ORDER  # noqa: E402
from visdetect.suite.plotting import setup_style                # noqa: E402
from visdetect.analysis.tracking_qc import (                    # noqa: E402
    UIDIntermediate, SessionRecord,
    badge_isi, badge_depth, badge_waveform, badge_fr, composite_verdict,
)

setup_style()

# Per-criterion colors
BADGE_COLORS = {"pass": "#2d5a2d", "warn": "#5a5a2d", "fail": "#5a2d2d"}
BADGE_SYMBOLS = {"pass": "✅", "warn": "⚠", "fail": "❌"}


def draw_header(ax, uid: UIDIntermediate,
                isi_score: float, depth_std: float,
                wave_corr: float, fr_cv_val: float) -> str:
    """Draw the 4-badge header strip + stage stripe.  Returns composite verdict."""
    ax.set_axis_off()

    b_isi   = badge_isi(isi_score)
    b_dep   = badge_depth(depth_std)
    b_wave  = badge_waveform(wave_corr)
    b_fr    = badge_fr(fr_cv_val)
    verdict = composite_verdict([b_isi, b_dep, b_wave, b_fr])

    ne_flag = " · N→E" if uid.has_naive_to_expert else ""
    suspect = " · ⚠ KNOWN SUSPECT" if uid.suspect_known else ""
    title = (f"UID {uid.global_uid} · span {uid.span}{ne_flag}{suspect}"
             f"   composite: {verdict.upper()}")
    ax.text(0.0, 0.92, title, fontsize=13, fontweight="bold",
            transform=ax.transAxes, va="top")

    # Badge row
    badges = [
        (f"ISI {isi_score:.2f}", b_isi),
        (f"depth {depth_std:.1f}µm", b_dep),
        (f"wave r={wave_corr:.2f}", b_wave),
        (f"FR CV {fr_cv_val:.2f}", b_fr),
    ]
    x = 0.0
    for label, level in badges:
        text = f"{BADGE_SYMBOLS[level]} {label}"
        ax.text(x, 0.55, text, fontsize=10,
                transform=ax.transAxes, va="center",
                bbox=dict(facecolor=BADGE_COLORS[level], edgecolor="none",
                          pad=4, alpha=0.85),
                color="white")
        x += 0.20

    # Stage stripe at the bottom: one cell per session in chronological order
    if uid.sessions:
        n = len(uid.sessions)
        bar_y = 0.05
        bar_h = 0.18
        for i, rec in enumerate(uid.sessions):
            color = STAGE_COLORS.get(rec.stage, "#888888")
            ax.add_patch(Rectangle((i / n, bar_y), 1.0 / n, bar_h,
                                    transform=ax.transAxes,
                                    facecolor=color, edgecolor="none"))

    return verdict


def _waveform_color(stage: str) -> str:
    return STAGE_COLORS.get(stage, "#888888")


def render_page1(uid: UIDIntermediate, um_pair_scores: Optional[np.ndarray],
                 isi_score: float, depth_std: float, wave_corr: float,
                 fr_cv_val: float) -> plt.Figure:
    """Render page 1 (physical) — returns the Figure."""
    fig = plt.figure(figsize=(8.5, 11.0))

    # Master gridspec: header / footprints / waveform / depth-amp / um-scores
    gs = gridspec.GridSpec(
        nrows=5, ncols=1,
        height_ratios=[0.9, 2.5, 1.5, 1.8, 0.9],
        hspace=0.55, top=0.96, bottom=0.04, left=0.08, right=0.96,
    )

    # Header
    ax_hdr = fig.add_subplot(gs[0])
    draw_header(ax_hdr, uid, isi_score, depth_std, wave_corr, fr_cv_val)

    # Footprints @ first / mid / last
    fp_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.25)
    n = len(uid.sessions)
    if n >= 1:
        idxs = [0, n // 2, n - 1]
        labels = ["first", "mid", "last"]
        for col, (idx, lab) in enumerate(zip(idxs, labels)):
            ax = fig.add_subplot(fp_gs[col])
            rec = uid.sessions[idx]
            # Footprint: lines per channel stacked vertically by channel index
            fp = rec.footprint                                # (n_samples, n_chans)
            offsets = np.arange(fp.shape[1])[None, :] * (np.abs(fp).max() + 1e-6) * 1.2
            ax.plot(fp + offsets, color=_waveform_color(rec.stage), linewidth=0.6)
            ax.set_title(f"{lab}: {rec.session_name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    # Peak-channel waveform overlay (~2:1)
    ax_wf = fig.add_subplot(gs[2])
    for rec in uid.sessions:
        ax_wf.plot(rec.waveform_peak, color=_waveform_color(rec.stage),
                   linewidth=0.6, alpha=0.6)
    ax_wf.set_title("Peak-channel waveform overlay", fontsize=10)
    ax_wf.set_xlabel("samples"); ax_wf.set_ylabel("µV (raw)")

    # Depth + amplitude side by side (~3:1 each)
    da_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], wspace=0.30)
    ax_d = fig.add_subplot(da_gs[0])
    ax_a = fig.add_subplot(da_gs[1])
    xs = np.arange(len(uid.sessions))
    depths = [r.peak_depth_um for r in uid.sessions]
    amps   = [r.amplitude for r in uid.sessions]
    colors = [_waveform_color(r.stage) for r in uid.sessions]
    ax_d.scatter(xs, depths, c=colors, s=18); ax_d.plot(xs, depths, color="0.5", linewidth=0.7)
    ax_d.set_xlabel("session #"); ax_d.set_ylabel("peak depth (µm)")
    ax_d.set_title("Depth on probe", fontsize=10)
    ax_a.scatter(xs, amps, c=colors, s=18);   ax_a.plot(xs, amps,   color="0.5", linewidth=0.7)
    ax_a.set_xlabel("session #"); ax_a.set_ylabel("amplitude (µV)")
    ax_a.set_title("Amplitude", fontsize=10)

    # UM pairwise scores
    ax_um = fig.add_subplot(gs[4])
    if um_pair_scores is not None and len(um_pair_scores) > 0:
        ax_um.bar(np.arange(len(um_pair_scores)), um_pair_scores, color="0.4")
        ax_um.set_ylim(0, 1)
        ax_um.set_title("UM consecutive-session match probability", fontsize=10)
        ax_um.set_xlabel("pair # (session i, i+1)")
    else:
        ax_um.text(0.5, 0.5, "UM pair scores unavailable", ha="center", va="center",
                   transform=ax_um.transAxes, fontsize=10, color="0.5")
        ax_um.set_axis_off()

    return fig
