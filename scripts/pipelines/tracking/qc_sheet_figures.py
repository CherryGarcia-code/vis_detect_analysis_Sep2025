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
