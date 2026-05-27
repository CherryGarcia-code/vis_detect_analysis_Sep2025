"""Figure-rendering helpers for the per-UID QC sheets.

Two pages per UID.  Layout reference: docs/superpowers/plans/
2026-05-22-tracking-qc-sheets-plan.md Tasks 11 (page 1) and 12 (page 2).
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

# Local extension: tracking_qc adds "Unknown" stage (sessions not in the
# tracking-QC filter, see spec §3.4). Light grey distinguishes from the
# dimmed-trace grey (0.7) used for trimmed-but-not-Unknown sessions.
STAGE_COLORS_LOCAL = {**STAGE_COLORS, "Unknown": "#bbbbbb"}

# Per-criterion colors
BADGE_COLORS = {"pass": "#2d5a2d", "warn": "#5a5a2d", "fail": "#5a2d2d"}
BADGE_SYMBOLS = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}


def draw_header(ax, uid: UIDIntermediate,
                isi_score: float, depth_std: float,
                wave_corr: float, fr_cv_val: float,
                *,
                dropped_indices: Optional[List[int]] = None,
                n_kept: Optional[int] = None,
                trimmed_verdict: Optional[str] = None) -> str:
    """Draw the 4-badge header strip + stage stripe.  Returns composite verdict.

    Optional trim-visualization kwargs:
      dropped_indices : session indices flagged by find_stable_subset; those
                         cells in the stage stripe are overlaid with a diagonal
                         hatch on top of the stage color.
      n_kept, trimmed_verdict : if both provided, append a secondary annotation
                         line with the kept-count and the trimmed-subset verdict.
    """
    ax.set_axis_off()
    dropped_set = set(dropped_indices or [])

    b_isi   = badge_isi(isi_score)
    b_dep   = badge_depth(depth_std)
    b_wave  = badge_waveform(wave_corr)
    b_fr    = badge_fr(fr_cv_val)
    verdict = composite_verdict([b_isi, b_dep, b_wave, b_fr])

    ne_flag = " · N→E" if uid.has_naive_to_expert else ""
    suspect = " · KNOWN SUSPECT" if uid.suspect_known else ""
    title = (f"UID {uid.global_uid} · span {uid.span}{ne_flag}{suspect}"
             f"   composite: {verdict.upper()}")
    ax.text(0.0, 0.95, title, fontsize=13, fontweight="bold",
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
        ax.text(x, 0.65, text, fontsize=10,
                transform=ax.transAxes, va="center",
                bbox=dict(facecolor=BADGE_COLORS[level], edgecolor="none",
                          pad=4, alpha=0.85),
                color="white")
        x += 0.20

    # Stage stripe: one cell per session in chronological order.
    # Dropped sessions get a diagonal hatch overlay so the user can SEE which
    # sessions were excluded by find_stable_subset without opening the CSV.
    if uid.sessions:
        n = len(uid.sessions)
        bar_y = 0.30
        bar_h = 0.18
        for i, rec in enumerate(uid.sessions):
            color = STAGE_COLORS_LOCAL.get(rec.stage, "#888888")
            ax.add_patch(Rectangle((i / n, bar_y), 1.0 / n, bar_h,
                                    transform=ax.transAxes,
                                    facecolor=color, edgecolor="none"))
            if i in dropped_set:
                ax.add_patch(Rectangle((i / n, bar_y), 1.0 / n, bar_h,
                                        transform=ax.transAxes,
                                        facecolor="none", edgecolor="black",
                                        hatch="///", linewidth=0.5))

    # Optional trim annotation at the BOTTOM of the header axes, below the
    # stage stripe.  Only shown when something was actually dropped; an
    # all-kept run would just be visual noise.
    if n_kept is not None and trimmed_verdict is not None and dropped_set:
        n_total = len(uid.sessions)
        ax.text(0.0, 0.10,
                f"Trimmed: kept {n_kept}/{n_total} sessions "
                f"(verdict on trimmed: {trimmed_verdict})",
                fontsize=9, transform=ax.transAxes,
                ha="left", va="bottom", color="0.4")

    # Stage-stripe legend (always shown, regardless of trim state)
    ax.text(
        0.0, -0.04,
        "stripe: Learning · Expert · Unknown · /// = trimmed",
        transform=ax.transAxes, fontsize=8, color="0.4",
        ha="left", va="top",
    )

    return verdict


def _waveform_color(stage: str) -> str:
    return STAGE_COLORS_LOCAL.get(stage, "#888888")


def _scatter_kept_dropped(ax, xs, ys, colors, dropped_set):
    """Scatter with kept sessions as filled stage-colored dots and dropped
    sessions as open circles (white face, stage-colored edge)."""
    xs = np.asarray(xs); ys = np.asarray(ys)
    kept_mask = np.array([i not in dropped_set for i in range(len(xs))])
    if kept_mask.any():
        ax.scatter(xs[kept_mask], ys[kept_mask],
                   c=[colors[i] for i in range(len(xs)) if kept_mask[i]],
                   s=18, zorder=3)
    if (~kept_mask).any():
        edge = [colors[i] for i in range(len(xs)) if not kept_mask[i]]
        ax.scatter(xs[~kept_mask], ys[~kept_mask],
                   facecolors="white", edgecolors=edge,
                   linewidths=1.0, s=22, zorder=3)


def render_page1(uid: UIDIntermediate, um_pair_scores: Optional[np.ndarray],
                 isi_score: float, depth_std: float, wave_corr: float,
                 fr_cv_val: float,
                 *,
                 dropped_indices: Optional[List[int]] = None,
                 n_kept: Optional[int] = None,
                 trimmed_verdict: Optional[str] = None) -> plt.Figure:
    """Render page 1 (physical) — returns the Figure.

    Layout (5 rows × 1 col master):
      Row 0: header
      Row 1: 3 footprints (first / mid / last)
      Row 2: peak-channel waveform overlay | UM consecutive-pair match probability
      Row 3: ISI distribution | Baseline FR
      Row 4: Depth on probe | Amplitude
    """
    dropped_set = set(dropped_indices or [])

    fig = plt.figure(figsize=(8.5, 11.0))

    gs = gridspec.GridSpec(
        nrows=5, ncols=1,
        height_ratios=[1.25, 1.75, 1.35, 1.35, 1.55],
        hspace=0.65, top=0.96, bottom=0.05, left=0.08, right=0.96,
    )

    # ── Row 0: Header
    ax_hdr = fig.add_subplot(gs[0])
    draw_header(ax_hdr, uid, isi_score, depth_std, wave_corr, fr_cv_val,
                dropped_indices=dropped_indices,
                n_kept=n_kept, trimmed_verdict=trimmed_verdict)

    # ── Row 1: Footprints @ first / mid / last
    fp_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.25)
    n = len(uid.sessions)
    if n >= 1:
        idxs = [0, n // 2, n - 1]
        labels = ["first", "mid", "last"]
        for col, (idx, lab) in enumerate(zip(idxs, labels)):
            ax = fig.add_subplot(fp_gs[col])
            rec = uid.sessions[idx]
            fp = rec.footprint                                # (n_samples, n_chans)
            offsets = np.arange(fp.shape[1])[None, :] * (np.abs(fp).max() + 1e-6) * 1.2
            ax.plot(fp + offsets, color=_waveform_color(rec.stage), linewidth=0.6)
            prefix = "[DROPPED] " if idx in dropped_set else ""
            ax.set_title(f"{prefix}{lab}: {rec.session_name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    # ── Row 2: Peak-channel waveform overlay | UM consecutive-pair match probability
    wf_um_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.30)
    ax_wf = fig.add_subplot(wf_um_gs[0])
    for i, rec in enumerate(uid.sessions):
        if i in dropped_set:
            ax_wf.plot(rec.waveform_peak, color="0.7",
                       linewidth=0.5, alpha=0.6)
        else:
            ax_wf.plot(rec.waveform_peak, color=_waveform_color(rec.stage),
                       linewidth=0.6, alpha=0.6)
    ax_wf.set_title("Peak-channel waveform overlay", fontsize=10)
    ax_wf.set_xlabel("samples"); ax_wf.set_ylabel("µV (raw)")

    ax_um = fig.add_subplot(wf_um_gs[1])
    if um_pair_scores is not None and len(um_pair_scores) > 0:
        ax_um.bar(np.arange(len(um_pair_scores)), um_pair_scores, color="0.4")
        ax_um.set_ylim(0, 1)
        ax_um.set_title("UM consecutive-pair match probability", fontsize=10)
        ax_um.set_xlabel("pair # (session i, i+1)")
        ax_um.set_ylabel("match prob")
    else:
        ax_um.text(0.5, 0.5, "UM pair scores unavailable", ha="center", va="center",
                   transform=ax_um.transAxes, fontsize=10, color="0.5")
        ax_um.set_title("UM consecutive-pair match probability", fontsize=10)
        ax_um.set_xticks([]); ax_um.set_yticks([])

    # ── Row 3: ISI distribution | Baseline FR (moved from old page 2)
    isi_fr_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], wspace=0.30)
    ax_isi = fig.add_subplot(isi_fr_gs[0])
    for i, rec in enumerate(uid.sessions):
        if i in dropped_set:
            ax_isi.semilogx(rec.isi_centers, rec.isi_hist,
                            color="0.7", linewidth=0.5, alpha=0.6)
        else:
            ax_isi.semilogx(rec.isi_centers, rec.isi_hist,
                            color=_waveform_color(rec.stage),
                            linewidth=0.7, alpha=0.6)
    ax_isi.set_xlabel("ISI (s, log)"); ax_isi.set_ylabel("prob")
    ax_isi.set_title("ISI distribution", fontsize=10)
    if dropped_set:
        ax_isi.text(0.98, 0.97, "grey traces = dropped",
                    transform=ax_isi.transAxes, fontsize=7, color="0.4",
                    ha="right", va="top")

    ax_fr = fig.add_subplot(isi_fr_gs[1])
    xs = np.arange(len(uid.sessions))
    colors = [_waveform_color(r.stage) for r in uid.sessions]
    fr_vals = [r.baseline_fr_hz for r in uid.sessions]
    _scatter_kept_dropped(ax_fr, xs, fr_vals, colors, dropped_set)
    ax_fr.plot(xs, fr_vals, color="0.5", linewidth=0.7, zorder=1)
    ax_fr.set_xlabel("session #"); ax_fr.set_ylabel("FR (Hz)")
    ax_fr.set_title("Baseline FR", fontsize=10)

    # ── Row 4: Depth on probe | Amplitude
    da_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4], wspace=0.30)
    ax_d = fig.add_subplot(da_gs[0])
    ax_a = fig.add_subplot(da_gs[1])
    depths = [r.peak_depth_um for r in uid.sessions]
    amps   = [r.amplitude for r in uid.sessions]
    _scatter_kept_dropped(ax_d, xs, depths, colors, dropped_set)
    ax_d.plot(xs, depths, color="0.5", linewidth=0.7, zorder=1)
    ax_d.set_xlabel("session #"); ax_d.set_ylabel("peak depth (µm)")
    ax_d.set_title("Depth on probe", fontsize=10)
    if dropped_set:
        ax_d.text(0.98, 0.97, "○ = dropped",
                  transform=ax_d.transAxes, fontsize=7, color="0.4",
                  ha="right", va="top")
    _scatter_kept_dropped(ax_a, xs, amps, colors, dropped_set)
    ax_a.plot(xs, amps, color="0.5", linewidth=0.7, zorder=1)
    ax_a.set_xlabel("session #"); ax_a.set_ylabel("amplitude (µV)")
    ax_a.set_title("Amplitude", fontsize=10)

    return fig


def _psth_matrix(uid: UIDIntermediate, key: str) -> Optional[tuple]:
    """Stack per-session PSTH rows into (n_sessions, n_bins) + bin_centers + stages.

    Returns (matrix, centers, stages, n_trials_per_session) or None if every session is empty.
    """
    rows, centers, stages, n_trials = [], None, [], []
    for rec in uid.sessions:
        psth, c, n = rec.psths.get(key, (None, None, 0))
        if psth is None:
            continue
        rows.append(psth)
        centers = c
        stages.append(rec.stage)
        n_trials.append(n)
    if not rows:
        return None
    return np.vstack(rows), centers, stages, n_trials


def _draw_heatmap(ax, uid: UIDIntermediate, key: str, title: str,
                  *, dropped_indices: Optional[List[int]] = None) -> None:
    """Render the chronological PSTH heatmap into `ax`.  No inset overlay.

    If dropped_indices is supplied, draw a thin red rectangle just to the
    LEFT of each dropped row to flag sessions excluded by find_stable_subset.
    Note: the heatmap row index corresponds to the sequence of sessions that
    *had trials for this key*, which may be shorter than uid.sessions.  We
    re-derive the mapping (uid-session-index -> heatmap-row-index) here so
    the markers line up with the row they refer to.
    """
    data = _psth_matrix(uid, key)
    if data is None:
        ax.text(0.5, 0.5, f"no trials for {key}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        return

    mat, centers, _stages, _ = data
    vmax = np.percentile(mat, 99)
    ax.imshow(mat, aspect="auto", origin="upper", cmap="magma",
              extent=[centers[0], centers[-1], mat.shape[0], 0],
              vmin=0, vmax=max(vmax, 1e-6))
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("session #")

    if dropped_indices:
        # Build uid_idx -> heatmap_row_idx mapping using the same filter as
        # _psth_matrix (only rows where psths[key] is non-None).
        heatmap_row = 0
        uid_to_row = {}
        for uid_idx, rec in enumerate(uid.sessions):
            psth, _c, _n = rec.psths.get(key, (None, None, 0))
            if psth is None:
                continue
            uid_to_row[uid_idx] = heatmap_row
            heatmap_row += 1
        x0, x1 = centers[0], centers[-1]
        pad = 0.03 * (x1 - x0)
        for uid_idx in dropped_indices:
            row = uid_to_row.get(uid_idx)
            if row is None:
                continue
            ax.add_patch(Rectangle((x0 - pad, row), pad, 1.0,
                                    facecolor="red", edgecolor="none",
                                    clip_on=False, zorder=5))
        # Extend the visible x-range slightly so the red stripe is not clipped
        ax.set_xlim(x0 - pad, x1)
        ax.text(0.02, 0.97, "red bar = dropped row",
                transform=ax.transAxes, fontsize=7, color="white",
                ha="left", va="top",
                bbox=dict(facecolor="0.1", edgecolor="none", alpha=0.5, pad=2))


def _draw_psth_summary(ax, uid: UIDIntermediate, key: str,
                        miss_keys: Optional[List[str]] = None) -> None:
    """Render L vs E stage-mean PSTH traces into `ax` as a normal (white) plot.

    miss_keys (optional): list of keys whose stage-mean traces to overlay as
    dashed lines for hit/miss comparison (e.g. ["change_on_big_miss"]).
    """
    data = _psth_matrix(uid, key)
    if data is None:
        ax.text(0.5, 0.5, f"no trials for {key}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
        ax.set_axis_off()
        return

    mat, centers, stages, _ = data
    has_label = False
    for st in STAGE_ORDER:
        mask = np.array([s == st for s in stages])
        if mask.sum() == 0:
            continue
        label_solid = f"{st} hit" if miss_keys else st
        ax.plot(centers, mat[mask].mean(axis=0), color=STAGE_COLORS_LOCAL[st],
                linewidth=1.2, label=label_solid)
        has_label = True

    if miss_keys:
        for mk in miss_keys:
            mdata = _psth_matrix(uid, mk)
            if mdata is None:
                continue
            mmat, mcenters, mstages, _ = mdata
            for st in STAGE_ORDER:
                mask = np.array([s == st for s in mstages])
                if mask.sum() == 0:
                    continue
                ax.plot(mcenters, mmat[mask].mean(axis=0),
                        color=STAGE_COLORS_LOCAL[st], linewidth=1.0,
                        linestyle="--", alpha=0.7,
                        label=f"{st} miss")
                has_label = True

    ax.axvline(0, color="0.5", linewidth=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    ax.tick_params(labelsize=8)
    if has_label:
        ax.legend(loc="upper right", fontsize=6 if miss_keys else 7, frameon=False)


def render_page2(uid: UIDIntermediate, isi_score: float, depth_std: float,
                 wave_corr: float, fr_cv_val: float,
                 *,
                 dropped_indices: Optional[List[int]] = None,
                 n_kept: Optional[int] = None,
                 trimmed_verdict: Optional[str] = None) -> plt.Figure:
    """Render page 2 (physical) — returns the Figure.

    Layout (5 rows × 2 cols master): each row pairs a heatmap (left) with its
    L vs E PSTH-summary panel (right) at matched row heights.
      Row 0: header (spans both cols)
      Row 1: Baseline_ON heatmap   | Baseline_ON L-vs-E PSTH
      Row 2: Change_ON Big-Hit     | Big-Hit PSTH (+ Big-Miss dashed)
      Row 3: Change_ON Small-Hit   | Small-Hit PSTH (+ Small-Miss dashed)
      Row 4: Hit-lick heatmap      | Hit-lick PSTH

    Trim-visualization: dropped sessions get a red marker on the LEFT edge
    of their heatmap row.  PSTH-summary panels are unchanged (they aggregate
    across sessions; trim viz isn't well-defined there).
    """
    fig = plt.figure(figsize=(8.5, 11.0))
    gs = gridspec.GridSpec(
        nrows=5, ncols=2,
        height_ratios=[1.25, 1.75, 1.75, 1.75, 1.75],
        width_ratios=[1, 1],
        hspace=0.70, wspace=0.30,
        top=0.96, bottom=0.05, left=0.09, right=0.96,
        figure=fig,
    )

    # ── Row 0: Header (spans both columns)
    ax_hdr = fig.add_subplot(gs[0, :])
    draw_header(ax_hdr, uid, isi_score, depth_std, wave_corr, fr_cv_val,
                dropped_indices=dropped_indices,
                n_kept=n_kept, trimmed_verdict=trimmed_verdict)

    # ── Row 1: Baseline_ON
    _draw_heatmap(
        fig.add_subplot(gs[1, 0]), uid, "baseline_on",
        title="PSTH · Baseline_ON · all outcomes pooled [TODO: split by outcome in v2]",
        dropped_indices=dropped_indices,
    )
    _draw_psth_summary(
        fig.add_subplot(gs[1, 1]), uid, "baseline_on",
    )

    # ── Row 2: Change_ON Big-Hit (+ Big-Miss dashed overlay in summary)
    _draw_heatmap(
        fig.add_subplot(gs[2, 0]), uid, "change_on_big_hit",
        title="Change_ON · Big-Hit (2.0× + 4.0×)",
        dropped_indices=dropped_indices,
    )
    _draw_psth_summary(
        fig.add_subplot(gs[2, 1]), uid, "change_on_big_hit",
        miss_keys=["change_on_big_miss"],
    )

    # ── Row 3: Change_ON Small-Hit (+ Small-Miss dashed overlay in summary)
    _draw_heatmap(
        fig.add_subplot(gs[3, 0]), uid, "change_on_sm_hit",
        title="Change_ON · Small-Hit (1.25× + 1.35×)",
        dropped_indices=dropped_indices,
    )
    _draw_psth_summary(
        fig.add_subplot(gs[3, 1]), uid, "change_on_sm_hit",
        miss_keys=["change_on_sm_miss"],
    )

    # ── Row 4: Hit-lick
    _draw_heatmap(
        fig.add_subplot(gs[4, 0]), uid, "hit_lick",
        title="PSTH · Hit lick",
        dropped_indices=dropped_indices,
    )
    _draw_psth_summary(
        fig.add_subplot(gs[4, 1]), uid, "hit_lick",
    )

    return fig


def write_uid_pdf(out_path: Path, uid: UIDIntermediate,
                  um_pair_scores: Optional[np.ndarray],
                  isi_score: float, depth_std: float,
                  wave_corr: float, fr_cv_val: float,
                  *,
                  dropped_indices: Optional[List[int]] = None,
                  n_kept: Optional[int] = None,
                  trimmed_verdict: Optional[str] = None) -> str:
    """Write the 2-page PDF; return the composite verdict string."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        f1 = render_page1(uid, um_pair_scores, isi_score, depth_std, wave_corr, fr_cv_val,
                          dropped_indices=dropped_indices,
                          n_kept=n_kept, trimmed_verdict=trimmed_verdict)
        pdf.savefig(f1); plt.close(f1)
        f2 = render_page2(uid, isi_score, depth_std, wave_corr, fr_cv_val,
                          dropped_indices=dropped_indices,
                          n_kept=n_kept, trimmed_verdict=trimmed_verdict)
        pdf.savefig(f2); plt.close(f2)
    # Re-run the composite using the same inputs (cheap; keeps the API tidy)
    from visdetect.analysis.tracking_qc import (
        badge_isi, badge_depth, badge_waveform, badge_fr, composite_verdict,
    )
    return composite_verdict([
        badge_isi(isi_score), badge_depth(depth_std),
        badge_waveform(wave_corr), badge_fr(fr_cv_val),
    ])
