#!/usr/bin/env python
"""Interactive TF cell labeling GUI — keyboard-driven manual classification.

Displays z-scored traces, splitter mirror overlay, spike density heatmaps,
and raster plots with PSTHs for each unit.  The reviewer assigns a tier
(Splitter / Unilateral / Omni / Non-responsive) and sub-type using keyboard
shortcuts.  Labels are auto-saved after every assignment.

Usage
-----
    py scripts/tf_labeling/run_labeling_gui.py
    py scripts/tf_labeling/run_labeling_gui.py --reviewer BG
    py scripts/tf_labeling/run_labeling_gui.py --include-labeled   # re-review mode

Keyboard shortcuts
------------------
    1          Tier 1 (Splitter)  → then: f = Fast+/Slow-, s = Slow+/Fast-
    2          Tier 2 (Unilateral) → then: f/F = Fast+/-, s/S = Slow+/-
    3          Tier 3 (Omni)      → then: + = Both+, - = Both-
    0          Non-responsive
    j / Right  Next unit
    k / Left   Previous unit
    n          Add a note (opens text input)
    c          Cycle confidence (high → medium → low → high)
    h          Toggle help overlay
    q          Quit

The GUI can run without pre-cached rasters (raster panels will show
"No raster cache — run precache_rasters.py"). Pre-caching is recommended
for smooth experience.
"""
import argparse
import os
import sys
import textwrap

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
# Use an interactive backend — TkAgg is standard on Windows with CPython.
# Must be set before importing pyplot.
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Ensure the project is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_script_dir))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from visdetect.analysis.tf_labeling import (
    VALID_TIERS, VALID_SUB_TYPES, TIER_COLORS,
    TIER_SPLITTER, TIER_UNILATERAL, TIER_OMNI, TIER_NONE,
    LabelRecord, load_labels, save_label, get_label_stats,
    get_labeling_queue, load_unit_traces, load_unit_rasters,
)

# Re-assert interactive backend. Some imports in the visdetect chain
# (e.g. qc.py, tf_pulse.py, unit_selection.py) call matplotlib.use("Agg")
# at module level, which overrides our TkAgg setting. Force it back.
matplotlib.use("TkAgg", force=True)
plt.switch_backend("TkAgg")

# Fail fast if the interactive backend didn't stick
_backend = matplotlib.get_backend()
if "agg" in _backend.lower() and "tk" not in _backend.lower():
    raise RuntimeError(
        f"Failed to activate interactive backend (got '{_backend}'). "
        "Ensure tkinter is installed: py -c \"import tkinter\""
    )

# ── Constants ──────────────────────────────────────────────────────────
FAST_COLOR = "#1f77b4"  # blue
SLOW_COLOR = "#d62728"  # red
BG_COLOR = "#fafafa"
PANEL_BG = "#ffffff"

# Tightened x-axis: focus on the response window, not empty flanks
X_LIM_MS = (-200, 400)

# Trace smoothing (matches analysis_suite SIGMA_SMOOTH=5 bins at 1 ms)
TRACE_SMOOTH_SIGMA = 5
CI_Z = 1.96  # 95% CI multiplier

# Raster subsampling
MAX_RASTER_DISPLAY = 500
RASTER_PRE_WINDOW = (-0.200, 0.0)     # seconds, baseline for ranking pulses
RASTER_POST_WINDOW = (0.0, 0.250)     # seconds, response for ranking pulses
PSTH_BIN_MS = 5                        # finer PSTH bins

# Spike density heatmap
DENSITY_BIN_MS = 25       # time bin for density (ms)
DENSITY_TRIAL_BIN = 50    # number of pulses per row in the heatmap
# Diverging colormap for baseline-subtracted density (blue = decrease, red = increase)
DENSITY_CMAP = "RdBu_r"

HELP_TEXT = """\
KEYBOARD SHORTCUTS
--------------------------
  1   Splitter (Tier 1)
  2   Unilateral (Tier 2)
  3   Omni (Tier 3)
  0   Non-responsive

  After tier key:
    f  Fast+/Slow- or Fast+
    s  Slow+/Fast- or Slow+
    F  Fast-
    S  Slow-
    +  Both+
    -  Both-

  j / Right  Next unit
  k / Left   Previous unit
  d          Toggle detrended view
  v          Toggle t=0 lines (blind mode)
  c          Cycle confidence
  n          Add note
  h          Toggle help
  q          Quit
"""

# Sub-type shortcut mapping for each tier
SUBTYPE_KEYS = {
    TIER_SPLITTER: {
        "f": "Fast+/Slow-",
        "s": "Slow+/Fast-",
    },
    TIER_UNILATERAL: {
        "f": "Fast+",
        "F": "Fast-",
        "s": "Slow+",
        "S": "Slow-",
    },
    TIER_OMNI: {
        "+": "Both+",
        "=": "Both+",  # unshifted + on most keyboards
        "-": "Both-",
    },
    TIER_NONE: {},
}


# ── Spike density helper ──────────────────────────────────────────────

def _compute_spike_density(raster, t_range_s, bin_ms=DENSITY_BIN_MS,
                           trial_bin=DENSITY_TRIAL_BIN):
    """Compute a 2D spike density heatmap from a raster (list of arrays).

    Returns
    -------
    density : 2D array (trial_groups × time_bins), firing rate in Hz
    t_edges : time bin edges in ms
    """
    t_lo, t_hi = t_range_s
    t_lo_ms, t_hi_ms = t_lo * 1000, t_hi * 1000
    n_time_bins = max(1, int((t_hi_ms - t_lo_ms) / bin_ms))
    t_edges = np.linspace(t_lo_ms, t_hi_ms, n_time_bins + 1)
    n_trials = len(raster)
    n_groups = max(1, n_trials // trial_bin)

    density = np.zeros((n_groups, n_time_bins), dtype=float)
    for i, spk in enumerate(raster):
        g = min(i // trial_bin, n_groups - 1)
        if len(spk) > 0:
            counts, _ = np.histogram(spk * 1000, bins=t_edges)
            density[g] += counts

    # Normalize: each group has `trial_bin` trials (last may have fewer)
    for g in range(n_groups):
        g_start = g * trial_bin
        g_end = min(g_start + trial_bin, n_trials)
        n_in_group = g_end - g_start
        if n_in_group > 0:
            density[g] /= (n_in_group * bin_ms / 1000)  # → Hz

    return density, t_edges


def _compute_detrended_traces(traces):
    """Remove linear baseline trend from z-scored traces.

    Delegates to :func:`visdetect.analysis.tf_pulse.detrend_tf_traces`
    for the linear-detrend fitting.  The GUI displays the full detrended
    trace, so we pass the detrended array back in a dict matching the
    input format.
    """
    from visdetect.analysis.tf_pulse import detrend_tf_traces

    t_vec = traces["t_vec"]
    result = {
        "t_vec": t_vec,
        "z_max_fast": traces.get("z_max_fast", 0),
        "z_max_slow": traces.get("z_max_slow", 0),
    }

    for direction in ("fast", "slow"):
        z = traces[f"{direction}_z"]
        sem = traces[f"{direction}_z_sem"].copy()

        # Library expects (n_units, n_time) — wrap single trace as 2D
        z_2d = z[np.newaxis, :] if z.ndim == 1 else z
        detrended_2d, _, _ = detrend_tf_traces(
            t_vec, z_2d,
            baseline_window=(-0.4, -0.01),
            post_window=(0.0, 0.3),
        )
        result[f"{direction}_z"] = detrended_2d[0] if z.ndim == 1 else detrended_2d
        result[f"{direction}_z_sem"] = sem

    return result


def _compute_psth_from_raster(raster, t_range_s, bin_ms=PSTH_BIN_MS):
    """Compute a binned PSTH (Hz) from raster data, with Gaussian smoothing."""
    t_lo_ms, t_hi_ms = t_range_s[0] * 1000, t_range_s[1] * 1000
    n_bins = max(1, int((t_hi_ms - t_lo_ms) / bin_ms))
    edges = np.linspace(t_lo_ms, t_hi_ms, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    total_counts = np.zeros(n_bins, dtype=float)
    n_trials = len(raster)
    for spk in raster:
        if len(spk) > 0:
            counts, _ = np.histogram(spk * 1000, bins=edges)
            total_counts += counts
    if n_trials > 0:
        total_counts /= (n_trials * bin_ms / 1000)  # → Hz
    # Smooth for cleaner appearance
    total_counts = gaussian_filter1d(total_counts, sigma=2, mode="nearest")
    return centers, total_counts


def _smooth_trace(arr, sigma=TRACE_SMOOTH_SIGMA):
    """Gaussian-smooth a 1-D trace (matches analysis suite convention)."""
    return gaussian_filter1d(arr, sigma=sigma, mode="nearest")


def _select_best_pulses(raster_data, max_display=MAX_RASTER_DISPLAY):
    """Select pulses with largest rate modulation (|post_rate - pre_rate|).

    Ranks by absolute firing-rate change so that actual modulation is
    prioritised over baseline rate.  Captures both excitation and suppression.
    """
    pre_dur = RASTER_PRE_WINDOW[1] - RASTER_PRE_WINDOW[0]
    post_dur = RASTER_POST_WINDOW[1] - RASTER_POST_WINDOW[0]
    modulation = np.empty(len(raster_data))
    for i, spk in enumerate(raster_data):
        n_pre = np.sum((spk >= RASTER_PRE_WINDOW[0]) & (spk < RASTER_PRE_WINDOW[1]))
        n_post = np.sum((spk >= RASTER_POST_WINDOW[0]) & (spk < RASTER_POST_WINDOW[1]))
        modulation[i] = abs(n_post / post_dur - n_pre / pre_dur)
    if len(modulation) <= max_display:
        return list(raster_data), len(raster_data)
    top_idx = np.argsort(modulation)[-max_display:]
    top_idx.sort()  # preserve temporal order
    return [raster_data[i] for i in top_idx], len(raster_data)


class LabelingGUI:
    """Interactive matplotlib-based labeling tool."""

    def __init__(self, queue_df, reviewer="BG"):
        self.queue = queue_df.reset_index(drop=True)
        self.reviewer = reviewer
        self.idx = 0
        self.total = len(self.queue)

        # State
        self.pending_tier = None     # Tier selected, waiting for sub-type
        self.confidence = "high"
        self.notes = ""
        self.show_help = False
        self.show_detrended = True   # Default: show detrended; d = toggle to raw
        self.show_vlines = True      # Toggle: v = hide/show t=0 dashed lines
        self._cached_traces = None   # NPZ traces for current unit
        self._cached_rasters = None  # Raster data for current unit
        self._cached_detrended = None  # Detrended traces (computed on demand)
        self._cached_row = None      # Current queue row

        # Load existing labels for display
        self.labels_df = load_labels()

        # ── Layout ─────────────────────────────────────────────────
        # Proportions: traces are tall, splitter shorter, rasters+density
        # at the bottom.  Right 20% is the info panel.
        self.fig = plt.figure(figsize=(8, 15), facecolor=BG_COLOR)
        self.fig.canvas.manager.set_window_title("TF Cell Labeling GUI")

        # Main grid: 4 rows.  Narrow figure + tall rows = dynamics
        # are visible.  Right ~38% is the info panel.
        self.gs = GridSpec(
            4, 1, figure=self.fig,
            left=0.10, right=0.60, top=0.93, bottom=0.04,
            height_ratios=[3, 2, 2.5, 1.5],
            hspace=0.30,
        )

        # Row 0: Z-scored traces (full width)
        self.ax_ztrace = self.fig.add_subplot(self.gs[0])

        # Row 1: Splitter mirror overlay (full width)
        self.ax_mirror = self.fig.add_subplot(self.gs[1])

        # Row 2: Fast raster | Slow raster (side by side, equal width)
        gs_raster = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=self.gs[2],
            height_ratios=[4, 1], hspace=0.05, wspace=0.25)
        self.ax_raster_fast = self.fig.add_subplot(gs_raster[0, 0])
        self.ax_psth_fast = self.fig.add_subplot(gs_raster[1, 0],
                                                  sharex=self.ax_raster_fast)
        self.ax_raster_slow = self.fig.add_subplot(gs_raster[0, 1])
        self.ax_psth_slow = self.fig.add_subplot(gs_raster[1, 1],
                                                  sharex=self.ax_raster_slow)

        # Row 3: Fast density heatmap | Slow density heatmap
        gs_density = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=self.gs[3], wspace=0.25)
        self.ax_density_fast = self.fig.add_subplot(gs_density[0, 0])
        self.ax_density_slow = self.fig.add_subplot(gs_density[0, 1])

        # Info panel (right side)
        self.ax_info = self.fig.add_axes([0.62, 0.04, 0.36, 0.89],
                                         facecolor=BG_COLOR)
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        for spine in self.ax_info.spines.values():
            spine.set_visible(False)

        # Help overlay axis (hidden by default)
        self.ax_help = self.fig.add_axes([0.25, 0.15, 0.50, 0.70],
                                         facecolor="white", zorder=100)
        self.ax_help.set_visible(False)

        # Disable ALL default matplotlib keybindings that conflict with
        # our labeling shortcuts (s=save, f=fullscreen, h=home, k=xscale,
        # q=quit, left/right=back/fwd, g=grid, l=yscale, etc.)
        for param in list(plt.rcParams):
            if param.startswith("keymap."):
                plt.rcParams[param] = []

        # Connect keyboard
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Initial draw
        self._draw_unit()

    # ── Drawing ────────────────────────────────────────────────────

    def _draw_unit(self):
        """Render the current unit across all panels."""
        if self.idx >= self.total:
            self._show_done()
            return

        row = self.queue.iloc[self.idx]
        session_name = int(row["session_name"])
        cluster_id = int(row["cluster_id"])

        # Reset state for new unit
        self.pending_tier = None
        self.confidence = "high"
        self.notes = ""
        self.show_detrended = True  # default: detrended view

        # Load and cache data for this unit
        traces = load_unit_traces(session_name, cluster_id)
        rasters = load_unit_rasters(session_name, cluster_id)
        self._cached_traces = traces
        self._cached_rasters = rasters
        self._cached_detrended = (_compute_detrended_traces(traces)
                                   if traces is not None else None)
        self._cached_row = row

        # Default display: detrended
        display_traces = self._cached_detrended if traces is not None else traces

        # ── Panel 1: Z-scored traces (or detrended) ─────────────────
        ax = self.ax_ztrace
        ax.clear()
        if display_traces is not None:
            t = display_traces["t_vec"] * 1000  # ms
            fz = _smooth_trace(display_traces["fast_z"])
            sz = _smooth_trace(display_traces["slow_z"])
            fz_ci = _smooth_trace(display_traces["fast_z_sem"]) * CI_Z
            sz_ci = _smooth_trace(display_traces["slow_z_sem"]) * CI_Z

            ax.fill_between(t, fz - fz_ci, fz + fz_ci,
                            alpha=0.18, color=FAST_COLOR, linewidth=0)
            ax.fill_between(t, sz - sz_ci, sz + sz_ci,
                            alpha=0.18, color=SLOW_COLOR, linewidth=0)
            ax.plot(t, fz, color=FAST_COLOR, lw=1.5, label="Fast")
            ax.plot(t, sz, color=SLOW_COLOR, lw=1.5, label="Slow")
            if self.show_vlines:
                ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)
            ax.set_ylabel("z-score", fontsize=9)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
            # Fit y-axis to mean traces in visible window (not CI bands)
            vis = (t >= X_LIM_MS[0]) & (t <= X_LIM_MS[1])
            if vis.any():
                yvals = np.concatenate([fz[vis], sz[vis]])
                ylo, yhi = np.nanmin(yvals), np.nanmax(yvals)
                pad = max((yhi - ylo) * 0.25, 1.0)
                ax.set_ylim(ylo - pad, yhi + pad)
            if self.show_detrended:
                ax.set_title("DETRENDED: per-pulse baseline corrected  [d=toggle]",
                             fontsize=10, fontweight="bold", loc="left",
                             color="#b71c1c")
            else:
                ax.set_title("Z-scored TF pulse traces  [d=toggle detrend]",
                             fontsize=10, fontweight="bold", loc="left")
        else:
            ax.text(0.5, 0.5, "No NPZ trace data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#999")
        ax.set_xlim(*X_LIM_MS)

        # ── Panel 2: Mirror overlay (Fast vs -Slow) ──────────────
        ax = self.ax_mirror
        ax.clear()
        if display_traces is not None:
            t = display_traces["t_vec"] * 1000
            fz = _smooth_trace(display_traces["fast_z"])
            sz = _smooth_trace(display_traces["slow_z"])
            ax.plot(t, fz, color=FAST_COLOR, lw=1.5, label="Fast")
            ax.plot(t, -sz, color=SLOW_COLOR, lw=1.5, ls="--",
                    label="-Slow (mirrored)")
            if self.show_vlines:
                ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)
            ax.set_ylabel("z-score", fontsize=9)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
            # Fit y-axis to mean traces in visible window
            vis = (t >= X_LIM_MS[0]) & (t <= X_LIM_MS[1])
            if vis.any():
                yvals = np.concatenate([fz[vis], -sz[vis]])
                ylo, yhi = np.nanmin(yvals), np.nanmax(yvals)
                pad = max((yhi - ylo) * 0.25, 1.0)
                ax.set_ylim(ylo - pad, yhi + pad)
            mirror_r = row.get("mirror_score", np.nan)
            mirror_str = f"  (r={mirror_r:.2f})" if not np.isnan(mirror_r) else ""
            ax.set_title(f"Splitter test: Fast vs -Slow{mirror_str}",
                         fontsize=10, fontweight="bold", loc="left")
        else:
            ax.text(0.5, 0.5, "No trace data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#999")
        ax.set_xlim(*X_LIM_MS)

        # ── Panels 3-4: Rasters + PSTH strips ────────────────────
        self._draw_raster_panel(
            self.ax_raster_fast, self.ax_psth_fast, rasters, "fast")
        self._draw_raster_panel(
            self.ax_raster_slow, self.ax_psth_slow, rasters, "slow")
        # ── Panels 5-6: Spike density heatmaps ───────────────────
        self._draw_density_panel(self.ax_density_fast, rasters, "fast")
        self._draw_density_panel(self.ax_density_slow, rasters, "slow")

        # ── Info panel ────────────────────────────────────────────
        self._draw_info_panel(row)

        # ── Title bar ─────────────────────────────────────────────
        stats = get_label_stats()
        n_done = stats["total"]
        tier_str = "  ".join(
            f"{t.split('(')[1].rstrip(')') if '(' in t else t}: {c}"
            for t, c in stats.get("by_tier", {}).items()
        )
        self.fig.suptitle(
            f"Unit {self.idx + 1} / {self.total}  |  "
            f"Labeled: {n_done}  |  {tier_str}",
            fontsize=11, fontweight="bold", y=0.97)

        self.fig.canvas.draw_idle()

    def _draw_raster_panel(self, ax_raster, ax_psth, rasters, direction):
        """Draw a raster + PSTH strip for one direction (fast or slow)."""
        ax_raster.clear()
        ax_psth.clear()

        key = f"{direction}_raster"
        n_key = f"n_{direction}_pulses"

        if rasters is not None and key in rasters:
            raster_data = rasters[key]
            t_range = rasters["t_range"]
            n_pulses = rasters[n_key]

            # Subsample to best-responding pulses for the raster display
            display_raster, total_n = _select_best_pulses(raster_data)
            n_display = len(display_raster)

            # Raster: eventplot (efficient single call)
            spike_times_ms = [spk * 1000 for spk in display_raster]
            ax_raster.eventplot(
                spike_times_ms, colors="k", linewidths=0.5,
                linelengths=0.8)
            if self.show_vlines:
                ax_raster.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax_raster.set_ylabel("Pulse #", fontsize=8)
            if n_display < total_n:
                title = (f"{direction.capitalize()} raster "
                         f"(top {n_display}/{total_n} pulses)")
            else:
                title = (f"{direction.capitalize()} raster "
                         f"({total_n} pulses)")
            ax_raster.set_title(title, fontsize=9, fontweight="bold",
                                loc="left")
            ax_raster.set_xlim(*X_LIM_MS)
            plt.setp(ax_raster.get_xticklabels(), visible=False)
            ax_raster.tick_params(axis='x', length=0)

            # PSTH strip below (computed from ALL pulses, no bias)
            centers, rate = _compute_psth_from_raster(raster_data, t_range)
            color = FAST_COLOR if direction == "fast" else SLOW_COLOR
            ax_psth.fill_between(centers, 0, rate, alpha=0.3, color=color)
            ax_psth.plot(centers, rate, color=color, lw=1.0)
            if self.show_vlines:
                ax_psth.axvline(0, color="k", ls="--", lw=0.6, alpha=0.4)
            ax_psth.set_ylabel("Hz", fontsize=7)
            ax_psth.set_xlabel("Time from pulse (ms)", fontsize=8)
            ax_psth.set_xlim(*X_LIM_MS)
            ax_psth.tick_params(labelsize=7)
        else:
            ax_raster.text(0.5, 0.5, "No raster cache\nrun precache_rasters.py",
                           ha="center", va="center", transform=ax_raster.transAxes,
                           fontsize=9, color="#999")
            ax_raster.set_xlim(*X_LIM_MS)
            ax_psth.set_xlim(*X_LIM_MS)
            ax_psth.set_xlabel("Time from pulse (ms)", fontsize=8)

    def _draw_density_panel(self, ax, rasters, direction):
        """Draw a baseline-subtracted spike density heatmap for one direction."""
        ax.clear()
        key = f"{direction}_raster"
        if rasters is not None and key in rasters:
            raster_data = rasters[key]
            t_range = rasters["t_range"]

            density, t_edges = _compute_spike_density(
                raster_data, t_range,
                bin_ms=DENSITY_BIN_MS, trial_bin=DENSITY_TRIAL_BIN)

            if density.size > 0:
                t_centers = (t_edges[:-1] + t_edges[1:]) / 2
                mask = (t_centers >= X_LIM_MS[0]) & (t_centers <= X_LIM_MS[1])
                density_clipped = density[:, mask]
                t_clipped = t_centers[mask]

                if density_clipped.size > 0:
                    # Baseline-subtract per row: subtract mean of pre-period bins
                    pre_mask = t_clipped < 0
                    if pre_mask.any():
                        baseline = density_clipped[:, pre_mask].mean(axis=1,
                                                                     keepdims=True)
                        density_bs = density_clipped - baseline
                    else:
                        density_bs = density_clipped

                    # Symmetric percentile clipping
                    abs_vals = np.abs(density_bs)
                    vmax = np.percentile(abs_vals[abs_vals > 0], 95) \
                        if np.any(abs_vals > 0) else 1.0
                    vmax = max(vmax, 0.5)  # floor to avoid degenerate colormap

                    im = ax.imshow(
                        density_bs,
                        aspect="auto", origin="lower",
                        extent=[t_clipped[0] - DENSITY_BIN_MS / 2,
                                t_clipped[-1] + DENSITY_BIN_MS / 2,
                                0, density.shape[0]],
                        cmap=DENSITY_CMAP,
                        vmin=-vmax, vmax=vmax,
                        interpolation="bilinear")
                    if self.show_vlines:
                        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)

            label = direction.capitalize()
            ax.set_title(f"{label} density (baseline-sub)",
                         fontsize=9, fontweight="bold", loc="left")
            ax.set_xlabel("Time from pulse (ms)", fontsize=8)
            ax.set_ylabel(f"Pulse group\n({DENSITY_TRIAL_BIN}/grp)",
                          fontsize=7)
            ax.set_xlim(*X_LIM_MS)
            ax.tick_params(labelsize=7)
        else:
            ax.text(0.5, 0.5, "No raster cache",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#999")
            ax.set_xlim(*X_LIM_MS)

    def _draw_info_panel(self, row):
        """Draw metadata + controls on the right panel."""
        ax = self.ax_info
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        session_name = int(row["session_name"])
        cluster_id = int(row["cluster_id"])
        algo_tier = row.get("tier", "?")
        algo_sub = row.get("sub_type", "?")
        stage = row.get("stage", "?")
        tier_color = TIER_COLORS.get(algo_tier, "#999")

        lines = [
            ("SESSION", f"{session_name}", 0.95),
            ("CLUSTER", f"{cluster_id}", 0.90),
            ("STAGE", f"{stage}", 0.85),
            ("", "", 0.82),
            ("ALGO TIER", algo_tier.replace("Tier ", "T"), 0.78),
            ("ALGO SUB", algo_sub, 0.73),
            ("", "", 0.70),
        ]

        # Key features
        features = [
            ("Peak Fast", f"{row.get('peak_fast', 0):.2f}"),
            ("Peak Slow", f"{row.get('peak_slow', 0):.2f}"),
            ("AUC Fast", f"{row.get('auc_fast', 0):.2f}"),
            ("AUC Slow", f"{row.get('auc_slow', 0):.2f}"),
            ("Mirror", f"{row.get('mirror_score', 0):.2f}"),
            ("Trend", f"{row.get('trend_ratio', 0):.2f}"),
            ("p Fast", f"{row.get('p_peak_fast', 1):.3f}"),
            ("p Slow", f"{row.get('p_peak_slow', 1):.3f}"),
            ("# Fast", f"{int(row.get('n_fast_pulses', 0))}"),
            ("# Slow", f"{int(row.get('n_slow_pulses', 0))}"),
            ("Priority", f"{row.get('priority', 0):.0f}"),
        ]

        for label, value, y_pos in lines:
            if label:
                ax.text(0.05, y_pos, label, fontsize=8.5, color="#666",
                        fontweight="bold", transform=ax.transAxes)
                color = tier_color if "ALGO" in label else "k"
                ax.text(0.95, y_pos, value, fontsize=11, color=color,
                        ha="right", transform=ax.transAxes)

        y = 0.65
        ax.text(0.05, y, "FEATURES", fontsize=8.5, color="#666",
                fontweight="bold", transform=ax.transAxes)
        y -= 0.03
        for fname, fval in features:
            y -= 0.032
            ax.text(0.08, y, fname, fontsize=9, color="#444",
                    transform=ax.transAxes)
            ax.text(0.95, y, fval, fontsize=9, color="k", ha="right",
                    fontfamily="monospace", transform=ax.transAxes)

        # Current assignment state
        y -= 0.06
        ax.plot([0.05, 0.95], [y + 0.01, y + 0.01], "-", color="#ddd",
                transform=ax.transAxes, lw=0.5)
        ax.text(0.05, y - 0.02, "CURRENT LABEL", fontsize=8.5, color="#666",
                fontweight="bold", transform=ax.transAxes)
        if self.pending_tier:
            tier_short = self.pending_tier.split("(")[1].rstrip(")") \
                if "(" in self.pending_tier else self.pending_tier
            ax.text(0.95, y - 0.02, f"{tier_short} (pick sub-type...)",
                    fontsize=11, ha="right",
                    color=TIER_COLORS.get(self.pending_tier, "k"),
                    fontweight="bold", transform=ax.transAxes)
        else:
            ax.text(0.95, y - 0.02, "\u2014", fontsize=11, ha="right",
                    color="#999", transform=ax.transAxes)

        ax.text(0.05, y - 0.06, "CONFIDENCE", fontsize=8.5, color="#666",
                fontweight="bold", transform=ax.transAxes)
        conf_colors = {"high": "#2e7d32", "medium": "#ef6c00", "low": "#c62828"}
        ax.text(0.95, y - 0.06, self.confidence, fontsize=11, ha="right",
                color=conf_colors.get(self.confidence, "k"),
                fontweight="bold", transform=ax.transAxes)

        # Shortcuts reminder
        y -= 0.14
        ax.plot([0.05, 0.95], [y + 0.01, y + 0.01], "-", color="#ddd",
                transform=ax.transAxes, lw=0.5)
        shortcuts = [
            "1=Splitter  2=Uni  3=Omni  0=NR",
            "j/k or arrows = nav",
            "d=detrend  v=blind  c=confidence",
            "n=note  h=help  q=quit",
        ]
        for i, s in enumerate(shortcuts):
            ax.text(0.5, y - 0.03 - i * 0.03, s, fontsize=8, color="#888",
                    ha="center", transform=ax.transAxes)

    def _show_done(self):
        """Show completion screen."""
        all_axes = [
            self.ax_ztrace, self.ax_mirror,
            self.ax_raster_fast, self.ax_psth_fast,
            self.ax_raster_slow, self.ax_psth_slow,
            self.ax_density_fast, self.ax_density_slow,
            self.ax_info,
        ]
        for ax in all_axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax_ztrace.text(
            0.5, 0.5, "All units in queue have been reviewed!",
            ha="center", va="center", fontsize=16, fontweight="bold",
            transform=self.ax_ztrace.transAxes)
        stats = get_label_stats()
        self.ax_mirror.text(
            0.5, 0.5, f"Total labeled: {stats['total']}\n"
            + "\n".join(f"  {t}: {c}" for t, c in
                        stats.get("by_tier", {}).items()),
            ha="center", va="center", fontsize=12,
            transform=self.ax_mirror.transAxes)
        self.fig.canvas.draw_idle()

    def _toggle_detrended(self):
        """Toggle between standard z-scored and trend-corrected traces.

        Fits and subtracts a linear baseline trend from the NPZ traces.
        Only redraws the top two trace panels — rasters/density are unchanged.
        """
        if self._cached_traces is None:
            return

        self.show_detrended = not self.show_detrended

        if self.show_detrended:
            if self._cached_detrended is None:
                self._cached_detrended = _compute_detrended_traces(
                    self._cached_traces)
            display_traces = self._cached_detrended
        else:
            display_traces = self._cached_traces

        row = self._cached_row

        # Redraw only the two trace panels (fast, no full redraw)
        ax = self.ax_ztrace
        ax.clear()
        if display_traces is not None:
            t = display_traces["t_vec"] * 1000
            fz = _smooth_trace(display_traces["fast_z"])
            sz = _smooth_trace(display_traces["slow_z"])
            fz_ci = _smooth_trace(display_traces["fast_z_sem"]) * CI_Z
            sz_ci = _smooth_trace(display_traces["slow_z_sem"]) * CI_Z
            ax.fill_between(t, fz - fz_ci, fz + fz_ci,
                            alpha=0.18, color=FAST_COLOR, linewidth=0)
            ax.fill_between(t, sz - sz_ci, sz + sz_ci,
                            alpha=0.18, color=SLOW_COLOR, linewidth=0)
            ax.plot(t, fz, color=FAST_COLOR, lw=1.5, label="Fast")
            ax.plot(t, sz, color=SLOW_COLOR, lw=1.5, label="Slow")
            if self.show_vlines:
                ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)
            ax.set_ylabel("z-score", fontsize=9)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
            # Fit y-axis to mean traces in visible window
            vis = (t >= X_LIM_MS[0]) & (t <= X_LIM_MS[1])
            if vis.any():
                yvals = np.concatenate([fz[vis], sz[vis]])
                ylo, yhi = np.nanmin(yvals), np.nanmax(yvals)
                pad = max((yhi - ylo) * 0.25, 1.0)
                ax.set_ylim(ylo - pad, yhi + pad)
            if self.show_detrended:
                ax.set_title(
                    "DETRENDED: per-pulse baseline corrected  [d=toggle]",
                    fontsize=10, fontweight="bold", loc="left", color="#b71c1c")
            else:
                ax.set_title("Z-scored TF pulse traces  [d=toggle detrend]",
                             fontsize=10, fontweight="bold", loc="left")
        ax.set_xlim(*X_LIM_MS)

        ax = self.ax_mirror
        ax.clear()
        if display_traces is not None:
            t = display_traces["t_vec"] * 1000
            fz = _smooth_trace(display_traces["fast_z"])
            sz = _smooth_trace(display_traces["slow_z"])
            ax.plot(t, fz, color=FAST_COLOR, lw=1.5, label="Fast")
            ax.plot(t, -sz, color=SLOW_COLOR, lw=1.5, ls="--",
                    label="-Slow (mirrored)")
            if self.show_vlines:
                ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)
            ax.set_ylabel("z-score", fontsize=9)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
            # Fit y-axis to mean traces in visible window
            vis = (t >= X_LIM_MS[0]) & (t <= X_LIM_MS[1])
            if vis.any():
                yvals = np.concatenate([fz[vis], -sz[vis]])
                ylo, yhi = np.nanmin(yvals), np.nanmax(yvals)
                pad = max((yhi - ylo) * 0.25, 1.0)
                ax.set_ylim(ylo - pad, yhi + pad)
            mirror_r = row.get("mirror_score", np.nan) if row is not None \
                else np.nan
            mirror_str = f"  (r={mirror_r:.2f})" if not np.isnan(mirror_r) \
                else ""
            prefix = "DETRENDED " if self.show_detrended else ""
            ax.set_title(
                f"{prefix}Splitter test: Fast vs -Slow{mirror_str}",
                fontsize=10, fontweight="bold", loc="left",
                color="#b71c1c" if self.show_detrended else "k")
        ax.set_xlim(*X_LIM_MS)

        self.fig.canvas.draw_idle()

    def _toggle_help(self):
        """Show/hide help overlay."""
        self.show_help = not self.show_help
        self.ax_help.set_visible(self.show_help)
        if self.show_help:
            self.ax_help.clear()
            self.ax_help.set_xticks([])
            self.ax_help.set_yticks([])
            self.ax_help.set_facecolor("white")
            self.ax_help.patch.set_alpha(0.95)
            for spine in self.ax_help.spines.values():
                spine.set_edgecolor("#ddd")
            self.ax_help.text(
                0.05, 0.95, HELP_TEXT, fontsize=10, fontfamily="monospace",
                verticalalignment="top", transform=self.ax_help.transAxes)
        self.fig.canvas.draw_idle()

    # ── Keyboard handler ───────────────────────────────────────────

    def _on_key(self, event):
        key = event.key

        # Help toggle
        if key == "h":
            self._toggle_help()
            return

        # Dismiss help if showing
        if self.show_help and key != "h":
            self._toggle_help()
            return

        # Quit
        if key == "q":
            plt.close(self.fig)
            return

        # Navigation
        if key in ("j", "right"):
            self._navigate(+1)
            return
        if key in ("k", "left"):
            self._navigate(-1)
            return

        # Confidence cycle
        if key == "c":
            cycle = {"high": "medium", "medium": "low", "low": "high"}
            self.confidence = cycle.get(self.confidence, "high")
            self._draw_info_panel(self.queue.iloc[self.idx])
            self.fig.canvas.draw_idle()
            return

        # Detrended view toggle
        if key == "d" and self.pending_tier is None:
            self._toggle_detrended()
            return

        # Vertical line toggle (blind mode for unbiased review)
        if key == "v" and self.pending_tier is None:
            self.show_vlines = not self.show_vlines
            self._draw_unit()
            return

        # Note entry (simple text input via console)
        if key == "n":
            print("\n--- Enter note (press Enter to confirm) ---")
            try:
                note = input("Note: ").strip()
                self.notes = note
                print(f"Note saved: '{note}'")
            except (EOFError, KeyboardInterrupt):
                pass
            return

        # Tier selection
        if self.pending_tier is None:
            if key == "1":
                self.pending_tier = TIER_SPLITTER
            elif key == "2":
                self.pending_tier = TIER_UNILATERAL
            elif key == "3":
                self.pending_tier = TIER_OMNI
            elif key == "0":
                # Non-responsive: no sub-type, save immediately
                self._save_current(TIER_NONE, "None")
                return

            if self.pending_tier:
                # Update info panel to show pending tier
                self._draw_info_panel(self.queue.iloc[self.idx])
                self.fig.canvas.draw_idle()
            return

        # Sub-type selection (tier already chosen)
        subtype_map = SUBTYPE_KEYS.get(self.pending_tier, {})
        if key in subtype_map:
            self._save_current(self.pending_tier, subtype_map[key])
        elif key == "escape":
            # Cancel tier selection
            self.pending_tier = None
            self._draw_info_panel(self.queue.iloc[self.idx])
            self.fig.canvas.draw_idle()

    def _save_current(self, tier, sub_type):
        """Save label for current unit and advance."""
        row = self.queue.iloc[self.idx]
        record = LabelRecord(
            session_name=int(row["session_name"]),
            cluster_id=int(row["cluster_id"]),
            manual_tier=tier,
            manual_sub_type=sub_type,
            confidence=self.confidence,
            notes=self.notes,
            algo_tier=row.get("tier", ""),
            algo_sub_type=row.get("sub_type", ""),
            reviewer=self.reviewer,
        )
        save_label(record)
        tier_short = tier.split("(")[1].rstrip(")") if "(" in tier else tier
        print(f"  Labeled {row['session_name']}:{row['cluster_id']} -> "
              f"{tier_short}/{sub_type} [{self.confidence}]")

        # Advance to next
        self._navigate(+1)

    def _navigate(self, delta):
        """Move to a different unit in the queue."""
        new_idx = self.idx + delta
        if 0 <= new_idx < self.total:
            self.idx = new_idx
            self._draw_unit()
        elif new_idx >= self.total:
            self.idx = self.total
            self._show_done()

    def run(self):
        """Start the GUI event loop."""
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TF cell manual labeling GUI")
    parser.add_argument("--reviewer", default="BG",
                        help="Reviewer name (default: BG)")
    parser.add_argument("--include-labeled", action="store_true",
                        help="Include already-labeled units for re-review")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start at this index in the queue")
    args = parser.parse_args()

    print("Loading labeling queue...")
    queue = get_labeling_queue(include_labeled=args.include_labeled)
    print(f"Queue: {len(queue)} units to review")

    if queue.empty:
        print("Nothing to label! All units have been reviewed, or "
              "classification CSV is missing.")
        return

    stats = get_label_stats()
    if stats["total"] > 0:
        print(f"Existing labels: {stats['total']} "
              f"({stats.get('by_tier', {})})")

    gui = LabelingGUI(queue, reviewer=args.reviewer)
    if args.start_idx > 0:
        gui.idx = min(args.start_idx, gui.total - 1)
        gui._draw_unit()

    print("\nStarting GUI... (press 'h' for help, 'q' to quit)")
    gui.run()

    # Print final stats
    final = get_label_stats()
    print(f"\nFinal: {final['total']} labels saved")
    for t, c in final.get("by_tier", {}).items():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
