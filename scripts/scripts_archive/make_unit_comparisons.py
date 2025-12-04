from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parents[1]

# Ensure src importable
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from visdetect.core.session import load_session
from visdetect.analysis import su_analysis as su
from visdetect.analysis import align as align_mod

# Reuse internal helpers for inline plotting (keeps font vector quality)
from visdetect.analysis.su_analysis import _get_outcome_colors, _normalize_outcome_label, _spikes_relative_to_events


def safe_read_png(path: Path):
    try:
        return mpimg.imread(path.as_posix())
    except Exception:
        return None


def draw_image(ax, path: Path, title: str | None = None):
    img = safe_read_png(path)
    ax.axis('off')
    if img is None:
        if title:
            ax.set_title(f"{title} (missing)", fontsize=9)
        return
    # Fill available axes space; avoid "tiny image centered" effect from aspect='equal'
    ax.imshow(img, aspect='auto')
    if title:
        ax.set_title(title, fontsize=9)


def build_unit_comparison(session_pkl: Path, cluster_id: int, out_png: Path,
                           include: List[str], compact_scale: float, raster_line_height: float,
                           window=(-0.5, 1.0), bin_size=0.02,
                           stack_change_outcome: bool = True,
                           panel_scale: float = 1.0,
                           panel_dpi: int = 220,
                           inline: bool = False,
                           raster_tick_width: float = 0.9) -> Path:
    session = load_session(str(session_pkl))
    key = f"{getattr(session, 'subject', 'unknown')}_{getattr(session, 'session_name', 'unknown')}"

    tmp = REPO / "tmp_demo_qc" / "unit_compare"
    tmp.mkdir(parents=True, exist_ok=True)

    panels: List[tuple[str, Path]] = []

    # 1) Baseline aligned, colored by outcome (grouped)
    if "baseline_outcome" in include:
        p = tmp / f"{key}_c{cluster_id}_baseline_outcome.png"
        su.plot_baseline_raster_psth_by_future_outcome(
            session, cluster_id,
            window=window, bin_size=bin_size,
            sort_trials="outcome", peth_scale="per_outcome",
            smooth_sigma=1.0,
            compact_scale=panel_scale, raster_line_height=raster_line_height, legend_ratio=0.10,
            export_dpi=panel_dpi,
            save_path=str(p))
        panels.append(("Baseline_ON — by outcome", p))

    # 2) Baseline aligned, chronological order (early→late)
    if "baseline_chrono" in include:
        p = tmp / f"{key}_c{cluster_id}_baseline_chrono.png"
        su.plot_baseline_raster_psth_by_future_outcome(
            session, cluster_id,
            window=window, bin_size=bin_size,
            sort_trials="none", peth_scale="none",
            smooth_sigma=1.0,
            compact_scale=panel_scale, raster_line_height=raster_line_height, legend_ratio=0.10,
            export_dpi=panel_dpi,
            save_path=str(p))
        panels.append(("Baseline_ON — chrono", p))

    # 3) Change_ON aligned, Hit vs Miss
    if "change_outcome" in include:
        p = tmp / f"{key}_c{cluster_id}_change_outcome.png"
        if stack_change_outcome:
            su.plot_change_raster_psth_stacked(
                session, cluster_id,
                window=window, bin_size=bin_size,
                smooth_sigma=1.0,
                compact_scale=panel_scale, raster_line_height=raster_line_height, legend_ratio=0.10,
                export_dpi=panel_dpi,
                save_path=str(p))
        else:
            su.plot_change_rasters_by_outcome(
                session, cluster_id,
                window=window, bin_size=bin_size,
                smooth_sigma=1.0,
                compact_scale=compact_scale, raster_line_height=raster_line_height,
                save_path=str(p))
        panels.append(("Change_ON — Hit vs Miss", p))

    # Optionally: generic raster aligned to First_Lick if event exists
    if "first_lick" in include:
        try:
            p = tmp / f"{key}_c{cluster_id}_firstlick.png"
            su.plot_raster_psth(
                session, cluster_id,
                event_name="First_Lick", window=window, bin_size=bin_size,
                compact_scale=compact_scale, raster_line_height=raster_line_height,
                save_path=str(p))
            panels.append(("First_Lick — raster/PSTH", p))
        except Exception:
            # silently skip if event not present
            pass

    # Inline path: compose directly without rasterized intermediate PNGs to avoid squished text
    if inline:
        # -------------------- Data assembly --------------------
        session_obj = session
        trials = getattr(session_obj, "trials", []) or []
        baseline_events = align_mod.get_event_times_by_trial(session_obj, "Baseline_ON")
        rows = []
        for i, t in enumerate(trials):
            try:
                et = float(baseline_events[i])
            except Exception:
                et = float('nan')
            if np.isnan(et):
                continue
            rows.append((i, et, _normalize_outcome_label(getattr(t, "trialoutcome", None))))
        import pandas as pd
        df_base = pd.DataFrame(rows, columns=["trial_idx", "event_time", "outcome"]) if rows else pd.DataFrame(columns=["trial_idx","event_time","outcome"])
        colors = _get_outcome_colors(None)
        present_outcomes = [o for o in ("Hit","FA","Abort","Miss","Ref","Other") if o in set(df_base['outcome'].unique())]
        n_out = max(1, len(present_outcomes))
        change_events = align_mod.get_event_times_by_trial(session_obj, "Change_ON")
        rows_ch = []
        for i, t in enumerate(trials):
            try:
                et = float(change_events[i])
            except Exception:
                et = float('nan')
            if np.isnan(et):
                continue
            rows_ch.append((i, et, _normalize_outcome_label(getattr(t, "trialoutcome", None))))
        df_change = pd.DataFrame(rows_ch, columns=["trial_idx","event_time","outcome"]) if rows_ch else pd.DataFrame(columns=["trial_idx","event_time","outcome"])

        # -------------------- Figure layout --------------------
        import matplotlib.gridspec as gridspec
        fig_w, fig_h = 8.27, 11.69  # A4 portrait
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=panel_dpi, constrained_layout=True)
        gs_main = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1], wspace=0.25, hspace=0.35)
        gs_top = gs_main[0, :].subgridspec(1, 3)
        gs_bot = gs_main[1, :].subgridspec(1, 3)
        gs_bot_left = gs_bot[0, 0].subgridspec(n_out, 1, hspace=0.12)
        gs_bot_right = gs_bot[0, 2].subgridspec(2, 1, hspace=0.20)

        # -------------------- Rasters --------------------
        cluster = None
        for c in session_obj.clusters:
            if int(c.cluster_id) == int(cluster_id):
                cluster = c
                break
        spike_times = getattr(cluster, 'spike_times', np.array([]))

        # Baseline outcome raster (sorted by outcome)
        ax_r_base = fig.add_subplot(gs_top[0, 0])
        order = ["Hit","FA","Abort","Miss","Ref","Other"]
        try:
            cat = pd.Categorical(df_base["outcome"], categories=order, ordered=True)
            df_base_sorted = df_base.assign(_o=cat).sort_values(["_o","trial_idx"]).drop(columns=["_o"]).reset_index(drop=True)
        except Exception:
            df_base_sorted = df_base
        spikes_trials = _spikes_relative_to_events(spike_times, df_base_sorted['event_time'].tolist(), window)
        for row_idx, (_, row) in enumerate(df_base_sorted.iterrows()):
            sp = spikes_trials[row_idx]
            col = colors.get(row['outcome'], colors['Other'])
            h = max(0.1, min(0.95, float(raster_line_height)))
            y1 = row_idx + 0.5 - h / 2
            y2 = row_idx + 0.5 + h / 2
            ax_r_base.vlines(sp, y1, y2, color=col, linewidth=float(raster_tick_width))
        ax_r_base.set_ylabel('Trial')
        ax_r_base.set_title('Baseline_ON — by outcome')
        ax_r_base.axvline(0, color='k', linestyle='--', linewidth=0.8)

        # Baseline chrono raster
        ax_r_chrono = fig.add_subplot(gs_top[0, 1], sharex=ax_r_base)
        spikes_trials_chrono = _spikes_relative_to_events(spike_times, df_base.sort_values('trial_idx')['event_time'].tolist(), window)
        for row_idx, (_, row) in enumerate(df_base.sort_values('trial_idx').iterrows()):
            sp = spikes_trials_chrono[row_idx]
            col = colors.get(row['outcome'], colors['Other'])
            h = max(0.1, min(0.95, float(raster_line_height)))
            y1 = row_idx + 0.5 - h / 2
            y2 = row_idx + 0.5 + h / 2
            ax_r_chrono.vlines(sp, y1, y2, color=col, linewidth=float(raster_tick_width)*0.85)
        ax_r_chrono.set_title('Baseline_ON — chrono')
        ax_r_chrono.set_ylabel('Trial')
        ax_r_chrono.axvline(0, color='k', linestyle='--', linewidth=0.8)

        # Change_ON raster (only Hit and Miss)
        ax_r_change = fig.add_subplot(gs_top[0, 2], sharex=ax_r_base)
        # keep only Hit and Miss trials for raster display
        df_change_hm = df_change[df_change['outcome'].isin(['Hit','Miss'])].copy()
        # sort right raster by outcome (Hit first, then Miss)
        order_change = ["Hit","Miss"]
        try:
            catc = pd.Categorical(df_change_hm["outcome"], categories=order_change, ordered=True)
            df_change_sorted = df_change_hm.assign(_o=catc).sort_values(["_o","trial_idx"]).drop(columns=["_o"]).reset_index(drop=True)
        except Exception:
            df_change_sorted = df_change_hm
        spikes_trials_change = _spikes_relative_to_events(spike_times, df_change_sorted['event_time'].tolist(), window)
        for row_idx, (_, row) in enumerate(df_change_sorted.iterrows()):
            sp = spikes_trials_change[row_idx]
            col = colors.get(row['outcome'], colors['Other'])
            h = max(0.1, min(0.95, float(raster_line_height)))
            y1 = row_idx + 0.5 - h / 2
            y2 = row_idx + 0.5 + h / 2
            ax_r_change.vlines(sp, y1, y2, color=col, linewidth=float(raster_tick_width))
        ax_r_change.set_title('Change_ON — Hit vs Miss')
        ax_r_change.set_ylabel('Trial')
        ax_r_change.axvline(0, color='k', linestyle='--', linewidth=0.8)

        # -------------------- Baseline PSTHs (stacked) --------------------
        _, bin_centers = align_mod.align_spikes_to_events(np.array([]), df_base['event_time'].tolist(), window=window, bin_size=bin_size)
        psth_axes_left = []
        for i, o in enumerate(present_outcomes):
            axp = fig.add_subplot(gs_bot_left[i, 0], sharex=ax_r_base)
            g = df_base.loc[df_base['outcome'] == o]
            ets = g['event_time'].tolist()
            m, _ = align_mod.align_spikes_to_events(spike_times, ets, window=window, bin_size=bin_size)
            if m.shape[0] > 0:
                psth = np.nanmean(m, axis=0)
                sem = np.nanstd(m, axis=0) / np.sqrt(max(1, m.shape[0]))
            else:
                psth = np.zeros_like(bin_centers)
                sem = np.zeros_like(bin_centers)
            if psth.size > 1:
                psth = gaussian_filter1d(psth, sigma=1.0)
                sem = gaussian_filter1d(sem, sigma=1.0)
            col = colors.get(o, colors['Other'])
            axp.plot(bin_centers, psth, color=col)
            axp.fill_between(bin_centers, psth - sem, psth + sem, color=col, alpha=0.18, linewidth=0)
            axp.axvline(0, color='k', linestyle='--', linewidth=0.6)
            axp.set_ylabel('FR (Hz)')
            if i == len(present_outcomes)-1:
                axp.set_xlabel('Time (s)')
            psth_axes_left.append(axp)

        # -------------------- Change_ON PSTHs (Hit vs Miss) --------------------
        _, bc_change = align_mod.align_spikes_to_events(np.array([]), df_change['event_time'].tolist(), window=window, bin_size=bin_size)
        hit_df = df_change.loc[df_change['outcome']=='Hit']
        miss_df = df_change.loc[df_change['outcome']=='Miss']
        psth_axes_right = []
        for j, (lbl, subdf) in enumerate([("Hit", hit_df),("Miss", miss_df)]):
            axp = fig.add_subplot(gs_bot_right[j, 0], sharex=ax_r_change)
            ets = subdf['event_time'].tolist()
            m,_ = align_mod.align_spikes_to_events(spike_times, ets, window=window, bin_size=bin_size)
            if m.shape[0] > 0:
                psth = np.nanmean(m, axis=0)
                sem = np.nanstd(m, axis=0) / np.sqrt(max(1, m.shape[0]))
            else:
                psth = np.zeros_like(bc_change)
                sem = np.zeros_like(bc_change)
            if psth.size > 1:
                psth = gaussian_filter1d(psth, sigma=1.0)
                sem = gaussian_filter1d(sem, sigma=1.0)
            col = colors.get(lbl, colors['Other'])
            axp.plot(bc_change, psth, color=col)
            axp.fill_between(bc_change, psth - sem, psth + sem, color=col, alpha=0.18, linewidth=0)
            axp.axvline(0, color='k', linestyle='--', linewidth=0.6)
            axp.set_ylabel('FR (Hz)')
            if j==1:
                axp.set_xlabel('Time (s)')
            psth_axes_right.append(axp)

        # -------- Manual PSTH width alignment to match rasters --------
        # Apply tight_layout to finalize all automatic positioning first
        fig.tight_layout(pad=1.5, h_pad=0.5, w_pad=0.5)
        fig.canvas.draw()
        
        # Now override PSTH axes positions to exactly match raster widths
        base_bbox = ax_r_base.get_position()
        change_bbox = ax_r_change.get_position()
        
        for axp in psth_axes_left:
            bb = axp.get_position()
            # Use exact x0 and width from baseline raster
            axp.set_position([base_bbox.x0, bb.y0, base_bbox.width, bb.height])
        
        for axp in psth_axes_right:
            bb = axp.get_position()
            # Use exact x0 and width from change raster
            axp.set_position([change_bbox.x0, bb.y0, change_bbox.width, bb.height])

        # -------------------- Inline Legend (swatches) --------------------
        if present_outcomes:
            counts = {o: int((df_base['outcome']==o).sum()) for o in present_outcomes}
            y = 0.98
            x_sw, x_txt = 1.02, 1.06  # to the right of the right raster
            for o in present_outcomes:
                col = colors.get(o, colors['Other'])
                ax_r_change.add_patch(plt.Rectangle((x_sw, y-0.02), 0.025, 0.015, transform=ax_r_change.transAxes,
                                                    facecolor=col, edgecolor='k', linewidth=0.3, clip_on=False, zorder=5))
                ax_r_change.text(x_txt, y, f"{o} (n={counts[o]})", transform=ax_r_change.transAxes,
                                 va='top', ha='left', fontsize=8)
                y -= 0.04

        out_png.parent.mkdir(parents=True, exist_ok=True)
        # Save without bbox_inches='tight' to preserve manual alignment
        fig.savefig(out_png.as_posix(), dpi=panel_dpi)
        plt.close(fig)
        return out_png

    # Compose into a single row figure (legacy PNG embedding path)
    n = len(panels)
    if n == 0:
        raise SystemExit("No panels requested or produced. Nothing to do.")

    # Use A4 landscape overall size and squeeze right column (Change_ON) to ~half its prior width
    # Equal thirds across three main columns
    widths = [1.0 for _ in panels]
    # Portrait A4 dimensions (width x height in inches)
    fig_w = 8.27
    fig_h = 11.69
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = gridspec.GridSpec(1, n, figure=fig, width_ratios=widths, wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    if n == 1:
        axes = [axes]
    for ax, (title, path) in zip(axes, panels):
        draw_image(ax, path, title=title)

    # constrained_layout already used
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png.as_posix(), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main(argv=None):
    p = argparse.ArgumentParser(description="Build a compact per-unit comparison figure across multiple alignments/sorts")
    p.add_argument("--session-pkl", default=None, help="Path to session .pkl; if omitted, use --session-key to find in data/")
    p.add_argument("--session-key", default=None, help="Key like BG_031_01052025 to locate data/<key>.pkl")
    p.add_argument("--cluster-id", type=int, required=True)
    p.add_argument("--out", default="png_output/unit_comparisons/comparison.png")
    p.add_argument("--include", nargs="*", default=["baseline_outcome", "baseline_chrono", "change_outcome"],
                   help="Which panels to include: baseline_outcome baseline_chrono change_outcome first_lick")
    p.add_argument("--compact-scale", type=float, default=0.5, help="Legacy: ignored for panel sizing; use --panel-scale instead")
    p.add_argument("--panel-scale", type=float, default=1.0, help="Scale factor for individual panel figures (higher avoids text distortion)")
    p.add_argument("--panel-dpi", type=int, default=220, help="DPI for individual panel images to keep text crisp when embedded")
    p.add_argument("--inline", action="store_true", help="Render panels directly in one figure (no PNG embedding) for crisp text")
    p.add_argument("--raster-line-height", type=float, default=0.6)
    p.add_argument("--raster-tick-width", type=float, default=0.9, help="Line width for spike ticks in rasters (inline mode)")
    p.add_argument("--window", nargs=2, type=float, default=[-0.5, 1.0])
    p.add_argument("--stack-change-outcome", action="store_true",
                   help="Use stacked layout for Change_ON Hit vs Miss panel (recommended)")
    p.add_argument("--bin-size", type=float, default=0.02)
    args = p.parse_args(argv)

    if args.session_pkl:
        pkl = Path(args.session_pkl)
    else:
        if not args.session_key:
            raise SystemExit("Provide --session-pkl or --session-key")
        pkl = REPO / "data" / f"{args.session_key}.pkl"
    out_png = REPO / args.out

    build_unit_comparison(
        pkl,
        args.cluster_id,
        out_png,
        include=list(args.include),
        compact_scale=float(args.compact_scale),
        raster_line_height=float(args.raster_line_height),
        window=tuple(map(float, args.window)),
        bin_size=float(args.bin_size),
        stack_change_outcome=bool(args.stack_change_outcome),
        panel_scale=float(args.panel_scale),
        panel_dpi=int(args.panel_dpi),
        inline=bool(args.inline),
        raster_tick_width=float(args.raster_tick_width)
    )
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    raise SystemExit(main())
