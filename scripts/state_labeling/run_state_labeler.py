"""Interactive matplotlib GUI to sparsely label behavioral-state episodes on the
outcome raster. Mirrors scripts/tf_labeling/run_labeling_gui.py.

Two stacked tracks: the outcome raster (top) and a live "your labels" strip
(bottom) that shows every span you've saved for the current session in its state
colour — so revisiting a session shows your prior work, and a freshly painted
span appears in the strip the moment you release.

Keys: 1=Impulsive 2=StimSens 3=Disengaged 4=Abort  | drag=paint span (saved immediately)
      c=toggle change-size shading  | left/right=prev/next session (Expert->Naive)  q=quit
(number keys are bound to STATE_LABELS in order, so the mapping stays in sync.)

Each painted span is appended to the labels CSV on release (no explicit save needed).
To correct a mislabel, paint over the region (later spans win) or edit the CSV directly.
"""
import argparse
import datetime as dt
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def main():
    ap = argparse.ArgumentParser(description="Behavioral-state labeling GUI.")
    ap.add_argument("--labels", default="data/state_labels/state_episodes.csv")
    ap.add_argument("--labeler", default=os.environ.get("USERNAME", "user"))
    args = ap.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.widgets import SpanSelector

    from visdetect.suite.loader import load_session, session_exists
    from visdetect.analysis.constants import STATE_LABELS
    from visdetect.analysis.state_labeling import (
        get_labeling_queue, build_outcome_raster, render_raster, render_state_strip,
        save_episode, load_episodes, episodes_to_trial_labels, StateEpisode,
        lick_valence_legend_handles,
    )

    # The visdetect imports above (qc.py, tf_pulse.py, unit_selection.py,
    # suite/plotting.py) call matplotlib.use("Agg") at module level, which clobbers
    # any backend we set earlier. Assert the interactive backend AFTER them.
    matplotlib.use("TkAgg", force=True)
    plt.switch_backend("TkAgg")
    if matplotlib.get_backend().lower() == "agg":
        raise SystemExit(
            "Could not activate an interactive matplotlib backend (still 'Agg'). "
            "This needs a desktop session with tkinter — don't run it headless/over SSH."
        )

    # Drop manifest sessions that have no pkl on disk (e.g. 05092025) so the GUI
    # never tries to load — and crash on — a session that isn't there.
    full_queue = get_labeling_queue()
    queue = [sn for sn in full_queue if session_exists(sn)]
    missing = [sn for sn in full_queue if sn not in queue]
    if missing:
        print(f"Skipping {len(missing)} manifest session(s) with no pkl: {missing}")
    if not queue:
        print("No loadable sessions in the labeling queue — check the QC-filtered staging manifest.")
        return
    keymap = {str(i + 1): s for i, s in enumerate(STATE_LABELS)}  # 1=first state, ...
    state = {"i": 0, "label": STATE_LABELS[0], "cs_shade": False}

    fig = plt.figure(figsize=(14, 3.4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12)
    ax_r = fig.add_subplot(gs[0])   # outcome raster
    ax_y = fig.add_subplot(gs[1])   # live "your labels" strip
    fig.subplots_adjust(left=0.14)  # room for the outcome legend left of the tracks

    def _title():
        sn = state.get("session_name", queue[state["i"]])
        suffix = "(unloadable)" if state.get("raster_len", 0) == 0 else \
                 f"active label: {state['label']}"
        ax_r.set_title(f"{sn}  [{state['i']+1}/{len(queue)}]  {suffix}")

    def refresh_strip():
        """Redraw the 'your labels' strip from the CSV for the current session."""
        ax_y.clear()
        labels = episodes_to_trial_labels(
            load_episodes(args.labels), state["session_name"], state["raster_len"])
        render_state_strip(ax_y, labels, ylabel="your\nlabels")
        ax_y.set_xlim(ax_r.get_xlim())
        ax_y.set_xlabel("trial index")

    def draw():
        ax_r.clear()
        ax_y.clear()
        sn = queue[state["i"]]
        state["session_name"] = sn
        try:
            sess = load_session(sn)
        except Exception as e:  # never let a bad session kill the window
            ax_r.text(0.5, 0.5, f"{sn}: could not load\n{type(e).__name__}: {e}\n"
                                "use left/right to skip", ha="center", va="center",
                      transform=ax_r.transAxes, color="crimson")
            state["raster_len"] = 0
            _title()
            fig.canvas.draw_idle()
            return
        raster = build_outcome_raster(sess)
        render_raster(ax_r, raster, change_size_shading=state["cs_shade"])
        ax_r.legend(handles=lick_valence_legend_handles(), loc="center right",
                    bbox_to_anchor=(-0.01, 0.5), fontsize=6, frameon=False,
                    handlelength=1.0, handleheight=1.0, labelspacing=0.3,
                    borderaxespad=0.0, title="outcome", title_fontsize=6.5)
        ax_r.set_xlabel("")            # trial-index axis lives under the strip
        ax_r.tick_params(labelbottom=False)
        state["raster_len"] = len(raster)
        refresh_strip()
        _title()
        fig.canvas.draw_idle()
        del sess
        gc.collect()

    def on_span(xmin, xmax):
        if state.get("raster_len", 0) == 0:
            return  # nothing to label on an unloadable session
        lo = max(0, int(round(xmin)))
        hi = min(state["raster_len"] - 1, int(round(xmax)))
        if hi < lo:
            return
        ep = StateEpisode(state["session_name"], lo, hi, state["label"], args.labeler,
                          dt.datetime.now().isoformat())
        save_episode(ep, args.labels)
        refresh_strip()            # the new span appears immediately in its state colour
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in keymap:
            state["label"] = keymap[event.key]
            _title()
            fig.canvas.draw_idle()
        elif event.key == "right":
            state["i"] = min(state["i"] + 1, len(queue) - 1); draw()
        elif event.key == "left":
            state["i"] = max(state["i"] - 1, 0); draw()
        elif event.key == "c":
            state["cs_shade"] = not state["cs_shade"]; draw()
        elif event.key == "q":
            plt.close(fig)

    span = SpanSelector(ax_r, on_span, "horizontal", useblit=True,
                        props=dict(alpha=0.2, facecolor="orange"))
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
