"""Interactive matplotlib GUI to sparsely label behavioral-state episodes on the
outcome raster. Mirrors scripts/tf_labeling/run_labeling_gui.py.

Keys: 1=Impulsive 2=StimSens 3=Disengaged 4=Abort  | drag=paint span (saved immediately)
      c=toggle change-size shading  | left/right=prev/next session (Expert->Naive)  q=quit
(number keys are bound to STATE_LABELS in order, so the mapping stays in sync.)

Each painted span is appended to the labels CSV on release (no explicit save needed).
To correct a mislabel, edit the labels CSV directly or paint over the region.
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
    from matplotlib.widgets import SpanSelector

    from visdetect.suite.loader import load_session, session_exists
    from visdetect.analysis.constants import STATE_LABELS
    from visdetect.analysis.state_labeling import (
        get_labeling_queue, build_outcome_raster, render_raster, save_episode,
        load_episodes, StateEpisode,
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

    fig, ax = plt.subplots(figsize=(14, 3))

    def draw():
        ax.clear()
        sn = queue[state["i"]]
        try:
            sess = load_session(sn)
        except Exception as e:  # never let a bad session kill the window
            ax.text(0.5, 0.5, f"{sn}: could not load\n{type(e).__name__}: {e}\n"
                              "use left/right to skip", ha="center", va="center",
                    transform=ax.transAxes, color="crimson")
            ax.set_title(f"{sn}  [{state['i']+1}/{len(queue)}]  (unloadable)")
            fig.canvas.draw_idle()
            state["raster_len"] = 0
            state["session_name"] = sn
            return
        raster = build_outcome_raster(sess)
        # show previously-saved spans for this session so revisits are iterative
        prior = [e for e in load_episodes(args.labels) if str(e.session_name) == str(sn)]
        render_raster(ax, raster, change_size_shading=state["cs_shade"], episodes=prior)
        ax.set_title(f"{sn}  [{state['i']+1}/{len(queue)}]  active label: {state['label']}")
        fig.canvas.draw_idle()
        state["raster_len"] = len(raster)
        state["session_name"] = sn
        del sess
        gc.collect()

    def on_span(xmin, xmax):
        lo, hi = int(round(xmin)), int(round(xmax))
        ep = StateEpisode(state["session_name"], lo, hi, state["label"], args.labeler,
                          dt.datetime.now().isoformat())
        save_episode(ep, args.labels)
        ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.18, color="orange", lw=0)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in keymap:
            state["label"] = keymap[event.key]
        elif event.key == "right":
            state["i"] = min(state["i"] + 1, len(queue) - 1); draw()
        elif event.key == "left":
            state["i"] = max(state["i"] - 1, 0); draw()
        elif event.key == "c":
            state["cs_shade"] = not state["cs_shade"]; draw()
        elif event.key == "q":
            plt.close(fig)
            return  # figure is gone; don't touch its title/canvas afterwards
        ax.set_title(ax.get_title().rsplit("active label:", 1)[0] + f"active label: {state['label']}")
        fig.canvas.draw_idle()

    span = SpanSelector(ax, on_span, "horizontal", useblit=True,
                        props=dict(alpha=0.2, facecolor="orange"))
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
