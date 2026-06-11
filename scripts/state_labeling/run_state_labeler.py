"""Interactive matplotlib GUI to sparsely label behavioral-state episodes on the
outcome raster. Mirrors scripts/tf_labeling/run_labeling_gui.py.

Keys: 1=Impulsive 2=StimSens 3=Disengaged  | drag=paint span (saved immediately)
      c=toggle change-size shading  | left/right=prev/next session (Expert->Naive)  q=quit

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
    matplotlib.use("TkAgg", force=True)  # assert interactive backend even if a lib set Agg
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    from visdetect.suite.loader import load_session
    from visdetect.analysis.state_labeling import (
        get_labeling_queue, build_outcome_raster, render_raster, save_episode, StateEpisode,
    )

    queue = get_labeling_queue()
    if not queue:
        print("No sessions in the labeling queue — check the QC-filtered staging manifest.")
        return
    state = {"i": 0, "label": "Impulsive", "cs_shade": False}
    keymap = {"1": "Impulsive", "2": "StimSens", "3": "Disengaged"}

    fig, ax = plt.subplots(figsize=(14, 3))

    def draw():
        ax.clear()
        sn = queue[state["i"]]
        sess = load_session(sn)
        raster = build_outcome_raster(sess)
        render_raster(ax, raster, change_size_shading=state["cs_shade"])
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
