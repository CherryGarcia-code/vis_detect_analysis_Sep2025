"""Fig B (talk substrate): event-aligned activity, split by modulation SIGN, faceted by cell type.

Rows = putative cell type (Narrow/FSI, Broad/MSN-Proj); columns = task events.
Within each panel, units are split into UP- vs DOWN-modulated, with the sign defined
on HELD-OUT (odd) trials in the canonical response window and the EVEN half plotted
(non-circular). Bands = bootstrap 95% CI across units (canonical utils.bootstrap_ci).

Averaging up- and down-units together cancels (old all-unit average looked flat);
this shows the real bidirectional structure, separately for each cell type.

Usage: py scripts/talk_substrate/fig_b_sign_split.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402

setup_style()
EVENTS = ["Baseline_ON", "Change_ON", "Hit", "FA"]


def main():
    cache = E.load_event_cache()
    masks = E.celltype_masks(cache)
    cts = [C.NARROW, C.BROAD]
    fig = plt.figure(figsize=(20, 8.5))
    gs = gridspec.GridSpec(2, 4, hspace=0.42, wspace=0.28)
    rows = []
    for ri, ct in enumerate(cts):
        for ci, ev in enumerate(EVENTS):
            ax = fig.add_subplot(gs[ri, ci])
            title = E.EVENT_DISPLAY[ev]["short"] if ri == 0 else None
            r = E.sign_panel(ax, cache, ev, row_mask=masks[ct], title=title)
            for d in r:
                d["celltype"] = ct
            rows += r
            if ci == 0:
                ax.set_ylabel(f"{ct}\nz-score (shared baseline)")
            else:
                ax.set_ylabel("")
    fig.suptitle(f"{C.SUBJECT} {C.region_label()}: event-aligned activity by modulation sign "
                 f"(rows = cell type)", fontsize=13, y=0.99)
    fig.text(0.5, 0.03,
             "Sign defined on held-out (odd) trials in the yellow response window; even half "
             "plotted (non-circular). Bands = bootstrap 95% CI across units. Cell-type "
             "proportions are unreliable (Fig A), but the up/down structure within each type is robust.",
             ha="center", fontsize=8, color="#555555", wrap=True)
    out = C.save_talk_figure(fig, "fig_b_sign_split")
    print(f"[fig] wrote {out}")
    sdf = pd.DataFrame(rows)
    sp = C.stats_csv_path("fig_b_sign_split")
    sdf.to_csv(sp, index=False)
    print(f"[fig] wrote {sp}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
