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
C.setup_talk_style()
EVENTS = ["Baseline_ON", "Change_ON", "Hit", "FA"]


def _parse_window(s):
    lo, hi = (float(x) for x in s.split(","))
    return (lo, hi)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # NOTE: pass with '=' so the leading '-' isn't parsed as a flag: --baseline-anchor=-0.3,-0.05
    ap.add_argument("--baseline-anchor", type=str, default=None,
                    help="Re-reference ONLY the Baseline_ON column to this window 'lo,hi' (s), e.g. "
                         "'--baseline-anchor=-0.3,-0.05' to anchor the up/down split at onset. "
                         "Default: canonical (-1.75,-1.25) — unchanged. Other events untouched.")
    ap.add_argument("--group", choices=["celltype", "tf"], default="celltype",
                    help="tf = overlay TF-responsive (solid) vs non-responsive (grey dashed) within "
                         "the up/down split, per cell-type row (3 striatum mice only; ignores --baseline-anchor)")
    args = ap.parse_args()
    anchor = _parse_window(args.baseline_anchor) if args.baseline_anchor else None
    tf = args.group == "tf"

    cache = E.load_event_cache()
    masks = E.celltype_masks(cache)
    cts = [C.NARROW, C.BROAD]
    if tf:
        if not C.has_tf_registry(C.SUBJECT):
            raise SystemExit(f"no TF registry for {C.SUBJECT} (3 striatum mice only)")
        resp, nonresp = E.tf_masks(cache, C.SUBJECT)
    fig = plt.figure(figsize=(20, 8.5))
    gs = gridspec.GridSpec(2, 4, hspace=0.42, wspace=0.28)
    rows = []
    for ri, ct in enumerate(cts):
        for ci, ev in enumerate(EVENTS):
            ax = fig.add_subplot(gs[ri, ci])
            title = E.EVENT_DISPLAY[ev]["short"] if ri == 0 else None
            if tf:
                r = E.sign_panel_tf(ax, cache, ev, masks[ct], resp, nonresp, title=title)
                for d in r:
                    d["celltype"] = ct
            else:
                # Anchor applies ONLY to the Baseline_ON column; other events keep canonical baseline.
                reref = anchor if (ev == "Baseline_ON" and anchor is not None) else None
                r = E.sign_panel(ax, cache, ev, row_mask=masks[ct], title=title, reref_window=reref)
                for d in r:
                    d["celltype"] = ct
                    d["baseline_anchor"] = (f"{anchor[0]},{anchor[1]}" if reref else "canonical")
            rows += r
            if ci == 0:
                ax.set_ylabel(f"{ct}\nz-score (shared baseline)")
            else:
                ax.set_ylabel("")
    anch_txt = (f"  [Baseline column re-referenced to ({anchor[0]}, {anchor[1]}) s]"
                if (anchor is not None and not tf) else "")
    tf_txt = "  [TF-responsive overlay]" if tf else ""
    fig.suptitle(f"{C.SUBJECT} {C.region_label()}: event-aligned activity by modulation sign "
                 f"(rows = cell type){anch_txt}{tf_txt}", fontsize=C.FS["suptitle"], y=0.99)
    cap = ("Sign defined on held-out (odd) trials in the yellow response window; even half "
           "plotted (non-circular). Bands = bootstrap 95% CI across units. Cell-type "
           "proportions are unreliable (Fig A), but the up/down structure within each type is robust.")
    if tf:
        cap = ("Up/down held-out sign (hue) x TF-responsive (solid + band) vs non-responsive (grey "
               "dashed). TF-responsive = Khilkevich-Lohse GLM (C1 corr>0.2 & C2 CV p<0.01; NOT "
               "movement-controlled). Even half plotted (non-circular). Bands = bootstrap 95% CI; "
               "TF+ n small (see legend).")
    elif anchor is not None:
        cap += (f" Baseline_ON column RE-REFERENCED (re-centred) to ({anchor[0]}, {anchor[1]}) s so the "
                "split anchors at onset (canonical far-ITI reference makes it drift before t=0); SD kept "
                "from the stable canonical window. Other columns unchanged.")
    fig.text(0.5, 0.03, cap, ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    name = "fig_b_sign_split_tf" if tf else ("fig_b_sign_split" if anchor is None else "fig_b_sign_split_blonset")
    out = C.save_talk_figure(fig, name)
    print(f"[fig] wrote {out}")
    sdf = pd.DataFrame(rows)
    sp = C.stats_csv_path(name)
    sdf.to_csv(sp, index=False)
    print(f"[fig] wrote {sp}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
