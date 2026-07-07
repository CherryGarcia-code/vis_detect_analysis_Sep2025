"""Cross-animal STRIATUM replication, TF-responsive overlay (talk substrate).

Same layout as ws_xanimal_striatum.py (cols = animal: DMS BG_046, BG_039; VMS BG_031 kept
separate; rows = outcome / change-size / push-pull), but every line is split into
TF-responsive (colour, solid + band) vs non-responsive (grey, dashed) using the per-subject
Khilkevich-Lohse registries. Population view (NOT cell-type faceted). Only the 3 striatum mice
have registries. NOT movement-controlled; never pool DMS with VMS.

Usage: py scripts/talk_substrate/ws_xanimal_striatum_tf.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402
from visdetect.analysis.config import OUTCOME_COLORS  # noqa: E402

C.setup_talk_style()
SMALL, BIG = C.CHANGE_COLORS["small"], C.CHANGE_COLORS["big"]   # canonical (config.CHANGE_SIZE_COLORS)
H, M = OUTCOME_COLORS["Hit"], OUTCOME_COLORS["Miss"]
ANIMALS = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
RESP_WIN = (0.0, 1.0)


def win_mean(cache, event, cond, win, mask):
    m = E.mat(cache, event, cond, "full")
    seg = m[:, E.win_mask(E.bc(cache, event), win)]
    fin = np.isfinite(seg).all(1) & mask
    return float(np.nanmean(seg[fin])) if fin.any() else np.nan


def main():
    caches = {}
    for subj, _site in ANIMALS:
        if C.has_tf_registry(subj):
            caches[subj] = E.load_event_cache(subj)
    subs = [(s, site) for (s, site) in ANIMALS if s in caches]

    fig = plt.figure(figsize=(5.4 * len(subs), 12))
    gs = gridspec.GridSpec(3, len(subs), hspace=0.42, wspace=0.28)
    rows = []
    for ci, (subj, site) in enumerate(subs):
        cache = caches[subj]
        resp, nonresp = E.tf_masks(cache, subj)
        allm = np.ones(len(cache["unit_meta_celltype"]), bool)
        hdr = (f"{subj} ({site})\n{int(allm.sum())} unit-sess · "
               f"{int(resp.sum())} TF+ / {int(nonresp.sum())} TF−")

        ax0 = fig.add_subplot(gs[0, ci])
        E.multi_cond_panel_tf(ax0, cache,
                              [("Change_ON", "hit", H, "hit"), ("Change_ON", "miss", M, "miss")],
                              "Change_ON", allm, resp, nonresp, title=hdr)
        if ci == 0:
            ax0.set_ylabel("OUTCOME\nz (shared baseline)")
        ax1 = fig.add_subplot(gs[1, ci])
        E.multi_cond_panel_tf(ax1, cache,
                              [("Change_ON", "small", SMALL, "small"), ("Change_ON", "big", BIG, "big")],
                              "Change_ON", allm, resp, nonresp)
        if ci == 0:
            ax1.set_ylabel("CHANGE SIZE\nz (shared baseline)")
        ax2 = fig.add_subplot(gs[2, ci])
        E.sign_panel_tf(ax2, cache, "Hit", allm, resp, nonresp)
        if ci == 0:
            ax2.set_ylabel("PUSH-PULL @lick\nz (shared baseline)")

        for grp, mask in [("TF+", resp), ("TF-", nonresp)]:
            rows.append(dict(
                subject=subj, site=site, group=grp, n=int(mask.sum()),
                hit_minus_miss=round(win_mean(cache, "Change_ON", "hit", RESP_WIN, mask)
                                     - win_mean(cache, "Change_ON", "miss", RESP_WIN, mask), 4),
                big_minus_small=round(win_mean(cache, "Change_ON", "big", RESP_WIN, mask)
                                      - win_mean(cache, "Change_ON", "small", RESP_WIN, mask), 4)))

    fig.suptitle("Cross-animal STRIATUM — TF-responsive vs non-responsive (DMS 046, 039 | VMS 031)",
                 fontsize=C.FS["suptitle"], y=0.995)
    fig.text(0.5, 0.005,
             "Population view (all cells), TF-responsive = colour (solid + band) vs non-responsive = grey "
             "dashed. TF-responsive = Khilkevich-Lohse GLM (NOT movement-controlled). DMS (046, 039) should "
             "agree; VMS (031) kept separate (never pooled). Bands = bootstrap 95% CI; TF+ n small.",
             ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.FIG_DIR.parent / "ws_xanimal_striatum_tf.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {out}")
    df = pd.DataFrame(rows)
    df.to_csv(C.FIG_DIR.parent / "ws_xanimal_striatum_tf_stats.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
