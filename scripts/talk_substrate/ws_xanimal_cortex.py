"""STRIATUM vs CORTEX comparison (talk substrate): does motor/somatosensory cortex carry the
same task signals as striatum, or are they striatum-specific?

Columns = region group; rows = key contrast. Population mean + bootstrap 95% CI across
unit-sessions. Cell-type gloss differs by region (caption): striatum broad=SPN/narrow=FSI;
cortex broad=pyramidal(RS)/narrow=FS interneuron — the split is spike-width either way.

Groups:
  - Striatum DMS  : BG_046 + BG_039 pooled (coordinate-compatible dorsal CP)
  - Striatum VMS  : BG_031 (ventromedial CP, separate)
  - Cortex M1/S1  : BG_038 (MOp/SSp)

Contrasts: outcome (hit vs miss @change), change-size (small vs big @change),
push-pull (up vs down @lick, held-out sign).

Usage: py scripts/talk_substrate/ws_xanimal_cortex.py
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
RESP_WIN = (0.0, 1.0)
GROUPS = [
    ("Striatum DMS (BG_046+039)", ["BG_046", "BG_039"]),
    ("Striatum VMS (BG_031)", ["BG_031"]),
    ("Cortex M1/S1 reference (BG_038)", ["BG_038"]),
]


def win_mean(cache, event, cond, win):
    m = E.mat(cache, event, cond, "full")
    seg = m[:, E.win_mask(E.bc(cache, event), win)]
    fin = np.isfinite(seg).all(1)
    return float(np.nanmean(seg[fin])) if fin.any() else np.nan


def main():
    groups = []
    for name, subs in GROUPS:
        try:
            groups.append((name, subs, E.pool_caches(subs) if len(subs) > 1
                           else E.load_event_cache(subs[0])))
        except FileNotFoundError as e:
            print(f"[xcortex] {name}: cache missing ({e}); skip")
    n = len(groups)
    fig = plt.figure(figsize=(5.2 * n, 12))
    gs = gridspec.GridSpec(3, n, hspace=0.40, wspace=0.28)
    rows = []
    for ci, (name, subs, cache) in enumerate(groups):
        nar, bro, _ = C.common_celltype(cache, subs, E.common_cut())
        hdr = f"{name}\n{len(cache['unit_meta_celltype'])} unit-sess · {int(nar.sum())}N/{int(bro.sum())}B"
        a0 = fig.add_subplot(gs[0, ci])
        E.multi_cond_panel(a0, cache,
                           [("Change_ON", "hit", H, "hit"), ("Change_ON", "miss", M, "miss")],
                           "Change_ON", title=hdr)
        if ci == 0:
            a0.set_ylabel("OUTCOME\nz (shared baseline)")
        a1 = fig.add_subplot(gs[1, ci])
        E.multi_cond_panel(a1, cache,
                           [("Change_ON", "small", SMALL, "small"), ("Change_ON", "big", BIG, "big")],
                           "Change_ON")
        if ci == 0:
            a1.set_ylabel("CHANGE SIZE\nz (shared baseline)")
        a2 = fig.add_subplot(gs[2, ci])
        E.sign_panel(a2, cache, "Hit")
        if ci == 0:
            a2.set_ylabel("PUSH-PULL @lick\nz (shared baseline)")
        rows.append(dict(
            group=name, n_units=len(cache["unit_meta_celltype"]),
            n_narrow=int(nar.sum()), n_broad=int(bro.sum()),
            hit_minus_miss=round(win_mean(cache, "Change_ON", "hit", RESP_WIN)
                                 - win_mean(cache, "Change_ON", "miss", RESP_WIN), 4),
            big_minus_small=round(win_mean(cache, "Change_ON", "big", RESP_WIN)
                                  - win_mean(cache, "Change_ON", "small", RESP_WIN), 4)))

    fig.suptitle("STRIATUM vs CORTEX — task signals by region", fontsize=C.FS["suptitle"], y=0.995)
    fig.text(0.5, 0.005,
             "Population mean + bootstrap 95% CI across unit-sessions. Cell-type gloss is "
             "region-specific: striatum broad=SPN/narrow=FSI; cortex broad=pyramidal(RS)/narrow=FS. "
             "DMS = BG_046+BG_039 pooled (dorsal CP); VMS = BG_031; cortex = BG_038 (M1/S1 — "
             "high-quality cortical REFERENCE probe, NOT the MOs source region; positive control + "
             "generic cortex-vs-striatum dynamics contrast).",
             ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.FIG_DIR.parent / "ws_xanimal_cortex.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {out}")
    df = pd.DataFrame(rows)
    df.to_csv(C.FIG_DIR.parent / "ws_xanimal_cortex_stats.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
