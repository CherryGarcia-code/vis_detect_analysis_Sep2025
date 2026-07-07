"""Cross-region CELL-TYPE comparison on a COMMON width cutoff (talk substrate).

Per-subject GMM cutoffs (0.41-0.53 ms) put narrow/broad on DIFFERENT width scales, so
cross-animal cell-type labels aren't comparable. Here we relabel EVERY unit with a SINGLE
common cutoff (one 2-component GMM on the POOLED trough-to-peak across all 4 subjects), then
compare narrow- vs broad-spiking dynamics across regions on that common scale.

Cols = region group (DMS=BG_046+039 pooled, VMS=BG_031, Cortex=BG_038); rows = alignment
(change onset, response lick). Lines = narrow vs broad (common cutoff). Bands = bootstrap 95%
CI across unit-sessions. Cell-type gloss is region-specific (legend): striatum FSI/SPN,
cortex FS/pyramidal — but the width boundary is now identical across all.

Usage: py scripts/talk_substrate/ws_xanimal_celltype.py
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
from visdetect.analysis.waveform_celltype import classify_celltype  # noqa: E402

C.setup_talk_style()
NARROW_C, BROAD_C = C.CELLTYPE_COLORS["Narrow (FSI)"], C.CELLTYPE_COLORS["Broad (MSN/Proj)"]
T2P_FILES = {"BG_046": "bg046_waveform_t2p.csv", "BG_039": "waveform_t2p_BG_039.csv",
             "BG_031": "waveform_t2p_BG_031.csv", "BG_038": "waveform_t2p_BG_038.csv"}
# region group -> (subjects, narrow gloss, broad gloss)
GROUPS = [
    ("Striatum DMS (046+039)", ["BG_046", "BG_039"], "FSI", "SPN"),
    ("Striatum VMS (031)", ["BG_031"], "FSI", "SPN"),
    ("Cortex M1/S1 ref (038)", ["BG_038"], "FS", "Pyr"),
]
ROWS = [("Change_ON", "all", "CHANGE onset (all go)"),
        ("Hit", "all", "RESPONSE lick (hits)")]


def load_t2p(subj):
    return pd.read_csv(C.CACHE_DIR / T2P_FILES[subj], dtype={"session_8": str})


def common_celltype(cache, subs, thr):
    lut = {}
    for s in subs:
        for r in load_t2p(s).itertuples():
            lut[(str(r.session_8), int(r.cluster_id))] = float(r.t2p_ms)
    sess = cache["unit_meta_session"].astype(str)
    cid = cache["unit_meta_cluster_id"].astype(int)
    t2p = np.array([lut.get((sess[i], int(cid[i])), np.nan) for i in range(len(sess))])
    narrow = np.isfinite(t2p) & (t2p < thr)
    broad = np.isfinite(t2p) & (t2p >= thr)
    return narrow, broad


def main():
    # common cutoff from pooled t2p across all subjects
    allt2p = np.concatenate([load_t2p(s)["t2p_ms"].values for s in T2P_FILES])
    _, info = classify_celltype(allt2p)
    THR = float(info["threshold_ms"])
    print(f"[xct] common cutoff (pooled GMM, n={info['n']}): {THR:.3f} ms")

    fig = plt.figure(figsize=(5.2 * len(GROUPS), 9))
    gs = gridspec.GridSpec(len(ROWS), len(GROUPS), hspace=0.36, wspace=0.28)
    rows_out = []
    for ci, (name, subs, ng, bg) in enumerate(GROUPS):
        cache = E.pool_caches(subs) if len(subs) > 1 else E.load_event_cache(subs[0])
        narrow, broad = common_celltype(cache, subs, THR)
        for ri, (ev, cond, rlabel) in enumerate(ROWS):
            ax = fig.add_subplot(gs[ri, ci])
            bcv = E.bc(cache, ev)
            E.decorate(ax, ev, baseline_win=E.EVENT_DISPLAY[ev]["baseline"])
            for mask, col, gloss in [(narrow, NARROW_C, ng), (broad, BROAD_C, bg)]:
                m, lo, hi, nU = E.mean_ci(E.mat(cache, ev, cond, "full"), mask)
                E.plot_band(ax, bcv, m, lo, hi, col, f"{gloss} (n={nU})")
                pk, pt = E.peak_stat(bcv, m)
                rows_out.append(dict(group=name, event=ev, celltype=gloss, n=nU,
                                     peak_z=round(pk, 3), peak_t=round(pt, 3)))
            if ri == 0:
                ax.set_title(name, fontsize=C.FS["title"])
            if ci == 0:
                ax.set_ylabel(f"{rlabel}\nz (shared baseline)")
            else:
                ax.set_ylabel("")
            ax.legend(frameon=False, fontsize=C.FS["legend"], loc="upper left")

    fig.suptitle(f"Cross-region cell types on a COMMON width cutoff ({THR:.2f} ms) — "
                 "narrow vs broad by region", fontsize=C.FS["suptitle"], y=0.99)
    fig.text(0.5, 0.005,
             f"Single pooled-GMM width cutoff {THR:.2f} ms applied to ALL units (so narrow/broad "
             "are comparable across animals). Gloss: striatum narrow=FSI/broad=SPN; cortex "
             "narrow=FS/broad=pyramidal. Bands = bootstrap 95% CI across unit-sessions.",
             ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.FIG_DIR.parent / "ws_xanimal_celltype.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {out}")
    df = pd.DataFrame(rows_out)
    df.to_csv(C.FIG_DIR.parent / "ws_xanimal_celltype_stats.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
