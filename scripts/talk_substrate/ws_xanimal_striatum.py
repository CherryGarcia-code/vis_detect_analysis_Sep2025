"""Cross-animal STRIATUM replication (talk substrate): does the BG_046 story hold in other animals?

Striatal subjects, grouped by recording site (never pool VMS with DMS):
  - DMS (dorsal CP): BG_046, BG_039   (coordinate-compatible -> may be pooled)
  - VMS (ventromedial CP): BG_031     (kept separate)

Layout: rows = key contrast, columns = animal (DMS animals first, then VMS). Replication =
columns BG_046 & BG_039 should agree; BG_031 may differ. Contrasts:
  1. Outcome      : change-aligned SDT hit vs miss (go trials)
  2. Change size  : change-aligned small vs big (sensory scaling)
  3. Push-pull    : response-lick up- vs down-modulated (held-out sign)
Population mean + bootstrap 95% CI across unit-sessions. Each animal's cell types are computed
per-subject (separate GMM); proportions differ across animals (probe/yield), so this is a
population view (cell-type faceting lives in the per-animal figures).

Usage: py scripts/talk_substrate/ws_xanimal_striatum.py
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

setup_style()
SMALL, BIG = "#fdae6b", "#d94801"
H, M = OUTCOME_COLORS["Hit"], OUTCOME_COLORS["Miss"]
# (subject, site label)
ANIMALS = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
RESP_WIN = (0.0, 1.0)


def win_mean(cache, event, cond, win):
    m = E.mat(cache, event, cond, "full")
    wm = E.win_mask(E.bc(cache, event), win)
    seg = m[:, wm]
    fin = np.isfinite(seg).all(1)
    return float(np.nanmean(seg[fin])) if fin.any() else np.nan


def main():
    caches = {}
    for subj, _site in ANIMALS:
        try:
            caches[subj] = E.load_event_cache(subj)
        except FileNotFoundError:
            print(f"[xanimal] cache for {subj} missing; skip")
    subs = [(s, site) for (s, site) in ANIMALS if s in caches]

    fig = plt.figure(figsize=(5.2 * len(subs), 12))
    gs = gridspec.GridSpec(3, len(subs), hspace=0.40, wspace=0.28)
    rows = []
    for ci, (subj, site) in enumerate(subs):
        cache = caches[subj]
        nar, bro, _ = C.common_celltype(cache, [subj], E.common_cut())
        n_units = len(cache["unit_meta_celltype"])
        n_nar, n_bro = int(nar.sum()), int(bro.sum())
        hdr = (f"{subj} ({site})\n{n_units} unit-sess · "
               f"{n_nar}N/{n_bro}B (common cut {E.common_cut():.2f}ms)")

        # row 0: outcome
        ax0 = fig.add_subplot(gs[0, ci])
        E.multi_cond_panel(ax0, cache,
                           [("Change_ON", "hit", H, "SDT hit"), ("Change_ON", "miss", M, "SDT miss")],
                           "Change_ON", title=hdr)
        if ci == 0:
            ax0.set_ylabel("OUTCOME\nz (shared baseline)")
        # row 1: change size
        ax1 = fig.add_subplot(gs[1, ci])
        E.multi_cond_panel(ax1, cache,
                           [("Change_ON", "small", SMALL, "small (1.25-1.5x)"),
                            ("Change_ON", "big", BIG, "big (2-4x)")],
                           "Change_ON", title=None)
        if ci == 0:
            ax1.set_ylabel("CHANGE SIZE\nz (shared baseline)")
        # row 2: push-pull at lick
        ax2 = fig.add_subplot(gs[2, ci])
        E.sign_panel(ax2, cache, "Hit", title=None)
        if ci == 0:
            ax2.set_ylabel("PUSH-PULL @lick\nz (shared baseline)")

        # quantify
        oe = win_mean(cache, "Change_ON", "hit", RESP_WIN) - win_mean(cache, "Change_ON", "miss", RESP_WIN)
        ce = win_mean(cache, "Change_ON", "big", RESP_WIN) - win_mean(cache, "Change_ON", "small", RESP_WIN)
        rows.append(dict(subject=subj, site=site, n_units=n_units, n_narrow=n_nar, n_broad=n_bro,
                         hit_minus_miss=round(oe, 4), big_minus_small=round(ce, 4)))

    fig.suptitle("Cross-animal STRIATUM replication — DMS (BG_046, BG_039) vs VMS (BG_031)",
                 fontsize=13, y=0.995)
    fig.text(0.5, 0.005,
             "Population mean + bootstrap 95% CI across unit-sessions. Replication = BG_046 & BG_039 "
             "(both dorsal CP) agree; BG_031 (ventromedial CP) shown separately (never pooled with DMS). "
             "Cell-type proportions differ across animals (per-subject GMM); this is a population view.",
             ha="center", fontsize=8, color="#555555", wrap=True)

    out = C.FIG_DIR.parent / "ws_xanimal_striatum.png"   # FIGURES/talk_substrate/ (cross-animal)
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {out}")
    df = pd.DataFrame(rows)
    sp = C.FIG_DIR.parent / "ws_xanimal_striatum_stats.csv"
    df.to_csv(sp, index=False)
    print(f"[fig] wrote {sp}")
    print(df.to_string(index=False))

    # verdict
    dms = df[df.site == "DMS"]
    dms_ok = (dms["hit_minus_miss"] > 0).all() and (dms["big_minus_small"] > 0).all()
    consistent = dms_ok and len(dms) >= 2
    print("\nDMS hit>miss & big>small in both 046/039:", dms_ok)
    print("STRIATUM REPLICATION:", "STRONG" if consistent else "PARTIAL/CHECK")


if __name__ == "__main__":
    main()
