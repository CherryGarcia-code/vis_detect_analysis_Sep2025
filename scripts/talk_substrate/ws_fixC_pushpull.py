"""FIX C: push-pull TIMING is a CHANGE-aligned claim (not lick).

The suppression-precedes-excitation OFFSET lives at CHANGE onset (at the lick the offset is
~0; the lick shows an up>down AMPLITUDE split, a different claim). Here we compute, per region
x animal x cell type (COMMON cutoff), the CHANGE-aligned per-unit peak latency for up- vs
down-modulated units (held-out sign), read OUTSIDE the sign-defining window, and the up-vs-down
median latency difference (Mann-Whitney + bootstrap CI). Tabulated so the slide line matches
the data — the offset is strongest in narrow/FSI but PRESENT (to a lesser degree) in broad/SPN
in some animals: do NOT claim SPN-absence.

HONESTY (caption): the up-unit peak (~1 s post-change) is movement-entangled; the clean claim
is "suppression is EARLY (down-trough well before the excitatory peak AND before movement),"
not a claim about absolute up-peak latency.

Usage: py scripts/talk_substrate/ws_fixC_pushpull.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402

C.setup_talk_style()
REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS", "BG_038": "Cortex(ref)"}
LAT_WIN = (0.25, 1.5)   # change-aligned latency window (excludes the 0-0.25 s sign window)


def change_latencies(cache, mask):
    full = E.mat(cache, "Change_ON", "all", "full")
    odd = E.mat(cache, "Change_ON", "all", "odd")
    bcv = E.bc(cache, "Change_ON")
    sw = E.EVENT_DISPLAY["Change_ON"]["sign"]      # (0, 0.25) — defines sign; excluded from latency
    s = E.unit_sign(odd, bcv, sw)
    lm = E.win_mask(bcv, LAT_WIN) & ~E.win_mask(bcv, sw)
    t = bcv[lm]
    seg = full[:, lm]
    fin = np.isfinite(seg).all(1) & np.isfinite(s) & mask
    up = fin & (s > 0)
    dn = fin & (s < 0)
    up_lat = t[np.argmax(seg[up], axis=1)] if up.any() else np.array([])
    dn_lat = t[np.argmin(seg[dn], axis=1)] if dn.any() else np.array([])
    return up_lat, dn_lat


def boot_diff(a, b, n_boot=1000, seed=42):
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    d = (np.median(a[rng.integers(0, len(a), (n_boot, len(a)))], 1)
         - np.median(b[rng.integers(0, len(b), (n_boot, len(b)))], 1))
    return float(np.median(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    thr, _ = C.common_t2p_cutoff()
    rows = []
    for subj in C.ALL_SUBJECTS:
        cache = E.load_event_cache(subj)
        narrow, broad, _ = C.common_celltype(cache, [subj], thr)
        for cell, mask in [("narrow", narrow), ("broad", broad)]:
            up, dn = change_latencies(cache, mask)
            if len(up) < 3 or len(dn) < 3:
                continue
            p = mannwhitneyu(up, dn).pvalue
            md, lo, hi = boot_diff(up, dn)
            rows.append(dict(animal=subj, region=REGION[subj], celltype=cell,
                             n_up=len(up), n_down=len(dn),
                             up_peak_med=round(float(np.median(up)), 3),
                             down_trough_med=round(float(np.median(dn)), 3),
                             up_minus_down=round(md, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             mwu_p=p))
    df = pd.DataFrame(rows)

    # forest plot of up-minus-down change-aligned latency diff
    fig, ax = plt.subplots(figsize=(10, 7))
    labels, ys = [], []
    for i, r in enumerate(df.itertuples()):
        y = len(df) - i
        ys.append(y)
        labels.append(f"{r.animal} {r.region} · {r.celltype}")
        col = "#e74c3c" if r.celltype == "narrow" else "#3498db"
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=2.2, zorder=2)
        ax.scatter([r.up_minus_down], [y], color=col, s=55, zorder=3,
                   edgecolors="k", linewidths=0.5)
        sig = "***" if r.mwu_p < 1e-3 else "**" if r.mwu_p < 1e-2 else "*" if r.mwu_p < 0.05 else "ns"
        ax.text(r.ci_hi + 0.02, y, f"{sig} (Δ{r.up_minus_down:+.2f}s)", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=1.0, ls=":")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("CHANGE-aligned up-peak − down-trough latency (s)   [positive = suppression earlier]")
    ax.set_title("Push-pull TIMING at change onset: up vs down peak latency\n"
                 "(red=narrow/FSI, blue=broad/SPN; common cutoff; bootstrap 95% CI)", fontsize=C.FS["title"])
    fig.text(0.5, -0.02,
             "Positive Δ = up-units peak LATER than down-units trough, i.e. suppression is earlier. "
             "Offset is strongest in narrow/FSI but PRESENT in broad/SPN in some animals (no SPN-absence "
             "claim). Caveat: up-peak (~1 s) is movement-entangled — the clean claim is EARLY suppression "
             "before the excitatory peak & movement.", ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.FIG_DIR.parent / "ws_fixC_pushpull_change_timing.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    df.to_csv(C.FIG_DIR.parent / "ws_fixC_pushpull_change_timing.csv", index=False)
    print(f"[fig] wrote {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
