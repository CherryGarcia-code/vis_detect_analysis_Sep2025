"""WS3 (talk substrate): relative TIMING of up- vs down-modulated populations (push-pull).

The held-out sign design (sign on odd trials) is non-circular. New question: do up- and
down-modulated units differ in WHEN they peak/trough (push-pull reading: suppression during
withholding/accumulation, release at commitment)?

Rigor:
 - Sign defined on HELD-OUT (odd) trials in the canonical response window.
 - Latency read OUTSIDE that sign-defining window (within it the two groups are maximally
   separated by construction, so peak-timing there is partly built in).
 - Per-UNIT peak latency (up->argmax, down->argmin) AND center-of-mass latency, referenced to
   change and to lick. Up-vs-down compared with Mann-Whitney U + bootstrap CI on the median
   latency difference (vectorised bootstrap). Faceted by cell type (narrow/broad).
 - SIGN-FLIP contingency: sign is recomputed per event, so a unit can be up at change and down
   at lick. Report the change-sign x lick-sign table + flip fraction.

Caption: sign != cell identity (an 'up' cell can be D1, D2, or FSI). A timing asymmetry MOTIVATES
the push-pull (direct/indirect, release/withhold) hypothesis; establishing it needs the identity
layer (optotagging) — this is the 'why optotagging matters' slide.

Reuses event_psth_cache_<SUBJECT>.npz (per-unit traces + held-out sign). Cell type = COMMON
width cutoff (FIX A, E.celltype_masks). No re-sorting.
Usage: py scripts/talk_substrate/ws3_updown_timing.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402

C.setup_talk_style()
UP, DOWN = C.SIGN_COLORS["up"], C.SIGN_COLORS["down"]   # canonical (config.MODULATION_SIGN_COLORS)
# latency-measurement window per event (excludes the sign-defining window)
LAT_WIN = {"Change_ON": (0.25, 1.5), "Hit": (-1.0, 0.5)}
REF = {"Change_ON": "change", "Hit": "lick"}


def latencies(cache, event):
    full = E.mat(cache, event, "all", "full")
    odd = E.mat(cache, event, "all", "odd")
    bcv = E.bc(cache, event)
    sign_win = E.EVENT_DISPLAY[event]["sign"]
    s = E.unit_sign(odd, bcv, sign_win)                       # held-out sign
    lm = E.win_mask(bcv, LAT_WIN[event]) & ~E.win_mask(bcv, sign_win)
    t = bcv[lm]
    seg = full[:, lm]
    finite = np.isfinite(seg).all(1) & np.isfinite(s)
    pk = np.full(len(s), np.nan)
    com = np.full(len(s), np.nan)
    up = finite & (s > 0)
    dn = finite & (s < 0)
    if up.any():
        pk[up] = t[np.argmax(seg[up], axis=1)]
    if dn.any():
        pk[dn] = t[np.argmin(seg[dn], axis=1)]
    if finite.any():
        w = np.abs(seg[finite])
        com[finite] = (w @ t) / w.sum(1)
    return s, pk, com, up, dn


def boot_median_diff(a, b, n_boot=1000, seed=42):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    da = (np.median(a[rng.integers(0, len(a), (n_boot, len(a)))], axis=1)
          - np.median(b[rng.integers(0, len(b), (n_boot, len(b)))], axis=1))
    return float(np.median(da)), float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))


def lat_panel(ax, pk, up, dn, event, title):
    a = pk[up]; a = a[np.isfinite(a)]
    b = pk[dn]; b = b[np.isfinite(b)]
    bins = np.linspace(LAT_WIN[event][0], LAT_WIN[event][1], 31)
    ax.hist(a, bins=bins, color=UP, alpha=0.5, density=True, label=f"up (n={len(a)})")
    ax.hist(b, bins=bins, color=DOWN, alpha=0.5, density=True, label=f"down (n={len(b)})")
    ma, mb = np.median(a), np.median(b)
    ax.axvline(ma, color=UP, lw=1.5)
    ax.axvline(mb, color=DOWN, lw=1.5)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    p = mannwhitneyu(a, b).pvalue if (len(a) and len(b)) else np.nan
    md, lo, hi = boot_median_diff(a, b)
    ax.set_title(f"{title}\nMWU p={p:.1e} · up-down med {md:+.2f}s [{lo:+.2f},{hi:+.2f}]",
                 fontsize=C.FS["title"] - 2)
    ax.set_xlabel(f"peak latency from {REF[event]} (s)"); ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=C.FS["legend"])
    return dict(event=event, up_med=float(ma), down_med=float(mb), mwu_p=float(p),
                med_diff=md, ci_lo=lo, ci_hi=hi, n_up=len(a), n_down=len(b))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["celltype", "tf"], default="celltype",
                    help="celltype (default) = narrow/broad rows; tf = TF-responsive / "
                         "non-responsive rows (3 striatum mice only)")
    args = ap.parse_args()
    cache = E.load_event_cache()
    if args.group == "tf":
        if not C.has_tf_registry(C.SUBJECT):
            raise SystemExit(f"no TF registry for {C.SUBJECT} (3 striatum mice only)")
        resp, nonresp = E.tf_masks(cache, C.SUBJECT)
        masks = {"TF-responsive": resp, "Non-responsive": nonresp}
        groups = ["TF-responsive", "Non-responsive"]
    else:
        masks = E.celltype_masks(cache)   # COMMON width cutoff (FIX A), not per-subject cache labels
        groups = [C.NARROW, C.BROAD]
    name = "ws3_updown_timing_tf" if args.group == "tf" else "ws3_updown_timing"
    gtxt = " — TF-responsive vs non-responsive rows" if args.group == "tf" else ""

    lat = {ev: latencies(cache, ev) for ev in ("Change_ON", "Hit")}

    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.30)
    rows = []
    for ri, cell in enumerate(groups):
        cm = masks[cell]
        for ci, ev in enumerate(("Change_ON", "Hit")):
            s, pk, com, up, dn = lat[ev]
            ax = fig.add_subplot(gs[ri, ci])
            r = lat_panel(ax, pk, up & cm, dn & cm, ev,
                          f"{cell.split()[0]} · {E.EVENT_DISPLAY[ev]['short']}")
            r["celltype"] = cell
            rows.append(r)

    # sign-flip contingency (all units with sign at BOTH events)
    s_ch = lat["Change_ON"][0]
    s_li = lat["Hit"][0]
    both = np.isfinite(s_ch) & np.isfinite(s_li)
    ch_up = s_ch[both] > 0
    li_up = s_li[both] > 0
    tab = np.array([[np.sum(ch_up & li_up), np.sum(ch_up & ~li_up)],
                    [np.sum(~ch_up & li_up), np.sum(~ch_up & ~li_up)]])
    flip_frac = (tab[0, 1] + tab[1, 0]) / tab.sum()

    axT = fig.add_subplot(gs[0, 2])
    im = axT.imshow(tab, cmap="Purples")
    axT.set_xticks([0, 1]); axT.set_xticklabels(["lick up", "lick down"])
    axT.set_yticks([0, 1]); axT.set_yticklabels(["change up", "change down"])
    for i in range(2):
        for j in range(2):
            axT.text(j, i, str(tab[i, j]), ha="center", va="center",
                     color="white" if tab[i, j] > tab.max() / 2 else "black", fontsize=11)
    axT.set_title(f"Sign-flip: change x lick sign\nflip fraction = {flip_frac:.2f}", fontsize=C.FS["title"] - 2)
    fig.colorbar(im, ax=axT, fraction=0.046)

    # verdict: scan all cell-type x event results. CLEAN if any has p<0.05, CI excludes 0,
    # and a sizeable median latency gap (>=150 ms); MARGINAL if significant but small.
    axV = fig.add_subplot(gs[1, 2]); axV.axis("off")
    sig = [r for r in rows if r["mwu_p"] < 0.05 and r["ci_lo"] * r["ci_hi"] > 0
           and abs(r["med_diff"]) >= 0.05]
    big = [r for r in sig if abs(r["med_diff"]) >= 0.15]
    verdict = "CLEAN" if big else ("MARGINAL" if sig else "NONE")
    carriers = ", ".join(sorted({f"{r['celltype'].split()[0]}/{REF[r['event']]}" for r in big})) or "—"
    txt = ["WS3 up vs down relative timing", "",
           "peak latency (outside sign window):"]
    for r in rows:
        txt.append(f"  {r['celltype'].split()[0]:6s} {REF[r['event']]:6s}: "
                   f"up {r['up_med']:+.2f}s vs down {r['down_med']:+.2f}s "
                   f"(d {r['med_diff']:+.2f} [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}], p={r['mwu_p']:.0e})")
    txt += ["", f"sign-flip change<->lick: {flip_frac:.2f}",
            f"  (up-at-change can be down-at-lick)", "",
            "sign != cell identity (D1/D2/FSI);",
            "timing asymmetry MOTIVATES push-pull,",
            "identity needs optotagging.", "",
            f"carried by: {carriers}",
            f"VERDICT (up-vs-down timing): {verdict}"]
    axV.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=C.FS["caption"] - 1, family="monospace")

    fig.suptitle(f"{C.SUBJECT}: up- vs down-modulated relative timing (push-pull motivation){gtxt}",
                 fontsize=C.FS["suptitle"], y=0.99)
    fig.text(0.5, 0.005,
             "Held-out sign (odd trials); latency read OUTSIDE the sign-defining window. Per-unit "
             "peak latency, up vs down, Mann-Whitney + bootstrap CI on the median difference. "
             "Sign != cell identity; this motivates (not proves) push-pull.",
             ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.save_talk_figure(fig, name)
    print(f"[fig] wrote {out}")
    df = pd.DataFrame(rows)
    df.to_csv(C.stats_csv_path(name), index=False)
    print(f"[fig] wrote {C.stats_csv_path(name)}")
    print(df.to_string(index=False))
    print(f"\nsign-flip fraction: {flip_frac:.3f}  contingency:\n{tab}")
    print(f"WS3 VERDICT: {verdict}")


if __name__ == "__main__":
    main()
