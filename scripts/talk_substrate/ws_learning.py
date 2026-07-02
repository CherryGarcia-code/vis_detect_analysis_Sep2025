"""ACROSS-LEARNING (talk substrate, DESCRIPTIVE): does the neural EVIDENCE signal sharpen
naive->expert — the neural correlate of the behavioural sensitivity-up story?

PRIMARY: change-aligned big-vs-small scaling (the WS1 quantity) PER STAGE, on ALL GO TRIALS
(comparable at every stage; naive is hit-starved so hit-PSTHs aren't comparable). Metric =
per-unit mean(big - small) over a fixed post-change window; population mean + bootstrap CI.
ALSO (per the request): response magnitude per stage for Hit lick, Early FA, Late FA.

Stages from the manifest RAW `stage` field (Naive/Learning/Expert; Excluded dropped) — a
deliberate, noted departure from SESSION_FILTER's Naive->Learning merge, for THIS figure only.
All from the existing event caches (cache rows = unit-sessions; map session->stage). NOT N1.

CONFOUND GUARDS (a naive-vs-expert difference can be a RECORDING difference):
  - YIELD per stage (unit-sessions + unique-neurons/session) + a unit-count-MATCHED scaling
    (subsample every stage to the smallest stage's n).
  - CELL-TYPE composition per stage (narrow/broad fraction, common cutoff).
  - metric on go-trials/change-size (definable at all stages).

Usage: py scripts/talk_substrate/ws_learning.py
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
from visdetect.analysis.config import canonical_session_id as canon, _ALL_STAGE_COLORS  # noqa: E402
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS  # noqa: E402
from visdetect.analysis.utils import bootstrap_ci  # noqa: E402

C.setup_talk_style()
STAGES = ["Naive", "Learning", "Expert"]
SCOL = {s: _ALL_STAGE_COLORS.get(s, "#888888") for s in STAGES}
ANIMALS = [("BG_046", "Striatum DMS"), ("BG_039", "Striatum DMS"),
           ("BG_031", "Striatum VMS"), ("BG_038", "Cortex M1/S1 ref")]
CHANGE_WIN = (0.0, 1.0)               # fixed post-change window (matches cross-animal figure)
LICK_WIN = (-0.1, 0.3)                # peri-lick magnitude window (Hit / FA)
MAG_SPECS = [("Hit lick", "Hit", "all"), ("Early FA", "FA", "early"), ("Late FA", "FA", "late")]


def stage_map(subj):
    m = pd.read_csv(f"data/{subj}_staging_manifest.csv", dtype={"session_name": str})
    return dict(zip(m["session_name"].map(canon), m["stage"].astype(str)))


def unit_window_mean(cache, event, cond, win):
    m = E.mat(cache, event, cond, "full")
    wm = E.win_mask(E.bc(cache, event), win)
    seg = m[:, wm]
    out = np.full(m.shape[0], np.nan)
    ok = np.isfinite(seg).all(1)
    out[ok] = np.nanmean(seg[ok], axis=1)
    return out


def mean_ci_1d(x):
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    lo, hi = bootstrap_ci(x, n_bootstrap=1000, ci_level=0.95, axis=0, seed=42)
    return float(np.mean(x)), float(lo), float(hi), len(x)


def matched(diff, stg, n_min, n_draws=300, seed=42):
    rng = np.random.default_rng(seed)
    out = {}
    for s in STAGES:
        d = diff[(stg == s) & np.isfinite(diff)]
        if len(d) < n_min or n_min < 3:
            out[s] = (np.nan, np.nan, np.nan)
            continue
        mu = [np.mean(rng.choice(d, n_min, replace=False)) for _ in range(n_draws)]
        out[s] = (float(np.median(mu)), float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5)))
    return out


def analyse(subj):
    cache = E.load_event_cache(subj)
    sess = cache["unit_meta_session"].astype(str)
    smap = stage_map(subj)
    stg = np.array([smap.get(canon(s), "NA") for s in sess])
    narrow, broad, _ = C.common_celltype(cache, [subj], E.common_cut())
    diff = (unit_window_mean(cache, "Change_ON", "big", CHANGE_WIN)
            - unit_window_mean(cache, "Change_ON", "small", CHANGE_WIN))
    mags = {lbl: unit_window_mean(cache, ev, cond, LICK_WIN) for (lbl, ev, cond) in MAG_SPECS}
    res = {"subj": subj, "stage": {}, "mag": {lbl: {} for lbl, _e, _c in MAG_SPECS}}
    counts = {s: int(np.sum(stg == s)) for s in STAGES}
    n_min = min([c for c in counts.values() if c > 0], default=0)
    for s in STAGES:
        m = stg == s
        res["stage"][s] = dict(
            scaling=mean_ci_1d(diff[m]),
            n_unitsess=int(m.sum()),
            unique_per_sess=int(pd.Series(sess[m]).value_counts().median()) if m.any() else 0,
            n_sessions=int(pd.Series(sess[m]).nunique()) if m.any() else 0,
            narrow_frac=float(np.nanmean(narrow[m])) if m.any() else np.nan)
        for lbl, _e, _c in MAG_SPECS:
            res["mag"][lbl][s] = mean_ci_1d(mags[lbl][m])
    res["matched"] = matched(diff, stg, n_min)
    return res


def _pts(ax, getter, color, label=None):
    xs, ys, los, his = [], [], [], []
    for i, s in enumerate(STAGES):
        mu, lo, hi = getter(s)
        if np.isfinite(mu):
            xs.append(i); ys.append(mu); los.append(mu - lo); his.append(hi - mu)
    if xs:
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=color, capsize=3, label=label, lw=1.8)


def per_animal_fig(res, region):
    subj = res["subj"]
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)
    st = res["stage"]
    # A change scaling
    axA = fig.add_subplot(gs[0, 0])
    _pts(axA, lambda s: st[s]["scaling"][:3], "#d94801")
    axA.axhline(0, color="k", lw=0.8, ls=":")
    axA.set_xticks(range(3)); axA.set_xticklabels([f"{s}\n(n={st[s]['n_unitsess']})" for s in STAGES])
    axA.set_ylabel("big - small change response (z)")
    axA.set_title("A. Change-size (evidence) scaling per stage", fontsize=C.FS["title"])
    # B magnitude
    axB = fig.add_subplot(gs[0, 1])
    for (lbl, _e, _c), col in zip(MAG_SPECS, ["#4CAF50", "#ef6548", "#3474ae"]):
        _pts(axB, lambda s, L=lbl: res["mag"][L][s][:3], col, lbl)
    axB.axhline(0, color="k", lw=0.8, ls=":")
    axB.set_xticks(range(3)); axB.set_xticklabels(STAGES)
    axB.set_ylabel("peri-lick response (z)"); axB.set_title("B. Lick/FA response per stage", fontsize=C.FS["title"])
    axB.legend(frameon=False, fontsize=C.FS["legend"])
    # C matched scaling
    axC = fig.add_subplot(gs[0, 2])
    _pts(axC, lambda s: res["matched"][s], "#7a3b8f")
    axC.axhline(0, color="k", lw=0.8, ls=":")
    nmin = min([st[s]["n_unitsess"] for s in STAGES if st[s]["n_unitsess"] > 0], default=0)
    axC.set_xticks(range(3)); axC.set_xticklabels(STAGES)
    axC.set_ylabel("big - small (z)"); axC.set_title(f"C. Scaling, unit-count MATCHED (n={nmin})", fontsize=C.FS["title"])
    # D yield
    axD = fig.add_subplot(gs[1, 0])
    axD.bar(range(3), [st[s]["n_unitsess"] for s in STAGES], color=[SCOL[s] for s in STAGES])
    for i, s in enumerate(STAGES):
        axD.text(i, st[s]["n_unitsess"], f"{st[s]['n_sessions']}sess\n~{st[s]['unique_per_sess']}u/sess",
                 ha="center", va="bottom", fontsize=7)
    axD.set_xticks(range(3)); axD.set_xticklabels(STAGES)
    axD.set_ylabel("unit-sessions"); axD.set_title("D. YIELD per stage (guard)", fontsize=C.FS["title"])
    # E composition
    axE = fig.add_subplot(gs[1, 1])
    axE.bar(range(3), [100 * st[s]["narrow_frac"] for s in STAGES], color="#e74c3c", alpha=0.7)
    axE.set_xticks(range(3)); axE.set_xticklabels(STAGES)
    axE.set_ylabel("% narrow (common cut)"); axE.set_ylim(0, 100)
    axE.set_title("E. Cell-type composition (guard)", fontsize=C.FS["title"])
    # F verdict
    axF = fig.add_subplot(gs[1, 2]); axF.axis("off")
    nv, ev_ = st["Naive"]["scaling"], st["Expert"]["scaling"]   # (mu, lo, hi, n)
    mn, mx = res["matched"]["Naive"], res["matched"]["Expert"]
    enough = np.isfinite(nv[0]) and np.isfinite(ev_[0])
    raw_rise = enough and ev_[0] > nv[0]
    nonoverlap = enough and ev_[1] > nv[2]                       # Expert CI-lo above Naive CI-hi
    matched_rise = np.isfinite(mn[0]) and np.isfinite(mx[0]) and mx[0] > mn[0]
    if not enough:
        verdict = "INSUFFICIENT (naive/expert n too low)"
    elif raw_rise and nonoverlap and matched_rise:
        verdict = "CLEAN (sharpens; matched + non-overlapping CIs)"
    elif raw_rise and matched_rise:
        verdict = "MARGINAL (rises; CIs overlap)"
    elif raw_rise:
        verdict = "MARGINAL (raw rise only; matched flat)"
    else:
        verdict = "NO CLEAR TREND"
    txt = [f"{subj} ({region})", "", "change-size scaling (z):"]
    for s in STAGES:
        v = st[s]["scaling"]
        txt.append(f"  {s:9s}: {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] (n={v[3]})"
                   if np.isfinite(v[0]) else f"  {s:9s}: n/a (n={v[3]})")
    txt += ["", "matched (z):"] + [f"  {s:9s}: {res['matched'][s][0]:+.3f}"
                                    if np.isfinite(res['matched'][s][0]) else f"  {s:9s}: n/a"
                                    for s in STAGES]
    txt += ["", f"VERDICT: {verdict}"]
    axF.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=C.FS["caption"], family="monospace")

    fig.suptitle(f"{subj} ({region}): change-size (evidence) scaling ACROSS LEARNING "
                 "(descriptive)", fontsize=C.FS["suptitle"], y=0.99)
    fig.text(0.5, 0.005,
             "Stages = manifest RAW stage (Naive/Learning/Expert). Metric = per-unit mean(big-small) "
             f"in {CHANGE_WIN}s post-change, ALL go trials; bands = bootstrap 95% CI over unit-sessions. "
             "Guards: yield (D) + unit-count-matched scaling (C) + composition (E). NOT N1; descriptive.",
             ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.FIG_DIR.parent / subj / "ws_learning_changesize.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {out}  verdict={verdict}")
    return verdict


def main():
    results = {}
    summ = []
    for subj, region in ANIMALS:
        try:
            res = analyse(subj)
        except Exception as e:  # noqa: BLE001
            print(f"{subj}: FAILED {e}"); continue
        v = per_animal_fig(res, region)
        results[subj] = res
        for s in STAGES:
            sc = res["stage"][s]["scaling"]
            summ.append(dict(subj=subj, region=region, stage=s, scaling=round(sc[0], 4) if np.isfinite(sc[0]) else None,
                             ci_lo=round(sc[1], 4) if np.isfinite(sc[1]) else None,
                             ci_hi=round(sc[2], 4) if np.isfinite(sc[2]) else None,
                             n_unitsess=res["stage"][s]["n_unitsess"],
                             n_sessions=res["stage"][s]["n_sessions"],
                             narrow_frac=round(res["stage"][s]["narrow_frac"], 3) if np.isfinite(res["stage"][s]["narrow_frac"]) else None,
                             matched=round(res["matched"][s][0], 4) if np.isfinite(res["matched"][s][0]) else None,
                             verdict=v))

    # cross-animal summary: scaling per stage, all animals
    fig, ax = plt.subplots(figsize=(10, 6))
    acol = {"BG_046": "#238b45", "BG_039": "#74c476", "BG_031": "#3474ae", "BG_038": "#969696"}
    for subj, region in ANIMALS:
        if subj not in results:
            continue
        st = results[subj]["stage"]
        xs = [i for i, s in enumerate(STAGES) if np.isfinite(st[s]["scaling"][0])]
        ys = [st[s]["scaling"][0] for s in STAGES if np.isfinite(st[s]["scaling"][0])]
        lo = [st[s]["scaling"][0] - st[s]["scaling"][1] for s in STAGES if np.isfinite(st[s]["scaling"][0])]
        hi = [st[s]["scaling"][2] - st[s]["scaling"][0] for s in STAGES if np.isfinite(st[s]["scaling"][0])]
        ax.errorbar([x + 0.04 * list(acol).index(subj) for x in xs], ys, yerr=[lo, hi],
                    fmt="o-", color=acol[subj], capsize=3, label=f"{subj} ({region})", lw=1.8)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels(STAGES)
    ax.set_ylabel("big - small change response (z)")
    ax.set_title("Change-size (evidence) scaling ACROSS LEARNING — all animals", fontsize=C.FS["title"])
    ax.legend(frameon=False, fontsize=C.FS["legend"])
    fig.text(0.5, -0.02, "Per-unit mean(big-small) in 0-1 s post-change, all go trials; bootstrap 95% CI "
             "over unit-sessions. Naive is thin (2-3 sessions); see per-animal yield guards.",
             ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.FIG_DIR.parent / "ws_learning_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {out}")
    df = pd.DataFrame(summ)
    df.to_csv(C.FIG_DIR.parent / "ws_learning_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
