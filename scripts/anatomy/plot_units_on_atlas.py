# scripts/anatomy/plot_units_on_atlas.py
"""Per-unit activity / state maps on a coronal atlas slice, for one session.

Colours localized units (peak-channel CCF) by a chosen per-unit metric on a coronal
Allen slice (context + striatum zoom). Metrics:
  - "fr"              : mean firing rate over the session (sequential viridis)
  - "change_response": Δfiring to the TF change (hit/miss, baseline-subtracted; RdBu_r)
  - "state_contrast" : baseline FR StimSens − Impulsive (custom Impulsive→StimSens map)
  - "preferred_state": state with highest baseline FR (categorical, state palette)

Alignment uses visdetect.analysis.align.get_event_times_by_trial (valid-outcome safe:
Change_ON only on hit/miss; Baseline_ON valid for all). State per trial comes from the
state-labeler tags. Reuses the coronal slice renderer from plot_sites_on_atlas.

Effect size: --effect auroc (default) colours evoked/state maps by signed auROC-0.5
(bounded, comparable across SPNs/FSIs — avoids high-FR units dominating the scale);
--effect raw uses Δ Hz. Non-significant units are hidden by default (per-unit Wilcoxon/
Mann-Whitney + Benjamini-Hochberg FDR; --show-nonsig to keep them). `fr` uses a log
colour scale (--fr-scale) and is never masked (it has no response test).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from scipy.stats import mannwhitneyu, wilcoxon

from visdetect.anatomy.atlas import AllenAtlas
from visdetect.anatomy.tracks import load_track_artifact
from visdetect.anatomy.stereotaxic import CoordMap, pia_dv_um
from visdetect.analysis.config import STATE_LABEL_COLORS
from visdetect.analysis.utils import compute_auroc, fdr_correct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_sites_on_atlas import coronal_coarse_image, dominant_region_label

MOOD_STATES = ["Impulsive", "StimSens", "Disengaged"]

METRIC_INFO = {
    "fr": dict(label="firing rate (Hz)", diverging=False, cmap="viridis"),
    "change_response": dict(label="Δ firing to change (Hz)", diverging=True, cmap="RdBu_r"),
    "state_contrast": dict(label="baseline ΔFR (Hz)   Impulsive ←→ StimSens",
                           diverging=True, cmap=None),  # custom below
    "preferred_state": dict(label="preferred state", diverging=False, cmap=None),
    "lick_hit": dict(label="Δ firing at Hit lick (Hz)", diverging=True, cmap="RdBu_r"),
    "lick_fa": dict(label="Δ firing at early (FA) lick (Hz)", diverging=True, cmap="RdBu_r"),
    "lick_contrast": dict(label="Δ firing  Hit-lick − early-lick (Hz)", diverging=True, cmap="RdBu_r"),
}

# peri-lick response windows (s, relative to lick onset): response vs pre-lick baseline
LICK_POST = (0.0, 0.2)
LICK_PRE = (-0.4, -0.2)


def _state_contrast_cmap():
    """Custom diverging map matching the state palette: Impulsive(red) → white → StimSens(blue)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "imp_stim", [STATE_LABEL_COLORS["Impulsive"], "#f7f7f7", STATE_LABEL_COLORS["StimSens"]])


def _counts(spikes, events, w0, w1):
    """Spike counts in [e+w0, e+w1) per event (NaN event -> NaN)."""
    st = np.sort(np.asarray(spikes, float))
    out = np.full(len(events), np.nan)
    for i, e in enumerate(events):
        if not np.isfinite(e):
            continue
        out[i] = np.searchsorted(st, e + w1) - np.searchsorted(st, e + w0)
    return out


def _rate(sp, ev, w):
    return _counts(sp, ev, w[0], w[1]) / (w[1] - w[0])


def _evoked(sp, ev, post_w, pre_w, effect, min_trials):
    """(value, pval) for an evoked response (post vs pre around ev).

    effect 'auroc' -> signed auROC-0.5 (>0 = excited); 'raw' -> mean Δ Hz.
    pval = paired Wilcoxon signed-rank on per-trial (post-pre)."""
    post, pre = _rate(sp, ev, post_w), _rate(sp, ev, pre_w)
    m = np.isfinite(post) & np.isfinite(pre)
    post, pre = post[m], pre[m]
    if len(post) < min_trials:
        return np.nan, np.nan
    d = post - pre
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(post, pre)
        except Exception:
            p = np.nan
    val = (compute_auroc(post, pre) - 0.5) if effect == "auroc" else float(np.mean(d))
    return val, p


def _two_state(sp, base, base_w, state, A, B, effect, min_trials):
    """(value, pval) for baseline FR in state A vs B.

    effect 'auroc' -> auROC(A,B)-0.5 (>0 = higher in A); 'raw' -> mean(A)-mean(B) Hz.
    pval = Mann-Whitney U (unpaired)."""
    fr = _rate(sp, base, base_w)
    a = fr[(state == A) & np.isfinite(fr)]
    b = fr[(state == B) & np.isfinite(fr)]
    if len(a) < min_trials or len(b) < min_trials:
        return np.nan, np.nan
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        p = np.nan
    val = (compute_auroc(a, b) - 0.5) if effect == "auroc" else float(np.mean(a) - np.mean(b))
    return val, p


def compute_unit_metric(session, tags, metric, *, effect="auroc", baseline_win=(0.0, 1.0),
                        change_post=(0.0, 0.3), change_pre=(-0.3, 0.0),
                        min_trials=5) -> pd.DataFrame:
    """Per-unit (value, pval) for `metric`. Returns df[cluster_id, value, pval].

    effect: "auroc" (signed auROC-0.5, comparable across the SPN/FSI population) or
    "raw" (Δ Hz / Hz difference). pval = per-unit significance (paired Wilcoxon for
    evoked, Mann-Whitney for state); NaN for `fr` and for units with too few trials.
    """
    from visdetect.analysis.align import get_event_times_by_trial
    n = len(session.trials)
    base = np.array(get_event_times_by_trial(session, "Baseline_ON"), float)
    chg = np.array(get_event_times_by_trial(session, "Change_ON"), float)
    hit_t = fa_t = None
    if metric in ("lick_hit", "lick_fa", "lick_contrast"):
        hit_t = np.array(get_event_times_by_trial(session, "Hit"), float)   # hit response-lick times
        fa_t = np.array(get_event_times_by_trial(session, "FA"), float)     # early (FA) lick times
    state = tags.set_index("trial_idx")["state_label"].reindex(range(n)).values
    units = session.good_and_stable_ids or [c.cluster_id for c in session.clusters]
    spk = {c.cluster_id: np.asarray(c.spike_times, float) for c in session.clusters}

    rows = []
    for cid in units:
        sp = spk.get(cid, np.array([]))
        val, pval = np.nan, np.nan
        if metric == "fr":
            t0 = np.nanmin(base)
            t1 = np.nanmax(np.where(np.isfinite(chg), chg, base)) + 2.0
            val = float(((sp >= t0) & (sp < t1)).sum() / max(t1 - t0, 1e-9))
        elif metric == "change_response":
            val, pval = _evoked(sp, chg, change_post, change_pre, effect, min_trials)
        elif metric == "lick_hit":
            val, pval = _evoked(sp, hit_t, LICK_POST, LICK_PRE, effect, min_trials)
        elif metric == "lick_fa":
            val, pval = _evoked(sp, fa_t, LICK_POST, LICK_PRE, effect, min_trials)
        elif metric == "lick_contrast":
            vh, ph = _evoked(sp, hit_t, LICK_POST, LICK_PRE, effect, min_trials)
            vf, pf = _evoked(sp, fa_t, LICK_POST, LICK_PRE, effect, min_trials)
            val = (vh - vf) if (np.isfinite(vh) and np.isfinite(vf)) else np.nan
            ps = [x for x in (ph, pf) if np.isfinite(x)]
            pval = min(ps) if ps else np.nan
        elif metric == "state_contrast":
            val, pval = _two_state(sp, base, baseline_win, state, "StimSens", "Impulsive", effect, min_trials)
        elif metric == "preferred_state":
            fr = _rate(sp, base, baseline_win)
            per = {}
            for stt in MOOD_STATES:
                m = (state == stt) & np.isfinite(fr)
                per[stt] = float(np.nanmean(fr[m])) if m.sum() >= min_trials else np.nan
            avail = {k: v for k, v in per.items() if np.isfinite(v)}
            val = max(avail, key=avail.get) if avail else None
            _, pval = _two_state(sp, base, baseline_win, state, "StimSens", "Impulsive", "raw", min_trials)
        rows.append({"cluster_id": int(cid), "value": val, "pval": pval})
    return pd.DataFrame(rows)


def fdr_significant(pvals, alpha=0.05):
    """Boolean significance after Benjamini-Hochberg FDR; NaN p-values -> False."""
    pv = np.asarray(pvals, float)
    sig = np.zeros(len(pv), bool)
    valid = np.isfinite(pv)
    if valid.any():
        sig[np.where(valid)[0]] = fdr_correct(pv[valid], alpha=alpha)
    return sig


def _draw(ax, atlas, ap_um, art, df, metric, *, zoom, sc, win, cmap, norm, cm):
    img, extent = coronal_coarse_image(atlas, ap_um)
    img, extent = cm.image(img, extent)
    ax.imshow(img, extent=extent, origin="upper", interpolation="nearest", aspect="equal")
    for s in art.shanks:
        poly = np.asarray(s.ccf_polyline, float)
        ax.plot(cm.x(poly[:, 1]), cm.y(poly[:, 2]), "-", lw=0.9, color="0.4", alpha=0.7, zorder=4)
    lw = 0.25 if zoom else 0.15
    if metric == "preferred_state":
        for stt in MOOD_STATES:
            d = df[df["value"] == stt]
            if len(d):
                ax.scatter(cm.x(d.ccf_ml), cm.y(d.ccf_dv), s=sc, c=STATE_LABEL_COLORS[stt],
                           edgecolors="white", linewidths=lw, alpha=0.85,
                           zorder=5, label=stt)
    else:
        good = df[np.isfinite(df["value"])]
        ax.scatter(cm.x(good.ccf_ml), cm.y(good.ccf_dv), c=good["value"], cmap=cmap, norm=norm,
                   s=sc, edgecolors="white", linewidths=lw, alpha=0.85, zorder=5)
    ax.set_xlabel(cm.xlabel); ax.set_ylabel(cm.ylabel)
    if zoom:
        ax.set_xlim(win[0], win[1]); ax.set_ylim(win[3], win[2])


def _cbar_label(metric, effect, fr_scale, info):
    if metric == "fr":
        return info["label"] + (" — log scale" if fr_scale == "log" else "")
    if effect == "auroc":
        if metric == "state_contrast":
            return "auROC − 0.5   Impulsive ←→ StimSens"
        return "auROC − 0.5 (response vs baseline)"
    return info["label"]


def plot_units_on_atlas(subject, session_name, metric, df, art, out_png,
                        atlas=None, *, effect="auroc", fr_scale="log", n_total=None,
                        coords="ccf", stage="Expert", masked=True, n_sig=None) -> str:
    """Render the 2-panel (whole section + zoom) map. `df` has ccf_ml/ccf_dv/ccf_ap/value
    (already significance-filtered upstream). n_total = pre-mask unit count for the label.
    coords="stereotaxic" relabels/flips axes to Bregma-referenced mm (left on the left).
    Region words in the titles come from `df.region_coarse` (so a cortical probe reads
    'cortex', not 'CPu')."""
    atlas = atlas or AllenAtlas()
    info = METRIC_INFO[metric]
    ap_um = float(df["ccf_ap"].median())
    cm = CoordMap(coords, pia_dv_um(art))
    reg = dominant_region_label(df)            # 'cortex' / 'CPu' / ... from the units
    reg_zoom = reg.split("/")[0]; reg_zoom = reg_zoom[:1].upper() + reg_zoom[1:]
    # zoom window in plot coordinates (sorted to absorb the ML flip)
    xs = np.sort(cm.x(np.array([df.ccf_ml.min() - 700, df.ccf_ml.max() + 700])))
    ys = np.sort(cm.y(np.array([df.ccf_dv.min() - 1500, df.ccf_dv.max() + 500])))
    xlo, xhi = float(xs[0]), float(xs[1]); ylo, yhi = float(ys[0]), float(ys[1])
    win = (xlo, xhi, ylo, yhi)

    cmap = norm = None
    if metric != "preferred_state":
        vals = pd.to_numeric(df["value"], errors="coerce").to_numpy()
        if metric == "fr":
            cmap = plt.get_cmap("viridis")
            pos = vals[np.isfinite(vals) & (vals > 0)]
            if fr_scale == "log" and pos.size:
                norm = mcolors.LogNorm(vmin=max(float(pos.min()), 0.1), vmax=float(np.nanmax(vals)))
            else:
                norm = mcolors.Normalize(np.nanmin(vals), np.nanmax(vals))
        elif info["diverging"]:
            vmax = np.nanmax(np.abs(vals)) or 1.0
            norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
            cmap = _state_contrast_cmap() if metric == "state_contrast" else plt.get_cmap(info["cmap"])
        else:
            norm = mcolors.Normalize(np.nanmin(vals), np.nanmax(vals))
            cmap = plt.get_cmap(info["cmap"])

    fig = plt.figure(figsize=(13, 6.2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax, zoom in ((axA, False), (axB, True)):
        _draw(ax, atlas, ap_um, art, df, metric, zoom=zoom,
              sc=13 if zoom else 5, win=win, cmap=cmap, norm=norm, cm=cm)
    axA.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, ec="k",
                            lw=1.0, ls="--", zorder=6))
    n_plot = len(df)
    if metric == "fr" or n_total is None:
        n_lab = f"n = {n_plot} units"
    elif masked:
        n_lab = f"n = {n_plot}/{n_total} units (FDR<0.05)"
    else:  # showing all units (incl. non-significant) — say how many actually pass FDR
        n_lab = f"n = {n_plot} units" + (f" · {n_sig} FDR-sig" if n_sig is not None else "")
    axA.set_title(f"A. Coronal section ({cm.ap_title(ap_um)})", fontweight="bold", fontsize=12)
    axA.text(0.02, 0.02, n_lab, transform=axA.transAxes, fontsize=8, color="0.3", va="bottom")
    axB.set_title(f"B. {reg_zoom} zoom", fontweight="bold", fontsize=12)
    # scale bar (500 um -> 0.5 mm in stereotaxic)
    x0 = xlo + 0.08 * (xhi - xlo); y0 = yhi - 0.08 * (yhi - ylo)
    axB.plot([x0, x0 + cm.length(500)], [y0, y0], "k-", lw=2.5)
    axB.text(x0 + cm.length(250), y0 - 0.02 * (yhi - ylo), "500 µm",
             ha="center", va="bottom", fontsize=8)

    if metric == "preferred_state":
        handles = [Line2D([0], [0], marker="o", ls="", mec="white", mew=0.3,
                          mfc=STATE_LABEL_COLORS[s], label=s) for s in MOOD_STATES
                   if (df["value"] == s).any()]
        axB.legend(handles=handles, title="preferred state", loc="upper right",
                   frameon=False, fontsize=8, title_fontsize=8)
    else:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axB, fraction=0.046, pad=0.04)
        cb.set_label(_cbar_label(metric, effect, fr_scale, info), fontsize=9)

    fig.suptitle(f"{subject} {session_name} ({stage}) — {info['label'].split('(')[0].strip()}"
                 f"  ·  {art.hemisphere} {reg}", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


def _build(subject, session_name, metric, anatomy_dir, out_png, session=None, tags=None,
           *, effect="auroc", mask_nonsig=True, fr_scale="log", coords="ccf",
           coarse_region=None, stage="Expert"):
    from visdetect.core.session import load_session
    # DDMMYYYY token; zero-pad so leading-zero dates (e.g. 08072025) resolve their
    # pkl/tag files (str(int(...)) would drop the zero -> 8072025 -> FileNotFound).
    tok = str(session_name).zfill(8)
    if session is None:
        session = load_session(os.path.join("data", "pkls", subject, f"{subject}_{tok}.pkl"))
    if tags is None:
        tags = pd.read_csv(os.path.join("data", "cache", "state_tags", subject, f"{tok}.csv"))
    met = compute_unit_metric(session, tags, metric, effect=effect)
    ua = pd.read_csv(os.path.join(anatomy_dir, "unit_anatomy.csv"))
    ua = ua[ua.session_name == int(tok)]
    if coarse_region:                          # e.g. restrict to cortical units (CTX)
        ua = ua[ua.region_coarse == coarse_region]
    df = ua.merge(met, on="cluster_id")
    # keep only units with a computable value
    if metric == "preferred_state":
        df = df[df["value"].isin(MOOD_STATES)]
    else:
        df = df[np.isfinite(pd.to_numeric(df["value"], errors="coerce"))]
    n_total = len(df)
    n_sig = int(fdr_significant(df["pval"].to_numpy()).sum()) if metric != "fr" else None
    # hide non-responsive units (FDR-significant only); fr has no significance test
    masked = mask_nonsig and metric != "fr"
    if masked:
        df = df[fdr_significant(df["pval"].to_numpy())]
    art = load_track_artifact(os.path.join(anatomy_dir, f"{subject}_shank_tracks.json"))
    if len(df) == 0:
        return None, session, tags
    out = plot_units_on_atlas(subject, tok, metric, df, art, out_png,
                              effect=effect, fr_scale=fr_scale, n_total=n_total, coords=coords,
                              stage=stage, masked=masked, n_sig=n_sig)
    return out, session, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--metric", required=True,
                    choices=list(METRIC_INFO), nargs="+")
    ap.add_argument("--anatomy-dir", default=None, help="defaults to data/anatomy/<subject>")
    ap.add_argument("--effect", choices=["auroc", "raw"], default="auroc",
                    help="auroc (signed auROC-0.5, cross-unit comparable; default) or raw (Δ Hz)")
    ap.add_argument("--show-nonsig", action="store_true",
                    help="also show units that are NOT FDR-significant (default: hide them)")
    ap.add_argument("--fr-scale", choices=["log", "linear"], default="log")
    ap.add_argument("--coords", choices=["ccf", "stereotaxic"], default="ccf",
                    help="ccf (raw Allen microns; default) or stereotaxic "
                         "(Bregma-referenced mm, ML flipped so anatomical left is on the left)")
    ap.add_argument("--coarse-region", default=None,
                    help="restrict to units in one coarse region (e.g. CTX, CP, GPe). "
                         "Titles auto-label the region either way.")
    ap.add_argument("--stage", default="Expert", help="stage label shown in the title")
    args = ap.parse_args()
    args.anatomy_dir = args.anatomy_dir or os.path.join("data", "anatomy", args.subject)
    coord_suffix = "_stereotaxic" if args.coords == "stereotaxic" else ""
    session = tags = None
    for m in args.metric:
        suffix = "" if m == "fr" else f"_{args.effect}"
        out = os.path.join("FIGURES", "anatomy", args.subject,
                           f"{args.subject}_{args.session}_{m}{suffix}{coord_suffix}.png")
        out, session, tags = _build(args.subject, args.session, m, args.anatomy_dir, out,
                                    session=session, tags=tags, effect=args.effect,
                                    mask_nonsig=not args.show_nonsig, fr_scale=args.fr_scale,
                                    coords=args.coords, coarse_region=args.coarse_region,
                                    stage=args.stage)
        print(f"wrote {out}" if out else f"{m}: no units to plot (none significant?)")


if __name__ == "__main__":
    main()
