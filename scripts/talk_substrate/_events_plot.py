"""Shared rendering helpers for the talk-substrate event-PSTH figures.

Reads the unified cache (build_event_cache.py) and provides small primitives:
per-unit -> population mean+/-SEM, held-out modulation-sign assignment, and a
trace-panel drawer. Keeps every figure script thin.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS  # noqa: E402
from visdetect.analysis.utils import bootstrap_ci  # noqa: E402  (canonical: 1000 boot, seed 42)

def cache_path(subject=None):
    return C.CACHE_DIR / f"event_psth_cache_{subject or C.SUBJECT}.npz"


CACHE = cache_path()  # subject-scoped (default = active subject)

# Display config per event (x-label + canonical baseline + sign/response window).
def _disp(event, xlabel, short):
    return dict(xlabel=xlabel, short=short,
                baseline=EVENT_RESPONSIVENESS_WINDOWS[event][0],
                sign=EVENT_RESPONSIVENESS_WINDOWS[event][1])


EVENT_DISPLAY = {
    "Baseline_ON": _disp("Baseline_ON", "time from baseline onset (s)", "Baseline onset"),
    "Change_ON":   _disp("Change_ON", "time from change onset (s)", "Change onset"),
    "Hit":         _disp("Hit", "time from response lick (s)", "Response lick (hit)"),
    "FA":          _disp("FA", "time from early lick (s)", "Early lick (FA)"),
}


def load_event_cache(subject=None) -> dict:
    d = np.load(cache_path(subject), allow_pickle=True)
    return {k: d[k] for k in d.files}


def pool_caches(subjects) -> dict:
    """Pool several subjects' caches into one (concatenate unit-session rows). Only keys
    present in ALL caches are kept (e.g. state conds, BG_046-only, are dropped). Use for
    coordinate-compatible groups only (e.g. DMS = BG_046 + BG_039)."""
    caches = [load_event_cache(s) for s in subjects]
    common = set(caches[0])
    for c in caches[1:]:
        common &= set(c)
    out = {}
    for k in sorted(common):
        if k.startswith("bc__"):
            out[k] = caches[0][k]
        elif caches[0][k].ndim == 2:          # trace matrices (n_units, n_bins)
            out[k] = np.vstack([c[k] for c in caches])
        else:                                  # unit_meta_* / ntr (1-D)
            out[k] = np.concatenate([c[k] for c in caches])
    return out


def bc(cache, event) -> np.ndarray:
    return cache[f"bc__{event}"]


def mat(cache, event, cond, half="full") -> np.ndarray:
    return cache[f"{event}__{cond}__{half}"]


def celltype(cache) -> np.ndarray:
    return cache["unit_meta_celltype"]


def win_mask(bcarr, w) -> np.ndarray:
    return (bcarr >= w[0]) & (bcarr <= w[1])


def mean_sem(matrix: np.ndarray, row_mask=None):
    """Population mean +/- SEM across units (rows). Drops all-NaN/partial rows.
    Returns (mean, sem, n_units)."""
    finite = np.isfinite(matrix).all(axis=1)
    if row_mask is not None:
        finite = finite & row_mask
    M = matrix[finite]
    n = M.shape[0]
    if n == 0:
        w = matrix.shape[1]
        return np.full(w, np.nan), np.full(w, np.nan), 0
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0) / np.sqrt(n), n


def unit_sign(odd_matrix: np.ndarray, bcarr: np.ndarray, sign_win) -> np.ndarray:
    """Signed mean z in the sign window, per unit (held-out odd half). NaN if undefined."""
    m = win_mask(bcarr, sign_win)
    return np.nanmean(odd_matrix[:, m], axis=1)


def mean_ci(matrix: np.ndarray, row_mask=None):
    """Population mean + bootstrap 95% CI across units (rows), reusing the canonical
    utils.bootstrap_ci (1000 resamples, seed 42, percentile). Drops partial-NaN rows.
    Returns (mean, ci_lo, ci_hi, n_units)."""
    finite = np.isfinite(matrix).all(axis=1)
    if row_mask is not None:
        finite = finite & row_mask
    M = matrix[finite]
    n = M.shape[0]
    w = matrix.shape[1]
    if n == 0:
        return np.full(w, np.nan), np.full(w, np.nan), np.full(w, np.nan), 0
    mean = M.mean(axis=0)
    lo, hi = bootstrap_ci(M, n_bootstrap=1000, ci_level=0.95, axis=0, seed=42)
    return mean, lo, hi, n


_COMMON_CUT = None


def common_cut():
    """Cached COMMON narrow/broad width cutoff (live pooled-GMM over all animals)."""
    global _COMMON_CUT
    if _COMMON_CUT is None:
        _COMMON_CUT, _ = C.common_t2p_cutoff()
    return _COMMON_CUT


def celltype_masks(cache, subjects=None) -> dict:
    """{display celltype -> boolean row mask} under the COMMON width cutoff (FIX A: one cutoff
    everywhere). subjects defaults to the active subject; pass a list for pooled caches."""
    subs = list(subjects) if subjects else [C.SUBJECT]
    narrow, broad, _ = C.common_celltype(cache, subs, common_cut())
    return {C.NARROW: narrow, C.BROAD: broad}


def celltype_array(cache, subjects=None) -> np.ndarray:
    """Per-unit display cell-type labels (NARROW/BROAD/UNKNOWN) under the COMMON cutoff —
    the array analogue of celltype_masks (FIX A). Use instead of celltype() (cache-baked
    per-subject labels) wherever cross-figure cutoff consistency matters."""
    masks = celltype_masks(cache, subjects)
    out = np.full(len(cache["unit_meta_celltype"]), C.UNKNOWN, dtype=object)
    out[masks[C.NARROW]] = C.NARROW
    out[masks[C.BROAD]] = C.BROAD
    return out


def signed_mask(cache, event, celltype_mask, sign):
    """Row mask = units of a cell type whose OVERALL modulation at this event is up/down.

    Sign = mean z in the event's canonical response window over ALL trials. Used as a
    fixed per-unit grouping so the *condition* contrast within a group stays non-circular
    (the existence of the up/down split itself is shown held-out in Fig B). Every analysis
    is thus split by BOTH cell type (narrow/broad) and modulation sign (up/down)."""
    full = mat(cache, event, "all", "full")
    s = unit_sign(full, bc(cache, event), EVENT_DISPLAY[event]["sign"])
    base = np.isfinite(full).all(axis=1) & np.isfinite(s)
    if celltype_mask is not None:
        base = base & celltype_mask
    return (base & (s > 0)) if sign == "up" else (base & (s < 0))


def plot_band(ax, bcarr, mean, lo, hi, color, label, lw=1.9):
    ax.plot(bcarr, mean, color=color, lw=lw, label=label, zorder=3)
    ax.fill_between(bcarr, lo, hi, color=color, alpha=0.2, zorder=2)


def cond_panel(ax, cache, event, cond_specs, row_mask=None, title=None):
    """Draw a condition contrast (mean + bootstrap CI) for one event into ax.
    cond_specs: list of (cond, color, label)."""
    disp = EVENT_DISPLAY[event]
    bcarr = bc(cache, event)
    decorate(ax, event, baseline_win=disp["baseline"])
    rows = []
    for cond, color, label in cond_specs:
        m, lo, hi, n = mean_ci(mat(cache, event, cond, "full"), row_mask)
        plot_band(ax, bcarr, m, lo, hi, color, f"{label} (n={n}u)")
        pk, pt = peak_stat(bcarr, m)
        rows.append({"event": event, "cond": cond, "n_units": n,
                     "peak_z": round(pk, 3), "peak_time_s": round(pt, 3)})
    if title:
        ax.set_title(title, fontsize=C.FS["title"])
    ax.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)
    return rows


def multi_cond_panel(ax, cache, specs, decor_event, row_mask=None, title=None):
    """Draw traces that may come from DIFFERENT events on one axis (e.g. response-aligned
    Hit vs FA). specs: list of (event, cond, color, label). Decorated by decor_event."""
    disp = EVENT_DISPLAY[decor_event]
    decorate(ax, decor_event, baseline_win=disp["baseline"])
    rows = []
    for event, cond, color, label in specs:
        m, lo, hi, n = mean_ci(mat(cache, event, cond, "full"), row_mask)
        plot_band(ax, bc(cache, event), m, lo, hi, color, f"{label} (n={n}u)")
        pk, pt = peak_stat(bc(cache, event), m)
        rows.append({"event": event, "cond": cond, "n_units": n,
                     "peak_z": round(pk, 3), "peak_time_s": round(pt, 3)})
    if title:
        ax.set_title(title, fontsize=C.FS["title"])
    ax.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)
    return rows


def sign_panel(ax, cache, event, row_mask=None, title=None,
               up_color=C.SIGN_COLORS["up"], down_color=C.SIGN_COLORS["down"], reref_window=None):
    """Up- vs down-modulated split (held-out sign), mean + bootstrap CI, into ax.

    reref_window: optional (lo, hi) s window to RE-REFERENCE (re-centre) this event's cached
    z-traces to, instead of the canonical baseline. Subtracts each unit's mean over that window
    from the odd & even traces (exact re-centring of the cached z — see demo_baseline_window.py).
    The SD stays the canonical (longer, more stable) estimate on purpose: a short near-onset
    window gives a noisy SD / z-blowup, so we re-centre on it but keep the stable denominator.
    Used to anchor the Baseline_ON split at onset WITHOUT touching any other event/column."""
    disp = EVENT_DISPLAY[event]
    bcarr = bc(cache, event)
    base_win = reref_window if reref_window is not None else disp["baseline"]
    decorate(ax, event, baseline_win=base_win)
    ax.axvspan(disp["sign"][0], disp["sign"][1], color="#ffe08a", alpha=0.35, zorder=0)
    odd = mat(cache, event, "all", "odd")
    even = mat(cache, event, "all", "even")
    if reref_window is not None:
        rw = win_mask(bcarr, reref_window)
        odd = odd - np.nanmean(odd[:, rw], axis=1, keepdims=True)
        even = even - np.nanmean(even[:, rw], axis=1, keepdims=True)
    s = unit_sign(odd, bcarr, disp["sign"])
    base = np.isfinite(odd).all(1) & np.isfinite(even).all(1) & np.isfinite(s)
    if row_mask is not None:
        base = base & row_mask
    mp, lop, hip, nu = mean_ci(even, base & (s > 0))
    mn, lon, hin, nd = mean_ci(even, base & (s < 0))
    plot_band(ax, bcarr, mp, lop, hip, up_color, f"Up-modulated (n={nu}u)")
    plot_band(ax, bcarr, mn, lon, hin, down_color, f"Down-modulated (n={nd}u)")
    if title:
        ax.set_title(title, fontsize=C.FS["title"])
    ax.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)
    return [{"event": event, "group": "up", "n_units": nu},
            {"event": event, "group": "down", "n_units": nd}]


def plot_trace(ax, bcarr, mean, sem, color, label, lw=1.9):
    ax.plot(bcarr, mean, color=color, lw=lw, label=label, zorder=3)
    ax.fill_between(bcarr, mean - sem, mean + sem, color=color, alpha=0.2, zorder=2)


def decorate(ax, event, baseline_win=None, ylabel="z-score (shared baseline)"):
    if baseline_win is not None:
        ax.axvspan(baseline_win[0], baseline_win[1], color="0.85", alpha=0.5, zorder=0)
    ax.axvline(0, color="k", lw=1.0, zorder=1)
    ax.axhline(0, color="0.6", lw=0.7, ls=":", zorder=1)
    ax.set_xlabel(EVENT_DISPLAY[event]["xlabel"])
    ax.set_ylabel(ylabel)


def faceted_signsplit_figure(cache, columns, name, suptitle, caption, figsize=None):
    """Grid: rows = cell type (Narrow, Broad) x cols = (alignment x sign).

    columns: list of dict(title, decor_event, specs=[(event,cond,color,label)], sign).
    If sign is 'up'/'down', the panel is restricted to that-sign units of the row's cell
    type; if sign is None, all units of the cell type. Saves PNG + stats CSV.
    """
    masks = celltype_masks(cache)
    cts = [C.NARROW, C.BROAD]
    ncol = len(columns)
    fig = plt.figure(figsize=figsize or (4.2 * ncol, 8.6))
    gs = gridspec.GridSpec(2, ncol, hspace=0.42, wspace=0.30)
    rows = []
    for ri, ct in enumerate(cts):
        for ci, col in enumerate(columns):
            ax = fig.add_subplot(gs[ri, ci])
            sign = col.get("sign")
            rm = (signed_mask(cache, col["decor_event"], masks[ct], sign)
                  if sign else masks[ct])
            title = col["title"] if ri == 0 else None
            r = multi_cond_panel(ax, cache, col["specs"], col["decor_event"],
                                 row_mask=rm, title=title)
            for d in r:
                d["celltype"] = ct
                d["sign"] = sign or "all"
                d["column"] = col["title"]
            rows += r
            ax.set_ylabel(f"{ct}\nz (shared baseline)" if ci == 0 else "")
    fig.suptitle(suptitle, fontsize=C.FS["suptitle"], y=0.99)
    fig.text(0.5, 0.02, caption, ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.save_talk_figure(fig, name)
    sdf = pd.DataFrame(rows)
    sp = C.stats_csv_path(name)
    sdf.to_csv(sp, index=False)
    print(f"[fig] wrote {out}")
    print(f"[fig] wrote {sp}")
    return out, sp, sdf


# ── TF-responsive overlay (keeps cell-type rows; adds TF+ vs TF- within each panel) ──
def tf_masks(cache, subject=None):
    """(resp, nonresp) boolean row masks under the TF-responsive registry (3 striatum mice)."""
    return C.tf_responsive_masks(cache, subject or C.SUBJECT)


_TFMINUS_GREY = C.TF_MINUS_GREY          # shared non-responsive greys (light cond0 / dark cond1)
_TF_MIN_BAND = 8                          # draw a CI band only when TF+ n >= this (else thin line)


def multi_cond_panel_tf(ax, cache, specs, decor_event, cell_mask, resp, nonresp, title=None):
    """Like multi_cond_panel, but each condition line is drawn twice: TF-responsive (the
    condition's COLOUR, solid + bootstrap-CI band when n>=8) and non-responsive (GREY, dashed,
    no band). So colour = TF-responsive (pops), grey = non-responsive (context); condition
    = hue/grey-level. cell_mask = the row's cell-type (x sign) mask."""
    disp = EVENT_DISPLAY[decor_event]
    decorate(ax, decor_event, baseline_win=disp["baseline"])
    rows = []
    for j, (event, cond, color, label) in enumerate(specs):
        bcv = bc(cache, event)
        M = mat(cache, event, cond, "full")
        m_r, lo_r, hi_r, n_r = mean_ci(M, cell_mask & resp)
        if n_r >= _TF_MIN_BAND:
            plot_band(ax, bcv, m_r, lo_r, hi_r, color, f"{label} · TF+ (n={n_r})")
        elif n_r > 0:
            ax.plot(bcv, m_r, color=color, lw=1.4, zorder=3, label=f"{label} · TF+ (n={n_r}, low)")
        g = _TFMINUS_GREY[j % len(_TFMINUS_GREY)]
        m_n, lo_n, hi_n, n_n = mean_ci(M, cell_mask & nonresp)
        if n_n > 0:
            ax.plot(bcv, m_n, color=g, lw=1.5, ls="--", zorder=2, label=f"{label} · TF− (n={n_n})")
            ax.fill_between(bcv, lo_n, hi_n, color=g, alpha=0.15, zorder=1)
        rows.append({"event": event, "cond": cond, "n_tfresp": n_r, "n_nonresp": n_n,
                     "peak_tfresp": round(peak_stat(bcv, m_r)[0], 3),
                     "peak_nonresp": round(peak_stat(bcv, m_n)[0], 3)})
    if title:
        ax.set_title(title, fontsize=C.FS["title"])
    ax.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)
    return rows


def faceted_signsplit_tf_figure(cache, subject, columns, name, suptitle, caption, figsize=None):
    """Grid rows = cell type (Narrow, Broad) x cols = (alignment x sign), with the TF-responsive
    overlay (TF+ solid+band vs TF- dashed) inside each panel. Saves PNG + stats CSV."""
    masks = celltype_masks(cache)
    resp, nonresp = tf_masks(cache, subject)
    cts = [C.NARROW, C.BROAD]
    ncol = len(columns)
    fig = plt.figure(figsize=figsize or (4.6 * ncol, 8.6))
    gs = gridspec.GridSpec(2, ncol, hspace=0.42, wspace=0.30)
    rows = []
    for ri, ct in enumerate(cts):
        for ci, col in enumerate(columns):
            ax = fig.add_subplot(gs[ri, ci])
            sign = col.get("sign")
            rm = (signed_mask(cache, col["decor_event"], masks[ct], sign)
                  if sign else masks[ct])
            title = col["title"] if ri == 0 else None
            r = multi_cond_panel_tf(ax, cache, col["specs"], col["decor_event"],
                                    rm, resp, nonresp, title=title)
            for d in r:
                d["celltype"] = ct
                d["sign"] = sign or "all"
                d["column"] = col["title"]
            rows += r
            ax.set_ylabel(f"{ct}\nz (shared baseline)" if ci == 0 else "")
    fig.suptitle(suptitle, fontsize=C.FS["suptitle"], y=0.99)
    fig.text(0.5, 0.02, caption, ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.save_talk_figure(fig, name)
    sdf = pd.DataFrame(rows)
    sp = C.stats_csv_path(name)
    sdf.to_csv(sp, index=False)
    print(f"[fig] wrote {out}")
    print(f"[fig] wrote {sp}")
    return out, sp, sdf


def sign_panel_tf(ax, cache, event, cell_mask, resp, nonresp, title=None,
                  up_color=C.SIGN_COLORS["up"], down_color=C.SIGN_COLORS["down"]):
    """fig_B panel with the TF overlay: up/down held-out sign (hue), TF+ solid+band vs TF-
    dashed. cell_mask = the row's cell-type mask."""
    disp = EVENT_DISPLAY[event]
    bcarr = bc(cache, event)
    decorate(ax, event, baseline_win=disp["baseline"])
    ax.axvspan(disp["sign"][0], disp["sign"][1], color="#ffe08a", alpha=0.35, zorder=0)
    odd = mat(cache, event, "all", "odd")
    even = mat(cache, event, "all", "even")
    s = unit_sign(odd, bcarr, disp["sign"])
    base = np.isfinite(odd).all(1) & np.isfinite(even).all(1) & np.isfinite(s)
    if cell_mask is not None:
        base = base & cell_mask
    out = []
    for j, (sgn, col, nm) in enumerate([(1, up_color, "Up"), (-1, down_color, "Down")]):
        sgn_mask = base & ((s > 0) if sgn > 0 else (s < 0))
        m_r, lo_r, hi_r, n_r = mean_ci(even, sgn_mask & resp)
        if n_r >= _TF_MIN_BAND:
            plot_band(ax, bcarr, m_r, lo_r, hi_r, col, f"{nm} · TF+ (n={n_r})")
        elif n_r > 0:
            ax.plot(bcarr, m_r, color=col, lw=1.4, zorder=3, label=f"{nm} · TF+ (n={n_r}, low)")
        g = _TFMINUS_GREY[j % len(_TFMINUS_GREY)]
        m_n, lo_n, hi_n, n_n = mean_ci(even, sgn_mask & nonresp)
        if n_n > 0:
            ax.plot(bcarr, m_n, color=g, lw=1.5, ls="--", zorder=2, label=f"{nm} · TF− (n={n_n})")
            ax.fill_between(bcarr, lo_n, hi_n, color=g, alpha=0.15, zorder=1)
        out.append({"event": event, "group": nm.lower(), "n_tfresp": n_r, "n_nonresp": n_n})
    if title:
        ax.set_title(title, fontsize=C.FS["title"])
    ax.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)
    return out


def peak_stat(bcarr, mean, post_only=True):
    """Peak |z| and its time, in the post-event window (t>=0) if post_only."""
    sel = bcarr >= 0 if post_only else np.ones_like(bcarr, bool)
    if not sel.any() or not np.isfinite(mean).any():
        return np.nan, np.nan
    seg = mean[sel]
    i = int(np.nanargmax(np.abs(seg)))
    return float(seg[i]), float(bcarr[sel][i])
