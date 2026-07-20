"""Stage-3 adversarial hardening battery for the Fig-5 e-h preparatory-by-cell-class
result (Khilkevich & Lohse 2024, ported to transient/sustained/non-TF within striatum).

HEADLINE UNDER TEST (from prep_hit.npz, corrected cache): the more SUSTAINED a
TF-responsive cell's kernel, the EARLIER it is recruited into pre-lick preparatory
activity. Per-class population activation onset (s from hit-lick, earlier = more
negative): sustained -0.738, transient -0.613, non-TF -0.338; per-cell onset~width
is negative. This module TRIES TO REFUTE that.

Every control is reported POOLED and PER-REGION (DMS = BG_046+BG_039, VMS = BG_031).
CACHE-ONLY (data/cache/preparatory_fig5/prep_<lick>.npz); no pkl reload. Reuses the
primitives in visdetect.analysis.preparatory (active_mask, fraction_active,
bootstrap_fraction_ci, population_onset, cell_onset, width_deciles).

Controls
  1. width_onset_corr(width, onset)                         -> Pearson r helper
  2. LABEL-SHUFFLE NULL   (per-class onset ordering)        -> observed must beat null
  3. WIDTH-SHUFFLE NULL   (per-cell onset~width Pearson)    -> observed must beat null
  4. MIXEDLM              (onset ~ width, session/subject RE) vs naive OLS + sign test
  5. PRE-LICK-ONLY        (active mask zeroed for t>=0)      -> ordering must survive
  6. LICK-RESPONSIVENESS STRATIFICATION (join lick_acquisition_cells.csv)
  7. INDEPENDENT RE-DERIVATION (different onset impl + seed) -> within primary CI

Deferred to the main session (too fragile / heavy for a subagent): a pkl-level
lick-time-shuffle that re-aligns spikes to random times and rebuilds the cache. It
is the cleanest destructive null for the fraction-active ramp but needs the ProcessPool
recompute, so it is NOT run here (noted in the report).

Usage:  .venv/Scripts/python.exe scripts/tf_responsiveness/preparatory_fig5/nulls_and_hardening.py [--lick hit|fa]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress, wilcoxon, binomtest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import prep_common as C  # noqa: E402
from visdetect.analysis.preparatory import (  # noqa: E402
    active_mask, bootstrap_fraction_ci, population_onset, cell_onset)

FIGROOT = C.REPO / "FIGURES/preparatory_fig5"
OUTDIR = FIGROOT / "hardening"
REGIONS = [("pooled", None), ("DMS", "DMS"), ("VMS", "VMS")]
ORDER_CLASSES = ("sustained", "transient", "non-TF")   # hypothesised recruitment order
CLASS_RANK = {"sustained": 0, "transient": 1, "non-TF": 2}
LICK_CSV = C.REPO / "FIGURES/tf_glm_bg046/lick_acquisition/lick_acquisition_cells.csv"
N_SHUFFLE = 1000
N_BOOT_CI = 5000
N_BOOT_ONSET = 2000   # per-group headline onset bootstrap (over neurons)


# ───────────────────────── primitives / helpers ──────────────────────────────
def width_onset_corr(width, onset) -> float:
    """Pearson r between width and onset over finite pairs (NaN if <3 pairs)."""
    w = np.asarray(width, float)
    o = np.asarray(onset, float)
    m = np.isfinite(w) & np.isfinite(o)
    if m.sum() < 3 or np.ptp(w[m]) == 0 or np.ptp(o[m]) == 0:
        return np.nan
    return float(pearsonr(w[m], o[m])[0])


def width_shuffle_corr_null(onset, width, n=N_SHUFFLE, seed=0) -> np.ndarray:
    """Permutation null of the per-cell onset~width Pearson r: onset fixed, width
    shuffled across cells n times. Returns the null distribution of r."""
    o = np.asarray(onset, float)
    w = np.asarray(width, float)
    m = np.isfinite(o) & np.isfinite(w)
    o, w = o[m], w[m]
    if o.size < 3:
        return np.full(n, np.nan)
    rng = np.random.default_rng(seed)
    return np.array([width_onset_corr(w[rng.permutation(w.size)], o) for _ in range(n)])


def _first_sustained_idx(cond, win=4, need=3) -> int:
    """Earliest i where cond[i] and >=need of cond[i:i+win] are True (100ms/80ms)."""
    cond = np.asarray(cond, bool)
    n = len(cond)
    cs = np.concatenate(([0], np.cumsum(cond.astype(int))))
    idx = np.arange(n)
    end = np.minimum(idx + win, n)
    ok = cond & ((cs[end] - cs[idx]) >= need)
    w = np.where(ok)[0]
    return int(w[0]) if w.size else -1


def _analytic_pop_onset(Ag, t, base_mask) -> float:
    """Fast Wald analytic population onset for a group active-matrix Ag (units x bins):
    lower-95%-CI(fraction) > 0 AND mean(fraction above baseline) > 0.1, sustained.
    Used for the shuffle nulls (matches fig5h._analytic_decile_onsets)."""
    n = Ag.shape[0]
    if n < 3:
        return np.nan
    p = Ag.mean(0)
    base = float(np.nanmean(p[base_mask]))
    frac = p - base
    se = np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / n)
    lo = (p - 1.96 * se) - base
    i = _first_sustained_idx((lo > 0) & (frac > 0.1))
    return float(t[i]) if i >= 0 else np.nan


def _full_pop_onset(Ag, t, base_mask) -> tuple[float, float, float]:
    """Headline population onset via bootstrap_fraction_ci (over neurons). Returns
    (onset, peak_frac, peak_t)."""
    if Ag.shape[0] < 3:
        return np.nan, np.nan, np.nan
    mean, lo, _hi = bootstrap_fraction_ci(Ag, baseline_bins=base_mask, n=N_BOOT_ONSET)
    onset = population_onset(t, mean, lo)
    pk = int(np.nanargmax(mean))
    return onset, float(mean[pk]), float(t[pk])


def _indep_cell_onset(t, z, thresh=2.576, win=4, need=3) -> float:
    """INDEPENDENT re-implementation of per-cell onset (direct loop): earliest bin
    where |z| exceeds thresh for >=need of the NEXT win bins. Deliberately does NOT
    require the first bin itself to cross (slightly different from the primitive
    cell_onset), so agreement is a genuine cross-check."""
    z = np.asarray(z, float)
    a = np.abs(z) > thresh
    nb = len(a)
    for i in range(nb):
        j = min(nb, i + win)
        if a[i:j].sum() >= need:
            return float(t[i])
    return np.nan


# ───────────────────────── control 2: label-shuffle null ──────────────────────
def label_shuffle_null(A, cls_arr, t, base_mask, n=N_SHUFFLE, seed=0):
    """Shuffle cls labels (transient/intermediate/sustained/non-TF) across cells,
    preserving group sizes; recompute per-class analytic onset and the ordering
    statistics. Returns dict with observed + null arrays."""
    rng = np.random.default_rng(seed)
    Af = A.astype(np.float32)

    def _class_onsets(labels):
        return {g: _analytic_pop_onset(Af[labels == g], t, base_mask) for g in ORDER_CLASSES}

    def _stats(on):
        diff = on["non-TF"] - on["sustained"]              # >0 = sustained earlier
        ranks = np.array([CLASS_RANK[g] for g in ORDER_CLASSES], float)
        ovals = np.array([on[g] for g in ORDER_CLASSES], float)
        if np.all(np.isfinite(ovals)) and np.ptp(ovals) > 0:
            rho = spearmanr(ranks, ovals)[0]              # >0 = onset increases with rank
            mono = float(ovals[0] < ovals[1] < ovals[2])  # sustained<transient<non-TF
        else:
            rho, mono = np.nan, np.nan
        return diff, rho, mono

    obs_on = _class_onsets(cls_arr)
    obs_diff, obs_rho, obs_mono = _stats(obs_on)
    null_diff = np.empty(n)
    null_rho = np.empty(n)
    for s in range(n):
        on = _class_onsets(rng.permutation(cls_arr))
        null_diff[s], null_rho[s], _ = _stats(on)
    return {"obs_onsets": obs_on, "obs_diff": obs_diff, "obs_rho": obs_rho,
            "obs_mono": obs_mono, "null_diff": null_diff, "null_rho": null_rho}


def _one_sided_upper(obs, null):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan, np.nan
    pct = 100.0 * float(np.mean(null < obs))
    p = (1.0 + float(np.sum(null >= obs))) / (1.0 + null.size)
    return pct, p


def _two_sided_abs(obs, null):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan, np.nan
    pct = 100.0 * float(np.mean(np.abs(null) < abs(obs)))
    p = (1.0 + float(np.sum(np.abs(null) >= abs(obs)))) / (1.0 + null.size)
    return pct, p


# ───────────────────────── control 4: mixedlm ─────────────────────────────────
def mixedlm_onset_width(df):
    """onset ~ interp_fwhm with session (and subject) random intercepts vs naive OLS.
    df columns: onset, interp_fwhm, session, subject. Returns dict of results."""
    d = df.dropna(subset=["onset", "interp_fwhm"]).copy()
    out = {"n": len(d), "n_sessions": d["session"].nunique(), "n_subjects": d["subject"].nunique()}
    if len(d) < 10 or np.ptp(d["interp_fwhm"]) == 0:
        out.update(dict(ols_slope=np.nan, ols_p=np.nan, mm_slope=np.nan, mm_p=np.nan,
                        mm2_slope=np.nan, mm2_p=np.nan, method="skip"))
        return out
    lr = linregress(d["interp_fwhm"], d["onset"])
    out["ols_slope"], out["ols_p"] = float(lr.slope), float(lr.pvalue)
    try:
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("onset ~ interp_fwhm", d, groups=d["session"]).fit(method="lbfgs")
        out["mm_slope"] = float(m.params.get("interp_fwhm", np.nan))
        out["mm_p"] = float(m.pvalues.get("interp_fwhm", np.nan))
        out["method"] = "mixedlm(session RE)"
    except Exception as e:
        out.update(dict(mm_slope=np.nan, mm_p=np.nan, method=f"mixedlm failed: {type(e).__name__}"))
    # nested subject / session (only meaningful when >1 subject present)
    out["mm2_slope"], out["mm2_p"] = np.nan, np.nan
    if d["subject"].nunique() >= 2:
        try:
            import statsmodels.formula.api as smf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m2 = smf.mixedlm("onset ~ interp_fwhm", d, groups=d["subject"],
                                 vc_formula={"session": "0 + C(session)"}).fit(method="lbfgs")
            out["mm2_slope"] = float(m2.params.get("interp_fwhm", np.nan))
            out["mm2_p"] = float(m2.pvalues.get("interp_fwhm", np.nan))
        except Exception:
            pass
    return out


def per_session_sign_test(df, min_cells=5):
    """Sign of the within-session OLS onset~width slope; binomial sign test over
    sessions with >=min_cells finite-onset cells."""
    d = df.dropna(subset=["onset", "interp_fwhm"])
    signs = []
    for _sess, g in d.groupby("session"):
        if len(g) >= min_cells and np.ptp(g["interp_fwhm"]) > 0:
            signs.append(np.sign(linregress(g["interp_fwhm"], g["onset"]).slope))
    signs = np.array([s for s in signs if s != 0], float)
    n = len(signs)
    n_neg = int((signs < 0).sum())
    if n < 4:
        return dict(n_sessions=n, n_neg=n_neg, sign_p=np.nan, wilcoxon_p=np.nan)
    sign_p = binomtest(n_neg, n, 0.5, alternative="greater").pvalue
    try:
        wp = float(wilcoxon(signs).pvalue)
    except Exception:
        wp = np.nan
    return dict(n_sessions=n, n_neg=n_neg, sign_p=float(sign_p), wilcoxon_p=wp)


# ───────────────────────── control 7: independent re-derivation ───────────────
def _boot_slope_ci(x, y, n=N_BOOT_CI, seed=42):
    """Bootstrap over cells -> percentile CI of OLS slope of y on x (onset on width)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    sl = []
    for _ in range(n):
        bi = rng.integers(0, x.size, x.size)
        if np.ptp(x[bi]) > 0:
            sl.append(np.polyfit(x[bi], y[bi], 1)[0])
    sl = np.asarray(sl, float)
    return (float(np.polyfit(x, y, 1)[0]), float(np.percentile(sl, 2.5)),
            float(np.percentile(sl, 97.5)))


# ───────────────────────── main battery ──────────────────────────────────────
def main(lick="hit"):
    path = C.REPO / f"data/cache/preparatory_fig5/prep_{lick}.npz"
    if not path.exists():
        raise SystemExit(f"cache missing: {path} — run build_prep_cache.py --lick {lick}")
    D = np.load(path, allow_pickle=True)
    t = np.asarray(D["t"], float)
    z = np.asarray(D["z"], float)
    resp = np.asarray(D["resp"], bool)
    region = D["region"].astype(str)
    subject = D["meta_subject"].astype(str)
    session = D["meta_session"].astype(str)
    unit = np.asarray(D["meta_unit"], int)
    interp = np.asarray(D["interp_fwhm"], float)
    cls = D["cls"].astype(str)
    A = active_mask(z)
    base_mask = (t >= C.BASE_FRAC_WIN[0]) & (t <= C.BASE_FRAC_WIN[1])
    prelick = t < 0.0
    lick_lbl = lick.upper()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# Fig-5 e-h preparatory-by-cell-class — Stage-3 hardening report ({lick_lbl} lick)",
             "",
             f"cache: `{path.name}`  |  {len(resp)} cells "
             f"({int(resp.sum())} TF-responsive / {int((~resp).sum())} non-TF)  |  "
             f"DMS {int((region == 'DMS').sum())} / VMS {int((region == 'VMS').sum())}",
             f"shuffles per null = {N_SHUFFLE}; bootstrap CI = {N_BOOT_CI}; base-frac window "
             f"[{C.BASE_FRAC_WIN[0]}, {C.BASE_FRAC_WIN[1]}] s ({int(base_mask.sum())} bins).",
             "",
             "Earlier onset = MORE NEGATIVE (s from lick). Headline: sustained leads "
             "transient leads non-TF; per-cell onset~width negative.",
             ""]

    csv = {"label": [], "width": [], "mixedlm": [], "prelick": [], "prelick_cell": [],
           "stratify": [], "rederiv": []}

    for rname, rval in REGIONS:
        rmask = np.ones(len(resp), bool) if rval is None else (region == rval)
        lines += [f"\n## Region: {rname}", ""]

        # ---- headline per-class population onsets (bootstrap over neurons) ----
        head = {}
        for grp in ORDER_CLASSES:
            sel = (rmask & (~resp)) if grp == "non-TF" else (rmask & resp & (cls == grp))
            on, pkf, pkt = _full_pop_onset(A[sel], t, base_mask)
            head[grp] = dict(n=int(sel.sum()), onset=on, peak_frac=pkf, peak_t=pkt)
        lines.append("**Per-class population onset (bootstrap-over-neurons, headline):**")
        for grp in ORDER_CLASSES:
            h = head[grp]
            lines.append(f"  - {grp:10s} n={h['n']:5d}  onset={h['onset']:+.3f} s  "
                         f"peak_frac={h['peak_frac']:+.3f} @ t={h['peak_t']:+.3f} s")
        mono = (head["sustained"]["onset"] < head["transient"]["onset"] < head["non-TF"]["onset"])
        lines.append(f"  ordering sustained<transient<non-TF holds: **{bool(mono)}**")
        lines.append("")

        # ================= CONTROL 2: LABEL-SHUFFLE NULL =================
        ls = label_shuffle_null(A[rmask], cls[rmask], t, base_mask, n=N_SHUFFLE, seed=1)
        pct_d, p_d = _one_sided_upper(ls["obs_diff"], ls["null_diff"])
        pct_r, p_r = _one_sided_upper(ls["obs_rho"], ls["null_rho"])
        nd = ls["null_diff"][np.isfinite(ls["null_diff"])]
        verdict = "SURVIVES" if (np.isfinite(p_d) and p_d < 0.05) else "*** FAILS — BUG REPORT ***"
        lines += ["### C2 label-shuffle null (per-class onset ordering)",
                  f"  observed diff (onset_nonTF - onset_sustained) = {ls['obs_diff']:+.3f} s "
                  f"(positive = sustained earlier)",
                  f"  null diff: mean={np.mean(nd):+.3f}, 95%=[{np.percentile(nd, 2.5):+.3f}, "
                  f"{np.percentile(nd, 97.5):+.3f}] (n_valid={nd.size})" if nd.size else "  null diff: all NaN",
                  f"  observed at percentile {pct_d:.1f} of null; one-sided p={p_d:.4g}",
                  f"  class-rank Spearman obs={ls['obs_rho']:+.2f} (p={p_r:.4g}); "
                  f"monotonic ordering obs={ls['obs_mono']}",
                  f"  VERDICT: label-shuffle null {verdict}", ""]
        csv["label"].append(dict(region=rname, obs_diff=ls["obs_diff"],
                                 null_diff_mean=float(np.mean(nd)) if nd.size else np.nan,
                                 null_diff_p2_5=float(np.percentile(nd, 2.5)) if nd.size else np.nan,
                                 null_diff_p97_5=float(np.percentile(nd, 97.5)) if nd.size else np.nan,
                                 obs_percentile=pct_d, p_one_sided=p_d,
                                 obs_rho=ls["obs_rho"], rho_p=p_r, monotonic=ls["obs_mono"],
                                 onset_sustained=head["sustained"]["onset"],
                                 onset_transient=head["transient"]["onset"],
                                 onset_nonTF=head["non-TF"]["onset"]))

        # ================= per-cell onset (shared by C3/C4/C6/C7) =================
        fsel = rmask & resp & np.isfinite(interp)
        zf = z[fsel]
        wf = interp[fsel]
        onf = np.array([cell_onset(t, zf[i]) for i in range(zf.shape[0])])
        cm = np.isfinite(onf) & np.isfinite(wf)

        # ================= CONTROL 3: WIDTH-SHUFFLE NULL =================
        r_obs = width_onset_corr(wf, onf)
        null_r = width_shuffle_corr_null(onf, wf, n=N_SHUFFLE, seed=2)
        pct_w, p_w = _two_sided_abs(r_obs, null_r)
        nr = null_r[np.isfinite(null_r)]
        w_verdict = ("SURVIVES" if (np.isfinite(p_w) and p_w < 0.05 and r_obs < 0)
                     else "does NOT beat null (weak/absent per-cell gradient)")
        lines += ["### C3 width-shuffle null (per-cell onset~width Pearson)",
                  f"  n_cells with finite onset = {int(cm.sum())} of {int(fsel.sum())} responsive",
                  f"  observed Pearson r(onset, width) = {r_obs:+.3f}",
                  f"  null |r|: 95th pct = {np.percentile(np.abs(nr), 95):.3f} (n={nr.size}); "
                  f"observed |r| at percentile {pct_w:.1f}; two-sided p={p_w:.4g}" if nr.size
                  else "  null: insufficient cells",
                  f"  VERDICT: width-shuffle null {w_verdict}", ""]
        # Spearman too (project standard)
        sp = spearmanr(onf[cm], wf[cm]) if cm.sum() >= 3 else (np.nan, np.nan)
        csv["width"].append(dict(region=rname, n_cells=int(cm.sum()), pearson_r=r_obs,
                                 null_absr_p95=float(np.percentile(np.abs(nr), 95)) if nr.size else np.nan,
                                 obs_percentile=pct_w, p_perm=p_w,
                                 spearman_r=float(sp[0]), spearman_p=float(sp[1])))

        # ================= CONTROL 4: MIXEDLM PSEUDOREPLICATION =================
        df_cell = pd.DataFrame({"onset": onf[cm], "interp_fwhm": wf[cm],
                                "session": session[fsel][cm], "subject": subject[fsel][cm]})
        mm = mixedlm_onset_width(df_cell)
        st = per_session_sign_test(df_cell)
        lines += ["### C4 mixedlm pseudoreplication (onset ~ width)",
                  f"  naive OLS: slope={mm['ols_slope']:+.4f} s per (fwhm unit), p={mm['ols_p']:.4g}",
                  f"  {mm['method']}: slope={mm['mm_slope']:+.4f}, p={mm['mm_p']:.4g}",
                  f"  nested subject/session RE: slope={mm['mm2_slope']:+.4f}, p={mm['mm2_p']:.4g}"
                  if np.isfinite(mm.get('mm2_slope', np.nan)) else "  nested subject/session RE: n/a",
                  f"  n_cells={mm['n']} over {mm['n_sessions']} sessions / {mm['n_subjects']} subjects",
                  f"  per-session sign test: {st['n_neg']}/{st['n_sessions']} sessions negative slope, "
                  f"binomial p={st['sign_p']:.4g}, Wilcoxon p={st['wilcoxon_p']:.4g}", ""]
        csv["mixedlm"].append(dict(region=rname, **{k: mm[k] for k in
                              ("n", "n_sessions", "n_subjects", "ols_slope", "ols_p",
                               "mm_slope", "mm_p", "mm2_slope", "mm2_p", "method")},
                              sign_n_sessions=st["n_sessions"], sign_n_neg=st["n_neg"],
                              sign_p=st["sign_p"], sign_wilcoxon_p=st["wilcoxon_p"]))

        # ================= CONTROL 5: PRE-LICK-ONLY =================
        A_pre = A & prelick[None, :]
        z_pre = z.copy()
        z_pre[:, ~prelick] = 0.0
        pre_head = {}
        for grp in ORDER_CLASSES:
            sel = (rmask & (~resp)) if grp == "non-TF" else (rmask & resp & (cls == grp))
            on_pre, _pf, _pt = _full_pop_onset(A_pre[sel], t, base_mask)
            # peak & pre-lick ramp fraction from the FULL (un-censored) fraction trace
            mean_full, _lo, _hi = bootstrap_fraction_ci(A[sel], baseline_bins=base_mask, n=500)
            pos = np.clip(mean_full, 0, None)
            ramp_pre = float(pos[prelick].sum() / pos.sum()) if pos.sum() > 0 else np.nan
            pk = int(np.nanargmax(mean_full))
            pre_head[grp] = dict(onset_pre=on_pre, peak_t=float(t[pk]), ramp_prelick_frac=ramp_pre,
                                 onset_full=head[grp]["onset"])
        mono_pre = (pre_head["sustained"]["onset_pre"] < pre_head["transient"]["onset_pre"]
                    < pre_head["non-TF"]["onset_pre"])
        lines += ["### C5 pre-lick-only control (active mask zeroed for t>=0)",
                  "  per-class pre-lick-only population onset (ordering must survive):"]
        for grp in ORDER_CLASSES:
            h = pre_head[grp]
            lines.append(f"    {grp:10s} onset_prelick={h['onset_pre']:+.3f} s "
                         f"(full={h['onset_full']:+.3f}); peak@{h['peak_t']:+.3f} s; "
                         f"pre-lick ramp fraction={h['ramp_prelick_frac']:.2f}")
            csv["prelick"].append(dict(region=rname, cls=grp, onset_prelick=h["onset_pre"],
                                       onset_full=h["onset_full"], peak_t=h["peak_t"],
                                       ramp_prelick_frac=h["ramp_prelick_frac"]))
        lines.append(f"  ordering sustained<transient<non-TF (pre-lick only): **{bool(mono_pre)}**")
        # per-cell onset~width using pre-lick-censored z
        onf_pre = np.array([cell_onset(t, z_pre[fsel][i]) for i in range(zf.shape[0])])
        r_pre = width_onset_corr(wf, onf_pre)
        lines += [f"  per-cell onset~width (pre-lick-only z): Pearson r={r_pre:+.3f} "
                  f"(n={int((np.isfinite(onf_pre) & np.isfinite(wf)).sum())})",
                  "  CAVEAT: peak fraction sits at ~+0.0 to +0.14 s (peri-lick); anticipatory "
                  "MOVEMENT vs decision-PREPARATION cannot be separated without video "
                  "(future extension — project has video_sync).", ""]
        csv["prelick_cell"].append(dict(region=rname, pearson_r_prelick=r_pre,
                                        n=int((np.isfinite(onf_pre) & np.isfinite(wf)).sum())))

        # ================= CONTROL 6: LICK-RESPONSIVENESS STRATIFICATION =========
        if LICK_CSV.exists():
            lk = pd.read_csv(LICK_CSV)[["subject", "session", "unit", "lick_sig"]]
            key = pd.DataFrame({"subject": subject[fsel], "session": session[fsel],
                                "unit": unit[fsel], "onset": onf, "width": wf})
            key["unit"] = key["unit"].astype(int)
            lk["unit"] = lk["unit"].astype(int)
            j = key.merge(lk, on=["subject", "session", "unit"], how="left")
            matched = j["lick_sig"].notna().sum()
            lines += ["### C6 lick-responsiveness stratification (join lick_acquisition_cells.csv)",
                      f"  matched {matched}/{len(j)} responsive cells to lick_sig"]
            for grp_name, gmask in [("lick-responsive", j["lick_sig"] == True),          # noqa: E712
                                    ("non-lick-responsive", j["lick_sig"] == False)]:      # noqa: E712
                sub = j[gmask]
                r_s = width_onset_corr(sub["width"].values, sub["onset"].values)
                nn = int((np.isfinite(sub["onset"]) & np.isfinite(sub["width"])).sum())
                sp2 = spearmanr(sub["onset"], sub["width"], nan_policy="omit") if nn >= 3 else (np.nan, np.nan)
                lines.append(f"    within {grp_name:20s}: n={nn:4d}  Pearson r={r_s:+.3f}  "
                             f"Spearman rho={float(sp2[0]):+.3f} (p={float(sp2[1]):.4g})")
                csv["stratify"].append(dict(region=rname, lick_group=grp_name, n=nn,
                                            pearson_r=r_s, spearman_r=float(sp2[0]),
                                            spearman_p=float(sp2[1])))
            lines.append("")
        else:
            lines += ["### C6 lick-responsiveness stratification",
                      f"  SKIPPED — {LICK_CSV} absent.", ""]

        # ================= CONTROL 7: INDEPENDENT RE-DERIVATION =================
        prim_slope, prim_lo, prim_hi = _boot_slope_ci(wf[cm], onf[cm], n=N_BOOT_CI, seed=42)
        onf_ind = np.array([_indep_cell_onset(t, zf[i]) for i in range(zf.shape[0])])
        im = np.isfinite(onf_ind) & np.isfinite(wf)
        ind_slope, ind_lo, ind_hi = _boot_slope_ci(wf[im], onf_ind[im], n=N_BOOT_CI, seed=123)
        within = bool(np.isfinite(ind_slope) and np.isfinite(prim_lo) and prim_lo <= ind_slope <= prim_hi)
        # agreement of the two onset vectors where both finite
        both = np.isfinite(onf) & np.isfinite(onf_ind)
        onset_mae = float(np.mean(np.abs(onf[both] - onf_ind[both]))) if both.any() else np.nan
        lines += ["### C7 independent re-derivation (different onset impl + seed)",
                  f"  primary  slope(onset~width) = {prim_slope:+.4f}  CI[{prim_lo:+.4f}, {prim_hi:+.4f}] "
                  f"(cell_onset primitive, seed 42, n={int(cm.sum())})",
                  f"  independent slope           = {ind_slope:+.4f}  CI[{ind_lo:+.4f}, {ind_hi:+.4f}] "
                  f"(direct-loop 3-of-4, seed 123, n={int(im.sum())})",
                  f"  independent slope within primary CI: **{within}**; "
                  f"onset MAE between implementations = {onset_mae:.4f} s", ""]
        csv["rederiv"].append(dict(region=rname, n=int(cm.sum()), primary_slope=prim_slope,
                                   primary_ci_lo=prim_lo, primary_ci_hi=prim_hi,
                                   indep_slope=ind_slope, indep_ci_lo=ind_lo, indep_ci_hi=ind_hi,
                                   within_primary_ci=within, onset_mae=onset_mae))

    # ---- deferred + bottom line ----
    lines += ["\n## Deferred / not run here",
              "  - pkl-level LICK-TIME-SHUFFLE (re-align spikes to random times, rebuild cache): "
              "deferred to main session (needs a heavy ProcessPool pkl recompute; fragile in a subagent).",
              ""]

    (OUTDIR / "hardening_report.md").write_text("\n".join(lines), encoding="utf-8")
    for name, rows in csv.items():
        if rows:
            pd.DataFrame(rows).to_csv(OUTDIR / f"hardening_{name}_{lick}.csv", index=False)
    print(f"wrote {OUTDIR}/hardening_report.md (+ {sum(1 for v in csv.values() if v)} CSVs)", flush=True)
    # echo the report to stdout (ascii-safe)
    print("\n".join(lines).encode("ascii", "replace").decode(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lick", choices=["hit", "fa"], default="hit")
    a = ap.parse_args()
    main(lick=a.lick)
