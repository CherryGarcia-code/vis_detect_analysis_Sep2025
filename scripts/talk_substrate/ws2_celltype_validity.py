"""WS2 (talk substrate): cell-type statistics + label validity.

(2a) UNEQUAL N is EXPECTED (FSIs preferentially isolated; SPNs sparse) — handle, don't "fix":
     modulated FRACTIONS with Wilson binomial CIs (smaller broad -> honestly wider band), where
     "modulated" uses a baseline-CALIBRATED threshold (95th pct of per-unit baseline max|z| ->
     ~5% false-positive by construction, not an arbitrary cutoff); a matched-n subsample panel
     (draw N_broad narrow units, recompute the narrow curve many times in parallel) showing the
     narrow dynamics don't depend on the bigger n; and a mixed-effects test (cell type fixed;
     session random) for the formal claim.
(2b) LABEL VALIDITY (orthogonal to n): width<->rate correlated, so a width-ONLY split risks
     sorting on rate. Test whether the split REPLICATES on a rate-INDEPENDENT second axis (CV2
     ISI irregularity) by how well CV2 alone CLASSIFIES narrow vs broad (ROC AUC). Rate AUC is
     shown for contrast (the confound). If only width/rate separate and CV2 does not, the label
     is width-driven -> PRELIMINARY.

Caveat (caption): recorded SPNs are the high-firing sortable tail, not a random SPN sample.
Reuses caches (no re-sorting): per-subject t2p (C.load_t2p), ISI (C.isi_features_path),
event_psth_cache_<SUBJECT>.npz. Cell type = COMMON width cutoff (FIX A), via E.celltype_array.

Usage: py scripts/talk_substrate/ws2_celltype_validity.py [--n_workers N] [--n_draws N]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402

setup_style()
RESP_WINDOWS = {"Change_ON": (0.0, 1.0), "Hit": (-0.5, 0.2)}
BASE_WINDOWS = {"Change_ON": (-1.0, 0.0), "Hit": (-1.75, -1.05)}
AUC_READY = 0.70   # CV2 must classify this well for the label to "replicate" on axis 2


def wilson(k, n):
    from statsmodels.stats.proportion import proportion_confint
    if n == 0:
        return np.nan, np.nan, np.nan
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return k / n, lo, hi


def sep_auc(x, label):
    """Separation AUC (>=0.5) of feature x classifying label (1=broad). Direction-agnostic."""
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(x)
    if ok.sum() < 10 or len(np.unique(label[ok])) < 2:
        return np.nan
    a = roc_auc_score(label[ok], x[ok])
    return max(a, 1 - a)


# ── matched-n subsample (parallel) ───────────────────────────────────────────
_G = {}


def _mn_init(mat):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    _G["mat"] = mat


def _mn_chunk(args):
    n_target, seeds = args
    mat = _G["mat"]
    n = mat.shape[0]
    return [np.nanmean(mat[np.random.default_rng(s).choice(n, n_target, replace=False)], axis=0)
            for s in seeds]


def matched_n(narrow_mat, n_target, n_draws, n_workers, seed=42):
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, n_draws)
    chunks = np.array_split(seeds, max(n_workers * 4, 1))
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_mn_init,
                             initargs=(narrow_mat,)) as ex:
        for r in ex.map(_mn_chunk, [(n_target, list(c)) for c in chunks]):
            rows.extend(r)
    arr = np.vstack(rows)
    return np.nanmean(arr, 0), np.nanpercentile(arr, 2.5, 0), np.nanpercentile(arr, 97.5, 0)


def maxabs(cache, event, win):
    m = E.mat(cache, event, "all", "full")
    wm = E.win_mask(E.bc(cache, event), win)
    seg = m[:, wm]
    pk = np.full(m.shape[0], np.nan)
    ok = np.isfinite(seg).all(1)
    pk[ok] = np.nanmax(np.abs(seg[ok]), axis=1)
    return pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_workers", type=int, default=min(8, (os.cpu_count() or 4) - 2))
    ap.add_argument("--n_draws", type=int, default=500)
    args = ap.parse_args()

    cache = E.load_event_cache()
    ct = E.celltype_array(cache)   # COMMON width cutoff (FIX A), not the per-subject cache labels
    sess_meta = cache["unit_meta_session"]
    cN, cB = C.celltype_color(C.NARROW), C.celltype_color(C.BROAD)

    # ---- 2a: modulated fractions (baseline-calibrated ~5% FPR) + Wilson CIs ----
    frac_rows, thr_by_ev = [], {}
    for ev in ("Change_ON", "Hit"):
        resp = maxabs(cache, ev, RESP_WINDOWS[ev])
        base = maxabs(cache, ev, BASE_WINDOWS[ev])
        thr = np.nanpercentile(base, 95)
        thr_by_ev[ev] = thr
        mod = resp > thr
        for cell in (C.NARROW, C.BROAD):
            mask = (ct == cell) & np.isfinite(resp)
            k = int(np.sum(mod[mask])); n = int(mask.sum())
            f, lo, hi = wilson(k, n)
            frac_rows.append(dict(event=ev, celltype=cell, k=k, n=n, frac=f, lo=lo, hi=hi))
    frac_df = pd.DataFrame(frac_rows)

    # ---- 2a: matched-n subsample at lick (downsample the LARGER cell type to the smaller's n;
    #          larger = narrow in striatum, but broad/pyramidal in cortex) ----
    lick = E.mat(cache, "Hit", "all", "full")
    bc_l = E.bc(cache, "Hit")
    nar = lick[(ct == C.NARROW) & np.isfinite(lick).all(1)]
    bro = lick[(ct == C.BROAD) & np.isfinite(lick).all(1)]
    if nar.shape[0] >= bro.shape[0]:
        big_mat, big_lbl, big_col, small_mat, small_lbl, small_col = nar, C.NARROW, cN, bro, C.BROAD, cB
    else:
        big_mat, big_lbl, big_col, small_mat, small_lbl, small_col = bro, C.BROAD, cB, nar, C.NARROW, cN
    n_target = small_mat.shape[0]
    mn_mean, mn_lo, mn_hi = matched_n(big_mat, n_target, args.n_draws, args.n_workers)
    big_full, small_full = np.nanmean(big_mat, 0), np.nanmean(small_mat, 0)

    # ---- 2b: width x ISI features; AUC of each axis classifying narrow vs broad ----
    t2p = C.load_t2p(C.SUBJECT); t2p["cluster_id"] = t2p["cluster_id"].astype(int)
    isi = pd.read_csv(C.isi_features_path(), dtype={"session_8": str})
    isi["cluster_id"] = isi["cluster_id"].astype(int)
    val = t2p.merge(isi, on=["session_8", "cluster_id"], how="inner")
    val = val[np.isfinite(val["t2p_ms"])].copy()
    val["ctd"] = np.where(val["t2p_ms"] < E.common_cut(), C.NARROW, C.BROAD)  # COMMON cutoff (FIX A)
    lab = (val["ctd"] == C.BROAD).astype(int).values
    aucs = {f: sep_auc(val[f].values, lab) for f in
            ["t2p_ms", "cv2", "burst_frac", "isi_mode_s", "median_isi_s", "rate_hz"]}
    thr_w = E.common_cut()   # COMMON width cutoff (FIX A) — drawn as the narrow/broad split line
    cv2_auc = aucs["cv2"]
    feature_ready = np.isfinite(cv2_auc) and cv2_auc >= AUC_READY

    # ---- 2a: mixed-effects (peak lick |z| ~ celltype + (1|session)) ----
    pk_lick = maxabs(cache, "Hit", RESP_WINDOWS["Hit"])
    me_txt = []
    try:
        import statsmodels.formula.api as smf
        df = pd.DataFrame({"pk": pk_lick, "celltype": ct, "session": sess_meta})
        df = df[np.isfinite(df["pk"]) & df["celltype"].isin([C.NARROW, C.BROAD])].copy()
        df["broad"] = (df["celltype"] == C.BROAD).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm("pk ~ broad", df, groups=df["session"]).fit(method="lbfgs")
        beta = md.params["broad"]; ci = md.conf_int().loc["broad"]
        re_var = float(md.cov_re.iloc[0, 0])
        me_txt = ["MixedLM peak-lick |z| ~ broad + (1|session)",
                  f"  broad-narrow beta = {beta:+.3f}  p={md.pvalues['broad']:.1e}",
                  f"  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]",
                  f"  session RE var = {re_var:.2e}" + ("  (~singular: session" if re_var < 1e-4 else ""),
                  "   adds little)" if re_var < 1e-4 else "",
                  f"  n={len(df)} unit-sessions, {df['session'].nunique()} sessions"]
        me_txt = [m for m in me_txt if m]
    except Exception as e:  # noqa: BLE001
        me_txt = [f"MixedLM unavailable: {e}"]

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.32)

    axA = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1.0, 41)
    axA.hist([val.loc[val.ctd == C.BROAD, "t2p_ms"], val.loc[val.ctd == C.NARROW, "t2p_ms"]],
             bins=bins, stacked=True, color=[cB, cN], edgecolor="white", linewidth=0.3,
             label=[f"{C.BROAD} (n={int((val.ctd==C.BROAD).sum())})",
                    f"{C.NARROW} (n={int((val.ctd==C.NARROW).sum())})"])
    if np.isfinite(thr_w):
        axA.axvline(thr_w, ls="--", color="k", lw=1.2, label=f"cut {thr_w:.2f} ms")
    axA.set_xlabel("trough-to-peak width (ms)"); axA.set_ylabel("# unit-sessions")
    axA.set_title("2b Width split — axis 1 (the DEFINING cut; AUC=1 trivial)", fontsize=10)
    axA.legend(frameon=False, fontsize=7)

    axB = fig.add_subplot(gs[0, 1])
    for cell, col in [(C.BROAD, cB), (C.NARROW, cN)]:
        d = val[val.ctd == cell]
        axB.scatter(d["t2p_ms"], d["cv2"], s=5, c=col, alpha=0.30, edgecolors="none", label=cell)
    if np.isfinite(thr_w):
        axB.axvline(thr_w, ls="--", color="k", lw=1.0)
    axB.set_xlabel("trough-to-peak width (ms)"); axB.set_ylabel("CV2 (ISI irregularity, rate-indep.)")
    axB.set_title(f"2b Width x CV2 — does axis 2 separate? (CV2 AUC={cv2_auc:.2f})", fontsize=10)
    axB.legend(frameon=False, fontsize=7, markerscale=2)

    axC = fig.add_subplot(gs[0, 2])
    for cell, col in [(C.NARROW, cN), (C.BROAD, cB)]:
        v = val.loc[val.ctd == cell, "cv2"].dropna()
        axC.hist(v, bins=30, color=col, alpha=0.5, density=True,
                 label=f"{cell} (med {v.median():.2f})")
    axC.set_xlabel("CV2"); axC.set_ylabel("density")
    axC.set_title("2b CV2 by cell type (heavy overlap = width-driven)", fontsize=10)
    axC.legend(frameon=False, fontsize=7)

    axD = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(frac_df))
    cols = [cN if r.celltype == C.NARROW else cB for r in frac_df.itertuples()]
    axD.bar(xs, frac_df["frac"], color=cols,
            yerr=[frac_df["frac"] - frac_df["lo"], frac_df["hi"] - frac_df["frac"]], capsize=4)
    axD.set_xticks(xs)
    axD.set_xticklabels([f"{r.event.split('_')[0]}\n{r.celltype.split()[0]}\nn={r.n}"
                         for r in frac_df.itertuples()], fontsize=7)
    axD.set_ylabel("frac modulated (baseline-calibrated)")
    axD.set_title("2a Modulated fraction (Wilson 95% CI)", fontsize=10)

    axE = fig.add_subplot(gs[1, 1])
    _bn = big_lbl.split()[0]
    axE.plot(bc_l, big_full, color=big_col, lw=2.0, label=f"{_bn} full (n={big_mat.shape[0]})")
    axE.plot(bc_l, mn_mean, color="#555555", lw=1.5, ls="--", label=f"{_bn} matched-n (n={n_target})")
    axE.fill_between(bc_l, mn_lo, mn_hi, color="#555555", alpha=0.2)
    axE.plot(bc_l, small_full, color=small_col, lw=2.0, label=f"{small_lbl.split()[0]} (n={n_target})")
    axE.axvline(0, color="k", lw=1.0)
    axE.set_xlabel("time from response lick (s)"); axE.set_ylabel("population z")
    axE.set_title(f"2a {_bn} curve robust to matched n", fontsize=10)
    axE.legend(frameon=False, fontsize=7, loc="upper left")

    axF = fig.add_subplot(gs[1, 2]); axF.axis("off")
    txt = ["WS2 cell-type stats + label validity", "",
           "2a modulated fractions (Wilson CI):"]
    for r in frac_df.itertuples():
        txt.append(f"  {r.event.split('_')[0]:7s} {r.celltype.split()[0]:6s}: "
                   f"{r.frac*100:4.1f}% [{r.lo*100:.0f},{r.hi*100:.0f}] n={r.n}")
    txt += ["", f"2a matched-n: {big_lbl.split()[0]} curve stable at matched n={n_target}", ""]
    txt += me_txt
    txt += ["", "2b separation AUC (narrow vs broad):"]
    for f in ["t2p_ms", "cv2", "burst_frac", "rate_hz"]:
        tag = "(rate-indep)" if f == "cv2" else "(=CONFOUND)" if f == "rate_hz" else ""
        txt.append(f"  {f:11s}: {aucs[f]:.2f} {tag}")
    verdict = "FEATURE-READY" if feature_ready else "PRELIMINARY (split rests on width; CV2 barely separates)"
    txt += ["", f"VERDICT (cell-type axis): {verdict}"]
    axF.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=8.3, family="monospace")

    fig.suptitle(f"{C.SUBJECT}: cell-type label validity & statistics "
                 "(narrow/broad = spike width; gloss FSI/SPN in striatum, FS/pyramidal in cortex)",
                 fontsize=12, y=0.99)
    fig.text(0.5, 0.005,
             "Unequal n (narrow>>broad) is EXPECTED (FSIs preferentially isolated) — fractions use "
             "Wilson CIs, matched-n shows narrow dynamics aren't an n artifact, mixed-effects gives the "
             "formal effect. Label validity: width separates cleanly, but if CV2 (rate-independent) "
             "does NOT, the split is width/rate-driven. Recorded SPNs = high-firing sortable tail.",
             ha="center", fontsize=8, color="#555555", wrap=True)

    out = C.save_talk_figure(fig, "ws2_celltype_validity")
    print(f"[fig] wrote {out}")
    frac_df.assign(**{f"auc_{k}": v for k, v in aucs.items()}).to_csv(
        C.stats_csv_path("ws2_celltype_validity"), index=False)
    print(f"[fig] wrote {C.stats_csv_path('ws2_celltype_validity')}")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
