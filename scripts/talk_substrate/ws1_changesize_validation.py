"""WS1 (talk substrate): is the change-aligned big>small scaling a graded EVIDENCE signal,
or just earlier reaction times on big trials pulling movement into the window?

Four converging controls (descriptive; NOT N1 — RT is treated as a CONFOUND to remove):
  1.1 Divergence onset vs earliest-lick boundary (cluster-based permutation, within-session
      label shuffles, parallel). If big/small neural curves diverge BEFORE the earliest
      licks (earliest-lick - ~150 ms motor-prep), movement can't explain the early scaling.
  1.2 Censor-before-lick: each trial contributes only bins before its own lick; if scaling
      survives it isn't lick timing. Per-bin trial count annotated.
  1.3 RT-matched: subsample big/small to a common RT distribution (parallel); show the
      big-small effect distribution at matched RT.
  1.4 Misses: big vs small on miss trials (no lick / zero movement) — converging, n-limited.

Sampling unit = TRIAL (population = mean z across that session's units, per trial). The
population trace is computed per cell type too (narrow/broad) for a descriptive panel.

Reuses: build_population_tensor, compute_zscore_normalized, get_trial_dataframe, canonical
SMALL_/BIG_CHANGE_SIZES + Change_ON baseline. Output: FIGURES/talk_substrate/BG_046/.

Usage: py scripts/talk_substrate/ws1_changesize_validation.py [--force] [--n_workers N]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gc
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

from visdetect.suite.loader import load_session                    # noqa: E402
from visdetect.suite.plotting import setup_style                   # noqa: E402
from visdetect.analysis.behavior import get_trial_dataframe        # noqa: E402
from visdetect.analysis.constants import (                         # noqa: E402
    DEFAULT_BIN_SIZE, DEFAULT_SIGMA_MS, EVENT_RESPONSIVENESS_WINDOWS,
)
from visdetect.analysis.utils import (                             # noqa: E402
    get_good_cluster_ids, build_population_tensor,
    compute_zscore_normalized, smooth_psth,
)

setup_style()

BIN = DEFAULT_BIN_SIZE
WINDOW = (-1.0, 2.5)                                  # wide enough to reach licks for censoring
BASELINE = EVENT_RESPONSIVENESS_WINDOWS["Change_ON"][0]   # canonical (-0.4, -0.05)
MOTOR_PREP_S = 0.15                                   # lick leads ~150 ms by motor prep
EARLY_WIN = (0.0, 0.5)                                # matched-effect measurement window
CACHE = C.CACHE_DIR / f"ws1_changesize_trials_{C.SUBJECT}.npz"
SMALL_C, BIG_C = "#fdae6b", "#d94801"

# ── Per-trial builder ────────────────────────────────────────────────────────
def build_trials():
    ct_lookup, sessions_8 = C.celltype_and_sessions(C.SUBJECT)
    pop_all, pop_n, pop_b = [], [], []
    grp, outc, rt, sess = [], [], [], []
    bc_ref = None
    for si, s8 in enumerate(sessions_8, 1):
        try:
            sess_o = load_session(s8)
        except Exception as e:  # noqa: BLE001
            print(f"  [{si}/{len(sessions_8)}] {s8}: load failed ({e}); skip"); continue
        cids = get_good_cluster_ids(sess_o)
        tdf = get_trial_dataframe(sess_o)
        if len(cids) == 0 or tdf.empty or "trial_idx" not in tdf.columns:
            print(f"  [{si}/{len(sessions_8)}] {s8}: no trials/units; skip")
            del sess_o; gc.collect(); continue
        tdf = tdf.set_index("trial_idx")
        ct = np.array([ct_lookup.get((s8, int(c)), C.UNKNOWN) for c in cids])
        try:
            tensor, bc, valid = build_population_tensor(
                sess_o, list(cids), event_name="Change_ON", window=WINDOW, bin_size=BIN)
        except ValueError:
            print(f"  [{si}/{len(sessions_8)}] {s8}: no Change_ON trials; skip")
            del sess_o; gc.collect(); continue
        bc_ref = bc
        z = compute_zscore_normalized(tensor, bc, BASELINE)        # (T, bins, U)
        z[:, :, np.nanmax(np.abs(z), axis=(0, 1)) > 50.0] = np.nan  # drop degenerate-baseline blowups
        n_mask = ct == C.NARROW
        b_mask = ct == C.BROAD
        sub = tdf.loc[valid]
        n_go = 0
        for ti, p in enumerate(valid):
            row = sub.loc[p]
            if not bool(row["is_go"]):
                continue
            o = str(row["outcome"]).lower()
            if o not in ("hit", "miss"):
                continue
            cs = float(row["change_size"])
            g = "big" if cs >= 2.0 else "small"
            tr = z[ti]                                              # (bins, U)
            pop_all.append(smooth_psth(np.nanmean(tr, axis=1), BIN, DEFAULT_SIGMA_MS))
            pop_n.append(smooth_psth(np.nanmean(tr[:, n_mask], axis=1), BIN, DEFAULT_SIGMA_MS)
                         if n_mask.any() else np.full(len(bc), np.nan))
            pop_b.append(smooth_psth(np.nanmean(tr[:, b_mask], axis=1), BIN, DEFAULT_SIGMA_MS)
                         if b_mask.any() else np.full(len(bc), np.nan))
            grp.append(g); outc.append(o); sess.append(s8)
            rt.append(float(row["rt"]) if o == "hit" else np.nan)  # rt = RT from change (hits)
            n_go += 1
        print(f"  [{si}/{len(sessions_8)}] {s8}: {n_go} go-trials, {len(cids)}u")
        del sess_o; gc.collect()

    out = dict(bc=bc_ref, pop_all=np.array(pop_all), pop_narrow=np.array(pop_n),
               pop_broad=np.array(pop_b), group=np.array(grp), outcome=np.array(outc),
               rt=np.array(rt, float), session=np.array(sess))
    np.savez_compressed(CACHE, **out)
    print(f"[WS1] wrote {CACHE}  ({len(grp)} go-trials)")
    return out


def load_trials():
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ── Cluster-based permutation (within-session label shuffle), parallel ────────
T_THRESH = 2.0  # ~p<0.05 cluster-forming threshold


def _per_bin_t(mat, big_bool):
    big = mat[big_bool]; small = mat[~big_bool]
    mb, ms = np.nanmean(big, 0), np.nanmean(small, 0)
    vb, vs = np.nanvar(big, 0, ddof=1), np.nanvar(small, 0, ddof=1)
    nb, ns = big.shape[0], small.shape[0]
    se = np.sqrt(vb / nb + vs / ns)
    se[se == 0] = np.inf
    return (mb - ms) / se


def _clusters(t):
    """Contiguous supra-threshold runs -> list of (start, end, signed mass)."""
    supra = np.abs(t) > T_THRESH
    out = []
    i = 0
    n = len(t)
    while i < n:
        if supra[i]:
            j = i
            while j < n and supra[j]:
                j += 1
            mass = float(np.sum(t[i:j]))
            out.append((i, j - 1, mass))
            i = j
        else:
            i += 1
    return out


_GG = {}


def _perm_init(mat, big_bool, sess_blocks):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    _GG["mat"] = mat
    _GG["big"] = big_bool
    _GG["blocks"] = sess_blocks


def _perm_chunk(seeds):
    mat, big, blocks = _GG["mat"], _GG["big"], _GG["blocks"]
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        g = big.copy()
        for idx in blocks:
            g[idx] = rng.permutation(g[idx])
        t = _per_bin_t(mat, g)
        cls = _clusters(t)
        out.append(max((m for _s, _e, m in cls), default=0.0))  # max positive mass (big>small)
    return out


def cluster_perm(mat, big_bool, session, bc, n_perm=1000, n_workers=8, seed=42):
    obs_t = _per_bin_t(mat, big_bool)
    obs = _clusters(obs_t)
    sess_blocks = [np.where(session == s)[0] for s in np.unique(session)]
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, n_perm)
    chunks = np.array_split(seeds, max(n_workers * 4, 1))
    null = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_perm_init,
                             initargs=(mat, big_bool, sess_blocks)) as ex:
        for r in ex.map(_perm_chunk, [list(c) for c in chunks]):
            null.extend(r)
    null = np.array(null)
    # significant positive clusters in the post-change region (t>=0), earliest onset
    div_onset = np.nan
    sig = []
    for s, e, m in obs:
        if m <= 0:
            continue
        p = (1 + np.sum(null >= m)) / (1 + len(null))
        t0 = bc[s]
        sig.append((t0, bc[e], m, p))
        if p < 0.05 and t0 >= 0 and (np.isnan(div_onset) or t0 < div_onset):
            div_onset = t0
    return div_onset, sig, obs_t, null


# ── RT-matched subsample (parallel) ──────────────────────────────────────────
def _match_init(mat, big_bool, rt, win_mask):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    _GG["mat"] = mat; _GG["big"] = big_bool; _GG["rt"] = rt; _GG["wm"] = win_mask


def _match_chunk(seeds):
    mat, big, rt, wm = _GG["mat"], _GG["big"], _GG["rt"], _GG["wm"]
    edges = np.quantile(rt[np.isfinite(rt)], np.linspace(0, 1, 11))
    edges[-1] += 1e-6
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        keep_big, keep_small = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            inb = np.where(big & (rt >= lo) & (rt < hi))[0]
            ins = np.where((~big) & (rt >= lo) & (rt < hi))[0]
            k = min(len(inb), len(ins))
            if k == 0:
                continue
            keep_big.extend(rng.choice(inb, k, replace=False))
            keep_small.extend(rng.choice(ins, k, replace=False))
        if not keep_big:
            out.append(np.nan); continue
        eb = np.nanmean(mat[np.array(keep_big)][:, wm])
        es = np.nanmean(mat[np.array(keep_small)][:, wm])
        out.append(eb - es)
    return out


def rt_match(mat, big_bool, rt, bc, n_match=500, n_workers=8, seed=42):
    wm = (bc >= EARLY_WIN[0]) & (bc <= EARLY_WIN[1])
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, n_match)
    chunks = np.array_split(seeds, max(n_workers * 4, 1))
    eff = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_match_init,
                             initargs=(mat, big_bool, rt, wm)) as ex:
        for r in ex.map(_match_chunk, [list(c) for c in chunks]):
            eff.extend(r)
    return np.array(eff, float)


def _mean_ci_trials(mat, n_boot=1000, seed=42):
    """Mean + bootstrap 95% CI over TRIALS (rows). NaN-aware."""
    n = mat.shape[0]
    if n == 0:
        w = mat.shape[1]; return np.full(w, np.nan), np.full(w, np.nan), np.full(w, np.nan)
    mean = np.nanmean(mat, 0)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, mat.shape[1]))
    for b in range(n_boot):
        boots[b] = np.nanmean(mat[rng.integers(0, n, n)], 0)
    return mean, np.nanpercentile(boots, 2.5, 0), np.nanpercentile(boots, 97.5, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n_workers", type=int, default=min(8, (os.cpu_count() or 4) - 2))
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--n_match", type=int, default=500)
    args = ap.parse_args()

    D = build_trials() if (args.force or not CACHE.exists()) else load_trials()
    bc = D["bc"]; group = D["group"]; outcome = D["outcome"]; rt = D["rt"]; session = D["session"]
    pop = D["pop_all"]
    big = group == "big"
    hit = outcome == "hit"
    miss = outcome == "miss"

    # ---- 1.1 divergence vs earliest-lick boundary (HIT trials) ----
    hb = hit & big; hs = hit & (~big)
    mat_hit = pop[hit]; big_hit = big[hit]; sess_hit = session[hit]
    div_onset, sig, obs_t, null = cluster_perm(
        mat_hit, big_hit, sess_hit, bc, n_perm=args.n_perm, n_workers=args.n_workers)
    rt_big = rt[hb]; rt_big = rt_big[np.isfinite(rt_big)]
    rt_small = rt[hs]; rt_small = rt_small[np.isfinite(rt_small)]
    earliest_lick = np.percentile(rt_big, 5) if rt_big.size else np.nan
    boundary = earliest_lick - MOTOR_PREP_S
    margin = boundary - div_onset

    # ---- 1.2 censor before lick (HIT trials) ----
    cens = pop[hit].copy()
    rt_h = rt[hit]
    for i in range(cens.shape[0]):
        if np.isfinite(rt_h[i]):
            cens[i, bc >= rt_h[i]] = np.nan
    nbig = ~np.isnan(cens[big_hit]).all(0)  # bins with any big trial
    cnt_big = np.sum(~np.isnan(cens[big_hit]), 0)
    cnt_small = np.sum(~np.isnan(cens[~big_hit]), 0)
    cb_m, cb_lo, cb_hi = _mean_ci_trials(cens[big_hit])
    cs_m, cs_lo, cs_hi = _mean_ci_trials(cens[~big_hit])

    # ---- 1.3 RT-matched effect ----
    eff = rt_match(mat_hit, big_hit, rt[hit], bc, n_match=args.n_match, n_workers=args.n_workers)
    eff = eff[np.isfinite(eff)]
    eff_lo, eff_hi = np.percentile(eff, [2.5, 97.5]) if eff.size else (np.nan, np.nan)
    eff_med = np.median(eff) if eff.size else np.nan
    frac_pos = float(np.mean(eff > 0)) if eff.size else np.nan

    # ---- 1.4 misses ----
    mb = miss & big; ms = miss & (~big)
    mm_b, mb_lo, mb_hi = _mean_ci_trials(pop[mb])
    mm_s, ms_lo, ms_hi = _mean_ci_trials(pop[ms])

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)

    # A: full big/small + divergence + boundary + RT rug
    axA = fig.add_subplot(gs[0, 0])
    for m, mask, col, lab in [("big", hb, BIG_C, "Big (2-4x)"), ("small", hs, SMALL_C, "Small (1.25-1.5x)")]:
        mu, lo, hi = _mean_ci_trials(pop[mask])
        axA.plot(bc, mu, color=col, lw=1.9, label=f"{lab} (n={int(mask.sum())} tr)")
        axA.fill_between(bc, lo, hi, color=col, alpha=0.2)
    axA.axvline(0, color="k", lw=1.0)
    if np.isfinite(div_onset):
        axA.axvline(div_onset, color="purple", ls="--", lw=1.3, label=f"divergence {div_onset:.2f}s")
    if np.isfinite(boundary):
        axA.axvline(boundary, color="green", ls=":", lw=1.3, label=f"earliest-lick-150ms {boundary:.2f}s")
    axA.set_title("1.1 Change-aligned big vs small (hits)\n+ divergence vs lick boundary", fontsize=9.5)
    axA.set_xlabel("time from change (s)"); axA.set_ylabel("population z (per-trial)")
    axA.legend(frameon=False, fontsize=7, loc="upper left")

    # B: RT distributions
    axB = fig.add_subplot(gs[0, 1])
    axB.hist(rt_small, bins=30, color=SMALL_C, alpha=0.6, density=True, label=f"small RT (n={rt_small.size})")
    axB.hist(rt_big, bins=30, color=BIG_C, alpha=0.6, density=True, label=f"big RT (n={rt_big.size})")
    if np.isfinite(earliest_lick):
        axB.axvline(earliest_lick, color="green", ls=":", lw=1.3, label=f"big 5th pct {earliest_lick:.2f}s")
    axB.set_title("1.1 RT (from change) distributions", fontsize=9.5)
    axB.set_xlabel("reaction time from change (s)"); axB.set_ylabel("density")
    axB.legend(frameon=False, fontsize=7)

    # C: censored curves + trial count
    axC = fig.add_subplot(gs[0, 2])
    axC.plot(bc, cb_m, color=BIG_C, lw=1.9, label="Big (pre-lick only)")
    axC.fill_between(bc, cb_lo, cb_hi, color=BIG_C, alpha=0.2)
    axC.plot(bc, cs_m, color=SMALL_C, lw=1.9, label="Small (pre-lick only)")
    axC.fill_between(bc, cs_lo, cs_hi, color=SMALL_C, alpha=0.2)
    axC.axvline(0, color="k", lw=1.0)
    axc2 = axC.twinx()
    axc2.plot(bc, cnt_big, color=BIG_C, lw=0.8, ls="--", alpha=0.6)
    axc2.plot(bc, cnt_small, color=SMALL_C, lw=0.8, ls="--", alpha=0.6)
    axc2.set_ylabel("# trials remaining (dashed)", fontsize=8)
    axC.set_title("1.2 Censored before each trial's lick", fontsize=9.5)
    axC.set_xlabel("time from change (s)"); axC.set_ylabel("population z")
    axC.legend(frameon=False, fontsize=7, loc="upper left")

    # D: RT-matched effect distribution
    axD = fig.add_subplot(gs[1, 0])
    if eff.size:
        axD.hist(eff, bins=40, color="#7b3294", alpha=0.8)
        axD.axvline(0, color="k", lw=1.0)
        axD.axvline(eff_med, color="purple", lw=1.5, label=f"median {eff_med:.3f}")
    axD.set_title(f"1.3 RT-matched big-small effect\n[{EARLY_WIN[0]}-{EARLY_WIN[1]}s], "
                  f"{frac_pos*100:.0f}% > 0", fontsize=9.5)
    axD.set_xlabel("big - small population z (matched RT)"); axD.set_ylabel("# matches")
    axD.legend(frameon=False, fontsize=7)

    # E: misses
    axE = fig.add_subplot(gs[1, 1])
    axE.plot(bc, mm_b, color=BIG_C, lw=1.9, label=f"Big miss (n={int(mb.sum())})")
    axE.fill_between(bc, mb_lo, mb_hi, color=BIG_C, alpha=0.2)
    axE.plot(bc, mm_s, color=SMALL_C, lw=1.9, label=f"Small miss (n={int(ms.sum())})")
    axE.fill_between(bc, ms_lo, ms_hi, color=SMALL_C, alpha=0.2)
    axE.axvline(0, color="k", lw=1.0)
    axE.set_title("1.4 Misses: big vs small (no lick)", fontsize=9.5)
    axE.set_xlabel("time from change (s)"); axE.set_ylabel("population z")
    axE.legend(frameon=False, fontsize=7, loc="upper left")

    # F: verdict text
    axF = fig.add_subplot(gs[1, 2]); axF.axis("off")
    verdict = ("CLEAN" if (np.isfinite(margin) and margin > 0 and frac_pos > 0.95)
               else "MARGINAL" if (frac_pos > 0.8 or (np.isfinite(margin) and margin > 0))
               else "FAILS")
    txt = [
        "WS1 change-size RT-confound checks",
        f"sampling unit = TRIAL; window {WINDOW}",
        "",
        f"1.1 divergence onset: {div_onset:.3f} s" if np.isfinite(div_onset) else "1.1 divergence: none sig",
        f"     earliest big lick (5pct): {earliest_lick:.3f} s",
        f"     clean boundary (-150ms): {boundary:.3f} s",
        f"     margin (boundary-div): {margin:+.3f} s",
        f"       -> divergence {'PRECEDES' if margin>0 else 'after'} licks",
        "",
        f"1.3 RT-matched big-small: {eff_med:+.3f} z",
        f"     95% CI [{eff_lo:+.3f}, {eff_hi:+.3f}]",
        f"     {frac_pos*100:.0f}% of matches > 0",
        "",
        f"1.4 miss big n={int(mb.sum())}, small n={int(ms.sum())}",
        "",
        f"VERDICT (RT-independent scaling): {verdict}",
    ]
    axF.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle(f"{C.SUBJECT}: is change-size scaling RT-independent? "
                 "(descriptive controls)", fontsize=13, y=0.99)
    out = C.save_talk_figure(fig, "ws1_changesize_validation")
    print(f"[fig] wrote {out}")

    # stats CSV
    rowsdf = pd.DataFrame([{
        "div_onset_s": div_onset, "earliest_big_lick_s": earliest_lick,
        "clean_boundary_s": boundary, "margin_s": margin,
        "rtmatch_median": eff_med, "rtmatch_ci_lo": eff_lo, "rtmatch_ci_hi": eff_hi,
        "rtmatch_frac_pos": frac_pos, "n_hit_big": int(hb.sum()), "n_hit_small": int(hs.sum()),
        "n_miss_big": int(mb.sum()), "n_miss_small": int(ms.sum()), "verdict": verdict,
    }])
    sp = C.stats_csv_path("ws1_changesize_validation")
    rowsdf.to_csv(sp, index=False)
    print(f"[fig] wrote {sp}")
    print(rowsdf.T.to_string())
    print(f"\nWS1 VERDICT: {verdict}")


if __name__ == "__main__":
    main()
