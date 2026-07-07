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

C.setup_talk_style()

BIN = DEFAULT_BIN_SIZE
WINDOW = (-1.0, 2.5)                                  # wide enough to reach licks for censoring
BASELINE = EVENT_RESPONSIVENESS_WINDOWS["Change_ON"][0]   # canonical (-0.4, -0.05)
MOTOR_PREP_S = 0.15                                   # lick leads ~150 ms by motor prep
EARLY_WIN = (0.0, 0.5)                                # matched-effect measurement window
CACHE = C.CACHE_DIR / f"ws1_changesize_trials_{C.SUBJECT}.npz"
SMALL_C, BIG_C = C.CHANGE_COLORS["small"], C.CHANGE_COLORS["big"]   # canonical (config.CHANGE_SIZE_COLORS)

# ── Per-trial builder (parallel over sessions; pkls are LOCAL so this is safe) ─────────────
# Each session's per-trial population traces are computed in a worker process. We store the
# population mean z per trial for: all units, narrow, broad (cell type, COMMON cutoff), and
# TF-responsive / non-responsive (Khilkevich-Lohse registry). One row per go hit/miss trial.
_WS1G = {}


def _bt_init(ct_lookup, tf_lut):
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"                       # pin BLAS per worker (project convention)
    _WS1G["ct"] = ct_lookup
    _WS1G["tf"] = tf_lut


def _popmean(tr, mask, nb):
    return (smooth_psth(np.nanmean(tr[:, mask], axis=1), BIN, DEFAULT_SIGMA_MS)
            if mask.any() else np.full(nb, np.nan))


def _build_one_session(s8):
    """Worker: return (bc, dict-of-per-trial-lists) for one session, or None to skip."""
    ct_lookup, tf_lut = _WS1G["ct"], _WS1G["tf"]
    try:
        sess_o = load_session(s8)
    except Exception:  # noqa: BLE001
        return None
    cids = get_good_cluster_ids(sess_o)
    tdf = get_trial_dataframe(sess_o)
    if len(cids) == 0 or tdf.empty or "trial_idx" not in tdf.columns:
        return None
    tdf = tdf.set_index("trial_idx")
    ct = np.array([ct_lookup.get((s8, int(c)), C.UNKNOWN) for c in cids])
    calls = [tf_lut.get((C.canon(s8), int(c))) for c in cids]
    resp = np.array([c is True for c in calls])
    nonresp = np.array([c is False for c in calls])
    try:
        tensor, bc, valid = build_population_tensor(
            sess_o, list(cids), event_name="Change_ON", window=WINDOW, bin_size=BIN)
    except ValueError:
        return None
    z = compute_zscore_normalized(tensor, bc, BASELINE)
    z[:, :, np.nanmax(np.abs(z), axis=(0, 1)) > 50.0] = np.nan
    nb = len(bc)
    n_mask, b_mask = ct == C.NARROW, ct == C.BROAD
    sub = tdf.loc[valid]
    R = {k: [] for k in ("pop_all", "pop_n", "pop_b", "pop_tf", "pop_non", "grp", "outc", "rt", "sess")}
    for ti, p in enumerate(valid):
        row = sub.loc[p]
        if not bool(row["is_go"]) or str(row["outcome"]).lower() not in ("hit", "miss"):
            continue
        o = str(row["outcome"]).lower()
        tr = z[ti]
        R["pop_all"].append(smooth_psth(np.nanmean(tr, axis=1), BIN, DEFAULT_SIGMA_MS))
        R["pop_n"].append(_popmean(tr, n_mask, nb)); R["pop_b"].append(_popmean(tr, b_mask, nb))
        R["pop_tf"].append(_popmean(tr, resp, nb)); R["pop_non"].append(_popmean(tr, nonresp, nb))
        R["grp"].append("big" if float(row["change_size"]) >= 2.0 else "small")
        R["outc"].append(o); R["rt"].append(float(row["rt"]) if o == "hit" else np.nan); R["sess"].append(s8)
    del sess_o; gc.collect()
    return bc, R


def build_trials(n_workers=None):
    ct_lookup, sessions_8 = C.celltype_and_sessions(C.SUBJECT)
    tf_lut = {}
    if C.has_tf_registry(C.SUBJECT):
        reg = C.load_tf_responsive(C.SUBJECT)
        tf_lut = {(C.canon(str(r.session_date)), int(r.unit)): bool(r.resp_log2)
                  for r in reg.itertuples()}
    n_workers = n_workers or min(8, (os.cpu_count() or 4) - 2)
    print(f"[WS1] building {len(sessions_8)} sessions on {n_workers} workers "
          f"(TF registry: {'yes' if tf_lut else 'no'})")
    acc = {k: [] for k in ("pop_all", "pop_n", "pop_b", "pop_tf", "pop_non", "grp", "outc", "rt", "sess")}
    bc_ref = None
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_bt_init,
                             initargs=(ct_lookup, tf_lut)) as ex:
        for res in ex.map(_build_one_session, sessions_8):
            done += 1
            if res is None:
                continue
            bc_ref, R = res
            for k in acc:
                acc[k].extend(R[k])
            print(f"  [{done}/{len(sessions_8)}] +{len(R['grp'])} go-trials (total {len(acc['grp'])})")

    out = dict(bc=bc_ref, pop_all=np.array(acc["pop_all"]), pop_narrow=np.array(acc["pop_n"]),
               pop_broad=np.array(acc["pop_b"]), pop_tfresp=np.array(acc["pop_tf"]),
               pop_nonresp=np.array(acc["pop_non"]), group=np.array(acc["grp"]),
               outcome=np.array(acc["outc"]), rt=np.array(acc["rt"], float), session=np.array(acc["sess"]))
    np.savez_compressed(CACHE, **out)
    print(f"[WS1] wrote {CACHE}  ({len(acc['grp'])} go-trials)")
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


def _tf_pair(ax, bc, tfp_m, non_m, col, g, lab):
    """Plot one condition for TF+ (colour, solid + band) and TF- (grey, dashed + light band)."""
    mu, lo, hi = _mean_ci_trials(tfp_m)
    ax.plot(bc, mu, color=col, lw=2.0, label=f"{lab}·TF+"); ax.fill_between(bc, lo, hi, color=col, alpha=0.22)
    mun, lon, hin = _mean_ci_trials(non_m)
    ax.plot(bc, mun, color=g, lw=1.6, ls="--", label=f"{lab}·TF−"); ax.fill_between(bc, lon, hin, color=g, alpha=0.15)


def render_tf(D, args):
    """TF-responsive vs non-responsive version of ws1 — the SAME RT-confound controls as the
    cell-type figure (change-aligned + divergence, RT distributions, censor-before-lick,
    RT-matched, misses), with the neural population split TF+ vs TF-. TF+ = colour (solid+band),
    TF- = grey (dashed + light band). Answers: is the RT-independent big>small scaling carried
    by TF-responsive cells?"""
    bc = D["bc"]; group = D["group"]; outcome = D["outcome"]; rt = D["rt"]; session = D["session"]
    tfp = D["pop_tfresp"]; non = D["pop_nonresp"]
    big = group == "big"; hit = outcome == "hit"; miss = outcome == "miss"
    hb, hs = hit & big, hit & (~big)
    CB, CS = C.CHANGE_COLORS["big"], C.CHANGE_COLORS["small"]
    GB, GS = C.TF_MINUS_GREY[1], C.TF_MINUS_GREY[0]

    # divergence (cluster-perm, within-session shuffle) per TF group, on HIT trials
    div_tf = cluster_perm(tfp[hit], big[hit], session[hit], bc, n_perm=args.n_perm, n_workers=args.n_workers)[0]
    div_non = cluster_perm(non[hit], big[hit], session[hit], bc, n_perm=args.n_perm, n_workers=args.n_workers)[0]
    rt_big = rt[hb]; rt_big = rt_big[np.isfinite(rt_big)]
    rt_small = rt[hs]; rt_small = rt_small[np.isfinite(rt_small)]
    earliest = np.percentile(rt_big, 5) if rt_big.size else np.nan
    boundary = earliest - MOTOR_PREP_S

    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)

    # 1.1 change-aligned big/small (hits): TF+ vs TF- + earliest-lick boundary
    axA = fig.add_subplot(gs[0, 0])
    _tf_pair(axA, bc, tfp[hb], non[hb], CB, GB, "Big")
    _tf_pair(axA, bc, tfp[hs], non[hs], CS, GS, "Small")
    axA.axvline(0, color="k", lw=1.0); axA.axhline(0, color="0.6", lw=0.7, ls=":")
    if np.isfinite(boundary):
        axA.axvline(boundary, color="green", ls=":", lw=1.3, label=f"lick bound {boundary:.2f}s")
    axA.set_xlabel("time from change (s)"); axA.set_ylabel("population z (per-trial)")
    axA.set_title("1.1 Change-aligned big vs small (hits)", fontsize=C.FS["title"])
    axA.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

    # 1.1 RT distributions (behavioural — identical across TF groups; context for RT-matching)
    axB = fig.add_subplot(gs[0, 1])
    axB.hist(rt_small, bins=30, color=CS, alpha=0.6, density=True, label=f"small RT (n={rt_small.size})")
    axB.hist(rt_big, bins=30, color=CB, alpha=0.6, density=True, label=f"big RT (n={rt_big.size})")
    if np.isfinite(earliest):
        axB.axvline(earliest, color="green", ls=":", lw=1.3, label=f"big 5th pct {earliest:.2f}s")
    axB.set_title("1.1 RT distributions (behavioural)", fontsize=C.FS["title"])
    axB.set_xlabel("reaction time from change (s)"); axB.set_ylabel("density")
    axB.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

    # 1.2 censor before each trial's lick (hits): TF+ vs TF-
    def _censor(mat):
        cm = mat[hit].copy(); r = rt[hit]
        for i in range(cm.shape[0]):
            if np.isfinite(r[i]):
                cm[i, bc >= r[i]] = np.nan
        return cm
    ctf, cno, bh = _censor(tfp), _censor(non), big[hit]
    axC = fig.add_subplot(gs[0, 2])
    _tf_pair(axC, bc, ctf[bh], cno[bh], CB, GB, "Big")
    _tf_pair(axC, bc, ctf[~bh], cno[~bh], CS, GS, "Small")
    axC.axvline(0, color="k", lw=1.0)
    axC.set_title("1.2 Censored before each trial's lick", fontsize=C.FS["title"])
    axC.set_xlabel("time from change (s)"); axC.set_ylabel("population z")
    axC.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

    # 1.3 RT-matched big-small effect: TF+ vs TF-
    eff_tf = rt_match(tfp[hit], big[hit], rt[hit], bc, n_match=args.n_match, n_workers=args.n_workers)
    eff_tf = eff_tf[np.isfinite(eff_tf)]
    eff_non = rt_match(non[hit], big[hit], rt[hit], bc, n_match=args.n_match, n_workers=args.n_workers)
    eff_non = eff_non[np.isfinite(eff_non)]
    axD = fig.add_subplot(gs[1, 0])
    if eff_non.size:
        axD.hist(eff_non, bins=40, color=GB, alpha=0.55, label=f"TF− (med {np.median(eff_non):+.3f})")
    if eff_tf.size:
        axD.hist(eff_tf, bins=40, color=C.RTMATCH_PURPLE, alpha=0.75, label=f"TF+ (med {np.median(eff_tf):+.3f})")
    axD.axvline(0, color="k", lw=1.0)
    axD.set_xlabel(f"big−small pop z (matched RT) [{EARLY_WIN[0]}–{EARLY_WIN[1]}s]"); axD.set_ylabel("# matches")
    axD.set_title("1.3 RT-matched big−small effect", fontsize=C.FS["title"])
    axD.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

    # 1.4 misses big vs small (no lick): TF+ vs TF-
    mb, ms = miss & big, miss & (~big)
    axE = fig.add_subplot(gs[1, 1])
    _tf_pair(axE, bc, tfp[mb], non[mb], CB, GB, f"Big(n={int(mb.sum())})")
    _tf_pair(axE, bc, tfp[ms], non[ms], CS, GS, f"Small(n={int(ms.sum())})")
    axE.axvline(0, color="k", lw=1.0)
    axE.set_title("1.4 Misses: big vs small (no lick)", fontsize=C.FS["title"])
    axE.set_xlabel("time from change (s)"); axE.set_ylabel("population z")
    axE.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

    # 1.5 verdict
    axF = fig.add_subplot(gs[1, 2]); axF.axis("off")
    fp_tf = float(np.mean(eff_tf > 0)) if eff_tf.size else np.nan
    fp_non = float(np.mean(eff_non > 0)) if eff_non.size else np.nan
    txt = ["WS1 × TF-responsive", "", "divergence onset (hits):",
           (f"  TF+ : {div_tf:.3f} s" if np.isfinite(div_tf) else "  TF+ : none sig"),
           (f"  TF− : {div_non:.3f} s" if np.isfinite(div_non) else "  TF− : none sig"),
           (f"  earliest-lick bound: {boundary:.3f} s" if np.isfinite(boundary) else ""),
           "", "RT-matched big−small (0–0.5 s):",
           (f"  TF+ : {np.median(eff_tf):+.3f} z, {fp_tf*100:.0f}% >0" if eff_tf.size else "  TF+ : n/a"),
           (f"  TF− : {np.median(eff_non):+.3f} z, {fp_non*100:.0f}% >0" if eff_non.size else "  TF− : n/a"),
           "", "TF-responsive carry the RT-", "independent scaling if TF+ >> TF−.",
           "", "NOT movement-controlled (GLM)."]
    axF.text(0.0, 1.0, "\n".join([t for t in txt if t != ""] if False else txt),
             va="top", ha="left", fontsize=C.FS["caption"], family="monospace")

    fig.suptitle(f"{C.SUBJECT} {C.region_label()}: change-size RT-confound controls — "
                 "TF-responsive vs non-responsive", fontsize=C.FS["suptitle"], y=0.99)
    fig.text(0.5, 0.01,
             "Per-trial population z over TF-responsive (colour, solid + band) vs non-responsive (grey "
             "dashed + light band); big/small = dark/light. Same RT-confound controls as the cell-type "
             "figure. TF-responsive = Khilkevich-Lohse GLM, NOT movement-controlled. Sampling unit = trial.",
             ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.save_talk_figure(fig, "ws1_changesize_validation_tf")
    print(f"[fig] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["celltype", "tf"], default="celltype",
                    help="tf = TF-responsive vs non-responsive population (3 striatum mice only)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n_workers", type=int, default=min(8, (os.cpu_count() or 4) - 2))
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--n_match", type=int, default=500)
    args = ap.parse_args()

    if args.group == "tf" and not C.has_tf_registry(C.SUBJECT):
        raise SystemExit(f"no TF registry for {C.SUBJECT} (3 striatum mice only)")
    if args.force or not CACHE.exists():
        D = build_trials(n_workers=args.n_workers)
    else:
        D = load_trials()
        if args.group == "tf" and "pop_tfresp" not in D:
            print("[WS1] cache lacks TF columns -> rebuilding with TF")
            D = build_trials(n_workers=args.n_workers)
    if args.group == "tf":
        render_tf(D, args); return
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
    axA.set_title("1.1 Change-aligned big vs small (hits)\n+ divergence vs lick boundary", fontsize=C.FS["title"])
    axA.set_xlabel("time from change (s)"); axA.set_ylabel("population z (per-trial)")
    axA.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

    # B: RT distributions
    axB = fig.add_subplot(gs[0, 1])
    axB.hist(rt_small, bins=30, color=SMALL_C, alpha=0.6, density=True, label=f"small RT (n={rt_small.size})")
    axB.hist(rt_big, bins=30, color=BIG_C, alpha=0.6, density=True, label=f"big RT (n={rt_big.size})")
    if np.isfinite(earliest_lick):
        axB.axvline(earliest_lick, color="green", ls=":", lw=1.3, label=f"big 5th pct {earliest_lick:.2f}s")
    axB.set_title("1.1 RT (from change) distributions", fontsize=C.FS["title"])
    axB.set_xlabel("reaction time from change (s)"); axB.set_ylabel("density")
    axB.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

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
    axC.set_title("1.2 Censored before each trial's lick", fontsize=C.FS["title"])
    axC.set_xlabel("time from change (s)"); axC.set_ylabel("population z")
    axC.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

    # D: RT-matched effect distribution
    axD = fig.add_subplot(gs[1, 0])
    if eff.size:
        axD.hist(eff, bins=40, color=C.RTMATCH_PURPLE, alpha=0.8)
        axD.axvline(0, color="k", lw=1.0)
        axD.axvline(eff_med, color="purple", lw=1.5, label=f"median {eff_med:.3f}")
    axD.set_title(f"1.3 RT-matched big-small effect\n[{EARLY_WIN[0]}-{EARLY_WIN[1]}s], "
                  f"{frac_pos*100:.0f}% > 0", fontsize=C.FS["title"])
    axD.set_xlabel("big - small population z (matched RT)"); axD.set_ylabel("# matches")
    axD.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

    # E: misses
    axE = fig.add_subplot(gs[1, 1])
    axE.plot(bc, mm_b, color=BIG_C, lw=1.9, label=f"Big miss (n={int(mb.sum())})")
    axE.fill_between(bc, mb_lo, mb_hi, color=BIG_C, alpha=0.2)
    axE.plot(bc, mm_s, color=SMALL_C, lw=1.9, label=f"Small miss (n={int(ms.sum())})")
    axE.fill_between(bc, ms_lo, ms_hi, color=SMALL_C, alpha=0.2)
    axE.axvline(0, color="k", lw=1.0)
    axE.set_title("1.4 Misses: big vs small (no lick)", fontsize=C.FS["title"])
    axE.set_xlabel("time from change (s)"); axE.set_ylabel("population z")
    axE.legend(fontsize=C.FS["legend"], **C.LEGEND_KW)

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
    axF.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=C.FS["caption"], family="monospace")

    fig.suptitle(f"{C.SUBJECT}: is change-size scaling RT-independent? "
                 "(descriptive controls)", fontsize=C.FS["suptitle"], y=0.99)
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
