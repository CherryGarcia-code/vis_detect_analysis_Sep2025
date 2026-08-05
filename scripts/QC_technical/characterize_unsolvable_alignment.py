"""QC1 Task 6a (DIAGNOSTIC): why do 41 pkls have no single-offset alignment?

`visdetect.core.run_alignment.solve_alignment` searches a SINGLE contiguous offset:
either the whole trial block slides along the events, or the whole event block is
covered by a sliding trial window. 41 / 253 pkls fail that search
(`neural_safe == False` in `trial_vs_baselineon_audit.csv`).

Hand-sampling two of them showed they are PIECEWISE, not broken:
  * BG_046_13082025 (483 trials / 488 events): best single offset 5 scores only 0.849
    overall, yet ~63% of 60-trial sliding windows score >0.95 (max 1.00).
  * BG_046_05082025 (461 / 463): offsets 1 AND 2 each align ~42-44% of windows at 1.00
    -- a trial was dropped mid-recording, so the correct offset CHANGES partway through.

This script measures that for all 41 so a scope decision can be made: is a SEGMENTED
solver worth building, or are these sessions genuinely unalignable?

Method (read-only; no pkl is written or repaired)
------------------------------------------------
Offset convention: `off` maps trial i -> event index i + off (so `off` is the solver's
`event_offset` when the trial block starts at 0, and negative `off` is the sign-A case
where trials outnumber events). Every offset with at least MIN_OVERLAP overlapping
trials is scanned -- not just the full-overlap ones the solver considers.

Per offset, the per-trial agreement vector is Check 1 of the solver, reused verbatim:
    isfinite(Change_ON[i + off]) == (outcome_i in CHANGE_PRESENTED_OUTCOMES)
CHANGE_PRESENTED_OUTCOMES is IMPORTED (case-sensitive; includes "Ref"), never redefined.

  1. best single-offset agreement, over the solver's own full-overlap search space
     (this is the number the solver saw and rejected);
  2. 60-trial sliding-window agreement at that offset: fraction of windows >0.95 and
     the max window -- long perfect stretches inside a mediocre overall score are the
     piecewise signature;
  3. GREEDY COVERAGE. Repeatedly take the offset that matches the most still-uncovered
     trials; report cumulative coverage after 1, 2, 3 offsets.
  4. SEGMENTED COVERAGE. Exact DP for the best CONTIGUOUS piecewise assignment:
     partition the trial axis into <= k blocks, give each block one offset, maximise
     matched trials. This is literally what a segmented solver would do, so its
     coverage-vs-k curve is the scope answer. Reported for k = 1, 2, 3 and the max.

     Greedy coverage (3) is the metric the task specified and is retained in the CSV,
     but it is NOT used to classify: it lets a trial be claimed by an offset that
     matches it coincidentally in no coherent block, and once run-hardened it is
     simultaneously PESSIMISTIC, because up to RUN_MIN-1 trials at every segment
     boundary are discarded (BG_046_05082025 scores 0.970 greedily, 0.983 segmented).

     Offset pool for the DP = the full-overlap ladder range(0, diff+1) (mirrored for
     diff < 0). Physically, each dropped or duplicated trial bumps the offset by one
     and never back, so nothing outside that range is a meaningful segment. A plan
     whose offsets march strictly one way is flagged `monotone_ladder` -- the direct
     signature of trials dropping one at a time.

  5. FUSED SEGMENTATION -- THE CLASSIFIER. The same DP run on a per-trial match that
     requires Check 1 AND Check 2 (see `fuse_mats`). Everything above scores Check 1
     only, and Check 1 is BINARY: a meaningless offset still matches ~50% of trials, so
     a Check-1 DP with enough segments reaches "100% coverage" on pure coincidence.
     Fusing collapses that -- the shuffle null falls from ~0.58 to ~0.20 -- and it is
     what the reported classification, `seg_plan` and segment counts are built from.

CHANCE CONTROL (mandatory here, not optional). Guards, all written to the CSV:
  * RUN-HARDENED greedy coverage: a trial counts only inside a run of >= RUN_MIN
    consecutive matches at that offset (a chance run of 20 is ~1e-6).
  * SHUFFLE NULL at the SAME offset pool and the SAME k, for both the Check-1 and the
    fused ladders, with outcome and change_time permuted JOINTLY. A k counts as a
    repair only if it beats its matched-k null by NULL_MARGIN, so no session is ever
    called piecewise on a number the null also reaches.

WHY THE FUSION MATTERS (it changed the answer, twice). Check 2 is near-deterministic:
an aligned trial residual is 0.0051 s, a misaligned one is seconds off. Scoring Check 1
alone called BG_031_15042025 and BG_041_09052025 repairable at >98% coverage on plans
whose segments were 0.3-4.2 s off -- fitted, not aligned. A post-hoc per-segment residual
filter still leaked BG_049_21082025, whose bad segments held only 2 and 4 testable trials
and so were "too thin to verify". Fusing at the TRIAL level removes the loophole by
construction rather than by threshold, which is why it, not the filter, is the classifier.

Run:  py scripts/QC_technical/characterize_unsolvable_alignment.py [--sessions BG_046_13082025 ...]
Out:  data/cache/qc_alignment/unsolvable_characterization.csv
"""
import os
import sys
import gc
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd

from visdetect.core.session import load_session
from visdetect.core.run_alignment import (ACCEPT_RESID_S, CHANGE_PRESENTED_OUTCOMES,
                                          _arr)

WINDOW = 60            # sliding-window length (trials), per the task spec
WIN_THRESH = 0.95      # "a window aligns" threshold
MIN_OVERLAP = 60       # an offset must overlap at least this many trials to be scanned
RUN_MIN = 20           # hardened coverage: min consecutive matches to count a segment
COVER_TARGET = 0.98    # coverage that counts as "piecewise-repairable"
MAX_OFFSETS = 10       # greedy cap; needing more than this is not a repair
MAX_SEG = 25           # segmentation DP cap (a 25-segment "repair" is not a repair)
MAX_POOL = 400         # bound the DP when |diff| is huge
NULL_MARGIN = 0.10     # observed coverage must beat the shuffle null by this much
MIN_SEG_RESID_N = 5    # a segment with fewer finite residuals cannot be residual-checked
VERIFIED_MIN = 0.80    # >= this fraction of trials must sit in residual-VERIFIED segments
N_SHUFFLES = 3
SEED = 42

AUDIT_CSV = os.path.join(_ROOT, "data", "cache", "qc_alignment",
                         "trial_vs_baselineon_audit.csv")
OUT_DIR = os.path.join(_ROOT, "data", "cache", "qc_alignment")
OUT_CSV = os.path.join(OUT_DIR, "unsolvable_characterization.csv")


# ── per-offset scoring ──────────────────────────────────────────────────────────
def _keep_long_runs(match, run_min):
    """Zero out isolated matches: keep only runs of >= run_min consecutive True."""
    m = np.asarray(match, dtype=bool)
    if run_min <= 1:
        return m.copy()
    out = np.zeros_like(m)
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s >= run_min:
            out[s:e] = True
    return out


def _window_stats(match, window=WINDOW):
    """(fraction of sliding windows > WIN_THRESH, max window agreement, n windows)."""
    m = np.asarray(match, dtype=float)
    if len(m) < window:
        return float("nan"), float("nan"), 0
    c = np.concatenate(([0.0], np.cumsum(m)))
    w = (c[window:] - c[:-window]) / float(window)
    return float(np.mean(w > WIN_THRESH)), float(np.max(w)), int(len(w))


def scan_offsets(expected, observed, min_overlap=MIN_OVERLAP):
    """Per-trial match vectors for every offset with >= min_overlap overlap.

    expected: bool[n_trials]  -- outcome says a change was presented
    observed: bool[n_events]  -- isfinite(Change_ON)
    Returns dict off -> (match_full_len_bool[n_trials], i0, i1) where [i0, i1) is the
    overlapping trial range (match is False outside it).
    """
    n_tr, n_ev = len(expected), len(observed)
    out = {}
    if n_tr == 0 or n_ev == 0:
        return out
    lo = -(n_tr - min_overlap)
    hi = n_ev - min_overlap
    for off in range(lo, hi + 1):
        i0 = max(0, -off)
        i1 = min(n_tr, n_ev - off)
        if i1 - i0 < min_overlap:
            continue
        m = np.zeros(n_tr, dtype=bool)
        m[i0:i1] = observed[i0 + off:i1 + off] == expected[i0:i1]
        out[off] = (m, i0, i1)
    return out


def greedy_cover(mats, n_trials, max_offsets=MAX_OFFSETS):
    """Greedily pick offsets covering the most still-uncovered trials.

    `mats` maps offset -> bool[n_trials] match vector.
    Returns list of (offset, newly_covered, cumulative_coverage_fraction).
    """
    covered = np.zeros(n_trials, dtype=bool)
    chosen = []
    for _ in range(max_offsets):
        best_off, best_gain = None, 0
        for off, m in mats.items():
            gain = int(np.count_nonzero(m & ~covered))
            if gain > best_gain:
                best_off, best_gain = off, gain
        if best_off is None or best_gain <= 0:
            break
        covered |= mats[best_off]
        chosen.append((best_off, best_gain, float(np.mean(covered))))
        if covered.all():
            break
    return chosen


def _cum(chosen, k):
    """Cumulative coverage after k greedy offsets (0.0 if fewer were found)."""
    if not chosen:
        return 0.0
    return float(chosen[min(k, len(chosen)) - 1][2])


def _n_to_target(chosen, target=COVER_TARGET):
    for i, (_, _, cov) in enumerate(chosen, start=1):
        if cov > target:
            return i
    return -1                                    # unreachable within MAX_OFFSETS


def fuse_mats(mats, bon, con, change_time):
    """AND Check 2 into the per-trial match, so the DP CANNOT overfit Check 1.

    Check 1 alone is binary: a wrong offset still matches ~50% of trials, and with
    enough segments the DP will happily stitch coincidences into "100% coverage".
    Check 2 is near-deterministic -- an aligned trial residual is 0.0051 s, a
    misaligned one is seconds off -- so a trial that carries a residual is required to
    pass it too. Trials with no presented change carry no residual and stay on Check 1.

    This is the honest classifier: it replaces a post-hoc filter (which leaked segments
    holding only 2-4 testable trials) with a per-trial constraint the DP must respect.
    """
    n_ev = min(len(bon), len(con))
    out = {}
    for off, (m, a, b) in mats.items():
        idx = np.arange(a, b)
        ev = idx + off
        keep = (ev >= 0) & (ev < n_ev)
        ok2 = np.ones(b - a, dtype=bool)
        if keep.any():
            r = (con[ev[keep]] - bon[ev[keep]]) - change_time[idx[keep]]
            ok2[keep] = ~np.isfinite(r) | (np.abs(r) < ACCEPT_RESID_S)
        mm = m.copy()
        mm[a:b] = m[a:b] & ok2
        out[off] = (mm, a, b)
    return out


def feasible_pool(mats, n_trials, n_events, max_pool=MAX_POOL):
    """Offsets a piecewise repair may draw on: the FULL-overlap offset range.

    If events outnumber trials by `diff`, every trial i maps to event i + off with
    0 <= off <= diff, and each dropped/extra trial bumps `off` by one -- so the whole
    admissible ladder is range(0, diff + 1). The sign-A case (trials outnumber events)
    is the mirror image, range(diff, 1). Nothing outside that range is a physically
    meaningful segment, which keeps the DP honest AND small (|diff| + 1 offsets).
    """
    diff = n_events - n_trials
    lo, hi = min(0, diff), max(0, diff)
    pool = [o for o in range(lo, hi + 1) if o in mats]
    if len(pool) > max_pool:                       # huge |diff|: keep the best-scoring
        pool = sorted(pool, key=lambda o: -float(
            np.mean(mats[o][0][mats[o][1]:mats[o][2]])))[:max_pool]
    return sorted(pool)


def candidate_offsets(mats, max_pool=50):
    """Offsets carrying POSITIVE evidence of being a real segment (reported, not used
    for the DP): some 60-trial window reaches WIN_THRESH. At chance (p ~ 0.5/trial)
    the probability that any 60-window scores >0.95 is ~1e-15, so this pool is
    essentially coincidence-free -- unlike naive greedy coverage.
    """
    scored = []
    for off, (m, a, b) in mats.items():
        _, mx, nw = _window_stats(m[a:b])
        if nw and np.isfinite(mx) and mx >= WIN_THRESH:
            scored.append((mx, float(np.mean(m[a:b])), off))
    scored.sort(reverse=True)
    return [off for _, _, off in scored[:max_pool]]


def segmented_cover(mats, pool, n_trials, max_seg=MAX_SEG):
    """Best CONTIGUOUS piecewise assignment of trials to offsets, for k = 1..max_seg.

    This is what a segmented solver would actually do: partition the trial axis into
    <= k contiguous blocks, give each block one offset, maximise matched trials.
    Exact DP (no greedy approximation, no boundary penalty).

    Returns list of dicts, index k-1: {k, coverage, n_distinct, offsets, plan}.
    """
    if not pool or n_trials == 0:
        return []
    P = len(pool)
    M = np.stack([mats[o][0].astype(np.int32) for o in pool])       # (P, n)
    NEG = -(10 ** 6)
    dp = np.full((max_seg + 1, P), NEG, dtype=np.int64)
    dp[1] = M[:, 0]
    ptr = np.zeros((n_trials, max_seg + 1, P), dtype=np.int16)
    ptr[0] = np.arange(P, dtype=np.int16)[None, :]
    for i in range(1, n_trials):
        new = np.full_like(dp, NEG)
        idx = np.arange(P)
        for k in range(1, max_seg + 1):
            stay = dp[k]
            if k > 1 and P > 1 and dp[k - 1].max() > NEG:
                # best switch SOURCE for each target offset o must exclude o itself,
                # so carry the top-2 of dp[k-1] rather than the global argmax alone
                order = np.argsort(dp[k - 1])[::-1]
                s1, s2 = int(order[0]), int(order[1])
                src = np.where(idx == s1, s2, s1)
                sw = dp[k - 1][src]
            else:
                src, sw = idx, np.full(P, NEG, dtype=np.int64)
            take_sw = sw > stay
            best = np.where(take_sw, sw, stay)
            ptr[i, k] = np.where(take_sw, src, idx).astype(np.int16)
            new[k] = np.where(best > NEG // 2, best + M[:, i], NEG)
        dp = new

    out = []
    for k in range(1, max_seg + 1):
        if dp[k].max() <= NEG // 2:
            continue
        o = int(np.argmax(dp[k]))
        seq = np.zeros(n_trials, dtype=np.int16)
        kk = k
        for i in range(n_trials - 1, -1, -1):
            seq[i] = o
            prev = int(ptr[i, kk, o])
            if prev != o:
                kk -= 1
            o = prev
        offs_seq = [pool[j] for j in seq]
        # run-length encode the plan
        plan, s = [], 0
        for i in range(1, n_trials + 1):
            if i == n_trials or offs_seq[i] != offs_seq[s]:
                plan.append(f"{offs_seq[s]}:{s}-{i}")
                s = i
        cov = float(dp[k].max()) / n_trials
        out.append({"k": k, "coverage": cov, "n_distinct": len(set(offs_seq)),
                    "n_segments": len(plan), "plan": ";".join(plan)})
    return out


def plan_stats(plan, mats, bon, con, change_time, n_trials):
    """Residual-verify a segmentation plan. THE guard against Check-1 overfitting.

    Check 1 is binary, so with a big offset pool and enough segments the DP can hit
    100% "coverage" on segments that are pure coincidence. Check 2 is an INDEPENDENT,
    continuous test: a genuinely aligned segment has median
    |(Change_ON - Baseline_ON) - change_time| ~ 0.005 s; a coincidental one lands
    seconds away. Measured on the 41: every monotone-ladder plan came in at 0.0051 s,
    while every non-monotone plan contained segments at 0.3-4.2 s. So the residual is
    what separates a real piecewise session from a fitted one.

    A segment PASSES if it has >= MIN_SEG_RESID_N finite residuals and its median is
    below the solver's own ACCEPT_RESID_S. Segments too thin to test are neither passed
    nor failed -- they simply do not count toward `verified_frac` (conservative).
    """
    resid, ns, ok, ver = [], [], True, 0
    for part in plan.split(";"):
        o, span = part.split(":")
        off = int(o)
        a, b = (int(x) for x in span.split("-"))
        r, rn = _seg_resid(off, np.arange(a, b), bon, con, change_time)
        resid.append("nan" if not np.isfinite(r) else f"{r:.4f}")
        ns.append(rn)
        if rn >= MIN_SEG_RESID_N and np.isfinite(r):
            if r < ACCEPT_RESID_S:
                m = mats.get(off)
                ver += int(np.count_nonzero(m[0][a:b])) if m is not None else (b - a)
            else:
                ok = False
    return {"resid_s": ";".join(resid), "resid_n": ";".join(str(x) for x in ns),
            "resid_ok": bool(ok), "verified_frac": float(ver) / max(n_trials, 1),
            "resid_max_s": max([float(x) for x in resid if x != "nan"], default=float("nan"))}


def _is_monotone_ladder(plan):
    """True if the segment offsets march strictly one way (0 -> 2 -> 3 -> 5 ...).

    That is the physical signature of trials being dropped (or duplicated) one at a
    time as the session runs: each drop bumps the offset by one and never back.
    """
    if not plan:
        return False
    offs = [int(p.split(":")[0]) for p in plan.split(";")]
    if len(offs) < 2:
        return True
    d = np.diff(offs)
    return bool(np.all(d > 0) or np.all(d < 0))


def _seg_resid(off, trial_idx, bon, con, change_time):
    """Median Check-2 residual over the trials assigned to `off`."""
    trial_idx = np.asarray(trial_idx, dtype=int)
    n_ev = min(len(bon), len(con))
    ok = (trial_idx + off >= 0) & (trial_idx + off < n_ev) & (trial_idx < len(change_time))
    trial_idx = trial_idx[ok]
    if len(trial_idx) == 0:
        return float("nan"), 0
    ev = trial_idx + off
    r = (con[ev] - bon[ev]) - change_time[trial_idx]
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan"), 0
    return float(np.median(np.abs(r))), int(len(r))


# ── one session ─────────────────────────────────────────────────────────────────
def characterize(path, rng):
    s = load_session(path)
    try:
        trials = list(s.trials or [])
        ni = s.ni_events or {}
        bon = _arr(ni.get("Baseline_ON"))
        con = _arr(ni.get("Change_ON"))
        outcomes = np.array([str(getattr(t, "trialoutcome", "") or "") for t in trials])
        change_time = np.array(
            [float(getattr(t, "change_time", np.nan))
             if getattr(t, "change_time", None) is not None else np.nan
             for t in trials], dtype=float)
    finally:
        del s
        gc.collect()

    n_tr, n_ev = len(outcomes), int(len(bon))
    rec = {"n_trials": n_tr, "n_events": n_ev, "n_change_on": int(len(con)),
           "diff": n_ev - n_tr,
           "event_len_mismatch": bool(len(con) != len(bon)),
           "classification": "", "note": ""}

    if n_tr == 0 or n_ev == 0 or len(con) == 0:
        rec["classification"] = "degenerate"
        rec["note"] = f"n_trials={n_tr} n_events={n_ev} n_change_on={len(con)}"
        return rec
    if n_tr < MIN_OVERLAP:
        rec["classification"] = "degenerate"
        rec["note"] = f"only {n_tr} trials (< window {MIN_OVERLAP}); not characterisable"
        return rec

    expected = np.isin(outcomes, list(CHANGE_PRESENTED_OUTCOMES))
    observed = np.isfinite(con)

    mats = scan_offsets(expected, observed)
    rec["n_offsets_scanned"] = len(mats)
    if not mats:
        rec["classification"] = "degenerate"
        rec["note"] = "no offset reaches the minimum overlap"
        return rec

    # ---- 1/2. best single offset, restricted to the SOLVER's full-overlap space ----
    full = {off: v for off, v in mats.items() if (v[2] - v[1]) == min(n_tr, n_ev)}
    space = full if full else mats
    best_off = max(space, key=lambda o: float(np.mean(space[o][0][space[o][1]:space[o][2]])))
    bm, i0, i1 = space[best_off]
    rec["best_offset"] = int(best_off)
    rec["best_agreement"] = float(np.mean(bm[i0:i1]))
    rec["best_overlap_n"] = int(i1 - i0)
    fw, mw, nw = _window_stats(bm[i0:i1])
    rec["best_frac_win_gt95"] = fw
    rec["best_max_win"] = mw
    rec["best_n_windows"] = nw

    # best window statistics achievable at ANY offset (piecewise sessions can hide the
    # good stretch at an offset whose OVERALL score is mediocre)
    fw_any, mw_any, off_any = -1.0, -1.0, np.nan
    for off, (m, a, b) in mats.items():
        f, mx, _ = _window_stats(m[a:b])
        if np.isfinite(f) and f > fw_any:
            fw_any, off_any = f, off
        if np.isfinite(mx) and mx > mw_any:
            mw_any = mx
    rec["max_frac_win_gt95_any"] = fw_any if fw_any >= 0 else float("nan")
    rec["offset_at_max_frac_win"] = int(off_any) if np.isfinite(off_any) else -999
    rec["max_win_any"] = mw_any if mw_any >= 0 else float("nan")

    # ---- 3. greedy coverage: naive (as specified) and run-hardened (primary) ----
    raw = {off: v[0] for off, v in mats.items()}
    hard = {off: _keep_long_runs(v[0], RUN_MIN) for off, v in mats.items()}

    ch_raw = greedy_cover(raw, n_tr)
    ch_run = greedy_cover(hard, n_tr)
    for tag, ch in (("", ch_raw), ("_run", ch_run)):
        for k in (1, 2, 3):
            rec[f"cov{k}{tag}"] = _cum(ch, k)
        rec[f"n_offsets_98{tag}"] = _n_to_target(ch)
    rec["cov_max_run"] = float(ch_run[-1][2]) if ch_run else 0.0
    rec["cov_max_raw"] = float(ch_raw[-1][2]) if ch_raw else 0.0

    # ---- 3b. exact contiguous segmentation (PRIMARY: what a segmented solver does) ----
    cands = candidate_offsets(mats)
    rec["n_candidate_offsets"] = len(cands)
    rec["candidate_offsets"] = ";".join(str(o) for o in sorted(cands)[:20])
    pool = feasible_pool(mats, n_tr, n_ev)
    rec["n_pool_offsets"] = len(pool)
    segs = segmented_cover(mats, pool, n_tr)
    by_k = {d["k"]: d for d in segs}
    for k in (1, 2, 3):
        rec[f"seg_cov{k}"] = float(by_k[k]["coverage"]) if k in by_k else float("nan")
    rec["seg_cov_max"] = float(max((d["coverage"] for d in segs), default=float("nan")))

    # ---- 3c. FUSED segmentation (Check 1 AND Check 2 per trial) -- the classifier ----
    fmats = fuse_mats(mats, bon, con, change_time)
    fsegs = segmented_cover(fmats, pool, n_tr)
    fby_k = {d["k"]: d for d in fsegs}
    for k in (1, 2, 3):
        rec[f"fused_cov{k}"] = float(fby_k[k]["coverage"]) if k in fby_k else float("nan")
    rec["fused_cov_max"] = float(max((d["coverage"] for d in fsegs), default=float("nan")))

    # ---- shuffle null on the SAME pool and the same k ladder ----
    # Outcome and change_time are permuted JOINTLY, so the null destroys the
    # trial<->event correspondence while preserving each trial's own structure.
    null_raw, null_run = [], []
    null_by_k = {k: [] for k in range(1, MAX_SEG + 1)}
    fnull_by_k = {k: [] for k in range(1, MAX_SEG + 1)}
    for _ in range(N_SHUFFLES):
        perm = rng.permutation(n_tr)
        exp_s, ct_s = expected[perm], change_time[perm]
        ms = scan_offsets(exp_s, observed)
        null_raw.append(_cum(greedy_cover({o: v[0] for o, v in ms.items()}, n_tr), 3))
        null_run.append(_cum(greedy_cover(
            {o: _keep_long_runs(v[0], RUN_MIN) for o, v in ms.items()}, n_tr), 3))
        p_s = [o for o in pool if o in ms]
        for d in segmented_cover(ms, p_s, n_tr):
            null_by_k[d["k"]].append(d["coverage"])
        for d in segmented_cover(fuse_mats(ms, bon, con, ct_s), p_s, n_tr):
            fnull_by_k[d["k"]].append(d["coverage"])
    rec["cov3_null"] = float(np.mean(null_raw))
    rec["cov3_run_null"] = float(np.mean(null_run))
    null_k = {k: (float(np.mean(v)) if v else 0.0) for k, v in null_by_k.items()}
    fnull_k = {k: (float(np.mean(v)) if v else 0.0) for k, v in fnull_by_k.items()}
    for k in (1, 2, 3):
        rec[f"seg_cov{k}_null"] = null_k.get(k, 0.0)
        rec[f"fused_cov{k}_null"] = fnull_k.get(k, 0.0)

    # ---- accept the smallest FUSED k clearing both gates ----
    #   (a) fused coverage > 98%   (b) beats the matched-k fused shuffle null.
    #   Because the fused match already carries Check 2 per trial, a plan that clears
    #   these is verified by construction -- plan_stats is then only a belt-and-braces
    #   re-check that every segment's median residual really is at the aligned value.
    seg_hit, best_ver, refuted = None, 0.0, False
    for d in fsegs:
        st = plan_stats(d["plan"], fmats, bon, con, change_time, n_tr)
        d.update(st)
        best_ver = max(best_ver, st["verified_frac"])
        gate_ab = (d["coverage"] > COVER_TARGET
                   and d["coverage"] - fnull_k.get(d["k"], 0.0) > NULL_MARGIN)
        if seg_hit is None and gate_ab and st["resid_ok"]:
            seg_hit = d
    rec["verified_coverage_max"] = best_ver
    # Check 1 alone would have called it repairable; the fused/Check-2 test says no
    rec["refuted_by_residual"] = bool(
        seg_hit is None and np.isfinite(rec["seg_cov_max"])
        and rec["seg_cov_max"] > COVER_TARGET)
    rec["n_segments_98"] = int(seg_hit["k"]) if seg_hit else -1
    rec["n_distinct_offsets_98"] = int(seg_hit["n_distinct"]) if seg_hit else -1
    rec["seg_plan"] = (seg_hit or (fsegs[-1] if fsegs else {})).get("plan", "")
    rec["monotone_ladder"] = bool(
        seg_hit and _is_monotone_ladder(seg_hit["plan"]))
    # the null AT THE ACCEPTED k -- the number that has to be beaten, not a k=3 proxy
    rec["seg_cov_hit"] = float(seg_hit["coverage"]) if seg_hit else float("nan")
    rec["seg_cov_hit_null"] = float(fnull_k.get(seg_hit["k"], 0.0)) if seg_hit else float("nan")
    rec["seg_null_margin"] = (rec["seg_cov_hit"] - rec["seg_cov_hit_null"]
                              if seg_hit else float("nan"))

    # per-segment Check-2 residuals of the reported plan (already computed above)
    reported = seg_hit or (segs[-1] if segs else {})
    rec["seg_resid_s"] = reported.get("resid_s", "")
    rec["seg_resid_n"] = reported.get("resid_n", "")
    rec["seg_resid_max_s"] = reported.get("resid_max_s", float("nan"))
    rec["seg_resid_ok"] = reported.get("resid_ok", False)

    # ---- orthogonal Check-2 residual per chosen (hardened) offset ----
    offs, gains, resids, resid_ns = [], [], [], []
    covered = np.zeros(n_tr, dtype=bool)
    for off, gain, _cov in ch_run:
        newly = np.where(hard[off] & ~covered)[0]
        covered |= hard[off]
        r, rn = _seg_resid(off, newly, bon, con, change_time)
        offs.append(int(off)); gains.append(int(gain))
        resids.append("nan" if not np.isfinite(r) else f"{r:.4f}"); resid_ns.append(int(rn))
    rec["chosen_offsets"] = ";".join(str(o) for o in offs)
    rec["chosen_gains"] = ";".join(str(g) for g in gains)
    rec["chosen_resid_s"] = ";".join(resids)
    rec["chosen_resid_n"] = ";".join(str(n) for n in resid_ns)

    # ---- classification: exact segmentation, on DISTINCT offsets (per the task) ----
    k = rec["n_distinct_offsets_98"]
    if k == 1:
        rec["classification"] = "piecewise-1"
        rec["note"] = (
            "one offset scores a perfect Check 1 -- solver rejected it on Check 2 "
            "(residual), so this is not a piecewise problem at all"
            if rec["best_agreement"] >= 1.0 else
            f"one offset covers >98% ({rec['seg_cov1']:.3f}) but misses the solver's "
            f"EXACT 1.0 Check-1 bar (ACCEPT_AGREEMENT); a handful of stray trials, "
            f"not a segmentation")
    elif k == 2:
        rec["classification"] = "piecewise-2"
    elif k == 3:
        rec["classification"] = "piecewise-3"
    elif k > 3:
        rec["classification"] = "piecewise-many"
    elif rec["refuted_by_residual"]:
        rec["classification"] = "unalignable"
        rec["note"] = (f"Check 1 ALONE reaches {rec['seg_cov_max']:.3f} coverage, but "
                       f"fusing the INDEPENDENT Check-2 residual collapses it to "
                       f"{rec['fused_cov_max']:.3f} -- the Check-1 plan is coincidence, "
                       f"not alignment")
    else:
        rec["classification"] = "unalignable"
        rec["note"] = (f">98% unreachable with {MAX_SEG} segments "
                       f"(fused max {rec['fused_cov_max']:.3f}, Check-1 max "
                       f"{rec['seg_cov_max']:.3f}, over {rec['n_pool_offsets']} "
                       f"feasible offsets; fused null3 {rec['fused_cov3_null']:.3f})")
    if rec["n_segments_98"] > 0:
        rec["note"] = (rec["note"] + " | " if rec["note"] else "") + \
            f"{rec['n_segments_98']} contiguous segments" + \
            (", monotone ladder" if rec["monotone_ladder"] else "")
    return rec


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="restrict to these session stems (file name without .pkl)")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    audit = pd.read_csv(AUDIT_CSV)
    bad = audit[~audit["neural_safe"].fillna(False).astype(bool)].copy()
    if args.sessions:
        want = {str(x).replace(".pkl", "") for x in args.sessions}
        bad = bad[bad["file"].str.replace(".pkl", "", regex=False).isin(want)]
    print(f"characterising {len(bad)} unsolvable pkls (of {len(audit)} audited)",
          flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for i, (_, r) in enumerate(bad.iterrows(), start=1):
        subj, fname = str(r["subject"]), str(r["file"])
        path = os.path.join(_ROOT, "data", "pkls", subj, fname)
        rec = {"subject": subj, "file": fname,
               "session": fname.replace(".pkl", "")}
        print(f"[{i}/{len(bad)}] {fname} ...", flush=True)
        try:
            rec.update(characterize(path, rng))
        except Exception as exc:
            rec.update({"classification": "error",
                        "note": f"{type(exc).__name__}: {exc}"})
        rows.append(rec)
        print(f"    -> {rec.get('classification')} "
              f"best_agr={rec.get('best_agreement', float('nan')):.4f} "
              f"off={rec.get('best_offset')} "
              f"win>.95={rec.get('best_frac_win_gt95', float('nan')):.3f} "
              f"| check1_cov1/2/3={rec.get('seg_cov1', float('nan')):.3f}/"
              f"{rec.get('seg_cov2', float('nan')):.3f}/"
              f"{rec.get('seg_cov3', float('nan')):.3f} "
              f"max={rec.get('seg_cov_max', float('nan')):.3f} "
              f"| FUSED1/2/3={rec.get('fused_cov1', float('nan')):.3f}/"
              f"{rec.get('fused_cov2', float('nan')):.3f}/"
              f"{rec.get('fused_cov3', float('nan')):.3f} "
              f"max={rec.get('fused_cov_max', float('nan')):.3f} "
              f"null3={rec.get('fused_cov3_null', float('nan')):.3f} "
              f"resid_max={rec.get('seg_resid_max_s', float('nan')):.4f} "
              f"| greedy cov1/2/3_run={rec.get('cov1_run', 0):.3f}/"
              f"{rec.get('cov2_run', 0):.3f}/{rec.get('cov3_run', 0):.3f} "
              f"| {rec.get('note','')}", flush=True)
        gc.collect()
        pd.DataFrame(rows).to_csv(args.out, index=False)   # incremental, crash-safe

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n=== SUMMARY ===", flush=True)
    cols = ["session", "n_trials", "n_events", "diff", "best_offset", "best_agreement",
            "best_frac_win_gt95", "n_pool_offsets", "n_candidate_offsets",
            "seg_cov1", "seg_cov2", "seg_cov3", "seg_cov_max",
            "fused_cov1", "fused_cov2", "fused_cov3", "fused_cov_max",
            "fused_cov3_null", "seg_resid_max_s", "refuted_by_residual",
            "n_distinct_offsets_98", "n_segments_98", "monotone_ladder",
            "cov1_run", "cov2_run", "cov3_run", "classification"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df[cols].to_string(index=False), flush=True)
    print("\nclassification counts:", flush=True)
    print(df["classification"].value_counts().to_string(), flush=True)
    print(f"\nSaved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
