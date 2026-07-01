"""Population geometry of TF encoding vs behavioural state — Lohse & Khilkevich
2025 (Fig 3) replication in BG data.

Per session, on the good/stable population:
  Sensory-input CD  = per-unit (fast-minus-slow) TF-pulse response in the window
                      0.122-0.167 s post-pulse (Lohse's window).
  Task-state CD     = per-unit (engaged - Disengaged) pre-pulse baseline firing
                      (-0.4-0 s), engaged = StimSens+Impulsive.
Both are N-neuron vectors (unit-normalised). Headline:
  cosine(Sensory, Task-state)  -> predicted ~0 (ORTHOGONAL), vs a
  neuron-shuffle null.
Then project each (state x fast/slow) mean population response into the
(Sensory, Task-state) plane: fast/slow separate along the sensory axis, the
three states separate (displace) along the task-state axis, with the sensory
separation PRESERVED across states (Lohse Fig 3f,i,j). Tracked across learning
(sessions ordered by date + d').

Parallel across sessions. No early/late blocks (absent in this task).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import argparse
import glob
import concurrent.futures as cf
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_bg046/src")
from visdetect.core.session import load_session  # noqa: E402
from visdetect.analysis.tf_glm import (  # noqa: E402
    TFGLMConfig, assemble_design, count_vector, pulse_times_from_tf, tf_pulse_peth)
from visdetect.analysis.tf_glm_data import session_trial_regressors  # noqa: E402

PKL_DIR = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/data/pkls/{subj}"
STATE_DIR = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/data/cache/state_tags/{subj}"
ENGAGED = {"StimSens", "Impulsive"}
STATES = ["StimSens", "Impulsive", "Disengaged"]
STATE_COLORS = {"StimSens": "#6baed6", "Impulsive": "#ef6548", "Disengaged": "#3474ae"}
SENSORY_WIN = (0.122, 0.167)   # Lohse's sensory-response window
BASELINE_WIN = (-0.40, 0.0)    # pre-pulse baseline for task-state
SEED = 42


def _cfg():
    return TFGLMConfig(include_movement=False, include_phase=False, include_tiled_baseline=True,
                       standardize_design=True, fast_fit=True, tf_encoding="log2",
                       min_pulses_per_label=20)


def _state_by_trial(subj, session):
    date = session.replace(f"{subj}_", "", 1)
    f = Path(STATE_DIR.format(subj=subj)) / f"{date}.csv"
    return pd.read_csv(f).set_index("trial_idx")["state_label"].to_dict() if f.exists() else None


def _pulse_states(ptimes, edges, ti, lab):
    idx = np.searchsorted(edges, ptimes, side="right") - 1
    ok = (idx >= 0) & (idx < ti.size)
    out = np.full(ptimes.shape, "UNTAGGED", dtype=object)
    out[ok] = [lab.get(int(t), "UNTAGGED") for t in ti[idx[ok]]]
    return out


def _win_mean(peth, t, win, bs):
    """mean firing (Hz) in a time window of a pulse-PETH (counts/bin -> Hz)."""
    m = (t >= win[0]) & (t <= win[1])
    return float(np.nanmean(peth[m]) / bs) if m.any() else np.nan


def session_geometry(args):
    subj, session = args
    lab = _state_by_trial(subj, session)
    if lab is None:
        return None
    cfg = _cfg()
    s = load_session(str(Path(PKL_DIR.format(subj=subj)) / f"{session}.pkl"))
    uids = list(getattr(s, "good_and_stable_ids", None) or getattr(s, "good_cluster_ids", None) or [])
    if len(uids) < 8:
        return None
    trials, units = session_trial_regressors(s, cfg)
    d = assemble_design(trials, cfg)
    ti, edges, bs = d.trial_index, d.bin_edges, cfg.bin_s
    win = cfg.pulse_eval_win
    fast, slow = pulse_times_from_tf(d, cfg)
    fs, ss = _pulse_states(fast, edges, ti, lab), _pulse_states(slow, edges, ti, lab)
    if fast.size < 40 or slow.size < 40:
        return None
    # pulse-time subsets by state
    def sub(p, st, want):
        return p[np.isin(st, want)]
    all_f, all_s = fast, slow
    eng_f, eng_s = sub(fast, fs, list(ENGAGED)), sub(slow, ss, list(ENGAGED))
    dis_f, dis_s = sub(fast, fs, ["Disengaged"]), sub(slow, ss, ["Disengaged"])
    if min(len(dis_f), len(dis_s), len(eng_f), len(eng_s)) < 20:
        return None
    per_state_f = {st: sub(fast, fs, [st]) for st in STATES}
    per_state_s = {st: sub(slow, ss, [st]) for st in STATES}

    # bin index of each pulse (once), + fixed window bin-offsets
    def pbins(p):
        return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 1)
    sw = np.arange(int(round(SENSORY_WIN[0] / bs)), int(round(SENSORY_WIN[1] / bs)) + 1)
    bw = np.arange(int(round(BASELINE_WIN[0] / bs)), int(round(BASELINE_WIN[1] / bs)))
    fb, sb = pbins(all_f), pbins(all_s)
    engb = pbins(np.concatenate([eng_f, eng_s])); disb = pbins(np.concatenate([dis_f, dis_s]))
    stf_b = {st: pbins(per_state_f[st]) for st in STATES}
    sts_b = {st: pbins(per_state_s[st]) for st in STATES}

    def win_hz(y, pb, offs):
        """mean firing (Hz) over the window `offs` (bin offsets) around pulse-bins pb."""
        if pb.size == 0:
            return np.nan
        idx = pb[:, None] + offs[None, :]
        val = (idx >= 0) & (idx < y.size)
        g = np.where(val, y[np.clip(idx, 0, y.size - 1)], 0.0)
        return float(g.sum() / (val.sum() * bs)) if val.sum() else np.nan

    # 2-fold split of every pulse subset (interleaved) -> CDs on fold A, measure
    # on fold B (and swap). Cosine uses sensory-CD and task-state-CD from
    # INDEPENDENT halves, so shared trial noise cannot inflate it.
    A = lambda b: b[0::2]; B = lambda b: b[1::2]  # noqa: E731
    stfA = {st: A(stf_b[st]) for st in STATES}; stfB = {st: B(stf_b[st]) for st in STATES}
    stsA = {st: A(sts_b[st]) for st in STATES}; stsB = {st: B(sts_b[st]) for st in STATES}

    uids = [u for u in uids if u in units]
    N = len(uids)
    senA = np.full(N, np.nan); senB = np.full(N, np.nan)
    tskA = np.full(N, np.nan); tskB = np.full(N, np.nan)
    pvA = {(st, tf): np.full(N, np.nan) for st in STATES for tf in ("fast", "slow")}
    pvB = {(st, tf): np.full(N, np.nan) for st in STATES for tf in ("fast", "slow")}
    for i, u in enumerate(uids):
        y = count_vector(trials, units[u], d).astype(float)
        senA[i] = win_hz(y, A(fb), sw) - win_hz(y, A(sb), sw)
        senB[i] = win_hz(y, B(fb), sw) - win_hz(y, B(sb), sw)
        tskA[i] = win_hz(y, A(engb), bw) - win_hz(y, A(disb), bw)
        tskB[i] = win_hz(y, B(engb), bw) - win_hz(y, B(disb), bw)
        for st in STATES:
            if stfA[st].size >= 5: pvA[(st, "fast")][i] = win_hz(y, stfA[st], sw)
            if stfB[st].size >= 5: pvB[(st, "fast")][i] = win_hz(y, stfB[st], sw)
            if stsA[st].size >= 5: pvA[(st, "slow")][i] = win_hz(y, stsA[st], sw)
            if stsB[st].size >= 5: pvB[(st, "slow")][i] = win_hz(y, stsB[st], sw)

    ok = np.isfinite(senA) & np.isfinite(senB) & np.isfinite(tskA) & np.isfinite(tskB)
    if ok.sum() < 6:
        return None

    def hat(v):
        vz = (v - v.mean()) / (v.std() + 1e-9)
        return vz / (np.linalg.norm(vz) + 1e-9)
    sA, sB = hat(senA[ok]), hat(senB[ok])
    tA, tB = hat(tskA[ok]), hat(tskB[ok])
    cosine = float(0.5 * (np.dot(sA, tB) + np.dot(sB, tA)))      # independent-halves
    rng = np.random.default_rng(SEED)
    null = np.array([0.5 * (np.dot(sA, tB[rng.permutation(len(tB))]) +
                            np.dot(sB, tA[rng.permutation(len(tA))])) for _ in range(500)])

    # held-out projections: fold-B condition-means onto fold-A CDs (and swap),
    # each centred by that fold's grand-mean activity (removes common-mode);
    # then z-score the six positions WITHIN session so scale is comparable
    # across sessions before pooling.
    def gm(pv):
        return np.nanmean(np.array([pv[(st, tf)][ok] for st in STATES for tf in ("fast", "slow")]), 0)
    gmA, gmB = gm(pvA), gm(pvB)
    raw = {}
    for st in STATES:
        for tf in ("fast", "slow"):
            b, a = pvB[(st, tf)][ok], pvA[(st, tf)][ok]
            if np.isfinite(b).all() and np.isfinite(a).all():
                x = 0.5 * (np.dot(b - gmB, sA) + np.dot(a - gmA, sB))
                yv = 0.5 * (np.dot(b - gmB, tA) + np.dot(a - gmA, tB))
                raw[(st, tf)] = (x, yv)
            else:
                raw[(st, tf)] = (np.nan, np.nan)
    xs = np.array([raw[k][0] for k in raw]); ys = np.array([raw[k][1] for k in raw])
    xm, xsd = np.nanmean(xs), np.nanstd(xs) + 1e-9
    ym, ysd = np.nanmean(ys), np.nanstd(ys) + 1e-9
    proj = {k: ((raw[k][0] - xm) / xsd, (raw[k][1] - ym) / ysd) for k in raw}
    sens_sep = {}
    for st in STATES:
        pf, ps = proj.get((st, "fast")), proj.get((st, "slow"))
        sens_sep[st] = (pf[0] - ps[0]) if (pf and np.isfinite(pf[0]) and np.isfinite(ps[0])) else np.nan
    return dict(subject=subj, session=session, n_units=int(ok.sum()), cosine=cosine,
                cosine_null_mean=float(null.mean()), cosine_null_sd=float(null.std()),
                proj={f"{k[0]}|{k[1]}": v for k, v in proj.items()},
                sens_sep=sens_sep, dprime=_dprime(s),
                sensory_load=sA.tolist(), taskstate_load=tB.tolist())


def _dprime(session):
    """Canonical SDT d' = z(hit_rate) - z(fa_rate) with hit_rate = SDT-hits /
    (SDT-hits + SDT-misses) — go trials with fa/ref/abort outcomes are EXCLUDED
    from the denominator (they are not valid SDT trials). Uses the project's
    canonical compute_session_performance so it matches the rest of the codebase
    (a naive hits/all-go denominator deflates d' ~2x)."""
    try:
        from visdetect.analysis.behavior import compute_session_performance
        d = compute_session_performance(session).get("d_prime")
        return float(d) if d is not None and np.isfinite(d) else np.nan
    except Exception:
        return np.nan


def _date_key(session):
    from visdetect.analysis import config
    try:
        return config.parse_session_date(session)
    except Exception:
        return session


def main(argv=None):
    from visdetect.viz.plotting import set_style
    try:
        set_style("talk")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    subj = a.subject
    out = Path(a.out_dir or f"E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/"
               f"tf_glm_bg046/population_geometry/{subj}")
    out.mkdir(parents=True, exist_ok=True)

    sessions = sorted(Path(p).stem for p in glob.glob(str(Path(PKL_DIR.format(subj=subj)) / "*.pkl")))
    tasks = [(subj, s) for s in sessions]
    print(f"{subj}: {len(tasks)} sessions", flush=True)
    res = []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(session_geometry, tasks):
            if r is not None:
                res.append(r); print(f"  {r['session']}: N={r['n_units']} cos={r['cosine']:+.3f} "
                                     f"d'={r['dprime']:.2f}", flush=True)
    if not res:
        raise SystemExit("no sessions produced geometry")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("proj", "sens_sep",
                        "sensory_load", "taskstate_load")} for r in res])
    df["date"] = df["session"].map(_date_key)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(out / "population_geometry.csv", index=False)

    # ---------- Fig 1: orthogonality ----------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    null_lo = np.median([r["cosine_null_mean"] - 2 * r["cosine_null_sd"] for r in res])
    null_hi = np.median([r["cosine_null_mean"] + 2 * r["cosine_null_sd"] for r in res])
    axA.axhspan(null_lo, null_hi, color="0.85", label="neuron-shuffle null (±2 SD)")
    axA.axhline(0, color="0.5", lw=0.8, ls=":")
    x = np.arange(len(df))
    axA.scatter(x, df["cosine"], c=df["dprime"], cmap="viridis", s=45, zorder=3)
    axA.plot(x, df["cosine"], color="0.6", lw=0.8, alpha=0.5, zorder=2)
    med = df["cosine"].median()
    axA.axhline(med, color="#d6322a", lw=1.5, label=f"median cosine = {med:+.3f}")
    axA.set_xlabel("session (chronological)"); axA.set_ylabel("cosine(Sensory CD, Task-state CD)")
    axA.set_title(f"{subj}: Sensory & Task-state axes are ~ORTHOGONAL\n"
                  f"(median cosine {med:+.3f}, {len(df)} sessions)")
    axA.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)
    # pooled per-unit loadings scatter
    sv = np.concatenate([r["sensory_load"] for r in res])
    tv = np.concatenate([r["taskstate_load"] for r in res])
    axB.scatter(sv, tv, s=6, alpha=0.25, color="#3474ae")
    axB.axhline(0, color="0.7", lw=0.8); axB.axvline(0, color="0.7", lw=0.8)
    rr = np.corrcoef(sv, tv)[0, 1]
    axB.set_xlabel("Sensory-CD loading (per unit)"); axB.set_ylabel("Task-state-CD loading")
    axB.set_title(f"per-unit loadings: uncorrelated (r={rr:+.2f})\n"
                  f"{len(sv)} units pooled")
    for sp in ("top", "right"): axB.spines[sp].set_visible(False)
    fig.suptitle(f"Population geometry — TF-sensory vs engagement axes ({subj})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig1_orthogonality.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---------- Fig 2: state-space (3 states x fast/slow), averaged across sessions ----------
    keys = [f"{st}|{tf}" for st in STATES for tf in ("fast", "slow")]
    agg = {k: np.array([r["proj"][k] for r in res if np.isfinite(r["proj"][k][0])]) for k in keys}
    fig2, ax = plt.subplots(figsize=(7.5, 6.5))
    rng = np.random.default_rng(SEED)
    for st in STATES:
        for tf, mk, fc in (("fast", "o", None), ("slow", "o", "white")):
            v = agg[f"{st}|{tf}"]
            if len(v) < 2:
                continue
            m = v.mean(0)
            boot = np.array([v[rng.integers(0, len(v), len(v))].mean(0) for _ in range(1000)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            ax.errorbar(m[0], m[1], xerr=[[m[0]-lo[0]], [hi[0]-m[0]]], yerr=[[m[1]-lo[1]], [hi[1]-m[1]]],
                        fmt=mk, ms=13, color=STATE_COLORS[st],
                        markerfacecolor=(fc or STATE_COLORS[st]), markeredgecolor=STATE_COLORS[st],
                        mew=2, capsize=3, elinewidth=1.2, zorder=3,
                        label=f"{st} {tf}")
        # connect fast->slow within state
        vf, vs = agg[f"{st}|fast"], agg[f"{st}|slow"]
        if len(vf) >= 2 and len(vs) >= 2:
            ax.plot([vf.mean(0)[0], vs.mean(0)[0]], [vf.mean(0)[1], vs.mean(0)[1]],
                    color=STATE_COLORS[st], lw=1.2, alpha=0.6, zorder=2)
    ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
    ax.set_xlabel("Sensory-input axis (z)   —   fast > slow TF")
    ax.set_ylabel("Task-state axis (z)   —   engaged > disengaged")
    ax.set_title(f"{subj}: state-space (mean ± 95% CI across {len(res)} sessions)\n"
                 "fast/slow separate on sensory axis; states displace on task-state axis")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="best")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(out / "fig2_state_space.png", dpi=150, bbox_inches="tight"); plt.close(fig2)

    # ---------- Fig 3: across learning ----------
    # x-axis = EXPERIENCE (chronological session order); d' is its own panel +
    # the colour, so performance (incl. late off-days) is decoupled from the
    # development axis (no double-encoding of "learning").
    df["sens_sep_mean"] = [np.nanmean([r["sens_sep"][st] for st in STATES]) for r in res]
    fig3, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.8))
    sc = a1.scatter(x, df["cosine"], c=df["dprime"], cmap="viridis", s=48, zorder=3)
    a1.axhspan(null_lo, null_hi, color="0.9", zorder=0, label="shuffle null")
    a1.axhline(0, color="0.6", lw=0.8, ls=":")
    a1.set_xlabel("session (experience →)"); a1.set_ylabel("cosine(Sensory, Task-state)")
    a1.set_title("Orthogonality holds across learning"); a1.legend(frameon=False, fontsize=8)
    a2.plot(x, df["dprime"], "-o", color="0.35", ms=5, lw=1.2)
    a2.axhline(1.0, color="#d6322a", lw=1, ls="--", alpha=0.6, label="d′=1")
    a2.set_xlabel("session (experience →)"); a2.set_ylabel("behavioural d′")
    a2.set_title("Learning curve (performance; off-days visible)")
    a2.legend(frameon=False, fontsize=8)
    a3.scatter(x, df["sens_sep_mean"], c=df["dprime"], cmap="viridis", s=48)
    a3.axhline(0, color="0.7", lw=0.8, ls=":")
    a3.set_xlabel("session (experience →)"); a3.set_ylabel("sensory fast−slow separation (z)")
    a3.set_title("TF (sensory) coding vs learning")
    fig3.colorbar(sc, ax=a3, label="d′ (performance)")
    for ax in (a1, a2, a3):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig3.suptitle(f"{subj}: population geometry across learning "
                  f"(experience axis; d′ decoupled)", fontsize=13)
    fig3.tight_layout(rect=(0, 0, 1, 0.95))
    fig3.savefig(out / "fig3_across_learning.png", dpi=150, bbox_inches="tight"); plt.close(fig3)

    print(f"\nwrote {out}/  (fig1_orthogonality, fig2_state_space, fig3_across_learning)")
    print(f"  median cosine = {df['cosine'].median():+.3f} (null median {np.median([r['cosine_null_mean'] for r in res]):+.3f})")
    print(f"  sessions with |cosine| < 0.2: {int((df['cosine'].abs()<0.2).sum())}/{len(df)}")


if __name__ == "__main__":
    main()

