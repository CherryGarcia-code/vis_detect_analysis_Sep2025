"""Descriptive first-look: is TF encoding STATE-GATED? Split the fast-minus-slow
TF-pulse response by behavioural state (engaged = StimSens+Impulsive vs
Disengaged) for responsive + borderline cells, WITHOUT re-fitting the GLM.

Hypothesis (user): a cell that encodes baseline TF only while engaged has its
session-wide C1/C2 diluted by disengaged trials and can fall below threshold.
If so, the fast-slow pulse response should SHARPEN in engaged trials and FLATTEN
in disengaged.

Per session: build the design, get fast/slow pulse times, tag each pulse with
its trial's state (trial_idx join VERIFIED to align with the design's 0-based
trial ordinal), compute the fast-slow pulse-PETH separately for engaged vs
disengaged pulses. Amplitude = mean post-pulse (0.05-0.6 s) of the
baseline-subtracted fast-slow trace; `amp_eng_matched` subsamples engaged pulses
to the disengaged count (power control). Population traces get 95% bootstrap CI
bands (over cells). Parallel across sessions.

NOT the rigorous test (that re-fits the GLM engaged-only); this is the cheap look.

Usage:
  py state_split_pulse_peth.py --subject BG_046 --n-resp 60 --n-border 120 --workers 12
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import argparse
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
REGISTRY = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/data/cache/tf_responsive/{low}_tf_responsive.csv"
ENGAGED = {"StimSens", "Impulsive"}
DISENGAGED = {"Disengaged"}
POST_WIN = (0.05, 0.6)
SEED = 42


def _cfg():
    return TFGLMConfig(include_movement=False, include_phase=False, include_tiled_baseline=True,
                       standardize_design=True, fast_fit=True, tf_encoding="log2",
                       min_pulses_per_label=20)


def select_cells(subj, n_resp, n_border, border_lo=0.13, border_hi=0.20):
    reg = pd.read_csv(REGISTRY.format(low=subj.lower().replace("_", "")), dtype={"session": str})
    reg["resp_log2"] = reg["resp_log2"].astype(str).str.lower().isin(["true", "1", "1.0"])
    resp = reg[reg.resp_log2].sort_values("c1_r_log2", ascending=False).head(n_resp).copy()
    resp["kind"] = "responsive"
    border = reg[(~reg.resp_log2) & (reg.c1_r_log2.between(border_lo, border_hi))] \
        .sort_values("c1_r_log2", ascending=False).head(n_border).copy()
    border["kind"] = "borderline"
    return pd.concat([resp, border], ignore_index=True)


def state_by_trial(subj, session):
    date = session.replace(f"{subj}_", "", 1)
    f = Path(STATE_DIR.format(subj=subj)) / f"{date}.csv"
    if not f.exists():
        return None
    st = pd.read_csv(f)
    return st.set_index("trial_idx")["state_label"].to_dict()


def pulse_states(ptimes, edges, ti, lab):
    idx = np.searchsorted(edges, ptimes, side="right") - 1
    ok = (idx >= 0) & (idx < ti.size)
    out = np.full(ptimes.shape, "UNTAGGED", dtype=object)
    tr = ti[idx[ok]]
    out[ok] = [lab.get(int(t), "UNTAGGED") for t in tr]
    return out


def _amp(fast, slow, t, bs):
    d = (fast - slow) / bs
    d = d - (np.nanmean(d[t < 0]) if (t < 0).any() else 0.0)
    m = (t >= POST_WIN[0]) & (t <= POST_WIN[1])
    return float(np.nanmean(d[m])), d


def process_session(args):
    subj, session, unit_ids = args
    cfg = _cfg()
    s = load_session(str(Path(PKL_DIR.format(subj=subj)) / f"{session}.pkl"))
    lab = state_by_trial(subj, session)
    if lab is None:
        return [], {}, None
    trials, units = session_trial_regressors(s, cfg)
    d = assemble_design(trials, cfg)
    ti, edges = d.trial_index, d.bin_edges
    win, bs = cfg.pulse_eval_win, cfg.bin_s
    fast, slow = pulse_times_from_tf(d, cfg)
    fs, ss = pulse_states(fast, edges, ti, lab), pulse_states(slow, edges, ti, lab)
    eng_f = fast[np.isin(fs, list(ENGAGED))]; dis_f = fast[np.isin(fs, list(DISENGAGED))]
    eng_s = slow[np.isin(ss, list(ENGAGED))]; dis_s = slow[np.isin(ss, list(DISENGAGED))]
    if len(dis_f) < 20 or len(eng_f) < 20:      # session lacks a state -> skip (can't split)
        return [], {}, None
    rng = np.random.default_rng(SEED)
    m_f, m_s = min(len(eng_f), len(dis_f)), min(len(eng_s), len(dis_s))
    mf = eng_f[rng.choice(len(eng_f), m_f, replace=False)] if len(eng_f) > m_f else eng_f
    ms = eng_s[rng.choice(len(eng_s), m_s, replace=False)] if len(eng_s) > m_s else eng_s

    def peth(y, p):
        _, h = tf_pulse_peth(y, edges, p, win, bs, trial_index=ti)
        return h
    rows, traces, t_axis = [], {}, None
    for uid in unit_ids:
        if uid not in units:
            continue
        y = count_vector(trials, units[uid], d)
        t_axis, ef = tf_pulse_peth(y, edges, eng_f, win, bs, trial_index=ti)
        es, df_, ds = peth(y, eng_s), peth(y, dis_f), peth(y, dis_s)
        mef, mes = peth(y, mf), peth(y, ms)
        amp_e, tr_e = _amp(ef, es, t_axis, bs)
        amp_d, tr_d = _amp(df_, ds, t_axis, bs)
        amp_em, _ = _amp(mef, mes, t_axis, bs)
        rows.append(dict(subject=subj, session=session, unit=int(uid),
                         amp_engaged=amp_e, amp_disengaged=amp_d, amp_eng_matched=amp_em,
                         n_eng_pulses=len(eng_f), n_dis_pulses=len(dis_f)))
        traces[(session, int(uid))] = (tr_e, tr_d)
    return rows, traces, t_axis


def boot_ci(traces, n=1000, seed=SEED):
    """(mean, lo, hi) per lag, bootstrapping over cells (rows)."""
    m = np.nanmean(traces, 0)
    if len(traces) < 3:
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(traces), size=(n, len(traces)))
    boots = np.stack([np.nanmean(traces[ix], 0) for ix in idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return m, lo, hi


def paired_ci(a, b, n=1000, seed=SEED):
    """95% bootstrap CI on median(a-b), paired (drop NaN pairs)."""
    d = (np.asarray(a) - np.asarray(b))
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return float(np.median(d)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--n-resp", type=int, default=60)
    ap.add_argument("--n-border", type=int, default=120)
    ap.add_argument("--max-sessions", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    subj = a.subject
    out = Path(a.out_dir or f"E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/"
               f"tf_glm_bg046/state_conditioned/{subj}")
    out.mkdir(parents=True, exist_ok=True)

    sel = select_cells(subj, a.n_resp, a.n_border)
    counts = sel.groupby("session").size().sort_values(ascending=False)
    sel = sel[sel.session.isin(list(counts.head(a.max_sessions).index))]
    print(f"{subj}: {len(sel)} candidate cells / {sel.session.nunique()} sessions "
          f"({int((sel.kind=='responsive').sum())} resp, {int((sel.kind=='borderline').sum())} border)",
          flush=True)

    tasks = [(subj, session, [int(u) for u in g.unit]) for session, g in sel.groupby("session")]
    all_rows, all_traces, t_axis = [], {}, None
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rows, traces, t in ex.map(process_session, tasks):
            all_rows += rows; all_traces.update(traces)
            if t is not None:
                t_axis = t
    df = pd.DataFrame(all_rows)
    kind = sel.set_index(["session", "unit"])["kind"].to_dict()
    c1 = sel.set_index(["session", "unit"])["c1_r_log2"].to_dict()
    df["kind"] = [kind.get((r.session, r.unit)) for r in df.itertuples()]
    df["c1_session"] = [c1.get((r.session, r.unit)) for r in df.itertuples()]
    df.to_csv(out / "state_split_amplitudes.csv", index=False)

    # ---- population with 95% bootstrap CI bands ----
    fig, (axP, axS) = plt.subplots(1, 2, figsize=(13.5, 5))
    for kd, col in [("responsive", "#d6322a"), ("borderline", "#5aa469")]:
        keys = [(r.session, r.unit) for r in df[df.kind == kd].itertuples()
                if (r.session, r.unit) in all_traces]
        if not keys:
            continue
        eng = np.vstack([all_traces[k][0] for k in keys])
        dis = np.vstack([all_traces[k][1] for k in keys])
        me, le, he = boot_ci(eng); md, ld, hd = boot_ci(dis)
        axP.plot(t_axis, me, color=col, lw=2, label=f"{kd} engaged (n={len(eng)})")
        axP.fill_between(t_axis, le, he, color=col, alpha=0.22)
        axP.plot(t_axis, md, color=col, lw=1.5, ls="--", alpha=0.9, label=f"{kd} disengaged")
        axP.fill_between(t_axis, ld, hd, color=col, alpha=0.10)
    axP.axvline(0, color="0.7", lw=0.8); axP.axhline(0, color="0.7", lw=0.8)
    axP.set_xlabel("time from TF pulse (s)"); axP.set_ylabel("Δ firing, fast−slow (Hz)")
    axP.set_title(f"{subj}: fast−slow pulse response, engaged vs disengaged\n(mean ± 95% bootstrap CI over cells)")
    axP.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"): axP.spines[sp].set_visible(False)

    for kd, col in [("responsive", "#d6322a"), ("borderline", "#5aa469")]:
        d = df[df.kind == kd]
        axS.scatter(d.amp_disengaged, d.amp_engaged, s=20, color=col, alpha=0.7, label=kd)
    v = df[["amp_engaged", "amp_disengaged"]].to_numpy()
    lim = np.nanmax(np.abs(v[np.isfinite(v)])) * 1.1 if np.isfinite(v).any() else 1
    axS.plot([-lim, lim], [-lim, lim], color="0.6", lw=1, ls=":")
    axS.axhline(0, color="0.85", lw=0.8); axS.axvline(0, color="0.85", lw=0.8)
    axS.set_xlim(-lim, lim); axS.set_ylim(-lim, lim)
    axS.set_xlabel("disengaged fast−slow amplitude (Hz)")
    axS.set_ylabel("engaged fast−slow amplitude (Hz)")
    valid = df.dropna(subset=["amp_engaged", "amp_disengaged"])
    n_gated = int((valid.amp_engaged > valid.amp_disengaged).sum())
    axS.set_title(f"per-cell amplitude (n={len(valid)} with both states)\n"
                  f"{n_gated}/{len(valid)} engaged > disengaged")
    axS.legend(frameon=False)
    for sp in ("top", "right"): axS.spines[sp].set_visible(False)
    fig.suptitle(f"State-split TF encoding — descriptive ({subj}, "
                 f"{df.session.nunique()} sessions, {len(df)} cells)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "state_split_population.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- exemplars: biggest engaged>disengaged gap ----
    df["gap"] = df.amp_engaged - df.amp_disengaged
    top = df.dropna(subset=["gap"]).sort_values("gap", ascending=False).head(12)
    figE, axes = plt.subplots(3, 4, figsize=(16, 8), squeeze=False)
    for i, r in enumerate(top.itertuples()):
        ax = axes[i//4][i%4]; tr_e, tr_d = all_traces[(r.session, r.unit)]
        ax.axvline(0, color="0.8", lw=0.8)
        ax.plot(t_axis, tr_e, color="#1a7f37", lw=2, label="engaged")
        ax.plot(t_axis, tr_d, color="0.5", lw=1.5, ls="--", label="disengaged")
        ax.set_title(f"{r.session[-8:]} u{r.unit} [{r.kind[:4]}]\n"
                     f"eng={r.amp_engaged:+.2f} dis={r.amp_disengaged:+.2f}", fontsize=8)
        if i == 0: ax.legend(fontsize=7, frameon=False)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    figE.suptitle(f"{subj}: cells with the largest engaged>disengaged TF gap", fontsize=12)
    figE.tight_layout(rect=(0, 0, 1, 0.96))
    figE.savefig(out / "state_split_exemplars.png", dpi=140, bbox_inches="tight"); plt.close(figE)

    # ---- stats ----
    print(f"\nwrote {out}/")
    for kd in ("responsive", "borderline"):
        d = df[df.kind == kd]
        med, lo, hi = paired_ci(d.amp_engaged, d.amp_disengaged)
        medm, lom, him = paired_ci(d.amp_engaged, d.amp_eng_matched)
        print(f"  {kd} (n={len(d)}): eng={d.amp_engaged.median():+.3f} "
              f"dis={d.amp_disengaged.median():+.3f} | "
              f"med(eng-dis)={med:+.3f} [95% {lo:+.3f},{hi:+.3f}] "
              f"{'*' if (lo>0 or hi<0) else 'ns'}")


if __name__ == "__main__":
    main()
