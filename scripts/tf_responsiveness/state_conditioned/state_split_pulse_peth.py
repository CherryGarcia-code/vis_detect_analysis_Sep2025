"""Descriptive first-look: is TF encoding STATE-GATED? Split the fast-minus-slow
TF-pulse response by behavioural state (engaged = StimSens+Impulsive vs
Disengaged) for responsive + borderline cells, WITHOUT re-fitting the GLM.

Hypothesis (user): a cell that encodes baseline TF only while engaged has its
session-wide C1/C2 diluted by disengaged trials and can fall below threshold.
If so, the fast-slow pulse response should SHARPEN in engaged trials and FLATTEN
in disengaged.

Selects positive-control (responsive) + borderline (C1~0.13-0.20, non-responsive)
cells from the registry, groups by session, and per session: builds the design,
gets fast/slow pulse times, tags each pulse with its trial's state (trial_idx
join VERIFIED to align with the design's 0-based trial ordinal), and computes the
fast-slow pulse-PETH separately for engaged vs disengaged pulses. Amplitude =
mean post-pulse (0.05-0.6 s) of the baseline-subtracted fast-slow trace;
`amp_eng_matched` subsamples engaged pulses to the disengaged count (power
control). Outputs exemplars + population (engaged vs disengaged) + a per-cell
amplitude scatter + CSV.

NOT the rigorous test (that re-fits the GLM engaged-only); this is the cheap look
to see if the signature exists before investing in re-fits.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import argparse
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
POST_WIN = (0.05, 0.6)      # post-pulse window for the amplitude metric
SEED = 42


def _cfg():
    return TFGLMConfig(include_movement=False, include_phase=False, include_tiled_baseline=True,
                       standardize_design=True, fast_fit=True, tf_encoding="log2",
                       min_pulses_per_label=20)


def select_cells(subj, n_resp=25, n_border=45, border_lo=0.13, border_hi=0.20):
    reg = pd.read_csv(REGISTRY.format(low=subj.lower().replace("_", "")), dtype={"session": str})
    reg["resp_log2"] = reg["resp_log2"].astype(str).str.lower().isin(["true", "1", "1.0"])
    resp = reg[reg.resp_log2].sort_values("c1_r_log2", ascending=False).head(n_resp).copy()
    resp["kind"] = "responsive"
    border = reg[(~reg.resp_log2) & (reg.c1_r_log2.between(border_lo, border_hi))] \
        .sort_values("c1_r_log2", ascending=False).head(n_border).copy()
    border["kind"] = "borderline"
    sel = pd.concat([resp, border], ignore_index=True)
    return sel


def state_by_trial(subj, session):
    """{trial_ordinal:int -> state_label}. state_tags file is named by the bare
    date (session minus the 'BG_0xx_' prefix)."""
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
    """post-pulse mean of baseline-subtracted (fast-slow), in Hz."""
    d = (fast - slow) / bs
    d = d - (np.nanmean(d[t < 0]) if (t < 0).any() else 0.0)
    m = (t >= POST_WIN[0]) & (t <= POST_WIN[1])
    return float(np.nanmean(d[m])), d  # amplitude, full baseline-sub trace (Hz)


def process_session(subj, session, unit_ids, verbose=True):
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
    rng = np.random.default_rng(SEED)
    # power control: engaged subsampled to disengaged counts
    m_f = min(len(eng_f), len(dis_f)); m_s = min(len(eng_s), len(dis_s))
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
        es = peth(y, eng_s); df = peth(y, dis_f); ds = peth(y, dis_s)
        mef = peth(y, mf); mes = peth(y, ms)
        amp_e, tr_e = _amp(ef, es, t_axis, bs)
        amp_d, tr_d = _amp(df, ds, t_axis, bs)
        amp_em, _ = _amp(mef, mes, t_axis, bs)
        rows.append(dict(subject=subj, session=session, unit=int(uid),
                         amp_engaged=amp_e, amp_disengaged=amp_d, amp_eng_matched=amp_em,
                         n_eng_pulses=len(eng_f), n_dis_pulses=len(dis_f)))
        traces[(session, int(uid))] = (tr_e, tr_d)
        if verbose:
            print(f"  [{session}] u{uid}: eng_amp={amp_e:+.3f} dis_amp={amp_d:+.3f} "
                  f"(matched {amp_em:+.3f}) Hz", flush=True)
    return rows, traces, t_axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_031")
    ap.add_argument("--max-sessions", type=int, default=10, help="top sessions by candidate count")
    ap.add_argument("--out-dir", default="E:/python_analysis/git_repos/vd_tf_bg046/"
                    "FIGURES/tf_glm_bg046/state_conditioned/BG_031")
    a = ap.parse_args()
    subj = a.subject
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    sel = select_cells(subj)
    counts = sel.groupby("session").size().sort_values(ascending=False)
    keep_sessions = list(counts.head(a.max_sessions).index)
    sel = sel[sel.session.isin(keep_sessions)]
    print(f"{subj}: {len(sel)} candidate cells across {sel.session.nunique()} sessions "
          f"({int((sel.kind=='responsive').sum())} responsive, "
          f"{int((sel.kind=='borderline').sum())} borderline)")

    all_rows, all_traces, t_axis = [], {}, None
    for session, g in sel.groupby("session"):
        print(f"session {session} ({len(g)} cells)...", flush=True)
        rows, traces, t = process_session(subj, session, [int(u) for u in g.unit])
        all_rows += rows; all_traces.update(traces)
        if t is not None:
            t_axis = t
    df = pd.DataFrame(all_rows)
    kind = sel.set_index(["session", "unit"])["kind"].to_dict()
    c1 = sel.set_index(["session", "unit"])["c1_r_log2"].to_dict()
    df["kind"] = [kind.get((r.session, r.unit)) for r in df.itertuples()]
    df["c1_session"] = [c1.get((r.session, r.unit)) for r in df.itertuples()]
    df.to_csv(out / "state_split_amplitudes.csv", index=False)

    # ---- population + scatter ----
    fig, (axP, axS) = plt.subplots(1, 2, figsize=(13, 5))
    for kd, col in [("responsive", "#d6322a"), ("borderline", "#5aa469")]:
        keys = [(r.session, r.unit) for r in df[df.kind == kd].itertuples() if (r.session, r.unit) in all_traces]
        eng = np.vstack([all_traces[k][0] for k in keys]) if keys else np.zeros((0, t_axis.size))
        dis = np.vstack([all_traces[k][1] for k in keys]) if keys else np.zeros((0, t_axis.size))
        if len(eng):
            axP.plot(t_axis, np.nanmean(eng, 0), color=col, lw=2, label=f"{kd} engaged (n={len(eng)})")
            axP.plot(t_axis, np.nanmean(dis, 0), color=col, lw=1.5, ls="--", alpha=0.7,
                     label=f"{kd} disengaged")
    axP.axvline(0, color="0.7", lw=0.8); axP.axhline(0, color="0.7", lw=0.8)
    axP.set_xlabel("time from TF pulse (s)"); axP.set_ylabel("Δ firing, fast−slow (Hz)")
    axP.set_title(f"{subj}: fast−slow pulse response, engaged vs disengaged")
    axP.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"): axP.spines[sp].set_visible(False)

    for kd, col in [("responsive", "#d6322a"), ("borderline", "#5aa469")]:
        d = df[df.kind == kd]
        axS.scatter(d.amp_disengaged, d.amp_engaged, s=22, color=col, alpha=0.75, label=kd)
    lim = np.nanmax(np.abs([df.amp_engaged.values, df.amp_disengaged.values])) * 1.1
    axS.plot([-lim, lim], [-lim, lim], color="0.6", lw=1, ls=":")
    axS.axhline(0, color="0.85", lw=0.8); axS.axvline(0, color="0.85", lw=0.8)
    axS.set_xlim(-lim, lim); axS.set_ylim(-lim, lim)
    axS.set_xlabel("disengaged fast−slow amplitude (Hz)")
    axS.set_ylabel("engaged fast−slow amplitude (Hz)")
    n_gated = int((df.amp_engaged > df.amp_disengaged).sum())
    axS.set_title(f"per-cell amplitude: engaged vs disengaged\n"
                  f"{n_gated}/{len(df)} above diagonal (engaged > disengaged)")
    axS.legend(frameon=False);
    for sp in ("top", "right"): axS.spines[sp].set_visible(False)
    fig.suptitle(f"State-gated TF encoding — descriptive ({subj}, {df.session.nunique()} sessions, "
                 f"{len(df)} cells)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "state_split_population.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- exemplar grid: cells with biggest engaged>disengaged gap ----
    df["gap"] = df.amp_engaged - df.amp_disengaged
    top = df.sort_values("gap", ascending=False).head(12)
    nc = 4; nr = 3
    figE, axes = plt.subplots(nr, nc, figsize=(4*nc, 2.6*nr), squeeze=False)
    for i, r in enumerate(top.itertuples()):
        ax = axes[i//nc][i%nc]; tr_e, tr_d = all_traces[(r.session, r.unit)]
        ax.axvline(0, color="0.8", lw=0.8)
        ax.plot(t_axis, tr_e, color="#1a7f37", lw=2, label="engaged")
        ax.plot(t_axis, tr_d, color="0.5", lw=1.5, ls="--", label="disengaged")
        ax.set_title(f"{r.session[-8:]} u{r.unit} [{r.kind[:4]}]\n"
                     f"eng={r.amp_engaged:+.2f} dis={r.amp_disengaged:+.2f}", fontsize=8)
        if i == 0: ax.legend(fontsize=7, frameon=False)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    figE.suptitle(f"{subj}: cells with the largest engaged>disengaged TF gap "
                  f"(fast−slow pulse response, Hz)", fontsize=12)
    figE.tight_layout(rect=(0, 0, 1, 0.96))
    figE.savefig(out / "state_split_exemplars.png", dpi=140, bbox_inches="tight"); plt.close(figE)

    print(f"\nwrote {out}/  (population, exemplars, amplitudes.csv)")
    print(f"  engaged>disengaged: {n_gated}/{len(df)} cells "
          f"({100*n_gated/len(df):.0f}%)")
    print(f"  median engaged amp={df.amp_engaged.median():+.3f} vs "
          f"disengaged={df.amp_disengaged.median():+.3f} vs "
          f"eng-count-matched={df.amp_eng_matched.median():+.3f} Hz")
    print(f"  by kind:\n{df.groupby('kind')[['amp_engaged','amp_disengaged','amp_eng_matched']].median()}")


if __name__ == "__main__":
    main()
