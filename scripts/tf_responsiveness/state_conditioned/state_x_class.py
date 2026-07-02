"""2a — does the transient/sustained (kernel-width) axis interact with behavioural
STATE, where the undifferentiated TF population showed no state effect?

Two tests, per TF-responsive cell (good_dates; class from transient_vs_sustained):

  (i) TASK-STATE CD LOADING by class (robust; baseline firing, so every trial
      contributes incl. Disengaged). Reuses population_geometry's EXACT definition:
      task_load = engaged − Disengaged pre-pulse baseline firing (BASELINE_WIN
      −0.4..0 s; engaged = StimSens+Impulsive), sens_load = fast − slow pulse
      (SENSORY_WIN 0.122..0.167 s). Prediction: sustained cells carry a larger
      |task-state loading| (the orthogonal engagement offset is an integrator-
      population phenomenon) — OR they don't (offset is population-wide/tonic,
      independent of the functional axis). Either is a clean result.

 (ii) STATE-SPLIT outcome ramps by class (coverage-limited): change response
      (Change_ON hit, canonical 0..0.25 vs −0.4..−0.05) and FA motor ramp
      (−0.3..−0.15 vs −1.75..−1.25) computed SEPARATELY per state. Prediction:
      sustained cells' decision/motor ramps are state-modulated; transient
      cells' (sensory) responses are not. Circularity caveat: states are
      behaviourally defined ([[state_labeler_circularity_caveat]]) — so the
      informative contrast is the NEURAL magnitude per matched event, and the
      class×state INTERACTION, not the state main effect.

One session load each; cache to CSV.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO, _spikes, load_session, get_event_times_by_trial  # noqa: E402
from transient_vs_sustained import load_cells, TCOL, SCOL                     # noqa: E402
from population_geometry import (_cfg, _state_by_trial, _pulse_states,        # noqa: E402
                                 ENGAGED, STATES, STATE_COLORS, SENSORY_WIN, BASELINE_WIN)
from visdetect.analysis.tf_glm import (assemble_design, count_vector,         # noqa: E402
                                       pulse_times_from_tf)
from visdetect.analysis.tf_glm_data import session_trial_regressors           # noqa: E402
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS          # noqa: E402

CH_BASE, CH_RESP = EVENT_RESPONSIVENESS_WINDOWS["Change_ON"]
RA_BASE, RA_RESP = EVENT_RESPONSIVENESS_WINDOWS["FA"]
MIN_EV = 5
OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/state_x_class")
CACHE = OUT / "state_x_class_metrics.csv"


def _win_rate(spk, times, win):
    if len(times) < MIN_EV:
        return np.nan
    times = np.asarray(times, float)
    lo = np.searchsorted(spk, times + win[0]); hi = np.searchsorted(spk, times + win[1])
    return float(((hi - lo) / (win[1] - win[0])).mean())


def _delta(spk, times, base, resp):
    if len(times) < MIN_EV:
        return np.nan
    return _win_rate(spk, times, resp) - _win_rate(spk, times, base)


def session_metrics(subj, sess, gcells):
    lab = _state_by_trial(subj, sess)
    if lab is None:
        return []
    s = load_session(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
    cfg = _cfg()
    trials, units = session_trial_regressors(s, cfg)
    d = assemble_design(trials, cfg)
    edges, bs, ti = d.bin_edges, cfg.bin_s, d.trial_index
    fast, slow = pulse_times_from_tf(d, cfg)
    fs, ss = _pulse_states(fast, edges, ti, lab), _pulse_states(slow, edges, ti, lab)

    def sub(p, st, want):
        return p[np.isin(st, want)]

    def pbins(p):
        return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 1)
    sw = np.arange(int(round(SENSORY_WIN[0] / bs)), int(round(SENSORY_WIN[1] / bs)) + 1)
    bw = np.arange(int(round(BASELINE_WIN[0] / bs)), int(round(BASELINE_WIN[1] / bs)))
    fb, sb = pbins(fast), pbins(slow)
    engb = pbins(np.concatenate([sub(fast, fs, list(ENGAGED)), sub(slow, ss, list(ENGAGED))]))
    disb = pbins(np.concatenate([sub(fast, fs, ["Disengaged"]), sub(slow, ss, ["Disengaged"])]))

    def win_hz(y, pb, offs):
        if pb.size == 0:
            return np.nan
        idx = pb[:, None] + offs[None, :]
        val = (idx >= 0) & (idx < y.size)
        g = np.where(val, y[np.clip(idx, 0, y.size - 1)], 0.0)
        return float(g.sum() / (val.sum() * bs)) if val.sum() else np.nan

    # per-trial event times + outcomes + state
    et_ch = np.asarray(get_event_times_by_trial(s, "Change_ON"), float)
    et_fa = np.asarray(get_event_times_by_trial(s, "FA"), float)
    outc = [str(getattr(t, "trialoutcome", "") or "").lower() for t in s.trials]
    st_of = {i: lab.get(i, "NA") for i in range(len(s.trials))}

    def times_for(et, want_outcome, state):
        return [et[i] for i in range(len(s.trials))
                if i < et.size and np.isfinite(et[i]) and outc[i] == want_outcome
                and st_of[i] == state]

    rows = []
    for _, r in gcells.iterrows():
        uid = int(r["unit"])
        if uid not in units:
            continue
        y = count_vector(trials, units[uid], d).astype(float)
        task_load = win_hz(y, engb, bw) - win_hz(y, disb, bw)
        sens_load = win_hz(y, fb, sw) - win_hz(y, sb, sw)
        spk = np.sort(_spikes(s, uid))
        rec = dict(subject=subj, session=sess, unit=uid, cls=r["class"],
                   kernel_fwhm=float(r["kernel_fwhm"]), task_load=task_load, sens_load=sens_load)
        for stt in STATES:
            rec[f"change_{stt}"] = _delta(spk, times_for(et_ch, "hit", stt), CH_BASE, CH_RESP)
            rec[f"fa_{stt}"] = _delta(spk, times_for(et_fa, "fa", stt), RA_BASE, RA_RESP)
        rows.append(rec)
    del s
    gc.collect()
    return rows


def compute_or_load(force=False):
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE)
    cells = load_cells()
    cells = cells[cells["class"].isin(["transient", "sustained"])]
    allrows = []
    for (subj, sess), g in cells.groupby(["subject", "session"]):
        if not Path(f"{REPO}/data/pkls/{subj}/{sess}.pkl").exists():
            continue
        allrows += session_metrics(subj, sess, g)
        print(f"  {subj}/{sess}: {len(g)} cells", flush=True)
    df = pd.DataFrame(allrows)
    # per-session z of task_load across responsive cells (comparable magnitude across sessions)
    df["task_load_z"] = df.groupby("session")["task_load"].transform(
        lambda v: (v - v.mean()) / (v.std() + 1e-9))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def _mwu(a, b):
    a = pd.Series(a).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(b).replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, len(a), len(b), np.nan
    return float(a.median()), float(b.median()), len(a), len(b), float(mannwhitneyu(a, b).pvalue)


def _mixed_p(df, col):
    """p for the class effect on `col` with a session random intercept + region
    fixed effect — the same pseudoreplication control used for the width claim."""
    d = df[df["cls"].isin(["transient", "sustained"])].copy()
    d = d.dropna(subset=[col]).replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
    d["is_sus"] = (d["cls"] == "sustained").astype(float)
    try:
        import warnings
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(f"{col} ~ is_sus + C(region)", d, groups=d["session"]).fit(method="lbfgs")
        return float(m.pvalues.get("is_sus", np.nan))
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    df = compute_or_load(force=a.force)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})
    lines = []
    df["region"] = df["subject"].map({"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"})
    tr, su = df[df.cls == "transient"], df[df.cls == "sustained"]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)

    # (i) task-state loading by class — FR-NORMALIZED (per-session z).
    # CRITICAL FIX (verification): raw-Hz |task_load| is FIRING-RATE-CONFOUNDED
    # (sustained cells fire faster) — an invalid cross-neuron comparison
    # (CLAUDE.md golden rule). The honest test uses task_load_z (per-session z
    # across the responsive pop). It is NULL. The earlier raw-Hz 3.65-vs-2.24
    # "p=4.9e-3" headline is RETRACTED.
    axa = fig.add_subplot(gs[0, 0])
    trz = tr["task_load_z"].abs().replace([np.inf, -np.inf], np.nan).dropna()
    suz = su["task_load_z"].abs().replace([np.inf, -np.inf], np.nan).dropna()
    for si, v in enumerate([trz, suz]):
        jit = (np.random.default_rng(si).random(len(v)) - 0.5) * 0.28
        axa.scatter(np.full(len(v), si) + jit, v, s=10, alpha=0.4,
                    color=(TCOL if si == 0 else SCOL), edgecolors="none")
        axa.hlines(np.median(v), si - 0.25, si + 0.25, color="k", lw=2.3, zorder=5)
    mtz, msz, ntz, nsz, pz = _mwu(tr["task_load_z"].abs(), su["task_load_z"].abs())
    axa.text(0.5, 0.97, f"FR-normalized (per-session z)\ntransient {mtz:.2f} vs sustained {msz:.2f}\nMWU p={pz:.2f}  —  NS",
             transform=axa.transAxes, ha="center", va="top", fontsize=9)
    axa.set_xticks([0, 1]); axa.set_xticklabels(["transient", "sustained"], fontsize=10)
    axa.set_ylabel("|task-state loading|  (z, FR-normalized)")
    axa.set_ylim(0, np.nanpercentile(pd.concat([trz, suz]), 97))
    axa.set_title("(i) task-state loading — FR-normalized = NULL", fontsize=10.5)

    # honest stats: raw (confounded) vs normalized, per-subject, per-region, mixed model
    mtr, msr, ntr_, nsr, praw = _mwu(tr.task_load.abs(), su.task_load.abs())
    lines.append(f"[task_load RAW-Hz POOLED] t={mtr:.3f} s={msr:.3f} MWU p={praw:.2e}  <-- FR-CONFOUNDED "
                 f"(raw Hz scales with firing rate; sustained fire faster); RETRACTED, not a valid cross-neuron test")
    lines.append(f"[task_load_z FR-NORM POOLED] t={mtz:.3f} s={msz:.3f} MWU p={pz:.3f}  <-- HONEST test = NULL")
    for subj in ["BG_046", "BG_039", "BG_031"]:
        a = df[(df.subject == subj) & (df.cls == "transient")]["task_load_z"].abs()
        b = df[(df.subject == subj) & (df.cls == "sustained")]["task_load_z"].abs()
        _, _, na, nb, ps = _mwu(a, b)
        lines.append(f"   z {subj}: t(n{na}) vs s(n{nb}) p={ps if ps == ps else float('nan'):.3f}")
    for reg in ["DMS", "VMS"]:
        a = df[(df.region == reg) & (df.cls == "transient")]["task_load_z"].abs()
        b = df[(df.region == reg) & (df.cls == "sustained")]["task_load_z"].abs()
        _, _, na, nb, ps = _mwu(a, b)
        lines.append(f"   z {reg}: t(n{na}) vs s(n{nb}) p={ps if ps == ps else float('nan'):.3f}")
    df["abs_task_load_z"] = df["task_load_z"].abs()
    lines.append(f"   |z| session-random-intercept mixedLM (is_sustained) p={_mixed_p(df, 'abs_task_load_z'):.3f}"
                 f"  [signed-z mixedLM p={_mixed_p(df, 'task_load_z'):.3f}, different question]")

    # (ii) change response by state x class
    def state_panel(ax, prefix, title):
        xs = np.arange(len(STATES)); w = 0.38
        for gi, (nm, d_, col) in enumerate([("transient", tr, TCOL), ("sustained", su, SCOL)]):
            meds, ns_ = [], []
            for stt in STATES:
                v = d_[f"{prefix}_{stt}"].replace([np.inf, -np.inf], np.nan).dropna()
                meds.append(v.median() if len(v) >= 5 else np.nan); ns_.append(len(v))
            ax.bar(xs + (gi - 0.5) * w, meds, w, color=col, label=nm, alpha=0.9)
            for x, m, n in zip(xs + (gi - 0.5) * w, meds, ns_):
                if np.isfinite(m):
                    ax.text(x, m, f"n{n}", ha="center", va="bottom", fontsize=6.5)
        # per-state transient vs sustained
        for si, stt in enumerate(STATES):
            _, _, _, _, ps = _mwu(tr[f"{prefix}_{stt}"], su[f"{prefix}_{stt}"])
            if ps == ps:
                ax.text(si, ax.get_ylim()[1] * 0.92, f"p={ps:.1e}", ha="center", fontsize=7)
                lines.append(f"[{prefix} {stt}] t vs s p={ps:.2e}")
        ax.axhline(0, color="0.7", lw=0.8, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels(STATES, fontsize=8, rotation=12)
        ax.set_ylabel("Δ firing (Hz)"); ax.set_title(title, fontsize=10.5)
        ax.legend(frameon=False, fontsize=8)

    state_panel(fig.add_subplot(gs[0, 1]), "change", "(ii) Change response × state")
    state_panel(fig.add_subplot(gs[0, 2]), "fa", "(ii) FA motor ramp × state")

    # within-class state modulation (paired across cells that have >=2 states)
    axm = fig.add_subplot(gs[1, 0])
    lines.append("--- within-class state modulation (sustained cells) ---")
    # engaged vs disengaged for change & fa within sustained (where both finite)
    for prefix in ("change", "fa"):
        for nm, d_ in [("sustained", su), ("transient", tr)]:
            eng = d_[[f"{prefix}_StimSens", f"{prefix}_Impulsive"]].mean(1)
            dis = d_[f"{prefix}_Disengaged"]
            both = pd.concat([eng, dis], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if len(both) >= 6:
                w = wilcoxon(both.iloc[:, 0], both.iloc[:, 1]).pvalue
                lines.append(f"[{prefix} {nm}] engaged {both.iloc[:,0].median():.2f} vs diseng "
                             f"{both.iloc[:,1].median():.2f} (n={len(both)}) Wilcoxon p={w:.2e}")
    axm.axis("off")
    axm.text(0.0, 1.0, "\n".join(lines), transform=axm.transAxes, va="top", ha="left",
             fontsize=7.2, family="monospace")

    # sens_load sanity by class (sustained should have >= sensory loading too)
    axs = fig.add_subplot(gs[1, 1])
    for si, (nm, d_, col) in enumerate([("transient", tr, TCOL), ("sustained", su, SCOL)]):
        v = d_["sens_load"].replace([np.inf, -np.inf], np.nan).dropna()
        jit = (np.random.default_rng(si + 5).random(len(v)) - 0.5) * 0.28
        axs.scatter(np.full(len(v), si) + jit, v, s=9, alpha=0.4, color=col, edgecolors="none")
        axs.hlines(np.median(v), si - 0.25, si + 0.25, color="k", lw=2.2, zorder=5)
    _, _, _, _, psl = _mwu(tr.sens_load, su.sens_load)
    axs.axhline(0, color="0.7", lw=0.8, ls=":")
    axs.set_xticks([0, 1]); axs.set_xticklabels(["transient", "sustained"], fontsize=10)
    axs.set_ylabel("fast − slow pulse (Hz)"); axs.set_title(f"sensory-CD loading  (p={psl:.1e})", fontsize=10.5)
    lines.append(f"[sens_load] t vs s p={psl:.2e}")

    # coverage note panel
    axc = fig.add_subplot(gs[1, 2]); axc.axis("off")
    cov = []
    for prefix in ("change", "fa"):
        for stt in STATES:
            n = df[f"{prefix}_{stt}"].replace([np.inf, -np.inf], np.nan).notna().sum()
            cov.append(f"{prefix}_{stt}: {n}/{len(df)} cells")
    axc.text(0.0, 1.0, "coverage (cells with >=5 events)\n" + "\n".join(cov)
             + f"\n\nn cells: transient={len(tr)}, sustained={len(su)}",
             transform=axc.transAxes, va="top", ha="left", fontsize=8, family="monospace")

    fig.suptitle("2a — does the transient/sustained axis interact with behavioural state?\n"
                 "(i) FR-NORMALIZED task-state loading is NULL (raw-Hz result was a firing-rate artifact) · "
                 "(ii) change & FA ramps split by state",
                 fontsize=12.5, y=1.005)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"state_x_class.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)
    (OUT / "state_x_class_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/state_x_class.png (+.pdf)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
