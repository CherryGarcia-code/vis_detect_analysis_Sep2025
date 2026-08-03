"""FA-lick (anticipatory early-lick) HAZARD across learning — 3 mice.

Motivation: the raw early-lick RT distribution confounds the lick HAZARD with
per-trial right-censoring by the change (a change is never presented before ~6 s,
so an FA is only possible while the trial's change is still ahead). The hazard
separates the two: h(t) = P(FA lick in [t, t+dt) | trial still in baseline at t).
A hazard that RISES toward the ~6 s change is genuine anticipatory timing; a flat
hazard is not. The question is whether that anticipatory rise DEVELOPS with
learning (Naive → Expert).

Reuses the canonical, already-debugged machinery (do NOT reinvent):
  - `ddm.build_trial_evidence`  — per-trial geometry (EXCLUDES aborts/ref).
  - `decision_latents.fa_lick_hazard` — cause-specific FA hazard; non-FA trials
    are censored at min(change_time_planned, decision_time), i.e. they leave the
    at-risk set at the change (the fix that stops the "hazard" collapsing onto the
    raw FA-time histogram — docstring there, 2026-06-18).
  - `decision_latents.change_onset_hazard` — when the change actually arrives (ref).

TWO versions (the existing pooled `fig_timing` only ever ran WITHOUT aborts and
never split by learning):
  - WITHOUT aborts (canonical): abort trials are dropped entirely.
  - WITH aborts: abort trials enter the risk set, censored at reactiontimes['abort']
    (they were genuinely at risk of FA-licking until they aborted). This dilutes
    the FA hazard most where aborts are frequent (early Naive).

Run: py scripts/analysis/behavior/fa_lick_hazard_learning.py [--force]
Out: FIGURES/behavior/fa_hazard/fa_lick_hazard_learning.png
     data/cache/behavior/fa_hazard_trials_<subject>.csv
     data/cache/behavior/fa_lick_hazard_summary.csv
"""
import os
import sys
import gc
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.analysis import ddm
from visdetect.analysis import decision_latents as dl
from visdetect.analysis.config import _ALL_STAGE_ORDER as STAGES, _ALL_STAGE_COLORS as STAGE_COLORS
from visdetect.analysis.evidence_learning_io import subject_sessions, SUBJECTS
from visdetect.suite.plotting import setup_style, save_figure

setup_style()

CACHE_DIR = os.path.join(_ROOT, "data", "cache", "behavior")
SUBJECTS_ORDER = ["BG_046", "BG_039", "BG_031"]
DT = 0.1          # hazard bin (s)
X_HI = 9.0        # display cap (s); late bins have few trials at risk
EARLIEST_CHANGE = 6.0
CHANGE_REF_COLOR = "#555555"
N_BOOT = 300      # session-clustered bootstrap draws for the 95% CI band
K = int(round(X_HI / DT))              # bins kept (centers up to X_HI)
CENTERS = DT * (np.arange(K) + 0.5)    # 0.05, 0.15, ... 8.95
_EDGES_L = DT * np.arange(K)           # bin left edges 0, 0.1, ...

HAZ_COLS = ["session_name", "stage", "outcome", "change_size", "change_time_planned",
            "decision_time", "change_reached", "is_abort"]


# ── Per-subject trial table (hit/miss/fa via canonical builder + abort rows) ──
def session_rows(sess, stage, session_name):
    ev = ddm.build_trial_evidence(sess)   # hit/miss/fa only (aborts excluded)
    rows = []
    if len(ev):
        ev = ev.rename(columns={"change_time": "change_time_planned"})
        for _, r in ev.iterrows():
            rows.append({"session_name": session_name, "stage": stage, "outcome": r["outcome"],
                         "change_size": float(r["change_size"]),
                         "change_time_planned": float(r["change_time_planned"]),
                         "decision_time": float(r["decision_time"]),
                         "change_reached": r["outcome"] in ("hit", "miss"),
                         "is_abort": False})
    # abort rows: censored at reactiontimes['abort'] (Baseline_ON-aligned)
    for t in getattr(sess, "trials", []):
        if (getattr(t, "trialoutcome", "") or "").lower() != "abort":
            continue
        rts = getattr(t, "reactiontimes", {}) or {}
        ab = rts.get("abort", np.nan)
        ct = float(getattr(t, "change_time", np.nan) or np.nan)
        if np.isfinite(ab) and ab > 0:
            rows.append({"session_name": session_name, "stage": stage, "outcome": "abort",
                         "change_size": float(getattr(t, "change_size", np.nan) or np.nan),
                         "change_time_planned": ct, "decision_time": float(ab),
                         "change_reached": False, "is_abort": True})
    return rows


def compute_subject(subject, force=False):
    cache = os.path.join(CACHE_DIR, f"fa_hazard_trials_{subject}.csv")
    if os.path.exists(cache) and not force:
        return pd.read_csv(cache)
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = []
    for skey, sname, stage, sess in subject_sessions(subject, stages=tuple(STAGES)):
        try:
            rows.extend(session_rows(sess, stage, str(sname)))
            print(f"  {subject} {sname} ({stage})")
        finally:
            del sess
            gc.collect()
    df = pd.DataFrame(rows, columns=HAZ_COLS)
    df.to_csv(cache, index=False)
    return df


# ── Hazard summary metrics ───────────────────────────────────────────
def ramp_metrics(centers, hz):
    """Peak time + ramp index (late 4-6 s vs early 0-2 s mean hazard)."""
    c = np.asarray(centers); h = np.asarray(hz)
    pre = c < EARLIEST_CHANGE
    peak_t = float(c[pre][np.argmax(h[pre])]) if pre.any() and h[pre].max() > 0 else np.nan
    early = h[(c >= 0) & (c < 2)].mean() if ((c >= 0) & (c < 2)).any() else np.nan
    late = h[(c >= 4) & (c < 6)].mean() if ((c >= 4) & (c < 6)).any() else np.nan
    ramp = float(late / early) if (early and np.isfinite(early) and early > 0) else np.nan
    return peak_t, ramp


# ── Discrete-time hazard on a FIXED grid + session-clustered bootstrap CI ──
def _censor(df, kind):
    """(censor_time, is_event) per trial. FA: non-FA leave the risk set at the
    change (min(change,decision)); all-lick: miss censored at decision_time.
    Reproduces dl.fa_lick_hazard / dl.lick_hazard censoring exactly."""
    oc = df["outcome"].values.astype(str)
    dtime = df["decision_time"].values.astype(float)
    ctime = df["change_time_planned"].values.astype(float)
    if kind == "fa":
        is_ev = (oc == "fa")
        chg = np.where(np.isnan(ctime), dtime, np.minimum(ctime, dtime))
        censor = np.where(is_ev, dtime, chg)
    else:
        is_ev = np.isin(oc, ("hit", "fa"))
        censor = dtime
    return censor.astype(float), is_ev


def _haz_grid(censor, is_ev):
    """Discrete-time cause-specific hazard on the fixed CENTERS grid. Vectorised
    reproduction of dl.censored_hazard (verified bit-for-bit, max|diff|=0)."""
    ct_sorted = np.sort(censor)
    at_risk = len(censor) - np.searchsorted(ct_sorted, _EDGES_L + 1e-12, side="right")
    raw_bin = np.ceil(censor / DT).astype(int) - 1
    okev = is_ev & (raw_bin >= 0) & (raw_bin < K)   # events beyond X_HI excluded from numerator
    ev = np.bincount(raw_bin[okev], minlength=K)[:K]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(at_risk > 0, ev / at_risk, 0.0)


def stage_hazard_ci(tab, stage, kind, n_boot=N_BOOT, seed=42):
    """Point hazard + SESSION-clustered bootstrap 95% CI (resamples whole
    sessions, honest to the ~3-Naive-session reality). None if under-powered."""
    st = tab[tab["stage"] == stage]
    n_fa = int((st["outcome"] == "fa").sum())
    n_lick = int(st["outcome"].isin(["hit", "fa"]).sum())
    if len(st) < 20 or (kind == "fa" and n_fa < 15) or (kind == "all" and n_lick < 15):
        return None
    censor, is_ev = _censor(st, kind)
    hz = _haz_grid(censor, is_ev)
    lo = hi = None
    sess = st["session_name"].astype(str).values
    uniq = np.unique(sess)
    if len(uniq) >= 2:
        by = {s: _censor(st[sess == s], kind) for s in uniq}
        rng = np.random.default_rng(seed)
        boot = np.empty((n_boot, K))
        for b in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            boot[b] = _haz_grid(np.concatenate([by[s][0] for s in pick]),
                                np.concatenate([by[s][1] for s in pick]))
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
    return {"centers": CENTERS, "hz": hz, "lo": lo, "hi": hi,
            "n_fa": n_fa, "n_lick": n_lick, "n_sessions": int(len(uniq))}


def session_fa46(tab, stage, min_fa=15):
    """Per-session fraction of FA licks landing in 4-6 s (depletion-free; the
    session is the replication unit). Sessions with < min_fa FAs are dropped."""
    st = tab[(tab["stage"] == stage) & (tab["outcome"] == "fa")]
    out = []
    for _, g in st.groupby("session_name"):
        if len(g) >= min_fa:
            dtv = g["decision_time"].values
            out.append(float(np.mean((dtv >= 4) & (dtv < 6))))
    return out


# ── Figure ────────────────────────────────────────────────────────────
def make_figure(data):
    versions = [("WITHOUT aborts", False), ("WITH aborts", True)]
    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.34, wspace=0.24)
    summary = []

    for r, (vlabel, with_ab) in enumerate(versions):
        for j, subject in enumerate(SUBJECTS_ORDER):
            df = data[subject]
            tab = df if with_ab else df[~df["is_abort"]]
            ax = fig.add_subplot(gs[r, j])
            for stage in STAGES:
                res = stage_hazard_ci(tab, stage, "fa")
                if res is None:
                    continue
                c, hz, lo, hi = res["centers"], res["hz"], res["lo"], res["hi"]
                col = STAGE_COLORS[stage]
                if lo is not None:
                    ax.fill_between(c, lo, hi, color=col, alpha=0.18, lw=0)
                ax.plot(c, hz, color=col, lw=2.0,
                        label=f"{stage} (n_fa={res['n_fa']}, {res['n_sessions']}s)")
                pk, ramp = ramp_metrics(c, hz)
                summary.append({"subject": subject, "region": SUBJECTS.get(subject, "?"),
                                "version": "with_aborts" if with_ab else "without_aborts",
                                "stage": stage, "n_sessions": res["n_sessions"], "n_fa": res["n_fa"],
                                "peak_hazard_time_s": pk, "ramp_index_late_over_early": ramp})
            ax.axvline(EARLIEST_CHANGE, color="k", ls="--", lw=0.9, alpha=0.55)
            ax.text(EARLIEST_CHANGE, ax.get_ylim()[1], " earliest change ~6 s", fontsize=7,
                    color="0.3", va="top", ha="left")
            ax.set_xlim(0, X_HI)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("time in baseline (s)")
            if j == 0:
                ax.set_ylabel(f"{vlabel}\nFA-lick hazard  P(lick | at risk)/{int(DT*1000)}ms")
            else:
                ax.set_ylabel(f"FA-lick hazard /{int(DT*1000)}ms")
            reg = SUBJECTS.get(subject, "?")
            ax.set_title(f"{subject} ({reg}) — {vlabel}", fontweight="bold", loc="left", fontsize=10.5)
            ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle("FA-lick (anticipatory early-lick) hazard across learning — pre-change ramp is "
                 "HIGHEST in Naive, suppressed with learning (DMS BG_046; BG_039 weaker); VMS BG_031 reverses",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, "Headline = ABSOLUTE pre-change (4-6 s) FA hazard (BG_046 Naive 0.019 -> Expert 0.007). "
             "The late/early 'ramp index' is inflated ~1.2-1.5x by survival censoring — do not headline it.",
             ha="center", fontsize=8, color="0.35")
    return fig, pd.DataFrame(summary)


def make_disentangle_figure(data):
    """Row 1: FA hazard (pre-change). Row 2: all first-lick hazard (FA+Hit) +
    change-onset reference. If row 2 ramps to ~6 s EQUALLY across stages while
    row 1 shrinks with learning, the FA change is Hit-reclassification (timing
    stable); if row 2's ramp also develops, that is real learned timing.
    WITHOUT aborts (aborts shown not to matter for the FA hazard)."""
    rows = [("FA-lick hazard\n(licks BEFORE change)", "fa"),
            ("all first-lick hazard\n(FA + Hit)", "all")]
    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.34, wspace=0.24)
    summary = []
    for r, (rlabel, kind) in enumerate(rows):
        for j, subject in enumerate(SUBJECTS_ORDER):
            tab = data[subject][~data[subject]["is_abort"]]
            ax = fig.add_subplot(gs[r, j])
            results = {}
            for stage in STAGES:
                res = stage_hazard_ci(tab, stage, kind)
                if res is not None:
                    results[stage] = res
            ymax = max([float(np.nanmax(r["hz"])) for r in results.values()] or [1.0])
            if kind == "all" and len(tab):
                cc, ch, _ = dl.change_onset_hazard(tab, dt=DT)
                mm = cc <= X_HI
                if ch[mm].max() > 0:
                    ax.plot(cc[mm], ch[mm] / ch[mm].max() * ymax, color="#555555",
                            ls=(0, (1, 1)), lw=1.1, alpha=0.65, zorder=1,
                            label="change-onset (scaled ref)")
            for stage, res in results.items():
                c, hz, lo, hi = res["centers"], res["hz"], res["lo"], res["hi"]
                col = STAGE_COLORS[stage]
                if lo is not None:
                    ax.fill_between(c, lo, hi, color=col, alpha=0.16, lw=0)
                lbl = (f"{stage} (n_fa={res['n_fa']}, {res['n_sessions']}s)" if kind == "fa"
                       else f"{stage} (n_lick={res['n_lick']}, {res['n_sessions']}s)")
                ax.plot(c, hz, color=col, lw=2.0, label=lbl)
                pk, ramp = ramp_metrics(c, hz)
                frac46 = np.nan
                stt = tab[(tab["stage"] == stage) & (tab["outcome"] == "fa")]
                if len(stt) >= 15:
                    dtv = stt["decision_time"].values
                    frac46 = float(np.mean((dtv >= 4) & (dtv < 6)))
                summary.append({"subject": subject, "region": SUBJECTS.get(subject, "?"),
                                "hazard": kind, "stage": stage, "n_fa": res["n_fa"], "n_lick": res["n_lick"],
                                "n_sessions": res["n_sessions"], "peak_hazard_time_s": pk,
                                "ramp_index_late_over_early": ramp, "fa_frac_in_4_6s": frac46})
            if kind == "fa":  # depletion-free effect, immune to the ramp-index inflation
                fr = []
                for stage in STAGES:
                    st = tab[(tab["stage"] == stage) & (tab["outcome"] == "fa")]
                    if len(st) >= 15:
                        dtv = st["decision_time"].values
                        fr.append(f"{stage[:3]} {np.mean((dtv >= 4) & (dtv < 6)):.2f}")
                if fr:
                    ax.text(0.97, 0.03, "FA-licks in 4-6 s (frac):\n" + ", ".join(fr),
                            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="right",
                            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.75))
            ax.axvline(EARLIEST_CHANGE, color="k", ls="--", lw=0.9, alpha=0.55)
            ax.text(EARLIEST_CHANGE, ax.get_ylim()[1], " earliest change ~6 s", fontsize=7,
                    color="0.3", va="top", ha="left")
            ax.set_xlim(0, X_HI)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("time in baseline (s)")
            ax.set_ylabel(rlabel + f"\nP(lick | at risk)/{int(DT*1000)}ms" if j == 0
                          else f"P(lick | at risk)/{int(DT*1000)}ms")
            ax.set_title(f"{subject} ({SUBJECTS.get(subject, '?')})", fontweight="bold",
                         loc="left", fontsize=10.5)
            ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("FA-lick vs ALL-lick hazard — the pre-change reduction is REAL; the post-6 s peak is "
                 "SURVIVORSHIP, not 'detection develops'", fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "Pre-6 s: all-lick == FA by definition (max|diff|=0) -> the Naive>Expert pre-change reduction is NOT a "
             "Hit-reclassification artifact. Post-6 s: the hazard peak is LARGEST in Naive (few trials survive to the "
             "change), so it does NOT show detection 'developing'; the learning shift is a COUNT shift (FA-fraction of "
             "first-licks 0.73->0.51 in BG_046).", ha="center", fontsize=8, color="0.35")
    return fig, pd.DataFrame(summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FA-lick hazard across learning (3 mice, ± aborts).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = {s: compute_subject(s, force=args.force) for s in SUBJECTS_ORDER}
    pd.set_option("display.width", 200, "display.max_columns", 20)

    # sanity: the vectorised fixed-grid hazard reproduces the canonical dl hazard
    _chk = data["BG_046"]
    _chk = _chk[(~_chk["is_abort"]) & (_chk["stage"] == "Expert")]
    _cc, _hdl, _ = dl.fa_lick_hazard(_chk, dt=DT)
    _hloc = _haz_grid(*_censor(_chk, "fa"))
    _kk = min(len(_hdl), K)
    _md = float(np.max(np.abs(_hloc[:_kk] - _hdl[:_kk])))
    print(f"[sanity] local vs dl.fa_lick_hazard max|diff| = {_md:.2e}")
    assert _md < 1e-9, "local hazard diverges from canonical dl.censored_hazard!"

    fig, summ = make_figure(data)
    paths = save_figure(fig, "fa_lick_hazard_learning", "behavior/fa_hazard")
    summ.to_csv(os.path.join(CACHE_DIR, "fa_lick_hazard_summary.csv"), index=False)
    print("\nSaved figure:", paths[0])
    print("=== FA-lick hazard ramp summary (± aborts) ===")
    print(summ.to_string(index=False))

    fig2, summ2 = make_disentangle_figure(data)
    paths2 = save_figure(fig2, "fa_vs_alllick_hazard_learning", "behavior/fa_hazard")
    summ2.to_csv(os.path.join(CACHE_DIR, "fa_vs_alllick_hazard_summary.csv"), index=False)
    print("\nSaved figure:", paths2[0])
    print("=== FA vs ALL-lick hazard ramp summary (without aborts) ===")
    print(summ2.to_string(index=False))

    # session-clustered test: per-session 4-6 s FA-lick fraction (session = replicate,
    # depletion-free; the honest unit given only 3 Naive sessions/mouse)
    from scipy.stats import mannwhitneyu
    srows = []
    for subj in SUBJECTS_ORDER:
        tab = data[subj][~data[subj]["is_abort"]]
        nai, exp = session_fa46(tab, "Naive"), session_fa46(tab, "Expert")
        p = (mannwhitneyu(nai, exp, alternative="two-sided").pvalue
             if len(nai) >= 2 and len(exp) >= 2 else np.nan)
        srows.append({"subject": subj, "region": SUBJECTS.get(subj, "?"),
                      "n_naive_sess": len(nai), "n_expert_sess": len(exp),
                      "naive_fa46_median": np.median(nai) if nai else np.nan,
                      "expert_fa46_median": np.median(exp) if exp else np.nan,
                      "naive_vs_expert_session_mwu_p": p})
    sdf = pd.DataFrame(srows)
    sdf.to_csv(os.path.join(CACHE_DIR, "fa_hazard_session_level.csv"), index=False)
    print("\n=== SESSION-CLUSTERED test: per-session 4-6 s FA-lick fraction, Naive vs Expert ===")
    print(sdf.to_string(index=False))
