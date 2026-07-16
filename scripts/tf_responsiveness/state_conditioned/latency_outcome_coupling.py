"""Does TF-kernel PEAK LATENCY predict OUTCOME-response magnitude across the
whole TF-responsive population?  (Population test of the early-vs-late
dissociation seen in the representative-cells figure.)

Motivation
----------
The exemplar cells suggested two functional classes among TF-responsive units:
  * EARLY / transient (short kernel_peak_t): sharp sensory pulse response but a
    WEAK change response and an FA *suppression* (dip at the impulsive lick);
  * LATE / ramping (long kernel_peak_t): strong change ramp + FA motor UP-ramp.
Here we test that across every TF-responsive cell (per-session GLM fits; units
are NOT cross-session tracked, so each cell-session is one observation).

Metrics — ALL canonical windows (EVENT_RESPONSIVENESS_WINDOWS), baseline-subtracted
Δrate (Hz), per cell:
  change_on : hit trials, resp (0,0.25)   - base (-0.4,-0.05)   [sensory-evoked]
  hit_ramp  : hit trials, resp (-0.3,-0.15)- base (-1.75,-1.25) [response lick]
  fa_ramp   : fa  trials, resp (-0.3,-0.15)- base (-1.75,-1.25) [impulsive ramp, early lick]
Predictor: kernel_peak_t (registry).  base_hz (Change_ON baseline rate) is the
firing-rate covariate for partial correlations.

⚠ hit_ramp is NOT an independent motor signal.  Hit licks follow the change by a
median ~0.64 s, so the (-0.3,-0.15) s "pre-lick" window lands at/after Change_ON on
~95% of hit trials (37% overlap the 0-0.25 s sensory response) — hit_ramp is largely
the change-evoked response measured again near the lick (empirically ρ≈+0.57 with
change_on, ρ≈+0.80 with fa_ramp).  Downstream width→coupling figures must treat
change_on / hit_ramp as ONE sensory/decision signal and fa_ramp (the only clean motor
probe: an early lick with NO change stimulus present) as the independent motor test.

BASELINE-CLAMP (Jul 2026 fix): the (-1.75,-1.25) s hit/FA-ramp baseline is taken
relative to the LICK, and FA licks are early (median ~4.6 s after Baseline_ON but
p5≈0.36 s), so on ~20% of FA trials that window started BEFORE the trial's own
Baseline_ON — sampling the ITI / previous trial (which can carry the prior change or
lick).  _delta now includes a trial only if BOTH its baseline AND response windows
start at/after that trial's Baseline_ON (complete-case).  Baseline_ON is read from the
event getter because Trial.t_start is NaN on many trials.

Stats: Spearman rho (pooled + per mouse) with 1000x bootstrap CI; PARTIAL
Spearman controlling base_hz (rules out "late cells just fire more"); early/late
split at the exemplar's 0.30 s threshold (Mann-Whitney U).  Non-parametric
throughout (project standard).

Population = TF-responsive cells in QC-pass, pre-breakdown sessions (identical to
the representative-cells figure: good_dates() with <50% Disengaged).
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
from scipy.stats import spearmanr, rankdata, mannwhitneyu, t as _tdist

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# reuse the representative-cells infrastructure (REPO, registry, good_dates, spikes)
from representative_cells import (REPO, _registry, good_dates, _spikes,  # noqa: E402
                                  load_session, get_event_times_by_trial)
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS  # noqa: E402

MICE = [("BG_046", "DMS", "#2c7fb8"), ("BG_039", "DMS", "#41ab5d"),
        ("BG_031", "VMS", "#ef6548")]
LATE_THRESH = 0.30                     # kernel_peak_t split (matches rep-cells 'LATE' tag)
MIN_TRIALS = 5                         # per outcome, to compute a Δrate
SEED = 42
CH_BASE, CH_RESP = EVENT_RESPONSIVENESS_WINDOWS["Change_ON"]     # (-0.4,-0.05),(0,0.25)
RAMP_BASE, RAMP_RESP = EVENT_RESPONSIVENESS_WINDOWS["FA"]        # (-1.75,-1.25),(-0.3,-0.15)
METRICS = [("change_on", "Change_ON response\n(0–0.25 s, hit)"),
           ("hit_ramp", "Hit pre-lick window\n(−0.3..−0.15 s; ≈change resp)"),
           ("fa_ramp", "FA motor ramp\n(−0.3..−0.15 s, early lick)")]
OUT = Path(str(REPO)) / "FIGURES/tf_glm_bg046/latency_outcome_coupling"
CACHE = OUT / "latency_outcome_metrics.csv"


def _win_rate(spk, times, win):
    """Mean firing rate (Hz) in `win` relative to each event time, averaged over trials."""
    if len(times) == 0:
        return np.nan
    times = np.asarray(times, float)
    lo = np.searchsorted(spk, times + win[0])
    hi = np.searchsorted(spk, times + win[1])
    return float(((hi - lo) / (win[1] - win[0])).mean())


def _delta(spk, times, bon, base, resp):
    """Baseline-subtracted Δrate (Hz), COMPLETE-CASE clamped to Baseline_ON.

    A trial is included only if BOTH its baseline and response windows start at/after
    that trial's Baseline_ON (`times + win[0] >= bon`), so neither window can sample the
    pre-trial ITI / previous trial.  This fixes the fa_ramp/hit_ramp baseline
    (-1.75,-1.25 s re the lick) crossing Baseline_ON on early-lick trials.  `bon` is the
    per-event Baseline_ON time (parallel to `times`)."""
    times = np.asarray(times, float)
    bon = np.asarray(bon, float)
    ok = (np.isfinite(times) & np.isfinite(bon)
          & (times + base[0] >= bon) & (times + resp[0] >= bon))
    sub = times[ok]
    if sub.size < MIN_TRIALS:
        return np.nan
    return _win_rate(spk, sub, resp) - _win_rate(spk, sub, base)


def _outcome_times(session, event, outcome):
    """(event_times, baseline_on_times) parallel float arrays for trials whose
    lowercased trialoutcome == outcome and both times are finite.  Baseline_ON is read
    from the event getter because Trial.t_start is NaN on many trials."""
    et = np.asarray(get_event_times_by_trial(session, event), float)
    bon = np.asarray(get_event_times_by_trial(session, "Baseline_ON"), float)
    ev, bo = [], []
    for i, t in enumerate(session.trials):
        if (str(getattr(t, "trialoutcome", "") or "").lower() == outcome
                and i < et.size and i < bon.size
                and np.isfinite(et[i]) and np.isfinite(bon[i])):
            ev.append(et[i])
            bo.append(bon[i])
    return np.asarray(ev, float), np.asarray(bo, float)


def session_metrics(subj, region, sess, reg_rows):
    s = load_session(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
    hit_change, hit_change_bon = _outcome_times(s, "Change_ON", "hit")  # sensory-evoked change
    hit_lick, hit_lick_bon = _outcome_times(s, "Hit", "hit")            # response-lick aligned
    fa_lick, fa_lick_bon = _outcome_times(s, "FA", "fa")                # early-lick aligned
    rows = []
    for _, r in reg_rows.iterrows():
        uid = int(r["unit"])
        spk = np.sort(_spikes(s, uid))
        if spk.size == 0:
            continue
        rows.append(dict(
            subject=subj, region=region, session=sess, unit=uid,
            kernel_peak_t=float(r["kernel_peak_t"]), c1_r=float(r["c1_r_log2"]),
            n_spikes=float(r["n_spikes"]),
            base_hz=_win_rate(spk, hit_change, CH_BASE),
            change_on=_delta(spk, hit_change, hit_change_bon, CH_BASE, CH_RESP),
            hit_ramp=_delta(spk, hit_lick, hit_lick_bon, RAMP_BASE, RAMP_RESP),
            fa_ramp=_delta(spk, fa_lick, fa_lick_bon, RAMP_BASE, RAMP_RESP),
            n_hit=len(hit_change), n_fa=len(fa_lick)))
    del s
    gc.collect()
    return rows


def compute_or_load(force=False):
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE)
    all_rows = []
    for subj, region, _ in MICE:
        reg = _registry(subj)
        reg = reg[reg.resp & reg.session_date.isin(good_dates(subj))]
        sessions = sorted(reg.session.unique())
        print(f"{subj}: {len(reg)} responsive cells in {len(sessions)} good sessions", flush=True)
        for k, sess in enumerate(sessions):
            rr = reg[reg.session == sess]
            pkl = Path(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
            if not pkl.exists():
                print(f"  [skip] {sess} (no pkl)", flush=True)
                continue
            all_rows += session_metrics(subj, region, sess, rr)
            print(f"  [{k+1}/{len(sessions)}] {sess}: {len(rr)} cells", flush=True)
    df = pd.DataFrame(all_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


# ── stats helpers ───────────────────────────────────────────────────────────
def _boot_rho(x, y, n=1000, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) == 0 or np.ptp(y[idx]) == 0:
            continue
        b.append(spearmanr(x[idx], y[idx]).statistic)
    return np.percentile(b, [2.5, 97.5]) if b else (np.nan, np.nan)


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling z (rank-residual method)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]

    def resid(a):
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    ex, ey = resid(rx), resid(ry)
    r = float(np.corrcoef(ex, ey)[0, 1])
    n = len(x)
    dof = n - 3
    tt = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    p = float(2 * _tdist.sf(abs(tt), dof))
    return r, p


def _clean(df, col):
    d = df[["kernel_peak_t", col, "base_hz", "subject", "region"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    return d


# ── figure ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    df = compute_or_load(force=a.force)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})
    cmap = {s: c for s, _, c in MICE}

    fig = plt.figure(figsize=(18, 10.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)
    stat_lines = []

    # Row 0: kernel_peak_t vs each metric (pooled, colored by mouse)
    for j, (col, label) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j])
        d = _clean(df, col)
        x, y = d.kernel_peak_t.values, d[col].values
        for s, _, c in MICE:
            m = d.subject == s
            ax.scatter(d.kernel_peak_t[m], d[col][m], s=16, color=c, alpha=0.55,
                       edgecolors="none", label=None)
        rho, p = spearmanr(x, y)
        lo, hi = _boot_rho(x, y)
        pr, pp = partial_spearman(x, y, d.base_hz.values)
        # per-mouse fit lines (region-dependence is the real story, so show it)
        for s, _, c in MICE:
            ds = d[d.subject == s]
            if len(ds) >= 8:
                mb1, mb0 = np.polyfit(ds.kernel_peak_t.values, ds[col].values, 1)
                mxs = np.linspace(ds.kernel_peak_t.min(), ds.kernel_peak_t.max(), 40)
                ax.plot(mxs, mb0 + mb1 * mxs, color=c, lw=1.6, alpha=0.9, zorder=4)
        # pooled fit (dashed, for reference)
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color="k", lw=1.6, ls="--", zorder=5)
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.axvline(LATE_THRESH, color="0.6", lw=0.8, ls="--")
        ax.set_title(label, fontsize=10.5)
        ax.set_xlabel("kernel peak latency (s)")
        ax.set_ylabel("Δ firing (Hz)")
        ax.text(0.03, 0.97,
                f"ρ={rho:+.2f} [{lo:+.2f},{hi:+.2f}]\np={p:.1e}  n={len(d)}\n"
                f"partial(base)={pr:+.2f}, p={pp:.1e}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.6,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # per-mouse
        pm = []
        for s, _, _ in MICE:
            ds = d[d.subject == s]
            if len(ds) >= 8:
                rs, ps = spearmanr(ds.kernel_peak_t, ds[col])
                pm.append(f"{s} ρ={rs:+.2f}(p={ps:.0e},n={len(ds)})")
        stat_lines.append(f"[{col}] pooled ρ={rho:+.3f} p={p:.2e} "
                          f"partial={pr:+.3f}(p={pp:.2e}) | " + " · ".join(pm))

    # Row 1, panel D: early vs late split (Mann-Whitney per metric)
    axd = fig.add_subplot(gs[1, 0])
    df["grp"] = np.where(df.kernel_peak_t >= LATE_THRESH, "late", "early")
    gcol = {"early": "#f0a202", "late": "#5e3c99"}
    xt = []
    for gi, (col, label) in enumerate(METRICS):
        for si, grp in enumerate(("early", "late")):
            vals = df.loc[df.grp == grp, col].replace([np.inf, -np.inf], np.nan).dropna()
            xc = gi + (si - 0.5) * 0.42
            xt.append((gi, label.split("\n")[0]))
            jit = (np.random.default_rng(SEED + gi * 2 + si).random(len(vals)) - 0.5) * 0.18
            axd.scatter(np.full(len(vals), xc) + jit, vals, s=10, color=gcol[grp],
                        alpha=0.4, edgecolors="none")
            axd.hlines(np.median(vals), xc - 0.19, xc + 0.19, color="k", lw=2.2, zorder=5)
        e = df.loc[df.grp == "early", col].replace([np.inf, -np.inf], np.nan).dropna()
        l = df.loc[df.grp == "late", col].replace([np.inf, -np.inf], np.nan).dropna()
        u, pu = mannwhitneyu(e, l)
        yy = max(l.max(), e.max())
        axd.text(gi, yy, f"p={pu:.1e}", ha="center", va="bottom", fontsize=8)
        stat_lines.append(f"[{col}] early(n={len(e)}) med={e.median():+.2f} vs "
                          f"late(n={len(l)}) med={l.median():+.2f}  MWU p={pu:.2e}")
    axd.axhline(0, color="0.6", lw=0.8, ls=":")
    axd.set_xticks(range(len(METRICS)))
    axd.set_xticklabels(["Change_ON", "Hit ramp", "FA ramp"], fontsize=9)
    axd.set_ylabel("Δ firing (Hz)")
    axd.set_title(f"early (<{LATE_THRESH}s) vs late (≥{LATE_THRESH}s) kernel", fontsize=10.5)
    from matplotlib.lines import Line2D
    axd.legend(handles=[Line2D([0], [0], marker="o", ls="", color=gcol[g], label=g)
                        for g in ("early", "late")], frameon=False, fontsize=9, loc="lower right")
    for sp in ("top", "right"):
        axd.spines[sp].set_visible(False)

    # Row 1, panel E: 2D change_on vs fa_ramp, colored by kernel_peak_t
    axe = fig.add_subplot(gs[1, 1])
    d2 = df[["change_on", "fa_ramp", "kernel_peak_t"]].replace([np.inf, -np.inf], np.nan).dropna()
    sc = axe.scatter(d2.change_on, d2.fa_ramp, c=d2.kernel_peak_t, cmap="viridis",
                     s=22, alpha=0.8, edgecolors="none", vmin=0, vmax=0.9)
    axe.axhline(0, color="0.6", lw=0.8, ls=":"); axe.axvline(0, color="0.6", lw=0.8, ls=":")
    axe.set_xlabel("Change_ON response (Hz)"); axe.set_ylabel("FA motor ramp (Hz)")
    axe.set_title("joint structure (color = kernel latency)", fontsize=10.5)
    plt.colorbar(sc, ax=axe, label="kernel peak latency (s)", fraction=0.046, pad=0.04)
    for sp in ("top", "right"):
        axe.spines[sp].set_visible(False)

    # Row 1, panel F: stats text
    axf = fig.add_subplot(gs[1, 2]); axf.axis("off")
    axf.text(0.0, 1.0, "Spearman ρ(kernel latency, response)\n" + "─" * 40,
             transform=axf.transAxes, va="top", ha="left", fontsize=9, family="monospace")
    axf.text(0.0, 0.90, "\n".join(_wrap(s) for s in stat_lines),
             transform=axf.transAxes, va="top", ha="left", fontsize=7.4, family="monospace")

    fig.suptitle("TF-kernel latency is largely DECOUPLED from outcome-response magnitude across the population\n"
                 "null when pooled (ρ≈0, dashed line); a weak POSITIVE relation in BG_046 DMS only, "
                 "flat/reversed in BG_031 VMS — coloured lines = per-mouse fits",
                 fontsize=13, y=1.005)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"latency_outcome_coupling.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)

    stats_csv = OUT / "latency_outcome_stats.txt"
    stats_csv.write_text("\n".join(stat_lines), encoding="utf-8")
    print(f"\nwrote {OUT}/latency_outcome_coupling.png (+.pdf)")
    print(f"n cells (with change_on): {df.change_on.notna().sum()}")
    for s in stat_lines:
        print("  " + s.encode("ascii", "replace").decode())  # console-safe (cp1252)


def _wrap(s, w=52):
    out, line = [], ""
    for tok in s.split(" "):
        if len(line) + len(tok) + 1 > w:
            out.append(line); line = tok
        else:
            line = (line + " " + tok).strip()
    out.append(line)
    return "\n   ".join(out)


if __name__ == "__main__":
    main()
