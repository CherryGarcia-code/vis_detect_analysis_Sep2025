"""Population state-conditioned sensory response — BG_046 17092025 (Expert).

Turns the single-unit observation (state modulates the Change response, most for
*small* changes) into a population test across all Change_ON-responsive units.

Per-unit metric
---------------
For each responsive unit, a single shared baseline (mean, SD) is computed from the
pre-change window (-0.4,-0.05 s) over ALL go trials (Hit+Miss) — one baseline per
unit, used for every condition, so the across-state comparison is not circular and
units are equalized for pooling (CLAUDE.md golden rule). The evoked response for a
condition is the z-scored firing in the early, pre-lick sensory window (0-0.25 s).

Statistics (each unit contributes one value per condition -> paired across units)
---------------------------------------------------------------------------------
  * State effect within Small / within Big change:
        Friedman omnibus (3 states, paired) -> Wilcoxon signed-rank post-hoc
        (3 pairs, Holm-corrected). Effect sizes: Kendall's W, matched-pairs r.
  * State x size interaction ("diverge for small / converge for big"):
        per-unit across-state SD for Small vs Big -> Wilcoxon signed-rank.
  * Hit vs Miss (go trials, pooled state):
        per-unit evoked z, Hit vs Miss -> Wilcoxon signed-rank.
Separate scientific questions are NOT cross-corrected (CLAUDE.md); only the 3
post-hoc state pairs within a size group are Holm-corrected.

Outputs
-------
  figures/state_labeler/BG_046/population_state_sensory_17092025.png
  figures/state_labeler/BG_046/population_state_sensory_stats.csv
"""

import os
import gc
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite import config as cfg
from visdetect.suite.config import STATE_LABEL_COLORS
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.utils import build_population_tensor, smooth_psth, bootstrap_ci
from visdetect.analysis.constants import DEFAULT_SIGMA_MS

setup_style()

SESSION = "17092025"
SUBJECT = "BG_046"
STATES = ["Impulsive", "StimSens", "Abort"]
BIN = 0.01
SIGMA_MS = DEFAULT_SIGMA_MS
WINDOW = (-0.5, 1.0)
BASELINE_WIN = (-0.4, -0.05)
SENSORY_WIN = (0.0, 0.25)
SIZE_GROUPS = [("Small", {1.25, 1.35, 1.5}), ("Big", {2.0, 4.0})]

OUTCOME_COLORS = getattr(cfg, "OUTCOME_COLORS",
                         {"Hit": "#4CAF50", "Miss": "#F44336"})

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_TAG_CSV = os.path.join(_REPO, "data", "cache", "state_tags", SUBJECT, f"{SESSION}.csv")
RESP_CACHE = os.path.join(_REPO, "data", "cache", "state_labeling", "responsiveness_all_sessions.csv")
STATS_CSV = os.path.join(
    _REPO, "FIGURES", "state_labeler", SUBJECT,
    "population_state_sensory_stats.csv",
)


def wilcoxon_paired(x, y):
    """Return (W, p, z, matched-pairs r, n_nonzero) for a paired Wilcoxon."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    d = x - y
    n = int(np.sum(d != 0))
    res = wilcoxon(x, y, method="approx", zero_method="wilcox")
    z = float(res.zstatistic)
    r = z / np.sqrt(n) if n > 0 else np.nan
    return float(res.statistic), float(res.pvalue), z, float(r), n


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    print(f"[pop-sensory] loading {SUBJECT} {SESSION} ...")
    sess = load_session(SESSION)

    # responsive units (Change_ON screen), restricted to this session
    resp = pd.read_csv(RESP_CACHE)
    resp = resp[(resp["session_name"].astype(str) == SESSION) & (resp["is_responsive"])]
    unit_ids = [int(c) for c in resp["cluster_id"].tolist()]
    # keep only units actually present in the session
    present = {c.cluster_id for c in sess.clusters}
    unit_ids = [u for u in unit_ids if u in present]
    print(f"  responsive units: {len(unit_ids)}")

    tags = pd.read_csv(STATE_TAG_CSV)
    state_of = dict(zip(tags["trial_idx"].astype(int), tags["state_label"]))
    csize_of = {i: float(getattr(t, "change_size", np.nan)) for i, t in enumerate(sess.trials)}
    outcome_of = {i: (getattr(t, "trialoutcome", "") or "").lower()
                  for i, t in enumerate(sess.trials)}

    tensor, bc, valid_trials = build_population_tensor(
        sess, unit_ids, event_name="Change_ON", window=WINDOW,
        bin_size=BIN, outcome_filter={"Hit", "Miss"},
    )
    del sess
    gc.collect()

    vt = np.array([int(t) for t in valid_trials])
    st_arr = np.array([state_of.get(t) for t in vt])
    sz_arr = np.array([csize_of.get(t, np.nan) for t in vt])
    oc_arr = np.array([outcome_of.get(t) for t in vt])
    go_mask = sz_arr > 1.0

    base_bins = (bc >= BASELINE_WIN[0]) & (bc < BASELINE_WIN[1])
    sens_bins = (bc >= SENSORY_WIN[0]) & (bc < SENSORY_WIN[1])
    n_units = len(unit_ids)

    # per-unit shared baseline from all go trials
    base_mean = np.zeros(n_units)
    base_std = np.ones(n_units)
    for j in range(n_units):
        vals = tensor[go_mask][:, base_bins, j].ravel()
        base_mean[j] = vals.mean()
        s = vals.std()
        base_std[j] = s if s > 1e-6 else 1.0

    def cond_ztrace(mask):
        """Mean z-scored PSTH trace per unit for a trial mask -> (n_units, n_bins)."""
        out = np.full((n_units, tensor.shape[1]), np.nan)
        if mask.sum() == 0:
            return out
        m = tensor[mask].mean(axis=0)               # (bins, units)
        out = ((m - base_mean[None, :]) / base_std[None, :]).T
        return out

    # population traces + per-unit evoked scalar, per (state, size) on Hit trials
    pop_trace = {}                                  # (state,size) -> (n_units,n_bins)
    evoked = {}                                     # (state,size) -> (n_units,)
    n_trials_cond = {}
    for s in STATES:
        for gname, gset in SIZE_GROUPS:
            mask = go_mask & (oc_arr == "hit") & (st_arr == s) \
                & np.array([c in gset for c in sz_arr])
            tr = cond_ztrace(mask)
            pop_trace[(s, gname)] = tr
            evoked[(s, gname)] = tr[:, sens_bins].mean(axis=1)
            n_trials_cond[(s, gname)] = int(mask.sum())

    # Hit vs Miss (go trials, pooled over state/size)
    hit_tr = cond_ztrace(go_mask & (oc_arr == "hit"))
    miss_tr = cond_ztrace(go_mask & (oc_arr == "miss"))
    evoked_hit = hit_tr[:, sens_bins].mean(axis=1)
    evoked_miss = miss_tr[:, sens_bins].mean(axis=1)
    n_hit = int((go_mask & (oc_arr == "hit")).sum())
    n_miss = int((go_mask & (oc_arr == "miss")).sum())

    # -------------------------------------------------------------- stats ---
    rows = []

    def add(test, sname, sval, p, esname, esval, n, npg, interp, notes=""):
        rows.append(dict(test=test, statistic_name=sname, statistic_value=sval,
                         p_value=p, effect_size_name=esname, effect_size_value=esval,
                         n=n, n_per_group=npg, interpretation=interp, notes=notes))

    posthoc_summary = {}
    for gname, _ in SIZE_GROUPS:
        arrs = [evoked[(s, gname)] for s in STATES]
        chi2, p = friedmanchisquare(*arrs)
        W = chi2 / (n_units * (len(STATES) - 1))           # Kendall's W
        npg = "|".join(f"{s}:{n_trials_cond[(s, gname)]}tr" for s in STATES)
        add(f"state_effect_{gname}", "chi2(Friedman)", round(chi2, 3), p,
            "Kendall_W", round(W, 3), n_units, npg,
            f"{'sig' if p < 0.05 else 'n.s.'} state effect ({gname} changes)",
            f"means z: " + ", ".join(f"{s}={np.mean(evoked[(s, gname)]):.2f}" for s in STATES))
        # post-hoc (Holm) regardless, flagged
        pairs = list(combinations(STATES, 2))
        pvals, recs = [], []
        for a, b in pairs:
            Wst, pp, z, r, nn = wilcoxon_paired(evoked[(a, gname)], evoked[(b, gname)])
            pvals.append(pp)
            recs.append((a, b, Wst, pp, z, r, nn))
        adj = holm(pvals)
        ph = []
        for (a, b, Wst, pp, z, r, nn), pa in zip(recs, adj):
            add(f"posthoc_{gname}_{a}_vs_{b}", "W(Wilcoxon)", round(Wst, 1), pp,
                "r_matched", round(r, 3), nn, "",
                f"{a} vs {b} ({gname}): {'sig' if pa < 0.05 else 'n.s.'} (Holm)",
                f"p_holm={pa:.4f}")
            ph.append((a, b, pa, r))
        posthoc_summary[gname] = (p, W, ph)

    # interaction: across-state spread (SD) Small vs Big
    spread_small = np.std(np.stack([evoked[(s, "Small")] for s in STATES]), axis=0, ddof=1)
    spread_big = np.std(np.stack([evoked[(s, "Big")] for s in STATES]), axis=0, ddof=1)
    Wsp, psp, zsp, rsp, nsp = wilcoxon_paired(spread_small, spread_big)
    diff_med = float(np.median(spread_small - spread_big))
    add("interaction_spread_small_vs_big", "W(Wilcoxon)", round(Wsp, 1), psp,
        "r_matched", round(rsp, 3), nsp, f"units:{n_units}",
        f"state spread {'larger' if diff_med > 0 else 'smaller'} for Small "
        f"({'sig' if psp < 0.05 else 'n.s.'})",
        f"median(spread_Small-spread_Big)={diff_med:.3f}")

    # Hit vs Miss
    Whm, phm, zhm, rhm, nhm = wilcoxon_paired(evoked_hit, evoked_miss)
    med_hm = float(np.median(evoked_hit - evoked_miss))
    add("hit_vs_miss_go", "W(Wilcoxon)", round(Whm, 1), phm,
        "r_matched", round(rhm, 3), nhm, f"Hit:{n_hit}tr|Miss:{n_miss}tr",
        f"Hit {'>' if med_hm > 0 else '<'} Miss evoked ({'sig' if phm < 0.05 else 'n.s.'})",
        f"median(Hit-Miss)={med_hm:.3f}")

    stats_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(STATS_CSV), exist_ok=True)
    stats_df.to_csv(STATS_CSV, index=False)
    print("\n=== STATS ===")
    print(stats_df[["test", "statistic_value", "p_value",
                    "effect_size_value", "interpretation"]].to_string(index=False))

    # --------------------------------------------------------------- plot ---
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.26,
                           left=0.09, right=0.97, top=0.90, bottom=0.08)

    def plot_pop(ax, gname, title):
        for s in STATES:
            tr = pop_trace[(s, gname)]
            mean = smooth_psth(np.nanmean(tr, axis=0), BIN, SIGMA_MS)
            sem = smooth_psth(np.nanstd(tr, axis=0) / np.sqrt(n_units), BIN, SIGMA_MS)
            ax.plot(bc, mean, color=STATE_LABEL_COLORS[s], lw=1.9, label=s, zorder=3)
            ax.fill_between(bc, mean - sem, mean + sem, color=STATE_LABEL_COLORS[s],
                            alpha=0.18, lw=0, zorder=2)
        ax.axvspan(*SENSORY_WIN, color="0.85", alpha=0.5, zorder=0)
        ax.axvline(0, color="k", ls="--", lw=0.9, alpha=0.6)
        ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)
        ax.set_xlim(bc[0], bc[-1])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time from change onset (s)")
        ax.set_ylabel("Population z-score")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.text(0.97, 0.04, f"n = {n_units} units", transform=ax.transAxes,
                fontsize=8, color="gray", ha="right")

    plot_pop(fig.add_subplot(gs[0, 0]), "Small", "A.  Population response — Small change (1.25–1.5×)")
    plot_pop(fig.add_subplot(gs[0, 1]), "Big", "B.  Population response — Big change (2.0–4.0×)")

    # C. interaction plot: mean evoked z (+/-95% CI) vs state, Small & Big
    axC = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(STATES))
    for gname, ls, mk in [("Small", "-", "o"), ("Big", "--", "s")]:
        means, los, his = [], [], []
        for s in STATES:
            vals = evoked[(s, gname)]
            lo, hi = bootstrap_ci(vals, n_bootstrap=1000, ci_level=0.95, seed=42)
            means.append(np.mean(vals)); los.append(lo); his.append(hi)
        means = np.array(means); los = np.array(los); his = np.array(his)
        axC.plot(xs, means, ls=ls, color="0.35", lw=1.6, zorder=2,
                 label=f"{gname} change")
        axC.errorbar(xs, means, yerr=[means - los, his - means], fmt="none",
                     ecolor="0.5", capsize=3, zorder=2)
        for i, s in enumerate(STATES):
            axC.scatter([xs[i]], [means[i]], s=70, color=STATE_LABEL_COLORS[s],
                        edgecolors="white", linewidths=0.8, marker=mk, zorder=4)
    axC.set_xticks(xs); axC.set_xticklabels(STATES)
    axC.set_ylabel("Evoked z (0–250 ms)")
    axC.set_title("C.  State × change-size  (mean ± 95% CI)", fontsize=11, fontweight="bold")
    axC.legend(frameon=False, fontsize=8.5, loc="upper right")
    pS, WS, _ = posthoc_summary["Small"]
    pB, WB, _ = posthoc_summary["Big"]
    spread_dir = "Big>Small" if diff_med < 0 else "Small>Big"
    axC.text(0.02, 0.98,
             f"Friedman  Small: p={pS:.1e}, W={WS:.2f}\n"
             f"Friedman  Big:   p={pB:.1e}, W={WB:.2f}\n"
             f"state spread {spread_dir}: p={psp:.1e}, r={abs(rsp):.2f}",
             transform=axC.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", fc="0.95", ec="none", alpha=0.85))

    # D. Hit vs Miss evoked z (paired across units)
    axD = fig.add_subplot(gs[1, 1])
    data = [evoked_hit, evoked_miss]
    labels = ["Hit", "Miss"]
    cols = [OUTCOME_COLORS.get("Hit", "#4CAF50"), OUTCOME_COLORS.get("Miss", "#F44336")]
    rng = np.random.default_rng(42)
    for i in range(n_units):
        axD.plot([0, 1], [evoked_hit[i], evoked_miss[i]], color="0.7",
                 lw=0.5, alpha=0.35, zorder=1)
    for k, (vals, c) in enumerate(zip(data, cols)):
        jx = rng.normal(k, 0.05, size=n_units)
        axD.scatter(jx, vals, s=18, color=c, edgecolors="white",
                    linewidths=0.3, zorder=3, alpha=0.9)
        axD.scatter([k], [np.mean(vals)], s=120, color=c, edgecolors="k",
                    linewidths=1.2, marker="D", zorder=4)
    axD.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)
    axD.set_xticks([0, 1]); axD.set_xticklabels(labels)
    axD.set_xlim(-0.4, 1.4)
    axD.set_ylabel("Evoked z (0–250 ms)")
    axD.set_title("D.  Hit vs Miss  (go trials, per unit)", fontsize=11, fontweight="bold")
    stars = ("***" if phm < 1e-3 else "**" if phm < 1e-2 else "*" if phm < 0.05 else "n.s.")
    axD.text(0.5, 0.97, f"Wilcoxon p={phm:.1e}  r={rhm:.2f}  {stars}",
             transform=axD.transAxes, ha="center", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round", fc="0.95", ec="none", alpha=0.85))
    axD.text(0.97, 0.04, f"n = {n_units} units", transform=axD.transAxes,
             fontsize=8, color="gray", ha="right")

    fig.suptitle(
        f"Population sensory response is state-dependent — {SUBJECT} {SESSION} (Expert)\n"
        f"{n_units} Change-responsive units · per-unit shared-baseline z · "
        "early pre-lick window (0–250 ms)",
        fontsize=12.5, fontweight="bold", y=0.985,
    )
    save_figure(fig, f"population_state_sensory_{SESSION}", f"state_labeler/{SUBJECT}")
    plt.close(fig)
    print("[pop-sensory] done.")


if __name__ == "__main__":
    main()
