"""Per-neuron BEHAVIOURAL profile across learning for the high-confidence consensus cells.

Now that these are established as the SAME neuron across learning (identity figures),
these figures ask the scientific question: does the neuron's task/behavioural response
CHANGE as the mouse learns? Four signal families, all along the Naive→Learning→Expert axis:

  Row 1  task-event PSTHs      : Baseline_ON, Change_ON (hit), Hit-lick, FA-lick
  Row 2  decision selectivity  : Change_ON Hit-vs-Miss, big/small-change tuning,
                                 choice AUROC, RT coding
  Row 3  behavioural state     : baseline firing by state, change-response by state,
                                 + an identity + GLM TF-encoding text badge

TF-encoding uses the Khilkevich-Lohse GLM registry (data/cache/tf_responsive/
bg046_tf_responsive.csv, `resp_log2`; 2.8% base rate) — the valid per-unit call.
The old single-pulse z-screen was stale/superseded and is NOT used.

Reads data/cache/tracking_consensus/BG_046/behavior_cache.pkl (built by
compute_behavior_cache.py).

Output: FIGURES/tracking_consensus/BG_046/behavior/behavior_um<G>_dant<D>.png
        FIGURES/tracking_consensus/BG_046/behavior/behavior_cohort_summary.png
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from visdetect.analysis.config import canonical_session_id, session_date_key  # noqa: E402

CACHE = ROOT / "data/cache/tracking_consensus/BG_046"
OUT_DIR = ROOT / "FIGURES/tracking_consensus/BG_046/behavior"

STAGE_COLORS = {"Naive": "#addd8e", "Learning": "#74c476", "Expert": "#238b45"}
STAGE_ORDER = ["Naive", "Learning", "Expert"]
STATE_COLORS = {"Impulsive": "#ef6548", "StimSens": "#6baed6", "Disengaged": "#3474ae"}
STATES = ["Impulsive", "StimSens", "Disengaged"]


# ----------------------------------------------------------------- aggregation
def _entries(cache, uid):
    return {sk: v for (u, sk), v in cache.items() if u == uid}


def _stage_of(entries):
    st = {}
    for sk, v in entries.items():
        st.setdefault(v["stage"], []).append(sk)
    return st


def _wmean_psth(entries, sessions, cond):
    """Trial-weighted mean PSTH across sessions for one condition."""
    accum, wsum, centers = None, 0.0, None
    for sk in sessions:
        e = entries[sk]
        d = e["fa"] if cond == "fa" else e["psths"].get(cond)
        if d is None or d["psth"] is None or d["n"] == 0:
            continue
        p = np.asarray(d["psth"], float) * d["n"]
        accum = p if accum is None else accum + p
        wsum += d["n"]; centers = d["centers"]
    if accum is None or wsum == 0:
        return None, None, 0
    return accum / wsum, centers, int(wsum)


def _stage_scalar(entries, sessions, field):
    vals = [entries[sk][field] for sk in sessions
            if entries[sk].get(field) is not None and np.isfinite(entries[sk].get(field, np.nan))]
    return (float(np.mean(vals)), float(np.std(vals) / max(len(vals) ** 0.5, 1)), len(vals)) if vals else (np.nan, np.nan, 0)


def _stage_state(entries, sessions, key, state):
    """Weighted-mean of a per-state field ('state_resp' or 'baseline_state')."""
    num, den = 0.0, 0
    for sk in sessions:
        d = entries[sk].get(key, {}).get(state, {})
        if d and np.isfinite(d.get("mean", np.nan)) and d.get("n", 0) > 0:
            num += d["mean"] * d["n"]; den += d["n"]
    return (num / den, den) if den else (np.nan, 0)


# ----------------------------------------------------------------- panels
def _plot_event(ax, entries, stages, cond, title, xline0="event"):
    any_data = False
    for stg in STAGE_ORDER:
        if stg not in stages:
            continue
        p, centers, n = _wmean_psth(entries, stages[stg], cond)
        if p is None:
            continue
        ax.plot(centers, p, color=STAGE_COLORS[stg], lw=2,
                label=f"{stg} (n={n})"); any_data = True
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("Hz")
    if any_data:
        ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return any_data


# TF-encoding: the VALID per-unit registry is the Khilkevich-Lohse GLM replication
# (resp_log2 = c1_r_log2>0.2 AND c2_p_log2<0.01). BG_046 base rate 2.8% (195/7047) --
# so a flagged session is genuinely notable, unlike the retired single-pulse z-screen.
TF_REGISTRY = ROOT / "data/cache/tf_responsive/bg046_tf_responsive.csv"


def _load_tf_lookup():
    if not TF_REGISTRY.exists():
        return None
    tf = pd.read_csv(TF_REGISTRY)
    return {(session_date_key(sd), int(u)): (bool(rp), float(c1))
            for sd, u, rp, c1 in zip(tf["session_date"], tf["unit"],
                                     tf["resp_log2"], tf["c1_r_log2"])}


def _tf_glm(entries, tf_lookup):
    """Per-session GLM TF-encoding call for this neuron's (session, ks_unit) nodes."""
    rows = []
    for sk, v in entries.items():
        hit = tf_lookup.get((session_date_key(sk), int(v["ks_unit_id"])))
        if hit is not None:
            rows.append((sk, v["stage"], hit[0], hit[1]))
    return len(rows), sum(r[2] for r in rows), rows


# ----------------------------------------------------------------- per neuron
def render_neuron(uid, cache, cohort, tf_lookup):
    entries = _entries(cache, uid)
    if not entries:
        return None
    stages = _stage_of(entries)
    row = cohort[cohort["um_uid"] == uid].iloc[0]
    dant = int(row["dant_uid"])
    n_tf, n_tf_resp, tf_rows = _tf_glm(entries, tf_lookup) if tf_lookup else (0, 0, [])
    tf_flag = (f"TF-ENCODING (GLM): {n_tf_resp}/{n_tf} sessions" if n_tf_resp > 0
               else (f"not TF-encoding ({n_tf} sessions in registry)" if n_tf
                     else "TF: not in registry"))

    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(3, 4, hspace=0.42, wspace=0.28,
                           left=0.05, right=0.98, top=0.87, bottom=0.06)
    present = [s for s in STAGE_ORDER if s in stages]
    fig.text(0.05, 0.955,
             f"Behavioural profile across learning  —  UM #{uid} ∩ DANT #{dant}"
             f"   ({int(row['n_agree'])} agreed sessions: {'→'.join(present)})",
             fontsize=15, fontweight="bold", ha="left")
    fig.text(0.05, 0.930,
             f"identity: Jaccard {row['jaccard']:.2f}, held-out ISI r {row['matched_isi_corr']:.2f} "
             f"(>{row['matched_isi_pctile']*100:.0f}% of null), DANT biophysical: {row['dant_composite']}"
             f"     |     {tf_flag}",
             fontsize=10, color=("#6a51a3" if n_tf_resp > 0 else "#333333"), ha="left")
    fig.text(0.05, 0.910,
             "each panel overlays the stage-averaged response (by learning stage); "
             "row 1 = task events, row 2 = decision, row 3 = state + identity/TF",
             fontsize=9, color="#777777", style="italic", ha="left")

    # Row 1: task-event PSTHs
    _plot_event(fig.add_subplot(gs[0, 0]), entries, stages, "baseline_on", "Baseline onset")
    _plot_event(fig.add_subplot(gs[0, 1]), entries, stages, "change_on_big_hit", "Change onset (large-change hit)")
    _plot_event(fig.add_subplot(gs[0, 2]), entries, stages, "hit_lick", "Hit lick (response)")
    _plot_event(fig.add_subplot(gs[0, 3]), entries, stages, "fa", "False-alarm lick (impulsive)")

    # Row 2a: Hit vs Miss (Change_ON) across stages
    ax = fig.add_subplot(gs[1, 0])
    for stg in present:
        ph, c, nh = _wmean_psth(entries, stages[stg], "change_on_big_hit")
        pm, _, nm = _wmean_psth(entries, stages[stg], "change_on_big_miss")
        if ph is not None:
            ax.plot(c, ph, color=STAGE_COLORS[stg], lw=2, label=f"{stg} hit")
        if pm is not None:
            ax.plot(c, pm, color=STAGE_COLORS[stg], lw=1.4, ls="--", label=f"{stg} miss")
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.set_title("Choice: Change_ON Hit vs Miss", fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("Hz"); ax.legend(fontsize=6.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # Row 2b: big/small change tuning
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(present)); w = 0.38
    big = [_stage_scalar(entries, stages[s], "big_resp")[0] for s in present]
    small = [_stage_scalar(entries, stages[s], "small_resp")[0] for s in present]
    ax.bar(x - w/2, big, w, color="#cb181d", label="large change")
    ax.bar(x + w/2, small, w, color="#fcae91", label="small change")
    ax.set_xticks(x); ax.set_xticklabels(present); ax.set_ylabel("evoked Hz")
    ax.set_title("Change-size tuning (hit)", fontsize=10); ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # Row 2c: choice AUROC by stage
    ax = fig.add_subplot(gs[1, 2])
    auroc = [_stage_scalar(entries, stages[s], "choice_auroc") for s in present]
    ax.bar(x, [a[0] for a in auroc], yerr=[a[1] for a in auroc],
           color=[STAGE_COLORS[s] for s in present], capsize=3)
    ax.axhline(0.5, color="0.4", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(present); ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC (hit vs miss)")
    ax.set_title("Choice coding across learning", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Row 2d: RT coding by stage
    ax = fig.add_subplot(gs[1, 3])
    rt = [_stage_scalar(entries, stages[s], "rt_spearman") for s in present]
    ax.bar(x, [a[0] for a in rt], yerr=[a[1] for a in rt],
           color=[STAGE_COLORS[s] for s in present], capsize=3)
    ax.axhline(0, color="0.4", ls="-", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(present)
    ax.set_ylabel("Spearman(resp, RT)")
    ax.set_title("Reaction-time coding (hits)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Row 3a: baseline firing by state
    ax = fig.add_subplot(gs[2, 0])
    xs = np.arange(len(present)); w = 0.26
    for j, st in enumerate(STATES):
        vals = [_stage_state(entries, stages[s], "baseline_state", st)[0] for s in present]
        ax.bar(xs + (j-1)*w, vals, w, color=STATE_COLORS[st], label=st)
    ax.set_xticks(xs); ax.set_xticklabels(present); ax.set_ylabel("baseline Hz")
    ax.set_title("Baseline firing by behavioural state", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)

    # Row 3b: change response by state
    ax = fig.add_subplot(gs[2, 1])
    for j, st in enumerate(STATES):
        vals = [_stage_state(entries, stages[s], "state_resp", st)[0] for s in present]
        ax.bar(xs + (j-1)*w, vals, w, color=STATE_COLORS[st], label=st)
    ax.set_xticks(xs); ax.set_xticklabels(present); ax.set_ylabel("evoked Hz")
    ax.set_title("Change response by behavioural state", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)

    # Row 3c: identity + TF-status text badge (span 2)
    ax = fig.add_subplot(gs[2, 2:]); ax.axis("off")
    lines = [
        f"NEURON  UM #{uid}  ∩  DANT #{dant}",
        f"  agreed span: {int(row['n_agree'])} sessions  ({'/'.join(present)})",
        f"  identity — Jaccard {row['jaccard']:.2f} | held-out ISI r {row['matched_isi_corr']:.2f}"
        f" (>{row['matched_isi_pctile']*100:.0f}% null) | DANT biophys {row['dant_composite']}",
        "",
        "TF-encoding (Khilkevich-Lohse GLM resp_log2; BG_046 base rate 2.8%):",
        f"  {n_tf_resp}/{n_tf} of this neuron's sessions are TF-responsive"
        + ("  <- longitudinal TF-encoder" if n_tf_resp >= 2 else ""),
        "",
        "sessions (stage | GLM TF | c1_r):",
    ]
    tf_by_sk = {r[0]: (r[2], r[3]) for r in tf_rows}
    for sk in sorted(entries, key=session_date_key):
        rc = tf_by_sk.get(sk)
        tfs = (f"TF={'Y' if rc[0] else 'n'} c1r={rc[1]:+.2f}" if rc else "TF=na")
        lines.append(f"   {sk}  {entries[sk]['stage']:9s} {tfs}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.5,
            family="monospace", transform=ax.transAxes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"behavior_um{uid}_dant{dant}.png"
    fig.savefig(out, dpi=135); plt.close(fig)
    return {"um_uid": uid, "dant_uid": dant, "n_sessions": len(entries),
            "stages": "/".join(present), "out": out.name}


def render_cohort_summary(cache, cohort, done, tf_lookup):
    """Do the well-tracked neurons collectively change their behaviour across learning?"""
    uids = sorted({u for (u, _) in cache})
    rows = []
    for u in uids:
        entries = _entries(cache, u)
        stages = _stage_of(entries)
        if "Learning" not in stages or "Expert" not in stages:
            continue
        la = _stage_scalar(entries, stages["Learning"], "choice_auroc")[0]
        ea = _stage_scalar(entries, stages["Expert"], "choice_auroc")[0]
        lb = _stage_scalar(entries, stages["Learning"], "big_resp")[0]
        eb = _stage_scalar(entries, stages["Expert"], "big_resp")[0]
        rows.append({"uid": u, "L_auroc": la, "E_auroc": ea, "L_big": lb, "E_big": eb})
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(15, 4.4))
    gs = gridspec.GridSpec(1, 3, wspace=0.32, left=0.06, right=0.97, top=0.80, bottom=0.16)
    fig.suptitle("High-confidence consensus neurons — behavioural change across learning "
                 f"({len(uids)} neurons both trackers agree on)", fontsize=13, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    for _, r in df.iterrows():
        if np.isfinite(r["L_auroc"]) and np.isfinite(r["E_auroc"]):
            ax.plot([0, 1], [r["L_auroc"], r["E_auroc"]], "-o", color="0.6", ms=4)
    ax.axhline(0.5, color="0.4", ls="--", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Learning", "Expert"]); ax.set_ylim(0, 1)
    ax.set_ylabel("choice AUROC (hit vs miss)")
    ax.set_title("Choice coding per neuron", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    for _, r in df.iterrows():
        if np.isfinite(r["L_big"]) and np.isfinite(r["E_big"]):
            ax.plot([0, 1], [r["L_big"], r["E_big"]], "-o", color="#cb181d", ms=4, alpha=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Learning", "Expert"])
    ax.set_ylabel("large-change evoked Hz")
    ax.set_title("Change-evoked response per neuron", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    if tf_lookup:
        ge2 = one = none = 0
        for u in uids:
            n_tf, n_resp, _ = _tf_glm(_entries(cache, u), tf_lookup)
            if n_resp >= 2: ge2 += 1
            elif n_resp == 1: one += 1
            elif n_tf: none += 1
        vals = [ge2, one, none]
        ax.bar(["TF-enc\n≥2 sess", "TF-enc\n1 sess", "not TF-enc"], vals,
               color=["#6a51a3", "#9e9ac8", "#bdbdbd"])
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("# neurons")
        ax.set_title("TF-encoding (GLM resp_log2, 2.8% base rate)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "TF registry\nnot found", ha="center", va="center", fontsize=10)
        ax.set_title("TF-encoding status", fontsize=10)

    out = OUT_DIR / "behavior_cohort_summary.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def main():
    with open(CACHE / "behavior_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    cohort = pd.read_csv(CACHE / "consensus_cohort.csv")
    tf_lookup = _load_tf_lookup()
    print("TF registry:", "loaded" if tf_lookup else "NOT FOUND (TF panels blank)")

    uids = sorted({u for (u, _) in cache})
    print(f"rendering behaviour for {len(uids)} neurons: {uids}")
    done = []
    for u in uids:
        r = render_neuron(u, cache, cohort, tf_lookup)
        if r:
            n_tf, n_resp, _ = _tf_glm(_entries(cache, u), tf_lookup) if tf_lookup else (0, 0, [])
            done.append(r); print(f"  wrote {r['out']}  (GLM TF {n_resp}/{n_tf})")
    render_cohort_summary(cache, cohort, done, tf_lookup)
    print(f"\nrendered {len(done)} behaviour figures + cohort summary")


if __name__ == "__main__":
    main()
