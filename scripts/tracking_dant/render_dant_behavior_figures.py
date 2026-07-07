"""Functional-activity-across-learning figures for DANT trusted tracks (BG_031).

Companion to render_dant_candidate_figures.py (which proves IDENTITY). This asks the
SCIENTIFIC question: how does each tracked neuron's task response change across learning?
Task-event PSTHs by stage + decision selectivity + reaction-time coding, PLUS each
session's Khilkevich-Lohse GLM TF-encoding call (data/cache/tf_responsive/<subj>_tf_
responsive.csv). Candidates are ranked TF-encoders first (the most interesting cells).

Reuses the behaviour feature extractors + panel helpers from the BG_046 consensus
behaviour pipeline (scripts/tracking_consensus/). One pkl pass over the candidate
sessions. All LOCAL. Subject-general via --subject (default BG_031; no state tags -> the
state panels are omitted).

Usage:  py scripts/tracking_dant/render_dant_behavior_figures.py --subject BG_031 [--n 10]
Output: FIGURES/tracking_dant/<subj>/curation/behavior_figs/dant_behavior_uid<U>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams.update({"xtick.labelsize": 10, "ytick.labelsize": 10, "axes.labelsize": 11})

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "pipelines" / "tracking"))
sys.path.insert(0, str(ROOT / "scripts" / "tracking_consensus"))
from visdetect.analysis.config import canonical_session_id, session_date_key  # noqa: E402
from visdetect.core.session import load_session  # noqa: E402
import _subject_paths as sjp  # noqa: E402
# reusable behaviour feature extractors + panel helpers (BG_046 pipeline)
from compute_behavior_cache import _psth_dict, _fa_psth, _change_trialwise, _go_indices  # noqa: E402
from render_behavior_figures import _plot_event, _wmean_psth, _stage_scalar, _stage_psth_ci  # noqa: E402

STAGE_COLORS = {"Naive": "#addd8e", "Learning": "#74c476", "Expert": "#238b45"}
STAGE_ORDER = ["Naive", "Learning", "Expert"]


def _stage_map(subject):
    p = ROOT / f"data/{subject}_staging_manifest.csv"
    st = pd.read_csv(p, dtype={"session_name": str})
    return {session_date_key(s): stg for s, stg in zip(st["session_name"], st["stage"])}


def _tf_lookup(subject):
    # registry files are named e.g. bg031_tf_responsive.csv (no underscore, no prefix)
    p = ROOT / f"data/cache/tf_responsive/{subject.lower().replace('_', '')}_tf_responsive.csv"
    if not p.exists():
        return None
    tf = pd.read_csv(p)
    return {(session_date_key(sd), int(u)): (bool(r), float(c1))
            for sd, u, r, c1 in zip(tf["session_date"], tf["unit"], tf["resp_log2"], tf["c1_r_log2"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_031")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    subj = args.subject

    cur = ROOT / f"FIGURES/tracking_dant/{subj}/curation"
    tracks = pd.read_csv(cur / "curated_tracks.csv")
    reg = pd.read_csv(ROOT / f"data/cache/dant/{subj}/dant_registry.csv", dtype={"session": str})
    reg["dant_uid"] = reg["dant_uid"].astype(int); reg["ks_unit_id"] = reg["ks_unit_id"].astype(int)
    stage_of = _stage_map(subj)
    tf_lookup = _tf_lookup(subj)
    pkl_dir = sjp.pkl_dir(subj)
    out_dir = cur / "behavior_figs"; out_dir.mkdir(parents=True, exist_ok=True)

    trusted = tracks[tracks["confidence_tier"] == "trusted"].copy()

    # per-track kept members (session token -> ks) + TF-encoding count -> rank TF-encoders first
    def members(uid):
        kept = [s for s in str(trusted[trusted.curated_uid == uid]["kept_sessions"].iloc[0]).split(";") if s]
        out = []
        for s in kept:
            m = reg[(reg.session == s) & (reg.dant_uid == int(uid))]
            if not m.empty:
                out.append((s, int(m.iloc[0]["ks_unit_id"])))
        return out

    ranked = []
    for uid in trusted["curated_uid"].astype(int):
        mem = members(uid)
        n_tf = sum((tf_lookup or {}).get((session_date_key(s), k), (False, 0))[0] for s, k in mem)
        ranked.append((uid, len(mem), n_tf))
    ranked.sort(key=lambda r: (-r[2], -r[1]))          # TF-encoders first, then span
    top = ranked[: args.n]
    print(f"{subj}: rendering {len(top)} candidates (TF-encoders first): "
          + ", ".join(f"uid{u}(span{n},TF{t})" for u, n, t in top))

    # ---- one pkl pass over the needed sessions ----
    need = {}   # session_token -> list of (uid, ks)
    for uid, _, _ in top:
        for s, k in members(uid):
            need.setdefault(s, []).append((uid, k))
    cache = {}  # (uid, session_token) -> feats
    for i, s in enumerate(sorted(need, key=session_date_key), 1):
        pkl = sjp.session_pkl(subj, s, pkl_dir)
        if pkl is None:
            print(f"  [{i}] {s}: no pkl"); continue
        S = load_session(str(pkl)); go = _go_indices(S)
        for uid, k in need[s]:
            feats = {"stage": stage_of.get(session_date_key(s), "Unknown"), "ks_unit_id": k,
                     "psths": _psth_dict(S, k), "fa": _fa_psth(S, k)}
            feats.update(_change_trialwise(S, k, go, {}))   # no state tags -> {}
            cache[(uid, s)] = feats
        print(f"  [{i}/{len(need)}] {s} ({stage_of.get(session_date_key(s),'?')}): {len(need[s])} units", flush=True)
        del S

    # ---- render per candidate ----
    done = []
    for uid, span, n_tf in top:
        entries = {s: cache[(uid, s)] for (u, s) in cache if u == uid}
        if len(entries) < 2:
            continue
        stages = {}
        for s, v in entries.items():
            stages.setdefault(v["stage"], []).append(s)
        present = [s for s in STAGE_ORDER if s in stages]

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 4, hspace=0.5, wspace=0.3, left=0.06, right=0.98,
                               top=0.87, bottom=0.06)
        tf_txt = (f"TF-ENCODING: {n_tf}/{span} sessions (GLM)" if n_tf > 0
                  else f"not TF-encoding ({span} sessions)")
        fig.text(0.06, 0.955, f"Behaviour across learning  —  {subj}  DANT dant_uid #{int(uid)}"
                 f"   ({span} sessions: {'/'.join(present)})",
                 fontsize=15, fontweight="bold", ha="left")
        fig.text(0.06, 0.928, f"DANT trusted (biophysical + held-out-ISI validated)     |     {tf_txt}",
                 fontsize=10.5, color=("#6a51a3" if n_tf > 0 else "#333"), ha="left")
        fig.text(0.06, 0.908, "row 1 = task events by stage · row 2 = decision · row 3 = TF-encoding across sessions",
                 fontsize=9, color="#777", style="italic", ha="left")

        # Row 1: task-event PSTHs by stage (reuse _plot_event)
        _plot_event(fig.add_subplot(gs[0, 0]), entries, stages, "baseline_on", "Baseline onset")
        _plot_event(fig.add_subplot(gs[0, 1]), entries, stages, "change_on_big_hit", "Change onset (large-change hit)")
        _plot_event(fig.add_subplot(gs[0, 2]), entries, stages, "hit_lick", "Hit lick (response)")
        _plot_event(fig.add_subplot(gs[0, 3]), entries, stages, "fa", "False-alarm lick (impulsive)")

        # Row 2a: hit vs miss by stage
        ax = fig.add_subplot(gs[1, 0])
        for stg in present:
            ph, c, _, sh = _stage_psth_ci(entries, stages[stg], "change_on_big_hit")
            pm, cm, _, sm = _stage_psth_ci(entries, stages[stg], "change_on_big_miss")
            if ph is not None:
                ax.plot(c, ph, color=STAGE_COLORS[stg], lw=2.2, label=f"{stg} hit")
                if sh is not None:
                    ax.fill_between(c, ph - 1.96*sh, ph + 1.96*sh, color=STAGE_COLORS[stg], alpha=0.15, lw=0)
            if pm is not None:
                ax.plot(cm, pm, color=STAGE_COLORS[stg], lw=1.5, ls="--", label=f"{stg} miss")
                if sm is not None:
                    ax.fill_between(cm, pm - 1.96*sm, pm + 1.96*sm, color=STAGE_COLORS[stg], alpha=0.08, lw=0)
        ax.axvline(0, color="0.4", lw=0.8, ls=":"); ax.set_title("Change_ON Hit vs Miss", fontsize=12)
        ax.set_xlabel("time (s)"); ax.set_ylabel("Hz"); ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        # Row 2b: change-size tuning
        ax = fig.add_subplot(gs[1, 1]); x = np.arange(len(present)); w = 0.38
        big = [_stage_scalar(entries, stages[s], "big_resp")[0] for s in present]
        small = [_stage_scalar(entries, stages[s], "small_resp")[0] for s in present]
        ax.bar(x - w/2, big, w, color="#cb181d", label="large"); ax.bar(x + w/2, small, w, color="#fcae91", label="small")
        ax.set_xticks(x); ax.set_xticklabels(present); ax.set_ylabel("evoked Hz")
        ax.set_title("Change-size tuning (hit)", fontsize=12); ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        # Row 2c: choice AUROC
        ax = fig.add_subplot(gs[1, 2])
        au = [_stage_scalar(entries, stages[s], "choice_auroc") for s in present]
        ax.bar(x, [a[0] for a in au], yerr=[a[1] for a in au], color=[STAGE_COLORS[s] for s in present], capsize=3)
        ax.axhline(0.5, color="0.4", ls="--", lw=1); ax.set_ylim(0, 1); ax.set_xticks(x); ax.set_xticklabels(present)
        ax.set_ylabel("AUROC (hit vs miss)"); ax.set_title("Choice coding", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

        # Row 2d: RT coding
        ax = fig.add_subplot(gs[1, 3])
        rt = [_stage_scalar(entries, stages[s], "rt_spearman") for s in present]
        ax.bar(x, [a[0] for a in rt], yerr=[a[1] for a in rt], color=[STAGE_COLORS[s] for s in present], capsize=3)
        ax.axhline(0, color="0.4", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(present)
        ax.set_ylabel("Spearman(resp, RT)"); ax.set_title("Reaction-time coding (hits)", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

        # Row 3a: TF-encoding across sessions (c1_r per session, colored by resp)
        ax = fig.add_subplot(gs[2, :2])
        sk_sorted = sorted(entries, key=session_date_key)
        c1s, cols, labs = [], [], []
        for s in sk_sorted:
            hit = (tf_lookup or {}).get((session_date_key(s), int(entries[s]["ks_unit_id"])))
            c1s.append(hit[1] if hit else np.nan)
            cols.append("#6a51a3" if (hit and hit[0]) else "#bdbdbd")
            labs.append(s.replace(subj + "_", ""))
        ax.bar(range(len(sk_sorted)), c1s, color=cols)
        ax.axhline(0.2, color="#d7301f", ls="--", lw=1, label="GLM C1 threshold (0.2)")
        ax.set_xticks(range(len(sk_sorted))); ax.set_xticklabels(labs, rotation=90, fontsize=7)
        ax.set_ylabel("GLM TF-kernel corr (c1_r_log2)")
        ax.set_title("TF-encoding per session (purple = resp_log2 True)", fontsize=12)
        ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)

        # Row 3b: text badge
        ax = fig.add_subplot(gs[2, 2:]); ax.axis("off")
        lines = [f"DANT dant_uid #{int(uid)}  (trusted, span {span})",
                 f"  {tf_txt}", "",
                 "TF = Khilkevich-Lohse GLM (resp_log2 = c1_r>0.2 AND c2_p<0.01);",
                 f"  {subj} base rate ~5.3% -> flagged sessions are notable.", "",
                 "sessions (stage | GLM TF | c1_r):"]
        for s in sk_sorted:
            hit = (tf_lookup or {}).get((session_date_key(s), int(entries[s]["ks_unit_id"])))
            tfs = (f"TF={'Y' if hit[0] else 'n'} c1r={hit[1]:+.2f}" if hit else "TF=na")
            lines.append(f"   {s.replace(subj+'_','')}  {entries[s]['stage']:9s} {tfs}")
        ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.5,
                family="monospace", transform=ax.transAxes)

        out = out_dir / f"dant_behavior_uid{int(uid)}.png"
        fig.savefig(out, dpi=135); plt.close(fig)
        print(f"  wrote {out.name}  (TF {n_tf}/{span})")
        done.append({"dant_uid": int(uid), "span": span, "n_tf_sessions": n_tf, "out": out.name})

    pd.DataFrame(done).to_csv(out_dir / "behavior_stats.csv", index=False)
    print(f"\nrendered {len(done)} DANT behaviour figures -> {out_dir}")


if __name__ == "__main__":
    main()
