"""Transfer panel: each tagged state keeps its defining outcome signature across mice.

Computes, per subject, each state's *defining* outcome-composition fraction
(Impulsive->inappropriate licks, StimSens->appropriate licks, Disengaged->no-lick,
Abort->aborts) pooled over that subject's tagged sessions, and draws a
states x subjects heatmap. Similar, high values across columns = the
BG_046-trained rule transfers.

IMPORTANT: the heatmap cell is a *signature* — the mean defining-outcome fraction
WITHIN trials tagged that state (high by construction) — NOT the state's occupancy
(fraction of trials in that state). Both are printed to stdout.

Session set:
  --qc-only               keep only QC-passing sessions from the subject's staging
                          manifest (d'/trial-count criterion) -- recommended.
  --exclude-late-breakdown drop the trailing run of heavily-disengaged sessions
                          (end-of-training breakdown) instead of full QC.

    py scripts/state_labeling/make_transfer_panel.py --subjects BG_046 BG_031 BG_039 \
       --qc-only --out figures/state_labeler/slide_state_transfer_qc.png
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis.config import (
    chronological_sort, canonical_session_id, load_staging_manifest,
)

# state -> (per-trial defining feature column, plain-language name)
DEFINING = {
    "Impulsive":  ("f_inapplick", "inappropriate licks"),
    "StimSens":   ("f_applick",   "appropriate licks"),
    "Disengaged": ("f_nolick",    "no-lick"),
    "Abort":      ("f_abort",     "aborts"),
}
STATE_ROWS = ["Impulsive", "StimSens", "Disengaged", "Abort"]


def subject_sessions(tags_root, subj):
    files = [f for f in glob.glob(os.path.join(tags_root, subj, "*.csv"))
             if "_tag_summary" not in os.path.basename(f)]
    tok = {os.path.basename(f)[:-4]: f for f in files}
    order = [str(s) for s in chronological_sort(list(tok.keys()))]
    return [(sn, tok[sn]) for sn in order]


def qc_pass_ids(subj):
    """Canonical ids of the subject's QC-passing sessions (d'/trial-count filter)."""
    m = load_staging_manifest(qc_only=True,
                              manifest_path=os.path.join("data", f"{subj}_staging_manifest.csv"))
    return set(canonical_session_id(s) for s in m["session_name"])


def peel_late_breakdown(sessions, dfs, competence_stim=0.35, breakdown_diseng=0.5):
    """Drop the trailing run of heavily-disengaged sessions (end-of-training breakdown)."""
    frac = {sn: d["state_label"].value_counts(normalize=True) for sn, d in dfs.items()}
    diseng = {sn: float(frac[sn].get("Disengaged", 0.0)) for sn, _ in sessions}
    stim = {sn: float(frac[sn].get("StimSens", 0.0)) for sn, _ in sessions}
    order = [sn for sn, _ in sessions]
    if not any(stim[sn] >= competence_stim for sn in order):
        return order, []
    kept, dropped = list(order), []
    while len(kept) > 3 and diseng[kept[-1]] >= breakdown_diseng \
            and any(stim[sn] >= competence_stim for sn in kept[:-1]):
        dropped.insert(0, kept.pop())
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description="Cross-subject state-signature transfer heatmap.")
    ap.add_argument("--tags-root", default=os.path.join("data", "cache", "state_tags"))
    ap.add_argument("--subjects", nargs="+", default=["BG_046", "BG_031", "BG_039"])
    ap.add_argument("--qc-only", action="store_true",
                    help="keep only QC-passing sessions from each subject's staging manifest")
    ap.add_argument("--exclude-late-breakdown", action="store_true")
    ap.add_argument("--out", default=os.path.join("figures", "state_labeler",
                                                  "slide_state_transfer.png"))
    args = ap.parse_args()

    sig = np.full((len(STATE_ROWS), len(args.subjects)), np.nan)   # defining-outcome signature
    occ = np.full((len(STATE_ROWS), len(args.subjects)), np.nan)   # occupancy (frac of trials)
    for j, subj in enumerate(args.subjects):
        sessions = subject_sessions(args.tags_root, subj)
        dropped = []
        if args.qc_only:
            qc = qc_pass_ids(subj)
            sessions = [(sn, f) for sn, f in sessions if canonical_session_id(sn) in qc]
            kept = [sn for sn, _ in sessions]
            dfs = {sn: pd.read_csv(f) for sn, f in sessions}
        else:
            dfs = {sn: pd.read_csv(f) for sn, f in sessions}
            if args.exclude_late_breakdown:
                kept, dropped = peel_late_breakdown(sessions, dfs)
            else:
                kept = [sn for sn, _ in sessions]

        pooled = pd.concat([dfs[sn] for sn in kept], ignore_index=True)
        vc = pooled["state_label"].value_counts(normalize=True)
        for i, state in enumerate(STATE_ROWS):
            occ[i, j] = float(vc.get(state, 0.0))
            sub = pooled.loc[pooled["state_label"] == state, DEFINING[state][0]]
            if len(sub):
                sig[i, j] = float(sub.mean())
        extra = f", dropped {len(dropped)} late-breakdown" if dropped else ""
        print(f"{subj}: kept {len(kept)} sessions, {len(pooled)} trials{extra}")

    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(args.subjects), 4.2))
    im = ax.imshow(sig, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(args.subjects)))
    ax.set_xticklabels(args.subjects, fontsize=9)
    ax.set_yticks(range(len(STATE_ROWS)))
    ax.set_yticklabels([f"{s}\n({DEFINING[s][1]})" for s in STATE_ROWS], fontsize=9)
    ax.set_xlabel("mouse", fontsize=10)
    for i in range(len(STATE_ROWS)):
        for j in range(len(args.subjects)):
            v = sig[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.55 else "0.15")
    scope = ("QC-passing sessions" if args.qc_only else
             "task-performing sessions" if args.exclude_late_breakdown else "all tagged sessions")
    ax.set_title("State definitions transfer across mice\n"
                 f"(mean defining-outcome fraction WITHIN each state; {scope})",
                 fontsize=10.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("defining-outcome fraction within state", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)
    idx = pd.Index(STATE_ROWS, name="state")
    print("\nSIGNATURE  (heatmap: mean defining-outcome fraction WITHIN state):")
    print(pd.DataFrame(sig, index=idx, columns=args.subjects).round(2).to_string())
    print("\nOCCUPANCY  (fraction of trials tagged that state):")
    print(pd.DataFrame(occ, index=idx, columns=args.subjects).round(2).to_string())


if __name__ == "__main__":
    main()
