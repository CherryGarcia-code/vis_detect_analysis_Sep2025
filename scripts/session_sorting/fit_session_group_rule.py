"""Learn an interpretable RULE that reproduces the manual session groups.

Runs AFTER scripts/session_sorting/run_session_sorter.py. Mirrors the trial-level
calibration pipeline (visdetect.analysis.state_calibration): shallow decision tree,
leave-one-session-out Cohen's kappa, exported rules text.

Reports three things, in order of importance:
  1. TEST-RETEST kappa — the human's own self-consistency on the silently repeated
     sessions. This is the CEILING: no fitted rule can be more reliable than the
     labels it learns from. If this is low, stop and fix the taxonomy.
  2. LOSO kappa of the fitted tree vs the human labels.
  3. The rules themselves + which features carry them.

If the tree does WELL, we have a rule that generalises to untagged sessions and to
BG_039/BG_031. If it does POORLY, the human is using something the features do not
capture — which is itself the finding, and tells us what to measure next.

⚠ SCOPE: groups are learned from BEHAVIOUR, so they are clean for NEURAL dependent
variables and CIRCULAR for behavioural ones. Do not use these groups to then claim
"impulsive sessions have more early licks" — that is definitional.

Run: py scripts/session_sorting/fit_session_group_rule.py --subject BG_046
Out: data/cache/session_sorting/session_group_features.csv
     data/cache/session_sorting/session_group_rule.txt
     FIGURES/session_sorting/<SUBJ>/session_group_rule.png
"""
import os
import sys
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from visdetect.suite.plotting import setup_style, save_figure

setup_style()

TAG_DIR = os.path.join(_ROOT, "data", "cache", "state_tags")
OUT_DIR = os.path.join(_ROOT, "data", "cache", "session_sorting")
LABELS_PATH = os.path.join(OUT_DIR, "manual_session_labels.csv")
FEATURES_PATH = os.path.join(OUT_DIR, "session_group_features.csv")
RULES_PATH = os.path.join(OUT_DIR, "session_group_rule.txt")
STATES = ["StimSens", "Impulsive", "Disengaged", "Abort"]
ROLL_W = 31


# ── Session-level features (all from the tag CSVs; no pkl needed) ─────
def session_features(tags):
    n = len(tags)
    is_go = tags["is_go"].astype(bool).values
    is_hit = tags["is_hit"].astype(bool).values
    is_fa = tags["is_fa"].astype(bool).values
    lab = tags["state_label"].astype(str).values
    f = {"n_trials": n,
         "early_lick_frac": float(is_fa.mean()),
         "hit_rate_go": float((is_hit & is_go).sum() / max(is_go.sum(), 1)),
         "abort_frac": float((tags["outcome"].astype(str) == "abort").mean()),
         "mean_state_conf": float(pd.to_numeric(tags.get("state_confidence"),
                                                errors="coerce").mean())}
    for s in STATES:                                   # occupancy + longest run
        m = (lab == s)
        f[f"occ_{s}"] = float(m.mean())
        run = best = 0
        for v in m:
            run = run + 1 if v else 0
            best = max(best, run)
        f[f"maxrun_{s}_frac"] = best / max(n, 1)       # normalised: max-run scales with n
    f["switch_rate"] = float(np.mean(lab[1:] != lab[:-1])) if n > 1 else np.nan
    # first- vs second-half contrasts — these are what can express "Deteriorating"
    h = n // 2
    if h >= 10:
        go1, go2 = is_go[:h], is_go[h:]
        f["d_hit_rate_2nd_minus_1st"] = float(
            (is_hit[h:] & go2).sum() / max(go2.sum(), 1) - (is_hit[:h] & go1).sum() / max(go1.sum(), 1))
        f["d_earlylick_2nd_minus_1st"] = float(is_fa[h:].mean() - is_fa[:h].mean())
        f["d_occ_StimSens_2nd_minus_1st"] = float((lab[h:] == "StimSens").mean()
                                                  - (lab[:h] == "StimSens").mean())
        f["d_occ_Disengaged_2nd_minus_1st"] = float((lab[h:] == "Disengaged").mean()
                                                    - (lab[:h] == "Disengaged").mean())
    else:
        for k in ("d_hit_rate_2nd_minus_1st", "d_earlylick_2nd_minus_1st",
                  "d_occ_StimSens_2nd_minus_1st", "d_occ_Disengaged_2nd_minus_1st"):
            f[k] = np.nan
    return f


REQUIRED_TAG_COLS = {"trial_idx", "outcome", "is_go", "is_hit", "is_fa", "state_label"}


def build_features(subject, sessions):
    """Per-session feature rows. Skips roll-up files (e.g. `_tag_summary.csv`) and
    any tag CSV missing the required per-trial columns."""
    rows, skipped = [], []
    for s in sessions:
        p = os.path.join(TAG_DIR, subject, f"{s}.csv")
        if not os.path.exists(p) or str(s).startswith("_"):
            skipped.append(s); continue
        tags = pd.read_csv(p, dtype={"session_name": str})
        if not REQUIRED_TAG_COLS <= set(tags.columns):
            skipped.append(s); continue
        f = session_features(tags)
        f["subject"], f["session_name"] = subject, s
        rows.append(f)
    if skipped:
        print(f"  skipped {len(skipped)} non-session/invalid tag file(s): {skipped[:5]}")
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Fit an interpretable session-group rule.")
    ap.add_argument("--subject", default="BG_046")
    ap.add_argument("--min-per-class", type=int, default=3,
                    help="drop groups with fewer labelled sessions than this")
    args = ap.parse_args()

    if not os.path.exists(LABELS_PATH):
        raise SystemExit(f"No manual labels yet at {LABELS_PATH}. Run run_session_sorter.py first.")
    lab = pd.read_csv(LABELS_PATH, dtype={"session_name": str})
    lab = lab[lab["subject"] == args.subject].copy()
    if lab.empty:
        raise SystemExit(f"No labels for {args.subject} yet.")

    # ── 1. TEST-RETEST (the ceiling) ─────────────────────────────────
    both = (lab.groupby("session_name")["group"].agg(list)
               .loc[lambda s: s.map(len) >= 2])
    tr_txt = "test-retest: not enough repeats labelled yet"
    if len(both):
        a = [v[0] for v in both]; b = [v[1] for v in both]
        agree = float(np.mean([x == y for x, y in zip(a, b)]))
        k = cohen_kappa_score(a, b) if len(set(a + b)) > 1 else float("nan")
        tr_txt = (f"test-retest on {len(both)} repeated sessions: "
                  f"raw agreement {agree:.2f}, Cohen's kappa {k:.3f}")
        for s, v in both.items():
            if v[0] != v[1]:
                tr_txt += f"\n    disagreement {s}: {v[0]} vs {v[1]}"

    # one label per session (first presentation) for fitting; drop Unsure
    first = lab.sort_values("presentation_idx").drop_duplicates("session_name", keep="first")
    first = first[first["group"] != "Unsure"]
    feats = build_features(args.subject, first["session_name"].tolist())
    if feats.empty:
        raise SystemExit("No tag CSVs matched the labelled sessions.")
    df = feats.merge(first[["session_name", "group"]], on="session_name")
    df.to_csv(FEATURES_PATH, index=False)

    keep = df["group"].value_counts().loc[lambda s: s >= args.min_per_class].index
    dropped = sorted(set(df["group"]) - set(keep))
    d = df[df["group"].isin(keep)].copy()
    fcols = [c for c in d.columns if c not in ("subject", "session_name", "group")]
    X = d[fcols].fillna(0.0).values
    y = d["group"].astype(str).values

    lines = [f"SESSION-GROUP RULE — {args.subject}",
             f"labelled sessions used: {len(d)}  (groups: {dict(d['group'].value_counts())})"]
    if dropped:
        lines.append(f"groups dropped (< {args.min_per_class} sessions): {dropped}")
    lines += ["", tr_txt, ""]

    # ── 2. LOSO kappa ────────────────────────────────────────────────
    loso = float("nan")
    if len(d) >= 6 and len(set(y)) >= 2:
        preds = []
        for i in range(len(d)):
            tr = np.ones(len(d), bool); tr[i] = False
            if len(set(y[tr])) < 2:
                preds.append(y[tr][0]); continue
            m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3,
                                       class_weight="balanced", random_state=42).fit(X[tr], y[tr])
            preds.append(m.predict(X[i:i + 1])[0])
        loso = cohen_kappa_score(y, preds)
        acc = float(np.mean(np.array(preds) == y))
        lines.append(f"leave-one-session-out: kappa = {loso:.3f}, accuracy = {acc:.2f}")
        lines.append("  (project precedent for the trial-level labeler was LOSO kappa 0.731)")
    else:
        preds = list(y)
        lines.append("too few labelled sessions for LOSO — label more, then re-run.")

    # ── 3. Rules on the full labelled set ────────────────────────────
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3,
                                  class_weight="balanced", random_state=42).fit(X, y)
    lines += ["", "RULES (depth-3 tree fitted on all labelled sessions):",
              export_text(tree, feature_names=fcols)]
    imp = sorted(zip(fcols, tree.feature_importances_), key=lambda t: -t[1])
    lines += ["top features: " + ", ".join(f"{k} {v:.2f}" for k, v in imp[:6] if v > 0), "",
              "SCOPE: groups are behaviour-derived -> clean for NEURAL DVs, CIRCULAR for",
              "behavioural ones. Do not test behavioural DVs that defined the groups."]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RULES_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ── Figure ───────────────────────────────────────────────────────
    labs = sorted(set(y))
    fig = plt.figure(figsize=(15, 5.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.2, 1.5], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    cm = confusion_matrix(y, preds, labels=labs)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted (LOSO)"); ax.set_ylabel("your label")
    ax.set_title(f"A. LOSO confusion — kappa={loso:.2f}", fontweight="bold", loc="left", fontsize=11)

    ax2 = fig.add_subplot(gs[1])
    top = [t for t in imp if t[1] > 0][:8][::-1]
    if top:
        ax2.barh([t[0] for t in top], [t[1] for t in top], color="#6baed6")
    ax2.set_xlabel("tree feature importance")
    ax2.set_title("B. What the rule uses", fontweight="bold", loc="left", fontsize=11)
    ax2.tick_params(axis="y", labelsize=8)

    ax3 = fig.add_subplot(gs[2]); ax3.axis("off")
    ax3.text(0, 1, export_text(tree, feature_names=fcols)[:1400], fontsize=6.5,
             family="monospace", va="top", transform=ax3.transAxes)
    ax3.set_title("C. Rules", fontweight="bold", loc="left", fontsize=11)
    fig.suptitle(f"{args.subject} — learned session-group rule vs manual labels",
                 fontsize=13, fontweight="bold")
    save_figure(fig, "session_group_rule", f"session_sorting/{args.subject}")

    print("\n".join(lines))
    print(f"\nSaved: {RULES_PATH}\n       {FEATURES_PATH}")


if __name__ == "__main__":
    main()
