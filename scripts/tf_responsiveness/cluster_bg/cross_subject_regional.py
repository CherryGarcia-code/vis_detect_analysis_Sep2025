"""Region-grouped cross-subject TF-responsive summary from the per-subject
registries. Honors the pooling rule: DMS = {BG_046, BG_039} shown together,
VMS = {BG_031} separate; NO cell pooling across sessions/subjects (bars are
subject-level pooled %, dots are per-session %), pending bank-position audit.

Usage:
  py cross_subject_regional.py <registry_dir> <out_png>
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGION_COLOR = {"DMS": "#3474ae", "VMS": "#ef6548"}


def main():
    reg_dir, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(str(Path(reg_dir) / "bg*_tf_responsive.csv")))
    df = pd.concat([pd.read_csv(f, dtype={"session": str}) for f in files], ignore_index=True)
    df["resp_log2"] = df["resp_log2"].astype(str).str.lower().isin(["true", "1", "1.0"])

    # order: DMS subjects first, then VMS; within region by subject name
    subj_region = df.groupby("subject")["region"].first()
    order = sorted(subj_region.index, key=lambda s: (subj_region[s] != "DMS", s))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- per-subject responsive %, grouped/colored by region, + per-session dots
    for i, subj in enumerate(order):
        d = df[df.subject == subj]
        reg = subj_region[subj]
        pooled = 100 * d.resp_log2.mean()
        axL.bar(i, pooled, color=REGION_COLOR.get(reg, "0.5"), alpha=0.85, width=0.7)
        # per-session %
        per_sess = d.groupby("session")["resp_log2"].mean().values * 100
        axL.scatter(np.full(len(per_sess), i) + np.linspace(-0.18, 0.18, len(per_sess)),
                    per_sess, s=10, color="0.2", alpha=0.5, zorder=3)
        axL.text(i, pooled, f"{pooled:.1f}%\n{int(d.resp_log2.sum())}/{len(d)}",
                 ha="center", va="bottom", fontsize=8)
    axL.set_xticks(range(len(order)))
    axL.set_xticklabels([f"{s}\n({subj_region[s]})" for s in order])
    axL.set_ylabel("% TF-responsive (log2, C1+C2)")
    axL.set_title("TF-responsive fraction by subject (dots = per-session)\n"
                  "DMS vs VMS kept separate — no cross-region pooling")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REGION_COLOR.values()]
    axL.legend(handles, REGION_COLOR.keys(), frameon=False, title="region")
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # ---- region-level (subject-averaged, NOT cell-pooled) with the caveat
    axR.axis("off")
    lines = ["Region summary (subject-level; cells NOT pooled — bank audit pending)\n"]
    for reg in ("DMS", "VMS"):
        subs = [s for s in order if subj_region[s] == reg]
        d = df[df.subject.isin(subs)]
        subj_pcts = [100 * df[df.subject == s].resp_log2.mean() for s in subs]
        lines.append(f"{reg}: subjects {', '.join(subs)}")
        lines.append(f"   per-subject %: {', '.join(f'{p:.1f}' for p in subj_pcts)}")
        lines.append(f"   n_units={len(d)}, n_resp={int(d.resp_log2.sum())} "
                     f"(pooled {100*d.resp_log2.mean():.1f}% — provisional)")
        lines.append("")
    lines.append("Caveats: no-movement GLM; region_bank_confirmed=False for all")
    lines.append("(chronic-probe drift → confirm recorded bank/depth per session")
    lines.append("before pooling cells within a region).")
    axR.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10,
             family="monospace", transform=axR.transAxes)

    fig.suptitle(f"TF-responsive cells across striatal subjects "
                 f"(n={df.subject.nunique()} mice, {df.session.nunique()} sessions, "
                 f"{len(df)} units)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for reg in ("DMS", "VMS"):
        subs = [s for s in order if subj_region[s] == reg]
        d = df[df.subject.isin(subs)]
        print(f"  {reg}: {', '.join(subs)} | {len(d)} units | "
              f"{100*d.resp_log2.mean():.1f}% resp (provisional pool)")


if __name__ == "__main__":
    main()
