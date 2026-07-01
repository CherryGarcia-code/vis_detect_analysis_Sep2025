"""Combined TALK figure: population geometry of TF encoding vs behavioural state
across mice/regions (Lohse 2025 Fig 3 replication). Three state-space panels
(BG_046 DMS, BG_039 DMS, BG_031 VMS) sharing axes + one orthogonality-summary
panel. Recomputes per-session geometry (fast) since only the per-mouse figures,
not the projection coordinates, were cached.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import glob
import concurrent.futures as cf
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_bg046/scripts/tf_responsiveness/state_conditioned")
from population_geometry import (session_geometry, qc_sessions, STATES, STATE_COLORS,  # noqa: E402
                                 PKL_DIR, SEED)

SUBJECTS = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/population_geometry")
try:
    from visdetect.viz.plotting import set_style
    set_style("talk")
except Exception:
    pass
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})


def run_subject(subj, qc=False):
    sess = sorted(Path(p).stem for p in glob.glob(str(Path(PKL_DIR.format(subj=subj)) / "*.pkl")))
    if qc:
        from visdetect.analysis import config
        keep = qc_sessions(subj)
        if keep is not None:
            sess = [s for s in sess
                    if config.canonical_session_id(s.replace(f"{subj}_", "", 1)) in keep]
    res = []
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(session_geometry, [(subj, s) for s in sess]):
            if r is not None:
                res.append(r)
    return res


def state_space_panel(ax, res, subj, region, show_legend=False):
    keys = [(st, tf) for st in STATES for tf in ("fast", "slow")]
    agg = {k: np.array([r["proj"][f"{k[0]}|{k[1]}"] for r in res
                        if np.isfinite(r["proj"][f"{k[0]}|{k[1]}"][0])]) for k in keys}
    rng = np.random.default_rng(SEED)
    for st in STATES:
        for tf, fc in (("fast", None), ("slow", "white")):
            v = agg[(st, tf)]
            if len(v) < 2:
                continue
            m = v.mean(0)
            boot = np.array([v[rng.integers(0, len(v), len(v))].mean(0) for _ in range(1000)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            ax.errorbar(m[0], m[1], xerr=[[m[0]-lo[0]], [hi[0]-m[0]]], yerr=[[m[1]-lo[1]], [hi[1]-m[1]]],
                        fmt="o", ms=12, color=STATE_COLORS[st],
                        markerfacecolor=(fc or STATE_COLORS[st]), markeredgecolor=STATE_COLORS[st],
                        mew=2, capsize=3, elinewidth=1.1, zorder=3,
                        label=(f"{st} {tf}" if show_legend else None))
        vf, vs = agg[(st, "fast")], agg[(st, "slow")]
        if len(vf) >= 2 and len(vs) >= 2:
            ax.plot([vf.mean(0)[0], vs.mean(0)[0]], [vf.mean(0)[1], vs.mean(0)[1]],
                    color=STATE_COLORS[st], lw=1.2, alpha=0.55, zorder=2)
    ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
    ax.set_title(f"{subj}  ({region})", fontweight="bold")
    ax.set_xlabel("Sensory axis (z)\nfast > slow TF")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="center left", bbox_to_anchor=(0.0, 0.28))
    return np.median([r["cosine"] for r in res]), res


def main():
    import argparse
    qc = argparse.ArgumentParser(); qc.add_argument("--qc", action="store_true")
    a = qc.parse_args()
    suffix = "_qc" if a.qc else ""
    data = {subj: run_subject(subj, qc=a.qc) for subj, _ in SUBJECTS}
    fig = plt.figure(figsize=(19, 5.6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # shared y-limits across the three state-spaces
    for i, (subj, region) in enumerate(SUBJECTS):
        state_space_panel(axes[i], data[subj], subj, region, show_legend=(i == 0))
    ylims = [ax.get_ylim() for ax in axes]; xl = [ax.get_xlim() for ax in axes]
    ylo, yhi = min(y[0] for y in ylims), max(y[1] for y in ylims)
    xlo, xhi = min(x[0] for x in xl), max(x[1] for x in xl)
    for ax in axes:
        ax.set_ylim(ylo, yhi); ax.set_xlim(xlo, xhi)
    axes[0].set_ylabel("Task-state axis (z)\nengaged > disengaged")

    # orthogonality summary panel
    axo = fig.add_subplot(gs[0, 3])
    nl = np.median([r["cosine_null_mean"] - 2 * r["cosine_null_sd"]
                    for res in data.values() for r in res])
    nh = np.median([r["cosine_null_mean"] + 2 * r["cosine_null_sd"]
                    for res in data.values() for r in res])
    axo.axhspan(nl, nh, color="0.88", label="shuffle null (±2 SD)")
    axo.axhline(0, color="0.5", lw=0.8, ls=":")
    rng = np.random.default_rng(SEED)
    for i, (subj, region) in enumerate(SUBJECTS):
        cos = np.array([r["cosine"] for r in data[subj]])
        col = "#3474ae" if region == "DMS" else "#ef6548"
        jit = i + (rng.random(len(cos)) - 0.5) * 0.5
        axo.scatter(jit, cos, s=20, color=col, alpha=0.55, zorder=3)
        axo.hlines(np.median(cos), i - 0.3, i + 0.3, color="k", lw=2.5, zorder=4)
        axo.text(i, 0.42, f"{np.median(cos):+.3f}", ha="center", fontsize=10)
    axo.set_xticks(range(3))
    axo.set_xticklabels([f"{s}\n({r})" for s, r in SUBJECTS], fontsize=10)
    axo.set_ylabel("cosine(Sensory, Task-state)")
    axo.set_ylim(-0.5, 0.5)
    axo.set_title("Axes are orthogonal\n(cosine ≈ 0, per session)", fontweight="bold")
    axo.legend(frameon=False, fontsize=8, loc="lower center")
    for sp in ("top", "right"):
        axo.spines[sp].set_visible(False)

    fig.suptitle("Striatal TF encoding and behavioural engagement occupy ORTHOGONAL population axes\n"
                 "fast/slow separate on the sensory axis (preserved across states) · "
                 "disengagement DISPLACES the population down the task-state axis",
                 fontsize=13.5, y=1.06)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"combined_population_geometry{suffix}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/combined_population_geometry{suffix}.png (+ .pdf)"
          + ("  [QC-passing sessions only]" if a.qc else ""))
    for subj, region in SUBJECTS:
        cos = [r["cosine"] for r in data[subj]]
        print(f"  {subj} ({region}): {len(cos)} sessions, median cosine {np.median(cos):+.3f}")


if __name__ == "__main__":
    main()
