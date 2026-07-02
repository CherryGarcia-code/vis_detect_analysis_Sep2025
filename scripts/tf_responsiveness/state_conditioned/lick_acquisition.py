"""Do TF-responsive cells (transient/sustained) also carry the codebase's canonical
FA-lick responsiveness, and does that overlap change across learning?

Lick-responsive = `visdetect.analysis.lick.compute_fa_lick_responsiveness`
(is_significant): FA/early-lick aligned, paired Wilcoxon of baseline (-1.75,-1.25)
vs pre-lick (-0.3,-0.15) s, p<0.05 — the SAME windows as the FA motor ramp, but the
canonical significance label. TF-responsive (transient/sustained by kernel_fwhm) is
defined from the TF-pulse GLM, INDEPENDENT of lick — so this is a genuine cross-tab
of two independent responsiveness criteria.

Per good_dates session (with a manifest stage; Naive merged into Learning): compute
lick-responsiveness for the good/stable units, join to the TF registry -> each unit
is {transient, sustained, intermediate, nonTF} x {lick-resp, not}. Then:
  (1) enrichment: P(lick-resp | class) — are sustained cells the lick cells?
  (2) reverse: P(TF-resp | lick-resp) — same population or different?
  (3) LEARNING: P(lick-resp | transient/sustained) Learning vs Expert — does the
      overlap grow (=acquisition, inferential without tracking; drift-confounded)?

⚠️ No cross-session tracking, so "acquire" = a fraction change across stages in
DIFFERENT neurons, confounded by chronic-probe drift. One session load each.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import gc
import argparse
import concurrent.futures as cf
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import chi2_contingency, fisher_exact

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO, _registry, good_dates, _pdate, load_session  # noqa: E402
from transient_vs_sustained import load_cells, NARROW, BROAD, TCOL, SCOL       # noqa: E402
from visdetect.analysis.lick import compute_fa_lick_responsiveness, MatlabLickConfig  # noqa: E402

CLS_COL = {"transient": TCOL, "sustained": SCOL, "intermediate": "#9e9e9e", "nonTF": "#d9d9d9"}
ORDER = ["nonTF", "transient", "intermediate", "sustained"]
OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/lick_acquisition")
CACHE = OUT / "lick_acquisition_cells.csv"


def _stage_map(subj):
    man = pd.read_csv(f"{REPO}/data/{subj}_staging_manifest.csv", dtype={"session_name": str})
    man = man[~man.qc_fail.astype(bool)]
    man["stage2"] = man["stage"].replace({"Naive": "Learning"})
    return man.set_index("session_name")["stage2"].to_dict()


def _class(resp, fwhm):
    if not resp:
        return "nonTF"
    if fwhm <= NARROW:
        return "transient"
    if fwhm >= BROAD:
        return "sustained"
    return "intermediate"


def session_rows(subj, sess, reg_s, stage):
    s = load_session(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
    gs = list(getattr(s, "good_and_stable_ids", None) or getattr(s, "good_cluster_ids", None) or [])
    if len(gs) < 8:
        del s; gc.collect(); return []
    res = compute_fa_lick_responsiveness(s, cfg=MatlabLickConfig(), good_ids=set(int(u) for u in gs))
    del s; gc.collect()
    tab = res.table
    if tab is None or tab.empty:
        return []
    lick = {int(r.cluster_id): (bool(r.is_significant), int(getattr(r, "n_events", 0)))
            for r in tab.itertuples()}
    rows = []
    for _, r in reg_s.iterrows():
        uid = int(r["unit"])
        if uid not in lick:
            continue
        sig, nev = lick[uid]
        rows.append(dict(subject=subj, session=sess, stage=stage, unit=uid,
                         cls=_class(bool(r["resp"]), float(r["kernel_fwhm"])),
                         lick_sig=sig, n_events=nev))
    return rows


def _worker(args):
    """Picklable per-session worker for the ProcessPool (BLAS pinned to 1/worker
    via the module-level os.environ.setdefault, re-run on child import)."""
    subj, sess, reg_s, stage = args
    try:
        rows = session_rows(subj, sess, reg_s, stage)
        print(f"  {subj}/{sess} [{stage}]: {len(reg_s)} reg units -> {len(rows)}", flush=True)
        return rows
    except Exception as e:
        print(f"  [FAIL] {subj}/{sess}: {e}", flush=True)
        return []


def compute_or_load(force=False, n_workers=10):
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE)
    tasks = []
    for subj in ["BG_046", "BG_039", "BG_031"]:
        reg = _registry(subj)
        gd = good_dates(subj)
        smap = _stage_map(subj)
        reg = reg[reg.session_date.isin(gd)]
        for sess, reg_s in reg.groupby("session"):
            date = str(sess).split(f"{subj}_", 1)[-1]
            stage = smap.get(date)
            if stage is None or not Path(f"{REPO}/data/pkls/{subj}/{sess}.pkl").exists():
                continue
            tasks.append((subj, sess, reg_s.copy(), stage))
    print(f"lick-responsiveness over {len(tasks)} sessions on {n_workers} workers", flush=True)
    allrows = []
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for rows in ex.map(_worker, tasks):
            allrows += rows
    df = pd.DataFrame(allrows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def _frac(df, cls, stage=None):
    d = df[df.cls == cls]
    if stage:
        d = d[d.stage == stage]
    return (d.lick_sig.mean() * 100 if len(d) else np.nan), len(d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    df = compute_or_load(force=a.force, n_workers=a.workers)
    df = df[df.n_events >= 10]      # need enough FA events for a meaningful lick test
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})
    lines = [f"n cells (>=10 FA events) = {len(df)}; "
             + str(df.cls.value_counts().to_dict())]

    fig = plt.figure(figsize=(17, 5.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.32)

    # (1) enrichment: P(lick-resp | class)
    ax1 = fig.add_subplot(gs[0, 0])
    fr, ns = [], []
    for c in ORDER:
        f, n = _frac(df, c); fr.append(f); ns.append(n)
    ax1.bar(range(len(ORDER)), fr, color=[CLS_COL[c] for c in ORDER])
    for i, (f, n) in enumerate(zip(fr, ns)):
        ax1.text(i, f + 1, f"{f:.0f}%\nn{n}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(len(ORDER))); ax1.set_xticklabels(ORDER, fontsize=9, rotation=15)
    ax1.set_ylabel("% lick-responsive"); ax1.set_ylim(0, max(fr) * 1.25 if fr else 100)
    ax1.set_title("(1) lick-responsiveness by TF class", fontsize=11, fontweight="bold")
    # sustained vs transient Fisher
    tab = pd.crosstab(df[df.cls.isin(["transient", "sustained"])].cls,
                      df[df.cls.isin(["transient", "sustained"])].lick_sig)
    try:
        orr, pf = fisher_exact(tab.reindex(index=["transient", "sustained"]).values)
        lines.append(f"[enrichment] P(lick|transient)={fr[1]:.1f}% P(lick|sustained)={fr[3]:.1f}% "
                     f"P(lick|nonTF)={fr[0]:.1f}%; sustained-vs-transient Fisher OR={orr:.2f} p={pf:.2e}")
    except Exception:
        pass

    # (2) reverse: composition of lick-responsive vs non-lick cells
    ax2 = fig.add_subplot(gs[0, 1])
    for xi, grp in enumerate([("lick-resp", df[df.lick_sig]), ("not", df[~df.lick_sig])]):
        nm, d = grp
        comp = d.cls.value_counts(normalize=True).reindex(ORDER).fillna(0)
        bottom = 0
        for c in ORDER:
            ax2.bar(xi, comp[c], bottom=bottom, color=CLS_COL[c], label=(c if xi == 0 else None))
            bottom += comp[c]
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["lick-resp", "not lick-resp"], fontsize=9)
    ax2.set_ylabel("fraction"); ax2.set_title("(2) TF composition of lick cells", fontsize=11, fontweight="bold")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    p_tf_given_lick = 100 * df[df.lick_sig].cls.isin(["transient", "sustained", "intermediate"]).mean()
    lines.append(f"[reverse] P(TF-responsive | lick-responsive) = {p_tf_given_lick:.1f}% "
                 f"(so {'largely separate' if p_tf_given_lick < 40 else 'substantial overlap'})")

    # (3) learning: P(lick-resp | class) by stage
    ax3 = fig.add_subplot(gs[0, 2])
    w = 0.35
    for si, stg in enumerate(["Learning", "Expert"]):
        vals, nsx = [], []
        for c in ["transient", "sustained"]:
            f, n = _frac(df, c, stg); vals.append(f); nsx.append(n)
        xs = np.arange(2) + (si - 0.5) * w
        ax3.bar(xs, vals, w, color=("#fdae6b" if stg == "Learning" else "#31a354"), label=stg)
        for x, f, n in zip(xs, vals, nsx):
            if np.isfinite(f):
                ax3.text(x, f + 1, f"{f:.0f}%\nn{n}", ha="center", va="bottom", fontsize=7.5)
    ax3.set_xticks(range(2)); ax3.set_xticklabels(["transient", "sustained"], fontsize=9)
    ax3.set_ylabel("% lick-responsive"); ax3.set_title("(3) acquisition across learning?", fontsize=11, fontweight="bold")
    ax3.legend(frameon=False, fontsize=9)
    for c in ["transient", "sustained"]:
        for stg in ["Learning", "Expert"]:
            f, n = _frac(df, c, stg)
            lines.append(f"[learning] P(lick|{c},{stg})={f:.1f}% (n={n})")
        # Fisher Learning vs Expert within class
        d = df[df.cls == c]
        ct = pd.crosstab(d.stage, d.lick_sig).reindex(index=["Learning", "Expert"])
        try:
            orr, pf = fisher_exact(ct.values)
            lines.append(f"[learning] {c}: Expert-vs-Learning lick-resp Fisher OR={orr:.2f} p={pf:.2e}")
        except Exception:
            pass

    for ax in (ax1, ax2, ax3):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Do TF-responsive (transient/sustained) cells carry canonical FA-lick responsiveness, and does it grow across learning?",
                 fontsize=12.5, y=1.04)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lick_acquisition.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (OUT / "lick_acquisition_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/lick_acquisition.png")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
