"""Continuum re-render of the LEARNING-axis transient/sustained figure.

The class figure (`learning_transient_sustained.py`) asked whether the
transient->sustained outcome-coupling GAP (a two-class contrast) tracks learning.
The spectrum result ([[tf_transient_sustained_state_jul2026]] / docs 2026-07
transient-sustained-spectrum) showed kernel width is a GRADED axis, not two
classes, so here the target is the WIDTH->COUPLING RELATIONSHIP itself (Spearman
of continuous kernel width `interp_fwhm` on each downstream coupling metric), and
the learning question becomes: does that graded width->coupling relationship hold
WITHIN each learning stage, and does the per-session width->coupling SLOPE relate
to behavioural d'?

⚠️ CONFOUND (carried from the class figure): units are NOT cross-session tracked,
and there is documented chronic-probe DRIFT (BG_046 broad/SPN% 89->15% across
stages), so cross-stage comparisons of the recorded population mix learning with
drift. Therefore:
  Row 1 (DRIFT-ROBUST) — within-stage: overlay the width->coupling binned trend
     for Learning vs Expert and report the within-stage Spearman(interp_fwhm,
     outcome) for EACH stage. A within-sample graded relationship at each stage is
     drift-robust (drift changes WHICH cells you record, not the within-sample
     width->coupling slope). This is the defensible result.
  Row 2 (CAVEATED) — per-session width->coupling SLOPE (= Spearman(interp_fwhm,
     outcome) over the >=5 responsive cells in a session) vs that session's
     behavioural d'. Cross-session => drift-confounded, so a session-order (chrono,
     the drift proxy) PARTIAL Spearman + slope~order Spearman are reported to probe
     learning-vs-drift.

Stages/d' from data/<SUBJ>_staging_manifest.csv (Naive merged into Learning per the
project SESSION_FILTER); chrono = chronological session order (via _pdate). Stage
is attached per cell EXACTLY as `learning_transient_sustained.attach_stage` (keyed
on the width-cache session `SUBJ_DDMMYYYY`). Cache-only (kernel_width_continuous.csv
+ registry c1_r via load_width_metrics); no session reloads — fast.

Usage:  py scripts/tf_responsiveness/state_conditioned/learning_continuum.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, rankdata

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import (  # noqa: E402
    load_width_metrics, binned_trend, WIDTH, OUTCOMES, REPO,
)
from representative_cells import _pdate                                  # noqa: E402

OUT = Path(REPO) / "FIGURES/tf_glm_bg046/learning_continuum"
XLABEL = "kernel width interp_fwhm (s)"
STAGES = ["Learning", "Expert"]
STAGE_C = {"Learning": "#fdae6b", "Expert": "#31a354"}
REGION_C = {"DMS": "#3474ae", "VMS": "#ef6548"}
MIN_CELLS_SESSION = 5   # per-session slope needs >=5 responsive cells (task spec)
MIN_SESS = 6            # need a handful of sessions for the slope~d' correlation


def attach_stage(cells):
    """Attach stage2 (Naive->Learning) + d_prime + chrono (chronological order) per
    cell — mirrors learning_transient_sustained.attach_stage exactly: per subject
    read the staging manifest, drop qc_fail, merge Naive into Learning, order by
    _pdate(date) for chrono, join on the width-cache session (SUBJ_DDMMYYYY split
    off SUBJ_ and matched to manifest session_name)."""
    frames = []
    for subj in cells.subject.unique():
        man = pd.read_csv(f"{REPO}/data/{subj}_staging_manifest.csv", dtype={"session_name": str})
        man = man[~man.qc_fail.astype(bool)]
        # One BG_031 date (19052025) appears TWICE in the manifest (a base session +
        # a "_b" re-recording, both keyed by the same session_name/date). Without this
        # de-dup the stage-join matches that session's cells to both rows and
        # double-counts them (520 -> 527). The two rows are identical in stage/d',
        # so keeping the first is safe.
        man = man.drop_duplicates(subset="session_name", keep="first")
        man["stage2"] = man["stage"].replace({"Naive": "Learning"})
        man["order"] = man["date"].map(_pdate)
        man = man.sort_values("order").reset_index(drop=True)
        man["chrono"] = np.arange(len(man))
        smap = man.set_index("session_name")[["stage2", "d_prime", "chrono"]]
        c = cells[cells.subject == subj].copy()
        c["date_str"] = [str(s).split(f"{subj}_", 1)[-1] for s in c.session]
        c = c.merge(smap, left_on="date_str", right_index=True, how="left")
        frames.append(c)
    return pd.concat(frames, ignore_index=True)


def partial_spearman(x, y, z):
    """Spearman(x, y) partialling out z — rank all three, regress ranks of x and y
    on ranks of z, correlate residuals (mirrors learning_transient_sustained)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz, dtype=float), rz]

    def resid(a):
        c, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ c
    ex, ey = resid(rx.astype(float)), resid(ry.astype(float))
    return float(np.corrcoef(ex, ey)[0, 1])


def session_slopes(df, col):
    """Per-session width->coupling SLOPE = Spearman(interp_fwhm, outcome) over the
    session's responsive cells (>=MIN_CELLS_SESSION with finite width AND outcome),
    plus that session's d', chrono (drift proxy) and region."""
    rows = []
    for sess, g in df.groupby("session"):
        x = g[WIDTH].to_numpy(float)
        y = g[col].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) >= MIN_CELLS_SESSION:
            r, _ = spearmanr(x[m], y[m])
            if np.isfinite(r):
                rows.append(dict(session=sess, slope=float(r),
                                 d_prime=g.d_prime.iloc[0], chrono=g.chrono.iloc[0],
                                 region=g.region.iloc[0], n=int(m.sum())))
    return pd.DataFrame(rows)


def main():
    d = load_width_metrics()
    cells = attach_stage(d)
    n_join = int(cells.stage2.notna().sum())
    cells = cells.dropna(subset=["stage2"]).copy()

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    stage_ct = cells.groupby("stage2").size().to_dict()
    stage_reg_ct = cells.groupby(["stage2", "region"]).size().to_dict()
    lines = [
        "LEARNING axis on the CONTINUOUS kernel-width->coupling relationship",
        "(continuum re-render of learning_transient_sustained.py; interp_fwhm vs each outcome)",
        "",
        f"cells with stage joined: {n_join}/{len(cells)}  |  "
        f"DMS={int((cells.region=='DMS').sum())} VMS={int((cells.region=='VMS').sum())}",
        f"stage cell counts: {stage_ct}",
        f"stage x region cell counts: {stage_reg_ct}",
        "",
        "ROW 1 (DRIFT-ROBUST) within-stage Spearman(interp_fwhm, outcome):",
    ]

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)

    # ── Row 1: within-stage width->coupling trend, Learning vs Expert overlaid ────
    for j, (col, lab) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[0, j])
        stg_res = {}
        for stg in STAGES:
            sub = cells[cells.stage2 == stg]
            bt = binned_trend(ax, sub[WIDTH].to_numpy(float), sub[col].to_numpy(float),
                              color=STAGE_C[stg], scatter=False, label=stg)
            # drop binned_trend's auto rho text (overlaps for the two overlaid stages)
            if ax.texts:
                ax.texts[-1].remove()
            stg_res[stg] = bt
        # per-stage, stage-coloured within-stage Spearman annotations (drift-robust)
        ypos = 0.97
        for stg in STAGES:
            sub = cells[cells.stage2 == stg]
            xx = sub[WIDTH].to_numpy(float); yy = sub[col].to_numpy(float)
            n_stg = int((np.isfinite(xx) & np.isfinite(yy)).sum())
            r = stg_res[stg]
            ax.text(0.03, ypos, f"{stg}: ρ={r['rho']:+.2f} p={r['p']:.1e} (n={n_stg})",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                    color=STAGE_C[stg], fontweight="bold")
            ypos -= 0.085
            lines.append(f"  [{col:10s}] {stg:8s}: rho={r['rho']:+.3f} p={r['p']:.2e} (n={n_stg})")
        ax.axhline(0, color="0.75", lw=0.8, ls=":")
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(f"{lab} (Hz)")
        ax.set_title(f"{lab}\nwithin-stage width->coupling (drift-robust)", fontsize=10.5)
        if j == 0:
            ax.legend(frameon=False, fontsize=9, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # ── Row 2: per-session width->coupling slope vs session d' (partial | order) ──
    lines.append("")
    lines.append("ROW 2 (CAVEATED, cross-session=drift-confounded) per-session slope vs d':")
    for j, (col, lab) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[1, j])
        sl = session_slopes(cells, col).dropna(subset=["d_prime", "chrono", "slope"]).copy()
        if len(sl) >= MIN_SESS:
            for reg in ("DMS", "VMS"):
                m = sl.region == reg
                ax.scatter(sl.d_prime[m], sl.slope[m], s=45, color=REGION_C[reg],
                           alpha=0.75, edgecolors="none", label=reg)
            rho, p = spearmanr(sl.d_prime, sl.slope)
            pr = partial_spearman(sl.slope.to_numpy(float), sl.d_prime.to_numpy(float),
                                  sl.chrono.to_numpy(float))
            rho_ord, p_ord = spearmanr(sl.chrono, sl.slope)
            b1, b0 = np.polyfit(sl.d_prime.to_numpy(float), sl.slope.to_numpy(float), 1)
            xs = np.linspace(sl.d_prime.min(), sl.d_prime.max(), 20)
            ax.plot(xs, b0 + b1 * xs, color="k", lw=1.5, zorder=1)
            ax.set_title(f"{lab}\nρ(slope,d')={rho:+.2f} p={p:.2f} | partial|order={pr:+.2f} | "
                         f"ρ(slope,order)={rho_ord:+.2f}", fontsize=9)
            lines.append(
                f"  [{col:10s}] slope~d' rho={rho:+.3f} p={p:.2e} (n={len(sl)} sessions); "
                f"partial|chrono={pr:+.3f}; slope~chrono rho={rho_ord:+.3f} p={p_ord:.2e}; "
                f"median slope={sl.slope.median():+.3f}")
            if j == 0:
                ax.legend(frameon=False, fontsize=9, loc="best")
        else:
            ax.text(0.5, 0.5, f"only {len(sl)} sessions\n(need >={MIN_SESS})",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
            lines.append(f"  [{col:10s}] only {len(sl)} sessions (<{MIN_SESS}); skipped")
        ax.axhline(0, color="0.75", lw=0.8, ls=":")
        ax.set_xlabel("session d′")
        ax.set_ylabel("per-session width->coupling slope\n(Spearman ρ, interp_fwhm vs outcome)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle(
        "Learning axis on CONTINUOUS kernel width (interp_fwhm)\n"
        "TOP: does the graded width->coupling relationship hold WITHIN each stage? (drift-robust)   ||   "
        "BOTTOM: per-session width->coupling slope vs d'  (caution: cross-session = drift-confounded; "
        "partial | session-order shown)",
        fontsize=12.5, y=1.005,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"learning_continuum.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)

    (OUT / "learning_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/learning_continuum.png (+.pdf, +_stats.txt)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
