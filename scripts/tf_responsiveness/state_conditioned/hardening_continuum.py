"""Continuum re-render of the §6 pseudoreplication-hardening figure.

The class figure (`hardening_pseudoreplication.py`) asked whether the
transient->sustained outcome-coupling GAP (a two-class contrast) survives
removing pseudoreplication. The spectrum result
([[tf_transient_sustained_state_jul2026]] / docs 2026-07
transient-sustained-spectrum) showed kernel width is a GRADED axis, not two
classes, so here the target is the WIDTH->COUPLING RELATIONSHIP itself
(Spearman/regression of continuous kernel width `interp_fwhm` on each downstream
coupling metric), and the question is whether that graded relationship is robust
to pseudoreplication rather than whether a class gap is.

Three complementary controls (per outcome: change_on / hit_ramp / fa_ramp), none
both high-confidence and high-coverage on its own:

  A. SESSION RANDOM-INTERCEPT REGRESSION (all 3 mice; full coverage).
     outcome ~ z(interp_fwhm) + C(region), groups=session -> the width slope
     beta_w with a session random intercept, removing within-session correlation
     (the main pseudoreplication source: units don't repeat within a session).
     On NON-CONVERGENCE of the mixed model (getattr(fit, "converged", True) is
     False) we ALSO fit and PREFER a session-cluster-robust OLS
     (cov_type="cluster", groups=session) — both coefficients + a convergence
     flag are emitted so the fallback is self-documenting (the Task-5 lesson from
     the width-vs-waveform independence test). The cluster-robust OLS is computed
     for EVERY outcome regardless, so the two session-level estimates can always
     be compared.

  B. PER-SESSION SPEARMAN + WILCOXON (all mice; SESSION = replication unit).
     For each session with >=5 responsive cells, Spearman(interp_fwhm, outcome);
     a Wilcoxon signed-rank of those per-session rho across sessions tests whether
     the median per-session width->coupling rho is > 0 (a within-session,
     pseudoreplication-free sign test).

  C. TRACKED-UNIT COLLAPSE (BG_046 only; cleanest, lowest n).
     Map each (session, ks_unit_id) -> stable UM/DANT consensus unit via
     data/cache/tracking_consensus/BG_046/consensus_members.csv, collapse to ONE
     value per um_uid (mean interp_fwhm + mean outcome over the sessions where
     that unit was TF-responsive), then Spearman(interp_fwhm, outcome) on the
     collapsed, non-repeating units.

Headline = the pooled Spearman width->coupling relationship compared against each
hardened estimate per outcome. Cache-only: kernel_width_continuous.csv (+ registry
c1_r via load_width_metrics) + consensus_members.csv; no session reloads.

Usage:  py scripts/tf_responsiveness/state_conditioned/hardening_continuum.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy.stats import zscore, spearmanr, wilcoxon

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import (  # noqa: E402
    load_width_metrics, binned_trend, WIDTH, OUTCOMES, REPO,
)
from waveform_celltype_join import _norm_date                       # noqa: E402

OUT = Path(REPO) / "FIGURES/tf_glm_bg046/hardening_continuum"
MEMBERS = Path(f"{REPO}/data/cache/tracking_consensus/BG_046/consensus_members.csv")
XLABEL = "kernel width interp_fwhm (s)"
MIN_CELLS_SESSION = 5   # per-session Spearman needs >=5 cells (task spec)
MIN_SESS_WILCOXON = 6   # Wilcoxon across sessions needs a handful of replicates

# bar-panel palette (one colour per hardening approach)
COL_RAW = "0.6"
COL_SESS = "#238b45"     # session-level model (mixed / cluster-robust)
COL_PSESS = "#08519c"    # per-session Spearman + Wilcoxon
COL_TRK = "#6a51a3"      # tracked-unit collapse


def _clean(d, col):
    """Rows with finite width AND finite outcome (drops +/-inf too)."""
    d2 = (d.dropna(subset=[WIDTH, col])
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[WIDTH, col]).copy())
    return d2[np.isfinite(d2[WIDTH]) & np.isfinite(d2[col])].copy()


# ── A. session random-intercept regression (+ cluster-robust companion) ───────
def session_model(d, col):
    """outcome ~ z(interp_fwhm) + C(region), groups=session.

    Fits the mixed model AND a session-cluster-robust OLS. Returns both width
    slopes/p-values, the convergence flag, and the PREFERRED estimate (mixed if
    it converged, else the cluster-robust OLS — the Task-5 fallback)."""
    d2 = _clean(d, col)
    d2 = d2.copy()
    d2["w"] = zscore(d2[WIDTH].to_numpy(float))
    import statsmodels.formula.api as smf
    mb = mp = np.nan
    conv = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.mixedlm(f"{col} ~ w + C(region)", d2, groups=d2["session"]).fit()
            mb = float(m.params.get("w", np.nan))
            mp = float(m.pvalues.get("w", np.nan))
            conv = bool(getattr(m, "converged", True))
        except Exception as e:  # pragma: no cover — defensive
            print(f"  [mixedlm {col}] failed ({type(e).__name__})", flush=True)
        # cluster-robust OLS is ALWAYS fit so the fallback is self-documenting
        ols = smf.ols(f"{col} ~ w + C(region)", d2).fit(
            cov_type="cluster", cov_kwds={"groups": d2["session"]})
        cb = float(ols.params.get("w", np.nan))
        cp = float(ols.pvalues.get("w", np.nan))
    prefer = "mixed" if conv else "cluster"
    beta, p = (mb, mp) if prefer == "mixed" else (cb, cp)
    return dict(mixed_b=mb, mixed_p=mp, converged=conv,
                clus_b=cb, clus_p=cp, prefer=prefer, beta=beta, p=p,
                n=len(d2), n_sess=int(d2["session"].nunique()))


# ── B. per-session Spearman + Wilcoxon of the per-session rho ─────────────────
def per_session_rho(d, col):
    d2 = _clean(d, col)
    rhos = []
    for _, g in d2.groupby("session"):
        if len(g) >= MIN_CELLS_SESSION:
            r, _ = spearmanr(g[WIDTH], g[col])
            if np.isfinite(r):
                rhos.append(float(r))
    rhos = np.array(rhos)
    p = float(wilcoxon(rhos).pvalue) if len(rhos) >= MIN_SESS_WILCOXON else np.nan
    return rhos, p


# ── C. tracked-unit collapse (BG_046 consensus cohort) ────────────────────────
def tracked_collapse(d):
    """BG_046 responsive cells collapsed to unique UM/DANT consensus um_uid."""
    if not MEMBERS.exists():
        return None
    mem = pd.read_csv(MEMBERS)
    mem["date_key"] = mem["session_key"].map(_norm_date)
    mem["unit"] = mem["ks_unit_id"].astype(int)
    mem = mem[["date_key", "unit", "um_uid"]].drop_duplicates(["date_key", "unit"])
    dd = d[d.subject == "BG_046"].copy()
    dd["date_key"] = [_norm_date(str(s).split("BG_046_", 1)[-1]) for s in dd.session]
    dd = dd.merge(mem, on=["date_key", "unit"], how="inner")
    if not len(dd):
        return None
    agg = dd.groupby("um_uid").agg(
        interp_fwhm=(WIDTH, "mean"),
        change_on=("change_on", "mean"), hit_ramp=("hit_ramp", "mean"),
        fa_ramp=("fa_ramp", "mean"), n_obs=("session", "nunique")).reset_index()
    return dd, agg


def tracked_rho(agg, col):
    gg = agg.dropna(subset=[WIDTH, col])
    if len(gg) < 5:
        return np.nan, np.nan, len(gg)
    r, p = spearmanr(gg[WIDTH], gg[col])
    return float(r), float(p), int(len(gg))


def _nl(p):
    return -np.log10(p) if (p is not None and p == p and p > 0) else 0.0


def main():
    d = load_width_metrics()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    n = int(np.isfinite(d[WIDTH]).sum())
    lines = [
        "Hardening the width->coupling relationship against pseudoreplication "
        "(CONTINUOUS width)",
        "(continuum re-render of the transient/sustained class-gap hardening figure)",
        "",
        f"n cells = {len(d)} (finite width = {n}) | "
        f"DMS={int((d.region=='DMS').sum())} VMS={int((d.region=='VMS').sum())} | "
        f"sessions={int(d.session.nunique())}",
        "target = the graded width->coupling relationship (Spearman/regression of",
        "  continuous interp_fwhm on each outcome), NOT a transient-vs-sustained gap.",
        "",
    ]

    tc = tracked_collapse(d)
    if tc is not None:
        dd_map, agg = tc
        lines.append(
            f"[C tracked-collapse BG_046] {len(dd_map)} responsive cell-sessions -> "
            f"{len(agg)} unique consensus units "
            f"(mean {dd_map.groupby('um_uid').size().mean():.1f} sess/unit)")
    else:
        agg = None
        lines.append("[C tracked-collapse] no consensus overlap")
    lines.append("")

    # per-outcome hardening estimates
    res = {}
    for col, lab in OUTCOMES:
        sm = session_model(d, col)
        rhos, wp = per_session_rho(d, col)
        if agg is not None:
            tr, tp, tn = tracked_rho(agg, col)
        else:
            tr, tp, tn = np.nan, np.nan, 0
        res[col] = dict(sm=sm, rhos=rhos, wp=wp, tr=tr, tp=tp, tn=tn)
        conv_tag = "converged" if sm["converged"] else "NON-CONVERGED->cluster-robust"
        lines.append(f"[{col}]")
        lines.append(
            f"  A mixed(session RE)  beta_w={sm['mixed_b']:+.3f} p={sm['mixed_p']:.2e} "
            f"[{conv_tag}]  (n={sm['n']}, sess={sm['n_sess']})")
        lines.append(
            f"    cluster-robust OLS beta_w={sm['clus_b']:+.3f} p={sm['clus_p']:.2e}  "
            f"| PREFERRED = {sm['prefer']} (beta_w={sm['beta']:+.3f} p={sm['p']:.2e})")
        lines.append(
            f"  B per-session Spearman: median rho={np.median(rhos):+.3f} over "
            f"{len(rhos)} sessions ({100*np.mean(rhos>0):.0f}% positive), "
            f"Wilcoxon p={wp:.2e}")
        lines.append(
            f"  C tracked collapse: Spearman rho={tr:+.3f} p={tp:.2e} (n={tn} units)")
        lines.append("")

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.30,
                           height_ratios=[1.0, 0.95])

    # Row 0: pooled binned_trend per outcome (the relationship being hardened)
    pooled = {}
    for oi, (col, lab) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[0, oi])
        bt = binned_trend(ax, d[WIDTH].to_numpy(float), d[col].to_numpy(float),
                          color=COL_SESS)
        pooled[col] = bt
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(f"{lab} (Hz)")
        ax.set_title(f"{lab}\npooled width->coupling", fontsize=10.5)
        ax.axhline(0, color="0.75", lw=0.8, ls=":")

    # Row 1, col 0: hardening comparison bars (-log10 p per approach)
    axb = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(OUTCOMES))
    w = 0.20
    raw_nl = [_nl(pooled[c]["p"]) for c, _ in OUTCOMES]
    sess_nl = [_nl(res[c]["sm"]["p"]) for c, _ in OUTCOMES]
    ps_nl = [_nl(res[c]["wp"]) for c, _ in OUTCOMES]
    trk_nl = [_nl(res[c]["tp"]) for c, _ in OUTCOMES]
    axb.bar(xs - 1.5 * w, raw_nl, w, label="raw pooled Spearman", color=COL_RAW)
    bars_sess = axb.bar(xs - 0.5 * w, sess_nl, w, label="session RE (mixed/cluster)", color=COL_SESS)
    axb.bar(xs + 0.5 * w, ps_nl, w, label="per-session Wilcoxon", color=COL_PSESS)
    axb.bar(xs + 1.5 * w, trk_nl, w, label="tracked units (BG_046)", color=COL_TRK)
    # tag session bars with M(ixed)/C(luster) so convergence is self-documenting
    for oi, (col, _) in enumerate(OUTCOMES):
        tag = "M" if res[col]["sm"]["prefer"] == "mixed" else "C"
        axb.text(bars_sess[oi].get_x() + bars_sess[oi].get_width() / 2,
                 bars_sess[oi].get_height() + 0.2, tag, ha="center",
                 fontsize=8, color=COL_SESS, fontweight="bold")
    axb.axhline(-np.log10(0.05), color="r", lw=1, ls="--")
    axb.text(len(OUTCOMES) - 1, -np.log10(0.05) + 0.2, "p=0.05", color="r",
             fontsize=7.5, ha="right")
    axb.set_xticks(xs)
    axb.set_xticklabels([o[1] for o in OUTCOMES], fontsize=9)
    axb.set_ylabel("-log10(p)  width->coupling")
    axb.set_title("does hardening break the width->coupling link?\n"
                  "(M=mixed converged, C=cluster-robust fallback)", fontsize=10)
    axb.legend(frameon=False, fontsize=7.6, loc="upper left")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    # Row 1, cols 1-2: stats text
    axt = fig.add_subplot(gs[1, 1:])
    axt.axis("off")
    axt.text(0.0, 1.0, "\n".join(lines), transform=axt.transAxes, va="top",
             ha="left", fontsize=8.4, family="monospace")

    fig.suptitle(
        "Hardening the width->coupling relationship against pseudoreplication "
        "(continuous width)\n"
        "session random-intercept regression (all mice) . per-session Spearman "
        "(session = replication unit) . tracked-unit collapse (BG_046)",
        fontsize=13, y=1.005,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"hardening_continuum.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)

    (OUT / "hardening_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/hardening_continuum.png (+.pdf, +_stats.txt)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
