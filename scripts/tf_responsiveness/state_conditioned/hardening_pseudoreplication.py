"""Hardening pass — does the transient->sustained outcome-coupling gap survive
removing PSEUDOREPLICATION (per-session GLM fits treat one physical neuron
recorded across sessions as many independent points)?

Three complementary controls, because no single one is both high-confidence and
high-coverage:

  A. TRACKED-UNIT COLLAPSE (BG_046 only; the cleanest, lowest n).
     Map each (session, cluster_id) -> stable UM∩DANT consensus unit via
     data/cache/tracking_consensus/BG_046/consensus_members.csv (312 tracked
     units, both trackers agree). Collapse to ONE value per um_uid (mean over
     the sessions where that unit was TF-responsive); reassign transient/
     sustained from the averaged kernel_fwhm; re-test.

  B. SESSION RANDOM-INTERCEPT MIXED MODEL (all 3 mice; full coverage).
     outcome ~ is_sustained + C(region), groups=session. Removes within-session
     correlation (the main pseudoreplication source given units don't repeat
     within a session). Falls back to a per-session sign test if statsmodels
     is unavailable / fails to converge.

  C. PER-SESSION SIGN TEST (all mice). Per session with >=2 cells in each class,
     Δ = median(sustained) - median(transient); Wilcoxon signed-rank across
     sessions treats SESSION as the replication unit.

Headline = compare RAW MWU p vs each hardened p per outcome.
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
from scipy.stats import mannwhitneyu, wilcoxon

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO                                    # noqa: E402
from transient_vs_sustained import load_cells, OUTCOMES, TCOL, SCOL, NARROW, BROAD  # noqa: E402
from waveform_celltype_join import _norm_date                           # noqa: E402

OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/hardening_pseudoreplication")
MEMBERS = Path(f"{REPO}/data/cache/tracking_consensus/BG_046/consensus_members.csv")
MIN_PER_CLASS = 2


def _mwu(a, b):
    a = pd.Series(a).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(b).replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, len(a), len(b), np.nan
    return float(a.median()), float(b.median()), len(a), len(b), float(mannwhitneyu(a, b).pvalue)


def tracked_collapse(cells):
    """BG_046 responsive cells collapsed to unique consensus um_uid."""
    if not MEMBERS.exists():
        return None
    mem = pd.read_csv(MEMBERS)
    mem["date_key"] = mem["session_key"].map(_norm_date)
    mem["unit"] = mem["ks_unit_id"].astype(int)
    mem = mem[["date_key", "unit", "um_uid"]].drop_duplicates(["date_key", "unit"])
    d = cells[cells.subject == "BG_046"].copy()
    d["date_key"] = [_norm_date(str(s).split("BG_046_", 1)[-1]) for s in d.session]
    d = d.merge(mem, on=["date_key", "unit"], how="inner")
    if not len(d):
        return None
    agg = d.groupby("um_uid").agg(
        kernel_fwhm=("kernel_fwhm", "mean"),
        change_on=("change_on", "mean"), hit_ramp=("hit_ramp", "mean"),
        fa_ramp=("fa_ramp", "mean"), n_obs=("session", "nunique")).reset_index()
    agg["class"] = np.where(agg.kernel_fwhm <= NARROW, "transient",
                            np.where(agg.kernel_fwhm >= BROAD, "sustained", "intermediate"))
    return d, agg


def mixed_model_p(cells, col):
    d = cells[cells["class"].isin(["transient", "sustained"])].copy()
    d = d.dropna(subset=[col]).replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
    d["is_sustained"] = (d["class"] == "sustained").astype(float)
    try:
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(f"{col} ~ is_sustained + C(region)", d, groups=d["session"]).fit(method="lbfgs")
        return float(m.pvalues.get("is_sustained", np.nan)), float(m.params.get("is_sustained", np.nan)), "mixedlm"
    except Exception as e:
        print(f"  [mixedlm {col}] fallback ({type(e).__name__})", flush=True)
        return np.nan, np.nan, "failed"


def per_session_delta(cells, col):
    d = cells[cells["class"].isin(["transient", "sustained"])].dropna(subset=[col])
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
    deltas = []
    for sess, g in d.groupby("session"):
        t = g[g["class"] == "transient"][col]
        s = g[g["class"] == "sustained"][col]
        if len(t) >= MIN_PER_CLASS and len(s) >= MIN_PER_CLASS:
            deltas.append(s.median() - t.median())
    deltas = np.array(deltas)
    if len(deltas) < 6:
        return deltas, np.nan
    return deltas, float(wilcoxon(deltas).pvalue)


def main():
    cells = load_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})
    lines = []

    tc = tracked_collapse(cells)
    if tc is not None:
        d_map, agg = tc
        n_cs = len(d_map)
        n_u = len(agg)
        lines.append(f"[A tracked-collapse BG_046] {n_cs} responsive cell-sessions -> {n_u} unique "
                     f"consensus units (mean {d_map.groupby('um_uid').size().mean():.1f} sess/unit); "
                     f"classes: {agg['class'].value_counts().to_dict()}")
    else:
        agg = None
        lines.append("[A tracked-collapse] no consensus overlap")

    # comparison table: raw vs mixed vs tracked, per outcome
    raw_p, mix_p, mix_b, trk_p = {}, {}, {}, {}
    for col, _ in OUTCOMES:
        _, _, _, _, pr = _mwu(cells.loc[cells["class"] == "transient", col],
                              cells.loc[cells["class"] == "sustained", col])
        raw_p[col] = pr
        pm, bm, tag = mixed_model_p(cells, col)
        mix_p[col], mix_b[col] = pm, bm
        if agg is not None:
            mt, ms, nt, ns, pt = _mwu(agg.loc[agg["class"] == "transient", col],
                                      agg.loc[agg["class"] == "sustained", col])
            trk_p[col] = pt
            lines.append(f"[{col}] RAW p={pr:.2e} | MIXED(session RE) p={pm:.2e} (β={bm:+.2f}Hz) | "
                         f"TRACKED p={pt if pt==pt else float('nan'):.2e} (t={mt}/s={ms}, n {nt}/{ns})")
        else:
            trk_p[col] = np.nan
            lines.append(f"[{col}] RAW p={pr:.2e} | MIXED p={pm:.2e} (β={bm:+.2f}Hz) | TRACKED n/a")

    fig = plt.figure(figsize=(18, 9.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.30)

    # Panel A: tracked-collapse strip (unique units)
    axa = fig.add_subplot(gs[0, 0])
    if agg is not None:
        for oi, (col, labn) in enumerate(OUTCOMES):
            for si, cls in enumerate(("transient", "sustained")):
                v = agg.loc[agg["class"] == cls, col].dropna()
                xc = oi + (si - 0.5) * 0.4
                jit = (np.random.default_rng(si).random(len(v)) - 0.5) * 0.18
                axa.scatter(np.full(len(v), xc) + jit, v, s=22,
                            color=(TCOL if cls == "transient" else SCOL), alpha=0.7, edgecolors="none")
                if len(v):
                    axa.hlines(np.median(v), xc - 0.18, xc + 0.18, color="k", lw=2.2, zorder=5)
            axa.text(oi, axa.get_ylim()[1] * 0.9 if oi else 8, f"p={trk_p[col]:.1e}", ha="center", fontsize=8)
        axa.axhline(0, color="0.7", lw=0.8, ls=":")
        axa.set_xticks(range(len(OUTCOMES))); axa.set_xticklabels([o[1] for o in OUTCOMES], fontsize=9)
        axa.set_ylabel("Δ firing (Hz), per tracked unit")
        n_u = len(agg[agg["class"].isin(["transient", "sustained"])])
        axa.set_title(f"A. tracked-unit collapse (BG_046)\n{n_u} unique consensus units", fontsize=10.5)
    from matplotlib.lines import Line2D
    axa.legend(handles=[Line2D([0], [0], marker="o", ls="", color=TCOL, label="transient"),
                        Line2D([0], [0], marker="o", ls="", color=SCOL, label="sustained")],
               frameon=False, fontsize=8, loc="upper left")

    # Panel B: p comparison
    axb = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(OUTCOMES)); w = 0.26

    def _nl(p):
        return -np.log10(p) if (p is not None and p == p and p > 0) else 0
    axb.bar(xs - w, [_nl(raw_p[c]) for c, _ in OUTCOMES], w, label="raw (cell-sessions)", color="0.6")
    axb.bar(xs, [_nl(mix_p[c]) for c, _ in OUTCOMES], w, label="mixed (session RE)", color="#238b45")
    axb.bar(xs + w, [_nl(trk_p[c]) for c, _ in OUTCOMES], w, label="tracked units", color="#6a51a3")
    axb.axhline(-np.log10(0.05), color="r", lw=1, ls="--")
    axb.text(len(OUTCOMES) - 1, -np.log10(0.05) + 0.15, "p=0.05", color="r", fontsize=7.5, ha="right")
    axb.set_xticks(xs); axb.set_xticklabels([o[1] for o in OUTCOMES], fontsize=9)
    axb.set_ylabel("-log10(p) transient vs sustained")
    axb.set_title("B. does hardening change it?", fontsize=10.5)
    axb.legend(frameon=False, fontsize=8)

    # Panel C: per-session Δ (all mice)
    axc = fig.add_subplot(gs[0, 2])
    for oi, (col, labn) in enumerate(OUTCOMES):
        deltas, pw = per_session_delta(cells, col)
        jit = (np.random.default_rng(oi).random(len(deltas)) - 0.5) * 0.2
        axc.scatter(np.full(len(deltas), oi) + jit, deltas, s=16, color="#08519c", alpha=0.5, edgecolors="none")
        if len(deltas):
            axc.hlines(np.median(deltas), oi - 0.2, oi + 0.2, color="k", lw=2.2, zorder=5)
        axc.text(oi, axc.get_ylim()[1] * 0.88 if oi else 10, f"p={pw:.1e}\nn={len(deltas)}", ha="center", fontsize=7.5)
        lines.append(f"[C per-session Δ] {col}: median Δ(sus-tra)={np.median(deltas):+.2f}Hz over "
                     f"{len(deltas)} sessions, Wilcoxon p={pw if pw==pw else float('nan'):.2e}")
    axc.axhline(0, color="0.7", lw=0.8, ls=":")
    axc.set_xticks(range(len(OUTCOMES))); axc.set_xticklabels([o[1] for o in OUTCOMES], fontsize=9)
    axc.set_ylabel("per-session Δ median (Hz)\nsustained − transient")
    axc.set_title("C. per-session sign test (all mice)", fontsize=10.5)

    # Panel D: stats text
    axd = fig.add_subplot(gs[1, :]); axd.axis("off")
    axd.text(0.0, 1.0, "\n".join(lines), transform=axd.transAxes, va="top", ha="left",
             fontsize=8.8, family="monospace")

    for ax in (axa, axb, axc):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Hardening the transient→sustained finding against pseudoreplication\n"
                 "tracked-unit collapse (BG_046) · session random-intercept (all mice) · per-session sign test",
                 fontsize=13, y=1.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"hardening_pseudoreplication.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)
    (OUT / "hardening_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/hardening_pseudoreplication.png (+.pdf)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
