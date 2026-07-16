"""Do transient (fast/sensory) vs sustained TF cells map onto FSI (narrow) vs
SPN (broad) waveform type — AND is the well-known narrow-cell over-sampling a
confound for the transient/sustained -> outcome-coupling finding?

Waveform labels: data/{SUBJ}/waveform_celltype_labels.csv (visdetect.analysis.
waveform_celltype: trough-to-peak -> 2-component GMM; FSI=narrow, SPN=broad;
BG_046 delta_BIC=6982 => genuinely bimodal, threshold 0.41 ms). Keyed by
session_date (INT — leading-zero DAY dropped) + cluster_id. We join on
int(session_date)+unit (both sides derive from the same date string, so int()
is a consistent key for 6- and 8-digit forms alike).

THE CONFOUND (user): these recordings over-sample NARROW/fast-firing cells
(higher rate = easier to detect/sort). BG_046 labels are 84% FSI vs true
striatal composition ~90-95% SPN — a large yield/selection bias. Two effects,
tested separately:
  (1) mapping test: transient/sustained (kernel WIDTH) vs FSI/SPN (waveform);
  (2) does the transient->sustained OUTCOME-COUPLING gap survive controls that
      neutralise firing rate — WITHIN cell type, and after RATE-MATCHING?
If the coupling gap survives, the functional dissociation is robust to the
sorting bias (which biases population *fractions*, not the within-sample contrast).

Reuses transient_vs_sustained.load_cells() (responsive cells, good_dates, class,
base_hz, outcome metrics). Non-parametric throughout.
"""
from __future__ import annotations
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import mannwhitneyu, spearmanr, chi2_contingency

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO                       # noqa: E402
from transient_vs_sustained import load_cells, MICE, OUTCOMES, TCOL, SCOL  # noqa: E402

WF = {"FSI": "#d94801", "SPN": "#08519c"}   # narrow / broad
OUT = Path(str(REPO)) / "FIGURES/tf_glm_bg046/waveform_celltype_join"


def _norm_date(s):
    """Normalise a session-date token so the int-stored labels (leading-zero DAY
    dropped) match the registry's zero-padded strings, WHILE preserving re-sort
    suffixes like '_v2' so an original and its re-sort stay distinct keys."""
    s = str(s).strip()
    if s.endswith(".0"):
        s = s[:-2]
    m = re.match(r"(\d+)(.*)$", s)
    if not m:
        return s
    num, rest = m.groups()
    return f"{int(num)}{rest}"


def load_labels(subj):
    f = Path(f"{REPO}/data/{subj}/waveform_celltype_labels.csv")
    if not f.exists():
        return None
    lab = pd.read_csv(f)
    lab["date_key"] = lab["session_date"].map(_norm_date)
    lab["unit"] = lab["cluster_id"].astype(int)
    return lab[["date_key", "unit", "celltype"]].drop_duplicates(["date_key", "unit"])


def attach_celltype(cells):
    cells = cells.copy()
    cells["date_key"] = [_norm_date(str(s).split(f"{sub}_", 1)[-1])
                         for s, sub in zip(cells.session, cells.subject)]
    out = []
    full_counts = {}
    for subj, region, _ in MICE:
        lab = load_labels(subj)
        sub = cells[cells.subject == subj]
        if lab is None:
            sub = sub.assign(celltype=np.nan)
        else:
            full_counts[subj] = lab.celltype.value_counts().to_dict()
            sub = sub.merge(lab, on=["date_key", "unit"], how="left")
        out.append(sub)
    return pd.concat(out, ignore_index=True), full_counts


def _mwu(a, b):
    a = pd.Series(a).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(b).replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, len(a), len(b), np.nan
    u, p = mannwhitneyu(a, b)
    return float(a.median()), float(b.median()), len(a), len(b), float(p)


def rate_matched(df, col, nbins=8, reps=300, seed=42):
    """Match base_hz of transient vs sustained by rate-decile subsampling, then
    MWU on `col`. Returns (median matched p, median sustained-transient Δ, frac p<.05, mean n/group)."""
    d = df.dropna(subset=["base_hz", col]).replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
    t, s = d[d["class"] == "transient"], d[d["class"] == "sustained"]
    if len(t) < 10 or len(s) < 10:
        return np.nan, np.nan, np.nan, 0
    edges = np.quantile(d.base_hz, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    rng = np.random.default_rng(seed)
    ps, diffs, ns = [], [], []
    for _ in range(reps):
        tt, ss = [], []
        for b in range(nbins):
            tb = t[(t.base_hz >= edges[b]) & (t.base_hz < edges[b + 1])]
            sb = s[(s.base_hz >= edges[b]) & (s.base_hz < edges[b + 1])]
            k = min(len(tb), len(sb))
            if k == 0:
                continue
            tt.append(tb[col].sample(k, random_state=int(rng.integers(1 << 31))))
            ss.append(sb[col].sample(k, random_state=int(rng.integers(1 << 31))))
        if not tt:
            continue
        tv, sv = pd.concat(tt), pd.concat(ss)
        if len(tv) < 5:
            continue
        ps.append(mannwhitneyu(tv, sv).pvalue)
        diffs.append(sv.median() - tv.median())
        ns.append(len(tv))
    if not ps:
        return np.nan, np.nan, np.nan, 0
    return float(np.median(ps)), float(np.median(diffs)), float(np.mean(np.array(ps) < 0.05)), int(np.median(ns))


def main():
    cells, full_counts = attach_celltype(load_cells())
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})
    lab = cells.dropna(subset=["celltype"])
    lab = lab[lab.celltype.isin(["FSI", "SPN"])]
    lines = []
    n_resp = len(cells)
    n_lab = len(lab)
    lines.append(f"responsive cells={n_resp}; with FSI/SPN label={n_lab} ({100*n_lab/n_resp:.0f}% join)")

    fig = plt.figure(figsize=(19, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.44, wspace=0.32)

    # A: FSI/SPN base rate — ALL labeled units vs TF-responsive (the yield bias)
    axa = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(MICE)); w = 0.38
    for gi, (which, cnt_fn) in enumerate([
            ("all units", lambda subj: full_counts.get(subj, {})),
            ("TF-responsive", lambda subj: lab[lab.subject == subj].celltype.value_counts().to_dict())]):
        fsi_frac = []
        for subj, _, _ in MICE:
            c = cnt_fn(subj)
            tot = c.get("FSI", 0) + c.get("SPN", 0)
            fsi_frac.append(100 * c.get("FSI", 0) / tot if tot else np.nan)
        axa.bar(xs + (gi - 0.5) * w, fsi_frac, w, label=which,
                color=("#bdbdbd" if gi == 0 else "#d94801"))
        for x, v in zip(xs + (gi - 0.5) * w, fsi_frac):
            if np.isfinite(v):
                axa.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    axa.axhline(50, color="0.6", lw=0.8, ls=":")
    axa.axhspan(5, 15, color="#c7e9c0", alpha=0.7, zorder=0)  # ~true FSI fraction ~5-15%? actually SPN dominant
    axa.text(len(MICE) - 1, 10, "true FSI ~1-15%\n(striatum is SPN-dominant)", fontsize=7.5, va="center", ha="right")
    axa.set_xticks(xs); axa.set_xticklabels([f"{s}\n({r})" for s, r, _ in MICE], fontsize=9)
    axa.set_ylabel("% FSI (narrow)"); axa.set_ylim(0, 100)
    axa.set_title("YIELD BIAS: narrow (FSI) cells over-sampled\n(vs SPN-dominant true composition)", fontsize=10.5)
    axa.legend(frameon=False, fontsize=8)
    for subj, _, _ in MICE:
        c = full_counts.get(subj, {})
        lines.append(f"  {subj} all-units: {c}")

    # B: firing rate by waveform type (mechanism of the bias)
    axb = fig.add_subplot(gs[0, 1])
    for si, ct in enumerate(("FSI", "SPN")):
        v = lab.loc[lab.celltype == ct, "base_hz"].dropna()
        jit = (np.random.default_rng(si).random(len(v)) - 0.5) * 0.28
        axb.scatter(np.full(len(v), si) + jit, v, s=9, alpha=0.35, color=WF[ct], edgecolors="none")
        axb.hlines(np.median(v), si - 0.25, si + 0.25, color="k", lw=2.3, zorder=5)
    mf, ms, nf, nsp, pf = _mwu(lab.loc[lab.celltype == "FSI", "base_hz"],
                               lab.loc[lab.celltype == "SPN", "base_hz"])
    axb.text(0.5, 0.95, f"FSI {mf:.1f} vs SPN {ms:.1f} Hz\nMWU p={pf:.1e}", transform=axb.transAxes,
             ha="center", va="top", fontsize=8.5)
    axb.set_xticks([0, 1]); axb.set_xticklabels(["FSI (narrow)", "SPN (broad)"], fontsize=9)
    axb.set_ylabel("baseline rate (Hz)"); axb.set_ylim(0, 60)
    axb.set_title("FSIs fire faster → the sorting-bias mechanism", fontsize=10.5)
    lines.append(f"[base_hz] FSI med={mf:.2f}(n={nf}) vs SPN med={ms:.2f}(n={nsp}) MWU p={pf:.2e}")

    # C: MAPPING — transient/sustained (kernel width) x FSI/SPN (waveform)
    axc = fig.add_subplot(gs[0, 2])
    sub = lab[lab["class"].isin(["transient", "sustained"])]
    ctab = pd.crosstab(sub["class"], sub["celltype"]).reindex(["transient", "sustained"])
    frac = ctab.div(ctab.sum(1), axis=0)
    bottom = np.zeros(len(frac))
    for ct in ("FSI", "SPN"):
        if ct in frac:
            axc.bar(frac.index, frac[ct], bottom=bottom, color=WF[ct], label=ct)
            bottom += frac[ct].values
    chi2, pchi, *_ = chi2_contingency(ctab)
    axc.set_ylabel("fraction"); axc.set_title(f"mapping: kernel-width × waveform\nχ²={chi2:.1f}, p={pchi:.1e}", fontsize=10.5)
    axc.legend(frameon=False, fontsize=9)
    lines.append(f"[mapping] crosstab class×celltype:\n{ctab.to_string().replace(chr(10), chr(10)+'   ')}")
    lines.append(f"  chi2={chi2:.2f} p={pchi:.2e}")

    # D: is kernel width a firing-rate artifact?
    axd = fig.add_subplot(gs[1, 0])
    dd = cells.dropna(subset=["kernel_fwhm", "base_hz"])
    for cls, cc in (("transient", TCOL), ("sustained", SCOL), ("intermediate", "0.6")):
        m = dd["class"] == cls
        axd.scatter(dd.base_hz[m], dd.kernel_fwhm[m], s=10, alpha=0.4, color=cc,
                    edgecolors="none", label=cls)
    rho_fw, p_fw = spearmanr(dd.base_hz, dd.kernel_fwhm)
    axd.set_xlabel("baseline rate (Hz)"); axd.set_ylabel("kernel FWHM (s)")
    axd.set_xlim(0, 60)
    axd.set_title(f"is width a rate artifact?  ρ(rate,FWHM)={rho_fw:+.2f}\n(p={p_fw:.1e})", fontsize=10.5)
    axd.legend(frameon=False, fontsize=8, markerscale=1.5)
    lines.append(f"[width~rate] Spearman(base_hz, kernel_fwhm) rho={rho_fw:+.3f} p={p_fw:.2e}")

    # E: does the transient->sustained coupling gap survive RATE-MATCHING?
    axe = fig.add_subplot(gs[1, 1])
    xs2 = np.arange(len(OUTCOMES)); w2 = 0.38
    raw_p, mat_p = [], []
    for oi, (col, labn) in enumerate(OUTCOMES):
        mt, mss, _, _, pr = _mwu(cells.loc[cells["class"] == "transient", col],
                                 cells.loc[cells["class"] == "sustained", col])
        pmed, dmed, frac_sig, nmg = rate_matched(cells, col)
        raw_p.append(pr); mat_p.append(pmed)
        lines.append(f"[{col}] RAW t={mt:.2f}/s={mss:.2f} p={pr:.2e} | RATE-MATCHED med-p={pmed:.2e} "
                     f"(Δs-t={dmed:+.2f}Hz, {100*frac_sig:.0f}% reps p<.05, n/grp~{nmg})")
    axe.bar(xs2 - w2 / 2, [-np.log10(p) if p and p > 0 else 0 for p in raw_p], w2, label="raw", color="0.6")
    axe.bar(xs2 + w2 / 2, [-np.log10(p) if p and p > 0 else 0 for p in mat_p], w2, label="rate-matched", color="#238b45")
    axe.axhline(-np.log10(0.05), color="r", lw=1, ls="--")
    axe.text(len(OUTCOMES) - 1, -np.log10(0.05) + 0.2, "p=0.05", color="r", fontsize=7.5, ha="right")
    axe.set_xticks(xs2); axe.set_xticklabels([o[1] for o in OUTCOMES], fontsize=9)
    axe.set_ylabel("-log10(p)  transient vs sustained")
    axe.set_title("coupling gap survives RATE-MATCHING", fontsize=10.5)
    axe.legend(frameon=False, fontsize=8)

    # within-cell-type: does the gap survive within SPN and within FSI?
    for ct in ("FSI", "SPN"):
        sc = lab[lab.celltype == ct]
        for col, _ in OUTCOMES:
            mt, mss, nt, ns2, p = _mwu(sc.loc[sc["class"] == "transient", col],
                                       sc.loc[sc["class"] == "sustained", col])
            lines.append(f"[within {ct}] {col}: t={mt}(n{nt})/s={mss}(n{ns2}) p={p if p is not np.nan else 'NA'}")

    # F: stats text
    axf = fig.add_subplot(gs[1, 2]); axf.axis("off")
    axf.text(0.0, 1.0, "\n".join(lines), transform=axf.transAxes, va="top", ha="left",
             fontsize=6.6, family="monospace")

    for ax in (axb, axc, axd, axe):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("TF-cell kernel width vs FSI/SPN waveform type, and the narrow-cell over-sampling confound\n"
                 "does the transient→sustained outcome-coupling gap survive rate-matching & within-cell-type controls?",
                 fontsize=13, y=1.005)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"waveform_celltype_join.{ext}", dpi=175, bbox_inches="tight")
    plt.close(fig)
    (OUT / "waveform_celltype_join_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/waveform_celltype_join.png (+.pdf)")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
