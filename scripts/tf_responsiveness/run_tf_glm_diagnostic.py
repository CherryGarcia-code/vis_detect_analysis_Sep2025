"""Positive-control DIAGNOSTIC: decompose C1 (firing-rate floor) vs C2 (the
FR-independent TF-specific ablation test) on the Khilkevich npx_converted data.

Motivation
----------
Our corrected TF-GLM positive control gives a CORTICAL region (VISp) ~0%
TF-responsive while Khilkevich & Lohse find visual cortex highly responsive.
``identify_tf_responsive`` flags a unit only when BOTH:
  C1: mean held-out predictive correlation of the FULL model  c1_r > 0.2
  C2: paired one-sided t-test (corr_full - corr_reduced > 0)  c2_p < 0.01
The 0.2 floor in C1 was calibrated on the authors' FULL 19-regressor model; on
our REDUCED model it can act as a firing-rate gate that knocks out lower-firing
units (e.g. VISp) regardless of TF. C2 is the FR-independent TF-specific test.

This script does NOT build a new model. It re-runs the existing pipeline on a
CORTICAL region (VISp) and a STRIATAL region (CP), and decomposes the
identification into:
  1. C1-alone fraction:  mean(c1_r > 0.2)
  2. C2-alone fraction:  mean(c2_p < 0.01)   <-- THE KEY NUMBER (TF test)
  3. Both (current is_responsive) fraction
  4. FR-matched: spike-count-binned matched subsamples of VISp vs CP, fractions
     recomputed within the matched sets.
  5. Distributions: MWU of c1_r / c2_p / (r_full-r_red) between regions; Spearman
     of c1_r vs n_spikes per region (quantifies the FR confound).

Sessions
--------
VISp is thin in any single Khilkevich session, so we POOL the two sessions of
animal 1116764 that carry both a visual cortical probe and a CP probe
(ML_1116764_S02_M2_V1 + ML_1116764_S03_M2_V1). Per-unit fits are independent, so
pooling per-unit rows across sessions is valid. CP is plentiful; we take the
highest-spike-count CP units to match the per-region budget.

Outputs
-------
  data/cache/tf_glm/khilkevich_diagnostic.csv
  figures/tf_responsiveness/glm_khilkevich_diagnostic.png

Run
---
PYTHONPATH=src py scripts/tf_responsiveness/run_tf_glm_diagnostic.py --max-units 25
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive,
)
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors,
)
from visdetect.viz.plotting import set_style, despine

ROOT = Path("X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted")
CACHE = _REPO / "data" / "cache" / "tf_glm" / "khilkevich_diagnostic.csv"
FIG = _REPO / "figures" / "tf_responsiveness" / "glm_khilkevich_diagnostic.png"

# Pool both 1116764 sessions (each carries a V1 probe + a CP probe).
SESSIONS = [
    "1116764/ML_1116764_S02_M2_V1",
    "1116764/ML_1116764_S03_M2_V1",
]
CORTEX_REGION = "VISp"
STRIATUM_REGION = "CP"
MIN_SPIKES = 500
C1_THRESH = 0.2
C2_THRESH = 0.01


def _qualifying_units(units, min_spikes=MIN_SPIKES):
    spk = {u: float(units[u].size) for u in units}
    uids = [u for u in sorted(units, key=lambda u: spk[u], reverse=True)
            if spk[u] >= min_spikes]
    return uids, spk


CSV_COLS = ["region", "session", "unit", "n_spikes", "c1_r", "c2_p",
            "r_full_mean", "r_red_mean", "is_responsive"]


def _append_row(row):
    """Checkpoint a single per-unit result to the CSV (resume-safe)."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CACHE.exists()
    pd.DataFrame([row])[CSV_COLS].to_csv(
        CACHE, mode="a", header=write_header, index=False)


def _done_units():
    """(region, session, unit) tuples already checkpointed in the CSV."""
    if not CACHE.exists():
        return set()
    d = pd.read_csv(CACHE)
    return set(zip(d["region"], d["session"], d["unit"].astype(int)))


def run_region_session(session_dir, region, cfg, uid_cap=None, verbose=True):
    """Fit full+reduced GLM for each qualifying unit in `region` of one session.

    Each completed unit is appended to the CSV immediately so a re-run resumes
    where it left off (skips units already present). Returns the new rows.
    """
    ks = load_khilkevich_session(session_dir)
    trials, units = khilkevich_trial_regressors(ks, cfg, region=region)
    sess_name = Path(session_dir).name
    uids, _ = _qualifying_units(units)
    if uid_cap is not None:
        uids = uids[:uid_cap]

    done = _done_units()
    todo = [u for u in uids if (region, sess_name, int(u)) not in done]
    if not todo:
        if verbose:
            print(f"[{region}|{sess_name}] all {len(uids)} units already cached, "
                  f"skipping", flush=True)
        return []

    design = assemble_design(trials, cfg)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    Xr = design.X.copy()
    Xr[:, design.col_groups["tf"]] = 0.0

    if verbose:
        print(f"[{region}|{sess_name}] {len(trials)} trials, X={design.X.shape}, "
              f"fitting {len(todo)}/{len(uids)} units "
              f"({len(uids)-len(todo)} cached)", flush=True)

    rows = []
    for k, uid in enumerate(todo):
        y = count_vector(trials, units[uid], design)
        if y.sum() < MIN_SPIKES:
            continue
        t0 = time.time()
        full = fit_poisson_cv(design.X, y, cfg, folds)
        red = fit_poisson_cv(Xr, y, cfg, folds)
        out = identify_tf_responsive(design, y, full, red, cfg)
        row = {
            "region": region, "session": sess_name, "unit": int(uid),
            "n_spikes": float(y.sum()),
            "c1_r": float(out["c1_r"]), "c2_p": float(out["c2_p"]),
            "r_full_mean": float(out["r_full_mean"]),
            "r_red_mean": float(out["r_red_mean"]),
            "is_responsive": bool(out["is_responsive"]),
        }
        rows.append(row)
        _append_row(row)
        if verbose:
            print(f"  [{region}|{sess_name}] {uid} ({k+1}/{len(todo)}): "
                  f"{int(y.sum())}spk c1_r={out['c1_r']:.3f} "
                  f"c2_p={out['c2_p']:.1e} resp={out['is_responsive']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return rows


def fractions(df):
    """C1-alone, C2-alone, both fractions for a per-unit DataFrame."""
    n = len(df)
    if n == 0:
        return dict(n=0, c1=np.nan, c2=np.nan, both=np.nan)
    c1 = float((df["c1_r"] > C1_THRESH).mean())
    c2 = float((df["c2_p"] < C2_THRESH).mean())
    both = float(((df["c1_r"] > C1_THRESH) & (df["c2_p"] < C2_THRESH)).mean())
    return dict(n=n, c1=c1, c2=c2, both=both)


def fr_matched(df_vis, df_cp, n_bins=4, seed=42):
    """Match VISp and CP on spike-count distribution, return matched subsets.

    Bin both regions by shared log-spike-count quantile edges; in each bin take
    min(n_vis, n_cp) units from each region (random subsample, seeded).
    """
    rng = np.random.default_rng(seed)
    s_all = np.log10(np.concatenate([df_vis["n_spikes"].to_numpy(),
                                     df_cp["n_spikes"].to_numpy()]))
    edges = np.quantile(s_all, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    keep_v, keep_c = [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        vb = df_vis[(np.log10(df_vis["n_spikes"]) > lo)
                    & (np.log10(df_vis["n_spikes"]) <= hi)]
        cb = df_cp[(np.log10(df_cp["n_spikes"]) > lo)
                   & (np.log10(df_cp["n_spikes"]) <= hi)]
        m = min(len(vb), len(cb))
        if m == 0:
            continue
        keep_v.append(vb.iloc[rng.permutation(len(vb))[:m]])
        keep_c.append(cb.iloc[rng.permutation(len(cb))[:m]])
    mv = pd.concat(keep_v) if keep_v else df_vis.iloc[:0]
    mc = pd.concat(keep_c) if keep_c else df_cp.iloc[:0]
    return mv, mc


def mwu(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def spearman_c1_fr(df):
    x = np.log10(df["n_spikes"].to_numpy(float))
    yv = df["c1_r"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(yv)
    if ok.sum() < 3:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x[ok], yv[ok])
    return float(rho), float(p)


def make_figure(df_vis, df_cp, fv, fc, fv_m, fc_m, out_path):
    set_style("talk")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # LEFT: grouped bars C1-alone / C2-alone / both for VISp vs CP.
    cats = ["C1-alone\n(FR floor r>0.2)", "C2-alone\n(TF test p<0.01)",
            "Both\n(current call)"]
    vis_vals = [fv["c1"], fv["c2"], fv["both"]]
    cp_vals = [fc["c1"], fc["c2"], fc["both"]]
    x = np.arange(len(cats))
    w = 0.36
    bv = axL.bar(x - w / 2, [100 * v for v in vis_vals], w,
                 label=f"VISp (cortex, n={fv['n']})", color="#3b7dd8")
    bc = axL.bar(x + w / 2, [100 * v for v in cp_vals], w,
                 label=f"CP (striatum, n={fc['n']})", color="#d8743b")
    axL.axhline(1.0, color="0.4", ls="--", lw=1.2, label="1% chance")
    for bars in (bv, bc):
        for bar in bars:
            axL.annotate(f"{bar.get_height():.0f}%",
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=10,
                         xytext=(0, 2), textcoords="offset points")
    axL.set_xticks(x)
    axL.set_xticklabels(cats, fontsize=11)
    axL.set_ylabel("% of neurons")
    axL.set_title("Does the speed (TF) signal survive each test?\n"
                  "C2-alone is the firing-rate-independent TF test",
                  fontsize=13)
    axL.legend(loc="upper right", frameon=False, fontsize=10)
    despine(axL)
    # annotate the FR-matched C2-alone under the C2 group
    axL.annotate(f"FR-matched C2-alone:\nVISp {100*fv_m['c2']:.0f}%  "
                 f"CP {100*fc_m['c2']:.0f}%",
                 (1, max(100 * fv["c2"], 100 * fc["c2"])),
                 ha="center", va="bottom", fontsize=9, color="0.3",
                 xytext=(0, 22), textcoords="offset points")

    # RIGHT: c1_r vs n_spikes scatter, colored by region, 0.2 floor line.
    axR.scatter(df_vis["n_spikes"], df_vis["c1_r"], s=46, color="#3b7dd8",
                alpha=0.85, label="VISp (cortex)", edgecolor="w", linewidth=0.5)
    axR.scatter(df_cp["n_spikes"], df_cp["c1_r"], s=46, color="#d8743b",
                alpha=0.85, label="CP (striatum)", edgecolor="w", linewidth=0.5)
    axR.axhline(C1_THRESH, color="0.3", ls="--", lw=1.4,
                label="C1 floor (r=0.2)")
    axR.set_xscale("log")
    axR.set_xlabel("Total spikes (log scale)")
    axR.set_ylabel("C1 held-out predictive r (full model)")
    rho_v, _ = spearman_c1_fr(df_vis)
    rho_c, _ = spearman_c1_fr(df_cp)
    axR.set_title("The C1 floor is a firing-rate gate\n"
                  f"Spearman(c1_r, spikes): VISp ρ={rho_v:.2f}, "
                  f"CP ρ={rho_c:.2f}", fontsize=13)
    axR.legend(loc="lower right", frameon=False, fontsize=10)
    despine(axR)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-units", type=int, default=25,
                   help="per-region budget (split across pooled sessions)")
    p.add_argument("--reuse-cache", action="store_true",
                   help="skip fitting and re-plot from the existing CSV")
    a = p.parse_args(argv)

    if not a.reuse_cache:
        cfg = TFGLMConfig(include_phase=False, fast_fit=True)
        # VISp: pool ALL qualifying units across both sessions (it is thin).
        for s in SESSIONS:
            run_region_session(ROOT / s, CORTEX_REGION, cfg, uid_cap=None)
        # CP: split the budget across the two sessions (highest-spike units).
        cp_cap = max(1, a.max_units // len(SESSIONS))
        for s in SESSIONS:
            run_region_session(ROOT / s, STRIATUM_REGION, cfg, uid_cap=cp_cap)
    # Read back the accumulated (resume-safe) per-unit checkpoint CSV.
    df = pd.read_csv(CACHE)
    print(f"\nLoaded {CACHE} ({len(df)} units)")

    df_vis = df[df["region"] == CORTEX_REGION].copy()
    df_cp = df[df["region"] == STRIATUM_REGION].copy()

    fv = fractions(df_vis)
    fc = fractions(df_cp)
    mv, mc = fr_matched(df_vis, df_cp)
    fv_m = fractions(mv)
    fc_m = fractions(mc)

    print("\n=== Fraction decomposition (full sets) ===")
    for name, f in [("VISp", fv), ("CP", fc)]:
        print(f"  {name} (n={f['n']}): C1-alone={f['c1']:.3f}  "
              f"C2-alone={f['c2']:.3f}  both={f['both']:.3f}")
    print("\n=== FR-matched sets ===")
    for name, f in [("VISp", fv_m), ("CP", fc_m)]:
        print(f"  {name} (n={f['n']}): C1-alone={f['c1']:.3f}  "
              f"C2-alone={f['c2']:.3f}  both={f['both']:.3f}")

    print("\n=== Distributions (Mann-Whitney U, VISp vs CP) ===")
    for col in ["c1_r", "c2_p"]:
        u, pp = mwu(df_vis[col], df_cp[col])
        print(f"  {col}: U={u:.1f} p={pp:.3g}")
    diff_v = df_vis["r_full_mean"] - df_vis["r_red_mean"]
    diff_c = df_cp["r_full_mean"] - df_cp["r_red_mean"]
    u, pp = mwu(diff_v, diff_c)
    print(f"  (r_full-r_red): U={u:.1f} p={pp:.3g}  "
          f"median VISp={np.nanmedian(diff_v):.4f} CP={np.nanmedian(diff_c):.4f}")

    print("\n=== Spearman(c1_r, log10 n_spikes) — FR confound ===")
    for name, d in [("VISp", df_vis), ("CP", df_cp)]:
        rho, pp = spearman_c1_fr(d)
        print(f"  {name}: rho={rho:.3f} p={pp:.3g}")

    fig = make_figure(df_vis, df_cp, fv, fc, fv_m, fc_m, FIG)
    print(f"\nFigure: {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
