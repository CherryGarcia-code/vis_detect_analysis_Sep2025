#!/usr/bin/env python3
"""QC sweep of completed concat-sort KS4 runs (the nblocks=1 resort).

Reads every run directory listed in ks4_run_manifest.json that carries a
ks4_complete.txt marker and extracts per-run quality metrics directly from
the raw KS4 output (no downstream stitch/pkl needed):

  - Unit yield      : n_total / n_good / n_mua from cluster_KSLabel.tsv
  - Contamination   : median ContamPct, fraction < 10% (cluster_ContamPct.tsv)
  - Drift quality   : lag-1 autocorrelation of dshift, drift range, # large
                      single-batch jumps (>50 um) -- from ops.npy
  - Runtime / params: elapsed, Th_universal, Th_learned, nblocks

Drift AC1 is the headline metric: the nblocks=5->1 fix was made specifically
to suppress the noisy per-batch drift estimates documented in
docs/AI_interaction/concat-sort/deep_audit_drift_and_nblocks.md. Real slow
drift gives AC1 > 0.95; noise-dominated estimates give AC1 < 0.7.

Usage:
    py scripts/pipelines/concat_sort/qc_ks4_runs.py
    py scripts/pipelines/concat_sort/qc_ks4_runs.py --manifest <path> --no-fig
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MANIFEST = ("X:/public/projects/BeJG_20230130_VisDetect/wEPhys/"
                    "BG_046/concat_sort/ks4_runs/ks4_run_manifest.json")
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "FIGURES" / "concat_sort_qc"

# nblocks=5 baseline from the March 2026 deep audit (deep_audit_drift_and_nblocks.md)
BASELINE_NB5 = {
    "drift_groups": {"good (AC>0.9)": 66, "mid (0.7-0.9)": 33, "bad (AC<0.7)": 37},
    "per_shank_good": {0: 42, 1: 44, 2: 67, 3: 104},
    "per_shank_ac1": {0: 0.839, 1: 0.826, 2: 0.803, 3: 0.834},
}

LARGE_JUMP_UM = 50.0


def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D series; nan if degenerate."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3 or np.std(x) < 1e-9:
        return np.nan
    a, b = x[:-1], x[1:]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def read_label_counts(run_dir: Path):
    """(n_total, n_good, n_mua) from cluster_KSLabel.tsv; falls back to group."""
    for fname in ("cluster_KSLabel.tsv", "cluster_group.tsv"):
        f = run_dir / fname
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        label_col = "KSLabel" if "KSLabel" in df.columns else df.columns[-1]
        labels = df[label_col].astype(str).str.lower()
        return len(df), int((labels == "good").sum()), int((labels == "mua").sum())
    return None, None, None


def read_contam(run_dir: Path):
    """(median_contam_pct, frac_below_10pct) from cluster_ContamPct.tsv."""
    f = run_dir / "cluster_ContamPct.tsv"
    if not f.exists():
        return np.nan, np.nan
    df = pd.read_csv(f, sep="\t")
    col = "ContamPct" if "ContamPct" in df.columns else df.columns[-1]
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.median(vals)), float(np.mean(vals < 10.0))


def read_ops_metrics(run_dir: Path):
    """Drift + runtime metrics from ops.npy."""
    out = dict(drift_ac1=np.nan, drift_range_um=np.nan, n_jumps_50um=np.nan,
               n_batches=np.nan, runtime_min=np.nan, th_universal=np.nan,
               th_learned=np.nan, nblocks=np.nan, n_spikes=np.nan,
               ops_good=np.nan, ops_total=np.nan)
    f = run_dir / "ops.npy"
    if not f.exists():
        return out
    try:
        ops = np.load(f, allow_pickle=True).item()
    except Exception:
        return out
    ds = np.asarray(ops.get("dshift", [])).ravel()
    if ds.size >= 3:
        out["drift_ac1"] = lag1_autocorr(ds)
        out["drift_range_um"] = float(np.ptp(ds))
        out["n_jumps_50um"] = int(np.sum(np.abs(np.diff(ds)) > LARGE_JUMP_UM))
        out["n_batches"] = int(ds.size)
    for k_src, k_dst in [("Th_universal", "th_universal"),
                         ("Th_learned", "th_learned"), ("nblocks", "nblocks"),
                         ("n_spikes", "n_spikes"), ("n_units_good", "ops_good"),
                         ("n_units_total", "ops_total")]:
        v = ops.get(k_src)
        if v is not None:
            try:
                out[k_dst] = float(v)
            except (TypeError, ValueError):
                pass
    rt = ops.get("runtime")
    if rt is not None:
        try:
            out["runtime_min"] = float(rt) / 60.0
        except (TypeError, ValueError):
            pass
    return out


def stage_for_window(w: int) -> str:
    """Audit's temporal grouping of the 34 windows."""
    if w <= 8:
        return "Early (Jun)"
    if w <= 20:
        return "Transition (Jul)"
    return "Late (Aug-Sep)"


def drift_group(ac1: float) -> str:
    if np.isnan(ac1):
        return "unknown"
    if ac1 > 0.9:
        return "good (AC>0.9)"
    if ac1 >= 0.7:
        return "mid (0.7-0.9)"
    return "bad (AC<0.7)"


def sweep(manifest_path: str) -> pd.DataFrame:
    with open(manifest_path) as fh:
        windows = json.load(fh)["windows"]
    rows = []
    for i, win in enumerate(windows):
        run_dir = Path(win["run_dir"])
        rec = dict(job_id=i + 1, window_idx=win["window_idx"],
                   shank_id=int(win["shank_id"]),
                   stage=stage_for_window(int(win["window_idx"])),
                   n_sessions=len(win.get("sessions", [])))
        if not (run_dir / "ks4_complete.txt").exists():
            rec["status"] = "FAILED"
            rows.append(rec)
            continue
        rec["status"] = "COMPLETED"
        n_total, n_good, n_mua = read_label_counts(run_dir)
        rec["n_total"] = n_total
        rec["n_good"] = n_good
        rec["n_mua"] = n_mua
        rec["good_frac"] = (n_good / n_total) if n_total else np.nan
        med_contam, frac_clean = read_contam(run_dir)
        rec["median_contam_pct"] = med_contam
        rec["frac_contam_below10"] = frac_clean
        rec.update(read_ops_metrics(run_dir))
        rec["drift_group"] = drift_group(rec["drift_ac1"])
        print(f"  job {i+1:3d}  W{win['window_idx']:>2} S{win['shank_id']}  "
              f"good={n_good}/{n_total}  AC1={rec['drift_ac1']:.3f}", flush=True)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    done = df[df.status == "COMPLETED"].copy()
    n_done = len(done)
    print("\n" + "=" * 72)
    print(f"KS4 RESORT QUALITY  --  {n_done}/{len(df)} runs completed")
    print("=" * 72)

    print("\nUNIT YIELD (per completed run):")
    for label, col in [("total units", "n_total"), ("good units", "n_good")]:
        s = done[col].dropna()
        print(f"  {label:<14} median {s.median():6.0f}   mean {s.mean():6.1f}"
              f"   sum {s.sum():7.0f}   range [{s.min():.0f}, {s.max():.0f}]")
    gf = done["good_frac"].dropna()
    print(f"  good fraction   median {gf.median():.3f}   mean {gf.mean():.3f}")

    print("\nDRIFT QUALITY (lag-1 autocorr of dshift):")
    grp = done["drift_group"].value_counts()
    for g in ("good (AC>0.9)", "mid (0.7-0.9)", "bad (AC<0.7)", "unknown"):
        n5 = BASELINE_NB5["drift_groups"].get(g)
        base = f"   (nblocks=5 baseline: {n5})" if n5 is not None else ""
        cnt = int(grp.get(g, 0))
        pct = 100 * cnt / n_done if n_done else 0
        print(f"  {g:<16} {cnt:3d} ({pct:4.1f}%){base}")
    ac = done["drift_ac1"].dropna()
    print(f"  AC1 median {ac.median():.3f}   mean {ac.mean():.3f}")
    jumps = done["n_jumps_50um"].dropna()
    print(f"  runs with >50um single-batch jumps: "
          f"{int((jumps > 0).sum())}/{len(jumps)}  (total jumps {int(jumps.sum())})")

    print("\nPER-SHANK (this resort vs nblocks=5 audit baseline):")
    print(f"  {'shank':<7}{'n':>4}{'good (med)':>12}{'AC1 (mean)':>12}"
          f"{'nb5 good':>11}{'nb5 AC1':>10}")
    for sh in sorted(done.shank_id.unique()):
        sub = done[done.shank_id == sh]
        print(f"  {sh:<7}{len(sub):>4}{sub.n_good.median():>12.0f}"
              f"{sub.drift_ac1.mean():>12.3f}"
              f"{BASELINE_NB5['per_shank_good'].get(sh, 0):>11}"
              f"{BASELINE_NB5['per_shank_ac1'].get(sh, float('nan')):>10.3f}")

    print("\nPER-STAGE (temporal):")
    for st in ("Early (Jun)", "Transition (Jul)", "Late (Aug-Sep)"):
        sub = done[done.stage == st]
        if len(sub):
            print(f"  {st:<20} n={len(sub):>3}   good(med) {sub.n_good.median():>5.0f}"
                  f"   AC1(mean) {sub.drift_ac1.mean():.3f}")

    print("\nCONTAMINATION:")
    mc = done["median_contam_pct"].dropna()
    fc = done["frac_contam_below10"].dropna()
    if len(mc):
        print(f"  per-run median ContamPct: median {mc.median():.1f}%")
        print(f"  fraction of clusters <10% contam: mean {fc.mean():.3f}")

    rt = done["runtime_min"].dropna()
    if len(rt):
        print(f"\nRUNTIME: median {rt.median()/60:.1f} h   "
              f"max {rt.max()/60:.1f} h")

    th = done.groupby(["th_universal", "th_learned"]).size()
    print("\nTHRESHOLDS USED (Th_universal, Th_learned) -> n runs:")
    for (tu, tl), n in th.items():
        print(f"  ({tu:.0f}, {tl:.0f}) -> {n}")


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    done = df[df.status == "COMPLETED"].copy()
    shank_colors = {0: "#2196F3", 1: "#4CAF50", 2: "#FF9800", 3: "#F44336"}

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Concat-sort KS4 resort QC (nblocks=1)  -  "
                 f"{len(done)}/{len(df)} runs completed", fontsize=13)

    # 1. drift AC1 histogram vs thresholds
    ax = axes[0, 0]
    ax.hist(done.drift_ac1.dropna(), bins=np.linspace(0, 1, 26),
            color="steelblue", edgecolor="white")
    ax.axvline(0.9, color="green", ls="--", label="0.9")
    ax.axvline(0.7, color="red", ls="--", label="0.7")
    ax.set_xlabel("drift lag-1 autocorr"); ax.set_ylabel("# runs")
    ax.set_title("Drift quality (higher = real drift)"); ax.legend(fontsize=8)

    # 2. good units per run, by shank
    ax = axes[0, 1]
    for sh in sorted(done.shank_id.unique()):
        sub = done[done.shank_id == sh]
        ax.scatter(sub.window_idx, sub.n_good, s=22, color=shank_colors.get(sh),
                   label=f"shank {sh}", alpha=0.8)
    ax.set_xlabel("window idx"); ax.set_ylabel("# good units")
    ax.set_title("Good-unit yield over windows"); ax.legend(fontsize=8)

    # 3. AC1 over windows
    ax = axes[0, 2]
    for sh in sorted(done.shank_id.unique()):
        sub = done[done.shank_id == sh].sort_values("window_idx")
        ax.plot(sub.window_idx, sub.drift_ac1, "o-", ms=4,
                color=shank_colors.get(sh), label=f"shank {sh}", alpha=0.8)
    ax.axhline(0.9, color="green", ls="--", alpha=0.5)
    ax.set_xlabel("window idx"); ax.set_ylabel("drift AC1")
    ax.set_title("Drift AC1 over windows"); ax.legend(fontsize=8)

    # 4. good fraction histogram
    ax = axes[1, 0]
    ax.hist(done.good_frac.dropna(), bins=np.linspace(0, 1, 26),
            color="seagreen", edgecolor="white")
    ax.set_xlabel("good / total units"); ax.set_ylabel("# runs")
    ax.set_title("KS4 good-label fraction")

    # 5. yield vs drift AC1
    ax = axes[1, 1]
    for sh in sorted(done.shank_id.unique()):
        sub = done[done.shank_id == sh]
        ax.scatter(sub.drift_ac1, sub.n_good, s=22, color=shank_colors.get(sh),
                   label=f"shank {sh}", alpha=0.8)
    ax.set_xlabel("drift AC1"); ax.set_ylabel("# good units")
    ax.set_title("Yield vs drift quality")

    # 6. completed vs failed per window
    ax = axes[1, 2]
    pivot = (df.assign(ok=(df.status == "COMPLETED").astype(int))
               .groupby("window_idx").ok.agg(["sum", "count"]))
    ax.bar(pivot.index, pivot["count"], color="lightgray", label="total")
    ax.bar(pivot.index, pivot["sum"], color="mediumseagreen", label="completed")
    ax.set_xlabel("window idx"); ax.set_ylabel("# shank-jobs (of 4)")
    ax.set_title("Completion per window"); ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="QC sweep of completed KS4 resort runs")
    ap.add_argument("--manifest", "-m", default=DEFAULT_MANIFEST)
    ap.add_argument("--no-fig", action="store_true", help="skip the figure")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        raise SystemExit(f"Manifest not found: {args.manifest}")

    print(f"Reading manifest: {args.manifest}\n")
    df = sweep(args.manifest)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "ks4_run_quality.csv"
    df.to_csv(csv_path, index=False)

    summarize(df)
    print(f"\nPer-run CSV -> {csv_path}")
    if not args.no_fig:
        make_figure(df, OUT_DIR / "ks4_run_quality.png")


if __name__ == "__main__":
    main()
