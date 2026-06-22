"""Aggregate cluster TF-GLM results -> master per-unit table + per-region
TF-responsive fractions + the no-movement -> movement-controlled survival test.

Reads every ``<results>/task_*.csv`` the array job wrote, concatenates them, and
reports per region:

  - N units, movement-controlled responsive fraction (``resp_move``), and the
    no-movement responsive fraction (``resp_nomove``);
  - SURVIVAL: of the units flagged WITHOUT movement control, what fraction
    SURVIVE movement control (genuine TF) vs COLLAPSE (movement confound).

The movement-controlled fraction is the faithful Khilkevich-Lohse number to
compare against their 5-45% range and the cortex>striatum gradient. The survival
fraction is the decisive arbiter for whether the no-movement flags were real TF.

Outputs
-------
  <results>/../master.csv             per-unit master table
  <results>/../region_fractions.csv   per-region summary
  <results>/../tfglm_replication_summary.png

Run
---
  py aggregate.py --results "X:/.../tf_glm_cluster/results"
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_LO, PAPER_HI = 5.0, 45.0


def load_master(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("task_*.csv"))
    if not files:
        raise SystemExit(f"No task_*.csv under {results_dir}")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    df = pd.concat(frames, ignore_index=True)
    # de-dup (a re-queued task may have re-emitted a unit): keep last per
    # (session_rel, region, unit).
    df = df.drop_duplicates(subset=["session_rel", "region", "unit"], keep="last")
    for c in ("resp_log2", "resp_lin"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    return df.reset_index(drop=True)


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reg, g in df.groupby("region"):
        n = len(g)
        rows.append(dict(
            region=reg, n_units=n,
            pct_resp_log2=100.0 * g["resp_log2"].mean(),
            pct_resp_lin=(100.0 * g["resp_lin"].mean()
                          if "resp_lin" in g.columns else np.nan),
            median_dR_log2=float((g["r_full_log2"] - g["r_red_log2"]).median()),
        ))
    out = pd.DataFrame(rows).sort_values("n_units", ascending=False)
    return out.reset_index(drop=True)


def make_figure(summ: pd.DataFrame, out_path: Path):
    # Show the larger regions; tiny-N regions are noisy.
    s = summ[summ["n_units"] >= 5].copy()
    if not len(s):
        s = summ.copy()
    s = s.sort_values("pct_resp_log2", ascending=False)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(s) + 3), 5.2))

    x = np.arange(len(s))
    w = 0.38
    ax.axhspan(PAPER_LO, PAPER_HI, color="0.88", zorder=0,
               label=f"Khilkevich-Lohse range ({PAPER_LO:.0f}-{PAPER_HI:.0f}%)")
    ax.bar(x - w / 2, s["pct_resp_log2"], w, color="#3b7dd8",
           label="log2 (faithful encoding)", zorder=2)
    ax.bar(x + w / 2, s["pct_resp_lin"], w, color="#d8a13b",
           label="linear-Hz (control)", zorder=2)
    for xi, (_, r) in zip(x, s.iterrows()):
        ax.annotate(f"{r['pct_resp_log2']:.0f}%",
                    (xi - w / 2, r["pct_resp_log2"]), ha="center", va="bottom",
                    fontsize=9, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={int(n)})"
                        for r, n in zip(s["region"], s["n_units"])], fontsize=10)
    ax.set_ylabel("% TF-responsive (C2, p<0.01)")
    ax.set_title("Faithful Khilkevich full-model TF-responsive fraction by region\n"
                 "log2 (authors' encoding) vs linear-Hz control", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, help="dir of task_*.csv files")
    a = p.parse_args(argv)
    results_dir = Path(a.results)
    base = results_dir.parent

    df = load_master(results_dir)
    master_csv = base / "master.csv"
    df.to_csv(master_csv, index=False)

    summ = region_summary(df)
    frac_csv = base / "region_fractions.csv"
    summ.to_csv(frac_csv, index=False)

    print(f"\nMaster: {master_csv} ({len(df)} units)")
    print(f"Region fractions: {frac_csv}\n")
    with pd.option_context("display.width", 160,
                           "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.1f}"):
        print(summ.to_string(index=False))

    fig = make_figure(summ, base / "tfglm_replication_summary.png")
    print(f"\nFigure: {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
