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
    for c in ("resp_move", "resp_nomove"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    return df.reset_index(drop=True)


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reg, g in df.groupby("region"):
        n = len(g)
        flagged_nomove = g[g["resp_nomove"]]
        n_nm = len(flagged_nomove)
        survived = int(flagged_nomove["resp_move"].sum()) if n_nm else 0
        rows.append(dict(
            region=reg, n_units=n,
            pct_resp_move=100.0 * g["resp_move"].mean(),
            pct_resp_nomove=100.0 * g["resp_nomove"].mean(),
            n_flagged_nomove=n_nm,
            n_survived_move=survived,
            pct_survived=(100.0 * survived / n_nm) if n_nm else np.nan,
        ))
    out = pd.DataFrame(rows).sort_values("n_units", ascending=False)
    return out.reset_index(drop=True)


def make_figure(summ: pd.DataFrame, out_path: Path):
    # Show the larger regions; tiny-N regions are noisy.
    s = summ[summ["n_units"] >= 5].copy()
    if not len(s):
        s = summ.copy()
    s = s.sort_values("pct_resp_move", ascending=False)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2))

    x = np.arange(len(s))
    w = 0.38
    axL.axhspan(PAPER_LO, PAPER_HI, color="0.88", zorder=0,
                label=f"Khilkevich-Lohse range ({PAPER_LO:.0f}-{PAPER_HI:.0f}%)")
    axL.bar(x - w / 2, s["pct_resp_move"], w, color="#3b7dd8",
            label="movement-controlled", zorder=2)
    axL.bar(x + w / 2, s["pct_resp_nomove"], w, color="#d8a13b",
            label="no movement control", zorder=2)
    for xi, (_, r) in zip(x, s.iterrows()):
        axL.annotate(f"{r['pct_resp_move']:.0f}%",
                     (xi - w / 2, r["pct_resp_move"]), ha="center", va="bottom",
                     fontsize=9, xytext=(0, 2), textcoords="offset points")
    axL.set_xticks(x)
    axL.set_xticklabels([f"{r}\n(n={int(n)})"
                         for r, n in zip(s["region"], s["n_units"])], fontsize=10)
    axL.set_ylabel("% TF-responsive (C2, p<0.01)")
    axL.set_title("TF-responsive fraction by region\n"
                  "movement-controlled vs no movement control", fontsize=13)
    axL.legend(frameon=False, fontsize=9, loc="upper right")
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)

    # survival: of no-movement-flagged units, % surviving movement control
    sv = s.dropna(subset=["pct_survived"])
    axR.bar(np.arange(len(sv)), sv["pct_survived"], 0.6, color="#5aa469")
    for xi, (_, r) in zip(np.arange(len(sv)), sv.iterrows()):
        axR.annotate(f"{r['pct_survived']:.0f}%\n({int(r['n_survived_move'])}/"
                     f"{int(r['n_flagged_nomove'])})",
                     (xi, r["pct_survived"]), ha="center", va="bottom",
                     fontsize=9, xytext=(0, 2), textcoords="offset points")
    axR.set_xticks(np.arange(len(sv)))
    axR.set_xticklabels(sv["region"], fontsize=10)
    axR.set_ylim(0, 105)
    axR.set_ylabel("% of no-movement flags surviving movement control")
    axR.set_title("Survival: are no-movement TF flags genuine?\n"
                  "high = real TF; low = movement confound", fontsize=13)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)

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
