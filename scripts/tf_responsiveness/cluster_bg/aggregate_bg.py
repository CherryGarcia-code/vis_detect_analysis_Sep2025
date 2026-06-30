"""Aggregate the BG-mouse TF-GLM array-task outputs into master tables +
cross-subject figures.

Reads every ``results_bg/task_*.csv`` (per-unit C1/C2 + responsive call) and the
matching ``task_*_peth.npz`` (fast/slow pulse PETHs), then writes:
  master_bg.csv          one row per (subject, session, unit), deduped
  subject_fractions.csv  TF-responsive fraction + C1 stats per subject
  session_fractions.csv  ... per session
  fig_bg_summary.png     per-subject responsive %, C1 distribution vs 0.2,
                         and the population fast-minus-slow pulse response for
                         responsive vs non-responsive units (the "do they look
                         TF-responsive" view, pooled across all mice)

Session ids are kept as strings throughout (never int-cast).

Usage:
  py aggregate_bg.py --results "X:/.../tf_glm_cluster/bg_mice/results_bg" \
      --out-dir "X:/.../tf_glm_cluster/bg_mice/agg"
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_master(results_dir):
    csvs = [f for f in sorted(glob.glob(str(Path(results_dir) / "task_*.csv")))
            if "_peth" not in f]
    if not csvs:
        raise SystemExit(f"no task_*.csv in {results_dir}")
    m = pd.concat([pd.read_csv(f, dtype={"session": str, "subject": str})
                   for f in csvs], ignore_index=True)
    for c in ("resp_log2", "resp_lin"):
        if c in m.columns:
            m[c] = m[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    m = m.drop_duplicates(subset=["session", "unit"], keep="last")
    return m


def load_peth(results_dir):
    """{(session, unit) -> 4xnlags}, plus a shared t_axis."""
    peth, t_axis = {}, None
    for f in sorted(glob.glob(str(Path(results_dir) / "task_*_peth.npz"))):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        sess = str(z["session"]) if "session" in z else Path(f).stem
        if "t_axis" in z and z["t_axis"].size:
            t_axis = z["t_axis"]
        for u in z["units"]:
            key = f"u{int(u)}"
            if key in z:
                peth[(sess, int(u))] = z[key]
    return peth, t_axis


def frac_table(m, by):
    g = m.groupby(by)
    out = g.agg(n_units=("unit", "size"),
                n_resp_log2=("resp_log2", "sum"),
                c1_med=("c1_r_log2", "median"),
                c1_gt02=("c1_r_log2", lambda s: int((s > 0.2).sum()))).reset_index()
    out["pct_resp_log2"] = 100 * out["n_resp_log2"] / out["n_units"]
    out["pct_c1_gt02"] = 100 * out["c1_gt02"] / out["n_units"]
    if "resp_lin" in m.columns:
        out["n_resp_lin"] = g["resp_lin"].sum().values
        out["pct_resp_lin"] = 100 * out["n_resp_lin"] / out["n_units"]
    return out


def fig_summary(m, peth, t, out_png):
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(17, 4.8))
    # per-subject responsive %
    st = frac_table(m, "subject").sort_values("subject")
    x = range(len(st))
    axL.bar(x, st["pct_resp_log2"], color="#5aa469")
    axL.set_xticks(list(x)); axL.set_xticklabels(st["subject"], rotation=45, ha="right")
    axL.set_ylabel("% TF-responsive (log2, C1+C2)")
    axL.set_title("TF-responsive fraction per subject")
    for i, (_, r) in enumerate(st.iterrows()):
        axL.text(i, r["pct_resp_log2"], f"{int(r['n_resp_log2'])}/{int(r['n_units'])}",
                 ha="center", va="bottom", fontsize=7)
    # C1 distribution
    axM.hist(m["c1_r_log2"].dropna(), bins=40, color="#6baed6", edgecolor="w")
    axM.axvline(0.2, color="#d6322a", ls="--", lw=1.5, label="C1=0.2")
    axM.set_xlabel("C1 (full-model fast-slow pulse-PETH corr)")
    axM.set_ylabel("units")
    axM.set_title(f"C1 distribution (all mice, n={len(m)})\n"
                  f"{100*np.mean(m.c1_r_log2>0.2):.0f}% > 0.2, "
                  f"{100*m.resp_log2.mean():.0f}% responsive")
    axM.legend(frameon=False)
    # population fast-slow PETH: responsive vs non
    if t is not None and len(peth):
        for grp, col, lab in [(True, "#d6322a", "responsive"),
                              (False, "0.5", "non-responsive")]:
            diffs = []
            for _, r in m[m.resp_log2 == grp].iterrows():
                k = (str(r.session), int(r.unit))
                if k in peth:
                    af, as_, _, _ = peth[k]
                    d = af - as_
                    d = d - (np.nanmean(d[t < 0]) if (t < 0).any() else 0)
                    diffs.append(d)
            if diffs:
                D = np.vstack(diffs); mu = np.nanmean(D, 0)
                se = np.nanstd(D, 0) / np.sqrt(len(D))
                axR.plot(t, mu, color=col, lw=2, label=f"{lab} (n={len(D)})")
                axR.fill_between(t, mu - se, mu + se, color=col, alpha=0.2)
        axR.axvline(0, color="0.7", lw=0.8); axR.axhline(0, color="0.7", lw=0.8)
        axR.set_xlabel("time from TF pulse (s)")
        axR.set_ylabel("Δ firing, fast−slow (Hz)")
        axR.set_title("Population fast−slow pulse response")
        axR.legend(frameon=False)
    for ax in (axL, axM, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_png)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, help="results_bg dir with task_*.csv")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    m = load_master(a.results)
    m.to_csv(out / "master_bg.csv", index=False)
    frac_table(m, "subject").to_csv(out / "subject_fractions.csv", index=False)
    frac_table(m, ["subject", "session"]).to_csv(out / "session_fractions.csv", index=False)
    peth, t = load_peth(a.results)
    print(f"master: {len(m)} units | {m.subject.nunique()} subjects | "
          f"{m.session.nunique()} sessions | {int(m.resp_log2.sum())} responsive | "
          f"{len(peth)} PETHs")
    print(frac_table(m, "subject").to_string(index=False))
    fig_summary(m, peth, t, str(out / "fig_bg_summary.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
