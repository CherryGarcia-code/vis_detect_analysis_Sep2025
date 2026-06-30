"""Eyeball-check: render the most TF-responsive units' fast/slow pulse PETHs from
a (possibly PARTIAL) cluster results dir, so we can sanity-check that
GLM-identified cells actually look TF-responsive while the sweep is still running.

Reads results_bg_<SUBJ>/task_*.csv (resp_log2 + c1_r_log2) and task_*_peth.npz
(per-unit 4xnlags: actual fast/slow + GLM-predicted fast/slow, Hz). Plots the
top-C1 responsive units: fast (red) vs slow (blue), actual solid + GLM dashed,
baseline-subtracted. A real TF cell separates fast from slow after the pulse.

Usage:
  py plot_cluster_exemplars.py <results_dir> <SubjectLabel> <out_png> [n]
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(results_dir):
    csvs = [f for f in glob.glob(str(Path(results_dir) / "task_*.csv")) if "_peth" not in f]
    if not csvs:
        raise SystemExit(f"no task_*.csv in {results_dir}")
    m = pd.concat([pd.read_csv(f, dtype={"session": str, "subject": str}) for f in csvs],
                  ignore_index=True)
    m["resp_log2"] = m["resp_log2"].astype(str).str.lower().isin(["true", "1", "1.0"])
    m = m.drop_duplicates(subset=["session", "unit"], keep="last")
    peth, t = {}, None
    for f in glob.glob(str(Path(results_dir) / "task_*_peth.npz")):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        sess = str(z["session"]) if "session" in z else ""
        if "t_axis" in z and z["t_axis"].size:
            t = z["t_axis"]
        for u in z["units"]:
            k = f"u{int(u)}"
            if k in z:
                peth[(sess, int(u))] = z[k]
    return m, peth, t


def panel(ax, t, arr, title):
    af, as_, pf, ps = arr

    def bs(v):
        pre = v[t < 0]
        return v - (np.nanmean(pre) if pre.size else 0.0)
    ax.axvline(0, color="0.7", lw=0.8, zorder=0)
    ax.plot(t, bs(af), color="#d6322a", lw=2, label="fast (actual)")
    ax.plot(t, bs(as_), color="#2b6fb3", lw=2, label="slow (actual)")
    ax.plot(t, bs(pf), color="#d6322a", lw=1, ls="--", alpha=0.8, label="fast (GLM)")
    ax.plot(t, bs(ps), color="#2b6fb3", lw=1, ls="--", alpha=0.8, label="slow (GLM)")
    ax.set_title(title, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    results, subj, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    m, peth, t = load(results)
    if t is None:
        raise SystemExit("no t_axis found in any peth npz yet")
    resp = m[m.resp_log2].sort_values("c1_r_log2", ascending=False)
    keys = [(r.session, int(r.unit)) for _, r in resp.iterrows()
            if (r.session, int(r.unit)) in peth][:n]
    n_done_sess = m.session.nunique()
    n_resp = int(m.resp_log2.sum())
    if not keys:
        raise SystemExit(f"{subj}: no responsive units with PETHs yet "
                         f"({len(m)} units / {n_done_sess} sessions done)")
    nc = 4
    nr = int(np.ceil(len(keys) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 2.6 * nr), squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // nc][i % nc]
        r = resp[(resp.session == k[0]) & (resp.unit == k[1])].iloc[0]
        panel(ax, t, peth[k], f"{k[0]} u{k[1]}\nC1={r.c1_r_log2:.2f} p={r.c2_p_log2:.1e}")
        if i == 0:
            ax.legend(fontsize=6, frameon=False, loc="best")
        if i % nc == 0:
            ax.set_ylabel("Δ firing (Hz)")
        if i // nc == nr - 1:
            ax.set_xlabel("time from TF pulse (s)")
    for j in range(len(keys), nr * nc):
        axes[j // nc][j % nc].axis("off")
    fig.suptitle(f"{subj}: top TF-responsive units (PARTIAL run: {n_done_sess} sessions "
                 f"done, {n_resp} responsive / {len(m)} units) — fast vs slow TF pulse",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n_done_sess} sessions, {n_resp} responsive, {len(keys)} shown)")


if __name__ == "__main__":
    main()
