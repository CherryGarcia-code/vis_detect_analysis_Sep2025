"""B9 Phase 0 — registry-only preliminary + per-session coverage (subject-parameterized).

For a subject with a TF-responsive registry, state tags, and a staging manifest:
  - join manifest (d', stage) + registry (responsive counts, whole-session c1_r)
    + state tags (confidence-gated StimSens/Disengaged coverage),
  - write an overview CSV,
  - draw the registry-only preliminary figure (encoding vs learning stage),
  - print per-stage c1_r + candidate early/late session picks.

No model fitting — this is the free, instant first look.

Run:  PYTHONPATH=src py scripts/state_tf_learning/b9_phase0_profile.py --subject BG_031
"""
import os, sys, glob, argparse, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from visdetect.analysis import state_tf_learning as stl
from visdetect.analysis.config import canonical_session_id as csid

STAGE_COL = {"Naive": "#c7c7c7", "Learning": "#f0a848", "Expert": "#3474ae", "Excluded": "#e05050"}
STAGE_ORDER = ["Naive", "Learning", "Expert"]


def robust_date(k):
    k = str(k).split("_")[0]
    if len(k) == 8 and k.isdigit():
        dd, mm, yyyy = int(k[:2]), int(k[2:4]), int(k[4:])
        if dd == 0:                    # 00280125 <- 6-digit DDMMYY zero-padded by csid
            s6 = k[2:]; dd, mm, yy = int(s6[:2]), int(s6[2:4]), int(s6[4:]); yyyy = 2000 + yy
        try:
            return datetime.date(yyyy, mm, dd)
        except Exception:
            return None
    return None


def build_overview(subject):
    m = pd.read_csv(stl.manifest_path(subject)); m["sess_key"] = m["session_name"].map(csid)
    man = m.drop_duplicates("sess_key").set_index("sess_key")[["d_prime", "stage"]]
    reg = stl.load_registry(stl.registry_path(subject))
    rmask = reg["resp_log2"] == True                                        # noqa: E712
    per = reg.groupby("sess_key").agg(n_units=("resp_log2", "count"), n_resp=("resp_log2", "sum")).join(
        reg[rmask].groupby("sess_key")["c1_r_log2"].mean().rename("mean_c1r_resp")).join(
        reg[~rmask].groupby("sess_key")["c1_r_log2"].mean().rename("mean_c1r_non"))
    rows = []
    for fp in glob.glob(str(stl.DEFAULT_STATES_DIR / subject / "*.csv")):
        b = os.path.basename(fp)
        if b.startswith("_"):
            continue
        t = pd.read_csv(fp); g = t[t["state_confidence"] >= 0.8]; vc = g["state_label"].value_counts()
        rows.append({"sess_key": csid(b.replace(".csv", "")),
                     "StimSens": int(vc.get("StimSens", 0)), "Diseng": int(vc.get("Disengaged", 0)),
                     "Impuls": int(vc.get("Impulsive", 0))})
    st = pd.DataFrame(rows).set_index("sess_key") if rows else pd.DataFrame()
    per = per.join(man, how="left")
    if len(st):
        per = per.join(st, how="left")
    per["n_resp"] = per["n_resp"].fillna(0).astype(int)
    per["date"] = [robust_date(k) for k in per.index]
    per = per[per["date"].notna()].sort_values("date")
    return per, reg.join(man, on="sess_key")


def main(subject):
    cache = stl._REPO / "data" / "cache" / "state_tf_learning"; cache.mkdir(parents=True, exist_ok=True)
    figdir = stl._REPO / "FIGURES" / "state_tf_learning" / subject; figdir.mkdir(parents=True, exist_ok=True)
    per, reg = build_overview(subject)
    per.to_csv(cache / f"b9_session_overview_{subject}.csv")

    ru = reg[reg["resp_log2"] == True].dropna(subset=["c1_r_log2"])          # noqa: E712
    stages = [s for s in STAGE_ORDER if s in set(per["stage"].dropna())]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.1, 1]})
    p = per.reset_index(); x = np.arange(len(p))
    for i, r in p.iterrows():
        axA.scatter(i, r["mean_c1r_resp"], s=25 + 30 * (r["n_resp"] if np.isfinite(r["n_resp"]) else 0),
                    color=STAGE_COL.get(str(r["stage"]), "#999"), edgecolor="k", lw=.4, zorder=3)
    axA.plot(x, p["mean_c1r_non"], "-", color="#888", lw=1, alpha=.7)
    axA.axhline(0.2, ls="--", color="k", lw=.8, alpha=.6)
    axA.set_xticks(x); axA.set_xticklabels([str(d) for d in p["date"]], rotation=90, fontsize=5)
    axA.set_ylabel("mean c1_r (responsive units)")
    axA.set_title(f"{subject}: baseline-TF encoding across recording (size ~ #responsive; registry)")
    axA.legend(handles=[plt.Line2D([0], [0], marker='o', ls='', mfc=STAGE_COL.get(s, "#999"), mec='k', label=s)
                        for s in stages] + [plt.Line2D([0], [0], color="#888", label="non-resp")],
               fontsize=8, loc="upper left")
    data = [ru.loc[ru["stage"] == s, "c1_r_log2"].to_numpy() for s in stages]
    bp = axB.boxplot(data, tick_labels=[f"{s}\n(n={len(d)})" for s, d in zip(stages, data)],
                     showfliers=False, widths=.5, patch_artist=True)
    for patch, s in zip(bp["boxes"], stages):
        patch.set_facecolor(STAGE_COL.get(s, "#999")); patch.set_alpha(.5)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        axB.scatter(np.full(len(d), i + 1) + rng.uniform(-.12, .12, len(d)), d, s=14, color="k", alpha=.55, zorder=3)
    axB.axhline(0.2, ls="--", color="k", lw=.8, alpha=.6); axB.set_ylabel("c1_r (per responsive unit)")
    axB.set_title("by learning stage")
    fig.tight_layout(); out = figdir / "b9_preliminary_trend.png"; fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[fig] {out}")

    print(f"\n=== {subject}: per-stage responsive-unit c1_r (whole-session registry) ===")
    for s in stages:
        d = ru.loc[ru["stage"] == s, "c1_r_log2"]
        print(f"  {s:9s} n_units={len(d):3d}  median={d.median():.3f}  mean={d.mean():.3f}")
    print(f"\n=== candidate sessions per stage (n_resp, StimSens, d') ===")
    cols = [c for c in ["n_resp", "StimSens", "Diseng", "d_prime"] if c in per.columns]
    for s in stages + ["Excluded"]:
        sub = per[per["stage"] == s].sort_values("n_resp", ascending=False)
        if not len(sub):
            continue
        print(f"  [{s}]")
        for k, r in sub.head(6).iterrows():
            ss = int(r["StimSens"]) if "StimSens" in r and np.isfinite(r["StimSens"]) else -1
            dp = r["d_prime"] if "d_prime" in r and np.isfinite(r["d_prime"]) else float("nan")
            print(f"      {k}  {r['date']}  n_resp={int(r['n_resp']):2d}  StimSens={ss:4d}  d'={dp:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=os.environ.get("VISDETECT_SUBJECT", "BG_039"))
    main(ap.parse_args().subject)
