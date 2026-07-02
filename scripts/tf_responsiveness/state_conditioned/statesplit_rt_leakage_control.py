"""Control (a): is the sustained-cell "change response is larger in Impulsive
than StimSens" (within-cell paired, p=1.4e-3) a genuine gain effect, or lick /
RT leakage into the (0,0.25) change window?

Two controls, both WITHIN-cell paired (StimSens vs Impulsive), per class:
  resp_full     : canonical Change_ON response (0,0.25) - base(-0.4,-0.05), hit trials.
  resp_censored : LICK-CENSORED — per trial the response window is (0, min(0.25, RT)),
                  i.e. spikes AFTER the lick are excluded so motor activity can't
                  leak in. Trials with pre-lick window < 0.05 s dropped.
  resp_clean    : RT-MATCHED-by-exclusion — only hits with RT > 0.25 s, so the whole
                  (0,0.25) window is pre-lick for BOTH states.
Also reports the RT distributions per state (does Impulsive really have shorter RTs?).
If Impulsive>StimSens survives censoring AND the RT>0.25 subset, it is a real
integrator-cell gain increase, not lick leakage. Non-parametric paired (Wilcoxon).

RT = Hit lick time - Change_ON time (both from get_event_times_by_trial, hit trials).
State from data/cache/state_tags. One session load each; cache to CSV.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import wilcoxon, mannwhitneyu

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO, _spikes, load_session, get_event_times_by_trial  # noqa: E402
from transient_vs_sustained import load_cells, TCOL, SCOL                     # noqa: E402
from population_geometry import _state_by_trial                              # noqa: E402
from visdetect.analysis.constants import EVENT_RESPONSIVENESS_WINDOWS         # noqa: E402

CH_BASE, CH_RESP = EVENT_RESPONSIVENESS_WINDOWS["Change_ON"]   # (-0.4,-0.05),(0,0.25)
RESP_CAP = CH_RESP[1]          # 0.25
RT_FLOOR = 0.05                # need >=50ms pre-lick window to keep a censored trial
STATES2 = ["StimSens", "Impulsive"]
MIN_TR = 5
OUT = Path("E:/python_analysis/git_repos/vd_tf_bg046/FIGURES/tf_glm_bg046/statesplit_rt_leakage")
CACHE = OUT / "rt_leakage_metrics.csv"


def _count(spk, a, b):
    return np.searchsorted(spk, b) - np.searchsorted(spk, a)


def _base_rate(spk, ct):
    return _count(spk, ct + CH_BASE[0], ct + CH_BASE[1]) / (CH_BASE[1] - CH_BASE[0])


def _metrics(spk, changes, rts):
    """(resp_full, resp_censored, resp_clean, n_full, n_cens, n_clean, med_rt)."""
    full, cens, clean = [], [], []
    for ct, rt in zip(changes, rts):
        b = _base_rate(spk, ct)
        full.append(_count(spk, ct + CH_RESP[0], ct + CH_RESP[1]) / (CH_RESP[1] - CH_RESP[0]) - b)
        w = min(RESP_CAP, rt) if np.isfinite(rt) else RESP_CAP
        if w >= RT_FLOOR:
            cens.append(_count(spk, ct, ct + w) / w - b)
        if np.isfinite(rt) and rt > RESP_CAP:
            clean.append(_count(spk, ct + CH_RESP[0], ct + CH_RESP[1]) / (CH_RESP[1] - CH_RESP[0]) - b)
    f = np.mean(full) if len(full) >= MIN_TR else np.nan
    c = np.mean(cens) if len(cens) >= MIN_TR else np.nan
    cl = np.mean(clean) if len(clean) >= MIN_TR else np.nan
    return f, c, cl, len(full), len(cens), len(clean), (np.nanmedian(rts) if len(rts) else np.nan)


def session_rows(subj, sess, gcells):
    lab = _state_by_trial(subj, sess)
    if lab is None:
        return []
    s = load_session(f"{REPO}/data/pkls/{subj}/{sess}.pkl")
    et_ch = np.asarray(get_event_times_by_trial(s, "Change_ON"), float)
    et_hit = np.asarray(get_event_times_by_trial(s, "Hit"), float)
    outc = [str(getattr(t, "trialoutcome", "") or "").lower() for t in s.trials]
    # hit-trial index sets per engaged state, with change time + RT
    per_state = {}
    for st in STATES2:
        idx = [i for i in range(len(s.trials))
               if i < et_ch.size and outc[i] == "hit" and np.isfinite(et_ch[i])
               and i < et_hit.size and np.isfinite(et_hit[i]) and lab.get(i) == st]
        ct = np.array([et_ch[i] for i in idx], float)
        rt = np.array([et_hit[i] - et_ch[i] for i in idx], float)
        ok = rt > 0                       # lick after change (sane)
        per_state[st] = (ct[ok], rt[ok])
    rows = []
    for _, r in gcells.iterrows():
        uid = int(r["unit"])
        spk = np.sort(_spikes(s, uid))
        if spk.size == 0:
            continue
        rec = dict(subject=subj, session=sess, unit=uid, cls=r["class"])
        for st in STATES2:
            ct, rt = per_state[st]
            f, c, cl, nf, nc, ncl, mrt = _metrics(spk, ct, rt)
            rec[f"full_{st}"] = f; rec[f"cens_{st}"] = c; rec[f"clean_{st}"] = cl
            rec[f"n_{st}"] = nf; rec[f"nclean_{st}"] = ncl; rec[f"rt_{st}"] = mrt
        rows.append(rec)
    del s
    gc.collect()
    return rows


def compute_or_load(force=False):
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE)
    cells = load_cells()
    cells = cells[cells["class"].isin(["transient", "sustained"])]
    allrows = []
    for (subj, sess), g in cells.groupby(["subject", "session"]):
        if not Path(f"{REPO}/data/pkls/{subj}/{sess}.pkl").exists():
            continue
        allrows += session_rows(subj, sess, g)
        print(f"  {subj}/{sess}: {len(g)} cells", flush=True)
    df = pd.DataFrame(allrows)
    df["region"] = df.subject.map({"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"})
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def _paired(d, a, b):
    dd = d[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(dd) < 6:
        return np.nan, np.nan, len(dd), np.nan
    p = wilcoxon(dd[a], dd[b]).pvalue
    return float(dd[a].median()), float(dd[b].median()), len(dd), float(p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    df = compute_or_load(force=a.force)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    lines = []
    METRICS = [("full", "Change resp (0-0.25, uncensored)"),
               ("cens", "LICK-CENSORED (0..lick)"),
               ("clean", "RT>0.25 clean subset")]

    # premise: are Impulsive hits faster?
    for cls in ["sustained", "transient"]:
        d = df[df.cls == cls]
        ss = d["rt_StimSens"].replace([np.inf, -np.inf], np.nan).dropna()
        im = d["rt_Impulsive"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ss) >= 6 and len(im) >= 6:
            _, pu = mannwhitneyu(ss, im)
            lines.append(f"[RT median per-cell {cls}] StimSens={ss.median():.3f}s Impulsive={im.median():.3f}s (MWU p={pu:.2e})")

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.32)
    for row, cls in enumerate(["sustained", "transient"]):
        d = df[df.cls == cls]
        for col, (m, mlab) in enumerate(METRICS):
            ax = fig.add_subplot(gs[row, col])
            sa, sb, n, p = _paired(d, f"{m}_StimSens", f"{m}_Impulsive")
            dd = d[[f"{m}_StimSens", f"{m}_Impulsive"]].replace([np.inf, -np.inf], np.nan).dropna()
            for i in range(len(dd)):
                ax.plot([0, 1], dd.iloc[i].values, color="0.7", lw=0.4, alpha=0.4, zorder=1)
            for xi, cc in [(0, "#6baed6"), (1, "#ef6548")]:
                v = dd.iloc[:, xi]
                ax.scatter(np.full(len(v), xi) + (np.random.default_rng(xi).random(len(v)) - 0.5) * 0.1,
                           v, s=12, color=cc, alpha=0.6, zorder=2, edgecolors="none")
                ax.hlines(v.median(), xi - 0.2, xi + 0.2, color="k", lw=2.5, zorder=3)
            col_c = SCOL if cls == "sustained" else TCOL
            ax.set_title(f"{cls}: {mlab}\nSS={sa:.2f} Imp={sb:.2f}  p={p:.1e}  n={n}",
                         fontsize=9.5, color=col_c)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["StimSens", "Impulsive"], fontsize=9)
            ax.set_ylabel("change resp (Hz)" if col == 0 else "")
            ax.axhline(0, color="0.7", lw=0.7, ls=":")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            lines.append(f"[{cls} {m}] SS={sa:.3f} Imp={sb:.3f} n={n} paired-Wilcoxon p={p:.2e}")
            # per region
            for reg in ["DMS", "VMS"]:
                sa2, sb2, n2, p2 = _paired(d[d.region == reg], f"{m}_StimSens", f"{m}_Impulsive")
                lines.append(f"    {cls} {m} {reg}: SS={sa2:.3f} Imp={sb2:.3f} n={n2} p={p2 if p2==p2 else float('nan'):.2e}")

    fig.suptitle("Control (a): sustained-cell change response StimSens vs Impulsive — lick-censoring & RT controls\n"
                 "if Impulsive>StimSens survives censoring + RT>0.25 subset, it is a gain effect, not lick leakage",
                 fontsize=12.5, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"statesplit_rt_leakage.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (OUT / "rt_leakage_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}/statesplit_rt_leakage.png")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
