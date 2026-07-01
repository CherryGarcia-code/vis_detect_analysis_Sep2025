"""B9 engagement test — PAIRED StimSens vs Disengaged TF-encoding, within-session, matched-N.

For sessions with enough of BOTH states, subsample each state to the SAME per-session N
(min of the two, capped), run the registry TF-GLM on the SAME responsive units in each
state, average over draws, and pair per unit. Because both states use the same N and the
same unit/session, c1_r's trial-count attenuation cancels in the paired difference
(StimSens - Disengaged), giving a clean engagement (attention-gating) estimate.

Run:  PYTHONPATH=src py scripts/state_tf_learning/b9_engagement.py --subjects BG_031,BG_039,BG_046
"""
import os, sys, gc, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
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
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
from visdetect.analysis import state_tf_learning as stl

FLOOR = 80       # min of (StimSens, Disengaged) trials per session
N_MAX = 200      # cap the matched N
K_DRAWS = 2
MIN_RESP = 5


def eligible(subject):
    reg = stl.load_registry(stl.registry_path(subject))
    nr = reg.groupby("sess_key")["resp_log2"].sum()
    meta = reg.drop_duplicates("sess_key").set_index("sess_key")[["session", "session_date"]]
    smap = stl.date_stage_map(subject)
    rows = []
    for key, m in meta.iterrows():
        try:
            tags = stl.load_state_tags(subject, str(m["session_date"]))
        except FileNotFoundError:
            continue
        g = tags[tags["state_confidence"] >= 0.8]; vc = g["state_label"].value_counts()
        nss, ndis = int(vc.get("StimSens", 0)), int(vc.get("Disengaged", 0))
        if min(nss, ndis) >= FLOOR and nr.get(key, 0) >= MIN_RESP:
            rows.append({"sess_key": key, "stem": m["session"], "orig": str(m["session_date"]),
                         "stage": smap.get(key, "?"), "n_resp": int(nr.get(key, 0)),
                         "N": int(min(nss, ndis, N_MAX))})
    return pd.DataFrame(rows), reg


def _run(args):
    subject, key, stem, orig, stage, state, resp, N, k, seed = args
    from visdetect.analysis import state_tf_learning as _stl
    from visdetect.core.session import load_session as _load
    import numpy as _np, pandas as _pd, gc as _gc
    pkl = _stl.PKL_ROOT / subject / f"{stem}.pkl"
    if not pkl.exists():
        return None
    sess = _load(str(pkl)); tags = _stl.load_state_tags(subject, orig)
    idx = _stl.state_trial_indices(tags, state)
    if len(idx) < N:
        del sess; _gc.collect(); return None
    cfg = _stl.b9_cfg(); rng = _np.random.default_rng(seed); draws = []
    try:
        for d in range(k):
            sub = sorted(int(i) for i in rng.choice(idx, size=N, replace=False))
            df = _stl.state_conditioned_encoding(sess, sub, cfg, unit_ids=list(resp))
            if not df.empty:
                draws.append(df)
    except Exception:
        del sess; _gc.collect(); return None
    del sess; _gc.collect()
    if not draws:
        return None
    agg = _pd.concat(draws, ignore_index=True).groupby("unit")["c1_r"].mean().reset_index()
    agg["subject"] = subject; agg["sess_key"] = key; agg["stage"] = stage; agg["state"] = state
    return agg


def main(subjects, n_workers):
    cache = stl._REPO / "data" / "cache" / "state_tf_learning"; cache.mkdir(parents=True, exist_ok=True)
    figdir = stl._REPO / "FIGURES" / "state_tf_learning"; figdir.mkdir(parents=True, exist_ok=True)
    tasks = []; seed = 0
    for subj in subjects:
        cand, reg = eligible(subj)
        print(f"{subj}: {len(cand)} eligible sessions (stages={dict(cand.stage.value_counts()) if len(cand) else {}})", flush=True)
        for _, r in cand.iterrows():
            rs = reg[reg.sess_key == r.sess_key]
            resp = rs.loc[rs.resp_log2 == True, "unit"].astype(int).tolist()   # noqa: E712
            for state in ["StimSens", "Disengaged"]:
                tasks.append((subj, r.sess_key, r.stem, r.orig, r.stage, state, resp, int(r.N), K_DRAWS, seed)); seed += 1
    print(f"=== running {len(tasks)} (session x state) tasks on {n_workers} workers ===", flush=True)
    frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_run, tasks):
            if res is not None:
                frames.append(res)
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if d.empty:
        print("[done] no rows", flush=True); return
    piv = d.pivot_table(index=["subject", "sess_key", "unit", "stage"], columns="state", values="c1_r").reset_index()
    piv = piv.dropna(subset=["StimSens", "Disengaged"])
    piv["delta"] = piv["StimSens"] - piv["Disengaged"]
    piv.to_csv(cache / "b9_engagement_paired.csv", index=False)

    W, p = stats.wilcoxon(piv["StimSens"], piv["Disengaged"]) if len(piv) >= 6 else (np.nan, np.nan)
    print(f"\n[RESULT] PAIRED engagement (all subjects, n={len(piv)} unit-sessions):", flush=True)
    print(f"  StimSens median c1_r={piv.StimSens.median():.3f}  Disengaged median={piv.Disengaged.median():.3f}  "
          f"delta(SS-Dis) median={piv.delta.median():.3f}  Wilcoxon p={p:.2e}", flush=True)
    for subj in subjects:
        s = piv[piv.subject == subj]
        if len(s):
            wp = stats.wilcoxon(s.StimSens, s.Disengaged)[1] if len(s) >= 6 else np.nan
            print(f"  {subj}: n={len(s)}  SS={s.StimSens.median():.3f}  Dis={s.Disengaged.median():.3f}  "
                  f"delta={s.delta.median():.3f}  p={wp:.3g}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for _, r in piv.iterrows():
        ax1.plot([0, 1], [r.StimSens, r.Disengaged], "-", color="#bbb", lw=.5, alpha=.5, zorder=1)
    ax1.scatter(np.zeros(len(piv)), piv.StimSens, s=14, color="#3474ae", zorder=3, label="StimSens")
    ax1.scatter(np.ones(len(piv)), piv.Disengaged, s=14, color="#ef6548", zorder=3, label="Disengaged")
    ax1.axhline(0, color="#999", lw=.6)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["StimSens\n(engaged)", "Disengaged"])
    ax1.set_ylabel("c1_r (matched N, paired within session)")
    ax1.set_title(f"Paired engagement (n={len(piv)} unit-sessions)\nWilcoxon p={p:.1e}")
    ax1.legend(fontsize=8)
    ax2.axvline(0, color="k", lw=.8)
    ax2.hist(piv.delta, bins=25, color="#6baed6", edgecolor="k", lw=.3)
    ax2.axvline(piv.delta.median(), color="r", lw=1.5, label=f"median Δ={piv.delta.median():.3f}")
    ax2.set_xlabel("Δ c1_r  (StimSens − Disengaged)"); ax2.set_ylabel("unit-sessions"); ax2.legend(fontsize=8)
    ax2.set_title("engagement gain per unit")
    fig.suptitle("B9: does engagement gate baseline-TF encoding? (paired, matched-N)", fontsize=12)
    fig.tight_layout(); fig.savefig(figdir / "b9_engagement_paired.png", dpi=150); plt.close(fig)
    print("[done] wrote b9_engagement_paired.csv + figure", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="BG_031,BG_039,BG_046")
    ap.add_argument("--n_workers", type=int, default=8)
    a = ap.parse_args()
    main([s.strip() for s in a.subjects.split(",")], a.n_workers)
