"""β_TF SUPPORT-matched control — the last cheap gate before the dense refit.

The count-matched control removed the β_FA-reliability confound but left the β_TF-SUPPORT
confound (responsive-cell count covaries with impulsivity; θ tracks support). Here we match
BOTH at once, per session: subsample β_FA to a fixed K FA-trials AND the responsive cells to a
fixed count S. With reliability AND support both constant, any residual θ-vs-impulsivity is
neither confound. If BG_031 survives → refit justified; if it dies → clean null, no refit needed.
S-sweep like the K-sweep. Reads PRIMARY read-only; writes worktree FIGURES.
"""
import os, sys, re
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor
WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WT, "src"))
from visdetect.analysis import config

PRIMARY = "e:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
D = f"{PRIMARY}/data"
OUT = f"{WT}/FIGURES/popgeom_theta"
MICE = [("BG_046", "DMS", "#2c7fb8"), ("BG_039", "DMS", "#31a354"), ("BG_031", "VMS", "#d6322a")]
PRE, POST = (-1.75, -1.25), (-0.30, -0.15)
K_FA = 60           # β_FA reliability match (from count-matched control)
M = 60              # subsamples
SEED = 42


def cid(x):
    try:
        return config.canonical_session_id(str(x))
    except Exception:
        return str(x)


def _counts(spk, los, his):
    return np.searchsorted(spk, his) - np.searchsorted(spk, los)


def _session(task):
    subj, msession, gs_units = task
    from visdetect.core.session import load_session
    from visdetect.analysis.align import get_event_times_by_trial, align_spikes_to_events
    from visdetect.analysis.preparatory import baseline_mean_sd
    pkl = f"{PRIMARY}/data/pkls/{subj}/{msession}.pkl"
    if not os.path.exists(pkl):
        return {"err": f"MISSING {pkl}"}
    try:
        s = load_session(pkl)
        bon = np.asarray(get_event_times_by_trial(s, "Baseline_ON"), float)
        fa = np.asarray(get_event_times_by_trial(s, "FA"), float)
        change = np.asarray(get_event_times_by_trial(s, "Change_ON"), float)
        change_t = change[np.isfinite(change)]
        cc = np.isfinite(fa) & np.isfinite(bon) & ((fa - bon) >= 1.75)
        fa_L = np.sort(fa[cc])
        if len(fa_L) < K_FA or len(change_t) < 5:
            return {"err": None, "subj": subj, "msession": msession, "units": [], "ramps": None}
        gs = set(int(u) for u in gs_units)
        units, ramps = [], []
        for c in s.clusters:
            uid = int(getattr(c, "cluster_id", -1))
            if uid not in gs:
                continue
            spk = np.sort(np.asarray(c.spike_times, float).ravel())
            if spk.size == 0:
                continue
            b_binned, _ = align_spikes_to_events(spk, list(change_t), window=(-2.0, 0.0), bin_size=0.025)
            _, sd = baseline_mean_sd(b_binned)
            if not np.isfinite(sd) or sd < 1e-6:
                continue
            pre = _counts(spk, fa_L + PRE[0], fa_L + PRE[1]) / (PRE[1] - PRE[0])
            post = _counts(spk, fa_L + POST[0], fa_L + POST[1]) / (POST[1] - POST[0])
            ramps.append((post - pre) / sd)
            units.append(uid)
        del s
        return {"err": None, "subj": subj, "msession": msession,
                "units": units, "ramps": np.asarray(ramps) if ramps else None}
    except Exception as e:
        import traceback
        return {"err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


if __name__ == "__main__":
    pf = np.load(f"{D}/cache/preparatory_fig5/prep_fa.npz", allow_pickle=True)
    meta = pd.DataFrame({"subj": pf["meta_subject"].astype(str), "msession": pf["meta_session"].astype(str),
                         "unit": pf["meta_unit"].astype(int)})
    tasks = [(subj, ms, g.unit.tolist()) for (subj, ms), g in meta.groupby(["subj", "msession"])]
    print(f"loading {len(tasks)} sessions ...", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(_session, tasks):
            if r.get("err"):
                print("  ERR", r["err"].splitlines()[0], flush=True)
            elif r.get("ramps") is not None:
                res.append(r)
    print(f"  {len(res)} sessions usable (>= {K_FA} FAs)", flush=True)

    kv_peak = {}
    for subj, _, _ in MICE:
        kv = np.load(f"{D}/cache/tf_glm_bg046/kernel_vectors_{subj}.npz", allow_pickle=True)
        for k in kv.keys():
            if k not in ("lags", "units"):
                kv_peak[k] = float(kv[k][np.argmax(np.abs(kv[k]))])

    beh = []
    for subj, _, _ in MICE:
        m = pd.read_csv(f"{D}/{subj}_staging_manifest.csv", dtype={"session_name": str})
        m["sid"] = m.session_name.map(cid)
        tot = m[["early_licks", "hits", "misses", "fas", "crs"]].sum(axis=1).replace(0, np.nan)
        beh.append(pd.DataFrame({"subject": subj, "sid": m.sid, "fa_rate_behav": m.early_licks / tot}))
    beh = pd.concat(beh, ignore_index=True)

    # precompute per-session: ramps, responsive indices, full bTF
    sess = []
    for r in res:
        units = np.array(r["units"]); ramps = r["ramps"]
        bTF = np.array([kv_peak.get(f"{r['msession']}_u{int(u)}", 0.0) for u in units])
        resp_idx = np.where(bTF != 0)[0]
        sess.append(dict(subj=r["subj"], sid=cid(re.sub(r"^BG_\d+_", "", r["msession"])),
                         ramps=ramps, bTF=bTF, resp_idx=resp_idx, support=len(resp_idx)))

    def theta_both_matched(S, rng):
        rows = []
        for e in sess:
            if e["support"] < S:
                continue
            ramps, bTF, ridx = e["ramps"], e["bTF"], e["resp_idx"]
            nfa = ramps.shape[1]
            coss = []
            for _ in range(M):
                bFA = np.nanmean(ramps[:, rng.choice(nfa, K_FA, replace=False)], axis=1)
                sub = rng.choice(ridx, S, replace=False)
                bT = np.zeros_like(bTF); bT[sub] = bTF[sub]
                nb, nf = np.linalg.norm(bT), np.linalg.norm(bFA)
                if nb > 1e-12 and nf > 1e-12:
                    coss.append(abs(float(np.dot(bT, bFA))) / (nb * nf))
            if coss:
                rows.append(dict(subject=e["subj"], sid=e["sid"], support=e["support"],
                                 theta=float(np.degrees(np.arccos(np.clip(np.mean(coss), 0, 1))))))
        return pd.DataFrame(rows).merge(beh, on=["subject", "sid"], how="left")

    print(f"\n=== S-SWEEP: θ (β_FA matched K={K_FA} AND β_TF support matched to S) vs impulsivity ===")
    supports = np.array([e["support"] for e in sess])
    print(f"  support/session: min={supports.min()} median={np.median(supports):.0f} max={supports.max()}")
    sweep = {}
    for S in [3, 4, 5]:
        u = theta_both_matched(S, np.random.default_rng(SEED)).dropna(subset=["fa_rate_behav"])
        sweep[S] = u
        parts = []
        for subj, _, _ in MICE:
            s = u[u.subject == subj]
            if len(s) >= 6:
                rr, pp = spearmanr(s.theta, s.fa_rate_behav)
                parts.append(f"{subj} ρ={rr:+.2f}(p{pp:.2f},n{len(s)})")
            else:
                parts.append(f"{subj} n{len(s)}")
        print(f"  S={S}: " + " | ".join(parts))

    Sp = 4
    u = sweep[Sp]; u.to_csv(f"{OUT}/theta_support_matched.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for subj, region, col in MICE:
        s = u[u.subject == subj].dropna(subset=["theta", "fa_rate_behav"])
        ax[0].scatter(s.fa_rate_behav, s.theta, s=42, color=col, alpha=0.8, label=f"{subj} ({region}) n={len(s)}")
        if len(s) >= 5:
            b, a0 = np.polyfit(s.fa_rate_behav, s.theta, 1); xx = np.linspace(s.fa_rate_behav.min(), s.fa_rate_behav.max(), 20)
            ax[0].plot(xx, a0 + b * xx, color=col, lw=1.6, alpha=0.7)
    ax[0].axhline(90, color="0.7", ls="--", lw=1)
    ax[0].set_xlabel("behavioural FA rate (impulsivity) →"); ax[0].set_ylabel(f"θ (K={K_FA}, S={Sp}, deg)")
    ax[0].set_title(f"θ with BOTH confounds matched\n(β_FA reliability + β_TF support), S={Sp}"); ax[0].legend(frameon=False, fontsize=8)
    Ss = [3, 4, 5]; x = np.arange(len(Ss)); w = 0.26; cols = {s: c for s, _, c in MICE}
    for i, (subj, _, col) in enumerate(MICE):
        vals = []
        for S in Ss:
            s = sweep[S][sweep[S].subject == subj].dropna(subset=["theta", "fa_rate_behav"])
            vals.append(spearmanr(s.theta, s.fa_rate_behav)[0] if len(s) >= 6 else np.nan)
        ax[1].bar(x + (i - 1) * w, vals, w, color=col, label=subj)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels([f"S={S}" for S in Ss])
    ax[1].set_ylabel("Spearman ρ (θ vs FA-rate)"); ax[1].set_title("S-sweep: does the effect survive\nmatched support?"); ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(f"Support-matched control — both β_FA reliability (K={K_FA}) AND β_TF support (S) held constant", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT}/theta_support_matched.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {OUT}/theta_support_matched.png + theta_support_matched.csv")
