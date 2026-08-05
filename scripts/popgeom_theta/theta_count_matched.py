"""COUNT-MATCHED β_FA control — the gating test for the θ prototype.

The prototype's θ↓-with-impulsivity was the β_FA attenuation artifact: FA-rate ≈ FA-count
(ρ~0.8), so impulsive sessions had cleaner β_FA and smaller θ. This breaks that collinearity:
recompute β_FA from spikes (per-FA-trial pre-lick ramp, complete-case ≥1.75s, z-normalised to the
pre-change baseline like prep_fa), then average over M subsamples of a FIXED K FA trials — every
session's β_FA now has EQUAL reliability. If θ_matched still tracks impulsivity → real geometry;
if it collapses like θ_unmatched → confirmed artifact. β_TF = cached signed GLM kernel (as prototype).
Reads PRIMARY tree read-only; writes worktree FIGURES.
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
os.makedirs(OUT, exist_ok=True)
MICE = [("BG_046", "DMS", "#2c7fb8"), ("BG_039", "DMS", "#31a354"), ("BG_031", "VMS", "#d6322a")]
PRE, POST = (-1.75, -1.25), (-0.30, -0.15)
MIN_FA_LOAD = 25          # skip sessions with < this many complete-case FAs
M_SUB = 30                # subsamples per session at matched K
SEED = 42


def cid(x):
    try:
        return config.canonical_session_id(str(x))
    except Exception:
        return str(x)


def _counts(spk, los, his):
    return np.searchsorted(spk, his) - np.searchsorted(spk, los)


def _session(task):
    """Return per-unit z-scored ramp for every complete-case FA trial: (n_units, n_fa)."""
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
        cc = np.isfinite(fa) & np.isfinite(bon) & ((fa - bon) >= 1.75)   # complete-case
        fa_L = np.sort(fa[cc])
        if len(fa_L) < MIN_FA_LOAD or len(change_t) < 5:
            return {"err": None, "subj": subj, "msession": msession, "n_fa": int(len(fa_L)), "units": [], "ramps": None}
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
            ramps.append((post - pre) / sd)     # z-scored ramp per FA trial
            units.append(uid)
        del s
        return {"err": None, "subj": subj, "msession": msession, "n_fa": int(len(fa_L)),
                "units": units, "ramps": np.asarray(ramps) if ramps else None}
    except Exception as e:
        import traceback
        return {"err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


def theta_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))) / (na * nb), 0.0, 1.0)))


if __name__ == "__main__":
    # session list + good_and_stable unit ids per session (from prep_fa meta)
    pf = np.load(f"{D}/cache/preparatory_fig5/prep_fa.npz", allow_pickle=True)
    meta = pd.DataFrame({"subj": pf["meta_subject"].astype(str), "msession": pf["meta_session"].astype(str),
                         "unit": pf["meta_unit"].astype(int)})
    tasks = [(subj, ms, g.unit.tolist()) for (subj, ms), g in meta.groupby(["subj", "msession"])]
    print(f"loading {len(tasks)} sessions (spikes) ...", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(_session, tasks):
            if r.get("err"):
                print("  ERR", r["err"].splitlines()[0], flush=True)
            elif r.get("ramps") is not None:
                res.append(r)
    print(f"  {len(res)} sessions with usable spikes", flush=True)

    # β_TF from cached signed kernels
    kv_peak = {}
    for subj, _, _ in MICE:
        kv = np.load(f"{D}/cache/tf_glm_bg046/kernel_vectors_{subj}.npz", allow_pickle=True)
        for k in kv.keys():
            if k not in ("lags", "units"):
                kv_peak[k] = float(kv[k][np.argmax(np.abs(kv[k]))])

    fa_counts = np.array([r["n_fa"] for r in res])
    print(f"FA-count per session: min={fa_counts.min()} p20={np.percentile(fa_counts,20):.0f} "
          f"median={np.median(fa_counts):.0f} max={fa_counts.max()}", flush=True)

    # behavioural FA rate
    beh = []
    for subj, _, _ in MICE:
        m = pd.read_csv(f"{D}/{subj}_staging_manifest.csv", dtype={"session_name": str})
        m["sid"] = m.session_name.map(cid)
        tot = m[["early_licks", "hits", "misses", "fas", "crs"]].sum(axis=1).replace(0, np.nan)
        beh.append(pd.DataFrame({"subject": subj, "sid": m.sid, "fa_rate_behav": m.early_licks / tot}))
    beh = pd.concat(beh, ignore_index=True)

    def theta_df(K, rng):
        rows = []
        for r in res:
            if r["n_fa"] < K:
                continue
            units = np.array(r["units"]); ramps = r["ramps"]
            bTF = np.array([kv_peak.get(f"{r['msession']}_u{int(u)}", 0.0) for u in units])
            bFA_unm = np.nanmean(ramps, axis=1)
            bFA_mat = np.mean([np.nanmean(ramps[:, rng.choice(ramps.shape[1], K, replace=False)], axis=1)
                               for _ in range(M_SUB)], axis=0)
            rows.append(dict(subject=r["subj"], sid=cid(re.sub(r"^BG_\d+_", "", r["msession"])),
                             n_fa=r["n_fa"], support=int((bTF != 0).sum()),
                             theta_unmatched=theta_deg(bTF, bFA_unm), theta_matched=theta_deg(bTF, bFA_mat)))
        return pd.DataFrame(rows).merge(beh, on=["subject", "sid"], how="left")

    def partial_spearman(x, y, z):
        d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
        if len(d) < 6:
            return np.nan
        rx, ry, rz = d.x.rank(), d.y.rank(), d.z.rank()
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return spearmanr(ex, ey)[0]

    print("\n=== K-SWEEP: θ_matched vs impulsivity at fixed β_FA reliability (robustness) ===")
    for K in [40, 60, 91]:
        u = theta_df(K, np.random.default_rng(SEED))
        u = u[u.support >= 3].dropna(subset=["fa_rate_behav"])
        parts = []
        for subj, _, _ in MICE:
            s = u[u.subject == subj]
            if len(s) >= 6:
                r, p = spearmanr(s.theta_matched, s.fa_rate_behav)
                parts.append(f"{subj} ρ={r:+.2f}(p{p:.2f},n{len(s)})")
            else:
                parts.append(f"{subj} n{len(s)}")
        print(f"  K={K:3d}: " + " | ".join(parts))

    K = 60                                  # primary: keeps more sessions than 91
    df = theta_df(K, np.random.default_rng(SEED))
    use = df[df.support >= 3].dropna(subset=["fa_rate_behav"])
    use.to_csv(f"{OUT}/theta_count_matched.csv", index=False)

    print(f"\n=== primary K={K}: unmatched vs matched + β_TF-SUPPORT confound check ===")
    summ = []
    for subj, _, _ in MICE:
        s = use[use.subject == subj]
        if len(s) < 6:
            print(f"  {subj}: n={len(s)} (<6, skip)"); continue
        ru, pu = spearmanr(s.theta_unmatched, s.fa_rate_behav)
        rm, pm = spearmanr(s.theta_matched, s.fa_rate_behav)
        r_sf = spearmanr(s.support, s.fa_rate_behav)[0]       # support ~ impulsivity?
        r_ts = spearmanr(s.theta_matched, s.support)[0]       # θ ~ support?
        par = partial_spearman(s.theta_matched.values, s.fa_rate_behav.values, s.support.values.astype(float))
        print(f"  {subj} (n={len(s)}): θ_unm ρ={ru:+.2f}(p{pu:.2f}) → θ_mat ρ={rm:+.2f}(p{pm:.2f}); "
              f"support~FA ρ={r_sf:+.2f}, θ~support ρ={r_ts:+.2f}, θ~FA|support ρ={par:+.2f}")
        summ.append((subj, ru, rm, len(s)))

    # figure: A) θ vs impulsivity at matched reliability; B) the β_TF-support confound; C) ρ ladder
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    cols = {s: c for s, _, c in MICE}
    ladder = []
    for subj, region, col in MICE:
        s = use[use.subject == subj].dropna(subset=["theta_matched", "fa_rate_behav", "support"])
        if not len(s):
            continue
        ax[0].scatter(s.fa_rate_behav, s.theta_matched, s=42, color=col, alpha=0.8, label=f"{subj} ({region}) n={len(s)}")
        ax[1].scatter(s.support, s.theta_matched, s=42, color=col, alpha=0.8, label=subj)
        if len(s) >= 5:
            for a_, xv in ((ax[0], s.fa_rate_behav), (ax[1], s.support)):
                b, a0 = np.polyfit(xv, s.theta_matched, 1); xx = np.linspace(xv.min(), xv.max(), 20)
                a_.plot(xx, a0 + b * xx, color=col, lw=1.6, alpha=0.7)
        if len(s) >= 6:
            ladder.append((subj, spearmanr(s.theta_matched, s.fa_rate_behav)[0],
                           partial_spearman(s.theta_matched.values, s.fa_rate_behav.values, s.support.values.astype(float))))
    ax[0].axhline(90, color="0.7", ls="--", lw=1); ax[0].set_xlabel("behavioural FA rate (impulsivity) →")
    ax[0].set_ylabel(f"θ_matched (K={K} FAs, deg)"); ax[0].set_title("A. θ (matched β_FA reliability)\nvs impulsivity"); ax[0].legend(frameon=False, fontsize=7.5)
    ax[1].axhline(90, color="0.7", ls="--", lw=1); ax[1].set_xlabel("β_TF support (# responsive cells) →")
    ax[1].set_ylabel("θ_matched (deg)"); ax[1].set_title("B. CONFOUND #2: θ tracks β_TF support\n(support also covaries with impulsivity)"); ax[1].legend(frameon=False, fontsize=7.5)
    x = np.arange(len(ladder)); w = 0.36
    ax[2].bar(x - w/2, [m for _, m, _ in ladder], w, color=[cols[s] for s, *_ in ladder], alpha=0.5, label="θ_matched ρ")
    ax[2].bar(x + w/2, [p for _, _, p in ladder], w, color=[cols[s] for s, *_ in ladder], edgecolor="k", label="ρ | support")
    ax[2].axhline(0, color="k", lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels([s for s, *_ in ladder])
    ax[2].set_ylabel("Spearman ρ (θ vs FA-rate)"); ax[2].set_title("C. Matched effect HALVES\nafter partialling support"); ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(f"Count-matched β_FA (K={K}): BG_046 null; BG_031 survives reliability-matching but is ~half β_TF-support confound "
                 f"(sparse responsive-only β_TF → refit needed for a clean test)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{OUT}/theta_count_matched.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {OUT}/theta_count_matched.png + theta_count_matched.csv | sessions used={len(use)}")
