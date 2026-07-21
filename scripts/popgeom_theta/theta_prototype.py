"""θ PROTOTYPE (cheap, no GLM refit) — does the sensory-motor population geometry
track behavioural regulation across sessions/mice?

  β_TF (sensory)  = signed peak of the cached GLM TF-kernel (kernel_vectors_*.npz),
                    responsive cells only (non-responsive → 0). No refit.
  β_FA (motor)    = FA-lick pre-lick ramp  z(-0.3,-0.15) − z(-1.75,-1.25)  read off
                    the cached prep_fa.npz lick-aligned z-traces (all good_and_stable cells).
  θ  = arccos(|β̂_TF · β̂_FA|), per session, over (a) the shared responsive support and
       (b) the full population; vs a neuron-identity shuffle null (500×).
  Test: θ per session vs that session's behavioural FA rate + d' (impulsivity/sensitivity),
        within-animal + pooled. Reads PRIMARY tree read-only; writes worktree FIGURES.
CAVEATS (prototype): β_TF is responsive-only & 8-day-stale cache; β_FA is LUMPED over all
FAs (not split at τ); per-session responsive support is thin (esp. BG_039). First look only.
"""
import os, sys, re
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WT, "src"))
from visdetect.analysis import config

PRIMARY = "e:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
D = f"{PRIMARY}/data"
OUT = f"{WT}/FIGURES/popgeom_theta"
os.makedirs(OUT, exist_ok=True)
MICE = [("BG_046", "DMS", "#2c7fb8"), ("BG_039", "DMS", "#31a354"), ("BG_031", "VMS", "#d6322a")]
N_SHUF = 500
MIN_SUPPORT = 3          # min responsive cells for a session's θ to be reported


def cid(x):
    try:
        return config.canonical_session_id(str(x))
    except Exception:
        return str(x)


def signed_peak(k):
    return float(k[np.argmax(np.abs(k))])


def theta_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    c = np.clip(abs(float(np.dot(a, b)) / (na * nb)), 0.0, 1.0)
    return np.degrees(np.arccos(c))


def shuffle_null_theta(a, b, rng, n=N_SHUF):
    """Neuron-identity shuffle: permute β_FA loadings, recompute θ. Returns null mean+sd (deg)."""
    out = []
    for _ in range(n):
        out.append(theta_deg(a, rng.permutation(b)))
    out = np.array([x for x in out if np.isfinite(x)])
    return (float(out.mean()), float(out.std())) if len(out) else (np.nan, np.nan)


# ── β_FA from prep_fa.npz ──────────────────────────────────────────────────────
pf = np.load(f"{D}/cache/preparatory_fig5/prep_fa.npz", allow_pickle=True)
t = pf["t"]
pre = (t >= -1.75) & (t <= -1.25); post = (t >= -0.30) & (t <= -0.15)
z = pf["z"]
bFA = np.nanmean(z[:, post], axis=1) - np.nanmean(z[:, pre], axis=1)   # per cell-session ramp
fa_df = pd.DataFrame({
    "subject": pf["meta_subject"].astype(str), "meta_session": pf["meta_session"].astype(str),
    "unit": pf["meta_unit"].astype(int), "region": pf["region"].astype(str),
    "resp": pf["resp"].astype(bool), "bFA": bFA,
})
fa_df["date"] = fa_df.meta_session.map(lambda s: re.sub(r"^BG_\d+_", "", s))
fa_df["sid"] = fa_df.date.map(cid)
fa_df["n_licks"] = pf["n_licks"].astype(float)   # FA-lick count = β_FA reliability proxy (attenuation confound)
fa_df = fa_df[np.isfinite(fa_df.bFA)]

# ── β_TF from kernel_vectors (signed peak; responsive cells) ────────────────────
kv_peak = {}
for subj, _, _ in MICE:
    kv = np.load(f"{D}/cache/tf_glm_bg046/kernel_vectors_{subj}.npz", allow_pickle=True)
    for key in kv.keys():
        if key in ("lags", "units"):
            continue
        kv_peak[key] = signed_peak(kv[key])
fa_df["bTF"] = [kv_peak.get(f"{ms}_u{u}", 0.0) for ms, u in zip(fa_df.meta_session, fa_df.unit)]

# ── per-session θ ──────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
rows = []
for (subj, sid), g in fa_df.groupby(["subject", "sid"]):
    bTF = g.bTF.values.astype(float); bFA = g.bFA.values.astype(float)
    supp = int((bTF != 0).sum())
    th_full = theta_deg(bTF, bFA)
    m = bTF != 0
    th_resp = theta_deg(bTF[m], bFA[m]) if supp >= 2 else np.nan
    null_mu, null_sd = shuffle_null_theta(bTF, bFA, rng) if supp >= 2 else (np.nan, np.nan)
    rows.append(dict(subject=subj, sid=sid, n_units=len(g), support=supp,
                     n_fa_licks=float(np.median(g.n_licks)),
                     theta_full=th_full, theta_resp=th_resp,
                     null_mu=null_mu, null_sd=null_sd,
                     dev_from_null=(th_full - null_mu) if np.isfinite(null_mu) else np.nan))
th_df = pd.DataFrame(rows)


def partial_spearman(x, y, z):
    """Spearman(x,y | z): correlate rank-residuals of x~z and y~z."""
    df = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(df) < 6:
        return np.nan, np.nan, len(df)
    rx, ry, rz = df.x.rank(), df.y.rank(), df.z.rank()
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = spearmanr(ex, ey)
    return r, p, len(df)

# ── behavioural regulation per session (manifest) ──────────────────────────────
beh = []
for subj, _, _ in MICE:
    m = pd.read_csv(f"{D}/{subj}_staging_manifest.csv", dtype={"session_name": str})
    m["sid"] = m.session_name.map(cid)
    tot = m[["early_licks", "hits", "misses", "fas", "crs"]].sum(axis=1).replace(0, np.nan)
    m["fa_rate_behav"] = m.early_licks / tot
    beh.append(m[["sid", "fa_rate_behav", "d_prime", "stage"]].assign(subject=subj))
beh = pd.concat(beh, ignore_index=True)
th_df = th_df.merge(beh, on=["subject", "sid"], how="left")
th_df.to_csv(f"{OUT}/theta_per_session.csv", index=False)

# ── report ─────────────────────────────────────────────────────────────────────
print("=== per-subject θ summary (sessions with support>=%d) ===" % MIN_SUPPORT)
use = th_df[th_df.support >= MIN_SUPPORT].copy()
for subj, _, _ in MICE:
    s = use[use.subject == subj]
    if not len(s):
        print(f"  {subj}: no sessions with support>={MIN_SUPPORT}"); continue
    print(f"  {subj}: n={len(s)} sess | median support={int(s.support.median())} | "
          f"θ_resp med={s.theta_resp.median():.1f}° | θ_full med={s.theta_full.median():.1f}° "
          f"(null med={s.null_mu.median():.1f}°)")
print("\n=== θ vs behavioural regulation (Spearman; session-level, per animal) ===")
for metric in ["theta_resp", "theta_full", "dev_from_null"]:
    print(f"  [{metric}]")
    for subj, _, _ in MICE:
        s = use[(use.subject == subj)].dropna(subset=[metric, "fa_rate_behav"])
        if len(s) >= 5:
            r1, p1 = spearmanr(s[metric], s.fa_rate_behav)
            r2, p2 = spearmanr(s[metric], s.d_prime.astype(float), nan_policy="omit")
            print(f"    {subj}: vs FA-rate ρ={r1:+.2f} (p={p1:.2f}, n={len(s)}) | vs d' ρ={r2:+.2f} (p={p2:.2f})")
        else:
            print(f"    {subj}: n={len(s)} (<5, skip)")

print("\n=== CONFOUND CONTROL: does θ_full vs FA-rate survive partialling FA-lick count (β_FA reliability)? ===")
print("    (raw ρ, then partial ρ | n_fa_licks; and how FA-count itself relates to FA-rate & θ)")
for subj, _, _ in MICE:
    s = use[use.subject == subj].dropna(subset=["theta_full", "fa_rate_behav", "n_fa_licks"])
    if len(s) < 6:
        print(f"    {subj}: n={len(s)} (<6, skip)"); continue
    raw_r, raw_p = spearmanr(s.theta_full, s.fa_rate_behav)
    par_r, par_p, n = partial_spearman(s.theta_full.values, s.fa_rate_behav.values, s.n_fa_licks.values)
    r_fc, p_fc = spearmanr(s.fa_rate_behav, s.n_fa_licks)     # is FA-rate confounded with FA-count?
    r_tc, p_tc = spearmanr(s.theta_full, s.n_fa_licks)        # is θ driven by FA-count (attenuation)?
    print(f"    {subj}: raw ρ={raw_r:+.2f}(p{raw_p:.2f}) → partial ρ={par_r:+.2f}(p{par_p:.2f}) | "
          f"FA-rate~count ρ={r_fc:+.2f} | θ~count ρ={r_tc:+.2f}")

# ── figure: the CONFOUND story (raw effect is FA-count attenuation) ─────────────
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
raws, pars = {}, {}
for subj, region, col in MICE:
    s = use[use.subject == subj].dropna(subset=["theta_full", "fa_rate_behav", "n_fa_licks"])
    # A: θ vs impulsivity (the raw, tempting view)
    ax[0].scatter(s.fa_rate_behav, s.theta_full, s=42, color=col, alpha=0.8, label=f"{subj} ({region}) n={len(s)}")
    if len(s) >= 5:
        b, a0 = np.polyfit(s.fa_rate_behav, s.theta_full, 1); xx = np.linspace(s.fa_rate_behav.min(), s.fa_rate_behav.max(), 20)
        ax[0].plot(xx, a0 + b * xx, color=col, lw=1.6, alpha=0.7)
    # B: θ vs FA-lick count (the attenuation confound)
    ax[1].scatter(s.n_fa_licks, s.theta_full, s=42, color=col, alpha=0.8, label=subj)
    if len(s) >= 5:
        b, a0 = np.polyfit(s.n_fa_licks, s.theta_full, 1); xx = np.linspace(s.n_fa_licks.min(), s.n_fa_licks.max(), 20)
        ax[1].plot(xx, a0 + b * xx, color=col, lw=1.6, alpha=0.7)
    if len(s) >= 6:
        raws[subj] = spearmanr(s.theta_full, s.fa_rate_behav)[0]
        pars[subj] = partial_spearman(s.theta_full.values, s.fa_rate_behav.values, s.n_fa_licks.values)[0]
ax[0].axhline(90, color="0.7", ls="--", lw=1); ax[0].set_xlabel("behavioural FA rate (impulsivity) →")
ax[0].set_ylabel("θ_full (deg)"); ax[0].set_title("A. RAW: θ vs impulsivity\n(looks like the hypothesis)"); ax[0].legend(frameon=False, fontsize=7.5)
ax[1].axhline(90, color="0.7", ls="--", lw=1); ax[1].set_xlabel("FA-lick count (β_FA reliability) →")
ax[1].set_ylabel("θ_full (deg)"); ax[1].set_title("B. CONFOUND: θ tracks FA-count\n(attenuation: more FAs → smaller θ)"); ax[1].legend(frameon=False, fontsize=7.5)
# C: raw vs partial ρ per animal
subjs = list(raws.keys()); x = np.arange(len(subjs)); w = 0.36
cols = {s: c for s, _, c in MICE}
ax[2].bar(x - w/2, [raws[s] for s in subjs], w, color=[cols[s] for s in subjs], alpha=0.55, label="raw ρ")
ax[2].bar(x + w/2, [pars[s] for s in subjs], w, color=[cols[s] for s in subjs], alpha=1.0, edgecolor="k", label="partial ρ | FA-count")
ax[2].axhline(0, color="k", lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels(subjs)
ax[2].set_ylabel("Spearman ρ (θ_full vs FA-rate)"); ax[2].set_title("C. Effect COLLAPSES after\ncontrolling FA-count"); ax[2].legend(frameon=False, fontsize=8)
for a in ax:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("θ prototype — the raw θ↓ with impulsivity is the β_FA attenuation artifact, not geometry "
             "(β_TF cached responsive-only, β_FA lumped; 3 core mice)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{OUT}/theta_prototype.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {OUT}/theta_prototype.png + theta_per_session.csv")
print(f"sessions total={len(th_df)}, with support>={MIN_SUPPORT}={len(use)}; "
      f"support dist: {th_df.support.describe()[['min','50%','max']].to_dict()}")
