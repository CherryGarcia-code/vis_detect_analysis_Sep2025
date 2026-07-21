"""Within-session behavioural-state DYNAMICS — a level-agnostic regulation readout.

Occupancy (fraction of the session in each state) can be flat while REGULATION improves as
longer engaged bouts + faster recovery from impulsive lapses. From the per-trial state sequence
(state_tags: StimSens / Impulsive / Disengaged; Abort dropped) we compute, per session:
  - dwell = mean run length of consecutive same-state trials (StimSens, Impulsive)
  - switch_rate = fraction of adjacent trials that change state (volatility)
  - recovery = P(next = StimSens | current = Impulsive)  — bounce-back to engaged
  - persistence = P(next = StimSens | current = StimSens)
Then per animal: do these track performance (d')? And does StimSens dwell improve with d' BEYOND
occupancy (partial | f_stimsens)? Behavioural-only (state_tags + manifest d'); reads PRIMARY read-only.
CAVEAT: states are behavioural + partly self-referential (state_labeler_circularity_caveat) — this
is a descriptive regulation-dynamics readout, not a neural claim.
"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WT, "src"))
from visdetect.analysis import config

PRIMARY = "e:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
D = f"{PRIMARY}/data"
OUT = f"{WT}/FIGURES/state_dynamics"
os.makedirs(OUT, exist_ok=True)
SUBJ = [("BG_046", "DMS", "#2c7fb8"), ("BG_039", "DMS", "#31a354"), ("BG_031", "VMS", "#d6322a"),
        ("BG_038", "cortex", "#7f7f7f"), ("BG_040", "TBD", "#9467bd"), ("BG_049", "TBD", "#17becf")]
STATES = ["StimSens", "Impulsive", "Disengaged"]
MIN_TRIALS = 50


def cid(x):
    try:
        return config.canonical_session_id(str(x))
    except Exception:
        return str(x)


def dynamics(seq):
    s = np.array([x for x in seq if x in STATES])   # drop Abort, collapse
    n = len(s)
    if n < MIN_TRIALS:
        return None
    # run lengths per state
    runs = {st: [] for st in STATES}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        runs[s[i]].append(j - i + 1)
        i = j + 1
    occ = {st: (s == st).mean() for st in STATES}
    dwell = {st: (float(np.mean(runs[st])) if len(runs[st]) >= 3 else np.nan) for st in STATES}
    switch = float(np.mean(s[:-1] != s[1:]))
    imp = s[:-1] == "Impulsive"; stim = s[:-1] == "StimSens"; nxt = s[1:]
    recovery = float((nxt[imp] == "StimSens").mean()) if imp.sum() >= 10 else np.nan
    persist = float((nxt[stim] == "StimSens").mean()) if stim.sum() >= 10 else np.nan
    return dict(n_trials=n, f_stimsens=occ["StimSens"], f_impulsive=occ["Impulsive"],
                dwell_stimsens=dwell["StimSens"], dwell_impulsive=dwell["Impulsive"],
                switch_rate=switch, recovery=recovery, persistence=persist)


def partial_spearman(x, y, z):
    d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(d) < 6:
        return np.nan
    rx, ry, rz = d.x.rank(), d.y.rank(), d.z.rank()
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(ex, ey)[0]


rows = []
for subj, region, col in SUBJ:
    manp = f"{D}/{subj}_staging_manifest.csv"
    if not os.path.exists(manp):
        continue
    man = pd.read_csv(manp, dtype={"session_name": str}); man["sid"] = man.session_name.map(cid)
    dpr = man.set_index("sid")["d_prime"].to_dict()
    for f in glob.glob(f"{D}/cache/state_tags/{subj}/*.csv"):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem.startswith("_"):
            continue
        d = pd.read_csv(f)
        if "state_label" not in d.columns:
            continue
        seq = d.sort_values("trial_idx")["state_label"].astype(str).tolist() if "trial_idx" in d.columns else d["state_label"].astype(str).tolist()
        m = dynamics(seq)
        if m is None:
            continue
        m.update(subject=subj, region=region, sid=cid(stem), d_prime=dpr.get(cid(stem), np.nan))
        rows.append(m)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/within_session_dynamics.csv", index=False)

print("=== within-session regulation dynamics vs performance (d'), per animal ===")
print("    (Spearman; and StimSens-dwell vs d' PARTIALLING occupancy — regulation beyond occupancy)")
for subj, region, _ in SUBJ:
    s = df[(df.subject == subj)].dropna(subset=["d_prime"])
    if len(s) < 6:
        print(f"  {subj} ({region}): n={len(s)} (<6, skip)"); continue
    def sp(col):
        t = s.dropna(subset=[col]);  return spearmanr(t[col], t.d_prime) if len(t) >= 6 else (np.nan, np.nan)
    rd, pd_ = sp("dwell_stimsens"); rr, pr = sp("recovery"); rs, ps = sp("switch_rate")
    par = partial_spearman(s.dwell_stimsens.values, s.d_prime.values, s.f_stimsens.values)
    par_r = partial_spearman(s.recovery.values, s.d_prime.values, s.f_stimsens.values)
    ro, po = sp("f_stimsens")
    print(f"  {subj} ({region}, n={len(s)}): occupancy~d' ρ={ro:+.2f} || StimSens-dwell~d' ρ={rd:+.2f}(p{pd_:.2f}) "
          f"[|occ ρ={par:+.2f}] | recovery~d' ρ={rr:+.2f}(p{pr:.2f}) [|occ ρ={par_r:+.2f}] | switch~d' ρ={rs:+.2f}(p{ps:.2f})")

# figure
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
for subj, region, col in SUBJ:
    s = df[df.subject == subj].dropna(subset=["d_prime"])
    ax[0].scatter(s.d_prime, s.dwell_stimsens, s=34, color=col, alpha=0.75, label=f"{subj} ({region})")
    ax[1].scatter(s.d_prime, s.recovery, s=34, color=col, alpha=0.75)
    for a_, yc in ((ax[0], "dwell_stimsens"), (ax[1], "recovery")):
        t = s.dropna(subset=[yc])
        if len(t) >= 6:
            b, a0 = np.polyfit(t.d_prime, t[yc], 1); xx = np.linspace(t.d_prime.min(), t.d_prime.max(), 20)
            a_.plot(xx, a0 + b * xx, color=col, lw=1.5, alpha=0.7)
ax[0].set_xlabel("d' (performance) →"); ax[0].set_ylabel("StimSens dwell (mean bout length, trials)")
ax[0].set_title("A. Do engaged bouts lengthen with performance?"); ax[0].legend(frameon=False, fontsize=7.5)
ax[1].set_xlabel("d' (performance) →"); ax[1].set_ylabel("recovery  P(next=StimSens | Impulsive)")
ax[1].set_title("B. Faster bounce-back from impulsive lapses?")
# panel C: per-animal ρ (dwell~d', dwell~d'|occupancy, recovery~d')
subs = [s for s, _, _ in SUBJ if (df.subject == s).sum() and df[df.subject == s].dropna(subset=["d_prime"]).shape[0] >= 6]
x = np.arange(len(subs)); w = 0.2; cols = {s: c for s, _, c in SUBJ}
def rho(subj, col, partial=False):
    s = df[df.subject == subj].dropna(subset=["d_prime", col])
    if len(s) < 6:
        return np.nan
    return partial_spearman(s[col].values, s.d_prime.values, s.f_stimsens.values) if partial else spearmanr(s[col], s.d_prime)[0]
ax[2].bar(x - 1.5 * w, [rho(s, "dwell_stimsens") for s in subs], w, color=[cols[s] for s in subs], alpha=0.45, label="dwell~d' (raw)")
ax[2].bar(x - 0.5 * w, [rho(s, "dwell_stimsens", True) for s in subs], w, color=[cols[s] for s in subs], edgecolor="k", label="dwell~d' | occupancy")
ax[2].bar(x + 0.5 * w, [rho(s, "recovery") for s in subs], w, color=[cols[s] for s in subs], alpha=0.45, hatch="//", label="recovery~d' (raw)")
ax[2].bar(x + 1.5 * w, [rho(s, "recovery", True) for s in subs], w, color=[cols[s] for s in subs], edgecolor="k", hatch="//", label="recovery~d' | occupancy")
ax[2].axhline(0, color="k", lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels(subs, rotation=30, fontsize=7.5)
ax[2].set_ylabel("Spearman ρ vs d'"); ax[2].set_title("C. Regulation vs performance:\nraw (occupancy) vs |occupancy (dynamics)"); ax[2].legend(frameon=False, fontsize=7)
for a in ax:
    for sp_ in ("top", "right"):
        a.spines[sp_].set_visible(False)
fig.suptitle("Within-session behavioural-state dynamics — regulation (dwell/recovery) vs performance, per animal", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{OUT}/within_session_dynamics.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {OUT}/within_session_dynamics.png + within_session_dynamics.csv | n sessions={len(df)}")
