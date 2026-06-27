"""Evaluate DANT tracking on BG_046: yield + survival vs UnitMatch, co-membership agreement,
and a held-out ISI-fingerprint AUC. Saves presentation-ready figures + a summary JSON.

⚠️ CAVEAT (pre-merge audit, Jun 2026): the DANT-vs-UM MEAN-TRACKED-LENGTH / survival
comparison is biased and OVERSTATES DANT's length advantage. DANT only emits multi-session
(>=2) clusters, whereas melt_cellregistry assigns a um_uid to EVERY unit, so ~80% of UM
UIDs are length-1 singletons that drag UM's mean down. The "~2x longer" framing is an
apples-to-oranges artifact of that singleton asymmetry; the matched-pool fix does not fully
cure it. The TRACKER-QUALITY conclusion does NOT rely on this — it rests on the held-out
ISI-fingerprint AUC (DANT ~0.94 vs UM ~0.96, comparable), which is unaffected. Read the
length/survival panels as descriptive, not as a head-to-head win.

Run with ANALYSIS_PY from the worktree root.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry  # noqa: E402

PRIMARY = "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025"
DEFAULT_UM = os.path.join(PRIMARY, "data", "unit_match", "output", "BG_046_um329_CellRegistry.csv")
FIGDIR = "FIGURES/tracking_dant/BG_046"


def _isi_hist(spike_ms, window=100, binwidth=1, sigma=1):
    isi = np.diff(np.sort(spike_ms))
    h = np.histogram(isi, bins=np.arange(0, window + binwidth, binwidth))[0].astype(float)
    s = h.sum()
    if s > 0:
        h /= s
    return gaussian_filter1d(h, sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dant-registry", default="data/cache/dant/BG_046/dant_registry.csv")
    ap.add_argument("--input-dir", default="data/cache/dant/BG_046/input")
    ap.add_argument("--um-registry", default=DEFAULT_UM)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(FIGDIR, exist_ok=True)

    dant = pd.read_csv(args.dant_registry, dtype={"session": str})
    dant["session"] = dant["session"].str.zfill(8)
    lookup = pd.read_csv(os.path.join(args.input_dir, "unit_lookup.csv"), dtype={"session": str})
    lookup["session"] = lookup["session"].str.zfill(8)
    n_sessions = int(lookup["session_index"].max())

    summary = {"n_units": int(len(dant)),
               "dant_n_clusters": int(dant.loc[dant["dant_uid"] > 0, "dant_uid"].nunique()),
               "dant_n_tracked_units": int((dant["dant_uid"] > 0).sum())}

    # --- DANT tracked-length survival ---
    dant_len = registry.tracked_lengths(dant)
    ks, dant_surv = registry.survival_function(dant_len, n_sessions)
    summary["dant_mean_tracked_len"] = float(dant_len.mean()) if len(dant_len) else 0.0

    # --- UnitMatch comparison (best-effort; skip cleanly if registry absent) ---
    um_surv = None
    if os.path.exists(args.um_registry):
        um_wide = pd.read_csv(args.um_registry)
        uid_col = "UID" if "UID" in um_wide.columns else um_wide.columns[0]
        um_long = registry.melt_cellregistry(um_wide, uid_col=uid_col)
        um_long["session"] = um_long["session"].astype(str).str.zfill(8)
        um_len = registry.tracked_lengths(um_long, uid_col="um_uid")
        _, um_surv = registry.survival_function(um_len, n_sessions)
        summary["um_mean_tracked_len"] = float(um_len.mean()) if len(um_len) else 0.0
        summary["um_n_tracked_units"] = int((um_long["um_uid"] > 0).sum())
        agree = registry.comembership_agreement(dant, um_long, "dant_uid", "um_uid")
        summary["comembership_vs_unitmatch"] = agree

        # --- MATCHED tracked-length on the shared (session, ks_unit_id) unit set ---
        # The own-pool means above are unfair: DANT's pool excludes ~1226 positive-going
        # units and session 13082025, while UM's registry includes them. Restrict BOTH
        # trackers to the units present in BOTH registries (dedup um_long on the key first),
        # then count distinct sessions among only those shared members -- symmetric & fair.
        dant_keys = dant.drop_duplicates(["session", "ks_unit_id"])
        um_keys = um_long.drop_duplicates(["session", "ks_unit_id"])
        a_idx = dant_keys.set_index(["session", "ks_unit_id"]).index
        b_idx = um_keys.set_index(["session", "ks_unit_id"]).index
        shared = a_idx.intersection(b_idx)
        dant_shared = dant_keys.set_index(["session", "ks_unit_id"]).loc[shared].reset_index()
        um_shared = um_keys.set_index(["session", "ks_unit_id"]).loc[shared].reset_index()
        dant_len_matched = registry.tracked_lengths(dant_shared, "dant_uid")
        um_len_matched = registry.tracked_lengths(um_shared, "um_uid")
        summary["n_shared_units"] = int(len(shared))
        summary["dant_mean_tracked_len_matched"] = (
            float(dant_len_matched.mean()) if len(dant_len_matched) else 0.0)
        summary["um_mean_tracked_len_matched"] = (
            float(um_len_matched.mean()) if len(um_len_matched) else 0.0)
        summary["matched_note"] = (
            "tracked length counted over the shared (session,ks_unit_id) unit set only")
    else:
        summary["um_note"] = f"UnitMatch registry not found at {args.um_registry}; comparison skipped."

    # --- Survival comparison figure ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ks, dant_surv, "-o", ms=3,
            label=f"DANT (own pool, mean {summary['dant_mean_tracked_len']:.1f})")
    if um_surv is not None:
        ax.plot(ks, um_surv, "-s", ms=3,
                label=f"UnitMatch (own pool, mean {summary.get('um_mean_tracked_len', float('nan')):.1f})")
        # Matched comparison on the shared unit set (fair, symmetric)
        txt = (f"Matched (shared {summary['n_shared_units']} units):\n"
               f"DANT {summary['dant_mean_tracked_len_matched']:.2f}  vs  "
               f"UM {summary['um_mean_tracked_len_matched']:.2f}")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_xlabel("Tracked length (# sessions)")
    ax.set_ylabel("Fraction of tracked neurons ≥ k sessions")
    ax.set_title("BG_046 cross-session tracking: survival")
    ax.legend(); ax.set_ylim(0, 1); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "survival_comparison.png"), dpi=200)
    plt.close(fig)

    # --- Held-out ISI-fingerprint AUC ---
    def isi_for(pooled_index):
        st = np.load(os.path.join(args.input_dir, "spike_times", f"Unit{int(pooled_index)}.npy"))
        return _isi_hist(st)

    key_to_pooled = {(r.session, int(r.ks_unit_id)): int(r.pooled_index) for r in lookup.itertuples()}
    tracked = dant[dant["dant_uid"] > 0].copy()
    matched_sims, nonmatched_sims = [], []
    # matched: cross-session pairs within the same dant_uid
    for uid, grp in tracked.groupby("dant_uid"):
        members = [(row.session, int(row.ks_unit_id)) for row in grp.itertuples()]
        if len(members) < 2:
            continue
        hists = {m: isi_for(key_to_pooled[m]) for m in members if m in key_to_pooled}
        ms = list(hists)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if ms[i][0] != ms[j][0]:  # different session
                    r = np.corrcoef(hists[ms[i]], hists[ms[j]])[0, 1]
                    if np.isfinite(r):
                        matched_sims.append(r)
    # non-matched: within-session pairs of different units (random sample, balanced)
    by_session = tracked.groupby("session")
    target = len(matched_sims)
    attempts = 0
    while len(nonmatched_sims) < target and attempts < target * 50:
        attempts += 1
        sess = rng.choice(tracked["session"].unique())
        g = by_session.get_group(sess)
        if len(g) < 2:
            continue
        rows = list(g.sample(2, random_state=int(rng.integers(1 << 30))).itertuples(index=False))
        ka = (rows[0].session, int(rows[0].ks_unit_id))
        kb = (rows[1].session, int(rows[1].ks_unit_id))
        if ka not in key_to_pooled or kb not in key_to_pooled:
            continue
        r = np.corrcoef(isi_for(key_to_pooled[ka]), isi_for(key_to_pooled[kb]))[0, 1]
        if np.isfinite(r):
            nonmatched_sims.append(r)

    if len(nonmatched_sims) < target:
        print(f"[warn] non-matched ISI sampling under-filled: {len(nonmatched_sims)} "
              f"of target {target} pairs (sparse subject / few within-session unit pairs).")

    if matched_sims and nonmatched_sims:
        y = np.r_[np.ones(len(matched_sims)), np.zeros(len(nonmatched_sims))]
        score = np.r_[matched_sims, nonmatched_sims]
        auc = float(roc_auc_score(y, score))
        summary["heldout_isi_auc"] = auc
        summary["n_matched_pairs"] = len(matched_sims)
        summary["n_nonmatched_pairs"] = len(nonmatched_sims)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bins = np.linspace(-1, 1, 41)
        ax.hist(nonmatched_sims, bins=bins, density=True, alpha=0.6, label="within-session, different unit")
        ax.hist(matched_sims, bins=bins, density=True, alpha=0.6, label="cross-session, same DANT id")
        ax.set_xlabel("ISI-histogram correlation"); ax.set_ylabel("density")
        ax.set_title(f"Held-out ISI fingerprint (AUC = {auc:.3f})")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "isi_auc.png"), dpi=200)
        plt.close(fig)
    else:
        summary["heldout_isi_auc"] = None

    with open(os.path.join(FIGDIR, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
