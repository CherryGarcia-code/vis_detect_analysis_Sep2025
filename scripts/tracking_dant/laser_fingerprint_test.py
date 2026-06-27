#!/usr/bin/env python3
"""Test the optotag LASER-RESPONSE fingerprint as a learning-invariant identity axis.

Hypothesis (user's idea): a unit's response to the optotagging laser — even a purely
SYNAPTIC excitation, not an antidromic spike — reflects its circuit position, which is a
STRUCTURAL property that should be stable day-to-day, UNLIKE the task PSTH (which changes
with learning). So the laser fingerprint should (a) be more similar for matched cross-
session pairs than random within-session pairs, and (b) be FLAT across learning epochs
(no decline into early learning, the distinguishing prediction vs the task PSTH).

Reuses the EXISTING optotagging caches (no recompute, no raw NIDAQ):
  <PRIMARY>/analysis_suite/cache/optotagging_results.csv   (per session,cluster,fiber)
  <PRIMARY>/analysis_suite/cache/optotagging_unit_tags.csv (per session,cluster: tier/pathway)
Matched pairs come from the DANT curated tracks; random pairs are within-session different
units. Join on session_date_key (session_name is an int -> leading zeros stripped).

Run from the worktree root with the analysis interpreter.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import curate_dant as cd            # noqa: E402

WT = cd.WORKTREE_ROOT
PRIMARY = cd.PRIMARY_DEFAULT
OPTO = PRIMARY / "analysis_suite" / "cache" / "optotagging_results.csv"
TAGS = PRIMARY / "analysis_suite" / "cache" / "optotagging_unit_tags.csv"
TRACKS = WT / "FIGURES" / "tracking_dant" / "BG_046" / "curation" / "curated_tracks.csv"
REG = WT / "data" / "cache" / "dant" / "BG_046" / "dant_registry_curation.csv"
OUT = WT / "FIGURES" / "tracking_dant" / "BG_046" / "prototype"
RNG = np.random.RandomState(42)

# Fingerprint features per fiber (drop the vacuous excess_jitter_ms; -log10 the p-values).
FEATS = ["excess_reliability", "peak_latency_ms", "neglog_salt_p", "collision_suppression_index"]


def build_fingerprints(skey):
    """{(session_key, cluster_id) -> z-scored fingerprint vector (len 2*len(FEATS))}.

    Pivots GPe + SNr rows into one vector; z-scores each feature across the population;
    NaN preserved (missing collision / latency). Also returns a 'responsive' set
    (salt-significant in either fiber) and the per-unit raw table for diagnostics.
    """
    df = pd.read_csv(OPTO)
    df["skey"] = df["session_name"].map(skey)
    df["neglog_salt_p"] = -np.log10(df["salt_p"].clip(lower=1e-300))
    # pivot per (skey, cluster_id) x fiber
    cols = {}
    resp = set()
    for (sk, cid), g in df.groupby(["skey", "cluster_id"]):
        vec = []
        salt_min = 1.0
        for fiber in ["GPe", "SNr"]:
            r = g[g["fiber"] == fiber]
            if len(r):
                r = r.iloc[0]
                vec += [float(r[f]) if pd.notna(r[f]) else np.nan for f in FEATS]
                if pd.notna(r["salt_p"]):
                    salt_min = min(salt_min, float(r["salt_p"]))
            else:
                vec += [np.nan] * len(FEATS)
        cols[(sk, int(cid))] = np.array(vec, float)
        if salt_min < 0.01:
            resp.add((sk, int(cid)))
    # z-score each feature dimension across the population
    M = np.vstack(list(cols.values()))
    mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0); sd[sd < 1e-9] = 1.0
    fp = {k: (v - mu) / sd for k, v in cols.items()}
    return fp, resp


def fp_sim(a, b):
    """Similarity of two z-scored fingerprints: -mean |za-zb| over shared finite dims."""
    if a is None or b is None:
        return np.nan
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    return float(-np.mean(np.abs(a[m] - b[m])))


def main() -> int:
    subj = "BG_046"
    sjp, _, _ = cd._import_pipeline(subj)
    skey = sjp.session_date_key

    fp, responsive = build_fingerprints(skey)
    n_units = len(fp)
    print(f"laser fingerprints: {n_units} (session,unit); responsive (salt<0.01) = {len(responsive)}",
          flush=True)

    # Matched pairs = DANT curated tracks' cross-session same-uid pairs (keyed via registry).
    tracks = pd.read_csv(TRACKS)
    reg = pd.read_csv(REG, dtype={"session": str})
    kept = cd_kept_pairs(tracks, reg, skey)  # {(uid, session_token) -> ks_id}, normalized join
    # group kept nodes by uid -> cross-session pairs
    by_uid = {}
    for (uid, stok), ks in kept.items():
        by_uid.setdefault(uid, []).append((skey(stok), int(ks)))
    matched = []
    for uid, nodes in by_uid.items():
        for a, b in combinations(sorted(set(nodes)), 2):
            if a[0] != b[0]:
                matched.append((a, b))
    # Random within-session different-unit pairs from the opto units.
    units_by_sess = {}
    for (sk, cid) in fp:
        units_by_sess.setdefault(sk, []).append(cid)
    sess_list = [s for s in units_by_sess if len(units_by_sess[s]) > 3]
    rnd = []
    for _ in range(min(8000, 40 * len(sess_list))):
        sk = sess_list[RNG.randint(len(sess_list))]
        c1, c2 = RNG.choice(units_by_sess[sk], 2, replace=False)
        rnd.append(((sk, int(c1)), (sk, int(c2))))

    def auc(pairs_pos, pairs_neg, restrict_resp=False):
        def score(pairs):
            out = []
            for a, b in pairs:
                if restrict_resp and not (a in responsive and b in responsive):
                    continue
                s = fp_sim(fp.get(a), fp.get(b))
                if np.isfinite(s):
                    out.append(s)
            return np.array(out)
        sp, sn = score(pairs_pos), score(pairs_neg)
        if len(sp) < 5 or len(sn) < 5:
            return np.nan, len(sp), len(sn)
        from visdetect.analysis import track_curation as tc
        return tc._auc(sp, sn), len(sp), len(sn)

    auc_all, np_all, nn_all = auc(matched, rnd, restrict_resp=False)
    # Responsive units are too sparse (~2-3/session) for a within-session random baseline,
    # so use a CROSS-session, DIFFERENT-track responsive baseline (fair: both matched and
    # random are cross-session, isolating identity from the time/drift confound).
    node_uid = {}
    for uid, nodes in by_uid.items():
        for nd in nodes:
            node_uid.setdefault(nd, set()).add(uid)
    resp_nodes = [n for n in responsive if n in fp]
    rnd_resp = []
    tries = 0
    while len(rnd_resp) < 4000 and tries < 200000 and len(resp_nodes) > 3:
        tries += 1
        a = resp_nodes[RNG.randint(len(resp_nodes))]
        b = resp_nodes[RNG.randint(len(resp_nodes))]
        if a[0] == b[0]:
            continue                                   # require cross-session
        if node_uid.get(a, set()) & node_uid.get(b, set()):
            continue                                   # require different track
        rnd_resp.append((a, b))
    auc_resp, np_r, nn_r = auc(matched, rnd_resp, restrict_resp=True)
    print(f"matched-vs-random laser-fingerprint AUC: all={auc_all:.3f} (n={np_all}/{nn_all}); "
          f"responsive-only(x-session baseline)={auc_resp if not np.isnan(auc_resp) else float('nan'):.3f} "
          f"(n={np_r}/{nn_r})", flush=True)

    # Per-feature single-axis AUCs (magnitude, latency) on responsive pairs — interpretable.
    feat_auc = {}
    from visdetect.analysis import track_curation as tc
    for fi, name in enumerate(["GPe_" + f for f in FEATS] + ["SNr_" + f for f in FEATS]):
        def one(pairs):
            o = []
            for a, b in pairs:
                if not (a in responsive and b in responsive):
                    continue
                va, vb = fp.get(a), fp.get(b)
                if va is None or vb is None or not (np.isfinite(va[fi]) and np.isfinite(vb[fi])):
                    continue
                o.append(-abs(va[fi] - vb[fi]))
            return np.array(o)
        sp, sn = one(matched), one(rnd)
        if len(sp) >= 5 and len(sn) >= 5:
            feat_auc[name] = (tc._auc(sp, sn), len(sp))

    # Epoch flatness: matched-pair similarity (responsive) by earlier session epoch.
    all_sk = sorted({sk for sk in units_by_sess})
    rank = {sk: i for i, sk in enumerate(all_sk)}
    n = len(all_sk); e0, e1 = n / 3.0, 2 * n / 3.0
    epoch_buckets = {"early": [], "mid": [], "late": []}
    for a, b in matched:
        if not (a in responsive and b in responsive):
            continue
        s = fp_sim(fp.get(a), fp.get(b))
        if not np.isfinite(s):
            continue
        earlier = a if rank.get(a[0], 0) <= rank.get(b[0], 0) else b
        r = rank.get(earlier[0], 0)
        ep = "early" if r < e0 else ("mid" if r < e1 else "late")
        epoch_buckets[ep].append(s)
    print("epoch (responsive matched, n, median sim): " + ", ".join(
        f"{k}: n={len(v)}, med={np.median(v):.3f}" if v else f"{k}: n=0" for k, v in epoch_buckets.items()),
        flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"n_units": n_units, "n_responsive": len(responsive),
           "auc_all": auc_all, "auc_responsive": auc_resp,
           "n_matched_resp": np_r, "n_random_resp": nn_r,
           **{f"featAUC_{k}": v[0] for k, v in feat_auc.items()}}
    pd.DataFrame([res]).to_csv(OUT / "laser_fingerprint_metrics.csv", index=False)
    _plot(auc_all, auc_resp, feat_auc, epoch_buckets, OUT / "laser_fingerprint.png")
    _write_findings(res, feat_auc, epoch_buckets, OUT / "laser_fingerprint_findings.md")
    print(f"wrote {OUT}", flush=True)
    return 0


def cd_kept_pairs(tracks, reg, skey):
    """(uid, kept_sessions_token) -> ks_unit_id, normalized join (handles leading zeros)."""
    reg = reg.copy(); reg["uid"] = reg["dant_uid"].astype(int)
    lut = {(int(u), skey(k)): int(ks) for u, k, ks in zip(reg["uid"], reg["session"], reg["ks_unit_id"])}
    out = {}
    for _, row in tracks.iterrows():
        uid = int(row["curated_uid"])
        for s in [s for s in str(row["kept_sessions"]).split(";") if s]:
            ks = lut.get((uid, skey(s)))
            if ks is not None:
                out[(uid, s)] = ks
    return out


def _plot(auc_all, auc_resp, feat_auc, epoch_buckets, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].bar(["all units", "responsive\n(salt<0.01)"], [auc_all, auc_resp],
              color=["#9e9e9e", "#2ca25f"])
    ax[0].axhline(0.5, ls=":", c="k", lw=.8); ax[0].set_ylim(0.4, 1.0)
    ax[0].set_title("Laser fingerprint:\nmatched vs random AUC"); ax[0].set_ylabel("AUC")
    if feat_auc:
        names = list(feat_auc); vals = [feat_auc[k][0] for k in names]
        ax[1].barh(range(len(names)), vals, color="#3474ae")
        ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=7)
        ax[1].axvline(0.5, ls=":", c="k", lw=.8); ax[1].set_xlim(0.4, 1.0)
        ax[1].set_title("Per-feature AUC (responsive)")
    order = ["early", "mid", "late"]
    meds = [np.median(epoch_buckets[e]) if epoch_buckets[e] else np.nan for e in order]
    ax[2].plot(range(3), meds, "-o", color="#2ca25f")
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels([f"{e}\n(n={len(epoch_buckets[e])})" for e in order])
    ax[2].set_title("Epoch flatness (the key test:\nlearning-invariant if FLAT)")
    ax[2].set_ylabel("matched laser-fp similarity")
    fig.suptitle("Optotag laser-response fingerprint as a tracking identity axis (BG_046)")
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)


def _write_findings(res, feat_auc, epoch_buckets, path):
    lines = ["# Optotag laser-response fingerprint — tracking identity axis (BG_046)\n",
             "Reuses the existing optotagging caches (no recompute). Fingerprint = z-scored",
             "[excess_reliability, peak_latency_ms, -log10 salt_p, collision_suppression_index]",
             "for GPe + SNr fibers. Matched = DANT cross-session same-track pairs; random =",
             "within-session different units. Similarity = -mean|z-diff| over shared dims.\n",
             "## Numbers"]
    for k, v in res.items():
        lines.append(f"- {k}: {round(v, 4) if isinstance(v, float) else v}")
    em = {e: (round(float(np.median(epoch_buckets[e])), 3) if epoch_buckets[e] else None)
          for e in ("early", "mid", "late")}
    lines += [f"- epoch median sim (responsive matched): {em}",
              "\n## Reading it",
              "- auc_responsive > 0.5: the laser fingerprint carries cross-session identity info",
              "  for laser-responsive units.",
              "- epoch FLAT (early ~= late) supports learning-INVARIANCE — the key advantage over",
              "  the task PSTH (which declined into early learning). A decline would undercut it.",
              "\n## Honest limitations",
              "- Optotag yield is low (162 candidates; ~30% collision-untestable), so the responsive",
              "  set is small -> watch n and CIs.",
              "- Fingerprint is a coarse scalar summary (the cache has no laser PSTH time-course);",
              "  a richer shape fingerprint would need to regenerate laser-aligned PSTHs (fast, reuses",
              "  OptoTagger) — do that if the scalar signal is promising.",
              "- Excitation-only (no inhibited-response sign in the pipeline).",
              "- Non-responsive units have near-null fingerprints (all alike) -> 'all units' AUC is",
              "  partly responsiveness-matching, not identity; the responsive-only AUC is the real test."]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
