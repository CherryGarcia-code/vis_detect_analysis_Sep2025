#!/usr/bin/env python3
"""Laser-locked PSTH-SHAPE fingerprint as a tracking identity axis (richer v2).

Unlike laser_fingerprint_test.py (which used the SALT-tuned scalar summaries and so only
spoke for the ~97 salt-significant units), this uses the FULL laser-aligned PSTH SHAPE
(0-20 ms post-pulse, baseline z-scored) for EVERY unit. The user's point: even units that
"fail" the antidromic/SALT criteria often have a reproducible, characteristic firing
pattern locked to the pulse (e.g. a bump at 5-15 ms) — and that shape can fingerprint a
neuron across sessions regardless of tagging.

Reuses existing components (no raw NIDAQ): OptoTagger -> split GPe/SNr pulse trains from
the pkl's ni_events; align_spikes_to_events -> per-pulse aligned rate matrix. Fingerprint
= concatenated baseline-z-scored post-pulse PSTH (GPe + SNr). Similarity = Pearson r of
shapes. Matched = DANT cross-session same-track pairs; random = cross-session different-
track pairs. Join via session_date_key.

Run from the worktree root with the analysis interpreter (loads each session once).
"""
from __future__ import annotations

import gc
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
TRACKS = WT / "FIGURES" / "tracking_dant" / "BG_046" / "curation" / "curated_tracks.csv"
REG = WT / "data" / "cache" / "dant" / "BG_046" / "dant_registry_curation.csv"
OPTO = PRIMARY / "data" / "cache" / "optotagging" / "optotagging_results.csv"
PKL = PRIMARY / "data" / "pkls" / "BG_046"
OUT = WT / "FIGURES" / "tracking_dant" / "BG_046" / "prototype"
RNG = np.random.RandomState(42)

WIN = (-0.050, 0.030)     # s around each pulse
BIN = 0.0005              # 0.5 ms
POST = (0.0005, 0.020)    # post-pulse window used as the fingerprint shape
PRE = (-0.050, -0.003)    # pre-pulse baseline for z-scoring


def _zscore_shape(psth, centers):
    pre = psth[(centers >= PRE[0]) & (centers < PRE[1])]
    mu, sd = float(np.nanmean(pre)), float(np.nanstd(pre))
    if sd < 1e-9:
        sd = 1.0
    z = (psth - mu) / sd
    return z[(centers >= POST[0]) & (centers < POST[1])]


def laser_fp_for_session(S, kids, gpe, snr):
    from visdetect.analysis.align import align_spikes_to_events
    out = {}
    cmap = {c.cluster_id: c for c in S.clusters}
    for k in kids:
        c = cmap.get(int(k))
        if c is None:
            continue
        st = np.asarray(c.spike_times, float).ravel()
        if st.size == 0:
            continue
        parts, structured = [], False
        for pulses in (gpe, snr):
            if pulses is None or len(pulses) == 0:
                parts.append(None); continue
            m, centers = align_spikes_to_events(st, np.asarray(pulses, float),
                                                window=WIN, bin_size=BIN)
            if m is None or len(m) == 0:
                parts.append(None); continue
            psth = np.asarray(m, float).mean(axis=0)
            z = _zscore_shape(psth, np.asarray(centers, float))
            parts.append(z)
            if np.nanmax(np.abs(z)) >= 3.0:
                structured = True
        if all(p is None for p in parts):
            continue
        n = next(len(p) for p in parts if p is not None)
        vec = np.concatenate([p if p is not None else np.full(n, np.nan) for p in parts])
        out[int(k)] = {"fp": vec.astype(np.float32), "structured": structured}
    return out


def _sim(a, b):
    if a is None or b is None:
        return np.nan
    fa, fb = a["fp"], b["fp"]
    m = np.isfinite(fa) & np.isfinite(fb)
    if m.sum() < 8 or np.std(fa[m]) < 1e-9 or np.std(fb[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(fa[m], fb[m])[0, 1])


def main() -> int:
    subj = "BG_046"
    sjp, _, load_session = cd._import_pipeline(subj)
    skey = sjp.session_date_key
    from visdetect.analysis.optotagging import OptoTagger

    # nodes from DANT registry (matched + random pool)
    reg = pd.read_csv(REG, dtype={"session": str})
    reg["uid"] = reg["dant_uid"].astype(int)
    by_sess = {}
    for _, r in reg.iterrows():
        by_sess.setdefault(str(r["session"]), set()).add(int(r["ks_unit_id"]))

    # responsive set (salt<0.01 either fiber) for a subgroup comparison
    opto = pd.read_csv(OPTO)
    opto["skey"] = opto["session_name"].map(skey)
    resp = set()
    for (sk, cid), g in opto.groupby(["skey", "cluster_id"]):
        if g["salt_p"].min() < 0.01:
            resp.add((sk, int(cid)))

    # one pass: per-session laser-PSTH fingerprints, keyed by (skey, kid)
    fp = {}
    for s in sorted(by_sess, key=skey):
        pkl = sjp.session_pkl(subj, s, PKL)
        if pkl is None:
            print(f"  skip {s}: no pkl", flush=True); continue
        S = load_session(str(pkl))
        try:
            tg = OptoTagger(S)
            gpe, snr = tg.gpe_pulses, tg.snr_pulses
        except Exception as e:
            print(f"  {s}: no laser ({e})", flush=True); del S; gc.collect(); continue
        d = laser_fp_for_session(S, by_sess[s], gpe, snr)
        for k, v in d.items():
            fp[(skey(s), k)] = v
        n_str = sum(v["structured"] for v in d.values())
        print(f"  {s}: {len(d)} fps ({n_str} structured)", flush=True)
        del S; gc.collect()

    n_struct = sum(v["structured"] for v in fp.values())
    print(f"laser-PSTH fingerprints: {len(fp)} (session,unit); {n_struct} structured "
          f"(|z|>=3 post-pulse)", flush=True)

    # matched (DANT cross-session same-track) + random (cross-session different-track)
    kept = cd_kept_pairs(pd.read_csv(TRACKS), reg, skey)
    by_uid = {}
    for (uid, stok), ks in kept.items():
        by_uid.setdefault(uid, []).append((skey(stok), int(ks)))
    node_uid = {}
    for uid, nodes in by_uid.items():
        for nd in nodes:
            node_uid.setdefault(nd, set()).add(uid)
    matched = []
    for uid, nodes in by_uid.items():
        for a, b in combinations(sorted(set(nodes)), 2):
            if a[0] != b[0]:
                matched.append((a, b))
    all_nodes = [n for n in fp]
    random_pairs = []
    tries = 0
    while len(random_pairs) < 8000 and tries < 400000:
        tries += 1
        a = all_nodes[RNG.randint(len(all_nodes))]
        b = all_nodes[RNG.randint(len(all_nodes))]
        if a[0] == b[0] or (node_uid.get(a, set()) & node_uid.get(b, set())):
            continue
        random_pairs.append((a, b))

    from visdetect.analysis import track_curation as tc

    def auc(pos, neg, subset=None):
        def sc(pairs):
            o = []
            for a, b in pairs:
                if subset is not None and not (a in subset and b in subset):
                    continue
                s = _sim(fp.get(a), fp.get(b))
                if np.isfinite(s):
                    o.append(s)
            return np.array(o)
        sp, sn = sc(pos), sc(neg)
        if len(sp) < 5 or len(sn) < 5:
            return np.nan, len(sp), len(sn)
        return tc._auc(sp, sn), len(sp), len(sn)

    structured_set = {k for k, v in fp.items() if v["structured"]}
    a_all = auc(matched, random_pairs)
    a_str = auc(matched, random_pairs, subset=structured_set)
    a_resp = auc(matched, random_pairs, subset=resp)
    print(f"laser-PSTH-shape AUC  all={a_all[0]:.3f} (n={a_all[1]}/{a_all[2]}) | "
          f"structured={a_str[0]:.3f} (n={a_str[1]}/{a_str[2]}) | "
          f"salt-responsive={a_resp[0]:.3f} (n={a_resp[1]}/{a_resp[2]})", flush=True)

    # epoch flatness (structured matched pairs)
    all_sk = sorted({n[0] for n in fp})
    rank = {sk: i for i, sk in enumerate(all_sk)}
    n = len(all_sk); e0, e1 = n / 3.0, 2 * n / 3.0
    buck = {"early": [], "mid": [], "late": []}
    for a, b in matched:
        if not (a in structured_set and b in structured_set):
            continue
        s = _sim(fp.get(a), fp.get(b))
        if not np.isfinite(s):
            continue
        earlier = a if rank.get(a[0], 0) <= rank.get(b[0], 0) else b
        r = rank.get(earlier[0], 0)
        buck["early" if r < e0 else ("mid" if r < e1 else "late")].append(s)
    print("epoch (structured matched, n, median r): " + ", ".join(
        f"{k}: n={len(v)}, med={np.median(v):.3f}" if v else f"{k}: n=0" for k, v in buck.items()),
        flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"n_fp": len(fp), "n_structured": n_struct,
           "auc_all": a_all[0], "auc_structured": a_str[0], "auc_responsive": a_resp[0],
           "n_matched_all": a_all[1], "n_matched_struct": a_str[1]}
    pd.DataFrame([res]).to_csv(OUT / "laser_psth_metrics.csv", index=False)
    _plot(a_all, a_str, a_resp, buck, OUT / "laser_psth_fingerprint.png")
    print(f"wrote {OUT}", flush=True)
    return 0


def cd_kept_pairs(tracks, reg, skey):
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


def _plot(a_all, a_str, a_resp, buck, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = [f"all\n(n={a_all[1]})", f"structured\n(n={a_str[1]})", f"salt-resp\n(n={a_resp[1]})"]
    ax[0].bar(labels, [a_all[0], a_str[0], a_resp[0]], color=["#9e9e9e", "#2ca25f", "#3474ae"])
    ax[0].axhline(0.5, ls=":", c="k", lw=.8); ax[0].set_ylim(0.4, 1.0)
    ax[0].set_title("Laser-PSTH-shape fingerprint\nmatched vs random AUC"); ax[0].set_ylabel("AUC")
    order = ["early", "mid", "late"]
    meds = [np.median(buck[e]) if buck[e] else np.nan for e in order]
    ax[1].plot(range(3), meds, "-o", color="#2ca25f")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"{e}\n(n={len(buck[e])})" for e in order])
    ax[1].axhline(0.0, ls=":", c="k", lw=.8)
    ax[1].set_title("Epoch flatness (learning-invariant if FLAT)")
    ax[1].set_ylabel("matched laser-PSTH-shape r")
    fig.suptitle("Laser-locked PSTH-shape fingerprint for tracking (BG_046) — all units")
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
