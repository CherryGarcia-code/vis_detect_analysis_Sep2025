#!/usr/bin/env python3
"""Inclusive-trusted re-tiering of the DANT curation + dual validation.

The shipped trusted rule (compute_tier) demands span>=3 AND zero warn-flagged kept
links AND zero bridges -> any single warn/bridge demotes an otherwise-long clean
track to review, pinning trusted near the span-3 floor. This script re-tiers the
EXISTING curated tracks under a looser "inclusive-trusted" rule (span>=3 AND
<=max_warn warn-flagged kept links; bridges allowed) WITHOUT re-running the sweep,
then validates every tier on two axes (NOTE their independence differs):

  * held-out ISI AUC  (identity fingerprint, even/odd partition split). This is the
    PRIMARY precision axis. Caveat: even/odd partitions are autocorrelated estimates of
    one stationary ISI distribution, and the tier itself is partly ISI-gated (the warn
    flag uses badge_isi_hist_corr), so it is QUASI-independent, not fully independent.
  * functional-PSTH AUC (matched cross-session vs random within-session PSTH
    similarity) -- CORROBORATIVE BUT ENTANGLED, not independent: the warn flags that
    define the tier are set partly by badge_func_resp (cross-session in-zone PSTH
    similarity), so this axis partly re-confirms the rule it validates. (Mitigation:
    here we pool ALL trials whereas the corroborator used in-zone-restricted trials, so
    it's partial not trivial self-prediction.) Treat held-out ISI as primary.

Reads only the canonical curation outputs + the session pkls (loaded once); writes
to FIGURES/tracking_dant/BG_046/curation/. Touches no shared pipeline code.

Run from the worktree root with the analysis interpreter:
    <PRIMARY>/.venv/Scripts/python.exe scripts/tracking_dant/inclusive_trusted.py
"""
from __future__ import annotations

import gc
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))            # sibling: curate_dant
import curate_dant as cd                        # noqa: E402

WT = cd.WORKTREE_ROOT
PRIMARY = cd.PRIMARY_DEFAULT
CUR = WT / "FIGURES" / "tracking_dant" / "BG_046" / "curation"
REG = WT / "data" / "cache" / "dant" / "BG_046" / "dant_registry_curation.csv"
PKL = PRIMARY / "data" / "pkls" / "BG_046"


# ── Pure re-tiering ──────────────────────────────────────────────────────────
def compute_inclusive_tier(span: int, n_warn_keep: int,
                           min_span: int = 3, max_warn: int = 1) -> str:
    """Looser tier: trusted if span>=min_span AND <=max_warn warn kept-links.

    Bridges are allowed (unlike the shipped rule). STOP is not checked here: a
    STOP only truncates which sessions are kept, not the quality of the kept run,
    and only 6 tracks contain one.
    """
    if span < 2:
        return "suspect"
    if span >= min_span and n_warn_keep <= max_warn:
        return "trusted"
    return "review"


def _as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda x: str(x).strip().lower() in ("true", "1"))


def assign_inclusive_tiers(tracks_df: pd.DataFrame, links_df: pd.DataFrame,
                           min_span: int = 3, max_warn: int = 1) -> pd.Series:
    """inclusive_tier per curated_uid, from kept-link warn counts."""
    lf = links_df.copy()
    lf["review_flag"] = _as_bool(lf["review_flag"])
    keep = lf[lf["link_decision"] == "KEEP"]
    warn_by_uid = (keep[keep["review_flag"]].groupby("liberal_uid").size().to_dict())
    tiers = []
    for _, row in tracks_df.iterrows():
        uid = int(row["curated_uid"])
        span = len([s for s in str(row["kept_sessions"]).split(";") if s])
        n_warn = int(warn_by_uid.get(uid, 0))
        tiers.append(compute_inclusive_tier(span, n_warn, min_span, max_warn))
    return pd.Series(tiers, index=tracks_df.index)


# ── Generic matched-vs-nonmatched AUC (mirrors tc.held_out_isi_auc_by_tier) ──
def auc_by_tier(tracks_df: pd.DataFrame, fp_by_key: Dict, sim_fn,
                tier_col: str = "confidence_tier") -> Dict[str, dict]:
    """Per-tier AUC of matched (cross-session, same uid) vs non-matched
    (within-session, different uid) fingerprint similarities."""
    from visdetect.analysis import track_curation as tc
    import numpy as np
    out: Dict[str, dict] = {}
    for tier, grp in tracks_df.groupby(tier_col):
        sess_by_uid: Dict[int, list] = {}
        matched = []
        for _, row in grp.iterrows():
            uid = int(row["curated_uid"])
            sess = [s for s in str(row["kept_sessions"]).split(";") if s]
            sess_by_uid[uid] = sess
            for s1, s2 in combinations(sess, 2):
                r = sim_fn(fp_by_key.get((uid, s1)), fp_by_key.get((uid, s2)))
                if np.isfinite(r):
                    matched.append(r)
        nonmatched = []
        uids = list(sess_by_uid)
        for u1, u2 in combinations(uids, 2):
            for s in set(sess_by_uid[u1]) & set(sess_by_uid[u2]):
                r = sim_fn(fp_by_key.get((u1, s)), fp_by_key.get((u2, s)))
                if np.isfinite(r):
                    nonmatched.append(r)
        out[str(tier)] = {
            "auc": tc._auc(np.array(matched), np.array(nonmatched)),
            "n_matched": len(matched), "n_nonmatched": len(nonmatched),
        }
    return out


def _isi_sim(a, b) -> float:
    from visdetect.analysis import track_curation as tc
    return tc._pearson(a, b)


def _func_sim(fa: Optional[dict], fb: Optional[dict], keys=None) -> float:
    """Median Pearson r over PSTH conditions present in both (shape similarity).

    keys: optional iterable restricting which conditions to use (e.g. baseline only).
    """
    from visdetect.analysis import track_curation as tc
    import numpy as np
    if not fa or not fb:
        return float("nan")
    use = list(fa.keys()) if keys is None else [k for k in keys if k in fa]
    rs = []
    for k in use:
        pb = fb.get(k)
        if pb is None:
            continue
        r = tc._pearson(fa[k], pb)
        if np.isfinite(r):
            rs.append(r)
    return float(np.median(rs)) if rs else float("nan")


# ── Heavy: load each session once -> ISI holdout + PSTH fingerprints ─────────
def collect_isi_and_psth(kept_pairs: Dict[Tuple[int, str], int], subj: str, pkl_dir):
    import numpy as np
    sjp, tc, load_session = cd._import_pipeline(subj)
    from visdetect.analysis.tracking_qc import extract_unit_psths
    holdout: Dict[Tuple[int, str], object] = {}
    psth_fp: Dict[Tuple[int, str], dict] = {}
    for sess in sorted({s for (_, s) in kept_pairs}):
        pkl = sjp.session_pkl(subj, sess, pkl_dir)
        if pkl is None:
            print(f"  skip {sess}: no pkl", flush=True)
            continue
        S = load_session(str(pkl))
        cmap = {c.cluster_id: c for c in S.clusters}
        for (uid, s), kid in kept_pairs.items():
            if s != sess or kid not in cmap:
                continue
            _, hold = tc.partitioned_isi_hists(np.asarray(cmap[kid].spike_times))
            holdout[(uid, s)] = hold
            psths = extract_unit_psths(S, int(kid))
            psth_fp[(uid, s)] = {k: v[0] for k, v in psths.items() if v[0] is not None}
        del S; gc.collect()
        print(f"  {sess}: features extracted", flush=True)
    return holdout, psth_fp


def kept_pairs_from(tracks_df: pd.DataFrame, reg: pd.DataFrame, norm) -> Dict[Tuple[int, str], int]:
    """{(uid, kept_sessions_token) -> ks_unit_id}.

    `norm` normalizes session tokens for the join (e.g. session_date_key). The
    registry here is read padded ("08092025") but curate_tracks writes kept_sessions
    7-digit ("8092025"), so raw string equality would drop the 14 single-digit-day
    sessions (~31% of pairs). The dict key keeps the ORIGINAL kept_sessions token
    (so downstream session_pkl/zfill lookups and PSTH keys stay consistent).
    """
    reg = reg.copy()
    reg["uid"] = reg["dant_uid"].astype(int)
    lut = {(int(u), norm(k)): int(ks)
           for u, k, ks in zip(reg["uid"], reg["session"], reg["ks_unit_id"])}
    pairs: Dict[Tuple[int, str], int] = {}
    for _, row in tracks_df.iterrows():
        uid = int(row["curated_uid"])
        for s in [s for s in str(row["kept_sessions"]).split(";") if s]:
            ks = lut.get((uid, norm(s)))
            if ks is not None:
                pairs[(uid, s)] = ks
    return pairs


def _baseline_keys(psth_fp: Dict) -> list:
    """PSTH condition keys for the trial-RICH baseline window (stage-robust)."""
    keys: set = set()
    for d in psth_fp.values():
        keys.update(k for k in d if "baseline" in k.lower())
    return sorted(keys)


def matched_func_by_epoch(tracks_df: pd.DataFrame, psth_fp: Dict,
                          rank: Dict[str, int], n_sess: int) -> Dict[str, Dict[str, list]]:
    """Matched (same-uid, cross-session) functional similarity, bucketed by the
    EARLIER session's learning epoch (chronological thirds; late == Expert end).

    Reports TWO series per epoch:
      * 'all'      — median r over all PSTH conditions (incl. trial-STARVED Hit
                     conditions, which are noise-dominated in early Naive sessions)
      * 'baseline' — median r over the trial-RICH Baseline_ON condition only,
                     which is roughly stage-INVARIANT in trial count.
    If 'all' declines into early learning but 'baseline' stays flat, the decline is
    a trial-count/PSTH-noise artifact, NOT neural plasticity (the early Naive
    sessions are hit-trial-starved). Do not read 'all' as plasticity on its own.
    """
    import numpy as np
    e0, e1 = n_sess / 3.0, 2 * n_sess / 3.0
    bkeys = _baseline_keys(psth_fp)

    def epoch(s: str) -> str:
        r = rank.get(s, 0)
        return "early" if r < e0 else ("mid" if r < e1 else "late")

    buckets: Dict[str, Dict[str, list]] = {
        ep: {"all": [], "baseline": []} for ep in ("early", "mid", "late")}
    for _, row in tracks_df.iterrows():
        uid = int(row["curated_uid"])
        sess = [s for s in str(row["kept_sessions"]).split(";") if s]
        for s1, s2 in combinations(sess, 2):
            fa, fb = psth_fp.get((uid, s1)), psth_fp.get((uid, s2))
            earlier = s1 if rank.get(s1, 0) <= rank.get(s2, 0) else s2
            ep = epoch(earlier)
            r_all = _func_sim(fa, fb)
            if np.isfinite(r_all):
                buckets[ep]["all"].append(r_all)
            r_base = _func_sim(fa, fb, keys=bkeys)
            if np.isfinite(r_base):
                buckets[ep]["baseline"].append(r_base)
    return buckets


def _rows_for(name: str, tier_df: pd.DataFrame, isi_fp, psth_fp) -> list:
    counts = tier_df["tier"].value_counts().to_dict()
    isi = auc_by_tier(tier_df, isi_fp, _isi_sim, tier_col="tier")
    func = auc_by_tier(tier_df, psth_fp, _func_sim, tier_col="tier")
    rows = []
    for tier in ["trusted", "review", "suspect"]:
        i = isi.get(tier, {}); f = func.get(tier, {})
        rows.append({
            "scenario": name, "tier": tier, "n_tracks": int(counts.get(tier, 0)),
            "isi_auc": i.get("auc", float("nan")), "isi_n_matched": i.get("n_matched", 0),
            "func_auc": f.get("auc", float("nan")), "func_n_matched": f.get("n_matched", 0),
        })
    return rows


def main() -> int:
    subj = "BG_046"
    sjp, _, _ = cd._import_pipeline(subj)        # session_date_key for the normalized join
    tracks = pd.read_csv(CUR / "curated_tracks.csv")
    links = pd.read_csv(CUR / "curated_links.csv")
    reg = pd.read_csv(REG, dtype={"session": str})

    tracks["inclusive_tier"] = assign_inclusive_tiers(tracks, links, max_warn=1)
    n_orig_tr = int((tracks["confidence_tier"] == "trusted").sum())
    n_inc_tr = int((tracks["inclusive_tier"] == "trusted").sum())
    newly = tracks[(tracks["inclusive_tier"] == "trusted")
                   & (tracks["confidence_tier"] != "trusted")].copy()
    print(f"original trusted={n_orig_tr}  inclusive trusted={n_inc_tr}  "
          f"newly promoted={len(newly)}", flush=True)

    # Load features once over the union of all kept pairs (normalized join on
    # session_date_key so single-digit-day sessions are not dropped).
    kept_pairs = kept_pairs_from(tracks, reg, sjp.session_date_key)
    print(f"loading features for {len(kept_pairs)} kept (uid,session) pairs...", flush=True)
    isi_fp, psth_fp = collect_isi_and_psth(kept_pairs, subj, PKL)

    # Build per-scenario tier tables.
    orig = tracks[["curated_uid", "kept_sessions"]].copy()
    orig["tier"] = tracks["confidence_tier"]
    inc = tracks[["curated_uid", "kept_sessions"]].copy()
    inc["tier"] = tracks["inclusive_tier"]
    new_df = newly[["curated_uid", "kept_sessions"]].copy()
    new_df["tier"] = "trusted"

    rows = []
    rows += _rows_for("original", orig, isi_fp, psth_fp)
    rows += _rows_for("inclusive", inc, isi_fp, psth_fp)
    # newly-promoted-only (precision of the tracks the loosening admits).
    np_rows = _rows_for("newly_promoted", new_df, isi_fp, psth_fp)
    rows += [r for r in np_rows if r["tier"] == "trusted"]

    out = pd.DataFrame(rows)
    CUR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CUR / "inclusive_trusted_validation.csv", index=False)
    print(out.to_string(index=False), flush=True)

    # Functional similarity vs learning epoch (all kept tracks span>=2, to span the
    # full Naive->Expert range; the long review tracks are what reach early learning).
    all_sess = sorted({s for (_, s) in kept_pairs}, key=sjp.session_date_key)
    rank = {s: i for i, s in enumerate(all_sess)}
    print(f"epoch ranking over {len(all_sess)} sessions (should be 41)", flush=True)
    all_tracks = tracks[["curated_uid", "kept_sessions"]].copy()
    epoch_buckets = matched_func_by_epoch(all_tracks, psth_fp, rank, len(all_sess))

    def _med(v):
        import numpy as np
        return float(np.median(v)) if v else float("nan")
    for ser in ("all", "baseline"):
        print(f"matched PSTH r by epoch [{ser}]: "
              + ", ".join(f"{ep}: n={len(epoch_buckets[ep][ser])}, "
                          f"med={_med(epoch_buckets[ep][ser]):.3f}"
                          for ep in ("early", "mid", "late")), flush=True)

    _plot(out, n_orig_tr, n_inc_tr, len(newly), epoch_buckets,
          CUR / "inclusive_trusted_validation.png")
    print(f"wrote {CUR / 'inclusive_trusted_validation.csv'} + .png", flush=True)
    return 0


def _plot(out: pd.DataFrame, n_orig: int, n_inc: int, n_new: int,
          epoch_buckets: Dict[str, Dict[str, list]], png) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def cell(scenario, tier, col):
        r = out[(out.scenario == scenario) & (out.tier == tier)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.4))

    ax0.bar([0, 1], [n_orig, n_inc], color=["#9e9e9e", "#3474ae"], width=0.6)
    ax0.bar([1], [n_new], bottom=[n_orig], color="#6baed6", width=0.6,
            label=f"newly promoted (+{n_new})")
    ax0.set_xticks([0, 1]); ax0.set_xticklabels(["shipped\ntrusted", "inclusive\ntrusted"])
    ax0.set_ylabel("trusted tracks"); ax0.set_title("Recall: trusted count")
    ax0.legend(frameon=False, fontsize=8)

    groups = [("original", "trusted", "shipped-trusted"),
              ("inclusive", "trusted", "inclusive-trusted"),
              ("newly_promoted", "trusted", "newly-promoted"),
              ("original", "review", "review")]
    x = np.arange(len(groups))
    isi = [cell(s, t, "isi_auc") for s, t, _ in groups]
    func = [cell(s, t, "func_auc") for s, t, _ in groups]
    ax1.bar(x - 0.2, isi, width=0.4, color="#3474ae", label="held-out ISI AUC")
    ax1.bar(x + 0.2, func, width=0.4, color="#ef6548", label="functional PSTH AUC")
    ax1.axhline(0.5, color="k", lw=0.8, ls=":", label="chance")
    ax1.set_xticks(x); ax1.set_xticklabels([g[2] for g in groups], fontsize=8)
    ax1.set_ylim(0.4, 1.0); ax1.set_ylabel("AUC (matched vs random)")
    ax1.set_title("Precision: held-out ISI (quasi-indep.) + functional (entangled)")
    ax1.legend(frameon=False, fontsize=8)

    # Panel 3: matched PSTH similarity vs learning epoch -- two series. If 'all'
    # declines but trial-rich 'baseline' stays flat, the decline is a trial-count
    # artifact (early Naive sessions are hit-trial-starved), not plasticity.
    order = ["early", "mid", "late"]
    x = np.arange(len(order))
    def meds(ser):
        return [float(np.median(epoch_buckets[ep][ser])) if epoch_buckets[ep][ser]
                else np.nan for ep in order]
    ax2.plot(x, meds("all"), "-o", color="#ef6548", label="all conditions")
    ax2.plot(x, meds("baseline"), "-s", color="#3474ae",
             label="baseline only (trial-rich)")
    ax2.axhline(0.0, color="k", lw=0.8, ls=":", label="no similarity (r=0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{ep}\n(n={len(epoch_buckets[ep]['all'])})" for ep in order])
    ax2.set_ylabel("matched PSTH median Pearson r")
    ax2.set_xlabel("learning epoch of earlier session (late = Expert)")
    ax2.set_title("Functional agreement vs epoch (trial-count check)")
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("DANT BG_046 — inclusive-trusted: recall vs precision (ISI + functional)")
    fig.tight_layout()
    png = Path(png); png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
