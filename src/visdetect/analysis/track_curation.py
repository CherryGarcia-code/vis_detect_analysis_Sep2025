"""Precision curation of UnitMatch cross-session tracks.

Expert->Naive backward sweep over the liberal UM registry: biophysical gate +
availability-gated in-zone functional corroborator, rolling anchor with
gap-bridge tolerance. Never alters the original registry. See
docs/superpowers/specs/2026-06-07-track-curation-design.md.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from visdetect.analysis.tracking_qc import isi_log_histogram


def partitioned_isi_hists(spike_times: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Two log-ISI histograms from disjoint spike partitions (even/odd index).

    The curation ISI feature uses one partition; validation uses the other, so
    ISI validation is statistically independent of the ISI curation feature
    (spec sec 8.1). Both estimate the same stationary fingerprint.

    Returns (curation_hist, holdout_hist), each shape (50,); all-NaN if a
    partition has too few spikes.
    """
    st = np.asarray(spike_times, dtype=float)
    st = np.sort(st)
    cur = st[0::2]
    hold = st[1::2]
    cur_h, _ = isi_log_histogram(cur)
    hold_h, _ = isi_log_histogram(hold)
    return cur_h, hold_h


from dataclasses import dataclass, field
from typing import Dict, List, Optional

from visdetect.analysis.tracking_qc import (
    load_raw_mean_waveform, extract_peak_channel, extract_footprint,
    extract_unit_psths,
)


@dataclass
class CurationFeature:
    session_name: str
    ks_unit_id: int
    stage: str
    waveform_peak: np.ndarray
    footprint: np.ndarray
    footprint_channels: np.ndarray
    peak_chan: int
    peak_depth_um: float
    peak_depth_corrected_um: float
    baseline_fr_hz: float
    isi_hist_curation: np.ndarray
    isi_hist_holdout: np.ndarray
    inzone_psths: Dict[str, Optional[np.ndarray]]
    n_inzone_trials: int


def _baseline_fr(cluster, session) -> float:
    st = np.asarray(cluster.spike_times, dtype=float)
    if st.size == 0:
        return 0.0
    dur = float(st.max() - st.min())
    return float(st.size / dur) if dur > 0 else 0.0


def extract_curation_feature(session, ks_unit_id: int, session_name: str,
                             stage: str, raw_wf_root,
                             channel_positions: Optional[np.ndarray],
                             in_zone_idx: List[int],
                             drift_offset: float = 0.0,
                             ) -> Optional[CurationFeature]:
    """Assemble a CurationFeature for one (session, uid). None if no waveform."""
    cluster_map = {c.cluster_id: c for c in session.clusters}
    cluster = cluster_map.get(int(ks_unit_id))
    if cluster is None:
        return None

    mean_wf = load_raw_mean_waveform(raw_wf_root, session_name, int(ks_unit_id))
    if mean_wf is None:
        return None
    peak_chan = extract_peak_channel(mean_wf)
    peak_wave = mean_wf[:, peak_chan]
    footprint, fp_chans = extract_footprint(mean_wf, peak_chan)

    if channel_positions is not None and peak_chan < channel_positions.shape[0]:
        depth_um = float(channel_positions[peak_chan, 1])
    else:
        depth_um = float("nan")
    depth_corr = depth_um - float(drift_offset) if np.isfinite(depth_um) else float("nan")

    cur_h, hold_h = partitioned_isi_hists(np.asarray(cluster.spike_times))

    in_zone_set = set(int(i) for i in in_zone_idx)
    psth_dict = extract_unit_psths(session, int(ks_unit_id),
                                   restrict_trials=in_zone_set)
    inzone_psths = {k: v[0] for k, v in psth_dict.items()}

    return CurationFeature(
        session_name=session_name, ks_unit_id=int(ks_unit_id), stage=stage,
        waveform_peak=peak_wave.astype(np.float32),
        footprint=footprint.astype(np.float32), footprint_channels=fp_chans,
        peak_chan=peak_chan, peak_depth_um=depth_um,
        peak_depth_corrected_um=depth_corr,
        baseline_fr_hz=_baseline_fr(cluster, session),
        isi_hist_curation=cur_h, isi_hist_holdout=hold_h,
        inzone_psths=inzone_psths, n_inzone_trials=len(in_zone_set),
    )


from visdetect.analysis.tracking_qc import (
    badge_waveform, badge_depth, badge_isi_hist_corr, badge_func_resp,
    FUNC_RESP_MIN_PSTH_STD,
)

MAX_BRIDGE_GAP = 2
MIN_INZONE_TRIALS = 20
MIN_TRUSTED_SPAN = 3


@dataclass
class CurationParams:
    max_bridge_gap: int = MAX_BRIDGE_GAP
    min_inzone_trials: int = MIN_INZONE_TRIALS
    min_trusted_span: int = MIN_TRUSTED_SPAN
    corroborator_ref: str = "rolling"     # "rolling" | "expert"


@dataclass
class LinkResult:
    anchor_session: str
    candidate_session: str
    gap_sessions: int
    wave_corr: float
    depth_jump_um: float
    depth_evaluable: bool
    isi_shape_corr: float
    func_corr: float
    func_evaluable: bool
    n_inzone_trials: int
    decision: str           # "KEEP" | "SKIP" | "STOP"
    review_flag: bool
    stop_reason: str = ""


def _pearson(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _func_corr(ref: CurationFeature, cand: CurationFeature) -> float:
    """Median pairwise Pearson r over conditions both have modulated PSTHs for."""
    rs: List[float] = []
    for key, ref_psth in ref.inzone_psths.items():
        cand_psth = cand.inzone_psths.get(key)
        if ref_psth is None or cand_psth is None:
            continue
        if (float(np.std(ref_psth)) < FUNC_RESP_MIN_PSTH_STD
                or float(np.std(cand_psth)) < FUNC_RESP_MIN_PSTH_STD):
            continue
        r = _pearson(ref_psth, cand_psth)
        if np.isfinite(r):
            rs.append(r)
    return float(np.median(rs)) if rs else float("nan")


def _depth_jump(anchor: CurationFeature, candidate: CurationFeature) -> float:
    """Cross-session depth difference (um), measured in a consistent frame.

    Prefer drift-CORRECTED depth when both sessions have it; otherwise fall back
    to RAW depth. On BG_046 whole-probe inter-session drift is ~0 (amplitude-depth
    fingerprint diagnostic, corr 0.88), so raw depth is comparable across days and
    the match-based correction — which starves on low-anchor sessions and returns
    NaN — must not be allowed to manufacture a depth "contradiction". NaN only if
    NEITHER frame is available for both units, in which case the caller treats
    depth as not-evaluable (abstain), not as a fail.
    """
    ac = float(anchor.peak_depth_corrected_um)
    bc = float(candidate.peak_depth_corrected_um)
    if np.isfinite(ac) and np.isfinite(bc):
        return abs(ac - bc)
    ar = float(anchor.peak_depth_um)
    br = float(candidate.peak_depth_um)
    if np.isfinite(ar) and np.isfinite(br):
        return abs(ar - br)
    return float("nan")


def score_link(anchor: CurationFeature, candidate: CurationFeature,
               corroborator_ref: CurationFeature, params: CurationParams,
               gap_sessions: int = 1) -> LinkResult:
    """Decide one cross-session link: biophysical gate + functional corroborator."""
    wave_corr = _pearson(anchor.waveform_peak, candidate.waveform_peak)
    depth_jump = _depth_jump(anchor, candidate)
    isi_corr = _pearson(anchor.isi_hist_curation, candidate.isi_hist_curation)

    w = badge_waveform(wave_corr)
    # Depth abstains (does not vote) when it cannot be measured in any frame.
    # "unknown depth" != "depths disagree" — abstain never fails/stops a link.
    depth_evaluable = bool(np.isfinite(depth_jump))
    d = badge_depth(depth_jump) if depth_evaluable else "na"

    # Functional corroborator (availability-gated).
    func_evaluable = candidate.n_inzone_trials >= params.min_inzone_trials
    func_corr = _func_corr(corroborator_ref, candidate) if func_evaluable else float("nan")
    if func_evaluable and not np.isfinite(func_corr):
        func_evaluable = False          # no modulated condition -> not evaluable

    base = dict(
        anchor_session=anchor.session_name,
        candidate_session=candidate.session_name,
        gap_sessions=int(gap_sessions),
        wave_corr=wave_corr, depth_jump_um=depth_jump,
        depth_evaluable=depth_evaluable, isi_shape_corr=isi_corr,
        func_corr=func_corr, func_evaluable=func_evaluable,
        n_inzone_trials=candidate.n_inzone_trials,
    )

    if w == "fail" and d == "fail":
        return LinkResult(**base, decision="STOP", review_flag=False,
                          stop_reason="hard_contradiction")
    if w == "pass" and d in ("pass", "na"):
        review = (badge_isi_hist_corr(isi_corr) != "pass")
        if func_evaluable and badge_func_resp(func_corr) != "pass":
            review = True
        if not depth_evaluable:
            review = True               # kept without depth corroboration
        return LinkResult(**base, decision="KEEP", review_flag=review)
    return LinkResult(**base, decision="SKIP", review_flag=False)


@dataclass
class SweepResult:
    liberal_uid: int
    anchor_session: str
    kept_sessions: List[str]
    skipped_sessions: List[str]
    dropped_sessions: List[str]
    links: List[LinkResult] = field(default_factory=list)
    confidence_tier: str = "suspect"


def compute_tier(kept_sessions: List[str], skipped_sessions: List[str],
                 kept_links: List[LinkResult], params: CurationParams) -> str:
    """trusted / review / suspect for a curated track (spec sec 6.2)."""
    span = len(kept_sessions)
    if span < 2:
        return "suspect"
    any_review = any(lr.review_flag for lr in kept_links)
    any_bridge = len(skipped_sessions) > 0
    if span >= params.min_trusted_span and not any_review and not any_bridge:
        return "trusted"
    return "review"


def sweep_uid(features_by_session: Dict[str, CurationFeature],
              session_order: List[str], params: CurationParams,
              liberal_uid: int = -1) -> SweepResult:
    """Expert->Naive backward sweep over one liberal-uid's sessions.

    session_order: chronological ascending; anchor = most-recent (last).
    """
    present = [s for s in session_order if s in features_by_session]
    if not present:
        return SweepResult(liberal_uid, "", [], [], list(session_order))
    anchor_sess = present[-1]
    expert_anchor = features_by_session[anchor_sess]
    anchor = expert_anchor
    anchor_pos = len(present) - 1

    kept = [anchor_sess]
    skipped: List[str] = []
    dropped: List[str] = []
    pending: List[str] = []
    links: List[LinkResult] = []
    n_bridge = 0

    i = len(present) - 2
    while i >= 0:
        cand_sess = present[i]
        cand = features_by_session[cand_sess]
        ref = anchor if params.corroborator_ref == "rolling" else expert_anchor
        lr = score_link(anchor, cand, ref, params, gap_sessions=anchor_pos - i)
        links.append(lr)
        if lr.decision == "KEEP":
            kept.append(cand_sess)
            skipped.extend(pending); pending = []
            anchor = cand; anchor_pos = i; n_bridge = 0
        elif lr.decision == "SKIP":
            pending.append(cand_sess); n_bridge += 1
            if n_bridge > params.max_bridge_gap:
                dropped.extend(pending); pending = []
                dropped.extend(present[:i])         # all earlier sessions
                break
        else:  # STOP
            dropped.extend(pending); pending = []
            dropped.append(cand_sess)
            dropped.extend(present[:i])
            break
        i -= 1
    dropped.extend(pending)                          # trailing unclosed skips

    kept_links = [lr for lr in links if lr.decision == "KEEP"]
    tier = compute_tier(kept, skipped, kept_links, params)
    return SweepResult(
        liberal_uid=liberal_uid, anchor_session=anchor_sess,
        kept_sessions=kept, skipped_sessions=skipped, dropped_sessions=dropped,
        links=links, confidence_tier=tier,
    )


import pandas as pd


def curate_registry(uid_to_sessions: Dict[int, List[str]],
                    features: Dict, params: CurationParams):
    """Run the sweep for every uid; return (links_df, tracks_df).

    uid_to_sessions: {liberal_uid -> chronological-ascending session list}.
    features: {(liberal_uid, session_name) -> CurationFeature}.
    """
    link_rows: List[dict] = []
    track_rows: List[dict] = []
    for uid in sorted(uid_to_sessions):
        order = uid_to_sessions[uid]
        feats = {s: features[(uid, s)] for s in order if (uid, s) in features}
        res = sweep_uid(feats, [s for s in order if s in feats], params,
                        liberal_uid=uid)
        for lr in res.links:
            link_rows.append({
                "liberal_uid": uid,
                "anchor_session": lr.anchor_session,
                "candidate_session": lr.candidate_session,
                "gap_sessions": lr.gap_sessions,
                "wave_corr": lr.wave_corr,
                "depth_jump_um": lr.depth_jump_um,
                "depth_evaluable": lr.depth_evaluable,
                "isi_shape_corr": lr.isi_shape_corr,
                "func_corr": lr.func_corr,
                "func_evaluable": lr.func_evaluable,
                "n_inzone_trials": lr.n_inzone_trials,
                "link_decision": lr.decision,
                "review_flag": lr.review_flag,
                "stop_reason": lr.stop_reason,
            })
        track_rows.append({
            "curated_uid": uid,            # 1:1 with liberal_uid (Expert-anchored)
            "liberal_uid": uid,
            "anchor_session": res.anchor_session,
            "kept_sessions": ";".join(res.kept_sessions),
            "skipped_sessions": ";".join(res.skipped_sessions),
            "dropped_sessions": ";".join(res.dropped_sessions),
            "trimmed_span": len(res.kept_sessions),
            "n_bridged": len(res.skipped_sessions),
            "confidence_tier": res.confidence_tier,
        })
    return pd.DataFrame(link_rows), pd.DataFrame(track_rows)


from itertools import combinations


def _auc(matched: np.ndarray, nonmatched: np.ndarray) -> float:
    """ROC AUC of matched (label 1) vs nonmatched (label 0) scores."""
    if len(matched) == 0 or len(nonmatched) == 0:
        return float("nan")
    scores = np.concatenate([matched, nonmatched])
    labels = np.concatenate([np.ones_like(matched), np.zeros_like(nonmatched)])
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    tpr = tp / max(1, labels.sum()); fpr = fp / max(1, (1 - labels).sum())
    return float(np.trapz(tpr, fpr))


def held_out_isi_auc_by_tier(tracks_df, holdout_isi: Dict) -> Dict[str, dict]:
    """Per-tier held-out-ISI AUC (spec sec 8.2).

    tracks_df: must have curated_uid, kept_sessions (';'-joined), confidence_tier.
    holdout_isi: {(curated_uid, session) -> holdout ISI hist (50,)}.
    Matched = cross-session pairs within a curated_uid's kept sessions.
    Non-matched = within-session pairs across different curated_uids.
    """
    out: Dict[str, dict] = {}
    for tier, grp in tracks_df.groupby("confidence_tier"):
        matched: List[float] = []
        # matched: cross-session, same uid
        sess_by_uid: Dict[int, List[str]] = {}
        for _, row in grp.iterrows():
            uid = int(row["curated_uid"])
            sess = [s for s in str(row["kept_sessions"]).split(";") if s]
            sess_by_uid[uid] = sess
            for s1, s2 in combinations(sess, 2):
                r = _pearson(holdout_isi.get((uid, s1)), holdout_isi.get((uid, s2)))
                if np.isfinite(r):
                    matched.append(r)
        # non-matched: within-session, different uid
        nonmatched: List[float] = []
        uids = list(sess_by_uid)
        for u1, u2 in combinations(uids, 2):
            shared = set(sess_by_uid[u1]) & set(sess_by_uid[u2])
            for s in shared:
                r = _pearson(holdout_isi.get((u1, s)), holdout_isi.get((u2, s)))
                if np.isfinite(r):
                    nonmatched.append(r)
        out[str(tier)] = {
            "auc": _auc(np.array(matched), np.array(nonmatched)),
            "n_matched": len(matched),
            "n_nonmatched": len(nonmatched),
        }
    return out
