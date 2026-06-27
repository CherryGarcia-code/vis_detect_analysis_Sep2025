#!/usr/bin/env python3
"""Composite-verdict re-tiering: how many DANT review tracks are actually composite-
pristine cells the link-by-link tiering wrongly demoted?

The curation sweep tiers link-by-link (any single warn link or bridge -> review). But a
long track can be GLOBALLY pristine (overall waveform r=0.99, FR CV<0.2) yet have one or
two noisy transitions. This counts those "hidden gems" with a whole-track verdict.

VERDICT DEFINITION (NOT identical to the QC-sheet header verdict): we use a 4-badge
BIOPHYSICAL composite over [badge_isi_hist_corr(isi-hist shape), badge_depth, badge_waveform,
badge_fr]. This INTENTIONALLY differs from the renderer's authoritative verdict
(build_qc_sheets), which uses 6 badges (adds badge_func_resp + a median-ISI badge_isi) and
then promotes via apply_isi_autopass. We deliberately EXCLUDE the functional badge (it would
re-introduce the PSTH->identity circularity this whole effort is trying to avoid) and the
median-ISI/autopass step (its isi_scores input comes from validate_long_tracks, not from
compute_uid_metrics). So read the output as "review tracks that pass the 4 core BIOPHYSICAL
badges", a biophysical-identity proxy — NOT "the sheet says trusted". The downstream held-out
ISI validation (validate_composite.py) is what actually certifies these are real, and it does
not depend on this verdict being renderer-exact.
Reuses extract_session_records -> compute_uid_metrics -> badge_* -> composite_verdict.
Run from the worktree root with the analysis interpreter (loads each session once).
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import curate_dant as cd            # noqa: E402

WT = cd.WORKTREE_ROOT
PRIMARY = cd.PRIMARY_DEFAULT
CUR = WT / "FIGURES" / "tracking_dant" / "BG_046" / "curation"
REG = WT / "data" / "cache" / "dant" / "BG_046" / "dant_registry_curation.csv"
RAWWF = PRIMARY / "data" / "unit_match" / "input" / "BG_046"
PKL = PRIMARY / "data" / "pkls" / "BG_046"

os.environ["VISDETECT_SUBJECT"] = "BG_046"
sys.path.insert(0, str(WT / "scripts" / "pipelines" / "tracking"))
sys.path.insert(0, str(WT / "src"))


def main() -> int:
    import _subject_paths as sjp
    from visdetect.analysis.tracking_qc import (
        extract_session_records, load_channel_positions, UIDIntermediate,
        badge_isi_hist_corr, badge_depth, badge_waveform, badge_fr, composite_verdict)
    from visdetect.core.session import load_session
    from build_qc_sheets import compute_uid_metrics

    tracks = pd.read_csv(CUR / "curated_tracks.csv")
    # Read registry WITHOUT dtype (unpadded) to match kept_sessions tokens, exactly as
    # render_curation_sheets does — both sides unpadded => consistent join (no zfill bug).
    reg = pd.read_csv(REG)
    reg["session"] = reg["session"].astype(str)
    reg["uid"] = reg["dant_uid"].astype(int)

    kept_by_uid, tier_by_uid = {}, {}
    for _, r in tracks.iterrows():
        kept = {s for s in str(r["kept_sessions"]).split(";") if s}
        if len(kept) >= 2:
            u = int(r["curated_uid"])
            kept_by_uid[u] = kept
            tier_by_uid[u] = str(r["confidence_tier"])

    uid_to_ks = {}
    for _, r in reg.iterrows():
        u = int(r["uid"])
        if u in kept_by_uid:
            uid_to_ks.setdefault(u, {})[str(r["session"])] = int(r["ks_unit_id"])

    intermediates = {
        u: UIDIntermediate(global_uid=u, span=len(ks), has_naive_to_expert=False,
                           suspect_known=False, sessions=[])
        for u, ks in uid_to_ks.items()}
    sess_set = sorted({s for ks in uid_to_ks.values() for s in ks}, key=sjp.session_date_key)
    for sess in sess_set:
        pkl = sjp.session_pkl("BG_046", sess, PKL)
        if pkl is None:
            print(f"  skip {sess}: no pkl", flush=True); continue
        S = load_session(str(pkl))
        cp = load_channel_positions(RAWWF, sess)
        uids_here = [u for u, ks in uid_to_ks.items() if sess in ks]
        ks_here = [uid_to_ks[u][sess] for u in uids_here]
        records = extract_session_records(S, ks_here, session_name=sess, stage="Unknown",
                                          raw_wf_root=RAWWF, channel_positions=cp)
        for u in uids_here:
            rec = records.get(int(uid_to_ks[u][sess]))
            if rec is not None:
                intermediates[u].sessions.append(rec)
        del S; gc.collect()
        print(f"  {sess}: {len(records)} records", flush=True)

    rows = []
    for u, iv in intermediates.items():
        kept = kept_by_uid[u]
        kr = [r for r in iv.sessions if r.session_name in kept]
        if len(kr) < 2:
            continue
        kiv = UIDIntermediate(global_uid=u, span=len(kr), has_naive_to_expert=False,
                              suspect_known=False, sessions=kr)
        m = compute_uid_metrics(kiv)
        bs = [badge_isi_hist_corr(m["isi_hist_corr"]), badge_depth(m["depth_std_um"]),
              badge_waveform(m["wave_corr"]), badge_fr(m["fr_cv"])]
        rows.append(dict(curated_uid=u, curation_tier=tier_by_uid[u], span=len(kr),
                         composite_verdict=composite_verdict(bs),
                         isi_hist_corr=m["isi_hist_corr"], depth_std_um=m["depth_std_um"],
                         wave_corr=m["wave_corr"], fr_cv=m["fr_cv"]))
    df = pd.DataFrame(rows)
    CUR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CUR / "composite_retier.csv", index=False)

    ct = pd.crosstab(df["curation_tier"], df["composite_verdict"])
    print("\n=== curation tier (rows) x composite verdict (cols) ===", flush=True)
    print(ct.to_string(), flush=True)
    gems = df[(df.curation_tier == "review") & (df.composite_verdict == "trusted")]
    print(f"\nHIDDEN GEMS (curation=review, composite=trusted): {len(gems)} tracks; "
          f"span median={gems['span'].median():.0f}, max={gems['span'].max() if len(gems) else 0}; "
          f"span>=10: {len(gems[gems.span >= 10])}", flush=True)
    _plot(df, ct, gems, CUR / "composite_retier.png")
    print(f"wrote {CUR / 'composite_retier.csv'} + .png", flush=True)
    return 0


def _plot(df, ct, gems, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    tiers = ["trusted", "review", "suspect"]
    verds = ["trusted", "review", "suspect"]
    cols = {"trusted": "#2ca25f", "review": "#fdae6b", "suspect": "#9e9e9e"}
    bottom = np.zeros(len(tiers))
    for v in verds:
        vals = [int(ct.loc[t, v]) if (t in ct.index and v in ct.columns) else 0 for t in tiers]
        ax[0].bar(tiers, vals, bottom=bottom, label=f"composite={v}", color=cols[v])
        bottom += np.array(vals)
    ax[0].set_ylabel("tracks"); ax[0].set_xlabel("curation (link-by-link) tier")
    ax[0].set_title("Composite verdict within each curation tier\n(orange-in-review-bar = hidden gems)")
    ax[0].legend(frameon=False, fontsize=8)

    if len(gems):
        ax[1].hist(gems["span"], bins=range(2, int(gems["span"].max()) + 2),
                   color="#2ca25f", align="left")
    ax[1].set_xlabel("kept span (sessions)"); ax[1].set_ylabel("hidden-gem tracks")
    ax[1].set_title(f"Hidden gems by span (n={len(gems)})\nreview tier but composite=TRUSTED")
    fig.suptitle("DANT BG_046 — composite-verdict re-tiering (how many clean cells the "
                 "link-strict rule demoted)")
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
