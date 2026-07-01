"""Clean talk-ready 1-page 'representative neuron' figure per top DANT trusted track.

Same presentation style as the BG_046 UM x DANT consensus figures, but for a single
tracker (DANT) — used where there is no clean two-tracker consensus (e.g. BG_031,
where UM/DANT agree only at ARI ~0.19). Each figure argues from FOUR biophysical
lines of evidence that ONE neuron was followed across sessions:
  * waveform-shape stability across sessions (peak-channel overlay)
  * multi-channel footprint stability (first vs last kept session)
  * probe-depth stability
  * held-out log-ISI fingerprint vs the simultaneously-recorded trusted population
    (the independent validation axis; DANT never uses spike timing to match)

Reads the DANT curation output (curated_tracks.csv, trusted tier) + registry.
Loads each needed session pkl once (all LOCAL). Subject-general via --subject.

Usage:  py scripts/tracking_dant/render_dant_candidate_figures.py --subject BG_031 [--n 8]
Output: FIGURES/tracking_dant/<subj>/curation/candidate_figs/dant_uid<U>_span<N>.png
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "pipelines" / "tracking"))
from visdetect.analysis.config import canonical_session_id, session_date_key  # noqa: E402
from visdetect.analysis.track_curation import partitioned_isi_hists  # noqa: E402
from visdetect.analysis.tracking_qc import (  # noqa: E402
    load_raw_mean_waveform, load_channel_positions,
    extract_peak_channel, extract_footprint, isi_log_histogram,
)
from visdetect.core.session import load_session  # noqa: E402
import _subject_paths as sjp  # noqa: E402

STAGE_COLORS = {"Naive": "#addd8e", "Learning": "#74c476", "Expert": "#238b45",
                "Excluded": "#d9d9d9", "Unknown": "#f0f0f0"}
STAGE_RANK = {"Naive": 0, "Learning": 1, "Expert": 2}
_, ISI_CENTERS = isi_log_histogram(np.array([]))


def _stage_map(subject):
    p = ROOT / f"data/{subject}_staging_manifest.csv"
    if not p.exists():
        return {}
    st = pd.read_csv(p, dtype={"session_name": str})
    return {session_date_key(s): stg for s, stg in zip(st["session_name"], st["stage"])}


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a is None or b is None or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    if a.std() == 0 or b.std() == 0:
        return np.nan
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_031")
    ap.add_argument("--n", type=int, default=8, help="how many top trusted tracks to render")
    ap.add_argument("--tier", default="trusted")
    args = ap.parse_args()
    subj = args.subject

    cur_dir = ROOT / f"FIGURES/tracking_dant/{subj}/curation"
    tracks = pd.read_csv(cur_dir / "curated_tracks.csv")
    reg = pd.read_csv(ROOT / f"data/cache/dant/{subj}/dant_registry.csv", dtype={"session": str})
    reg["dant_uid"] = reg["dant_uid"].astype(int)
    reg["ks_unit_id"] = reg["ks_unit_id"].astype(int)
    raw_wf_root = ROOT / f"data/unit_match/input/{subj}"
    pkl_dir = sjp.pkl_dir(subj)
    stage_of = _stage_map(subj)
    out_dir = cur_dir / "candidate_figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tier = tracks[tracks["confidence_tier"] == args.tier].copy()
    tier = tier.sort_values("trimmed_span", ascending=False)
    top = list(tier["curated_uid"].head(args.n))
    print(f"{subj}: {len(tier)} {args.tier} tracks; rendering top {len(top)} by kept span: {top}")

    # ---- one pkl pass: holdout log-ISI for ALL trusted members (validation population) ----
    trusted_uids = set(tier["curated_uid"].astype(int))
    tmembers = reg[reg["dant_uid"].isin(trusted_uids)].copy()
    tmembers["sk"] = tmembers["session"].map(canonical_session_id)
    holdout = {}   # (session_token, ks) -> holdout hist
    fullisi = {}   # (session_token, ks) -> full isi hist
    sessions = sorted(tmembers["session"].unique(), key=session_date_key)
    for i, sess in enumerate(sessions, 1):
        pkl = sjp.session_pkl(subj, sess, pkl_dir)
        if pkl is None:
            print(f"  [{i}/{len(sessions)}] {sess}: no pkl -> skip"); continue
        S = load_session(str(pkl))
        by_id = {int(c.cluster_id): c for c in S.clusters}
        need = tmembers.loc[tmembers["session"] == sess, "ks_unit_id"].unique()
        for kid in need:
            c = by_id.get(int(kid))
            if c is None:
                continue
            st = np.asarray(c.spike_times, float)
            _, hold = partitioned_isi_hists(st)
            holdout[(sess, int(kid))] = np.asarray(hold, np.float32)
            fh, _ = isi_log_histogram(st)
            fullisi[(sess, int(kid))] = np.asarray(fh, np.float32)
        del S
        print(f"  [{i}/{len(sessions)}] {sess}: {len(need)} units", flush=True)

    # non-matched null: within-session pairs across DIFFERENT trusted tracks
    node_uid = {(r["session"], int(r["ks_unit_id"])): int(r["dant_uid"])
                for _, r in tmembers.iterrows()}
    bysess = {}
    for (sess, kid), h in holdout.items():
        if h is None or not np.all(np.isfinite(h)):
            continue
        bysess.setdefault(sess, []).append((node_uid[(sess, kid)], h))
    nonmatched = []
    for sess, items in bysess.items():
        for (u1, h1), (u2, h2) in combinations(items, 2):
            if u1 != u2:
                c = _corr(h1, h2)
                if np.isfinite(c):
                    nonmatched.append(c)
    nonmatched = np.array(nonmatched, float)

    # ---- render each top candidate ----
    done = []
    for uid in top:
        row = tier[tier["curated_uid"] == uid].iloc[0]
        kept = [s for s in str(row["kept_sessions"]).split(";") if s]
        kept = sorted(kept, key=session_date_key)
        # per-session ks unit + stage
        recs = []
        for s in kept:
            m = reg[(reg["session"] == s) & (reg["dant_uid"] == int(uid))]
            if m.empty:
                continue
            recs.append((s, int(m.iloc[0]["ks_unit_id"]), stage_of.get(session_date_key(s), "Unknown")))
        if len(recs) < 2:
            print(f"  uid {uid}: <2 resolvable sessions -> skip"); continue

        # waveforms / footprints / depth (npy)
        traces, depths, footprints, vsess, vstage = [], [], [], [], []
        for s, kid, stg in recs:
            wf = load_raw_mean_waveform(raw_wf_root, s, kid)
            if wf is None:
                continue
            pos = load_channel_positions(raw_wf_root, s)
            pc = extract_peak_channel(wf)
            traces.append(wf[:, pc].astype(float))
            depths.append(float(pos[pc, 1]) if pos is not None and pc < len(pos) else np.nan)
            snip, _ = extract_footprint(wf, pc)
            footprints.append(snip); vsess.append(s); vstage.append(stg)
        if len(vsess) < 2:
            print(f"  uid {uid}: <2 usable waveforms -> skip"); continue

        # wave r (pairwise) + matched ISI corr (holdout) + population percentile
        wpairs = [np.corrcoef(traces[i][:min(len(traces[i]),len(traces[j]))],
                              traces[j][:min(len(traces[i]),len(traces[j]))])[0,1]
                  for i, j in combinations(range(len(traces)), 2)]
        wave_r = float(np.nanmean(wpairs)) if wpairs else np.nan
        hh = [holdout.get((s, kid)) for s, kid, _ in recs]
        mcorr = [_corr(a, b) for a, b in combinations([h for h in hh if h is not None], 2)]
        mcorr = [c for c in mcorr if np.isfinite(c)]
        matched_isi = float(np.mean(mcorr)) if mcorr else np.nan
        pctile = float((nonmatched < matched_isi).mean()) if (np.isfinite(matched_isi) and len(nonmatched)) else np.nan
        auc = _auc_matched_vs_null(mcorr, nonmatched)

        colors = [plt.get_cmap("viridis")(i / max(len(vsess) - 1, 1)) for i in range(len(vsess))]
        stages_seq = [stage_of.get(session_date_key(s), "Unknown") for s in kept]
        sset = {s for s in stages_seq if s in STAGE_RANK}
        n2e = "Naive→Expert" if ("Naive" in sset and "Expert" in sset) else (
            "Learning→Expert" if (("Learning" in sset or "Naive" in sset) and "Expert" in sset)
            else "single/other")

        fig = plt.figure(figsize=(15, 8.8))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1.0, 1.0, 0.55], hspace=0.5, wspace=0.3,
                               left=0.06, right=0.975, top=0.82, bottom=0.08)
        fig.text(0.06, 0.945, f"DANT-tracked neuron  —  {subj}  dant_uid #{int(uid)}"
                 f"   ({len(vsess)} sessions; {n2e})",
                 fontsize=15, fontweight="bold", ha="left")
        depth_rng = np.nanmax(depths) - np.nanmin(depths)
        fig.text(0.06, 0.905,
                 f"waveform shape r = {wave_r:.2f}      "
                 f"held-out ISI r = {matched_isi:.2f} "
                 f"(> {pctile*100:.0f}% of unrelated pairs; trusted-pop AUC {auc:.2f})      "
                 f"depth range {depth_rng:.0f} µm      span type: {n2e}",
                 fontsize=10.5, color="#222")
        fig.text(0.06, 0.882,
                 f"DANT curation tier: {args.tier}  (biophysical gate; held-out-ISI validated). "
                 "Single-tracker — no UM consensus on this subject.",
                 fontsize=9.5, color="#666", style="italic")

        # (0,0) waveform overlay
        ax = fig.add_subplot(gs[0, 0])
        for tr, c in zip(traces, colors):
            ax.plot(tr, color=c, lw=1.2, alpha=0.85)
        ax.set_title(f"Spike waveform, every session\n(peak channel; r = {wave_r:.2f})", fontsize=11)
        ax.set_xlabel("sample"); ax.set_ylabel("µV"); ax.spines[["top", "right"]].set_visible(False)
        cax = ax.inset_axes([0.60, 0.07, 0.36, 0.05])   # tucked bottom-right, clear of header
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), cax=cax, orientation="horizontal")
        cb.set_ticks([0, 1]); cb.set_ticklabels(["early", "late"]); cb.ax.tick_params(labelsize=6.5)
        cb.set_label("session order", fontsize=6.5)

        # (0,1) ISI overlay (holdout)
        ax = fig.add_subplot(gs[0, 1]); n_isi = 0
        for (s, kid, _), c in zip(recs, colors):
            h = holdout.get((s, kid))
            if h is None or not np.all(np.isfinite(h)):
                continue
            ax.plot(ISI_CENTERS, h, color=c, lw=1.1, alpha=0.8); n_isi += 1
        ax.set_xscale("log")
        ax.set_title(f"ISI fingerprint\n(held-out spikes, {n_isi} sessions)", fontsize=11)
        ax.set_xlabel("ISI (s)"); ax.set_ylabel("prob."); ax.spines[["top", "right"]].set_visible(False)

        # (0,2) ISI vs population
        ax = fig.add_subplot(gs[0, 2])
        if len(nonmatched):
            ax.hist(nonmatched, bins=40, color="#cccccc", density=True, label="unrelated pairs (null)")
        if np.isfinite(matched_isi):
            ax.axvline(matched_isi, color="#d7301f", lw=2.5, label=f"this neuron (r={matched_isi:.2f})")
        ax.set_title(f"ISI match vs population\n(AUC {auc:.2f}; beats {pctile*100:.0f}% of null)", fontsize=11)
        ax.set_xlabel("cross-session ISI corr"); ax.set_ylabel("density")
        ax.legend(fontsize=8, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)

        # (1,0)/(1,1) footprints first/last
        for col, idx, lab in [(0, 0, "first"), (1, len(vsess) - 1, "last")]:
            ax = fig.add_subplot(gs[1, col])
            snip = footprints[idx]; mx = np.abs(snip).max() or 1.0
            ax.imshow(snip.T, aspect="auto", cmap="RdBu_r", origin="lower", vmin=-mx, vmax=mx)
            ax.set_title(f"Footprint — {lab} session\n{vsess[idx]} ({vstage[idx]})", fontsize=10)
            ax.set_xlabel("sample"); ax.set_ylabel("channel (near peak)")

        # (1,2) depth trajectory
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(range(len(vsess)), depths, "-o", color="#238b45", lw=1.5, ms=5)
        ax.set_title(f"Probe depth stability\n(range {depth_rng:.0f} µm)", fontsize=11)
        ax.set_xlabel("session (early → late)"); ax.set_ylabel("depth (µm)")
        ax.spines[["top", "right"]].set_visible(False)

        # (2,:) DANT track strip across all its sessions (kept highlighted, stage-coloured)
        ax = fig.add_subplot(gs[2, :])
        all_sess = sorted(reg[reg["dant_uid"] == int(uid)]["session"].unique(), key=session_date_key)
        keptset = set(kept)
        x = np.arange(len(all_sess))
        for xi, s in zip(x, all_sess):
            ax.add_patch(plt.Rectangle((xi - 0.45, 1.05), 0.9, 0.4,
                         color=STAGE_COLORS.get(stage_of.get(session_date_key(s), "Unknown"), "#eee")))
            ax.plot(xi, 0.5, "s", ms=13, color=("#d7301f" if s in keptset else "#fdae6b"))
        ax.set_xlim(-0.7, len(all_sess) - 0.3); ax.set_ylim(0.2, 1.6)
        ax.set_yticks([0.5, 1.25]); ax.set_yticklabels(["DANT", "stage"])
        ax.set_xticks(x); ax.set_xticklabels([s.replace(subj + "_", "") for s in all_sess],
                                             rotation=90, fontsize=6.5)
        ax.set_title(f"DANT track across sessions (red = trusted-kept, {len(kept)}; "
                     f"orange = trimmed; raw DANT span {len(all_sess)})", fontsize=10)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)

        out = out_dir / f"dant_uid{int(uid)}_span{len(vsess)}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"  wrote {out.name}  (wave_r {wave_r:.2f}, ISI {matched_isi:.2f}, AUC {auc:.2f})")
        done.append({"dant_uid": int(uid), "kept_span": len(vsess), "wave_r": round(wave_r, 3),
                     "matched_isi": round(matched_isi, 3) if np.isfinite(matched_isi) else np.nan,
                     "isi_pctile": round(pctile, 3) if np.isfinite(pctile) else np.nan, "out": out.name})

    pd.DataFrame(done).to_csv(out_dir / "candidate_stats.csv", index=False)
    print(f"\nrendered {len(done)} DANT candidate figures + candidate_stats.csv "
          f"(non-matched null n={len(nonmatched)})")


def _auc_matched_vs_null(matched, nonmatched):
    matched = np.asarray([m for m in matched if np.isfinite(m)], float)
    nonmatched = np.asarray(nonmatched, float)
    if len(matched) == 0 or len(nonmatched) == 0:
        return float("nan")
    allv = np.concatenate([matched, nonmatched])
    ranks = pd.Series(allv).rank().to_numpy()
    u = ranks[:len(matched)].sum() - len(matched) * (len(matched) + 1) / 2
    return float(u / (len(matched) * len(nonmatched)))


if __name__ == "__main__":
    main()
