"""Registration TRUST check for one subject.

A single whole-probe "~0 drift" number from an amplitude-fingerprint correlation is
not self-evidently trustworthy (the method is brittle and confounds drift with yield
change). This script produces four independent trust signals so the result can be
judged:

  1. per-SHANK shifts   -- does the whole-probe estimate hide per-shank drift?
  2. per-session CONFIDENCE (consecutive-pair corr) -- which sessions are weak?
  3. fingerprint OVERLAYS -- do the raw landscapes visually coincide across months?
  4. cross-check vs the independent diagnose_intersession_drift result.

Local only; run with the data junctions live. Output:
  FIGURES/population_field/<SUBJ>/registration_qc.png  (+ a printed summary).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis import population_field as pf
from visdetect.analysis.config import canonical_session_id, session_date_key, ROOT
from visdetect.analysis.tracking_qc import load_channel_positions, load_raw_mean_waveform
from visdetect.anatomy.channel_geometry import chanmap_signature, assign_shanks

SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "BG_046"
CORR_GATE = 0.6
root = os.path.join(ROOT, "data", "unit_match", "input", SUBJECT)


def good_stable_ids(session):
    from visdetect.core.session import load_session
    path = os.path.join(ROOT, "data", "pkls", SUBJECT, f"{SUBJECT}_{session}.pkl")
    if not os.path.exists(path):
        return None
    sess = load_session(path)
    ids = sorted(int(x) for x in (sess.good_and_stable_ids or []))
    del sess
    return ids


def whole_and_pershank_fp(session, unit_ids, y_edges, shank_of_chan):
    """Return (whole_smoothed, {shank: smoothed}) amplitude-depth fingerprints."""
    pos = load_channel_positions(root, session)
    n_bins = len(y_edges) - 1
    chan_bin = np.clip(np.searchsorted(y_edges, pos[:, 1]) - 1, 0, n_bins - 1)
    whole = np.zeros(n_bins)
    shanks = sorted(set(int(s) for s in shank_of_chan))
    per = {sh: np.zeros(n_bins) for sh in shanks}
    for uid in unit_ids:
        mw = load_raw_mean_waveform(root, session, uid)
        if mw is None:
            continue
        ptp = mw.max(axis=0) - mw.min(axis=0)
        np.add.at(whole, chan_bin, ptp)
        for sh in shanks:
            m = shank_of_chan == sh
            np.add.at(per[sh], chan_bin[m], ptp[m])
    sm = pf.smooth_fingerprint
    return sm(whole, pf.REG_SMOOTH_BINS), {sh: sm(per[sh], pf.REG_SMOOTH_BINS) for sh in shanks}


def main():
    # kept sessions: dominant chanmap signature, pkl present, chronological
    sessions = [canonical_session_id(d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]
    sig = {}
    for s in sessions:
        pos = load_channel_positions(root, s)
        if pos is not None:
            sig[s] = chanmap_signature(pos)
    chosen, kept = pf.select_dominant_signature(sig)
    kept = sorted(kept, key=session_date_key)
    kept = [s for s in kept if good_stable_ids(s) is not None]

    ref_pos = load_channel_positions(root, kept[0])
    y_edges = pf.registration_y_edges(ref_pos)
    centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    shank_of_chan = assign_shanks(ref_pos)
    shanks = sorted(set(int(s) for s in shank_of_chan))

    whole_fps, shank_fps = {}, {sh: {} for sh in shanks}
    nunits = {}
    for s in kept:
        ids = good_stable_ids(s)
        nunits[s] = len(ids)
        w, per = whole_and_pershank_fp(s, ids, y_edges, shank_of_chan)
        whole_fps[s] = w
        for sh in shanks:
            shank_fps[sh][s] = per[sh]

    # register (consecutive chaining, anchored at latest) -- re-anchor to session 0
    def rereferenced(shift_dict):
        s0 = shift_dict[kept[0]][0]
        return {s: (shift_dict[s][0] - s0, shift_dict[s][1]) for s in kept}

    whole_reg = rereferenced(pf.session_shift_um_chained(whole_fps, kept))
    shank_reg = {sh: rereferenced(pf.session_shift_um_chained(shank_fps[sh], kept)) for sh in shanks}

    # cross-check vs diagnose
    diag_csv = os.path.join(ROOT, "FIGURES", "tracking_qc", "intersession_drift.csv")
    diag = pd.read_csv(diag_csv, dtype={"session": str}) if os.path.exists(diag_csv) else None

    x = np.arange(len(kept))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{SUBJECT} — registration trust check (whole-probe result: "
                 f"max |shift| = {max(abs(whole_reg[s][0]) for s in kept):.0f} um)", fontsize=13)

    # A: fingerprint overlays
    ax = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(kept):
        ax.plot(centres, whole_fps[s] / (whole_fps[s].max() or 1), color=cmap(i / max(1, len(kept) - 1)), lw=0.8, alpha=0.7)
    ax.set(title="A. Whole-probe fingerprints overlaid (norm.)  — early=purple, late=yellow",
           xlabel="probe depth (um)", ylabel="amplitude density (norm.)")

    # B: per-session shifts (whole + per-shank)
    ax = axes[0, 1]
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.plot(x, [whole_reg[s][0] for s in kept], "k-o", ms=3, lw=2, label="whole-probe")
    for sh in shanks:
        ax.plot(x, [shank_reg[sh][s][0] for s in kept], "-", lw=1, alpha=0.8, label=f"shank {sh}")
    ax.set(title="B. Estimated shift per session (re-anchored to first session)",
           xlabel="session (chronological)", ylabel="shift (um)")
    ax.legend(fontsize=8, ncol=3)

    # C: confidence
    ax = axes[1, 0]
    ax.axhline(CORR_GATE, color="r", lw=0.8, ls="--", label=f"gate {CORR_GATE}")
    ax.plot(x, [whole_reg[s][1] for s in kept], "k-o", ms=3, lw=2, label="whole-probe corr")
    ax.plot(x, [min(shank_reg[sh][s][1] for sh in shanks) for s in kept], "-", color="tab:orange", lw=1.2, label="min per-shank corr")
    ax.set(title="C. Consecutive-pair confidence (low = untrustworthy)",
           xlabel="session (chronological)", ylabel="correlation", ylim=(0, 1.02))
    ax.legend(fontsize=8)

    # D: cross-check vs diagnose
    ax = axes[1, 1]
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.plot(x, [whole_reg[s][0] for s in kept], "k-o", ms=3, lw=2, label="ours (whole-probe)")
    if diag is not None:
        dmap = {canonical_session_id(r): float(c) for r, c in zip(diag["session"], diag["cum_um"])}
        ax.plot(x, [dmap.get(s, np.nan) for s in kept], "s--", color="tab:red", ms=3, lw=1.2, label="diagnose cum_um")
    ax.set(title="D. Cross-check vs independent diagnose_intersession_drift",
           xlabel="session (chronological)", ylabel="cumulative shift (um)")
    ax.legend(fontsize=8)

    out_dir = os.path.join(ROOT, "FIGURES", "population_field", SUBJECT)
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figpath = os.path.join(out_dir, "registration_qc.png")
    fig.savefig(figpath, dpi=130)
    print("figure:", figpath)

    # printed summary
    print(f"\n{SUBJECT}: {len(kept)} sessions, signature {chosen}, shanks {shanks}")
    print(f"whole-probe max |shift| = {max(abs(whole_reg[s][0]) for s in kept):.0f} um")
    for sh in shanks:
        print(f"  shank {sh}: max |shift| = {max(abs(shank_reg[sh][s][0]) for s in kept):.0f} um")
    flagged = [s for s in kept if whole_reg[s][1] < CORR_GATE or min(shank_reg[sh][s][1] for sh in shanks) < CORR_GATE]
    print(f"sessions below corr gate {CORR_GATE}: {len(flagged)} -> {flagged}")
    if diag is not None:
        dmap = {canonical_session_id(r): float(c) for r, c in zip(diag["session"], diag["cum_um"])}
        agree = [abs(whole_reg[s][0] - dmap.get(s, np.nan)) for s in kept if s in dmap]
        print(f"agreement with diagnose cum_um: max |diff| = {np.nanmax(agree):.0f} um over {len(agree)} shared sessions")
        # per-shank max jump reported by diagnose (independent)
        for sh in shanks:
            col = f"step_shank{sh}_um"
            if col in diag.columns:
                print(f"  diagnose {col}: max |step| = {diag[col].abs().max():.0f} um")


if __name__ == "__main__":
    main()
