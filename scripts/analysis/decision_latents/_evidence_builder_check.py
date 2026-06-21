"""B8 Phase-2 Task 0.1 — validate the corrected 60 Hz / collapse-runs-of-3 evidence builder.

Plain English: each trial stores a temporal-frequency (TF) trace for the baseline
grating. The OLD evidence builder (``ddm.build_trial_evidence``) guessed the TF
update period as ``change_time / len(baseline_values)`` and mis-sampled the trace.
The CORRECTED builder (``decision_latents.build_trial_evidence_corrected``) uses the
verified data facts: ``baseline_values`` is stored at 60 Hz, each TF value is held
for 3 frames (50 ms), so on the dt = 0.05 s grid bin ``k`` reads frame ``3k``.

This figure overlays, for a few real trials, the reconstructed evidence (one point
per 50 ms bin) on top of the raw 60 Hz baseline trace (downsampled to every 3rd
frame). They must lie on top of each other pre-change, and the reconstruction must
show the change-size step at the planned change time. Confirms the builder reads
TF at the true 50 ms cadence established by ``_tf_sampling_check.py``.

Run (from the worktree):
    PYTHONPATH="$(pwd)/src" py scripts/analysis/decision_latents/_evidence_builder_check.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style          # styling only (NOT save_figure)
from visdetect.analysis.config import ROOT, SUBJECT
from visdetect.analysis import decision_latents as dl

setup_style()

FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
os.makedirs(FIG_DIR, exist_ok=True)

MONITOR_HZ = 60.0
DT = 0.05
FRAMES_PER_BIN = int(round(DT * MONITOR_HZ))   # == 3
N_SESSIONS = 3
TRIALS_PER_SESSION = 3                          # go-trials shown per session


def save_fig(fig, name):
    """Local saver (deliberately NOT suite.plotting.save_figure): writes a
    presentation-ready PNG into FIGURES/decision_latents/<subject>/."""
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _pick_go_trials(df, n):
    """Pick up to n go-trials (change_size > 1) that actually reach the change
    within their decision window, preferring hits so the change-step is visible."""
    go = df[(df["change_size"] > 1.0) & (df["change_time"] < df["decision_time"])]
    hits = go[go["outcome"] == "hit"]
    chosen = hits if len(hits) >= n else go
    return chosen.head(n)


def main():
    sessions = dl.enumerate_valid_sessions(subject=SUBJECT)[:N_SESSIONS]
    panels = []   # (session_name, trial_row, bv)
    for sname in sessions:
        sess = load_session(sname)
        df = dl.build_trial_evidence_corrected(sess, dt=DT, tf_base=None)
        if df.empty:
            del sess
            continue
        picks = _pick_go_trials(df, TRIALS_PER_SESSION)
        for _, r in picks.iterrows():
            bv = np.asarray(sess.trials[int(r["trial_idx"])].baseline_values, float).ravel()
            panels.append((sname, r, bv))
        del sess   # free the session (sessions are large)

    if not panels:
        raise RuntimeError("No go-trials found in the first sessions — cannot validate.")

    n = len(panels)
    ncol = TRIALS_PER_SESSION
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow), squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for idx, (sname, r, bv) in enumerate(panels):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_visible(True)
        ev = np.asarray(r["evidence"], float)
        n_bins = int(r["n_bins"])
        t_bins = np.arange(n_bins) * DT                       # bin centers on the dt grid

        # Reconstructed evidence: one point per 50 ms bin (the canonical signal).
        ax.plot(t_bins, ev, color="#3474ae", lw=1.6, zorder=3,
                label="reconstructed evidence\n(50 ms bins, frame 3k)")

        # Raw 60 Hz baseline, collapsed to every 3rd frame, as log2(TF/base) — the
        # ground truth the builder must reproduce PRE-change (no change-size applied).
        base = float(np.nanmedian(bv)) or 1.0
        frames = np.arange(0, bv.size, FRAMES_PER_BIN)
        t_raw = frames / MONITOR_HZ
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_log2 = np.where(bv[frames] > 0, np.log2(bv[frames] / base), 0.0)
        m = t_raw <= t_bins[-1] + DT                          # clip to the shown window
        ax.plot(t_raw[m], raw_log2[m], color="#d9d9d9", lw=3.0, zorder=1,
                label="raw 60 Hz baseline\n(every 3rd frame, pre-change TF)")

        ct = float(r["change_time"])
        if np.isfinite(ct):
            ax.axvline(ct, color="#ef6548", ls="--", lw=1.2, zorder=2,
                       label="planned change")
        dec_t = float(r["decision_time"])
        ax.axvline(dec_t, color="#444444", ls=":", lw=1.0, zorder=2,
                   label="decision time")

        ax.set_title(f"{sname}  trial {int(r['trial_idx'])}  "
                     f"({r['outcome']}, ×{r['change_size']:.2f})", fontsize=9)
        ax.set_xlabel("time from baseline onset (s)")
        ax.set_ylabel("evidence  log2(TF / base)")
        if idx == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.suptitle("B8 Phase-2 — corrected evidence builder reads TF at the true 50 ms cadence",
                 fontsize=13, y=0.995)
    caption = (
        "Reconstructed per-trial evidence (blue) read at the verified 50 ms TF cadence "
        "(bin k reads 60 Hz baseline frame 3k) sits exactly on the raw baseline trace "
        "(grey, every 3rd frame) before the change, then steps up by the change size at "
        "the planned change time (orange). This replaces the old change_time/len(bv) "
        "mis-sampling and matches the cadence established by _tf_sampling_check.py."
    )
    fig.text(0.5, -0.02 / max(nrow, 1), caption, ha="center", va="top",
             fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out = save_fig(fig, "fig_b8_P2_evidence_builder_check")
    print(f"wrote {out}")
    print(f"sessions: {sessions}")
    print(f"panels (trials shown): {len(panels)}")
    # Spot-check: pre-change reconstructed evidence == raw every-3rd-frame log2 value.
    sname0, r0, bv0 = panels[0]
    base0 = float(np.nanmedian(bv0)) or 1.0
    ev0 = np.asarray(r0["evidence"], float)
    ct0 = float(r0["change_time"])
    k_pre = max(0, min(int(ct0 / DT) - 2, len(ev0) - 1))      # a bin safely pre-change
    j = k_pre * FRAMES_PER_BIN
    expect = np.log2(bv0[j] / base0) if bv0[j] > 0 else 0.0
    print(f"pre-change spot-check (trial {int(r0['trial_idx'])}, bin {k_pre}): "
          f"reconstructed={ev0[k_pre]:.4f}  expected log2(bv[{j}]/base)={expect:.4f}  "
          f"match={np.isclose(ev0[k_pre], expect)}")


if __name__ == "__main__":
    main()
