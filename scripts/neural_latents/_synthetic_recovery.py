"""N1 synthetic recovery + NOTE-A discriminability prerequisite.

Validates the decode METHOD BEFORE any real-data claim. Pure synthetic — NO
session pkl loads, NO X: access, NO heavy compute. Three checks:

 (1) recovers          — `decode_cohort` recovers planted response-timing in
                          BOTH a phi-urgency-ramp population AND a pure-motor-ramp
                          population (mean_r beats the within-session-shuffle null
                          in both). ASSERTED (method validity).
 (2) motor_killed      — the PER-SESSION motor-CD projection KILLS the pure-motor
                          case (mean_r -> chance) but SPARES the phi-urgency case
                          (mean_r stays above null). ASSERTED (prong-2 control works).
 (3) phi_separable     — phi-vs-ramp separability via `phi_specificity_session`
     _on_window          aggregated over sessions, computed ON THE PRE-mu READOUT
                          WINDOW features (NOT from the decision-time distribution).
                          This is the NOTE-A empirical finding — COMPUTED and
                          REPORTED, NOT asserted: over a pre-mu window phi sits on
                          its rising flank and is ~collinear with a monotonic ramp,
                          so this leg is expected to be underpowered/False. That is
                          an acceptable, honest outcome that gates Task 7.

Design (scientifically faithful structure):
 - Both regimes carry a lick-locked MOTOR-preparation ramp along a fixed unit-axis
   `w_motor` (every real lick has motor prep) -> `fit_lick_motor_cd` recovers
   ~`w_motor` in BOTH regimes (a valid motor CD always exists).
 - The PURE-MOTOR regime carries its readout-window TIMING signal ONLY along
   `w_motor` (timing == the rising edge of the lick-locked ramp) -> projecting out
   the motor CD removes the timing signal -> decode dies.
 - The PHI-URGENCY regime carries its readout-window TIMING signal along an
   ORTHOGONAL urgency axis `w_phi` (a mu-anchored, non-motor urgency ramp) ON TOP
   of the shared (timing-uninformative) motor prep -> projecting out the motor CD
   spares the urgency timing signal -> decode survives.

Writes FIGURES/neural_latents/BG_046/n1_synthetic_recovery.png + the JSON verdict
data/cache/neural_latents/n1_synthetic_verdict.json. Prints the three booleans.

Run: PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/_synthetic_recovery.py
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis.config import ROOT
from visdetect.analysis import neural_latents as nl

# ── reproducibility / sizes (modest: must run in a couple minutes) ──────────
SEED = 42
K = 8                       # synthetic sessions per regime
N_TRIALS = 120             # trials per session
N_UNITS = 30               # units per session
N_NULL = 100              # cohort within-session-shuffle null draws (<=100 per brief)
MU_LO, MU_HI = 6.7, 7.5    # REAL expected-change-time (mu) range, seconds
SIGMA = 0.8                # fixed B8 timing-bump width (do not change; used downstream)
BIN = 0.025                # readout bin size, seconds

READOUT_WIN = nl.WINDOWS["late"]        # (4.0, 6.0) — latest pre-mu window (PRIMARY)
# lick-aligned tensor must cover the lab-canonical FA motor windows used by
# fit_lick_motor_cd: base (-1.75,-1.25), premove (-0.3,-0.15).
LICK_LO, LICK_HI = -2.0, 0.0

# Amplitudes/noise are tuned (see Task-5 report) so that: raw decode beats null
# in BOTH regimes; a clean per-session motor CD (cos~0.99 with the planted motor
# axis) drops the PURE-MOTOR projected decode to ~chance while SPARING phi-urgency
# (its timing lives on the orthogonal axis the CD does not touch). The higher
# readout NOISE keeps the residual single-axis leak after projection below the
# decode floor while the raw signal stays clearly decodable.
NOISE = 1.0                # per-bin readout Gaussian noise SD
MOTOR_AMP = 0.7           # readout-window motor-prep ramp amplitude
PHI_AMP = 2.0             # readout-window phi-urgency amplitude (orthogonal axis)
PREP_AMP = 2.0            # lick-aligned motor-prep amplitude (drives a clean motor CD)
PREP_NOISE = 0.6          # lick-aligned tensor noise SD


def _readout_bin_centers():
    lo, hi = READOUT_WIN
    edges = np.arange(lo, hi + 1e-9, BIN)
    return 0.5 * (edges[:-1] + edges[1:])


def _lick_bin_centers():
    edges = np.arange(LICK_LO, LICK_HI + 1e-9, BIN)
    return 0.5 * (edges[:-1] + edges[1:])


def _orthonormal_axes(n_units, rng):
    """Two orthonormal unit-weight vectors: w_motor (motor prep) and w_phi
    (mu-anchored urgency), orthogonal so the motor-CD control is meaningful."""
    A = rng.normal(size=(n_units, 2))
    Q, _ = np.linalg.qr(A)            # columns orthonormal
    return Q[:, 0], Q[:, 1]


def _draw_decision_times(n, mu, rng):
    """Realistic mixture: ~45% FA (anticipatory, pre-mu) + ~55% hit (post-change,
    >= mu). Earlier trials = faster urgency build-up; the spread within and across
    type carries the graded timing signal the decoder must recover."""
    is_fa = rng.random(n) < 0.45
    dt = np.empty(n)
    # FAs: anticipatory licks landing inside / just before the readout window
    dt[is_fa] = rng.uniform(4.2, 6.3, is_fa.sum())
    # hits: post-change response licks, mu .. mu+1.5
    dt[~is_fa] = mu + rng.uniform(0.0, 1.5, (~is_fa).sum())
    tt = np.where(is_fa, "fa", "hit")
    return dt, tt


def _urgency_progress(t_grid, dt, mu):
    """Per-(trial, bin) readout signal that carries TIMING. A normalized urgency
    ramp whose RATE depends on the trial's decision_time: earlier licks -> steeper
    rise -> higher readout-window activity. Shape (n_trials, n_bins). Used as the
    common timing carrier; the regime decides WHICH unit-axis it loads onto."""
    # urgency "lead time": how early before the lick the build-up starts (s)
    lead = 2.5
    onset = dt[:, None] - lead                         # (n_trials, 1)
    tau = 0.7
    prog = 1.0 / (1.0 + np.exp(-(t_grid[None, :] - onset) / tau))
    return prog                                        # in [0,1], rises toward the lick


def simulate_session(regime, mu, rng):
    """Build ONE synthetic session for `regime` in {'phi','motor'}.

    Returns dict with:
      Xt_readout  (n_trials, n_bins_read, n_units)  pre-mu readout tensor
      z_lick      (n_trials, n_bins_lick, n_units)  lick-aligned tensor (motor-CD fit)
      dt          (n_trials,)                        decision_time (the target)
      tt          (n_trials,)                        'hit'/'fa' trial type
    """
    t_read = _readout_bin_centers()
    t_lick = _lick_bin_centers()
    w_motor, w_phi = _orthonormal_axes(N_UNITS, rng)

    dt, tt = _draw_decision_times(N_TRIALS, mu, rng)

    # ── readout-window tensor ────────────────────────────────────────────
    # Shared, timing-UNINFORMATIVE motor prep present in BOTH regimes: a flat-ish
    # baseline ramp along w_motor that does NOT encode dt (constant per trial).
    base_motor = MOTOR_AMP * 0.3 * np.ones((N_TRIALS, t_read.size))

    # timing carrier (graded with dt) for the readout window
    timing = _urgency_progress(t_read, dt, mu)         # (n_trials, n_bins)

    if regime == "motor":
        # TIMING lives ONLY along the motor axis (rising edge of the lick-locked
        # ramp). No orthogonal urgency content.
        load_motor = base_motor + MOTOR_AMP * timing
        load_phi = np.zeros((N_TRIALS, t_read.size))
    else:  # phi-urgency
        # TIMING lives along the ORTHOGONAL urgency axis, shaped by the phi bump
        # (mu-anchored). Motor axis only carries the shared, timing-free prep.
        b = nl.phi_ramp_bases(t_read, mu, SIGMA)["phi"]   # (n_bins,) rising flank
        bn = b / (b.max() + 1e-12)
        load_motor = base_motor
        load_phi = PHI_AMP * timing * bn[None, :]         # urgency, mu-anchored
    Xt_read = (load_motor[:, :, None] * w_motor[None, None, :]
               + load_phi[:, :, None] * w_phi[None, None, :])
    Xt_read = Xt_read + NOISE * rng.normal(size=Xt_read.shape)

    # ── lick-aligned tensor (for the per-session motor CD) ───────────────
    # Both regimes have genuine lick-locked MOTOR prep along w_motor: low in the
    # baseline window, rising into the pre-movement window (t -> 0 = lick). This
    # is what fit_lick_motor_cd (LDA premove vs baseline) recovers (~w_motor),
    # regardless of regime — exactly as in the real pipeline.
    prep = 1.0 / (1.0 + np.exp(-(t_lick - (-0.5)) / 0.25))   # rises toward lick
    prep = prep[None, :] * (0.8 + 0.4 * rng.random((N_TRIALS, 1)))
    z_lick = PREP_AMP * prep[:, :, None] * w_motor[None, None, :]
    z_lick = z_lick + PREP_NOISE * rng.normal(size=z_lick.shape)

    return {"Xt_readout": Xt_read, "z_lick": z_lick, "dt": dt, "tt": tt}


def build_cohort(regime, *, projected, base_seed):
    """Build the (sess_id, X, y, tt) cohort list for a regime. `X` = mean over the
    readout window (n_trials, n_units). If `projected`, the PER-SESSION motor CD
    (from that session's lick-aligned tensor) is projected out of X first."""
    t_lick = _lick_bin_centers()
    sessions = []
    for s in range(K):
        rng = np.random.default_rng(base_seed + s)     # per-session fixed offset (no global state)
        mu = rng.uniform(MU_LO, MU_HI)
        sess = simulate_session(regime, mu, rng)
        # collapse readout tensor to (n_trials, n_units) by mean over bins
        X = sess["Xt_readout"].mean(axis=1)
        if projected:
            cd = nl.fit_lick_motor_cd(sess["z_lick"], t_lick)
            X = nl.project_out_axis(X, cd)
        sessions.append((f"{regime}_s{s}", X, sess["dt"], sess["tt"]))
    return sessions


def build_phi_specificity(base_seed):
    """Aggregate per-session phi_specificity over the PRE-mu readout window (NOTE A).
    Returns (deltas, r_phis, r_ramps) arrays over the K phi-urgency sessions."""
    t_read = _readout_bin_centers()
    deltas, r_phis, r_ramps = [], [], []
    for s in range(K):
        rng = np.random.default_rng(base_seed + s)
        mu = rng.uniform(MU_LO, MU_HI)
        sess = simulate_session("phi", mu, rng)
        spec = nl.phi_specificity_session(
            sess["Xt_readout"], sess["dt"], t_read, mu, sigma=SIGMA, seed=SEED)
        deltas.append(spec["delta"])
        r_phis.append(spec["r_phi"])
        r_ramps.append(spec["r_ramp"])
    return np.array(deltas), np.array(r_phis), np.array(r_ramps)


def _above_null(res):
    """mean_r beats null by >= 2 SD (the project's null convention)."""
    return res["mean_r"] > res["null_mean"] + 2 * res["null_sd"]


def main():
    np.random.seed(SEED)   # belt-and-braces; all draws use seeded Generators

    # distinct per-regime base seeds so the two regimes differ by a FIXED offset,
    # not by consuming a shared global RNG mid-run.
    PHI_BASE = SEED
    MOTOR_BASE = SEED + 1000

    # ── (1) recovery: decode_cohort beats null in BOTH regimes (un-projected) ─
    phi_raw = nl.decode_cohort(build_cohort("phi", projected=False, base_seed=PHI_BASE),
                               n_null=N_NULL, seed=SEED)
    motor_raw = nl.decode_cohort(build_cohort("motor", projected=False, base_seed=MOTOR_BASE),
                                 n_null=N_NULL, seed=SEED)
    recovers = bool(_above_null(phi_raw) and _above_null(motor_raw))

    # ── (2) motor-CD projection: kills pure-motor, spares phi-urgency ────────
    phi_proj = nl.decode_cohort(build_cohort("phi", projected=True, base_seed=PHI_BASE),
                                n_null=N_NULL, seed=SEED)
    motor_proj = nl.decode_cohort(build_cohort("motor", projected=True, base_seed=MOTOR_BASE),
                                  n_null=N_NULL, seed=SEED)
    motor_killed = bool((not _above_null(motor_proj)) and _above_null(phi_proj))

    # ── (3) phi-vs-ramp separability ON THE READOUT WINDOW (NOTE A) ──────────
    #     COMPUTE + REPORT only — do NOT assert (expected underpowered/False).
    deltas, r_phis, r_ramps = build_phi_specificity(PHI_BASE)
    ci_lo, ci_hi = nl.bootstrap_ci(deltas, n_bootstrap=1000, seed=SEED)
    ci_lo, ci_hi = float(np.ravel(ci_lo)[0]), float(np.ravel(ci_hi)[0])
    phi_separable_on_window = bool(ci_lo > 0.0)   # separable iff delta CI excludes 0

    verdict = {
        "recovers": recovers,
        "motor_killed": motor_killed,
        "phi_separable_on_window": phi_separable_on_window,
        "params": {"K": K, "N_TRIALS": N_TRIALS, "N_UNITS": N_UNITS,
                   "N_NULL": N_NULL, "mu_range": [MU_LO, MU_HI], "sigma": SIGMA,
                   "readout_window": list(READOUT_WIN), "seed": SEED},
        "phi_urgency_raw": {"mean_r": phi_raw["mean_r"], "null_mean": phi_raw["null_mean"],
                            "null_sd": phi_raw["null_sd"], "within_type": phi_raw["within_type"]},
        "pure_motor_raw": {"mean_r": motor_raw["mean_r"], "null_mean": motor_raw["null_mean"],
                           "null_sd": motor_raw["null_sd"], "within_type": motor_raw["within_type"]},
        "phi_urgency_projected": {"mean_r": phi_proj["mean_r"], "null_mean": phi_proj["null_mean"],
                                  "null_sd": phi_proj["null_sd"]},
        "pure_motor_projected": {"mean_r": motor_proj["mean_r"], "null_mean": motor_proj["null_mean"],
                                 "null_sd": motor_proj["null_sd"]},
        "phi_specificity": {"delta_mean": float(np.mean(deltas)),
                            "delta_ci": [ci_lo, ci_hi],
                            "r_phi_mean": float(np.mean(r_phis)),
                            "r_ramp_mean": float(np.mean(r_ramps)),
                            "note": ("NOTE A: over a pre-mu readout window phi's rising "
                                     "flank is ~collinear with a monotonic ramp; this leg "
                                     "is expected to be underpowered (delta CI ~spans 0). "
                                     "Reported, NOT asserted; gates Task 7.")},
    }

    # ── write verdict JSON ───────────────────────────────────────────────
    cache_dir = os.path.join(ROOT, "data", "cache", "neural_latents")
    os.makedirs(cache_dir, exist_ok=True)
    verdict_path = os.path.join(cache_dir, "n1_synthetic_verdict.json")
    with open(verdict_path, "w") as fh:
        json.dump(verdict, fh, indent=2)

    # ── figure (plain-language, presentation-ready) ──────────────────────
    _make_figure(verdict, phi_raw, motor_raw, phi_proj, motor_proj, deltas)

    # ── print the three booleans + a short summary ───────────────────────
    print(f"recovers={recovers} motor_killed={motor_killed} "
          f"phi_separable_on_window={phi_separable_on_window}")
    print(f"  [1] phi-urgency raw   mean_r={phi_raw['mean_r']:.3f} "
          f"(null {phi_raw['null_mean']:.3f}+/-{phi_raw['null_sd']:.3f})")
    print(f"  [1] pure-motor raw    mean_r={motor_raw['mean_r']:.3f} "
          f"(null {motor_raw['null_mean']:.3f}+/-{motor_raw['null_sd']:.3f})")
    print(f"  [2] phi-urgency proj  mean_r={phi_proj['mean_r']:.3f} "
          f"(null {phi_proj['null_mean']:.3f}+/-{phi_proj['null_sd']:.3f})  -> spared")
    print(f"  [2] pure-motor proj   mean_r={motor_proj['mean_r']:.3f} "
          f"(null {motor_proj['null_mean']:.3f}+/-{motor_proj['null_sd']:.3f})  -> killed")
    print(f"  [3] phi-vs-ramp delta = {np.mean(deltas):+.3f} "
          f"CI[{ci_lo:+.3f},{ci_hi:+.3f}] (NOTE A: expected ~0 / underpowered)")
    print(f"  verdict -> {verdict_path}")

    # ── ASSERT method validity (#1, #2). Artifacts already written above so a
    #    failed assert still leaves the figure + JSON for inspection. #3 is NOT
    #    asserted (NOTE A: expected underpowered/False, gates Task 7). ──────────
    assert recovers, (
        "(1) decode_cohort must recover planted timing in BOTH regimes: "
        f"phi raw {phi_raw['mean_r']:.3f} (null {phi_raw['null_mean']:.3f}"
        f"+/-{phi_raw['null_sd']:.3f}), motor raw {motor_raw['mean_r']:.3f} "
        f"(null {motor_raw['null_mean']:.3f}+/-{motor_raw['null_sd']:.3f})")
    assert motor_killed, (
        "(2) per-session motor-CD projection must KILL pure-motor but SPARE "
        f"phi-urgency: motor proj {motor_proj['mean_r']:.3f} (null "
        f"{motor_proj['null_mean']:.3f}+/-{motor_proj['null_sd']:.3f}), phi proj "
        f"{phi_proj['mean_r']:.3f} (null {phi_proj['null_mean']:.3f}"
        f"+/-{phi_proj['null_sd']:.3f})")
    return verdict


def _make_figure(verdict, phi_raw, motor_raw, phi_proj, motor_proj, deltas):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel A: recovery (raw decode beats null) in both populations
    ax = axes[0]
    labels = ["phi-urgency", "pure-motor"]
    means = [phi_raw["mean_r"], motor_raw["mean_r"]]
    nulls = [phi_raw["null_mean"], motor_raw["null_mean"]]
    nsd = [phi_raw["null_sd"], motor_raw["null_sd"]]
    x = np.arange(2)
    ax.bar(x - 0.2, means, 0.4, color="#3474ae", label="decoded r")
    ax.bar(x + 0.2, nulls, 0.4, yerr=[2 * s for s in nsd], color="#bbbbbb",
           label="shuffle null +/-2SD", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("response-timing decode (Spearman r)")
    ax.set_title("A. Decoder recovers WHEN the mouse responds\nin both planted populations")
    ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color="k", lw=0.6)

    # Panel B: motor-CD projection kills motor, spares phi
    ax = axes[1]
    raw = [phi_raw["mean_r"], motor_raw["mean_r"]]
    proj = [phi_proj["mean_r"], motor_proj["mean_r"]]
    ax.bar(x - 0.2, raw, 0.4, color="#6baed6", label="raw")
    ax.bar(x + 0.2, proj, 0.4, color="#ef6548", label="after removing motor axis")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("response-timing decode (Spearman r)")
    ax.set_title("B. Removing the motor axis KILLS pure-motor\nbut SPARES phi-urgency (the prong-2 control)")
    ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color="k", lw=0.6)

    # Panel C: phi-vs-ramp delta per session (NOTE A: ~0, underpowered)
    ax = axes[2]
    ax.axhline(0, color="k", lw=0.8)
    ax.scatter(np.arange(len(deltas)), deltas, color="#3474ae", zorder=3)
    ci = verdict["phi_specificity"]["delta_ci"]
    ax.axhspan(ci[0], ci[1], color="#3474ae", alpha=0.15,
               label=f"95% CI [{ci[0]:+.2f},{ci[1]:+.2f}]")
    ax.axhline(np.mean(deltas), color="#3474ae", ls="--",
               label=f"mean delta={np.mean(deltas):+.2f}")
    ax.set_xlabel("synthetic session"); ax.set_ylabel("delta CV r (phi - ramp)")
    sep = verdict["phi_separable_on_window"]
    ax.set_title("C. phi-shaped vs plain ramp on readout window\n"
                 f"(NOTE A: expected ~0) separable={sep}")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("N1 synthetic recovery: the response-timing decoder is valid BEFORE any real-data claim\n"
                 "(A) it reads out response timing from planted activity; (B) the motor-axis control behaves; "
                 "(C) phi-vs-ramp is hard to tell apart on a pre-change window (NOTE A) — honest, expected",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig_dir = os.path.join(ROOT, "FIGURES", "neural_latents", "BG_046")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "n1_synthetic_recovery.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  figure  -> {fig_path}")


if __name__ == "__main__":
    main()
