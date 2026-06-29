"""N1 / C1 real-data existence gate — does a striatal urgency ramp predict
WHEN the mouse will respond?  (Task 6 of the N1 plan.)

Plain-language summary
----------------------
Each expert session is decoded ON ITS OWN (the recorded units are NOT tracked
across days, so a unit's column number means a different neuron in every
session).  For a swept set of post-Baseline_ON time windows we ask, per session:
"from the population activity in this window, can a cross-validated linear
read-out predict the trial's response time (the lick, relative to Baseline_ON)?"
We aggregate the per-session prediction r OVER SESSIONS (session = the unit of
replication) and compare it to a within-session-shuffle null.  That is **prong 1**
(the urgency ramp carries timing information).

**Prong 2** asks whether that timing signal is more than just "the animal is
about to lick": we remove (project out) a fresh per-session PREPARATORY-MOTOR
axis — an LDA direction fit on a separate LICK-aligned tensor (the 200 ms-
corrected lick) — and re-decode.  Real urgency is only PARTLY aligned with the
motor axis, so we expect the decode to drop SOMEWHAT; the gate's prong-2
criterion is that the projected decode STILL BEATS the null (not that r is
unchanged).

Window selection (movement-free):  among the swept windows we report as PRIMARY
the LATEST window whose per-session motor-axis signal is NOT significantly above
its own pre-trial baseline (i.e. the population is not yet visibly committing to
a lick) — the most stringent test of a *preparatory* (not peri-movement) ramp.

Outputs (all under the canonical cache / FIGURES trees):
  * data/cache/neural_latents/n1_c1_gate.json
  * FIGURES/neural_latents/BG_046/fig_n1_c1_gate.png
  * data/cache/neural_latents/n1_c1_gate_stats.csv

Pure core: ``evaluate_window(sessions, motor_axes, *, n_null, seed)``.
Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/n1_c1_gate.py
"""
import os
import gc
import json

import numpy as np

from visdetect.analysis import neural_latents as nl

# ── Pure core ───────────────────────────────────────────────────────────────
def evaluate_window(sessions, motor_axes, *, n_null=200, seed=42, n_workers=1):
    """C1 gate statistics for ONE swept window.

    Parameters
    ----------
    sessions : list of (sess_id, X, y, trial_type)
        Per-session windowed feature matrix ``X`` (trials x that session's units,
        already z-scored in ``join_session``), target ``y`` = decision_time on
        lick trials (hit+fa), and per-trial ``trial_type`` (outcome string).
    motor_axes : dict {sess_id -> axis}
        One unit-norm preparatory-motor axis per session, living in the SAME
        unit space (same units, same order) as that session's ``X``.

    Returns
    -------
    dict with:
        mean_r, median_r, ci, null_mean, null_sd, within_type, per_session
            — the prong-1 per-session cohort decode + within-session-shuffle null.
        mean_r_after_projection
            — the same mean-over-sessions decode after projecting out each
              session's motor axis (prong 2; reuses the SAME base null).
        survives_projection
            — bool: ``mean_r_after_projection > null_mean + 2*null_sd``.

    Per-session decode + within-session-shuffle null (prong 1) + per-session
    motor-CD projection survival (prong 2).  NO cross-session pooling, NO global
    Spearman (Simpson's-paradox inflation from between-session offsets).
    """
    base = nl.decode_cohort(sessions, n_null=n_null, seed=seed, n_workers=n_workers)
    proj = [(sid, nl.project_out_axis(X, motor_axes[sid]), y, tt)
            for sid, X, y, tt in sessions]
    proj_mean = float(np.nanmean(nl._per_session_rs(proj, seed)))   # reuse; no extra null
    survives = proj_mean > base["null_mean"] + 2 * base["null_sd"]
    return {"mean_r": base["mean_r"], "median_r": base["median_r"], "ci": base["ci"],
            "null_mean": base["null_mean"], "null_sd": base["null_sd"],
            "mean_r_after_projection": proj_mean, "survives_projection": bool(survives),
            "within_type": base["within_type"], "per_session": base["per_session"]}


# ── Real-data wiring (main) ──────────────────────────────────────────────────
# Imports below are deferred to main() so the pure core (and the test) load
# WITHOUT pulling in matplotlib / the suite loader / heavy session I/O.

_JOIN_WINDOW = (-1.3, 6.0)          # Baseline_ON-aligned span covering all WINDOWS
_BASELINE_WINDOW = (-1.3, -0.3)     # shared pre-Baseline_ON baseline (join_session default)
_LICK_SPAN = (-2.0, 0.75)           # lick-aligned tensor span for the motor CD
_LICK_BASELINE = (-1.75, -1.25)     # pre-lick baseline for the lick-aligned z-score (_FA_BASE)
_LICK_TRIALS = ("hit", "fa")        # licks come from hit (Hit) + fa (FA) trials


def _build_motor_axis(session, good_ids):
    """Fresh per-session preparatory-motor axis in the unit space `good_ids`.

    Builds a LICK-aligned tensor by stacking FA- and Hit-aligned trials (both
    aligned to the 200 ms-corrected lick inside ``build_population_tensor`` via
    ``get_event_times_by_trial`` / ``compute_true_reaction_time``), restricts FA
    to LATE licks (>= FA_RT_SPLIT s after Baseline_ON) so the pre-lick baseline
    window does not run before Baseline_ON, per-unit z-scores against the pre-lick
    baseline, then fits the LDA preparatory-motor direction.  Returns
    (axis, n_lick_trials) or (None, 0) if too few lick trials.
    """
    from visdetect.analysis.utils import build_population_tensor, compute_zscore_normalized
    from visdetect.analysis.align import get_event_times_by_trial
    from visdetect.analysis.constants import DEFAULT_BIN_SIZE, FA_RT_SPLIT

    base_on = np.asarray(session.ni_events.get("Baseline_ON", []), float).ravel()
    blocks, bc_ref = [], None
    for ev in ("FA", "Hit"):
        # late-lick filter (FA only): keep trials whose corrected lick is >= 3 s
        # after Baseline_ON (a clean pre-lick baseline; matches lick.py min_fa_delay).
        trial_idx = None
        if ev == "FA":
            ev_times = np.asarray(get_event_times_by_trial(session, "FA"), float)
            trial_idx = [i for i, t in enumerate(ev_times)
                         if i < len(base_on) and np.isfinite(t)
                         and (t - base_on[i]) >= FA_RT_SPLIT]
            if not trial_idx:
                continue
        try:
            tensor, bc, _ = build_population_tensor(
                session, cluster_ids=good_ids, event_name=ev,
                window=_LICK_SPAN, bin_size=DEFAULT_BIN_SIZE, trial_indices=trial_idx)
        except ValueError:
            continue                              # no valid trials for this event
        bc_ref = bc if bc_ref is None else bc_ref
        blocks.append(tensor)
    if not blocks:
        return None, 0
    z_lick = compute_zscore_normalized(np.concatenate(blocks, axis=0), bc_ref, _LICK_BASELINE)
    if z_lick.shape[0] < 3:                       # too few lick trials to fit an LDA
        return None, int(z_lick.shape[0])
    axis = nl.fit_lick_motor_cd(z_lick, bc_ref)
    return axis, int(z_lick.shape[0])


def _movement_free_check(X_win, X_pre, axis):
    """Is the motor-axis signal in the window NOT significantly above its
    pre-trial baseline?  Paired (per-trial) Wilcoxon of motor_axis_signal(X_win)
    vs motor_axis_signal(X_pre).  Returns (p_value, delta) where delta = mean
    window projection minus mean pre-trial projection.  Movement-free <=> the
    window is NOT significantly *greater* (one-sided)."""
    from scipy.stats import wilcoxon
    sig_win = nl.motor_axis_signal(X_win, axis)
    sig_pre = nl.motor_axis_signal(X_pre, axis)
    delta = float(np.mean(sig_win) - np.mean(sig_pre))
    diff = sig_win - sig_pre
    if np.allclose(diff, 0.0) or len(diff) < 6:
        return 1.0, delta
    try:
        p = float(wilcoxon(sig_win, sig_pre, alternative="greater").pvalue)
    except ValueError:
        p = 1.0
    return p, delta


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from visdetect.suite.loader import load_session
    from visdetect.analysis.config import ROOT, SUBJECT, OUTCOME_COLORS

    CACHE_DIR = os.path.join(ROOT, "data", "cache", "neural_latents")
    FIG_DIR = os.path.join(ROOT, "FIGURES", "neural_latents", SUBJECT)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    n_workers = max(1, (os.cpu_count() or 2) - 2)
    df = nl.load_latent_table()
    sess_ids = nl.fitted_expert_sessions(df)
    print(f"[n1_c1_gate] {len(sess_ids)} fitted-expert sessions; "
          f"null n_workers={n_workers}", flush=True)

    win_names = list(nl.WINDOWS)
    # per-window cohort: list of (sid, X, y, tt); per-window motor axes {sid: axis}
    cohort = {w: [] for w in win_names}
    motor_axes = {w: {} for w in win_names}
    # movement-free bookkeeping: per window, per session (p_value, delta)
    mf = {w: [] for w in win_names}
    join_failures = []
    contributing = set()

    for sid in sess_ids:
        try:
            s = load_session(int(sid))
        except Exception as e:                    # noqa: BLE001
            join_failures.append((sid, f"load_session: {e}"))
            continue
        try:
            try:
                jr = nl.join_session(s, df[df.sess_canon == sid], window=_JOIN_WINDOW)
            except (ValueError, AssertionError) as e:
                join_failures.append((sid, f"join_session: {e}"))
                continue

            mask = jr.y["outcome"].isin(_LICK_TRIALS).to_numpy()
            if mask.sum() < 8:                    # too few lick trials to decode
                join_failures.append((sid, f"only {int(mask.sum())} lick trials"))
                continue
            y = jr.y.loc[mask, "decision_time"].to_numpy(float)
            tt = jr.y.loc[mask, "outcome"].to_numpy()
            z_lick_trials = jr.z[mask]

            # fresh per-session motor axis in jr.unit_ids order (== window-feature units)
            axis, n_lick = _build_motor_axis(s, jr.unit_ids)
            if axis is None or axis.shape[0] != z_lick_trials.shape[2]:
                join_failures.append(
                    (sid, f"motor axis unavailable/mismatched (n_lick={n_lick})"))
                continue

            # pre-trial baseline features (same z tensor, baseline window) for the
            # movement-free check — one row per lick trial.
            X_pre = nl.window_feature_matrix(z_lick_trials, jr.bin_centers, _BASELINE_WINDOW)
            for w in win_names:
                X = nl.window_feature_matrix(z_lick_trials, jr.bin_centers, nl.WINDOWS[w])
                cohort[w].append((sid, X, y, tt))
                motor_axes[w][sid] = axis
                p, delta = _movement_free_check(X, X_pre, axis)
                mf[w].append({"sess_id": sid, "p": p, "delta": delta})
            contributing.add(sid)
            print(f"  [{sid}] {mask.sum()} lick trials, {len(jr.unit_ids)} units, "
                  f"motor axis from {n_lick} licks", flush=True)
        finally:
            del s
            gc.collect()

    if not contributing:
        raise RuntimeError("n1_c1_gate: NO sessions contributed — every join failed. "
                           f"failures: {join_failures}")

    # ── per-window gate statistics ───────────────────────────────────────────
    results = {}
    for w in win_names:
        res = evaluate_window(cohort[w], motor_axes[w], n_null=200, seed=42,
                              n_workers=n_workers)
        # movement-free: fraction of sessions where motor signal is significantly
        # above its pre-trial baseline (one-sided, alpha=0.05).
        ps = np.array([m["p"] for m in mf[w]])
        frac_sig = float(np.mean(ps < 0.05)) if len(ps) else 1.0
        median_delta = float(np.median([m["delta"] for m in mf[w]])) if mf[w] else np.nan
        movement_free = frac_sig < 0.5            # majority of sessions NOT committing
        results[w] = {
            "window": list(nl.WINDOWS[w]),
            "n_sessions": len(cohort[w]),
            "mean_r": res["mean_r"], "median_r": res["median_r"], "ci": list(res["ci"]),
            "null_mean": res["null_mean"], "null_sd": res["null_sd"],
            "null_upper": res["null_mean"] + 2 * res["null_sd"],
            "beats_null": bool(res["mean_r"] > res["null_mean"] + 2 * res["null_sd"]),
            "mean_r_after_projection": res["mean_r_after_projection"],
            "survives_projection": res["survives_projection"],
            "within_type": res["within_type"],
            "motor_frac_sessions_significant": frac_sig,
            "motor_median_delta": median_delta,
            "movement_free": bool(movement_free),
        }
        print(f"[{w}] {nl.WINDOWS[w]} n={len(cohort[w])} | mean_r={res['mean_r']:.3f} "
              f"(null {res['null_mean']:.3f}+2*{res['null_sd']:.3f}="
              f"{res['null_mean']+2*res['null_sd']:.3f}) beats={results[w]['beats_null']} "
              f"| after-proj r={res['mean_r_after_projection']:.3f} "
              f"survives={res['survives_projection']} | within-hit="
              f"{res['within_type'].get('hit', float('nan')):.3f} "
              f"| movement_free={movement_free} (frac sig {frac_sig:.2f})", flush=True)

    # ── PRIMARY = latest movement-free window (fallback: latest of all) ───────
    mf_windows = [w for w in win_names if results[w]["movement_free"]]
    primary = mf_windows[-1] if mf_windows else win_names[-1]
    primary_fallback = not mf_windows
    print(f"[PRIMARY] {primary} {nl.WINDOWS[primary]} "
          f"{'(fallback: no movement-free window)' if primary_fallback else ''}", flush=True)

    gate_pass = (results[primary]["beats_null"] and results[primary]["survives_projection"])

    out = {
        "subject": SUBJECT,
        "n_fitted_expert_sessions": len(sess_ids),
        "n_contributing_sessions": len(contributing),
        "contributing_sessions": sorted(contributing, key=lambda x: list(sess_ids).index(x)),
        "join_failures": [{"sess_id": s, "reason": r} for s, r in join_failures],
        "windows": results,
        "primary_window": primary,
        "primary_is_fallback": primary_fallback,
        "gate_pass": bool(gate_pass),
    }
    json_path = os.path.join(CACHE_DIR, "n1_c1_gate.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[n1_c1_gate] wrote {json_path}", flush=True)

    # ── stats CSV ────────────────────────────────────────────────────────────
    rows = []
    for w in win_names:
        r = results[w]
        rows.append({
            "window": w, "lo": r["window"][0], "hi": r["window"][1],
            "n_sessions": r["n_sessions"], "mean_r": r["mean_r"],
            "ci_lo": r["ci"][0], "ci_hi": r["ci"][1],
            "null_mean": r["null_mean"], "null_sd": r["null_sd"],
            "null_upper": r["null_upper"], "beats_null": r["beats_null"],
            "mean_r_after_projection": r["mean_r_after_projection"],
            "survives_projection": r["survives_projection"],
            "within_hit": r["within_type"].get("hit", np.nan),
            "within_fa": r["within_type"].get("fa", np.nan),
            "motor_frac_sig": r["motor_frac_sessions_significant"],
            "motor_median_delta": r["motor_median_delta"],
            "movement_free": r["movement_free"], "is_primary": (w == primary),
        })
    csv_path = os.path.join(CACHE_DIR, "n1_c1_gate_stats.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[n1_c1_gate] wrote {csv_path}", flush=True)

    # ── figure ───────────────────────────────────────────────────────────────
    _make_figure(win_names, results, primary, primary_fallback, gate_pass,
                 len(contributing), len(sess_ids), OUTCOME_COLORS, FIG_DIR)

    verdict = "PASSES" if gate_pass else "DOES NOT PASS"
    print(f"\n[n1_c1_gate] C1 gate {verdict} on PRIMARY window '{primary}' "
          f"{nl.WINDOWS[primary]}: mean_r={results[primary]['mean_r']:.3f} "
          f"vs null_upper={results[primary]['null_upper']:.3f}; "
          f"after-projection r={results[primary]['mean_r_after_projection']:.3f} "
          f"(survives={results[primary]['survives_projection']}); "
          f"within-hit={results[primary]['within_type'].get('hit', float('nan')):.3f}; "
          f"{len(contributing)}/{len(sess_ids)} sessions contributed.", flush=True)
    return out


def _make_figure(win_names, results, primary, primary_fallback, gate_pass,
                 n_contrib, n_total, outcome_colors, fig_dir):
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(win_names))
    labels = [f"{w}\n{tuple(results[w]['window'])} s" for w in win_names]

    # ── Left: per-window decode r (raw + after motor-CD projection) vs null band
    mean_r = [results[w]["mean_r"] for w in win_names]
    ci_lo = [results[w]["mean_r"] - results[w]["ci"][0] for w in win_names]
    ci_hi = [results[w]["ci"][1] - results[w]["mean_r"] for w in win_names]
    proj_r = [results[w]["mean_r_after_projection"] for w in win_names]
    null_up = [results[w]["null_upper"] for w in win_names]

    axL.errorbar(x - 0.07, mean_r, yerr=[np.abs(ci_lo), np.abs(ci_hi)], fmt="o",
                 ms=9, capsize=4, color="#1f4e79", lw=2, label="decode r (±95% CI over sessions)")
    axL.scatter(x + 0.07, proj_r, marker="D", s=70, color="#c0392b", zorder=5,
                label="after removing motor axis")
    # null band: chance = null_mean, significance line = null_mean + 2 SD
    for xi, w in zip(x, win_names):
        nm, nsd = results[w]["null_mean"], results[w]["null_sd"]
        axL.add_patch(plt.Rectangle((xi - 0.32, nm - 2 * nsd), 0.64, 4 * nsd,
                                    color="0.8", zorder=0,
                                    label="shuffle null (mean ±2 SD)" if xi == 0 else None))
        axL.hlines(nm, xi - 0.32, xi + 0.32, color="0.45", lw=1, zorder=1)
    axL.plot(x, null_up, "k--", lw=1, alpha=0.6, label="null + 2 SD (significance line)")
    axL.set_xticks(x); axL.set_xticklabels(labels)
    axL.set_ylabel("response-time decoding accuracy\n(Spearman r, cross-validated)")
    axL.set_xlabel("time window after baseline onset")
    axL.axhline(0, color="0.6", lw=0.8)
    # mark PRIMARY
    pi = win_names.index(primary)
    axL.axvspan(pi - 0.42, pi + 0.42, color="#fff2cc", zorder=-1)
    axL.annotate("PRIMARY\n(movement-free)" if not primary_fallback else "PRIMARY\n(fallback)",
                 xy=(pi, axL.get_ylim()[1]), xytext=(pi, axL.get_ylim()[1]),
                 ha="center", va="top", fontsize=9, color="#7f6000", fontweight="bold")
    axL.legend(loc="upper left", fontsize=8, framealpha=0.9)
    axL.set_title("Can striatal activity predict WHEN the mouse responds?", fontsize=11)

    # ── Right: within-hit graded decode per window
    wh = [results[w]["within_type"].get("hit", np.nan) for w in win_names]
    wf = [results[w]["within_type"].get("fa", np.nan) for w in win_names]
    axR.bar(x - 0.18, wh, width=0.36, color=outcome_colors.get("Hit", "#4CAF50"),
            label="within HIT trials")
    axR.bar(x + 0.18, wf, width=0.36, color=outcome_colors.get("FA", "#FF9800"),
            label="within FA trials")
    axR.axhline(0, color="0.6", lw=0.8)
    axR.set_xticks(x); axR.set_xticklabels(labels)
    axR.set_ylabel("graded prediction WITHIN a trial type\n(Spearman r)")
    axR.set_xlabel("time window after baseline onset")
    axR.legend(loc="upper left", fontsize=8)
    axR.set_title("Is the timing read-out graded within outcome (NOTE B)?", fontsize=11)

    verdict = "PASSES" if gate_pass else "does NOT pass"
    fig.suptitle(
        f"C1 existence gate: striatal urgency ramp -> response timing  "
        f"[gate {verdict} on '{primary}']\n"
        f"{n_contrib}/{n_total} expert sessions; decoded within-session, "
        f"aggregated over sessions; null = within-session response-time shuffle",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "fig_n1_c1_gate.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[n1_c1_gate] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
