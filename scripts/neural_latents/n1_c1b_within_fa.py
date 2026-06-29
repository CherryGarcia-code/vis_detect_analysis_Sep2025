"""N1 / C1b — within-FA timing: leakage filter FIRST, then movement-MATCHING.
The DECISIVE prong-2 test.  (Task 6b of the N1 plan.)

Plain-language summary
----------------------
The C1 gate (Task 6) showed striatal pre-decision activity predicts WHEN the
mouse responds.  The real, interesting signal is SELF-TIMED licks: on false-alarm
(FA) trials the mouse licks early, of its own accord, and we can read off *when*
from the population ramp.  (On HIT trials the lick always comes >6 s in, after
every read-out window — within-hit prediction is ~0, a clean leakage-free
NEGATIVE control that validates the decode.)

But the raw within-FA prediction is PARTLY CIRCULAR: 15-28% of FA trials lick
INSIDE the read-out window, so the decoder can read the response time off
peri/post-lick activity rather than a genuine pre-decision ramp.  This script
runs the decisive cascade, in the MANDATORY order:

  (1) RAW within-FA r            — all FA trials (circular; the starting point).
  (2) LEAKAGE-FILTERED r         — keep only FA trials whose lick is comfortably
                                   AFTER the read-out window (>= window-end + 0.25 s);
                                   the decoder can no longer see the lick.  Expect a drop.
  (3) MATCHED (PRIMARY)          — on the leakage-filtered set, decode once per
                                   session, then hold the per-trial MOTOR-AXIS
                                   signal ~constant: within-strata Spearman AND the
                                   continuous rank-partial Spearman.  This asks: does
                                   timing prediction survive when "how much the
                                   animal is moving" is matched out?
  (4) SUBSPACE-PROJECTED (2ndary) — remove the top-k motor PCs from the features and
                                   re-decode (caveat: a high-dim projection can also
                                   remove genuine signal; secondary, not decisive).

We NEVER match on / partial out `decision_time` itself (it is the target); we
control only for the motor-axis projection magnitude.  We LEAD with the EARLY
window: it has the best raw signal, the fewest in-window licks, and early activity
predicting a lick seconds later is itself evidence AGAINST pure peri-movement
motor-prep (which peaks at the lick).  All three windows are reported.

Headline reframe (locked): "pre-change striatal ramp predicts self-timed
(anticipatory FA) lick timing."

Honest verdict (both ways):
  * survives leakage-filter + matching  -> "self-timed urgency predicts FA timing
    beyond generic motor prep."
  * collapses                           -> "for self-timed licks the
    urgency/commitment ramp and motor preparation are not separable" — a REAL,
    meaningful basal-ganglia action-commitment conclusion, NOT a control failure.

Outputs (canonical cache / FIGURES trees):
  * data/cache/neural_latents/n1_c1b_within_fa.json
  * FIGURES/neural_latents/BG_046/fig_n1_c1b_within_fa.png
  * data/cache/neural_latents/n1_c1b_within_fa_stats.csv

Pure core: ``evaluate_window_within_fa(sessions, motor_subspaces, *, n_null, seed,
n_workers)`` (FA-only cohort already leakage-filtered upstream by ``main``).
Run (CONTROLLER ONLY — slow):
  PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/n1_c1b_within_fa.py
"""
import os
import gc
import json

import numpy as np

from visdetect.analysis import neural_latents as nl

LEAKAGE_GUARD = 0.25     # s past the read-out window before a lick counts as clean
MOTOR_K = 5              # top-k motor PCs for the secondary subspace projection
MIN_CLEAN_FA = 8         # min leakage-free FA trials/session to decode that window


# ── Pure core ────────────────────────────────────────────────────────────────
def evaluate_window_within_fa(sessions, motor_signals, motor_subspaces,
                              *, n_null=200, seed=42, n_workers=1):
    """C1b cascade statistics for ONE swept window, FA trials only.

    Parameters
    ----------
    sessions : list of (sess_id, X, y, trial_type)
        Per-session LEAKAGE-FILTERED FA cohort: windowed feature matrix ``X``
        (clean-FA trials x that session's units, z-scored in ``join_session``),
        target ``y`` = decision_time, ``trial_type`` all == "fa".  (Whoever calls
        this has ALREADY applied the leakage filter — order is mandatory.)
    motor_signals : dict {sess_id -> np.ndarray}
        Per-trial motor-axis projection magnitude for each clean-FA trial in that
        session's ``X`` (the matching control).  Aligned row-for-row with ``X``/``y``.
    motor_subspaces : dict {sess_id -> (n_units, k) orthonormal basis}
        Per-session top-k motor subspace (for the secondary projection leg).

    Returns
    -------
    dict with the matched-primary aggregate + the secondary subspace aggregate:
        mean_r, median_r, ci, null_mean, null_sd, within_type, per_session
            — the leakage-filtered within-FA decode + within-session-shuffle null
              (this is the leakage-filtered r the cascade reports at stage 2).
        matched_strata_mean, matched_strata_ci
            — mean over sessions of within_strata_spearman(y_pred, y, motor_signal),
              with bootstrap CI over sessions (PRIMARY, movement matched).
        matched_partial_mean, matched_partial_ci
            — mean over sessions of partial_spearman(y_pred, y, motor_signal),
              with bootstrap CI over sessions (PRIMARY complement).
        matched_null_mean, matched_null_sd
            — within-session-shuffle null for the matched statistic (the strata
              statistic, aggregated identically over sessions).
        survives_matching
            — bool: matched_strata_mean > matched_null_mean + 2*matched_null_sd.
        subspace_mean_r, subspace_ci
            — mean over sessions of the re-decode after removing the top-k motor
              subspace (SECONDARY; can remove genuine signal).
    """
    from visdetect.analysis.utils import bootstrap_ci

    base = nl.decode_cohort(sessions, n_null=n_null, seed=seed, n_workers=n_workers)

    # ── per-session OOF predictions on the leakage-filtered set ───────────────
    strata_rs, partial_rs, subspace_rs = [], [], []
    per_match = []
    for sid, X, y, tt in sessions:
        y_pred = nl.decode_session(X, y, seed=seed)["y_pred"]
        ms = np.asarray(motor_signals[sid], float)
        s_strata = nl.within_strata_spearman(y_pred, y, ms)
        s_partial = nl.partial_spearman(y_pred, y, ms)
        strata_rs.append(s_strata)
        partial_rs.append(s_partial)
        # secondary: remove the top-k motor subspace from the features, re-decode
        Xp = nl.project_out_subspace(X, motor_subspaces[sid])
        r_sub = nl.decode_session(Xp, y, seed=seed)["r"]
        subspace_rs.append(r_sub)
        per_match.append({"sess_id": sid, "strata_r": float(s_strata),
                          "partial_r": float(s_partial), "subspace_r": float(r_sub)})

    strata_rs = np.asarray(strata_rs, float)
    partial_rs = np.asarray(partial_rs, float)
    subspace_rs = np.asarray(subspace_rs, float)

    def _ci(arr):
        lo, hi = bootstrap_ci(arr, n_bootstrap=1000, seed=seed)
        return (float(lo), float(hi))

    # ── matched-statistic null: shuffle y within session, re-decode, re-match ─
    rng = np.random.default_rng(seed + 1)
    shuffle_seeds = rng.integers(0, 2**31 - 1, size=n_null)
    matched_null = np.empty(n_null)
    for i, ss in enumerate(shuffle_seeds):
        sr = np.random.default_rng(int(ss))
        vals = []
        for sid, X, y, tt in sessions:
            ysh = sr.permutation(y)
            yp = nl.decode_session(X, ysh, seed=seed)["y_pred"]
            vals.append(nl.within_strata_spearman(yp, ysh, np.asarray(motor_signals[sid], float)))
        matched_null[i] = float(np.nanmean(vals))

    matched_strata_mean = float(np.nanmean(strata_rs))
    matched_null_mean = float(matched_null.mean())
    matched_null_sd = float(matched_null.std())
    survives = matched_strata_mean > matched_null_mean + 2 * matched_null_sd

    return {
        # stage 2 (leakage-filtered) decode + its null
        "mean_r": base["mean_r"], "median_r": base["median_r"], "ci": base["ci"],
        "null_mean": base["null_mean"], "null_sd": base["null_sd"],
        "within_type": base["within_type"], "per_session": base["per_session"],
        # stage 3 (matched PRIMARY)
        "matched_strata_mean": matched_strata_mean,
        "matched_strata_ci": _ci(strata_rs),
        "matched_partial_mean": float(np.nanmean(partial_rs)),
        "matched_partial_ci": _ci(partial_rs),
        "matched_null_mean": matched_null_mean,
        "matched_null_sd": matched_null_sd,
        "survives_matching": bool(survives),
        # stage 4 (subspace SECONDARY)
        "subspace_mean_r": float(np.nanmean(subspace_rs)),
        "subspace_ci": _ci(subspace_rs),
        "per_session_matching": per_match,
    }


def _verdict_string(early):
    """Both-ways honest verdict from the EARLY-window cascade result `early`."""
    if early["survives_matching"]:
        return ("PASSES (early window): self-timed urgency predicts FA lick timing "
                "BEYOND generic motor prep — the leakage-filtered within-FA decode "
                f"(r={early['mean_r']:.3f}) survives movement-matching "
                f"(within-strata r={early['matched_strata_mean']:.3f} > null+2SD="
                f"{early['matched_null_mean'] + 2 * early['matched_null_sd']:.3f}; "
                f"partial r={early['matched_partial_mean']:.3f}).")
    return ("COLLAPSES (early window): for self-timed (FA) licks the urgency/"
            "commitment ramp and motor preparation are NOT separable — movement-"
            f"matching reduces the leakage-filtered within-FA decode "
            f"(r={early['mean_r']:.3f}) to within-strata r="
            f"{early['matched_strata_mean']:.3f} (null+2SD="
            f"{early['matched_null_mean'] + 2 * early['matched_null_sd']:.3f}). "
            "This is a REAL basal-ganglia action-commitment conclusion, not a "
            "control failure.")


# ── Real-data wiring (main) ──────────────────────────────────────────────────
# Imports below are deferred to main() so the pure core (and the test) load
# WITHOUT pulling in matplotlib / the suite loader / heavy session I/O.

_JOIN_WINDOW = (-1.3, 6.0)          # Baseline_ON-aligned span covering all WINDOWS
_LICK_SPAN = (-2.0, 0.75)           # lick-aligned tensor span for the motor CD / subspace
_LICK_BASELINE = (-1.75, -1.25)     # pre-lick baseline for the lick-aligned z-score (_FA_BASE)


def _build_motor_blocks(session, good_ids):
    """Fresh per-session LICK-aligned z tensor in the unit space `good_ids`.

    Mirrors the Task-6 motor wiring: stack FA- and Hit-aligned trials (both aligned
    to the 200 ms-corrected lick inside ``build_population_tensor``), restrict FA to
    LATE licks (>= FA_RT_SPLIT s after Baseline_ON) so the pre-lick baseline does
    not run before Baseline_ON, per-unit z-score against the pre-lick baseline.
    Returns (z_lick, bin_centers) or (None, None) if too few lick trials."""
    from visdetect.analysis.utils import build_population_tensor, compute_zscore_normalized
    from visdetect.analysis.align import get_event_times_by_trial
    from visdetect.analysis.constants import DEFAULT_BIN_SIZE, FA_RT_SPLIT

    base_on = np.asarray(session.ni_events.get("Baseline_ON", []), float).ravel()
    blocks, bc_ref = [], None
    for ev in ("FA", "Hit"):
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
            continue
        bc_ref = bc if bc_ref is None else bc_ref
        blocks.append(tensor)
    if not blocks:
        return None, None
    z_lick = compute_zscore_normalized(np.concatenate(blocks, axis=0), bc_ref, _LICK_BASELINE)
    if z_lick.shape[0] < 3:
        return None, None
    return z_lick, bc_ref


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
    print(f"[n1_c1b] {len(sess_ids)} fitted-expert sessions; "
          f"null n_workers={n_workers}", flush=True)

    win_names = list(nl.WINDOWS)
    # RAW cohort (all FA), LEAKAGE-FILTERED cohort, motor signals, motor subspaces
    raw_cohort = {w: [] for w in win_names}
    clean_cohort = {w: [] for w in win_names}
    motor_signals = {w: {} for w in win_names}
    motor_subspaces = {w: {} for w in win_names}
    n_clean = {w: [] for w in win_names}        # per-session (sid, n_clean, n_fa)
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

            fa_mask = (jr.y["outcome"] == "fa").to_numpy()
            if fa_mask.sum() < MIN_CLEAN_FA:
                join_failures.append((sid, f"only {int(fa_mask.sum())} FA trials"))
                continue
            y_fa = jr.y.loc[fa_mask, "decision_time"].to_numpy(float)
            tt_fa = jr.y.loc[fa_mask, "outcome"].to_numpy()
            z_fa = jr.z[fa_mask]

            # fresh per-session motor axis + subspace in jr.unit_ids order
            z_lick, lick_bc = _build_motor_blocks(s, jr.unit_ids)
            if z_lick is None or z_lick.shape[2] != z_fa.shape[2]:
                join_failures.append((sid, "motor axis/subspace unavailable or mismatched"))
                continue
            axis = nl.fit_lick_motor_cd(z_lick, lick_bc)
            subspace = nl.motor_subspace(z_lick, lick_bc, k=MOTOR_K)

            contributed_any = False
            for w in win_names:
                win = nl.WINDOWS[w]
                X_fa = nl.window_feature_matrix(z_fa, jr.bin_centers, win)
                # RAW: all FA trials
                raw_cohort[w].append((sid, X_fa, y_fa, tt_fa))
                # LEAKAGE FILTER FIRST: licks comfortably AFTER the read-out window
                clean = nl.leakage_free_mask(y_fa, win, guard=LEAKAGE_GUARD)
                n_clean[w].append({"sess_id": sid, "n_clean": int(clean.sum()),
                                   "n_fa": int(fa_mask.sum())})
                if clean.sum() < MIN_CLEAN_FA:
                    continue                      # too few clean trials for this window
                Xc = X_fa[clean]
                yc = y_fa[clean]
                ttc = tt_fa[clean]
                clean_cohort[w].append((sid, Xc, yc, ttc))
                # motor-axis signal per CLEAN trial = the matching control
                motor_signals[w][sid] = nl.motor_axis_signal(Xc, axis)
                motor_subspaces[w][sid] = subspace
                contributed_any = True
            if contributed_any:
                contributing.add(sid)
                print(f"  [{sid}] {int(fa_mask.sum())} FA, {len(jr.unit_ids)} units; "
                      f"clean/window="
                      f"{ {w: int(nl.leakage_free_mask(y_fa, nl.WINDOWS[w], guard=LEAKAGE_GUARD).sum()) for w in win_names} }",
                      flush=True)
        finally:
            del s
            gc.collect()

    if not contributing:
        raise RuntimeError("n1_c1b: NO sessions contributed — every join failed. "
                           f"failures: {join_failures}")

    # ── per-window cascade ────────────────────────────────────────────────────
    results = {}
    for w in win_names:
        win = nl.WINDOWS[w]
        # stage 1: raw within-FA r (all FA) — decode_cohort, no null needed beyond mean
        raw = nl.decode_cohort(raw_cohort[w], n_null=1, seed=42, n_workers=1) \
            if raw_cohort[w] else None
        raw_mean = raw["mean_r"] if raw else float("nan")
        # stages 2-4: leakage-filtered + matched + subspace
        if clean_cohort[w]:
            res = evaluate_window_within_fa(
                clean_cohort[w], motor_signals[w], motor_subspaces[w],
                n_null=200, seed=42, n_workers=n_workers)
        else:
            res = None
        ncl = [d["n_clean"] for d in n_clean[w]]
        results[w] = {
            "window": list(win),
            "n_sessions_raw": len(raw_cohort[w]),
            "n_sessions_clean": len(clean_cohort[w]),
            "median_clean_fa_per_session": float(np.median(ncl)) if ncl else 0.0,
            "n_clean_per_session": n_clean[w],
            "raw_mean_r": float(raw_mean),
            "filtered_mean_r": res["mean_r"] if res else float("nan"),
            "filtered_ci": list(res["ci"]) if res else [float("nan")] * 2,
            "filtered_null_mean": res["null_mean"] if res else float("nan"),
            "filtered_null_sd": res["null_sd"] if res else float("nan"),
            "matched_strata_mean": res["matched_strata_mean"] if res else float("nan"),
            "matched_strata_ci": list(res["matched_strata_ci"]) if res else [float("nan")] * 2,
            "matched_partial_mean": res["matched_partial_mean"] if res else float("nan"),
            "matched_partial_ci": list(res["matched_partial_ci"]) if res else [float("nan")] * 2,
            "matched_null_mean": res["matched_null_mean"] if res else float("nan"),
            "matched_null_sd": res["matched_null_sd"] if res else float("nan"),
            "matched_null_upper": (res["matched_null_mean"] + 2 * res["matched_null_sd"])
            if res else float("nan"),
            "survives_matching": res["survives_matching"] if res else False,
            "subspace_mean_r": res["subspace_mean_r"] if res else float("nan"),
            "subspace_ci": list(res["subspace_ci"]) if res else [float("nan")] * 2,
            "within_type": res["within_type"] if res else {},
        }
        print(f"[{w}] {win} | raw r={raw_mean:.3f} -> filtered r="
              f"{results[w]['filtered_mean_r']:.3f} (n_clean median "
              f"{results[w]['median_clean_fa_per_session']:.0f}) -> matched strata="
              f"{results[w]['matched_strata_mean']:.3f} partial="
              f"{results[w]['matched_partial_mean']:.3f} "
              f"(null+2SD={results[w]['matched_null_upper']:.3f}, "
              f"survives={results[w]['survives_matching']}) -> subspace r="
              f"{results[w]['subspace_mean_r']:.3f}", flush=True)

    # LEAD with EARLY (best raw signal / least movement / most clean trials)
    early = results["early"]
    verdict = _verdict_string({
        "survives_matching": early["survives_matching"],
        "mean_r": early["filtered_mean_r"],
        "matched_strata_mean": early["matched_strata_mean"],
        "matched_partial_mean": early["matched_partial_mean"],
        "matched_null_mean": early["matched_null_mean"],
        "matched_null_sd": early["matched_null_sd"],
    })
    print(f"\n[n1_c1b] VERDICT: {verdict}", flush=True)

    out = {
        "subject": SUBJECT,
        "n_fitted_expert_sessions": len(sess_ids),
        "n_contributing_sessions": len(contributing),
        "contributing_sessions": sorted(contributing, key=lambda x: list(sess_ids).index(x)),
        "join_failures": [{"sess_id": s, "reason": r} for s, r in join_failures],
        "leakage_guard_s": LEAKAGE_GUARD,
        "motor_k": MOTOR_K,
        "lead_window": "early",
        "windows": results,
        "verdict": verdict,
    }
    json_path = os.path.join(CACHE_DIR, "n1_c1b_within_fa.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[n1_c1b] wrote {json_path}", flush=True)

    # ── stats CSV ──────────────────────────────────────────────────────────────
    rows = []
    for w in win_names:
        r = results[w]
        rows.append({
            "window": w, "lo": r["window"][0], "hi": r["window"][1],
            "is_lead": (w == "early"),
            "n_sessions_clean": r["n_sessions_clean"],
            "median_clean_fa": r["median_clean_fa_per_session"],
            "raw_r": r["raw_mean_r"],
            "filtered_r": r["filtered_mean_r"],
            "filtered_ci_lo": r["filtered_ci"][0], "filtered_ci_hi": r["filtered_ci"][1],
            "matched_strata_r": r["matched_strata_mean"],
            "matched_strata_ci_lo": r["matched_strata_ci"][0],
            "matched_strata_ci_hi": r["matched_strata_ci"][1],
            "matched_partial_r": r["matched_partial_mean"],
            "matched_null_mean": r["matched_null_mean"],
            "matched_null_upper": r["matched_null_upper"],
            "survives_matching": r["survives_matching"],
            "subspace_r": r["subspace_mean_r"],
        })
    csv_path = os.path.join(CACHE_DIR, "n1_c1b_within_fa_stats.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[n1_c1b] wrote {csv_path}", flush=True)

    # ── figure ──────────────────────────────────────────────────────────────────
    _make_figure(win_names, results, verdict, len(contributing), len(sess_ids),
                 OUTCOME_COLORS, FIG_DIR)
    return out


def _make_figure(win_names, results, verdict, n_contrib, n_total, outcome_colors, fig_dir):
    import matplotlib.pyplot as plt

    # lead-early ordering for the panels
    order = ["early"] + [w for w in win_names if w != "early"]
    fig, axes = plt.subplots(1, len(order), figsize=(5.2 * len(order), 5.4), sharey=True)
    if len(order) == 1:
        axes = [axes]
    fa_color = outcome_colors.get("FA", "#FF9800")
    cascade_labels = ["raw\nwithin-FA", "leakage-\nfiltered",
                      "matched\n(strata)", "matched\n(partial)", "subspace-\nprojected"]
    cascade_colors = [fa_color, "#d18b2c", "#1f4e79", "#2e6da4", "#c0392b"]

    for ax, w in zip(axes, order):
        r = results[w]
        vals = [r["raw_mean_r"], r["filtered_mean_r"], r["matched_strata_mean"],
                r["matched_partial_mean"], r["subspace_mean_r"]]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=cascade_colors, edgecolor="0.2", width=0.72)
        # null+2SD line for the matched stage (the decisive bar)
        nu = r["matched_null_upper"]
        if np.isfinite(nu):
            ax.hlines(nu, 1.5, 3.5, color="k", linestyle="--", lw=1.2,
                      label="matched null + 2 SD")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cascade_labels, fontsize=8)
        lead = "  (LEAD)" if w == "early" else ""
        ax.set_title(f"{w} window {tuple(r['window'])} s{lead}\n"
                     f"clean FA/session ~{r['median_clean_fa_per_session']:.0f}",
                     fontsize=10, fontweight=("bold" if w == "early" else "normal"))
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (xi, v), ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=8)
        if w == "early":
            ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("self-timed (FA) lick-timing prediction\n(Spearman r)")

    fig.suptitle(
        "Within-FA timing decode: leakage filter FIRST, then movement-MATCHING "
        "(decisive prong-2)\n"
        f"{n_contrib}/{n_total} expert sessions; FA trials only; decoded "
        "within-session, aggregated over sessions\n" + verdict,
        fontsize=11, y=1.04)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "fig_n1_c1b_within_fa.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[n1_c1b] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
