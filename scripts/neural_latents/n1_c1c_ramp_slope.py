"""N1 / C1c — within-FA timing from the per-unit ramp SLOPE.
The ONE pre-specified ramp-appropriate readout; the LAST readout before N1
finalizes.  (Task 6c of the N1 plan.)

Plain-language summary
----------------------
Task 6b ran the decisive within-FA cascade (leakage filter FIRST, then movement-
MATCHING) using the MEAN z-rate inside each read-out window as the decode feature,
and found a fully-controlled NULL: once the lick is filtered out of the window and
"how much the animal is moving" is matched away, the mean-level signal no longer
predicts WHEN the self-timed (false-alarm) lick happens.

But a ramp-to-threshold's decision-relevant content is the SLOPE (the rate of rise
-> the implied time-to-threshold), not the average level.  A null on the MEAN is
NOT a null on the SLOPE.  So this script re-runs the SAME leakage-filtered +
movement-matched within-FA cascade, changing ONLY the decode feature: per trial and
unit we use the OLS slope of z vs time across the in-window bins
(``nl.ramp_slope_feature_matrix``) instead of the mean (``nl.window_feature_matrix``).
Everything else — the join, the FA-only cohort, the leakage filter
(``decision_time >= window-end + 0.25 s``), the per-session preparatory-motor coding
direction, the movement-matching control (still the MEAN-window motor-axis MAGNITUDE,
i.e. movement magnitude, which is independent of the decode feature), the
within-strata + partial-Spearman primaries, the secondary subspace projection, the
per-session aggregation, the bootstrap-over-sessions CI, and the within-session
shuffle nulls — is REUSED verbatim from Task 6b's cascade core
(``n1_c1b_within_fa.evaluate_window_within_fa``).

We LEAD with the EARLY window (best raw signal, fewest in-window licks; early
activity predicting a lick seconds later is itself evidence AGAINST pure
peri-movement motor-prep).  All three windows are reported.

Discipline (locked, post-6b null): this is the LAST readout.
  * If the ramp-slope readout is ALSO null  -> N1 finalizes as a CONTROLLED
    NEGATIVE.  No further readouts (that would be fishing).
  * If it REVIVES                            -> treat with SUSPICION (scrutiny /
    replication, NOT a headline).

(Optional SNR variant — slope of the projection onto a TRAIN-FOLD timing coding
direction — is deliberately NOT implemented: it would add a circularity surface.
The per-unit slope is the clean, pre-specified check.)

Verdict rule (SAME as 6b): report null-or-revive based on whether the MATCHED
partial-Spearman robustly beats its within-session-shuffle null AND the
bootstrap-over-sessions CI excludes 0.  We note explicitly that the within-strata
statistic ALONE is lenient (coarse 4-quantile matching) — the partial-Spearman is
the conservative primary.

Outputs (canonical cache / FIGURES trees):
  * data/cache/neural_latents/n1_c1c_ramp_slope.json
  * FIGURES/neural_latents/BG_046/fig_n1_c1c_ramp_slope.png
  * data/cache/neural_latents/n1_c1c_ramp_slope_stats.csv

Pure core: REUSED from the sibling Task-6b script
``n1_c1b_within_fa.evaluate_window_within_fa`` (the feature only changes upstream
in ``main``).
Run (CONTROLLER ONLY — slow):
  PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/n1_c1c_ramp_slope.py
"""
import os
import gc
import json

import numpy as np

from visdetect.analysis import neural_latents as nl

# Reuse Task-6b's cascade core + motor wiring + verdict (DO NOT duplicate).
# scripts/neural_latents/ has an __init__.py, so this is a clean package import.
from scripts.neural_latents.n1_c1b_within_fa import (
    LEAKAGE_GUARD,
    MOTOR_K,
    MIN_CLEAN_FA,
    _JOIN_WINDOW,
    _build_motor_blocks,
    evaluate_window_within_fa,
)


def _ramp_slope_verdict(early):
    """Null-or-revive verdict from the EARLY-window cascade result `early`, using
    the SAME rule as 6b: the MATCHED partial-Spearman must robustly beat its
    within-session-shuffle null AND the bootstrap-over-sessions CI must exclude 0.
    We lead with the conservative partial-Spearman; the within-strata statistic
    alone is lenient (coarse 4-quantile matching) and is reported for context."""
    partial = early["matched_partial_mean"]
    lo, hi = early["matched_partial_ci"]
    null_upper = early["matched_null_mean"] + 2 * early["matched_null_sd"]
    ci_excludes_0 = (lo > 0) or (hi < 0)
    beats_null = partial > null_upper
    revives = bool(beats_null and ci_excludes_0)
    if revives:
        return ("REVIVES (early window, TREAT WITH SUSPICION): the per-unit ramp-"
                "SLOPE readout predicts self-timed (FA) lick timing beyond generic "
                "motor prep where the MEAN readout was null — the leakage-filtered "
                f"within-FA slope decode (r={early['mean_r']:.3f}) survives movement-"
                f"matching (partial r={partial:.3f}, CI [{lo:.3f},{hi:.3f}] excludes 0; "
                f"> matched null+2SD={null_upper:.3f}; within-strata r="
                f"{early['matched_strata_mean']:.3f}, the LENIENT statistic). Per the "
                "locked discipline this is the LAST readout: do NOT headline — "
                "scrutinize / replicate.")
    return ("NULL (early window): the per-unit ramp-SLOPE readout is ALSO a fully-"
            "controlled NULL — like the MEAN readout, once leakage is filtered and "
            "movement is matched out the slope does not predict FA lick timing "
            f"(leakage-filtered within-FA slope decode r={early['mean_r']:.3f}; matched "
            f"partial r={partial:.3f}, CI [{lo:.3f},{hi:.3f}]; matched null+2SD="
            f"{null_upper:.3f}; within-strata r={early['matched_strata_mean']:.3f}, the "
            "LENIENT statistic). This was the ONE pre-specified ramp-appropriate "
            "readout and the LAST one — N1 finalizes as a CONTROLLED NEGATIVE (no "
            "further readouts).")


# ── Real-data wiring (main) ──────────────────────────────────────────────────
# Imports below are deferred to main() so the module (and the test) load WITHOUT
# pulling in matplotlib / the suite loader / heavy session I/O.
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
    print(f"[n1_c1c] {len(sess_ids)} fitted-expert sessions; "
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
                # ONLY CHANGE vs 6b: the decode FEATURE is the per-unit ramp SLOPE.
                X_fa = nl.ramp_slope_feature_matrix(z_fa, jr.bin_centers, win)
                # the movement-matching CONTROL stays the MEAN-window motor-axis
                # magnitude (movement magnitude is independent of the decode feature).
                X_mean = nl.window_feature_matrix(z_fa, jr.bin_centers, win)
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
                # motor-axis signal per CLEAN trial = the matching control, defined on
                # the MEAN window features (movement magnitude, NOT the slope feature).
                motor_signals[w][sid] = nl.motor_axis_signal(X_mean[clean], axis)
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
        raise RuntimeError("n1_c1c: NO sessions contributed — every join failed. "
                           f"failures: {join_failures}")

    # ── per-window cascade (REUSED Task-6b core) ──────────────────────────────
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
        print(f"[{w}] {win} | raw slope-r={raw_mean:.3f} -> filtered r="
              f"{results[w]['filtered_mean_r']:.3f} (n_clean median "
              f"{results[w]['median_clean_fa_per_session']:.0f}) -> matched strata="
              f"{results[w]['matched_strata_mean']:.3f} partial="
              f"{results[w]['matched_partial_mean']:.3f} "
              f"(null+2SD={results[w]['matched_null_upper']:.3f}, "
              f"survives={results[w]['survives_matching']}) -> subspace r="
              f"{results[w]['subspace_mean_r']:.3f}", flush=True)

    # LEAD with EARLY (best raw signal / least movement / most clean trials)
    early = results["early"]
    verdict = _ramp_slope_verdict({
        "mean_r": early["filtered_mean_r"],
        "matched_strata_mean": early["matched_strata_mean"],
        "matched_partial_mean": early["matched_partial_mean"],
        "matched_partial_ci": early["matched_partial_ci"],
        "matched_null_mean": early["matched_null_mean"],
        "matched_null_sd": early["matched_null_sd"],
    })
    print(f"\n[n1_c1c] VERDICT: {verdict}", flush=True)

    out = {
        "subject": SUBJECT,
        "readout": "per-unit ramp SLOPE (OLS slope of z vs time over the window)",
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
    json_path = os.path.join(CACHE_DIR, "n1_c1c_ramp_slope.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[n1_c1c] wrote {json_path}", flush=True)

    # ── stats CSV ──────────────────────────────────────────────────────────────
    rows = []
    for w in win_names:
        r = results[w]
        rows.append({
            "window": w, "lo": r["window"][0], "hi": r["window"][1],
            "is_lead": (w == "early"),
            "n_sessions_clean": r["n_sessions_clean"],
            "median_clean_fa": r["median_clean_fa_per_session"],
            "raw_slope_r": r["raw_mean_r"],
            "filtered_r": r["filtered_mean_r"],
            "filtered_ci_lo": r["filtered_ci"][0], "filtered_ci_hi": r["filtered_ci"][1],
            "matched_strata_r": r["matched_strata_mean"],
            "matched_strata_ci_lo": r["matched_strata_ci"][0],
            "matched_strata_ci_hi": r["matched_strata_ci"][1],
            "matched_partial_r": r["matched_partial_mean"],
            "matched_partial_ci_lo": r["matched_partial_ci"][0],
            "matched_partial_ci_hi": r["matched_partial_ci"][1],
            "matched_null_mean": r["matched_null_mean"],
            "matched_null_upper": r["matched_null_upper"],
            "survives_matching": r["survives_matching"],
            "subspace_r": r["subspace_mean_r"],
        })
    csv_path = os.path.join(CACHE_DIR, "n1_c1c_ramp_slope_stats.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[n1_c1c] wrote {csv_path}", flush=True)

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
    axes[0].set_ylabel("self-timed (FA) lick-timing prediction\nfrom per-unit RAMP SLOPE (Spearman r)")

    fig.suptitle(
        "Within-FA timing decode from the per-unit RAMP SLOPE: leakage filter FIRST, "
        "then movement-MATCHING\n"
        "(the ONE pre-specified ramp-appropriate readout — the LAST readout before "
        "N1 finalizes)\n"
        f"{n_contrib}/{n_total} expert sessions; FA trials only; decoded "
        "within-session, aggregated over sessions\n" + verdict,
        fontsize=11, y=1.06)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "fig_n1_c1c_ramp_slope.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[n1_c1c] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
