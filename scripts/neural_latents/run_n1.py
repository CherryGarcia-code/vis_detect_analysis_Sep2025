"""N1 — write-up orchestrator for the controlled-NEGATIVE result (Task 9, repurposed).

Plain-language summary
----------------------
N1 asked: in expert BG_046 medial striatum, does pre-decision (waiting-window)
population activity carry an "urgency ramp" that predicts WHEN the animal will
respond (a self-timed lick), over and above generic motor preparation?

The settled answer is a **fully-controlled NEGATIVE**, and the headline is really
a reusable METHODS CAUTIONARY TALE:

  * The pooled response-time decode beats chance (mean r ~ 0.44) but that is mostly
    hit-vs-FA *trial-type* separation.  The clean read is WITHIN trial type.
  * WITHIN-HIT ~ 0.05 is a leakage-free NEGATIVE CONTROL: hits lick > 6 s, after
    every read-out window, so the exogenous, change-set hit time cannot be (and is
    not) predicted from pre-change activity.  This validates the decode.
  * The genuine signal is self-timed (false-alarm, FA) lick timing.  The RAW
    within-FA decode looks strong (r ~ 0.56 mean / ~0.33 slope) but is OVERWHELMINGLY
    LICK LEAKAGE: 15-28 % of FA licks land inside the read-out window, so the decoder
    reads `decision_time` off peri/post-lick activity.
  * Once the leakage is filtered out (lick >= window-end + 0.25 s) AND movement is
    matched out (partial-Spearman primary), BOTH the MEAN and the per-unit SLOPE
    read-outs collapse to ~0 with bootstrap-over-sessions CIs that span zero.

This orchestrator does NOT re-run any heavy analysis.  It reads the four cached
result JSONs, assembles a compact `n1_results.json`, draws one plain-language
summary figure, and (separately) backs the results write-up.

Pure core: ``assemble_n1_summary(paths_or_dicts)`` — load the JSONs (or accept
pre-loaded dicts) and return the compact summary.  Fully unit-testable.

Outputs (canonical trees):
  * data/cache/neural_latents/n1_results.json
  * FIGURES/neural_latents/BG_046/fig_n1_summary.png

Run (light — reads cached JSONs only; NO session pkl, NO X:):
  PYTHONPATH=src .venv/Scripts/python.exe scripts/neural_latents/run_n1.py
"""
import os
import json

import numpy as np


# ── canonical cache paths (resolved lazily so the pure core can be tested with
#    in-memory dicts without importing the suite config) ───────────────────────
def _default_paths():
    from visdetect.analysis.config import ROOT
    cache = os.path.join(ROOT, "data", "cache", "neural_latents")
    return {
        "gate": os.path.join(cache, "n1_c1_gate.json"),
        "within_fa": os.path.join(cache, "n1_c1b_within_fa.json"),
        "ramp_slope": os.path.join(cache, "n1_c1c_ramp_slope.json"),
        "synthetic": os.path.join(cache, "n1_synthetic_verdict.json"),
    }


def _load(path_or_dict):
    """Accept either a path to a JSON file or an already-loaded dict (for tests)."""
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict) as f:
        return json.load(f)


def _within_fa_early(within_fa):
    """The EARLY window block of a within-FA cascade JSON (6b mean OR 6c slope).
    EARLY leads: best raw signal, least movement, most clean trials; early activity
    predicting a lick seconds later is itself evidence against pure peri-movement
    motor prep."""
    early = within_fa["windows"]["early"]
    return {
        "window": list(early["window"]),
        "raw_mean_r": early["raw_mean_r"],
        "filtered_mean_r": early["filtered_mean_r"],
        "filtered_ci": list(early["filtered_ci"]),
        "matched_strata_mean": early["matched_strata_mean"],
        "matched_strata_ci": list(early["matched_strata_ci"]),
        "matched_partial_mean": early["matched_partial_mean"],
        "matched_partial_ci": list(early["matched_partial_ci"]),
        "matched_null_mean": early["matched_null_mean"],
        "matched_null_sd": early["matched_null_sd"],
        "matched_null_upper": early["matched_null_mean"] + 2 * early["matched_null_sd"],
        "subspace_mean_r": early["subspace_mean_r"],
        "median_clean_fa_per_session": early["median_clean_fa_per_session"],
    }


def assemble_n1_summary(paths):
    """Pure: load the four cached N1 result JSONs (paths OR pre-loaded dicts) and
    return a compact, presentation-ready summary of the CONTROLLED-NEGATIVE verdict.

    `paths` is a dict with keys {gate, within_fa, ramp_slope, synthetic}; each value
    is a filesystem path OR an already-loaded dict (the latter for unit tests).
    """
    gate = _load(paths["gate"])
    within_fa = _load(paths["within_fa"])
    ramp_slope = _load(paths["ramp_slope"])
    synthetic = _load(paths["synthetic"])

    # ── provenance / sessions (identical across the three real-data JSONs) ────
    n_fitted = gate.get("n_fitted_expert_sessions")
    n_contrib = gate.get("n_contributing_sessions")
    join_failures = gate.get("join_failures", [])

    # ── within-HIT clean negative control (the decode validation) ────────────
    # Read from the gate's per-window within_type["hit"] (the leakage-free control:
    # hits lick > 6 s, after every read-out window).
    gate_windows = gate.get("windows", {})
    within_hit_by_window = {
        w: gw.get("within_type", {}).get("hit")
        for w, gw in gate_windows.items()
    }
    within_fa_raw_by_window = {
        w: gw.get("within_type", {}).get("fa")
        for w, gw in gate_windows.items()
    }

    # ── the leakage-collapse cascade, EARLY window, for MEAN and SLOPE ────────
    mean_cascade = _within_fa_early(within_fa)
    slope_cascade = _within_fa_early(ramp_slope)

    # ── synthetic method-validation booleans ─────────────────────────────────
    method_validation = {
        "recovers": bool(synthetic.get("recovers")),
        "motor_killed": bool(synthetic.get("motor_killed")),
        "phi_separable_on_window": bool(synthetic.get("phi_separable_on_window")),
    }

    # ── one-line plain-language conclusion ────────────────────────────────────
    conclusion = (
        "Controlled NEGATIVE: in expert BG_046 striatum there is no leakage-free, "
        "movement-matched correlate of self-timed (FA) lick timing in pre-change "
        "population activity — neither the mean level nor the per-unit ramp slope. "
        "The within-HIT decode (~0.05) is a clean leakage-free negative control that "
        "validates the method; the raw within-FA decode (~0.5) is overwhelmingly lick "
        "leakage that collapses to ~0.02 once leakage is filtered and movement matched. "
        "Methods lesson: within-trial-type timing decodes inflate via lick leakage; the "
        "fix is the leakage filter + movement-matching + the within-hit negative control."
    )

    return {
        "question_id": "N1",
        "subject": gate.get("subject", "BG_046"),
        "verdict": "controlled negative",
        "n_fitted_expert_sessions": n_fitted,
        "n_contributing_sessions": n_contrib,
        "join_failures": join_failures,
        # the leakage-free decode validation
        "within_hit_negative_control": {
            "by_window": within_hit_by_window,
            "within_fa_raw_by_window": within_fa_raw_by_window,
            "note": ("within-HIT ~0.05 is a clean leakage-free NEGATIVE CONTROL "
                     "(hits lick >6 s, after every read-out window); raw within-FA "
                     "~0.34-0.56 is the leakage-inflated signal."),
        },
        # the two leakage-collapse cascades (EARLY window leads)
        "lead_window": "early",
        "mean_cascade_early": mean_cascade,
        "slope_cascade_early": slope_cascade,
        "method_validation": method_validation,
        "conclusion": conclusion,
    }


# ── figure ───────────────────────────────────────────────────────────────────
def _make_summary_figure(summary, fig_path, outcome_colors=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    outcome_colors = outcome_colors or {}
    fa_color = outcome_colors.get("FA", "#FF9800")
    hit_color = outcome_colors.get("Hit", "#4CAF50")

    fig = plt.figure(figsize=(15, 6.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1.0, 1.15],
                           wspace=0.32, left=0.06, right=0.98,
                           top=0.80, bottom=0.13)

    # ── Panel (a): leakage-collapse cascade, descending bars, MEAN and SLOPE ──
    ax_a = fig.add_subplot(gs[0, 0])
    stages = ["raw\nwithin-FA", "leakage-\nfiltered", "matched\n(partial)"]
    mean_c = summary["mean_cascade_early"]
    slope_c = summary["slope_cascade_early"]
    mean_vals = [mean_c["raw_mean_r"], mean_c["filtered_mean_r"],
                 mean_c["matched_partial_mean"]]
    slope_vals = [slope_c["raw_mean_r"], slope_c["filtered_mean_r"],
                  slope_c["matched_partial_mean"]]
    x = np.arange(len(stages))
    w = 0.38
    b1 = ax_a.bar(x - w / 2, mean_vals, w, label="MEAN read-out",
                  color="#1f4e79", edgecolor="0.2")
    b2 = ax_a.bar(x + w / 2, slope_vals, w, label="SLOPE read-out",
                  color="#c0392b", edgecolor="0.2")
    ax_a.axhline(0, color="0.6", lw=0.8)
    for bars, vals in ((b1, mean_vals), (b2, slope_vals)):
        for bar, v in zip(bars, vals):
            ax_a.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                          ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(stages, fontsize=9)
    ax_a.set_ylabel("self-timed (FA) lick-timing\nprediction (Spearman r)")
    ax_a.set_title("(a) Leakage collapses the within-FA decode\n"
                   "early window [0.5, 2.5] s; r ~0.5 -> ~0.02",
                   fontsize=10)
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.set_ylim(min(-0.05, min(mean_vals + slope_vals) - 0.05),
                  max(mean_vals + slope_vals) * 1.18)

    # ── Panel (b): within-HIT ~0 (negative control) vs within-FA-raw ──────────
    ax_b = fig.add_subplot(gs[0, 1])
    wh = summary["within_hit_negative_control"]
    # use the EARLY window for the comparison (the lead window)
    hit_r = wh["by_window"].get("early")
    fa_raw_r = wh["within_fa_raw_by_window"].get("early")
    bvals = [hit_r if hit_r is not None else 0.0,
             fa_raw_r if fa_raw_r is not None else 0.0]
    bx = np.arange(2)
    bars = ax_b.bar(bx, bvals, 0.6,
                    color=[hit_color, fa_color], edgecolor="0.2")
    ax_b.axhline(0, color="0.6", lw=0.8)
    for bar, v in zip(bars, bvals):
        ax_b.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                      ha="center", va="bottom", fontsize=9)
    ax_b.set_xticks(bx)
    ax_b.set_xticklabels(["within-HIT\n(neg. control)", "within-FA\n(RAW)"],
                         fontsize=9)
    ax_b.set_ylabel("within-trial-type timing\nprediction (Spearman r)")
    ax_b.set_title("(b) Within-HIT ~0 validates the decode;\n"
                   "raw within-FA is LICK LEAKAGE", fontsize=10)
    ax_b.annotate("raw within-FA is\nlick leakage\n(15-28% of FA licks\n"
                  "fall in the window)",
                  xy=(1, bvals[1]), xytext=(0.55, bvals[1] * 0.62),
                  fontsize=8, ha="center", color="0.15",
                  arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0))
    ax_b.set_ylim(-0.05, max(bvals) * 1.25)

    # ── Panel (c): text — verdict + behavioural anchor + methods lesson ───────
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    n_c = summary["n_contributing_sessions"]
    n_f = summary["n_fitted_expert_sessions"]
    mv = summary["method_validation"]
    text = (
        "(c) VERDICT: CONTROLLED NEGATIVE\n"
        "\n"
        "No leakage-free, movement-matched striatal\n"
        "correlate of self-timed (FA) lick timing —\n"
        "neither the mean level nor the ramp slope.\n"
        f"({n_c}/{n_f} expert sessions; 11092025 skipped\n"
        " fail-safe on a Baseline_ON/trials mismatch.)\n"
        "\n"
        "BEHAVIOUR ANCHOR (B8 figure F4):\n"
        "a survival-corrected anticipatory FA-hazard\n"
        "RISES toward the expected change time, so\n"
        "temporal expectation is established BEHAVIOURALLY.\n"
        "The neural null is therefore structurally\n"
        "expected, not a weakness.\n"
        "\n"
        "METHODS CAUTIONARY TALE (the reusable result):\n"
        "within-trial-type timing decodes INFLATE\n"
        "~0.5 -> ~0.02 via lick leakage. The fix =\n"
        "  (1) leakage filter (lick >= window-end + 0.25 s)\n"
        "  (2) movement-matching (partial-Spearman)\n"
        "  (3) the within-HIT leakage-free negative control.\n"
        "\n"
        "Method validated on synthetic data:\n"
        f"  recovers timing = {mv['recovers']}\n"
        f"  motor-CD projection kills pure-motor = {mv['motor_killed']}\n"
        f"  phi separable on window = {mv['phi_separable_on_window']}\n"
        "  (phi-specificity underpowered by construction)."
    )
    ax_c.text(0.0, 1.0, text, va="top", ha="left", fontsize=8.4,
              family="monospace", transform=ax_c.transAxes)

    fig.suptitle(
        "N1 — Does an expert striatal urgency ramp predict WHEN the mouse responds, "
        "beyond movement?  An HONEST CONTROLLED NEGATIVE.\n"
        "Self-timed (FA) lick-timing decode collapses from ~0.5 to ~0.02 once lick "
        "leakage is filtered and movement is matched (mean AND ramp-slope read-outs).",
        fontsize=11.5, y=0.97)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    from visdetect.analysis.config import ROOT, SUBJECT, OUTCOME_COLORS

    paths = _default_paths()
    summary = assemble_n1_summary(paths)

    cache_dir = os.path.join(ROOT, "data", "cache", "neural_latents")
    fig_dir = os.path.join(ROOT, "FIGURES", "neural_latents", SUBJECT)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    json_path = os.path.join(cache_dir, "n1_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[run_n1] wrote {json_path}", flush=True)

    fig_path = os.path.join(fig_dir, "fig_n1_summary.png")
    _make_summary_figure(summary, fig_path, outcome_colors=OUTCOME_COLORS)
    print(f"[run_n1] wrote {fig_path}", flush=True)

    print(f"\n[run_n1] VERDICT: {summary['verdict']}", flush=True)
    print(f"[run_n1] {summary['conclusion']}", flush=True)
    return summary


if __name__ == "__main__":
    main()
