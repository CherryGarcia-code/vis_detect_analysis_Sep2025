"""Fig35k: Lick-hazard GLM — decompose lick probability into sensory,
temporal, and state components using a discrete-time survival model.

Scientific question:
  What drives the animal's decision to lick at each moment within a trial?
  Is it temporal expectation (internal clock), ongoing sensory input
  (stochastic TF fluctuations / fast pulses), change-specific sensory
  evidence (TF ratio change), or behavioral state (impulsivity)?

Approach:
  Discrete-time hazard model: bin each trial at 50 ms resolution from
  Baseline_ON until first lick (event) or end of response window (censored).
  At each bin: P(lick | no lick yet) = logistic(time_splines + log2_tf
  + post_change + change_evidence + behavioral_features + HMM_state + stage).

  Nested model comparison (M0-M5) quantifies incremental contribution of
  each component.  Per-lick decomposition attributes each observed lick to
  its dominant driver.

Produces:
  - Fig35k A: Empirical hazard by outcome
  - Fig35k B: Fitted vs empirical hazard by stage
  - Fig35k C: Nested model % deviance explained
  - Fig35k D: Temporal hazard curve (learned clock)
  - Fig35k E: Component decomposition by outcome (FA / Hit-small / Hit-large)
  - Fig35k F: Stage comparison of component weights
  - Fig35k G: Example per-trial predicted hazard curves
  - Fig35k H: Neural CD projection vs model residual

Saves:
  figures/07_advanced/fig35k_lick_hazard_glm.png
  figures/07_advanced/hazard_glm_model_comparison.csv
  cache/hazard_glm_dataset.csv
"""

import os
import sys
import gc
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import statsmodels.api as sm
from patsy import dmatrix

from config import (
    STAGE_ORDER, STAGE_COLORS, CACHE_DIR,
    HMM_STATE_ORDER, HMM_STATE_COLORS,
    OUTCOME_COLORS,
)
from loader import (
    load_staging_manifest, load_session,
    load_hmm_assignments,
)
from plotting import setup_style, save_figure

# Ensure visdetect is importable
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "src"))

setup_style()
warnings.filterwarnings("ignore")

# ── Parameters ────────────────────────────────────────────────────────
BIN_SIZE = 0.05               # 50 ms bins (matches TF baseline sample period)
BASELINE_STRIDE = 3           # Subsample baseline_values (legacy convention)
MISS_RESPONSE_WINDOW = 2.155  # Response window limit for miss trials (s)
ROLLING_FA_WINDOW = 15        # Trailing window for rolling FA rate
SPLINE_DF = 6                 # Degrees of freedom for temporal basis
MAX_TRIAL_TIME = 25.0         # Truncate very long trials (s) to avoid perfect separation
IMPULSIVE_STATE_LABEL = "Impulsive"
ENGAGED_STATE_LABEL = "Engaged"

# Cache paths
CACHE_CSV = os.path.join(CACHE_DIR, "hazard_glm_dataset.csv")
CACHE_RESULTS = os.path.join(CACHE_DIR, "hazard_glm_results.npz")


# =====================================================================
# Section 1: Build expanded discrete-time hazard dataset
# =====================================================================

def _find_hmm_state_columns(hmm_assign):
    """Find the p_state_X columns for Engaged and Impulsive states."""
    engaged_col = impulsive_col = None
    for state_idx in [0, 1, 2]:
        labels = hmm_assign[hmm_assign["hmm_state"] == state_idx][
            "hmm_state_label"
        ].unique()
        if ENGAGED_STATE_LABEL in labels or "Engaged_2" in labels:
            engaged_col = f"p_state_{state_idx}"
        if IMPULSIVE_STATE_LABEL in labels or "Biased" in labels:
            impulsive_col = f"p_state_{state_idx}"
    if engaged_col is None or impulsive_col is None:
        raise ValueError(
            f"Cannot find Engaged/Impulsive columns. "
            f"Found labels: {hmm_assign['hmm_state_label'].unique()}"
        )
    return engaged_col, impulsive_col


def _get_observation_window(outcome, change_time, reactiontimes, change_size):
    """Return (end_time_from_baseline_on, is_lick_event) for a trial.

    Returns (None, None) for trials to exclude (abort, ref).
    """
    outcome_lower = str(outcome).lower()

    if outcome_lower == "fa":
        fa_rt = reactiontimes.get("FA")
        if fa_rt is None or fa_rt <= 0:
            return None, None
        return float(fa_rt), True

    elif outcome_lower == "hit":
        rt = reactiontimes.get("RT")
        if rt is None or change_time is None:
            return None, None
        return float(change_time) + float(rt), True

    elif outcome_lower == "miss":
        if change_time is None:
            return None, None
        return float(change_time) + MISS_RESPONSE_WINDOW, False

    else:  # abort, ref, unknown
        return None, None


def _process_one_session(sname, stage, hmm_sess_df, engaged_col, impulsive_col):
    """Process a single session into hazard-dataset rows.

    Returns (blocks_list, n_trials, n_skipped) or None on failure.
    """
    stage_binary = 1.0 if stage == "Expert" else 0.0

    try:
        sess = load_session(str(sname))
    except FileNotFoundError:
        print(f"    {sname}: pkl not found, skipping")
        return [], 0, 0

    hmm_sess = hmm_sess_df.set_index("trial_idx") if len(hmm_sess_df) > 0 else pd.DataFrame()

    # Build per-session FA history
    outcomes_seq = []
    for t in sess.trials:
        out = str(t.trialoutcome).lower() if t.trialoutcome else ""
        outcomes_seq.append(out)

    is_fa_seq = np.array([1.0 if o == "fa" else 0.0 for o in outcomes_seq])
    rolling_fa = pd.Series(is_fa_seq).rolling(
        ROLLING_FA_WINDOW, min_periods=3
    ).mean().shift(1).values
    prev_fa_arr = np.concatenate([[np.nan], is_fa_seq[:-1]])

    blocks = []
    n_trials = 0
    n_skipped = 0

    for tidx, trial in enumerate(sess.trials):
        outcome = str(trial.trialoutcome).lower() if trial.trialoutcome else ""
        if outcome in ("abort", "ref", ""):
            n_skipped += 1
            continue

        end_time, is_event = _get_observation_window(
            outcome, trial.change_time, trial.reactiontimes,
            trial.change_size,
        )
        if end_time is None or end_time <= 0:
            n_skipped += 1
            continue

        end_time = min(end_time, MAX_TRIAL_TIME)

        bv = getattr(trial, "baseline_values", None)
        if bv is None:
            n_skipped += 1
            continue
        bv_arr = np.asarray(bv).flatten()
        strided = bv_arr[::BASELINE_STRIDE]
        if trial.n_seen is not None and trial.n_seen < len(strided):
            strided = strided[: trial.n_seen]
        if len(strided) == 0:
            n_skipped += 1
            continue

        change_time = float(trial.change_time) if trial.change_time else 0.0
        change_size = float(trial.change_size) if trial.change_size else 1.0
        change_presented = outcome in ("hit", "miss")

        # HMM posteriors
        if not hmm_sess.empty and tidx in hmm_sess.index:
            p_eng = float(hmm_sess.loc[tidx, engaged_col])
            p_imp = float(hmm_sess.loc[tidx, impulsive_col])
        else:
            p_eng = np.nan
            p_imp = np.nan

        bin_centers = np.arange(0.5 * BIN_SIZE, end_time, BIN_SIZE)
        n_bins = len(bin_centers)
        if n_bins == 0:
            n_skipped += 1
            continue

        y_arr = np.zeros(n_bins, dtype=np.int8)
        if is_event:
            y_arr[-1] = 1

        tf_indices = np.minimum(
            (bin_centers / BIN_SIZE).astype(int), len(strided) - 1
        )
        tf_vals = strided[tf_indices].astype(float)

        if change_presented and change_time > 0:
            last_pre_idx = max(0, min(
                int(change_time / BIN_SIZE) - 1, len(strided) - 1
            ))
            post_mask = bin_centers >= change_time
            if post_mask.any():
                tf_vals[post_mask] = strided[last_pre_idx] * change_size

        log2_tf = np.log2(np.clip(tf_vals, 0.01, None))

        if change_presented and change_time > 0:
            post_change = (bin_centers >= change_time).astype(float)
            change_evidence = post_change * np.log2(max(change_size, 1.0))
        else:
            post_change = np.zeros(n_bins)
            change_evidence = np.zeros(n_bins)

        trial_block = pd.DataFrame({
            "session_name": sname,
            "trial_idx": tidx,
            "outcome": outcome,
            "stage": stage,
            "stage_binary": stage_binary,
            "bin_idx": np.arange(n_bins),
            "bin_time": bin_centers,
            "y": y_arr,
            "log2_tf": log2_tf,
            "post_change": post_change,
            "change_evidence": change_evidence,
            "p_engaged": p_eng,
            "p_impulsive": p_imp,
            "rolling_fa_rate": rolling_fa[tidx]
            if tidx < len(rolling_fa) else np.nan,
            "prev_fa": prev_fa_arr[tidx]
            if tidx < len(prev_fa_arr) else np.nan,
            "change_size": change_size,
        })
        blocks.append(trial_block)
        n_trials += 1

    del sess
    gc.collect()
    return blocks, n_trials, n_skipped


def build_hazard_dataset(force=False, n_workers=1):
    """Build the expanded discrete-time hazard dataset.

    One row per 50 ms bin per trial, from Baseline_ON until first lick
    (event) or end of response window (censored).  Excludes abort and ref.
    """
    if os.path.exists(CACHE_CSV) and not force:
        print(f"  Loading cached dataset: {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)

    print("  Building hazard dataset from scratch...")
    manifest = load_staging_manifest(qc_only=True)
    hmm = load_hmm_assignments()
    engaged_col, impulsive_col = _find_hmm_state_columns(hmm)
    print(f"    Engaged: {engaged_col}, Impulsive: {impulsive_col}")

    # Pre-group HMM by session
    hmm_by_session = {
        int(sname): grp for sname, grp in hmm.groupby("session_name")
    }

    # Prepare session args
    session_args = []
    for _, mrow in manifest.iterrows():
        sname = int(mrow["session_name"])
        stage = mrow["stage"]
        hmm_sess_df = hmm_by_session.get(sname, pd.DataFrame())
        session_args.append((sname, stage, hmm_sess_df, engaged_col, impulsive_col))

    blocks = []
    total_trials = 0
    total_skipped = 0

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"  Using {n_workers} workers for {len(session_args)} sessions")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_one_session, *sa): sa[0]
                for sa in session_args
            }
            for fut in as_completed(futures):
                sname = futures[fut]
                try:
                    sess_blocks, nt, ns = fut.result()
                except Exception as e:
                    print(f"    {sname}: FAILED ({e})")
                    continue
                blocks.extend(sess_blocks)
                total_trials += nt
                total_skipped += ns
                print(f"    {sname}: {nt} trials")
    else:
        for sa in session_args:
            sess_blocks, nt, ns = _process_one_session(*sa)
            blocks.extend(sess_blocks)
            total_trials += nt
            total_skipped += ns
            print(f"    {sa[0]} ({sa[1]}): {total_trials} trials so far")

    print(f"  Total trials: {total_trials}, skipped: {total_skipped}")
    df = pd.concat(blocks, ignore_index=True)

    # Create unique trial ID for cluster-robust SEs
    df["trial_id"] = df["session_name"] * 10000 + df["trial_idx"]

    print(f"  Dataset shape: {df.shape}")
    df.to_csv(CACHE_CSV, index=False)
    print(f"  Saved to {CACHE_CSV}")
    return df


# =====================================================================
# Section 2: Add spline basis
# =====================================================================

def add_spline_basis(df):
    """Add natural cubic spline basis columns for time-in-trial."""
    basis = dmatrix(
        f"cr(bin_time, df={SPLINE_DF}) - 1",
        data=df,
        return_type="dataframe",
    )
    basis.columns = [f"time_basis_{i}" for i in range(basis.shape[1])]
    basis.index = df.index
    df = pd.concat([df, basis], axis=1)

    # Stage × time interactions for M5
    for i in range(SPLINE_DF):
        df[f"stage_x_time_{i}"] = df["stage_binary"] * df[f"time_basis_{i}"]

    return df


# =====================================================================
# Section 3: Fit nested hazard models
# =====================================================================

def fit_nested_hazard_models(df):
    """Fit M0-M5 nested discrete-time hazard models via statsmodels GLM."""
    time_cols = [f"time_basis_{i}" for i in range(SPLINE_DF)]
    stage_int_cols = [f"stage_x_time_{i}" for i in range(SPLINE_DF)]

    model_specs = {
        "M0": [],
        "M1": time_cols,
        "M2": time_cols + ["log2_tf"],
        "M3": time_cols + ["log2_tf", "post_change", "change_evidence"],
        "M4a": time_cols + ["log2_tf", "post_change", "change_evidence",
                            "rolling_fa_rate", "prev_fa"],
        "M4b": time_cols + ["log2_tf", "post_change", "change_evidence",
                            "rolling_fa_rate", "prev_fa",
                            "p_engaged", "p_impulsive"],
        "M5": time_cols + ["log2_tf", "post_change", "change_evidence",
                           "rolling_fa_rate", "prev_fa",
                           "p_engaged", "p_impulsive",
                           "stage_binary"] + stage_int_cols,
    }

    # All covariates needed (for NaN cleanup)
    all_cols = list(set(
        col for cols in model_specs.values() for col in cols
    ))

    # Drop rows with NaN in any covariate
    mask = df[all_cols].notna().all(axis=1)
    df_clean = df[mask].copy().reset_index(drop=True)
    y = df_clean["y"].values.astype(float)
    trial_ids = df_clean["trial_id"].values

    print(f"  Fitting on {len(df_clean):,} rows "
          f"({mask.sum() / len(df) * 100:.1f}% of total)")
    print(f"  Event rate: {y.mean():.6f} ({y.sum():.0f} licks)")

    results = {}
    for name, cols in model_specs.items():
        print(f"    Fitting {name} ({len(cols)} predictors)...", end=" ")
        if cols:
            X = np.column_stack([
                np.ones(len(df_clean)), df_clean[cols].values.astype(float)
            ])
        else:
            X = np.ones((len(df_clean), 1))

        glm = sm.GLM(y, X, family=sm.families.Binomial())
        try:
            res = glm.fit(maxiter=500, tol=1e-6)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        results[name] = {
            "llf": res.llf,
            "deviance": res.deviance,
            "null_deviance": res.null_deviance,
            "aic": res.aic,
            "params": res.params,
            "pvalues": res.pvalues,
            "n_obs": int(res.nobs),
            "df_model": int(res.df_model),
            "converged": res.converged,
            "cols": cols,
        }
        pde = 1.0 - res.deviance / res.null_deviance
        print(f"LLF={res.llf:.1f}  %DE={pde:.4f}  "
              f"AIC={res.aic:.1f}  converged={res.converged}")

    return results, df_clean


# =====================================================================
# Section 4: Component decomposition
# =====================================================================

def decompose_components(df_clean, params, cols):
    """Attribute each lick to component drivers.

    Returns DataFrame with one row per lick event and columns for each
    component's fractional contribution.
    """
    covariate_groups = {
        "temporal": [0] + [1 + cols.index(f"time_basis_{i}")
                           for i in range(SPLINE_DF) if f"time_basis_{i}" in cols],
        "ongoing_sensory": [1 + cols.index("log2_tf")] if "log2_tf" in cols else [],
        "change_sensory": [1 + cols.index(c) for c in ["post_change", "change_evidence"]
                           if c in cols],
        "behavioral_history": [1 + cols.index(c)
                               for c in ["rolling_fa_rate", "prev_fa"]
                               if c in cols],
        "hmm_state": [1 + cols.index(c) for c in ["p_engaged", "p_impulsive"]
                      if c in cols],
        "learning": [1 + cols.index(c) for c in cols
                     if c.startswith("stage") and c in cols],
    }

    # Filter to lick events only
    event_mask = df_clean["y"] == 1
    X_events = np.column_stack([
        np.ones(event_mask.sum()),
        df_clean.loc[event_mask, cols].values.astype(float),
    ])

    decomp_rows = []
    for i, (row_idx, row) in enumerate(df_clean[event_mask].iterrows()):
        x = X_events[i]
        eta_total = float(x @ params)

        fractions = {}
        denom = 0.0
        for group_name, col_indices in covariate_groups.items():
            if col_indices:
                eta_group = float(x[col_indices] @ params[col_indices])
            else:
                eta_group = 0.0
            fractions[group_name] = abs(eta_group)
            denom += abs(eta_group)

        # Normalize to fractions
        if denom > 0:
            for k in fractions:
                fractions[k] /= denom

        fractions["outcome"] = row["outcome"]
        fractions["stage"] = row["stage"]
        fractions["bin_time"] = row["bin_time"]
        fractions["eta_total"] = eta_total
        fractions["p_lick"] = expit(eta_total)
        fractions["change_size"] = row["change_size"]
        fractions["session_name"] = row["session_name"]
        fractions["trial_idx"] = row["trial_idx"]
        decomp_rows.append(fractions)

    return pd.DataFrame(decomp_rows)


# =====================================================================
# Section 5: Figure generation
# =====================================================================

def compute_empirical_hazard(df, groupby_col, max_time=20.0, n_bins=200):
    """Compute empirical hazard: P(lick in bin | at risk)."""
    time_edges = np.linspace(0, max_time, n_bins + 1)
    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])
    results = {}

    for group_val, grp in df.groupby(groupby_col):
        hazard = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (grp["bin_time"] >= time_edges[b]) & (
                grp["bin_time"] < time_edges[b + 1]
            )
            n_at_risk = mask.sum()
            n_events = grp.loc[mask, "y"].sum()
            if n_at_risk > 10:
                hazard[b] = n_events / n_at_risk
        results[group_val] = hazard

    return time_centers, results


def make_figure(df, df_clean, model_results, decomp_df):
    """Generate the 8-panel figure (2×4 layout)."""
    fig = plt.figure(figsize=(28, 14))
    gs = gridspec.GridSpec(2, 4, hspace=0.40, wspace=0.35)

    # Color maps
    outcome_cmap = {"fa": "#fb6a4a", "hit": "#6baed6", "miss": "#bdbdbd"}
    component_colors = {
        "temporal": "#4292c6",
        "ongoing_sensory": "#fd8d3c",
        "change_sensory": "#41ab5d",
        "behavioral_history": "#d94801",
        "hmm_state": "#9e9ac8",
        "learning": "#969696",
    }

    # ── Panel A: Empirical hazard by outcome ──────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    tc, hazards = compute_empirical_hazard(df, "outcome")
    for outcome in ["fa", "hit", "miss"]:
        if outcome in hazards:
            ax_a.plot(tc, hazards[outcome], label=outcome.upper(),
                      color=outcome_cmap.get(outcome, "gray"), linewidth=1.5)
    ax_a.set_xlabel("Time from Baseline_ON (s)")
    ax_a.set_ylabel("P(lick | at risk)")
    ax_a.set_title("A  Empirical lick hazard")
    ax_a.legend(frameon=False)
    ax_a.set_xlim(0, 15)

    # ── Panel B: Per-session hazard traces + stage mean ± SEM ─────────
    ax_b = fig.add_subplot(gs[0, 1])
    stage_session_curves = {s: [] for s in STAGE_ORDER}
    n_bins_b = 150
    common_tc = None

    for sname in df["session_name"].unique():
        sess_df = df[df["session_name"] == sname]
        stage = sess_df["stage"].iloc[0]
        if stage not in stage_session_curves:
            continue
        tc_s, hz_s = compute_empirical_hazard(
            sess_df, "outcome", max_time=15.0, n_bins=n_bins_b
        )
        if common_tc is None:
            common_tc = tc_s
        # Combine outcomes for session-level hazard
        all_hz = np.zeros_like(tc_s)
        for outcome_hz in hz_s.values():
            all_hz += outcome_hz
        all_hz /= max(len(hz_s), 1)
        stage_session_curves[stage].append(all_hz)
        # Thin per-session traces
        ax_b.plot(tc_s, all_hz,
                  color=STAGE_COLORS.get(stage, "gray"),
                  linewidth=0.3, alpha=0.25)

    # Mean ± SEM overlay + peakiness metric
    peakiness_stats = {}
    for stage in STAGE_ORDER:
        curves = stage_session_curves.get(stage, [])
        if not curves or common_tc is None:
            continue
        arr = np.array(curves)
        mean_hz = arr.mean(axis=0)
        sem_hz = arr.std(axis=0) / np.sqrt(len(curves))
        color = STAGE_COLORS.get(stage, "gray")
        ax_b.plot(common_tc, mean_hz, color=color,
                  linewidth=2.5, label=stage)
        ax_b.fill_between(common_tc, mean_hz - sem_hz, mean_hz + sem_hz,
                          color=color, alpha=0.2)
        # Peakiness: peak / mean ratio (higher = sharper peak)
        mean_nz = mean_hz[mean_hz > 0]
        if len(mean_nz) > 0:
            peakiness_stats[stage] = mean_hz.max() / mean_nz.mean()

    # Annotate peakiness
    if peakiness_stats:
        txt = "Peak/mean: " + ", ".join(
            f"{s}={v:.1f}" for s, v in peakiness_stats.items())
        ax_b.text(0.02, 0.97, txt, transform=ax_b.transAxes,
                  va="top", fontsize=7,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white",
                            ec="gray", alpha=0.7))

    ax_b.set_xlabel("Time from Baseline_ON (s)")
    ax_b.set_ylabel("P(lick | at risk)")
    ax_b.set_title("B  Hazard by learning stage")
    ax_b.legend(frameon=False, fontsize=8)
    ax_b.set_xlim(0, 15)

    # ── Panel C: Nested model % deviance explained ────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    model_names = ["M0", "M1", "M2", "M3", "M4a", "M4b", "M5"]
    pde_values = []
    for mn in model_names:
        if mn in model_results:
            r = model_results[mn]
            pde_values.append(1.0 - r["deviance"] / r["null_deviance"])
        else:
            pde_values.append(0.0)

    # Incremental deviance
    incremental = [pde_values[0]]
    for i in range(1, len(pde_values)):
        incremental.append(pde_values[i] - pde_values[i - 1])

    component_names = ["Baseline", "Temporal", "Ongoing sensory",
                       "Change sensory", "Behav. history",
                       "HMM state", "Learning"]
    bar_colors = ["#bdbdbd", "#4292c6", "#fd8d3c", "#41ab5d",
                  "#d94801", "#9e9ac8", "#969696"]

    bars = ax_c.bar(range(len(component_names)), incremental,
                    color=bar_colors, edgecolor="white")
    ax_c.set_xticks(range(len(component_names)))
    ax_c.set_xticklabels(component_names, rotation=45, ha="right", fontsize=8)
    ax_c.set_ylabel("Incremental % deviance explained")
    ax_c.set_title("C  Model comparison")

    # Annotate cumulative on top of bars
    cum = 0
    for b, inc in zip(bars, incremental):
        cum += inc
        if inc > 0.0001:
            ax_c.text(b.get_x() + b.get_width() / 2, b.get_height(),
                      f"{inc:.4f}", ha="center", va="bottom", fontsize=7)

    # ── Panel D: Temporal hazard curve ────────────────────────────────
    ax_d = fig.add_subplot(gs[0, 3])
    if "M1" in model_results:
        r1 = model_results["M1"]
        time_grid = np.linspace(0.5 * BIN_SIZE, 15.0, 300)
        basis_grid = dmatrix(
            f"cr(time_grid, df={SPLINE_DF}) - 1",
            {"time_grid": time_grid},
            return_type="dataframe",
        ).values
        X_grid = np.column_stack([np.ones(len(time_grid)), basis_grid])
        eta_temporal = X_grid @ r1["params"]
        ax_d.plot(time_grid, expit(eta_temporal),
                  color="#4292c6", linewidth=2)
        ax_d.set_xlabel("Time from Baseline_ON (s)")
        ax_d.set_ylabel("P(lick | temporal only)")
        ax_d.set_title("D  Learned temporal hazard")
    else:
        ax_d.text(0.5, 0.5, "M1 not fitted", transform=ax_d.transAxes,
                  ha="center")

    # ── Panel E: Component decomposition by outcome (split by Δ) ──────
    ax_e = fig.add_subplot(gs[1, 0])
    if len(decomp_df) > 0:
        comp_cols = ["temporal", "ongoing_sensory", "change_sensory",
                     "behavioral_history", "hmm_state", "learning"]

        # Split hits by change_size: small (≤1.5) vs large (>1.5)
        decomp_df = decomp_df.copy()
        decomp_df["outcome_split"] = decomp_df["outcome"]
        hit_mask = decomp_df["outcome"] == "hit"
        if hit_mask.any():
            small_hit = hit_mask & (decomp_df["change_size"] <= 1.5)
            large_hit = hit_mask & (decomp_df["change_size"] > 1.5)
            decomp_df.loc[small_hit, "outcome_split"] = "hit_small"
            decomp_df.loc[large_hit, "outcome_split"] = "hit_large"

        outcome_order = ["fa", "hit_small", "hit_large"]
        outcome_labels = ["FA", "Hit\n(small Δ)", "Hit\n(large Δ)"]

        x_pos = np.arange(len(outcome_order))
        bottom = np.zeros(len(outcome_order))
        for comp in comp_cols:
            means = []
            for out in outcome_order:
                sub = decomp_df[decomp_df["outcome_split"] == out]
                means.append(sub[comp].mean() if len(sub) > 0 else 0)
            means = np.array(means)
            ax_e.bar(x_pos, means, bottom=bottom,
                     color=component_colors.get(comp, "gray"),
                     label=comp.replace("_", " ").title(),
                     edgecolor="white", linewidth=0.5)
            bottom += means

        ax_e.set_xticks(x_pos)
        ax_e.set_xticklabels(outcome_labels)
        ax_e.set_ylabel("Fraction of log-odds")
        ax_e.set_title("E  Component attribution by outcome")
        ax_e.legend(fontsize=6, frameon=False, loc="upper right")
        ax_e.set_ylim(0, 1.05)

        # Annotate tiny ongoing_sensory values (invisible in stacked bars)
        for i, out in enumerate(outcome_order):
            sub = decomp_df[decomp_df["outcome_split"] == out]
            if len(sub) > 0:
                os_val = sub["ongoing_sensory"].mean()
                if os_val < 0.01:  # Only annotate if truly tiny
                    ax_e.annotate(f"ong.sens.\n{os_val:.1e}",
                                  xy=(i, sub["temporal"].mean()),
                                  fontsize=5, ha="center", va="bottom",
                                  color="#e6550d")

    # ── Panel F: Stage comparison ─────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 1])
    if len(decomp_df) > 0:
        comp_cols_f = ["temporal", "ongoing_sensory", "change_sensory",
                       "behavioral_history", "hmm_state"]
        stage_order = [s for s in STAGE_ORDER if s in decomp_df["stage"].unique()]

        x_pos = np.arange(len(comp_cols_f))
        width = 0.35
        for si, stage in enumerate(stage_order):
            sub = decomp_df[decomp_df["stage"] == stage]
            means = [sub[c].mean() if len(sub) > 0 else 0 for c in comp_cols_f]
            offset = (si - 0.5) * width
            ax_f.bar(x_pos + offset, means, width,
                     color=STAGE_COLORS.get(stage, "gray"),
                     label=stage, edgecolor="white")

        ax_f.set_xticks(x_pos)
        ax_f.set_xticklabels(
            [c.replace("_", " ").title() for c in comp_cols_f],
            rotation=35, ha="right", fontsize=8,
        )
        ax_f.set_ylabel("Mean fraction of log-odds")
        ax_f.set_title("F  Component weights by stage")
        ax_f.legend(frameon=False)

        # Annotate tiny ongoing_sensory bars with exact values
        os_idx = comp_cols_f.index("ongoing_sensory")
        for si, stage in enumerate(stage_order):
            sub = decomp_df[decomp_df["stage"] == stage]
            os_val = sub["ongoing_sensory"].mean() if len(sub) > 0 else 0
            if os_val < 0.01:
                bar_x = os_idx + (si - 0.5) * width
                ax_f.text(bar_x, os_val + 0.005, f"{os_val:.1e}",
                          ha="center", va="bottom", fontsize=5,
                          color="#e6550d", rotation=90)

    # ── Panel G: Example per-trial predicted hazard curves ────────────
    ax_g = fig.add_subplot(gs[1, 2])
    _plot_example_trials(ax_g, df_clean, model_results, outcome_cmap)

    # ── Panel H: Neural CD projection vs model residual ───────────────
    ax_h = fig.add_subplot(gs[1, 3])
    _plot_neural_residual_correlation(ax_h, df_clean, model_results, decomp_df)

    return fig


def _plot_example_trials(ax, df_clean, model_results, outcome_cmap):
    """Panel G: Show predicted hazard curves for a few example trials."""
    full_model = "M5" if "M5" in model_results else "M4b"
    if full_model not in model_results:
        ax.text(0.5, 0.5, "No model", transform=ax.transAxes, ha="center")
        return

    r = model_results[full_model]
    cols = r["cols"]

    # Pick 4 example trials: 1 FA, 1 Hit (small), 1 Hit (large), 1 Miss
    # Select trials near median length for each category
    trial_ids_by_type = {}
    for outcome_want, label in [("fa", "fa"), ("miss", "miss")]:
        sub = df_clean[df_clean["outcome"] == outcome_want]
        if len(sub) == 0:
            continue
        trial_lens = sub.groupby("trial_id").size()
        if len(trial_lens) == 0:
            continue
        median_len = trial_lens.median()
        closest = (trial_lens - median_len).abs().idxmin()
        trial_ids_by_type[label] = closest

    # Hit trials: split by change_size
    hit_data = df_clean[df_clean["outcome"] == "hit"]
    for label, cs_mask in [("hit_small", hit_data["change_size"] <= 1.5),
                           ("hit_large", hit_data["change_size"] > 1.5)]:
        sub = hit_data[cs_mask]
        if len(sub) == 0:
            continue
        trial_lens = sub.groupby("trial_id").size()
        if len(trial_lens) == 0:
            continue
        median_len = trial_lens.median()
        closest = (trial_lens - median_len).abs().idxmin()
        trial_ids_by_type[label] = closest

    label_colors = {
        "fa": ("#fb6a4a", "FA"),
        "hit_small": ("#93c4e0", "Hit (small Δ)"),
        "hit_large": ("#2171b5", "Hit (large Δ)"),
        "miss": ("#bdbdbd", "Miss"),
    }

    for key, tid in trial_ids_by_type.items():
        trial_data = df_clean[df_clean["trial_id"] == tid].copy()
        if len(trial_data) == 0:
            continue
        if cols:
            X_trial = np.column_stack([
                np.ones(len(trial_data)),
                trial_data[cols].values.astype(float),
            ])
        else:
            X_trial = np.ones((len(trial_data), 1))
        eta = X_trial @ r["params"]
        p_lick = expit(eta)
        color, label = label_colors.get(key, ("gray", key))
        ax.plot(trial_data["bin_time"].values, p_lick,
                color=color, label=label, linewidth=1.2, alpha=0.8)
        # Mark the lick event
        lick_bin = trial_data[trial_data["y"] == 1]
        if len(lick_bin) > 0:
            lick_t = lick_bin["bin_time"].values[0]
            lick_p = expit(
                X_trial[trial_data["y"].values == 1][0] @ r["params"]
            )
            ax.plot(lick_t, lick_p, "v", color=color, markersize=8)

    ax.set_xlabel("Time from Baseline_ON (s)")
    ax.set_ylabel("P(lick | at risk)")
    ax.set_title("G  Example trial predicted hazard")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_xlim(0, 15)


def _plot_neural_residual_correlation(ax, df_clean, model_results, decomp_df):
    """Panel H: Correlate per-trial GLM residual with pre-trial CD projection.

    The CD projection measures neural task-state (Hit vs Miss engagement).
    The GLM residual captures lick probability unexplained by behavioral
    covariates. A positive correlation means the neural CD adds predictive
    power beyond the behavioral GLM.
    """
    from scipy.stats import spearmanr
    import glob

    full_model = "M5" if "M5" in model_results else "M4b"
    if full_model not in model_results:
        ax.text(0.5, 0.5, "No model", transform=ax.transAxes, ha="center")
        return

    r = model_results[full_model]
    cols = r["cols"]

    # Compute per-trial predicted P(lick) using survival formula:
    # P(lick in trial) = 1 - prod(1 - h_t)  where h_t is per-bin hazard.
    # This is the correct discrete-time survival probability, NOT mean(h_t).
    if cols:
        X_full = np.column_stack([
            np.ones(len(df_clean)), df_clean[cols].values.astype(float)
        ])
    else:
        X_full = np.ones((len(df_clean), 1))
    df_clean = df_clean.copy()
    df_clean["p_hat"] = expit(X_full @ r["params"])

    # Per-trial survival probability: P(lick) = 1 - prod(1 - h_t)
    trial_pred = df_clean.groupby("trial_id").agg(
        observed=("y", "max"),       # 1 if trial had a lick event
        p_trial_lick=("p_hat", lambda x: 1.0 - np.prod(1.0 - x.values)),
        session_name=("session_name", "first"),
        trial_idx=("trial_idx", "first"),
        outcome=("outcome", "first"),
        stage=("stage", "first"),
    ).reset_index()

    # Residual: observed - predicted trial-level P(lick)
    trial_pred["residual"] = trial_pred["observed"] - trial_pred["p_trial_lick"]

    # Load CD projections from cache
    cd_cache_dir = os.path.join(CACHE_DIR, "cd_results")
    cd_files = sorted(glob.glob(os.path.join(cd_cache_dir, "*_hit_miss_cd.npz")))

    if not cd_files:
        ax.text(0.5, 0.5, "No CD cache\nRun a_coding_direction.py first",
                transform=ax.transAxes, ha="center", fontsize=9)
        return

    # For each session with a CD cache, load the mean pre-change CD projection
    # per trial from the cached projections.
    # mean_hit / mean_miss are time-course arrays; compute scalar separation
    # as the average difference in a post-pulse window (0.1–0.5 s).
    POST_CD_WINDOW = (0.1, 0.5)
    cd_rows = []
    for cd_file in cd_files:
        d = np.load(cd_file, allow_pickle=True)
        # Extract session name from filename
        fname = os.path.basename(cd_file)
        sname = int(fname.split("_hit_miss_cd")[0].rsplit("_", 1)[-1])

        # The CD cache has mean hit/miss/fa projections but not per-trial.
        # Use the cached hit vs miss mean projections as session-level CD strength.
        if "mean_hit" in d and "mean_miss" in d:
            mh = np.asarray(d["mean_hit"]).ravel()
            mm = np.asarray(d["mean_miss"]).ravel()
            bc = np.asarray(d["bin_centers"]).ravel() if "bin_centers" in d else None
            if bc is not None and len(bc) == len(mh):
                win_mask = (bc >= POST_CD_WINDOW[0]) & (bc <= POST_CD_WINDOW[1])
                if win_mask.any():
                    sep = float(mh[win_mask].mean() - mm[win_mask].mean())
                else:
                    sep = float(mh.mean() - mm.mean())
            else:
                sep = float(mh.mean() - mm.mean())
            cd_rows.append({
                "session_name": sname,
                "cd_separation": sep,
                "cv_accuracy": float(d["cv_accuracy"]) if "cv_accuracy" in d else np.nan,
            })

    if not cd_rows:
        ax.text(0.5, 0.5, "No valid CD data", transform=ax.transAxes,
                ha="center", fontsize=9)
        return

    cd_df = pd.DataFrame(cd_rows)

    # Merge: per-session mean residual with CD separation
    session_resid = trial_pred.groupby("session_name").agg(
        mean_residual=("residual", "mean"),
        lick_rate=("observed", "mean"),
        n_trials=("trial_id", "count"),
        stage=("stage", "first"),
    ).reset_index()
    session_resid["session_name"] = session_resid["session_name"].astype(int)

    merged = session_resid.merge(cd_df, on="session_name", how="inner")

    if len(merged) < 5:
        ax.text(0.5, 0.5, f"Too few sessions ({len(merged)})",
                transform=ax.transAxes, ha="center", fontsize=9)
        return

    # Plot CD separation vs GLM mean residual
    for stage in STAGE_ORDER:
        sub = merged[merged["stage"] == stage]
        if len(sub) == 0:
            continue
        ax.scatter(sub["cd_separation"], sub["mean_residual"],
                   color=STAGE_COLORS.get(stage, "gray"),
                   label=stage, s=50, alpha=0.7, edgecolors="white",
                   linewidth=0.5)

    # Correlation
    rho, p = spearmanr(merged["cd_separation"], merged["mean_residual"])
    ax.set_xlabel("CD separation (Hit - Miss mean)")
    ax.set_ylabel("Mean residual (obs - P_survival)")
    ax.set_title("H  Neural CD vs behavioral residual")
    ax.legend(frameon=False, fontsize=8)

    # Add stats text
    ax.text(0.05, 0.95, f"ρ = {rho:.3f}\np = {p:.3f}\nn = {len(merged)}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                      alpha=0.8))


# =====================================================================
# Section 6: Stats export
# =====================================================================

def export_stats(model_results, decomp_df, out_dir):
    """Save model comparison and decomposition stats."""
    os.makedirs(out_dir, exist_ok=True)

    # Model comparison table
    rows = []
    model_names = ["M0", "M1", "M2", "M3", "M4a", "M4b", "M5"]
    prev_llf = None
    prev_df = 0
    for mn in model_names:
        if mn not in model_results:
            continue
        r = model_results[mn]
        pde = 1.0 - r["deviance"] / r["null_deviance"]
        row = {
            "model": mn,
            "n_predictors": len(r["cols"]),
            "log_likelihood": r["llf"],
            "deviance": r["deviance"],
            "pct_deviance_explained": pde,
            "aic": r["aic"],
            "converged": r["converged"],
        }
        # LR test vs previous model
        if prev_llf is not None:
            lr_stat = 2 * (r["llf"] - prev_llf)
            df_diff = len(r["cols"]) + 1 - prev_df
            if df_diff > 0 and lr_stat > 0:
                row["lr_chi2"] = lr_stat
                row["lr_df"] = df_diff
                row["lr_pvalue"] = chi2.sf(lr_stat, df_diff)
        prev_llf = r["llf"]
        prev_df = len(r["cols"]) + 1
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join(out_dir, "hazard_glm_model_comparison.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved stats: {stats_path}")

    # Decomposition summary
    if len(decomp_df) > 0:
        comp_cols = ["temporal", "ongoing_sensory", "change_sensory",
                     "behavioral_history", "hmm_state", "learning"]
        summary = decomp_df.groupby(["outcome", "stage"])[comp_cols].mean()
        decomp_path = os.path.join(out_dir, "hazard_glm_decomposition.csv")
        summary.to_csv(decomp_path)
        print(f"  Saved decomposition: {decomp_path}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rebuild dataset cache")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel workers for dataset build (default 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("Fig35k: Lick-Hazard GLM")
    print("=" * 60)

    # Step 1: Build dataset
    print("\n[1/5] Building hazard dataset...")
    df = build_hazard_dataset(force=args.force, n_workers=args.n_workers)

    # Sanity checks
    n_trials = df[["session_name", "trial_idx"]].drop_duplicates().shape[0]
    n_sessions = df["session_name"].nunique()
    n_licks = int(df["y"].sum())
    print(f"  Dataset: {len(df):,} rows, {n_trials:,} trials, "
          f"{n_sessions} sessions, {n_licks:,} lick events")

    # Verify no aborts
    assert "abort" not in df["outcome"].unique(), \
        "Abort trials should be excluded!"
    # Verify FA has no post_change
    fa_post = df.loc[df["outcome"] == "fa", "post_change"]
    assert fa_post.sum() == 0, "FA trials must have post_change=0 everywhere"
    print("  Sanity checks passed")

    # Step 2: Add spline basis
    print("\n[2/5] Adding spline basis...")
    df = add_spline_basis(df)

    # Step 3: Fit models
    print("\n[3/5] Fitting nested models...")
    model_results, df_clean = fit_nested_hazard_models(df)

    # Step 4: Component decomposition (using full model M5, or M4b if M5 fails)
    print("\n[4/5] Decomposing lick components...")
    full_model = "M5" if "M5" in model_results else "M4b"
    if full_model in model_results:
        decomp_df = decompose_components(
            df_clean,
            model_results[full_model]["params"],
            model_results[full_model]["cols"],
        )
        print(f"  Decomposed {len(decomp_df)} lick events using {full_model}")
    else:
        decomp_df = pd.DataFrame()
        print("  WARNING: No full model available for decomposition")

    # Step 5: Figure
    print("\n[5/5] Generating figure...")
    fig = make_figure(df, df_clean, model_results, decomp_df)
    paths = save_figure(fig, "fig35k_lick_hazard_glm", "07_advanced")
    print(f"  Saved: {paths}")

    # Export stats
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "07_advanced",
    )
    export_stats(model_results, decomp_df, out_dir)

    print("\nDone.")
