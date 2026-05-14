"""Expert-anchor HMM diagnostic for BG_046.

Fits a Bernoulli GLM-HMM on Expert sessions only, then:
  1. Compares BIC / K-selection to the joint (all-sessions) model
  2. Measures psychometric and GLM-weight alignment between models
     (cosine similarity after optimal K! state permutation matching)
  3. Decodes ALL sessions with the fixed Expert-only model weights
  4. Tests whether the Expert model degrades on Learning sessions
     (held-out LL per trial, state fractions, Cohen's κ vs joint model)

Output
------
  data/hmm/BG_046/expert_only/        model PKLs + state assignments
  FIGURES/behavior/BG_046/hmm/
      expert_vs_joint_diagnostic.png
      expert_vs_joint_stats.csv

Usage
-----
    py scripts/analysis/behavior/expert_anchor_diagnostic.py
    py scripts/analysis/behavior/expert_anchor_diagnostic.py --force
"""

import argparse
import gc
import itertools
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import mannwhitneyu, spearmanr

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from visdetect.analysis.config import (
    HMM_DIR, HMM_MODEL_SEL_PATH, HMM_STATE_COLORS, HMM_STATE_ORDER,
    PKL_DIR, ROOT, STAGE_COLORS, SUBJECT,
    load_staging_manifest,
)
from visdetect.analysis.hmm import (
    GLMHMM, GLMHMMConfig,
    auto_label_states, fit_best_model, prepare_session_data,
)
from visdetect.core.session import load_session
from visdetect.viz.plotting import despine, set_style

# ── Paths ────────────────────────────────────────────────────────────
EXPERT_DIR = Path(HMM_DIR) / "expert_only"
FIG_DIR    = Path(ROOT) / "FIGURES" / "behavior" / SUBJECT / "hmm"

N_RESTARTS = 20
K_RANGE    = [2, 3, 4]
SEED       = 0

# ── Helpers ──────────────────────────────────────────────────────────

def _lighten_hex(hex_color: str, factor: float = 0.45) -> str:
    """Return a lighter version of a hex color (mix with white)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r2 = int(r + (255 - r) * factor)
    g2 = int(g + (255 - g) * factor)
    b2 = int(b + (255 - b) * factor)
    return f"#{r2:02x}{g2:02x}{b2:02x}"


def _build_label_colors(all_labels: list) -> dict:
    """Assign colors handling numbered state variants (e.g. 'Engaged_1', 'Engaged_2').

    Canonical names ('Disengaged', 'Engaged', 'Impulsive') get their project
    colors directly.  Numbered variants of the same base get progressively
    lighter shades so they remain distinguishable.
    """
    from collections import defaultdict
    colors: dict = {}
    base_groups: dict = defaultdict(list)
    for lbl in dict.fromkeys(all_labels):   # deduplicate while preserving order
        base = lbl.split("_")[0]
        base_groups[base].append(lbl)
    for base, group in base_groups.items():
        base_col = HMM_STATE_COLORS.get(base, "#888888")
        for i, lbl in enumerate(group):
            colors[lbl] = _lighten_hex(base_col, factor=0.45 * i) if i > 0 else base_col
    return colors


def _load_sessions(manifest_rows, label=""):
    """Load sessions from manifest rows; return list of session dicts."""
    sessions_data, meta = [], []
    for _, row in manifest_rows.iterrows():
        sname = str(row["session_name"])
        candidates = list(Path(PKL_DIR).glob(f"*{sname}*.pkl"))
        if not candidates:
            print(f"  SKIP {sname}: pkl not found")
            continue
        try:
            sess = load_session(str(candidates[0]))
            sd = prepare_session_data(sess)
            if len(sd["y"]) < 10:
                print(f"  SKIP {sname}: only {len(sd['y'])} valid trials")
                del sess; gc.collect()
                continue
            sd["session_name"] = sname
            sessions_data.append(sd)
            meta.append({"session_name": sname, "stage": row.get("stage", ""), "session_idx": row.get("session_idx", 0)})
            del sess; gc.collect()
        except Exception as exc:
            print(f"  SKIP {sname}: {exc}")
    print(f"  Loaded {len(sessions_data)} {label}sessions ({sum(len(s['y']) for s in sessions_data)} trials)")
    return sessions_data, pd.DataFrame(meta)


def _align_states(model_a: GLMHMM, model_b: GLMHMM) -> list[int]:
    """Return a permutation mapping model_b states → model_a states.

    Finds the K! permutation of model_b's state indices that maximises
    mean cosine similarity between matched weight vectors.
    """
    Ka, Kb = model_a.n_states, model_b.n_states
    K = min(Ka, Kb)
    best_perm, best_score = list(range(K)), -np.inf
    for perm in itertools.permutations(range(Kb)):
        score = 0.0
        for i, j in enumerate(perm[:K]):
            wa = model_a.weights[i]
            wb = model_b.weights[j]
            denom = (np.linalg.norm(wa) * np.linalg.norm(wb))
            score += (np.dot(wa, wb) / denom) if denom > 0 else 0.0
        if score > best_score:
            best_score, best_perm = score, list(perm)
    return best_perm[:K]


def _cosine_sim(w1: np.ndarray, w2: np.ndarray) -> float:
    d = np.linalg.norm(w1) * np.linalg.norm(w2)
    return float(np.dot(w1, w2) / d) if d > 0 else 0.0


def _per_session_ll(model: GLMHMM, sessions_data: list[dict]) -> dict[str, float]:
    """Mean log-likelihood per trial per session."""
    out = {}
    for sd in sessions_data:
        _, _, ll = model._e_step_session(sd["y"], sd["X"])
        out[sd["session_name"]] = ll / max(len(sd["y"]), 1)
    return out


def _decode_all(model: GLMHMM, sessions_data: list[dict], labels: list[str]) -> pd.DataFrame:
    """Viterbi + posteriors for all sessions; return combined DataFrame."""
    rows = []
    for sd in sessions_data:
        states = model.most_likely_states(sd)
        posts  = model.state_posteriors(sd)
        df = sd["df"].copy()
        df.insert(0, "session_name", sd["session_name"])
        df["hmm_state_exp"] = states
        df["hmm_state_label_exp"] = [labels[s] for s in states]
        for k in range(model.n_states):
            df[f"p_exp_{k}"] = posts[:, k]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's κ between two integer label arrays."""
    labels = np.unique(np.concatenate([a, b]))
    n = len(a)
    if n == 0:
        return np.nan
    p_o = np.mean(a == b)
    p_e = sum((np.mean(a == k) * np.mean(b == k)) for k in labels)
    return float((p_o - p_e) / (1 - p_e)) if (1 - p_e) > 1e-9 else 1.0


def _mw_effect(x, y):
    """Mann-Whitney U + rank-biserial r (two-sided)."""
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan
    U, p = mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - 2 * U / (len(x) * len(y))
    return float(U), float(p), float(r)


def _bootstrap_spearman_ci(x, y, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    rho, _ = spearmanr(x, y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r, _ = spearmanr(x[idx], y[idx])
        boots.append(r)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return float(rho), float(ci_lo), float(ci_hi)


# ── Main ─────────────────────────────────────────────────────────────

def main(force: bool = False):
    set_style(context="talk")
    EXPERT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 0. Load manifest ────────────────────────────────────────────
    manifest = load_staging_manifest(qc_only=True)
    manifest = manifest.sort_values("session_idx").reset_index(drop=True)
    expert_rows  = manifest[manifest["stage"] == "Expert"]
    all_rows     = manifest

    print(f"Manifest: {len(manifest)} sessions  ({len(expert_rows)} Expert)")

    # ── 1. Load existing joint model ────────────────────────────────
    sel_df   = pd.read_csv(HMM_MODEL_SEL_PATH)
    best_K_joint = int(sel_df.loc[sel_df["bic"].idxmin(), "K"])
    joint_model  = GLMHMM.load(Path(HMM_DIR) / f"model_K{best_K_joint}.pkl")
    joint_labels_path = Path(HMM_DIR) / f"state_labels_K{best_K_joint}.json"
    with open(joint_labels_path) as f:
        joint_labels = json.load(f)["labels"]
    print(f"Joint model: K={best_K_joint}, labels={joint_labels}")

    # ── 2. Fit Expert-only model (or load from cache) ───────────────
    exp_sel_path = EXPERT_DIR / "model_selection.csv"
    if exp_sel_path.exists() and not force:
        print("Expert-only model already fitted (use --force to refit).")
        exp_sel_df   = pd.read_csv(exp_sel_path)
        best_K_exp   = int(exp_sel_df.loc[exp_sel_df["bic"].idxmin(), "K"])
        expert_model = GLMHMM.load(EXPERT_DIR / f"model_K{best_K_exp}.pkl")
        with open(EXPERT_DIR / f"state_labels_K{best_K_exp}.json") as f:
            exp_labels = json.load(f)["labels"]
    else:
        print("\nFitting Expert-only model...")
        print(f"  Loading {len(expert_rows)} Expert sessions...")
        exp_sessions, _ = _load_sessions(expert_rows, label="Expert ")
        if len(exp_sessions) == 0:
            print("ERROR: No Expert sessions loaded.")
            sys.exit(1)

        cfg = GLMHMMConfig(max_iter=200, n_restarts=N_RESTARTS, verbose=True)
        expert_model, exp_sel_df, exp_all_models = fit_best_model(
            exp_sessions, K_range=K_RANGE, config=cfg, verbose=True, seed=SEED,
        )
        best_K_exp = expert_model.n_states
        exp_labels = auto_label_states(expert_model)

        # Save
        exp_sel_df.to_csv(exp_sel_path, index=False)
        for Kv, m in exp_all_models.items():
            lbl = auto_label_states(m)
            m.save(EXPERT_DIR / f"model_K{Kv}.pkl")
            with open(EXPERT_DIR / f"state_labels_K{Kv}.json", "w") as f:
                json.dump({"K": Kv, "labels": lbl}, f, indent=2)
        print(f"Expert-only best K={best_K_exp}, labels={exp_labels}")

    # ── 3. State alignment (cosine similarity) ──────────────────────
    K_match = min(best_K_joint, best_K_exp)
    perm    = _align_states(joint_model, expert_model)  # expert state perm → joint order
    cos_sims = {
        joint_labels[i]: _cosine_sim(joint_model.weights[i], expert_model.weights[perm[i]])
        for i in range(K_match)
    }
    print(f"\nState alignment (cosine sim): {cos_sims}")

    # Build a per-label color map that handles numbered variants ('Engaged_1', 'Engaged_2')
    all_labels_combined = list(dict.fromkeys(joint_labels + exp_labels))
    label_colors = _build_label_colors(all_labels_combined)

    # ── 4. Decode ALL sessions with Expert-anchor model ─────────────
    assign_path = EXPERT_DIR / "state_assignments_all_sessions.csv"
    if assign_path.exists() and not force:
        print("Loading cached Expert-anchor assignments...")
        exp_assign_df = pd.read_csv(assign_path, dtype={"session_name": str})
    else:
        print("\nDecoding all sessions with Expert-anchor model...")
        all_sessions, all_meta = _load_sessions(all_rows, label="all ")
        exp_assign_df = _decode_all(expert_model, all_sessions, exp_labels)
        exp_assign_df.to_csv(assign_path, index=False)

    # ── 5. Per-session statistics ────────────────────────────────────
    print("\nComputing per-session statistics...")
    all_sessions_reload, all_meta = _load_sessions(all_rows)

    joint_ll_map = _per_session_ll(joint_model, all_sessions_reload)
    exp_ll_map   = _per_session_ll(expert_model, all_sessions_reload)

    # Load joint assignments for κ
    joint_assign_path = Path(HMM_DIR) / f"state_assignments_K{best_K_joint}.csv"
    joint_assign_df   = pd.read_csv(joint_assign_path, dtype={"session_name": str})

    rows_stats = []
    for _, mrow in manifest.iterrows():
        sn    = str(mrow["session_name"])
        stage = mrow["stage"]
        sidx  = mrow["session_idx"]

        jll = joint_ll_map.get(sn, np.nan)
        ell = exp_ll_map.get(sn, np.nan)

        # Expert-anchor state fractions
        exp_sub = exp_assign_df[exp_assign_df["session_name"] == sn]
        fracs = {lbl: (exp_sub["hmm_state_label_exp"] == lbl).mean()
                 for lbl in exp_labels}
        # Engaged posterior
        eng_col = f"p_exp_{exp_labels.index('Engaged')}" if "Engaged" in exp_labels else None
        p_engaged = exp_sub[eng_col].mean() if eng_col and eng_col in exp_sub.columns else np.nan

        # Cohen's κ
        j_sub = joint_assign_df[joint_assign_df["session_name"] == sn]
        kappa = np.nan
        if len(j_sub) > 0 and len(exp_sub) > 0:
            min_len = min(len(j_sub), len(exp_sub))
            j_states = j_sub["hmm_state"].values[:min_len]
            # Remap Expert-anchor states to joint ordering via perm
            e_states_raw = exp_sub["hmm_state_exp"].values[:min_len]
            inv_perm = {v: k for k, v in enumerate(perm[:K_match])}
            e_states_remapped = np.array([inv_perm.get(s, s) for s in e_states_raw])
            kappa = _cohens_kappa(j_states, e_states_remapped)

        row = {"session_name": sn, "stage": stage, "session_idx": sidx,
               "joint_ll_per_trial": jll, "exp_ll_per_trial": ell,
               "p_engaged_exp": p_engaged, "cohens_kappa": kappa}
        row.update({f"frac_{lbl}": fracs.get(lbl, np.nan) for lbl in exp_labels})
        rows_stats.append(row)

    stats_df = pd.DataFrame(rows_stats)
    l_mask = stats_df["stage"] == "Learning"
    e_mask = stats_df["stage"] == "Expert"

    # ── 6. Statistical tests ─────────────────────────────────────────
    test_records = []

    # T1: Held-out LL — Expert-anchor model, Learning vs Expert sessions
    U, p, r = _mw_effect(
        stats_df.loc[l_mask, "exp_ll_per_trial"].dropna().values,
        stats_df.loc[e_mask, "exp_ll_per_trial"].dropna().values,
    )
    test_records.append({"test": "exp_anchor_ll_learning_vs_expert",
                         "statistic_name": "U", "statistic_value": U,
                         "p_value": p, "effect_size_name": "rank_biserial_r",
                         "effect_size_value": r,
                         "n": int(l_mask.sum() + e_mask.sum()),
                         "n_per_group": f"L:{int(l_mask.sum())}|E:{int(e_mask.sum())}",
                         "notes": "Per-session mean LL per trial; Expert model trained on Expert only"})

    # T2: P(Engaged | Expert model) trajectory
    valid = stats_df[["session_idx", "p_engaged_exp"]].dropna()
    if len(valid) >= 5:
        rho, ci_lo, ci_hi = _bootstrap_spearman_ci(
            valid["session_idx"].values, valid["p_engaged_exp"].values)
        _, sp_p = spearmanr(valid["session_idx"], valid["p_engaged_exp"])
        test_records.append({"test": "p_engaged_vs_session_idx",
                              "statistic_name": "rho", "statistic_value": rho,
                              "p_value": sp_p, "effect_size_name": "rho",
                              "effect_size_value": rho,
                              "n": len(valid),
                              "n_per_group": f"CI=[{ci_lo:.3f},{ci_hi:.3f}]",
                              "notes": "Spearman rho + 1000-resample bootstrap CI (seed=42)"})

    # T3: P(Engaged) Learning vs Expert (categorical cross-check)
    U2, p2, r2 = _mw_effect(
        stats_df.loc[l_mask, "p_engaged_exp"].dropna().values,
        stats_df.loc[e_mask, "p_engaged_exp"].dropna().values,
    )
    test_records.append({"test": "p_engaged_learning_vs_expert",
                         "statistic_name": "U", "statistic_value": U2,
                         "p_value": p2, "effect_size_name": "rank_biserial_r",
                         "effect_size_value": r2,
                         "n": int(l_mask.sum() + e_mask.sum()),
                         "n_per_group": f"L:{int(l_mask.sum())}|E:{int(e_mask.sum())}",
                         "notes": "Categorical cross-check for trajectory test"})

    # T4: State fractions Learning vs Expert (Holm-Bonferroni)
    frac_pvals = []
    for lbl in exp_labels:
        col = f"frac_{lbl}"
        if col not in stats_df.columns:
            continue
        Uf, pf, rf = _mw_effect(
            stats_df.loc[l_mask, col].dropna().values,
            stats_df.loc[e_mask, col].dropna().values,
        )
        frac_pvals.append((lbl, Uf, pf, rf))
    # Holm-Bonferroni correction
    frac_pvals_sorted = sorted(frac_pvals, key=lambda x: x[2])
    m = len(frac_pvals_sorted)
    for rank, (lbl, Uf, pf, rf) in enumerate(frac_pvals_sorted):
        p_adj = min(pf * (m - rank), 1.0)
        test_records.append({"test": f"frac_{lbl}_learning_vs_expert",
                              "statistic_name": "U", "statistic_value": Uf,
                              "p_value": pf, "p_value_adjusted": p_adj,
                              "effect_size_name": "rank_biserial_r",
                              "effect_size_value": rf,
                              "n": int(l_mask.sum() + e_mask.sum()),
                              "n_per_group": f"L:{int(l_mask.sum())}|E:{int(e_mask.sum())}",
                              "notes": "Holm-Bonferroni corrected across 3 state fractions"})

    # T5: Cohen's κ Learning vs Expert
    U3, p3, r3 = _mw_effect(
        stats_df.loc[l_mask, "cohens_kappa"].dropna().values,
        stats_df.loc[e_mask, "cohens_kappa"].dropna().values,
    )
    test_records.append({"test": "cohens_kappa_learning_vs_expert",
                         "statistic_name": "U", "statistic_value": U3,
                         "p_value": p3, "effect_size_name": "rank_biserial_r",
                         "effect_size_value": r3,
                         "n": int(l_mask.sum() + e_mask.sum()),
                         "n_per_group": f"L:{int(l_mask.sum())}|E:{int(e_mask.sum())}",
                         "notes": "κ between joint and Expert-anchor assignments; low κ in Learning = Expert model misses early structure"})

    # T6: ΔBIC (descriptive)
    bic_joint = sel_df.loc[sel_df["K"] == best_K_joint, "bic"].values[0]
    bic_exp   = exp_sel_df.loc[exp_sel_df["K"] == best_K_exp, "bic"].values[0]
    test_records.append({"test": "delta_bic_joint_vs_expert_model",
                         "statistic_name": "delta_bic", "statistic_value": bic_exp - bic_joint,
                         "p_value": np.nan, "effect_size_name": "kass_raftery",
                         "effect_size_value": abs(bic_exp - bic_joint),
                         "n": np.nan, "n_per_group": "",
                         "notes": "ΔBIC>10=very strong; models trained on different data (all vs Expert); descriptive only"})

    tests_out = pd.DataFrame(test_records)
    tests_out.to_csv(FIG_DIR / "expert_vs_joint_stats.csv", index=False)
    print("\nStatistical tests:")
    for _, row in tests_out.iterrows():
        p_str = f"p={row['p_value']:.4f}" if pd.notna(row['p_value']) else "descriptive"
        r_str = f"r={row['effect_size_value']:.3f}" if pd.notna(row.get('effect_size_value')) else ""
        print(f"  {row['test']}: {p_str} {r_str}")

    # ── 7. Figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 13))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32,
                             height_ratios=[1, 1, 1.3])

    stim_log2 = np.array([0.0, np.log2(1.25), np.log2(1.35),
                           np.log2(1.5), np.log2(2.0), np.log2(4.0)])

    # ── Panel A: BIC/AIC vs K ────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(sel_df["K"], sel_df["bic"], "o-", color="#2166ac", lw=2, ms=7, label="Joint BIC")
    ax_a.plot(sel_df["K"], sel_df["aic"], "s--", color="#2166ac", lw=1.5, ms=5,
              alpha=0.6, label="Joint AIC")
    ax_a.plot(exp_sel_df["K"], exp_sel_df["bic"], "o-", color="#d6604d", lw=2, ms=7,
              label="Expert-only BIC")
    ax_a.plot(exp_sel_df["K"], exp_sel_df["aic"], "s--", color="#d6604d", lw=1.5, ms=5,
              alpha=0.6, label="Expert-only AIC")
    ax_a.axvline(best_K_joint, color="#2166ac", lw=0.8, ls=":", alpha=0.7)
    ax_a.axvline(best_K_exp,   color="#d6604d", lw=0.8, ls=":", alpha=0.7)
    ax_a.set_xlabel("Number of states (K)")
    ax_a.set_ylabel("Information criterion")
    ax_a.set_xticks(K_RANGE)
    ax_a.legend(fontsize=7, frameon=False)
    ax_a.set_title("A. Model selection: joint vs Expert-only", fontweight="bold", fontsize=11)
    despine(ax_a)

    # ── Panel D: Held-out LL per trial by stage ──────────────────────
    ax_d = fig.add_subplot(gs[0, 1])
    for model_name, ll_col, color, ls_kwargs in [
        ("Joint",        "joint_ll_per_trial", "#2166ac", {}),
        ("Expert-anchor","exp_ll_per_trial",   "#d6604d", {}),
    ]:
        for si, (stage, mask, xpos) in enumerate(
            [("Learning", l_mask, 0), ("Expert", e_mask, 1)]
        ):
            vals = stats_df.loc[mask, ll_col].dropna().values
            offset = -0.2 if model_name == "Joint" else 0.2
            jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
            ax_d.scatter(xpos + offset + jitter, vals, s=25, color=color, alpha=0.6, zorder=3)
            ax_d.plot([xpos + offset - 0.08, xpos + offset + 0.08],
                      [np.median(vals)] * 2, color=color, lw=2.5, zorder=4)
    # Legend via proxy
    from matplotlib.lines import Line2D
    ax_d.legend(handles=[
        Line2D([0], [0], color="#2166ac", lw=2, label="Joint"),
        Line2D([0], [0], color="#d6604d", lw=2, label="Expert-anchor"),
    ], fontsize=8, frameon=False, loc="upper right")
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["Learning\n(incl. Naive)", "Expert"])
    ax_d.set_ylabel("Mean LL per trial (nats)")
    ax_d.set_title("D. Held-out LL by stage", fontweight="bold", fontsize=11)
    if pd.notna(p):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax_d.text(0.5, 0.97, f"L vs E: {stars} (r={r:.2f})",
                  transform=ax_d.transAxes, ha="center", va="top", fontsize=9, color="dimgrey")
    despine(ax_d)

    # ── Panel B: Psychometric curves ─────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    for model, labels, ls, lw in [
        (joint_model,  joint_labels,  "-",   2.0),
        (expert_model, exp_labels,    "--",  1.5),
    ]:
        for k, lbl in enumerate(labels):
            color = label_colors.get(lbl, "#888888")
            p_lick = [expit(model.weights[k] @ np.array([1.0, sv] + [0.0] * (model.n_features - 2)))
                      for sv in stim_log2]
            ax_b.plot(stim_log2, p_lick, color=color, ls=ls, lw=lw, alpha=0.9,
                      label=lbl if ls == "-" else None)
    # Legend: one patch per unique label + linestyle key
    from matplotlib.patches import Patch
    seen = {}
    for lbl in dict.fromkeys(joint_labels + exp_labels):
        if lbl not in seen:
            seen[lbl] = Patch(facecolor=label_colors.get(lbl, "#888"), label=lbl)
    handles = list(seen.values())
    handles += [Line2D([0], [0], color="k", ls="-",  lw=2, label="Joint"),
                Line2D([0], [0], color="k", ls="--", lw=1.5, label="Expert-only")]
    ax_b.legend(handles=handles, fontsize=7, frameon=False, ncol=2, loc="upper left")
    ax_b.set_xticks(stim_log2)
    ax_b.set_xticklabels(["1.0", "1.25", "1.35", "1.5", "2.0", "4.0"], fontsize=8)
    ax_b.set_xlabel("Change size (TF ratio)")
    ax_b.set_ylabel("P(lick)")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_title("B. Psychometric curves: joint (—) vs Expert-anchor (- -)",
                   fontweight="bold", fontsize=11)
    despine(ax_b)

    # ── Panel C: GLM weights comparison ─────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    feat_names = joint_model.feature_names
    D = joint_model.n_features
    x_pos = np.arange(D)
    bar_w = 0.35
    for ki in range(K_match):
        jlbl = joint_labels[ki]
        elbl = exp_labels[perm[ki]]
        color_j = label_colors.get(jlbl, "#888888")
        color_e = _lighten_hex(color_j, factor=0.45)
        offset = (ki - K_match / 2 + 0.5) * (bar_w * 2 + 0.05)
        ax_c.bar(x_pos + offset,         joint_model.weights[ki],  bar_w,
                 color=color_j, label=f"{jlbl} (joint)", edgecolor="white", lw=0.3)
        ax_c.bar(x_pos + offset + bar_w, expert_model.weights[perm[ki]], bar_w,
                 color=color_e, label=f"{elbl} (expert)", edgecolor="white", lw=0.3, hatch="//")
    ax_c.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(feat_names, rotation=25, ha="right", fontsize=8)
    ax_c.set_ylabel("GLM weight")
    ax_c.set_title("C. GLM weights (filled=joint, hatch=Expert-only)",
                   fontweight="bold", fontsize=11)
    # Cosine sim annotation — show matched label pairs
    sim_txt = "  ".join(
        f"{joint_labels[i]}↔{exp_labels[perm[i]]}: cos={cos_sims[joint_labels[i]]:.2f}"
        for i in range(K_match)
    )
    ax_c.text(0.02, 0.97, sim_txt, transform=ax_c.transAxes,
              va="top", fontsize=7, color="dimgrey")
    ax_c.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
    despine(ax_c)

    # ── Panel E: Learning score trajectory (full width) ──────────────
    ax_e = fig.add_subplot(gs[2, :])
    valid_e = stats_df.dropna(subset=["p_engaged_exp"])
    ax_e.scatter(valid_e["session_idx"], valid_e["p_engaged_exp"],
                 c=[STAGE_COLORS[s] for s in valid_e["stage"]],
                 s=55, zorder=3, edgecolors="white", lw=0.5)
    ax_e.plot(valid_e["session_idx"], valid_e["p_engaged_exp"],
              color="#6baed6", lw=1.2, alpha=0.7)
    # Stage background
    for stage, color in STAGE_COLORS.items():
        sidxs = manifest.loc[manifest["stage"] == stage, "session_idx"].values
        if len(sidxs):
            ax_e.axvspan(sidxs.min() - 0.5, sidxs.max() + 0.5,
                         color=color, alpha=0.08, zorder=0)
    # Spearman annotation
    if len(valid_e) >= 5:
        rho_e, ci_lo_e, ci_hi_e = _bootstrap_spearman_ci(
            valid_e["session_idx"].values, valid_e["p_engaged_exp"].values)
        _, sp_p_e = spearmanr(valid_e["session_idx"], valid_e["p_engaged_exp"])
        ax_e.text(0.02, 0.95,
                  f"ρ = {rho_e:.2f} [95% CI {ci_lo_e:.2f}–{ci_hi_e:.2f}], p = {sp_p_e:.4f}",
                  transform=ax_e.transAxes, va="top", fontsize=9, color="dimgrey",
                  bbox=dict(fc="white", ec="none", alpha=0.7))
    ax_e.set_xlabel("Session index (chronological)")
    ax_e.set_ylabel("P(Engaged | Expert model)")
    ax_e.set_ylim(-0.02, 1.02)
    ax_e.set_title("E. Learning score: P(Engaged | Expert-anchor model) across sessions",
                   fontweight="bold", fontsize=11)
    ax_e.text(0.02, 0.02, f"n = {len(valid_e)} sessions", transform=ax_e.transAxes,
              va="bottom", fontsize=8, color="gray")
    despine(ax_e)

    fig.savefig(FIG_DIR / "expert_vs_joint_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {FIG_DIR / 'expert_vs_joint_diagnostic.png'}")
    print(f"Stats saved:  {FIG_DIR / 'expert_vs_joint_stats.csv'}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Refit model even if cached files exist.")
    args = parser.parse_args()
    main(force=args.force)
