"""Leave-One-Session-Out cross-validation for GLM-HMM.

Evaluates held-out log-likelihood and prediction accuracy for each K
to complement BIC-based model selection.

Supports parallel fold processing via ``ProcessPoolExecutor`` with a
``tqdm`` progress bar (see ``--n-workers``).

Usage
-----
    python scripts/analysis/behavior/hmm_cross_validation.py \\
        --manifest data/BG_046_staging_manifest_v2.csv \\
        --pkl-dir  data/pkls/BG_046 \\
        --data-out data/hmm/BG_046 \\
        --out      FIGURES/behavior/BG_046/hmm \\
        --exclude-qc-fail

    # Parallel folds (12 workers):
    python scripts/analysis/behavior/hmm_cross_validation.py \\
        ... --n-workers 12

Outputs
-------
  data/hmm/BG_046/loso_cv_results.csv   — per-fold held-out metrics
  FIGURES/…/hmm/loso_cv_summary.png     — bar chart of test LL per K
"""

import argparse
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from visdetect.core.session import load_session
from visdetect.analysis.hmm import GLMHMM, GLMHMMConfig, prepare_session_data
from visdetect.viz.plotting import set_style, despine


# =====================================================================
# Per-fold worker (picklable for ProcessPoolExecutor)
# =====================================================================

@dataclass
class FoldTask:
    """Everything needed to run a single LOSO fold (picklable)."""
    sessions_data: list     # list of session dicts
    fold_idx: int           # which session to hold out
    K: int
    n_restarts: int
    max_iter: int
    seed: int


def _run_single_fold(task: FoldTask) -> dict:
    """Run one LOSO fold: fit on N-1 sessions, evaluate on held-out.

    Returns a result dict matching the columns of the final DataFrame,
    plus a ``status`` key for error handling.
    """
    result = {
        "fold": task.fold_idx,
        "K": task.K,
        "status": "ok",
        "message": "",
    }
    try:
        held_out = task.sessions_data[task.fold_idx]
        train = [s for i, s in enumerate(task.sessions_data) if i != task.fold_idx]
        sname = held_out.get("session_name", f"session_{task.fold_idx}")
        n_features = task.sessions_data[0]["X"].shape[1]

        cfg = GLMHMMConfig(
            max_iter=task.max_iter,
            n_restarts=task.n_restarts,
            verbose=False,
        )

        best_ll = -np.inf
        best_model = None
        for r in range(task.n_restarts):
            model = GLMHMM(task.K, n_features, config=cfg)
            try:
                ll = model.fit(train, seed=task.seed + r * 137 + task.fold_idx * 7)
            except Exception:
                continue
            if ll > best_ll:
                best_ll = ll
                best_model = model

        if best_model is None:
            result["status"] = "error"
            result["message"] = f"All {task.n_restarts} restarts failed"
            return result

        # Evaluate on held-out session
        test_ll = best_model.log_likelihood([held_out])
        n_test = len(held_out["y"])

        # Prediction accuracy
        states = best_model.most_likely_states(held_out)
        X_test = held_out["X"]
        y_test = held_out["y"]
        p_lick = np.array([
            expit(best_model.weights[states[t]] @ X_test[t])
            for t in range(n_test)
        ])
        pred_choice = (p_lick >= 0.5).astype(float)
        accuracy = float(np.mean(pred_choice == y_test))

        result.update({
            "held_out_session": sname,
            "n_trials_test": n_test,
            "train_ll": best_ll,
            "test_ll": test_ll,
            "test_ll_per_trial": test_ll / max(n_test, 1),
            "test_accuracy": accuracy,
        })

    except Exception as exc:
        result["status"] = "error"
        result["message"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return result


def _print_fold_result(r: dict, n_sessions: int) -> None:
    """Print a one-line status for a completed fold."""
    fi = r.get("fold", "?")
    K = r.get("K", "?")
    if r["status"] == "ok":
        sname = r.get("held_out_session", "?")
        ll = r.get("test_ll_per_trial", float("nan"))
        acc = r.get("test_accuracy", float("nan"))
        print(f"  [OK]  K={K} fold {fi + 1}/{n_sessions}  "
              f"held-out={sname}  LL/trial={ll:.3f}  acc={acc:.3f}")
    else:
        print(f"  [ERR] K={K} fold {fi + 1}/{n_sessions}: {r.get('message', '')}")


def main():
    parser = argparse.ArgumentParser(
        description="LOSO cross-validation for GLM-HMM."
    )
    parser.add_argument("--manifest", default=None,
                        help="Path to staging manifest CSV (default: canonical path).")
    parser.add_argument("--pkl-dir", required=True)
    parser.add_argument("--data-out", default="data/hmm")
    parser.add_argument("--out", default="FIGURES/behavior/hmm")
    parser.add_argument("--K-min", type=int, default=2)
    parser.add_argument("--K-max", type=int, default=5)
    parser.add_argument("--n-restarts", type=int, default=10,
                        help="Restarts per fold (lower than fitting for speed).")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--exclude-qc-fail", action="store_true",
                        help="DEPRECATED: SESSION_FILTER handles QC.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Bypass SESSION_FILTER and use the full manifest.")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers for fold processing "
                             "(default 1 = serial).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_out = Path(args.data_out)
    data_out.mkdir(parents=True, exist_ok=True)
    set_style(context="talk")

    # ------------------------------------------------------------------
    # 1. Load sessions
    # ------------------------------------------------------------------
    manifest = load_staging_manifest(
        manifest_path=args.manifest,
        apply_filter=not getattr(args, 'no_filter', False),
    )

    pkl_dir = Path(args.pkl_dir)
    sessions_data = []
    print(f"Loading {len(manifest)} sessions...")
    for _, row in manifest.iterrows():
        sname = str(row["session_name"])
        if "path" in row and pd.notna(row["path"]):
            pkl_path = Path(row["path"])
            if not pkl_path.exists():
                pkl_path = pkl_dir / pkl_path.name
        else:
            candidates = list(pkl_dir.glob(f"*{sname}*.pkl"))
            pkl_path = candidates[0] if candidates else None

        if pkl_path is None or not pkl_path.exists():
            continue
        try:
            session = load_session(str(pkl_path))
            sd = prepare_session_data(session)
            if len(sd["y"]) < 10:
                continue
            sd["session_name"] = sname
            sessions_data.append(sd)
        except Exception as exc:
            print(f"  SKIP {sname}: {exc}")

    print(f"Loaded {len(sessions_data)} sessions "
          f"({sum(len(s['y']) for s in sessions_data)} trials)")

    # ------------------------------------------------------------------
    # 2. Run LOSO for each K
    # ------------------------------------------------------------------
    K_range = list(range(args.K_min, args.K_max + 1))
    n_sessions = len(sessions_data)
    all_cv = []

    for K in K_range:
        print(f"\n{'='*50}")
        print(f"LOSO Cross-Validation: K={K}  ({n_sessions} folds, "
              f"workers={args.n_workers})")
        print(f"{'='*50}")

        # Build one FoldTask per fold
        fold_tasks = [
            FoldTask(
                sessions_data=sessions_data,
                fold_idx=fi,
                K=K,
                n_restarts=args.n_restarts,
                max_iter=args.max_iter,
                seed=args.seed,
            )
            for fi in range(n_sessions)
        ]

        fold_results: list = []

        if args.n_workers <= 1:
            # ---- Serial ----
            for ft in tqdm(fold_tasks, desc=f"K={K} folds", unit="fold"):
                r = _run_single_fold(ft)
                fold_results.append(r)
                _print_fold_result(r, n_sessions)
        else:
            # ---- Parallel ----
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                futures = {pool.submit(_run_single_fold, ft): ft
                           for ft in fold_tasks}
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"K={K} folds", unit="fold"):
                    ft = futures[future]
                    try:
                        r = future.result()
                    except Exception as exc:
                        r = {
                            "fold": ft.fold_idx,
                            "K": K,
                            "status": "error",
                            "message": str(exc),
                        }
                    fold_results.append(r)
                    _print_fold_result(r, n_sessions)

        # Collect successful folds into a DataFrame
        ok_records = [r for r in fold_results if r["status"] == "ok"]
        n_err = sum(1 for r in fold_results if r["status"] == "error")
        if n_err:
            print(f"  WARNING: {n_err}/{n_sessions} folds failed for K={K}")

        if ok_records:
            cv_df = pd.DataFrame(ok_records)
            cv_df["K"] = K
            all_cv.append(cv_df)

            mean_ll = cv_df["test_ll_per_trial"].mean()
            mean_acc = cv_df["test_accuracy"].mean()
            print(f"  K={K}  mean test LL/trial={mean_ll:.3f}  "
                  f"mean accuracy={mean_acc:.3f}")
        else:
            print(f"  K={K}  NO successful folds")

    if not all_cv:
        print("\nNo successful CV folds — nothing to save.")
        return

    results = pd.concat(all_cv, ignore_index=True)
    csv_path = data_out / "loso_cv_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # ------------------------------------------------------------------
    # 3. Summary table
    # ------------------------------------------------------------------
    summary = (
        results.groupby("K")
        .agg(
            mean_test_ll_per_trial=("test_ll_per_trial", "mean"),
            std_test_ll_per_trial=("test_ll_per_trial", "std"),
            mean_accuracy=("test_accuracy", "mean"),
            std_accuracy=("test_accuracy", "std"),
            n_folds=("fold", "count"),
        )
        .reset_index()
    )
    print("\n" + "=" * 60)
    print("LOSO CV Summary")
    print("=" * 60)
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Test LL per trial
    ax1.errorbar(
        summary["K"], summary["mean_test_ll_per_trial"],
        yerr=summary["std_test_ll_per_trial"],
        fmt="o-", color="tab:blue", linewidth=2, markersize=8, capsize=5,
    )
    ax1.set_xlabel("Number of states (K)")
    ax1.set_ylabel("Test log-likelihood / trial")
    ax1.set_title("LOSO Cross-Validation: Held-Out LL")
    ax1.set_xticks(K_range)
    despine(ax1)

    # Prediction accuracy
    ax2.errorbar(
        summary["K"], summary["mean_accuracy"],
        yerr=summary["std_accuracy"],
        fmt="s-", color="tab:orange", linewidth=2, markersize=8, capsize=5,
    )
    ax2.set_xlabel("Number of states (K)")
    ax2.set_ylabel("Prediction accuracy")
    ax2.set_title("LOSO Cross-Validation: Accuracy")
    ax2.set_xticks(K_range)
    ax2.set_ylim(0.4, 1.0)
    despine(ax2)

    plt.tight_layout()
    fig_path = out_dir / "loso_cv_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {fig_path}")


if __name__ == "__main__":
    main()
