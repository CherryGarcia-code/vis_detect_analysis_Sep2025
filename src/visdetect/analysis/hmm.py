"""Bernoulli GLM-HMM for behavioral state identification.

Implements the Generalized Linear Model Hidden Markov Model framework
described in:

    Ashwood, Roy, Stone et al. (2022). "Mice alternate between discrete
    strategies during perceptual decision-making."
    Nature Neuroscience 25, 201-212.

Each hidden state k defines a distinct behavioral strategy, parameterized
by a logistic regression mapping trial covariates to P(lick):

    P(lick_t = 1 | z_t = k, x_t) = sigmoid(w_k^T x_t)

Default covariates x_t = [1, log2(change_size), prev_choice, prev_reward].

The model is fit via Expectation-Maximization (EM):
    E-step  : Forward-backward algorithm for state posteriors
    M-step  : Transition matrix MLE + weighted logistic regression per state

Multi-session fitting shares all parameters across sessions while allowing
independent state sequences per session (forward-backward resets at session
boundaries).

Usage
-----
    from visdetect.analysis.hmm import GLMHMM, prepare_session_data, fit_best_model

    sessions_data = [prepare_session_data(s) for s in sessions]
    best_model, selection_df = fit_best_model(sessions_data, K_range=[2, 3, 4, 5])
    states = best_model.most_likely_states(sessions_data[0])
"""

from __future__ import annotations

import json
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from tqdm import tqdm

from visdetect.analysis.behavior import get_trial_dataframe
from visdetect.core.session import Session

# =====================================================================
# Constants
# =====================================================================
_EPS = 1e-300  # prevent log(0)

FEATURE_NAMES = ["bias", "stimulus", "prev_choice", "prev_reward", "prev_early_lick"]


# =====================================================================
# Numerics helpers
# =====================================================================

def _log_bernoulli(y: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Log probability of Bernoulli observations given logits.

    log p(y | logits) = y * logits - log(1 + exp(logits))

    Uses ``np.logaddexp(0, logits)`` for numerically-stable softplus.
    """
    return y * logits - np.logaddexp(0, logits)


def _nll_and_grad(w, X, y, gamma_k, l2):
    """Negative weighted log-likelihood and gradient for one state's GLM.

    Minimised during the M-step to update w_k.
    """
    logits = X @ w                              # (T,)
    p = expit(logits)                           # (T,)
    nll = -np.sum(gamma_k * (y * logits - np.logaddexp(0, logits)))
    grad = -X.T @ (gamma_k * (y - p))
    if l2 > 0:
        nll += 0.5 * l2 * np.dot(w, w)
        grad += l2 * w
    return nll, grad


# =====================================================================
# Data preparation
# =====================================================================

def prepare_session_data(
    session: Session,
    *,
    exclude_outcomes: Sequence[str] = ("abort", "ref"),
) -> Dict[str, Any]:
    """Extract binary choice vector *y* and covariate matrix *X* from a Session.

    Parameters
    ----------
    session : Session
        Loaded session object.
    exclude_outcomes : sequence of str
        Trial outcomes to discard (default: abort, ref).

    Returns
    -------
    dict with keys:
        y               : ndarray (T,)  binary choice (1=licked, 0=no-lick)
        X               : ndarray (T, D) design matrix [bias, stim, prev_choice, prev_reward]
        df              : DataFrame      trial-level metadata for the *included* trials
        session_name    : str
        feature_names   : list[str]
    """
    df = get_trial_dataframe(session)
    if df.empty:
        return {"y": np.array([]), "X": np.empty((0, len(FEATURE_NAMES))),
                "df": df, "session_name": session.session_name or "",
                "feature_names": list(FEATURE_NAMES)}

    # Filter excluded outcomes
    mask = ~df["outcome"].isin([o.lower() for o in exclude_outcomes])
    df = df[mask].reset_index(drop=True)
    if df.empty:
        return {"y": np.array([]), "X": np.empty((0, len(FEATURE_NAMES))),
                "df": df, "session_name": session.session_name or "",
                "feature_names": list(FEATURE_NAMES)}

    # --- Binary choice: 1 = licked, 0 = no-lick ---
    y = (df["is_hit"] | df["is_fa"]).astype(float).values

    # --- Stimulus strength ---
    # log2(change_size); catch trials have change_size=1 -> log2(1)=0.
    # We keep the *scheduled* change_size for ALL trials, including
    # early-lick ("fa") ones.  The early lick happened before the
    # change, so the stimulus didn't drive the response — but the
    # scheduled value is still an unconditional trial property and
    # leaving it intact lets the model learn ~zero stimulus weight
    # for impulsive states instead of the degenerate large-negative
    # weight that arose when stim was forced to 0 for FAs.
    stim = np.log2(np.clip(df["change_size"].values.astype(float), 1.0, None))
    stim = np.nan_to_num(stim, nan=0.0)

    # --- History features ---
    prev_choice = np.zeros(len(df))
    prev_choice[1:] = y[:-1]

    prev_reward = np.zeros(len(df))
    hit_on_go = (df["is_hit"] & df["is_go"]).astype(float).values
    prev_reward[1:] = hit_on_go[:-1]

    # --- Impulsivity history ---
    # Whether the *previous* trial was an early/anticipatory lick.
    # This lets the model capture serial impulsivity (runs of early
    # licks) via a dedicated coefficient, disentangling impulsive
    # responding from stimulus-driven detection.
    prev_early_lick = np.zeros(len(df))
    prev_early_lick[1:] = df["is_fa"].values[:-1].astype(float)

    # --- Design matrix ---
    X = np.column_stack([
        np.ones(len(df)),   # bias / intercept
        stim,               # stimulus strength
        prev_choice,        # previous choice
        prev_reward,        # previous reward
        prev_early_lick,    # previous trial was early lick
    ])

    return {
        "y": y,
        "X": X,
        "df": df,
        "session_name": session.session_name or "",
        "feature_names": list(FEATURE_NAMES),
    }


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class GLMHMMConfig:
    """Hyper-parameters for GLM-HMM fitting."""
    max_iter: int = 200
    tol: float = 1e-4
    n_restarts: int = 20
    self_transition_prior: float = 0.8
    l2_penalty: float = 0.0
    glm_max_iter: int = 100
    verbose: bool = True


# =====================================================================
# Core model
# =====================================================================

class GLMHMM:
    """Bernoulli GLM-HMM for trial-by-trial behavioral state inference.

    Parameters
    ----------
    n_states : int
        Number of latent states K.
    n_features : int
        Dimensionality D of the covariate vector (including bias).
    config : GLMHMMConfig, optional
        Training hyper-parameters.
    """

    def __init__(self, n_states: int, n_features: int,
                 config: Optional[GLMHMMConfig] = None):
        self.n_states = n_states
        self.n_features = n_features
        self.config = config or GLMHMMConfig()

        # Parameters (set by _init_params or fit)
        self._weights: Optional[np.ndarray] = None   # (K, D)
        self._log_A: Optional[np.ndarray] = None     # (K, K)
        self._log_pi: Optional[np.ndarray] = None    # (K,)

        # Diagnostics
        self.train_ll_history: List[float] = []
        self.converged: bool = False
        self.feature_names: List[str] = list(FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """GLM weight matrix (K, D)."""
        return self._weights

    @property
    def transition_matrix(self) -> np.ndarray:
        """Transition matrix A (K, K) in probability space."""
        return np.exp(self._log_A)

    @property
    def initial_state_dist(self) -> np.ndarray:
        """Initial state distribution pi (K,)."""
        return np.exp(self._log_pi)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self, seed: Optional[int] = None, smart: bool = False):
        """Randomly initialise model parameters."""
        rng = np.random.default_rng(seed)
        K, D = self.n_states, self.n_features

        # Transition matrix: strong self-transition
        p_self = self.config.self_transition_prior
        A = np.full((K, K), (1 - p_self) / max(K - 1, 1))
        np.fill_diagonal(A, p_self)
        # Add small noise
        A += rng.uniform(0, 0.02, (K, K))
        A /= A.sum(axis=1, keepdims=True)
        self._log_A = np.log(A + _EPS)

        # Initial state: uniform
        pi = np.ones(K) / K
        self._log_pi = np.log(pi + _EPS)

        # GLM weights
        if smart and K >= 2:
            # Spread bias values evenly across logit space
            bias_vals = np.linspace(-2, 2, K)
            self._weights = np.zeros((K, D))
            self._weights[:, 0] = bias_vals
            # Small stimulus sensitivity to all states
            if D > 1:
                self._weights[:, 1] = rng.uniform(0.5, 1.5, K)
            # Add noise
            self._weights += rng.normal(0, 0.1, (K, D))
        else:
            self._weights = rng.normal(0, 0.5, (K, D))

    # ------------------------------------------------------------------
    # Emission model
    # ------------------------------------------------------------------

    def _emission_log_likes(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute emission log-likelihoods for all states.

        Returns (T, K) array: log P(y_t | z_t=k, x_t).
        """
        T = len(y)
        K = self.n_states
        logits = X @ self._weights.T   # (T, K)
        # log P(y | logit) = y * logit - softplus(logit)
        ll = np.empty((T, K))
        for k in range(K):
            ll[:, k] = _log_bernoulli(y, logits[:, k])
        return ll

    # ------------------------------------------------------------------
    # Forward-backward
    # ------------------------------------------------------------------

    def _forward(self, log_likes: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward pass in log-space.

        Parameters
        ----------
        log_likes : (T, K) emission log-likelihoods.

        Returns
        -------
        log_alpha : (T, K)
        log_marginal : float   log P(y_{1:T})
        """
        T, K = log_likes.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = self._log_pi + log_likes[0]
        for t in range(1, T):
            # log_alpha[t, k] = log_likes[t, k]
            #   + logsumexp_j( log_alpha[t-1, j] + log_A[j, k] )
            log_alpha[t] = log_likes[t] + logsumexp(
                log_alpha[t - 1, :, None] + self._log_A, axis=0
            )
        log_marginal = float(logsumexp(log_alpha[-1]))
        return log_alpha, log_marginal

    def _backward(self, log_likes: np.ndarray) -> np.ndarray:
        """Backward pass in log-space.

        Returns log_beta (T, K).
        """
        T, K = log_likes.shape
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            # log_beta[t, k] = logsumexp_j(
            #     log_A[k, j] + log_likes[t+1, j] + log_beta[t+1, j] )
            log_beta[t] = logsumexp(
                self._log_A + log_likes[t + 1] + log_beta[t + 1], axis=1
            )
        return log_beta

    # ------------------------------------------------------------------
    # E-step (single session)
    # ------------------------------------------------------------------

    def _e_step_session(
        self, y: np.ndarray, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run E-step for one session.

        Returns
        -------
        gamma   : (T, K)  state posterior  P(z_t=k | y_{1:T})
        xi_sum  : (K, K)  summed transition posterior  sum_t P(z_t=i, z_{t+1}=j | y)
        log_marginal : float
        """
        T = len(y)
        if T == 0:
            K = self.n_states
            return (np.empty((0, K)), np.zeros((K, K)), 0.0)

        log_likes = self._emission_log_likes(y, X)         # (T, K)
        log_alpha, log_Z = self._forward(log_likes)         # (T,K), scalar
        log_beta = self._backward(log_likes)                # (T, K)

        # State posteriors
        log_gamma = log_alpha + log_beta - log_Z
        gamma = np.exp(log_gamma)
        # Clamp for safety
        gamma = np.clip(gamma, _EPS, None)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # Transition sufficient statistics (vectorised)
        if T > 1:
            # (T-1, K, 1) + (1, K, K) + (T-1, 1, K) + (T-1, 1, K) - scalar
            log_xi = (
                log_alpha[:-1, :, None]
                + self._log_A[None, :, :]
                + log_likes[1:, None, :]
                + log_beta[1:, None, :]
                - log_Z
            )
            xi_sum = np.exp(log_xi).sum(axis=0)  # (K, K)
        else:
            xi_sum = np.zeros((self.n_states, self.n_states))

        return gamma, xi_sum, log_Z

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _fit_glm_state(self, X: np.ndarray, y: np.ndarray,
                       gamma_k: np.ndarray) -> np.ndarray:
        """Weighted logistic regression for one state (scipy L-BFGS-B)."""
        w0 = self._weights[0] if self._weights is not None else np.zeros(self.n_features)
        w0 = np.array(w0, dtype=float)
        result = minimize(
            _nll_and_grad,
            x0=w0,
            args=(X, y, gamma_k, self.config.l2_penalty),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.config.glm_max_iter, "ftol": 1e-8},
        )
        return result.x

    def _m_step(
        self,
        sessions_data: List[Dict[str, Any]],
        all_gamma: List[np.ndarray],
        total_xi: np.ndarray,
        total_init: np.ndarray,
    ):
        """Update all parameters from sufficient statistics."""
        K = self.n_states

        # --- Transition matrix ---
        # Row-normalise xi
        row_sums = total_xi.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, _EPS)
        A = total_xi / row_sums
        self._log_A = np.log(A + _EPS)

        # --- Initial state distribution ---
        pi = total_init / max(total_init.sum(), _EPS)
        self._log_pi = np.log(pi + _EPS)

        # --- GLM weights per state ---
        y_all = np.concatenate([s["y"] for s in sessions_data])
        X_all = np.concatenate([s["X"] for s in sessions_data])
        gamma_all = np.concatenate(all_gamma)

        for k in range(K):
            self._weights[k] = self._fit_glm_state(X_all, y_all, gamma_all[:, k])

    # ------------------------------------------------------------------
    # EM fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        sessions_data: List[Dict[str, Any]],
        seed: Optional[int] = None,
        smart_init: bool = False,
    ) -> float:
        """Fit via EM on (possibly multiple) sessions.

        Parameters
        ----------
        sessions_data : list of dicts from ``prepare_session_data``.
        seed : int, optional
            Random seed for parameter initialisation.
        smart_init : bool
            Use heuristic initialisation (spread biases).

        Returns
        -------
        best_ll : float
            Final training log-likelihood.
        """
        self._init_params(seed=seed, smart=smart_init)
        K = self.n_states
        prev_ll = -np.inf
        self.train_ll_history = []

        for iteration in range(self.config.max_iter):
            # ---- E-step ----
            all_gamma: List[np.ndarray] = []
            total_xi = np.zeros((K, K))
            total_init = np.zeros(K)
            total_ll = 0.0

            for s in sessions_data:
                if len(s["y"]) == 0:
                    continue
                gamma, xi_sum, ll = self._e_step_session(s["y"], s["X"])
                all_gamma.append(gamma)
                total_xi += xi_sum
                total_init += gamma[0]
                total_ll += ll

            self.train_ll_history.append(total_ll)

            # ---- Convergence check ----
            rel_change = abs(total_ll - prev_ll) / max(abs(prev_ll), 1.0)
            if iteration > 0 and rel_change < self.config.tol:
                self.converged = True
                if self.config.verbose:
                    print(f"  EM converged at iteration {iteration + 1}  "
                          f"(LL={total_ll:.2f}, delta={rel_change:.2e})")
                break
            prev_ll = total_ll

            # ---- M-step ----
            self._m_step(sessions_data, all_gamma, total_xi, total_init)

        if not self.converged and self.config.verbose:
            print(f"  EM did not converge after {self.config.max_iter} iterations "
                  f"(LL={total_ll:.2f})")

        return total_ll

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def most_likely_states(self, session_data: Dict[str, Any]) -> np.ndarray:
        """Viterbi decoding — most likely state sequence.

        Returns integer array (T,).
        """
        y, X = session_data["y"], session_data["X"]
        T = len(y)
        if T == 0:
            return np.array([], dtype=int)

        K = self.n_states
        log_likes = self._emission_log_likes(y, X)

        # log delta & back-pointers
        log_delta = np.empty((T, K))
        psi = np.zeros((T, K), dtype=int)

        log_delta[0] = self._log_pi + log_likes[0]
        for t in range(1, T):
            candidates = log_delta[t - 1, :, None] + self._log_A  # (K, K)
            psi[t] = candidates.argmax(axis=0)
            log_delta[t] = log_likes[t] + candidates.max(axis=0)

        # Back-trace
        z = np.empty(T, dtype=int)
        z[-1] = int(log_delta[-1].argmax())
        for t in range(T - 2, -1, -1):
            z[t] = psi[t + 1, z[t + 1]]
        return z

    def state_posteriors(self, session_data: Dict[str, Any]) -> np.ndarray:
        """Posterior state probabilities gamma (T, K) via forward-backward."""
        y, X = session_data["y"], session_data["X"]
        if len(y) == 0:
            return np.empty((0, self.n_states))
        gamma, _, _ = self._e_step_session(y, X)
        return gamma

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def log_likelihood(self, sessions_data: List[Dict[str, Any]]) -> float:
        """Total log-likelihood across all sessions."""
        ll = 0.0
        for s in sessions_data:
            if len(s["y"]) == 0:
                continue
            _, _, ll_s = self._e_step_session(s["y"], s["X"])
            ll += ll_s
        return ll

    def n_params(self) -> int:
        """Number of free parameters."""
        K, D = self.n_states, self.n_features
        n_glm = K * D                            # GLM weights
        n_trans = K * (K - 1)                     # transition (rows sum to 1)
        n_init = K - 1                            # initial dist (sums to 1)
        return n_glm + n_trans + n_init

    def _total_trials(self, sessions_data: List[Dict[str, Any]]) -> int:
        return sum(len(s["y"]) for s in sessions_data)

    def bic(self, sessions_data: List[Dict[str, Any]]) -> float:
        """Bayesian Information Criterion (lower is better)."""
        ll = self.log_likelihood(sessions_data)
        n = self._total_trials(sessions_data)
        return -2 * ll + self.n_params() * np.log(max(n, 1))

    def aic(self, sessions_data: List[Dict[str, Any]]) -> float:
        """Akaike Information Criterion (lower is better)."""
        ll = self.log_likelihood(sessions_data)
        return -2 * ll + 2 * self.n_params()

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def state_psychometrics(
        self,
        stim_values: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """P(lick) vs stimulus for each state (with prev_choice=0, prev_reward=0).

        Returns DataFrame with columns: state, stimulus, p_lick.
        """
        if stim_values is None:
            stim_values = np.array([0, 0.32, 0.43, 0.58, 1.0, 2.0])  # log2 of task values
        rows = []
        for k in range(self.n_states):
            for sv in stim_values:
                # Build x with the right dimensionality; set all
                # history/covariate terms to 0 (baseline psychometric).
                x = np.zeros(self.n_features)
                x[0] = 1.0   # bias
                x[1] = sv    # stimulus
                p = float(expit(self._weights[k] @ x))
                rows.append({"state": k, "stimulus": sv, "p_lick": p})
        return pd.DataFrame(rows)

    def sort_states_by_bias(self):
        """Re-order states so that State 0 has the lowest bias (most disengaged)
        and State K-1 has the highest bias (most lick-biased).

        Modifies the model in-place.
        """
        order = np.argsort(self._weights[:, 0])  # ascending bias
        self._weights = self._weights[order]
        self._log_A = self._log_A[order][:, order]
        self._log_pi = self._log_pi[order]

    def summary(self) -> str:
        """Human-readable model summary."""
        K, D = self.n_states, self.n_features
        lines = [
            f"GLM-HMM  K={K}  D={D}  params={self.n_params()}  "
            f"converged={self.converged}",
            "",
            "GLM weights (rows=states, cols=features):",
            f"  features: {self.feature_names}",
        ]
        for k in range(K):
            w_str = "  ".join(f"{v:+.3f}" for v in self._weights[k])
            lines.append(f"  State {k}: [{w_str}]")
        lines.append("")
        lines.append("Transition matrix A:")
        A = self.transition_matrix
        for k in range(K):
            row = "  ".join(f"{v:.3f}" for v in A[k])
            lines.append(f"  [{row}]")
        lines.append("")
        lines.append(f"Initial state dist: "
                      f"[{' '.join(f'{v:.3f}' for v in self.initial_state_dist)}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        """Save fitted model to pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "GLMHMM":
        """Load a fitted model from pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)


# =====================================================================
# Model selection
# =====================================================================

@dataclass
class KFitTask:
    """Parameters for fitting a single K value."""
    K: int
    sessions_data: List[Dict[str, Any]]
    n_features: int
    config: GLMHMMConfig
    n_restarts: int
    base_seed: int = 0


def _fit_single_K(task: KFitTask) -> Tuple[int, Optional["GLMHMM"], float, int]:
    """Worker function to fit a single K value with multiple random restarts.

    Parameters
    ----------
    task : KFitTask
        Contains K, sessions_data, config, n_restarts, base_seed.

    Returns
    -------
    K : int
    best_model : GLMHMM or None
    best_ll : float
    n_failures : int
    """
    K = task.K
    best_ll_K = -np.inf
    best_model_K = None
    n_failures = 0

    for r in range(task.n_restarts):
        model = GLMHMM(K, task.n_features, config=task.config)
        smart = (r == 0)  # first restart uses heuristic init
        seed = task.base_seed + r * 137 + K * 7
        try:
            ll = model.fit(task.sessions_data, seed=seed, smart_init=smart)
            if ll > best_ll_K:
                best_ll_K = ll
                best_model_K = model
        except Exception:
            n_failures += 1
            continue

    if best_model_K is not None:
        best_model_K.sort_states_by_bias()

    return K, best_model_K, best_ll_K, n_failures


def fit_best_model(
    sessions_data: List[Dict[str, Any]],
    K_range: Sequence[int] = (2, 3, 4, 5),
    config: Optional[GLMHMMConfig] = None,
    verbose: bool = True,
    n_workers: int = 1,
    seed: int = 0,
) -> Tuple["GLMHMM", pd.DataFrame, Dict[int, "GLMHMM"]]:
    """Fit GLM-HMMs for each K, selecting the best by BIC.

    For each K, ``config.n_restarts`` random restarts are run and the
    restart with the highest log-likelihood is kept. K values can be
    fit in parallel using multiple workers.

    Parameters
    ----------
    sessions_data : list of session dicts.
    K_range : sequence of int
        Candidate numbers of states to try.
    config : GLMHMMConfig, optional
    verbose : bool
    n_workers : int, default=1
        Number of parallel workers. If > 1, K values are fit in parallel.
    seed : int, default=0
        Base random seed for reproducibility.

    Returns
    -------
    best_model : GLMHMM
        Fitted model with the lowest BIC.
    selection_df : DataFrame
        Columns: K, best_ll, bic, aic, n_params.
    all_models : dict[int, GLMHMM]
        Mapping from K to the best fitted model at that K.
    """
    cfg = config or GLMHMMConfig()
    cfg_copy = GLMHMMConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
    # suppress per-restart verbosity for workers
    cfg_copy.verbose = False

    n_features = sessions_data[0]["X"].shape[1] if len(sessions_data) > 0 else len(FEATURE_NAMES)

    # Build tasks for each K
    tasks = [
        KFitTask(
            K=K,
            sessions_data=sessions_data,
            n_features=n_features,
            config=cfg_copy,
            n_restarts=cfg.n_restarts,
            base_seed=seed,
        )
        for K in K_range
    ]

    records = []
    all_models: Dict[int, GLMHMM] = {}

    if n_workers > 1:
        # Parallel execution
        if verbose:
            print(f"\nFitting {len(K_range)} K values in parallel with {n_workers} workers...")
            print(f"Each K: {cfg.n_restarts} random restarts\n")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_fit_single_K, task) for task in tasks]
            for future in tqdm(futures, desc="Fitting K values", disable=not verbose):
                try:
                    K, best_model_K, best_ll_K, n_failures = future.result()
                    
                    if best_model_K is not None:
                        bic_val = best_model_K.bic(sessions_data)
                        aic_val = best_model_K.aic(sessions_data)
                        all_models[K] = best_model_K
                        records.append({
                            "K": K,
                            "best_ll": best_ll_K,
                            "bic": bic_val,
                            "aic": aic_val,
                            "n_params": best_model_K.n_params(),
                        })
                        if verbose:
                            msg = f"K={K}  LL={best_ll_K:.2f}  BIC={bic_val:.2f}  AIC={aic_val:.2f}"
                            if n_failures > 0:
                                msg += f"  ({n_failures}/{cfg.n_restarts} restarts failed)"
                            print(f"  {msg}")
                    else:
                        if verbose:
                            print(f"  K={K}: All restarts failed.")
                except Exception as exc:
                    if verbose:
                        print(f"  K={K}: FAILED - {exc}")
    else:
        # Sequential execution (original behavior)
        for task in tasks:
            K = task.K
            if verbose:
                print(f"\n{'='*50}")
                print(f"Fitting K={K} states  ({cfg.n_restarts} restarts)")
                print(f"{'='*50}")
            
            K, best_model_K, best_ll_K, n_failures = _fit_single_K(task)

            if best_model_K is not None:
                bic_val = best_model_K.bic(sessions_data)
                aic_val = best_model_K.aic(sessions_data)
                all_models[K] = best_model_K
                records.append({
                    "K": K,
                    "best_ll": best_ll_K,
                    "bic": bic_val,
                    "aic": aic_val,
                    "n_params": best_model_K.n_params(),
                })
                if verbose:
                    msg = f">>> K={K}  best LL={best_ll_K:.2f}  BIC={bic_val:.2f}  AIC={aic_val:.2f}"
                    if n_failures > 0:
                        msg += f"  ({n_failures}/{cfg.n_restarts} restarts failed)"
                    print(f"  {msg}")

    selection_df = pd.DataFrame(records)
    if selection_df.empty:
        raise RuntimeError("All model fits failed.")

    best_K = int(selection_df.loc[selection_df["bic"].idxmin(), "K"])
    best_model = all_models[best_K]
    if verbose:
        print(f"\n*** Best model: K={best_K} (by BIC) ***\n")
        print(best_model.summary())

    return best_model, selection_df, all_models


# =====================================================================
# Auto-labelling
# =====================================================================

def auto_label_states(model: GLMHMM) -> List[str]:
    """Assign human-readable labels based on each state's psychometric profile.

    Heuristic (after states are sorted by ascending bias):
      - Compute P(lick | stimulus=0) and P(lick | stimulus=max)
      - "Disengaged": low P at both catch and max stimulus
      - "Engaged":    low P at catch, high P at max stimulus
      - "Biased":     high P at catch (always licking)

    Falls back to generic "State_k" if K > 3 and thresholds are ambiguous.
    """
    K = model.n_states
    D = model.n_features
    labels = []
    # Psychometric at catch (stim=0) and high stim (stim=2.0, i.e. log2(4))
    for k in range(K):
        x_catch = np.zeros(D); x_catch[0] = 1.0
        x_high  = np.zeros(D); x_high[0] = 1.0; x_high[1] = 2.0
        p_catch = float(expit(model.weights[k] @ x_catch))
        p_high = float(expit(model.weights[k] @ x_high))

        if p_catch > 0.65:
            labels.append("Biased")
        elif p_high < 0.40:
            labels.append("Disengaged")
        else:
            labels.append("Engaged")

    # De-duplicate if needed (e.g. two "Engaged" states)
    seen = {}
    for i, lab in enumerate(labels):
        if lab in seen:
            seen[lab] += 1
            labels[i] = f"{lab}_{seen[lab]}"
        else:
            seen[lab] = 1
    # Fix first occurrence if there were duplicates
    for lab_base, count in seen.items():
        if count > 1:
            first_idx = next(j for j, l in enumerate(labels) if l == lab_base)
            labels[first_idx] = f"{lab_base}_1"

    return labels


# =====================================================================
# Session decoding (key downstream interface)
# =====================================================================

def decode_session(
    model: GLMHMM,
    session: Session,
    state_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Decode a session: return a DataFrame with per-trial state assignments.

    Columns added to the trial DataFrame:
      hmm_state          : int    (Viterbi)
      hmm_state_label    : str    (if *state_labels* provided)
      p_state_0 … K-1   : float  (posterior probabilities)

    The returned DataFrame only contains valid (non-excluded) trials.
    """
    data = prepare_session_data(session)
    if len(data["y"]) == 0:
        return data["df"]

    states = model.most_likely_states(data)
    posteriors = model.state_posteriors(data)

    df = data["df"].copy()
    df["hmm_state"] = states
    if state_labels is not None:
        df["hmm_state_label"] = [state_labels[s] for s in states]
    for k in range(model.n_states):
        df[f"p_state_{k}"] = posteriors[:, k]

    return df
