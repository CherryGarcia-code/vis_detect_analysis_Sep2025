"""Coding direction utilities: regularized CD, time-resolved CD with cross-validation, and permutation testing.

Functions
- compute_cd_shrinkage(X, y, reg): compute CD = inv(Cov + reg I) (mean1 - mean0)
- compute_cd_ridge(X, y, alpha): fit ridge regression and use weights as CD
- time_resolved_cd(pop, cond_mask, method='shrinkage', reg=1.0, n_splits=5, n_permutations=200, random_state=0)
  where pop is trials x bins x units; returns cds (bins x units), proj (trials x bins), pvals (bins)
"""
import numpy as np
from typing import Tuple, Dict, Any


def _stratified_kfold_indices(y: np.ndarray, n_splits: int, random_state: int = 0):
    """Simple stratified K-fold indices generator.

    y: 1D array of binary labels (0/1)
    Returns list of (train_idx, test_idx)
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    classes = np.unique(y)
    # Group indices per class
    per_class_idxs = {c: np.where(y == c)[0] for c in classes}
    # Determine actual number of splits limited by smallest class count
    min_count = min(len(v) for v in per_class_idxs.values())
    n_splits = max(2, min(n_splits, min_count))
    folds = []
    # Shuffle per-class indices and split
    per_class_folds = {}
    for c, idxs in per_class_idxs.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        per_class_folds[c] = np.array_split(idxs, n_splits)

    for k in range(n_splits):
        test_idx = np.concatenate([per_class_folds[c][k] for c in classes if len(per_class_folds[c][k]) > 0])
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=True)
        folds.append((train_idx, test_idx))
    return folds


def _ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression weights (no intercept).

    Returns weight vector of shape (n_features,)
    """
    # center X and y (fit intercept separately) to improve numerical stability
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    n_features = XtX.shape[0]
    A = XtX + alpha * np.eye(n_features)
    try:
        w = np.linalg.solve(A, Xc.T @ yc)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ (Xc.T @ yc)
    return w


def compute_cd_shrinkage(X: np.ndarray, y: np.ndarray, reg: float = 1.0) -> np.ndarray:
    """Compute coding direction using shrinkage (regularized covariance inverse).

    X: trials x units
    y: binary labels (0/1)
    Returns unit vector CD (units,)
    """
    # Means
    mu1 = X[y == 1].mean(axis=0)
    mu0 = X[y == 0].mean(axis=0)
    diff = mu1 - mu0
    # Covariance (pooled)
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    # Regularize
    cov_reg = cov + reg * np.eye(cov.shape[0])
    try:
        inv = np.linalg.inv(cov_reg)
        w = inv @ diff
    except np.linalg.LinAlgError:
        # fallback to ridge-like pseudo-inverse
        w = np.linalg.pinv(cov_reg) @ diff
    norm = np.linalg.norm(w)
    if norm == 0:
        return w
    return w / norm


def compute_cd_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Compute coding direction by fitting a ridge regression to predict y from X.
    Returns normalized weight vector.
    """
    # Use closed-form ridge (centered)
    from .coding_direction import _ridge_closed_form as _rcf  # type: ignore
    w = _rcf(X, y, alpha=alpha)
    norm = np.linalg.norm(w)
    if norm == 0:
        return w
    return w / norm


def time_resolved_cd(pop: np.ndarray, cond_mask: np.ndarray, method: str = 'shrinkage', reg: float = 1.0, n_splits: int = 5, n_permutations: int = 200, random_state: int = 0) -> Dict[str, Any]:
    """Compute time-resolved coding directions.

    pop: trials x bins x units
    cond_mask: boolean array length trials, True for condition A (or class 1)
    method: 'shrinkage' or 'ridge'
    Returns dict with keys:
      'cds': bins x units array (average CD across folds)
      'proj': trials x bins projections onto CD (using cross-validated CDs)
      'pvals': bins array of permutation p-values for difference in mean projections between conditions
      'effect': true effect per bin (mean_proj_cond1 - mean_proj_cond0)
    """
    rng = np.random.RandomState(random_state)
    trials, bins, units = pop.shape
    y = cond_mask.astype(int)
    # Ensure both classes present
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError('Need both classes present in cond_mask')

    folds = _stratified_kfold_indices(y, n_splits=n_splits, random_state=random_state)
    proj = np.zeros((trials, bins), dtype=float)
    cds_accum = np.zeros((bins, units), dtype=float)
    fold_counts = np.zeros(bins, dtype=int)

    for train_idx, test_idx in folds:
        # Skip folds that are empty
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        for b in range(bins):
            # Defensive: ensure indices are within bounds and there are samples
            try:
                X_train = pop[train_idx, b, :]
                X_test = pop[test_idx, b, :]
            except Exception:
                # Skip this bin/fold if indexing fails
                continue
            y_train = y[train_idx]
            # Require both classes present in training fold to compute CD
            if len(np.unique(y_train)) < 2:
                # cannot compute a discriminative direction with only one class
                continue
            if method == 'shrinkage':
                cd = compute_cd_shrinkage(X_train, y_train, reg=reg)
            else:
                # compute ridge via closed form
                w = _ridge_closed_form(X_train, y_train, alpha=reg)
                cd = w
            # normalize
            if np.linalg.norm(cd) > 0:
                cd = cd / np.linalg.norm(cd)
            cds_accum[b] += cd
            fold_counts[b] += 1
            # project test data (only if sizes align)
            if X_test.shape[0] == len(test_idx):
                try:
                    proj[test_idx, b] = X_test @ cd
                except Exception:
                    # skip projection on failure
                    pass

    # Average cds across folds
    cds = cds_accum / np.maximum(fold_counts[:, None], 1)

    # Compute true effect per bin
    mean0 = proj[y == 0].mean(axis=0)
    mean1 = proj[y == 1].mean(axis=0)
    effect = mean1 - mean0

    # Permutation test: shuffle labels and compute effect distribution using the same cross-validation scheme
    perm_effects = np.zeros((n_permutations, bins), dtype=float)
    for i in range(n_permutations):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        # build cross-validated projections under permuted labels
        proj_perm = np.zeros((trials, bins), dtype=float)
        for train_idx, test_idx in folds:
            # Skip empty folds
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            for b in range(bins):
                try:
                    X_train = pop[train_idx, b, :]
                    X_test = pop[test_idx, b, :]
                except Exception:
                    continue
                y_train = y_perm[train_idx]
                if len(np.unique(y_train)) < 2:
                    continue
                if method == 'shrinkage':
                    cd_perm = compute_cd_shrinkage(X_train, y_train, reg=reg)
                else:
                    cd_perm = _ridge_closed_form(X_train, y_train, alpha=reg)
                if np.linalg.norm(cd_perm) > 0:
                    cd_perm = cd_perm / np.linalg.norm(cd_perm)
                if X_test.shape[0] == len(test_idx):
                    try:
                        proj_perm[test_idx, b] = X_test @ cd_perm
                    except Exception:
                        pass
        # compute effect for this permutation
        perm_effects[i] = proj_perm[y_perm == 1].mean(axis=0) - proj_perm[y_perm == 0].mean(axis=0)

    # p-values (two-sided)
    pvals = np.array([((np.abs(perm_effects[:, b]) >= abs(effect[b])).sum() + 1) / (n_permutations + 1) for b in range(bins)])

    return {
        'cds': cds,
        'proj': proj,
        'pvals': pvals,
        'effect': effect,
        'mean0': mean0,
        'mean1': mean1
    }
