"""Modality tests + segmented-vs-linear regression for the spectrum-vs-classes
question. GMM ΔBIC is the primary modality test (same method the repo uses for the
T2P waveform bimodality check); Silverman + Sarle's coefficient are secondary; the
Hartigan dip test is optional (only if the `diptest` package is installed).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def gmm_delta_bic(x, random_state: int = 42) -> dict:
    from sklearn.mixture import GaussianMixture
    x = _clean(x)
    if x.size < 4:
        return {"delta_bic": float("nan"), "n": int(x.size), "means": [], "weights": []}
    X = x.reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=random_state).fit(X)
    g2 = GaussianMixture(2, random_state=random_state).fit(X)
    order = np.argsort(g2.means_.flatten())
    return {
        "delta_bic": float(g1.bic(X) - g2.bic(X)),
        "n": int(x.size),
        "means": [float(m) for m in g2.means_.flatten()[order]],
        "weights": [float(w) for w in g2.weights_[order]],
    }


def bimodality_coefficient(x) -> float:
    """Sarle's BC = (skew^2 + 1) / (kurtosis + 3(n-1)^2/((n-2)(n-3)))."""
    x = _clean(x)
    n = x.size
    if n < 4:
        return float("nan")
    g = stats.skew(x)
    k = stats.kurtosis(x, fisher=True)  # excess kurtosis
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def silverman_bootstrap(x, n_boot: int = 500, seed: int = 42) -> dict:
    """Silverman critical-bandwidth test of H0: unimodal. Small p rejects unimodality."""
    x = _clean(x)
    if x.size < 10:
        return {"crit_bw": float("nan"), "p_unimodal": float("nan")}

    def _n_modes(sample, bw):
        grid = np.linspace(sample.min(), sample.max(), 512)
        dens = stats.gaussian_kde(sample, bw_method=bw / sample.std(ddof=1))(grid)
        return int(np.sum((dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])))

    lo, hi = 1e-3 * x.std(ddof=1), x.std(ddof=1) * 2
    for _ in range(60):  # bisection for the smallest bw giving a unimodal KDE
        mid = 0.5 * (lo + hi)
        if _n_modes(x, mid) <= 1:
            hi = mid
        else:
            lo = mid
    h_crit = hi
    rng = np.random.default_rng(seed)
    n = x.size
    count = 0
    for _ in range(n_boot):
        samp = rng.choice(x, n, replace=True)
        samp = samp + h_crit * rng.standard_normal(n)  # smoothed bootstrap
        if _n_modes(samp, h_crit) > 1:
            count += 1
    return {"crit_bw": float(h_crit), "p_unimodal": float(count / n_boot)}


def dip_test(x) -> dict:
    x = _clean(x)
    try:
        import diptest as _dt
        dip, p = _dt.diptest(x)
        return {"dip": float(dip), "p": float(p)}
    except Exception:
        return {"dip": float("nan"), "p": float("nan")}


def _ols_bic(y, yhat, k_params) -> float:
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + k_params * np.log(n)


def segmented_vs_linear(x, y, n_grid: int = 40) -> dict:
    """Compare a straight line vs a continuous 2-segment (broken-stick) fit by BIC.
    delta_bic = bic_linear - bic_segmented (positive => breakpoint preferred)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10:
        return {"breakpoint": float("nan"), "bic_linear": float("nan"),
                "bic_segmented": float("nan"), "delta_bic": float("nan"),
                "slope_lo": float("nan"), "slope_hi": float("nan")}
    b1, b0 = np.polyfit(x, y, 1)
    bic_lin = _ols_bic(y, b1 * x + b0, 2)
    best = None
    for bp in np.quantile(x, np.linspace(0.15, 0.85, n_grid)):
        h = np.maximum(0.0, x - bp)                 # continuous hinge basis
        A = np.column_stack([np.ones_like(x), x, h])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        bic = _ols_bic(y, A @ coef, 4)
        if best is None or bic < best[0]:
            best = (bic, float(bp), float(coef[1]), float(coef[1] + coef[2]))
    bic_seg, bp, slope_lo, slope_hi = best
    return {"breakpoint": bp, "bic_linear": float(bic_lin),
            "bic_segmented": float(bic_seg), "delta_bic": float(bic_lin - bic_seg),
            "slope_lo": slope_lo, "slope_hi": slope_hi}
