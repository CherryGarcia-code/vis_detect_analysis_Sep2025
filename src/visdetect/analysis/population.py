"""Population-level utilities: coding direction and trajectory analysis.

Functions
- compute_condition_averages(population_matrix, condition_trials)
- compute_coding_direction(avgA, avgB)
- project_onto_cd(population_matrix, cd)
"""

import numpy as np


def compute_condition_averages(
    population_matrix: np.ndarray, condition_trials: np.ndarray
) -> np.ndarray:
    """population_matrix: units x (trials x time) or trials x bins x units depending on usage.
    Here we expect trials x units (averaged in a given time window) for simplicity.
    condition_trials: boolean mask over trials.
    Returns avg vector of shape (units,)
    """
    if population_matrix.ndim == 3:
        # trials x bins x units -> average over trials then bins
        trials = population_matrix[condition_trials]
        return trials.mean(axis=(0, 1))
    else:
        trials = population_matrix[condition_trials]
        return trials.mean(axis=0)


def compute_coding_direction(avgA: np.ndarray, avgB: np.ndarray) -> np.ndarray:
    """Compute coding direction vector from avgA to avgB (unit vector)."""
    diff = (avgB - avgA).astype(float)
    norm = np.linalg.norm(diff)
    if norm == 0:
        return diff
    return diff / norm


def project_onto_cd(population_matrix: np.ndarray, cd: np.ndarray) -> np.ndarray:
    """Project population activity onto coding direction.

    population_matrix: trials x units or trials x bins x units
    cd: units vector
    Returns scalar projection per trial or trials x bins projection.
    """
    if population_matrix.ndim == 3:
        # trials x bins x units -> project along units
        return np.tensordot(population_matrix, cd, axes=([2], [0]))
    else:
        # trials x units
        return population_matrix @ cd
