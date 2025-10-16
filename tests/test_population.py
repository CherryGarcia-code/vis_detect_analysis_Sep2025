import numpy as np
from src.population import (
    compute_condition_averages,
    compute_coding_direction,
    project_onto_cd,
)


def test_cd_projection_simple():
    # Create a toy population: 10 trials x 5 units
    rng = np.random.RandomState(0)
    trials = rng.randn(20, 5)
    # condition A: first 10, condition B: last 10
    condA = np.zeros(20, dtype=bool)
    condA[:10] = True
    condB = ~condA
    avgA = compute_condition_averages(trials, condA)
    avgB = compute_condition_averages(trials, condB)
    cd = compute_coding_direction(avgA, avgB)
    proj = project_onto_cd(trials, cd)
    assert proj.shape == (20,)
    # Projections of A should have different mean than B
    assert abs(proj[:10].mean() - proj[10:].mean()) > 1e-6
