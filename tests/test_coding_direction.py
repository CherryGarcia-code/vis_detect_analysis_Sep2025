import numpy as np
from src.coding_direction import time_resolved_cd


def test_time_resolved_cd_recovery():
    rng = np.random.RandomState(42)
    trials = 100
    bins = 20
    units = 50
    # Build pop: trials x bins x units
    pop = rng.normal(0, 1, size=(trials, bins, units))
    # Define condition: first 50 trials = class 0, last 50 = class 1
    cond = np.zeros(trials, dtype=bool)
    cond[50:] = True
    # Inject a time-resolved effect in bins 8-12 for a larger subset of units
    effected_units = np.arange(15)
    pop[cond, 8:13, :][:, :, effected_units] += (
        3.0  # stronger boost for class 1 to ensure detectability
    )

    # Use a small permutation count for speed; the test asserts recovery by magnitude and location only
    res = time_resolved_cd(
        pop,
        cond,
        method="shrinkage",
        reg=1.0,
        n_splits=5,
        n_permutations=50,
        random_state=0,
    )
    eff = res["effect"]
    # Check max effect occurs in the effected bins
    max_bin = np.argmax(eff)
    assert 8 <= max_bin <= 12
    # Check the effect magnitude in the central effected bin is substantially larger than the mean of other bins
    central_effect = eff[10]
    other_mean = np.mean(np.concatenate([eff[:6], eff[14:]]))
    assert (central_effect - other_mean) > 0.15
