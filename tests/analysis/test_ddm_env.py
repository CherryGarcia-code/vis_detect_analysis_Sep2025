"""B0 — pyddm environment + API smoke test.

Locks the *reconciled* pyddm 0.9.0 API surface that ``visdetect.analysis.ddm``
(Tasks 2-7) depends on. If pyddm is upgraded and a signature drifts, this test
fails first and points at exactly what must be reconciled in ``ddm.py``.

Reconciled facts for pyddm 0.9.0 (see ddm.py module docstring):
  * ``Drift.get_drift(self, t, x, conditions, **kwargs)``  -- order is (t, x)!
  * ``Model(..., choice_names=("lick", "nolick"))`` MUST match the Sample's
    choice_names or ``fit_adjust_model`` asserts.
  * ``Sample.from_pandas_dataframe(df, rt_column_name=, choice_column_name=,
    choice_names=)``; choice value 1 -> first name (upper), 0 -> second (lower).
  * ``LossRobustLikelihood(sample, required_conditions=, dt=, T_dur=).loss(model)``
    returns the loss (neg log-lik); CV log-lik = ``-loss``.
  * ``Solution.resample(k, seed=None)`` -> a Sample with ``.choice_upper`` /
    ``.choice_lower`` RT arrays.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

pyddm = pytest.importorskip("pyddm")


def test_pyddm_version_pinned():
    # We pin pyddm==0.9.0 in setup.cfg; warn loudly here if it drifts so the
    # reconciled signatures below get re-checked.
    assert pyddm.__version__ == "0.9.0", (
        f"pyddm version drifted to {pyddm.__version__}; re-verify the API "
        "surface reconciled in ddm.py and this test."
    )


def test_required_symbols_exist():
    from pyddm import (  # noqa: F401
        Model, Fittable, Drift, Bound, Noise, InitialCondition, Sample,
        NoiseConstant, BoundConstant, ICPoint, OverlayNonDecision, Overlay,
        fit_adjust_model, LossRobustLikelihood, LossLikelihood,
    )


def test_get_drift_signature_is_t_then_x():
    # The argument ORDER is (t, x) in 0.9.0 -- ddm.py's DriftTwoRoute must match,
    # else the leak term -lam*x and time-indexing get swapped.
    params = list(inspect.signature(pyddm.Drift.get_drift).parameters)
    assert params[:4] == ["self", "t", "x", "conditions"]


def test_dependence_param_names():
    assert pyddm.BoundConstant.required_parameters == ["B"]
    assert pyddm.NoiseConstant.required_parameters == ["noise"]
    assert pyddm.ICPoint.required_parameters == ["x0"]
    assert pyddm.OverlayNonDecision.required_parameters == ["nondectime"]


def test_sample_from_pandas_choice_mapping():
    from pyddm import Sample

    df = pd.DataFrame({"RT": [0.4, 0.6, 0.5], "lick": [1, 1, 0], "trial_uid": [0, 1, 2]})
    samp = Sample.from_pandas_dataframe(
        df, rt_column_name="RT", choice_column_name="lick",
        choice_names=("lick", "nolick"),
    )
    assert len(samp) == 3
    assert samp.choice_names == ("lick", "nolick")
    # value 1 -> upper (lick), value 0 -> lower (nolick)
    assert sorted(np.asarray(samp.choice_upper)) == [0.4, 0.6]
    assert sorted(np.asarray(samp.choice_lower)) == [0.5]


def test_custom_drift_solve_and_resample():
    """The exact pattern ddm.py uses: per-trial evidence indexed by a condition."""
    from pyddm import (Model, Drift, NoiseConstant, BoundConstant, ICPoint,
                       OverlayNonDecision)

    evmap = {0: np.r_[np.zeros(25), np.ones(75) * 2.0], 1: np.zeros(100)}

    class _DriftProbe(Drift):
        name = "probe"
        required_parameters = ["v", "u", "dt", "evmap"]   # non-numeric params OK
        required_conditions = ["trial_uid"]

        def get_drift(self, t, x, conditions, **kwargs):
            ev = self.evmap.get(conditions["trial_uid"])
            i = int(round(t / self.dt))
            e_t = ev[i] if (ev is not None and 0 <= i < len(ev)) else 0.0
            return self.v * max(e_t, 0.0) + self.u * t

    def _mk(v, u):
        return Model(
            name="probe", drift=_DriftProbe(v=v, u=u, dt=0.02, evmap=evmap),
            noise=NoiseConstant(noise=1.0), bound=BoundConstant(B=1.0),
            IC=ICPoint(x0=0.0), overlay=OverlayNonDecision(nondectime=0.05),
            dx=0.01, dt=0.02, T_dur=3.0, choice_names=("lick", "nolick"),
        )

    sol_ev = _mk(3.0, 0.3).solve(conditions={"trial_uid": 0})
    sol_flat = _mk(3.0, 0.3).solve(conditions={"trial_uid": 1})
    # TF-evidence trial crosses (licks) more than the flat-baseline trial
    assert sol_ev.prob("lick") > sol_flat.prob("lick")

    rs = sol_ev.resample(200, seed=0)
    assert len(np.asarray(rs.choice_upper)) + len(np.asarray(rs.choice_lower)) == 200
