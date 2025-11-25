"""Utilities that reproduce legacy MATLAB analyses in Python."""

from .lick import MatlabLickConfig, compute_fa_lick_responsiveness  # noqa: F401
from .tf_pulse import TFPulseConfig, compute_tf_pulse_responsiveness  # noqa: F401
