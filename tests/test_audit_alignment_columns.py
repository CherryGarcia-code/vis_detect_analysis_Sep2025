import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "QC_technical"))
import audit_trial_baselineon_alignment as audit  # noqa: E402


def test_audit_exposes_measured_columns():
    assert hasattr(audit, "audit_pkl")
    assert hasattr(audit, "TOL_BENIGN")
    # the measured columns the repair depends on
    for col in ("agreement", "median_resid_s", "resid_n", "aligned"):
        assert col in audit.MEASURED_COLUMNS
