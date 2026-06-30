"""Focused tests for the talk_substrate cell-type glue.

The cell-type column has TWO on-disk vocabularies and the colour dict key for
broad uses a BACKSLASH ("Broad (MSN\\Proj)"), so the normaliser + colour lookup
are the genuinely error-prone bits. Run: ``py -m pytest scripts/talk_substrate/test_common.py``
"""
import numpy as np
import pytest

import _common as C


@pytest.mark.parametrize("raw,expected", [
    ("FSI", C.NARROW),
    ("SPN", C.BROAD),
    ("Narrow (FSI)", C.NARROW),
    ("Broad (MSN/Proj)", C.BROAD),
    ("Broad (MSN\\Proj)", C.BROAD),   # backslash producer variant
    ("  fsi ", C.NARROW),              # whitespace + case
    ("Unclassified", C.UNKNOWN),
    ("", C.UNKNOWN),
    (None, C.UNKNOWN),
    (np.nan, C.UNKNOWN),
])
def test_normalize_celltype(raw, expected):
    assert C.normalize_celltype(raw) == expected


def test_celltype_color_distinct_and_nonempty():
    cn = C.celltype_color(C.NARROW)
    cb = C.celltype_color(C.BROAD)
    assert cn and cb and cn != cb
    # narrow should resolve to the config's narrow colour (#e74c3c)
    assert cn.lower() == "#e74c3c"


def test_canon_strips_leading_zero_form():
    # the latents/labels CSVs store the leading-zero-stripped int form
    assert C.canon(1072025) == "01072025"
    assert C.canon("1072025") == "01072025"
    assert C.canon("01072025") == "01072025"
    assert C.canon(1072025.0) == "01072025"
