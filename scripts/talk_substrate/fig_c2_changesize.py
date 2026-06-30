"""Fig C2 (talk substrate): CHANGE-SIZE contrast, split by cell type AND modulation sign.

Grid: rows = cell type; columns = (alignment x sign): Change-up | Change-down |
Response-up | Response-down. Lines = SMALL (1.25-1.5x) vs BIG (2-4x) change
(canonical SMALL_/BIG_CHANGE_SIZES). Bands = bootstrap 95% CI.
(Small/big colours are a local display choice — no canonical change-size palette.)

Usage: py scripts/talk_substrate/fig_c2_changesize.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402

setup_style()
SMALL, BIG = "#fdae6b", "#d94801"   # local display choice (no canonical change-size palette)
CHANGE = [("Change_ON", "small", SMALL, "Small (1.25-1.5x)"), ("Change_ON", "big", BIG, "Big (2-4x)")]
RESP = [("Hit", "small", SMALL, "Small (1.25-1.5x)"), ("Hit", "big", BIG, "Big (2-4x)")]

COLUMNS = [
    dict(title="Change · up",     decor_event="Change_ON", specs=CHANGE, sign="up"),
    dict(title="Change · down",   decor_event="Change_ON", specs=CHANGE, sign="down"),
    dict(title="Response · up",   decor_event="Hit",       specs=RESP,   sign="up"),
    dict(title="Response · down", decor_event="Hit",       specs=RESP,   sign="down"),
]

if __name__ == "__main__":
    cache = E.load_event_cache()
    _o, _s, sdf = E.faceted_signsplit_figure(
        cache, COLUMNS, "fig_c2_changesize",
        f"{C.SUBJECT} {C.region_label()}: scaling with change size — cell type (rows) x up/down (cols)",
        "Bigger change-aligned response for big changes = sensory scaling; matched response-"
        "aligned curves = change-size-invariant motor signal. Bands = bootstrap 95% CI.")
    print(sdf.to_string(index=False))
