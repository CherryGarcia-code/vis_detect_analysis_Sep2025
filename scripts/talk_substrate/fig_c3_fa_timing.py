"""Fig C3 (talk substrate): EARLY vs LATE false alarms, split by cell type AND modulation sign.

Grid: rows = cell type; columns = FA-up | FA-down. Lines = early FA (<3 s) vs late FA
(>=3 s), split at canonical FA_RT_SPLIT (classify_fa_type). Response-aligned (lick at 0).
Bands = bootstrap 95% CI. Colours = canonical FA_SUBTYPE_COLORS (early~impulsive,
late~stimulus-driven).

Usage: py scripts/talk_substrate/fig_c3_fa_timing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402
from visdetect.analysis.config import FA_SUBTYPE_COLORS  # noqa: E402
from visdetect.analysis.constants import FA_RT_SPLIT  # noqa: E402

setup_style()
EARLY, LATE = FA_SUBTYPE_COLORS["Impulsive"], FA_SUBTYPE_COLORS["Stimulus-driven"]
FA = [("FA", "early", EARLY, f"Early FA (<{FA_RT_SPLIT:g}s)"),
      ("FA", "late", LATE, f"Late FA (>={FA_RT_SPLIT:g}s)")]

COLUMNS = [
    dict(title="FA · up",   decor_event="FA", specs=FA, sign="up"),
    dict(title="FA · down", decor_event="FA", specs=FA, sign="down"),
]

if __name__ == "__main__":
    cache = E.load_event_cache()
    _o, _s, sdf = E.faceted_signsplit_figure(
        cache, COLUMNS, "fig_c3_fa_timing",
        f"{C.SUBJECT} {C.region_label()}: early vs late FAs — cell type (rows) x up/down (cols)",
        "Response-aligned (lick at 0). Split at FA_RT_SPLIT = 3 s latency. Colours = "
        "canonical FA_SUBTYPE_COLORS. Bands = bootstrap 95% CI.",
        figsize=(9, 8.6))
    print(sdf.to_string(index=False))
