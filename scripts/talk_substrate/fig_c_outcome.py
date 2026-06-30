"""Fig C (talk substrate): OUTCOME contrast, split by cell type AND modulation sign.

Grid: rows = cell type (Narrow/FSI, Broad/MSN-Proj); columns = (alignment x sign):
Change-up | Change-down | Response-up | Response-down. Within each panel:
  - Change columns: SDT HIT vs SDT MISS (go trials, change_size>1).
  - Response columns: true HIT lick vs FALSE-ALARM lick (catch licks excluded).
Showing up- and down-modulated populations separately exposes potential push-pull
between opposing striatal populations. Bands = bootstrap 95% CI across units.

Usage: py scripts/talk_substrate/fig_c_outcome.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402
from visdetect.analysis.config import OUTCOME_COLORS  # noqa: E402

setup_style()
H, M, FA = OUTCOME_COLORS["Hit"], OUTCOME_COLORS["Miss"], OUTCOME_COLORS["FA"]
CHANGE = [("Change_ON", "hit", H, "SDT hit"), ("Change_ON", "miss", M, "SDT miss")]
RESP = [("Hit", "all", H, "True hit lick"), ("FA", "all", FA, "FA lick")]

COLUMNS = [
    dict(title="Change · up",     decor_event="Change_ON", specs=CHANGE, sign="up"),
    dict(title="Change · down",   decor_event="Change_ON", specs=CHANGE, sign="down"),
    dict(title="Response · up",   decor_event="Hit",       specs=RESP,   sign="up"),
    dict(title="Response · down", decor_event="Hit",       specs=RESP,   sign="down"),
]

if __name__ == "__main__":
    cache = E.load_event_cache()
    _o, _s, sdf = E.faceted_signsplit_figure(
        cache, COLUMNS, "fig_c_outcome",
        f"{C.SUBJECT} {C.region_label()}: activity by outcome — cell type (rows) x up/down (cols)",
        "Change cols: hit vs miss (go trials). Response cols: true hit vs FA lick "
        "(catch licks excluded). up/down = unit's overall modulation sign at that event "
        "(grouping only; the split's existence is shown held-out in Fig B). Bands = bootstrap 95% CI.")
    print(sdf.to_string(index=False))
