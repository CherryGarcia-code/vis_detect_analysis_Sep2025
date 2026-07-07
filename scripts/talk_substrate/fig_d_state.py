"""Fig D (talk substrate): engagement-STATE contrast (DESCRIPTIVE), by cell type AND modulation sign.

Grid: rows = cell type; columns = (alignment x sign): Change-up | Change-down |
Response-up | Response-down. Lines = behavioural state (Impulsive / StimSens /
Disengaged), joined to trials by trial_idx (verified) from the decision-latent model.
Bands = bootstrap 95% CI.

CAVEAT: states are DEFINED from behaviour, so a within-session neural-vs-state contrast
is partly circular by construction. Descriptive "does it visibly differ?" look, NOT an
independence claim — the rigorous version is N1's across-session graded test.

Usage: py scripts/talk_substrate/fig_d_state.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402
from visdetect.analysis.config import STATE_LABEL_COLORS  # noqa: E402

C.setup_talk_style()
STATES = ["Impulsive", "StimSens", "Disengaged"]
CHANGE = [("Change_ON", f"state_{s}", STATE_LABEL_COLORS[s], s) for s in STATES]
RESP = [("Hit", f"state_{s}", STATE_LABEL_COLORS[s], s) for s in STATES]

COLUMNS = [
    dict(title="Change · up",     decor_event="Change_ON", specs=CHANGE, sign="up"),
    dict(title="Change · down",   decor_event="Change_ON", specs=CHANGE, sign="down"),
    dict(title="Response · up",   decor_event="Hit",       specs=RESP,   sign="up"),
    dict(title="Response · down", decor_event="Hit",       specs=RESP,   sign="down"),
]

if __name__ == "__main__":
    cache = E.load_event_cache()
    _o, _s, sdf = E.faceted_signsplit_figure(
        cache, COLUMNS, "fig_d_state",
        "BG_046 striatum (CP): activity by engagement state (DESCRIPTIVE) — "
        "cell type (rows) x up/down (cols)",
        "DESCRIPTIVE only: states are defined from behaviour, so within-session neural-vs-state "
        "is partly circular (rigorous test is across-session, graded). Change cols = go trials; "
        "Response cols = true hits. Colours = STATE_LABEL_COLORS. Bands = bootstrap 95% CI.")
    print(sdf.to_string(index=False))
