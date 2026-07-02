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
from visdetect.analysis.config import OUTCOME_COLORS  # noqa: E402

C.setup_talk_style()
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["celltype", "tf"], default="celltype",
                    help="tf = overlay TF-responsive (solid) vs non-responsive (dashed) within "
                         "cell-type rows (3 striatum mice only)")
    args = ap.parse_args()
    cache = E.load_event_cache()
    if args.group == "tf":
        if not C.has_tf_registry(C.SUBJECT):
            raise SystemExit(f"no TF registry for {C.SUBJECT} (3 striatum mice only)")
        _o, _s, sdf = E.faceted_signsplit_tf_figure(
            cache, C.SUBJECT, COLUMNS, "fig_c_outcome_tf",
            f"{C.SUBJECT} {C.region_label()}: activity by outcome — TF-responsive overlay "
            "(cell type rows x up/down cols)",
            "Outcome (hit/miss @change; true-hit/FA @lick) = hue; TF-responsive = solid + band, "
            "non-responsive = grey dashed. TF-responsive = Khilkevich-Lohse GLM (NOT movement-controlled). "
            "Bands = bootstrap 95% CI; TF+ n small (see legend).")
    else:
        _o, _s, sdf = E.faceted_signsplit_figure(
            cache, COLUMNS, "fig_c_outcome",
            f"{C.SUBJECT} {C.region_label()}: activity by outcome — cell type (rows) x up/down (cols)",
            "Change cols: hit vs miss (go trials). Response cols: true hit vs FA lick "
            "(catch licks excluded). up/down = unit's overall modulation sign at that event "
            "(grouping only; the split's existence is shown held-out in Fig B). Bands = bootstrap 95% CI.")
    print(sdf.to_string(index=False))
