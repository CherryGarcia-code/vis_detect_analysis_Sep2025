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
from visdetect.analysis.config import FA_SUBTYPE_COLORS  # noqa: E402
from visdetect.analysis.constants import FA_RT_SPLIT  # noqa: E402

C.setup_talk_style()
EARLY, LATE = FA_SUBTYPE_COLORS["Impulsive"], FA_SUBTYPE_COLORS["Stimulus-driven"]
FA = [("FA", "early", EARLY, f"Early FA (<{FA_RT_SPLIT:g}s)"),
      ("FA", "late", LATE, f"Late FA (>={FA_RT_SPLIT:g}s)")]

COLUMNS = [
    dict(title="FA · up",   decor_event="FA", specs=FA, sign="up"),
    dict(title="FA · down", decor_event="FA", specs=FA, sign="down"),
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
            cache, C.SUBJECT, COLUMNS, "fig_c3_fa_timing_tf",
            f"{C.SUBJECT} {C.region_label()}: early vs late FAs — TF-responsive overlay "
            "(cell type rows x up/down cols)",
            "Early/late FA = hue; TF-responsive = solid + band, non-responsive = grey dashed. "
            "TF-responsive = Khilkevich-Lohse GLM (NOT movement-controlled). Response-aligned (lick at 0); "
            "split at FA_RT_SPLIT = 3 s. Bands = bootstrap 95% CI; TF+ n small (see legend).",
            figsize=(9.2, 8.6))
    else:
        _o, _s, sdf = E.faceted_signsplit_figure(
            cache, COLUMNS, "fig_c3_fa_timing",
            f"{C.SUBJECT} {C.region_label()}: early vs late FAs — cell type (rows) x up/down (cols)",
            "Response-aligned (lick at 0). Split at FA_RT_SPLIT = 3 s latency. Colours = "
            "canonical FA_SUBTYPE_COLORS. Bands = bootstrap 95% CI.",
            figsize=(9, 8.6))
    print(sdf.to_string(index=False))
