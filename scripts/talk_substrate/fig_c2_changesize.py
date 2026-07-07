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

C.setup_talk_style()
SMALL, BIG = C.CHANGE_COLORS["small"], C.CHANGE_COLORS["big"]   # canonical (config.CHANGE_SIZE_COLORS)
CHANGE = [("Change_ON", "small", SMALL, "Small (1.25-1.5x)"), ("Change_ON", "big", BIG, "Big (2-4x)")]
RESP = [("Hit", "small", SMALL, "Small (1.25-1.5x)"), ("Hit", "big", BIG, "Big (2-4x)")]

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
                    help="celltype (default) = small/big lines by narrow/broad rows; "
                         "tf = overlay TF-responsive (solid) vs non-responsive (dashed) within "
                         "cell-type rows (3 striatum mice only).")
    args = ap.parse_args()
    cache = E.load_event_cache()
    if args.group == "tf":
        if not C.has_tf_registry(C.SUBJECT):
            raise SystemExit(f"no TF registry for {C.SUBJECT} (3 striatum mice only)")
        _o, _s, sdf = E.faceted_signsplit_tf_figure(
            cache, C.SUBJECT, COLUMNS, "fig_c2_changesize_tf",
            f"{C.SUBJECT} {C.region_label()}: change-size scaling — TF-responsive overlay "
            "(cell type rows x up/down cols)",
            "Small/big change = hue; TF-responsive = solid + band, non-responsive = dashed. "
            "TF-responsive = Khilkevich-Lohse GLM (C1 corr>0.2 & C2 CV p<0.01; NOT movement-controlled). "
            "Bands = bootstrap 95% CI; TF+ n is small (see legend).")
    else:
        _o, _s, sdf = E.faceted_signsplit_figure(
            cache, COLUMNS, "fig_c2_changesize",
            f"{C.SUBJECT} {C.region_label()}: scaling with change size — cell type (rows) x up/down (cols)",
            "Bigger change-aligned response for big changes = sensory scaling; matched response-"
            "aligned curves = change-size-invariant motor signal. Bands = bootstrap 95% CI.")
    print(sdf.to_string(index=False))
