"""sync_status.py — batch progress tracker for the manual video-sync pipeline.

Read-only. Cross-references the QC staging manifest (the analysis roster) with
the video-sync cache and reports, per session, whether it has been anchored and
clock-fitted, plus the fitted drift and quality. Use it to see where you are in
the multi-session sync batch (click_anchor -> fit_sync -> tag_trials).

  py scripts/video/sync_status.py                # QC-passed roster, Expert->Naive
  py scripts/video/sync_status.py --remaining    # only sessions not yet DONE
  py scripts/video/sync_status.py --all          # include QC-excluded manifest rows
  py scripts/video/sync_status.py --order chrono # Naive->Expert instead of reverse

Status per session:
  DONE     {session}_video_sync.json present (anchored + fitted)
  PARTIAL  anchor JSON present but not yet fitted (run --anchor-last and/or fit_sync)
  TODO     no anchor yet

Nothing is modified; this only reads the manifest and data/cache/video_sync/.
"""
import argparse
import sys

from visdetect.analysis.config import (
    load_staging_manifest,
    chronological_sort,
    canonical_camera_session,
    subject_video_sync_dir,
)
from visdetect.core.video_sync import load_anchor, load_video_sync


def _session_status(session_int: int, sync_dir: str) -> dict:
    """Gather anchor + sync state for one session (read-only)."""
    name = canonical_camera_session(session_int)

    anchor = load_anchor(name, sync_dir=sync_dir)
    n_anchors = len(anchor["anchors"]) if anchor else 0

    sync = load_video_sync(name, sync_dir=sync_dir)
    if sync is not None:
        eye = sync.get("eye_cam") or {}
        slope_ppm = eye.get("slope_ppm")
        cv_rmse = eye.get("cv_rmse_ms")
        quality = sync.get("quality") or eye.get("quality")
        n_over = len(eye.get("per_trial_overrides") or {})
        status = "DONE"
    else:
        slope_ppm = quality = cv_rmse = None
        n_over = 0
        status = "PARTIAL" if n_anchors >= 1 else "TODO"

    # Derive iso from the 8-digit `name` (DDMMYYYY) directly. Routing the raw
    # int through session_int_to_iso zero-pads 6-digit subjects wrongly
    # (50325 -> 00050325 -> 0325-05-00); `name` already canonicalised above.
    iso = f"{name[4:]}-{name[2:4]}-{name[:2]}"
    return {
        "session": name,
        "iso": iso,
        "status": status,
        "n_anchors": n_anchors,
        "slope_ppm": slope_ppm,
        "cv_rmse": cv_rmse,
        "quality": quality,
        "n_over": n_over,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Video-sync batch progress tracker (read-only)."
    )
    p.add_argument("--remaining", action="store_true",
                   help="Show only sessions that are not yet DONE.")
    p.add_argument("--all", action="store_true",
                   help="Include QC-excluded manifest rows (default: QC-passed only).")
    p.add_argument("--order", choices=["reverse", "chrono"], default="reverse",
                   help="reverse = Expert->Naive (default, matches batch order); "
                        "chrono = Naive->Expert.")
    p.add_argument("--subject", default=None,
                   help="Subject (default: config.SUBJECT).")
    args = p.parse_args()

    # Per-subject sync cache: data/cache/video_sync/<SUBJECT>/.
    sync_dir = subject_video_sync_dir(args.subject)
    # NOTE: load_staging_manifest keys the roster off config.SUBJECT; for a
    # non-default subject set VISDETECT_SUBJECT (or pass a manifest_path). E.g.:
    #   VISDETECT_SUBJECT=BG_031 py scripts/video/sync_status.py --subject BG_031

    if args.all:
        manifest = load_staging_manifest(qc_only=False, apply_filter=False)
    else:
        manifest = load_staging_manifest(qc_only=True)
    sessions = chronological_sort(int(s) for s in manifest["session_name"].tolist())
    if args.order == "reverse":
        sessions = list(reversed(sessions))

    rows = [_session_status(s, sync_dir) for s in sessions]
    shown = [r for r in rows if not (args.remaining and r["status"] == "DONE")]

    roster = "all manifest rows" if args.all else "QC-passed roster"
    order_lbl = "Expert->Naive" if args.order == "reverse" else "Naive->Expert"
    print(f"\nVideo-sync status: {roster}, {order_lbl}  ({len(rows)} sessions)\n")
    hdr = (f"{'#':>3}  {'date':<10}  {'session':<8}  {'status':<7}  "
           f"{'anc':>3}  {'ppm':>8}  {'cv_ms':>6}  {'quality':<9}  {'ovr':>3}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(shown, 1):
        ppm = (f"{r['slope_ppm']:+.1f}"
               if isinstance(r["slope_ppm"], (int, float)) else "-")
        cv_ms = (f"{r['cv_rmse']:.1f}"
                 if isinstance(r["cv_rmse"], (int, float)) else "-")
        qual = r["quality"] or "-"
        if qual not in ("good", "-"):
            qual = qual + " !"           # flag review/failed fits
        over = str(r["n_over"]) if r["n_over"] else "-"
        print(f"{i:>3}  {r['iso']:<10}  {r['session']:<8}  {r['status']:<7}  "
              f"{r['n_anchors']:>3}  {ppm:>8}  {cv_ms:>6}  {qual:<9}  {over:>3}")

    n_done = sum(1 for r in rows if r["status"] == "DONE")
    n_part = sum(1 for r in rows if r["status"] == "PARTIAL")
    n_todo = sum(1 for r in rows if r["status"] == "TODO")
    n_bad = sum(1 for r in rows
                if r["status"] == "DONE" and r["quality"] not in ("good", None))
    print("-" * len(hdr))
    summary = f"DONE {n_done}/{len(rows)}   PARTIAL {n_part}   TODO {n_todo}"
    if n_bad:
        summary += f"   (! {n_bad} fitted but quality != good)"
    print(summary)
    nxt = next((r for r in rows if r["status"] != "DONE"), None)
    print(f"Next: {nxt['session']} ({nxt['iso']}, {nxt['status']})"
          if nxt else "All sessions DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
