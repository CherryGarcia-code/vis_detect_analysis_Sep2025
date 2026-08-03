"""Pure sidecar schema + IO for per-session ROIs and per-frame pupil labels.

Plan 2b: everything the unified tagger's ROI/label capture needs that is NOT
GUI — the JSON schema (v1), atomic load/save, frame-label upsert, ROI setters,
cross-session ROI seeding, and the crop/ellipse geometry helpers. No cv2, no
matplotlib: fully unit-testable in isolation.

Sidecar location: ``data/cache/video_labels/<subject>/<session>.json`` (see
``config.subject_video_labels_dir``), decoupled from the anchor/sync JSON so the
label schema can evolve with sub-projects B/C independently of the sync contract.

Schema (v1)::

    {
      "schema_version": 1,
      "subject": "BG_031",
      "session": "09042025",
      "camera": "eye_cam",
      "frame_size": [H, W],
      "rois": {"eye": {"box": [y0,y1,x0,x1], "source": "drawn|inherited:<sess>"}, ...},
      "frames": [{"frame_idx": int, "verdict": "confirmed|corrected|blink",
                  "proposed_ellipse": {..}|null, "corrected_ellipse": {..}|null,
                  "labeled_at": "<iso8601>"}]
    }
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

from visdetect.analysis.config import (
    subject_video_labels_dir,
    canonical_camera_session,
    session_date_key,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

VERDICT_CONFIRMED = "confirmed"
VERDICT_CORRECTED = "corrected"
VERDICT_BLINK = "blink"


def label_sidecar_path(session, subject: Optional[str] = None,
                       labels_dir: Optional[str] = None) -> str:
    """Absolute path of the label sidecar for *session* / *subject*.

    Filename is the canonical 8-digit ``DDMMYYYY`` session id so 6-digit
    ``DDMMYY`` subjects (BG_031/039) and leading-zero-day ids never collide.
    """
    d = labels_dir or subject_video_labels_dir(subject)
    return os.path.join(d, f"{canonical_camera_session(session)}.json")


def new_sidecar(subject: str, session, frame_size,
                camera: str = "eye_cam") -> dict:
    """Fresh schema-v1 sidecar dict (empty rois + frames). ``frame_size`` is
    ``(H, W)`` and is stored as ``[H, W]``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "subject": str(subject),
        "session": canonical_camera_session(session),
        "camera": str(camera),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "rois": {},
        "frames": [],
    }


def load_sidecar(session, subject: Optional[str] = None,
                 labels_dir: Optional[str] = None) -> Optional[dict]:
    """Read the sidecar JSON, or ``None`` if it does not exist."""
    path = label_sidecar_path(session, subject, labels_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_sidecar(sidecar: dict, session, subject: Optional[str] = None,
                 labels_dir: Optional[str] = None) -> None:
    """Atomically write *sidecar* (temp file + ``os.replace``), mirroring
    ``tag_trials._persist_overrides`` — a crash mid-write never corrupts the
    prior file and never leaves a partial one in place.

    The temp file has a DETERMINISTIC name (``<target>.json.tmp``) rather than a
    unique ``mkstemp`` one: a hard process kill (Windows ``TerminateProcess``)
    skips the ``except`` cleanup, so a unique name would leak one stray ``.tmp``
    per crash. Reusing a single per-session path bounds the leak to at most one
    stale temp, which the next successful save overwrites (``open(..., "w")``
    truncates it) before ``os.replace``. Readers only ever open ``<session>.json``
    so a lingering ``.tmp`` is never mistaken for data.
    """
    path = label_sidecar_path(session, subject, labels_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(sidecar, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def set_roi(sidecar: dict, name: str, box, source: str) -> dict:
    """Set ``sidecar['rois'][name] = {'box': [y0,y1,x0,x1], 'source': source}``.

    ``name`` is ``'eye'`` or ``'mouth'``; ``box`` is a FULL-FRAME
    ``(y0,y1,x0,x1)``; ``source`` is ``'drawn'`` or ``'inherited:<session>'``.
    Re-drawing an inherited ROI is the caller's cue to pass ``source='drawn'``.
    Returns *sidecar* (mutated in place).
    """
    sidecar.setdefault("rois", {})[name] = {
        "box": [int(v) for v in box],
        "source": str(source),
    }
    return sidecar


def upsert_frame_label(sidecar: dict, frame_idx: int, verdict: str,
                       proposed_ellipse: Optional[dict] = None,
                       corrected_ellipse: Optional[dict] = None,
                       labeled_at: Optional[str] = None) -> dict:
    """Insert or REPLACE the label for *frame_idx* (keyed on ``frame_idx``;
    never duplicates). A ``corrected`` verdict is expected to carry BOTH
    ``proposed_ellipse`` and ``corrected_ellipse``. Returns *sidecar*."""
    entry = {
        "frame_idx": int(frame_idx),
        "verdict": str(verdict),
        "proposed_ellipse": proposed_ellipse,
        "corrected_ellipse": corrected_ellipse,
        "labeled_at": labeled_at or datetime.now(timezone.utc).isoformat(),
    }
    frames = sidecar.setdefault("frames", [])
    for i, fr in enumerate(frames):
        if int(fr.get("frame_idx", -1)) == int(frame_idx):
            frames[i] = entry
            return sidecar
    frames.append(entry)
    return sidecar


def seed_rois_from_previous(session, subject, current_frame_size,
                            labels_dir: Optional[str] = None) -> Optional[dict]:
    """Return the most-recent PRIOR session's ROIs, or ``None``.

    Camera geometry is usually fixed within a subject, so a new session inherits
    the last session's ROIs as editable seeds instead of drawing from scratch.

    "Most recent prior" is a **date** comparison via
    :func:`config.session_date_key`, never a lexical/int sort — ``'1072025'``
    sorts before ``'23062025'`` lexically though 1 Jul is after 23 Jun, and
    6-digit ``DDMMYY`` ids exist. Only sidecars strictly EARLIER than *session*
    are eligible; a later session is never chosen.

    Provenance: every returned ROI is marked ``source='inherited:<prior>'`` so a
    silently-copied-forward ROI is distinguishable from one a human drew. The
    caller flips it to ``'drawn'`` (via :func:`set_roi`) the moment it is re-dragged.

    Frame-size guard: an absolute-pixel box is meaningless at a different
    resolution, so ``applied`` is ``True`` only when the prior sidecar's
    ``frame_size`` equals ``current_frame_size`` (as ``(H, W)``). On a mismatch the
    ROIs are still returned (``applied=False``) so the caller can warn/offer them.

    Corruption resilience: a prior sidecar whose CONTENT will not read/parse
    (malformed JSON, unreadable file) is SKIPPED and the next-most-recent eligible
    prior is tried, so a single bad file can never abort tagger startup for a new
    session. This mirrors the existing skip of files whose FILENAME will not parse
    as a date. Each skip is ``logger.warning``-ed so the corrupt sidecar is
    discoverable rather than silently swallowed.

    Returns ``{"source_session", "rois", "frame_size", "applied"}`` or ``None``.
    """
    d = labels_dir or subject_video_labels_dir(subject)
    if not os.path.isdir(d):
        return None
    cur = canonical_camera_session(session)
    cur_key = session_date_key(cur)
    # Collect every strictly-earlier eligible prior, then walk them most-recent
    # first so a corrupt winner can fall through to the next-best by DATE (never
    # iteration order).
    eligible = []  # (date_key, stem)
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-len(".json")]
        if stem == cur:
            continue
        try:
            k = session_date_key(stem)
        except ValueError:
            continue
        if k >= cur_key:            # not strictly earlier -> ineligible
            continue
        eligible.append((k, stem))
    eligible.sort(key=lambda t: t[0], reverse=True)  # most-recent prior first
    for _, stem in eligible:
        fp = os.path.join(d, stem + ".json")
        try:
            with open(fp, "r") as f:
                prior = json.load(f)
        except (ValueError, OSError) as exc:
            # json.JSONDecodeError subclasses ValueError; OSError covers
            # unreadable files. Skip this prior and try the next-most-recent.
            logger.warning("Skipping unreadable prior sidecar %s: %s", fp, exc)
            continue
        prior_fs = list(prior.get("frame_size") or [])
        rois = {}
        for name, r in (prior.get("rois") or {}).items():
            rois[name] = {"box": [int(v) for v in r["box"]],
                          "source": f"inherited:{stem}"}
        applied = prior_fs == [int(v) for v in current_frame_size]
        return {"source_session": stem, "rois": rois,
                "frame_size": prior_fs, "applied": applied}
    return None


def clamp_crop(crop, H: int, W: int) -> Optional[Tuple[int, int, int, int]]:
    """Clamp a ``(y0,y1,x0,x1)`` crop into the frame, or ``None`` if it misses it.

    HARD REQUIREMENT (design §7): ``tagging.eye_zoom_crop`` returns UNCLAMPED
    coords — padding an ROI near a frame edge can yield negative or out-of-frame
    values, and numpy slicing with a negative index does NOT error: it silently
    WRAPS from the far edge and returns the WRONG crop. Always clamp here before
    indexing a frame.

    Contract:
      * When the box intersects the frame, return the clamped
        ``(y0,y1,x0,x1)`` with ``0 <= y0 < y1 <= H`` and ``0 <= x0 < x1 <= W``
        (guaranteed non-empty — slicing a frame with it yields a real sub-image).
      * Return ``None`` when there is NO intersection: the clamped width or
        height would be zero (box entirely off-frame), OR the box is malformed.
        ``None`` means "no valid crop — the caller MUST fall back". For the GUI
        that means staying on / reverting to the full frame, never a zoom onto an
        empty array.

    Inverted (malformed) inputs — ``y1 < y0`` or ``x1 < x0`` — are NOT
    order-normalized: silently swapping the coordinates would invent an ROI the
    user never drew, so a malformed box is treated as non-intersecting → ``None``.
    """
    y0, y1, x0, x1 = (int(v) for v in crop)
    # Malformed / degenerate box (inverted or zero-area before clamping): reject
    # outright rather than swapping — a swap would fabricate an unintended ROI.
    if y1 <= y0 or x1 <= x0:
        return None
    y0 = max(0, min(y0, H))
    y1 = max(0, min(y1, H))
    x0 = max(0, min(x0, W))
    x1 = max(0, min(x1, W))
    # After clamping the box may collapse (it lay wholly outside the frame): a
    # zero-width/height slice is empty, so there is no valid crop.
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, y1, x0, x1)


def image_extent_for_crop(crop, frame_h: int, frame_w: int) -> Tuple[float, float, float, float]:
    """``(left, right, bottom, top)`` imshow extent placing the displayed image in
    FULL-FRAME data coords, for a crop ``(y0,y1,x0,x1)`` or ``None`` (whole frame).

    The tagger shows either the whole frame or a cropped eye-zoom in the SAME
    ``imshow`` artist (``cfg.crop`` toggles live). The image is created once, so
    its extent is frozen unless updated on every redraw. If a cropped (smaller)
    array is drawn under a frozen full-frame extent, matplotlib STRETCHES it
    across the whole frame — every ROI / pupil coordinate then read off the axes
    is silently rescaled (neither full-frame nor crop-local). Re-deriving the
    extent from the live crop keeps the displayed image in full-frame data
    coords in BOTH views, so a drag reads true full-frame pixels with no scale
    and no offset — which is why the crop-origin rebasing hacks are unnecessary.

    Matplotlib image extent is ``(left, right, bottom, top)`` and this axis is
    inverted (image origin at the top-left), so ``top`` uses the SMALLER y
    (``y0``) and ``bottom`` the larger (``y1``). The extent spans exactly
    ``x0..x1`` / ``y0..y1`` in full-frame pixel edges, introducing no scaling.
    """
    if crop is None:
        return (-0.5, float(frame_w) - 0.5, float(frame_h) - 0.5, -0.5)
    y0, y1, x0, x1 = (int(v) for v in crop)
    return (float(x0) - 0.5, float(x1) - 0.5, float(y1) - 0.5, float(y0) - 0.5)


def ellipse_from_box(box) -> dict:
    """Inscribed axis-aligned ellipse ``{cx,cy,major,minor,angle}`` from a drag
    box ``(y0,y1,x0,x1)``.

    ``major`` = larger of (width, height), ``minor`` = smaller, ``angle`` = 0.0
    when wider-than-tall else 90.0. Rotation beyond 0/90 is intentionally lost
    (design §5): negligible for a near-circular rodent pupil, and the two-drag
    major/minor variant can follow if it ever matters.

    Contract (differs from :func:`clamp_crop` deliberately):
      * Inverted coords are order-NORMALIZED — ``(y1,y0,x1,x0)`` yields the SAME
        ellipse as ``(y0,y1,x0,x1)``. An ellipse box is symmetric in intent (its
        centre and axes do not depend on drag direction), so normalizing cannot
        invent an ROI the user never drew. This is why swapping is safe here but
        NOT in ``clamp_crop`` (where a swap would fabricate an unintended crop).
      * A degenerate box — zero width OR zero height (zero area) after
        normalization — raises ``ValueError``. A zero/negative-diameter
        "ground-truth" ellipse is scientifically meaningless, so it is rejected
        loudly rather than returned silently.
    """
    y0, y1, x0, x1 = box
    y0, y1 = sorted((float(y0), float(y1)))   # normalize inverted drag
    x0, x1 = sorted((float(x0), float(x1)))
    w = x1 - x0
    h = y1 - y0
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"degenerate ellipse box (zero area): {tuple(box)!r}")
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    if w >= h:
        return {"cx": cx, "cy": cy, "major": w, "minor": h, "angle": 0.0}
    return {"cx": cx, "cy": cy, "major": h, "minor": w, "angle": 90.0}


def ellipse_from_detection(det: Optional[dict]) -> Optional[dict]:
    """Map a :func:`video_sync.detect_pupil_in_frame` result to the sidecar
    ellipse schema ``{cx,cy,major,minor,angle}``, or ``None`` if ``det`` is None.

    The detector surfaces only ``center_x``, ``center_y`` and ``radius``
    (``radius = max(axes)/2`` from the internal ``cv2.fitEllipse``); the minor
    axis and rotation are not exposed. The proposed ellipse is therefore stored
    as a CIRCLE of diameter ``2*radius`` (``angle=0``). This preserves the
    scientifically-critical quantity — the pupil's major diameter, which the
    too-small-diameter eyelid-occlusion bias is measured against — while
    collapsing the (unavailable) minor axis. A human ``correct`` supplies a true
    two-axis ellipse via :func:`ellipse_from_box` when the shape matters.
    """
    if det is None:
        return None
    diameter = 2.0 * float(det["radius"])
    return {"cx": float(det["center_x"]), "cy": float(det["center_y"]),
            "major": diameter, "minor": diameter, "angle": 0.0}
