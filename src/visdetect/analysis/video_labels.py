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
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

from visdetect.analysis.config import (
    subject_video_labels_dir,
    canonical_camera_session,
    session_date_key,
)

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
    prior file and never leaves a partial one in place."""
    path = label_sidecar_path(session, subject, labels_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
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

    Returns ``{"source_session", "rois", "frame_size", "applied"}`` or ``None``.
    """
    d = labels_dir or subject_video_labels_dir(subject)
    if not os.path.isdir(d):
        return None
    cur = canonical_camera_session(session)
    cur_key = session_date_key(cur)
    best = None  # (date_key, stem)
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
        if best is None or k > best[0]:
            best = (k, stem)
    if best is None:
        return None
    with open(os.path.join(d, best[1] + ".json"), "r") as f:
        prior = json.load(f)
    prior_fs = list(prior.get("frame_size") or [])
    rois = {}
    for name, r in (prior.get("rois") or {}).items():
        rois[name] = {"box": [int(v) for v in r["box"]],
                      "source": f"inherited:{best[1]}"}
    applied = prior_fs == [int(v) for v in current_frame_size]
    return {"source_session": best[1], "rois": rois,
            "frame_size": prior_fs, "applied": applied}
