"""organize_bg012_behavior.py — populate Raw data/<session>/Session/ for BG_012.

WHY a BG_012-specific organizer (vs organize_subject_data.py):
  - BG_012 keeps behaviour in ``BG_012/FSM_behavior_data/`` (timestamped
    ``BG_012_YYYYMMDD_HHMMSS__{trials,session_settings,computer_settings}.json``),
    NOT in the subject top-level or ``wEPhys/FSMdata/`` that organize_subject_data
    scans, and NOT in ``Raw data/<session>/Session/`` where
    ``ingest.load_behavioral_trials`` looks.
  - BG_012 has same-day restarts (``_b``/``_c``) and multi-protocol days, so the
    date-only matching in organize_subject_data (which copies *every* block of a
    date into *every* session dir of that date) is ambiguous/wrong here.

MATCHING: each ephys recording -> its behavioural block BY TIME. The SpikeGLX
meta's ``fileCreateTime_original`` (true recording start; ``fileCreateTime`` is the
later CatGT-run time) is matched to the nearest FSM filename ``HHMMSS`` on the same
date. Verified: 03112023 main=14:25:21->…_142531, _b=15:06:13->…_150647.

SAFETY: copy-only, DRY-RUN by default (``--execute`` to copy), size-verified,
skip-existing. Platform-aware root (X: locally / /ceph on the cluster) — run the
``--execute`` CLUSTER-SIDE so the (small) JSON writes go to native CephFS, not the
Samba gateway.

Usage:
  py scripts/data_management/organize_bg012_behavior.py                # dry-run (preview matches)
  py scripts/data_management/organize_bg012_behavior.py --execute      # copy (run cluster-side)
  py scripts/data_management/organize_bg012_behavior.py --tolerance-min 45
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SUBJECT = "BG_012"
JSON_TYPES = ("__trials.json", "__session_settings.json", "__computer_settings.json")


def wephys_root() -> Path:
    if platform.system() == "Linux":
        return Path("/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys")
    return Path("X:/public/projects/BeJG_20230130_VisDetect/wEPhys")


# ── parsing ──────────────────────────────────────────────────────────────
def session_date(name: str) -> str | None:
    """DDMMYYYY (first 8 digits after BG_012_) -> YYYYMMDD."""
    m = re.match(r"BG_012_(\d{2})(\d{2})(\d{4})", name)
    if not m:
        return None
    dd, mm, yyyy = m.groups()
    return f"{yyyy}{mm}{dd}"


def fsm_stamp(name: str) -> tuple[str, dt.datetime] | None:
    """(YYYYMMDD, datetime) from a FSM filename. Tolerates the stray-space
    variant 'BG_012 _YYYYMMDD_HHMMSS__...'."""
    m = re.match(r"BG_012\s*_(\d{8})_(\d{6})__", name)
    if not m:
        return None
    ymd, hms = m.groups()
    try:
        return ymd, dt.datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def recording_start(meta_path: Path) -> dt.datetime | None:
    """fileCreateTime_original (preferred) or fileCreateTime from a SpikeGLX meta."""
    orig = None
    fallback = None
    try:
        for line in meta_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("fileCreateTime_original="):
                orig = line.split("=", 1)[1]
            elif line.startswith("fileCreateTime="):
                fallback = line.split("=", 1)[1]
    except OSError:
        return None
    val = orig or fallback
    if not val:
        return None
    try:
        return dt.datetime.strptime(val.strip(), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


# ── discovery ────────────────────────────────────────────────────────────
def index_fsm_blocks(fsm_dir: Path) -> dict[str, list[tuple[str, dt.datetime]]]:
    """{YYYYMMDD: [(stamp_key 'YYYYMMDD_HHMMSS', datetime), ...]} sorted by time.

    stamp_key identifies a behavioural block; its triplet is recovered by globbing
    the FSM dir for that stamp + each JSON_TYPE.
    """
    blocks: dict[str, set[tuple[str, dt.datetime]]] = defaultdict(set)
    for f in fsm_dir.iterdir():
        if not f.is_file() or not f.name.endswith(JSON_TYPES):
            continue
        st = fsm_stamp(f.name)
        if not st:
            continue
        ymd, when = st
        blocks[ymd].add((when.strftime("%Y%m%d_%H%M%S"), when))
    return {d: sorted(v, key=lambda x: x[1]) for d, v in blocks.items()}


def block_triplet(fsm_dir: Path, stamp_key: str) -> dict[str, Path]:
    """Map JSON_TYPE -> existing file for one block (handles the stray-space name)."""
    ymd, hms = stamp_key.split("_")
    out = {}
    for t in JSON_TYPES:
        cands = [p for p in fsm_dir.iterdir()
                 if p.is_file() and p.name.endswith(t)
                 and (st := fsm_stamp(p.name)) and st[1].strftime("%Y%m%d_%H%M%S") == stamp_key]
        if cands:
            out[t] = cands[0]
    return out


def session_imec_meta(session_dir: Path) -> Path | None:
    ksphy = session_dir / "Kilosort&Phy"
    if not ksphy.is_dir():
        return None
    for probe in sorted(ksphy.iterdir()):
        if probe.is_dir() and "_imec" in probe.name:
            metas = list(probe.glob("*tcat.imec0.ap.meta")) or list(probe.glob("*.ap.meta"))
            if metas:
                return metas[0]
    return None


# ── main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Time-match BG_012 FSM behaviour into Session/ dirs.")
    ap.add_argument("--execute", action="store_true", help="Perform copies (default: dry-run).")
    ap.add_argument("--tolerance-min", type=float, default=30.0,
                    help="Flag matches whose |rec - behaviour| exceeds this (min). Default 30.")
    ap.add_argument("--wephys-root", type=Path, default=None)
    args = ap.parse_args()

    root = args.wephys_root or wephys_root()
    subj = root / SUBJECT
    fsm_dir = subj / "FSM_behavior_data"
    proc = subj / "Processed data"
    raw = subj / "Raw data"
    for p in (subj, fsm_dir, proc):
        if not p.is_dir():
            print(f"ERROR: missing {p}")
            sys.exit(1)

    fsm = index_fsm_blocks(fsm_dir)
    print(f"FSM blocks: {sum(len(v) for v in fsm.values())} across {len(fsm)} dates")

    sessions = sorted(d for d in proc.iterdir() if d.is_dir() and (d / "Kilosort&Phy").is_dir())
    print(f"Ephys sessions (Processed): {len(sessions)}\n")

    # Pass 1: nearest FSM block per session (delta=None if no meta time / no block)
    assign: dict[str, dict] = {}
    for sdir in sessions:
        s = sdir.name
        ymd = session_date(s)
        meta = session_imec_meta(sdir)
        rec = recording_start(meta) if meta else None
        cands = fsm.get(ymd, []) if ymd else []
        if not cands:
            assign[s] = dict(ymd=ymd, stamp=None, delta=None, status="no-FSM")
        elif rec is None:
            assign[s] = dict(ymd=ymd, stamp=cands[0][0], delta=None, status="no-meta-time")
        else:
            stamp, when = min(cands, key=lambda c: abs((c[1] - rec).total_seconds()))
            assign[s] = dict(ymd=ymd, stamp=stamp,
                             delta=abs((when - rec).total_seconds()) / 60.0, status="ok")

    # Pass 2: each FSM block's CLOSEST owning session (resolve same-day ambiguity)
    owner: dict[str, str] = {}
    for s, a in assign.items():
        if a["stamp"] is None or a["delta"] is None:
            continue
        k = a["stamp"]
        if k not in owner or a["delta"] < assign[owner[k]]["delta"]:
            owner[k] = s

    # Pass 3: finalize status + plan copies for CONFIDENT, UNIQUE, in-tolerance matches only
    plan: list[tuple[Path, Path]] = []
    rows, anomalies = [], []
    for s, a in assign.items():
        k, d = a["stamp"], a["delta"]
        if a["status"] == "no-FSM":
            anomalies.append(f"{s}: no FSM block for date {a['ymd']} -> SKIP (no behaviour)")
        elif a["status"] == "no-meta-time":
            anomalies.append(f"{s}: meta has no fileCreateTime -> SKIP (cannot time-match)")
        elif d is not None and d > args.tolerance_min:
            a["status"] = "FAR"
            anomalies.append(f"{s}: nearest block {k} is {d:.0f} min away (>{args.tolerance_min:.0f}) -> SKIP")
        elif owner.get(k) != s:
            a["status"] = "AMBIG"
            anomalies.append(f"{s}: block {k} better matches '{owner.get(k)}' -> SKIP (ambiguous same-day)")
        else:
            trip = block_triplet(fsm_dir, k)
            if "__trials.json" not in trip:
                a["status"] = "NO-TRIALS"
                anomalies.append(f"{s}: block {k} has no __trials.json -> SKIP")
            else:
                a["status"] = "OK"
                dest = raw / s / "Session"
                for _t, src in trip.items():
                    plan.append((src, dest / src.name))
        rows.append((s, a["ymd"], k or "-", f"{d:.1f}" if d is not None else "n/a", a["status"]))

    # report
    print(f"{'session':52s} {'date':9s} {'FSM block':18s} {'dt(min)':8s} status")
    for s, ymd, blk, d, st in rows:
        print(f"{s:52s} {str(ymd):9s} {blk:18s} {d:8s} {st}")

    n_ok = sum(1 for _s, a in assign.items() if a["status"] == "OK")
    print(f"\nConfident matches: {n_ok}/{len(sessions)} sessions | Planned JSON copies: {len(plan)}")
    if anomalies:
        print(f"\n-- Skipped / flagged ({len(anomalies)}) --")
        for a in anomalies:
            print("  " + a)

    if not args.execute:
        print("\nDRY RUN — no files copied. Re-run with --execute (cluster-side) to copy.")
        return

    copied = skipped = errors = 0
    for src, dst in plan:
        try:
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            if dst.stat().st_size == src.stat().st_size:
                copied += 1
            else:
                errors += 1
                print(f"  SIZE MISMATCH: {dst}")
        except Exception as e:
            errors += 1
            print(f"  ERROR {src} -> {dst}: {e}")
    print(f"\nCopied={copied}  Skipped={skipped}  Errors={errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
