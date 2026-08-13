# scripts/audit/d7_work_at_risk_supplement.py
"""D7 supplement: the three at-risk facts the disposition table rests on.

Mirrors the d6_retraction_survey.py precedent - a small companion to the main
D7 script so that every number cited in docs/audit/07-work-at-risk.md and
docs/audit/branch-disposition.md is a recorded measurement id, not loose prose.

STRICTLY READ-ONLY: git plumbing queries + os.lstat sizing. Nothing under
.claude/worktrees/, data/ or FIGURES/ is created, moved or deleted, and no
`git worktree` command is ever issued.
"""
import os
import stat as st
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d7_work_at_risk_supplement.py"
S = "d7_work_at_risk_supplement.py"


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout


def shb(*a):
    """Raw bytes, newline-normalised. text=True would decode with the console
    codec (cp1252) while the file is utf-8 -> spurious 'DIVERGENT' verdicts."""
    out = subprocess.run(a, capture_output=True, cwd=REPO).stdout
    return out.replace(b"\r\n", b"\n")


def _is_reparse(path):
    try:
        return bool(os.lstat(path).st_file_attributes & st.FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        return True


def _count_bytes(top):
    """(files, bytes, pruned) with the same every-level reparse discipline as
    _sized() in d7_work_at_risk.py. Path.rglob would descend THROUGH an inner
    junction and .stat() would follow it - the 2026-06-07 data-loss shape."""
    if _is_reparse(top):
        return 0, 0, [str(top)]
    n, size, pruned = 0, 0, []
    for root, dirs, fs in os.walk(top):
        keep = []
        for d in dirs:
            fp = os.path.join(root, d)
            if _is_reparse(fp):
                pruned.append(fp)
            else:
                keep.append(d)
        dirs[:] = keep
        for fname in fs:
            n += 1
            try:
                size += os.lstat(os.path.join(root, fname)).st_size
            except OSError:
                pass
    return n, size, pruned


# ---- 1. commits that exist on no origin ref (the sub-project -1 exposure) ----
local_only = sh("git", "rev-list", "--count", "--all", "--not", "--remotes").strip()
per_branch = []
for b in sh("git", "for-each-ref", "--format=%(refname:short)", "refs/heads").split():
    if sh("git", "rev-parse", "--verify", "-q", f"origin/{b}").strip():
        ahead = sh("git", "rev-list", "--count", f"origin/{b}..{b}").strip()
        if ahead != "0":
            per_branch.append(f"{b}:+{ahead}")
    else:
        per_branch.append(f"{b}:no-origin-ref")
fetch_head = REPO / ".git" / "FETCH_HEAD"
last_fetch = (date.fromtimestamp(fetch_head.stat().st_mtime).isoformat()
              if fetch_head.exists() else "unknown")
origin_tip = sh("git", "log", "-1", "--format=%cs", "origin/main").strip()
record("d7.local_only.commits", "D7",
       "commits reachable from local refs but no origin ref",
       f"{local_only} (" + ("; ".join(per_branch) if per_branch else "none") + ")",
       "commits", CMD, S,
       notes=f"origin/* are CACHED remote-tracking refs; last fetch {last_fetch}, "
             f"origin/main tip dated {origin_tip}; ls-remote unavailable "
             "(ssh agent not running) so remote state is unverifiable here")

# ---- 2. untracked working-tree entries: does the content exist on any ref? ----
refs = sh("git", "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/tags").split()
trees = {r: set(sh("git", "ls-tree", "-r", "--name-only", r).splitlines()) for r in refs}
untracked = [l[3:].strip() for l in sh("git", "status", "--porcelain").splitlines()
             if l.startswith("??")]
verdicts = []
for entry in untracked:
    p = REPO / entry
    files = ([f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p])
    for f in files:
        rel = f.relative_to(REPO).as_posix()
        if rel.startswith("scripts/audit/"):
            continue          # this audit's own new scripts, committed by this task
        base = rel.rsplit("/", 1)[-1]
        # match on basename, not path: the same work can sit at a different path on a branch
        hits = [(r, x) for r in refs for x in trees[r]
                if x == rel or x.rsplit("/", 1)[-1] == base]
        if hits:
            mine = f.read_bytes().replace(b"\r\n", b"\n")
            same = any(shb("git", "show", f"{r}:{x}") == mine for r, x in hits)
            verdicts.append(f"{rel}=on-ref({hits[0][0]}:{hits[0][1]}"
                            f"{'/identical' if same else '/DIVERGENT'})")
        else:
            verdicts.append(f"{rel}=NO-REF")
at_risk = sum(1 for v in verdicts if v.endswith("NO-REF") or "DIVERGENT" in v)
record("d7.untracked.at_risk", "D7",
       "untracked working-tree files whose content exists on no ref",
       f"{at_risk}/{len(verdicts)} at risk: " + "; ".join(verdicts), "files", CMD, S,
       notes="primary checkout only; audit's own new scripts excluded (committed by Task 12)")

# ---- 3. irreplaceable gitignored hand-label sets + off-tree backup coverage ----
sets = {"data/cache/tf_labeling": "TF unit labels",
        "data/cache/state_tags": "behavioural state tags",
        "data/cache/session_sorting": "blinded session sorter",
        "data/cache/video_sync": "video-sync / pupil labels"}
parts, hl_pruned = [], []
for rel, label in sets.items():
    p = REPO / rel
    if not p.exists():
        parts.append(f"{rel}: absent")
        continue
    n, size, pruned = _count_bytes(str(p))
    hl_pruned += pruned
    tracked = len([x for x in sh("git", "ls-files", rel).splitlines() if x])
    parts.append(f"{rel} ({label}): {n} files, {size/1e6:.1f} MB, {tracked} tracked"
                 + (f" [pruned {len(pruned)} junction(s)]" if pruned else ""))
backup = Path("e:/python_analysis/_handlabel_backup_20260806")
bn, bsize, bpruned = _count_bytes(str(backup)) if backup.exists() else (0, 0, [])
hl_pruned += bpruned
record("d7.handlabels.exposure", "D7",
       "irreplaceable gitignored hand-label sets and their backup coverage",
       " | ".join(parts) + f" || backup {backup.as_posix()}: "
       + (f"{bn} files, {bsize/1e6:.1f} MB" if bn else "MISSING"),
       "inventory", CMD, S,
       notes="no code can regenerate these; backup is on the SAME physical disk (E:) "
             "as the repo, so it is a second copy, not an off-disk copy"
             # only widens the recorded string if a junction actually appears, so the
             # walk-discipline fix does not by itself churn measurements.csv
             + (f"; junctions pruned: {'; '.join(hl_pruned)}" if hl_pruned else ""))
print("done")
