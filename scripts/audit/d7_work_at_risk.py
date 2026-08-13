# scripts/audit/d7_work_at_risk.py
import os
import stat as st
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d7_work_at_risk.py"; S = "d7_work_at_risk.py"

def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout


def _is_reparse(path):
    # ROUND-2 FIX (blocker): NTFS junctions are NOT symlinks on Python 3.10 —
    # is_symlink() misses them, and rglob happily descends THROUGH them.
    # Live case: .claude/worktrees/qc1-alignment/data/pkls IS a junction into
    # the primary ~30 GB pkl tree (the Jun-7 data-loss shape). Prune reparse
    # points at EVERY level.
    try:
        return bool(os.lstat(path).st_file_attributes & st.FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        return True   # unreadable: treat as boundary, never descend


def _sized(top):
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


# gitignored artefact volume per worktree + primary (junction-pruned)
wt_root = REPO / ".claude" / "worktrees"
lines, all_pruned = [], []
for wt in ([REPO] + (sorted(wt_root.iterdir()) if wt_root.exists() else [])):
    for sub in ("data", "FIGURES"):
        p = wt / sub
        if not p.exists():
            continue
        n, size, pruned = _sized(str(p))
        all_pruned += pruned
        lines.append(f"{wt.name}/{sub}: {n} files, {size/1e9:.1f} GB"
                     + (f" [pruned {len(pruned)} junction(s)]" if pruned else ""))
record("d7.gitignored.volume", "D7", "gitignored data/FIGURES volume per tree (junction-pruned)",
       " | ".join(lines), "inventory", CMD, S,
       notes="no branch migration carries any of this; junctions pruned: "
             + ("; ".join(all_pruned) if all_pruned else "none"))

# unmerged branches with unique commits
branches = sh("git", "for-each-ref", "--format=%(refname:short)", "refs/heads").split()
uniq = {}
for b in branches:
    if b == "main":
        continue
    n = sh("git", "rev-list", "--count", f"main..{b}").strip()
    cherry = sh("git", "cherry", "main", b)
    real = sum(1 for l in cherry.splitlines() if l.startswith("+"))
    uniq[b] = (n, real)
record("d7.branches.unmerged", "D7", "branches ahead of main (raw/cherry-verified unique)",
       "; ".join(f"{b}:{n}/{r}" for b, (n, r) in uniq.items()), "commits", CMD, S,
       notes="cherry-verified 0 == already applied under a rewritten sha (safe to drop)")

# stash-tags content
for tag in ["pre-tidy-20260628/stash-0", "pre-tidy-20260628/stash-1"]:
    stat = sh("git", "show", "--stat", "--format=%cs %s", tag).strip().splitlines()
    record(f"d7.stash.{tag[-1]}", "D7", f"stash-tag {tag} content",
           " | ".join(stat[:6]), "difftext", CMD, S)
print("done")
