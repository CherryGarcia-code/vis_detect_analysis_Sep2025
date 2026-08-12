# scripts/audit/d4_staleness.py
"""D4b: (1) rank caches by mtime-vs-producing-code (2) SESSION_FILTER
divergence for direct-manifest readers (3) twin date-key collisions
against the real pkl tree."""
import csv, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

CMD = "py scripts/audit/d4_staleness.py"; S = "d4_staleness.py"

# (1) staleness: cache mtime vs last commit touching the writer that names it.
# ROUND-2 FIXES: --untracked (two topics' writers are untracked scripts);
# splitlines (tracked paths contain spaces); tri-state — 7 of 23 topics have NO
# literal path reference in code, and 'no writer found' must never read as
# 'not stale'.
writers = {}   # topic -> (newest_commit, newest_artefact, n_files, verdict)
for topic_dir in sorted((REPO / "data/cache").iterdir()):
    if not topic_dir.is_dir() or topic_dir.name == "audit":
        continue
    hits = subprocess.run(["git", "grep", "--untracked", "-l",
                           f"data/cache/{topic_dir.name}", "--", "scripts/", "src/"],
                          capture_output=True, text=True, cwd=REPO).stdout.splitlines()
    newest = ""
    for h in hits:
        d = subprocess.run(["git", "log", "-1", "--format=%cs", "--", h],
                           capture_output=True, text=True, cwd=REPO).stdout.strip()
        newest = max(newest, d)
    # DEVIATION from the plan's code block, authorized by its own Step-2 gate
    # ("tf_responsive MUST rank as stale ...; if it does not, the heuristic is
    # broken - fix before committing"): exclude documentation (.md) files from
    # the artefact-mtime scan — tf_responsive's in-place STALE banner (README.md,
    # amended 2026-08-03) otherwise refreshes the topic's max mtime past the
    # newest writer commit (2026-07-20) and hides the staleness.
    files = [p for p in topic_dir.rglob("*")
             if p.is_file() and p.suffix.lower() != ".md"]
    if not files:
        continue
    newest_artefact = max(datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                          for p in files).date().isoformat()
    verdict = ("no-writer-found" if not newest
               else "stale" if newest_artefact < newest else "current")
    writers[topic_dir.name] = (newest, newest_artefact, len(files), verdict)
rows = sorted(((t, *v) for t, v in writers.items()), key=lambda r: r[4], reverse=True)
with (REPO / "data/cache/audit/stale_caches.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["topic", "newest_writer_commit", "newest_artefact_mtime", "n_files", "verdict"])
    w.writerows(rows)
n_stale = sum(1 for r in rows if r[4] == "stale")
n_unknown = sum(1 for r in rows if r[4] == "no-writer-found")
record("d4.stale.topics", "D4",
       "cache topics stale / writer-untraceable (excluded from denominator)",
       f"{n_stale}/{n_unknown}", "topics", CMD, S, "data/cache/audit/stale_caches.csv",
       notes="mtime heuristic over measurable topics only; known-stale tf_responsive "
             "must appear as stale or the metric is wrong")

# (2) SESSION_FILTER divergence
from visdetect.analysis.config import load_staging_manifest
import pandas as pd
filt = set(load_staging_manifest(qc_only=True)["session_name"].astype(str))
raw = set(pd.read_csv(REPO / "data/BG_046_staging_manifest.csv", dtype=str)["session_name"])
record("d4.filter.divergence", "D4",
       "sessions a direct-manifest reader sees that load_staging_manifest(qc_only=True) filters out",
       len(raw - filt), "sessions", CMD, S,
       notes="28 scripts read the CSV directly (recon); each sees this many extra sessions")

# (3) twin collisions against the real pkl tree.
# PRE-FLIGHT FIX (blocker): every pkl on disk is SUBJECT-PREFIXED
# (BG_046_01072025.pkl, BG_012_01112023_prot4_lickEndsTrial.pkl), so naive
# stem slicing produces garbage keys. Extract the date as the first standalone
# 6-8 digit token, and record WHICH TWIN WINS per the resolver semantics
# (suite/loader.py:120-135: plain form wins; unique suffix falls back;
# genuinely ambiguous -> None).
import re as _re
coll = {}
for subj_dir in sorted((REPO / "data/pkls").iterdir()):
    if not subj_dir.is_dir():
        continue
    seen = {}
    for p in subj_dir.glob("*.pkl"):
        m = _re.search(r"(?<!\d)\d{6,8}(?!\d)", p.stem)
        if not m:
            continue
        seen.setdefault(m.group(0), []).append(p.name)
    coll[subj_dir.name] = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
total = sum(len(v) for v in coll.values())
record("d4.twins.colliding_date_keys", "D4", "date keys with >1 pkl (twins) across subjects",
       total, "keys", CMD, S,
       notes="; ".join(f"{s}:{len(v)}" for s, v in coll.items() if v))
winners = []
for subj, keys in coll.items():
    for k, files in keys.items():
        plain = [f for f in files if _re.fullmatch(rf"{subj}_{k}\.pkl", f)]
        if plain:
            verdict = plain[0]
        else:
            # round-2 fix: replicate the resolver's unique-suffix fallback
            # (loader.py:129-135) rather than declaring all no-plain sets ambiguous
            suffixed = [f for f in files
                        if not f[:-4].rsplit("_", 1)[-1].isdigit()]
            verdict = suffixed[0] if len(suffixed) == 1 else "AMBIGUOUS(None)"
        winners.append(f"{subj}/{k}->{verdict}")
record("d4.twins.winners", "D4",
       "which twin the resolver serves per colliding key (deterministic by code path)",
       " | ".join(winners[:12]) + (f" (+{len(winners)-12} more)" if len(winners) > 12 else ""),
       "verdicts", CMD, S, "src/visdetect/suite/loader.py:120-135",
       notes="plain form wins; no plain form and >1 suffix -> resolver returns None")

# (4) PRE-FLIGHT ADDITION: the spec's named staleness value-check — the chron
# column of predicted_session_groups.csv holds tuples current code cannot produce
psg = REPO / "data/cache/session_sorting/predicted_session_groups.csv"
import pandas as _pd
chron = _pd.read_csv(psg, dtype=str)["chron"]
bad = chron[chron.str.contains(r"\(\s*\d{1,4},\s*(1[3-9]|[2-9]\d),", regex=True, na=False)]
record("d4.stale.chron_impossible", "D4",
       "predicted_session_groups.csv chron tuples with month>12 (LOWER BOUND: "
       "same-parse rows with mis-placed day <=12 are not countable by pattern)",
       len(bad), "rows", CMD, S, str(psg.relative_to(REPO)),
       notes="frozen output of the pre-fix parser; proves the staleness class "
             "beyond the mtime heuristic. Per-script SESSION_FILTER diffs are an "
             "upper-bound proxy (some direct readers apply own filters) - stated "
             "as such in 04-artefacts.md")
print("done")
