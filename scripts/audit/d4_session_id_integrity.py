# scripts/audit/d4_session_id_integrity.py
"""D4a: extend the session-id integrity check to FIGURES/ and table_output/,
classify the key domain of every CSV carrying a session-id column, and
QUANTIFY rows lost when joined to the staging manifest."""
import csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record, classify_token, canonical

CMD = "py scripts/audit/d4_session_id_integrity.py"; S = "d4_session_id_integrity.py"
# 'k' removed (round-2 fix): k-means hyperparameter columns would be misread as ids
ID_COLS = {"session_name", "session", "session_id", "sess", "sid", "session_key"}
ROOTS = ["data", "FIGURES", "table_output"]
SKIP = {"audit", "__pycache__"}

import pandas as pd
manifest = pd.read_csv(REPO / "data/BG_046_staging_manifest.csv", dtype=str)
manifest_keys = {canonical(x) for x in manifest["session_name"]}

out_rows, bad_files, total_bad_rows = [], 0, 0
for root in ROOTS:
    for f in (REPO / root).rglob("*.csv"):
        if any(x in f.parts for x in SKIP) or f.stat().st_size > 200_000_000:
            continue
        try:
            head = pd.read_csv(f, nrows=5, dtype=str)
        except Exception:
            continue
        col = next((c for c in head.columns if c.lower() in ID_COLS), None)
        if col is None:
            continue
        subj_col = next((c for c in head.columns if c.lower() == "subject"), None)
        try:   # round-2 fix: full read guarded too — one ragged/non-UTF8 legacy
            usecols = [col] + ([subj_col] if subj_col else [])   # CSV must not kill the scan
            df = pd.read_csv(f, usecols=usecols, dtype=str)
        except Exception:
            out_rows.append([str(f.relative_to(REPO)), -1, "READ-ERROR", False, 0])
            continue
        toks = df[col].dropna()
        domains = toks.map(classify_token).value_counts().to_dict()
        n_bad = sum(v for k, v in domains.items()
                    if k in ("7digit-stripped", "float-string", "00-padded"))
        # join-loss vs the BG_046 manifest — only for rows that are actually
        # BG_046-scoped. Round-2 fix: multi-subject caches carry the subject in
        # a COLUMN (session_group_features.csv, popgeom_theta deliverables), not
        # the filename — join only subject==BG_046 rows where a column exists;
        # fall back to the filename heuristic otherwise.
        if subj_col is not None:
            mask = df[subj_col] == "BG_046"
            scoped = df.loc[mask, col].dropna()
            lost = (-1 if scoped.empty else int(
                (~scoped.map(lambda x: canonical(re.sub(r"^BG_\d{3}_", "", str(x))))
                 .isin(manifest_keys)).sum()))
        else:
            name_l = f.name.lower()
            other_subject = any(s in name_l for s in
                                ("bg_012", "bg_031", "bg_038", "bg_039",
                                 "bg_040", "bg_041", "bg_049"))
            if other_subject:
                lost = -1
            else:
                lost = int((~toks.map(lambda x: canonical(
                    re.sub(r"^BG_\d{3}_", "", str(x)))).isin(manifest_keys)).sum())
        out_rows.append([str(f.relative_to(REPO)), len(toks), str(domains),
                         lost >= 0, max(lost, 0)])
        if n_bad:
            bad_files += 1; total_bad_rows += n_bad

with (REPO / "data/cache/audit/csv_key_domains.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["file", "n_rows", "domains", "joinable_to_manifest", "rows_lost_on_join"])
    w.writerows(out_rows)

record("d4.ids.files_scanned", "D4", "CSV files with a session-id column (data+FIGURES+table_output)",
       len(out_rows), "files", CMD, S, "data/cache/audit/csv_key_domains.csv")
record("d4.ids.files_corrupt", "D4", "files containing stripped/float/00-padded tokens",
       bad_files, "files", CMD, S)
record("d4.ids.rows_corrupt", "D4", "total corrupted-token rows",
       total_bad_rows, "rows", CMD, S,
       notes="recon baseline 15,802 across 6 caches; this extends scope to FIGURES+table_output")
print(f"files={len(out_rows)} corrupt_files={bad_files} corrupt_rows={total_bad_rows}")
