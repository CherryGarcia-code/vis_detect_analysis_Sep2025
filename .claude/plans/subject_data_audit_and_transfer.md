# Plan: Multi-Subject Data Organization & Channel Map Scan

## Context

Three priority subjects (BG_031, BG_038, BG_039) need data organized to match the BG_046 gold-standard structure before Kilosort and .pkl conversion can proceed.

## Current Status

| Subject | Session dirs | Session/ populated | Session/ empty | raw_backup → Raw data gaps | KS4 sorted | TPrime done | IMRO mapped |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BG_046 (ref) | 53 | 48 | 5 | 0 | All | All | 1 IMRO |
| BG_031 | 91 | 62 | 28 | 0 | 44 | No | 2 IMROs |
| BG_038 | 53 | 2 | 51 | 0 | 0 | No | **Need scan** |
| BG_039 | 35 | 0 | 35 | **8 sessions** | 0 | 32 (CatGT) | **Need scan** |

---

## Step 1: Write `scripts/data_management/organize_subject_data.py`

A safe, reusable Python script that:

1. **Scans** a subject's top-level directory for loose behavioral JSONs
2. **Matches** JSONs to `Raw data/` session dirs by date:
   - JSON: `YYYYMMDD` format → Session dir: `DDMMYYYY` format
   - Handles multi-run dates (multiple triplets → same Session/ dir)
3. **Identifies** raw_backup sessions missing from `Raw data/`
4. **Generates dry-run report** showing all planned operations:
   - Section A: Copy FSMdata JSONs → `Raw data/{session}/Session/`
   - Section B: Copy raw_backup sessions → `Raw data/` with proper structure
   - Section C: Anomalies (mismatched dates, duplicates, issues)
5. **Executes** only with `--execute` flag

**Safety features:**
- Copy-only (never deletes/moves)
- Logs every operation to a timestamped log file
- Skips files that already exist at destination
- Verifies file sizes match after copy
- `--dry-run` is default behavior

## Step 2: Run dry-run for each subject

```bash
py scripts/data_management/organize_subject_data.py --subject BG_031 --dry-run
py scripts/data_management/organize_subject_data.py --subject BG_038 --dry-run
py scripts/data_management/organize_subject_data.py --subject BG_039 --dry-run
```

User reviews output, then executes with `--execute`.

**BG_039 special case:** 8 sessions in raw_backup need to be copied into `Raw data/` with subdirectory structure (EphysNidaq/, Session/, Cameras/). These are large (~90 GB AP bins each = ~720 GB total). Consider running overnight.

## Step 3: Run channel map scan for BG_038 and BG_039

Modify `check_session_chanmaps.py` to accept path as CLI arg (currently uses `input()`), then run:

```bash
py scripts/kilosort_related/check_session_chanmaps.py BG_038 --path "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_038"
py scripts/kilosort_related/check_session_chanmaps.py BG_039 --path "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_039"
```

Output → `data/subject_session_imro_matching/{subject}/` CSVs.

---

## Deliverables

1. `scripts/data_management/organize_subject_data.py` — reusable organization script
2. Dry-run reports for BG_031, BG_038, BG_039
3. Channel map CSVs for BG_038 and BG_039

## Deferred (Not This Phase)

- Kilosort spike sorting (compute-heavy, external)
- TPrime correction (BG_031: run on existing KS4; BG_038/039: after KS4)
- .pkl conversion (after KS4 + TPrime)
- BG_012 (low priority per user)
- Anomaly cleanup (stray files, misnamed dirs)

## Risk Assessment

- **Data safety**: Copy-only. Size verification. Dry-run default. Never deletes.
- **Network drive**: Large files; 720 GB transfer for BG_039 raw_backup sessions.
- **Existing files**: Script skips destinations that already exist.
