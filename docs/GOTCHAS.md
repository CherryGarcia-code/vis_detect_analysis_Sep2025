# Gotchas and Pitfalls

Common traps in this codebase. Read before starting any new analysis.

| Gotcha | Detail |
|--------|--------|
| `py` not `python` | Windows + Git Bash requires `py` to invoke Python |
| Legacy pickle paths | `RenamingUnpickler` handles 10+ historical module paths. Don't panic about import errors on load. |
| pre-TPrime = stale | Files in `preTprime/` directories are from before spike time correction. Do NOT use for new analyses. |
| Session name format | DDMMYYYY as integer (e.g., `7072025` = July 7, 2025). Use `parse_session_date()` and `chronological_sort()`. |
| `change_size` determines trial type | Go vs catch is from `change_size`, NOT from the `trialoutcome` label. |
| `fa` ≠ SDT false alarm | The `fa` behavioral label means early/anticipatory lick. SDT FAs are `hit` outcomes on catch trials. |
| Memory management | Always `del sess; gc.collect()` after processing each session in loops. Sessions are large (~100+ MB). |
| Search before writing | **Always search the codebase for existing functions before writing new ones.** |
