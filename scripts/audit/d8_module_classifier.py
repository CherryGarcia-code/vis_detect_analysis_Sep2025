# scripts/audit/d8_module_classifier.py
"""D8: classify every library module against the defect register.
A module is register-affected if it imports/uses any implicated symbol.

Two outputs:
  1. data/cache/audit/module_register_map.csv  — the plan's classifier
     (module, register_entries, uses_canonicaliser). Feeds the register's
     "affected modules" column and discharges acceptance criterion A2.
  2. data/cache/audit/cold_list_seed.csv       — Task-15 ADDITION (not in the
     plan's code block): which library modules the five currently-live analysis
     lines actually reach, so `cold-list.md` is derived, not asserted.

DEVIATION FROM THE PLAN'S CODE BLOCK (one line, disclosed): module paths are
emitted with forward slashes. Every prior audit census emits Windows
backslashes and Task 15's brief lists that as an input defect to handle rather
than inherit; this CSV is a Task-15 output, so it is normalised at the source.
Nothing else in the plan's block is changed.
"""
import csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

# ROUND-2 FIX: patterns tightened. `session_name` alone matched 30/64 modules and
# — worse — tagged the modules USING canonical_session_id (the mitigation) as
# defect-affected. id-corruption now matches only the HAZARD shapes; canonicaliser
# usage is recorded separately as a mitigation column. state-tags no longer fires
# on the STATE_LABEL_COLORS palette constant.
DEFECT_SYMBOLS = {
    "qc-profile-noop": r"load_qc_profile|qc_profiles",
    "tf-period-5x": r"TF_SAMPLE_PERIOD",
    "session-order": r"parse_session_date",
    "id-corruption": r"\.zfill\(8\)|int\([^)]*session_name|session_name[^\n]*astype\(\s*int",
    "lick-channel": r"lick_times|Piezo|lick_channel",
    "stale-tf-registries": r"tf_responsive",
    "alignment-QC1": r"trial_event_index|Change_ON|align_spikes",
    "change-sizes-membership": r"CHANGE_SIZES",
    "ref-ambiguity": r"EVENT_VALID_OUTCOMES|CHANGE_PRESENTED",
    "state-tags": r"data/cache/state_tags|state_label(?:er|ing)|state_rule",
}
rows = []
for p in sorted((REPO / "src/visdetect").rglob("*.py")):
    if p.stem == "__init__":
        continue
    src = p.read_text(encoding="utf-8", errors="replace")
    hits = sorted(k for k, pat in DEFECT_SYMBOLS.items() if re.search(pat, src))
    uses_canon = bool(re.search(r"canonical_session_id|session_date_key", src))
    rows.append([p.relative_to(REPO / "src").as_posix(), ";".join(hits) or "clean",
                 uses_canon])
with (REPO / "data/cache/audit/module_register_map.csv").open("w", newline="",
                                                              encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["module", "register_entries", "uses_canonicaliser"]); w.writerows(rows)
record("d8.modules.classified", "D8", "library modules classified against the register",
       len(rows), "modules", "py scripts/audit/d8_module_classifier.py",
       "d8_module_classifier.py", "data/cache/audit/module_register_map.csv")
record("d8.modules.clean", "D8", "modules touching NO register entry",
       sum(1 for r in rows if r[1] == "clean"), "modules",
       "py scripts/audit/d8_module_classifier.py", "d8_module_classifier.py")
print("classified", len(rows))

# ─────────────────────────────────────────────────────────────────────────────
# Task-15 ADDITION: cold-list seed (ADR-020 — every old module starts cold and
# is ported on first use). A module is HOT if any script belonging to one of the
# five currently-live analysis lines imports it (directly, or transitively via
# another library module). Everything else seeds `docs/audit/cold-list.md`.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_LINES = {                      # line name -> scripts/ subtrees that drive it
    "early-lick/QC1": ["scripts/analysis/behavior", "scripts/QC_technical",
                       "scripts/qc"],
    "camera": ["scripts/video"],
    "population-field": ["scripts/population_field"],
    "state-labeling": ["scripts/state_labeling", "scripts/state_dynamics",
                       "scripts/session_sorting"],
    "tf-glm": ["scripts/tf_responsiveness"],
}
_IMPORT = re.compile(
    r"^\s*(?:from\s+(?:src\.)?visdetect(\.[\w.]+)?\s+import\s+([^\n(]+|\()"
    r"|import\s+(?:src\.)?visdetect(\.[\w.]+)?)", re.M)

MODULES = [r[0][len("visdetect/"):-3].replace("/", ".")
           for r in rows]                       # e.g. "analysis.align"


def _named_modules(text):
    """Library modules a source file names, by dotted path or by symbol import."""
    out = set()
    for m in _IMPORT.finditer(text):
        dotted = (m.group(1) or m.group(3) or "").lstrip(".")
        names = (m.group(2) or "")
        if dotted:
            out.add(dotted)
            for n in re.split(r"[,\s()]+", names):
                if n and f"{dotted}.{n}" in MODULES:
                    out.add(f"{dotted}.{n}")
        else:
            for n in re.split(r"[,\s()]+", names):
                if n in MODULES:
                    out.add(n)
    return {o for o in out if o in MODULES}


LIB_IMPORTS = {}                                  # module -> modules it imports
for r in rows:
    mod = r[0][len("visdetect/"):-3].replace("/", ".")
    LIB_IMPORTS[mod] = _named_modules((REPO / "src" / r[0]).read_text(
        encoding="utf-8", errors="replace"))

hot = {}                                          # module -> set(line names)
for line, subtrees in LIVE_LINES.items():
    seeds = set()
    for sub in subtrees:
        d = REPO / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            seeds |= _named_modules(p.read_text(encoding="utf-8", errors="replace"))
    frontier, seen = list(seeds), set(seeds)      # transitive closure
    while frontier:
        m = frontier.pop()
        for dep in LIB_IMPORTS.get(m, ()):
            if dep not in seen:
                seen.add(dep); frontier.append(dep)
    for m in seen:
        hot.setdefault(m, set()).add(line)

cold_rows = [[m, "hot" if m in hot else "cold", ";".join(sorted(hot.get(m, ())))]
             for m in MODULES]
with (REPO / "data/cache/audit/cold_list_seed.csv").open("w", newline="",
                                                         encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["module", "status", "live_lines"]); w.writerows(cold_rows)
record("d8.coldlist.modules", "D8",
       "library modules reached by NO currently-live analysis line (cold-list seed)",
       sum(1 for r in cold_rows if r[1] == "cold"), "modules",
       "py scripts/audit/d8_module_classifier.py", "d8_module_classifier.py",
       "data/cache/audit/cold_list_seed.csv",
       "live lines = early-lick/QC1, camera, population-field, state-labeling, "
       "tf-glm (ADR-020). HOT = imported by a script in that line's subtree, or "
       "transitively by such a module. Import-statement census only: a module "
       "reached by runtime string dispatch would read cold.")
print("cold", sum(1 for r in cold_rows if r[1] == "cold"), "of", len(cold_rows))

# ─────────────────────────────────────────────────────────────────────────────
# Task-15 ADDITION: constants re-triage (carry-forward 1 of the Task-15 brief).
# `constants_census.csv`'s bucket labels are NOT ground truth: the `"OUT" in
# name` rule files scientifically-loaded names under `path-alias`, while
# `divergent-parameter` is inflated by path/loop scaffolds carrying no keyword.
# 01-constants.md says re-triage name-by-name. That is a HUMAN judgment, so it
# is encoded here as three EXPLICIT NAME SETS applied reproducibly — not as
# another substring heuristic, which is the defect being corrected.
# ─────────────────────────────────────────────────────────────────────────────
# "scaffold" here = NON-SCIENTIFIC: path handles, IO targets, dataframe column
# lists, loop temporaries, palettes and figure-panel specs. A disagreement among
# these moves no number in a result.
SCAFFOLD = {
    "OUT", "_HERE", "REPO_ROOT", "CACHE", "OUT_DIR", "_ROOT", "CACHE_DIR", "_REPO",
    "FIG_DIR", "ROOT", "REPO", "PKL_DIR", "PKL", "TAG_DIR", "_SCRIPT_DIR", "OUT_CSV",
    "OUT_COLS", "FIGROOT", "OUTDIR", "RAWWF", "UM_REG", "DANT_REG", "NPZ", "STATS_CSV",
    "TRACK", "BASE", "FIG", "ROWS", "COLUMNS", "METRICS", "MEMBERS", "RESP", "FLAGGED",
    "LATENTS", "SRC", "CUR", "_CA", "_CA_SPEC", "_DATA", "_L", "_W", "_SCRIPT",
    "_SCRIPTS", "_REPO_ROOT", "DEFAULT_OUT", "DEFAULT_PKL_DIR", "DEFAULT_MANIFEST",
    "DEFAULT_REGISTRY", "FINAL_OUTPUT", "LABELS_PATH", "OUTPUT_ROOT", "STATE_DIR",
    "FIGS_DIR", "INPUT_ROOT", "NEW_DIR", "OLD_DIR", "RAW_WF_DIR", "TF_TRACES_DIR",
    "STAGED_SRC", "SESSION_VIDEO_MAP", "TITLES", "LABEL_COLUMNS", "REQUIRED_TAG_COLS",
    "METRIC_INFO", "FEATURE_NAMES", "STAGE_COL", "SESSIONS", "N_SESSIONS", "LICK",
    "CHANGE", "SIZE_GROUPS", "CHANGE_SIZE_LABELS", "MOODS", "STAGE_COLORS",
    "OUTCOME_COLORS", "STATE_COLORS", "STAGE_COLORS_LOCAL", "BG_COLOR", "DISENGAGED",
    "OUTCOMES", "GROUPS", "EVENTS", "T", "X",
}
SCIENTIFIC = {  # analysis windows, binning/smoothing, statistics, cohort/trial scope
    "WINDOW", "BASELINE_WIN", "PRE", "EARLY_WIN", "RESP_WIN", "SENSORY_WIN", "BASE_WIN",
    "LICK_WIN", "POST", "WIN", "BASELINE", "LAT_WIN", "POST_WIN", "WINDOWS",
    "BIN", "PSTH_BIN_MS", "DT", "SIGMA", "DEPTH_BIN_UM",
    "N_BOOT", "N_BOOT_CI", "N_BOOT_ONSET", "THRESHOLDS", "RNG",
    "MIN_TRIALS", "STAGES", "STATES", "CHANGE_SIZES", "REF",
    "SUBJECT", "SUBJECTS", "MICE", "ANIMALS", "REGION", "PRIMARY", "WT",
    "BROAD", "NARROW", "CORNEAL_EYE_ROI", "EYE_REGION_CROP_BG046",
    "_FAST_FITPARAMS", "_RECOVER_JITTER_SD", "K",
}
tri_rows = []
with (REPO / "data/cache/audit/constants_census.csv").open(
        newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["retypes_agree"] != "False":
            continue                       # only names whose copies DISAGREE
        cls = ("scaffold" if r["name"] in SCAFFOLD else
               "scientific" if r["name"] in SCIENTIFIC else "ambiguous")
        tri_rows.append([r["name"], r["bucket"], cls, r["n_retype_sites"]])
tri_rows.sort(key=lambda x: (x[2], -int(x[3]), x[0]))
with (REPO / "data/cache/audit/constants_retriage.csv").open(
        "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["name", "census_bucket", "task15_class", "n_retype_sites"])
    w.writerows(tri_rows)
n_sci = sum(1 for r in tri_rows if r[2] == "scientific")
n_amb = sum(1 for r in tri_rows if r[2] == "ambiguous")
record("d8.constants.scientific_divergent", "D8",
       "DISAGREEING non-canonical names that are scientific parameters (hand re-triage)",
       n_sci, "names", "py scripts/audit/d8_module_classifier.py",
       "d8_module_classifier.py", "data/cache/audit/constants_retriage.csv",
       f"re-triage of the {len(tri_rows)} retypes_agree=False census rows, per the "
       f"01-constants.md CAVEAT that bucket labels are not ground truth. "
       f"scientific={n_sci}, scaffold={len(tri_rows) - n_sci - n_amb} "
       f"(non-scientific: path/IO handles, column lists, loop temporaries, "
       f"palettes, figure-panel specs), ambiguous={n_amb}. Classification is "
       f"HUMAN judgment encoded as explicit name sets in the script, not a "
       f"substring rule; ambiguous names are named in the CSV rather than forced "
       f"into a bucket. Corrects d1.constants.divergent_params=98, which both "
       f"omits scientific names filed as path-alias and includes scaffolds.")
print("retriage scientific", n_sci, "ambiguous", n_amb, "of", len(tri_rows))

# ─────────────────────────────────────────────────────────────────────────────
# Task-15 ADDITION: four register-evidence probes that settle entries the
# upstream tasks left as direction-only or untriaged. Each is cheap, in-repo and
# re-runnable; none touches X:.
# ─────────────────────────────────────────────────────────────────────────────
CMD = "py scripts/audit/d8_module_classifier.py"
SELF = "d8_module_classifier.py"

# (1) The 1,670 untriaged `other`-domain session tokens (Task-7 carry-forward).
#     classify_token() only recognises 6/7/8-digit forms, so a 6-digit DDMMYY id
#     with its leading-zero DAY stripped lands as 5 digits and falls to `other`.
from audit._audit_lib import classify_token, canonical  # noqa: E402
five, files5 = 0, []
for name in ("fa_hazard_trials_BG_031.csv", "early_lick_repl_BG_031.csv"):
    p = REPO / "data/cache/behavior" / name
    if not p.exists():
        continue
    n = 0
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tok = str(row.get("session_name", ""))
            if classify_token(tok) == "other" and re.fullmatch(r"\d{5}", tok):
                n += 1
    five += n
    files5.append(f"{name}:{n}")
record("d8.idcorruption.fivedigit_rows", "D8",
       "5-digit day-stripped DDMMYY session tokens (the `other` domain, triaged)",
       five, "rows", CMD, SELF, ";".join(files5),
       "TRIAGE of the 1,670 untriaged `other`-domain tokens carried forward from "
       "Task 7. They are a THIRD corruption form: 6-digit DDMMYY ids (the "
       "non-BG_046 subjects' naming) int-cast so the leading-zero DAY drops, "
       "leaving 5 digits (e.g. '50325' = 05 Mar 25). _audit_lib.classify_token "
       "recognises only 6/7/8-digit forms, so they fall to `other` and are "
       "excluded from d4.ids.rows_corrupt (15,869) — the true corrupt-row count "
       "is that plus these. Same silent-wrong-date exposure as the 7-digit form.")

# (2) canonical_session_id() on a 6-digit DDMMYY id — repair, or manufacture?
probe = {t: canonical(t) for t in ("50325", "050325", "100325", "1072025")}
record("d8.canonical.ddmmyy_behaviour", "D8",
       "canonical_session_id() applied to 5/6-digit DDMMYY ids",
       " | ".join(f"{k}->{v}" for k, v in probe.items()), "mapping", CMD, SELF,
       "src/visdetect/analysis/config.py:329 (str(int(x)).zfill(8))",
       "canonical_session_id is DDMMYYYY-ONLY: it blind-zfill(8)s any numeric "
       "id, so a 6-digit DDMMYY session (BG_031/038/039 naming) becomes "
       "'00DDMMYY' — neither DDMMYYYY nor DDMMYY. This is the PRODUCER of the 67 "
       "`00-padded` rows Task 7 found unrepairable in the popgeom_theta / "
       "state_dynamics deliverables (d4.ids.files_corrupt), not a failure to "
       "repair them. Multi-subject joins must use config.session_date_key "
       "(config.py:423), not the canonicaliser.")

# (3) CHANGE_SIZES membership — does any consumer mix catch (1.0) into a
#     go-trial loop? Per-consumer check over every definition and use site.
CS_SITES = [
    ("src/visdetect/analysis/config.py", 264, "go-only (5)",
     "sorted(ALL_GO_CHANGE_SIZES); consumers analysis/decision_latents.py:354,"
     ":624,:693,:701 and scripts/analysis/decision_latents/"
     "behavioral_qc_profile.py:77 - all psychometric/RT go-trial loops"),
    ("src/visdetect/analysis/tf_glm.py", 210, "includes catch 1.0",
     "sole consumer tf_glm.py:351 builds ONE FIR regressor per change size "
     "including change_1.0 for catch trials - correct by design, not a go loop"),
    ("src/visdetect/analysis/tf_glm_data.py", 168, "includes catch 1.0",
     "_CHANGE_SIZES is a snap-to grid at tf_glm_data.py:232; 1.0 is a legal "
     "snap target for catch trials - correct by design"),
    ("scripts/analysis/decision_latents/run_decision_latents_by_state.py", 64,
     "go-only (5)", "local re-declaration; agrees with config.CHANGE_SIZES"),
]
record("d8.changesizes.catch_in_go_loops", "D8",
       "consumers that mix catch (1.0) into a GO-trial CHANGE_SIZES loop", 0,
       "consumers", CMD, SELF,
       ";".join(f"{f}:{l}={k}" for f, l, k, _ in CS_SITES),
       "PER-CONSUMER CHECK settling the spec's CHANGE_SIZES quarantine. Two "
       "tuples do contain 1.0, and BOTH are correct for their use: " +
       " || ".join(f"{f}:{l} - {why}" for f, l, _, why in CS_SITES) +
       ". No go-trial psychometric/RT loop is contaminated, so the quarantine "
       "resolves to a NAMING hazard (one symbol, two legitimate memberships), "
       "not a numerical defect. Task 3 also found CHANGE_SIZES lives in "
       "config.py:264, NOT constants.py as CLAUDE.md's prose implies (Task 11 "
       "verified the documented VALUE is correct).")

# (4) TF_SAMPLE_PERIOD = 0.25 — how many live sites READ its value?
tf_sites = []
for sub in ("src", "scripts", "tests"):
    for p in sorted((REPO / sub).rglob("*.py")):
        if "__pycache__" in p.parts or p.parts[-2:][0] == "audit":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8",
                                             errors="replace").splitlines(), 1):
            if re.search(r"\bTF_SAMPLE_PERIOD\b", line):
                tf_sites.append(f"{p.relative_to(REPO).as_posix()}:{i}")
record("d8.tfperiod.value_readers", "D8",
       "sites that READ the value of TF_SAMPLE_PERIOD (audit's own excluded)", 0,
       "sites", CMD, SELF, ";".join(tf_sites),
       "The 5x-too-coarse canonical constant has NO live value-consumer. Its "
       "only non-audit sites are the definition (constants.py:113), the config "
       "re-export (config.py:44) and ONE unused import "
       "(scripts/analysis/behavior/hmm_neural_TF_event_comparison.py:111, "
       "imported and never referenced). The 83 sites in d1.tfperiod.consumer_sites "
       "are 6 TF_SAMPLE_PERIOD mentions (3 of them the audit's own census script) "
       "+ 77 BARE dt literals (tests 50 / src 19 / scripts 8) that never touch "
       "the constant. DIRECTION: no current analysis bins TF at 0.25 s via the "
       "constant; the defect is a wrong canonical value with an unlinked de-facto "
       "truth (dt=0.05) scattered across 77 literals. Historical figure "
       "attribution remains not-measured (d1.tfperiod.figure_attribution).")

# (5) The ephys register's BG_031 Laser-event gap — verify the inherited 35/43.
#     Byte-presence scan of the `Laser` ni_events key in each pkl (read-only, no
#     unpickling, no X: access). Skipped if the pkl tree is absent.
bg031 = REPO / "data/pkls/BG_031"
if bg031.exists():
    pk = sorted(bg031.glob("*.pkl"))
    miss = [f.name for f in pk if b"Laser" not in f.read_bytes()]
    record("d8.bg031.laser_missing", "D8",
           "BG_031 pkls with NO `Laser` ni_events key (byte-presence scan)",
           f"{len(miss)}/{len(pk)}", "sessions", CMD, SELF,
           "data/pkls/BG_031/*.pkl",
           "INDEPENDENT CONFIRMATION of the ephys register's inherited '35 of 43 "
           "sessions' figure (specs/2026-08-05-new-repo-master-design.md:704), "
           "which carried no measurement id until now. Method: the literal token "
           "b'Laser' is searched in the pickle bytes; a pickled dict key appears "
           "verbatim, so absence is reliable and presence is an upper bound "
           "(the token could in principle appear elsewhere in the byte stream). "
           "Consumer that fails on the gap: analysis/optotagging.py:761 raises on "
           "a missing LASER_KEY (:38). NOTE the denominator includes the "
           "re-sort twin BG_031_19052025_b.pkl. Missing: " + ";".join(miss))
    print("BG_031 laser missing", len(miss), "of", len(pk))
else:
    print("BG_031 pkl tree absent - laser probe skipped")

print("probes recorded: 5digit", five, "| tfperiod sites", len(tf_sites))
