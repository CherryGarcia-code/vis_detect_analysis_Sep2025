# scripts/audit/d1_executed_checks.py
"""D1 executed checks on ONE real session (BG_046 01072025 - also exercises the
day-1-9 id path) + ref-trial check over 5 sessions.

(1) load_qc_profile for all 4 named profiles -> recorded dicts
(2) unit count per selection path on the same session
(3) ref trials: was the change presented? (RT relative to change_time)
(4) TF sample period measured from the stimulus log
"""
import gc, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit._audit_lib import REPO, record

from visdetect.core.qc import load_qc_profile, apply_unit_filters, find_good_stable_units
from visdetect.suite.loader import load_session
from visdetect.analysis.utils import get_good_cluster_ids

CMD = "py scripts/audit/d1_executed_checks.py"
S = "d1_executed_checks.py"

# --- (1) profiles ---
for name in ["default", "qc_only", "striatal_strict", "striatal_lenient"]:
    prof = load_qc_profile(name)
    record(f"d1.qcprofile.{name}", "D1", f"load_qc_profile('{name}') returned dict",
           repr(prof), "dict", CMD, S, "src/visdetect/core/qc.py:218",
           "empty dict == silent no-op defect confirmed")

# --- (2) unit counts per path on one session ---
sess = load_session("01072025")
n_good_and_stable = len(getattr(sess, "good_and_stable_ids", []) or [])
n_getgood_1hz = len(get_good_cluster_ids(sess, min_rate_hz=1.0))
n_getgood_01hz = len(get_good_cluster_ids(sess, min_rate_hz=0.1))
record("d1.frfloor.good_and_stable", "D1", "units via good_and_stable_ids (0.5Hz ingest gate)",
       n_good_and_stable, "units", CMD, S, "core/qc.py:269 find_good_stable_units")
record("d1.frfloor.getgood_1hz", "D1", "units via get_good_cluster_ids(min 1.0Hz)",
       n_getgood_1hz, "units", CMD, S, "analysis/utils.py:216")
record("d1.frfloor.getgood_01hz", "D1", "units via get_good_cluster_ids(min 0.1Hz, yml value)",
       n_getgood_01hz, "units", CMD, S, "config/qc_profiles.yml:8")
record("d1.frfloor.spread", "D1", "unit-count spread across live selection paths (session 01072025)",
       f"{n_good_and_stable}/{n_getgood_1hz}/{n_getgood_01hz}", "units", CMD, S,
       notes="one session, three floors, three different populations")

# --- (2b) PRE-FLIGHT FIX: the spec's headline blast-radius number — unit counts
# under each named profile's INTENDED thresholds (read from the YAML directly)
# vs the DEFAULTED gate that the load_qc_profile() -> {} defect actually applies.
# Metrics are built explicitly here so no unverified qc-module builder is assumed.
import yaml
import pandas as pd
mets = []
for c in sess.clusters:
    spk = np.asarray(c.spike_times, float)
    dur = float(spk[-1] - spk[0]) if len(spk) > 1 else 1.0
    isi = np.diff(spk) if len(spk) > 1 else np.array([np.inf])
    mets.append({"cluster_id": c.cluster_id, "n_spikes": len(spk),
                 "mean_rate_hz": len(spk) / max(dur, 1e-9),
                 "isi_viol_frac": float(np.mean(isi < 0.002))})
mdf = pd.DataFrame(mets)
profiles_yaml = yaml.safe_load((REPO / "config/qc_profiles.yml").read_text(encoding="utf-8"))
for pname, prof in profiles_yaml.items():
    keep = ((mdf["mean_rate_hz"] >= prof.get("min_mean_rate_hz", 0.0)) &
            (mdf["isi_viol_frac"] <= prof.get("max_isi_viol_frac", 1.0)) &
            (mdf["n_spikes"] >= prof.get("min_total_spikes", 0)))
    record(f"d1.qcprofile.diff.{pname}", "D1",
           f"units passing the YAML-INTENDED '{pname}' thresholds (session 01072025)",
           int(keep.sum()), "units", CMD, S, "config/qc_profiles.yml",
           notes="compare across profiles: under the live {} defect all named "
                 "profiles collapse to the function defaults, so these intended "
                 "counts differ from what any --profile run actually used")

# --- (3) ref trials across 5 sessions ---
# PRE-FLIGHT FIX (blocker): Trial has NO attribute RT/rt. Reaction times live in
# Trial.reactiontimes: Dict[str, float], keyed by the RAW capitalized outcome
# token ("Ref"/"FA"; "RT" for Hit) — see session.py:22-35 and align.py:115-127.
REF_SESSIONS = ["01072025", "23062025", "08072025", "15072025", "30062025"]
del sess; gc.collect()
tot_ref, ref_with_change, rts, rt_keys_seen = 0, 0, [], set()
for sname in REF_SESSIONS:
    try:
        s = load_session(sname)
    except Exception as e:
        print(f"skip {sname}: {e}"); continue
    n_ref_s, n_ct_s = 0, 0
    for t in s.trials:
        raw = str(getattr(t, "trialoutcome", getattr(t, "outcome", "")))
        if raw.lower() == "ref":
            tot_ref += 1; n_ref_s += 1
            ct = getattr(t, "change_time", None)
            rtd = getattr(t, "reactiontimes", None) or {}
            rt_keys_seen |= set(rtd.keys())
            rt = rtd.get(raw, rtd.get(raw.capitalize(), rtd.get("RT")))
            if ct is not None and not (isinstance(ct, float) and np.isnan(ct)):
                ref_with_change += 1; n_ct_s += 1
                if rt is not None and not (isinstance(rt, float) and np.isnan(rt)):
                    rts.append(float(rt))
    record(f"d1.ref.per_session.{sname}", "D1",
           f"ref trials / with change_time (session {sname})",
           f"{n_ref_s}/{n_ct_s}", "trials", CMD, S)
    del s; gc.collect()
record("d1.ref.total", "D1", "ref trials across 5 sessions", tot_ref, "trials", CMD, S)
record("d1.ref.with_change_time", "D1", "ref trials with a valid change_time",
       ref_with_change, "trials", CMD, S,
       notes="if ~=total, the change WAS presented on ref trials -> "
             "CHANGE_PRESENTED_OUTCOMES incl. Ref is factually right and "
             "EVENT_VALID_OUTCOMES excluding ref is a scientific choice, not a fact")
record("d1.ref.rt_dict_keys", "D1", "reactiontimes dict keys observed on ref trials",
       ";".join(sorted(rt_keys_seen)) or "none", "keys", CMD, S)
if rts:
    record("d1.ref.rt_median_ms", "D1", "median RT on ref trials (from change)",
           round(1000 * float(np.median(rts))), "ms", CMD, S,
           notes="small positive RT = lick AFTER change onset = reflex")
else:
    record("d1.ref.rt_median_ms", "D1", "median RT on ref trials", "not-measured",
           "ms", CMD, S, notes="no RT found under observed reactiontimes keys - "
           "report the rt_dict_keys row and settle via ni_events lick times")

# --- (4) TF sample period from the stimulus log ---
s = load_session("01072025")
period = None
for t in s.trials[:80]:
    vbl = getattr(t, "stim_vbl", None)
    tfd = getattr(t, "stim_tf_disp", None)
    if vbl is None or tfd is None:
        continue
    vbl, tfd = np.asarray(vbl, float).ravel(), np.asarray(tfd, float).ravel()
    if len(vbl) < 20 or len(tfd) < 20:
        continue
    changes = np.where(np.diff(tfd) != 0)[0]
    if len(changes) > 5:
        period = float(np.median(np.diff(vbl[changes])))
        break
if period is not None:
    record("d1.tfperiod.measured_s", "D1", "measured TF update period from stim log",
           round(period, 4), "s", CMD, S, "constants.py:113 TF_SAMPLE_PERIOD=0.25",
           notes="expected ~0.05; 0.25 is the known-wrong canonical value")
else:
    record("d1.tfperiod.measured_s", "D1", "measured TF update period", "not-measured",
           "s", CMD, S, notes="stim logs None on this pkl (legacy, pre-backfill); "
           "fall back to psychophysical_kernel.py:18 documentary evidence")
del s; gc.collect()
print("done")
