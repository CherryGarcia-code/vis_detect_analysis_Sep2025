"""Diagnostic plots for optotagging waveform metrics

This script loads the merged units CSV and Bombcell templates/raw waveforms
and produces:
 - histograms of peak width (us), peak-valley (us), amplitude, mean FR
 - scatter of (PW vs PV) colored by class
 - exemplar waveform plots for percentile units

Saves outputs to notebooks/BG_031_260325/optotagging/diagnostics/
"""

import numpy as np
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "notebooks" / "BG_031_260325" / "optotagging" / "diagnostics"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Filenames (adjust if different)
UNITS_CSV = (
    ROOT
    / "notebooks"
    / "BG_031_260325"
    / "optotagging"
    / "units_optotagging_merged.csv"
)
TEMPLATES_NPY = (
    ROOT / "notebooks" / "260325" / "bombcell" / "templates._bc_rawWaveforms.npy"
)

print("Reading", UNITS_CSV)
if not UNITS_CSV.exists():
    raise SystemExit(f"Missing units CSV: {UNITS_CSV}")

units = []
with open(UNITS_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        units.append(r)

print("Loaded", len(units), "units from CSV")


# Convert numeric fields if present
def tofloat(x, default=np.nan):
    try:
        return float(x) if x is not None and x != "" else default
    except:
        return default


for u in units:
    u["unit_id"] = int(u.get("unit_id") or u.get("cluster_id") or -1)
    u["mean_fr"] = tofloat(
        u.get("mean_fr") or u.get("meanFR") or u.get("mean_firing_rate")
    )
    u["peak_width_us"] = tofloat(
        u.get("peak_width_us") or u.get("peak_width") or u.get("PW_us")
    )
    u["peak_valley_us"] = tofloat(
        u.get("peak_valley_us") or u.get("pv_us") or u.get("PV_us")
    )
    u["template_amp"] = tofloat(
        u.get("template_amp") or u.get("amp") or u.get("template_amplitude")
    )
    u["snr"] = tofloat(u.get("snr") or u.get("SNR"))
    u["class"] = u.get("class") or u.get("label") or u.get("optotag_class") or "Unknown"
# Prefer filtering using session-level QC if available in the pickle.
orig_count = len(units)
good_ids = set()
try:
    # session IO provides load_session which normalizes a pickle into Session dataclass
    from src.session_io import load_session

    PKL_PATH = ROOT / "data" / "BG_031_260325.pkl"
    if PKL_PATH.exists():
        try:
            sess = load_session(str(PKL_PATH))
            # prefer explicit good_cluster_ids list
            if getattr(sess, "good_cluster_ids", None):
                good_ids.update([int(x) for x in sess.good_cluster_ids])
            else:
                # fall back to per-cluster quality labels in the session.clusters dataclass
                for c in getattr(sess, "clusters", []):
                    q = getattr(c, "quality", None)
                    if q is not None and str(q).lower() == "good":
                        good_ids.add(int(getattr(c, "cluster_id", -1)))
        except Exception:
            # If pickle load fails, we silently fall through to CSV-based check below
            good_ids = set()
    else:
        # no pickle present; leave good_ids empty to allow CSV fallback
        good_ids = set()
except Exception:
    good_ids = set()

# If no good ids detected in pickle, fall back to CSV 'quality' fields if present
if len(good_ids) == 0:

    def quality_field(u):
        for k in ("quality", "cluster_quality", "quality_label", "unit_quality"):
            if k in u and u.get(k) not in (None, ""):
                return str(u.get(k))
        return None

    for u in units:
        qf = quality_field(u)
        if qf is not None and qf.lower() == "good":
            try:
                good_ids.add(
                    int(
                        u.get("unit_id")
                        if u.get("unit_id") not in (None, "")
                        else u.get("cluster_id")
                    )
                )
            except Exception:
                pass

if len(good_ids) > 0:
    # filter units to only those whose cluster/unit id is in the good set
    units = [
        u
        for u in units
        if int(u.get("unit_id") or u.get("cluster_id") or -1) in good_ids
    ]
    print(
        f"Filtering units to {len(units)} good clusters (found {len(good_ids)} good ids, original {orig_count})"
    )
else:
    print(
        f"No explicit good cluster labels found in pickle or CSV (checked {orig_count} units). Proceeding with all units."
    )

# Load templates if available
templates = None
if TEMPLATES_NPY.exists():
    print("Loading templates npy", TEMPLATES_NPY)
    templates = np.load(str(TEMPLATES_NPY), allow_pickle=False)
    print("templates shape", templates.shape, templates.dtype)
else:
    print("No templates npy found at", TEMPLATES_NPY)

# Quick histograms
pw = np.array([u["peak_width_us"] for u in units], dtype=float)
pv = np.array([u["peak_valley_us"] for u in units], dtype=float)
fr = np.array([u["mean_fr"] for u in units], dtype=float)
amp = np.array([u["template_amp"] for u in units], dtype=float)
snr = np.array([u["snr"] for u in units], dtype=float)
classes = [u["class"] for u in units]

# Save a CSV summary
summary_csv = OUTDIR / "units_metrics_summary.csv"
with open(summary_csv, "w", newline="") as f:
    fieldnames = [
        "unit_id",
        "class",
        "mean_fr",
        "peak_width_us",
        "peak_valley_us",
        "template_amp",
        "snr",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for u in units:
        writer.writerow({k: u.get(k) for k in fieldnames})
print("Wrote summary CSV to", summary_csv)

# Histograms
plt.figure(figsize=(6, 4))
plt.hist(pw[~np.isnan(pw)], bins=50)
plt.xlabel("peak width (us)")
plt.ylabel("count")
plt.title("Peak width histogram")
plt.tight_layout()
plt.savefig(OUTDIR / "hist_peak_width_us.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(pv[~np.isnan(pv)], bins=50)
plt.xlabel("peak-valley (us)")
plt.ylabel("count")
plt.title("Peak-Valley histogram")
plt.tight_layout()
plt.savefig(OUTDIR / "hist_peak_valley_us.png")
plt.close()

plt.figure(figsize=(6, 4))
# FR: clip extreme NaNs
plt.hist(fr[~np.isnan(fr)], bins=50)
plt.xlabel("mean firing rate (Hz)")
plt.ylabel("count")
plt.title("Mean FR histogram")
plt.tight_layout()
plt.savefig(OUTDIR / "hist_mean_fr.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(amp[~np.isnan(amp)], bins=50)
plt.xlabel("template amplitude (arb)")
plt.ylabel("count")
plt.title("Template amplitude histogram")
plt.tight_layout()
plt.savefig(OUTDIR / "hist_template_amp.png")
plt.close()

# Scatter colored by class
unique_classes = sorted(set(classes))
color_map = {c: plt.cm.tab10(i % 10) for i, c in enumerate(unique_classes)}
plt.figure(figsize=(6, 6))
for c in unique_classes:
    mask = [i for i, cc in enumerate(classes) if cc == c]
    if len(mask) == 0:
        continue
    plt.scatter(
        pw[mask], pv[mask], label=f"{c} (n={len(mask)})", alpha=0.6, c=[color_map[c]]
    )
plt.xlabel("peak width (us)")
plt.ylabel("peak-valley (us)")
plt.legend()
plt.title("PW vs PV by class")
plt.tight_layout()
plt.savefig(OUTDIR / "scatter_pw_vs_pv_by_class.png")
plt.close()

# Exemplars: pick units at percentiles of PW
percentiles = [5, 25, 50, 75, 95]
valid_pw = pw[~np.isnan(pw)]
if valid_pw.size == 0:
    pw_vals = []
else:
    pw_vals = np.nanpercentile(valid_pw, percentiles)
    # ensure iterable (when only one percentile and numpy returns scalar)
    if np.isscalar(pw_vals):
        pw_vals = [float(pw_vals)]
exemplar_units = []
for val in pw_vals:
    # pick nearest unit
    # skip NaN val
    if np.isnan(val):
        continue
    diffs = np.abs(pw - float(val))
    # if all diffs are nan, skip
    if np.all(np.isnan(diffs)):
        continue
    idx = int(np.nanargmin(diffs))
    exemplar_units.append(units[idx]["unit_id"])

# Plot exemplar waveforms if templates are available
if templates is not None:
    # templates array shape guessed as (n_templates, n_samples, n_channels) or similar
    # We'll compute mean across channels for plotting raw waveform shape
    for uid in exemplar_units:
        # mapping: assume template index == unit_id or unit_id is in a column 'template_idx'
        tidx = None
        # try direct find
        if 0 <= uid < templates.shape[0]:
            tidx = uid
        else:
            # try matching on units list with a field 'template_index'
            for u in units:
                if (
                    int(u["unit_id"]) == uid
                    and "template_index" in u
                    and u["template_index"] != ""
                ):
                    tidx = int(u["template_index"])
                    break
        if tidx is None or tidx < 0 or tidx >= templates.shape[0]:
            print("cannot find template for unit", uid)
            continue
        twf = templates[tidx]
        # if templates are 2D (samples x channels) or 3D (n_templates x samples x channels)
        if twf.ndim == 3:
            # twf shape (samples, channels) or (channels, samples)? We'll try to handle common shapes
            # assume twf is (n_samples, n_channels)
            if twf.shape[0] < twf.shape[1]:
                # assume samples x channels
                samples = twf.shape[0]
                mean_wf = twf.mean(axis=1)
            else:
                mean_wf = twf.mean(axis=1)
        elif twf.ndim == 2:
            mean_wf = twf.mean(axis=1)
        else:
            mean_wf = twf
        # If sampling rate known, convert to time in ms; default assume 30000 Hz
        sr = 30000.0
        t = np.arange(len(mean_wf)) / sr * 1000.0  # ms
        plt.figure(figsize=(5, 3))
        plt.plot(t, mean_wf)
        plt.xlabel("time (ms)")
        plt.ylabel("amplitude (arb)")
        plt.title(f"Unit {uid} exemplar waveform")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"unit_{uid}_exemplar_waveform.png")
        plt.close()

print("Diagnostics saved to", OUTDIR)
print("Done")
