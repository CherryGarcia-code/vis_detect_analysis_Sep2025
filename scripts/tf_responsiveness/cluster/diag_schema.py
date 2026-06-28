"""Find the baseline-TF fluctuation signal in the npx_converted export."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_phase0/src")
from visdetect.analysis.tf_glm_data import load_khilkevich_session

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
SESS = "X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/1116764/ML_1116764_S02_M2_V1"
ks = load_khilkevich_session(SESS)

print("================ trials.parquet ================")
tr = ks.trials
print("cols:", list(tr.columns))
print("\ndtypes:")
print(tr.dtypes)
# Find array/object-valued columns (candidate baseline-TF vectors).
print("\n--- object/array columns (candidate per-trial TF vectors) ---")
for c in tr.columns:
    v = tr[c].iloc[0]
    if isinstance(v, (list, np.ndarray)) or tr[c].dtype == object:
        try:
            arr = np.asarray(v, float)
            print(f"  {c}: type={type(v).__name__} len={getattr(arr,'size','?')} "
                  f"first8={np.round(arr.ravel()[:8],3)}")
        except Exception as e:
            print(f"  {c}: type={type(v).__name__} (non-numeric) sample={str(v)[:60]}")

print("\n================ stim.csv: trial_idx == 0 ================")
stim = ks.stim
s0 = stim[stim["trial_idx"] == 0] if "trial_idx" in stim.columns else stim.iloc[:0]
print(f"n frames for trial 0: {len(s0)}")
if "tag" in stim.columns:
    print("\ntag value_counts (whole stim):")
    print(stim["tag"].value_counts())
    print("\nTF stats by tag (whole stim):")
    print(stim.groupby("tag")["TF"].agg(["count", "mean", "std", "min", "max"]))
print("\ntrial 0 frames (first 20 rows):")
cols = [c for c in ["trial_idx", "frame_idx", "TF", "tag", "vbl", "frame_time"] if c in s0.columns]
print(s0[cols].head(20).to_string(index=False))
print("\ntrial 0 TF: nonzero frac =", float(np.mean(s0["TF"].to_numpy(float) > 0)),
      "| mean =", float(np.nanmean(s0["TF"].to_numpy(float))))
