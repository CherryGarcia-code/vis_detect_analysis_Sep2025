"""Quick diagnostic: quantify stitch fragmentation."""
import pandas as pd, numpy as np

can = pd.read_csv('X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/final_output/global_registry_canonical.csv')
spans = can.groupby('global_uid').session.nunique()

sessions_per_shank = can.groupby(['shank_id','session']).global_uid.nunique()
avg_per_shank_session = sessions_per_shank.groupby('shank_id').mean()
print('Avg units per session by shank:')
print(avg_per_shank_session.to_string())
print()

uids_per_shank = can.groupby('shank_id').global_uid.nunique()
print('Total UIDs per shank:')
print(uids_per_shank.to_string())
print()

print('Fragmentation ratio (UIDs / avg_units_per_session):')
for sh in sorted(can.shank_id.unique()):
    frag = uids_per_shank[sh] / avg_per_shank_session[sh]
    print(f'  Shank {sh}: {frag:.1f}x')
print()

n_total = spans.shape[0]
n_gt5 = (spans > 5).sum()
n_gt10 = (spans > 10).sum()
print(f'Units spanning >5 sessions (cross-window match): {n_gt5} ({n_gt5/n_total*100:.1f}%)')
print(f'Units spanning >10 sessions: {n_gt10} ({n_gt10/n_total*100:.1f}%)')
print(f'Units spanning EXACTLY 5: {(spans==5).sum()} ({(spans==5).mean()*100:.1f}%)')
