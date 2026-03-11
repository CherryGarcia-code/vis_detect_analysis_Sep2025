"""Quick yield summary: old pipeline vs new concat-sort with stability filter."""
import sys, pickle
sys.path.insert(0, 'src')
from pathlib import Path

OLD_DIR = Path('data/pkls/BG_046')
NEW_DIR = Path('data/pkls/BG_046_concat_sort')


def main():
    new_pkls = sorted(NEW_DIR.glob('*.pkl'))
    rows = []
    for np_ in new_pkls:
        sess = np_.stem
        op = OLD_DIR / np_.name
        with open(np_, 'rb') as f:
            ns = pickle.load(f)
        n_stable = len(getattr(ns, 'good_and_stable_ids', None) or [])
        n_good = len(getattr(ns, 'good_cluster_ids', None) or [])
        if op.exists():
            with open(op, 'rb') as f:
                os_ = pickle.load(f)
            o_good = len(getattr(os_, 'good_cluster_ids', None) or [])
        else:
            o_good = 0
        rows.append((sess, o_good, n_good, n_stable))
        print(f'  {sess}  old={o_good:3d}  new_good={n_good:3d}  new_stable={n_stable:3d}',
              flush=True)

    old_vals = [r[1] for r in rows]
    good_vals = [r[2] for r in rows]
    stable_vals = [r[3] for r in rows]
    n = len(rows)
    print()
    print(f'--- Summary ({n} sessions) ---')
    print(f'Old pipeline (good, >=1Hz):      mean={sum(old_vals)/n:.1f}  total={sum(old_vals)}')
    print(f'New concat-sort (good only):     mean={sum(good_vals)/n:.1f}  total={sum(good_vals)}')
    print(f'New concat-sort (good+stable):   mean={sum(stable_vals)/n:.1f}  total={sum(stable_vals)}')


if __name__ == '__main__':
    main()
