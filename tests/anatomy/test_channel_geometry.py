# tests/anatomy/test_channel_geometry.py
import numpy as np
from visdetect.anatomy.channel_geometry import assign_shanks, chanmap_signature

def _np2_positions(y0=1515.0):
    # 4 shanks at x-base {0,250,500,750}, 2 cols per shank (+0,+32), 48 rows @15um
    xs_base = [0, 250, 500, 750]
    rows = np.arange(48) * 15.0 + y0
    pos = []
    for xb in xs_base:
        for col in (27.0, 59.0):
            for y in rows:
                pos.append([xb + col, y])
    return np.array(pos)

def test_assign_shanks_four_groups():
    pos = _np2_positions()
    sh = assign_shanks(pos)
    assert set(np.unique(sh)) == {0, 1, 2, 3}
    # lowest-x channels are shank 0
    assert sh[np.argmin(pos[:, 0])] == 0
    assert sh[np.argmax(pos[:, 0])] == 3
    # ~equal counts per shank
    counts = np.bincount(sh)
    assert counts.min() == counts.max()

def test_signature_stable_under_reorder():
    pos = _np2_positions()
    perm = np.random.RandomState(0).permutation(len(pos))
    assert chanmap_signature(pos) == chanmap_signature(pos[perm])

def test_signature_changes_with_y_offset():
    a = _np2_positions(1515.0)
    b = _np2_positions(765.0)
    assert chanmap_signature(a) != chanmap_signature(b)

def test_signature_invariant_to_subum_jitter():
    base = _np2_positions()
    jittered = base + 0.3            # < 0.5 um -> same integer-um -> same signature
    assert chanmap_signature(base) == chanmap_signature(jittered)

def test_signature_changes_at_1um_shift():
    base = _np2_positions()
    shifted = base.copy()
    shifted[:, 1] = shifted[:, 1] + 1.0   # +1 um in y -> different signature
    assert chanmap_signature(base) != chanmap_signature(shifted)
