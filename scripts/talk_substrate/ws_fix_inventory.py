"""FIX A + FIX D: cross-figure reconciliation table.

ONE common width cutoff (live pooled-GMM over all animals) applied everywhere. Emits, per
animal: narrow/broad n at the COMMON cutoff (+ each animal's OWN cutoff as a sensitivity
column), baseline-calibrated modulated fractions (change & lick), and the unit-count
convention reconciled: UNIQUE neurons per session (median [range] of sorted good clusters
that session) vs total UNIT-SESSIONS (the pooled n used in PSTH bands) vs n sessions.

Unique neurons = per-session good-cluster count (we do NOT cross-session match — tracking is
only partly trustworthy). Bands in PSTH figures are bootstrap CIs over UNIT-SESSIONS.

Usage: py scripts/talk_substrate/ws_fix_inventory.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import _events_plot as E  # noqa: E402

REGION = {"BG_046": "Striatum DMS", "BG_039": "Striatum DMS",
          "BG_031": "Striatum VMS", "BG_038": "Cortex M1/S1 (reference)"}
RESP = {"Change_ON": (0.0, 1.0), "Hit": (-0.5, 0.2)}
BASE = {"Change_ON": (-1.0, 0.0), "Hit": (-1.75, -1.05)}


def maxabs(cache, event, win):
    m = E.mat(cache, event, "all", "full")
    seg = m[:, E.win_mask(E.bc(cache, event), win)]
    pk = np.full(m.shape[0], np.nan)
    ok = np.isfinite(seg).all(1)
    pk[ok] = np.nanmax(np.abs(seg[ok]), axis=1)
    return pk


def modulated_frac(cache, event, mask):
    resp = maxabs(cache, event, RESP[event])
    base = maxabs(cache, event, BASE[event])
    thr = np.nanpercentile(base, 95)
    sel = mask & np.isfinite(resp)
    n = int(sel.sum())
    return (float(np.mean(resp[sel] > thr)) if n else np.nan), n


def main():
    thr, info = C.common_t2p_cutoff()
    print(f"COMMON width cutoff (pooled GMM, all animals, n={info['n']}): {thr:.3f} ms "
          f"[narrow mean {info['narrow_mean_ms']:.3f}, broad {info['broad_mean_ms']:.3f}]\n")
    rows = []
    for subj in C.ALL_SUBJECTS:
        cache = E.load_event_cache(subj)
        narrow, broad, t2p = C.common_celltype(cache, [subj], thr)
        own_thr, _ = C.common_t2p_cutoff([subj])
        own_narrow = np.isfinite(t2p) & (t2p < own_thr)
        # unique neurons per session
        sess = cache["unit_meta_session"].astype(str)
        per_sess = pd.Series(sess).value_counts()
        # modulated fractions (common cutoff)
        ch_n, _ = modulated_frac(cache, "Change_ON", narrow)
        ch_b, _ = modulated_frac(cache, "Change_ON", broad)
        hit_n, _ = modulated_frac(cache, "Hit", narrow)
        hit_b, _ = modulated_frac(cache, "Hit", broad)
        rows.append(dict(
            animal=subj, region=REGION[subj], n_sessions=int(per_sess.size),
            unit_sessions=int(len(sess)),
            unique_neurons_per_sess=f"{int(per_sess.median())} [{int(per_sess.min())}-{int(per_sess.max())}]",
            common_cut_ms=round(thr, 3), own_cut_ms=round(own_thr, 3),
            n_narrow_common=int(narrow.sum()), n_broad_common=int(broad.sum()),
            n_narrow_own=int(own_narrow.sum()),
            frac_mod_change_narrow=None if ch_n is None else round(ch_n, 3),
            frac_mod_change_broad=None if ch_b is None else round(ch_b, 3),
            frac_mod_lick_narrow=None if hit_n is None else round(hit_n, 3),
            frac_mod_lick_broad=None if hit_b is None else round(hit_b, 3)))
    df = pd.DataFrame(rows)
    out = C.FIG_DIR.parent / "inventory_reconciliation.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(df.to_string(index=False))
    print(f"\n[fix] wrote {out}")


if __name__ == "__main__":
    main()
