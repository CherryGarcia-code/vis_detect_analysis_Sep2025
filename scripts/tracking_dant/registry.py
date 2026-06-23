"""DANT output -> comparable long registry, plus UnitMatch-comparison metrics. Pure functions."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def idxcluster_to_registry(idx_cluster, lookup):
    """Per-unit cluster ids (-1 = untracked) + lookup -> long [session, ks_unit_id, dant_uid]."""
    idx = np.asarray(idx_cluster).astype(int)
    if len(idx) != len(lookup):
        raise ValueError(f"idx_cluster len {len(idx)} != lookup rows {len(lookup)}")
    df = lookup.copy().reset_index(drop=True)
    df["dant_uid"] = idx
    out = df[["session", "ks_unit_id", "dant_uid"]].reset_index(drop=True)
    if out.duplicated(subset=["session", "ks_unit_id"]).any():
        raise ValueError("duplicate (session, ks_unit_id) in registry")
    return out


def tracked_lengths(registry, uid_col="dant_uid"):
    """uid -> number of distinct sessions, for tracked uids (uid > 0)."""
    tracked = registry[registry[uid_col] > 0]
    return tracked.groupby(uid_col)["session"].nunique()


def survival_function(lengths, n_sessions):
    """(k, fraction of tracked neurons appearing in >= k sessions) for k=1..n_sessions."""
    lengths = np.asarray(lengths, dtype=float)
    ks = np.arange(1, n_sessions + 1)
    n = len(lengths)
    if n == 0:
        return ks, np.zeros(n_sessions)
    frac = np.array([(lengths >= k).sum() / n for k in ks])
    return ks, frac


def _relabel_singletons(labels):
    """Replace untracked (<=0) entries with unique negative singleton labels."""
    out = np.asarray(labels).astype(np.int64).copy()
    nxt = -1
    for i in range(len(out)):
        if out[i] <= 0:
            out[i] = nxt
            nxt -= 1
    return out


def _pair_count(sizes):
    sizes = np.asarray(sizes, dtype=np.int64)
    return int((sizes * (sizes - 1) // 2).sum())


def comembership_agreement(reg_a, reg_b, uid_a="dant_uid", uid_b="um_uid"):
    """Agreement between two registries on shared (session, ks_unit_id) units.

    Returns ARI plus pairwise precision/recall treating reg_b (UnitMatch) as reference:
    precision = (pairs same in BOTH) / (pairs same in A); recall = same / (pairs same in B).
    """
    reg_a = reg_a.drop_duplicates(["session", "ks_unit_id"])
    reg_b = reg_b.drop_duplicates(["session", "ks_unit_id"])
    a = reg_a.set_index(["session", "ks_unit_id"])[uid_a]
    b = reg_b.set_index(["session", "ks_unit_id"])[uid_b]
    shared = a.index.intersection(b.index)
    a = a.loc[shared]
    b = b.loc[shared]
    la = _relabel_singletons(a.to_numpy())
    lb = _relabel_singletons(b.to_numpy())
    ari = float(adjusted_rand_score(la, lb)) if len(shared) > 1 else float("nan")

    cont = pd.crosstab(la, lb).to_numpy()
    tp = _pair_count(cont.ravel())
    pairs_a = _pair_count(cont.sum(axis=1))
    pairs_b = _pair_count(cont.sum(axis=0))
    precision = tp / pairs_a if pairs_a else float("nan")
    recall = tp / pairs_b if pairs_b else float("nan")
    return {
        "n_shared": int(len(shared)),
        "ari": ari,
        "pairwise_precision": float(precision),
        "pairwise_recall": float(recall),
    }


def melt_cellregistry(wide, uid_col="UID"):
    """UnitMatch wide CellRegistry (UID + per-session-date columns of ks ids) -> long.

    Cells may be empty/NaN/0 (absent) or ';'-joined (merged) ks ids. Output columns:
    [session, ks_unit_id, um_uid].
    """
    session_cols = [c for c in wide.columns if c != uid_col]
    rows = []
    for _, r in wide.iterrows():
        uid = int(r[uid_col])
        for sess in session_cols:
            cell = r[sess]
            if pd.isna(cell):
                continue
            text = str(cell).strip()
            if text in ("", "0", "0.0", "nan"):
                continue
            for part in text.split(";"):
                part = part.strip()
                if not part or part in ("0", "0.0"):
                    continue
                rows.append({"session": str(sess), "ks_unit_id": int(float(part)), "um_uid": uid})
    return pd.DataFrame(rows, columns=["session", "ks_unit_id", "um_uid"])
