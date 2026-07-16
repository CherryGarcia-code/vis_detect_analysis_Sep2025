"""Null controls for the continuum width->coupling result (project hard rule: a shuffle
must make the effect go flat). This is the companion the continuum figure set was missing.

Two nulls, both non-circular:
  1. WIDTH-LABEL PERMUTATION (primary): shuffle interp_fwhm across cells and recompute
     Spearman(width, outcome). The observed correlation must sit far outside the permuted
     null. Directly tests the headline width->coupling result; the permutation destroys the
     width<->coupling pairing while preserving both marginals, so any residual correlation
     from selection / marginal shape shows up in the null.
  2. WITHIN-MOUSE PERMUTATION: same, permuting only WITHIN each mouse, so the null cannot
     borrow the between-mouse (region) difference — a stricter control for pseudoreplication.

Cache-only (kernel_width_continuous.csv); no session reloads. Writes a small figure +
stats to FIGURES/tf_glm_bg046/null_controls/.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO                                  # noqa: E402

CACHE = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/null_controls"
WIDTH = "interp_fwhm"
OUTCOMES = [("change_on", "Change_ON (sensory)"),
            ("hit_ramp", "Hit pre-lick (≈change resp)"),
            ("fa_ramp", "FA motor ramp (independent)")]
N_PERM = 2000
SEED = 42


def _perm_null(x, y, groups=None, n=N_PERM, seed=SEED):
    """Permutation null of Spearman(x, y). If `groups` given, permute y only WITHIN each
    group (mouse), so between-group structure cannot leak into the null."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    obs = spearmanr(x, y).statistic
    null = np.empty(n)
    idx_by = ([np.where(groups == g)[0] for g in np.unique(groups)]
              if groups is not None else None)
    for i in range(n):
        if idx_by is None:
            yp = rng.permutation(y)
        else:
            yp = y.copy()
            for ix in idx_by:
                yp[ix] = rng.permutation(y[ix])
        null[i] = spearmanr(x, yp).statistic
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)
    z = (obs - null.mean()) / (null.std() + 1e-12)
    return obs, null, p, z


def main():
    d = pd.read_csv(CACHE, dtype={"session": str})
    d["mouse"] = d.subject.values
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    lines = ["WIDTH -> COUPLING NULL CONTROLS (must be flat under permutation)", "=" * 60]
    for ax, (col, lab) in zip(axes, OUTCOMES):
        sub = d[[WIDTH, col, "mouse"]].replace([np.inf, -np.inf], np.nan).dropna()
        obs, null, p, z = _perm_null(sub[WIDTH], sub[col])
        _, _, p_wm, z_wm = _perm_null(sub[WIDTH], sub[col], groups=sub.mouse.values)
        ax.hist(null, bins=40, color="0.7", edgecolor="none", density=True)
        ax.axvline(obs, color="#d7301f", lw=2.4,
                   label=f"observed ρ={obs:+.3f}")
        ax.axvline(0, color="0.4", lw=0.8, ls=":")
        ax.set_title(f"{lab}\nglobal p_perm={p:.1e} (z={z:+.1f}) | within-mouse z={z_wm:+.1f}",
                     fontsize=9.5)
        ax.set_xlabel("Spearman ρ (width, outcome)")
        ax.set_ylabel("null density")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lines.append(f"[{col}] observed ρ={obs:+.3f} | null mean={null.mean():+.4f} "
                     f"sd={null.std():.4f} | global p_perm={p:.2e} z={z:+.2f} | "
                     f"within-mouse p={p_wm:.2e} z={z_wm:+.2f} | n={len(sub)}")

    fig.suptitle("Null control: shuffling the width label across cells makes width→coupling go FLAT\n"
                 "(observed red line sits far outside the permuted null → the correlation is not a "
                 "selection/marginal artifact)", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"null_controls_continuum.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    (OUT / "null_controls_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    for ln in lines:                       # console-safe (Windows cp1252 can't encode rho)
        print(ln.encode("ascii", "replace").decode())
    print(f"\nwrote {OUT}/null_controls_continuum.png")


if __name__ == "__main__":
    main()
