"""Sensitivity of the width->coupling headline to the DISENG_MAX session gate.

THE CONCERN. `good_dates()` keeps a session only if <50% of its trials are labelled
Disengaged. 50% is a round number somebody picked — no SNR argument, no convergence
criterion — yet 11 scripts call it, so it silently decides which sessions -> which cells ->
every headline number. That is an unexamined researcher degree of freedom sitting under all
the results. This script closes it the only way that counts: sweep the threshold and show
whether the answer moves.

THE DESIGN. To sweep BOTH directions we need the cells the current gate EXCLUDES, which have
no cached width (their GLM was never refit) and no coupling metrics. Removing the gate entirely
adds only ~31 responsive cells (520 -> 551, +6%), so we refit exactly those and compute their
metrics with the IDENTICAL code, then sweep by subsetting a complete 551-cell table. Every cell
carries its session's %Disengaged, so each threshold is a pure subset — no refit per threshold.

READ IT AS: if rho is flat from a strict gate (20%) through no gate at all (101%), the 50% cut
is arbitrary but demonstrably irrelevant, and no headline depends on it.

Usage: py diseng_sensitivity.py [--workers 10] [--force]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = str(Path(__file__).resolve().parent)
_CB = str(Path(_HERE).parents[0] / "cluster_bg")
for _p in (_HERE, _CB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from representative_cells import REPO, _registry, good_dates              # noqa: E402
from latency_outcome_coupling import session_metrics                      # noqa: E402
from tf_glm_bg_task import _cfg                                           # noqa: E402
from visdetect.core.session import load_session                          # noqa: E402
from visdetect.analysis.tf_glm import (                                  # noqa: E402
    assemble_design, fit_poisson_cv, make_trial_folds, _tf_kernel, _lag_offsets,
    count_vector,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors       # noqa: E402
from visdetect.analysis.kernel_width import interpolated_fwhm             # noqa: E402

MICE = [("BG_046", "DMS"), ("BG_039", "DMS"), ("BG_031", "VMS")]
WIDTH_CSV = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_continuous.csv"
FULL_CSV = Path(REPO) / "data/cache/tf_glm_bg046/kernel_width_no_diseng_gate.csv"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/diseng_sensitivity"
THRESHOLDS = [20, 30, 40, 50, 60, 70, 80, 101]   # 101 == no gate at all
OUTCOMES = [("change_on", "Change_ON (sensory)", "#238b45"),
            ("fa_ramp", "FA motor ramp (independent)", "#ce1256")]
CURRENT = 50


def _session_date(subj, sess):
    """'BG_046_01072025' -> '01072025'.

    ⚠️ The width CSV keys sessions by the FULL name (`BG_046_01072025`) while the manifest and
    the state-tag files key by the DATE alone (`01072025.csv`). Looking up a state file with the
    full name silently misses EVERY session -> every cell gets NaN -> a threshold sweep that
    quietly keeps everything and reports a perfectly 'flat' result. That is exactly what this
    script did on its first run. (Same family as the project's canonical session-id footgun.)
    Note BG_031 uses the 6-digit DDMMYY form (`050325`), which this handles by construction —
    it strips the prefix rather than parsing the date."""
    s = str(sess)
    return s[len(subj) + 1:] if s.startswith(f"{subj}_") else s


def _pct_diseng(subj, sess):
    """% Disengaged trials for a session; NaN if it has no state-tag file (those sessions are
    kept UNCONDITIONALLY by good_dates — a quiet asymmetry worth surfacing)."""
    f = Path(REPO) / f"data/cache/state_tags/{subj}/{_session_date(subj, sess)}.csv"
    if not f.exists():
        return np.nan
    return 100.0 * (pd.read_csv(f).state_label == "Disengaged").mean()


def _refit_cell(task):
    """interp_fwhm for one cell, via the same refit the cached width came from."""
    subj, sess, uids = task
    try:
        s = load_session(str(Path(REPO) / "data/pkls" / subj / f"{sess}.pkl"))
        cfg = _cfg("log2")
        trials, units = session_trial_regressors(s, cfg)
        d = assemble_design(trials, cfg)
        folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
        lags = _lag_offsets(cfg.kern["tf"], cfg.bin_s) * cfg.bin_s
        out = []
        for uid in uids:
            if int(uid) not in units:
                continue
            y = count_vector(trials, units[int(uid)], d)
            K = _tf_kernel(fit_poisson_cv(d.X, y, cfg, folds), d, cfg)
            if K is None or not np.any(np.isfinite(K)):
                continue
            out.append({"subject": subj, "session": sess, "unit": int(uid),
                        "interp_fwhm": interpolated_fwhm(K, lags)})
        del s; gc.collect()
        return out
    except Exception as e:
        import traceback
        return [{"err": f"{subj}/{sess}: {type(e).__name__}: {e}\n{traceback.format_exc()}"}]


def build_full_table(workers=10, force=False):
    """The 551-cell table = the cached 520 + the ~31 the 50% gate currently excludes."""
    if FULL_CSV.exists() and not force:
        return pd.read_csv(FULL_CSV, dtype={"session": str})
    from concurrent.futures import ProcessPoolExecutor, as_completed
    have = pd.read_csv(WIDTH_CSV, dtype={"session": str})
    have_keys = set(zip(have.subject, have.session, have.unit.astype(int)))

    # every responsive cell in ANY QC-pass session (gate fully open)
    missing = []
    for subj, region in MICE:
        r = _registry(subj)
        r = r[r.resp & r.session_date.isin(good_dates(subj, max_diseng=101))]
        for _, row in r.iterrows():
            if (subj, str(row.session), int(row.unit)) not in have_keys:
                missing.append((subj, region, str(row.session), int(row.unit)))
    print(f"cells to add (excluded by the {CURRENT}% gate): {len(missing)}", flush=True)

    add_rows = []
    if missing:
        # 1) refit their kernels -> interp_fwhm
        by_sess = {}
        for subj, region, sess, uid in missing:
            by_sess.setdefault((subj, sess), []).append(uid)
        tasks = [(subj, sess, uids) for (subj, sess), uids in by_sess.items()]
        widths = []
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
            for fut in as_completed([ex.submit(_refit_cell, t) for t in tasks]):
                for r in fut.result():
                    if "err" in r:
                        print("  " + r["err"].splitlines()[0], flush=True)
                    else:
                        widths.append(r)
        wdf = pd.DataFrame(widths)
        print(f"  refit {len(wdf)} kernels", flush=True)

        # 2) their coupling metrics — same session_metrics, same windows, same clamp
        mrows = []
        for (subj, sess), uids in by_sess.items():
            region = dict((m[0], m[1]) for m in MICE)[subj]
            reg = _registry(subj)
            g = reg[(reg.session == sess) & (reg.unit.astype(int).isin(uids))]
            if len(g):
                mrows += session_metrics(subj, region, sess, g)
        mdf = pd.DataFrame(mrows)
        if len(wdf) and len(mdf):
            add_rows = wdf.merge(mdf[["subject", "session", "unit", "base_hz", "change_on",
                                      "hit_ramp", "fa_ramp"]],
                                 on=["subject", "session", "unit"], how="inner")
            print(f"  + coupling metrics -> {len(add_rows)} complete new cells", flush=True)

    keep = ["subject", "session", "unit", "interp_fwhm", "base_hz", "change_on",
            "hit_ramp", "fa_ramp"]
    full = pd.concat([have[keep], pd.DataFrame(add_rows)[keep]], ignore_index=True) \
        if len(add_rows) else have[keep].copy()
    # attach each cell's session %Disengaged (the sweep variable)
    dis = {}
    for subj, _ in MICE:
        for sess in full.loc[full.subject == subj, "session"].unique():
            dis[(subj, str(sess))] = _pct_diseng(subj, str(sess))
    full["pct_diseng"] = [dis.get((r.subject, str(r.session)), np.nan) for r in full.itertuples()]
    # COVERAGE GATE. If the state-file lookup misses, every cell gets NaN, every threshold keeps
    # everything, and the sweep reports a perfectly flat rho — a vacuous PASS that looks like
    # evidence. (This fired on the first run: the CSV keys sessions by full name, the state files
    # by date.) Only ~2 sessions genuinely lack a state file, so demand near-total coverage.
    cov = float(full.pct_diseng.notna().mean())
    assert cov > 0.90, (
        f"diseng_sensitivity: only {cov:.1%} of cells resolved a %Disengaged value — the "
        f"session-key join is broken, and the sweep would silently keep every cell at every "
        f"threshold and report a fake 'flat' result. Check _session_date().")
    FULL_CSV.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(FULL_CSV, index=False)
    print(f"wrote {FULL_CSV} (n={len(full)})", flush=True)
    return full


def main(workers=10, force=False):
    full = build_full_table(workers=workers, force=force)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass

    lines = ["DISENG_MAX SENSITIVITY — does the width->coupling headline depend on the gate?",
             "=" * 74,
             "A session is kept if %Disengaged < threshold. Sessions with NO state file are kept",
             "unconditionally (good_dates' `else` branch) at EVERY threshold — noted, not hidden.",
             f"n cells with no state label: {int(full.pct_diseng.isna().sum())}", ""]
    res = {c: {"thr": [], "rho": [], "p": [], "n": []} for c, _, _ in OUTCOMES}
    for thr in THRESHOLDS:
        # NaN pct_diseng = no state file = kept unconditionally, matching good_dates
        sel = full.pct_diseng.isna() | (full.pct_diseng < thr)
        sub = full[sel]
        row = [f"  thr={thr:>3}%  n_cells={len(sub):>4}"]
        for c, _, _ in OUTCOMES:
            s = sub[["interp_fwhm", c]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p = spearmanr(s.interp_fwhm, s[c])
            res[c]["thr"].append(thr); res[c]["rho"].append(rho)
            res[c]["p"].append(p); res[c]["n"].append(len(s))
            row.append(f"{c}: rho={rho:+.3f} (p={p:.1e})")
        lines.append("  |  ".join(row))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for c, lab, col in OUTCOMES:
        ax.plot(res[c]["thr"], res[c]["rho"], "o-", color=col, lw=2.2, ms=6, label=lab)
    ax.axvline(CURRENT, color="k", ls="--", lw=1.4, alpha=0.7)
    ax.annotate("current gate (50%)", xy=(CURRENT, 0.04), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", rotation=90, va="bottom",
                ha="left", fontsize=8.5, color="0.3")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("DISENG_MAX threshold (% Disengaged trials allowed;  101 = no gate at all)")
    ax.set_ylabel("Spearman ρ (kernel width, coupling)")
    ax.set_ylim(0, max(max(res[c]["rho"]) for c, _, _ in OUTCOMES) * 1.35)
    for c, _, _ in OUTCOMES:
        for x, y, n in zip(res[c]["thr"], res[c]["rho"], res[c]["n"]):
            if x in (THRESHOLDS[0], CURRENT, THRESHOLDS[-1]):
                ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7, color="0.35")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("The 50% Disengaged session gate is arbitrary — and demonstrably irrelevant\n"
                 "width→coupling is flat from a strict gate to no gate at all",
                 fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"diseng_sensitivity.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    lines += ["", "VERDICT:"]
    for c, lab, _ in OUTCOMES:
        r = np.array(res[c]["rho"])
        lines.append(f"  [{c}] rho range {r.min():+.3f}..{r.max():+.3f} "
                     f"(spread {r.max()-r.min():.3f}); all p<0.05: {all(p < 0.05 for p in res[c]['p'])}")
    (OUT / "diseng_sensitivity_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    for ln in lines:
        print(ln.encode("ascii", "replace").decode())
    print(f"\nwrote {OUT}/diseng_sensitivity.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(workers=a.workers, force=a.force)
