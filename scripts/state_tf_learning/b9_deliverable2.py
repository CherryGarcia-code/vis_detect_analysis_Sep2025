"""B9 deliverable 2 — TRIAL-MATCHED state-conditioned TF-encoding, early vs late (subject-generic).

Auto-picks responsive-rich, StimSens-covered sessions per stage-group
(early = Naive+Learning, late = Expert), then for a FAIR comparison subsamples
every session to a COMMON StimSens trial count (--n_match) and averages c1_r over
--k_draws random draws (c1_r attenuates with trial count, so early/late must be
trial-matched). Responsive vs a non-responsive control. Includes a faithfulness
spot-check (whole-session re-run reproduces the registry c1_r_log2). Sessions run
in parallel (BLAS pinned per worker).

Run:  PYTHONPATH=src py scripts/state_tf_learning/b9_deliverable2.py --subject BG_031 --n_match 140
"""
import os, sys, gc, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ProcessPoolExecutor

from visdetect.analysis import state_tf_learning as stl
from visdetect.analysis.config import canonical_session_id as csid
from visdetect.core.session import load_session

MIN_RESP = 5         # min responsive units in the session
N_PICK = 3           # sessions per stage-group
N_NONRESP = 8        # non-responsive control units per session


def build_candidates(subject):
    reg = stl.load_registry(stl.registry_path(subject))
    man = pd.read_csv(stl.manifest_path(subject)); man["sess_key"] = man["session_name"].map(csid)
    stage_map = dict(zip(man.sess_key, man.stage.astype(str)))
    meta = reg.drop_duplicates("sess_key").set_index("sess_key")[["session", "session_date"]]
    nresp = reg.groupby("sess_key")["resp_log2"].sum()
    rows = []
    for key, mrow in meta.iterrows():
        try:
            tags = stl.load_state_tags(subject, str(mrow["session_date"]))
        except FileNotFoundError:
            continue
        g = tags[tags["state_confidence"] >= 0.8]
        rows.append({"sess_key": key, "stem": mrow["session"], "orig": str(mrow["session_date"]),
                     "stage": stage_map.get(key, "?"), "n_resp": int(nresp.get(key, 0)),
                     "StimSens": int((g["state_label"] == "StimSens").sum())})
    return pd.DataFrame(rows), reg


def pick(cand, n_match):
    def top(mask):
        return cand[mask & (cand.StimSens >= n_match) & (cand.n_resp >= MIN_RESP)] \
            .sort_values("n_resp", ascending=False).head(N_PICK)
    return top(cand.stage.isin(["Naive", "Learning"])), top(cand.stage == "Expert")


def _run_one(args):
    """Trial-matched draws for one session: subsample StimSens to n_match, K times, mean c1_r/unit."""
    subject, key, stem, orig, group, resp, nonresp, n_match, k_draws, seed = args
    from visdetect.analysis import state_tf_learning as _stl
    from visdetect.core.session import load_session as _load
    import numpy as _np, pandas as _pd, gc as _gc
    pkl = _stl.PKL_ROOT / subject / f"{stem}.pkl"
    if not pkl.exists():
        return ("MISS", key, str(pkl))
    sess = _load(str(pkl)); tags = _stl.load_state_tags(subject, orig)
    idx_full = _stl.state_trial_indices(tags, "StimSens")
    if len(idx_full) < n_match:
        del sess; _gc.collect(); return ("SKIP", key, f"{len(idx_full)} StimSens < n_match {n_match}")
    cfg = _stl.b9_cfg(); rng = _np.random.default_rng(seed)
    draws = []
    try:
        for d in range(k_draws):
            sub = sorted(int(i) for i in rng.choice(idx_full, size=n_match, replace=False))
            df = _stl.state_conditioned_encoding(sess, sub, cfg, unit_ids=list(resp) + list(nonresp))
            if not df.empty:
                df["draw"] = d; draws.append(df)
    except Exception as e:
        del sess; _gc.collect(); return ("ERR", key, str(e))
    del sess; _gc.collect()
    if not draws:
        return ("EMPTY", key, "no units cleared min_spikes")
    alld = _pd.concat(draws, ignore_index=True)
    agg = alld.groupby("unit").agg(c1_r=("c1_r", "mean"), c2_p=("c2_p", "mean"),
                                   n_draws=("c1_r", "size")).reset_index()
    rset = set(int(u) for u in resp)
    agg["sess_key"] = key; agg["group"] = group; agg["n_match"] = n_match
    agg["resp_class"] = agg["unit"].map(lambda u: "responsive" if int(u) in rset else "nonresponsive")
    return ("OK", key, group, len(idx_full), agg)


def faithfulness(subject, cand, reg, cache):
    best = cand.sort_values("n_resp", ascending=False).iloc[0]
    sess = load_session(str(stl.PKL_ROOT / subject / f"{best['stem']}.pkl"))
    rs = reg[reg.sess_key == best["sess_key"]]
    resp = rs.loc[rs.resp_log2 == True, "unit"].astype(int).tolist()      # noqa: E712
    df = stl.state_conditioned_encoding(sess, list(range(len(sess.trials))), stl.b9_cfg(), unit_ids=resp)
    want = rs.set_index("unit")["c1_r_log2"]; got = df.set_index("unit")["c1_r"]
    common = [u for u in want.index if u in got.index]
    del sess; gc.collect()
    if not common:
        print("[FAITHFULNESS] no common units", flush=True); return None
    diff = np.abs(got.loc[common].to_numpy() - want.loc[common].to_numpy())
    med = float(np.nanmedian(diff))
    print(f"[FAITHFULNESS] {best['sess_key']} responsive n={len(common)} | median abs-diff c1_r={med:.4f} "
          f"max={float(np.nanmax(diff)):.4f}  (want < ~0.02)", flush=True)
    return med


def main(subject, n_workers, n_match, k_draws):
    cache = stl._REPO / "data" / "cache" / "state_tf_learning"; cache.mkdir(parents=True, exist_ok=True)
    figdir = stl._REPO / "FIGURES" / "state_tf_learning" / subject; figdir.mkdir(parents=True, exist_ok=True)
    tag = f"N{n_match}"
    cand, reg = build_candidates(subject)
    early, late = pick(cand, n_match)
    print(f"=== {subject}: picks (StimSens >= n_match={n_match}) ===", flush=True)
    for lab, d in [("early(Naive/Learning)", early), ("late(Expert)", late)]:
        print(f"  {lab}: " + ", ".join(f"{r.sess_key}(r{r.n_resp}/ss{r.StimSens})" for _, r in d.iterrows()), flush=True)
    if early.empty or late.empty:
        print("[ABORT] no usable early or late picks at this n_match", flush=True); return

    print("=== FAITHFULNESS SPOT-CHECK (whole-session) ===", flush=True)
    med = faithfulness(subject, cand, reg, cache)
    if med is not None and med > 0.1:
        print(f"[ABORT] faithfulness {med:.3f} > 0.1", flush=True); return

    rng = np.random.default_rng(42)
    tasks = []
    for gi, (grp, dfp) in enumerate([("early", early), ("late", late)]):
        for si, (_, r) in enumerate(dfp.iterrows()):
            rs = reg[reg.sess_key == r.sess_key]
            resp = rs.loc[rs.resp_log2 == True, "unit"].astype(int).tolist()   # noqa: E712
            non = rs.loc[rs.resp_log2 == False, "unit"].astype(int).tolist()   # noqa: E712
            nsamp = list(rng.choice(non, size=min(N_NONRESP, len(non)), replace=False)) if non else []
            tasks.append((subject, r.sess_key, r.stem, r.orig, grp, resp, nsamp, n_match, k_draws, 100 * gi + si))

    print(f"=== TRIAL-MATCHED STATE-CONDITIONED (StimSens, n_match={n_match}, k_draws={k_draws}): "
          f"{len(tasks)} sessions x {n_workers} workers ===", flush=True)
    frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_run_one, tasks):
            if res[0] == "OK":
                _, key, grp, nfull, agg = res
                print(f"[ok] {key} ({grp}): {nfull} StimSens -> matched {n_match}, {len(agg)} units", flush=True)
                frames.append(agg)
            else:
                print(f"[{res[0]}] {res[1]}: {res[2] if len(res) > 2 else ''}", flush=True)
    res_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    res_df.to_csv(cache / f"b9_deliverable2_encoding_{subject}_{tag}.csv", index=False)
    if res_df.empty:
        print("[done] no rows", flush=True); return

    rr = res_df[res_df.resp_class == "responsive"]
    e = rr[rr.group == "early"]["c1_r"].dropna(); l = rr[rr.group == "late"]["c1_r"].dropna()
    nr = res_df[res_df.resp_class == "nonresponsive"]
    p = stats.mannwhitneyu(e, l, alternative="two-sided")[1] if len(e) and len(l) else np.nan
    print(f"\n[RESULT] {subject} trial-matched (N={n_match}) StimSens responsive c1_r: "
          f"early median={e.median():.3f} (n={len(e)})  late median={l.median():.3f} (n={len(l)})  MWU p={p:.4f}", flush=True)
    print(f"[RESULT] non-responsive control: early={nr[nr.group=='early']['c1_r'].median():.3f} "
          f"late={nr[nr.group=='late']['c1_r'].median():.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    xmap = {("early", "responsive"): 0, ("late", "responsive"): 1,
            ("early", "nonresponsive"): 2.4, ("late", "nonresponsive"): 3.4}
    col = {("early", "responsive"): "#c7c7c7", ("late", "responsive"): "#3474ae",
           ("early", "nonresponsive"): "#ececec", ("late", "nonresponsive"): "#aecbe6"}
    for (g, c), xp in xmap.items():
        d = res_df[(res_df.group == g) & (res_df.resp_class == c)]["c1_r"].dropna()
        if len(d):
            ax.boxplot(d, positions=[xp], widths=.55, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=col[(g, c)], alpha=.6), medianprops=dict(color="k"))
            ax.scatter(np.full(len(d), xp) + rng.uniform(-.12, .12, len(d)), d, s=16, color="k", alpha=.55, zorder=3)
    ax.axhline(0.2, ls="--", color="k", lw=.8, alpha=.5)
    ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(["early", "late", "early", "late"])
    ax.text(0.5, ax.get_ylim()[1], "RESPONSIVE", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.text(2.9, ax.get_ylim()[1], "non-responsive (ctrl)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"trial-matched c1_r (StimSens, N={n_match})")
    ax.set_title(f"{subject}: StimSens baseline-TF encoding, early vs late (trial-matched)\n"
                 f"responsive MWU p={p:.4f}  (early n={len(e)}, late n={len(l)})")
    fig.tight_layout(); fig.savefig(figdir / f"b9_deliverable2_state_conditioned_{tag}.png", dpi=150); plt.close(fig)
    print(f"[done] wrote encoding CSV + figure for {subject} (n_match={n_match})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="BG_039")
    ap.add_argument("--n_workers", type=int, default=6)
    ap.add_argument("--n_match", type=int, default=140)
    ap.add_argument("--k_draws", type=int, default=3)
    a = ap.parse_args()
    main(a.subject, a.n_workers, a.n_match, a.k_draws)
