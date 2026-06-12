"""Fig0N (B0): Which DDM knob does learning turn? Two-route change-detection accumulator.

Decomposes BG_046's Naive->Expert behavioural improvement into drift-diffusion
parameters: sensitivity (drift gain ``v``) vs caution (bound ``a``) vs impulsivity
(urgency/start ``u``/``z``). Models are fit per stage and a nested model comparison
(:func:`visdetect.analysis.ddm.compare_stage_models`) names the parameter learning
turns. Early licks (FAs) are tested as TF-evidence-driven vs time-driven
(``route_attribution``, Step 0b), and a state-resolved secondary asks whether the
TF share of early licks is higher in engaged than impulsive states (spec sec 5).

Behaviour-only (spec B0, tier T1). DDM fitting is slow, so trial counts are CAPPED
per stage (a few hundred trials give stable DDM parameters) and the optimizer is
bounded + seeded for tractable, reproducible runs. Tune the caps / FITPARAMS below.

Outputs:
  - figures/01_behavior/fig0N_ddm_learning_knob.png   (panels A-F)
  - figures/01_behavior/ddm_learning_stats.csv        (per-stage params, deltas, model comp)
  - cache/ddm_per_stage_fits.csv                       (headline summary row)
"""
import os
import gc
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import STAGE_ORDER, STAGE_COLORS, CACHE_DIR, FIGURE_DIR
from visdetect.suite.loader import load_session
from visdetect.suite.plotting import setup_style, save_figure
from visdetect.analysis.config import load_staging_manifest
from visdetect.analysis import ddm

warnings.filterwarnings("ignore")
setup_style()

# ── Config (tune for runtime vs thoroughness) ────────────────────────
# Bounded + seeded differential evolution: tractable on large trial sets, reproducible.
FITPARAMS = {"seed": 0, "maxiter": 15, "popsize": 8, "polish": False}
UID_STRIDE = 100000     # per-session offset so trial_uid is globally unique
STAGE_MAX = 600         # cap trials/stage for the (slow) per-stage fits
POOLED_MAX = 300        # cap pooled trials for Step 0 structure selection / Step 0b route attr
STATE_MAX = 300         # cap trials/state for the route-mixture secondary
CV_K = 2                # folds for CV log-likelihood on real data
RUN_STRUCTURE = True    # Step 0 structural grid (expensive: 6 specs x CV_K fits)
RUN_STATE = True        # spec sec 5 state-resolved secondary (needs HMM labels)
SEED = 42
PARAM_KEYS = ("v", "a", "z", "u")

OUTDIR = os.path.join(FIGURE_DIR, "01_behavior")
CACHE = os.path.join(CACHE_DIR, "ddm_per_stage_fits.csv")
STATS = os.path.join(OUTDIR, "ddm_learning_stats.csv")


# ── Data assembly ────────────────────────────────────────────────────
def _subsample(df, n, seed=SEED):
    if df is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _evmap_for(df, evmap):
    """Restrict an evmap to trial_uids present in df (keeps solves to what's used)."""
    uids = set(int(u) for u in df["trial_uid"].tolist())
    return {u: evmap[u] for u in uids if u in evmap}


def _bucket_state(label):
    """Map a raw behavioural-state label to {'engaged','impulsive', None}.

    Pluggable/heuristic: 'impulsive' if the label looks impulsive; 'engaged' if it
    looks engaged/stimulus-sensitive; otherwise None (excluded from the secondary).
    """
    if not isinstance(label, str):
        return None
    s = label.lower()
    if "impuls" in s:
        return "impulsive"
    if "engag" in s or "sens" in s or "stim" in s or "attent" in s:
        return "engaged"
    return None


def build_stage_samples(state_control=False):
    """Return {stage: (sample_df, evmap)} pooled across that stage's sessions.

    sample_df is tidy (trial_uid, RT, lick, change_size[, state]). RT = decision_time,
    the first-passage time measured from Baseline_ON (the model's t=0) — NOT relative
    to change onset. evmap maps trial_uid -> per-trial evidence trace e(t).
    """
    manifest = load_staging_manifest(qc_only=True)
    by_stage = {}
    for stage in manifest["stage"].unique():
        frames, evmap = [], {}
        for _, row in manifest[manifest["stage"] == stage].iterrows():
            sname = int(row["session_name"])
            try:
                sess = load_session(sname)
            except Exception:
                continue
            try:
                ev_df = ddm.build_trial_evidence(sess)
            except Exception:
                del sess; gc.collect(); continue
            if ev_df is None or len(ev_df) == 0:
                del sess; gc.collect(); continue
            ev_df = ev_df.copy()
            if state_control:
                try:
                    states = ddm.load_state_labels(sname)
                    ev_df["state"] = ev_df["trial_uid"].map(states)  # keyed by trial_idx (pre-offset)
                except Exception:
                    ev_df["state"] = np.nan
            ev_df["trial_uid"] = ev_df["trial_uid"].astype(int) + sname * UID_STRIDE
            for _, r in ev_df.iterrows():
                evmap[int(r["trial_uid"])] = r["evidence"]
            ev_df["RT"] = ev_df["decision_time"].astype(float)
            keep = ["trial_uid", "RT", "lick", "change_size"]
            if state_control:
                keep.append("state")
            frames.append(ev_df[keep])
            del sess; gc.collect()
        if frames:
            by_stage[stage] = (pd.concat(frames, ignore_index=True), evmap)
    return by_stage


# ── Figure panels (each defensive: a panel failure must not kill the figure) ──
def _safe_panel(fn, ax, *args):
    try:
        fn(ax, *args)
    except Exception as e:  # noqa: BLE001
        ax.text(0.5, 0.5, f"{fn.__name__} failed:\n{e}", ha="center", va="center",
                fontsize=8, transform=ax.transAxes, color="crimson")
        ax.set_xticks([]); ax.set_yticks([])


def _panel_rt(ax, capped, per_stage, R, U):
    """A. Empirical vs model RT distributions per stage (lick trials)."""
    for stage, (df, ev) in capped.items():
        emp = df.loc[df["lick"] == 1, "RT"].to_numpy()
        emp = emp[np.isfinite(emp) & (emp > 0)]
        color = STAGE_COLORS.get(stage, None)
        if len(emp):
            ax.hist(emp, bins=30, range=(0, 3.5), density=True, histtype="step",
                    color=color, lw=2, label=f"{stage} data")
        # model overlay: simulate from the stage's fitted params on its own trials
        try:
            p = {k: per_stage[stage][k] for k in ("v", "a", "z", "u", "t0", "lam")}
            conds = {u: {"trial_uid": u} for u in list(ev)[:200]}
            sim = ddm.simulate_sample(_evmap_for_keys(ev, conds), conds, p, R=R,
                                      urgency=U, n_per_trial=20, seed=0)
            mrt = sim.loc[sim["lick"] == 1, "RT"].to_numpy()
            mrt = mrt[np.isfinite(mrt) & (mrt > 0)]
            if len(mrt):
                ax.hist(mrt, bins=30, range=(0, 3.5), density=True, histtype="step",
                        color=color, lw=1, ls="--")
        except Exception:
            pass
    ax.set_xlabel("RT from Baseline_ON (s)"); ax.set_ylabel("density")
    ax.set_title("A. RT distributions (— data, -- model)"); ax.legend(fontsize=7)


def _evmap_for_keys(evmap, conds):
    return {u: evmap[u] for u in conds if u in evmap}


def _panel_psych(ax, capped):
    """B. Empirical P(lick) vs change_size per stage."""
    for stage, (df, _) in capped.items():
        g = df.copy()
        g["cs"] = g["change_size"].round(2)
        agg = g.groupby("cs")["lick"].mean()
        ax.plot(agg.index, agg.values, "o-", color=STAGE_COLORS.get(stage), label=stage, ms=4)
    ax.set_xlabel("change_size"); ax.set_ylabel("P(lick)")
    ax.set_title("B. Psychometric (data)"); ax.legend(fontsize=7)


def _panel_params(ax, per_stage, stages):
    """C. Fitted v, a, z, u across stages (point estimates)."""
    x = np.arange(len(stages))
    width = 0.2
    for i, k in enumerate(PARAM_KEYS):
        vals = [per_stage[s].get(k, np.nan) for s in stages]
        ax.bar(x + (i - 1.5) * width, vals, width, label=k)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("fitted value"); ax.set_title("C. Parameters by stage")
    ax.legend(fontsize=7, ncol=4)


def _panel_modelcomp(ax, comp, attr):
    """D. Nested model comparison AIC (lower = better) + route-attribution note."""
    names = list(comp["aic"]); vals = [comp["aic"][n] for n in names]
    colors = ["crimson" if n == comp["winner"] else "0.6" for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("AIC")
    ax.set_title(f"D. Model comparison (winner={comp['winner']})")
    ax.tick_params(axis="x", rotation=30)
    note = (f"two-route wins: {attr.get('two_route_wins')}\n"
            f"CVLL two={attr.get('two_route_cvll', float('nan')):.1f} "
            f"tf_only={attr.get('tf_only_cvll', float('nan')):.1f}")
    ax.text(0.98, 0.95, note, ha="right", va="top", fontsize=7, transform=ax.transAxes)


def _panel_recovery(ax, R, U):
    """E. Parameter recovery: recovered vs true on synthetic data (identifiability check)."""
    rng = np.random.default_rng(0)
    evmap, conds = {}, {}
    for uid in range(250):
        ct = rng.uniform(0.8, 1.8); n = 150; e = np.zeros(n); e[int(ct / 0.02):] = 2.0
        evmap[uid] = e; conds[uid] = {"trial_uid": uid, "change_time": ct}
    true = dict(v=2.5, a=1.0, z=0.0, u=0.4, t0=0.05, lam=0.0)
    rec = ddm.recover_parameters(true, evmap, conds, R=R, urgency=U, n_per_trial=1,
                                 seed=1, fitparams=FITPARAMS)
    xs = [true[k] for k in PARAM_KEYS]; ys = [rec.get(k, np.nan) for k in PARAM_KEYS]
    ax.scatter(xs, ys, c="k")
    for k, xv, yv in zip(PARAM_KEYS, xs, ys):
        ax.annotate(k, (xv, yv), fontsize=8)
    lim = [0, max([v for v in xs + ys if np.isfinite(v)] + [1]) * 1.2]
    ax.plot(lim, lim, "0.6", ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true"); ax.set_ylabel("recovered"); ax.set_title("E. Parameter recovery")


def _panel_state(ax, state_mix):
    """F. TF (route-1) share of early licks by behavioural state (engaged vs impulsive)."""
    if not state_mix:
        ax.text(0.5, 0.5, "state mixture unavailable", ha="center", va="center",
                transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]);
        ax.set_title("F. TF share by state"); return
    names = list(state_mix); vals = [state_mix[n]["tf_share"] for n in names]
    ax.bar(names, vals, color=[STAGE_COLORS.get(n, "0.5") for n in names])
    ax.set_ylim(0, 1); ax.set_ylabel("TF-route share of early licks")
    ax.set_title("F. TF share by state (engaged>impulsive?)")


# ── Outputs ──────────────────────────────────────────────────────────
def _write_outputs(comp, attr, struct, per_stage, stages, state_mix):
    rows = []
    for s in stages:
        row = {"stage": s}
        row.update({k: per_stage[s].get(k) for k in ("v", "a", "z", "u", "t0", "lam")})
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats["winner"] = comp["winner"]
    stats["delta_v"] = comp["delta_v"]; stats["delta_u"] = comp["delta_u"]
    stats["R"] = struct["R"]; stats["urgency"] = struct["urgency"]
    stats["two_route_wins"] = attr.get("two_route_wins")
    stats.to_csv(STATS, index=False)

    summary = {"winner": comp["winner"], "delta_v": comp["delta_v"],
               "delta_u": comp["delta_u"], "R": struct["R"], "urgency": struct["urgency"],
               "two_route_cvll": attr.get("two_route_cvll"),
               "tf_only_cvll": attr.get("tf_only_cvll"),
               "two_route_wins": attr.get("two_route_wins")}
    for s in stages:
        for k in PARAM_KEYS:
            summary[f"{k}_{s}"] = per_stage[s].get(k)
    for st, d in (state_mix or {}).items():
        summary[f"tf_share_{st}"] = d.get("tf_share")
    summary.update({f"aic_{n}": v for n, v in comp["aic"].items()})
    pd.DataFrame([summary]).to_csv(CACHE, index=False)


def main():
    print("[01h] B0 DDM learning-knob ...")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    by_stage = build_stage_samples(state_control=RUN_STATE)
    stages = [s for s in STAGE_ORDER if s in by_stage] or list(by_stage)
    by_stage = {s: by_stage[s] for s in stages}
    print("  stages: " + ", ".join(f"{s}:{len(df)}" for s, (df, _) in by_stage.items()))
    if len(by_stage) < 2:
        print("  Need >=2 stages. Exiting."); return

    # Step 0 / 0b on a pooled subsample
    pooled = pd.concat([df for df, _ in by_stage.values()], ignore_index=True)
    pooled_ev = {}
    for _, ev in by_stage.values():
        pooled_ev.update(ev)
    pooled_s = _subsample(pooled, POOLED_MAX)
    pooled_ev_s = _evmap_for(pooled_s, pooled_ev)

    if RUN_STRUCTURE:
        struct = ddm.select_structure(pooled_s, pooled_ev_s, dt=ddm.DT, k=CV_K,
                                       fitparams=FITPARAMS)
    else:
        struct = {"R": "halfwave", "urgency": "rising", "scores": {}}
    R, U = struct["R"], struct["urgency"]
    print(f"  structure: R={R} urgency={U}")

    attr = ddm.route_attribution(pooled_s, pooled_ev_s, R=R, urgency=U, dt=ddm.DT,
                                 k=CV_K, fitparams=FITPARAMS)
    print(f"  route: two={attr['two_route_cvll']:.1f} tf_only={attr['tf_only_cvll']:.1f} "
          f"two_wins={attr['two_route_wins']}")

    # Headline: per-stage nested comparison (capped per stage)
    capped = {}
    for s, (df, ev) in by_stage.items():
        sub = _subsample(df, STAGE_MAX)
        capped[s] = (sub, _evmap_for(sub, ev))
    comp = ddm.compare_stage_models(capped, R=R, urgency=U, dt=ddm.DT, fitparams=FITPARAMS)
    per_stage = comp["per_stage"]
    print(f"  WINNER={comp['winner']}  dv={comp['delta_v']:.3f}  du={comp['delta_u']:.3f}")

    # State-resolved route mixture (secondary)
    state_mix = {}
    if RUN_STATE and "state" in pooled.columns:
        try:
            sc = pooled.assign(_bucket=pooled["state"].map(_bucket_state))
            sbs = {}
            for b in ("engaged", "impulsive"):
                sub = _subsample(sc[sc["_bucket"] == b], STATE_MAX)
                if sub is not None and len(sub) >= 30:
                    sbs[b] = (sub, _evmap_for(sub, pooled_ev))
            if len(sbs) == 2:
                state_mix = ddm.route_mixture_by_state(sbs, R=R, urgency=U, dt=ddm.DT,
                                                       fitparams=FITPARAMS)
                print("  state tf_share: " + ", ".join(
                    f"{k}={v['tf_share']:.2f}" for k, v in state_mix.items()))
        except Exception as e:  # noqa: BLE001
            print(f"  state mixture skipped: {e}")

    # Figure
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.30)
    _safe_panel(_panel_rt, fig.add_subplot(gs[0, 0]), capped, per_stage, R, U)
    _safe_panel(_panel_psych, fig.add_subplot(gs[0, 1]), capped)
    _safe_panel(_panel_params, fig.add_subplot(gs[0, 2]), per_stage, stages)
    _safe_panel(_panel_modelcomp, fig.add_subplot(gs[1, 0]), comp, attr)
    _safe_panel(_panel_recovery, fig.add_subplot(gs[1, 1]), R, U)
    _safe_panel(_panel_state, fig.add_subplot(gs[1, 2]), state_mix)
    fig.suptitle("B0 — Which DDM knob does learning turn?", fontsize=14)
    save_figure(fig, "fig0N_ddm_learning_knob", "01_behavior")
    plt.close(fig)

    _write_outputs(comp, attr, struct, per_stage, stages, state_mix)
    print(f"  saved figure + {STATS} + {CACHE}")


if __name__ == "__main__":
    main()
