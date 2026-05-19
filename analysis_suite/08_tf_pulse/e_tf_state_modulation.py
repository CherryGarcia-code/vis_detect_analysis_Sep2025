"""Fig39: TF state modulation — TF responses split by HMM behavioral state.

REQUIRES session pickles for raw spike times + trial structure.
Uses the NPZ cache to identify TF-responsive units (avoids recomputing
`collect_tf_pulse_traces()`).  HMM state assignments come from the
standard CSV file.

Groups fast TF pulses by the HMM state of the trial they belong to,
then builds state-conditioned PSTHs per unit.

Produces fig39_tf_state_modulation.png:
  - Panel A: Population TF PSTHs split by HMM state
  - Panel B: Heatmap of state modulation index per unit
  - Panel C: Distribution of state modulation index
  - Panel D: State modulation by cell type and learning stage
"""

import argparse
import os
import sys


import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visdetect.suite.config import (
    STAGE_ORDER, STAGE_COLORS, HMM_STATE_ORDER, HMM_STATE_COLORS,
    CELLTYPE_COLORS, CACHE_DIR, DEFAULT_Z_THRESH_TF,
)
from visdetect.analysis.constants import TF_PULSE_PRE_WINDOW, TF_PULSE_POST_WINDOW
from visdetect.suite.loader import (
    load_staging_manifest, load_waveform_labels, load_hmm_assignments,
    load_tf_traces_npz, session_iterator,
)
from visdetect.suite.plotting import setup_style, save_figure

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

setup_style()

Z_THRESH = DEFAULT_Z_THRESH_TF

# ── Parameters ─────────────────────────────────────────────────────
DT = 0.001
SIGMA_MS = 17.0
PRE_WIN = TF_PULSE_PRE_WINDOW
POST_WIN = TF_PULSE_POST_WINDOW


def _vectorized_psth(spike_times, pulses, t_vec, sigma_bins):
    """Fast vectorized pulse-triggered histogram."""
    if pulses.size == 0:
        return np.zeros_like(t_vec)
    dt = t_vec[1] - t_vec[0]
    full0, full1 = t_vec[0], t_vec[-1] + dt
    all_rel = []
    for tp in pulses:
        lo = np.searchsorted(spike_times, tp + full0)
        hi = np.searchsorted(spike_times, tp + full1)
        if hi > lo:
            all_rel.append(spike_times[lo:hi] - tp)
    if not all_rel:
        return np.zeros_like(t_vec)
    all_rel = np.concatenate(all_rel)
    counts, _ = np.histogram(all_rel, bins=np.append(t_vec, t_vec[-1] + dt))
    rate = counts.astype(float) / pulses.size
    return gaussian_filter1d(rate, sigma=sigma_bins)


def _smooth(rel_times, t_vec, sigma_bins):
    train = np.zeros_like(t_vec)
    if rel_times.size == 0:
        return train
    idx = np.searchsorted(t_vec, rel_times)
    idx = idx[(idx >= 0) & (idx < train.size)]
    train[idx] = 1.0
    return gaussian_filter1d(train, sigma=sigma_bins)


def _zscore(trace, t_vec, pre_win):
    pre_mask = (t_vec >= pre_win[0]) & (t_vec < pre_win[1])
    mu = np.nanmean(trace[pre_mask]) if np.any(pre_mask) else 0.0
    sd = np.nanstd(trace[pre_mask]) if np.any(pre_mask) else 0.0
    if not np.isfinite(sd) or sd <= 0:
        return trace * 0.0
    return (trace - mu) / sd


def main():
    parser = argparse.ArgumentParser(description="TF state modulation")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print("[08e] TF State Modulation  [requires session pkls + HMM]")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    hmm = load_hmm_assignments()
    print(f"  Sessions: {len(manifest)}, HMM trials: {len(hmm)}")

    # Build responsive unit set from NPZ
    responsive_set = set()
    for _, row in manifest.iterrows():
        sname = int(row["session_name"])
        npz = load_tf_traces_npz(sname)
        if npz is None:
            continue
        for i, cid in enumerate(npz["cluster_ids"]):
            z_abs = max(abs(npz["z_max_fast"][i]), abs(npz["z_min_fast"][i]),
                        abs(npz["z_max_slow"][i]), abs(npz["z_min_slow"][i]))
            if z_abs >= Z_THRESH:
                responsive_set.add((sname, int(cid)))
    print(f"  TF-responsive units (NPZ): {len(responsive_set)}")

    ct_lookup = {}
    try:
        wf = load_waveform_labels()
        for _, r in wf.iterrows():
            ct_lookup[(int(r["session_name"]), int(r["cluster_id"]))] = r["cell_type"]
    except Exception:
        pass

    from visdetect.analysis.tf_pulse import _collect_pulses, TFRespPulseConfig
    from visdetect.analysis.align import get_event_times_by_trial

    cfg = TFRespPulseConfig()
    t_vec = np.arange(PRE_WIN[0], POST_WIN[1], DT, dtype=float)
    sigma_bins = (SIGMA_MS / 1000.0) / DT
    post_mask = (t_vec >= POST_WIN[0]) & (t_vec < POST_WIN[1])

    records = []
    pop_by_state = {st: [] for st in HMM_STATE_ORDER}

    # Build session_idx lookup from manifest
    sidx_lookup = {int(r["session_name"]): r["session_idx"] for _, r in manifest.iterrows()}

    for sname_int, stage, session in session_iterator():
        sidx = sidx_lookup.get(sname_int, -1)

        # Get HMM state for each trial
        hmm_sess = hmm[hmm["session_name"].astype(int) == sname_int]
        if len(hmm_sess) == 0:
            print(f"    {sname_int}: no HMM labels – skip")
            continue

        trials = getattr(session, "trials", []) or []
        base_on = np.array(get_event_times_by_trial(session, "Baseline_ON"), dtype=float)

        # Map trial index (1-based) to HMM state
        trial_state = {}
        for _, hr in hmm_sess.iterrows():
            trial_state[int(hr["trial_idx"])] = hr["hmm_state_label"]

        # Collect fast pulses per HMM state
        fast_by_state = {st: [] for st in HMM_STATE_ORDER}

        for ti, t in enumerate(trials, 1):
            bv = getattr(t, "baseline_values", None)
            if bv is None:
                continue
            state = trial_state.get(ti, None)
            if state is None or state not in HMM_STATE_ORDER:
                continue
            arr = np.array(bv).flatten()
            if cfg.baseline_stride > 1:
                arr = arr[::cfg.baseline_stride]
            n_seen = getattr(t, "n_seen", None)
            if isinstance(n_seen, (int, np.integer)) and n_seen and n_seen > 0:
                arr = arr[:int(n_seen)]
            from visdetect.analysis.tf_pulse import _safe_log2
            log2_tf = _safe_log2(arr)
            t0 = float(base_on[ti]) if ti < len(base_on) and np.isfinite(base_on[ti]) else None
            if t0 is None:
                continue
            for bi, l2 in enumerate(log2_tf):
                if not np.isfinite(l2):
                    continue
                if l2 >= cfg.fast_thresh_log2:
                    t_pulse = t0 + bi * cfg.sample_period
                    if t_pulse >= t0 + cfg.min_after_baseline:
                        fast_by_state[state].append(float(t_pulse))

        n_pulses = {st: len(v) for st, v in fast_by_state.items()}
        n_total = sum(n_pulses.values())
        if n_total < 20:
            continue
        print(f"    {sname_int}: {n_total} fast pulses ({', '.join(f'{s}={n}' for s,n in n_pulses.items())})")

        # Per-unit PSTHs by state
        for c in session.clusters:
            cid = int(c.cluster_id)
            if (sname_int, cid) not in responsive_set:
                continue
            st_arr = np.sort(np.asarray(c.spike_times, dtype=float).flatten())
            if st_arr.size == 0:
                continue

            state_amps = {}
            for state_name in HMM_STATE_ORDER:
                pulses = np.sort(np.array(fast_by_state[state_name]))
                if pulses.size < 5:
                    continue
                mean_trace = _zscore(_vectorized_psth(st_arr, pulses, t_vec, sigma_bins), t_vec, PRE_WIN)
                amp = float(np.nanmax(np.abs(mean_trace[post_mask])))
                state_amps[state_name] = amp
                pop_by_state[state_name].append(mean_trace.copy())

            if len(state_amps) >= 2:
                amp_eng = state_amps.get("Engaged", np.nan)
                amp_dis = state_amps.get("Disengaged", np.nan)
                if np.isfinite(amp_eng) and np.isfinite(amp_dis) and (amp_eng + amp_dis) > 0:
                    smi = (amp_eng - amp_dis) / (amp_eng + amp_dis)
                else:
                    smi = np.nan
                records.append({
                    "session_name": sname_int, "cluster_id": cid,
                    "stage": stage, "session_idx": sidx,
                    "cell_type": ct_lookup.get((sname_int, cid), "Unknown"),
                    "amp_engaged": amp_eng,
                    "amp_disengaged": amp_dis,
                    "amp_impulsive": state_amps.get("Impulsive", np.nan),
                    "state_mod_index": smi,
                })

    df = pd.DataFrame(records)
    print(f"\n  Units with state data: {len(df)}")

    if len(df) == 0:
        print("  No data. Exiting.")
        return

    cache_path = os.path.join(CACHE_DIR, "tf_state_modulation.csv")
    df.to_csv(cache_path, index=False)
    print(f"  Cached: {cache_path}")

    # ── Create figure ─────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Population PSTHs by state ────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for state in HMM_STATE_ORDER:
        traces = pop_by_state[state]
        if not traces:
            continue
        mn = np.nanmean(np.stack(traces), axis=0)
        se = np.nanstd(np.stack(traces), axis=0) / np.sqrt(len(traces))
        c = HMM_STATE_COLORS.get(state, "#999")
        ax_a.plot(t_vec * 1000, mn, color=c, linewidth=1.5,
                  label=f"{state} (n={len(traces)})")
        ax_a.fill_between(t_vec * 1000, mn - se, mn + se, color=c, alpha=0.15)
    ax_a.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.set_xlabel("Time from fast TF pulse (ms)")
    ax_a.set_ylabel("Mean z-score")
    ax_a.set_title("A – Population TF PSTH by HMM state")
    ax_a.legend(fontsize=8)

    # ── Panel B: Heatmap ──────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    smi_vals = df["state_mod_index"].dropna().values
    if len(smi_vals) > 5:
        sorted_idx = np.argsort(smi_vals)
        ax_b.barh(np.arange(len(smi_vals)), smi_vals[sorted_idx],
                  color=np.where(smi_vals[sorted_idx] > 0, "#E53935", "#1565C0"),
                  edgecolor="none", height=1.0)
        ax_b.axvline(0, color="k", linewidth=0.5)
        ax_b.set_xlabel("State modulation index\n(+Engaged, −Disengaged)")
        ax_b.set_ylabel("Unit (sorted)")
    ax_b.set_title(f"B – State modulation index (n={len(smi_vals)})")

    # ── Panel C: SMI distribution ─────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    smi_clean = smi_vals[np.isfinite(smi_vals)]
    if len(smi_clean):
        ax_c.hist(smi_clean, bins=30, color="#7E57C2", edgecolor="white",
                  linewidth=0.5, alpha=0.8)
        med = np.median(smi_clean)
        ax_c.axvline(0, color="grey", linewidth=1, linestyle=":")
        ax_c.axvline(med, color="#E53935", linewidth=1.5, linestyle="--",
                     label=f"Median={med:.3f}")
    ax_c.set_xlabel("State modulation index")
    ax_c.set_ylabel("Count")
    ax_c.set_title("C – SMI distribution")
    ax_c.legend(fontsize=8)

    # ── Panel D: SMI by cell type and stage ───────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    stages = [s for s in STAGE_ORDER if s in df["stage"].values]
    cell_types = sorted([c for c in df["cell_type"].unique() if c != "Unknown"])
    x = np.arange(len(stages))
    if cell_types:
        w = 0.3 / len(cell_types)
        for ci, ct in enumerate(cell_types):
            meds = [df[(df["stage"]==s)&(df["cell_type"]==ct)]["state_mod_index"].median()
                    for s in stages]
            ax_d.bar(x + ci * w, meds, w,
                     color=CELLTYPE_COLORS.get(ct, "#999"),
                     edgecolor="black", linewidth=0.3, label=ct)
    else:
        meds = [df[df["stage"]==s]["state_mod_index"].median() for s in stages]
        ax_d.bar(x, meds, 0.4, color=[STAGE_COLORS[s] for s in stages],
                 edgecolor="black", linewidth=0.5)
    ax_d.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax_d.set_xticks(x + 0.15 if cell_types else x)
    ax_d.set_xticklabels(stages)
    ax_d.set_ylabel("Median SMI")
    ax_d.set_title("D – State modulation by stage & cell type")
    if cell_types:
        ax_d.legend(fontsize=7)

    fig.suptitle(
        "TF Pulse State Modulation\n"
        "(HMM behavioral states: Engaged vs. Disengaged)",
        fontsize=13, fontweight="bold", y=0.98)
    save_figure(fig, "fig39_tf_state_modulation", "08_tf_pulse")
    print("\n  Saved fig39_tf_state_modulation.png")


if __name__ == "__main__":
    main()
