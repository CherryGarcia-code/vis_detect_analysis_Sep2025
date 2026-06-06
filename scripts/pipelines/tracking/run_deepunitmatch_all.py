#!/usr/bin/env python3
"""Canonical DeepUnitMatch pipeline on all 42 BG_046 sessions, single batch.

Mirrors the official DeepUnitMatch.ipynb demo end-to-end:
  1. Load waveforms (UM)
  2. param_fun.get_snippets        official preprocessing (writes HDF5)
  3. test.load_trained_model       shipped CLIP-trained CNN
  4. test.inference                similarity matrix
  5. split_units.merge_and_remove_splits   (optional, canonical)
  6. UM extract_metric_scores (centroid_dist only)  -> distance matrix
  7. Per session-pair Naive Bayes (similarity x distance)  -> prob matrix
  8. assign_unique_id              tracking

One adapter (NOT custom preprocessing): the per-session sorts excluded the
dead probe channel 127, leaving 383 channels. get_snippets strictly requires
384. We re-insert chan 127 as a zero waveform at its interpolated position
(midpoint of chan 126 / chan 128). A dead channel produces zero signal at
its real location -- physically truthful, neutral for the network.

Must run under unitmatch_env (UnitMatchPy 3.2.9 + torch, mat73, numpy<2)
via `conda run -n unitmatch_env --no-capture-output python -u ...`.

Output (separate from UM):
  data/unit_match/output/BG_046_all42_deep/
    cell_registry.csv, unit_index.csv,
    prob_matrix.npy, sim_matrix.npy, run_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import importlib

import UnitMatchPy as _umpy

# DeepUnitMatch is a PEP-420 namespace dir (no __init__.py) shipped as a
# sibling of the UnitMatchPy python package, inside the vendored repo.
# To `import DeepUnitMatch.*` we need the repo dir on sys.path.
# To satisfy DeepUnitMatch/testing/test.py's `from utils.losses import ...`
# we also need DeepUnitMatch/ itself on sys.path.
#
# Locate the repo dir by trying, in order:
#   1. SIBLING of this script (cluster staging: unit_match/UnitMatchPy/).
#      This is the most reliable -- script path is a hard fact.
#   2. parent(UnitMatchPy.__path__[0]) (editable install introspection;
#      works on legacy editable installs, may not point to source on PEP-660).
# Whichever has `DeepUnitMatch/` as a real subdir wins. Fail LOUDLY if neither
# does -- so we never silently get an empty namespace package.
_SCRIPT_DIR = Path(__file__).resolve().parent
_CANDIDATES = [_SCRIPT_DIR / "UnitMatchPy"]
try:
    _CANDIDATES.append(Path(next(iter(_umpy.__path__))).resolve().parent)
except Exception:  # noqa: BLE001
    pass

_REPO_DIR = next((c for c in _CANDIDATES if (c / "DeepUnitMatch").is_dir()), None)
if _REPO_DIR is None:
    raise RuntimeError(
        "Cannot locate DeepUnitMatch source. Tried: "
        + " | ".join(str(c) for c in _CANDIDATES)
    )
print(f"  DeepUnitMatch source -> {_REPO_DIR / 'DeepUnitMatch'}", flush=True)

for p in (_REPO_DIR, _REPO_DIR / "DeepUnitMatch"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
importlib.invalidate_caches()

import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as ov
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params

from DeepUnitMatch.utils import param_fun
from DeepUnitMatch.utils import helpers as dum_helpers
from DeepUnitMatch.testing import test as dum_test
from DeepUnitMatch.preprocess import split_units


# Monkey-patch DeepUnitMatch's sort_good_channels / extract_Rwaveforms to
# tolerate the case where within-radius good channels split unequally between
# the two columns of a NPx 2.0 shank (e.g. 16+14 of 30 when the max-site is
# near a shank edge). Shipped code's broadcast-write assumes exact 15+15 and
# crashed at session 1 unit 0 on BG_046 (job 3022569). We interleave only as
# many pairs as the smaller column allows and compute padding from the
# actually-used channel count so Rwaveform still pads back to (60, 30, 2).
def _patched_sort_good_channels(goodChannelMap, goodpos):
    unique_y_values = np.unique(goodpos[:, 0])
    unique_y_values.sort()
    if len(unique_y_values) != 2:
        return [-1], [-1]
    idx_a = np.where(goodpos[:, 0] == unique_y_values[0])[0]
    idx_b = np.where(goodpos[:, 0] == unique_y_values[1])[0]
    chan_a, chan_b = goodChannelMap[idx_a], goodChannelMap[idx_b]
    pos_a, pos_b = goodpos[idx_a], goodpos[idx_b]
    chan_a = chan_a[np.argsort(pos_a[:, 1])]
    chan_b = chan_b[np.argsort(pos_b[:, 1])]
    pos_a = pos_a[np.argsort(pos_a[:, 1])]
    pos_b = pos_b[np.argsort(pos_b[:, 1])]
    n_pairs = min(len(chan_a), len(chan_b))
    n_out = 2 * n_pairs
    out_chan = np.empty(n_out, dtype=goodChannelMap.dtype)
    out_chan[::2] = chan_a[:n_pairs]
    out_chan[1::2] = chan_b[:n_pairs]
    out_pos = np.empty((n_out, goodpos.shape[1]), dtype=goodpos.dtype)
    out_pos[::2, :] = pos_a[:n_pairs]
    out_pos[1::2, :] = pos_b[:n_pairs]
    return out_chan, out_pos


def _patched_extract_Rwaveforms(waveform, ChannelPos, ChannelMap, param):
    nChannels = param['nChannels']
    nTime = param['nTime']
    RnChannels = param['RnChannels']
    RnTime = param['RnTime']
    ChannelRadius = param['ChannelRadius']
    start_time, end_time = (nTime - RnTime) // 2, (nTime + RnTime) // 2
    if waveform.ndim == 2:
        waveform = np.stack([waveform, waveform], axis=2)
    waveform = waveform[start_time:end_time, :, :]
    waveform = param_fun.detrend_waveform(waveform)
    MeanCV = np.mean(waveform, axis=2)
    SpatialFootprint = param_fun.get_spatialfp(MeanCV)
    MaxSiteMean = param_fun.get_max_site(SpatialFootprint)
    MaxSitepos = ChannelPos[MaxSiteMean, :]
    goodidx = np.empty(nChannels, dtype=bool)
    for i in range(ChannelPos.shape[0]):
        dist = np.linalg.norm(ChannelPos[MaxSiteMean, :] - ChannelPos[i, :])
        goodidx[i] = dist < ChannelRadius
    goodChannelMap = ChannelMap[goodidx]
    goodpos = ChannelPos * np.tile(goodidx, (2, 1)).T
    goodpos = goodpos[goodidx, :]
    sorted_goodChannelMap, sorted_goodpos = _patched_sort_good_channels(
        goodChannelMap, goodpos
    )
    if sorted_goodChannelMap[0] == -1 and sorted_goodpos[0] == -1:
        return np.array([-1, -1]), np.array([-1, -1]), [0], [0], np.zeros((1, 1, 1))
    Rwaveform = waveform[:, sorted_goodChannelMap, :]
    GlobalMean = np.mean(Rwaveform)
    Rwaveform = Rwaveform - GlobalMean
    NewGlobalMean = np.mean(Rwaveform)
    z_sorted_goodpos = np.unique(sorted_goodpos[:, 1])
    mean_z_sorted_goodpos = np.mean(z_sorted_goodpos)
    z_MaxSitepos = MaxSitepos[1]
    # PATCH: use the actually-used channel count, not raw within-radius count.
    num_good_channels = len(sorted_goodChannelMap)
    padding_needed = RnChannels - num_good_channels
    pad_before = padding_needed if z_MaxSitepos < mean_z_sorted_goodpos else 0
    pad_after = padding_needed if z_MaxSitepos >= mean_z_sorted_goodpos else 0
    Rwaveform = np.pad(
        Rwaveform, ((0, 0), (pad_before, pad_after), (0, 0)),
        'constant', constant_values=(NewGlobalMean, NewGlobalMean),
    )
    return MaxSiteMean, MaxSitepos, sorted_goodChannelMap, sorted_goodpos, Rwaveform


param_fun.sort_good_channels = _patched_sort_good_channels
param_fun.extract_Rwaveforms = _patched_extract_Rwaveforms
print("  patched param_fun.sort_good_channels / extract_Rwaveforms", flush=True)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
DEFAULT_OUT = REPO_ROOT / "data" / "unit_match" / "output" / "BG_046_all42_deep"


def parse_session_date(name: str) -> datetime:
    s = str(name)
    if len(s) == 7:
        s = "0" + s
    return datetime.strptime(s, "%d%m%Y")


def pad_to_384_channels(waveform, channel_pos_list, channel_maps):
    """Re-insert dropped channels so DeepUM gets the expected (82, 384, 2)
    layout. Channels missing from channel_map are inserted at their canonical
    index with zero waveform and a position interpolated from kept neighbours
    (midpoint of nearest-by-ID below + above).

    Returns (padded_waveform (..., 384, 2), padded channel_pos list).
    """
    n_units, n_t, n_chan, n_rep = waveform.shape
    cm0 = np.asarray(channel_maps[0])
    missing = sorted(set(range(384)) - set(cm0.tolist()))
    if not missing:
        return waveform, channel_pos_list
    print(f"  padding {len(missing)} missing channel(s): {missing}", flush=True)
    assert n_chan + len(missing) == 384, "channel count mismatch after pad"

    new_wave = np.zeros((n_units, n_t, 384, n_rep), dtype=waveform.dtype)
    new_wave[:, :, cm0, :] = waveform                       # cm0 are the kept IDs

    kept = np.asarray(sorted(set(range(384)) - set(missing)))
    new_pos_list = []
    for cp in channel_pos_list:
        cp = np.asarray(cp)
        new_cp = np.zeros((384, cp.shape[1]), dtype=cp.dtype)
        new_cp[cm0, :] = cp
        for c in missing:
            below = kept[kept < c]
            above = kept[kept > c]
            if len(below) and len(above):
                new_cp[c] = 0.5 * (new_cp[below[-1]] + new_cp[above[0]])
            elif len(below):
                new_cp[c] = new_cp[below[-1]]
            else:
                new_cp[c] = new_cp[above[0]]
        new_pos_list.append(new_cp)
    return new_wave, new_pos_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-merge-splits", action="store_true",
                    help="Skip split_units.merge_and_remove_splits step")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Fine-tuned export checkpoint; if set, used instead of the "
                         "shipped DeepUM model.")
    args = ap.parse_args()

    t0 = time.time()
    sess_dirs = sorted([d for d in args.input.iterdir() if d.is_dir()],
                       key=lambda d: parse_session_date(d.name))
    ks_dirs = [str(d) for d in sess_dirs]
    session_names = [d.name for d in sess_dirs]
    print(f"DeepUnitMatch on {len(ks_dirs)} sessions: "
          f"{session_names[0]} .. {session_names[-1]}", flush=True)

    # ── Load waveforms (canonical UM) ───────────────────────────────
    param = default_params.get_default_param()
    param["KS_dirs"] = ks_dirs
    wave_paths, label_paths, channel_pos = util.paths_from_KS(ks_dirs)
    param = util.get_probe_geometry(channel_pos[0], param)
    print("Loading good waveforms ...", flush=True)
    waveform, session_id, session_switch, within_session, good_units, param = \
        util.load_good_waveforms(wave_paths, label_paths, param, good_units_only=True)
    n_units = int(param["n_units"])
    print(f"  n_units={n_units}, waveform shape={waveform.shape}", flush=True)

    clus_info = {"good_units": good_units, "session_switch": session_switch,
                 "session_id": session_id,
                 "original_ids": np.concatenate(good_units)}

    # ── Pad 383 → 384 channels ──────────────────────────────────────
    channel_maps = [np.load(Path(ks) / "channel_map.npy") for ks in ks_dirs]
    waveform, channel_pos = pad_to_384_channels(waveform, channel_pos, channel_maps)
    # Keep param consistent with the padded waveform; otherwise UM's
    # extract_parameters allocates good_idx with the old n_channels=383 and
    # crashes when iterating to channel index 383 (job 3023802).
    param["n_channels"] = waveform.shape[2]
    print(f"  padded waveform shape={waveform.shape}  (param n_channels={param['n_channels']})",
          flush=True)

    # ── DeepUM canonical preprocessing (HDF5 cache) ─────────────────
    print("DeepUM step 1  get_snippets ...", flush=True)
    snippets, positions = param_fun.get_snippets(waveform, channel_pos, session_id)
    print(f"  snippets shape={snippets.shape}, positions shape={positions.shape}",
          flush=True)
    data_dir = Path(param_fun.__file__).parent.parent / "processed_waveforms"
    print(f"  HDF5 cache: {data_dir}", flush=True)

    # ── Network inference ──────────────────────────────────────────
    print("DeepUM step 2  load_trained_model + inference ...", flush=True)
    if args.ckpt is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_deepum_common import load_finetuned_encoder
        print(f"DeepUM step 2  load_finetuned_encoder({args.ckpt}) ...", flush=True)
        model = load_finetuned_encoder(str(args.ckpt), device="cpu")
    else:
        model = dum_test.load_trained_model(device="cpu")
    sim_matrix = dum_test.inference(model, str(data_dir))
    print(f"  sim_matrix shape={sim_matrix.shape} "
          f"(diag median {np.median(np.diag(sim_matrix)):.3f})", flush=True)

    # ── EARLY SAVE: preserve the expensive DeepUM output before any
    # ── downstream step that might fail. Re-saved at the end too.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "sim_matrix.npy", sim_matrix.astype(np.float32))
    np.save(args.out_dir / "positions.npy", positions.astype(np.float32))
    print(f"  early-saved sim_matrix + positions -> {args.out_dir}", flush=True)

    # ── Optional split-unit merging ────────────────────────────────
    # merge_and_remove_splits, when it succeeds, populates
    # param["good_units"] / param["n_units"] which the Bayes block below
    # then reads. When it fails (e.g. missing spike_clusters.npy in the
    # UM input dir, as in BG_046 job 3025315) those keys never get added
    # and the Bayes block KeyErrors. Backfill in both paths so the rest
    # of the pipeline runs regardless.
    if not args.no_merge_splits:
        print("DeepUM step 3  merge_and_remove_splits ...", flush=True)
        try:
            sim_matrix, param, session_id, session_switch = \
                split_units.merge_and_remove_splits(param, sim_matrix, session_id,
                                                    model, str(data_dir))
            n_units = int(param["n_units"])
            clus_info["session_switch"] = session_switch
            clus_info["session_id"] = session_id
            clus_info["good_units"] = param["good_units"]
            clus_info["original_ids"] = np.concatenate(param["good_units"])
            print(f"  post-merge n_units={n_units}", flush=True)
        except Exception as e:
            print(f"  WARNING split-merge failed ({e}); continuing un-merged",
                  flush=True)
            param["good_units"] = good_units
            param["n_units"] = n_units
    else:
        param["good_units"] = good_units
        param["n_units"] = n_units

    # ── UM distance matrix + Bayes fusion + tracking ───────────────
    # If DeepUM dropped units (e.g. NaN/Inf waveforms) sim_matrix has
    # fewer rows than UM's n_units and the Bayes/tracking step below
    # will fail on a mask shape mismatch. We detect that up front, and
    # wrap the whole block so the run still completes and produces the
    # DeepUM sim_matrix even if fusion is impossible.
    bayes_ok = False
    probs = None
    uids = None
    if sim_matrix.shape[0] != n_units:
        print(f"  SKIP UM/Bayes: sim_matrix has {sim_matrix.shape[0]} units "
              f"but UM has {n_units} (DeepUM skipped "
              f"{n_units - sim_matrix.shape[0]} units, likely NaN/Inf "
              f"waveforms). DeepUM sim_matrix already saved -- use that "
              f"for sim-only tracking.", flush=True)
    else:
        try:
            print("UM extract_parameters + extract_metric_scores (centroid_dist) ...",
                  flush=True)
            extracted = ov.extract_parameters(waveform, channel_pos, clus_info, param)
            within_session_mat = 1 - (session_id[:, None] == session_id).astype(int)
            distance_matrix, _, _, _ = ov.extract_metric_scores(
                extracted, session_switch, within_session_mat, param,
                niter=2, to_use=["centroid_dist"])

            print("Per session-pair Naive Bayes (similarity x distance) ...",
                  flush=True)
            sessions = np.unique(session_id)
            probs = np.zeros(sim_matrix.shape)
            n_pairs = len(sessions) * (len(sessions) - 1) // 2
            done = 0
            for r1 in sessions:
                for r2 in sessions:
                    if r1 >= r2:
                        continue
                    mask = np.isin(session_id, [r1, r2])
                    sm = sim_matrix[mask][:, mask]
                    dm = distance_matrix[mask][:, mask]
                    idx = np.where(np.isin(session_id, [r1, r2]))[0]
                    df = dum_helpers.create_dataframe(
                        [param["good_units"][r1], param["good_units"][r2]],
                        sm, session_list=[int(r1), int(r2)])
                    matches = dum_test.get_matches(
                        df, sm, session_id[idx], str(data_dir), positions[idx],
                        dist_thresh=20)
                    labels = np.eye(sm.shape[0])
                    subsess = np.array(
                        [r1] * len(param["good_units"][r1]) +
                        [r2] * len(param["good_units"][r2]))
                    for (rs1, rs2), grp in matches.groupby(by=["RecSes1", "RecSes2"]):
                        asm = (grp["match"].values
                               .reshape(len(param["good_units"][rs1]),
                                        len(param["good_units"][rs2]))
                               .astype(int))
                        labels[np.ix_(subsess == rs1, subsess == rs2)] = asm
                    sti = {"similarity": sm, "distance": dm}
                    n_pair = int(np.sqrt(len(df)))
                    priors = np.array([1 - 2 / n_pair, 2 / n_pair])
                    kernels = bf.get_parameter_kernels(sti, labels,
                                                       np.unique(labels), param)
                    preds = np.stack(list(sti.values()), axis=2)
                    probability = bf.apply_naive_bayes(kernels, priors, preds,
                                                       param, np.unique(labels))
                    probs[np.ix_(mask, mask)] = probability[:, 1].reshape(n_pair, n_pair)
                    done += 1
                    if done % 50 == 0:
                        print(f"  Bayes {done}/{n_pairs} pairs", flush=True)

            print("assign_unique_id ...", flush=True)
            uid_lists = aid.assign_unique_id(probs, param, clus_info)
            uids = np.asarray(uid_lists[1]).ravel()           # intermediate / default
            bayes_ok = True
        except Exception as e:
            import traceback
            print(f"  WARNING UM/Bayes/tracking failed: {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()
            print("  DeepUM sim_matrix is still saved; sim-only tracking can be done offline.",
                  flush=True)

    # ── Outputs ────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # sim_matrix + positions were already saved early; re-save here in case
    # downstream changed shapes (e.g. split-merge would have).
    np.save(args.out_dir / "sim_matrix.npy", sim_matrix.astype(np.float32))
    np.save(args.out_dir / "positions.npy", positions.astype(np.float32))

    summary = dict(
        n_sessions=len(ks_dirs), n_units=n_units,
        n_units_deepum=int(sim_matrix.shape[0]),
        elapsed_min=round((time.time() - t0) / 60, 1),
        unitmatchpy_version="3.2.9",
        bayes_completed=bool(bayes_ok),
    )

    if bayes_ok:
        sess_idx = np.asarray(session_id).ravel().astype(int)
        ks_ids = np.asarray(clus_info["original_ids"]).ravel().astype(int)
        unit_index = pd.DataFrame({
            "session": [session_names[s] for s in sess_idx],
            "ks_unit_id": ks_ids,
            "global_uid": uids,
        })
        unit_index.to_csv(args.out_dir / "unit_index.csv", index=False)

        reg = unit_index.pivot_table(index="global_uid", columns="session",
                                     values="ks_unit_id", aggfunc="first")
        reg = reg.reindex(columns=session_names)
        reg.to_csv(args.out_dir / "cell_registry.csv")
        np.save(args.out_dir / "prob_matrix.npy", probs.astype(np.float32))

        spans = reg.notna().sum(axis=1).values
        summary.update(dict(
            n_tracked_ids=int(len(spans)),
            max_span=int(spans.max()),
            ge_2=int((spans >= 2).sum()), ge_5=int((spans >= 5).sum()),
            ge_10=int((spans >= 10).sum()), ge_15=int((spans >= 15).sum()),
            ge_20=int((spans >= 20).sum()),
        ))

    with open(args.out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Report ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 68)
    print(f"DEEPUNITMATCH (canonical, 1 batch, {len(ks_dirs)} sessions) "
          f"-- {elapsed/60:.1f} min")
    print("=" * 68)
    if bayes_ok:
        spans = (pd.read_csv(args.out_dir / "cell_registry.csv", index_col=0)
                 .notna().sum(axis=1).values)
        print(f"  units: {n_units}   tracked IDs: {len(spans)}   max span: {spans.max()}")
        print("  span distribution  --  side-by-side benchmarks (% of tracked IDs):")
        OLD = {2: 17.4, 5: 3.4, 10: 1.0, 15: 0.4, 20: 0.1}
        UM_NEW = {2: 19.8, 5: 4.9, 10: 1.6, 15: 0.9, 20: 0.5}
        for thr in (2, 5, 10, 15, 20):
            c = summary[f"ge_{thr}"]
            pct = 100 * c / len(spans) if len(spans) else 0
            print(f"    >= {thr:2d} sess: {c:5d}  ({pct:5.2f}%)   "
                  f"UM 3.2.9 {UM_NEW[thr]:.1f}%   old-batched-2.41 {OLD[thr]:.1f}%")
    else:
        print(f"  units (UM loaded): {n_units}   DeepUM sim_matrix: "
              f"{sim_matrix.shape[0]}x{sim_matrix.shape[0]}")
        print(f"  Bayes/tracking SKIPPED. Use sim_matrix.npy for sim-only tracking.")
    print(f"\n  output -> {args.out_dir}")


if __name__ == "__main__":
    main()
