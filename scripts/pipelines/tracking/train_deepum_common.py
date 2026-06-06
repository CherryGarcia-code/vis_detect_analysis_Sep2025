"""Shared helpers for DeepUnitMatch fine-tuning (tracked; vendor stays read-only)."""
from __future__ import annotations
import random
import sys
from pathlib import Path
import numpy as np


class AverageMeter:
    """Running average (matches upstream DeepUnitMatch utils.metric.AverageMeter)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def augment_channel_roll(data, choice=None, rng=random):
    """Port of upstream NeuropixelsDataset._augment_original (non-mutating).

    data: array [T, C]. Randomly shifts odd/even channel columns up or down by one
    electrode row (simulates a single-row probe drift), or leaves it unchanged.
    """
    data = np.asarray(data).copy()
    if choice is None:
        choice = rng.choice(["roll_up", "roll_down", "none"])
    C = data.shape[1]
    if choice == "roll_up":
        odd = np.arange(0, C - 1, 2)
        even = np.arange(1, C - 1, 2)
        if len(odd) > 1:
            data[:, odd[:-1]] = data[:, odd[1:]]
        if len(even) > 1:
            data[:, even[:-1]] = data[:, even[1:]]
    elif choice == "roll_down":
        odd = np.arange(2, C, 2)
        even = np.arange(3, C, 2)
        if len(odd) > 0:
            data[:, odd] = data[:, odd - 2]
        if len(even) > 0:
            data[:, even] = data[:, even - 2]
    return data


def add_deepum_to_path():
    """Locate the vendored DeepUnitMatch source via the installed UnitMatchPy
    package and put it on sys.path. The editable-install layout differs between
    machines (DeepUnitMatch/ may sit at __path__[0] itself, its parent, or its
    grandparent), so check all three -- mirrors run_deepunitmatch_all.py."""
    import UnitMatchPy as _umpy
    candidates = []
    try:
        base = Path(next(iter(_umpy.__path__))).resolve()
        candidates += [base, base.parent, base.parent.parent]
    except Exception:
        pass
    repo = next((c for c in candidates if (c / "DeepUnitMatch").is_dir()), None)
    if repo is None:
        raise RuntimeError(
            "Cannot locate DeepUnitMatch source. Tried: "
            + " | ".join(str(c) for c in candidates))
    for p in (repo, repo / "DeepUnitMatch"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return repo


def shipped_checkpoint_path():
    """Absolute path of the shipped pre-trained DeepUM checkpoint (utils/model)."""
    repo = add_deepum_to_path()
    return repo / "DeepUnitMatch" / "utils" / "model"


def build_export_checkpoint(model, clip_loss, projector=None):
    """Assemble an inference-format checkpoint readable by load_finetuned_encoder
    and the vendored load_trained_model (which only needs 'model' + 'clip_loss')."""
    out = {"model": model.state_dict(), "clip_loss": clip_loss.state_dict()}
    if projector is not None:
        out["projector"] = projector.state_dict()
    return out


def load_finetuned_encoder(ckpt_path, device="cpu"):
    """Build SpatioTemporalCNN_V2 and load encoder weights from a fine-tuned
    checkpoint (accepts a dict with 'model' or a bare state_dict)."""
    import torch
    add_deepum_to_path()
    from DeepUnitMatch.utils.mymodel import SpatioTemporalCNN_V2
    model = SpatioTemporalCNN_V2(n_channel=30, n_time=60, n_output=256).to(device).double()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def patch_augmentation(dataset_cls):
    """Install the missing _augment_original on the vendored dataset class so
    mode='train' applies channel-roll augmentation. Vendor file untouched."""
    def _augment_original(self, data):
        return augment_channel_roll(data)
    dataset_cls._augment_original = _augment_original
    return dataset_cls


def write_synthetic_cache(root, n_sessions=2, units_per_session=5, seed=0):
    """Write a tiny NeuropixelsDataset-layout HDF5 cache for tests.
    root/<session>/Unit<i>.npy each holds waveform (60,30,2) + MaxSitepos (2,)."""
    import os
    import h5py
    rng = np.random.default_rng(seed)
    for s in range(n_sessions):
        sd = os.path.join(root, str(s))
        os.makedirs(sd, exist_ok=True)
        for u in range(units_per_session):
            wf = rng.standard_normal((60, 30, 2)).astype(np.float64)
            pos = np.array([rng.uniform(0, 70), rng.uniform(0, 3840)], dtype=np.float64)
            with h5py.File(os.path.join(sd, f"Unit{u}.npy"), "w") as f:
                f.create_dataset("waveform", data=wf)
                f.create_dataset("MaxSitepos", data=pos)
    return root


def within_session_top1_accuracy(model, val_loader, device="cpu"):
    """Proxy metric: per session batch, does each unit's CV-half-1 encoding pick its
    own CV-half-2 as nearest? Returns mean accuracy across batches."""
    import torch
    add_deepum_to_path()
    from DeepUnitMatch.utils.losses import clip_prob
    model.eval()
    accs = []
    with torch.no_grad():
        for estimates, candidates, *_ in val_loader:
            estimates = estimates.to(device).double()
            candidates = candidates.to(device).double()
            bsz = estimates.shape[0]
            if bsz < 2:
                continue
            probs = clip_prob(model(estimates), model(candidates))
            pred = torch.argmax(probs, dim=1)
            gt = torch.arange(bsz, device=probs.device)
            accs.append((pred == gt).float().mean().item())
    return float(np.mean(accs)) if accs else 0.0
