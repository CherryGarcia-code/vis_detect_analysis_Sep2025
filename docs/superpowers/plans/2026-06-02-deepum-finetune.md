# DeepUnitMatch Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the DeepUnitMatch waveform encoder on BG_046 striatal data (three configurations) and measure whether cross-session tracking beats stock DeepUM (6.3% ≥2-sess) and approaches UnitMatch (19.8%).

**Architecture:** All new code lives as **tracked scripts** in `scripts/pipelines/tracking/` on branch `feature/deepum-finetune` (in the worktree). The gitignored vendored `_DeepUnitMatch_repo/` is treated **read-only**: we import its model/loss/dataset and **monkeypatch at runtime** (mirroring the existing `param_fun` monkeypatch in `run_deepunitmatch_all.py`) — the missing `_augment_original` augmentation and `AverageMeter` are supplied by our code, not by editing the vendor. Training is self-supervised CLIP on CV-half pairs; we run an AE-pretrain stage only for the from-scratch rung. Validation during training is a fast within-session top-1 proxy; the real metric is the ≥2-sess span produced by the existing tracking pipeline run with `--ckpt`.

**Tech Stack:** Python, PyTorch (double precision), the vendored `DeepUnitMatch` package (`SpatioTemporalCNN_V2`, `CustomClipLoss`, `Projector`, `NeuropixelsDataset`), h5py, conda env `unitmatch_env` (local) / `unitmatch` (cluster ceph), SLURM A100.

---

## Working agreement (read before any task)

- **All work happens in the worktree:** `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.claude/worktrees/deepum-finetune/`. Treat that as repo root for every path below.
- **Branch discipline (the user runs parallel chats):** before EVERY `git commit`, run `git branch --show-current` and confirm it prints `feature/deepum-finetune`. If it prints anything else, stop — you are in the wrong working tree.
- **Running anything that imports torch / UnitMatchPy / DeepUnitMatch MUST use conda** (running the env python.exe directly crashes scipy on Windows):
  `conda run -n unitmatch_env --no-capture-output python <file>`
  `conda run` rejects multi-line `python -c`; always run a file.
- **Do NOT edit `_DeepUnitMatch_repo/`** (gitignored vendor). If you think you need to, you don't — monkeypatch from our code instead.
- **DeepUM import resolution** is via the installed `UnitMatchPy` package (`_umpy.__path__` → parent has `DeepUnitMatch/`), so our worktree scripts import DeepUM regardless of cwd. The worktree does not contain `_DeepUnitMatch_repo/` and does not need to.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## File structure (what each file owns)

| File | Responsibility |
|---|---|
| `scripts/pipelines/tracking/train_deepum_common.py` | Shared, import-light helpers: DeepUM path resolution, `AverageMeter`, `augment_channel_roll`, `patch_augmentation`, `build_export_checkpoint`, `load_finetuned_encoder`, `write_synthetic_cache` (test fixture), `within_session_top1_accuracy`. |
| `scripts/pipelines/tracking/train_deepum_clip.py` | Stage-2 CLIP fine-tune. `--init shipped|<ae_ckpt>`, `--freeze fcblock|none`. Writes per-epoch training + export checkpoints + proxy log. |
| `scripts/pipelines/tracking/train_deepum_ae.py` | Stage-1 AE reconstruction pretrain (from-scratch rung only). Writes encoder checkpoints. |
| `scripts/pipelines/tracking/eval_deepum_checkpoints.py` | Runs the tracking pipeline per export checkpoint, parses `run_summary.json`, writes `finetune_comparison.csv`. |
| `scripts/pipelines/tracking/run_deepunitmatch_all.py` | **Modify:** add `--ckpt` to use a fine-tuned encoder instead of the shipped one. |
| `scripts/pipelines/tracking/slurm/train_deepum.sbatch` | A100 submission (ceph-conda pattern). |
| `scripts/pipelines/tracking/tests/test_deepum_common.py` | Unit tests for common helpers (standalone asserts, conda-run). |
| `scripts/pipelines/tracking/tests/test_deepum_smoke.py` | Integration smoke: synthetic cache → 1-epoch CLIP → export → reload → inference. |
| `scripts/pipelines/tracking/tests/test_eval_assembly.py` | Unit test for the comparison-CSV assembly. |

Tests live under `scripts/pipelines/tracking/tests/` (NOT the repo-root `tests/`) so the main `.venv` pytest run does not try to collect torch-dependent tests.

---

## Task 0: Cluster prerequisite — verify (and if needed fix) GPU PyTorch

**Files:** none (cluster verification gate).

This is BLOCKING for Tasks 5 and 9 (AE pretrain + real training). Prior DeepUM jobs ran CPU torch.

- [ ] **Step 1: Write a one-line GPU check file** at `scripts/pipelines/tracking/tools/check_gpu_torch.py`:

```python
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
```

- [ ] **Step 2: Run it on a cluster A100 node** (interactive or a 2-line sbatch) under the ceph env:

```bash
srun -p a100 --gres=gpu:1 --mem=8G --time=00:10:00 \
  bash -lc 'export PATH="${CONDA_ENV}/bin:${PATH}"; python scripts/pipelines/tracking/tools/check_gpu_torch.py'
```
Expected: `cuda_available True` and a device name.

- [ ] **Step 3: If `cuda_available False`**, install a CUDA build into the ceph `unitmatch` env (match the cluster's CUDA), e.g.:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```
Re-run Step 2 until it prints `True`. Record the working torch version + CUDA tag in the spec's §11 open items.

- [ ] **Step 4: Commit the check tool** (confirm branch first):

```bash
git branch --show-current   # must be feature/deepum-finetune
git add scripts/pipelines/tracking/tools/check_gpu_torch.py
git commit -m "Add GPU torch check tool for DeepUM training prereq" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1: Common helpers — `AverageMeter` + `augment_channel_roll`

**Files:**
- Create: `scripts/pipelines/tracking/train_deepum_common.py`
- Create: `scripts/pipelines/tracking/tests/test_deepum_common.py`

- [ ] **Step 1: Write failing tests** in `scripts/pipelines/tracking/tests/test_deepum_common.py`:

```python
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_deepum_common import AverageMeter, augment_channel_roll


def test_average_meter():
    m = AverageMeter()
    m.update(2.0, n=1); m.update(4.0, n=3)
    assert m.count == 4
    assert abs(m.avg - (2.0 + 4.0 * 3) / 4) < 1e-9
    print("test_average_meter PASS")


def test_augment_none_is_identity():
    data = np.arange(60 * 30, dtype=float).reshape(60, 30)
    out = augment_channel_roll(data, choice="none")
    assert out.shape == data.shape
    assert np.array_equal(out, data)
    assert out is not data  # must not mutate caller's array
    print("test_augment_none_is_identity PASS")


def test_augment_roll_preserves_shape_and_changes_values():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((60, 30))
    for choice in ("roll_up", "roll_down"):
        out = augment_channel_roll(data, choice=choice)
        assert out.shape == (60, 30)
        assert not np.array_equal(out, data)  # a non-constant input must change
    # original is untouched
    assert np.array_equal(data, data)
    print("test_augment_roll_preserves_shape_and_changes_values PASS")


if __name__ == "__main__":
    test_average_meter()
    test_augment_none_is_identity()
    test_augment_roll_preserves_shape_and_changes_values()
    print("ALL COMMON-BASIC TESTS PASSED")
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_deepum_common.py
```
Expected: FAIL — `ImportError: cannot import name 'AverageMeter'` (file does not exist yet).

- [ ] **Step 3: Create `train_deepum_common.py`** with these two helpers (port augmentation verbatim from upstream `_augment_original`, made standalone + non-mutating):

```python
"""Shared helpers for DeepUnitMatch fine-tuning (tracked; vendor stays read-only)."""
from __future__ import annotations
import random
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
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_deepum_common.py
```
Expected: `ALL COMMON-BASIC TESTS PASSED`.

- [ ] **Step 5: Commit** (confirm branch first):

```bash
git branch --show-current   # must be feature/deepum-finetune
git add scripts/pipelines/tracking/train_deepum_common.py scripts/pipelines/tracking/tests/test_deepum_common.py
git commit -m "DeepUM common: AverageMeter + channel-roll augmentation (TDD)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Common — DeepUM path resolution + checkpoint export/load

**Files:**
- Modify: `scripts/pipelines/tracking/train_deepum_common.py`
- Modify: `scripts/pipelines/tracking/tests/test_deepum_common.py`

- [ ] **Step 1: Add failing tests** to `tests/test_deepum_common.py` (append before the `__main__` block, and add the calls into `__main__`):

```python
def test_add_deepum_to_path_imports_model():
    from train_deepum_common import add_deepum_to_path
    repo = add_deepum_to_path()
    assert (repo / "DeepUnitMatch").is_dir()
    from DeepUnitMatch.utils.mymodel import SpatioTemporalCNN_V2  # noqa: F401
    print("test_add_deepum_to_path_imports_model PASS")


def test_export_and_load_roundtrip(tmp_path_str=None):
    import tempfile, os
    import torch
    from train_deepum_common import (add_deepum_to_path, build_export_checkpoint,
                                      load_finetuned_encoder)
    add_deepum_to_path()
    from DeepUnitMatch.utils.mymodel import SpatioTemporalCNN_V2
    from DeepUnitMatch.utils.losses import CustomClipLoss, Projector
    model = SpatioTemporalCNN_V2(30, 60, 256).double()
    clip_loss = CustomClipLoss().double()
    projector = Projector(256, 128, 128, 1, 0.1).double()
    export = build_export_checkpoint(model, clip_loss, projector)
    assert set(export.keys()) == {"model", "clip_loss", "projector"}
    d = tmp_path_str or tempfile.mkdtemp()
    p = os.path.join(d, "export_epoch_0.pt")
    torch.save(export, p)
    reloaded = load_finetuned_encoder(p, device="cpu")
    # weights identical to the originals
    for k, v in model.state_dict().items():
        assert torch.allclose(reloaded.state_dict()[k], v)
    print("test_export_and_load_roundtrip PASS")
```
Add to `__main__`:
```python
    test_add_deepum_to_path_imports_model()
    test_export_and_load_roundtrip()
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_deepum_common.py
```
Expected: FAIL — `ImportError: cannot import name 'add_deepum_to_path'`.

- [ ] **Step 3: Append helpers** to `train_deepum_common.py`:

```python
import sys
from pathlib import Path


def add_deepum_to_path():
    """Locate the vendored DeepUnitMatch source via the installed UnitMatchPy
    package and put it on sys.path. Mirrors run_deepunitmatch_all.py."""
    import UnitMatchPy as _umpy
    candidates = []
    try:
        candidates.append(Path(next(iter(_umpy.__path__))).resolve().parent)
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
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_deepum_common.py
```
Expected: `ALL COMMON-BASIC TESTS PASSED` (now includes the two new tests).

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must be feature/deepum-finetune
git add scripts/pipelines/tracking/train_deepum_common.py scripts/pipelines/tracking/tests/test_deepum_common.py
git commit -m "DeepUM common: path resolution + checkpoint export/load roundtrip (TDD)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Common — augmentation patch + synthetic cache fixture + proxy metric

**Files:**
- Modify: `scripts/pipelines/tracking/train_deepum_common.py`
- Modify: `scripts/pipelines/tracking/tests/test_deepum_common.py`

- [ ] **Step 1: Add failing tests** (append; add calls to `__main__`):

```python
def test_patch_augmentation_enables_train_mode():
    import tempfile
    from train_deepum_common import (add_deepum_to_path, patch_augmentation,
                                      write_synthetic_cache)
    add_deepum_to_path()
    from DeepUnitMatch.utils.npdataset import NeuropixelsDataset
    patch_augmentation(NeuropixelsDataset)
    d = tempfile.mkdtemp()
    write_synthetic_cache(d, n_sessions=2, units_per_session=5, seed=1)
    ds = NeuropixelsDataset(d, batch_size=4, mode="train")
    fh, sh, pos, exp, fp = ds[0]
    assert fh.shape == (60, 30) and sh.shape == (60, 30)
    print("test_patch_augmentation_enables_train_mode PASS")


def test_write_synthetic_cache_layout():
    import os, tempfile
    from train_deepum_common import write_synthetic_cache
    d = tempfile.mkdtemp()
    write_synthetic_cache(d, n_sessions=2, units_per_session=3, seed=0)
    sess = sorted(os.listdir(d))
    assert sess == ["0", "1"]
    files = sorted(os.listdir(os.path.join(d, "0")))
    assert files == ["Unit0.npy", "Unit1.npy", "Unit2.npy"]
    print("test_write_synthetic_cache_layout PASS")
```
Add to `__main__`:
```python
    test_write_synthetic_cache_layout()
    test_patch_augmentation_enables_train_mode()
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_deepum_common.py
```
Expected: FAIL — `cannot import name 'patch_augmentation'`.

- [ ] **Step 3: Append helpers** to `train_deepum_common.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_deepum_common.py
```
Expected: `ALL COMMON-BASIC TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must be feature/deepum-finetune
git add scripts/pipelines/tracking/train_deepum_common.py scripts/pipelines/tracking/tests/test_deepum_common.py
git commit -m "DeepUM common: augmentation patch + synthetic cache + proxy metric (TDD)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `train_deepum_clip.py` — CLIP fine-tune + integration smoke

**Files:**
- Create: `scripts/pipelines/tracking/train_deepum_clip.py`
- Create: `scripts/pipelines/tracking/tests/test_deepum_smoke.py`

- [ ] **Step 1: Write the failing smoke test** `scripts/pipelines/tracking/tests/test_deepum_smoke.py`:

```python
import os, sys, glob, tempfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(HERE)
sys.path.insert(0, TRACK)
from train_deepum_common import (add_deepum_to_path, write_synthetic_cache,
                                 load_finetuned_encoder)


def test_clip_one_epoch_smoke():
    cache = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    write_synthetic_cache(cache, n_sessions=2, units_per_session=6, seed=2)
    cmd = [
        "python", os.path.join(TRACK, "train_deepum_clip.py"),
        "--train-root", cache, "--out-dir", out,
        "--init", "shipped", "--freeze", "fcblock",
        "--epochs", "1", "--batch", "4", "--save-freq", "1", "--device", "cpu",
    ]
    subprocess.run(cmd, check=True)
    exports = glob.glob(os.path.join(out, "export_epoch_*.pt"))
    assert exports, "no export checkpoint written"
    # the export reloads into an encoder and runs inference to finite output
    import numpy as np, torch
    add_deepum_to_path()
    from DeepUnitMatch.utils.losses import clip_sim
    model = load_finetuned_encoder(exports[0], device="cpu")
    x = torch.from_numpy(np.zeros((3, 60, 30), dtype=np.float64))
    sim = clip_sim(model(x), model(x)).detach().numpy()
    assert np.isfinite(sim).all()
    print("test_clip_one_epoch_smoke PASS")


if __name__ == "__main__":
    test_clip_one_epoch_smoke()
    print("ALL SMOKE TESTS PASSED")
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_deepum_smoke.py
```
Expected: FAIL — `train_deepum_clip.py` does not exist (`No such file`/nonzero exit).

- [ ] **Step 3: Create `train_deepum_clip.py`:**

```python
#!/usr/bin/env python3
"""Stage-2 CLIP fine-tune of the DeepUnitMatch encoder on BG_046.

Self-supervised: positives = (aug CV-half-1, aug CV-half-2) of the same unit;
negatives = other units in the same-session batch. Loss = CLIP on projector outputs.
Inference later uses the 256-dim encoder via clip_sim, so fine-tuning FcBlock reshapes
the embedding space tracking consumes.

Run under unitmatch_env:
  conda run -n unitmatch_env --no-capture-output python train_deepum_clip.py \
    --train-root <processed_waveforms> --out-dir <out> --init shipped --freeze fcblock
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_deepum_common import (add_deepum_to_path, shipped_checkpoint_path,
                                 patch_augmentation, build_export_checkpoint,
                                 within_session_top1_accuracy, AverageMeter)


def build_model(device):
    add_deepum_to_path()
    from DeepUnitMatch.utils.mymodel import SpatioTemporalCNN_V2
    from DeepUnitMatch.utils.losses import CustomClipLoss, Projector
    model = SpatioTemporalCNN_V2(n_channel=30, n_time=60, n_output=256).to(device).double()
    clip_loss = CustomClipLoss().to(device)
    projector = Projector(256, 128, 128, 1, 0.1).to(device).double()
    return model, clip_loss, projector


def init_weights(model, clip_loss, init, device):
    if init == "shipped":
        ck = torch.load(shipped_checkpoint_path(), map_location=device)
        model.load_state_dict(ck["model"])
        if "clip_loss" in ck:
            clip_loss.load_state_dict(ck["clip_loss"])
        print(f"  init: shipped checkpoint", flush=True)
    else:
        ck = torch.load(init, map_location=device)
        model.load_state_dict(ck["encoder"])
        print(f"  init: AE encoder from {init}", flush=True)


def apply_freeze(model, freeze):
    if freeze == "fcblock":
        for name, p in model.named_parameters():
            p.requires_grad = "FcBlock" in name
    elif freeze == "none":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown freeze policy {freeze}")
    n_train = sum(p.requires_grad for p in model.parameters())
    print(f"  freeze={freeze}: {n_train} encoder params trainable", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init", default="shipped", help="'shipped' or path to AE ckpt")
    ap.add_argument("--freeze", default="fcblock", choices=["fcblock", "none"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-proj", type=float, default=1.1e-4)
    ap.add_argument("--save-freq", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    add_deepum_to_path()
    from DeepUnitMatch.utils.npdataset import (NeuropixelsDataset,
                                               TrainExperimentBatchSampler,
                                               ValidationExperimentBatchSampler)
    patch_augmentation(NeuropixelsDataset)

    model, clip_loss, projector = build_model(device)
    init_weights(model, clip_loss, args.init, device)
    apply_freeze(model, args.freeze)

    enc_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam([
        {"params": enc_params, "lr": args.lr_enc},
        {"params": list(projector.parameters()) + list(clip_loss.parameters()),
         "lr": args.lr_proj},
    ])

    train_ds = NeuropixelsDataset(args.train_root, batch_size=args.batch, mode="train")
    train_loader = DataLoader(
        train_ds, batch_sampler=TrainExperimentBatchSampler(train_ds, args.batch, shuffle=True))
    val_ds = NeuropixelsDataset(args.train_root, batch_size=args.batch, mode="val")
    val_loader = DataLoader(
        val_ds, batch_sampler=ValidationExperimentBatchSampler(val_ds, shuffle=False))

    proxy_log = []
    for epoch in range(args.epochs):
        model.train(); clip_loss.train()
        losses = AverageMeter()
        for estimates, candidates, *_ in train_loader:
            estimates = estimates.to(device).double()
            candidates = candidates.to(device).double()
            optimizer.zero_grad()
            loss = clip_loss(projector(model(estimates)), projector(model(candidates)))
            loss.backward(); optimizer.step()
            losses.update(loss.item(), estimates.shape[0])
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "clip_loss": clip_loss.state_dict(), "epoch": epoch},
                       os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"))
            torch.save(build_export_checkpoint(model, clip_loss, projector),
                       os.path.join(args.out_dir, f"export_epoch_{epoch}.pt"))
            acc = within_session_top1_accuracy(model, val_loader, device)
            proxy_log.append({"epoch": epoch, "train_loss": losses.avg, "val_top1": acc})
            print(f"epoch {epoch}: train_loss={losses.avg:.6f} val_top1={acc:.4f}", flush=True)
    with open(os.path.join(args.out_dir, "proxy_log.json"), "w") as f:
        json.dump({"init": args.init, "freeze": args.freeze, "log": proxy_log}, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_deepum_smoke.py
```
Expected: `ALL SMOKE TESTS PASSED`.

- [ ] **Step 5: Verify freeze policy** with a quick check file `scripts/pipelines/tracking/tests/test_freeze_policy.py`:

```python
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from train_deepum_common import add_deepum_to_path
from train_deepum_clip import build_model, apply_freeze

model, _, _ = build_model("cpu")
apply_freeze(model, "fcblock")
assert all(("FcBlock" in n) == p.requires_grad for n, p in model.named_parameters())
apply_freeze(model, "none")
assert all(p.requires_grad for p in model.parameters())
print("test_freeze_policy PASS")
```
Run:
```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_freeze_policy.py
```
Expected: `test_freeze_policy PASS`.

- [ ] **Step 6: Commit**

```bash
git branch --show-current   # must be feature/deepum-finetune
git add scripts/pipelines/tracking/train_deepum_clip.py scripts/pipelines/tracking/tests/test_deepum_smoke.py scripts/pipelines/tracking/tests/test_freeze_policy.py
git commit -m "DeepUM CLIP fine-tune script + integration smoke + freeze-policy test" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `train_deepum_ae.py` — AE pretrain (from-scratch rung)

**Files:**
- Create: `scripts/pipelines/tracking/train_deepum_ae.py`
- Create: `scripts/pipelines/tracking/tests/test_ae_smoke.py`

- [ ] **Step 1: Write failing smoke test** `tests/test_ae_smoke.py`:

```python
import os, sys, glob, tempfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(HERE)
sys.path.insert(0, TRACK)
from train_deepum_common import write_synthetic_cache, add_deepum_to_path


def test_ae_one_epoch_then_clip_init():
    cache = tempfile.mkdtemp(); out = tempfile.mkdtemp(); clip_out = tempfile.mkdtemp()
    write_synthetic_cache(cache, n_sessions=2, units_per_session=6, seed=3)
    subprocess.run(["python", os.path.join(TRACK, "train_deepum_ae.py"),
                    "--train-root", cache, "--out-dir", out,
                    "--epochs", "1", "--batch", "4", "--save-freq", "1",
                    "--device", "cpu"], check=True)
    ae_ckpts = sorted(glob.glob(os.path.join(out, "ae_epoch_*.pt")))
    assert ae_ckpts, "no AE checkpoint written"
    # the AE encoder must initialise the CLIP stage (key 'encoder' loads into the CNN)
    subprocess.run(["python", os.path.join(TRACK, "train_deepum_clip.py"),
                    "--train-root", cache, "--out-dir", clip_out,
                    "--init", ae_ckpts[-1], "--freeze", "fcblock",
                    "--epochs", "1", "--batch", "4", "--save-freq", "1",
                    "--device", "cpu"], check=True)
    assert glob.glob(os.path.join(clip_out, "export_epoch_*.pt"))
    print("test_ae_one_epoch_then_clip_init PASS")


if __name__ == "__main__":
    test_ae_one_epoch_then_clip_init()
    print("ALL AE SMOKE TESTS PASSED")
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_ae_smoke.py
```
Expected: FAIL — `train_deepum_ae.py` missing.

- [ ] **Step 3: Create `train_deepum_ae.py`:**

```python
#!/usr/bin/env python3
"""Stage-1 autoencoder pretrain for DeepUnitMatch (from-scratch rung only).

Reconstructs each CV half with SpatioTemporalAutoEncoder_V2 (AELoss lambda1=0,
lambda2=1.0 = pure MSE). Saves checkpoints with an 'encoder' key that
train_deepum_clip.py --init <ckpt> loads into the SpatioTemporalCNN_V2 encoder.
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_deepum_common import add_deepum_to_path, AverageMeter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-freq", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    add_deepum_to_path()
    from DeepUnitMatch.utils.mymodel import SpatioTemporalAutoEncoder_V2
    from DeepUnitMatch.utils.losses import AELoss
    from DeepUnitMatch.utils.npdataset import (NeuropixelsDataset,
                                               TrainExperimentBatchSampler)

    ae = SpatioTemporalAutoEncoder_V2(n_channel=30, n_time=60, n_output=256).to(device).double()
    ae_loss = AELoss(lambda1=0.0, lambda2=1.0)
    optimizer = optim.Adam(ae.parameters(), lr=args.lr)

    ds = NeuropixelsDataset(args.train_root, batch_size=args.batch, mode="val")  # no aug for AE target
    loader = DataLoader(ds, batch_sampler=TrainExperimentBatchSampler(ds, args.batch, shuffle=True))

    log = []
    for epoch in range(args.epochs):
        ae.train()
        losses = AverageMeter()
        for estimates, candidates, *_ in loader:
            for x in (estimates, candidates):  # reconstruct both CV halves
                x = x.to(device).double()
                optimizer.zero_grad()
                loss = ae_loss(ae(x), x)
                loss.backward(); optimizer.step()
                losses.update(loss.item(), x.shape[0])
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            torch.save({"model": ae.state_dict(), "encoder": ae.encoder.state_dict(),
                        "optimizer": optimizer.state_dict(), "epoch": epoch},
                       os.path.join(args.out_dir, f"ae_epoch_{epoch}.pt"))
            log.append({"epoch": epoch, "recon_loss": losses.avg})
            print(f"AE epoch {epoch}: recon_loss={losses.avg:.6f}", flush=True)
    with open(os.path.join(args.out_dir, "ae_log.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_ae_smoke.py
```
Expected: `ALL AE SMOKE TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must be feature/deepum-finetune
git add scripts/pipelines/tracking/train_deepum_ae.py scripts/pipelines/tracking/tests/test_ae_smoke.py
git commit -m "DeepUM AE pretrain script + smoke (AE encoder inits CLIP stage)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Wire `--ckpt` into `run_deepunitmatch_all.py`

**Files:**
- Modify: `scripts/pipelines/tracking/run_deepunitmatch_all.py`

The runner currently loads the model at one line:
`model = dum_test.load_trained_model(device="cpu")` (inside `main()`, the "Network inference" block). We add an optional `--ckpt`.

- [ ] **Step 1: Add a guard test** `scripts/pipelines/tracking/tests/test_runner_ckpt_arg.py`:

```python
import os, sys, subprocess
TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = subprocess.run(["python", os.path.join(TRACK, "run_deepunitmatch_all.py"), "--help"],
                     capture_output=True, text=True)
assert "--ckpt" in out.stdout, "runner is missing --ckpt"
print("test_runner_ckpt_arg PASS")
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_runner_ckpt_arg.py
```
Expected: FAIL — assertion error (`--ckpt` not in help).

- [ ] **Step 3: Edit `run_deepunitmatch_all.py`.**

(3a) In `main()`, add the argument next to the existing ones:
```python
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Fine-tuned export checkpoint; if set, used instead of the "
                         "shipped DeepUM model.")
```

(3b) Replace the model-load line:
```python
    model = dum_test.load_trained_model(device="cpu")
```
with:
```python
    if args.ckpt is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_deepum_common import load_finetuned_encoder
        print(f"DeepUM step 2  load_finetuned_encoder({args.ckpt}) ...", flush=True)
        model = load_finetuned_encoder(str(args.ckpt), device="cpu")
    else:
        model = dum_test.load_trained_model(device="cpu")
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_runner_ckpt_arg.py
```
Expected: `test_runner_ckpt_arg PASS`.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must be feature/deepum-finetune
git add scripts/pipelines/tracking/run_deepunitmatch_all.py scripts/pipelines/tracking/tests/test_runner_ckpt_arg.py
git commit -m "run_deepunitmatch_all: add --ckpt to track with a fine-tuned encoder" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: `eval_deepum_checkpoints.py` — run pipeline per checkpoint + comparison CSV

**Files:**
- Create: `scripts/pipelines/tracking/eval_deepum_checkpoints.py`
- Create: `scripts/pipelines/tracking/tests/test_eval_assembly.py`

- [ ] **Step 1: Write failing test** for the CSV-assembly helper `tests/test_eval_assembly.py`:

```python
import os, sys
TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRACK)
from eval_deepum_checkpoints import summarize_run, BASELINE_ROWS


def test_summarize_run_row():
    summary = {"n_tracked_ids": 1000, "ge_2": 80, "ge_5": 10, "ge_10": 2,
               "ge_15": 0, "ge_20": 0, "max_span": 12}
    row = summarize_run("warmstart_ep10", summary)
    assert row["label"] == "warmstart_ep10"
    assert abs(row["ge_2_pct"] - 8.0) < 1e-9      # 80/1000
    assert row["max_span"] == 12
    print("test_summarize_run_row PASS")


def test_baseline_rows_present():
    labels = {r["label"] for r in BASELINE_ROWS}
    assert {"UM 3.2.9", "DeepUM stock"} <= labels
    print("test_baseline_rows_present PASS")


if __name__ == "__main__":
    test_summarize_run_row()
    test_baseline_rows_present()
    print("ALL EVAL-ASSEMBLY TESTS PASSED")
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_eval_assembly.py
```
Expected: FAIL — `eval_deepum_checkpoints` missing.

- [ ] **Step 3: Create `eval_deepum_checkpoints.py`:**

```python
#!/usr/bin/env python3
"""Run the DeepUM tracking pipeline for each fine-tuned export checkpoint and
tabulate the span distribution vs the UM and stock-DeepUM baselines.

For each export_epoch_*.pt (optionally strided), invokes run_deepunitmatch_all.py
--ckpt and reads its run_summary.json, then writes finetune_comparison.csv.
"""
from __future__ import annotations
import argparse, os, sys, json, glob, subprocess
from pathlib import Path
import pandas as pd

# Hard reference numbers (from memory/neuron_tracking_may2026.md side-by-side).
BASELINE_ROWS = [
    {"label": "UM 3.2.9", "ge_2_pct": 19.8, "ge_5_pct": 4.9, "ge_10_pct": 1.6,
     "ge_15_pct": 0.9, "ge_20_pct": 0.5, "max_span": 28},
    {"label": "DeepUM stock", "ge_2_pct": 6.3, "ge_5_pct": 0.4, "ge_10_pct": 0.03,
     "ge_15_pct": 0.0, "ge_20_pct": 0.0, "max_span": 14},
]
THRESHOLDS = [2, 5, 10, 15, 20]


def summarize_run(label, summary):
    n = summary.get("n_tracked_ids", 0) or 0
    row = {"label": label, "n_tracked_ids": n, "max_span": summary.get("max_span", 0)}
    for t in THRESHOLDS:
        c = summary.get(f"ge_{t}", 0)
        row[f"ge_{t}_pct"] = round(100 * c / n, 3) if n else 0.0
    return row


def select_checkpoints(ckpt_dir, stride):
    exports = sorted(glob.glob(os.path.join(ckpt_dir, "export_epoch_*.pt")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not exports:
        return []
    epochs = [int(p.split("_")[-1].split(".")[0]) for p in exports]
    chosen = [p for p, e in zip(exports, epochs) if e % stride == 0]
    if exports[-1] not in chosen:      # always include the last epoch
        chosen.append(exports[-1])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="dir with export_epoch_*.pt")
    ap.add_argument("--input", required=True, help="UM input dir (BG_046 sessions)")
    ap.add_argument("--out-root", required=True, help="where per-ckpt outputs + CSV go")
    ap.add_argument("--runner", default=str(Path(__file__).with_name("run_deepunitmatch_all.py")))
    ap.add_argument("--stride", type=int, default=10, help="eval every Nth epoch + last")
    ap.add_argument("--label-prefix", default="ft")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    rows = list(BASELINE_ROWS)
    for ckpt in select_checkpoints(args.ckpt_dir, args.stride):
        epoch = int(ckpt.split("_")[-1].split(".")[0])
        label = f"{args.label_prefix}_ep{epoch}"
        out_dir = os.path.join(args.out_root, label)
        subprocess.run([sys.executable, args.runner, "--input", args.input,
                        "--out-dir", out_dir, "--ckpt", ckpt], check=True)
        with open(os.path.join(out_dir, "run_summary.json")) as f:
            summary = json.load(f)
        rows.append(summarize_run(label, summary))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_root, "finetune_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n unitmatch_env --no-capture-output python scripts/pipelines/tracking/tests/test_eval_assembly.py
```
Expected: `ALL EVAL-ASSEMBLY TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # must be feature/deepum-finetune
git add scripts/pipelines/tracking/eval_deepum_checkpoints.py scripts/pipelines/tracking/tests/test_eval_assembly.py
git commit -m "DeepUM eval: per-checkpoint tracking runs + comparison CSV (TDD assembly)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: SLURM submission script

**Files:**
- Create: `scripts/pipelines/tracking/slurm/train_deepum.sbatch`

Paths marked `# CONFIRM` are validated in Task 0 / Task 9 against the real ceph layout; the values below are the best-known defaults from `memory/neuron_tracking_may2026.md`.

- [ ] **Step 1: Create `slurm/train_deepum.sbatch`:**

```bash
#!/bin/bash
#SBATCH --job-name=deepum_ft
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=deepum_ft_%j.out

set -euo pipefail
echo "SLURM job $SLURM_JOB_ID on $(hostname)"

# --- ceph-conda activation (mirrors KS4/UM run scripts) ---
CONDA_ENV="/ceph/.../conda_envs/unitmatch"        # CONFIRM
export PATH="${CONDA_ENV}/bin:${PATH}"

# --- repo + data ---
REPO="/ceph/.../vis_detect_analysis_Sep2025"      # CONFIRM (feature/deepum-finetune checkout)
TRACK="${REPO}/scripts/pipelines/tracking"
TRAIN_ROOT="/ceph/.../UnitMatchPy/DeepUnitMatch/processed_waveforms"   # CONFIRM
OUT_BASE="${REPO}/data/unit_match/output/BG_046_finetune"

python "${TRACK}/tools/check_gpu_torch.py"   # fail fast if CUDA missing

MODE="${1:-clip_warm_frozen}"   # clip_warm_frozen | clip_warm_unfrozen | ae | clip_scratch

case "${MODE}" in
  clip_warm_frozen)
    python "${TRACK}/train_deepum_clip.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/rung1_warm_frozen" --init shipped --freeze fcblock ;;
  clip_warm_unfrozen)
    python "${TRACK}/train_deepum_clip.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/rung2_warm_unfrozen" --init shipped --freeze none ;;
  ae)
    python "${TRACK}/train_deepum_ae.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/ae" ;;
  clip_scratch)
    AE_CKPT=$(ls -t "${OUT_BASE}/ae"/ae_epoch_*.pt | head -1)
    python "${TRACK}/train_deepum_clip.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/rung3_scratch" --init "${AE_CKPT}" --freeze fcblock ;;
  *) echo "unknown MODE ${MODE}"; exit 2 ;;
esac
echo "done ${MODE}"
```

- [ ] **Step 2: Lint the bash** (local, no execution needed):

```bash
bash -n scripts/pipelines/tracking/slurm/train_deepum.sbatch && echo "syntax OK"
```
Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
git branch --show-current   # must be feature/deepum-finetune
git add scripts/pipelines/tracking/slurm/train_deepum.sbatch
git commit -m "DeepUM A100 sbatch: 3 rungs (warm-frozen/warm-unfrozen/scratch) + AE" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Cluster runbook — train 3 rungs, evaluate, compare (REAL run)

**Files:** none new (operational). Produces `data/unit_match/output/BG_046_finetune/finetune_comparison.csv`.

Prereq: Task 0 passed (GPU torch). The branch `feature/deepum-finetune` must be checked out on the cluster, and `# CONFIRM` paths in the sbatch corrected to the real ceph layout (cross-check against the prior UM/DeepUM run scripts for jobs 3026330 / 3025315).

- [ ] **Step 1: Sync the branch + verify the HDF5 cache** exists and has 42 session dirs:

```bash
ls "${TRAIN_ROOT}" | wc -l        # expect ~42
ls "${TRAIN_ROOT}"/0 | head        # expect Unit*.npy
```

- [ ] **Step 2: Submit AE pretrain (rung 3 prerequisite) and both warm-start rungs:**

```bash
sbatch scripts/pipelines/tracking/slurm/train_deepum.sbatch clip_warm_frozen   # rung 1
sbatch scripts/pipelines/tracking/slurm/train_deepum.sbatch clip_warm_unfrozen # rung 2
sbatch scripts/pipelines/tracking/slurm/train_deepum.sbatch ae                 # for rung 3
```
Expected: each ~1–6 h. Confirm `export_epoch_*.pt` (rungs 1/2) and `ae_epoch_*.pt` appear.

- [ ] **Step 3: After AE finishes, submit from-scratch CLIP (rung 3):**

```bash
sbatch scripts/pipelines/tracking/slurm/train_deepum.sbatch clip_scratch
```

- [ ] **Step 4: Evaluate each rung's checkpoints** (CPU job is fine; ~40 min per checkpoint, so stride keeps it bounded):

```bash
for RUNG in rung1_warm_frozen rung2_warm_unfrozen rung3_scratch; do
  python scripts/pipelines/tracking/eval_deepum_checkpoints.py \
    --ckpt-dir   data/unit_match/output/BG_046_finetune/${RUNG} \
    --input      data/unit_match/input/BG_046 \
    --out-root   data/unit_match/output/BG_046_finetune/${RUNG}_eval \
    --stride 10 --label-prefix ${RUNG}
done
```
Expected: a `finetune_comparison.csv` per rung with `ge_2_pct` rows alongside UM 19.8 and stock 6.3.

- [ ] **Step 5: Read off the result.** Best `ge_2_pct` per rung vs baselines. Record in the spec §9 and the memory file. Success = rung-1 clearly > 6.3; target = any rung ≥ ~19.8 and `ge_20 > 0`.

- [ ] **Step 6: Functional cross-check (optional but recommended).** For the winning rung's checkpoint, run the ISI-fingerprint validation used for the UM cohort to confirm long tracks are real:

```bash
python scripts/pipelines/tracking/validate_long_tracks.py \
  --registry data/unit_match/output/BG_046_finetune/<winning>_eval/<best_label>/cell_registry.csv
```
(Bring `validate_long_tracks.py` onto the branch first if needed — it is currently untracked in the main workspace, same situation as `run_deepunitmatch_all.py`; confirm with the user before committing another chat's WIP.)

- [ ] **Step 7: Commit the comparison artifacts** (CSV + proxy logs; not the large checkpoints):

```bash
git branch --show-current   # must be feature/deepum-finetune
git add data/unit_match/output/BG_046_finetune/**/finetune_comparison.csv \
        data/unit_match/output/BG_046_finetune/**/proxy_log.json
git commit -m "DeepUM fine-tune results: 3-rung comparison vs UM/stock" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Wrap-up — memory update + branch finalization

**Files:**
- Modify: `C:/Users/Ben/.claude/projects/e--python-analysis-git-repos-vis-detect-analysis-Sep2025/memory/neuron_tracking_may2026.md`

- [ ] **Step 1: Append a "DeepUM fine-tuning results (Jun 2026)" section** to the memory file summarizing the three-rung `ge_2_pct` outcome vs UM 19.8 / stock 6.3, the winning config, and the functional cross-check. (One paragraph; convert any relative dates to absolute.)

- [ ] **Step 2: Update `MEMORY.md` pointer** line for `neuron_tracking_may2026.md` to mention fine-tuning is done.

- [ ] **Step 3: Invoke `superpowers:finishing-a-development-branch`** to choose merge/PR/cleanup for `feature/deepum-finetune` (rebasing onto current `main` first, since this branch was based on `main`@7f438d4 which has since advanced).

---

## Self-review

**Spec coverage:**
- §2 missing pieces (`_augment_original`, `AverageMeter`) → Tasks 1, 3 (supplied via monkeypatch, not vendor edit — documented refinement).
- §3 method (CLIP on CV-halves, projector, encoder-at-inference) → Task 4.
- §4 three rungs → Tasks 4 (rungs 1/2), 5 (rung 3 AE), 9 (run all).
- §5 components → Tasks 1–8 (all files created).
- §6 checkpoint reconciliation (train vs export format) → Task 2 (`build_export_checkpoint`/`load_finetuned_encoder`), Task 4 (writes both).
- §7.1 untracked `run_deepunitmatch_all.py` → already committed; `validate_long_tracks.py` flagged in Task 9 Step 6. §7.2 GPU torch → Task 0. §7.3 SLURM/data path → Task 8/9.
- §8 testing (augmentation, dataset shape, integration smoke, export round-trip) → Tasks 1–5 tests.
- §9 success criteria + ISI cross-check → Task 9 Steps 5–6.
- §10 scope (no multi-subject driver; paths are args) → respected (all scripts take `--train-root`/`--input`).

**Placeholder scan:** sbatch `# CONFIRM` ceph paths are concrete best-known defaults with an explicit verification gate (Task 0/9), not forbidden "TODO" placeholders. No "TBD"/"implement later" anywhere.

**Type/name consistency:** `add_deepum_to_path`, `augment_channel_roll`, `patch_augmentation`, `build_export_checkpoint`, `load_finetuned_encoder`, `within_session_top1_accuracy`, `write_synthetic_cache`, `summarize_run`, `BASELINE_ROWS`, `select_checkpoints` are defined once and referenced consistently across tasks. Checkpoint keys: training `{model,optimizer,clip_loss,epoch}`, AE `{model,encoder,optimizer,epoch}`, export `{model,clip_loss,projector}` — used consistently in Tasks 2/4/5/6.
